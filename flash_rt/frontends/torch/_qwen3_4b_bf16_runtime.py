"""FlashRT -- Qwen3-4B BF16 prefill runtime shared by the per-device frontends.

The runtime owns what the per-device frontends have in common: the weights,
the RoPE tables, every activation buffer (sized once for ``max_seq``), input
staging and the CUDA graphs. The layer sequence itself is
``flash_rt.models.qwen3.prefill_bf16``; which device it runs on and which
attention it uses comes from the frontend's ``TARGET``.

Everything runs inside the instance's device context on that device's
current stream, so results are ordered with the caller's other work there.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

from flash_rt.frontends.torch._qwen3_4b_bf16_weights import (
    load_qwen3_4b_weights,
    qwen3_4b_dims,
)
from flash_rt.models.qwen3.prefill_bf16 import (
    Qwen3_4BDims,
    Qwen3Bf16Prefill,
    Qwen3Bf16Target,
    require_kernels,
    torch_reduce_tree,
)

BF16 = torch.bfloat16

#: How the HF baseline holds RoPE ``inv_freq``. ``from_pretrained(dtype=bf16)``
#: leaves the buffer fp32; casting a constructed model with ``.to(bfloat16)``
#: rounds it to bf16, which moves every angle at long positions.
ROPE_INV_FREQ = ("fp32", "bf16")


def _resolve_device(device: str | int | torch.device) -> torch.device:
    dev = torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)
    if dev.type != "cuda":
        raise ValueError(f"Qwen3-4B BF16 prefill runs on a CUDA device, got {dev}")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def rope_half_tables(dims: Qwen3_4BDims, positions: int, inv_freq: str,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``[positions, head_dim / 2]`` bf16 cos / sin, computed as HF computes them.

    HF builds ``inv_freq`` on the CPU in fp32 when the module is created, then
    per forward takes ``inv_freq @ position_ids`` in fp32, concatenates the
    result with itself and casts cos / sin to the activation dtype. The two
    halves are equal, so the kernels read one.
    """
    if inv_freq not in ROPE_INV_FREQ:
        raise ValueError(f"rope_inv_freq must be one of {ROPE_INV_FREQ}, got {inv_freq!r}")
    hd = dims.head_dim
    inv = 1.0 / (dims.rope_theta ** (
        torch.arange(0, hd, 2, dtype=torch.int64).to(dtype=torch.float) / hd))
    if inv_freq == "bf16":
        inv = inv.to(BF16)
    inv = inv[None, :, None].float().to(device)
    pos = torch.arange(positions, device=device)[None, None, :].float()
    freqs = (inv @ pos).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)[0]
    half = hd // 2
    return (emb.cos().to(BF16)[:, :half].contiguous(),
            emb.sin().to(BF16)[:, :half].contiguous())


class Qwen3_4BBf16Runtime:
    """Weights + persistent buffers + the bound prefill sequence (batch 1)."""

    def __init__(self, target: Qwen3Bf16Target, checkpoint_path: str | None, *,
                 device: str | int | torch.device, max_seq: int,
                 state_dict: Mapping[str, torch.Tensor] | None,
                 config: Any, rope_inv_freq: str, cuda_graphs: bool,
                 graph_cache_size: int, autotune_gemms: bool) -> None:
        self.target = target
        self.device = _resolve_device(device)
        self.max_seq = int(max_seq)
        if self.max_seq < 1:
            raise ValueError("max_seq must be positive")
        self.cuda_graphs = bool(cuda_graphs)
        self._graph_cap = int(graph_cache_size)
        if self._graph_cap < 1:
            raise ValueError("graph_cache_size must be at least 1")
        self._graphs: OrderedDict[int, torch.cuda.CUDAGraph] = OrderedDict()
        with torch.cuda.device(self.device):
            cc = torch.cuda.get_device_capability(self.device)
            if tuple(cc) != tuple(target.compute_capability):
                raise RuntimeError(
                    f"{target.name} target is validated on SM{target.compute_capability[0]}"
                    f"{target.compute_capability[1]}; {self.device} is "
                    f"SM{cc[0]}{cc[1]} ({torch.cuda.get_device_name(self.device)})")
            try:
                from flash_rt import flash_rt_kernels as fvk
            except ImportError as exc:
                raise RuntimeError(
                    f"flash_rt_kernels is not importable: {exc}. "
                    f"{target.build_hint}") from exc
            require_kernels(fvk, target)
            self._fvk = fvk
            self.weights, self.config = load_qwen3_4b_weights(
                device=self.device, checkpoint_path=checkpoint_path,
                state_dict=state_dict, config=config)
            self.dims = qwen3_4b_dims(self.config)
            if self.max_seq > self.dims.max_pos:
                raise ValueError(
                    f"max_seq {self.max_seq} exceeds max_position_embeddings "
                    f"{self.dims.max_pos}")
            self._alloc()
            self.rope_inv_freq = rope_inv_freq
            self._cos, self._sin = rope_half_tables(
                self.dims, self.max_seq, rope_inv_freq, self.device)
            self.attention = target.make_attention(
                self.dims, self.max_seq, self._q, self._k, self._v, self._o,
                self.device)
            self.pipeline = Qwen3Bf16Prefill(
                fvk, self.dims, self._layer_ptrs(), self._model_ptrs(),
                self._buf_ptrs(), self.attention, pdl=target.pdl,
                torch_reduce=torch_reduce_tree(), tuned_gemms=autotune_gemms)
            # Graphs are captured on a stream of this device; replay runs on
            # the caller's current stream.
            self._capture_stream = torch.cuda.Stream(device=self.device)

    # ── setup ──

    def _alloc(self) -> None:
        d, S = self.dims, self.max_seq
        e = dict(device=self.device, dtype=BF16)
        qd, kd = d.num_q_heads * d.head_dim, d.num_kv_heads * d.head_dim
        self._ids = torch.zeros(S, device=self.device, dtype=torch.int64)
        self._h = torch.empty(S, d.hidden, **e)
        self._xn = torch.empty(S, d.hidden, **e)
        self._t = torch.empty(S, d.hidden, **e)
        self._q = torch.empty(S * qd, **e)
        self._k = torch.empty(S * kd, **e)
        self._v = torch.empty(S * kd, **e)
        self._o = torch.empty(S * qd, **e)
        self._g = torch.empty(S, d.intermediate, **e)
        self._u = torch.empty(S, d.intermediate, **e)

    def _layer_ptrs(self) -> list[dict[str, int]]:
        return [{name: t.data_ptr() for name, t in layer.items()}
                for layer in self.weights.layers]

    def _model_ptrs(self) -> dict[str, int]:
        w = self.weights
        return {"embed": w.embed.data_ptr(), "final_norm": w.final_norm.data_ptr(),
                "lm_head": w.lm_head.data_ptr()}

    def _buf_ptrs(self) -> dict[str, int]:
        return {"ids": self._ids.data_ptr(), "h": self._h.data_ptr(),
                "xn": self._xn.data_ptr(), "t": self._t.data_ptr(),
                "q": self._q.data_ptr(), "k": self._k.data_ptr(),
                "v": self._v.data_ptr(), "o": self._o.data_ptr(),
                "g": self._g.data_ptr(), "u": self._u.data_ptr(),
                "cos": self._cos.data_ptr(), "sin": self._sin.data_ptr()}

    # ── inputs ──

    def _stage_ids(self, input_ids: Sequence[int] | torch.Tensor) -> int:
        """Copy the prompt into the persistent id buffer; return its length.

        A CUDA tensor must be on this instance's device; it is copied on the
        device without a host round-trip, so its token ids are not
        range-checked. A CPU tensor or a sequence of ints is checked against
        the vocabulary first.
        """
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.detach()
            if ids.dim() == 2 and ids.shape[0] == 1:
                ids = ids[0]
            if ids.dim() != 1:
                raise ValueError(
                    f"input_ids must be [S] or [1, S] (batch 1), got {tuple(input_ids.shape)}")
            if ids.dtype not in (torch.int64, torch.int32):
                raise TypeError(f"input_ids must be int64 or int32, got {ids.dtype}")
            if ids.is_cuda and ids.device != self.device:
                raise ValueError(f"input_ids on {ids.device}, runtime on {self.device}")
        else:
            ids = torch.as_tensor(list(input_ids), dtype=torch.int64)
            if ids.dim() != 1:
                raise ValueError("input_ids must be a flat sequence of token ids")
        S = int(ids.numel())
        if S == 0:
            raise ValueError("empty input_ids")
        if S > self.max_seq:
            raise ValueError(f"prompt length {S} > max_seq {self.max_seq}")
        if not ids.is_cuda:
            lo, hi = int(ids.min()), int(ids.max())
            if lo < 0 or hi >= self.dims.vocab_size:
                raise ValueError(
                    f"token id out of range [0, {self.dims.vocab_size}): {lo}..{hi}")
        self._ids[:S].copy_(ids)
        return S

    # ── execution ──

    def _run_stack(self, S: int, stream: int) -> None:
        self.pipeline.run(S, stream)

    def _stack(self, S: int) -> None:
        current = torch.cuda.current_stream(self.device)
        if not self.cuda_graphs:
            self._run_stack(S, current.cuda_stream)
            return
        graph = self._graphs.get(S)
        if graph is None:
            cs = self._capture_stream
            cs.wait_stream(current)
            # An eager pass on the capture stream first: it builds the GEMM
            # plans, the stream's GEMM workspace and any JIT kernels, none of
            # which may be created during capture.
            with torch.cuda.stream(cs):
                self._run_stack(S, cs.cuda_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=cs):
                self._run_stack(S, cs.cuda_stream)
            current.wait_stream(cs)
            self._graphs[S] = graph
            while len(self._graphs) > self._graph_cap:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(S)
        graph.replay()

    @torch.no_grad()
    def logits(self, input_ids: Sequence[int] | torch.Tensor,
               logits_to_keep: int) -> torch.Tensor:
        with torch.cuda.device(self.device):
            S = self._stage_ids(input_ids)
            keep = int(logits_to_keep)
            if keep < 0 or keep > S:
                raise ValueError(f"logits_to_keep must be in [0, {S}], got {keep}")
            rows = keep or S
            self._stack(S)
            # A fresh tensor per call; the LM head runs outside the graph so
            # the result never aliases a buffer the next call overwrites.
            out = torch.empty(rows, self.dims.vocab_size, device=self.device, dtype=BF16)
            self.pipeline.lm_head(S, rows, out.data_ptr(),
                                  torch.cuda.current_stream(self.device).cuda_stream)
            return out

    @torch.no_grad()
    def hidden(self, input_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        with torch.cuda.device(self.device):
            S = self._stage_ids(input_ids)
            self._stack(S)
            return self._xn[:S].clone()

    def clear_graphs(self) -> None:
        self._graphs.clear()


class Qwen3_4BBf16Frontend:
    """Qwen/Qwen3-4B BF16 prefill -> logits (batch 1), bound to ``TARGET``.

    Not constructed directly: each per-device subclass (the
    ``qwen3_4b_<hw>`` frontends) fixes a target.

    Args:
      checkpoint_path: HF checkpoint directory (config.json + safetensors).
        May be None when ``state_dict`` and ``config`` are given.
      device: CUDA device; ``"cuda"`` means the current device.
      max_seq: longest prompt; sizes every activation buffer.
      state_dict: in-memory HF state_dict to use instead of the checkpoint's.
      config: HF config (dict or ``Qwen3Config``); default config.json.
      rope_inv_freq: precision of RoPE ``inv_freq`` in the HF baseline this
        must match -- ``"fp32"`` for ``from_pretrained(dtype=torch.bfloat16)``,
        ``"bf16"`` for a model cast with ``.to(torch.bfloat16)``.
      cuda_graphs: replay one captured graph per prompt length.
      graph_cache_size: number of prompt lengths whose graphs are kept.
      autotune_gemms: ``False`` (default) runs every GEMM on cuBLASLt's
        heuristic first choice, the algorithm torch's matmul runs, so the
        GEMMs equal HF's bit for bit. ``True`` lets each shape run a faster
        candidate that matched the first choice on random probe operands at
        its first eager call -- faster where that choice is slow for a shape,
        but matching HF is then checked, not guaranteed.
    """

    TARGET: Qwen3Bf16Target | None = None

    def __init__(self, checkpoint_path: str | None = None, *,
                 device: str | int | torch.device = "cuda", max_seq: int = 2048,
                 state_dict: Mapping[str, torch.Tensor] | None = None,
                 config: Any = None, rope_inv_freq: str = "fp32",
                 cuda_graphs: bool = True, graph_cache_size: int = 16,
                 autotune_gemms: bool = False) -> None:
        if self.TARGET is None:
            raise TypeError(
                f"{type(self).__name__} has no TARGET; construct a per-device "
                "frontend (flash_rt.frontends.torch.qwen3_4b_<hw>)")
        self._rt = Qwen3_4BBf16Runtime(
            self.TARGET, checkpoint_path, device=device, max_seq=max_seq,
            state_dict=state_dict, config=config, rope_inv_freq=rope_inv_freq,
            cuda_graphs=cuda_graphs, graph_cache_size=graph_cache_size,
            autotune_gemms=autotune_gemms)

    @property
    def device(self) -> torch.device:
        return self._rt.device

    @property
    def dims(self) -> Qwen3_4BDims:
        return self._rt.dims

    @property
    def max_seq(self) -> int:
        return self._rt.max_seq

    def forward_prefill(self, input_ids: Sequence[int] | torch.Tensor, *,
                        logits_to_keep: int = 0) -> torch.Tensor:
        """Causal prefill -> logits ``[S, vocab]`` bf16 (a new tensor).

        ``logits_to_keep=k`` returns only the last ``k`` rows, computing the
        LM head for those rows alone, like HF's argument of the same name.
        """
        return self._rt.logits(input_ids, logits_to_keep)

    def forward_hidden(self, input_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Causal prefill -> final-normed hidden states ``[S, hidden]`` bf16."""
        return self._rt.hidden(input_ids)

    def clear_graphs(self) -> None:
        """Drop the captured per-length CUDA graphs."""
        self._rt.clear_graphs()
