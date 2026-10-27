"""FlashRT -- GR00T N1.5 torch frontend, RTX SM89 (RTX 4090), FP8 / BF16.

``config="groot_n15"``, ``framework="torch"``, ``hardware="rtx_sm89"``.

GR00T N1.5 (``GR00T_N1_5``: Eagle2.5 SigLIP + 12-layer Qwen3 backbone,
VL self-attention, 16-layer flow-matching DiT) runs as FlashRT kernels:
the stage sequence is :class:`flash_rt.models.groot_n15.GrootN15Forward`,
bound to this device by ``flash_rt.models.groot_n15.pipeline_rtx``. This
frontend owns what surrounds it:

* input validation and the upstream observation / action transforms
  (``_groot_n15_io``: Isaac-GR00T's data-config transform, unchanged);
* the checkpoint adapter (``_groot_n15_weights``);
* persistent buffers, the input-independent tables (timestep AdaLayerNorm
  modulation, the sinusoidal step encoding, position embeddings, future
  tokens, RoPE) built once with the reference's expressions;
* CUDA graphs: one per prompt length, captured on the first call of that
  length and replayed on the caller's current stream (LRU of
  ``graph_cache_size``).

Two tiers: FP8 (``use_fp8=True``, the default as for ``load_model``) and
BF16 (``use_fp8=False``, the reference's algorithms); see the target in
``pipeline_rtx`` for what each quantizes. Construction checks the device
generation, every kernel entry the forward and the tier launch and each
attention instantiation, and raises on a configuration it does not
implement. Nothing falls back to the upstream model.

Requirements: ``flash_rt_kernels``, ``flash_rt_fa2`` and ``flash_rt_groot_n15``
built for SM89 (``-DGPU_ARCH=89 -DFLASHRT_ENABLE_GROOT_N15=ON``), and the
upstream ``gr00t`` package for IO (``groot_root=`` or ``$GROOT_N15_ROOT``).

Usage (``load_model`` reads the data config and the source root from
``$GROOT_N15_DATA_CONFIG`` / ``$GROOT_N15_ROOT``)::

    import flash_rt
    model = flash_rt.load_model(
        "/path/to/gr00t-n1.5-checkpoint", config="groot_n15",
        hardware="rtx_sm89", num_views=3)
    actions = model.predict(
        images=[left, right, wrist],          # data-config video-key order
        prompt="open the microwave",
        state={"base_position": ..., ...})    # dict keyed by state name
"""
from __future__ import annotations

import dataclasses
import logging
import time
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np
import torch

from flash_rt.frontends.torch._groot_n15_io import GrootN15IO
from flash_rt.frontends.torch._groot_n15_weights import (
    fp8_per_channel,
    load_weights,
    parse_dims,
    read_config,
)
from flash_rt.frontends.torch._qwen3_4b_bf16_runtime import rope_half_tables
from flash_rt.models.groot_n15._linear import LinearWeights
from flash_rt.models.groot_n15.forward import (
    GrootN15Forward,
    GrootN15Target,
    require_kernels,
)
from flash_rt.models.groot_n15.pipeline_rtx import GROOT_N15_RTX_SM89
from flash_rt.models.groot_n15.tables import build_step_tables, table_pointers
from flash_rt.models.qwen3._bf16_attention import Fa2CausalAttention
from flash_rt.models.qwen3.prefill_bf16 import Qwen3_4BDims, torch_reduce_tree

logger = logging.getLogger(__name__)

BF16 = torch.bfloat16
F32 = torch.float32


def _resolve_device(device: str | int | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"GR00T N1.5 runs on a CUDA device, got {dev}")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


class GrootN15TorchFrontendRtxSm89:
    """GR00T N1.5 on RTX SM89, batch 1, BF16 or FP8.

    Args:
        checkpoint_dir: GR00T N1.5 checkpoint directory.
        num_views: Camera views per observation; must equal the data config's.
        embodiment_tag: Embodiment the checkpoint metadata and MLPs are read for.
        data_config: Isaac-GR00T ``DATA_CONFIG_MAP`` key (or
            ``$GROOT_N15_DATA_CONFIG``).
        groot_root: Isaac-GR00T source root providing ``gr00t`` (or
            ``$GROOT_N15_ROOT``); not needed when ``gr00t`` is importable.
        num_steps: Flow-matching steps (upstream ``denoising_steps``; default:
            the checkpoint's).
        action_horizon: When given, must equal the data config's (and the
            checkpoint's) action horizon; it is not a free parameter.
        max_prompt_tokens: Longest prompt (image tokens included) to size
            buffers for.
        cuda_graphs: Capture and replay one CUDA graph per prompt length.
        graph_cache_size: Graphs kept (least recently used evicted).
        device: CUDA device.
        use_fp8: The FP8 tier (default, like ``load_model``'s ``use_fp8``):
            the target's ``fp8_projections`` in E4M3 with per-channel weight
            and per-token activation scales, the rest BF16. ``False`` selects
            the BF16 tier, which runs the reference's algorithms.
    """

    target: GrootN15Target = GROOT_N15_RTX_SM89

    def __init__(self, checkpoint_dir: str, num_views: int = 3,
                 embodiment_tag: str = "new_embodiment", *,
                 data_config: str | None = None, groot_root: str | None = None,
                 num_steps: int | None = None, action_horizon: int | None = None,
                 max_prompt_tokens: int = 1024,
                 cuda_graphs: bool = True, graph_cache_size: int = 8,
                 device: str | int | torch.device = "cuda",
                 use_fp8: bool = True) -> None:
        self.device = _resolve_device(device)
        self.precision = "fp8" if use_fp8 else "bf16"
        self.cuda_graphs = bool(cuda_graphs)
        self._graph_cap = int(graph_cache_size)
        if self._graph_cap < 1:
            raise ValueError("graph_cache_size must be at least 1")
        self.max_tokens = int(max_prompt_tokens)
        self._graphs: OrderedDict[int, torch.cuda.CUDAGraph] = OrderedDict()
        self._prompt: str | None = None
        self.latency_records: list[float] = []
        target = self.target
        with torch.cuda.device(self.device):
            cc = torch.cuda.get_device_capability(self.device)
            if tuple(cc) != tuple(target.compute_capability):
                raise RuntimeError(
                    f"{target.name} target is validated on SM{target.compute_capability[0]}"
                    f"{target.compute_capability[1]}; {self.device} is SM{cc[0]}{cc[1]} "
                    f"({torch.cuda.get_device_name(self.device)})")
            self._fvk, self._g = self._kernel_modules()
            if self.precision not in target.precisions:
                raise ValueError(f"{target.name} has no {self.precision} tier; "
                                 f"available: {target.precisions}")
            require_kernels(self._fvk, self._g, target, self.precision)

            self.io = GrootN15IO(checkpoint_dir, data_config, embodiment_tag, groot_root)
            self.num_views = int(num_views)
            if self.num_views != len(self.io.video_keys):
                raise ValueError(f"num_views={num_views} but the data config has "
                                 f"{len(self.io.video_keys)} views {self.io.video_keys}")
            config = read_config(checkpoint_dir)
            dims = parse_dims(config, self.io.eagle_config(), self.io.action_horizon)
            if action_horizon is not None and int(action_horizon) != dims.action_horizon:
                raise ValueError(f"action_horizon={action_horizon}, but the data config and the "
                                 f"checkpoint use {dims.action_horizon}")
            if num_steps is not None:
                if int(num_steps) < 1:
                    raise ValueError("num_steps must be at least 1")
                dims = dataclasses.replace(dims, denoising_steps=int(num_steps))
            self.dims = dims
            self._image_rows = self.num_views * dims.patches_per_tile
            if self.max_tokens < self._image_rows + 1:
                raise ValueError(f"max_prompt_tokens={self.max_tokens} cannot hold "
                                 f"{self._image_rows} image tokens and a prompt")
            self._image_token = self.io.image_token_id()

            self.weights = load_weights(checkpoint_dir, dims, self.io.embodiment_id,
                                        self.device)
            self._alloc()
            self._build_tables()
            self.forward = self._bind()
            self._capture_stream = torch.cuda.Stream(device=self.device)
        logger.info("GR00T N1.5 %s: %d views, %d image tokens, embodiment %s (%d)",
                    target.name, self.num_views, self._image_rows, self.io.embodiment_tag,
                    self.io.embodiment_id)

    # ── setup ──

    def _kernel_modules(self) -> tuple[Any, Any]:
        """The kernel modules, checked before anything is loaded."""
        from flash_rt import _extensions

        mods = {}
        for name in ("flash_rt_kernels", "flash_rt_fa2", "flash_rt_groot_n15"):
            try:
                mods[name] = _extensions.require(name, config="groot_n15")
            except ImportError as exc:
                raise RuntimeError(f"GR00T N1.5 needs {name}: {exc}") from None
        fa2 = mods["flash_rt_fa2"]
        missing = [n for n in ("fwd_bf16", "fwd_bf16_causal") if not hasattr(fa2, n)]
        if missing:
            raise RuntimeError(f"flash_rt_fa2 has no {missing}; build it with FA2_DTYPES "
                               "including bf16 and FA2_HDIMS including 64, 96 and 128")
        return mods["flash_rt_kernels"], mods["flash_rt_groot_n15"]

    def _alloc(self) -> None:
        d, S, dev = self.dims, self.max_tokens, self.device
        e, f = dict(device=dev, dtype=BF16), dict(device=dev, dtype=F32)
        Mv, Dv, Fv = self._image_rows, d.vision_hidden, d.vision_intermediate
        T, De, Fd = d.dit_tokens, d.dit_hidden, d.dit_intermediate
        H, A, Hh = d.action_horizon, d.action_dim, d.head_hidden
        Dl, Fl = d.vl_hidden, d.vl_intermediate
        z = torch.empty
        b: dict[str, torch.Tensor] = {
            "pixels": z(self.num_views, 3, d.image_size, d.image_size, **e),
            "v_cols": z(Mv, d.patch_features, **e),
            "v_t": z(Mv, Dv, **e), "v_h": z(Mv, Dv, **e), "v_xn": z(Mv, Dv, **e),
            "v_q": z(Mv, Dv, **e), "v_k": z(Mv, Dv, **e), "v_v": z(Mv, Dv, **e),
            "v_o": z(Mv, Dv, **e), "v_f": z(Mv, Fv, **e),
            "l_h32": z(S, Dl, **f), "l_xn": z(S, Dl, **e),
            "l_q": z(S, Dl, **e), "l_k": z(S, Dl, **e), "l_v": z(S, Dl, **e),
            "l_o": z(S, Dl, **e), "l_t": z(S, Dl, **e), "l_f": z(S, Fl, **e),
            "vl": z(S, Dl, **e),
            "state": z(1, d.state_dim, **e), "s_t1": z(1, Hh, **e), "s_h": z(1, Hh, **e),
            "s_t2": z(1, De, **e), "sa": z(T, De, **e),
            "noise": z(H, A, **f), "actions": z(H, A, **f), "a_bf16": z(H, A, **e),
            "a_t1": z(H, De, **e), "a_cat": z(H, 2 * De, **e), "a_t2": z(H, De, **e),
            "a_h": z(H, De, **e), "a_t3": z(H, De, **e),
            "d_xn": z(T, De, **e), "d_q": z(T, De, **e), "d_k": z(T, De, **e),
            "d_v": z(T, De, **e), "d_o": z(T, De, **e), "d_t": z(T, De, **e),
            "d_h": z(T, De, **e), "d_f": z(T, Fd, **e), "d_out": z(T, d.dit_output, **e),
            "p_t1": z(T, Hh, **e), "p_h": z(T, Hh, **e), "p_t2": z(T, A, **e),
            "pred": z(T, A, **e),
        }
        # Language model: the token table followed by one row per image token,
        # which mlp1 writes; image positions of the prompt index those rows.
        H_l, I_l = d.llm_hidden, d.llm_intermediate
        qd, kd = d.llm_q_heads * d.llm_head_dim, d.llm_kv_heads * d.llm_head_dim
        table = z(d.llm_vocab + Mv, H_l, **e)
        table[:d.llm_vocab].copy_(self.weights.llm_embed)
        self.weights.llm_embed = None
        b["embed_table"] = table
        b["ids"] = torch.zeros(S, device=dev, dtype=torch.int64)
        b["features"] = z(S, H_l, **e)
        for name, width in (("h", H_l), ("t", H_l), ("q", qd), ("o", qd),
                            ("k", kd), ("v", kd), ("g", I_l), ("u", I_l)):
            b["lm_" + name] = z(S, width, **e)
        if self.precision == "fp8":
            # One quantized activation is live at a time; the widest is the
            # VL feed-forward input, or the vision fc2 input for short prompts.
            self._x8_bytes = max(S * max(Fl, I_l, H_l), Mv * max(Fv, Dv))
            b["x8"] = torch.empty(self._x8_bytes, device=dev, dtype=torch.uint8)
            b["x8_rows"] = torch.zeros(max(S, Mv), **f)
            b["one"] = torch.ones(1, **f)
        self._kv_cache = [
            (z(S, De, **e), z(S, De, **e)) if lw["cross"] else None
            for lw in self.weights.dit_layers]
        self._b = b

    @torch.no_grad()
    def _build_tables(self) -> None:
        """Input-independent tables (``models/groot_n15/tables.py``) and the
        persistent rows they seed."""
        d, w, dev, b = self.dims, self.weights, self.device, self._b
        self._steps = build_step_tables(d, w.head["t_lin1"], w.head["t_lin2"],
                                        [x["ada"] for x in w.dit_layers], w.head["proj_out1"],
                                        dev)
        b["sa"][1:1 + d.future_tokens].copy_(w.head["future"])
        # The dense Qwen3 BF16 dims contract at Eagle's geometry; the token
        # table carries the image rows after the vocabulary.
        self._lm_dims = Qwen3_4BDims(
            hidden=d.llm_hidden, num_layers=d.llm_layers,
            vocab_size=d.llm_vocab + self._image_rows, intermediate=d.llm_intermediate,
            num_q_heads=d.llm_q_heads, num_kv_heads=d.llm_kv_heads,
            head_dim=d.llm_head_dim, rope_theta=d.rope_theta, rms_norm_eps=d.rms_norm_eps)
        # Eagle builds the language model inside a bf16 from_pretrained, which
        # leaves RoPE's inv_freq buffer in fp32.
        self._cos, self._sin = rope_half_tables(self._lm_dims, self.max_tokens, "fp32", dev)

    def _bind(self) -> GrootN15Forward:
        d, w, b, dev, tgt = self.dims, self.weights, self._b, self.device, self.target
        p = lambda t: t.data_ptr()  # noqa: E731
        fp8 = self.precision == "fp8"
        self._fp8_weights: list[torch.Tensor] = []

        def lw(pair: tuple[torch.Tensor, torch.Tensor | None], *,
               quantize: bool = False) -> LinearWeights:
            W, bias = pair
            n, k = int(W.shape[0]), int(W.shape[1])
            bp = p(bias) if bias is not None else 0
            if not (fp8 and quantize):
                return LinearWeights(p(W), bp, n, k)
            W8, scale = fp8_per_channel(W)
            self._fp8_weights += [W8, scale]
            return LinearWeights(0, bp, n, k, p(W8), p(scale))

        def nw(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[int, int]:
            return p(pair[0]), p(pair[1])

        scope = tgt.fp8_projections
        vis, vl = dict(quantize="vision" in scope), dict(quantize="vl" in scope)
        attn, mlp = dict(quantize="llm_attention" in scope), dict(quantize="llm_mlp" in scope)
        weights = {
            "vision": {"patch": lw(w.vision["patch"]), "position": p(w.vision["position"]),
                       "post_ln": nw(w.vision["post_ln"]), "mlp1": lw(w.vision["mlp1"], **vis)},
            "vision_layers": [
                {"ln1": nw(x["ln1"]), "ln2": nw(x["ln2"]),
                 **{k: lw(x[k], **vis) for k in ("q", "k", "v", "o", "fc1", "fc2")}}
                for x in w.vision_layers],
            "llm": {"embed": p(b["embed_table"]), "final_norm": p(w.llm_final_norm)},
            "llm_layers": [
                {**{k: p(x[k]) for k in ("in_norm", "post_norm", "q_norm", "k_norm")},
                 **{k: lw((x[k], None), **attn) for k in ("q", "k", "v", "o")},
                 **{k: lw((x[k], None), **mlp) for k in ("gate", "up", "down")}}
                for x in w.llm_layers],
            "vl": {"vlln": nw(w.vl["vlln"])},
            "vl_layers": [
                {"norm1": nw(x["norm1"]), "norm3": nw(x["norm3"]),
                 **{k: lw(x[k], **vl) for k in ("q", "k", "v", "o", "ff1", "ff2")}}
                for x in w.vl_layers],
            "dit_layers": [
                {"cross": x["cross"],
                 **{k: lw(x[k]) for k in ("q", "o", "ff1", "ff2")},
                 **{k: lw(x[k], quantize=x["cross"] and "cross_kv" in scope) for k in ("k", "v")},
                 "k_cache": p(kv[0]) if kv else 0, "v_cache": p(kv[1]) if kv else 0}
                for x, kv in zip(w.dit_layers, self._kv_cache)],
            "head": {**{k: (p(w.head[k][0]), p(w.head[k][1]))
                        for k in ("state1", "state2", "act1", "act2", "act3", "dec1", "dec2")},
                     "proj_out": lw(w.head["proj_out"])},
        }
        if fp8:
            # Packed projections no longer read their bf16 weights.
            self._drop_packed_bf16(scope)
        tables = table_pointers(self._steps, w.head["position"])
        bufs = {k: p(v) for k, v in b.items()}
        bufs["image_rows"] = p(b["embed_table"]) + d.llm_vocab * d.llm_hidden * 2
        bufs["cos"], bufs["sin"] = p(self._cos), p(self._sin)

        site = tgt.make_attention
        first_cross = next(kv for kv in self._kv_cache if kv)
        attention = {
            "vision": site("vision", q=p(b["v_q"]), k=p(b["v_k"]), v=p(b["v_v"]),
                           o=p(b["v_o"]), batch=self.num_views, heads=d.vision_heads,
                           head_dim=d.vision_head_dim, max_q=d.patches_per_tile,
                           max_k=d.patches_per_tile, device=dev),
            "language": Fa2CausalAttention(self._lm_dims, self.max_tokens, b["lm_q"], b["lm_k"],
                                           b["lm_v"], b["lm_o"], dev),
            "vl": site("vl", q=p(b["l_q"]), k=p(b["l_k"]), v=p(b["l_v"]), o=p(b["l_o"]),
                       batch=1, heads=d.vl_heads, head_dim=d.vl_head_dim,
                       max_q=self.max_tokens, max_k=self.max_tokens, device=dev),
            "dit_cross": site("dit_cross", q=p(b["d_q"]), k=p(first_cross[0]),
                              v=p(first_cross[1]), o=p(b["d_o"]), batch=1,
                              heads=d.dit_heads, head_dim=d.dit_head_dim,
                              max_q=d.dit_tokens, max_k=self.max_tokens, device=dev),
            "dit_self": site("dit_self", q=p(b["d_q"]), k=p(b["d_k"]), v=p(b["d_v"]),
                             o=p(b["d_o"]), batch=1, heads=d.dit_heads,
                             head_dim=d.dit_head_dim, max_q=d.dit_tokens,
                             max_k=d.dit_tokens, device=dev),
        }
        scratch = (dict(x8=p(b["x8"]), x8_bytes=self._x8_bytes, row_scale=p(b["x8_rows"]),
                        rows=int(b["x8_rows"].numel()), one=p(b["one"])) if fp8 else {})
        linear = tgt.make_linear(self.precision, self._g, self._fvk, **scratch)
        return GrootN15Forward(self._g, self._fvk, d, self.num_views, weights, tables, bufs,
                               attention, linear, pdl=tgt.pdl, torch_reduce=torch_reduce_tree())

    def _drop_packed_bf16(self, scope: frozenset[str]) -> None:
        w = self.weights
        groups = [("vision", w.vision_layers, ("q", "k", "v", "o", "fc1", "fc2")),
                  ("vl", w.vl_layers, ("q", "k", "v", "o", "ff1", "ff2"))]
        for name, layers, keys in groups:
            if name in scope:
                for x in layers:
                    for k in keys:
                        x[k] = (None, x[k][1])
        if "vision" in scope:
            w.vision["mlp1"] = (None, w.vision["mlp1"][1])
        for name, keys in (("llm_attention", ("q", "k", "v", "o")),
                           ("llm_mlp", ("gate", "up", "down"))):
            if name in scope:
                for x in w.llm_layers:
                    for k in keys:
                        x[k] = None
        if "cross_kv" in scope:
            for x in w.dit_layers:
                if x["cross"]:
                    x["k"], x["v"] = (None, x["k"][1]), (None, x["v"][1])
        torch.cuda.empty_cache()

    # ── inputs ──

    def _stage(self, normalized: Mapping[str, Any], noise: torch.Tensor | None) -> int:
        """Copy one normalized observation into the persistent buffers; return
        the prompt length."""
        d, b = self.dims, self._b
        ids = torch.as_tensor(normalized["eagle_input_ids"])
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"expected eagle_input_ids [1, S], got {tuple(ids.shape)}")
        mask = normalized.get("eagle_attention_mask")
        if mask is not None and not bool(torch.as_tensor(mask).bool().all()):
            raise ValueError("padded prompts are not supported (batch 1)")
        ids = ids[0].to("cpu", torch.int64)
        S = int(ids.numel())
        if S > self.max_tokens:
            raise ValueError(f"prompt of {S} tokens exceeds max_prompt_tokens={self.max_tokens}")
        image = ids == self._image_token
        n_img = int(image.sum())
        if n_img != self._image_rows:
            raise ValueError(f"prompt has {n_img} image tokens, expected "
                             f"{self._image_rows} ({self.num_views} views)")
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= d.llm_vocab:
            raise ValueError(f"token id out of range [0, {d.llm_vocab}): {lo}..{hi}")
        staged = ids.clone()
        staged[image] = d.llm_vocab + torch.arange(n_img, dtype=torch.int64)
        pixels = torch.as_tensor(normalized["eagle_pixel_values"])
        if tuple(pixels.shape) != tuple(b["pixels"].shape):
            raise ValueError(f"expected eagle_pixel_values {tuple(b['pixels'].shape)} (one "
                             f"tile per view), got {tuple(pixels.shape)}")
        state = torch.as_tensor(normalized["state"])
        if state.numel() != d.state_dim:
            raise ValueError(f"expected a {d.state_dim}-wide state, got {tuple(state.shape)}")
        emb = int(torch.as_tensor(normalized["embodiment_id"]).reshape(-1)[0])
        if emb != self.io.embodiment_id:
            raise ValueError(f"observation is for embodiment {emb}, this frontend serves "
                             f"{self.io.embodiment_id}")
        b["ids"][:S].copy_(staged)
        b["pixels"].copy_(pixels.to(self.device, dtype=BF16))
        b["state"].copy_(state.to(self.device, dtype=BF16).reshape(1, d.state_dim))
        if noise is None:
            noise = torch.randn((1, d.action_horizon, d.action_dim), dtype=F32,
                                device=self.device)
        noise = torch.as_tensor(noise)
        if noise.numel() != d.action_horizon * d.action_dim:
            raise ValueError(f"noise must hold {d.action_horizon}x{d.action_dim} values, "
                             f"got {tuple(noise.shape)}")
        b["noise"].copy_(noise.to(self.device, F32).reshape(d.action_horizon, d.action_dim))
        return S

    # ── execution ──

    def _run(self, S: int) -> None:
        current = torch.cuda.current_stream(self.device)
        if not self.cuda_graphs:
            self.forward.run(S, current.cuda_stream)
            return
        graph = self._graphs.get(S)
        if graph is None:
            cs = self._capture_stream
            cs.wait_stream(current)
            # An eager pass on the capture stream first: it settles the GEMM
            # plans and the stream's GEMM workspaces, which may not be created
            # during capture.
            with torch.cuda.stream(cs):
                self.forward.run(S, cs.cuda_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=cs):
                self.forward.run(S, cs.cuda_stream)
            current.wait_stream(cs)
            self._graphs[S] = graph
            while len(self._graphs) > self._graph_cap:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(S)
        graph.replay()

    @torch.no_grad()
    def predict_normalized(self, normalized: Mapping[str, Any],
                           noise: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized model inputs (the upstream transform's output) ->
        normalized actions ``[1, H, action_dim]`` fp32 on the device.

        ``noise`` is the initial action noise ``[1, H, action_dim]``; it is
        drawn with ``torch.randn`` on the device when omitted.
        """
        with torch.cuda.device(self.device):
            S = self._stage(normalized, noise)
            self._run(S)
            return self._b["actions"].reshape(1, self.dims.action_horizon,
                                              self.dims.action_dim).clone()

    # ── public API ──

    def set_prompt(self, prompt: str) -> None:
        """Set the language instruction used by subsequent ``infer`` calls."""
        self._prompt = str(prompt)

    def infer(self, obs: Mapping[str, Any], noise: torch.Tensor | None = None) -> dict:
        """Run one policy call.

        Args:
            obs: Either data-config keys (``video.*`` uint8 ``[H, W, 3]``,
                ``state.*`` vectors), or ``images`` (a list in the data
                config's video-key order) plus ``state`` (a dict keyed by
                ``state.*`` or bare state names).
            noise: Optional initial action noise ``[1, H, action_dim]``.

        Returns:
            ``{"actions": {action_key: np.ndarray [H, dim]}}``, the
            unnormalized per-key action dict.
        """
        if self._prompt is None:
            raise RuntimeError("set_prompt() must be called before infer()")
        t0 = time.perf_counter()
        normalized = self.io.prepare(self._observation(obs))
        actions = self.predict_normalized(normalized, noise)
        out = self.io.unnormalize(actions.cpu())
        self.latency_records.append((time.perf_counter() - t0) * 1e3)
        return {"actions": out}

    def clear_graphs(self) -> None:
        """Drop every captured graph (they are recaptured on demand)."""
        self._graphs.clear()

    @property
    def graph_lengths(self) -> tuple[int, ...]:
        """Prompt lengths with a captured graph, least recently used first."""
        return tuple(self._graphs)

    def _observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        images = obs.get("images")
        for i, key in enumerate(self.io.video_keys):
            if key in obs:
                frame = obs[key]
            elif images is not None and i < len(images):
                frame = images[i]
            else:
                raise ValueError(f"missing camera frame for {key!r}")
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"{key}: expected a uint8 [H, W, 3] frame, got "
                                 f"{frame.dtype} {frame.shape}")
            out[key] = frame[None]
        state = obs.get("state")
        for key in self.io.state_keys:
            bare = key.removeprefix("state.")
            if key in obs:
                value = obs[key]
            elif isinstance(state, Mapping) and key in state:
                value = state[key]
            elif isinstance(state, Mapping) and bare in state:
                value = state[bare]
            else:
                raise ValueError(f"missing state {key!r}: pass obs['state'] as a dict keyed by "
                                 f"{self.io.state_keys}")
            value = np.asarray(value)
            if not np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float64)
            out[key] = value.reshape(1, -1)
        out[self.io.language_key] = [self._prompt]
        return out
