"""FlashRT -- causal GQA attention bindings for the dense Qwen3 BF16 prefill.

Each binding attends the runtime's ``q [S, nq, hd]`` over ``k`` / ``v``
``[S, nkv, hd]`` into ``o [S, nq, hd]`` and owns whatever scratch its kernel
needs. A binding is built once per runtime; construction loads the backend
and runs it once, so a missing module, an absent instantiation or a build for
another architecture fails there instead of in the middle of a prefill.
Nothing falls back to another backend.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def _fa2_split_rows(num_q_heads: int, num_sms: int, max_seq: int) -> int:
    """Upper bound of ``num_splits * seqlen_q`` FA2's causal split-KV takes.

    ``fvk_attention_fa2_fwd_bf16_causal`` keeps one split while
    ``nq * ceil(S / 64) >= 0.8 * 2 * num_sms`` and never splits a key range
    into more than ``ceil(S / 128)`` pieces; the bound covers every S below
    that cutoff, so the accumulators fit whatever split count it picks.
    """
    rows = 0
    for s in range(1, max_seq + 1):
        if num_q_heads * math.ceil(s / 64) >= 0.8 * 2 * num_sms:
            break
        rows = max(rows, min(128, 2 * num_sms, math.ceil(s / 128)) * s)
    return rows


class Fa2CausalAttention:
    """``flash_rt_fa2.fwd_bf16_causal`` with its split-KV heuristic enabled.

    The split count follows FA2's wave-efficiency heuristic -- the one torch's
    flash SDPA uses -- which also keeps short prompts from leaving most SMs
    idle. The split partials are combined in a fixed order, so the result is
    deterministic.
    """

    name = "fa2"

    def __init__(self, dims: Any, max_seq: int, q: torch.Tensor,
                 k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                 device: torch.device) -> None:
        try:
            from flash_rt import flash_rt_fa2
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 BF16 prefill needs flash_rt_fa2 (FA2 causal, bf16, "
                "head_dim 128), which is not importable: "
                f"{exc}") from exc
        if not hasattr(flash_rt_fa2, "fwd_bf16_causal"):
            raise RuntimeError(
                "flash_rt_fa2 has no fwd_bf16_causal; rebuild with "
                "FA2_HDIMS including 128 and FA2_DTYPES including bf16")
        self._fwd = flash_rt_fa2.fwd_bf16_causal
        self._nq, self._nkv, self._hd = (
            dims.num_q_heads, dims.num_kv_heads, dims.head_dim)
        self._scale = dims.head_dim ** -0.5
        self._q, self._k, self._v, self._o = (
            q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr())
        self._num_sms = torch.cuda.get_device_properties(
            device).multi_processor_count
        f32 = dict(device=device, dtype=torch.float32)
        self._lse = torch.empty(self._nq * (math.ceil(max_seq / 128) * 128),
                                **f32)
        rows = _fa2_split_rows(self._nq, self._num_sms, max_seq)
        self._lse_accum = torch.empty(max(rows, 1) * self._nq, **f32)
        self._o_accum = torch.empty(max(rows, 1) * self._nq * self._hd, **f32)
        # Construction-time check that the instantiation exists and runs on
        # this device (a wrong-arch build fails here, not mid-prefill).
        self.run(1, torch.cuda.current_stream(device).cuda_stream)
        torch.cuda.synchronize(device)

    def run(self, S: int, stream: int) -> None:
        qd, kd, hd = self._nq * self._hd, self._nkv * self._hd, self._hd
        self._fwd(
            Q=self._q, K=self._k, V=self._v, O=self._o,
            softmax_lse=self._lse.data_ptr(),
            softmax_lse_accum=self._lse_accum.data_ptr(),
            o_accum=self._o_accum.data_ptr(),
            batch=1, seqlen_q=S, seqlen_k=S,
            num_heads_q=self._nq, num_heads_kv=self._nkv, head_dim=hd,
            q_strides=(S * qd, qd, hd), k_strides=(S * kd, kd, hd),
            v_strides=(S * kd, kd, hd), o_strides=(S * qd, qd, hd),
            softmax_scale=self._scale, num_sms=self._num_sms, stream=stream)


class Fa4CausalAttention:
    """FlashAttention-4 (CuTe-DSL) forward through ``hardware.thor.fa4_backend``.

    FA4 launches on torch's current stream, which is the stream the runtime
    runs the prefill on; ``run`` takes ``stream`` only for the common
    signature. GQA is packed by FA4 itself.
    """

    name = "fa4"

    def __init__(self, dims: Any, max_seq: int, q: torch.Tensor,
                 k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                 device: torch.device) -> None:
        from flash_rt.hardware.thor import fa4_backend
        if not fa4_backend.is_available():
            raise RuntimeError(
                "Qwen3 BF16 prefill on Thor needs the vendored FA4 forward "
                f"(nvidia-cutlass-dsl + quack-kernels): {fa4_backend.status()}")
        self._fwd = fa4_backend.fa4_fwd()
        self._nq, self._nkv, self._hd = (
            dims.num_q_heads, dims.num_kv_heads, dims.head_dim)
        self._scale = dims.head_dim ** -0.5
        self._q, self._k, self._v, self._o = q, k, v, o
        self._views: dict[int, tuple[torch.Tensor, ...]] = {}
        self.run(1, torch.cuda.current_stream(device).cuda_stream)
        torch.cuda.synchronize(device)

    def _view(self, S: int) -> tuple[torch.Tensor, ...]:
        views = self._views.get(S)
        if views is None:
            nq, nkv, hd = self._nq, self._nkv, self._hd
            views = (self._q[:S * nq * hd].view(1, S, nq, hd),
                     self._k[:S * nkv * hd].view(1, S, nkv, hd),
                     self._v[:S * nkv * hd].view(1, S, nkv, hd),
                     self._o[:S * nq * hd].view(1, S, nq, hd))
            self._views[S] = views
        return views

    def run(self, S: int, stream: int) -> None:
        q, k, v, o = self._view(S)
        self._fwd(q, k, v, softmax_scale=self._scale, causal=True, out=o)
