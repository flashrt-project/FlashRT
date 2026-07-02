"""FlashRT -- MiniMax-Remover kernel attention processor.

Replaces ``torch.nn.functional.scaled_dot_product_attention`` in every
attention block with a pointer-based kernel backend:

* ``FlashRTFA2Processor.__call__`` runs the QKV projection (through the
  NVFP4 Linears installed by ``_nvfp4_linear``), Q/K RMSNorm
  (fp32-stat Triton), interleaved RoPE on the native [B, S, H, D]
  layout (no transpose / copy), then the chosen attention kernel.

The attention backend is selected by ``FLASHRT_ATTN_MODE``:

  * ``sage`` / ``sage_auto``    -- SageAttention auto-dispatcher.
  * ``sage_fp8`` / ``sage2``    -- SageAttention QK-int8 + PV-fp8 CUDA
                                   (default; ~5x vs FA2, cos ~0.9993).
  * ``sage_fp16`` / ``sage1``   -- SageAttention QK-int8 + PV-fp16 CUDA.
  * ``sage_triton``             -- SageAttention QK-int8 + PV-fp16 Triton.
  * ``fa2``                     -- FlashRT FA2 (``flash_rt_fa2.fwd_fp16``).

``sageattention`` is an optional third-party dependency (pip); only the
selected backend is imported lazily. ``fa2`` uses FlashRT's own vendored
``flash_rt_fa2.so``. No MiniMax-Remover imports -- tensors only.
"""

import math
import os

import torch

from ._kernels import rms_norm_fp32stat, rope_apply_bshd, freqs_to_cos_sin

try:
    _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    _NUM_SMS = 0


def _attention_mode():
    return os.environ.get("FLASHRT_ATTN_MODE", "sage_fp8").lower()


_SAGE = None


def _get_sage():
    global _SAGE
    if _SAGE is None:
        try:
            import sageattention as _m
        except ImportError as e:
            raise RuntimeError(
                "FLASHRT_ATTN_MODE=sage_* requires the 'sageattention' package "
                "(the default attention backend). Install the model extra:\n"
                "    pip install -e \".[minimax-remover]\"\n"
                "or switch to the dependency-light FlashRT FA2 backend:\n"
                "    FLASHRT_ATTN_MODE=fa2") from e
        _SAGE = _m
    return _SAGE


def _sage_attn(q, k, v, scale, mode):
    """Dispatch to the requested SageAttention variant.

    All variants accept and return [B, S, H, D] (NHD) fp16.
    """
    sa = _get_sage()
    kw = dict(tensor_layout="NHD", is_causal=False, sm_scale=scale)
    if mode in ("sage", "sage_auto"):
        return sa.sageattn(q, k, v, **kw)
    if mode in ("sage_fp8", "sage2"):
        return sa.sageattn_qk_int8_pv_fp8_cuda(q, k, v, **kw)
    if mode in ("sage_fp16", "sage1"):
        return sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, **kw)
    if mode == "sage_triton":
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, **kw)
    return sa.sageattn(q, k, v, **kw)


def _fa2_attn(q, k, v, scale, lse_cache=None):
    """Run FlashRT's vendored FA2 backend on [B, S, H, D] fp16 tensors."""
    B, S, H, Dd = q.shape
    out = torch.empty_like(q)
    if lse_cache is None:
        lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    else:
        lse = lse_cache.get((B, S, H))
        if lse is None:
            lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
            lse_cache[(B, S, H)] = lse
    from flash_rt import flash_rt_fa2 as fa2
    qs, ks, vs, os_ = (q.stride(), k.stride(), v.stride(), out.stride())
    fa2.fwd_fp16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        lse.data_ptr(), 0, 0,
        B, S, S, H, H, Dd,
        (qs[0], qs[1], qs[2]), (ks[0], ks[1], ks[2]), (vs[0], vs[1], vs[2]),
        (os_[0], os_[1], os_[2]),
        scale, _NUM_SMS, torch.cuda.current_stream().cuda_stream)
    return out


def attention_forward(q, k, v, scale, mode=None, *, lse_cache=None):
    """Dispatch MiniMax-Remover attention without importing unused backends."""
    mode = _attention_mode() if mode is None else str(mode).lower()
    if mode.startswith("sage"):
        return _sage_attn(q, k, v, scale, mode)
    return _fa2_attn(q, k, v, scale, lse_cache)


class FlashRTFA2Processor:
    """Kernel attention processor for the native [B, S, H, D] layout.

    QKV projection goes through the installed NVFP4 Linears; Q/K
    RMSNorm uses the fp32-stat Triton kernel; RoPE is applied in place
    on the native layout (zero transpose / copy); the attention backend
    is chosen by ``FLASHRT_ATTN_MODE``. Per-shape cos/sin RoPE tables
    and FA2 log-sum-exp buffers are cached.
    """

    def __init__(self):
        self._lse_bufs = {}
        self._cos_sin = {}

    def __call__(self, attn, hidden_states, rotary_emb=None,
                 attention_mask=None, encoder_hidden_states=None):
        mode = _attention_mode()
        B, S, _ = hidden_states.shape
        H = attn.heads
        Dd = attn.inner_dim // H
        scale = 1.0 / math.sqrt(float(Dd))

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            q = rms_norm_fp32stat(q, attn.norm_q.weight, attn.norm_q.eps)
        if attn.norm_k is not None:
            k = rms_norm_fp32stat(k, attn.norm_k.weight, attn.norm_k.eps)

        q = q.view(B, S, H, Dd)
        k = k.view(B, S, H, Dd)
        v = v.view(B, S, H, Dd)

        if rotary_emb is not None:
            cs = self._cos_sin.get(S)
            if cs is None:
                cs = freqs_to_cos_sin(rotary_emb)
                self._cos_sin[S] = cs
            rope_apply_bshd(q, cs[0], cs[1])
            rope_apply_bshd(k, cs[0], cs[1])

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        out = attention_forward(q, k, v, scale, mode,
                                lse_cache=self._lse_bufs)

        hidden_states = out.view(B, S, H * Dd)
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states


def install_attention(transformer):
    """Install the kernel attention processor on every transformer block."""
    proc = FlashRTFA2Processor()
    n = 0
    for block in transformer.blocks:
        block.attn1.processor = proc
        n += 1
    return n
