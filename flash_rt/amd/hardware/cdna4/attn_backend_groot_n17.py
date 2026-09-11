"""FlashRT AMD — CDNA4 attention backend for GROOT N1.7 (all 5 sites).

One backend serves the whole model on AMD:

  * ``vit``          multi-view batched ViT self-attention (16h x 64d)
  * ``llm``          truncated-LLM causal self-attention (16q/8kv x 128d)
  * ``vl_self_attn`` VL adapter self-attention (32h x 64d)
  * ``dit_self``     DiT action self-attention (32h x 48d, Sa tokens)
  * ``dit_cross``    DiT cross-attention (32h x 48d Q, per-block K/V)

Protocol mirrors the RTX pair (``RtxGrootN17BackboneAttn`` +
``RtxFlashAttnBackendGrootN17``): pre-allocated layer-shared Q/K/V/O
slots, ``get_slot_ptrs(site, layer)`` for the pipeline's raw-pointer
writes, ``run(site, layer, q_seq, kv_seq=..., stream=...)`` returning a
stable O pointer.

DISPATCH
--------
Every site runs on aiter ``mha_fwd`` (CK-tile asm flash attention).
The isolated-shape survey on the exact N1.7 geometries measured aiter
faster than torch sdpa at every site in both dtypes (1.19-1.83x, the
GQA-causal llm site largest) with clean parity, and confirmed head_dim
48 and native-GQA causal support. ``FVK_AMD_ATTN=sdpa`` is the escape
hatch (torch sdpa fallback, GQA expanded on the fly).

AMD-native GQA: unlike the RTX backend — whose cuBLAS-decomposed llm
kernel needs K/V pre-expanded to 16 heads by the forward — the llm K/V
slots here keep the model's 8 KV heads and aiter consumes them
natively, halving llm K/V slot traffic. The AMD pipeline forward must
therefore SKIP the ``gpu_repeat_interleave_heads`` step.

Capture safety follows the pi05 aiter backend's contract: aiter may
allocate LSE/workspace through torch's caching allocator per call, so
the pipeline must run warmup iterations on the capture stream before
``hipStreamBeginCapture`` (allocator steady state), and all calls land
on torch's current stream.
"""

from __future__ import annotations

import math
import os

# Fixed N1.7 attention geometry.
_VIT_NH, _VIT_HD = 16, 64
_LLM_NHQ, _LLM_NHKV, _LLM_HD = 16, 8, 128
_VLSA_NH, _VLSA_HD = 32, 64
_DIT_NH, _DIT_HD = 32, 48


def _resolve_mha_fwd():
    try:
        import aiter  # noqa: F401
    except ImportError as ex:
        raise ImportError(
            "Cdna4GrootN17AttnBackend requires the 'aiter' package "
            "(AMD asm flash attention); import failed — install aiter "
            "or use the sdpa fallback (FVK_AMD_ATTN=sdpa).") from ex
    fn = getattr(aiter, "mha_fwd", None)
    if fn is None:
        try:
            from aiter.ops.mha import mha_fwd as fn  # type: ignore
        except ImportError as ex:
            raise ImportError(
                "aiter imported but exposes no 'mha_fwd' — version "
                "mismatch; use FVK_AMD_ATTN=sdpa.") from ex
    return fn


class Cdna4GrootN17AttnBackend:
    """aiter-backed attention slots for GROOT N1.7 on AMD CDNA4."""

    SITES = ("vit", "llm", "vl_self_attn", "dit_self", "dit_cross")

    def __init__(
        self,
        *,
        num_vit_views: int,
        vit_seq: int,
        llm_seq: int,
        vl_self_attn_seq: int,
        sa: int,
        dit_kv_seq: int,
        num_dit_cross_blocks: int = 16,
        device: str = "cuda",
        backbone_dtype=None,
        dit_dtype=None,
    ):
        import torch

        self._torch = torch
        self._device = device
        bdt = backbone_dtype if backbone_dtype is not None else torch.float16
        ddt = dit_dtype if dit_dtype is not None else torch.bfloat16

        self._nv = int(num_vit_views)
        self._vit_seq = int(vit_seq)
        self._llm_seq = int(llm_seq)
        self._vlsa_seq = int(vl_self_attn_seq)
        self._sa = int(sa)
        self._dit_kv_seq = int(dit_kv_seq)
        self._num_dit_cross_blocks = int(num_dit_cross_blocks)
        if self._vit_seq % self._nv != 0:
            raise ValueError(
                f"vit_seq={self._vit_seq} not divisible by "
                f"num_vit_views={self._nv}")
        self._vit_per = self._vit_seq // self._nv
        if self._sa <= 0 or self._dit_kv_seq <= 0:
            raise ValueError("sa and dit_kv_seq must be positive")

        def slot(S, NH, HD, dt):
            return torch.empty(S, NH, HD, dtype=dt, device=device)

        # Backbone slots (layer-shared).
        self.vit_Q = slot(self._vit_seq, _VIT_NH, _VIT_HD, bdt)
        self.vit_K = slot(self._vit_seq, _VIT_NH, _VIT_HD, bdt)
        self.vit_V = slot(self._vit_seq, _VIT_NH, _VIT_HD, bdt)
        self.vit_O = slot(self._vit_seq, _VIT_NH, _VIT_HD, bdt)

        # llm K/V keep the native 8 KV heads (see module docstring).
        self.llm_Q = slot(self._llm_seq, _LLM_NHQ, _LLM_HD, bdt)
        self.llm_K = slot(self._llm_seq, _LLM_NHKV, _LLM_HD, bdt)
        self.llm_V = slot(self._llm_seq, _LLM_NHKV, _LLM_HD, bdt)
        self.llm_O = slot(self._llm_seq, _LLM_NHQ, _LLM_HD, bdt)

        self.vlsa_Q = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD, bdt)
        self.vlsa_K = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD, bdt)
        self.vlsa_V = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD, bdt)
        self.vlsa_O = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD, bdt)

        # DiT slots. Cross K/V are per-block (precomputed once per prompt
        # and read by all denoise steps), self K/V layer-shared.
        self.dit_self_Q = slot(self._sa, _DIT_NH, _DIT_HD, ddt)
        self.dit_self_K = torch.empty_like(self.dit_self_Q)
        self.dit_self_V = torch.empty_like(self.dit_self_Q)
        self.dit_self_O = torch.empty_like(self.dit_self_Q)

        self.dit_cross_Q = slot(self._sa, _DIT_NH, _DIT_HD, ddt)
        self.dit_cross_O = torch.empty_like(self.dit_cross_Q)
        self.dit_cross_K = [
            slot(self._dit_kv_seq, _DIT_NH, _DIT_HD, ddt)
            for _ in range(self._num_dit_cross_blocks)]
        self.dit_cross_V = [
            slot(self._dit_kv_seq, _DIT_NH, _DIT_HD, ddt)
            for _ in range(self._num_dit_cross_blocks)]

        self._use_sdpa = (
            os.environ.get("FVK_AMD_ATTN", "aiter").strip().lower() == "sdpa")
        self._mha_fwd = None if self._use_sdpa else _resolve_mha_fwd()

    # ── Protocol surface ──

    def sites(self) -> tuple:
        return self.SITES

    def get_ptrs(self) -> dict:
        return {
            "vit_Q": self.vit_Q.data_ptr(), "vit_K": self.vit_K.data_ptr(),
            "vit_V": self.vit_V.data_ptr(), "vit_O": self.vit_O.data_ptr(),
            "llm_Q": self.llm_Q.data_ptr(), "llm_K": self.llm_K.data_ptr(),
            "llm_V": self.llm_V.data_ptr(), "llm_O": self.llm_O.data_ptr(),
            "vlsa_Q": self.vlsa_Q.data_ptr(), "vlsa_K": self.vlsa_K.data_ptr(),
            "vlsa_V": self.vlsa_V.data_ptr(), "vlsa_O": self.vlsa_O.data_ptr(),
            "dit_self_Q": self.dit_self_Q.data_ptr(),
            "dit_self_K": self.dit_self_K.data_ptr(),
            "dit_self_V": self.dit_self_V.data_ptr(),
            "dit_self_O": self.dit_self_O.data_ptr(),
            "dit_cross_Q": self.dit_cross_Q.data_ptr(),
            "dit_cross_O": self.dit_cross_O.data_ptr(),
            "dit_cross_K": [t.data_ptr() for t in self.dit_cross_K],
            "dit_cross_V": [t.data_ptr() for t in self.dit_cross_V],
        }

    def get_slot_ptrs(self, site: str, layer_idx: int = 0) -> dict:
        if site == "vit":
            return {"Q": self.vit_Q.data_ptr(), "K": self.vit_K.data_ptr(),
                    "V": self.vit_V.data_ptr(), "O": self.vit_O.data_ptr()}
        if site == "llm":
            return {"Q": self.llm_Q.data_ptr(), "K": self.llm_K.data_ptr(),
                    "V": self.llm_V.data_ptr(), "O": self.llm_O.data_ptr()}
        if site == "vl_self_attn":
            return {"Q": self.vlsa_Q.data_ptr(), "K": self.vlsa_K.data_ptr(),
                    "V": self.vlsa_V.data_ptr(), "O": self.vlsa_O.data_ptr()}
        if site == "dit_self":
            self._check_layer(site, layer_idx, 16)
            return {"Q": self.dit_self_Q.data_ptr(),
                    "K": self.dit_self_K.data_ptr(),
                    "V": self.dit_self_V.data_ptr(),
                    "O": self.dit_self_O.data_ptr()}
        if site == "dit_cross":
            self._check_layer(site, layer_idx, self._num_dit_cross_blocks)
            return {"Q": self.dit_cross_Q.data_ptr(),
                    "K": self.dit_cross_K[layer_idx].data_ptr(),
                    "V": self.dit_cross_V[layer_idx].data_ptr(),
                    "O": self.dit_cross_O.data_ptr()}
        raise KeyError(f"unknown site {site!r}; known: {self.SITES}")

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq=None, stream: int = 0) -> int:
        if kv_seq is None:
            kv_seq = q_seq

        if site == "vit":
            if int(q_seq) != self._vit_per or int(kv_seq) != self._vit_per:
                raise ValueError(
                    f"vit q_seq/kv_seq must equal per-view len {self._vit_per}")
            nv, per = self._nv, self._vit_per
            q = self.vit_Q.view(nv, per, _VIT_NH, _VIT_HD)
            k = self.vit_K.view(nv, per, _VIT_NH, _VIT_HD)
            v = self.vit_V.view(nv, per, _VIT_NH, _VIT_HD)
            o = self.vit_O.view(nv, per, _VIT_NH, _VIT_HD)
            return self._attn(q, k, v, o, _VIT_HD, causal=False)

        if site == "llm":
            S = int(q_seq)
            self._check_seq("llm", S, self._llm_seq)
            if int(kv_seq) != S:
                raise ValueError("llm is self-attention; kv_seq must equal q_seq")
            q = self.llm_Q[:S].unsqueeze(0)
            k = self.llm_K[:S].unsqueeze(0)   # 8 KV heads, native GQA
            v = self.llm_V[:S].unsqueeze(0)
            o = self.llm_O[:S].unsqueeze(0)
            return self._attn(q, k, v, o, _LLM_HD, causal=True)

        if site == "vl_self_attn":
            S = int(q_seq)
            self._check_seq("vl_self_attn", S, self._vlsa_seq)
            q = self.vlsa_Q[:S].unsqueeze(0)
            k = self.vlsa_K[:S].unsqueeze(0)
            v = self.vlsa_V[:S].unsqueeze(0)
            o = self.vlsa_O[:S].unsqueeze(0)
            return self._attn(q, k, v, o, _VLSA_HD, causal=False)

        if site == "dit_self":
            self._check_layer(site, layer_idx, 16)
            self._check_seq("dit_self q_seq", q_seq, self._sa)
            if int(kv_seq) != int(q_seq):
                raise ValueError(
                    "dit_self is self-attention; kv_seq must equal q_seq")
            q = self.dit_self_Q[:q_seq].unsqueeze(0)
            k = self.dit_self_K[:q_seq].unsqueeze(0)
            v = self.dit_self_V[:q_seq].unsqueeze(0)
            o = self.dit_self_O[:q_seq].unsqueeze(0)
            return self._attn(q, k, v, o, _DIT_HD, causal=False)

        if site == "dit_cross":
            self._check_layer(site, layer_idx, self._num_dit_cross_blocks)
            self._check_seq("dit_cross q_seq", q_seq, self._sa)
            self._check_seq("dit_cross kv_seq", kv_seq, self._dit_kv_seq)
            q = self.dit_cross_Q[:q_seq].unsqueeze(0)
            k = self.dit_cross_K[layer_idx][:kv_seq].unsqueeze(0)
            v = self.dit_cross_V[layer_idx][:kv_seq].unsqueeze(0)
            o = self.dit_cross_O[:q_seq].unsqueeze(0)
            return self._attn(q, k, v, o, _DIT_HD, causal=False)

        raise KeyError(f"unknown site {site!r}; known: {self.SITES}")

    # ── Dispatch ──

    def _attn(self, q, k, v, o, hd, *, causal: bool) -> int:
        scale = 1.0 / math.sqrt(hd)
        if self._mha_fwd is not None:
            self._mha_fwd(
                q, k, v,
                0.0,        # dropout_p
                scale,      # softmax_scale
                causal,
                -1, -1,     # no local window
                False,      # return_softmax_lse
                False,      # return_dropout_randval
                out=o,
            )
            return o.data_ptr()
        # sdpa fallback: (B, S, H, D) -> (B, H, S, D), GQA expanded.
        torch = self._torch
        qt = q.transpose(1, 2)
        kt = k.transpose(1, 2)
        vt = v.transpose(1, 2)
        if kt.shape[1] != qt.shape[1]:
            rep = qt.shape[1] // kt.shape[1]
            kt = kt.repeat_interleave(rep, dim=1)
            vt = vt.repeat_interleave(rep, dim=1)
        res = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, is_causal=causal, scale=scale)
        o.copy_(res.transpose(1, 2))
        return o.data_ptr()

    @staticmethod
    def _check_layer(site: str, layer_idx: int, num_layers: int) -> None:
        if not (0 <= int(layer_idx) < int(num_layers)):
            raise IndexError(
                f"{site} layer_idx={layer_idx} out of range [0, {num_layers})")

    @staticmethod
    def _check_seq(name: str, seq: int, limit: int) -> None:
        if not (1 <= int(seq) <= int(limit)):
            raise ValueError(f"{name}={seq} out of range [1, {limit}]")
