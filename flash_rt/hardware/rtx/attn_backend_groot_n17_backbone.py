"""FlashRT -- RTX backbone attention backend for GROOT N1.7.

Provides the ``vit`` / ``llm`` / ``vl_self_attn`` attention sites needed to
run the N1.7 VLM backbone (ViT / truncated LLM / VL self-attn) through
FlashRT kernels on RTX, so ``set_prompt`` produces ``backbone_features`` from
kernels instead of a PyTorch reference forward.

Routing (RTX has no SM100 strided FMHA; uses in-SO FA2 + cuBLAS MHA):
  * ``vit``          → vendored FA2 ``fwd_fp16``, batch = num_views
                       (multi-view per-image attention, non-causal).
  * ``vl_self_attn`` → vendored FA2 ``fwd_fp16``, batch = 1 (non-causal).
  * ``llm``          → ``fvk.attention_mha_causal_fp16`` (causal; the FA2
                       causal entry is bf16-only, so the truncated LLM
                       decoder uses the cuBLAS-decomposed fp16 MHA kernel).

The forward functions in ``pipeline_rtx_fp16`` (and ``pipeline_thor``) write
Q/K/V into the slot pointers returned by ``get_slot_ptrs(site, layer)`` and
read O from the same slot after ``run(site, layer, q_seq, kv_seq)``. Slots
are layer-shared (one buffer set per site, reused across layers).
"""

from __future__ import annotations

# Fixed N1.7 backbone attention dims.
_VIT_NH, _VIT_HD = 16, 64
_LLM_NH, _LLM_HD = 16, 128      # K/V are GQA-expanded to NHQ heads by the forward
_VLSA_NH, _VLSA_HD = 32, 64


class RtxGrootN17BackboneAttn:
    """FA2 + cuBLAS-MHA backbone attention slots for GROOT N1.7 on RTX."""

    SITES = ("vit", "llm", "vl_self_attn")

    def __init__(
        self,
        *,
        num_vit_views: int,
        vit_seq: int,
        llm_seq: int,
        vl_self_attn_seq: int,
        device: str = "cuda",
        slot_dtype=None,
    ):
        import torch

        self._torch = torch
        self._device = device
        dt = slot_dtype if slot_dtype is not None else torch.float16

        self._nv = int(num_vit_views)
        self._vit_seq = int(vit_seq)
        self._llm_seq = int(llm_seq)
        self._vlsa_seq = int(vl_self_attn_seq)
        if self._vit_seq % self._nv != 0:
            raise ValueError(
                f"vit_seq={self._vit_seq} not divisible by num_vit_views={self._nv}")
        self._vit_per = self._vit_seq // self._nv

        def slot(S, NH, HD):
            return torch.empty(S, NH, HD, dtype=dt, device=device)

        # ViT slots (multi-view batched).
        self.vit_Q = slot(self._vit_seq, _VIT_NH, _VIT_HD)
        self.vit_K = slot(self._vit_seq, _VIT_NH, _VIT_HD)
        self.vit_V = slot(self._vit_seq, _VIT_NH, _VIT_HD)
        self.vit_O = slot(self._vit_seq, _VIT_NH, _VIT_HD)

        # LLM slots (K/V hold GQA-expanded 16-head data written by the forward).
        self.llm_Q = slot(self._llm_seq, _LLM_NH, _LLM_HD)
        self.llm_K = slot(self._llm_seq, _LLM_NH, _LLM_HD)
        self.llm_V = slot(self._llm_seq, _LLM_NH, _LLM_HD)
        self.llm_O = slot(self._llm_seq, _LLM_NH, _LLM_HD)

        # VL self-attn slots.
        self.vlsa_Q = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD)
        self.vlsa_K = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD)
        self.vlsa_V = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD)
        self.vlsa_O = slot(self._vlsa_seq, _VLSA_NH, _VLSA_HD)

        # FA2 (vit / vl_self_attn) + LSE scratch.
        try:
            from flash_rt import flash_rt_fa2 as fa2
        except ImportError:
            raise RuntimeError(
                "GROOT N1.7 RTX backbone backend requires FlashRT's vendored "
                "FA2 module (`flash_rt.flash_rt_fa2`). Build it with CMake on "
                "RTX targets.")
        self._fa2 = fa2
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count

        def lse(B, heads, seq):
            return torch.empty(
                B, heads, ((int(seq) + 127) // 128) * 128,
                dtype=torch.float32, device=device)

        self._vit_lse = lse(self._nv, _VIT_NH, self._vit_per)
        self._vlsa_lse = lse(1, _VLSA_NH, self._vlsa_seq)

        # cuBLAS MHA (llm, causal fp16): context + neginf-prefilled logits.
        import flash_rt.flash_rt_kernels as fvk
        self._fvk = fvk
        self._llm_ctx = fvk.FvkContext()
        self._llm_logits = torch.empty(
            _LLM_NH, self._llm_seq, self._llm_seq, dtype=dt, device=device)

    def sites(self) -> tuple[str, ...]:
        return self.SITES

    def get_slot_ptrs(self, site: str, layer_idx: int = 0) -> dict[str, int]:
        if site == "vit":
            return {"Q": self.vit_Q.data_ptr(), "K": self.vit_K.data_ptr(),
                    "V": self.vit_V.data_ptr(), "O": self.vit_O.data_ptr()}
        if site == "llm":
            return {"Q": self.llm_Q.data_ptr(), "K": self.llm_K.data_ptr(),
                    "V": self.llm_V.data_ptr(), "O": self.llm_O.data_ptr()}
        if site == "vl_self_attn":
            return {"Q": self.vlsa_Q.data_ptr(), "K": self.vlsa_K.data_ptr(),
                    "V": self.vlsa_V.data_ptr(), "O": self.vlsa_O.data_ptr()}
        raise KeyError(f"unknown site {site!r}; known: {self.SITES}")

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq=None, stream: int = 0) -> int:
        if kv_seq is None:
            kv_seq = q_seq

        if site == "vit":
            if int(q_seq) != self._vit_per or int(kv_seq) != self._vit_per:
                raise ValueError(
                    f"vit q_seq/kv_seq must equal per-view len {self._vit_per}")
            q = self.vit_Q.view(self._nv, self._vit_per, _VIT_NH, _VIT_HD)
            k = self.vit_K.view(self._nv, self._vit_per, _VIT_NH, _VIT_HD)
            v = self.vit_V.view(self._nv, self._vit_per, _VIT_NH, _VIT_HD)
            o = self.vit_O.view(self._nv, self._vit_per, _VIT_NH, _VIT_HD)
            return self._run_fa2(q, k, v, o, self._vit_lse, _VIT_HD, stream)

        if site == "vl_self_attn":
            S = int(q_seq)
            self._check_seq("vl_self_attn", S, self._vlsa_seq)
            q = self.vlsa_Q[:S].unsqueeze(0)
            k = self.vlsa_K[:S].unsqueeze(0)
            v = self.vlsa_V[:S].unsqueeze(0)
            o = self.vlsa_O[:S].unsqueeze(0)
            return self._run_fa2(q, k, v, o, self._vlsa_lse, _VLSA_HD, stream)

        if site == "llm":
            S = int(q_seq)
            self._check_seq("llm", S, self._llm_seq)
            if int(kv_seq) != S:
                raise ValueError("llm is self-attention; kv_seq must equal q_seq")
            nh, hd = _LLM_NH, _LLM_HD
            # attention_mha_* reads leftover logits bytes into softmax; the
            # full (NH, S, S) slab must be -inf-prefilled (cos collapses
            # otherwise — same contract as the Thor backend).
            self._fvk.gpu_fill_neginf_fp16(
                self._llm_logits.data_ptr(),
                nh * self._llm_seq * self._llm_seq, int(stream))
            self._fvk.attention_mha_causal_fp16(
                self._llm_ctx,
                self.llm_Q.data_ptr(), self.llm_K.data_ptr(), self.llm_V.data_ptr(),
                self._llm_logits.data_ptr(), self.llm_O.data_ptr(),
                S, S, nh, hd, 1.0 / (hd ** 0.5), int(stream))
            return self.llm_O.data_ptr()

        raise KeyError(f"unknown site {site!r}; known: {self.SITES}")

    def _run_fa2(self, q, k, v, o, lse, hd, stream) -> int:
        fwd = self._fa2.fwd_fp16
        B, Sq, Hq, D = q.shape
        Sk, Hk = k.shape[1], k.shape[2]
        fwd(
            Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
            O=o.data_ptr(), softmax_lse=lse.data_ptr(),
            softmax_lse_accum=0, o_accum=0,
            batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_heads_q=Hq, num_heads_kv=Hk, head_dim=D,
            q_strides=(q.stride(0), q.stride(1), q.stride(2)),
            k_strides=(k.stride(0), k.stride(1), k.stride(2)),
            v_strides=(v.stride(0), v.stride(1), v.stride(2)),
            o_strides=(o.stride(0), o.stride(1), o.stride(2)),
            softmax_scale=1.0 / (hd ** 0.5),
            num_sms=self._num_sms,
            stream=int(stream),
        )
        return o.data_ptr()

    @staticmethod
    def _check_seq(name: str, seq: int, limit: int) -> None:
        if not (1 <= int(seq) <= int(limit)):
            raise ValueError(f"{name} seq={seq} out of range [1, {limit}]")
