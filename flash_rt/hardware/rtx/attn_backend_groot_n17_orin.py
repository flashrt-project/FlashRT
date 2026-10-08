"""FlashRT -- Orin SM87 backbone attention backend for GROOT N1.7.

BF16 sibling of :mod:`flash_rt.hardware.rtx.attn_backend_groot_n17_backbone`.
Two differences, both forced by SM87:

* Slots are BF16 and FA2 dispatches on ``q.dtype``, so ``vit`` /
  ``vl_self_attn`` reuse the parent's ``_run_fa2`` unchanged (the vendored
  FA2 is built with ``bf16`` in ``FA2_DTYPES`` for ``GPU_ARCH=87``).
* ``llm`` moves off ``fvk.attention_mha_causal_fp16`` — that kernel has no
  BF16 sibling in the SM87 build — onto FA2 ``fwd_bf16_causal``. FA2 takes
  ``num_heads_q`` and ``num_heads_kv`` separately, so the parent's
  GQA pre-expansion (``gpu_repeat_interleave_heads``, FP16-only) is dropped
  and K/V slots hold the native 8 heads instead of 16.

The FP16-only ``gpu_fill_neginf_fp16`` logits-slab contract also disappears:
FA2 computes softmax online and needs no pre-filled slab.
"""

from __future__ import annotations

from flash_rt.hardware.rtx.attn_backend_groot_n17_backbone import (
    RtxGrootN17BackboneAttn,
    _LLM_HD,
    _LLM_NH,
)

_LLM_NHKV = 8  # Qwen3-VL-2B: 16 query heads, 8 KV heads (GQA factor 2)


class OrinGrootN17BackboneAttn(RtxGrootN17BackboneAttn):
    """BF16 FA2 backbone attention slots for GROOT N1.7 on Jetson Orin."""

    def __init__(self, *, slot_dtype=None, **kwargs):
        import torch

        super().__init__(
            slot_dtype=torch.bfloat16 if slot_dtype is None else slot_dtype,
            **kwargs)
        dt = self.llm_Q.dtype
        if dt != torch.bfloat16:
            raise ValueError(
                f"OrinGrootN17BackboneAttn requires bf16 slots, got {dt}; "
                "SM87 has no FP8 path and the INT8 kernels are bf16-in/bf16-out.")
        # Native-GQA causal KV slots replace the parent's 16-head expanded ones.
        self.llm_K = torch.empty(
            self._llm_seq, _LLM_NHKV, _LLM_HD, dtype=dt, device=self._device)
        self.llm_V = torch.empty_like(self.llm_K)
        self._llm_lse = torch.empty(
            1, _LLM_NH, ((self._llm_seq + 127) // 128) * 128,
            dtype=torch.float32, device=self._device)
        # FP16 cuBLAS-MHA scaffolding is unused on this path; drop it so a
        # mistaken fallthrough fails loudly instead of reading bf16 as fp16.
        del self._llm_logits, self._llm_ctx

    def _run_fa2(self, q, k, v, o, lse, hd, stream) -> int:
        fwd = (self._fa2.fwd_bf16
               if q.dtype == self._torch.bfloat16 else self._fa2.fwd_fp16)
        return self._call_fa2(fwd, q, k, v, o, lse, hd, stream, causal=False)

    def _run_fa2_causal(self, q, k, v, o, lse, hd, stream) -> int:
        return self._call_fa2(
            self._fa2.fwd_bf16_causal, q, k, v, o, lse, hd, stream, causal=True)

    def _call_fa2(self, fwd, q, k, v, o, lse, hd, stream, *, causal) -> int:
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

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq=None, stream: int = 0) -> int:
        if site != "llm":
            return super().run(
                site, layer_idx, q_seq, kv_seq=kv_seq, stream=stream)

        S = int(q_seq)
        self._check_seq("llm", S, self._llm_seq)
        if kv_seq is not None and int(kv_seq) != S:
            raise ValueError("llm is self-attention; kv_seq must equal q_seq")
        q = self.llm_Q[:S].unsqueeze(0)
        k = self.llm_K[:S].unsqueeze(0)
        v = self.llm_V[:S].unsqueeze(0)
        o = self.llm_O[:S].unsqueeze(0)
        self._run_fa2_causal(q, k, v, o, self._llm_lse, _LLM_HD, stream)
        return self.llm_O.data_ptr()
