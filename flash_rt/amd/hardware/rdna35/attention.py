"""PyTorch SDPA attention provider for AMD RDNA 3.5."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


class Rdna35AttentionBackend:
    """BF16 attention backend with shape-specialized RDNA 3.5 kernels."""

    def __init__(self, kernels):
        self.fvk = kernels
        self.hip_encoder_gqa = (
            os.getenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1") == "1"
        )
        self.hip_gqa = os.getenv("FLASHRT_RDNA35_HIP_GQA", "1") == "1"
        self.hip_gqa_split_key = (
            os.getenv("FLASHRT_RDNA35_HIP_GQA_SPLIT_KEY", "1") == "1"
        )
        self.hip_gqa_keys = int(
            os.getenv("FLASHRT_RDNA35_HIP_GQA_KEYS", "4"))
        if self.hip_gqa_keys not in (1, 2, 4, 8):
            raise ValueError(
                "FLASHRT_RDNA35_HIP_GQA_KEYS must be 1, 2, 4, or 8")

    @staticmethod
    def _stream(tensor: torch.Tensor) -> int:
        return int(torch.cuda.current_stream(tensor.device).cuda_stream)

    @staticmethod
    def _check_bf16(*tensors: torch.Tensor) -> None:
        if any(t.device.type != "cuda" for t in tensors):
            raise ValueError("RDNA 3.5 attention tensors must be on the ROCm device")
        if any(t.device != tensors[0].device for t in tensors[1:]):
            raise ValueError(
                "RDNA 3.5 attention tensors must be on the same device")
        if any(t.dtype != torch.bfloat16 for t in tensors):
            raise TypeError("RDNA 3.5 attention tensors must be BF16")

    def vision(self, qkv: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        self._check_bf16(qkv)
        if qkv.ndim != 3 or qkv.shape[1:] != (256, 3 * 1152):
            raise ValueError(f"unsupported vision QKV shape: {tuple(qkv.shape)}")
        views, seq, _ = qkv.shape
        q, k, v = qkv.view(views, seq, 3, 16, 72).unbind(dim=2)
        value = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        value = value.transpose(1, 2).reshape(views, seq, 1152)
        if out is None:
            return value
        self._check_bf16(qkv, out)
        if out.shape != value.shape:
            raise ValueError(
                f"unsupported vision output shape: {tuple(out.shape)}")
        out.copy_(value)
        return out

    def gqa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        valid_prefix: int,
        prefix_capacity: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_bf16(q, k, v)
        if (
            q.ndim != 3
            or not 1 <= q.shape[1] <= 16
            or q.shape[2] != 256
        ):
            raise ValueError(f"unsupported GQA query shape: {tuple(q.shape)}")
        if k.shape != v.shape or k.ndim != 3 or k.shape[1:] != (1, 256):
            raise ValueError(
                f"unsupported GQA KV shapes: k={tuple(k.shape)} v={tuple(v.shape)}")
        if not 0 < valid_prefix <= prefix_capacity:
            raise ValueError(
                f"invalid prefix lengths: valid={valid_prefix} "
                f"capacity={prefix_capacity} kv={k.shape[0]}")
        if out is not None:
            self._check_bf16(q, k, v, out)
            if out.shape != (q.shape[0], q.shape[1] * 256):
                raise ValueError(
                    f"unsupported GQA output shape: {tuple(out.shape)}")

        compact_kv = k.shape[0] in (valid_prefix, valid_prefix + q.shape[0])
        if not compact_kv and prefix_capacity > k.shape[0]:
            raise ValueError(
                f"invalid padded KV layout: capacity={prefix_capacity} "
                f"kv={k.shape[0]}")
        if (
            self.hip_encoder_gqa
            and q.shape[0] > 16
            and q.shape[0] <= 4096
            # The native kernel covers the CDNA shape contract, but dense
            # sequences above the old 1024-row boundary remain faster in
            # AOTriton SDPA. Large padded inputs with a short valid prefix do
            # benefit from the native loop, which skips invalid K/V rows.
            and (q.shape[0] <= 1024 or valid_prefix <= 640)
            and q.shape[0] == k.shape[0]
            and (
                valid_prefix == k.shape[0]
                or prefix_capacity == k.shape[0]
            )
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            if out is None:
                out = torch.empty(
                    q.shape[0], q.shape[1] * 256,
                    dtype=q.dtype, device=q.device)
            self.fvk.attention_encoder_gqa_rdna(
                out.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
                q.shape[0], valid_prefix, q.shape[1], 256, self._stream(q))
            return out
        if (
            self.hip_gqa
            and q.shape[0] <= 16
            and k.shape[0] <= 2048
            and (self.hip_gqa_split_key or k.shape[0] <= 1024)
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            if out is None:
                out = torch.empty(
                    q.shape[0], q.shape[1] * 256,
                    dtype=q.dtype, device=q.device)
            arguments = (
                out.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
                q.shape[0], k.shape[0], q.shape[1], 256, valid_prefix,
                valid_prefix if compact_kv else prefix_capacity, 0.0625,
            )
            if self.hip_gqa_split_key:
                self.fvk.attention_decoder_gqa_splitkey_rdna(
                    *arguments, self.hip_gqa_keys, self._stream(q))
            else:
                self.fvk.attention_decoder_gqa_rdna(
                    *arguments, self._stream(q))
            return out
        mask = None
        if not compact_kv:
            mask = torch.zeros(
                q.shape[0], k.shape[0], dtype=torch.bool, device=q.device)
            mask[:, :valid_prefix] = True
            mask[:, prefix_capacity:] = True
        value = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            enable_gqa=True,
        )
        value = value.squeeze(0).transpose(0, 1).reshape(
            q.shape[0], q.shape[1] * 256)
        if out is None:
            return value
        out.copy_(value)
        return out
