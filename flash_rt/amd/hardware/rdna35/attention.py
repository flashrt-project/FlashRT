"""Native HIP and PyTorch SDPA attention providers for AMD RDNA 3.5."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def sdpa(q, k, v, *, causal=False, mask=None):
    """BSHD attention shared by RDNA providers, including grouped KV heads.

    Return a BSHD view; callers that flatten heads or require contiguous output
    materialize that layout at their own boundary. mask follows Torch SDPA
    semantics (True means attend), including broadcast over batch and heads.
    """
    value = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=causal, enable_gqa=q.shape[2] != k.shape[2])
    return value.transpose(1, 2)


class Rdna35AttentionBackend:
    """BF16 attention backend with shape-specialized RDNA 3.5 kernels."""

    def __init__(self, kernels):
        self.fvk = kernels
        self.hip_encoder_gqa = (
            os.getenv("FLASHRT_RDNA35_HIP_ENCODER_ATTN", "1") == "1"
        )
        build_info = getattr(kernels, "build_info", None)
        arch = (
            str(dict(build_info()).get("gpu_arch", ""))
            if callable(build_info) else ""
        )
        # On gfx12 SDPA beats the native encoder kernel on dense prefixes
        # (66 vs 263 us at 661 rows on R9700); the native loop still wins
        # when it can skip padded K/V rows (239 vs 602 us at 712/661).
        self.hip_dense_encoder_gqa = not arch.startswith("gfx12")
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
        value = sdpa(q, k, v).reshape(views, seq, 1152)
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
            and (self.hip_dense_encoder_gqa or valid_prefix < k.shape[0])
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
        value = sdpa(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask=mask)
        value = value.squeeze(0).reshape(
            q.shape[0], q.shape[1] * 256)
        if out is None:
            return value
        out.copy_(value)
        return out


class RdnaKvCache(dict):
    """Immutable FP32 prefix states and packed attention workspace."""

    def __init__(self, key, value, packed_key, packed_value, groups=1):
        super().__init__(key_states=key, value_states=value)
        self.packed_key, self.packed_value = packed_key, packed_value
        self.groups = groups


class RdnaAttention:
    """Batch-1 BF16 vision attention and FP32 language/action attention.

    GQA preserves FP32 RoPE products and uniform fully masked rows. Layers
    share score workspace and retain packed prefix K/V for suffix updates.
    """

    def __init__(self, kernels):
        self.fvk = kernels
        self._runner = kernels.RdnaGemmRunner()
        self._workspace = {}
        self._grouped_gqa = dict(kernels.build_info())["gpu_arch"] == "gfx1151"

    def _groups(self, heads, kv_heads, dim):
        return (
            heads // kv_heads
            if self._grouped_gqa and (heads, kv_heads, dim) in ((32, 8, 128), (20, 4, 128))
            else 1
        )

    def begin_exact_calibration(self, num_algorithms=16):
        """Screen action PV algorithms against the original FP32 output."""
        if self._grouped_gqa:
            self._runner.begin_fp32_gqa_pv_calibration(num_algorithms)

    def end_exact_calibration(self):
        """Select the fastest algorithm matching all warmup attention calls."""
        if self._grouped_gqa:
            return self._runner.end_fp32_gqa_pv_calibration()
        return 0

    def prepare_cache(self, k, v, heads, suffix_rows):
        prefix, kv_heads, dim = k.shape[1:]
        groups = self._groups(heads, kv_heads, dim)
        packed_heads = heads // groups
        packed_k = torch.empty(
            (prefix + suffix_rows, packed_heads, dim), device=k.device, dtype=torch.float32
        )
        packed_v = torch.empty_like(packed_k)
        self.fvk.attention_pack_kv_rdna(
            k.data_ptr(), v.data_ptr(), 0, 0, packed_k.data_ptr(), packed_v.data_ptr(), prefix, 0,
            packed_heads, kv_heads, dim, k.stride(1), v.stride(1), 0, 0,
            int(torch.cuda.current_stream(k.device).cuda_stream),
        )
        return RdnaKvCache(k, v, packed_k, packed_v, groups)

    def prepare_prefix_cache(self, k, v, weight, cos, sin, row_map, heads, suffix_rows, eps):
        """Normalize 128-wide BF16 K and write FP32 K/V at original key positions."""
        prefix, kv_heads, dim = row_map.numel(), k.shape[2], k.shape[3]
        groups = self._groups(heads, kv_heads, dim)
        packed_k = torch.empty(
            (prefix + suffix_rows, kv_heads, dim), device=k.device, dtype=torch.float32
        )
        packed_v = torch.empty_like(packed_k)
        self.fvk.k_norm_rope_cache_rdna(
            k.data_ptr(), v.data_ptr(), weight.data_ptr(), cos.data_ptr(), sin.data_ptr(),
            row_map.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(), prefix, kv_heads, eps,
            int(torch.cuda.current_stream(k.device).cuda_stream),
        )
        return RdnaKvCache(
            packed_k[:prefix].unsqueeze(0), packed_v[:prefix].unsqueeze(0), packed_k, packed_v,
            groups,
        )

    def prepare_prefix_query(self, q, weight, cos, sin, groups, eps):
        """Normalize 128-wide BF16 Q and rotate into grouped FP32 attention layout."""
        rows, heads, dim = q.shape[1:]
        packed = torch.empty(
            (heads // groups, rows * groups, dim), device=q.device, dtype=torch.float32
        )
        self.fvk.q_norm_rope_pack_rdna(
            q.data_ptr(), weight.data_ptr(), cos.data_ptr(), sin.data_ptr(), packed.data_ptr(),
            rows, heads, groups, eps, int(torch.cuda.current_stream(q.device).cuda_stream),
        )
        return packed

    def __call__(
        self, q, k, v, mask,
        *,
        suffix=None, cache=None, rotary=None, reference_queries=None, packed_query=None,
    ):
        queries, heads, dim = q.shape[1:]
        prefix, kv_heads = k.shape[1:3]
        suffix_rows = 0 if suffix is None else suffix[0].shape[1]
        keys = prefix + suffix_rows
        groups = self._groups(heads, kv_heads, dim) if cache is None else cache.groups
        packed_heads = heads // groups
        if rotary is not None and (cache is None or suffix is None or groups == 1):
            raise ValueError("rotary packing requires cached FP32 grouped attention")
        shape = (queries, keys, heads, dim, groups, q.device, cache is None, packed_query is None)
        workspace = self._workspace.get(shape)
        if workspace is None:
            packed_k = packed_v = packed_q = None
            if cache is None:
                packed_k = torch.empty(
                    (keys, packed_heads, dim), device=q.device, dtype=torch.float32
                )
                packed_v = torch.empty_like(packed_k)
            if groups > 1 and packed_query is None:
                packed_q = torch.empty(
                    (packed_heads, queries * groups, dim), device=q.device, dtype=torch.float32
                )
            scores = torch.empty(
                (packed_heads, queries * groups, keys), device=q.device, dtype=torch.float32
            )
            probs = torch.empty_like(scores)
            attended = torch.empty(
                (packed_heads, queries * groups, dim), device=q.device, dtype=torch.float32
            )
            workspace = self._workspace[shape] = (
                packed_k, packed_v, packed_q, scores, probs, attended,
            )
        packed_k, packed_v, packed_q, scores, probs, attended = workspace
        if packed_query is not None:
            packed_q = packed_query
        sk, sv = (None, None) if suffix is None else suffix
        stream = int(torch.cuda.current_stream(q.device).cuda_stream)
        query_stride = heads * dim if rotary is not None else q.stride(1)
        if rotary is not None:
            cos, sin = rotary
            packed_k, packed_v = cache.packed_key, cache.packed_value
            offset = prefix * packed_heads * dim * packed_k.element_size()
            self.fvk.attention_rope_pack_fp32_rdna(
                q.data_ptr(), sk.data_ptr(), sv.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                packed_q.data_ptr(), packed_k.data_ptr() + offset, packed_v.data_ptr() + offset,
                queries, heads, kv_heads, dim, q.stride(1), sk.stride(1), sv.stride(1),
                cos.stride(-2), sin.stride(-2), stream,
            )
        elif cache is None:
            self.fvk.attention_pack_kv_rdna(
                k.data_ptr(), v.data_ptr(), 0 if sk is None else sk.data_ptr(),
                0 if sv is None else sv.data_ptr(), packed_k.data_ptr(), packed_v.data_ptr(),
                prefix, suffix_rows, packed_heads, kv_heads, dim, k.stride(1), v.stride(1),
                0 if sk is None else sk.stride(1), 0 if sv is None else sv.stride(1), stream,
            )
        else:
            packed_k, packed_v = cache.packed_key, cache.packed_value
            if suffix_rows:
                offset = prefix * packed_heads * dim * packed_k.element_size()
                self.fvk.attention_pack_kv_rdna(
                    0, 0, sk.data_ptr(), sv.data_ptr(), packed_k.data_ptr() + offset,
                    packed_v.data_ptr() + offset, 0, suffix_rows, packed_heads, kv_heads, dim, 0,
                    0, sk.stride(1), sv.stride(1), stream,
                )
        out = torch.empty((1, queries, heads * dim), dtype=torch.bfloat16, device=q.device)
        if groups > 1:
            if rotary is None and packed_query is None:
                self.fvk.attention_pack_q_fp32_rdna(
                    q.data_ptr(), packed_q.data_ptr(), queries, heads, groups, dim, q.stride(1),
                    stream,
                )
            bmm = (
                self._runner.fp32_gqa_bmm
                if reference_queries is None
                else self._runner.fp32_gqa_bmm_rows
            )
            query_rows = (queries,) if reference_queries is None else (queries, reference_queries)
            bmm(
                packed_q.data_ptr(), packed_k.data_ptr(), scores.data_ptr(), *query_rows, keys,
                dim, heads, kv_heads, query_stride, True, stream,
            )
            self.fvk.attention_softmax_gqa_fp32_rdna(
                scores.data_ptr(), mask.data_ptr(), probs.data_ptr(), queries, keys, heads, groups,
                dim**-0.5, stream,
            )
            bmm(
                probs.data_ptr(), packed_v.data_ptr(), attended.data_ptr(), *query_rows, keys, dim,
                heads, kv_heads, query_stride, False, stream,
            )
            self.fvk.attention_store_gqa_bf16_rdna(
                attended.data_ptr(), out.data_ptr(), queries, heads, groups, dim, stream
            )
        else:
            self._runner.fp32_bmm(
                q.data_ptr(), packed_k.data_ptr(), scores.data_ptr(), queries, keys, dim,
                q.stride(1), heads * dim, keys, dim, dim, queries * keys, heads, True, stream,
            )
            self.fvk.attention_softmax_fp32_rdna(
                scores.data_ptr(), mask.data_ptr(), probs.data_ptr(), queries, keys, heads,
                dim**-0.5, stream,
            )
            self._runner.fp32_bmm(
                probs.data_ptr(), packed_v.data_ptr(), attended.data_ptr(), queries, dim, keys,
                keys, heads * dim, dim, queries * keys, dim, queries * dim, heads, False, stream,
            )
            self.fvk.attention_store_bf16_rdna(
                attended.data_ptr(), out.data_ptr(), queries, heads, dim, stream
            )
        return out

    def vision(self, q, k, v):
        queries, heads, dim = q.shape[1:]
        keys = k.shape[1]
        shape = ("bf16", queries, keys, heads, dim, q.device)
        workspace = self._workspace.get(shape)
        if workspace is None:
            scores = torch.empty((heads, queries, keys), dtype=torch.bfloat16, device=q.device)
            probs = torch.empty_like(scores)
            attended = torch.empty((heads, queries, dim), dtype=torch.bfloat16, device=q.device)
            workspace = self._workspace[shape] = scores, probs, attended
        scores, probs, attended = workspace
        stream = int(torch.cuda.current_stream(q.device).cuda_stream)
        self._runner.bf16_bmm(
            q.data_ptr(), k.data_ptr(), scores.data_ptr(), queries, keys, dim, q.stride(1),
            k.stride(1), keys, dim, dim, queries * keys, heads, True, stream,
        )
        self.fvk.attention_softmax_bf16_rdna(
            scores.data_ptr(), probs.data_ptr(), queries, keys, heads, dim**-0.5, stream
        )
        self._runner.bf16_bmm(
            probs.data_ptr(), v.data_ptr(), attended.data_ptr(), queries, dim, keys, keys,
            v.stride(1), dim, queries * keys, dim, queries * dim, heads, False, stream,
        )
        out = torch.empty((1, queries, heads, dim), dtype=torch.bfloat16, device=q.device)
        self.fvk.attention_store_bshd_rdna(
            attended.data_ptr(), out.data_ptr(), queries, heads, dim, stream
        )
        return out
