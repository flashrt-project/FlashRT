"""ROCm attention backend for Qwen3 owned BF16 buffers."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - optional package
    flash_attn_func = None

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - depends on PyTorch build
    SDPBackend = None
    sdpa_kernel = None


class RocmQwen3AttnBackend:
    """Qwen3 GQA attention backend with owned Q/O scratch and KV cache.

    Tensor layout follows the Qwen3 kernels:

    - Q scratch: ``(max_q_seq, 32, 128)``
    - K/V cache: ``(num_layers, max_seq, 8, 128)``
    - O scratch: ``(max_q_seq, 32, 128)``
    """

    def __init__(
        self,
        *,
        num_layers: int = 36,
        max_seq: int = 4096,
        max_q_seq: int = 1,
        q_heads: int = 32,
        kv_heads: int = 8,
        head_dim: int = 128,
        dtype=None,
        preferred_backend: str = "flash_attn",
    ) -> None:
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("RocmQwen3AttnBackend requires ROCm PyTorch")

        bf16 = dtype if dtype is not None else torch.bfloat16
        device = "cuda"
        self.q = torch.empty(max_q_seq, q_heads, head_dim, dtype=bf16, device=device)
        self.o = torch.empty(max_q_seq, q_heads, head_dim, dtype=bf16, device=device)
        self.k_cache = torch.empty(
            num_layers, max_seq, kv_heads, head_dim, dtype=bf16, device=device
        )
        self.v_cache = torch.empty(
            num_layers, max_seq, kv_heads, head_dim, dtype=bf16, device=device
        )

        self.num_layers = int(num_layers)
        self.max_seq = int(max_seq)
        self.max_q_seq = int(max_q_seq)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self._preferred_backend = str(preferred_backend).lower()
        self._use_decode_kernel = self._preferred_backend in {
            "decode_kernel",
            "decode-kernel",
            "raw_decode",
            "rocm_decode",
        }
        self._use_flash_attn_func = self._preferred_backend in {
            "flash_attn",
            "flash-attn",
            "fa2",
            "fa_rocm",
            "decode_kernel",
            "decode-kernel",
            "raw_decode",
            "rocm_decode",
        }
        if self._use_flash_attn_func and flash_attn_func is None:
            raise RuntimeError("preferred_backend='flash_attn' requires flash_attn")
        self._active_backend = (
            None if self._use_flash_attn_func else self._resolve_backend(preferred_backend)
        )

    @staticmethod
    def _resolve_backend(name: str):
        if SDPBackend is None:
            return None
        normalized = str(name).lower()
        if normalized in {"auto", "default", "sdpa"}:
            return None
        if normalized in {"flash", "flash_attention", "fa"}:
            return SDPBackend.FLASH_ATTENTION
        if normalized in {"efficient", "mem_efficient", "memory_efficient"}:
            return SDPBackend.EFFICIENT_ATTENTION
        if normalized == "math":
            return SDPBackend.MATH
        raise ValueError(f"unknown ROCm SDPA backend: {name!r}")

    @property
    def active_backend_name(self) -> str:
        if self._use_decode_kernel:
            return "qwen3_decode_kernel+flash_attn_func"
        if self._use_flash_attn_func:
            return "flash_attn_func"
        if self._active_backend is None:
            return "auto"
        return getattr(self._active_backend, "name", str(self._active_backend))

    def get_ptrs(self) -> dict[str, int]:
        return {
            "q": self.q.data_ptr(),
            "o": self.o.data_ptr(),
            "k_cache": self.k_cache.data_ptr(),
            "v_cache": self.v_cache.data_ptr(),
            "num_layers": self.num_layers,
            "max_seq": self.max_seq,
            "max_q_seq": self.max_q_seq,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "k_layer_stride_bytes": self.k_cache.stride(0) * self.k_cache.element_size(),
            "v_layer_stride_bytes": self.v_cache.stride(0) * self.v_cache.element_size(),
        }

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool):
        if self._use_flash_attn_func:
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=self.head_dim ** -0.5,
                causal=causal,
            )

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if k_t.shape[1] != q_t.shape[1]:
            repeat = q_t.shape[1] // k_t.shape[1]
            k_t = k_t.repeat_interleave(repeat, dim=1)
            v_t = v_t.repeat_interleave(repeat, dim=1)
        if self._active_backend is None or sdpa_kernel is None:
            ctx = nullcontext()
        else:
            ctx = sdpa_kernel([self._active_backend])
        with ctx:
            return F.scaled_dot_product_attention(
                q_t, k_t, v_t, dropout_p=0.0, is_causal=causal
            ).transpose(1, 2)

    def run(self, layer_idx: int, q_seq: int, kv_seq: int, *, causal: bool = False) -> int:
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError("layer_idx out of range")
        if q_seq <= 0 or q_seq > self.max_q_seq:
            raise ValueError("q_seq out of range")
        if kv_seq <= 0 or kv_seq > self.max_seq:
            raise ValueError("kv_seq out of range")
        if self._use_decode_kernel and q_seq == 1:
            import flash_rt.flash_rt_rocm_kernels as kernels

            stream = torch.cuda.current_stream().cuda_stream
            kernels.qwen3_decode_attention_bf16_ptr(
                self.q.data_ptr(),
                self.k_cache.data_ptr(),
                self.v_cache.data_ptr(),
                self.o.data_ptr(),
                int(layer_idx),
                self.max_seq,
                int(kv_seq),
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                stream,
            )
            return int(self.o.data_ptr())
        q = self.q[:q_seq].unsqueeze(0)
        k = self.k_cache[layer_idx, :kv_seq].unsqueeze(0)
        v = self.v_cache[layer_idx, :kv_seq].unsqueeze(0)
        out = self._sdpa(q, k, v, causal=causal).squeeze(0)
        self.o[:q_seq].copy_(out)
        return int(self.o.data_ptr())


__all__ = ["RocmQwen3AttnBackend"]
