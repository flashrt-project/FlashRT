"""ROCm attention backend for FlashRT-owned Pi0.5 buffers.

This mirrors the RTX attention backend contract: the backend owns Q/K/V/O
scratch tensors and exposes raw device pointers. The first AMD implementation
uses PyTorch ROCm SDPA as a correctness bridge while the rest of the BF16
pipeline is being kernelized. The call boundary is intentionally stable so the
implementation can later be swapped to AOTriton, CK, or a custom HIP kernel
without changing the Pi0.5 pipeline.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - depends on optional ROCm FA package
    flash_attn_func = None

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - depends on PyTorch build
    SDPBackend = None
    sdpa_kernel = None


def make_pi05_attention_spec(
    *,
    num_views: int,
    encoder_seq_max: int,
    chunk_size: int = 10,
    num_encoder_layers: int = 18,
) -> AttentionSpec:
    spec = AttentionSpec()
    spec.add_site(
        "siglip",
        num_layers=27,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=72,
        max_q_seq=256,
        batch_axis=int(num_views),
    )
    spec.add_site(
        "encoder",
        num_layers=int(num_encoder_layers),
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=256,
        max_q_seq=int(encoder_seq_max),
    )
    spec.add_site(
        "decoder",
        num_layers=int(num_encoder_layers),
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=256,
        max_q_seq=int(chunk_size),
        max_kv_seq=int(encoder_seq_max) + int(chunk_size),
    )
    return spec


class RocmSdpaAttnBackend(AttentionBackendBase):
    """Pi0.5 ROCm attention backend using preallocated BF16 scratch."""

    def __init__(
        self,
        num_views: int,
        encoder_seq_max: int,
        chunk_size: int,
        num_encoder_layers: int = 18,
        dtype=None,
        preferred_backend: str = "flash",
        decoder_preferred_backend: str | None = "flash",
    ):
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("RocmSdpaAttnBackend requires ROCm PyTorch")

        super().__init__(
            make_pi05_attention_spec(
                num_views=num_views,
                encoder_seq_max=encoder_seq_max,
                chunk_size=chunk_size,
                num_encoder_layers=num_encoder_layers,
            )
        )
        bf16 = dtype if dtype is not None else torch.bfloat16
        device = "cuda"
        self.vis_Q = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=device)
        self.vis_K = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=device)
        self.vis_V = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=device)
        self.vis_O = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=device)

        total_kv = encoder_seq_max + chunk_size
        self.enc_Q = torch.empty(encoder_seq_max, 8, 256, dtype=bf16, device=device)
        self.enc_K = torch.empty(
            num_encoder_layers, total_kv, 1, 256, dtype=bf16, device=device
        )
        self.enc_V = torch.empty(
            num_encoder_layers, total_kv, 1, 256, dtype=bf16, device=device
        )
        self.enc_O = torch.empty(encoder_seq_max, 8, 256, dtype=bf16, device=device)

        self.dec_Q = torch.empty(chunk_size, 8, 256, dtype=bf16, device=device)
        self.dec_O = torch.empty(chunk_size, 8, 256, dtype=bf16, device=device)

        self._num_views = int(num_views)
        self._encoder_seq_max = int(encoder_seq_max)
        self._chunk_size = int(chunk_size)
        self._num_encoder_layers = int(num_encoder_layers)
        self._preferred_backend = str(preferred_backend).lower()
        self._use_flash_attn_func = self._preferred_backend in {
            "flash_attn",
            "flash-attn",
            "fa2",
            "fa_rocm",
        }
        if self._use_flash_attn_func and flash_attn_func is None:
            raise RuntimeError("preferred_backend='flash_attn' requires flash_attn")
        self._active_backend = (
            None if self._use_flash_attn_func else self._resolve_backend(preferred_backend)
        )
        decoder_backend_name = (
            preferred_backend
            if decoder_preferred_backend is None
            else decoder_preferred_backend
        )
        self._decoder_preferred_backend = str(decoder_backend_name).lower()
        self._use_decoder_flash_attn_func = self._decoder_preferred_backend in {
            "flash_attn",
            "flash-attn",
            "fa2",
            "fa_rocm",
        }
        if self._use_decoder_flash_attn_func and flash_attn_func is None:
            raise RuntimeError(
                "decoder_preferred_backend='flash_attn' requires flash_attn"
            )
        self._decoder_active_backend = (
            None
            if self._use_decoder_flash_attn_func
            else self._resolve_backend(decoder_backend_name)
        )
        self._enc_kv_layer_stride_bytes = (
            total_kv * 1 * 256 * self.enc_K.element_size()
        )

        try:
            self.warmup()
        except Exception:
            can_fallback = (
                self._active_backend is not None
                or self._decoder_active_backend is not None
            )
            if not can_fallback:
                raise
            self._active_backend = None
            self._decoder_active_backend = None
            self.warmup()

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
        return self._backend_name(self._use_flash_attn_func, self._active_backend)

    @staticmethod
    def _backend_name(use_flash_attn_func: bool, backend) -> str:
        if use_flash_attn_func:
            return "flash_attn_func"
        if backend is None:
            return "auto"
        return getattr(backend, "name", str(backend))

    @property
    def decoder_backend_name(self) -> str:
        return self._backend_name(
            self._use_decoder_flash_attn_func, self._decoder_active_backend
        )

    def get_ptrs(self) -> dict[str, int]:
        return {
            "vis_Q": self.vis_Q.data_ptr(),
            "vis_K": self.vis_K.data_ptr(),
            "vis_V": self.vis_V.data_ptr(),
            "enc_Q": self.enc_Q.data_ptr(),
            "enc_K": self.enc_K.data_ptr(),
            "enc_V": self.enc_V.data_ptr(),
            "dec_Q": self.dec_Q.data_ptr(),
            "enc_k_layer_stride_bytes": self._enc_kv_layer_stride_bytes,
            "enc_v_layer_stride_bytes": self._enc_kv_layer_stride_bytes,
        }

    def _check_layer_idx(self, site: str, layer_idx: int) -> None:
        spec = self._spec.site(site)
        if layer_idx < 0 or layer_idx >= spec.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r}"
            )

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        self._check_layer_idx(site, layer_idx)
        if site == "siglip":
            return {
                "Q": self.vis_Q.data_ptr(),
                "K": self.vis_K.data_ptr(),
                "V": self.vis_V.data_ptr(),
                "O": self.vis_O.data_ptr(),
            }
        if site == "encoder":
            offset = layer_idx * self._enc_kv_layer_stride_bytes
            return {
                "Q": self.enc_Q.data_ptr(),
                "K": self.enc_K.data_ptr() + offset,
                "V": self.enc_V.data_ptr() + offset,
                "O": self.enc_O.data_ptr(),
            }
        if site == "decoder":
            offset = layer_idx * self._enc_kv_layer_stride_bytes
            return {
                "Q": self.dec_Q.data_ptr(),
                "K": self.enc_K.data_ptr() + offset,
                "V": self.enc_V.data_ptr() + offset,
                "O": self.dec_O.data_ptr(),
            }
        raise ValueError(f"unknown attention site: {site!r}")

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        use_flash_attn_func: bool,
        active_backend,
    ) -> torch.Tensor:
        if use_flash_attn_func:
            return flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                softmax_scale=q.shape[-1] ** -0.5,
                causal=False,
            ).transpose(1, 2)
        if active_backend is None or sdpa_kernel is None:
            ctx = nullcontext()
        else:
            ctx = sdpa_kernel([active_backend])
        if k.shape[1] != q.shape[1]:
            k = k.expand(q.shape[0], q.shape[1], k.shape[2], k.shape[3])
            v = v.expand(q.shape[0], q.shape[1], v.shape[2], v.shape[3])
        with ctx:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )

    def warmup(self) -> None:
        """Compile/warm selected ROCm SDPA kernels for the fixed Pi0.5 shapes."""
        self.vis_Q.zero_()
        self.vis_K.zero_()
        self.vis_V.zero_()
        self.vision_attn()

        enc_seq = min(self._encoder_seq_max, 64)
        self.enc_Q[:enc_seq].zero_()
        self.enc_K[0, :enc_seq].zero_()
        self.enc_V[0, :enc_seq].zero_()
        self.encoder_attn(0, enc_seq)

        dec_seq = min(self._chunk_size, 10)
        self.dec_Q[:dec_seq].zero_()
        self.enc_K[0, : enc_seq + dec_seq].zero_()
        self.enc_V[0, : enc_seq + dec_seq].zero_()
        self.decoder_attn(0, enc_seq=enc_seq, dec_seq=dec_seq)
        torch.cuda.synchronize()

    def vision_attn(self, stream: int = 0) -> int:
        del stream
        q = self.vis_Q.transpose(1, 2)
        k = self.vis_K.transpose(1, 2)
        v = self.vis_V.transpose(1, 2)
        out = self._sdpa(
            q,
            k,
            v,
            use_flash_attn_func=self._use_flash_attn_func,
            active_backend=self._active_backend,
        ).transpose(1, 2)
        self.vis_O.copy_(out)
        return self.vis_O.data_ptr()

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        del stream
        q = self.enc_Q[:seq].transpose(0, 1).unsqueeze(0)
        k = self.enc_K[layer_idx, :seq].transpose(0, 1).unsqueeze(0)
        v = self.enc_V[layer_idx, :seq].transpose(0, 1).unsqueeze(0)
        out = self._sdpa(
            q,
            k,
            v,
            use_flash_attn_func=self._use_flash_attn_func,
            active_backend=self._active_backend,
        ).squeeze(0).transpose(0, 1)
        self.enc_O[:seq].copy_(out)
        return self.enc_O.data_ptr()

    def decoder_attn(
        self,
        layer_idx: int,
        enc_seq: int,
        dec_seq: int,
        stream: int = 0,
    ) -> int:
        total_kv = enc_seq + dec_seq
        q = self.dec_Q[:dec_seq].transpose(0, 1).unsqueeze(0)
        k = self.enc_K[layer_idx, :total_kv].transpose(0, 1).unsqueeze(0)
        v = self.enc_V[layer_idx, :total_kv].transpose(0, 1).unsqueeze(0)
        out = self._sdpa(
            q,
            k,
            v,
            use_flash_attn_func=self._use_decoder_flash_attn_func,
            active_backend=self._decoder_active_backend,
        ).squeeze(0).transpose(0, 1)
        self.dec_O[:dec_seq].copy_(out)
        return self.dec_O.data_ptr()

    def run(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq: int | None = None,
        stream: int = 0,
        state_nk: int | None = None,
    ) -> int:
        del state_nk
        if site == "siglip":
            return self.vision_attn(stream=stream)
        if site == "encoder":
            return self.encoder_attn(layer_idx, q_seq, stream=stream)
        if site == "decoder":
            if kv_seq is None:
                raise ValueError("decoder attention requires kv_seq")
            return self.decoder_attn(layer_idx, kv_seq - q_seq, q_seq, stream=stream)
        raise ValueError(f"unknown attention site: {site!r}")


__all__ = ["RocmSdpaAttnBackend", "make_pi05_attention_spec"]
