"""ROCm CK attention backend for FlashRT-owned Pi0.5 buffers."""

from __future__ import annotations

import torch

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec
from flash_rt import flash_rt_rocm_kernels as rocm_kernels


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
    """Pi0.5 ROCm attention backend using preallocated BF16 CK/HIP kernels."""

    def __init__(
        self,
        num_views: int,
        encoder_seq_max: int,
        chunk_size: int,
        num_encoder_layers: int = 18,
        dtype=None,
        preferred_backend: str = "ck_wmma",
        decoder_preferred_backend: str | None = "ck_wmma",
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
        self.gqa_K_full = torch.empty(total_kv, 8, 256, dtype=bf16, device=device)
        self.gqa_V_full = torch.empty(total_kv, 8, 256, dtype=bf16, device=device)

        self.dec_Q = torch.empty(chunk_size, 8, 256, dtype=bf16, device=device)
        self.dec_O = torch.empty(chunk_size, 8, 256, dtype=bf16, device=device)

        self._num_views = int(num_views)
        self._encoder_seq_max = int(encoder_seq_max)
        self._chunk_size = int(chunk_size)
        self._num_encoder_layers = int(num_encoder_layers)
        del preferred_backend, decoder_preferred_backend
        self._kernels = rocm_kernels
        self._enc_kv_layer_stride_bytes = (
            total_kv * 1 * 256 * self.enc_K.element_size()
        )

        self.warmup()

    @property
    def active_backend_name(self) -> str:
        return "ck_wmma"

    @property
    def decoder_backend_name(self) -> str:
        return "ck_wmma"

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

    def warmup(self) -> None:
        """Warm selected ROCm kernels for the fixed Pi0.5 shapes."""
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
        self._kernels.pi05_siglip_attention_bf16_ptr(
            self.vis_Q.data_ptr(),
            self.vis_K.data_ptr(),
            self.vis_V.data_ptr(),
            self.vis_O.data_ptr(),
            self._num_views,
            256,
            stream,
        )
        return self.vis_O.data_ptr()

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        self._kernels.qwen36_full_v_broadcast_bf16_out(
            self.enc_K[layer_idx, :seq],
            self.gqa_K_full[:seq],
            seq,
            1,
            8,
            256,
            stream,
        )
        self._kernels.qwen36_full_v_broadcast_bf16_out(
            self.enc_V[layer_idx, :seq],
            self.gqa_V_full[:seq],
            seq,
            1,
            8,
            256,
            stream,
        )
        self._kernels.pi05_gqa8_attention_bf16_ptr(
            self.enc_Q.data_ptr(),
            self.gqa_K_full.data_ptr(),
            self.gqa_V_full.data_ptr(),
            self.enc_O.data_ptr(),
            1,
            seq,
            seq,
            stream,
        )
        return self.enc_O.data_ptr()

    def decoder_attn(
        self,
        layer_idx: int,
        enc_seq: int,
        dec_seq: int,
        stream: int = 0,
    ) -> int:
        total_kv = enc_seq + dec_seq
        self._kernels.qwen36_full_v_broadcast_bf16_out(
            self.enc_K[layer_idx, :total_kv],
            self.gqa_K_full[:total_kv],
            total_kv,
            1,
            8,
            256,
            stream,
        )
        self._kernels.qwen36_full_v_broadcast_bf16_out(
            self.enc_V[layer_idx, :total_kv],
            self.gqa_V_full[:total_kv],
            total_kv,
            1,
            8,
            256,
            stream,
        )
        self._kernels.pi05_gqa8_attention_bf16_ptr(
            self.dec_Q.data_ptr(),
            self.gqa_K_full.data_ptr(),
            self.gqa_V_full.data_ptr(),
            self.dec_O.data_ptr(),
            1,
            dec_seq,
            total_kv,
            stream,
        )
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
