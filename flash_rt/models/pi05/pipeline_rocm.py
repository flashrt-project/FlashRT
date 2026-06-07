"""FlashRT Pi0.5 ROCm BF16/FP8 pipeline.

This is the AMD counterpart to :mod:`flash_rt.models.pi05.pipeline_rtx`.
The pipeline owns stable HIP buffers, fixed Pi0.5 geometry, and the ROCm
kernel sequence. BF16 remains the reference path; FP8 is layered on top through
static activation scales and pre-baked GEMM plans.
"""

from __future__ import annotations

import ctypes
import os

import ml_dtypes
import numpy as np

from flash_rt.core.hip_buffer import HipBuffer, _hip, _check


VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3

ENC_L = 18
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8
ENC_NKV = 1
ENC_HD = 256

DEC_L = 18
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256

ACTION_DIM = 32
NUM_STEPS_DEFAULT = 10

BF16 = np.float16
BF16_NP = ml_dtypes.bfloat16
FP32 = np.float32
FP8_BYTES = np.uint8


class _TorchTensorBuffer:
    """Tiny HipBuffer-compatible wrapper for PyTorch-owned HIP tensors."""

    def __init__(self, shape, dtype):
        import torch

        self.tensor = torch.empty(tuple(int(x) for x in shape), device="cuda", dtype=dtype)
        self._ptr = ctypes.c_void_p(int(self.tensor.data_ptr()))

    @property
    def ptr(self) -> ctypes.c_void_p:
        return self._ptr

    @property
    def nbytes(self) -> int:
        return int(self.tensor.numel() * self.tensor.element_size())

    def zero_(self, stream=None) -> None:
        del stream
        self.tensor.zero_()


def _weight_ptr(weights, name: str, layer: int | None = None) -> int:
    value = weights[name] if layer is None else weights[name][layer]
    return int(value.data_ptr() if hasattr(value, "data_ptr") else value)


def _fp8_key(name: str, layer: int | None = None) -> str:
    return name if layer is None else f"{name}_{int(layer)}"


def _check_layer_count(name: str, value: int, max_value: int) -> int:
    value = int(value)
    if not 1 <= value <= int(max_value):
        raise ValueError(f"{name} must be in [1, {max_value}], got {value}")
    return value


def expected_fp8_scale_names(
    *,
    vision_num_layers: int = VIS_L,
    encoder_num_layers: int = ENC_L,
    decoder_num_layers: int = DEC_L,
) -> tuple[str, ...]:
    """Return the static FP8 activation scale contract for this Pi0.5 run.

    These names are the producer/consumer boundaries that calibration owns.
    Decoder scales are shared across diffusion steps because the graph replays
    the same fixed decoder layers with one persistent scale per GEMM site.
    """
    vision_num_layers = _check_layer_count(
        "vision_num_layers", vision_num_layers, VIS_L
    )
    encoder_num_layers = _check_layer_count(
        "encoder_num_layers", encoder_num_layers, ENC_L
    )
    decoder_num_layers = _check_layer_count(
        "decoder_num_layers", decoder_num_layers, DEC_L
    )

    names: list[str] = []
    for i in range(vision_num_layers):
        names.extend(
            (
                _fp8_key("vision_attn_qkv_w", i),
                _fp8_key("vision_attn_o_w", i),
                _fp8_key("vision_ffn_up_w", i),
                _fp8_key("vision_ffn_down_w", i),
            )
        )
    names.append("encoder_multi_modal_projector_w")

    for i in range(encoder_num_layers):
        names.append(_fp8_key("encoder_attn_qkv_w", i))
        if i != encoder_num_layers - 1:
            names.extend(
                (
                    _fp8_key("encoder_attn_o_w", i),
                    _fp8_key("encoder_ffn_gate_up_w", i),
                    _fp8_key("encoder_ffn_down_w", i),
                )
            )

    for i in range(decoder_num_layers):
        names.extend(
            (
                _fp8_key("decoder_attn_qkv_w", i),
                _fp8_key("decoder_attn_o_w", i),
                _fp8_key("decoder_ffn_gate_up_w", i),
                _fp8_key("decoder_ffn_down_w", i),
            )
        )

    return tuple(names)


def fp8_scale_coverage(
    actual_names,
    *,
    vision_num_layers: int = VIS_L,
    encoder_num_layers: int = ENC_L,
    decoder_num_layers: int = DEC_L,
) -> dict[str, object]:
    """Compare collected FP8 scales against the static quant-site contract."""
    expected = expected_fp8_scale_names(
        vision_num_layers=vision_num_layers,
        encoder_num_layers=encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
    )
    actual = tuple(str(name) for name in actual_names)
    expected_set = set(expected)
    actual_set = set(actual)
    return {
        "expected_scale_count": len(expected),
        "actual_scale_count": len(actual_set),
        "expected_scales": expected,
        "actual_scales": tuple(sorted(actual_set)),
        "missing_scales": tuple(name for name in expected if name not in actual_set),
        "unexpected_scales": tuple(
            name for name in sorted(actual_set) if name not in expected_set
        ),
    }


class Pi05PipelineRocm:
    """FlashRT-owned Pi0.5 ROCm pipeline."""

    @staticmethod
    def _vision_fp8_fusion_enabled() -> bool:
        """Return whether SigLIP FP8 boundary fusion is enabled."""
        return os.environ.get("FLASHRT_ROCM_ENABLE_VISION_FP8_FUSION", "0") == "1"

    @staticmethod
    def _aiter_decoder_fp8_gemm_enabled() -> bool:
        """Return whether the AITER small-M decoder FP8 GEMM backend is enabled."""
        return os.environ.get("FLASHRT_ROCM_ENABLE_AITER_DECODER_FP8", "0") == "1"

    expected_fp8_scale_names = staticmethod(expected_fp8_scale_names)
    fp8_scale_coverage_for = staticmethod(fp8_scale_coverage)

    def __init__(
        self,
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = NUM_STEPS_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
        vision_pool_factor: int = 1,
    ):
        self.num_views = int(num_views)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)
        self.num_steps = int(num_steps)
        self.vision_pool_factor = int(vision_pool_factor)
        if self.num_views <= 0:
            raise ValueError(f"num_views must be positive, got {self.num_views}")
        if self.max_prompt_len <= 0:
            raise ValueError(
                f"max_prompt_len must be positive, got {self.max_prompt_len}"
            )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.vision_pool_factor not in (1, 2, 4):
            raise ValueError(
                "vision_pool_factor must be one of {1, 2, 4}; "
                f"got {self.vision_pool_factor}"
            )

        self.vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        pf = self.vision_pool_factor
        self.vision_seq_enc = self.vision_seq // (pf * pf)
        self.encoder_seq_len = self.vision_seq_enc + self.max_prompt_len
        self.total_kv = self.encoder_seq_len + self.chunk_size

        self.bufs = self._allocate_buffers()
        self.fp8_act_scales = {}
        self.fp8_calibrated = False
        self._aiter_scale_tensors = {}
        self._build_rope_table()
        self._rocm_kernels = None
        self._weights = None
        self._runtime_use_fp8 = False
        self._runtime_vision_num_layers = VIS_L
        self._runtime_encoder_num_layers = ENC_L
        self._runtime_decoder_num_layers = DEC_L

    @classmethod
    def with_sdpa_attention(cls, **kwargs):
        """Construct the BF16 pipeline plus the ROCm SDPA attention bridge."""
        from flash_rt.hardware.rocm.attn_backend import RocmSdpaAttnBackend

        pipe = cls(**kwargs)
        pipe.attn = RocmSdpaAttnBackend(
            pipe.num_views,
            pipe.encoder_seq_len,
            pipe.chunk_size,
            num_encoder_layers=ENC_L,
        )
        pipe._attn_ptrs = pipe.attn.get_ptrs()
        return pipe

    def _allocate_buffers(self) -> dict[str, HipBuffer]:
        nv = self.num_views
        vs = self.vision_seq
        vs_enc = self.vision_seq_enc
        es = self.encoder_seq_len
        ds = self.chunk_size
        buffers: dict[str, HipBuffer] = {}

        buffers["observation_images_normalized"] = HipBuffer.device_empty(
            nv * 224 * 224 * 3, BF16
        )
        buffers["vision_patches"] = HipBuffer.device_empty(vs * VIS_PATCH_FLAT, BF16)
        buffers["vision_x"] = HipBuffer.device_empty(vs * VIS_D, BF16)
        buffers["vision_x_norm"] = HipBuffer.device_empty(vs * VIS_D, BF16)
        if self.vision_pool_factor > 1:
            buffers["vision_x_pooled"] = HipBuffer.device_empty(vs_enc * VIS_D, BF16)
        else:
            buffers["vision_x_pooled"] = buffers["vision_x"]
        buffers["vision_QKV"] = HipBuffer.device_empty(vs * 3 * VIS_D, BF16)
        buffers["vision_hidden"] = HipBuffer.device_empty(vs * VIS_H, BF16)
        buffers["vision_pos_embed_expanded"] = HipBuffer.device_empty(vs * VIS_D, BF16)
        buffers["vision_act_fp8"] = HipBuffer.device_empty(vs * VIS_D, FP8_BYTES)
        buffers["vision_act_fp8_large"] = HipBuffer.device_empty(vs * VIS_H, FP8_BYTES)

        buffers["encoder_rope_weights"] = HipBuffer.device_empty(
            es * 2 * ENC_HD // 2, BF16
        )
        buffers["encoder_x"] = HipBuffer.device_empty(es * ENC_D, BF16)
        buffers["encoder_x_norm"] = HipBuffer.device_empty(es * ENC_D, BF16)
        buffers["encoder_QKV"] = HipBuffer.device_empty(
            es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, BF16
        )
        buffers["encoder_hidden"] = HipBuffer.device_empty(es * ENC_H, BF16)
        buffers["encoder_gate_merged"] = HipBuffer.device_empty(
            es * 2 * ENC_H, BF16
        )
        buffers["encoder_gate_buf"] = HipBuffer.device_empty(es * ENC_H, BF16)
        buffers["encoder_act_fp8"] = HipBuffer.device_empty(es * ENC_D, FP8_BYTES)
        buffers["encoder_act_fp8_large"] = HipBuffer.device_empty(
            es * ENC_H, FP8_BYTES
        )

        buffers["decoder_rope_weights"] = HipBuffer.device_empty(ds * 256, BF16)
        buffers["decoder_x"] = HipBuffer.device_empty(ds * DEC_D, BF16)
        buffers["decoder_action_buf"] = HipBuffer.device_empty(ds * ACTION_DIM, BF16)
        buffers["decoder_time_emb"] = HipBuffer.device_empty(
            self.num_steps * ds * DEC_D, BF16
        )
        buffers["decoder_style_attn"] = HipBuffer.device_empty(
            self.num_steps * DEC_L * ds * 3 * DEC_D, BF16
        )
        buffers["decoder_style_ffn"] = HipBuffer.device_empty(
            self.num_steps * DEC_L * ds * 3 * DEC_D, BF16
        )
        buffers["decoder_style_final"] = HipBuffer.device_empty(
            self.num_steps * ds * 3 * DEC_D, BF16
        )
        buffers["decoder_QKV"] = HipBuffer.device_empty(
            ds * (DEC_NH + 2 * DEC_NKV) * DEC_HD, BF16
        )
        buffers["decoder_hidden"] = HipBuffer.device_empty(ds * DEC_H, BF16)
        buffers["decoder_gate_merged"] = HipBuffer.device_empty(
            ds * 2 * DEC_H, BF16
        )
        buffers["decoder_gate_buf"] = HipBuffer.device_empty(ds * DEC_H, BF16)
        buffers["decoder_act_fp8"] = HipBuffer.device_empty(ds * DEC_D, FP8_BYTES)
        buffers["decoder_act_fp8_large"] = HipBuffer.device_empty(
            ds * DEC_H, FP8_BYTES
        )
        buffers["decoder_rms_ones"] = HipBuffer.from_numpy(
            np.ones((DEC_D,), dtype=BF16_NP)
        )
        buffers["diffusion_noise"] = HipBuffer.device_empty(ds * ACTION_DIM, BF16)
        buffers["x_normed_buf"] = HipBuffer.device_empty(ds * DEC_D, BF16)
        buffers["gate_buf"] = HipBuffer.device_empty(ds * DEC_D, BF16)
        buffers["fp8_dynamic_partial"] = HipBuffer.device_empty(4096, FP32)

        if self._aiter_decoder_fp8_gemm_enabled():
            import torch

            buffers["decoder_QKV"] = _TorchTensorBuffer(
                (ds, (DEC_NH + 2 * DEC_NKV) * DEC_HD), torch.bfloat16
            )
            buffers["decoder_gate_merged"] = _TorchTensorBuffer(
                (ds, 2 * DEC_H), torch.bfloat16
            )
            buffers["decoder_act_fp8"] = _TorchTensorBuffer(
                (ds, DEC_D), torch.float8_e4m3fnuz
            )
            buffers["decoder_act_fp8_large"] = _TorchTensorBuffer(
                (ds, DEC_H), torch.float8_e4m3fnuz
            )
            buffers["x_normed_buf"] = _TorchTensorBuffer(
                (ds, DEC_D), torch.bfloat16
            )

        return buffers

    def _build_rope_table(self) -> None:
        max_pos = self.encoder_seq_len + self.chunk_size
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]
        cos = np.cos(phase).astype(np.float32)
        sin = np.sin(phase).astype(np.float32)
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)

        enc = interleaved[: self.encoder_seq_len].astype(BF16_NP)
        dec = interleaved[
            self.encoder_seq_len : self.encoder_seq_len + self.chunk_size
        ].astype(BF16_NP)
        self.bufs["encoder_rope_weights"].upload(enc)
        self.bufs["decoder_rope_weights"].upload(dec)

    @property
    def input_images_buf(self) -> HipBuffer:
        return self.bufs["observation_images_normalized"]

    @property
    def input_noise_buf(self) -> HipBuffer:
        return self.bufs["diffusion_noise"]

    @property
    def input_encoder_x_buf(self) -> HipBuffer:
        return self.bufs["encoder_x"]

    def configure_runtime(
        self,
        rocm_kernels,
        weights,
        *,
        use_fp8: bool = False,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
    ) -> None:
        """Bind kernels, weights, and layer counts for ``forward``.

        Frontends own framework-specific weight loading. The pipeline owns the
        steady-state execution contract, so the hot path can replay the same
        graph without passing Python-side execution arguments.
        """
        self._rocm_kernels = rocm_kernels
        self._weights = weights
        self._runtime_use_fp8 = bool(use_fp8)
        self._runtime_vision_num_layers = _check_layer_count(
            "vision_num_layers", vision_num_layers, VIS_L
        )
        self._runtime_encoder_num_layers = _check_layer_count(
            "encoder_num_layers", encoder_num_layers, ENC_L
        )
        self._runtime_decoder_num_layers = _check_layer_count(
            "decoder_num_layers", decoder_num_layers, DEC_L
        )

    def set_language_embeds(self, lang_embeds_np) -> None:
        """Store prompt embeddings and copy them into ``encoder_x``.

        The encoder residual stream overwrites the language slice in-place.
        Keeping a persistent device copy lets every graph replay restore the
        exact prompt embedding bytes before running the encoder.
        """
        arr = np.ascontiguousarray(lang_embeds_np)
        if arr.ndim != 2 or arr.shape[1] != ENC_D:
            raise ValueError(
                f"language embeds must have shape (prompt_len, {ENC_D}), got {arr.shape}"
            )
        if arr.shape[0] > self.max_prompt_len:
            raise ValueError(
                f"prompt_len {arr.shape[0]} exceeds max_prompt_len {self.max_prompt_len}"
            )
        if hasattr(self, "_lang_embeds_buf") and self._lang_embeds_buf.nbytes == arr.nbytes:
            self._lang_embeds_buf.upload(arr)
        else:
            self._lang_embeds_buf = HipBuffer.from_numpy(arr)
        self._current_prompt_len = int(arr.shape[0])
        self._copy_lang_embeds_to_encoder_x()

    def _copy_lang_embeds_to_encoder_x(self, stream: int = 0) -> None:
        if not hasattr(self, "_lang_embeds_buf"):
            return
        start_byte = self.vision_seq_enc * ENC_D * 2
        dst_ptr = ctypes.c_void_p(self.bufs["encoder_x"].ptr.value + start_byte)
        src_ptr = self._lang_embeds_buf.ptr
        if stream:
            _check(
                _hip.hipMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    self._lang_embeds_buf.nbytes,
                    3,
                    ctypes.c_void_p(int(stream)),
                ),
                "hipMemcpyAsync language embeds",
            )
        else:
            _check(
                _hip.hipMemcpy(
                    dst_ptr,
                    src_ptr,
                    self._lang_embeds_buf.nbytes,
                    3,
                ),
                "hipMemcpy language embeds",
            )

    def copy_device_to_buffer(self, dst_name: str, src_ptr: int, nbytes: int) -> None:
        dst = self.bufs[dst_name]
        if nbytes > dst.nbytes:
            raise ValueError(f"{dst_name} has {dst.nbytes} bytes, copy asked {nbytes}")
        _check(
            _hip.hipMemcpy(dst.ptr, ctypes.c_void_p(int(src_ptr)), nbytes, 3),
            f"hipMemcpy D2D into {dst_name}",
        )

    def set_fp8_act_scale(self, name: str, value: float) -> None:
        """Set a persistent static FP8 activation scale for owned-buffer paths."""
        import torch

        self.fp8_act_scales[str(name)] = torch.tensor(
            [float(value)], device="cuda", dtype=torch.float32
        )

    def fp8_scale_coverage(
        self,
        *,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
    ) -> dict[str, object]:
        """Compare this pipeline's collected scales with the static contract."""
        return fp8_scale_coverage(
            self.fp8_act_scales,
            vision_num_layers=vision_num_layers,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
        )

    def _fp8_scale_tensor(self, name: str):
        import torch

        scale = self.fp8_act_scales.get(name)
        if scale is None:
            scale = torch.zeros((1,), device="cuda", dtype=torch.float32)
            self.fp8_act_scales[str(name)] = scale
        return scale

    def _fp8_scale_ptr(self, name: str, *, require_static: bool = True) -> int:
        scale = self.fp8_act_scales.get(name)
        if scale is None:
            if require_static:
                raise RuntimeError(
                    f"missing static FP8 activation scale for {name!r}; "
                    "calibrate or call set_fp8_act_scale() before FP8 inference"
                )
            scale = self._fp8_scale_tensor(name)
        return int(scale.data_ptr())

    def _fp8_quantize_bf16_ptr(
        self,
        rocm_kernels,
        *,
        src_ptr: int,
        dst_buf: str,
        scale_name: str,
        n: int,
        stream: int = 0,
    ) -> int:
        scale_ptr = self._fp8_scale_ptr(
            scale_name, require_static=self.fp8_calibrated
        )
        if self.fp8_calibrated:
            rocm_kernels.quantize_bf16_to_fp8_e4m3fnuz_ptr(
                int(src_ptr),
                scale_ptr,
                self.bufs[dst_buf].ptr.value,
                int(n),
                stream,
            )
        else:
            partial_count = max(1, min(4096, (int(n) + 255) // 256))
            rocm_kernels.dynamic_quantize_bf16_to_fp8_e4m3fnuz_ptr(
                int(src_ptr),
                self.bufs[dst_buf].ptr.value,
                scale_ptr,
                self.bufs["fp8_dynamic_partial"].ptr.value,
                partial_count,
                int(n),
                stream,
            )
        return scale_ptr

    def _weight_fp8_ptrs(
        self, weights, name: str, layer: int | None = None
    ) -> tuple[int, int]:
        fp8_weights = weights.get("fp8")
        if fp8_weights is None:
            raise RuntimeError(
                "weights['fp8'] is missing; rebuild ROCm weights with include_fp8=True"
            )
        key = _fp8_key(name, layer)
        if key not in fp8_weights:
            raise RuntimeError(f"missing FP8 weight {key!r}")
        w_fp8, w_scale = fp8_weights[key]
        return int(w_fp8.data_ptr()), int(w_scale.data_ptr())

    def _aiter_decoder_fp8_linear(
        self,
        *,
        act_fp8_buf: str,
        act_scale_name: str,
        weight_name: str,
        out_bf16_buf: str,
        weights,
        layer: int | None,
    ) -> bool:
        if not self._aiter_decoder_fp8_gemm_enabled():
            return False
        if not self.fp8_calibrated:
            return False
        if not weight_name.startswith("decoder_"):
            return False
        act_buf = self.bufs.get(act_fp8_buf)
        out_buf = self.bufs.get(out_bf16_buf)
        act = getattr(act_buf, "tensor", None)
        out = getattr(out_buf, "tensor", None)
        if act is None or out is None:
            return False
        fp8_weights = weights.get("fp8")
        if fp8_weights is None:
            raise RuntimeError(
                "weights['fp8'] is missing; rebuild ROCm weights with include_fp8=True"
            )
        key = _fp8_key(weight_name, layer)
        w_fp8, w_scale = fp8_weights[key]
        x_scale = self._aiter_expanded_scale(
            ("act", act_scale_name, int(act.shape[0])),
            self._fp8_scale_tensor(act_scale_name),
            (int(act.shape[0]), 1),
        )
        weight_scale = self._aiter_expanded_scale(
            ("weight", key, int(out.shape[1])),
            w_scale,
            (int(out.shape[1]),),
        )
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ck

        gemm_a8w8_ck(
            act,
            w_fp8,
            x_scale,
            weight_scale,
            out,
            None,
            0,
        )
        return True

    def _aiter_expanded_scale(self, cache_key, scalar, shape):
        import torch

        cached = self._aiter_scale_tensors.get(cache_key)
        if cached is None:
            cached = torch.empty(tuple(int(x) for x in shape), device="cuda", dtype=torch.float32)
            cached.copy_(scalar.reshape(1).expand_as(cached))
            self._aiter_scale_tensors[cache_key] = cached
        return cached

    def _fp8_linear(
        self,
        rocm_kernels,
        *,
        act_bf16_ptr: int,
        act_fp8_buf: str,
        act_scale_name: str,
        weight_name: str,
        out_bf16_buf: str,
        m: int,
        n: int,
        k: int,
        weights,
        layer: int | None = None,
        bias_ptr: int = 0,
        stream: int = 0,
    ) -> None:
        act_scale_ptr = self._fp8_quantize_bf16_ptr(
            rocm_kernels,
            src_ptr=int(act_bf16_ptr),
            dst_buf=act_fp8_buf,
            scale_name=act_scale_name,
            n=int(m) * int(k),
            stream=stream,
        )
        if bias_ptr == 0 and self._aiter_decoder_fp8_linear(
            act_fp8_buf=act_fp8_buf,
            act_scale_name=act_scale_name,
            weight_name=weight_name,
            out_bf16_buf=out_bf16_buf,
            weights=weights,
            layer=layer,
        ):
            return
        w_fp8_ptr, w_scale_ptr = self._weight_fp8_ptrs(weights, weight_name, layer)
        rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
            self.bufs[act_fp8_buf].ptr.value,
            w_fp8_ptr,
            act_scale_ptr,
            w_scale_ptr,
            int(bias_ptr),
            self.bufs[out_bf16_buf].ptr.value,
            int(m),
            int(n),
            int(k),
            stream,
        )

    def _fp8_linear_prequantized(
        self,
        rocm_kernels,
        *,
        act_fp8_buf: str,
        act_scale_name: str,
        weight_name: str,
        out_bf16_buf: str,
        m: int,
        n: int,
        k: int,
        weights,
        layer: int | None = None,
        bias_ptr: int = 0,
        stream: int = 0,
    ) -> None:
        if bias_ptr == 0 and self._aiter_decoder_fp8_linear(
            act_fp8_buf=act_fp8_buf,
            act_scale_name=act_scale_name,
            weight_name=weight_name,
            out_bf16_buf=out_bf16_buf,
            weights=weights,
            layer=layer,
        ):
            return
        act_scale_ptr = self._fp8_scale_ptr(act_scale_name)
        w_fp8_ptr, w_scale_ptr = self._weight_fp8_ptrs(weights, weight_name, layer)
        rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
            self.bufs[act_fp8_buf].ptr.value,
            w_fp8_ptr,
            act_scale_ptr,
            w_scale_ptr,
            int(bias_ptr),
            self.bufs[out_bf16_buf].ptr.value,
            int(m),
            int(n),
            int(k),
            stream,
        )

    def upload_precomputed_decoder_styles(self, precomputed: dict[str, np.ndarray]) -> None:
        """Upload BF16 bit-pattern style/time buffers from a frontend helper."""
        self.bufs["decoder_time_emb"].upload(np.ascontiguousarray(precomputed["time_emb"]))
        self.bufs["decoder_style_attn"].upload(
            np.ascontiguousarray(precomputed["style_attn"])
        )
        self.bufs["decoder_style_ffn"].upload(
            np.ascontiguousarray(precomputed["style_ffn"])
        )
        self.bufs["decoder_style_final"].upload(
            np.ascontiguousarray(precomputed["style_final"])
        )
        self._decoder_styles_uploaded = True

    def _style_slice_ptr(self, buf_name: str, step: int, layer: int | None = None) -> int:
        base = self.bufs[buf_name].ptr.value
        ds = self.chunk_size
        if buf_name == "decoder_time_emb":
            return base + int(step) * ds * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + int(step) * ds * 3 * DEC_D * 2
        if layer is None:
            raise ValueError(f"{buf_name} requires a layer")
        per_layer = ds * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + int(step) * per_step + int(layer) * per_layer

    def _enc_kv_layer_ptrs(self, layer: int, offset_tokens: int = 0) -> tuple[int, int]:
        attn_ptrs = getattr(self, "_attn_ptrs", None)
        if attn_ptrs is None:
            raise RuntimeError("no attention backend is attached")
        token_offset_bytes = offset_tokens * ENC_NKV * ENC_HD * 2
        return (
            int(attn_ptrs["enc_K"])
            + layer * int(attn_ptrs["enc_k_layer_stride_bytes"])
            + token_offset_bytes,
            int(attn_ptrs["enc_V"])
            + layer * int(attn_ptrs["enc_v_layer_stride_bytes"])
            + token_offset_bytes,
        )

    def vision_patch_im2col(self, rocm_kernels, stream: int = 0) -> None:
        """Run vision patch extraction into ``vision_patches``.

        Input must already be normalized BF16/NHWC in
        ``observation_images_normalized``. The kernel is a bitwise 16-bit copy,
        matching the RTX patch embedding im2col order.
        """
        rocm_kernels.patch_im2col_ptr(
            self.bufs["observation_images_normalized"].ptr.value,
            self.bufs["vision_patches"].ptr.value,
            self.num_views,
            stream,
        )

    def vision_patch_embed(
        self,
        rocm_kernels,
        patch_embedding_w_ptr: int,
        patch_embedding_b_ptr: int = 0,
        stream: int = 0,
    ) -> None:
        """Run patch im2col + BF16 patch projection into ``vision_x``.

        ``patch_embedding_w_ptr`` must point to a BF16 row-major weight in
        ``[VIS_D, VIS_PATCH_FLAT]`` layout, matching the ROCm raw linear ABI.
        This method intentionally stops before bias/position fusion; that will
        be consumed by the following projection.
        """
        self.vision_patch_im2col(rocm_kernels, stream=stream)
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["vision_patches"].ptr.value,
            int(patch_embedding_w_ptr),
            int(patch_embedding_b_ptr),
            self.bufs["vision_x"].ptr.value,
            self.vision_seq,
            VIS_D,
            VIS_PATCH_FLAT,
            stream,
        )

    def vision_patch_bias_pos(
        self,
        rocm_kernels,
        patch_embedding_b_ptr: int,
        position_embedding_ptr: int,
        stream: int = 0,
    ) -> None:
        """Add patch projection bias and repeated SigLIP position embedding."""
        rocm_kernels.patch_embed_bias_pos_bf16_ptr(
            self.bufs["vision_x"].ptr.value,
            int(patch_embedding_b_ptr),
            int(position_embedding_ptr),
            self.vision_seq,
            VIS_D,
            VIS_SEQ_PER_VIEW,
            stream,
        )

    def vision_patch_embed_with_bias_pos(
        self,
        rocm_kernels,
        patch_embedding_w_ptr: int,
        patch_embedding_b_ptr: int,
        position_embedding_ptr: int,
        stream: int = 0,
    ) -> None:
        """Run patch embedding into ``vision_x``."""
        self.vision_patch_embed(
            rocm_kernels,
            patch_embedding_w_ptr,
            0,
            stream=stream,
        )
        self.vision_patch_bias_pos(
            rocm_kernels,
            patch_embedding_b_ptr,
            position_embedding_ptr,
            stream=stream,
        )

    def vision_pre_attn_layer_norm(
        self,
        rocm_kernels,
        norm_weight_ptr: int,
        norm_bias_ptr: int,
        eps: float = 1e-5,
        stream: int = 0,
    ) -> None:
        """LayerNorm ``vision_x`` into ``vision_x_norm`` for SigLIP attention."""
        rocm_kernels.layer_norm_bf16_ptr(
            self.bufs["vision_x"].ptr.value,
            int(norm_weight_ptr),
            int(norm_bias_ptr),
            self.bufs["vision_x_norm"].ptr.value,
            self.vision_seq,
            VIS_D,
            float(eps),
            stream,
        )

    def vision_qkv_bf16(
        self,
        rocm_kernels,
        qkv_weight_ptr: int,
        qkv_bias_ptr: int,
        stream: int = 0,
    ) -> None:
        """Project normalized SigLIP tokens into packed QKV."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["vision_x_norm"].ptr.value,
            int(qkv_weight_ptr),
            int(qkv_bias_ptr),
            self.bufs["vision_QKV"].ptr.value,
            self.vision_seq,
            3 * VIS_D,
            VIS_D,
            stream,
        )

    def vision_qkv_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        prequantized: bool = False,
        stream: int = 0,
    ) -> None:
        """Project normalized SigLIP tokens into packed QKV with static FP8."""
        if prequantized:
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="vision_act_fp8",
                act_scale_name=_fp8_key("vision_attn_qkv_w", layer),
                weight_name="vision_attn_qkv_w",
                out_bf16_buf="vision_QKV",
                m=self.vision_seq,
                n=3 * VIS_D,
                k=VIS_D,
                weights=weights,
                layer=layer,
                bias_ptr=_weight_ptr(weights, "vision_attn_qkv_b", layer),
                stream=stream,
            )
            return

        self._fp8_linear(
            rocm_kernels,
            act_bf16_ptr=self.bufs["vision_x_norm"].ptr.value,
            act_fp8_buf="vision_act_fp8",
            act_scale_name=_fp8_key("vision_attn_qkv_w", layer),
            weight_name="vision_attn_qkv_w",
            out_bf16_buf="vision_QKV",
            m=self.vision_seq,
            n=3 * VIS_D,
            k=VIS_D,
            weights=weights,
            layer=layer,
            bias_ptr=_weight_ptr(weights, "vision_attn_qkv_b", layer),
            stream=stream,
        )

    def vision_qkv_split(
        self,
        rocm_kernels,
        q_ptr: int,
        k_ptr: int,
        v_ptr: int,
        stream: int = 0,
    ) -> None:
        """Split packed SigLIP QKV into attention backend buffers."""
        rocm_kernels.qkv_split_bf16_ptr(
            self.bufs["vision_QKV"].ptr.value,
            int(q_ptr),
            int(k_ptr),
            int(v_ptr),
            self.vision_seq,
            VIS_D,
            VIS_D,
            VIS_D,
            stream,
        )

    def vision_qkv_split_to_attn(self, rocm_kernels, stream: int = 0) -> None:
        """Split packed SigLIP QKV directly into the attached attention backend."""
        attn_ptrs = getattr(self, "_attn_ptrs", None)
        if attn_ptrs is None:
            raise RuntimeError("no attention backend is attached")
        self.vision_qkv_split(
            rocm_kernels,
            attn_ptrs["vis_Q"],
            attn_ptrs["vis_K"],
            attn_ptrs["vis_V"],
            stream=stream,
        )

    def vision_attention(self, stream: int = 0) -> int:
        """Run attached SigLIP attention and return its BF16 output pointer."""
        attn = getattr(self, "attn", None)
        if attn is None:
            raise RuntimeError("no attention backend is attached")
        return int(attn.run("siglip", 0, q_seq=VIS_SEQ_PER_VIEW, stream=stream))

    def vision_attn_output_residual_norm(
        self,
        rocm_kernels,
        attn_out_ptr: int,
        out_weight_ptr: int,
        out_bias_ptr: int,
        next_norm_weight_ptr: int,
        next_norm_bias_ptr: int,
        eps: float = 1e-5,
        stream: int = 0,
    ) -> None:
        """Project SigLIP attention output, then fuse residual and next LN."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            int(attn_out_ptr),
            int(out_weight_ptr),
            0,
            self.bufs["vision_x_norm"].ptr.value,
            self.vision_seq,
            VIS_D,
            VIS_D,
            stream,
        )
        self.vision_bias_residual_layer_norm(
            rocm_kernels,
            "vision_x_norm",
            out_bias_ptr,
            next_norm_weight_ptr,
            next_norm_bias_ptr,
            eps,
            stream=stream,
        )

    def vision_attn_output_residual_norm_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        attn_out_ptr: int,
        next_norm_weight_ptr: int,
        next_norm_bias_ptr: int,
        eps: float = 1e-5,
        stream: int = 0,
    ) -> None:
        """FP8 attention output projection, then BF16 residual + next LN."""
        self._fp8_linear(
            rocm_kernels,
            act_bf16_ptr=int(attn_out_ptr),
            act_fp8_buf="vision_act_fp8",
            act_scale_name=_fp8_key("vision_attn_o_w", layer),
            weight_name="vision_attn_o_w",
            out_bf16_buf="vision_x_norm",
            m=self.vision_seq,
            n=VIS_D,
            k=VIS_D,
            weights=weights,
            layer=layer,
            bias_ptr=0,
            stream=stream,
        )
        if self.fp8_calibrated and self._vision_fp8_fusion_enabled():
            rocm_kernels.bias_residual_layer_norm_fp8_e4m3fnuz_ptr(
                self.bufs["vision_x"].ptr.value,
                self.bufs["vision_x_norm"].ptr.value,
                _weight_ptr(weights, "vision_attn_o_b", layer),
                int(next_norm_weight_ptr),
                int(next_norm_bias_ptr),
                self.bufs["vision_act_fp8"].ptr.value,
                self._fp8_scale_ptr(_fp8_key("vision_ffn_up_w", layer)),
                self.vision_seq,
                VIS_D,
                float(eps),
                stream,
            )
        else:
            self.vision_bias_residual_layer_norm(
                rocm_kernels,
                "vision_x_norm",
                _weight_ptr(weights, "vision_attn_o_b", layer),
                next_norm_weight_ptr,
                next_norm_bias_ptr,
                eps,
                stream=stream,
            )

    def vision_ffn_up_gelu(
        self,
        rocm_kernels,
        up_weight_ptr: int,
        up_bias_ptr: int,
        stream: int = 0,
    ) -> None:
        """Run SigLIP FFN up projection, bias add, and GELU into hidden."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["vision_x_norm"].ptr.value,
            int(up_weight_ptr),
            0,
            self.bufs["vision_hidden"].ptr.value,
            self.vision_seq,
            VIS_H,
            VIS_D,
            stream,
        )
        rocm_kernels.add_bias_bf16_ptr(
            self.bufs["vision_hidden"].ptr.value,
            int(up_bias_ptr),
            self.vision_seq,
            VIS_H,
            stream,
        )
        rocm_kernels.gelu_tanh_bf16_ptr(
            self.bufs["vision_hidden"].ptr.value,
            self.bufs["vision_hidden"].ptr.value,
            self.vision_seq * VIS_H,
            stream,
        )

    def vision_ffn_up_gelu_fp8(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run SigLIP FFN up projection with FP8 GEMM, then BF16 GELU."""
        if self.fp8_calibrated and self._vision_fp8_fusion_enabled():
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="vision_act_fp8",
                act_scale_name=_fp8_key("vision_ffn_up_w", layer),
                weight_name="vision_ffn_up_w",
                out_bf16_buf="vision_hidden",
                m=self.vision_seq,
                n=VIS_H,
                k=VIS_D,
                weights=weights,
                layer=layer,
                bias_ptr=_weight_ptr(weights, "vision_ffn_up_b", layer),
                stream=stream,
            )
            rocm_kernels.gelu_tanh_quantize_fp8_e4m3fnuz_ptr(
                self.bufs["vision_hidden"].ptr.value,
                self._fp8_scale_ptr(_fp8_key("vision_ffn_down_w", layer)),
                self.bufs["vision_act_fp8_large"].ptr.value,
                self.vision_seq * VIS_H,
                stream,
            )
        else:
            self._fp8_linear(
                rocm_kernels,
                act_bf16_ptr=self.bufs["vision_x_norm"].ptr.value,
                act_fp8_buf="vision_act_fp8",
                act_scale_name=_fp8_key("vision_ffn_up_w", layer),
                weight_name="vision_ffn_up_w",
                out_bf16_buf="vision_hidden",
                m=self.vision_seq,
                n=VIS_H,
                k=VIS_D,
                weights=weights,
                layer=layer,
                bias_ptr=_weight_ptr(weights, "vision_ffn_up_b", layer),
                stream=stream,
            )
            rocm_kernels.gelu_tanh_bf16_ptr(
                self.bufs["vision_hidden"].ptr.value,
                self.bufs["vision_hidden"].ptr.value,
                self.vision_seq * VIS_H,
                stream,
            )

    def vision_ffn_down_residual_norm(
        self,
        rocm_kernels,
        down_weight_ptr: int,
        down_bias_ptr: int,
        next_norm_weight_ptr: int,
        next_norm_bias_ptr: int,
        eps: float = 1e-5,
        stream: int = 0,
    ) -> None:
        """Run SigLIP FFN down projection and fuse residual + next LN."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["vision_hidden"].ptr.value,
            int(down_weight_ptr),
            0,
            self.bufs["vision_x_norm"].ptr.value,
            self.vision_seq,
            VIS_D,
            VIS_H,
            stream,
        )
        self.vision_bias_residual_layer_norm(
            rocm_kernels,
            "vision_x_norm",
            down_bias_ptr,
            next_norm_weight_ptr,
            next_norm_bias_ptr,
            eps,
            stream=stream,
        )

    def vision_ffn_down_residual_norm_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        next_norm_weight_ptr: int,
        next_norm_bias_ptr: int,
        next_layer: int,
        eps: float = 1e-5,
        stream: int = 0,
    ) -> None:
        """Run SigLIP FFN down FP8 GEMM and fuse residual + next LN."""
        if self.fp8_calibrated and self._vision_fp8_fusion_enabled():
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="vision_act_fp8_large",
                act_scale_name=_fp8_key("vision_ffn_down_w", layer),
                weight_name="vision_ffn_down_w",
                out_bf16_buf="vision_x_norm",
                m=self.vision_seq,
                n=VIS_D,
                k=VIS_H,
                weights=weights,
                layer=layer,
                bias_ptr=0,
                stream=stream,
            )
            rocm_kernels.bias_residual_layer_norm_fp8_e4m3fnuz_ptr(
                self.bufs["vision_x"].ptr.value,
                self.bufs["vision_x_norm"].ptr.value,
                _weight_ptr(weights, "vision_ffn_down_b", layer),
                int(next_norm_weight_ptr),
                int(next_norm_bias_ptr),
                self.bufs["vision_act_fp8"].ptr.value,
                self._fp8_scale_ptr(_fp8_key("vision_attn_qkv_w", next_layer)),
                self.vision_seq,
                VIS_D,
                float(eps),
                stream,
            )
        else:
            self._fp8_linear(
                rocm_kernels,
                act_bf16_ptr=self.bufs["vision_hidden"].ptr.value,
                act_fp8_buf="vision_act_fp8_large",
                act_scale_name=_fp8_key("vision_ffn_down_w", layer),
                weight_name="vision_ffn_down_w",
                out_bf16_buf="vision_x_norm",
                m=self.vision_seq,
                n=VIS_D,
                k=VIS_H,
                weights=weights,
                layer=layer,
                bias_ptr=0,
                stream=stream,
            )
            self.vision_bias_residual_layer_norm(
                rocm_kernels,
                "vision_x_norm",
                _weight_ptr(weights, "vision_ffn_down_b", layer),
                next_norm_weight_ptr,
                next_norm_bias_ptr,
                eps,
                stream=stream,
            )

    def vision_ffn_down_residual(
        self,
        rocm_kernels,
        down_weight_ptr: int,
        down_bias_ptr: int,
        stream: int = 0,
    ) -> None:
        """Run final SigLIP FFN down projection and residual without next LN."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["vision_hidden"].ptr.value,
            int(down_weight_ptr),
            0,
            self.bufs["vision_x_norm"].ptr.value,
            self.vision_seq,
            VIS_D,
            VIS_H,
            stream,
        )
        rocm_kernels.bias_residual_bf16_ptr(
            self.bufs["vision_x"].ptr.value,
            self.bufs["vision_x_norm"].ptr.value,
            int(down_bias_ptr),
            self.vision_seq,
            VIS_D,
            stream,
        )

    def vision_ffn_down_residual_fp8(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run final SigLIP FFN down FP8 GEMM and residual without next LN."""
        if self.fp8_calibrated and self._vision_fp8_fusion_enabled():
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="vision_act_fp8_large",
                act_scale_name=_fp8_key("vision_ffn_down_w", layer),
                weight_name="vision_ffn_down_w",
                out_bf16_buf="vision_x_norm",
                m=self.vision_seq,
                n=VIS_D,
                k=VIS_H,
                weights=weights,
                layer=layer,
                bias_ptr=0,
                stream=stream,
            )
        else:
            self._fp8_linear(
                rocm_kernels,
                act_bf16_ptr=self.bufs["vision_hidden"].ptr.value,
                act_fp8_buf="vision_act_fp8_large",
                act_scale_name=_fp8_key("vision_ffn_down_w", layer),
                weight_name="vision_ffn_down_w",
                out_bf16_buf="vision_x_norm",
                m=self.vision_seq,
                n=VIS_D,
                k=VIS_H,
                weights=weights,
                layer=layer,
                bias_ptr=0,
                stream=stream,
            )
        rocm_kernels.bias_residual_bf16_ptr(
            self.bufs["vision_x"].ptr.value,
            self.bufs["vision_x_norm"].ptr.value,
            _weight_ptr(weights, "vision_ffn_down_b", layer),
            self.vision_seq,
            VIS_D,
            stream,
        )

    def vision_layer_bf16(
        self,
        rocm_kernels,
        *,
        qkv_weight_ptr: int,
        qkv_bias_ptr: int,
        attn_o_weight_ptr: int,
        attn_o_bias_ptr: int,
        pre_ffn_norm_weight_ptr: int,
        pre_ffn_norm_bias_ptr: int,
        ffn_up_weight_ptr: int,
        ffn_up_bias_ptr: int,
        ffn_down_weight_ptr: int,
        ffn_down_bias_ptr: int,
        next_pre_attn_norm_weight_ptr: int = 0,
        next_pre_attn_norm_bias_ptr: int = 0,
        is_last: bool = False,
        stream: int = 0,
    ) -> None:
        """Run one BF16 SigLIP layer once ``vision_x_norm`` is pre-attn LN."""
        self.vision_qkv_bf16(rocm_kernels, qkv_weight_ptr, qkv_bias_ptr, stream=stream)
        self.vision_qkv_split_to_attn(rocm_kernels, stream=stream)
        attn_out_ptr = self.vision_attention(stream=stream)
        self.vision_attn_output_residual_norm(
            rocm_kernels,
            attn_out_ptr,
            attn_o_weight_ptr,
            attn_o_bias_ptr,
            pre_ffn_norm_weight_ptr,
            pre_ffn_norm_bias_ptr,
            stream=stream,
        )
        self.vision_ffn_up_gelu(
            rocm_kernels, ffn_up_weight_ptr, ffn_up_bias_ptr, stream=stream
        )
        if is_last:
            self.vision_ffn_down_residual(
                rocm_kernels,
                ffn_down_weight_ptr,
                ffn_down_bias_ptr,
                stream=stream,
            )
        else:
            if not next_pre_attn_norm_weight_ptr or not next_pre_attn_norm_bias_ptr:
                raise ValueError(
                    "next_pre_attn_norm_weight_ptr and "
                    "next_pre_attn_norm_bias_ptr are required unless is_last=True"
                )
            self.vision_ffn_down_residual_norm(
                rocm_kernels,
                ffn_down_weight_ptr,
                ffn_down_bias_ptr,
                next_pre_attn_norm_weight_ptr,
                next_pre_attn_norm_bias_ptr,
                stream=stream,
            )

    def vision_layer_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        next_pre_attn_norm_weight_ptr: int = 0,
        next_pre_attn_norm_bias_ptr: int = 0,
        is_last: bool = False,
        prequantized_qkv: bool = False,
        stream: int = 0,
    ) -> None:
        """Run one SigLIP layer with FP8 GEMMs at the NV quantization sites."""
        self.vision_qkv_fp8(
            rocm_kernels,
            weights,
            layer,
            prequantized=prequantized_qkv,
            stream=stream,
        )
        self.vision_qkv_split_to_attn(rocm_kernels, stream=stream)
        attn_out_ptr = self.vision_attention(stream=stream)
        self.vision_attn_output_residual_norm_fp8(
            rocm_kernels,
            weights,
            layer,
            attn_out_ptr,
            _weight_ptr(weights, "vision_pre_ffn_norm_w", layer),
            _weight_ptr(weights, "vision_pre_ffn_norm_b", layer),
            stream=stream,
        )
        self.vision_ffn_up_gelu_fp8(rocm_kernels, weights, layer, stream=stream)
        if is_last:
            self.vision_ffn_down_residual_fp8(
                rocm_kernels, weights, layer, stream=stream
            )
        else:
            if not next_pre_attn_norm_weight_ptr or not next_pre_attn_norm_bias_ptr:
                raise ValueError(
                    "next_pre_attn_norm_weight_ptr and "
                    "next_pre_attn_norm_bias_ptr are required unless is_last=True"
                )
            self.vision_ffn_down_residual_norm_fp8(
                rocm_kernels,
                weights,
                layer,
                next_pre_attn_norm_weight_ptr,
                next_pre_attn_norm_bias_ptr,
                layer + 1,
                stream=stream,
            )

    def vision_encoder_bf16(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        stream: int = 0,
    ) -> None:
        """Run the BF16 SigLIP vision encoder with a ROCm weight table."""
        if not 1 <= int(vision_num_layers) <= VIS_L:
            raise ValueError(
                f"vision_num_layers must be in [1, {VIS_L}], got {vision_num_layers}"
            )
        n_layers = int(vision_num_layers)

        self.vision_patch_embed_with_bias_pos(
            rocm_kernels,
            _weight_ptr(weights, "vision_patch_embedding_w"),
            _weight_ptr(weights, "vision_patch_embedding_b"),
            _weight_ptr(weights, "vision_position_embedding"),
            stream=stream,
        )
        self.vision_pre_attn_layer_norm(
            rocm_kernels,
            _weight_ptr(weights, "vision_pre_attn_norm_w", 0),
            _weight_ptr(weights, "vision_pre_attn_norm_b", 0),
            stream=stream,
        )

        for i in range(n_layers):
            is_last = i == n_layers - 1
            self.vision_layer_bf16(
                rocm_kernels,
                qkv_weight_ptr=_weight_ptr(weights, "vision_attn_qkv_w", i),
                qkv_bias_ptr=_weight_ptr(weights, "vision_attn_qkv_b", i),
                attn_o_weight_ptr=_weight_ptr(weights, "vision_attn_o_w", i),
                attn_o_bias_ptr=_weight_ptr(weights, "vision_attn_o_b", i),
                pre_ffn_norm_weight_ptr=_weight_ptr(
                    weights, "vision_pre_ffn_norm_w", i
                ),
                pre_ffn_norm_bias_ptr=_weight_ptr(
                    weights, "vision_pre_ffn_norm_b", i
                ),
                ffn_up_weight_ptr=_weight_ptr(weights, "vision_ffn_up_w", i),
                ffn_up_bias_ptr=_weight_ptr(weights, "vision_ffn_up_b", i),
                ffn_down_weight_ptr=_weight_ptr(weights, "vision_ffn_down_w", i),
                ffn_down_bias_ptr=_weight_ptr(weights, "vision_ffn_down_b", i),
                next_pre_attn_norm_weight_ptr=(
                    0
                    if is_last
                    else _weight_ptr(weights, "vision_pre_attn_norm_w", i + 1)
                ),
                next_pre_attn_norm_bias_ptr=(
                    0
                    if is_last
                    else _weight_ptr(weights, "vision_pre_attn_norm_b", i + 1)
                ),
                is_last=is_last,
                stream=stream,
            )

    def vision_encoder_fp8(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        stream: int = 0,
    ) -> None:
        """Run the SigLIP vision encoder with static FP8 GEMM sites."""
        if not 1 <= int(vision_num_layers) <= VIS_L:
            raise ValueError(
                f"vision_num_layers must be in [1, {VIS_L}], got {vision_num_layers}"
            )
        n_layers = int(vision_num_layers)
        self.vision_patch_embed_with_bias_pos(
            rocm_kernels,
            _weight_ptr(weights, "vision_patch_embedding_w"),
            _weight_ptr(weights, "vision_patch_embedding_b"),
            _weight_ptr(weights, "vision_position_embedding"),
            stream=stream,
        )
        if self.fp8_calibrated and self._vision_fp8_fusion_enabled():
            rocm_kernels.layer_norm_fp8_e4m3fnuz_ptr(
                self.bufs["vision_x"].ptr.value,
                _weight_ptr(weights, "vision_pre_attn_norm_w", 0),
                _weight_ptr(weights, "vision_pre_attn_norm_b", 0),
                self.bufs["vision_act_fp8"].ptr.value,
                self._fp8_scale_ptr(_fp8_key("vision_attn_qkv_w", 0)),
                self.vision_seq,
                VIS_D,
                1e-5,
                stream,
            )
        else:
            self.vision_pre_attn_layer_norm(
                rocm_kernels,
                _weight_ptr(weights, "vision_pre_attn_norm_w", 0),
                _weight_ptr(weights, "vision_pre_attn_norm_b", 0),
                stream=stream,
            )
        for i in range(n_layers):
            is_last = i == n_layers - 1
            self.vision_layer_fp8(
                rocm_kernels,
                weights,
                i,
                next_pre_attn_norm_weight_ptr=(
                    0
                    if is_last
                    else _weight_ptr(weights, "vision_pre_attn_norm_w", i + 1)
                ),
                next_pre_attn_norm_bias_ptr=(
                    0
                    if is_last
                    else _weight_ptr(weights, "vision_pre_attn_norm_b", i + 1)
                ),
                is_last=is_last,
                prequantized_qkv=(
                    self.fp8_calibrated and self._vision_fp8_fusion_enabled()
                ),
                stream=stream,
            )

    def vision_project_to_encoder_bf16(
        self, rocm_kernels, weights, stream: int = 0
    ) -> None:
        """Final SigLIP LayerNorm and projector into ``encoder_x`` prefix."""
        rocm_kernels.layer_norm_bf16_ptr(
            self.bufs["vision_x_pooled"].ptr.value,
            _weight_ptr(weights, "vision_final_norm_w"),
            _weight_ptr(weights, "vision_final_norm_b"),
            self.bufs["vision_x_norm"].ptr.value,
            self.vision_seq_enc,
            VIS_D,
            1e-5,
            stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["vision_x_norm"].ptr.value,
            _weight_ptr(weights, "encoder_multi_modal_projector_w"),
            0,
            self.bufs["encoder_x"].ptr.value,
            self.vision_seq_enc,
            ENC_D,
            VIS_D,
            stream,
        )
        rocm_kernels.add_bias_bf16_ptr(
            self.bufs["encoder_x"].ptr.value,
            _weight_ptr(weights, "encoder_multi_modal_projector_b"),
            self.vision_seq_enc,
            ENC_D,
            stream,
        )

    def vision_project_to_encoder_fp8(
        self, rocm_kernels, weights, stream: int = 0
    ) -> None:
        """Final SigLIP LayerNorm and FP8 projector into ``encoder_x`` prefix."""
        if self.fp8_calibrated and self._vision_fp8_fusion_enabled():
            rocm_kernels.layer_norm_fp8_e4m3fnuz_ptr(
                self.bufs["vision_x_pooled"].ptr.value,
                _weight_ptr(weights, "vision_final_norm_w"),
                _weight_ptr(weights, "vision_final_norm_b"),
                self.bufs["vision_act_fp8"].ptr.value,
                self._fp8_scale_ptr("encoder_multi_modal_projector_w"),
                self.vision_seq_enc,
                VIS_D,
                1e-5,
                stream,
            )
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="vision_act_fp8",
                act_scale_name="encoder_multi_modal_projector_w",
                weight_name="encoder_multi_modal_projector_w",
                out_bf16_buf="encoder_x",
                m=self.vision_seq_enc,
                n=ENC_D,
                k=VIS_D,
                weights=weights,
                bias_ptr=_weight_ptr(weights, "encoder_multi_modal_projector_b"),
                stream=stream,
            )
        else:
            rocm_kernels.layer_norm_bf16_ptr(
                self.bufs["vision_x_pooled"].ptr.value,
                _weight_ptr(weights, "vision_final_norm_w"),
                _weight_ptr(weights, "vision_final_norm_b"),
                self.bufs["vision_x_norm"].ptr.value,
                self.vision_seq_enc,
                VIS_D,
                1e-5,
                stream,
            )
            self._fp8_linear(
                rocm_kernels,
                act_bf16_ptr=self.bufs["vision_x_norm"].ptr.value,
                act_fp8_buf="vision_act_fp8",
                act_scale_name="encoder_multi_modal_projector_w",
                weight_name="encoder_multi_modal_projector_w",
                out_bf16_buf="encoder_x",
                m=self.vision_seq_enc,
                n=ENC_D,
                k=VIS_D,
                weights=weights,
                bias_ptr=_weight_ptr(weights, "encoder_multi_modal_projector_b"),
                stream=stream,
            )

    def vision_encoder_to_encoder_x_bf16(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        stream: int = 0,
    ) -> None:
        """Run SigLIP BF16 encoder and write projected tokens to encoder_x."""
        self.vision_encoder_bf16(
            rocm_kernels,
            weights,
            vision_num_layers=vision_num_layers,
            stream=stream,
        )
        self.vision_project_to_encoder_bf16(rocm_kernels, weights, stream=stream)

    def vision_encoder_to_encoder_x_fp8(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        stream: int = 0,
    ) -> None:
        """Run SigLIP FP8 encoder and write projected tokens to encoder_x."""
        self.vision_encoder_fp8(
            rocm_kernels,
            weights,
            vision_num_layers=vision_num_layers,
            stream=stream,
        )
        self.vision_project_to_encoder_fp8(rocm_kernels, weights, stream=stream)

    def encoder_qkv_bf16(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run encoder RMSNorm and QKV projection for one Gemma layer."""
        rocm_kernels.rms_norm_bf16_ptr(
            self.bufs["encoder_x"].ptr.value,
            _weight_ptr(weights, "encoder_input_norm_w", layer),
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len,
            ENC_D,
            1e-6,
            stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["encoder_x_norm"].ptr.value,
            _weight_ptr(weights, "encoder_attn_qkv_w", layer),
            0,
            self.bufs["encoder_QKV"].ptr.value,
            self.encoder_seq_len,
            (ENC_NH + 2 * ENC_NKV) * ENC_HD,
            ENC_D,
            stream,
        )

    def encoder_qkv_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        skip_norm: bool = False,
        stream: int = 0,
    ) -> None:
        """Run encoder RMSNorm and FP8 QKV projection for one Gemma layer."""
        qkv_name = _fp8_key("encoder_attn_qkv_w", layer)
        if self.fp8_calibrated:
            if not skip_norm:
                rocm_kernels.rms_norm_fp8_e4m3fnuz_ptr(
                    self.bufs["encoder_x"].ptr.value,
                    _weight_ptr(weights, "encoder_input_norm_w", layer),
                    self.bufs["encoder_act_fp8"].ptr.value,
                    self._fp8_scale_ptr(qkv_name),
                    self.encoder_seq_len,
                    ENC_D,
                    1e-6,
                    stream,
                )
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="encoder_act_fp8",
                act_scale_name=qkv_name,
                weight_name="encoder_attn_qkv_w",
                out_bf16_buf="encoder_QKV",
                m=self.encoder_seq_len,
                n=(ENC_NH + 2 * ENC_NKV) * ENC_HD,
                k=ENC_D,
                weights=weights,
                layer=layer,
                stream=stream,
            )
        else:
            rocm_kernels.rms_norm_bf16_ptr(
                self.bufs["encoder_x"].ptr.value,
                _weight_ptr(weights, "encoder_input_norm_w", layer),
                self.bufs["encoder_x_norm"].ptr.value,
                self.encoder_seq_len,
                ENC_D,
                1e-6,
                stream,
            )
            self._fp8_linear(
                rocm_kernels,
                act_bf16_ptr=self.bufs["encoder_x_norm"].ptr.value,
                act_fp8_buf="encoder_act_fp8",
                act_scale_name=qkv_name,
                weight_name="encoder_attn_qkv_w",
                out_bf16_buf="encoder_QKV",
                m=self.encoder_seq_len,
                n=(ENC_NH + 2 * ENC_NKV) * ENC_HD,
                k=ENC_D,
                weights=weights,
                layer=layer,
                stream=stream,
            )

    def encoder_qkv_split_rope_to_attn(
        self, rocm_kernels, layer: int, stream: int = 0
    ) -> None:
        """Split encoder QKV, apply RoPE to Q/K, and write attention buffers."""
        attn_ptrs = getattr(self, "_attn_ptrs", None)
        if attn_ptrs is None:
            raise RuntimeError("no attention backend is attached")
        k_ptr, v_ptr = self._enc_kv_layer_ptrs(layer, offset_tokens=0)
        rocm_kernels.qkv_split_rope_bf16_ptr(
            self.bufs["encoder_QKV"].ptr.value,
            self.bufs["encoder_rope_weights"].ptr.value,
            int(attn_ptrs["enc_Q"]),
            k_ptr,
            v_ptr,
            self.encoder_seq_len,
            ENC_NH * ENC_HD,
            ENC_NKV * ENC_HD,
            ENC_NKV * ENC_HD,
            ENC_HD,
            stream,
        )

    def encoder_qkv_rope_bf16(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run the B1 encoder path through attention Q/K/V cache writes."""
        self.encoder_qkv_bf16(rocm_kernels, weights, layer, stream=stream)
        self.encoder_qkv_split_rope_to_attn(rocm_kernels, layer, stream=stream)

    def encoder_qkv_rope_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        skip_norm: bool = False,
        stream: int = 0,
    ) -> None:
        """Run encoder FP8 QKV through attention Q/K/V cache writes."""
        self.encoder_qkv_fp8(
            rocm_kernels, weights, layer, skip_norm=skip_norm, stream=stream
        )
        self.encoder_qkv_split_rope_to_attn(rocm_kernels, layer, stream=stream)

    def encoder_attention(self, layer: int, stream: int = 0) -> int:
        """Run attached Gemma encoder attention and return its BF16 output ptr."""
        attn = getattr(self, "attn", None)
        if attn is None:
            raise RuntimeError("no attention backend is attached")
        return int(
            attn.run("encoder", int(layer), q_seq=self.encoder_seq_len, stream=stream)
        )

    def encoder_attention_output_residual_norm(
        self,
        rocm_kernels,
        weights,
        layer: int,
        attn_out_ptr: int,
        stream: int = 0,
    ) -> None:
        """Project attention output, add residual, then write post-attn RMSNorm."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            int(attn_out_ptr),
            _weight_ptr(weights, "encoder_attn_o_w", layer),
            0,
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len,
            ENC_D,
            ENC_D,
            stream,
        )
        rocm_kernels.residual_add_rms_norm_bf16_ptr(
            self.bufs["encoder_x"].ptr.value,
            self.bufs["encoder_x_norm"].ptr.value,
            _weight_ptr(weights, "encoder_post_attn_norm_w", layer),
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len,
            ENC_D,
            1e-6,
            stream,
        )

    def encoder_attention_output_residual_norm_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        attn_out_ptr: int,
        stream: int = 0,
    ) -> None:
        """FP8 attention output projection, residual add, and post-attn RMSNorm."""
        self._fp8_linear(
            rocm_kernels,
            act_bf16_ptr=int(attn_out_ptr),
            act_fp8_buf="encoder_act_fp8",
            act_scale_name=_fp8_key("encoder_attn_o_w", layer),
            weight_name="encoder_attn_o_w",
            out_bf16_buf="encoder_x_norm",
            m=self.encoder_seq_len,
            n=ENC_D,
            k=ENC_D,
            weights=weights,
            layer=layer,
            stream=stream,
        )
        rocm_kernels.residual_add_rms_norm_bf16_ptr(
            self.bufs["encoder_x"].ptr.value,
            self.bufs["encoder_x_norm"].ptr.value,
            _weight_ptr(weights, "encoder_post_attn_norm_w", layer),
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len,
            ENC_D,
            1e-6,
            stream,
        )

    def encoder_mlp_bf16(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run Gemma gated MLP and add its down-projected output to encoder_x."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["encoder_x_norm"].ptr.value,
            _weight_ptr(weights, "encoder_ffn_gate_w", layer),
            0,
            self.bufs["encoder_gate_buf"].ptr.value,
            self.encoder_seq_len,
            ENC_H,
            ENC_D,
            stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["encoder_x_norm"].ptr.value,
            _weight_ptr(weights, "encoder_ffn_up_w", layer),
            0,
            self.bufs["encoder_hidden"].ptr.value,
            self.encoder_seq_len,
            ENC_H,
            ENC_D,
            stream,
        )
        rocm_kernels.gelu_tanh_mul_bf16_ptr(
            self.bufs["encoder_gate_buf"].ptr.value,
            self.bufs["encoder_hidden"].ptr.value,
            self.bufs["encoder_hidden"].ptr.value,
            self.encoder_seq_len * ENC_H,
            stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["encoder_hidden"].ptr.value,
            _weight_ptr(weights, "encoder_ffn_down_w", layer),
            0,
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len,
            ENC_D,
            ENC_H,
            stream,
        )
        rocm_kernels.residual_add_bf16_ptr(
            self.bufs["encoder_x"].ptr.value,
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len * ENC_D,
            stream,
        )

    def encoder_mlp_fp8_static(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run the encoder MLP with static FP8 GEMMs and BF16 residual output."""
        gate_up_name = _fp8_key("encoder_ffn_gate_up_w", layer)
        down_name = _fp8_key("encoder_ffn_down_w", layer)

        act_scale_ptr = self._fp8_quantize_bf16_ptr(
            rocm_kernels,
            src_ptr=self.bufs["encoder_x_norm"].ptr.value,
            dst_buf="encoder_act_fp8",
            scale_name=gate_up_name,
            n=self.encoder_seq_len * ENC_D,
            stream=stream,
        )
        w_fp8_ptr, w_scale_ptr = self._weight_fp8_ptrs(
            weights, "encoder_ffn_gate_up_w", layer
        )
        rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
            self.bufs["encoder_act_fp8"].ptr.value,
            w_fp8_ptr,
            act_scale_ptr,
            w_scale_ptr,
            0,
            self.bufs["encoder_gate_merged"].ptr.value,
            self.encoder_seq_len,
            2 * ENC_H,
            ENC_D,
            stream,
        )

        hidden_scale_ptr = self._fp8_scale_ptr(
            down_name, require_static=self.fp8_calibrated
        )
        if self.fp8_calibrated:
            rocm_kernels.gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr(
                self.bufs["encoder_gate_merged"].ptr.value,
                hidden_scale_ptr,
                self.bufs["encoder_act_fp8_large"].ptr.value,
                self.encoder_seq_len,
                ENC_H,
                stream,
            )
        else:
            rocm_kernels.gelu_tanh_merged_bf16_ptr(
                self.bufs["encoder_gate_merged"].ptr.value,
                self.bufs["encoder_hidden"].ptr.value,
                self.encoder_seq_len,
                ENC_H,
                stream,
            )
            hidden_scale_ptr = self._fp8_quantize_bf16_ptr(
                rocm_kernels,
                src_ptr=self.bufs["encoder_hidden"].ptr.value,
                dst_buf="encoder_act_fp8_large",
                scale_name=down_name,
                n=self.encoder_seq_len * ENC_H,
                stream=stream,
            )
        w_fp8_ptr, w_scale_ptr = self._weight_fp8_ptrs(
            weights, "encoder_ffn_down_w", layer
        )
        rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
            self.bufs["encoder_act_fp8_large"].ptr.value,
            w_fp8_ptr,
            hidden_scale_ptr,
            w_scale_ptr,
            0,
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len,
            ENC_D,
            ENC_H,
            stream,
        )
        rocm_kernels.residual_add_bf16_ptr(
            self.bufs["encoder_x"].ptr.value,
            self.bufs["encoder_x_norm"].ptr.value,
            self.encoder_seq_len * ENC_D,
            stream,
        )

    def encoder_mlp_fp8_prequantized(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        residual_add: bool = True,
        stream: int = 0,
    ) -> None:
        """Run encoder MLP when encoder_act_fp8 already holds normalized input."""
        gate_up_name = _fp8_key("encoder_ffn_gate_up_w", layer)
        down_name = _fp8_key("encoder_ffn_down_w", layer)
        self._fp8_linear_prequantized(
            rocm_kernels,
            act_fp8_buf="encoder_act_fp8",
            act_scale_name=gate_up_name,
            weight_name="encoder_ffn_gate_up_w",
            out_bf16_buf="encoder_gate_merged",
            m=self.encoder_seq_len,
            n=2 * ENC_H,
            k=ENC_D,
            weights=weights,
            layer=layer,
            stream=stream,
        )
        hidden_scale_ptr = self._fp8_scale_ptr(down_name)
        rocm_kernels.gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr(
            self.bufs["encoder_gate_merged"].ptr.value,
            hidden_scale_ptr,
            self.bufs["encoder_act_fp8_large"].ptr.value,
            self.encoder_seq_len,
            ENC_H,
            stream,
        )
        self._fp8_linear_prequantized(
            rocm_kernels,
            act_fp8_buf="encoder_act_fp8_large",
            act_scale_name=down_name,
            weight_name="encoder_ffn_down_w",
            out_bf16_buf="encoder_x_norm",
            m=self.encoder_seq_len,
            n=ENC_D,
            k=ENC_H,
            weights=weights,
            layer=layer,
            stream=stream,
        )
        if residual_add:
            rocm_kernels.residual_add_bf16_ptr(
                self.bufs["encoder_x"].ptr.value,
                self.bufs["encoder_x_norm"].ptr.value,
                self.encoder_seq_len * ENC_D,
                stream,
            )

    def encoder_layer_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        skip_b1: bool = False,
        is_last: bool = False,
        stream: int = 0,
    ) -> None:
        """Run one Gemma encoder layer through the FP8 kernel pipeline."""
        self.encoder_qkv_rope_fp8(
            rocm_kernels, weights, layer, skip_norm=skip_b1, stream=stream
        )
        if is_last:
            return
        attn_out_ptr = self.encoder_attention(layer, stream=stream)
        self._fp8_linear(
            rocm_kernels,
            act_bf16_ptr=int(attn_out_ptr),
            act_fp8_buf="encoder_act_fp8",
            act_scale_name=_fp8_key("encoder_attn_o_w", layer),
            weight_name="encoder_attn_o_w",
            out_bf16_buf="encoder_x_norm",
            m=self.encoder_seq_len,
            n=ENC_D,
            k=ENC_D,
            weights=weights,
            layer=layer,
            stream=stream,
        )
        if self.fp8_calibrated:
            gate_up_name = _fp8_key("encoder_ffn_gate_up_w", layer)
            rocm_kernels.residual_add_rms_norm_fp8_e4m3fnuz_ptr(
                self.bufs["encoder_x"].ptr.value,
                self.bufs["encoder_x_norm"].ptr.value,
                _weight_ptr(weights, "encoder_post_attn_norm_w", layer),
                self.bufs["encoder_act_fp8"].ptr.value,
                self._fp8_scale_ptr(gate_up_name),
                self.encoder_seq_len,
                ENC_D,
                1e-6,
                stream,
            )
            self.encoder_mlp_fp8_prequantized(
                rocm_kernels, weights, layer, residual_add=False, stream=stream
            )
            next_qkv_name = _fp8_key("encoder_attn_qkv_w", layer + 1)
            rocm_kernels.residual_add_rms_norm_fp8_e4m3fnuz_ptr(
                self.bufs["encoder_x"].ptr.value,
                self.bufs["encoder_x_norm"].ptr.value,
                _weight_ptr(weights, "encoder_input_norm_w", layer + 1),
                self.bufs["encoder_act_fp8"].ptr.value,
                self._fp8_scale_ptr(next_qkv_name),
                self.encoder_seq_len,
                ENC_D,
                1e-6,
                stream,
            )
        else:
            rocm_kernels.residual_add_rms_norm_bf16_ptr(
                self.bufs["encoder_x"].ptr.value,
                self.bufs["encoder_x_norm"].ptr.value,
                _weight_ptr(weights, "encoder_post_attn_norm_w", layer),
                self.bufs["encoder_x_norm"].ptr.value,
                self.encoder_seq_len,
                ENC_D,
                1e-6,
                stream,
            )
            self.encoder_mlp_fp8_static(rocm_kernels, weights, layer, stream=stream)

    def encoder_layer_bf16(
        self,
        rocm_kernels,
        weights,
        layer: int,
        *,
        is_last: bool = False,
        stream: int = 0,
    ) -> None:
        """Run one Gemma encoder layer through the BF16 kernel pipeline.

        The final layer stops after QKV/RoPE cache population, matching the RTX
        pipeline contract where the decoder consumes the final encoder KV cache.
        """
        self.encoder_qkv_rope_bf16(rocm_kernels, weights, layer, stream=stream)
        if is_last:
            return
        attn_out_ptr = self.encoder_attention(layer, stream=stream)
        self.encoder_attention_output_residual_norm(
            rocm_kernels,
            weights,
            layer,
            attn_out_ptr,
            stream=stream,
        )
        self.encoder_mlp_bf16(rocm_kernels, weights, layer, stream=stream)

    def encoder_bf16(
        self,
        rocm_kernels,
        weights,
        *,
        encoder_num_layers: int = ENC_L,
        stream: int = 0,
    ) -> None:
        """Run the Gemma encoder BF16 kernel pipeline over fixed buffers."""
        if not 1 <= int(encoder_num_layers) <= ENC_L:
            raise ValueError(
                f"encoder_num_layers must be in [1, {ENC_L}], got {encoder_num_layers}"
            )
        n_layers = int(encoder_num_layers)
        for i in range(n_layers):
            self.encoder_layer_bf16(
                rocm_kernels,
                weights,
                i,
                is_last=i == n_layers - 1,
                stream=stream,
            )

    def encoder_fp8(
        self,
        rocm_kernels,
        weights,
        *,
        encoder_num_layers: int = ENC_L,
        stream: int = 0,
    ) -> None:
        """Run the Gemma encoder FP8 kernel pipeline over fixed buffers."""
        if not 1 <= int(encoder_num_layers) <= ENC_L:
            raise ValueError(
                f"encoder_num_layers must be in [1, {ENC_L}], got {encoder_num_layers}"
            )
        n_layers = int(encoder_num_layers)
        for i in range(n_layers):
            self.encoder_layer_fp8(
                rocm_kernels,
                weights,
                i,
                skip_b1=(self.fp8_calibrated and i > 0),
                is_last=i == n_layers - 1,
                stream=stream,
            )

    def decoder_action_in_bf16(self, rocm_kernels, weights, stream: int = 0) -> None:
        """Project current diffusion noise into decoder_x."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["diffusion_noise"].ptr.value,
            _weight_ptr(weights, "decoder_action_in_proj_w"),
            _weight_ptr(weights, "decoder_action_in_proj_b"),
            self.bufs["decoder_x"].ptr.value,
            self.chunk_size,
            DEC_D,
            ACTION_DIM,
            stream,
        )

    def decoder_ada_rms_norm_style(
        self,
        rocm_kernels,
        style_ptr: int,
        *,
        src_name: str = "decoder_x",
        stream: int = 0,
    ) -> None:
        """Apply precomputed AdaRMS style to decoder tokens."""
        rocm_kernels.ada_rms_norm_style_bf16_ptr(
            self.bufs[src_name].ptr.value,
            self.bufs["decoder_rms_ones"].ptr.value,
            int(style_ptr),
            self.bufs["x_normed_buf"].ptr.value,
            self.bufs["gate_buf"].ptr.value,
            self.chunk_size,
            DEC_D,
            1e-6,
            stream,
        )

    def decoder_qkv_bf16(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run decoder AdaRMSNorm-normalized QKV projection."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["x_normed_buf"].ptr.value,
            _weight_ptr(weights, "decoder_attn_qkv_w", layer),
            0,
            self.bufs["decoder_QKV"].ptr.value,
            self.chunk_size,
            (DEC_NH + 2 * DEC_NKV) * DEC_HD,
            DEC_D,
            stream,
        )

    def decoder_qkv_fp8(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run decoder AdaRMSNorm-normalized FP8 QKV projection."""
        self._fp8_linear(
            rocm_kernels,
            act_bf16_ptr=self.bufs["x_normed_buf"].ptr.value,
            act_fp8_buf="decoder_act_fp8",
            act_scale_name=_fp8_key("decoder_attn_qkv_w", layer),
            weight_name="decoder_attn_qkv_w",
            out_bf16_buf="decoder_QKV",
            m=self.chunk_size,
            n=(DEC_NH + 2 * DEC_NKV) * DEC_HD,
            k=DEC_D,
            weights=weights,
            layer=layer,
            stream=stream,
        )

    def decoder_qkv_split_rope_to_attn(
        self, rocm_kernels, layer: int, stream: int = 0
    ) -> None:
        """Split decoder QKV, apply RoPE, and append K/V after encoder cache."""
        attn_ptrs = getattr(self, "_attn_ptrs", None)
        if attn_ptrs is None:
            raise RuntimeError("no attention backend is attached")
        k_ptr, v_ptr = self._enc_kv_layer_ptrs(layer, offset_tokens=self.encoder_seq_len)
        rocm_kernels.qkv_split_rope_bf16_ptr(
            self.bufs["decoder_QKV"].ptr.value,
            self.bufs["decoder_rope_weights"].ptr.value,
            int(attn_ptrs["dec_Q"]),
            k_ptr,
            v_ptr,
            self.chunk_size,
            DEC_NH * DEC_HD,
            DEC_NKV * DEC_HD,
            DEC_NKV * DEC_HD,
            DEC_HD,
            stream,
        )

    def decoder_attention(self, layer: int, stream: int = 0) -> int:
        """Run attached decoder cross-attention and return BF16 output ptr."""
        attn = getattr(self, "attn", None)
        if attn is None:
            raise RuntimeError("no attention backend is attached")
        return int(
            attn.run(
                "decoder",
                int(layer),
                q_seq=self.chunk_size,
                kv_seq=self.encoder_seq_len + self.chunk_size,
                stream=stream,
            )
        )

    def decoder_attention_output_bf16(
        self,
        rocm_kernels,
        weights,
        layer: int,
        attn_out_ptr: int,
        stream: int = 0,
    ) -> None:
        """Project decoder attention output into x_normed_buf."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            int(attn_out_ptr),
            _weight_ptr(weights, "decoder_attn_o_w", layer),
            0,
            self.bufs["x_normed_buf"].ptr.value,
            self.chunk_size,
            DEC_D,
            DEC_NH * DEC_HD,
            stream,
        )

    def decoder_attention_output_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        attn_out_ptr: int,
        stream: int = 0,
    ) -> None:
        """Project decoder attention output into x_normed_buf with FP8 GEMM."""
        self._fp8_linear(
            rocm_kernels,
            act_bf16_ptr=int(attn_out_ptr),
            act_fp8_buf="decoder_act_fp8",
            act_scale_name=_fp8_key("decoder_attn_o_w", layer),
            weight_name="decoder_attn_o_w",
            out_bf16_buf="x_normed_buf",
            m=self.chunk_size,
            n=DEC_D,
            k=DEC_NH * DEC_HD,
            weights=weights,
            layer=layer,
            stream=stream,
        )

    def decoder_gate_residual_then_ffn_norm(
        self,
        rocm_kernels,
        style_ptr: int,
        stream: int = 0,
    ) -> None:
        """Apply gated attention residual, then AdaRMSNorm for the FFN."""
        rocm_kernels.gate_residual_ada_norm_bf16_ptr(
            self.bufs["decoder_x"].ptr.value,
            self.bufs["x_normed_buf"].ptr.value,
            self.bufs["gate_buf"].ptr.value,
            self.bufs["decoder_rms_ones"].ptr.value,
            int(style_ptr),
            self.bufs["x_normed_buf"].ptr.value,
            self.bufs["gate_buf"].ptr.value,
            self.chunk_size,
            DEC_D,
            1e-6,
            stream,
        )

    def decoder_mlp_bf16(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run decoder gated MLP down projection into x_normed_buf."""
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["x_normed_buf"].ptr.value,
            _weight_ptr(weights, "decoder_ffn_gate_w", layer),
            0,
            self.bufs["decoder_gate_buf"].ptr.value,
            self.chunk_size,
            DEC_H,
            DEC_D,
            stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["x_normed_buf"].ptr.value,
            _weight_ptr(weights, "decoder_ffn_up_w", layer),
            0,
            self.bufs["decoder_hidden"].ptr.value,
            self.chunk_size,
            DEC_H,
            DEC_D,
            stream,
        )
        rocm_kernels.gelu_tanh_mul_bf16_ptr(
            self.bufs["decoder_gate_buf"].ptr.value,
            self.bufs["decoder_hidden"].ptr.value,
            self.bufs["decoder_hidden"].ptr.value,
            self.chunk_size * DEC_H,
            stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["decoder_hidden"].ptr.value,
            _weight_ptr(weights, "decoder_ffn_down_w", layer),
            0,
            self.bufs["x_normed_buf"].ptr.value,
            self.chunk_size,
            DEC_D,
            DEC_H,
            stream,
        )

    def decoder_mlp_fp8_static(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run the decoder MLP with static FP8 GEMMs and BF16 residual input."""
        gate_up_name = _fp8_key("decoder_ffn_gate_up_w", layer)
        down_name = _fp8_key("decoder_ffn_down_w", layer)

        act_scale_ptr = self._fp8_quantize_bf16_ptr(
            rocm_kernels,
            src_ptr=self.bufs["x_normed_buf"].ptr.value,
            dst_buf="decoder_act_fp8",
            scale_name=gate_up_name,
            n=self.chunk_size * DEC_D,
            stream=stream,
        )
        w_fp8_ptr, w_scale_ptr = self._weight_fp8_ptrs(
            weights, "decoder_ffn_gate_up_w", layer
        )
        rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
            self.bufs["decoder_act_fp8"].ptr.value,
            w_fp8_ptr,
            act_scale_ptr,
            w_scale_ptr,
            0,
            self.bufs["decoder_gate_merged"].ptr.value,
            self.chunk_size,
            2 * DEC_H,
            DEC_D,
            stream,
        )

        hidden_scale_ptr = self._fp8_scale_ptr(
            down_name, require_static=self.fp8_calibrated
        )
        if self.fp8_calibrated:
            rocm_kernels.gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr(
                self.bufs["decoder_gate_merged"].ptr.value,
                hidden_scale_ptr,
                self.bufs["decoder_act_fp8_large"].ptr.value,
                self.chunk_size,
                DEC_H,
                stream,
            )
        else:
            rocm_kernels.gelu_tanh_merged_bf16_ptr(
                self.bufs["decoder_gate_merged"].ptr.value,
                self.bufs["decoder_hidden"].ptr.value,
                self.chunk_size,
                DEC_H,
                stream,
            )
            hidden_scale_ptr = self._fp8_quantize_bf16_ptr(
                rocm_kernels,
                src_ptr=self.bufs["decoder_hidden"].ptr.value,
                dst_buf="decoder_act_fp8_large",
                scale_name=down_name,
                n=self.chunk_size * DEC_H,
                stream=stream,
            )
        w_fp8_ptr, w_scale_ptr = self._weight_fp8_ptrs(
            weights, "decoder_ffn_down_w", layer
        )
        rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
            self.bufs["decoder_act_fp8_large"].ptr.value,
            w_fp8_ptr,
            hidden_scale_ptr,
            w_scale_ptr,
            0,
            self.bufs["x_normed_buf"].ptr.value,
            self.chunk_size,
            DEC_D,
            DEC_H,
            stream,
        )

    def decoder_mlp_fp8_prequantized(
        self, rocm_kernels, weights, layer: int, stream: int = 0
    ) -> None:
        """Run decoder MLP when decoder_act_fp8 already holds normalized input."""
        gate_up_name = _fp8_key("decoder_ffn_gate_up_w", layer)
        down_name = _fp8_key("decoder_ffn_down_w", layer)
        self._fp8_linear_prequantized(
            rocm_kernels,
            act_fp8_buf="decoder_act_fp8",
            act_scale_name=gate_up_name,
            weight_name="decoder_ffn_gate_up_w",
            out_bf16_buf="decoder_gate_merged",
            m=self.chunk_size,
            n=2 * DEC_H,
            k=DEC_D,
            weights=weights,
            layer=layer,
            stream=stream,
        )
        hidden_scale_ptr = self._fp8_scale_ptr(down_name)
        rocm_kernels.gelu_tanh_merged_quantize_fp8_e4m3fnuz_ptr(
            self.bufs["decoder_gate_merged"].ptr.value,
            hidden_scale_ptr,
            self.bufs["decoder_act_fp8_large"].ptr.value,
            self.chunk_size,
            DEC_H,
            stream,
        )
        self._fp8_linear_prequantized(
            rocm_kernels,
            act_fp8_buf="decoder_act_fp8_large",
            act_scale_name=down_name,
            weight_name="decoder_ffn_down_w",
            out_bf16_buf="x_normed_buf",
            m=self.chunk_size,
            n=DEC_D,
            k=DEC_H,
            weights=weights,
            layer=layer,
            stream=stream,
        )

    def decoder_layer_bf16(
        self,
        rocm_kernels,
        weights,
        layer: int,
        step: int,
        *,
        stream: int = 0,
    ) -> None:
        """Run one Gemma expert decoder layer in the BF16 kernel pipeline."""
        self.decoder_ada_rms_norm_style(
            rocm_kernels,
            self._style_slice_ptr("decoder_style_attn", step, layer),
            stream=stream,
        )
        self.decoder_qkv_bf16(rocm_kernels, weights, layer, stream=stream)
        self.decoder_qkv_split_rope_to_attn(rocm_kernels, layer, stream=stream)
        attn_out_ptr = self.decoder_attention(layer, stream=stream)
        self.decoder_attention_output_bf16(
            rocm_kernels,
            weights,
            layer,
            attn_out_ptr,
            stream=stream,
        )
        self.decoder_gate_residual_then_ffn_norm(
            rocm_kernels,
            self._style_slice_ptr("decoder_style_ffn", step, layer),
            stream=stream,
        )
        self.decoder_mlp_bf16(rocm_kernels, weights, layer, stream=stream)
        rocm_kernels.gate_mul_residual_bf16_ptr(
            self.bufs["decoder_x"].ptr.value,
            self.bufs["x_normed_buf"].ptr.value,
            self.bufs["gate_buf"].ptr.value,
            self.chunk_size * DEC_D,
            stream,
        )

    def decoder_layer_fp8(
        self,
        rocm_kernels,
        weights,
        layer: int,
        step: int,
        *,
        skip_c1: bool = False,
        is_last: bool = False,
        stream: int = 0,
    ) -> None:
        """Run one Gemma expert decoder layer through the FP8 kernel pipeline."""
        qkv_name = _fp8_key("decoder_attn_qkv_w", layer)
        if self.fp8_calibrated:
            if not skip_c1:
                rocm_kernels.ada_rms_norm_style_fp8_e4m3fnuz_ptr(
                    self.bufs["decoder_x"].ptr.value,
                    self.bufs["decoder_rms_ones"].ptr.value,
                    self._style_slice_ptr("decoder_style_attn", step, layer),
                    self.bufs["decoder_act_fp8"].ptr.value,
                    self.bufs["gate_buf"].ptr.value,
                    self._fp8_scale_ptr(qkv_name),
                    self.chunk_size,
                    DEC_D,
                    1e-6,
                    stream,
                )
            self._fp8_linear_prequantized(
                rocm_kernels,
                act_fp8_buf="decoder_act_fp8",
                act_scale_name=qkv_name,
                weight_name="decoder_attn_qkv_w",
                out_bf16_buf="decoder_QKV",
                m=self.chunk_size,
                n=(DEC_NH + 2 * DEC_NKV) * DEC_HD,
                k=DEC_D,
                weights=weights,
                layer=layer,
                stream=stream,
            )
        else:
            self.decoder_ada_rms_norm_style(
                rocm_kernels,
                self._style_slice_ptr("decoder_style_attn", step, layer),
                stream=stream,
            )
            self.decoder_qkv_fp8(rocm_kernels, weights, layer, stream=stream)
        self.decoder_qkv_split_rope_to_attn(rocm_kernels, layer, stream=stream)
        attn_out_ptr = self.decoder_attention(layer, stream=stream)
        self.decoder_attention_output_fp8(
            rocm_kernels,
            weights,
            layer,
            attn_out_ptr,
            stream=stream,
        )
        if self.fp8_calibrated:
            gate_up_name = _fp8_key("decoder_ffn_gate_up_w", layer)
            rocm_kernels.gate_residual_ada_norm_fp8_e4m3fnuz_ptr(
                self.bufs["decoder_x"].ptr.value,
                self.bufs["x_normed_buf"].ptr.value,
                self.bufs["gate_buf"].ptr.value,
                self.bufs["decoder_rms_ones"].ptr.value,
                self._style_slice_ptr("decoder_style_ffn", step, layer),
                self.bufs["decoder_act_fp8"].ptr.value,
                self.bufs["gate_buf"].ptr.value,
                self._fp8_scale_ptr(gate_up_name),
                self.chunk_size,
                DEC_D,
                1e-6,
                stream,
            )
            self.decoder_mlp_fp8_prequantized(
                rocm_kernels, weights, layer, stream=stream
            )
        else:
            self.decoder_gate_residual_then_ffn_norm(
                rocm_kernels,
                self._style_slice_ptr("decoder_style_ffn", step, layer),
                stream=stream,
            )
            self.decoder_mlp_fp8_static(rocm_kernels, weights, layer, stream=stream)
        if self.fp8_calibrated and not is_last:
            next_qkv_name = _fp8_key("decoder_attn_qkv_w", layer + 1)
            rocm_kernels.gate_residual_ada_norm_fp8_e4m3fnuz_ptr(
                self.bufs["decoder_x"].ptr.value,
                self.bufs["x_normed_buf"].ptr.value,
                self.bufs["gate_buf"].ptr.value,
                self.bufs["decoder_rms_ones"].ptr.value,
                self._style_slice_ptr("decoder_style_attn", step, layer + 1),
                self.bufs["decoder_act_fp8"].ptr.value,
                self.bufs["gate_buf"].ptr.value,
                self._fp8_scale_ptr(next_qkv_name),
                self.chunk_size,
                DEC_D,
                1e-6,
                stream,
            )
        else:
            rocm_kernels.gate_mul_residual_bf16_ptr(
                self.bufs["decoder_x"].ptr.value,
                self.bufs["x_normed_buf"].ptr.value,
                self.bufs["gate_buf"].ptr.value,
                self.chunk_size * DEC_D,
                stream,
            )

    def decoder_final_bf16(
        self, rocm_kernels, weights, step: int, stream: int = 0
    ) -> None:
        """Apply final AdaRMSNorm, project actions, and update diffusion noise."""
        self.decoder_ada_rms_norm_style(
            rocm_kernels,
            self._style_slice_ptr("decoder_style_final", step),
            stream=stream,
        )
        rocm_kernels.hipblaslt_linear_bf16_ptr(
            self.bufs["x_normed_buf"].ptr.value,
            _weight_ptr(weights, "decoder_action_out_proj_w"),
            _weight_ptr(weights, "decoder_action_out_proj_b"),
            self.bufs["decoder_action_buf"].ptr.value,
            self.chunk_size,
            ACTION_DIM,
            DEC_D,
            stream,
        )
        rocm_kernels.residual_add_bf16_ptr(
            self.bufs["diffusion_noise"].ptr.value,
            self.bufs["decoder_action_buf"].ptr.value,
            self.chunk_size * ACTION_DIM,
            stream,
        )

    def decoder_step_bf16(
        self,
        rocm_kernels,
        weights,
        step: int,
        *,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> None:
        """Run one diffusion denoise step through the BF16 decoder."""
        if not 1 <= int(decoder_num_layers) <= DEC_L:
            raise ValueError(
                f"decoder_num_layers must be in [1, {DEC_L}], got {decoder_num_layers}"
            )
        self.decoder_action_in_bf16(rocm_kernels, weights, stream=stream)
        for i in range(int(decoder_num_layers)):
            self.decoder_layer_bf16(rocm_kernels, weights, i, step, stream=stream)
        self.decoder_final_bf16(rocm_kernels, weights, step, stream=stream)

    def decoder_step_fp8(
        self,
        rocm_kernels,
        weights,
        step: int,
        *,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> None:
        """Run one diffusion denoise step through the FP8 decoder."""
        if not 1 <= int(decoder_num_layers) <= DEC_L:
            raise ValueError(
                f"decoder_num_layers must be in [1, {DEC_L}], got {decoder_num_layers}"
            )
        self.decoder_action_in_bf16(rocm_kernels, weights, stream=stream)
        n_layers = int(decoder_num_layers)
        for i in range(n_layers):
            self.decoder_layer_fp8(
                rocm_kernels,
                weights,
                i,
                step,
                skip_c1=(self.fp8_calibrated and i > 0),
                is_last=(i == n_layers - 1),
                stream=stream,
            )
        self.decoder_final_bf16(rocm_kernels, weights, step, stream=stream)

    def decoder_bf16(
        self,
        rocm_kernels,
        weights,
        *,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> None:
        """Run the fixed-step Pi0.5 BF16 decoder over diffusion_noise."""
        precomputed = weights.get("precomputed")
        if precomputed is not None and not getattr(self, "_decoder_styles_uploaded", False):
            self.upload_precomputed_decoder_styles(precomputed)
        for step in range(self.num_steps):
            self.decoder_step_bf16(
                rocm_kernels,
                weights,
                step,
                decoder_num_layers=decoder_num_layers,
                stream=stream,
            )

    def decoder_fp8(
        self,
        rocm_kernels,
        weights,
        *,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> None:
        """Run the fixed-step Pi0.5 FP8 decoder over diffusion_noise."""
        precomputed = weights.get("precomputed")
        if precomputed is not None and not getattr(self, "_decoder_styles_uploaded", False):
            self.upload_precomputed_decoder_styles(precomputed)
        for step in range(self.num_steps):
            self.decoder_step_fp8(
                rocm_kernels,
                weights,
                step,
                decoder_num_layers=decoder_num_layers,
                stream=stream,
            )

    def run_bf16_pipeline(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> None:
        """Run vision, Gemma encoder, and diffusion decoder BF16 phases.

        The language prompt slice of ``encoder_x`` must already be populated
        by the frontend before this call; the vision projector overwrites only
        the visual prefix.
        """
        self._copy_lang_embeds_to_encoder_x(stream)
        self.vision_encoder_to_encoder_x_bf16(
            rocm_kernels,
            weights,
            vision_num_layers=vision_num_layers,
            stream=stream,
        )
        self.encoder_bf16(
            rocm_kernels,
            weights,
            encoder_num_layers=encoder_num_layers,
            stream=stream,
        )
        self.decoder_bf16(
            rocm_kernels,
            weights,
            decoder_num_layers=decoder_num_layers,
            stream=stream,
        )

    def run_fp8_pipeline(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> None:
        """Run vision, Gemma encoder, and diffusion decoder with FP8 GEMM sites.

        If ``fp8_calibrated`` is false, activation quantization uses the dynamic
        calibration kernels and writes each site's scale into ``fp8_act_scales``.
        Once calibrated, the same graph uses static scales only.
        """
        if "fp8" not in weights:
            raise RuntimeError(
                "run_fp8_pipeline requires weights built with include_fp8=True"
            )
        self._copy_lang_embeds_to_encoder_x(stream)
        self.vision_encoder_to_encoder_x_fp8(
            rocm_kernels,
            weights,
            vision_num_layers=vision_num_layers,
            stream=stream,
        )
        self.encoder_fp8(
            rocm_kernels,
            weights,
            encoder_num_layers=encoder_num_layers,
            stream=stream,
        )
        self.decoder_fp8(
            rocm_kernels,
            weights,
            decoder_num_layers=decoder_num_layers,
            stream=stream,
        )

    def calibrate_fp8(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
        stream: int = 0,
    ) -> dict[str, object]:
        """Collect static FP8 activation scales with one calibration run."""
        self.fp8_act_scales.clear()
        self._aiter_scale_tensors.clear()
        self.fp8_calibrated = False
        self.run_fp8_pipeline(
            rocm_kernels,
            weights,
            vision_num_layers=vision_num_layers,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            stream=stream,
        )
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        self.fp8_calibrated = True
        coverage = self.fp8_scale_coverage(
            vision_num_layers=vision_num_layers,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
        )
        missing = coverage["missing_scales"]
        unexpected = coverage["unexpected_scales"]
        if missing or unexpected:
            raise RuntimeError(
                "FP8 scale coverage mismatch: "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        return {
            "scale_count": len(self.fp8_act_scales),
            "scales": sorted(self.fp8_act_scales),
            **coverage,
        }

    def capture_bf16_graph(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
    ) -> None:
        """Capture the current BF16 pipeline with PyTorch ROCm CUDAGraph.

        The attention backend still uses torch SDPA, so the first AMD graph
        capture uses PyTorch's ROCm graph wrapper while compiled HIP kernels
        and hipBLASLt calls are routed onto the active capture stream.
        """
        import torch

        self.bake_bf16_gemms(rocm_kernels)
        precomputed = weights.get("precomputed")
        if precomputed is not None:
            self.upload_precomputed_decoder_styles(precomputed)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream = int(torch.cuda.current_stream().cuda_stream)
            self.run_bf16_pipeline(
                rocm_kernels,
                weights,
                vision_num_layers=vision_num_layers,
                encoder_num_layers=encoder_num_layers,
                decoder_num_layers=decoder_num_layers,
                stream=stream,
            )
        self._bf16_graph = graph
        self._bf16_graph_config = {
            "vision_num_layers": int(vision_num_layers),
            "encoder_num_layers": int(encoder_num_layers),
            "decoder_num_layers": int(decoder_num_layers),
        }

    def replay_bf16_graph(self) -> None:
        """Replay a previously captured BF16 graph."""
        graph = getattr(self, "_bf16_graph", None)
        if graph is None:
            raise RuntimeError("BF16 graph has not been captured")
        graph.replay()

    def capture_fp8_graph(
        self,
        rocm_kernels,
        weights,
        *,
        vision_num_layers: int = VIS_L,
        encoder_num_layers: int = ENC_L,
        decoder_num_layers: int = DEC_L,
    ) -> None:
        """Capture the calibrated FP8 pipeline with PyTorch ROCm CUDAGraph."""
        import torch

        if not self.fp8_calibrated:
            raise RuntimeError("call calibrate_fp8() before capture_fp8_graph()")
        self.bake_fp8_gemms(rocm_kernels)
        precomputed = weights.get("precomputed")
        if precomputed is not None:
            self.upload_precomputed_decoder_styles(precomputed)
        self.bake_aiter_decoder_fp8_gemms(weights)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream = int(torch.cuda.current_stream().cuda_stream)
            self.run_fp8_pipeline(
                rocm_kernels,
                weights,
                vision_num_layers=vision_num_layers,
                encoder_num_layers=encoder_num_layers,
                decoder_num_layers=decoder_num_layers,
                stream=stream,
            )
        self._fp8_graph = graph
        self._fp8_graph_config = {
            "vision_num_layers": int(vision_num_layers),
            "encoder_num_layers": int(encoder_num_layers),
            "decoder_num_layers": int(decoder_num_layers),
        }

    def replay_fp8_graph(self) -> None:
        """Replay a previously captured FP8 graph."""
        graph = getattr(self, "_fp8_graph", None)
        if graph is None:
            raise RuntimeError("FP8 graph has not been captured")
        graph.replay()

    def capture_decoder_fp8_graph(
        self,
        rocm_kernels,
        weights,
        *,
        decoder_num_layers: int = DEC_L,
    ) -> None:
        """Capture only the calibrated FP8 diffusion decoder.

        The caller must have already populated encoder K/V caches by running
        the vision + encoder phases. This mirrors the RTX decoder-only graph
        path used when visual/prompt context is stable and only diffusion noise
        changes between replays.
        """
        import torch

        if not self.fp8_calibrated:
            raise RuntimeError(
                "call calibrate_fp8() before capture_decoder_fp8_graph()"
            )
        self.bake_fp8_gemms(rocm_kernels)
        precomputed = weights.get("precomputed")
        if precomputed is not None:
            self.upload_precomputed_decoder_styles(precomputed)
        self.bake_aiter_decoder_fp8_gemms(weights)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream = int(torch.cuda.current_stream().cuda_stream)
            self.decoder_fp8(
                rocm_kernels,
                weights,
                decoder_num_layers=decoder_num_layers,
                stream=stream,
            )
        self._decoder_fp8_graph = graph
        self._decoder_fp8_graph_config = {
            "decoder_num_layers": int(decoder_num_layers),
        }

    def replay_decoder_fp8_graph(self) -> None:
        """Replay a previously captured FP8 decoder-only graph."""
        graph = getattr(self, "_decoder_fp8_graph", None)
        if graph is None:
            raise RuntimeError("FP8 decoder graph has not been captured")
        graph.replay()

    def bake_aiter_decoder_fp8_gemms(self, weights) -> dict[str, object]:
        """Warm AITER decoder FP8 GEMMs before graph capture."""
        if not self._aiter_decoder_fp8_gemm_enabled():
            return {"enabled": False, "shapes": []}
        import torch

        shapes = [
            (
                "decoder_attn_qkv_w",
                "decoder_act_fp8",
                "decoder_QKV",
                (DEC_NH + 2 * DEC_NKV) * DEC_HD,
                DEC_D,
            ),
            (
                "decoder_attn_o_w",
                "decoder_act_fp8",
                "x_normed_buf",
                DEC_D,
                DEC_NH * DEC_HD,
            ),
            (
                "decoder_ffn_gate_up_w",
                "decoder_act_fp8",
                "decoder_gate_merged",
                2 * DEC_H,
                DEC_D,
            ),
            (
                "decoder_ffn_down_w",
                "decoder_act_fp8_large",
                "x_normed_buf",
                DEC_D,
                DEC_H,
            ),
        ]
        warmed = []
        for weight_name, act_buf, out_buf, _n, _k in shapes:
            self._aiter_decoder_fp8_linear(
                act_fp8_buf=act_buf,
                act_scale_name=_fp8_key(weight_name, 0),
                weight_name=weight_name,
                out_bf16_buf=out_buf,
                weights=weights,
                layer=0,
            )
            warmed.append(weight_name)
        torch.cuda.synchronize()
        return {"enabled": True, "shapes": warmed}

    def capture_decoder_bf16_graph(
        self,
        rocm_kernels,
        weights,
        *,
        decoder_num_layers: int = DEC_L,
    ) -> None:
        """Capture only the BF16 diffusion decoder.

        The caller must have already populated encoder K/V caches by running
        the vision + encoder phases. This mirrors the RTX decoder-only graph
        path used when visual/prompt context is stable and only diffusion noise
        changes between replays.
        """
        import torch

        self.bake_bf16_gemms(rocm_kernels)
        precomputed = weights.get("precomputed")
        if precomputed is not None:
            self.upload_precomputed_decoder_styles(precomputed)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stream = int(torch.cuda.current_stream().cuda_stream)
            self.decoder_bf16(
                rocm_kernels,
                weights,
                decoder_num_layers=decoder_num_layers,
                stream=stream,
            )
        self._decoder_bf16_graph = graph
        self._decoder_bf16_graph_config = {
            "decoder_num_layers": int(decoder_num_layers),
        }

    def replay_decoder_bf16_graph(self) -> None:
        """Replay a previously captured BF16 decoder-only graph."""
        graph = getattr(self, "_decoder_bf16_graph", None)
        if graph is None:
            raise RuntimeError("BF16 decoder graph has not been captured")
        graph.replay()

    def bake_bf16_gemms(self, rocm_kernels) -> dict[str, object]:
        """Warm fixed Pi0.5 BF16 hipBLASLt Linear shapes into the algo cache."""
        shapes = [
            ("vision_patch", self.vision_seq, VIS_D, VIS_PATCH_FLAT, False),
            ("vision_qkv", self.vision_seq, 3 * VIS_D, VIS_D, True),
            ("vision_attn_o", self.vision_seq, VIS_D, VIS_D, False),
            ("vision_ffn_up", self.vision_seq, VIS_H, VIS_D, False),
            ("vision_ffn_down", self.vision_seq, VIS_D, VIS_H, False),
            ("vision_projector", self.vision_seq_enc, ENC_D, VIS_D, False),
            (
                "encoder_qkv",
                self.encoder_seq_len,
                (ENC_NH + 2 * ENC_NKV) * ENC_HD,
                ENC_D,
                False,
            ),
            ("encoder_attn_o", self.encoder_seq_len, ENC_D, ENC_D, False),
            ("encoder_ffn_gate_up", self.encoder_seq_len, ENC_H, ENC_D, False),
            ("encoder_ffn_down", self.encoder_seq_len, ENC_D, ENC_H, False),
            ("decoder_action_in", self.chunk_size, DEC_D, ACTION_DIM, True),
            (
                "decoder_qkv",
                self.chunk_size,
                (DEC_NH + 2 * DEC_NKV) * DEC_HD,
                DEC_D,
                False,
            ),
            ("decoder_attn_o", self.chunk_size, DEC_D, DEC_NH * DEC_HD, False),
            ("decoder_ffn_gate_up", self.chunk_size, DEC_H, DEC_D, False),
            ("decoder_ffn_down", self.chunk_size, DEC_D, DEC_H, False),
            ("decoder_action_out", self.chunk_size, ACTION_DIM, DEC_D, True),
        ]

        max_mk = max(m * k for _name, m, _n, k, _bias in shapes)
        max_nk = max(n * k for _name, _m, n, k, _bias in shapes)
        max_mn = max(m * n for _name, m, n, _k, _bias in shapes)
        max_n = max(n for _name, _m, n, _k, bias in shapes if bias)

        x = HipBuffer.device_zeros(max_mk, BF16_NP)
        w = HipBuffer.device_zeros(max_nk, BF16_NP)
        out = HipBuffer.device_empty(max_mn, BF16_NP)
        bias = HipBuffer.device_zeros(max_n, BF16_NP)

        before = (
            rocm_kernels.hipblaslt_linear_plan_cache_size()
            if hasattr(rocm_kernels, "hipblaslt_linear_plan_cache_size")
            else None
        )
        for _name, m, n, k, has_bias in shapes:
            rocm_kernels.hipblaslt_linear_bf16_ptr(
                x.ptr.value,
                w.ptr.value,
                bias.ptr.value if has_bias else 0,
                out.ptr.value,
                int(m),
                int(n),
                int(k),
            )
        after = (
            rocm_kernels.hipblaslt_linear_plan_cache_size()
            if hasattr(rocm_kernels, "hipblaslt_linear_plan_cache_size")
            else None
        )
        return {
            "shapes": [name for name, *_rest in shapes],
            "plan_cache_size_before": before,
            "plan_cache_size_after": after,
        }

    def bake_fp8_gemms(self, rocm_kernels) -> dict[str, object]:
        """Warm fixed Pi0.5 FP8 hipBLASLt Linear shapes into the algo cache."""
        import torch

        shapes = [
            ("vision_qkv", self.vision_seq, 3 * VIS_D, VIS_D, True),
            ("vision_attn_o", self.vision_seq, VIS_D, VIS_D, False),
            ("vision_ffn_up", self.vision_seq, VIS_H, VIS_D, True),
            ("vision_ffn_down", self.vision_seq, VIS_D, VIS_H, False),
            ("vision_projector", self.vision_seq_enc, ENC_D, VIS_D, True),
            (
                "encoder_qkv",
                self.encoder_seq_len,
                (ENC_NH + 2 * ENC_NKV) * ENC_HD,
                ENC_D,
                False,
            ),
            ("encoder_attn_o", self.encoder_seq_len, ENC_D, ENC_D, False),
            ("encoder_ffn_gate_up", self.encoder_seq_len, 2 * ENC_H, ENC_D, False),
            ("encoder_ffn_down", self.encoder_seq_len, ENC_D, ENC_H, False),
            (
                "decoder_qkv",
                self.chunk_size,
                (DEC_NH + 2 * DEC_NKV) * DEC_HD,
                DEC_D,
                False,
            ),
            ("decoder_attn_o", self.chunk_size, DEC_D, DEC_NH * DEC_HD, False),
            ("decoder_ffn_gate_up", self.chunk_size, 2 * DEC_H, DEC_D, False),
            ("decoder_ffn_down", self.chunk_size, DEC_D, DEC_H, False),
        ]

        max_mk = max(m * k for _name, m, _n, k, _bias in shapes)
        max_nk = max(n * k for _name, _m, n, k, _bias in shapes)
        max_mn = max(m * n for _name, m, n, _k, _bias in shapes)
        max_n = max(n for _name, _m, n, _k, bias in shapes if bias)

        x = HipBuffer.device_zeros(max_mk, FP8_BYTES)
        w = HipBuffer.device_zeros(max_nk, FP8_BYTES)
        out = HipBuffer.device_empty(max_mn, BF16_NP)
        bias = HipBuffer.device_zeros(max_n, BF16_NP)
        x_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        w_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)

        before = (
            rocm_kernels.hipblaslt_algo_cache_size()
            if hasattr(rocm_kernels, "hipblaslt_algo_cache_size")
            else None
        )
        for _name, m, n, k, has_bias in shapes:
            rocm_kernels.hipblaslt_linear_fp8_e4m3fnuz_bf16_ptr(
                x.ptr.value,
                w.ptr.value,
                x_scale.data_ptr(),
                w_scale.data_ptr(),
                bias.ptr.value if has_bias else 0,
                out.ptr.value,
                int(m),
                int(n),
                int(k),
            )
        after = (
            rocm_kernels.hipblaslt_algo_cache_size()
            if hasattr(rocm_kernels, "hipblaslt_algo_cache_size")
            else None
        )
        return {
            "shapes": [name for name, *_rest in shapes],
            "algo_cache_size_before": before,
            "algo_cache_size_after": after,
        }

    def vision_bias_residual_layer_norm(
        self,
        rocm_kernels,
        residual_src_name: str,
        bias_ptr: int,
        norm_weight_ptr: int,
        norm_bias_ptr: int,
        eps: float = 1e-5,
        stream: int = 0,
    ) -> None:
        """Fuse ``vision_x += residual + bias`` and next SigLIP LayerNorm."""
        rocm_kernels.bias_residual_layer_norm_bf16_ptr(
            self.bufs["vision_x"].ptr.value,
            self.bufs[residual_src_name].ptr.value,
            int(bias_ptr),
            int(norm_weight_ptr),
            int(norm_bias_ptr),
            self.bufs["vision_x_norm"].ptr.value,
            self.vision_seq,
            VIS_D,
            float(eps),
            stream,
        )

    def run_pipeline(self, stream: int = 0) -> None:
        if self._rocm_kernels is None or self._weights is None:
            raise RuntimeError("Pi05PipelineRocm runtime is not configured")
        if self._runtime_use_fp8:
            self.run_fp8_pipeline(
                self._rocm_kernels,
                self._weights,
                vision_num_layers=self._runtime_vision_num_layers,
                encoder_num_layers=self._runtime_encoder_num_layers,
                decoder_num_layers=self._runtime_decoder_num_layers,
                stream=stream,
            )
        else:
            self.run_bf16_pipeline(
                self._rocm_kernels,
                self._weights,
                vision_num_layers=self._runtime_vision_num_layers,
                encoder_num_layers=self._runtime_encoder_num_layers,
                decoder_num_layers=self._runtime_decoder_num_layers,
                stream=stream,
            )

    def forward(self) -> int:
        if self._runtime_use_fp8:
            graph = getattr(self, "_fp8_graph", None)
            if graph is not None:
                self.replay_fp8_graph()
                return self.bufs["diffusion_noise"].ptr.value
        else:
            graph = getattr(self, "_bf16_graph", None)
            if graph is not None:
                self.replay_bf16_graph()
                return self.bufs["diffusion_noise"].ptr.value
        self.run_pipeline(stream=0)
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        return self.bufs["diffusion_noise"].ptr.value


__all__ = [
    "Pi05PipelineRocm",
    "VIS_L",
    "VIS_D",
    "VIS_H",
    "ENC_L",
    "ENC_D",
    "ENC_H",
    "DEC_L",
    "DEC_D",
    "DEC_H",
    "ACTION_DIM",
    "NUM_STEPS_DEFAULT",
]
