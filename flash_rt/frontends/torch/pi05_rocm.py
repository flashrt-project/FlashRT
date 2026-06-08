"""ROCm Pi0.5 torch frontend.

This frontend runs the openpi PyTorch model on ROCm and installs FlashRT ROCm
kernels at stable module boundaries. The optimized ROCm pipeline lives in
``flash_rt.models.pi05.pipeline_rocm``; this frontend provides the public
``load_model`` route with real prompt and observation handling.
"""

from __future__ import annotations

import dataclasses
import ctypes
import logging
import os
import pathlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def _ensure_openpi_transformers_patch() -> None:
    """Best-effort check for openpi's patched transformers modules.

    The openpi PyTorch model expects its replacement Gemma/SigLIP modules to
    be installed into the active transformers package. Keep this check explicit
    so failures explain the environment issue instead of surfacing as a deep
    modeling error.
    """
    try:
        from transformers.models.siglip import check
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "openpi transformers patch is not visible. Copy "
            "openpi/src/openpi/models_pytorch/transformers_replace/* into "
            "the active transformers package before loading Pi0.5 ROCm."
        ) from exc
    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise RuntimeError(
            "openpi transformers patch check failed for the active environment."
        )


def _find_safetensors(checkpoint: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(checkpoint)
    if path.is_file():
        return path
    direct = path / "model.safetensors"
    if direct.exists():
        return direct
    shards = sorted(path.glob("*.safetensors"))
    if len(shards) == 1:
        return shards[0]
    raise FileNotFoundError(
        f"Could not find a single safetensors checkpoint under {path}"
    )


def _load_openpi_model(checkpoint: str | pathlib.Path):
    try:
        from openpi.training import config as openpi_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Pi05TorchFrontendRocm currently requires openpi to be importable. "
            "Use a ROCm environment with openpi available on PYTHONPATH."
        ) from exc

    _ensure_openpi_transformers_patch()

    cfg = openpi_config.get_config("pi05_libero")
    model_cfg = dataclasses.replace(cfg.model, pytorch_compile_mode=None)
    model = PI0Pytorch(config=model_cfg)

    weight_path = _find_safetensors(checkpoint)
    state = load_file(str(weight_path), device="cpu")
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}

    embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if embed_key not in state and lm_key in state:
        state[embed_key] = state[lm_key]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load Pi0.5 ROCm checkpoint: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval().to("cuda")
    torch.cuda.synchronize()
    return model, model_cfg, weight_path


def _probe_rocm_fp8_matmul(rocm_kernels) -> bool:
    if not hasattr(torch, "float8_e4m3fnuz"):
        return False

    try:
        a = torch.randn(16, 32, device="cuda", dtype=torch.float32) * 0.25
        b = torch.randn(32, 16, device="cuda", dtype=torch.float32) * 0.25
        a_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        b_scale = torch.tensor([0.05], device="cuda", dtype=torch.float32)
        a8 = (a / a_scale).to(torch.float8_e4m3fnuz)
        b8 = (b / b_scale).to(torch.float8_e4m3fnuz)
        out = rocm_kernels.hipblaslt_matmul_fp8_e4m3fnuz_bf16(
            a8, b8, a_scale, b_scale
        )
        rocm_kernels.hip_sync()
    except RuntimeError as exc:
        if "hipBLASLt did not return a usable" not in str(exc):
            raise
        return False
    return bool(torch.isfinite(out).all().item())


class Pi05TorchFrontendRocm:
    """Pi0.5 ROCm frontend.

    Public methods match the FlashRT VLA frontend contract used by
    :class:`flash_rt.api.VLAModel`: ``set_prompt``, ``infer``,
    ``calibrate_with_real_data``, and ``get_latency_stats``.
    """

    def __init__(
        self,
        checkpoint: str,
        num_views: int = 2,
        chunk_size: Optional[int] = None,
        autotune: int = 3,
        hardware: Optional[str] = None,
        num_steps: Optional[int] = None,
        use_fp8: bool = False,
        attn_backend: str = "ck_wmma",
        decoder_attn_backend: Optional[str] = "ck_wmma",
        **_unused,
    ):
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError(
                "Pi05TorchFrontendRocm requires ROCm PyTorch "
                "(torch.version.hip must be set)."
            )

        from flash_rt import flash_rt_rocm_kernels as rocm_kernels

        self.checkpoint = str(checkpoint)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size or 10)
        self.autotune = autotune
        self.hardware = hardware or "rocm"
        self._num_steps = int(num_steps or 10)
        self._attn_backend = str(attn_backend)
        self._decoder_attn_backend = decoder_attn_backend
        self._prompt_text: str | None = None
        self._latency_records: list[float] = []
        self.fvk = rocm_kernels
        self.use_fp8 = bool(use_fp8)
        self._rocm_fp8_matmul_available = False
        if self.use_fp8:
            self._rocm_fp8_matmul_available = _probe_rocm_fp8_matmul(self.fvk)
            if not self._rocm_fp8_matmul_available:
                logger.warning(
                    "ROCm FP8 hipBLASLt matmul is unavailable; using BF16 "
                    "execution for Pi0.5."
                )
                self.use_fp8 = False

        t0 = time.perf_counter()
        self._model, self._model_cfg, self._weight_path = _load_openpi_model(checkpoint)
        from openpi.models.tokenizer import PaligemmaTokenizer
        from flash_rt.core.hip_buffer import _hip, _check
        from flash_rt.frontends.torch.pi05_rocm_weights import (
            build_rocm_vision_weights_from_openpi_model,
        )
        from flash_rt.models.pi05.pipeline_rocm import (
            ACTION_DIM,
            ENC_D,
            Pi05PipelineRocm,
        )

        self._tokenizer = PaligemmaTokenizer(max_len=self._model_cfg.max_token_len)
        self._hip = _hip
        self._hip_check = _check
        self._action_dim = ACTION_DIM
        self._enc_dim = ENC_D
        self._pipeline_cls = Pi05PipelineRocm
        self._embedding_weight = (
            self._model.paligemma_with_expert
            .paligemma
            .model
            .language_model
            .embed_tokens
            .weight
            .detach()
            .to(device="cuda", dtype=torch.bfloat16)
            .contiguous()
        )
        self._weights = build_rocm_vision_weights_from_openpi_model(
            self._model,
            chunk_size=self.chunk_size,
            num_steps=self._num_steps,
            include_fp8=self.use_fp8,
        )
        self._pipeline = None
        self._current_prompt_len = 0
        self._graph_recorded = False
        self._graph_torch_stream = torch.cuda.Stream()
        self._img_buf = torch.empty(
            self.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"
        )
        self._noise_buf = torch.zeros(
            self.chunk_size, ACTION_DIM, dtype=torch.bfloat16, device="cuda"
        )
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=torch.bfloat16, device="cuda"
        )
        self._rocm_rms_norm_modules = 0
        self._rocm_linear_modules = 0
        self._rocm_fp8_linear_modules = 0
        self._rocm_fp8_mlp_fused_modules = 0
        self._rocm_bf16_mlp_fused_modules = 0
        self._rocm_mlp_activation_modules = 0
        self._rocm_siglip_mlp_activation_modules = 0
        self._rocm_fp8_sites = []
        self._rocm_fp8_calibrated = not self.use_fp8
        self.fvk.hip_sync()
        logger.info(
            "Pi05TorchFrontendRocm loaded %s in %.3fs (pipeline bf16 graph path)",
            self._weight_path,
            time.perf_counter() - t0,
        )

    def _embed_prompt(self, prompt_text: str, state=None):
        tokens_np, mask_np = self._tokenizer.tokenize(prompt_text, state=state)
        prompt_len = int(mask_np.sum())
        token_ids = torch.as_tensor(
            tokens_np[:prompt_len],
            dtype=torch.long,
            device="cuda",
        )
        embeds = F.embedding(token_ids, self._embedding_weight)
        embeds = embeds * float(embeds.shape[-1] ** 0.5)
        return embeds.contiguous(), prompt_len

    def _ensure_pipeline(self, prompt_len: int):
        if self._pipeline is not None and prompt_len == self._current_prompt_len:
            return
        self._pipeline = self._pipeline_cls.with_rocm_attention(
            num_views=self.num_views,
            max_prompt_len=int(prompt_len),
            chunk_size=self.chunk_size,
            num_steps=self._num_steps,
            preferred_backend=self._attn_backend,
            decoder_preferred_backend=self._decoder_attn_backend,
        )
        self._pipeline.configure_runtime(
            self.fvk,
            self._weights,
            use_fp8=self.use_fp8,
        )
        self._current_prompt_len = int(prompt_len)
        self._graph_recorded = False

    def _set_prompt_embeds(self, prompt_text: str, state=None) -> None:
        embeds, prompt_len = self._embed_prompt(prompt_text, state=state)
        self._ensure_pipeline(prompt_len)
        embeds_np = embeds.view(torch.uint16).cpu().numpy()
        self._pipeline.set_language_embeds(embeds_np)

    def set_prompt(self, prompt_text: str, state=None) -> None:
        self._prompt_text = prompt_text
        self._set_prompt_embeds(prompt_text, state=state)

    def calibrate_with_real_data(self, sample_observations) -> None:
        observations = list(sample_observations or [])
        if not observations:
            raise ValueError("ROCm graph preparation requires at least one observation")
        observation = observations[0]
        state = observation.get("state")
        prompt_text = "" if self._prompt_text is None else self._prompt_text
        self._set_prompt_embeds(prompt_text, state=state)
        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = int(self._graph_torch_stream.cuda_stream)
            self._fill_img_buf(observation)
            self._noise_buf.zero_()
            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf, self._pipeline.input_images_buf, stream_int
            )
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf, self._pipeline.input_noise_buf, stream_int
            )
            if self.use_fp8:
                self._pipeline.calibrate_fp8(self.fvk, self._weights, stream=stream_int)
                self._pipeline.capture_fp8_graph(self.fvk, self._weights)
                self._rocm_fp8_calibrated = True
            else:
                self._pipeline.capture_bf16_graph(self.fvk, self._weights)
        torch.cuda.synchronize()
        self._graph_recorded = True
        return None

    calibrate = calibrate_with_real_data

    def _images_from_observation(self, observation: dict) -> list[np.ndarray]:
        if "images" in observation:
            images = list(observation["images"])
        else:
            images = [observation["image"]]
            if "wrist_image" in observation:
                images.append(observation["wrist_image"])
            if "wrist_image_right" in observation:
                images.append(observation["wrist_image_right"])

        if len(images) == 1:
            images = [images[0], images[0], images[0]]
        elif len(images) == 2:
            images = [images[0], images[1], images[1]]
        else:
            images = images[:3]
        return [np.asarray(img) for img in images]

    def _fill_img_buf(self, observation: dict) -> None:
        images = self._images_from_observation(observation)
        for v, img in enumerate(images[: self.num_views]):
            if img.shape != (224, 224, 3):
                raise ValueError(
                    f"ROCm Pi0.5 frontend expects 224x224x3 uint8 images, got {img.shape}"
                )
            norm = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(device="cuda", dtype=torch.bfloat16))

    def _copy_tensor_to_pipeline_buf_stream(self, src: torch.Tensor, dst_buf, stream_int: int):
        nbytes = src.numel() * src.element_size()
        if nbytes != dst_buf.nbytes:
            raise ValueError(f"size mismatch: src {nbytes} vs dst {dst_buf.nbytes}")
        self._hip_check(
            self._hip.hipMemcpyAsync(
                dst_buf.ptr,
                ctypes.c_void_p(int(src.data_ptr())),
                nbytes,
                3,
                ctypes.c_void_p(int(stream_int)),
            ),
            "hipMemcpyAsync tensor to Pi0.5 ROCm buffer",
        )

    def _make_observation(self, observation: dict):
        from openpi.models import model as openpi_model

        images = self._images_from_observation(observation)
        batch = 1
        image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        image_tensors = {}
        for key, img in zip(image_keys, images, strict=True):
            if img.shape != (224, 224, 3):
                raise ValueError(
                    f"ROCm Pi0.5 frontend expects 224x224x3 uint8 images, got {img.shape}"
                )
            image_tensors[key] = torch.as_tensor(
                img, device="cuda", dtype=torch.uint8
            ).reshape(batch, 224, 224, 3)

        if "state" in observation and observation["state"] is not None:
            state_np = np.asarray(observation["state"], dtype=np.float32)
            if state_np.ndim == 1:
                state_np = state_np[None, :]
            if state_np.shape[-1] < self._model_cfg.action_dim:
                pad = self._model_cfg.action_dim - state_np.shape[-1]
                state_np = np.pad(state_np, ((0, 0), (0, pad)))
            state_np = state_np[:, : self._model_cfg.action_dim]
            state = torch.as_tensor(state_np, device="cuda", dtype=torch.float32)
        else:
            state = torch.zeros(
                (batch, self._model_cfg.action_dim),
                device="cuda",
                dtype=torch.float32,
            )

        prompt_text = "" if self._prompt_text is None else self._prompt_text
        state_for_prompt = None
        if "state" in observation and observation["state"] is not None:
            state_for_prompt = np.asarray(observation["state"], dtype=np.float32)
            if state_for_prompt.ndim > 1:
                state_for_prompt = state_for_prompt[0]
        tokens_np, mask_np = self._tokenizer.tokenize(
            prompt_text,
            state=state_for_prompt,
        )
        tokenized_prompt = torch.as_tensor(
            tokens_np[None, :],
            dtype=torch.int64,
            device="cuda",
        )
        tokenized_prompt_mask = torch.as_tensor(
            mask_np[None, :],
            dtype=torch.bool,
            device="cuda",
        )

        return openpi_model.Observation.from_dict(
            {
                "image": image_tensors,
                "image_mask": {
                    key: torch.ones((batch,), dtype=torch.bool, device="cuda")
                    for key in image_keys
                },
                "state": state,
                "tokenized_prompt": tokenized_prompt,
                "tokenized_prompt_mask": tokenized_prompt_mask,
            }
        )

    def infer(self, observation: dict, debug: bool = False) -> dict:
        state = observation.get("state")
        prompt_text = "" if self._prompt_text is None else self._prompt_text
        self._set_prompt_embeds(prompt_text, state=state)

        with torch.inference_mode(), torch.cuda.stream(self._graph_torch_stream):
            stream_int = int(self._graph_torch_stream.cuda_stream)
            self._fill_img_buf(observation)
            self._noise_buf.zero_()
            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf, self._pipeline.input_images_buf, stream_int
            )
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf, self._pipeline.input_noise_buf, stream_int
            )
            if not self._graph_recorded:
                if self.use_fp8:
                    self._pipeline.calibrate_fp8(self.fvk, self._weights, stream=stream_int)
                    self._pipeline.capture_fp8_graph(self.fvk, self._weights)
                    self._rocm_fp8_calibrated = True
                else:
                    self._pipeline.capture_bf16_graph(self.fvk, self._weights)
                self._graph_recorded = True

            t0 = time.perf_counter()
            out_ptr = self._pipeline.forward()
            self._hip_check(
                self._hip.hipMemcpyAsync(
                    ctypes.c_void_p(int(self._noise_out.data_ptr())),
                    ctypes.c_void_p(int(out_ptr)),
                    self._noise_out.numel() * self._noise_out.element_size(),
                    3,
                    ctypes.c_void_p(stream_int),
                ),
                "hipMemcpyAsync Pi0.5 ROCm output",
            )
            self._hip_check(
                self._hip.hipStreamSynchronize(ctypes.c_void_p(stream_int)),
                "hipStreamSynchronize Pi0.5 ROCm infer",
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._latency_records.append(elapsed_ms)
        actions = self._noise_out.float().cpu().numpy()
        result = {"actions": actions}
        if debug:
            result["debug"] = {
                "backend": "rocm",
                "infer_ms": elapsed_ms,
                "prompt_tokens": int(self._current_prompt_len),
                "pipeline_graph_recorded": bool(self._graph_recorded),
                "pipeline_path": "fp8_graph" if self.use_fp8 else "bf16_graph",
                "attn_backend": self._pipeline.attn.active_backend_name,
                "decoder_attn_backend": self._pipeline.attn.decoder_backend_name,
                "rocm_rms_norm_modules": self._rocm_rms_norm_modules,
                "rocm_linear_modules": self._rocm_linear_modules,
                "rocm_fp8_linear_modules": self._rocm_fp8_linear_modules,
                "rocm_fp8_mlp_fused_modules": self._rocm_fp8_mlp_fused_modules,
                "rocm_bf16_mlp_fused_modules": self._rocm_bf16_mlp_fused_modules,
                "rocm_mlp_activation_modules": self._rocm_mlp_activation_modules,
                "rocm_siglip_mlp_activation_modules": self._rocm_siglip_mlp_activation_modules,
                "fp8_requested": self.use_fp8,
                "fp8_matmul_available": self._rocm_fp8_matmul_available,
                "fp8_model_weights": self._rocm_fp8_linear_modules > 0,
                "fp8_static_calibrated": self._rocm_fp8_calibrated,
                "fp8_linear_scope": os.environ.get(
                    "FLASHRT_ROCM_FP8_LINEAR_SCOPE", "gemma_mlp"
                ),
            }
        return result

    def get_latency_stats(self) -> dict:
        if not self._latency_records:
            return {}
        arr = np.asarray(self._latency_records, dtype=np.float64)
        return {
            "count": int(arr.size),
            "last_ms": float(arr[-1]),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }
