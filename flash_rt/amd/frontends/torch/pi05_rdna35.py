"""BF16 Pi0.5 torch frontend for AMD RDNA 3.5.

The backend loads the RDNA extension once, then passes it directly to the
pipeline, attention provider, and instance-local hipBLASLt BF16 runner. ATen
and SDPA remain available as correctness fallbacks.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch

from flash_rt.amd.frontends.torch.pi05 import (
    ACTION_DIM,
    IMG_HW,
    NUM_STEPS_DEFAULT,
    _embed_prompt,
    convert_pi05_safetensors,
)
from flash_rt.amd.hardware.rdna35 import (
    Rdna35AttentionBackend,
    Rdna35GemmBackend,
)
from flash_rt.amd.models.pi05_rdna35.pipeline import (
    DEC_D,
    ENC_D,
    Pi05PipelineRdna35,
)
from flash_rt.core.utils.actions import unnormalize_actions
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates


def _build_time_embeddings(num_steps: int) -> torch.Tensor:
    """Build the Pi0.5 flow-matching time schedule for ``num_steps``.

    The converted checkpoint contains the default ten-step table.  The ODE
    step size and the sampled times are coupled, so supporting a different
    step count requires regenerating the table rather than slicing it.
    """
    if (
        not isinstance(num_steps, int)
        or isinstance(num_steps, bool)
        or num_steps < 1
    ):
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}")
    fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    time = torch.tensor(1.0, dtype=torch.float32)
    step = -1.0 / num_steps
    rows = []
    for _ in range(num_steps):
        sinusoid = time * period.reciprocal() * (2.0 * math.pi)
        rows.append(
            torch.cat((sinusoid.sin(), sinusoid.cos())).to(torch.bfloat16)
        )
        time = time + step
    return torch.stack(rows)


class Pi05TorchFrontendAmdRdna35:
    """Correctness-first Pi0.5 frontend for AMD RDNA 3.5."""

    _VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        num_steps: int = NUM_STEPS_DEFAULT,
        max_prompt_len: int = 200,
        chunk_size: Optional[int] = None,
        use_fp8: bool = False,
        hardware: Optional[str] = None,
    ):
        if hardware not in (None, "amd_rdna35"):
            raise ValueError(
                "Pi05TorchFrontendAmdRdna35 requires hardware='amd_rdna35', "
                f"got {hardware!r}")
        if use_fp8:
            raise ValueError("amd_rdna35 currently supports BF16 only")
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
            raise RuntimeError("amd_rdna35 requires a ROCm build of PyTorch")
        device_arch = str(
            getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unknown"))
        if device_arch.split(":", 1)[0] != "gfx1151":
            raise RuntimeError(
                "Pi05TorchFrontendAmdRdna35 is gfx1151-only: running device "
                f"architecture is {device_arch!r}")

        from flash_rt.amd import flash_rt_amd_kernels as fvk

        build_info = dict(fvk.build_info())
        build_arch = str(build_info.get("gpu_arch", "")).split(":", 1)[0]
        extension_device_arch = str(fvk.device_arch()).split(":", 1)[0]
        if (
            build_info.get("backend") != "rdna35"
            or build_info.get("wave_size") != 32
        ):
            raise RuntimeError(
                "flash_rt_amd_kernels was not built with the RDNA source set")
        if build_arch != "gfx1151" or extension_device_arch != "gfx1151":
            raise RuntimeError(
                "RDNA 3.5 HIP kernels require a gfx1151 build and device: "
                f"build={build_arch!r} device={extension_device_arch!r}")

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint = checkpoint_dir / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pi0.5 safetensors not found: {checkpoint}")
        if num_views not in (2, 3):
            raise ValueError(
                "num_views must be 2 (base + wrist camera) or 3 "
                f"(+ right wrist camera), got {num_views}")
        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 1
        ):
            raise ValueError(
                f"num_steps must be a positive integer, got {num_steps!r}")
        if (
            not isinstance(max_prompt_len, int)
            or isinstance(max_prompt_len, bool)
            or max_prompt_len < 1
        ):
            raise ValueError(
                f"max_prompt_len must be a positive integer, got {max_prompt_len!r}")

        if chunk_size is None:
            chunk_size = self._checkpoint_action_horizon(checkpoint_dir)
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.hardware = "amd_rdna35"
        self.num_views = int(num_views)
        self.num_steps = int(num_steps)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)
        self.dtype = torch.bfloat16
        self.latency_records: list[float] = []
        self._prompt_len = 0
        self._prompt_text: Optional[str] = None
        self._checkpoint_path = str(checkpoint)

        self.norm_stats = load_norm_stats(
            pi05_candidates(checkpoint_dir), checkpoint_dir=checkpoint_dir)
        self.action_dim = self._infer_action_dim(self.norm_stats)

        cpu_weights = convert_pi05_safetensors(checkpoint)
        cpu_weights["decoder_time_embeds"] = _build_time_embeddings(
            self.num_steps)
        # The native CDNA4 pipeline consumes flattened HWIO patch weights;
        # torch.conv2d expects OIHW.
        cpu_weights["vision_patch_embedding_w"] = (
            cpu_weights["vision_patch_embedding_w"]
            .permute(3, 2, 0, 1)
            .contiguous()
        )
        scale = -1.0 / self.num_steps
        cpu_weights["decoder_action_out_proj_w"] = (
            cpu_weights["decoder_action_out_proj_w"] * scale).contiguous()
        cpu_weights["decoder_action_out_proj_b"] = (
            cpu_weights["decoder_action_out_proj_b"] * scale).contiguous()
        cpu_weights["decoder_ffn_gate_up_w"] = torch.cat(
            (cpu_weights.pop("decoder_ffn_gate_w"),
             cpu_weights.pop("decoder_ffn_up_w")),
            dim=-1,
        ).contiguous()
        cpu_weights["encoder_ffn_gate_up_w"] = torch.cat(
            (cpu_weights.pop("encoder_ffn_gate_w"),
             cpu_weights.pop("encoder_ffn_up_w")),
            dim=-1,
        ).contiguous()
        self.weights = {
            key: value.to("cuda", non_blocking=False).contiguous()
            for key, value in cpu_weights.items()
        }
        del cpu_weights

        gemm = Rdna35GemmBackend(fvk, self.dtype)
        gemm.prepare_smallm_weight(
            self.weights["decoder_action_out_proj_w"])
        self.pipeline = Pi05PipelineRdna35(
            self.weights,
            fvk,
            gemm,
            Rdna35AttentionBackend(fvk),
            num_views=self.num_views,
            max_prompt_len=self.max_prompt_len,
            chunk_size=self.chunk_size,
            num_steps=self.num_steps,
            dtype=self.dtype,
        )
        self._prompt_buf = torch.empty(
            self.max_prompt_len, ENC_D, dtype=self.dtype, device="cuda")
        self._image_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3,
            dtype=self.dtype, device="cuda")
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=self.dtype, device="cuda")

    @staticmethod
    def _checkpoint_action_horizon(checkpoint_dir: pathlib.Path) -> int:
        config_path = checkpoint_dir / "config.json"
        if config_path.is_file():
            try:
                value = json.loads(config_path.read_text()).get("action_horizon")
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return 10

    @staticmethod
    def _infer_action_dim(norm_stats: dict) -> int:
        actions = norm_stats.get("actions", {})
        q01 = np.asarray(actions.get("q01", []), dtype=np.float32)
        q99 = np.asarray(actions.get("q99", []), dtype=np.float32)
        if q01.shape != q99.shape or q01.ndim != 1 or q01.size == 0:
            raise ValueError("Pi0.5 norm_stats actions must contain 1-D q01/q99")
        active = np.flatnonzero(np.abs(q99 - q01) > 1e-8)
        return int(active[-1] + 1) if active.size else int(q01.size)

    def set_prompt(self, prompt_text: str, state=None) -> None:
        embeds, prompt_len = _embed_prompt(
            prompt_text,
            self.weights["embedding_weight"],
            max_len=self.max_prompt_len,
            state=state,
        )
        if prompt_len > self.max_prompt_len:
            raise ValueError(
                f"prompt length {prompt_len} exceeds capacity "
                f"{self.max_prompt_len}")
        self._prompt_buf[:prompt_len].copy_(embeds.to(self.dtype))
        self._prompt_len = prompt_len
        self._prompt_text = prompt_text

    def _gather_view_images(self, observation: dict) -> list[np.ndarray]:
        if "images" in observation:
            images = list(observation["images"])
            if len(images) != self.num_views:
                raise ValueError(
                    f"observation['images'] has {len(images)} entries; "
                    f"expected {self.num_views}")
        else:
            required = self._VIEW_KEYS[:self.num_views]
            missing = [key for key in required if key not in observation]
            if missing:
                raise ValueError(f"observation is missing image key(s) {missing}")
            if self.num_views < 3 and "wrist_image_right" in observation:
                raise ValueError(
                    "observation provides 'wrist_image_right' but this "
                    f"frontend was built with num_views={self.num_views}; "
                    "construct it with num_views=3 to use a third view")
            images = [observation[key] for key in required]
        for index, image in enumerate(images):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise ValueError(f"image {index} must be a uint8 numpy array")
            if image.shape != (IMG_HW, IMG_HW, 3):
                raise ValueError(
                    f"image {index} has shape {image.shape}; expected "
                    f"({IMG_HW}, {IMG_HW}, 3)")
        return images

    def _fill_images(self, observation: dict) -> None:
        for index, image in enumerate(self._gather_view_images(observation)):
            value = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
            self._image_buf[index].copy_(value.to(self.dtype))

    @torch.inference_mode()
    def forward_with_fixed_noise(
        self,
        images_nhwc: torch.Tensor,
        noise: torch.Tensor,
        *,
        capture_probes: bool = False,
        use_graph: bool = False,
    ) -> torch.Tensor:
        if self._prompt_len == 0:
            raise RuntimeError("set_prompt must be called before inference")
        return self.pipeline.forward_with_inputs(
            images_nhwc,
            self._prompt_buf[:self._prompt_len],
            self._prompt_len,
            noise,
            capture_probes=capture_probes,
            use_graph=use_graph,
        )

    @torch.inference_mode()
    def infer(
        self,
        observation: dict,
        debug: bool = False,
        noise: Optional[np.ndarray] = None,
    ) -> dict:
        if self._prompt_len == 0:
            raise RuntimeError("set_prompt must be called before infer")
        started = time.perf_counter()
        self._fill_images(observation)
        if noise is None:
            self._noise_buf.normal_()
        else:
            value = torch.as_tensor(np.asarray(noise))
            if value.shape != self._noise_buf.shape:
                raise ValueError(
                    f"noise has shape {tuple(value.shape)}; expected "
                    f"{tuple(self._noise_buf.shape)}")
            self._noise_buf.copy_(value.to(self.dtype))
        raw = self.forward_with_fixed_noise(
            self._image_buf,
            self._noise_buf,
        )[0]
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        self.latency_records.append(latency_ms)

        raw_actions = raw.float().cpu().numpy()
        actions = unnormalize_actions(raw_actions, self.norm_stats)[
            :, :self.action_dim]
        result = {"actions": actions}
        if debug:
            result.update({
                "raw_actions": raw_actions,
                "latency_ms": latency_ms,
            })
        return result

    @property
    def precision_spec(self):
        return None

    def get_latency_stats(self) -> dict:
        if not self.latency_records:
            return {}
        values = np.asarray(self.latency_records)
        return {
            "count": len(values),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "mean_ms": float(values.mean()),
            "hz": float(1000.0 / values.mean()),
        }
