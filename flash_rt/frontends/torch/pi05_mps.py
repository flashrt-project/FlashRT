"""Experimental Pi0.5 BF16 frontend for Apple Silicon, without native extensions."""

from __future__ import annotations

import json
import math
import pathlib
import platform
import time

import numpy as np
import torch

from flash_rt.core.utils.actions import unnormalize_actions
from flash_rt.core.utils.norm_stats import load_norm_stats, pi05_candidates
from flash_rt.frontends.torch.pi05_checkpoint import (
    ACTION_DIM, IMG_HW, _embed_prompt, convert_pi05_safetensors,
)
from flash_rt.models.pi05.pipeline_mps import Pi05PipelineMps
from flash_rt.models.pi05.torch_pipeline import DEC_D, ENC_D


def require_mps() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("apple_mps requires a native arm64 Python on Apple Silicon macOS")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; install an MPS-enabled PyTorch build")
    try:
        # Check the actual runtime before loading multi-GB checkpoint weights.
        torch.ones(1, dtype=torch.bfloat16, device="mps").square()
        torch.mps.synchronize()
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError("Apple MPS BF16 requires macOS 14+ and a compatible PyTorch build") from exc


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _prepare_weights(weights, num_steps):
    # The converter supplies HWIO patches and unscaled action projections;
    # the tensor pipeline consumes OIHW and Euler increments, respectively.
    weights["vision_patch_embedding_w"] = (
        weights["vision_patch_embedding_w"].permute(3, 2, 0, 1).contiguous())
    for key in ("decoder_action_out_proj_w", "decoder_action_out_proj_b"):
        weights[key] = weights[key] * (-1.0 / num_steps)
    for stage in ("encoder", "decoder"):
        weights[f"{stage}_ffn_gate_up_w"] = torch.cat(
            (weights.pop(f"{stage}_ffn_gate_w"), weights.pop(f"{stage}_ffn_up_w")), -1)
    fraction = torch.linspace(0, 1, DEC_D // 2, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    t = torch.tensor(1.0)
    rows = []
    for _ in range(num_steps):
        phase = t * period.reciprocal() * (2.0 * math.pi)
        rows.append(torch.cat((phase.sin(), phase.cos())).bfloat16())
        t = t + (-1.0 / num_steps)
    weights["decoder_time_embeds"] = torch.stack(rows)
    return weights


class Pi05TorchFrontendMps:
    """Shared Pi0.5 model execution on MPS; checkpoint accuracy is experimental.

    Requires model.safetensors, normalization statistics, and the usual
    PaliGemma tokenizer. Images are 224x224 RGB uint8 arrays. Robot state,
    when supplied, follows the existing normalized state-in-prompt contract.
    """

    _VIEW_KEYS = ("image", "wrist_image", "wrist_image_right")

    def __init__(self, checkpoint_dir, num_views=2, num_steps=10,
                 max_prompt_len=200, chunk_size=None, action_dim=None):
        if num_views not in (2, 3):
            raise ValueError("num_views must be 2 or 3")
        self.num_steps = _positive_int("num_steps", num_steps)
        self.max_prompt_len = _positive_int("max_prompt_len", max_prompt_len)
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint = checkpoint_dir / "model.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pi0.5 safetensors not found: {checkpoint}")
        config_path = checkpoint_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        self.chunk_size = _positive_int(
            "chunk_size", config.get("action_horizon", 10) if chunk_size is None else chunk_size)
        if action_dim is None:
            action_dim = config.get("output_action_dim")
        if (not isinstance(action_dim, int) or isinstance(action_dim, bool)
                or not 1 <= action_dim <= ACTION_DIM):
            raise ValueError("Set action_dim (1..32) or output_action_dim in checkpoint config.json")
        self.action_dim = action_dim
        self.norm_stats = load_norm_stats(
            pi05_candidates(checkpoint_dir), checkpoint_dir=checkpoint_dir)
        stats = self.norm_stats.get("actions", {})
        q01 = np.asarray(stats.get("q01", []), dtype=np.float32)
        q99 = np.asarray(stats.get("q99", []), dtype=np.float32)
        if (q01.ndim != 1 or q01.shape != q99.shape
                or not action_dim <= q01.size <= ACTION_DIM
                or not np.isfinite(q01).all() or not np.isfinite(q99).all()
                or np.any(q99 < q01)):
            raise ValueError("Action q01/q99 must be finite, ordered, and cover action_dim")
        require_mps()
        self.hardware = "apple_mps"
        self.num_views = num_views
        self.dtype = torch.bfloat16
        self._prompt_len = 0
        self._prompt_text = None
        self.latency_records = []
        cpu_weights = _prepare_weights(convert_pi05_safetensors(checkpoint), num_steps)
        # Embed infrequently on CPU to avoid keeping the large vocabulary table
        # on the GPU. The model's transformer weights execute entirely on MPS.
        self._embedding_weight = cpu_weights.pop("embedding_weight")
        self.weights = {key: value.to("mps").contiguous() for key, value in cpu_weights.items()}
        del cpu_weights
        self.pipeline = Pi05PipelineMps(
            self.weights,
            num_views=num_views, max_prompt_len=max_prompt_len,
            chunk_size=self.chunk_size, num_steps=num_steps,
        )
        self._prompt_buf = torch.empty(max_prompt_len, ENC_D, dtype=self.dtype, device="mps")
        self._prompt_cpu = None
        self._image_buf = torch.empty(num_views, IMG_HW, IMG_HW, 3, dtype=self.dtype, device="mps")
        self._noise_buf = torch.empty(self.chunk_size, ACTION_DIM, dtype=self.dtype, device="mps")

    @torch.inference_mode()
    def set_prompt(self, prompt_text, state=None):
        self._prompt_len = 0
        self.pipeline.invalidate_encoder_cache()
        embeds, length = _embed_prompt(
            prompt_text, self._embedding_weight, max_len=self.max_prompt_len,
            state=state, device="cpu")
        if not 0 < length <= self.max_prompt_len:
            raise ValueError(f"Prompt length {length} exceeds capacity {self.max_prompt_len}")
        self._prompt_buf[:length].copy_(embeds)
        self._prompt_cpu = embeds.detach().cpu().contiguous()
        self._prompt_len, self._prompt_text = length, prompt_text

    @torch.inference_mode()
    def forward_with_fixed_noise(self, images_nhwc, noise, *, capture_probes=False, use_graph=False):
        if use_graph:
            raise ValueError("Apple MPS uses eager execution; CUDA graphs are unavailable")
        if not self._prompt_len:
            raise RuntimeError("set_prompt must be called before inference")
        return self.pipeline.forward_with_inputs(
            images_nhwc, self._prompt_buf[:self._prompt_len], self._prompt_len,
            noise, capture_probes=capture_probes)

    @torch.inference_mode()
    def infer(self, observation, debug=False, noise=None):
        if not self._prompt_len:
            raise RuntimeError("set_prompt must be called before infer")
        if "images" in observation:
            images = list(observation["images"])
        else:
            required = self._VIEW_KEYS[:self.num_views]
            if any(key not in observation for key in required):
                raise ValueError(f"Observation requires image keys {required}")
            if self.num_views == 2 and "wrist_image_right" in observation:
                raise ValueError("Construct with num_views=3 to use a third camera")
            images = [observation[key] for key in required]
        if len(images) != self.num_views:
            raise ValueError(f"Expected {self.num_views} camera images")
        for image in images:
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.shape != (IMG_HW, IMG_HW, 3)):
                raise ValueError("Images must be (224, 224, 3) RGB uint8 numpy arrays")
        native = self.pipeline._native_pipeline
        if native is not None and native.prompt_len != self._prompt_len:
            raise ValueError("Native prompt length changed; rebuild with enable_native_pipeline")
        if native is None:
            torch.mps.synchronize()
        started = time.perf_counter()
        if native is not None:
            # Camera pixels and prompt embeddings originate on the CPU. Write
            # shared Metal inputs directly; do not bounce through Torch MPS.
            normalized = np.stack(images).astype(np.float32) / 127.5 - 1.0
            value = torch.randn(self.chunk_size, ACTION_DIM, dtype=self.dtype) if noise is None else torch.as_tensor(noise)
            if value.shape != self._noise_buf.shape:
                raise ValueError(f"Noise must have shape {tuple(self._noise_buf.shape)}")
            raw_actions = self.pipeline.forward_native(normalized, self._prompt_cpu[:self._prompt_len], value)
        else:
            for index, image in enumerate(images):
                self._image_buf[index].copy_(torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0))
            if noise is None:
                self._noise_buf.normal_()
            else:
                value = torch.as_tensor(noise)
                if value.shape != self._noise_buf.shape:
                    raise ValueError(f"Noise must have shape {tuple(self._noise_buf.shape)}")
                self._noise_buf.copy_(value)
            raw = self.forward_with_fixed_noise(self._image_buf, self._noise_buf)[0]
            raw_actions = raw.float().cpu().numpy()
        latency_ms = (time.perf_counter() - started) * 1000
        self.latency_records.append(latency_ms)
        result = {"actions": unnormalize_actions(raw_actions, self.norm_stats)[:, :self.action_dim]}
        if debug:
            result.update(raw_actions=raw_actions, latency_ms=latency_ms)
        return result

    @property
    def precision_spec(self):
        return None

    def get_latency_stats(self):
        if not self.latency_records:
            return {}
        values = np.asarray(self.latency_records)
        return dict(count=len(values), mean_ms=float(values.mean()),
                    p50_ms=float(np.percentile(values, 50)),
                    p95_ms=float(np.percentile(values, 95)),
                    p99_ms=float(np.percentile(values, 99)), hz=float(1000 / values.mean()))
