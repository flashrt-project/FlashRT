"""Opt-in Pi0.5 BF16/FP32 frontend for a single RGB view on Jetson AGX Thor."""

from pathlib import Path
from threading import Lock

import numpy as np
import torch

from flash_rt.core.utils.norm_stats import load_norm_stats
from flash_rt.utils.paligemma_tokenizer import (
    encode_pi05_prompt,
    load_paligemma_sentencepiece,
)


class Pi05TorchFrontendThorBf16:
    def __init__(self, checkpoint_dir, *, use_cuda_graph=True):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
            raise RuntimeError("The Pi0.5 Thor BF16 frontend requires a CUDA SM110 device")
        from flash_rt.models.pi05.pipeline_thor_bf16 import Pi05ThorNativeBf16
        from flash_rt.models.pi05.thor_bf16_weights import load_thor_bf16_weights

        checkpoint_dir = Path(checkpoint_dir)
        stats = load_norm_stats(
            [checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json"]
        )
        action_stats = stats.get("actions", {})
        self._q01 = np.asarray(action_stats.get("q01", []), dtype=np.float64)
        self._q99 = np.asarray(action_stats.get("q99", []), dtype=np.float64)
        if (
            self._q01.shape != (7,)
            or self._q99.shape != (7,)
            or not np.isfinite(self._q01).all()
            or not np.isfinite(self._q99).all()
            or np.any(self._q99 < self._q01)
        ):
            raise ValueError(
                "Pi0.5 Thor BF16 requires finite, ordered q01/q99 for 7 LIBERO actions"
            )

        self._weights = load_thor_bf16_weights(checkpoint_dir / "model.safetensors")
        self._pipeline_cls = Pi05ThorNativeBf16
        self._tokenizer = load_paligemma_sentencepiece()
        self._use_cuda_graph = use_cuda_graph
        self._lock = Lock()
        self._prompt = None
        self._pipeline = None
        self._graph = None
        self._image = torch.zeros((1, 3, 224, 224), device="cuda", dtype=torch.float32)
        self._noise = torch.zeros((1, 10, 32), device="cuda", dtype=torch.float32)

    @torch.no_grad()
    def set_prompt(self, prompt, state=None):
        if state is not None:
            raise ValueError("The single-view Pi0.5 Thor BF16 frontend supports text-only prompts")
        if not isinstance(prompt, str):
            raise TypeError("Pi0.5 Thor BF16 expects a text prompt")
        with self._lock:
            if prompt == self._prompt:
                return
            ids = encode_pi05_prompt(self._tokenizer, prompt)
            if len(ids) > 200:
                raise ValueError("Pi0.5 Thor BF16 accepts at most 200 prompt tokens")
            tokens = np.zeros(200, dtype=np.int32)
            tokens[: len(ids)] = ids
            mask = np.arange(200) < len(ids)
            self._graph = None
            self._graph_output = None
            self._pipeline = None
            self._prompt = None
            pipeline = self._pipeline_cls(self._weights, tokens, mask)
            if self._use_cuda_graph:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        pipeline.execute(self._image, self._noise)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = pipeline.execute(self._image, self._noise)
                torch.cuda.current_stream().wait_stream(stream)
                self._graph, self._graph_output = graph, output
            self._pipeline, self._prompt = pipeline, prompt

    @torch.no_grad()
    def infer(self, observation):
        if any(
            observation.get(key) is not None
            for key in ("state", "wrist_image", "wrist_image_right")
        ):
            raise ValueError("Pi0.5 Thor BF16 accepts one RGB image and a text-only prompt")
        images = observation.get("images")
        if images is not None and len(images) != 1:
            raise ValueError("Pi0.5 Thor BF16 accepts exactly one RGB image")
        rgb = np.asarray(observation.get("image") if images is None else images[0])
        if rgb.dtype != np.uint8 or rgb.shape != (224, 224, 3):
            raise ValueError("Pi0.5 Thor BF16 expects a uint8 RGB image of shape (224, 224, 3)")
        image = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
        image = np.ascontiguousarray(
            (image * np.float32(2.0) - np.float32(1.0)).transpose(2, 0, 1)[None]
        )

        noise = observation.get("noise")
        if noise is None:
            noise = np.random.standard_normal((10, 32)).astype(np.float32)
        noise = np.asarray(noise)
        if noise.dtype != np.float32 or noise.shape != (10, 32) or not np.isfinite(noise).all():
            raise ValueError("Pi0.5 Thor BF16 expects finite FP32 noise of shape (10, 32)")

        with self._lock:
            if self._pipeline is None:
                raise RuntimeError("Set a prompt before calling Pi0.5 Thor BF16 inference")
            self._image.copy_(torch.from_numpy(image))
            self._noise.copy_(torch.from_numpy(noise.copy())[None])
            if self._graph is not None:
                self._graph.replay()
                output = self._graph_output
            else:
                output = self._pipeline.execute(self._image, self._noise)
            raw = output[0].cpu().numpy()
        actions = (raw[:, :7] + np.float32(1.0)) / np.float32(2.0)
        actions = actions * (self._q99 - self._q01 + 1e-6) + self._q01
        return {"actions": actions, "raw_actions": raw}
