"""FlashRT -- LTX-2.5 22B distilled (audio+video) torch frontend for RTX SM120.

Wraps the official ``ltx-pipelines`` two-stage distilled pipeline behind
FlashRT's ``set_prompt`` / ``infer`` surface and installs FlashRT compute
swaps (attention backends now; fused NVFP4 epilogues and CUDA graph capture
in later stages).

Scope:
    * Official LTX-2.5 split-pack checkpoints (one safetensors per component).
    * NVFP4 prequantized transformer by default (static activation scales ship
      in the checkpoint -- no calibration pass).
    * RTX SM120 registration only. No CMake or pybind changes.

The LTX-2 monorepo is located through ``FLASH_RT_LTX2_ROOT`` (checkout root;
``packages/*/src`` are added to ``sys.path``) unless ``ltx_pipelines`` is
already importable in the environment.
"""

from __future__ import annotations

import gc
import logging
import os
import pathlib
import sys
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_PACKAGES = ("ltx-core", "ltx-pipelines", "ltx-kernels")


class Ltx25TorchFrontendRtx:
    """LTX-2.5 distilled two-stage pipeline frontend for RTX SM120."""

    DEFAULT_WIDTH = 1536
    DEFAULT_HEIGHT = 1024
    DEFAULT_FRAMES = 121
    DEFAULT_FPS = 24.0

    def __init__(
        self,
        checkpoint_dir: str,
        num_views: int = 1,
        attention: Optional[str] = None,
        quantization: str = "nvfp4-prequant",
        fuse: Optional[bool] = None,
        compile_mode: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        **_: Any,
    ) -> None:
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"LTX-2.5 checkpoint pack not found: {self.checkpoint_dir}")
        self.num_views = num_views
        self.dtype = dtype
        self.quantization = quantization
        self.attention = attention or os.environ.get(
            "FLASH_RT_LTX25_ATTN", "auto")
        if fuse is None:
            fuse = os.environ.get("FLASH_RT_LTX25_FUSE", "1") == "1"
        self.fuse = bool(fuse)
        self.compile_mode = compile_mode if compile_mode is not None else (
            os.environ.get("FLASH_RT_LTX25_COMPILE", "") or None)
        self.device = torch.device("cuda")
        self.prompt: Optional[str] = None
        self._pipe = None
        self._attn_label: Optional[str] = None
        self._load_seconds: Optional[float] = None
        self._last_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Official package discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_roots() -> list[pathlib.Path]:
        roots: list[pathlib.Path] = []
        for key in ("FLASH_RT_LTX2_ROOT", "LTX2_ROOT"):
            value = os.environ.get(key)
            if value:
                roots.append(pathlib.Path(value).expanduser())
        return roots

    @classmethod
    def _ensure_ltx_importable(cls) -> None:
        try:
            import ltx_pipelines  # noqa: F401
            return
        except ModuleNotFoundError as exc:
            if exc.name not in ("ltx_pipelines", "ltx_core"):
                raise

        for root in cls._candidate_roots():
            added = []
            for pkg in _PACKAGES:
                src = root / "packages" / pkg / "src"
                if src.is_dir() and str(src) not in sys.path:
                    sys.path.insert(0, str(src))
                    added.append(str(src))
            try:
                import ltx_pipelines  # noqa: F401
                return
            except ModuleNotFoundError as exc:
                if exc.name not in ("ltx_pipelines", "ltx_core"):
                    raise
                for p in added:
                    sys.path.remove(p)
                continue

        raise ModuleNotFoundError(
            "Cannot import the official LTX-2 packages. Install "
            "ltx-pipelines into the environment, or set FLASH_RT_LTX2_ROOT "
            "to an LTX-2 monorepo checkout (the directory containing "
            "packages/ltx-core and packages/ltx-pipelines).")

    # ------------------------------------------------------------------
    # Checkpoint pack resolution
    # ------------------------------------------------------------------
    def _find_one(self, subdir: str, patterns: list[str],
                  required: bool = True) -> Optional[str]:
        base = self.checkpoint_dir / subdir
        for pattern in patterns:
            hits = sorted(base.glob(pattern))
            if hits:
                return str(hits[0])
        if required:
            raise FileNotFoundError(
                f"No file matching {patterns} under {base}. The frontend "
                "expects the official LTX-2.5 split pack layout.")
        return None

    def _resolve_paths(self) -> dict[str, str]:
        transformer = self._find_one(
            "diffusion_models",
            ["*distilled-transformer-nvfp4.safetensors",
             "*distilled-transformer-bf16.safetensors"])
        text_encoder = self._find_one(
            "text_encoders", ["*with-proj*bf16.safetensors"])
        video_vae = self._find_one("vae", ["*video-vae-bf16.safetensors"])
        audio_vae = self._find_one("vae", ["*audio-vae-bf16.safetensors"])
        duration_head = self._find_one(
            "model_patches", ["*duration-head*.safetensors"], required=False)
        spatial_upsampler = self._find_one(
            "latent_upscale_models",
            ["*latent-spatial-upscaler-x2*.safetensors",
             "*spatial-upscaler*.safetensors"])
        return {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "duration_head": duration_head,
            "spatial_upsampler": spatial_upsampler,
        }

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------
    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        self._ensure_ltx_importable()
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.quantization_factory import QuantizationKind

        paths = self._resolve_paths()
        quant = None
        if self.quantization and paths["transformer"].endswith(
                "nvfp4.safetensors"):
            quant = QuantizationKind(self.quantization).to_policy(
                paths["transformer"])

        compilation = None
        if self.compile_mode:
            from ltx_core.model.transformer.compiling import CompilationConfig
            if self.compile_mode == "capture":
                compilation = CompilationConfig(
                    capture=True, seq_dim_dynamic=False)
            elif self.compile_mode == "default":
                # Per-shape specialization: the two-stage pipeline sees a
                # fixed (stage1, stage2) sequence-length pair, and the
                # dynamic-seq marks conflict with the graph breaks our
                # raw-kernel swaps introduce.
                compilation = CompilationConfig(seq_dim_dynamic=False)
            else:
                compilation = CompilationConfig(mode=self.compile_mode)

        t0 = time.perf_counter()
        pipe = DistilledPipeline(
            model_paths=ModelPaths.from_split(
                transformer_path=paths["transformer"],
                text_encoder_path=paths["text_encoder"],
                video_vae_path=paths["video_vae"],
                audio_vae_path=paths["audio_vae"],
                duration_head_path=paths["duration_head"],
            ),
            spatial_upsampler_path=paths["spatial_upsampler"],
            loras=[],
            quantization=quant,
            compilation_config=compilation,
        )

        from flash_rt.models.ltx25._attn_swap import make_ltx25_attention
        attn = make_ltx25_attention(self.attention)
        if attn is not None and getattr(attn, "label", "") != "sdpa":
            pipe.stage = pipe.stage.with_attention(attn)
        self._attn_label = getattr(attn, "label", str(self.attention))

        if self.fuse:
            from flash_rt.models.ltx25._nvfp4_ffn_swap import (
                SwapInstallingBuilder, install_nvfp4_ffn)
            if self.compile_mode == "capture":
                import functools
                from flash_rt.models.ltx25._resident_graph import (
                    CachingPromptEncoder, ResidentSwapBuilder)
                pipe.stage = pipe.stage.with_builder(ResidentSwapBuilder(
                    pipe.stage._transformer_builder,
                    [functools.partial(install_nvfp4_ffn, free_upstream=True)]))
                # A prompt the cache does not hold needs the text encoder,
                # which does not fit beside the resident transformer: the
                # encoder ends the residency lease before it runs, and the
                # stage's next build takes a fresh one.
                pipe.prompt_encoder = CachingPromptEncoder(
                    pipe.prompt_encoder, on_miss=self.release_resident)
            else:
                pipe.stage = pipe.stage.with_builder(SwapInstallingBuilder(
                    pipe.stage._transformer_builder, [install_nvfp4_ffn]))
        self._load_seconds = time.perf_counter() - t0
        logger.info("[ltx25] pipeline ready in %.1fs (attention=%s)",
                    self._load_seconds, self._attn_label)
        self._pipe = pipe
        return pipe

    # ------------------------------------------------------------------
    # FlashRT surface
    # ------------------------------------------------------------------
    def set_prompt(self, prompt: str, **_: Any) -> None:
        self.prompt = prompt
        self._load_pipe()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def release_resident(self) -> int:
        """Release the resident transformer and its captured graphs.

        Idempotent, and returns the device bytes freed. Only capture mode
        holds a residency lease; every other mode disposes the transformer
        after each stage, so there is nothing here to release and this
        returns 0. The pipeline stays usable either way: the next call
        builds a transformer again.
        """
        pipe = self._pipe
        if pipe is None:
            return 0
        builder = getattr(pipe.stage, "_transformer_builder", None)
        release = getattr(builder, "release", None)
        return release() if callable(release) else 0

    def close(self) -> int:
        """Release everything this frontend holds. Idempotent.

        The resident transformer and its graphs, the cached prompt
        embeddings, and the pipeline itself. A later ``set_prompt`` or
        ``infer`` reloads from the checkpoint, so this is a release, not a
        teardown of the object.
        """
        freed = self.release_resident()
        pipe = self._pipe
        if pipe is not None:
            cache = getattr(pipe, "prompt_encoder", None)
            clear = getattr(cache, "clear", None)
            if callable(clear):
                clear()
        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        return freed

    @torch.inference_mode()
    def infer(
        self,
        prompt: Optional[str] = None,
        seed: int = 42,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[float] = None,
        output_path: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        prompt = prompt or self.prompt
        if not prompt:
            raise ValueError("No prompt: call set_prompt() or pass prompt=")
        pipe = self._load_pipe()

        from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number

        height = height or self.DEFAULT_HEIGHT
        width = width or self.DEFAULT_WIDTH
        num_frames = num_frames or self.DEFAULT_FRAMES
        frame_rate = frame_rate or self.DEFAULT_FPS

        tiling_config = AUTO_TILING
        if self.compile_mode == "capture":
            # AUTO tiling sizes itself from free memory before the resident
            # transformer (and its capture pools) exist; resolve it against
            # the memory decode will actually see.
            from ltx_pipelines.utils.helpers import tiling_config_for_vae
            free, total = torch.cuda.mem_get_info()
            reserved_slack = (torch.cuda.memory_reserved()
                              - torch.cuda.memory_allocated())
            builder = pipe.stage._transformer_builder
            if getattr(builder, "is_resident", False):
                # The transformer and its pools are already paid for, so
                # what the allocator reports free really is decode's.
                budget = max(5 << 30, free + reserved_slack - (2 << 30))
            else:
                # It is not built yet, and its reserve (weights, capture
                # pools, denoise peak) is measured against the *device*, not
                # against what happens to be free right now. Free is the
                # wrong base here because part of that same reserve --- the
                # VAEs, the upsampler, cached embeddings --- may already be
                # allocated, and subtracting the reserve from free would
                # then count it twice. It reads the same on a first run and
                # collapses the budget on a rebuild after a release, which
                # is where the difference showed up.
                budget = max(5 << 30, total - (23 << 30))
            tiling_config = tiling_config_for_vae(
                self._resolve_paths()["video_vae"],
                height=height, width=width, num_frames=num_frames,
                device=self.device, free_bytes=budget,
            )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video, audio, frames, tiling = pipe(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            images=[], tiling_config=tiling_config,
        )
        torch.cuda.synchronize()
        denoise_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if output_path:
            from ltx_pipelines.utils.media_io import encode_video
            encode_video(
                video=video, fps=frame_rate, audio=audio,
                output_path=output_path,
                video_chunks_number=get_video_chunks_number(frames, tiling))
        else:
            for _chunk in video:
                pass
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t1

        self._last_stats = {
            "attention": self._attn_label,
            "quantization": self.quantization,
            "resolution": f"{width}x{height}x{num_frames}",
            "denoise_and_prep_s": round(denoise_s, 3),
            "vae_decode_encode_s": round(decode_s, 3),
            "total_s": round(denoise_s + decode_s, 3),
            "peak_mem_gb": round(
                torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "output_path": output_path,
        }
        return dict(self._last_stats)

    def get_latency_stats(self) -> dict[str, Any]:
        return dict(self._last_stats)
