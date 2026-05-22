"""FlashRT -- Wan2.2 TI2V-5B official torch frontend for RTX SM120.

This frontend exposes the official Wan text/image-to-video pipeline through
FlashRT's stable ``set_prompt`` / ``infer`` wrapper. ComfyUI integrations
can use this API from an external custom-node package.

Scope:
    * Official Wan pipeline baseline for T2V/I2V.
    * RTX SM120 registration only.
    * No CMake or pybind changes.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from copy import deepcopy
from typing import Any, Optional

import torch


class Wan22TorchFrontendRtx:
    """Wan2.2 TI2V-5B official pipeline frontend for RTX."""

    DEFAULT_WIDTH = 832
    DEFAULT_HEIGHT = 480
    DEFAULT_FRAMES = 81
    DEFAULT_STEPS = 20
    DEFAULT_SHIFT = 5.0
    DEFAULT_GUIDE_SCALE = 5.0
    DEFAULT_SOLVER = "unipc"

    def __init__(
        self,
        checkpoint_dir: str,
        num_views: int = 1,
        autotune: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        t5_cpu: bool = True,
        init_on_cpu: bool = True,
        convert_model_dtype: bool = True,
        **_: Any,
    ) -> None:
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Wan2.2 checkpoint not found: {self.checkpoint_dir}")
        self.num_views = num_views
        self.autotune = autotune
        self.dtype = dtype
        self.t5_cpu = bool(t5_cpu)
        self.init_on_cpu = bool(init_on_cpu)
        self.convert_model_dtype = bool(convert_model_dtype)
        self.device = torch.device("cuda")
        self.prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self._pipe = None
        self._load_seconds: Optional[float] = None

    @staticmethod
    def _candidate_wan_roots() -> list[pathlib.Path]:
        roots: list[pathlib.Path] = []
        for key in ("FLASH_RT_WAN22_ROOT", "WAN22_ROOT", "MOTUS_ROOT",
                    "FLASH_RT_MOTUS_ROOT"):
            value = os.environ.get(key)
            if value:
                p = pathlib.Path(value).expanduser()
                roots.extend([p, p / "bak"])
        return roots

    @classmethod
    def _ensure_wan_importable(cls) -> None:
        try:
            import wan.textimage2video  # noqa: F401
            return
        except ModuleNotFoundError as exc:
            if exc.name != "wan":
                raise
            pass

        for root in cls._candidate_wan_roots():
            if (root / "wan").is_dir():
                root_s = str(root)
                if root_s not in sys.path:
                    sys.path.insert(0, root_s)
                try:
                    import wan.textimage2video  # noqa: F401
                    return
                except ModuleNotFoundError as exc:
                    if exc.name != "wan":
                        raise
                    continue

        raise ModuleNotFoundError(
            "Cannot import official Wan modules. Install the Wan2.2 source "
            "package, add it to PYTHONPATH, or set FLASH_RT_WAN22_ROOT to a "
            "directory containing the 'wan' package. A Motus checkout also "
            "works via FLASH_RT_MOTUS_ROOT/MOTUS_ROOT because it vendors "
            "Wan under bak/wan.")

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        self._ensure_wan_importable()
        from wan.configs.wan_ti2v_5B import ti2v_5B
        from wan.textimage2video import WanTI2V

        cfg = deepcopy(ti2v_5B)
        cfg.param_dtype = self.dtype
        t0 = time.perf_counter()
        self._pipe = WanTI2V(
            config=cfg,
            checkpoint_dir=str(self.checkpoint_dir),
            device_id=torch.cuda.current_device(),
            t5_cpu=self.t5_cpu,
            init_on_cpu=self.init_on_cpu,
            convert_model_dtype=self.convert_model_dtype,
        )
        self._load_seconds = time.perf_counter() - t0
        return self._pipe

    def set_prompt(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    def infer(
        self,
        *,
        mode: str = "t2v",
        image=None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        frames: int = DEFAULT_FRAMES,
        steps: int = DEFAULT_STEPS,
        shift: float = DEFAULT_SHIFT,
        guide_scale: float = DEFAULT_GUIDE_SCALE,
        seed: int = 1234,
        sample_solver: str = DEFAULT_SOLVER,
        offload_model: bool = True,
        save_path: Optional[str] = None,
        return_metadata: bool = False,
    ):
        """Generate video frames with the official Wan pipeline.

        Returns a ``torch.Tensor`` in ``[C, F, H, W]`` format by default. If
        ``return_metadata=True``, returns a dict with the tensor plus timing and
        generation settings.
        """
        if self.prompt is None:
            raise ValueError("set_prompt(prompt=...) must be called first")
        if mode not in ("t2v", "i2v"):
            raise ValueError("mode must be 't2v' or 'i2v'")
        if mode == "i2v" and image is None:
            raise ValueError("mode='i2v' requires image=")
        if frames < 1 or (frames - 1) % 4 != 0:
            raise ValueError("frames must be 4n+1 for Wan2.2")
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError("width and height must be multiples of 32")
        if steps <= 0:
            raise ValueError("steps must be positive")

        pipe = self._load_pipe()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            video = pipe.generate(
                self.prompt,
                img=image if mode == "i2v" else None,
                size=(int(width), int(height)),
                max_area=int(width) * int(height),
                frame_num=int(frames),
                shift=float(shift),
                sample_solver=sample_solver,
                sampling_steps=int(steps),
                guide_scale=float(guide_scale),
                n_prompt=self.negative_prompt or "",
                seed=int(seed),
                offload_model=bool(offload_model),
            )
        infer_seconds = time.perf_counter() - t0

        if save_path is not None:
            self._save_video(video, save_path)

        if not return_metadata:
            return video

        return {
            "video": video,
            "metadata": {
                "load_seconds": self._load_seconds,
                "infer_seconds": infer_seconds,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 1024 ** 3),
                "mode": mode,
                "width": int(width),
                "height": int(height),
                "frames": int(frames),
                "steps": int(steps),
                "shift": float(shift),
                "guide_scale": float(guide_scale),
                "sample_solver": sample_solver,
                "seed": int(seed),
            },
        }

    @classmethod
    def _save_video(cls, video: torch.Tensor, path: str) -> None:
        cls._ensure_wan_importable()
        from wan.utils.utils import save_video

        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=str(out),
            fps=24,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    @staticmethod
    def write_summary(result: dict[str, Any], path: str) -> None:
        payload = dict(result.get("metadata", result))
        pathlib.Path(path).write_text(json.dumps(payload, indent=2))
