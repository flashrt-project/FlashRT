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
import types
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
        self._teacache_installed = False

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
        teacache: bool = False,
        teacache_threshold: float = 0.0,
        teacache_start_step: int = 1,
        teacache_end_step: int = -1,
        teacache_cache_device: str = "cuda",
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
        if teacache and teacache_threshold <= 0:
            raise ValueError(
                "teacache_threshold must be positive when teacache=True")

        pipe = self._load_pipe()
        teacache_meta = self._configure_teacache(
            pipe.model,
            enabled=bool(teacache),
            threshold=float(teacache_threshold),
            start_step=int(teacache_start_step),
            end_step=int(teacache_end_step),
            cache_device=teacache_cache_device,
        )
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
        teacache_meta = self._teacache_metadata(pipe.model, teacache_meta)
        self._release_teacache_cache(pipe.model)

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
                "teacache": teacache_meta,
            },
        }

    def _configure_teacache(
        self,
        model,
        *,
        enabled: bool,
        threshold: float,
        start_step: int,
        end_step: int,
        cache_device: str,
    ) -> dict[str, Any]:
        if not enabled:
            state = getattr(model, "_flashrt_teacache", None)
            if state is not None:
                state["enabled"] = False
            return {"enabled": False}

        if not self._teacache_installed:
            self._install_teacache(model)
            self._teacache_installed = True

        if cache_device in ("cuda", "gpu", "main_device"):
            resolved_device = self.device
            device_name = "cuda"
        elif cache_device in ("cpu", "offload_device"):
            resolved_device = torch.device("cpu")
            device_name = "cpu"
        else:
            raise ValueError(
                "teacache_cache_device must be 'cuda'/'main_device' or "
                "'cpu'/'offload_device'")

        model._flashrt_teacache = self._new_teacache_state(
            enabled=True,
            threshold=threshold,
            start_step=start_step,
            end_step=end_step,
            cache_device=resolved_device,
        )
        return {
            "enabled": True,
            "threshold": threshold,
            "start_step": start_step,
            "end_step": end_step,
            "cache_device": device_name,
        }

    @staticmethod
    def _new_teacache_state(
        *,
        enabled: bool,
        threshold: float,
        start_step: int,
        end_step: int,
        cache_device: torch.device,
    ) -> dict[str, Any]:
        def slot() -> dict[str, Any]:
            return {
                "previous_modulated_input": None,
                "previous_residual": None,
                "accumulated": torch.tensor(0.0, device=cache_device),
                "calculated": 0,
                "skipped": [],
            }

        return {
            "enabled": enabled,
            "threshold": threshold,
            "start_step": start_step,
            "end_step": end_step,
            "cache_device": cache_device,
            "last_t": None,
            "step": -1,
            "call_index": 0,
            "states": {0: slot(), 1: slot()},
        }

    @staticmethod
    def _relative_l1(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        denom = prev.abs().mean().clamp_min(1e-6)
        return (cur - prev).abs().mean() / denom

    @classmethod
    def _install_teacache(cls, model) -> None:
        cls._ensure_wan_importable()
        from wan.modules.model import sinusoidal_embedding_1d

        original_forward = model.forward

        def teacache_forward(self, x, t, context, seq_len, y=None):
            tc = getattr(self, "_flashrt_teacache", None)
            if tc is None or not tc.get("enabled", False):
                return original_forward(x, t, context, seq_len, y=y)

            if self.model_type == "i2v":
                assert y is not None

            device = self.patch_embedding.weight.device
            if self.freqs.device != device:
                self.freqs = self.freqs.to(device)

            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([
                torch.tensor(u.shape[2:], dtype=torch.long) for u in x
            ])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat(
                    [u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1)
                for u in x
            ])

            if t.dim() == 1:
                t = t.expand(t.size(0), seq_len)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                bt = t.size(0)
                t_flat = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t_flat)
                    .unflatten(0, (bt, seq_len))
                    .float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat([
                        u,
                        u.new_zeros(self.text_len - u.size(0), u.size(1))
                    ])
                    for u in context
                ]))

            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
            )

            t_key = float(t_flat[0].detach().cpu())
            if tc["last_t"] != t_key:
                tc["last_t"] = t_key
                tc["step"] += 1
                tc["call_index"] = 0
            pred_id = tc["call_index"] % 2
            tc["call_index"] += 1
            state = tc["states"][pred_id]

            active = tc["start_step"] <= tc["step"]
            if tc["end_step"] >= 0:
                active = active and tc["step"] <= tc["end_step"]

            cache_device = tc["cache_device"]
            modulated = e.detach().to(cache_device)
            should_calc = True
            if active and state["previous_modulated_input"] is not None:
                state["accumulated"] = (
                    state["accumulated"] + cls._relative_l1(
                        modulated, state["previous_modulated_input"]))
                if (state["accumulated"] < tc["threshold"]
                        and state["previous_residual"] is not None):
                    should_calc = False
                else:
                    state["accumulated"] = torch.tensor(
                        0.0, device=cache_device)
            state["previous_modulated_input"] = modulated.clone()

            if should_calc:
                x_before = x.detach().to(cache_device).clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                state["previous_residual"] = (
                    x.detach().to(cache_device) - x_before)
                state["calculated"] += 1
            else:
                x = x + state["previous_residual"].to(x.device)
                state["skipped"].append(tc["step"])

            x = self.head(x, e)
            x = self.unpatchify(x, grid_sizes)
            return [u.float() for u in x]

        model._flashrt_teacache_original_forward = original_forward
        model.forward = types.MethodType(teacache_forward, model)

    @staticmethod
    def _teacache_metadata(model, base: dict[str, Any]) -> dict[str, Any]:
        state = getattr(model, "_flashrt_teacache", None)
        if state is None or not base.get("enabled", False):
            return {"enabled": False}
        meta = dict(base)
        meta.update({
            "cond_calculated": state["states"][0]["calculated"],
            "cond_skipped": list(state["states"][0]["skipped"]),
            "uncond_calculated": state["states"][1]["calculated"],
            "uncond_skipped": list(state["states"][1]["skipped"]),
        })
        return meta

    @staticmethod
    def _release_teacache_cache(model) -> None:
        state = getattr(model, "_flashrt_teacache", None)
        if state is None:
            return
        state["enabled"] = False
        for slot in state.get("states", {}).values():
            slot["previous_modulated_input"] = None
            slot["previous_residual"] = None

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
