"""Alpamayo 1.5 with FlashDrive's streaming rollout, FlashRT execution, on Thor.

FlashDrive (z-lab, https://github.com/z-lab/flashdrive) runs Alpamayo 1.5 as a
stream of windows: the first window prefills the whole prompt and returns no
trajectory; every later window encodes only each camera's newest frame, reuses
the cached keys/values of the older frames, speculates its reasoning with a
DFlash draft and samples the trajectory with adaptive action caching, on a W4A8
(ParoQuant) language model and a fine-tuned action expert. This frontend keeps
that algorithm, with every control explicit, and executes its primitives (all
of them by default, or those selected in ``backends``) with
:class:`flash_rt.models.alpamayo15_flashdrive.pipeline_thor.FlashDriveThor`.

FlashDrive's ``scripts/infer.py`` also sets
``torch.set_float32_matmul_precision("high")`` for the process; this frontend
leaves that global setting to the caller (the documented measurements use it).

Inputs are prepared as FlashDrive's ``scripts/infer.py`` does (official chat
message and processor; windows after the first are reduced to each camera's
newest frame with ``convert_to_streaming_window``).

Example::

    fe = Alpamayo15FlashDriveTorchFrontendThor()          # z-lab/Alpamayo-1.5-10B (+ -PARO, -DFlash)
    for window in clip_windows:                            # consecutive windows, 0.1 s apart
        out = fe.step(window["image_frames"], window["camera_indices"],
                      window["ego_history_xyz"], window["ego_history_rot"])
        if out is not None:                                # None for a stream's first window
            out["pred_xyz"], out["cot"]
    fe.reset()                                             # before the next, unrelated stream
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from flash_rt.models.alpamayo15_flashdrive.pipeline_thor import FlashDriveThor, euler_cache_schedule, load_flashdrive


@dataclass(frozen=True)
class FlashDriveRollout:
    """FlashDrive's rollout controls (defaults: FlashDrive ``scripts/infer.py``).

    ``max_new_tokens``, ``num_traj_samples`` and the sampling parameters are
    fixed for a stream by its first window (they size FlashDrive's cache and are
    bound into its graphs); change them only after :meth:`reset`.
    """

    max_new_tokens: int = 128
    temperature: float = 0.6
    top_p: float = 0.98
    inference_step: int = 8
    cache_steps: tuple[int, ...] = (3, 4, 5, 6)
    num_traj_samples: int = 1

    def diffusion_kwargs(self) -> dict[str, Any]:
        return {"inference_step": self.inference_step, "cache_steps": list(self.cache_steps),
                "int_method": "euler_with_cache"}


@dataclass(frozen=True)
class FlashDriveCheckpoints:
    """Checkpoint locations. FlashDrive derives the W4A8 and draft companions from a ``z-lab/`` repo id;
    here they are explicit, so local directories load the same optimized stack."""

    base: str = "z-lab/Alpamayo-1.5-10B"
    paro: str = "z-lab/Alpamayo-1.5-10B-PARO"
    dflash: str = "z-lab/Alpamayo-1.5-10B-DFlash"


class Alpamayo15FlashDriveTorchFrontendThor:
    """FlashDrive rollout of Alpamayo 1.5 on Thor (SM110) with FlashRT primitives.

    Args:
        checkpoints: base, W4A8 (``-PARO``) and DFlash draft locations.
        hardware: ``None`` or ``"thor"``.
        rollout: FlashDrive's algorithm controls.
        backends: per-primitive execution choice passed to ``FlashDriveThor``
            (default: every group on FlashRT with fused W4A8 activations and
            the tcgen05 prefill GEMM; groups set to ``"flashdrive"`` keep
            FlashDrive's execution).
        torch_compile: FlashDrive's compile mode for the primitives it executes
            (``None`` = eager).
    """

    def __init__(self, checkpoints: FlashDriveCheckpoints = FlashDriveCheckpoints(), *, hardware: str | None = None,
                 rollout: FlashDriveRollout = FlashDriveRollout(), backends: dict[str, Any] | None = None,
                 torch_compile: str | None = "max-autotune") -> None:
        if hardware not in (None, "thor"):
            raise ValueError(f"the Alpamayo 1.5 FlashDrive frontend is registered for Thor only; got hardware={hardware!r}")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
            raise RuntimeError("Alpamayo15FlashDriveTorchFrontendThor requires a Jetson AGX Thor GPU (SM110)")
        if rollout.num_traj_samples != 1:
            raise ValueError("the FlashRT FlashDrive backend serves one trajectory sample per window")
        euler_cache_schedule(rollout.inference_step, rollout.cache_steps)     # validate before loading
        try:
            from alpamayo1_5 import helper
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
            from flashdrive import resolve_model_class
            from flashdrive.streaming import convert_to_streaming_window
        except ImportError as exc:
            raise ImportError("needs NVIDIA's alpamayo1_5 and z-lab's flashdrive packages") from exc
        if resolve_model_class(checkpoints.base) is not Alpamayo1_5:
            raise ValueError(f"expected an Alpamayo 1.5 checkpoint at {checkpoints.base!r}")
        self._helper = helper
        self._to_streaming = convert_to_streaming_window
        self.rollout = rollout
        self.model = load_flashdrive(checkpoints.base, checkpoints.paro, checkpoints.dflash, torch_compile=torch_compile)
        self.engine = FlashDriveThor(self.model, **(backends or {}))
        self.processor = helper.get_processor(self.model.tokenizer)
        tok = self.processor.tokenizer
        self._vs = tok.convert_tokens_to_ids("<|vision_start|>")
        self._ve = tok.convert_tokens_to_ids("<|vision_end|>")

    @property
    def stream_started(self) -> bool:
        return self.model._past_key_values is not None

    def build_inputs(self, image_frames: torch.Tensor, camera_indices: torch.Tensor, ego_history_xyz: torch.Tensor,
                     ego_history_rot: torch.Tensor, *, first: bool) -> dict[str, Any]:
        """FlashDrive window inputs on CUDA: the full prompt for a stream's first window, else the streaming form.

        ``image_frames`` is ``(n_cameras, n_frames, 3, H, W)`` uint8 (oldest frame first),
        ``camera_indices`` ``(n_cameras,)``, ``ego_history_*`` ``(1, 1, T, ...)`` in the t0 ego frame.
        """
        messages = self._helper.create_message(image_frames.flatten(0, 1), camera_indices=camera_indices)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                    continue_final_message=True, return_dict=True, return_tensors="pt")
        window = {"tokenized_data": inputs, "ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        if not first:
            window = self._to_streaming(window, self._vs, self._ve)
        return self._helper.to_device(window, "cuda")

    def step(self, image_frames: torch.Tensor, camera_indices: torch.Tensor, ego_history_xyz: torch.Tensor,
             ego_history_rot: torch.Tensor) -> dict[str, Any] | None:
        """One window of the current stream; ``None`` for the stream's first (prefill-only) window."""
        data = self.build_inputs(image_frames, camera_indices, ego_history_xyz, ego_history_rot,
                                 first=not self.stream_started)
        r = self.rollout
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_streaming(
                data, max_new_tokens=r.max_new_tokens, temperature=r.temperature, top_p=r.top_p,
                num_traj_samples=r.num_traj_samples, diffusion_kwargs=r.diffusion_kwargs(), return_extra=True)
        if pred_xyz is None:
            return None
        return {"pred_xyz": pred_xyz, "pred_rot": pred_rot, "cot": [str(c) for c in extra["cot"][0, 0]],
                "meta_action": [str(c) for c in extra["meta_action"][0, 0]],
                "answer": [str(c) for c in extra["answer"][0, 0]]}

    def reset(self) -> None:
        """End the current stream; the next :meth:`step` starts a new one from a full prefill
        (no cache, shape cache, mask or compiled-step state carries over)."""
        self.engine.reset()
