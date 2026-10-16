"""Alpamayo 1.5 (10B) frontend for Jetson AGX Thor.

Inputs are prepared exactly as NVIDIA's ``alpamayo1_5`` package does (chat
message, Qwen3-VL processor, trajectory-history tokens); the request then runs
on :class:`flash_rt.models.alpamayo15.pipeline_thor.Alpamayo15Thor`.

The official package is required (it owns the checkpoint classes, tokenizer
extensions, trajectory tokenizer and action space). The model is loaded in
BF16 with SDPA attention, the upstream fallback that needs no flash-attn
build.

Example::

    fe = Alpamayo15TorchFrontendThor("nvidia/Alpamayo-1.5-10B")
    out = fe.infer(image_frames, camera_indices, ego_history_xyz, ego_history_rot, seed=42)
    out["pred_xyz"]   # (1, 1, 1, 64, 3) future ego positions, ego frame at t0
    out["cot"][0]     # chain-of-causation reasoning text
"""

from __future__ import annotations

from typing import Any

import torch

from flash_rt.models.alpamayo15.pipeline_thor import Alpamayo15Thor


class Alpamayo15TorchFrontendThor:
    """Alpamayo 1.5 on Thor (SM110): official preprocessing + FlashRT engine.

    Args:
        checkpoint: Hugging Face id or local directory of the Alpamayo 1.5 weights.
        hardware: ``None`` or ``"thor"``.
        max_prompt, max_new_tokens, top_p, temperature: see ``Alpamayo15Thor``;
            the defaults are the official inference settings.
        max_samples: largest ``num_traj_samples`` served by ``infer``.
        capture: capture the decode and diffusion CUDA graphs at construction
            (for ``num_traj_samples=1``; other counts are captured on first use).
        engine_kwargs: extra keyword arguments for ``Alpamayo15Thor``.
    """

    def __init__(self, checkpoint: str = "nvidia/Alpamayo-1.5-10B", *, hardware: str | None = None,
                 max_prompt: int = 4096, max_new_tokens: int = 256, top_p: float = 0.98,
                 temperature: float = 0.6, max_samples: int = 1, capture: bool = True,
                 engine_kwargs: dict[str, Any] | None = None) -> None:
        if hardware not in (None, "thor"):
            raise ValueError(f"Alpamayo 1.5 frontend is registered for Thor only; got hardware={hardware!r}")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
            raise RuntimeError("Alpamayo15TorchFrontendThor requires a Jetson AGX Thor GPU (SM110)")
        try:
            from alpamayo1_5 import helper
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        except ImportError as exc:
            raise ImportError("Alpamayo 1.5 needs NVIDIA's alpamayo1_5 package "
                              "(https://github.com/NVlabs/alpamayo1.5)") from exc
        self._helper = helper
        self.model = Alpamayo1_5.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        kwargs = {"max_prompt": max_prompt, "max_new_tokens": max_new_tokens, "max_samples": max_samples,
                  "top_p": top_p, "temperature": temperature}
        kwargs.update(engine_kwargs or {})
        self.engine = Alpamayo15Thor(self.model, **kwargs)
        if capture:
            self.engine.capture()

    def build_inputs(self, image_frames: torch.Tensor, camera_indices: torch.Tensor,
                     ego_history_xyz: torch.Tensor, ego_history_rot: torch.Tensor,
                     nav_text: str | None = None) -> dict[str, Any]:
        """Official model inputs on CUDA.

        ``image_frames`` is ``(n_cameras, n_frames, 3, H, W)`` uint8,
        ``camera_indices`` ``(n_cameras,)``, ``ego_history_xyz`` ``(1, 1, T, 3)``
        and ``ego_history_rot`` ``(1, 1, T, 3, 3)`` in the t0 ego frame.
        """
        messages = self._helper.create_message(
            frames=image_frames.flatten(0, 1), camera_indices=camera_indices,
            num_frames_per_camera=image_frames.shape[1], nav_text=nav_text)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt")
        return self._helper.to_device({"tokenized_data": inputs, "ego_history_xyz": ego_history_xyz,
                                       "ego_history_rot": ego_history_rot}, "cuda")

    def infer(self, image_frames: torch.Tensor, camera_indices: torch.Tensor,
              ego_history_xyz: torch.Tensor, ego_history_rot: torch.Tensor, *,
              seed: int | None = None, nav_text: str | None = None,
              num_traj_samples: int = 1) -> dict[str, Any]:
        """One request: reasoning text and ``num_traj_samples`` sampled future trajectories.

        ``seed`` re-seeds the CUDA generator exactly as the official example
        (``torch.cuda.manual_seed_all``) before sampling. ``pred_xyz`` is
        ``(1, 1, num_traj_samples, T, 3)``; ``cot`` is a list with one reasoning
        string per sample.
        """
        data = self.build_inputs(image_frames, camera_indices, ego_history_xyz, ego_history_rot, nav_text)
        if seed is not None:
            torch.cuda.manual_seed_all(seed)
        pred_xyz, pred_rot, extra = self.engine.sample_trajectories(data, num_traj_samples=num_traj_samples)
        return {"pred_xyz": pred_xyz, "pred_rot": pred_rot,
                "cot": [str(c) for c in extra["cot"][0, 0]],
                "meta_action": [str(c) for c in extra["meta_action"][0, 0]],
                "answer": [str(c) for c in extra["answer"][0, 0]],
                "generated_tokens": extra["generated_tokens"]}
