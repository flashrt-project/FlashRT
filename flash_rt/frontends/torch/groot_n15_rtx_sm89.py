"""FlashRT -- RTX SM89 GR00T N1.5 torch frontend (CUDA Graph, bf16).

GR00T N1.5 (``GR00T_N1_5``: Eagle2.5 = SigLIP2 + 12-layer Qwen3, 4-step flow
matching DiT head) is served by the upstream ``gr00t`` ``Gr00tPolicy`` so
Eagle pre-processing and normalisation stay identical to the reference.
After the checkpoint is loaded, the model call is replaced by
:class:`~flash_rt.frontends.torch._groot_n15_graph.Gr00tN15GraphRunner`, which
replays one captured CUDA Graph per request shape (SDPA attention, DiT
cross-attention K/V cache, eager fallback).

This is a bf16 torch-level tier: it does not use FlashRT's FP8 kernels, so no
calibration is needed and the compiled ``flash_rt_kernels`` extension is not
required. Measured on an RTX 4090 with the RoboCasa-365 N1.5 checkpoint:
``get_action`` ~70-80 ms eager -> ~46-50 ms, max action difference vs eager
0.005 (see ``docs/groot_n15_rtx_sm89.md``).

Requirements: the upstream ``gr00t`` package (Isaac-GR00T, ``groot_root`` or
``$GROOT_N15_ROOT``) and an importable ``flash_attn`` (Eagle's remote code
imports it at module level).

Usage::

    import flash_rt
    model = flash_rt.load_model(
        "/path/to/gr00t-n1.5-checkpoint", config="groot_n15",
        hardware="rtx_sm89", num_views=3)
    actions = model.predict(
        images=[main, extra, wrist],           # data-config video-key order
        prompt="pick up the cup",
        state={"state.base_position": ...})    # dict keyed by state name
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import time
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

LANGUAGE_KEY = "annotation.human.task_description"
_ENV_ROOT = "GROOT_N15_ROOT"
_ENV_DATA_CONFIG = "GROOT_N15_DATA_CONFIG"


class GrootN15TorchFrontendRtxSm89:
    """GR00T N1.5 on RTX 4090 (SM89), CUDA-Graph bf16 tier.

    Mirrors the ``set_prompt`` + ``infer`` frontend API used by the other
    GROOT frontends so ``flash_rt.load_model(...).predict(...)`` works.

    Args:
        checkpoint_dir: GR00T N1.5 checkpoint directory.
        num_views: Number of camera views; checked against the data config.
        embodiment_tag: Embodiment tag of the checkpoint.
        groot_root: Isaac-GR00T source root (default ``$GROOT_N15_ROOT``).
        data_config: Name in ``gr00t.experiment.data_config.DATA_CONFIG_MAP``
            (default ``$GROOT_N15_DATA_CONFIG``; required).
        denoising_steps: Flow-matching steps (checkpoint default 4).
        max_graphs: LRU size of the shape-specialised graph cache.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        embodiment_tag: str = "new_embodiment",
        *,
        groot_root: Union[str, pathlib.Path, None] = None,
        data_config: str | None = None,
        denoising_steps: int = 4,
        max_graphs: int = 16,
    ):
        root = groot_root or os.environ.get(_ENV_ROOT)
        data_config = data_config or os.environ.get(_ENV_DATA_CONFIG)
        if not root:
            raise ValueError(
                f"GR00T N1.5 needs the Isaac-GR00T source: pass groot_root or "
                f"set ${_ENV_ROOT}")
        if not data_config:
            raise ValueError(
                f"GR00T N1.5 needs a data config name: pass data_config or "
                f"set ${_ENV_DATA_CONFIG}")
        root = pathlib.Path(root).expanduser().resolve()
        checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
        if not root.is_dir() or not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"GR00T source {root} or checkpoint {checkpoint_dir} not found")
        sys.path.insert(0, str(root))
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        if data_config not in DATA_CONFIG_MAP:
            raise ValueError(
                f"unknown data_config {data_config!r}; "
                f"available: {sorted(DATA_CONFIG_MAP)}")
        cfg = DATA_CONFIG_MAP[data_config]
        modality = cfg.modality_config()
        self._video_keys = tuple(modality["video"].modality_keys)
        self._state_keys = tuple(modality["state"].modality_keys)
        if int(num_views) != len(self._video_keys):
            raise ValueError(
                f"num_views={num_views} but data_config {data_config!r} "
                f"expects {len(self._video_keys)} views {self._video_keys}")
        self._policy = Gr00tPolicy(
            model_path=str(checkpoint_dir),
            modality_config=modality,
            modality_transform=cfg.transform(),
            embodiment_tag=embodiment_tag,
            denoising_steps=int(denoising_steps),
        )
        from flash_rt.frontends.torch._groot_n15_graph import (
            install_graph_runner,
        )

        self.graph_runner = install_graph_runner(
            self._policy, max_graphs=int(max_graphs))
        self._prompt: str | None = None
        self.latency_records: list[float] = []
        logger.info(
            "GrootN15TorchFrontendRtxSm89 initialised (views=%s, state=%s)",
            self._video_keys, self._state_keys)

    def set_prompt(self, prompt: str) -> None:
        """Set the language instruction used by subsequent ``infer`` calls."""
        self._prompt = str(prompt)

    def infer(self, obs: dict) -> dict:
        """Run one inference.

        Args:
            obs: Either GR00T-native keys (``video.*`` uint8 ``[H, W, 3]``,
                ``state.*`` float vectors), or ``images`` (list, in the data
                config's video-key order) plus ``state`` (dict keyed by
                ``state.*`` or bare state names).

        Returns:
            ``{"actions": {action_key: np.ndarray[horizon, dim]}}`` -- the
            GR00T action dict, since N1.5 actions are per-key.
        """
        if self._prompt is None:
            raise RuntimeError("set_prompt() must be called before infer()")
        t0 = time.perf_counter()
        actions = self._policy.get_action(self._build_observation(obs))
        self.latency_records.append((time.perf_counter() - t0) * 1000)
        return {"actions": actions}

    def _build_observation(self, obs: dict) -> dict[str, Any]:
        out: dict[str, Any] = {}
        images = obs.get("images")
        for i, key in enumerate(self._video_keys):
            if key in obs:
                frame = obs[key]
            elif images is not None and i < len(images):
                frame = images[i]
            else:
                raise ValueError(f"missing camera frame for {key!r}")
            out[key] = np.asarray(frame, dtype=np.uint8)[None]
        state = obs.get("state")
        for key in self._state_keys:
            bare = key.removeprefix("state.")
            if key in obs:
                value = obs[key]
            elif isinstance(state, dict) and key in state:
                value = state[key]
            elif isinstance(state, dict) and bare in state:
                value = state[bare]
            else:
                raise ValueError(
                    f"missing state {key!r}: pass obs['state'] as a dict "
                    f"keyed by {self._state_keys}")
            out[key] = np.asarray(value, dtype=np.float64).reshape(1, -1)
        out[LANGUAGE_KEY] = [self._prompt]
        return out
