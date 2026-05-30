"""Async batch preparation for the RECAP training driver.

Why this exists: a step-level ``torch.profiler`` trace
(B=4, lora_rank=16, RTX 5090 SM120) showed that ~47 % of the per-step
wall time was main-thread CPU work — parquet reads via
``LeRobotLiberoDataset.get_frame``, PIL JPEG decode of 8 images per
step, and ``np.stack`` of action chunks. The GPU sat idle for that
entire window because the driver loop was strictly synchronous.

This module wraps that prep into a map-style ``Dataset`` whose
``__getitem__(step)`` returns one fully-prepared batch — frames
read, images decoded into ``uint8 (B, 224, 224, 3)`` arrays, action
chunks padded to ``(B, action_horizon, action_dim)``. A
``torch.utils.data.DataLoader`` with ``num_workers > 0`` pushes the
prep into worker processes; the driver's main thread keeps only
``tokenize`` + ``decoded_to_observation`` H2D + the model step.

Sampling determinism: the chunk-start indices for every step are
*precomputed* in the main process from a seeded ``np.random.Generator``
before the loader starts. The dataset is index-only at the worker
boundary, so loss-curve byte-equality vs the synchronous driver is
preserved across any ``num_workers`` value (workers can fetch in any
order; the per-step batch identity is fixed by the precomputed list).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.rl.lerobot_libero import LeRobotLiberoDataset
from training.rl.observation import decode_frame_images, pad_states


@dataclass
class PreparedBatch:
    """One step's worth of prepared, on-CPU training inputs.

    All numpy / torch tensors are host-side; the main thread does the
    H2D and dtype conversion via :func:`decoded_to_observation`.
    """

    step: int                                   # 0-based step index
    starts: np.ndarray                          # int64 (B,) — chunk-start global frame indices
    tasks: list[str]                            # length B
    decoded_images: dict[str, np.ndarray | None]  # pi05 cam key → uint8 (B, 224, 224, 3) or None
    states_padded: np.ndarray                   # float32 (B, 32)
    action_chunks: np.ndarray                   # float32 (B, action_horizon, action_dim)


class _Pi05StepDataset(Dataset):
    """Map-style dataset: ``__getitem__(step)`` → one full :class:`PreparedBatch`.

    The set of chunk-start indices for each step is fixed at
    construction so worker processes only do *deterministic* work
    given a step index.
    """

    def __init__(
        self,
        dataset: LeRobotLiberoDataset,
        per_step_starts: np.ndarray,
        *,
        action_horizon: int,
        action_dim_target: int,
        camera_map: dict[str, str | None] | None = None,
    ):
        if per_step_starts.ndim != 2:
            raise ValueError(
                f"per_step_starts must be (num_steps, batch_size), got shape "
                f"{per_step_starts.shape}"
            )
        self._dataset = dataset
        self._starts = per_step_starts.astype(np.int64, copy=False)
        self._horizon = int(action_horizon)
        self._adim = int(action_dim_target)
        self._camera_map = camera_map

    def __len__(self) -> int:
        return self._starts.shape[0]

    def __getitem__(self, step: int) -> PreparedBatch:
        starts = self._starts[step]
        frames = [self._dataset.get_frame(int(s)) for s in starts]

        action_chunks = np.stack(
            [self._dataset.get_action_chunk(int(s), self._horizon) for s in starts],
            axis=0,
        )
        if action_chunks.shape[-1] < self._adim:
            pad = np.zeros(
                (
                    action_chunks.shape[0],
                    self._horizon,
                    self._adim - action_chunks.shape[-1],
                ),
                dtype=np.float32,
            )
            action_chunks = np.concatenate([action_chunks, pad], axis=-1)
        elif action_chunks.shape[-1] > self._adim:
            raise ValueError(
                f"action_dim {action_chunks.shape[-1]} > target {self._adim}"
            )

        decoded = decode_frame_images(frames, camera_map=self._camera_map)
        states_np = np.stack([f.state for f in frames], axis=0).astype(np.float32)
        states_pad = pad_states(states_np)
        tasks = [f.task_name for f in frames]

        return PreparedBatch(
            step=int(step),
            starts=starts,
            tasks=tasks,
            decoded_images=decoded,
            states_padded=states_pad,
            action_chunks=action_chunks.astype(np.float32, copy=False),
        )


def _identity_collate(batch):
    """``DataLoader`` collate that returns the single :class:`PreparedBatch`."""
    if len(batch) != 1:
        raise RuntimeError(
            f"async loader expects batch_size=1 over PreparedBatch items, got {len(batch)}"
        )
    return batch[0]


def precompute_per_step_starts(
    *,
    rng: np.random.Generator,
    num_chunk_starts: int,
    num_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Generate the full ``(num_steps, batch_size)`` chunk-start matrix.

    Mirrors the synchronous driver's per-step
    ``rng.integers(0, len(chunk_starts), size=batch_size)`` calls so
    the byte-identical sequence comes out for the same seed.
    """
    return rng.integers(
        0, num_chunk_starts, size=(num_steps, batch_size), dtype=np.int64
    )


def make_step_dataloader(
    dataset: LeRobotLiberoDataset,
    *,
    per_step_starts: np.ndarray,
    action_horizon: int,
    action_dim_target: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    camera_map: dict[str, str | None] | None = None,
) -> DataLoader:
    """Wrap :class:`_Pi05StepDataset` in a ``DataLoader``.

    Args:
        per_step_starts: Output of :func:`precompute_per_step_starts`.
        num_workers: Worker process count. ``0`` keeps everything on
            the main thread (the synchronous reference path).
        prefetch_factor: Number of batches each worker prefetches.
            Ignored when ``num_workers == 0``.
        camera_map: Mapping from pi0.5 camera keys to source keys in
            ``frame.image_bytes``. ``None`` falls back to the LIBERO
            default (see decode_frame_images() in training.rl.observation).
            Pass a custom map for non-LIBERO datasets.
    """
    ds = _Pi05StepDataset(
        dataset,
        per_step_starts,
        action_horizon=action_horizon,
        action_dim_target=action_dim_target,
        camera_map=camera_map,
    )
    kwargs: dict = dict(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        pin_memory=False,  # CPU dict outputs; H2D happens after tokenize anyway
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)
