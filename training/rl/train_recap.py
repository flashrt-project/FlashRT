"""Generic RECAP / ACP-conditioned policy training driver.

Dataset-agnostic core of what used to be ``train_libero_recap.py``.
Any dataset that satisfies :class:`training.rl.dataset_protocol.RecapPolicyDataset`
plugs in — LIBERO is the first concrete implementation, but
nothing in this driver hardcodes LIBERO. The training math (AdamW
+ warmup-cosine + grad-clip 1.0 + flow-matching loss + ACP
prompt injection at 30 % dropout) is paper-aligned (π\\*0.6
arXiv:2511.14759, Section V-B + Appendix E + Appendix F) and
matches the openpi JAX baseline 1-for-1.

The data path is parametrised via two callables the caller
supplies:

* ``observation_builder(decoded_images_dict, states_padded,
                       tokenized_prompt, mask, *, device)`` →
  pi0.5 :class:`Observation` for one mini-batch. The async loader
  hands off the decoded images already normalised to pi0.5's
  ``(B, H, W, 3)`` channel-last layout; this callable just stacks
  them into the openpi Observation dataclass. The LIBERO entry
  point uses :func:`training.rl.observation.decoded_to_observation`.

The async loader expects the dataset to expose ``get_frame``,
``get_action_chunk``, and the camera-name → bytes-decoder
contract that's currently fixed inside
``training.rl.async_loader.make_step_dataloader``. New datasets
that need a different camera mapping should set
``RecapTrainConfig.camera_map``; the async loader applies that map
before handing decoded pi0.5 camera slots to ``observation_builder``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.rl.acp_hook import ACPPromptHook
from training.rl.async_loader import (
    make_step_dataloader,
    precompute_per_step_starts,
)
from training.rl.checkpoint import save_lora_state
from training.rl.dataset_protocol import RecapPolicyDataset
from training.rl.train_policy import policy_train_step
from training.trainers.pi05_torch_trainer import Pi05Trainer

logger = logging.getLogger(__name__)


# ── Public dataclasses ─────────────────────────────────────────────


@dataclass
class RecapTrainConfig:
    """Hyper-parameters for :func:`train_recap_policy`.

    Defaults match the openpi JAX baseline
    (``RL/scripts/train_jax_lora_recap.py``) so a head-to-head run
    produces directly comparable loss curves.
    """

    num_steps: int = 1_000
    batch_size: int = 4
    lr: float = 2.5e-5
    weight_decay: float = 1e-10
    grad_clip_norm: float = 1.0
    action_horizon: int = 10
    acp_dropout: float = 0.30
    log_every: int = 10
    save_every: int = 0   # 0 → save only at end
    seed: int = 42

    # LR schedule (warmup-cosine).
    end_value_factor: float = 0.10  # end LR = peak * 0.1
    # warmup_steps_max — final warmup_steps = min(this, num_steps // 30)
    warmup_steps_max: int = 100

    # ACP (RECAP) toggle. When False the driver runs as a vanilla
    # supervised LoRA fine-tune: no acp_indicator lookup, no
    # ACPPromptHook, no per-step prompt re-tokenisation. The
    # flow-matching loss + AdamW + LR schedule + LoRA adapters
    # are exactly the same — only the prompt-conditioning side is
    # bypassed.
    use_acp: bool = True

    # Async data-prep pipeline. When > 0, parquet read + JPEG
    # decode + np.stack run in DataLoader worker processes,
    # overlapped with the GPU step.
    dataloader_workers: int = 0
    dataloader_prefetch_factor: int = 2
    camera_map: dict[str, str | None] | None = None

    # ``torch.compile`` mode applied to ``trainer.model`` for the
    # training step. ``None`` keeps eager mode.
    compile_mode: str | None = None


@dataclass
class RecapTrainResult:
    """Output of :func:`train_recap_policy`."""

    loss_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    seconds_total: float = 0.0
    final_lora_dir: Path | None = None


# ── LR schedule (matches optax.warmup_cosine_decay_schedule) ───────


def make_warmup_cosine_lr_lambda(
    *,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    end_value_factor: float = 0.10,
) -> Callable[[int], float]:
    """Return a step→multiplier callable for ``LambdaLR``.

    Multiplier is normalised so that:
    * ``step=0``  → 0;
    * ``step=warmup_steps`` → 1.0 (LR = peak);
    * ``step=total_steps`` → ``end_value_factor`` (LR = peak * 0.1 by default).

    This matches the JAX baseline's
    ``optax.warmup_cosine_decay_schedule`` 1-for-1 in shape.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    warmup_steps = max(warmup_steps, 1)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_value_factor + (1.0 - end_value_factor) * cosine

    return schedule


# ── ACP indicator source ─────────────────────────────────────────


def _resolve_acp_indicators(
    dataset: RecapPolicyDataset,
    *,
    derive_if_missing: bool,
    derive_steps: int = 500,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Return per-frame ``acp_indicator`` from the dataset, deriving
    via one RECAP iteration if the dataset does not ship them.

    Derivation imports :mod:`training.rl.recap_iter` lazily because
    it pulls in numpy + dataset-specific helpers; callers running
    pre-annotated data (the common case) avoid the import.
    """
    if dataset.has_acp_column():
        logger.info("ACP indicators read from dataset's annotated column.")
        return dataset.ensure_acp_indicators()
    if not derive_if_missing:
        raise RuntimeError(
            "Dataset has no acp_indicator column and derive_if_missing=False"
        )
    logger.info(
        "ACP indicators derived on the fly via one RECAP iteration "
        "(derive_steps=%d).", derive_steps,
    )
    from training.rl.recap_iter import run_recap_iter
    iter_result = run_recap_iter(
        dataset, num_steps=derive_steps, seed=seed, device=device,
    )
    return iter_result.annotation.indicators


# ── Driver ─────────────────────────────────────────────────────────


def train_recap_policy(
    trainer: Pi05Trainer,
    dataset: RecapPolicyDataset,
    tokenizer,
    observation_builder: Callable[..., Any],
    *,
    config: RecapTrainConfig | None = None,
    output_dir: str | Path | None = None,
    derive_acp_if_missing: bool = True,
    progress_cb: Callable[[int, float, float], None] | None = None,
) -> RecapTrainResult:
    """Run a fixed-step RECAP / ACP-conditioned policy training pass.

    Args:
        trainer: Compiled :class:`Pi05Trainer` (FP8 + LoRA injected).
        dataset: Any object satisfying
            :class:`training.rl.dataset_protocol.RecapPolicyDataset`.
            LIBERO ships as the first concrete implementation
            (:class:`training.rl.lerobot_libero.LeRobotLiberoDataset`).
        tokenizer: PaliGemma SentencePiece tokenizer (or any callable
            with the same ``(list[str]) → (tokens, mask)`` shape).
        observation_builder: Callable that turns one mini-batch's
            decoded images + padded states + tokenized prompt into
            a pi0.5 :class:`Observation`. The LIBERO entry point
            passes :func:`training.rl.observation.decoded_to_observation`.
        config: Hyper-parameters; defaults match openpi JAX baseline.
        output_dir: When set, the final LoRA ckpt is saved there;
            set ``config.save_every`` for periodic intermediate
            saves.
        derive_acp_if_missing: If True (default) and the dataset is
            missing the ACP indicator column, one quick RECAP
            iteration runs before training to populate them.
        progress_cb: Optional ``(step, loss, lr) → None`` callback.

    Returns:
        :class:`RecapTrainResult` with loss + lr histories, peak
        memory, total wall-clock seconds, and the saved LoRA dir.
    """
    if not trainer.is_compiled:
        raise RuntimeError("Pi05Trainer must be compiled before training")
    cfg = config or RecapTrainConfig()

    rng = np.random.default_rng(cfg.seed)
    device = trainer.device

    chunk_starts = dataset.build_chunk_starts(action_horizon=cfg.action_horizon)
    if len(chunk_starts) == 0:
        raise RuntimeError(
            f"No valid chunk starts for action_horizon={cfg.action_horizon}; "
            f"all episodes are shorter."
        )

    if cfg.use_acp:
        indicators_full = _resolve_acp_indicators(
            dataset,
            derive_if_missing=derive_acp_if_missing,
            seed=cfg.seed,
            device=device,
        )
    else:
        indicators_full = None

    # Optimiser + schedule.
    optimizer = torch.optim.AdamW(
        trainer.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    warmup_steps = min(cfg.warmup_steps_max, max(cfg.num_steps // 30, 1))
    lr_lambda = make_warmup_cosine_lr_lambda(
        peak_lr=cfg.lr,
        warmup_steps=warmup_steps,
        total_steps=cfg.num_steps,
        end_value_factor=cfg.end_value_factor,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    hook = (
        ACPPromptHook(dropout=cfg.acp_dropout, seed=cfg.seed)
        if cfg.use_acp
        else None
    )

    output_dir_p = Path(output_dir) if output_dir is not None else None
    if output_dir_p is not None:
        output_dir_p.mkdir(parents=True, exist_ok=True)

    eager_model = trainer.model
    if cfg.compile_mode is not None:
        logger.info("torch.compile(mode=%s) applied to trainer.model", cfg.compile_mode)
        trainer.model = torch.compile(trainer.model, mode=cfg.compile_mode)

    per_step_idx = precompute_per_step_starts(
        rng=rng,
        num_chunk_starts=len(chunk_starts),
        num_steps=cfg.num_steps,
        batch_size=cfg.batch_size,
    )
    per_step_starts = chunk_starts[per_step_idx]   # (num_steps, batch_size)
    loader = make_step_dataloader(
        dataset,
        per_step_starts=per_step_starts,
        action_horizon=cfg.action_horizon,
        action_dim_target=trainer.model.config.action_dim,
        num_workers=cfg.dataloader_workers,
        prefetch_factor=cfg.dataloader_prefetch_factor,
        camera_map=cfg.camera_map,
    )

    losses: list[float] = []
    lrs: list[float] = []
    trainer.reset_peak_memory()
    t_start = time.perf_counter()

    for step, batch in enumerate(loader):
        if step >= cfg.num_steps:
            break

        actions = torch.from_numpy(batch.action_chunks).to(
            device=device, dtype=torch.float32,
        )
        tokens, mask = tokenizer(batch.tasks)
        observation = observation_builder(
            batch.decoded_images,
            batch.states_padded,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=mask,
            device=device,
        )

        if cfg.use_acp:
            indicators = torch.from_numpy(
                indicators_full[batch.starts].astype(np.int64)
            ).to(device=device)
        else:
            indicators = None

        loss = policy_train_step(
            trainer,
            optimizer,
            observation,
            actions,
            indicators=indicators,
            tasks=batch.tasks if cfg.use_acp else None,
            hook=hook,
            tokenize_fn=tokenizer if cfg.use_acp else None,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        scheduler.step()

        losses.append(loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        lrs.append(cur_lr)

        if progress_cb is not None:
            progress_cb(step, loss, cur_lr)

        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.num_steps - 1):
            logger.info(
                "step %5d/%d | loss=%.4f | lr=%.2e",
                step,
                cfg.num_steps,
                loss,
                cur_lr,
            )

        if (
            output_dir_p is not None
            and cfg.save_every > 0
            and step > 0
            and step % cfg.save_every == 0
        ):
            save_lora_state(trainer, output_dir_p / f"step_{step:06d}")

    seconds_total = time.perf_counter() - t_start
    if cfg.compile_mode is not None:
        trainer.model = eager_model

    final_lora_dir: Path | None = None
    if output_dir_p is not None:
        final_lora_dir = save_lora_state(trainer, output_dir_p / "final")

    return RecapTrainResult(
        loss_history=losses,
        lr_history=lrs,
        peak_memory_bytes=trainer.peak_memory_bytes(),
        seconds_total=seconds_total,
        final_lora_dir=final_lora_dir,
    )
