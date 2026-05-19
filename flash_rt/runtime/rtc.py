"""Lightweight asynchronous execution for action chunk policies.

This module stays outside model frontends. A frontend only needs to expose a
callable that maps the latest observation to an action chunk. The runner
handles background chunk generation and foreground action consumption.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np


class ActionChunkAdapter(Protocol):
    """Minimal adapter contract for a chunked action model."""

    def infer_actions(self, observation: Any) -> np.ndarray:
        """Return an action chunk shaped ``[horizon, action_dim]``."""


@dataclass(frozen=True)
class CallablePolicyAdapter:
    """Wrap a Python callable as an :class:`ActionChunkAdapter`.

    ``output_key`` covers frontends that return ``{"actions": array}``.
    ``tuple_index`` covers frontends that return tuples such as
    ``(frames, actions)``.
    """

    fn: Callable[[Any], Any]
    output_key: str | None = "actions"
    tuple_index: int | None = None

    def infer_actions(self, observation: Any) -> np.ndarray:
        out = self.fn(observation)
        if self.tuple_index is not None:
            out = out[self.tuple_index]
        elif self.output_key is not None and isinstance(out, Mapping):
            out = out[self.output_key]
        actions = np.asarray(out)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(
                f"expected action chunk [horizon, action_dim], got {actions.shape}")
        return actions


@dataclass(frozen=True)
class RTCConfig:
    """Configuration for asynchronous chunk execution."""

    target_hz: float = 20.0
    action_horizon: int | None = None
    start_next_at: int | None = None
    miss_policy: str = "hold_last"
    blend_steps: int = 0
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.target_hz <= 0:
            raise ValueError("target_hz must be positive")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.start_next_at is not None and self.start_next_at < 0:
            raise ValueError("start_next_at must be non-negative")
        if self.miss_policy not in {"hold_last", "block"}:
            raise ValueError("miss_policy must be 'hold_last' or 'block'")
        if self.blend_steps < 0:
            raise ValueError("blend_steps must be non-negative")
        if self.max_workers != 1:
            raise ValueError("RTC-lite supports exactly one model worker")

    @property
    def period_s(self) -> float:
        return 1.0 / self.target_hz


@dataclass
class ChunkResult:
    actions: np.ndarray
    latency_s: float
    observation_time_s: float
    ready_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RTCStats:
    chunks_started: int = 0
    chunks_completed: int = 0
    actions_served: int = 0
    deadline_misses: int = 0
    held_actions: int = 0
    swaps: int = 0
    last_latency_s: float = 0.0
    max_latency_s: float = 0.0


class AsyncChunkRunner:
    """Run an action chunk model asynchronously while actions are consumed.

    The runner does not sleep or own the controller loop. Call ``next_action``
    once per controller tick with the latest observation. The first call blocks
    to produce the initial chunk. Later calls trigger background inference when
    enough of the current chunk has been consumed.
    """

    def __init__(self, adapter: ActionChunkAdapter, config: RTCConfig):
        self.adapter = adapter
        self.config = config
        self.stats = RTCStats()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._current: ChunkResult | None = None
        self._pending: Future[ChunkResult] | None = None
        self._idx = 0
        self._last_action: np.ndarray | None = None
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "AsyncChunkRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self, observation: Any) -> None:
        """Synchronously initialize the first action chunk."""
        result = self._run_inference(observation)
        with self._lock:
            self._current = result
            self._pending = None
            self._idx = 0
            self._last_action = None

    def next_action(self, observation: Any, *, block_if_empty: bool = True) -> np.ndarray:
        """Return the next action for the foreground control loop."""
        self._raise_if_closed()
        if self._current is None:
            if not block_if_empty:
                raise RuntimeError("RTC runner has no current chunk")
            self.reset(observation)

        with self._lock:
            self._promote_ready_locked()
            current = self._current
            if current is None:
                raise RuntimeError("RTC runner failed to initialize")
            horizon = self._configured_horizon(current.actions)
            start_next_at = self._start_next_at(horizon)
            if self._idx >= start_next_at:
                self._submit_locked(observation)
            if self._idx >= horizon:
                self._handle_exhausted_locked(observation)
                current = self._current
                if current is None:
                    raise RuntimeError("RTC runner has no chunk after exhaustion")
            action = np.asarray(current.actions[self._idx]).copy()
            if self.config.blend_steps > 0:
                action = self._maybe_blend_locked(action)
            self._idx += 1
            self._last_action = action
            self.stats.actions_served += 1
            return action

    def _run_inference(self, observation: Any) -> ChunkResult:
        t0 = time.perf_counter()
        actions = self.adapter.infer_actions(observation)
        t1 = time.perf_counter()
        return ChunkResult(
            actions=np.asarray(actions),
            latency_s=t1 - t0,
            observation_time_s=t0,
            ready_time_s=t1,
        )

    def _submit_locked(self, observation: Any) -> None:
        if self._pending is not None:
            return
        self.stats.chunks_started += 1
        self._pending = self._executor.submit(self._run_inference, observation)

    def _promote_ready_locked(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        result = self._pending.result()
        self._pending = None
        self._current = result
        self._idx = 0
        self.stats.chunks_completed += 1
        self.stats.swaps += 1
        self.stats.last_latency_s = result.latency_s
        self.stats.max_latency_s = max(self.stats.max_latency_s, result.latency_s)

    def _handle_exhausted_locked(self, observation: Any) -> None:
        self._promote_ready_locked()
        current = self._current
        if current is not None and self._idx < self._configured_horizon(current.actions):
            return
        self.stats.deadline_misses += 1
        if self.config.miss_policy == "block":
            self._submit_locked(observation)
            if self._pending is None:
                raise RuntimeError("failed to submit recovery chunk")
            result = self._pending.result()
            self._pending = None
            self._current = result
            self._idx = 0
            self.stats.chunks_completed += 1
            self.stats.swaps += 1
            return
        self.stats.held_actions += 1
        if self._last_action is None:
            raise RuntimeError("cannot hold last action before any action was served")
        now = time.perf_counter()
        self._current = ChunkResult(
            actions=self._last_action[None, :],
            latency_s=0.0,
            observation_time_s=now,
            ready_time_s=now,
            metadata={"held": True},
        )
        self._idx = 0

    def _maybe_blend_locked(self, action: np.ndarray) -> np.ndarray:
        if self._last_action is None:
            return action
        current = self._current
        if current is None:
            return action
        horizon = self._configured_horizon(current.actions)
        remaining = horizon - self._idx
        if remaining > self.config.blend_steps:
            return action
        alpha = 1.0 / (remaining + 1)
        return (1.0 - alpha) * self._last_action + alpha * action

    def _configured_horizon(self, actions: np.ndarray) -> int:
        horizon = actions.shape[0]
        if self.config.action_horizon is not None:
            horizon = min(horizon, self.config.action_horizon)
        return horizon

    def _start_next_at(self, horizon: int) -> int:
        if self.config.start_next_at is not None:
            return min(self.config.start_next_at, horizon)
        return max(1, horizon // 2)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("RTC runner is closed")
