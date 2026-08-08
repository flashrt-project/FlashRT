"""Host-family adapter protocol for fixed iterative schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import torch


class FixedIterationRefused(RuntimeError):
    """A candidate schedule was recognized but cannot be lowered safely."""


@dataclass(frozen=True)
class FixedIterationLowering:
    """A graph-safe callable plus the reference output that qualified it."""

    forward: Callable[[], Any]
    reference_output: Any
    family: str
    steps: int
    exact: bool = True
    compile_before_capture: bool = True
    windows: Mapping[str, torch.Tensor] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)


class FixedIterationAdapter(Protocol):
    """One host-family realization of the fixed-iteration contract."""

    def lower(
        self,
        forward: Callable[[], Any],
        model: Any | None,
    ) -> FixedIterationLowering | None: ...


_ADAPTERS: list[FixedIterationAdapter] = []


def register_fixed_iteration_adapter(adapter: FixedIterationAdapter) -> None:
    """Register a host-family schedule adapter (last registration runs last)."""
    _ADAPTERS.append(adapter)


def normalize_fixed_iteration(
    forward: Callable[[], Any],
    model: Any | None = None,
) -> FixedIterationLowering | None:
    """Return a qualified lowering, or ``None`` for an ordinary callable.

    ``None`` is not a fallback: it means no adapter recognized the callable,
    so the existing capture path receives it unchanged.  Once an adapter
    recognizes its family, an unsafe form raises :class:`FixedIterationRefused`
    rather than silently returning to the original graph-unsafe loop.
    """
    for adapter in _ADAPTERS:
        lowering = adapter.lower(forward, model)
        if lowering is not None:
            if lowering.exact:
                want = _first_tensor(lowering.reference_output)
                with torch.no_grad():
                    got = _first_tensor(lowering.forward())
                if want is None or got is None:
                    raise FixedIterationRefused(
                        f"{lowering.family}: exact schedule qualification "
                        "needs one tensor output")
                if not torch.equal(got, want):
                    raise FixedIterationRefused(
                        f"{lowering.family}: fixed-iteration lowering is "
                        "not bit-exact")
            return lowering
    return None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    logits = getattr(value, "logits", None)
    if torch.is_tensor(logits):
        return logits
    if isinstance(value, Mapping):
        for key in sorted(value):
            found = _first_tensor(value[key])
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None
