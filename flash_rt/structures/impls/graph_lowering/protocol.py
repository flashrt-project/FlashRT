"""Host-family adapter protocol for whole-graph shape lowering.

CUDA graph capture needs every shape-derived quantity of the request to
be a constant: position tables, token routing, sequence cumsums — the
things a host recomputes per call, often through a synchronize the
capture cannot record. Which functions those are is host-family
knowledge, exactly like where the attention math runs — so the lowering
lives in registered adapters, not in user harnesses. A user asks to
capture; the family that recognizes its host pins its own glue.

Every pin is a shape-derived constant of one fixed request, never a
value-dependent quantity, and every application returns an ``undo``
that restores the host bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


class GraphLoweringRefused(RuntimeError):
    """A recognized host cannot be pinned safely."""


@dataclass(frozen=True)
class GraphLowering:
    """One applied family lowering and how to take it back off."""

    undo: Callable[[], None]
    family: str
    pins: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)


class GraphLoweringAdapter(Protocol):
    """One host-family realization of the lowering contract."""

    def lower(
        self,
        model: Any,
        forward: Callable[[], Any],
    ) -> GraphLowering | None: ...


_ADAPTERS: list[GraphLoweringAdapter] = []


def register_graph_lowering_adapter(adapter: GraphLoweringAdapter) -> None:
    """Register a host-family lowering adapter."""
    _ADAPTERS.append(adapter)


def lower_for_capture(
    model: Any,
    forward: Callable[[], Any],
) -> list[GraphLowering]:
    """Apply every registered lowering that recognizes this host.

    An empty list is not a fallback: it means no family recognized the
    model, and capture proceeds on the host's own forward — which is
    correct for hosts that are already graph-safe. A recognized family
    that cannot pin safely raises :class:`GraphLoweringRefused` instead
    of leaving the host half-pinned; adapters must apply atomically.
    """
    applied: list[GraphLowering] = []
    try:
        for adapter in _ADAPTERS:
            lowering = adapter.lower(model, forward)
            if lowering is not None:
                applied.append(lowering)
    except Exception:
        for lowering in reversed(applied):
            lowering.undo()
        raise
    return applied
