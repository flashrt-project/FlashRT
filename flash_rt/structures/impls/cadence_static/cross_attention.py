"""Cross-attention K/V addressing for the existing cadence-static structure.

Cross attention has a stable structural signature: Q consumes the current
hidden width, while K/V consume a different encoder width.  The encoder-side
K/V projections may therefore be refreshed once at the encoder cadence and
read from static buffers inside a repeated denoise loop.  Self attention has
equal Q/K/V input widths and is deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from .buffers import StaticOutput, bind_cadence_static


@dataclass(frozen=True)
class CrossKvCandidate:
    path: str
    module: torch.nn.Module


def discover_cross_attention_kv(
    model: torch.nn.Module,
) -> tuple[CrossKvCandidate, ...]:
    """Find encoder-side K/V projections without host or model names."""
    found = []
    for path, attention in model.named_modules():
        q_proj = getattr(attention, "to_q", None)
        k_proj = getattr(attention, "to_k", None)
        v_proj = getattr(attention, "to_v", None)
        if not all(
            isinstance(module, torch.nn.Linear)
            for module in (q_proj, k_proj, v_proj)
        ):
            continue
        if q_proj.in_features == k_proj.in_features:
            continue
        if (
            k_proj.in_features != v_proj.in_features
            or k_proj.out_features != v_proj.out_features
        ):
            continue
        found.extend(
            (
                CrossKvCandidate(f"{path}.to_k", k_proj),
                CrossKvCandidate(f"{path}.to_v", v_proj),
            )
        )
    return tuple(found)


def capture_cross_attention_kv(
    candidates: Sequence[CrossKvCandidate],
    forward: Callable[[], object],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Capture each candidate across one complete repeated-loop forward."""
    rows = [[] for _ in candidates]
    hooks = []
    for candidate, outputs in zip(candidates, rows):
        hooks.append(
            candidate.module.register_forward_hook(
                lambda _module, _args, output, outputs=outputs:
                outputs.append(output.detach().clone())
            )
        )
    try:
        with torch.no_grad():
            forward()
    finally:
        for hook in hooks:
            hook.remove()
    return tuple(tuple(outputs) for outputs in rows)


def bind_cross_attention_kv(
    candidates: Sequence[CrossKvCandidate],
    captures: Sequence[Sequence[torch.Tensor]],
    *,
    replacements: Mapping[str, torch.nn.Module] | None = None,
) -> tuple[dict[str, StaticOutput], tuple[StaticOutput, ...]]:
    """Bind K/V buffers, optionally consuming already-bound projections."""
    replacements = replacements or {}
    modules = []
    for candidate in candidates:
        replacement = replacements.get(candidate.path, candidate.module)
        if getattr(replacement, "_frt_requires_sibling_order", False):
            # A StashReader (and any equivalent composed tail) is not a
            # projection in isolation: it reads data produced by a sibling.
            # Refresh happens outside that sibling call order, so recompute
            # from the candidate's real projection instead of copying stale
            # stash contents into the cadence buffer.
            replacement = candidate.module
        modules.append(replacement)
    statics, _ = bind_cadence_static(modules, captures)
    return (
        {
            candidate.path: static
            for candidate, static in zip(candidates, statics)
        },
        tuple(statics),
    )


def refresh_cross_attention_kv(
    statics: Sequence[StaticOutput],
    encoder_hidden_states: torch.Tensor,
) -> None:
    """Refresh all K/V buffers once before the repeated attention loop."""
    with torch.no_grad():
        for static in statics:
            static.buffer.copy_(
                static.host_module(encoder_hidden_states)
            )
            static.refreshed()


def wire_refresh_to_producer(
    model: torch.nn.Module,
    statics: Sequence[StaticOutput],
    forward: Callable[[], object],
):
    """Wire the K/V refresh into the producing module's own forward.

    The manual :func:`refresh_cross_attention_kv` form leaves the
    refresh outside the hot path. That is the right split when only the
    fast loop is captured — but a *whole-pipeline* capture then records
    an encoder whose output feeds nothing: the banks are written outside
    the graph, so replaying on a new observation silently reuses the old
    encoding. This wires the split shut: one probe forward identifies
    the module whose output tensor the statics' host projections consume
    (by object identity), and a forward hook on that producer refreshes
    every bank whenever it runs. Eager, compiled and captured forms all
    carry the observation through; within one call the banks are still
    written once and read every loop step, so the cadence saving stands.

    Returns ``(producer, handle)``; ``handle.remove()`` unwires.
    Raises ``ValueError`` when no single producer can be identified —
    the caller keeps the explicit-refresh contract in that case.
    """
    if not statics:
        raise ValueError("cadence_static: no statics to wire")
    consumed: dict[int, None] = {}
    probes = []
    for static in statics:
        def grab(_module, args, _consumed=consumed):
            if args and torch.is_tensor(args[0]):
                _consumed[id(args[0])] = None
        probes.append(static.register_forward_pre_hook(grab))
    produced: dict[int, torch.nn.Module] = {}

    def note(module, _args, output):
        if torch.is_tensor(output):
            # parents fire after children, so an identity-preserving
            # wrapper chain resolves to its outermost module
            produced[id(output)] = module

    watchers = [module.register_forward_hook(note)
                for _, module in model.named_modules()]
    try:
        with torch.no_grad():
            forward()
    finally:
        for hook in probes + watchers:
            hook.remove()
    producers = {id(produced[x]): produced[x]
                 for x in consumed if x in produced}
    if len(producers) != 1:
        raise ValueError(
            "cadence_static: could not identify one producer module for "
            f"the cross-attention statics ({len(producers)} candidate(s) "
            "matched by tensor identity)")
    (producer,) = producers.values()

    def refresh(_module, _args, output):
        if not torch.is_tensor(output):
            return None
        with torch.no_grad():
            for static in statics:
                static.buffer.copy_(static.host_module(output))
                static.refreshed()
        return None

    return producer, producer.register_forward_hook(refresh)
