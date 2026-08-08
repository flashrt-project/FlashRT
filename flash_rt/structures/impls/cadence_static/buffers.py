"""cadence_static — hold work that changes slower than the hot loop.

A tick pipeline usually runs several cadences at once: a denoise loop
that repeats every tick, and encoder-side work that only changes when a
new observation or prompt arrives. Modules on the slower cadence still
sit inside the hot path, so a graph captures them and pays for them
every tick even though their inputs are unchanged.

This structure moves such a module out of the loop: its output becomes
a static buffer the captured graph reads, and the real computation runs
in an update function the host calls at the module's own cadence. The
resulting split is explicit — the update callable is returned to the
caller so it can be registered as a recipe ``outside_update`` rather
than hidden inside the replacement.

Qualification is empirical: the wrapped output must actually be
constant across the fast loop. Callers pass calibration captures from
several iterations of the hot loop and binding refuses when they
disagree — a module whose output moves per step is not a cadence
substructure, and freezing it would silently change the model.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from ...guard import PROCEED, GuardedSeam


class StaticOutput(GuardedSeam, torch.nn.Module):
    """Return a buffer the host refreshes at the slower cadence.

    This structure has no input contract to check — ignoring its inputs
    is the point. Its failure mode is the other one: the host stops
    calling the update function and the buffer goes stale while every
    read still succeeds. So the ledger counts reads against refreshes,
    and a seam read many times and refreshed never is visible as exactly
    that instead of as a seam that worked.
    """

    _frt_host_attr = "host_module"
    _frt_can_fallback = True
    # the retained host is not a fallback copy: the cadence refresh
    # projects through it every observation, so its weights must stay
    # resident — consumption would corrupt the banks silently
    _frt_host_serving = True

    def __init__(self, original: torch.nn.Module, value: torch.Tensor):
        super().__init__()
        self.host_module = original
        self.register_buffer("buffer", value.contiguous().clone())
        guard = self._frt_arm(dtypes=None, device=self.buffer.device)
        guard.notes.update(reads=0, refreshes=0)

    def refreshed(self) -> None:
        """Record that the slower cadence ran. Called by the updater."""
        guard = self._frt_guard
        if guard is not None and not torch.compiler.is_compiling():
            guard.notes["refreshes"] += 1

    def forward(self, *args, **kwargs):
        # same rule as the contract check: no Python bookkeeping while a
        # compiler is tracing this forward
        if not torch.compiler.is_compiling():
            self._frt_touch()
            self._frt_guard.notes["reads"] += 1
        return self.buffer

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_module"), name)


def bind_cadence_static(
    modules: Sequence[torch.nn.Module],
    captures: Sequence[Sequence[torch.Tensor]],
    *,
    recompute: Callable[[], Sequence[torch.Tensor]] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """Freeze module outputs into buffers plus one update function.

    ``captures[i]`` holds the outputs module ``i`` produced across
    several iterations of the hot loop; they must agree, or the module
    is not on a slower cadence and binding refuses. ``recompute``
    returns fresh outputs for the same modules — typically by running
    the host's own encoder path — and the returned update callable
    copies them into the buffers. Register that callable as the
    recipe's outside update so the split cadence stays explicit and
    the slower work is still timed.
    """
    if len(modules) != len(captures):
        raise ValueError("cadence_static: modules/captures mismatch")
    statics = []
    for i, (mod, caps) in enumerate(zip(modules, captures)):
        if not caps:
            raise ValueError(f"cadence_static: module {i} has no captures")
        first = caps[0]
        for other in caps[1:]:
            if not torch.allclose(first, other, rtol=rtol, atol=atol):
                raise ValueError(
                    f"cadence_static: module {i} output varies within "
                    "the hot loop — not a cadence substructure")
        statics.append(StaticOutput(mod, first))

    def update() -> None:
        if recompute is None:
            raise RuntimeError(
                "cadence_static: no recompute function was supplied, so "
                "the buffers cannot be refreshed for a new observation")
        with torch.no_grad():
            fresh = recompute()
            for static, value in zip(statics, fresh):
                static.buffer.copy_(value)
                static.refreshed()

    return statics, update
