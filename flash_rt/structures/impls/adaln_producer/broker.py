"""Step-scoped style materialisation shared across one conditioning stream.

The adaptive-norm kernel takes ``style`` as a contiguous ``(rows, W)``
tensor, so each producer expands its one style row to the row count and
materialises it. That is correct and, done per producer, wasteful in a
way no single producer can see: every producer on one conditioning
stream resolves the *same* step, and the whole stream's styles are
therefore fixed for the duration of a step. Measured on pi05, the
per-producer form spends 0.68 ms in 720 launches moving 221 MB — a
volume worth about 0.15 ms at this card's bandwidth, so the cost is the
launches, not the bytes.

The broker turns per-call work into per-step work. One index lookup and
one copy fill a ``(P, rows, W)`` buffer for every producer in the
stream; each producer then reads ``buf[slot]``, which is a contiguous
view, so the kernel's contract is untouched. This is the
``cadence_static`` idea applied inside a structure rather than around a
module: hold what changes at step cadence, not at call cadence.

Two things make it safe rather than merely fast:

- the writer is the producer the host actually calls first, taken from
  the observed call order of the calibration pass, not from the order
  the modules happen to sit in the tree;
- the readers depend on the writer through the buffer itself, the same
  ordering the packed-projection stash relies on inside compiled and
  captured graphs.
"""

from __future__ import annotations

import torch
from torch import nn


@torch.library.custom_op("flash_rt_structures::style_broadcast",
                         mutates_args={"out"})
def _style_broadcast(src: torch.Tensor, out: torch.Tensor) -> None:
    """Fill ``out`` (P, rows, W) by repeating each row of ``src`` (P, W).

    Opaque on purpose. Written as plain tensor work, the compiler sees a
    buffer whose only consumers are slices of it and inlines the fill
    into each consumer — a correct buffer elimination that happens to
    undo the sharing this broker exists for. Measured: the fill stayed
    at 720 launches, they merely moved bucket. Behind an opaque op the
    fill happens once and the readers read.
    """
    out.copy_(src.unsqueeze(1).expand_as(out))


@_style_broadcast.register_fake
def _style_broadcast_fake(src: torch.Tensor, out: torch.Tensor) -> None:
    return None


class StyleBroker(nn.Module):
    """One conditioning stream's styles, materialised once per step."""

    def __init__(self, locator, tables, rows: int):
        super().__init__()
        widths = {t.shape[-1] for t in tables}
        if len(widths) != 1:
            raise ValueError(
                f"style_broker: producers differ in style width {widths}")
        steps = {t.shape[0] for t in tables}
        if len(steps) != 1:
            raise ValueError(
                f"style_broker: producers differ in step count {steps}")
        self.locator = locator
        self.slots = len(tables)
        self.rows = rows
        # [steps, slots, W]: one index_select picks the whole stream's
        # styles for the current step
        self.register_buffer("stack", torch.stack(
            [t.to(torch.bfloat16) for t in tables], dim=1).contiguous())
        self.register_buffer("buf", torch.empty(
            self.slots, rows, widths.pop(), device=self.stack.device,
            dtype=torch.bfloat16))

    def refresh(self, cond: torch.Tensor) -> torch.Tensor:
        """Resolve the step and materialise every slot. Writer only."""
        idx = self.locator(cond)
        sel = self.stack.index_select(0, idx).reshape(self.slots, -1)
        torch.ops.flash_rt_structures.style_broadcast(sel, self.buf)
        return idx

    def slice(self, slot: int) -> torch.Tensor:
        """This producer's style for the current step, contiguous."""
        return self.buf[slot]


def bind_style_broker(producers, rows: int) -> StyleBroker | None:
    """Attach one broker to producers already bound on the same stream.

    ``producers`` must be in the order the host calls them; the first is
    the writer. Returns ``None`` when there is nothing to share (a single
    producer pays the same either way), leaving every producer on its own
    materialisation — this composes onto bound producers and can only
    remove work, never add a requirement.
    """
    # a form that never reads a materialised style has nothing to share:
    # attaching anyway would hold a buffer nobody reads and still report
    # a broker as active. Found on the second host, where every producer
    # is the layer form.
    producers = [p for p in producers if p.takes_style_rows]
    if len(producers) < 2:
        return None
    locator = producers[0].locator
    if any(p.locator is not locator for p in producers):
        raise ValueError(
            "style_broker: producers do not share a step locator, so they "
            "are not one conditioning stream")
    broker = StyleBroker(locator, [p.styles for p in producers], rows)
    for slot, producer in enumerate(producers):
        producer.attach_broker(broker, slot, writer=(slot == 0))
    return broker
