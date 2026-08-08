"""Step-table memoization for step-constant conditioning producers.

Diffusion-style hosts recompute ``dense(cond)`` in every layer at every
denoise step, yet ``cond`` depends only on the timestep: over a tick the
producer emits a small fixed set of vectors. This implementation
replaces such a producer with a calibrated table — the distinct
conditioning vectors seen during calibration and the outputs the host's
own producer computed for them. At runtime the module locates the
current step by nearest-neighbour match against the stored vectors
(pure tensor ops: safe under both compile tracing and graph capture,
no Python state) and gathers the stored row instead of running the
GEMV. Outputs are bit-identical to calibration by construction; the
match itself is arbitrated by the caller's parity gate.

Qualification refuses hosts whose conditioning is not actually
step-quantized: if calibration sees more distinct vectors than
``max_steps``, the producer depends on more than the step and a table
would silently mis-hit — that host keeps its GEMV.
"""

from __future__ import annotations

import torch


class StepTableLinear(torch.nn.Module):
    """Drop-in for a ``nn.Linear`` whose input is step-constant."""

    def __init__(self, original: torch.nn.Module, conds: torch.Tensor,
                 table: torch.Tensor,
                 locator: "StepTableLinear | None" = None):
        super().__init__()
        self.host_linear = original
        # match score: argmax(2 c·k - |k|^2) == nearest neighbour.
        # Sibling tables fed by the same conditioning stream share the
        # locator buffers (same tensor objects), so a compiling host
        # sees one common subexpression per step instead of one locate
        # per table — the redundant matches fold away.
        if locator is not None:
            self.register_buffer("conds_t", locator.conds_t)
            self.register_buffer("cond_sq", locator.cond_sq)
        else:
            self.register_buffer("conds_t", conds.float().t().contiguous())
            self.register_buffer("cond_sq",
                                 (conds.float() ** 2).sum(-1).contiguous())
        self.register_buffer("table", table.contiguous())

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        flat = cond.reshape(-1, cond.shape[-1]).float()
        scores = 2.0 * (flat @ self.conds_t) - self.cond_sq
        idx = scores.argmax(dim=-1)
        out = self.table.index_select(0, idx)
        return out.reshape(*cond.shape[:-1], out.shape[-1])

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


def bind_step_table(original: torch.nn.Module,
                    calibration: list[tuple[torch.Tensor, torch.Tensor]],
                    *, max_steps: int = 64,
                    dedup_rtol: float = 1e-5,
                    share_locator_with: StepTableLinear | None = None
                    ) -> StepTableLinear:
    """Build a step table from real ``(cond, out)`` calibration pairs.

    Pairs come from hooking the host's own producer over at least one
    full tick, so the table rows are exactly what the host computed.
    Refuses (``ValueError``) when the distinct-vector count exceeds
    ``max_steps`` — the producer is then not step-constant and a table
    would alias different inputs onto one row.

    ``share_locator_with``: a sibling table bound from the same
    conditioning stream; when its stored vectors match this
    calibration exactly (same set, same order), the new table reuses
    the sibling's locator buffers so redundant per-table step matches
    can fold into one. On any mismatch the table keeps its own
    locator — sharing is an optimization, never an assumption.
    """
    if not calibration:
        raise ValueError("step_table: no calibration pairs captured")
    conds: list[torch.Tensor] = []
    outs: list[torch.Tensor] = []
    for cond, out in calibration:
        c = cond.detach().reshape(-1, cond.shape[-1])
        o = out.detach().reshape(-1, out.shape[-1])
        for row in range(c.shape[0]):
            cr = c[row]
            if any(torch.allclose(cr, seen, rtol=dedup_rtol,
                                  atol=1e-6 * cr.abs().max().item() + 1e-12)
                   for seen in conds):
                continue
            conds.append(cr.clone())
            outs.append(o[row].clone())
            if len(conds) > max_steps:
                raise ValueError(
                    f"step_table: >{max_steps} distinct conditioning "
                    "vectors — producer is not step-constant, keeping "
                    "the host GEMV")
    stacked = torch.stack(conds)
    locator = None
    if (share_locator_with is not None
            and share_locator_with.conds_t.shape[1] == stacked.shape[0]
            and torch.allclose(share_locator_with.conds_t.t(),
                               stacked.float().to(
                                   share_locator_with.conds_t.device),
                               rtol=dedup_rtol, atol=1e-6)):
        locator = share_locator_with
    return StepTableLinear(original, stacked, torch.stack(outs),
                           locator=locator)
