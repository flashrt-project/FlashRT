"""The house parity judge: cosine / max-abs / p99-abs between an
implementation's output and a reference.

One definition, shared by the structures layer's gates
(flashrt-structures), the NPU verification harness and any other
consumer that needs to say how far an output is from its truth. Keep
the formula here; a second copy drifts.
"""

from __future__ import annotations

import torch


def parity_metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    """Cosine / max-abs / p99-abs between ``got`` and ``want``.

    Both tensors are compared in float64 over their flattened values.
    ``p99_abs`` uses ``kthvalue`` rather than ``quantile``: exact, and free
    of quantile's input-size limit (qualification outputs can exceed it,
    e.g. LLM logits).
    """
    if got.shape != want.shape:
        raise ValueError(
            f"output shape mismatch: impl {tuple(got.shape)} vs "
            f"reference {tuple(want.shape)}"
        )
    diff = (got.double() - want.double()).abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        got.double().flatten(), want.double().flatten(), dim=0
    )
    k = max(1, int(0.99 * diff.numel()))
    return {
        "cosine": float(cosine),
        "max_abs": float(diff.max()),
        "p99_abs": float(diff.kthvalue(k).values),
    }
