"""Ground-truth reference for the stateful Gated Delta core."""

from __future__ import annotations

import torch


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + 1e-6)


def gated_delta_core_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    qk_l2norm: bool = True,
    variant: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the recurrence with FP32 internal state accumulation."""
    del variant
    if not (q.shape == k.shape == v.shape):
        raise ValueError("gated_delta_core: Q/K/V shapes differ")
    if q.ndim != 4 or log_decay.shape != q.shape[:3] \
            or beta.shape != log_decay.shape:
        raise ValueError("gated_delta_core: incompatible boundary shapes")
    batch, _, heads, width = q.shape
    if state is None:
        state_f = torch.zeros(
            batch, heads, width, width, device=q.device, dtype=torch.float32)
    else:
        if state.shape != (batch, heads, width, width):
            raise ValueError("gated_delta_core: state shape differs")
        state_f = state.float().clone()
    query = _l2norm(q) if qk_l2norm else q.float()
    key = _l2norm(k) if qk_l2norm else k.float()
    query = query * (width ** -0.5)
    outputs = []
    for index in range(q.shape[1]):
        decay = log_decay[:, index].float().exp()[..., None, None]
        key_i = key[:, index]
        value_i = v[:, index].float()
        query_i = query[:, index]
        state_f = state_f * decay
        memory = torch.einsum("bhdt,bhd->bht", state_f, key_i)
        delta = (value_i - memory) * beta[:, index].float()[..., None]
        state_f = state_f + key_i[..., :, None] * delta[..., None, :]
        outputs.append(torch.einsum(
            "bhdt,bhd->bht", state_f, query_i).to(q.dtype))
    return torch.stack(outputs, dim=1), state_f.to(q.dtype)
