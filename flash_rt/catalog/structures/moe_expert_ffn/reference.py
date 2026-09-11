"""Ground-truth reference for the sparse MoE feed-forward block.

Plainest possible PyTorch, never executed on a serving hot path. Routing
replicates the Qwen3.x convention exactly: softmax over all experts,
iterative top-k with lower-index tie-break, clamp, renormalize.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _route_softmax_topk_clamp_renorm(
    logits: torch.Tensor, k_top: int, clamp_min: float = 1e-20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over E -> iterative top-k (lower index wins ties) -> renorm."""
    probs = torch.softmax(logits.float(), dim=-1)
    m, _ = probs.shape
    ids = torch.empty(m, k_top, dtype=torch.int64, device=logits.device)
    vals = torch.empty(m, k_top, dtype=torch.float32, device=logits.device)
    work = probs.clone()
    for j in range(k_top):
        # argmax returns the first (lowest-index) maximum, matching the
        # host convention this structure binds to.
        idx = work.argmax(dim=-1)
        ids[:, j] = idx
        vals[:, j] = work.gather(-1, idx[:, None]).squeeze(-1)
        work.scatter_(-1, idx[:, None], float("-inf"))
    vals = vals.clamp_min(clamp_min)
    vals = vals / vals.sum(dim=-1, keepdim=True)
    return ids, vals


def moe_expert_ffn_ref(
    x: torch.Tensor,
    w_gate_exps: torch.Tensor,
    w_up_exps: torch.Tensor,
    w_down_exps: torch.Tensor,
    *,
    w_router: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    expert_weights: torch.Tensor | None = None,
    k_top: int = 8,
    w_gate_shexp: torch.Tensor | None = None,
    w_up_shexp: torch.Tensor | None = None,
    w_down_shexp: torch.Tensor | None = None,
    w_gate_inp_shexp: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed expert GLU-FFN with optional sigmoid-gated shared expert.

    Either ``w_router`` (fused routing) or ``expert_ids``/``expert_weights``
    (external routing) must be provided.
    """
    if expert_ids is None:
        assert w_router is not None, "fused routing needs w_router"
        logits = x.float() @ w_router.float()
        expert_ids, expert_weights = _route_softmax_topk_clamp_renorm(logits, k_top)
    assert expert_weights is not None

    m = x.shape[0]
    y = torch.zeros(m, w_down_exps.shape[-1], dtype=torch.float32, device=x.device)
    for t in range(m):
        xt = x[t].float()
        for j in range(expert_ids.shape[1]):
            e = int(expert_ids[t, j])
            g = xt @ w_gate_exps[e].float()
            u = xt @ w_up_exps[e].float()
            h = F.silu(g) * u
            y[t] += float(expert_weights[t, j]) * (h @ w_down_exps[e].float())

    if w_gate_shexp is not None:
        assert w_up_shexp is not None and w_down_shexp is not None
        assert w_gate_inp_shexp is not None
        for t in range(m):
            xt = x[t].float()
            sig = torch.sigmoid(xt @ w_gate_inp_shexp.float())
            h = F.silu(xt @ w_gate_shexp.float()) * (xt @ w_up_shexp.float())
            y[t] += sig * (h @ w_down_shexp.float())

    if residual is not None:
        y = y + residual.float()
    return y.to(x.dtype)
