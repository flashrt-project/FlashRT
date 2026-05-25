"""Equivalence test for the FLA-style recompute_wu kernel.

Compares ``linear_attn_gdn_wy_recompute_wu_b64_bf16_mma_fla`` against the
existing cublasLt packed_rhs baseline on identical inputs.
"""

import pytest
import torch

from tests.test_qwen36_gdn_wy_stages import (
    _load_fvk,
    _local_cumsum_bf16,
    _ptr,
    _recompute_wu_ref,
)


def _pack_recompute_inputs(k_l2, v, beta, g_cumsum, chunks, H=48):
    """Compute Ai by solving tril A and stash rhs_w/rhs_u for cublasLt path."""
    # Same Ai computation as in the existing tests.
    K = k_l2.shape[-1]
    Hk = k_l2.shape[1]
    return None  # placeholder; tests below build Ai directly


@pytest.mark.parametrize("S", [64, 128, 256, 512])
def test_recompute_wu_mma_fla_matches_cublaslt(S):
    fvk = _load_fvk()
    if not hasattr(fvk,
                   "linear_attn_gdn_wy_recompute_wu_b64_bf16_mma_fla"):
        pytest.skip("recompute_wu mma_fla kernel not built")

    torch.manual_seed(20260525 + S)
    Hk, H, K, V = 16, 48, 128, 128
    qk_group = H // Hk
    chunks = (S + 63) // 64

    k = torch.randn(S, Hk, K, device="cuda", dtype=torch.bfloat16)
    v = (torch.randn(S, H, V, device="cuda", dtype=torch.bfloat16) * 0.3)
    beta = (torch.randn(S, H, device="cuda", dtype=torch.bfloat16) * 0.5)
    g = (torch.randn(S, H, device="cuda") * 0.02).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)

    # Build A and Ai via solve_tril, just like the production pipeline.
    # For test purposes we build a deterministic Ai directly.
    Ai = torch.zeros(chunks, H, 64, 64, device="cuda", dtype=torch.float32)
    for ci in range(chunks):
        # Random lower-triangular Ai with unit diagonal (kept small).
        m = torch.zeros(H, 64, 64, device="cuda")
        for vh in range(H):
            tri = torch.randn(64, 64, device="cuda") * 0.05
            tri = torch.tril(tri, diagonal=-1)
            tri.diagonal().fill_(1.0)
            m[vh] = tri
        Ai[ci] = m
    Ai_pack = Ai.to(torch.bfloat16)

    # mma_fla output
    w_mma = torch.empty(chunks, H, 64, K, device="cuda", dtype=torch.bfloat16)
    u_mma = torch.empty(chunks, H, 64, V, device="cuda", dtype=torch.bfloat16)
    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_mma_fla(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(Ai_pack),
        _ptr(w_mma), _ptr(u_mma),
        S, Hk, H, K, qk_group, 0)
    torch.cuda.synchronize()

    # cublasLt packed_rhs baseline. Needs scratch buffers.
    rhs_w = torch.empty_like(w_mma)
    rhs_u = torch.empty_like(u_mma)
    w_base = torch.empty_like(w_mma)
    u_base = torch.empty_like(u_mma)
    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
        _ptr(k_l2), _ptr(v), _ptr(beta), _ptr(g_cumsum), _ptr(Ai_pack),
        _ptr(rhs_w), _ptr(rhs_u), _ptr(w_base), _ptr(u_base),
        S, Hk, H, K, qk_group, 0)
    torch.cuda.synchronize()

    # torch fp32 ref (from existing test helper which expects "unpacked" Ai).
    # Build per-chunk Ai stack in the right shape.
    w_ref, u_ref = _recompute_wu_ref(k_l2, v, beta, g_cumsum, Ai)
    # ref is (S, H, V/K) un-packed. Pack to (chunks, H, 64, K/V).
    def _pack(x, K_or_V):
        out = torch.zeros(chunks, H, 64, K_or_V, device="cuda",
                          dtype=torch.bfloat16)
        for ci, start in enumerate(range(0, S, 64)):
            end = min(start + 64, S)
            out[ci, :, :end - start] = x[start:end].transpose(0, 1)
        return out

    w_ref_pack = _pack(w_ref, K)
    u_ref_pack = _pack(u_ref, V)

    def _max_diff(a, b):
        return (a.float() - b.float()).abs().max().item()

    d_w_mma = _max_diff(w_mma, w_ref_pack)
    d_u_mma = _max_diff(u_mma, u_ref_pack)
    d_w_base = _max_diff(w_base, w_ref_pack)
    d_u_base = _max_diff(u_base, u_ref_pack)
    print(f"S={S}  w: mma={d_w_mma:.4g} base={d_w_base:.4g}  "
          f"u: mma={d_u_mma:.4g} base={d_u_base:.4g}")

    slack = 0.05
    assert d_w_mma <= d_w_base + slack
    assert d_u_mma <= d_u_base + slack
