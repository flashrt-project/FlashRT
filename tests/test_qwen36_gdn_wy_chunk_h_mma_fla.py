"""Equivalence test for the FLA-style hand-tuned chunk_h kernel.

Compares ``linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla`` (RAW-input
mma.sync + cp.async kernel) against the existing
``linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu`` baseline and
the torch fp32 reference, on identical inputs.

Both kernels use fp32 mma accumulators but different reduction tile
orders, so per-element drift across NT chunks accumulates differently.
The test asserts the new kernel is no worse than packed_wu relative to
the fp32 reference (and direct mma_fla vs packed_wu stays within ~2x
of the packed_wu-vs-ref drift).
"""

import pytest
import torch

from tests.test_qwen36_gdn_wy_stages import (
    _chunk_h_ref,
    _load_fvk,
    _local_cumsum_bf16,
    _ptr,
)


def _pack_wu(w, u, chunks):
    w_pack = torch.zeros(chunks, 48, 64, 128, device=w.device,
                         dtype=torch.bfloat16)
    u_pack = torch.zeros_like(w_pack)
    for ci, start in enumerate(range(0, w.shape[0], 64)):
        end = min(start + 64, w.shape[0])
        w_pack[ci, :, :end - start] = w[start:end].transpose(0, 1)
        u_pack[ci, :, :end - start] = u[start:end].transpose(0, 1)
    return w_pack, u_pack


@pytest.mark.parametrize("S", [64, 128, 256, 512])
def test_mma_fla_matches_packed_wu_and_reference(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla"):
        pytest.skip("mma_fla kernel not built")

    torch.manual_seed(20260525 + S)
    Hk, H, K, V = 16, 48, 128, 128
    # Use bounded inputs so the recurrence does not overflow at large NT.
    scale = 0.2
    k = torch.randn(S, Hk, K, device="cuda", dtype=torch.bfloat16) * scale
    u = torch.randn(S, H,  V, device="cuda", dtype=torch.bfloat16) * scale
    w = torch.randn(S, H,  K, device="cuda", dtype=torch.bfloat16) * scale
    g = (torch.randn(S, H, device="cuda") * 0.02).to(torch.bfloat16)
    state0 = (torch.randn(H, K, V, device="cuda") * 0.02
              ).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    g_cumsum_bf16 = _local_cumsum_bf16(g)

    # --- mma_fla path (PACKED w/u inputs, bf16 g, bf16 state in-place) ---
    chunks = (S + 63) // 64
    w_pack_mma, u_pack_mma = _pack_wu(w, u, chunks)
    state_mma = state0.clone()
    h_mma = torch.empty(chunks, H, K, V, device="cuda",
                        dtype=torch.bfloat16)
    v_mma = torch.empty(S, H, V, device="cuda", dtype=torch.bfloat16)
    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla(
        _ptr(k_l2), _ptr(w_pack_mma), _ptr(u_pack_mma), _ptr(g_cumsum_bf16),
        _ptr(state_mma), _ptr(h_mma), _ptr(v_mma), 0, 0,
        S, Hk, H, K, H // Hk, 0)
    torch.cuda.synchronize()

    # --- packed_wu baseline ---
    w_pack, u_pack = _pack_wu(w, u, chunks)
    state_base = state0.clone()
    h_base = torch.empty_like(h_mma)
    v_base = torch.empty_like(v_mma)
    k_pack = torch.empty_like(w_pack)
    wh_pack = torch.empty_like(w_pack)
    decayed = torch.empty_like(w_pack)
    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
        _ptr(k_l2), _ptr(w_pack), _ptr(u_pack), _ptr(g_cumsum_bf16),
        _ptr(state_base), _ptr(h_base), _ptr(v_base),
        _ptr(k_pack), _ptr(wh_pack), _ptr(decayed),
        S, Hk, H, K, H // Hk, 0)
    torch.cuda.synchronize()

    # --- torch reference (fp32 math) ---
    h_ref, v_ref, state_ref = _chunk_h_ref(
        k_l2, u, w, g_cumsum_bf16, state0)

    # Both mma_fla and packed_wu use fp32 mma accumulators on bf16 inputs.
    # The reduction order differs (FLA: 64-element inner-K mma chain;
    # packed_wu: cuBLASLt cublasLtMatmul tiling). Per-chunk drift is
    # bounded by bf16 ULP * matrix-dim ~= O(2e-3). Over NT chunks this
    # compounds into the running state proportionally to NT * state-norm.
    # Scale tolerance with NT and the input magnitude (scale=0.2).
    # Both mma_fla and packed_wu accumulate in fp32 mma over bf16 inputs but
    # with different reduction tile orders. Per-element diff is roughly
    # bf16-ULP * sqrt(NT * inner-dim), absolute, plus a scale-dependent
    # constant. The existing packed_wu has the same drift vs the fp32 ref.
    # Test that the new kernel is no worse than packed_wu relative to ref,
    # within a small slack.
    def _max_diff(a, b):
        return (a.float() - b.float()).abs().max().item()

    base_diff_h = _max_diff(h_base, h_ref)
    base_diff_v = _max_diff(v_base, v_ref)
    base_diff_s = _max_diff(state_base.float(), state_ref.float())
    mma_diff_h  = _max_diff(h_mma, h_ref)
    mma_diff_v  = _max_diff(v_mma, v_ref)
    mma_diff_s  = _max_diff(state_mma, state_ref.float())

    # Floor for small-NT cases where the baseline diff itself is sub-ULP.
    slack = 0.05
    assert mma_diff_h <= base_diff_h + slack, (
        f"S={S} mma_diff_h={mma_diff_h} vs base_diff_h={base_diff_h}")
    assert mma_diff_v <= base_diff_v + slack, (
        f"S={S} mma_diff_v={mma_diff_v} vs base_diff_v={base_diff_v}")
    assert mma_diff_s <= base_diff_s + slack, (
        f"S={S} mma_diff_s={mma_diff_s} vs base_diff_s={base_diff_s}")

    # Direct mma_fla vs packed_wu comparison: should be in the same ballpark
    # (twice the baseline-vs-ref drift, since both kernels can drift in
    # opposite directions). Keep a reasonable upper bound.
    h_kk = _max_diff(h_mma, h_base)
    v_kk = _max_diff(v_mma, v_base)
    s_kk = _max_diff(state_mma, state_base.float())
    assert h_kk <= 2 * base_diff_h + slack
    assert v_kk <= 2 * base_diff_v + slack
    assert s_kk <= 2 * base_diff_s + slack
