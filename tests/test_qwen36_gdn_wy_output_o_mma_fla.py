"""Equivalence test for the FLA-style hand-tuned output_o kernel.

Compares ``linear_attn_gdn_wy_output_o_b64_bf16_mma_fla`` against the
existing ``linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv``
baseline on identical pre-packed inputs.
"""

import pytest
import torch

from tests.test_qwen36_gdn_wy_stages import _load_fvk, _local_cumsum_bf16, _ptr


def _pack_q_kh_v(q, k, v, chunks, H=48, Hk=16, K=128, V=128, qk_group=3):
    """Pack with GQA expansion: q,k are (S, Hk, K); v is (S, H, V)."""
    q_pack = torch.zeros(chunks, H, 64, K, device=q.device, dtype=torch.bfloat16)
    k_pack = torch.zeros(chunks, H, 64, K, device=q.device, dtype=torch.bfloat16)
    v_pack = torch.zeros(chunks, H, 64, V, device=q.device, dtype=torch.bfloat16)
    S = q.shape[0]
    for ci, start in enumerate(range(0, S, 64)):
        end = min(start + 64, S)
        v_pack[ci, :, :end - start] = v[start:end].transpose(0, 1)
        for vh in range(H):
            kh = vh // qk_group
            q_pack[ci, vh, :end - start] = q[start:end, kh]
            k_pack[ci, vh, :end - start] = k[start:end, kh]
    return q_pack, k_pack, v_pack


@pytest.mark.parametrize("S", [64, 128, 256, 512])
def test_output_o_mma_fla_matches_cublaslt(S):
    fvk = _load_fvk()
    if not hasattr(fvk, "linear_attn_gdn_wy_output_o_b64_bf16_mma_fla"):
        pytest.skip("output_o mma_fla kernel not built")

    torch.manual_seed(20260525 + S)
    Hk, H, K, V = 16, 48, 128, 128
    qk_group = H // Hk
    chunks = (S + 63) // 64
    scale = 1.0 / (K ** 0.5)

    q = torch.randn(S, Hk, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, Hk, K, device="cuda", dtype=torch.bfloat16)
    v_new = (torch.randn(S, H, V, device="cuda", dtype=torch.bfloat16) * 0.3)
    h0 = (torch.randn(chunks, H, K, V, device="cuda", dtype=torch.bfloat16) * 0.1)
    g = (torch.randn(S, H, device="cuda") * 0.02).to(torch.bfloat16)
    g_cumsum = _local_cumsum_bf16(g)

    q_l2 = (q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)
    k_l2 = (k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6)
    ).to(torch.bfloat16)

    q_pack, k_pack, v_pack = _pack_q_kh_v(
        q_l2, k_l2, v_new, chunks, H=H, Hk=Hk, K=K, V=V, qk_group=qk_group)

    # mma_fla path
    out_mma = torch.empty(S, H, V, device="cuda", dtype=torch.bfloat16)
    fvk.linear_attn_gdn_wy_output_o_b64_bf16_mma_fla(
        _ptr(q_pack), _ptr(k_pack), _ptr(v_pack),
        _ptr(h0), _ptr(g_cumsum), _ptr(out_mma),
        S, H, K, float(scale), 0)
    torch.cuda.synchronize()

    out_rawk = None
    if hasattr(fvk, "linear_attn_gdn_wy_output_o_b64_bf16_mma_fla_rawk"):
        out_rawk = torch.empty(S, H, V, device="cuda", dtype=torch.bfloat16)
        fvk.linear_attn_gdn_wy_output_o_b64_bf16_mma_fla_rawk(
            _ptr(q_pack), _ptr(k_l2), _ptr(v_pack),
            _ptr(h0), _ptr(g_cumsum), _ptr(out_rawk),
            S, Hk, H, K, qk_group, float(scale), 0)
        torch.cuda.synchronize()

    # cublasLt packed_qkv baseline. Needs scratch buffers.
    qk_base = torch.empty(chunks * H, 64, 64, device="cuda", dtype=torch.float32)
    local_a_pack = torch.empty(chunks, H, 64, 64, device="cuda", dtype=torch.bfloat16)
    qh_pack = torch.empty(chunks, H, 64, V, device="cuda", dtype=torch.bfloat16)
    local_pack = torch.empty(chunks, H, 64, V, device="cuda", dtype=torch.bfloat16)
    out_base = torch.empty(S, H, V, device="cuda", dtype=torch.bfloat16)
    fvk.linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_qkv(
        _ptr(q_pack), _ptr(k_pack), _ptr(v_pack),
        _ptr(h0), _ptr(g_cumsum),
        _ptr(qk_base), _ptr(local_a_pack),
        _ptr(qh_pack), _ptr(local_pack), _ptr(out_base),
        S, Hk, H, K, qk_group, 0)
    torch.cuda.synchronize()

    diff = (out_mma.float() - out_base.float()).abs()
    print(f"S={S}  max_diff={diff.max().item():.4g}  "
          f"mean_diff={diff.mean().item():.4g}  "
          f"out.abs.max={out_base.float().abs().max().item():.4g}")
    # Both kernels do fp32 mma over bf16 inputs with different reduction orders.
    # Bound by bf16 ulp * inner-dim. Loose tolerance like the chunk_h test.
    torch.testing.assert_close(out_mma, out_base, rtol=0,
                               atol=0.05 + 0.02 * (S // 64))
    if out_rawk is not None:
        torch.testing.assert_close(out_rawk, out_mma, rtol=0, atol=0.001)
