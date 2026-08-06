"""Numerical contracts for the Pi0.5 Thor FP4 fusion kernels.

Each test pins a fused kernel against the unfused chain it replaces:

* the seqused softmax fold against the standalone mask + softmax chain
  (exact equality — the fold reproduces the masked path's arithmetic),
* the fused GeGLU epilogue against the gate/up GEMM + combiner chain
  (at least the chain's accuracy, since it quantizes once instead of twice),
* the vectorized SigLIP LayerNorms against the reference norm + quantize
  pair (the reduction order differs at ulp level).
"""

import pytest
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_rt.flash_rt_kernels as fvk
from flash_rt.executors.fp4_utils import (
    FP4ActScratch, FP4Buffer, quant_act_nvfp4, quant_weight_nvfp4)


def _require_thor():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA Thor"
    assert tuple(torch.cuda.get_device_capability(0)) == (11, 0)
    assert fvk_fp4.has_nvfp4()


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(),
                               0).item()


def test_seqused_softmax_fold_matches_masked_chain():
    """The folded mask must reproduce the standalone mask + softmax result.

    Positions at or past the valid length decode to exactly zero either way:
    the reference writes -65504 and exponentiates it to zero, the fold skips
    them and stores zero directly.
    """
    _require_thor()
    torch.manual_seed(20260728)
    S, NH, HD, S_kv_max = 10, 8, 256, 992
    ctx = fvk.FvkContext()
    q = (torch.randn(S, NH * HD, dtype=torch.float16, device='cuda') * 0.7)
    k = (torch.randn(S_kv_max, HD, dtype=torch.float16, device='cuda') * 0.7)
    v = (torch.randn(S_kv_max, HD, dtype=torch.float16, device='cuda') * 0.7)
    scale = 1.0 / (HD ** 0.5)

    for valid in (S_kv_max, 978, 700):
        seqused = torch.tensor([valid], dtype=torch.int32, device='cuda')
        logits = torch.zeros(S * NH, S_kv_max, dtype=torch.float16,
                             device='cuda')
        out_ref = torch.zeros(S, NH * HD, dtype=torch.float16, device='cuda')
        out_new = torch.zeros_like(out_ref)

        fvk.attention_qkv_fp16_seqused(
            ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
            logits.data_ptr(), out_ref.data_ptr(),
            S, S_kv_max, NH, HD, seqused.data_ptr(), scale, 0)
        torch.cuda.synchronize()

        logits.zero_()
        fvk.attention_qkv_fp16_seqused_v2(
            ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
            logits.data_ptr(), out_new.data_ptr(),
            S, S_kv_max, NH, HD, seqused.data_ptr(), scale, 0)
        torch.cuda.synchronize()

        assert torch.equal(out_ref, out_new), (
            f"seqused fold diverged at valid={valid}: "
            f"max |delta| = {(out_ref.float() - out_new.float()).abs().max()}")


@pytest.mark.parametrize("S_kv_max", [1024, 1025, 2048])
def test_seqused_fold_handles_wide_logits(S_kv_max):
    """Rows wider than the fold's register tile still normalize correctly.

    1024 is where the register-tiled softmax hands over to the multi-pass
    one; past it the tiled path would leave the tail unnormalized while the
    PV GEMM still consumed it.
    """
    _require_thor()
    torch.manual_seed(20260805)
    S, NH, HD = 2, 4, 128
    ctx = fvk.FvkContext()
    q = (torch.randn(S, NH * HD, dtype=torch.float16, device='cuda') * 0.7)
    k = (torch.randn(S_kv_max, HD, dtype=torch.float16, device='cuda') * 0.7)
    v = (torch.randn(S_kv_max, HD, dtype=torch.float16, device='cuda') * 0.7)
    scale = 1.0 / (HD ** 0.5)

    for valid in (S_kv_max, S_kv_max - 37):
        seqused = torch.tensor([valid], dtype=torch.int32, device='cuda')
        logits = torch.zeros(S * NH, S_kv_max, dtype=torch.float16,
                             device='cuda')
        out_new = torch.zeros(S, NH * HD, dtype=torch.float16, device='cuda')
        fvk.attention_qkv_fp16_seqused_v2(
            ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
            logits.data_ptr(), out_new.data_ptr(),
            S, S_kv_max, NH, HD, seqused.data_ptr(), scale, 0)
        torch.cuda.synchronize()

        # Reference the math directly: the unfused sibling shares the same
        # 1024-column tile, so it cannot arbitrate past it on its own.
        qh = q.view(S, NH, HD).permute(1, 0, 2).float()
        kh = k[:valid].unsqueeze(0).expand(NH, valid, HD).float()
        vh = v[:valid].unsqueeze(0).expand(NH, valid, HD).float()
        ref = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, scale=scale).permute(1, 0, 2).reshape(S, NH * HD)
        got = out_new.float()
        assert torch.isfinite(got).all(), (
            f"S_kv_max={S_kv_max} valid={valid}: non-finite output")
        a, b = got.flatten().double(), ref.flatten().double()
        cos = float(a @ b / (a.norm() * b.norm()))
        assert cos >= 0.999, (
            f"S_kv_max={S_kv_max} valid={valid}: cosine {cos:.6f} vs torch "
            f"SDPA (max abs error "
            f"{float((got - ref).abs().max()):.6f})")


def test_fused_geglu_epilogue_matches_split_chain():
    """The interleaved GeGLU GEMM must be at least as accurate as the
    gate GEMM + up GEMM + combiner chain it replaces."""
    _require_thor()
    torch.manual_seed(20260728)
    M, D, H = 968, 2048, 8192
    N_il = 2 * H

    x = torch.randn(M, D, dtype=torch.float16, device='cuda')
    w_g = torch.randn(H, D, dtype=torch.float16, device='cuda') * 0.02
    w_u = torch.randn(H, D, dtype=torch.float16, device='cuda') * 0.02

    hidden_ref = (F.gelu(x.float() @ w_g.float().T, approximate="tanh")
                  * (x.float() @ w_u.float().T))

    act = FP4ActScratch(M, D, device='cuda')
    quant_act_nvfp4(x, act, M)

    # Reference chain: two fp4-out GEMMs plus the combiner.
    q_g = quant_weight_nvfp4(w_g)
    q_u = quant_weight_nvfp4(w_u)
    gate_b = FP4Buffer(M, H, device='cuda')
    up_b = FP4Buffer(M, H, device='cuda')
    chain = FP4ActScratch(M, H, device='cuda')
    assert fvk_fp4.cutlass_fp4_gemm_fp4out(
        act.packed.data_ptr(), act.sfa.data_ptr(),
        q_g['packed'].data_ptr(), q_g['sfb'].data_ptr(),
        gate_b.packed.data_ptr(), gate_b.sfa.data_ptr(), M, H, D, 0) == 0
    assert fvk_fp4.cutlass_fp4_gemm_fp4out(
        act.packed.data_ptr(), act.sfa.data_ptr(),
        q_u['packed'].data_ptr(), q_u['sfb'].data_ptr(),
        up_b.packed.data_ptr(), up_b.sfa.data_ptr(), M, H, D, 0) == 0
    fvk_fp4.geglu_two_fp4_to_fp4(
        gate_b.packed.data_ptr(), gate_b.sfa.data_ptr(),
        up_b.packed.data_ptr(), up_b.sfa.data_ptr(),
        chain.packed.data_ptr(), chain.sfa.data_ptr(), M, H, 0)

    # Fused chain: one GEMM over the pairwise-interleaved weight.
    w_il = torch.empty(N_il, D, dtype=torch.float16, device='cuda')
    w_il[0::2] = w_g
    w_il[1::2] = w_u
    q_il = quant_weight_nvfp4(w_il.contiguous())
    fused = FP4ActScratch(M, H, device='cuda')
    dummy = torch.zeros(M, H, dtype=torch.uint8, device='cuda')
    assert fvk_fp4.cutlass_fp4_gemm_geglu_il_hw(
        act.packed.data_ptr(), act.sfa.data_ptr(),
        q_il['packed'].data_ptr(), q_il['sfb'].data_ptr(),
        dummy.data_ptr(), fused.packed.data_ptr(), fused.sfa.data_ptr(),
        M, N_il, D, 0) == 0
    torch.cuda.synchronize()

    # Compare through the shared down projection so both paths are read the
    # way the pipeline reads them.
    w_d = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    q_d = quant_weight_nvfp4(w_d)
    out_chain = torch.empty(M, D, dtype=torch.float16, device='cuda')
    out_fused = torch.empty_like(out_chain)
    for src, dst in ((chain, out_chain), (fused, out_fused)):
        assert fvk_fp4.cutlass_fp4_gemm_variant(
            1, src.packed.data_ptr(), src.sfa.data_ptr(),
            q_d['packed'].data_ptr(), q_d['sfb'].data_ptr(),
            dst.data_ptr(), M, D, H, 1.0, 0.0, 0) == 0
    torch.cuda.synchronize()

    reference = hidden_ref @ w_d.float().T
    cos_chain = _cos(out_chain, reference)
    cos_fused = _cos(out_fused, reference)
    assert cos_fused >= cos_chain - 5e-4, (
        f"fused epilogue lost accuracy: {cos_fused:.6f} < {cos_chain:.6f}")


def test_vectorized_siglip_layernorms_match_reference():
    """The single-pass LayerNorms must agree with the reference norm +
    quantize pair (reduction order differs at ulp level)."""
    _require_thor()
    torch.manual_seed(20260728)
    S, D, eps = 768, 1152, 1e-5
    x = torch.randn(S, D, dtype=torch.float16, device='cuda') * 1.5
    gamma = (torch.randn(D, dtype=torch.float16, device='cuda') * 0.2 + 1.0)
    beta = torch.randn(D, dtype=torch.float16, device='cuda') * 0.1
    inv_s = (torch.rand(D, dtype=torch.float16, device='cuda') + 0.5)

    ref_fp8 = torch.zeros(S, D, dtype=torch.uint8, device='cuda')
    vec_fp8 = torch.zeros_like(ref_fp8)
    fvk.layer_norm_fp8(x.data_ptr(), ref_fp8.data_ptr(), gamma.data_ptr(),
                       beta.data_ptr(), S, D, eps, 0)
    assert fvk_fp4.layer_norm_fp8_vec_fp16(
        x.data_ptr(), gamma.data_ptr(), beta.data_ptr(),
        vec_fp8.data_ptr(), S, D, eps, 0) == 0
    torch.cuda.synchronize()
    agree = (ref_fp8 == vec_fp8).float().mean().item()
    assert agree > 0.999, f"LN->fp8 byte agreement {agree:.5f}"

    ref = FP4ActScratch(S, D, device='cuda')
    vec = FP4ActScratch(S, D, device='cuda')
    assert fvk_fp4.layer_norm_mul_fp4_sfa_fp16(
        x.data_ptr(), gamma.data_ptr(), beta.data_ptr(), inv_s.data_ptr(),
        ref.packed.data_ptr(), ref.sfa.data_ptr(), S, D, eps, 0) == 0
    assert fvk_fp4.layer_norm_mul_fp4_sfa_vec_fp16(
        x.data_ptr(), gamma.data_ptr(), beta.data_ptr(), inv_s.data_ptr(),
        vec.packed.data_ptr(), vec.sfa.data_ptr(), S, D, eps, 0) == 0
    torch.cuda.synchronize()
    packed_agree = (ref.packed == vec.packed).float().mean().item()
    sfa_agree = (ref.sfa == vec.sfa).float().mean().item()
    assert packed_agree > 0.999, f"LN->fp4 packed agreement {packed_agree:.5f}"
    assert sfa_agree > 0.999, f"LN->fp4 SFA agreement {sfa_agree:.5f}"
