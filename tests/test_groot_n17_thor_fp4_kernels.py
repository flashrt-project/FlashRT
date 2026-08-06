"""Kernel contracts for the GROOT N1.7 Thor NVFP4 tier.

Pins the new kernels against the chains they replace, at the shapes the
N1.7 DiT and backbone actually run (M = 41 action tokens, D = 1536,
FF = 6144; backbone S = 277/1024):

* NVFP4 GEMMs with fused bf16 bias / bias+residual / bias+GELU+FP4-out
  epilogues vs a torch fp32 reference.
* bf16 -> NVFP4 activation quantize (dequant fidelity via an identity
  GEMM).
* Fused DiT norms emitting FP4 vs the two-step norm-then-quantize chain
  (bit-exact contract).
* Vectorized backbone helpers vs their scalar originals (rope /
  quantize / head-expand are bit-exact; the norms match within fp16
  rounding of a different reduction order).
* Masked-softmax MHA vs the pre-filled -inf variant.

Skips cleanly when CUDA or the optional ``flash_rt_fp4`` extension is
unavailable.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")

M, D, FF = 41, 1536, 6144
DEV = "cuda"


@pytest.fixture(scope="module")
def fp4():
    fp4mod = pytest.importorskip(
        "flash_rt.flash_rt_fp4",
        reason="flash_rt_fp4 extension not built")
    if not fp4mod.has_nvfp4():
        pytest.skip("NVFP4 requires an SM100-class GPU")
    return fp4mod


@pytest.fixture(scope="module")
def fvk():
    return pytest.importorskip("flash_rt.flash_rt_kernels")


def _cos(a, b):
    a = a.float().flatten().double()
    b = b.float().flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def _quant_act_bf16(fp4, x):
    S, Dd = x.shape
    packed = torch.empty(S, Dd // 2, dtype=torch.uint8, device=DEV)
    sfa = torch.zeros(fp4.sfa_size_bytes(S, Dd, False), dtype=torch.uint8,
                      device=DEV)
    rc = fp4.quantize_fp4_dynamic_sfa_bf16_vec(
        x.data_ptr(), packed.data_ptr(), sfa.data_ptr(), S, Dd, False, 0)
    assert rc == 0, f"activation quantize rc={rc}"
    return packed, sfa


def _quant_weight(fp4, w_nk_fp16):
    N, K = w_nk_fp16.shape
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=DEV)
    # Scale-factor buffers must be zero-initialized: the tile-interleaved
    # layout pads rows/K to atom boundaries and the quantize kernel only
    # writes real coordinates.
    sfb = torch.zeros(fp4.sfa_size_bytes(N, K, True), dtype=torch.uint8,
                      device=DEV)
    rc = fp4.quantize_fp4_dynamic_sfa_mse_fp16(
        w_nk_fp16.data_ptr(), packed.data_ptr(), sfb.data_ptr(), N, K, True, 0)
    assert rc == 0, f"weight quantize rc={rc}"
    return packed, sfb


@pytest.fixture(scope="module")
def act(fp4):
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=torch.bfloat16, device=DEV) * 2.0
    packed, sfa = _quant_act_bf16(fp4, x)
    return x, packed, sfa


def test_bf16_activation_quantize_roundtrip(fp4, act):
    """An identity-weight NVFP4 GEMM recovers the quantized activation."""
    x, packed, sfa = act
    w = torch.eye(D, dtype=torch.float16, device=DEV)
    wp, wsf = _quant_weight(fp4, w)
    bias = torch.zeros(D, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(M, D, dtype=torch.bfloat16, device=DEV)
    rc = fp4.cutlass_fp4_gemm_bias_bf16(
        packed.data_ptr(), sfa.data_ptr(), wp.data_ptr(), wsf.data_ptr(),
        bias.data_ptr(), out.data_ptr(), M, D, D, 0)
    assert rc == 0, f"gemm rc={rc:#x}"
    torch.cuda.synchronize()
    assert _cos(out, x) >= 0.995


def test_gemm_bias_bf16(fp4, act):
    """Fused per-column bias epilogue matches A @ B^T + bias."""
    x, packed, sfa = act
    N = 3 * D  # the DiT's fused QKV projection
    w = (torch.randn(N, D, dtype=torch.float16, device=DEV) * 0.02).contiguous()
    b = torch.randn(N, dtype=torch.bfloat16, device=DEV) * 0.1
    wp, wsf = _quant_weight(fp4, w)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    rc = fp4.cutlass_fp4_gemm_bias_bf16(
        packed.data_ptr(), sfa.data_ptr(), wp.data_ptr(), wsf.data_ptr(),
        b.data_ptr(), out.data_ptr(), M, N, D, 0)
    assert rc == 0, f"gemm rc={rc:#x}"
    torch.cuda.synchronize()
    ref = x.float() @ w.float().t() + b.float()
    assert _cos(out, ref) >= 0.99


def test_gemm_bias_res_bf16_aliased_output(fp4, act):
    """Fused residual epilogue matches A @ B^T + bias + C with C aliasing D."""
    x, packed, sfa = act
    w = (torch.randn(D, D, dtype=torch.float16, device=DEV) * 0.02).contiguous()
    b = torch.randn(D, dtype=torch.bfloat16, device=DEV) * 0.1
    h = torch.randn(M, D, dtype=torch.bfloat16, device=DEV)
    h_before = h.clone()
    wp, wsf = _quant_weight(fp4, w)
    rc = fp4.cutlass_fp4_gemm_bias_res_bf16(
        packed.data_ptr(), sfa.data_ptr(), wp.data_ptr(), wsf.data_ptr(),
        b.data_ptr(), h.data_ptr(), h.data_ptr(), M, D, D, 0)
    assert rc == 0, f"gemm rc={rc:#x}"
    torch.cuda.synchronize()
    ref = x.float() @ w.float().t() + b.float() + h_before.float()
    assert _cos(h, ref) >= 0.99


def test_ffn_chain_bias_gelu_fp4out(fp4, act):
    """Up GEMM (bias + tanh-GELU + FP4 out) feeding the Down GEMM matches
    the torch chain."""
    x, packed, sfa = act
    w_up = (torch.randn(FF, D, dtype=torch.float16, device=DEV)
            * 0.02).contiguous()
    b_up = torch.randn(FF, dtype=torch.bfloat16, device=DEV) * 0.1
    wp_up, wsf_up = _quant_weight(fp4, w_up)
    hid = torch.empty(M, FF // 2, dtype=torch.uint8, device=DEV)
    hid_sfa = torch.zeros(fp4.sfa_size_bytes(M, FF, False), dtype=torch.uint8,
                          device=DEV)
    rc = fp4.cutlass_fp4_gemm_bias_gelu_fp4out_bf16(
        packed.data_ptr(), sfa.data_ptr(), wp_up.data_ptr(), wsf_up.data_ptr(),
        b_up.data_ptr(), hid.data_ptr(), hid_sfa.data_ptr(), M, FF, D, 0)
    assert rc == 0, f"up gemm rc={rc:#x}"

    w_dn = (torch.randn(D, FF, dtype=torch.float16, device=DEV)
            * 0.02).contiguous()
    b_dn = torch.randn(D, dtype=torch.bfloat16, device=DEV) * 0.1
    h = torch.randn(M, D, dtype=torch.bfloat16, device=DEV)
    h_before = h.clone()
    wp_dn, wsf_dn = _quant_weight(fp4, w_dn)
    rc = fp4.cutlass_fp4_gemm_bias_res_bf16(
        hid.data_ptr(), hid_sfa.data_ptr(), wp_dn.data_ptr(),
        wsf_dn.data_ptr(), b_dn.data_ptr(), h.data_ptr(), h.data_ptr(),
        M, D, FF, 0)
    assert rc == 0, f"down gemm rc={rc:#x}"
    torch.cuda.synchronize()

    gelu = torch.nn.functional.gelu(
        x.float() @ w_up.float().t() + b_up.float(), approximate="tanh")
    ref = gelu @ w_dn.float().t() + b_dn.float() + h_before.float()
    assert _cos(h, ref) >= 0.98


def test_ada_layer_norm_fp4_is_bit_exact(fp4, fvk):
    """Fused AdaLN -> FP4 equals the bf16 AdaLN kernel then quantize."""
    torch.manual_seed(1)
    h = torch.randn(M, D, dtype=torch.bfloat16, device=DEV)
    scale = torch.randn(D, dtype=torch.bfloat16, device=DEV) * 0.3
    shift = torch.randn(D, dtype=torch.bfloat16, device=DEV) * 0.3

    packed = torch.empty(M, D // 2, dtype=torch.uint8, device=DEV)
    sfa = torch.zeros(fp4.sfa_size_bytes(M, D, False), dtype=torch.uint8,
                      device=DEV)
    rc = fp4.ada_layer_norm_fp4_sfa_bf16(
        h.data_ptr(), scale.data_ptr(), shift.data_ptr(),
        packed.data_ptr(), sfa.data_ptr(), M, D, 1e-5, 0)
    assert rc == 0, f"fused adaln rc={rc}"

    xn = torch.empty(M, D, dtype=torch.bfloat16, device=DEV)
    fvk.ada_layer_norm_bf16(h.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                            xn.data_ptr(), M, D, 1e-5, 0)
    ref_packed, ref_sfa = _quant_act_bf16(fp4, xn)
    torch.cuda.synchronize()
    assert torch.equal(packed, ref_packed)
    assert torch.equal(sfa, ref_sfa)


def test_layer_norm_no_affine_fp4_is_bit_exact(fp4, fvk):
    """Fused pre-FFN LayerNorm -> FP4 equals the bf16 kernel then quantize."""
    torch.manual_seed(2)
    h = torch.randn(M, D, dtype=torch.bfloat16, device=DEV)

    packed = torch.empty(M, D // 2, dtype=torch.uint8, device=DEV)
    sfa = torch.zeros(fp4.sfa_size_bytes(M, D, False), dtype=torch.uint8,
                      device=DEV)
    rc = fp4.layer_norm_no_affine_fp4_sfa_bf16(
        h.data_ptr(), packed.data_ptr(), sfa.data_ptr(), M, D, 1e-5, 0)
    assert rc == 0, f"fused ln rc={rc}"

    xn = torch.empty(M, D, dtype=torch.bfloat16, device=DEV)
    fvk.layer_norm_no_affine_bf16(h.data_ptr(), xn.data_ptr(), M, D, 1e-5, 0)
    ref_packed, ref_sfa = _quant_act_bf16(fp4, xn)
    torch.cuda.synchronize()
    assert torch.equal(packed, ref_packed)
    assert torch.equal(sfa, ref_sfa)


# ── Vectorized backbone helpers vs their scalar originals ──────────────

def test_rope_vec_is_bit_exact(fvk):
    torch.manual_seed(3)
    S, NH, HD = 277, 16, 128
    x = torch.randn(S, NH * HD, dtype=torch.float16, device=DEV)
    cos_t = torch.randn(S, HD, dtype=torch.float16, device=DEV)
    sin_t = torch.randn(S, HD, dtype=torch.float16, device=DEV)
    a, b = x.clone(), x.clone()
    fvk.rope_rotate_half_fp16(a.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                              S, NH, HD, 0)
    rc = fvk.rope_rotate_half_fp16_vec(
        b.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(), S, NH, HD, 0)
    assert rc == 0
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def test_quantize_fp8_vec_is_bit_exact(fvk):
    torch.manual_seed(4)
    n = 277 * 2048
    x = torch.randn(n, dtype=torch.float16, device=DEV)
    scale = torch.tensor([x.abs().max().item() / 448.0], dtype=torch.float32,
                         device=DEV)
    a = torch.empty(n, dtype=torch.float8_e4m3fn, device=DEV)
    b = torch.empty(n, dtype=torch.float8_e4m3fn, device=DEV)
    fvk.quantize_fp8_static_fp16(x.data_ptr(), a.data_ptr(), scale.data_ptr(),
                                 n, 0)
    rc = fvk.quantize_fp8_static_fp16_vec(
        x.data_ptr(), b.data_ptr(), scale.data_ptr(), n, 0)
    assert rc == 0
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_repeat_interleave_heads_vec_is_bit_exact(fvk):
    torch.manual_seed(5)
    S, NHKV, HD, GQA = 277, 8, 128, 2
    src = torch.randn(S, NHKV * HD, dtype=torch.float16, device=DEV)
    a = torch.empty(S, NHKV * GQA * HD, dtype=torch.float16, device=DEV)
    b = torch.empty_like(a)
    fvk.gpu_repeat_interleave_heads(src.data_ptr(), a.data_ptr(), S, NHKV, HD,
                                    GQA, 0)
    rc = fvk.gpu_repeat_interleave_heads_vec(
        src.data_ptr(), b.data_ptr(), S, NHKV, HD, GQA, 0)
    assert rc == 0
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@pytest.mark.parametrize("rows,dim", [(277 * 16, 128), (277, 2048)])
def test_rms_norm_vec_matches_scalar(fvk, rows, dim):
    torch.manual_seed(6)
    x = torch.randn(rows, dim, dtype=torch.float16, device=DEV)
    w = torch.randn(dim, dtype=torch.float16, device=DEV)
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    fvk.rms_norm_fp16(x.data_ptr(), w.data_ptr(), a.data_ptr(), rows, dim,
                      1e-6, 0)
    rc = fvk.rms_norm_fp16_vec(x.data_ptr(), w.data_ptr(), b.data_ptr(), rows,
                               dim, 1e-6, 0)
    assert rc == 0
    torch.cuda.synchronize()
    assert _cos(a, b) >= 0.99999


def test_layer_norm_vec_matches_scalar(fvk):
    torch.manual_seed(7)
    rows, dim = 1024, 1024
    x = torch.randn(rows, dim, dtype=torch.float16, device=DEV)
    w = torch.randn(dim, dtype=torch.float16, device=DEV)
    bias = torch.randn(dim, dtype=torch.float16, device=DEV)
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    fvk.layer_norm_fp16(x.data_ptr(), w.data_ptr(), bias.data_ptr(),
                        a.data_ptr(), rows, dim, 1e-6, 0)
    rc = fvk.layer_norm_fp16_vec(x.data_ptr(), w.data_ptr(), bias.data_ptr(),
                                 b.data_ptr(), rows, dim, 1e-6, 0)
    assert rc == 0
    torch.cuda.synchronize()
    assert _cos(a, b) >= 0.99999


@pytest.mark.parametrize("S_kv", [1024, 1025, 2048])
def test_masked_softmax_mha_wide_rows(fvk, S_kv):
    """Rows wider than the register-tiled path's reach still softmax
    correctly — the boundary (1024) and the first row past it (1025) are
    pinned explicitly, in both dtypes."""
    torch.manual_seed(9)
    S_q, NH, HD = 1, 1, 16
    S_pad = ((S_kv + 7) // 8) * 8
    ctx = fvk.FvkContext()

    for dtype, masked, plain, fill in (
        (torch.float16, fvk.attention_mha_fp16_masked,
         fvk.attention_mha_fp16, fvk.gpu_fill_neginf_fp16),
        (torch.bfloat16, fvk.attention_mha_bf16_masked,
         fvk.attention_mha_bf16, fvk.gpu_fill_neginf_bf16),
    ):
        q = torch.randn(S_q, NH * HD, dtype=dtype, device=DEV)
        k = torch.randn(S_kv, NH * HD, dtype=dtype, device=DEV)
        v = torch.randn(S_kv, NH * HD, dtype=dtype, device=DEV)
        logits = torch.empty(NH, S_q, S_pad, dtype=dtype, device=DEV)
        out_ref = torch.empty(S_q, NH * HD, dtype=dtype, device=DEV)
        out_new = torch.empty_like(out_ref)
        scale = 1.0 / (HD ** 0.5)

        fill(logits.data_ptr(), NH * S_q * S_pad, 0)
        if dtype is torch.float16:
            plain(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                  logits.data_ptr(), out_ref.data_ptr(),
                  S_q, S_kv, NH, HD, scale, 0)
        else:
            plain(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                  logits.data_ptr(), out_ref.data_ptr(),
                  S_q, S_kv, NH, HD, scale, S_pad, 0)
        logits.fill_(float("nan"))
        if dtype is torch.float16:
            masked(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                   logits.data_ptr(), out_new.data_ptr(),
                   S_q, S_kv, NH, HD, scale, 0)
        else:
            masked(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                   logits.data_ptr(), out_new.data_ptr(),
                   S_q, S_kv, NH, HD, scale, S_pad, 0, 0)
        torch.cuda.synchronize()

        # Reference the math directly too: the pre-filled kernel shares the
        # 1024-column ceiling, so it cannot arbitrate past it on its own.
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.view(S_q, NH, HD).permute(1, 0, 2).float(),
            k.view(S_kv, NH, HD).permute(1, 0, 2).float(),
            v.view(S_kv, NH, HD).permute(1, 0, 2).float(),
        ).permute(1, 0, 2).reshape(S_q, NH * HD)
        assert torch.isfinite(out_new.float()).all(), (
            f"{dtype} S_kv={S_kv}: masked softmax produced non-finite output")
        cos = _cos(out_new, ref)
        max_err = float((out_new.float() - ref).abs().max())
        assert cos >= 0.999, (
            f"{dtype} S_kv={S_kv}: cosine {cos:.6f} vs torch SDPA "
            f"(max abs error {max_err:.6f})")


def test_masked_softmax_mha_matches_prefilled(fvk):
    """The masked-softmax MHA needs no -inf logits pre-fill and matches the
    pre-filled variant."""
    torch.manual_seed(8)
    S, NH, HD = 41, 32, 48
    S_pad = ((S + 7) // 8) * 8
    ctx = fvk.FvkContext()
    q = torch.randn(S, NH * HD, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(S, NH * HD, dtype=torch.bfloat16, device=DEV)
    v = torch.randn(S, NH * HD, dtype=torch.bfloat16, device=DEV)
    logits = torch.empty(NH, S, S_pad, dtype=torch.bfloat16, device=DEV)
    out_ref = torch.empty(S, NH * HD, dtype=torch.bfloat16, device=DEV)
    out_new = torch.empty_like(out_ref)
    scale = 1.0 / (HD ** 0.5)

    fvk.gpu_fill_neginf_bf16(logits.data_ptr(), NH * S * S_pad, 0)
    fvk.attention_mha_bf16(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                           logits.data_ptr(), out_ref.data_ptr(), S, S, NH, HD,
                           scale, S_pad, 0)
    # Poison the scratch so a missing mask would show up as garbage.
    logits.fill_(float("nan"))
    fvk.attention_mha_bf16_masked(ctx, q.data_ptr(), k.data_ptr(),
                                  v.data_ptr(), logits.data_ptr(),
                                  out_new.data_ptr(), S, S, NH, HD, scale,
                                  S_pad, 0, 0)
    torch.cuda.synchronize()
    assert torch.isfinite(out_new.float()).all()
    assert _cos(out_new, out_ref) >= 0.9999
