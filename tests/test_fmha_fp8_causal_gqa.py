"""Numerical check for the SM120 FP8 causal GQA prefill attention.

`fmha_fp8_causal_gqa_nhd_d128` serves the Qwen3 prefill backend and (with the
64x64 entries) the MolmoAct2 text stage. The kernel consumes raw e4m3 Q/K/V
(NHD layout) and writes bf16 O. This test dequantizes the same bytes, runs a
plain float causal softmax reference, and requires the bf16 output to match at
bf16-rounding scale.

Skips cleanly on builds without the SM120 FP8 prefill object
(FLASHRT_ENABLE_QWEN3_FP8_PREFILL=OFF) or without a CUDA device.

Run:
    PYTHONPATH=. python -m pytest tests/test_fmha_fp8_causal_gqa.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


def _kernel_or_skip():
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:                  # pragma: no cover
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    if not hasattr(fvk, "fmha_fp8_causal_gqa_nhd_d128"):
        pytest.skip("SM120 FP8 prefill attention not built "
                    "(FLASHRT_ENABLE_QWEN3_FP8_PREFILL=OFF)")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return fvk


def _reference(q, k, v, scale):
    qf, kf, vf = q.float(), k.float(), v.float()
    rep = q.shape[1] // k.shape[1]
    scores = torch.einsum("qhd,khd->hqk", qf,
                          kf.repeat_interleave(rep, dim=1)) * scale
    causal = torch.ones(q.shape[0], k.shape[0], device=q.device,
                        dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", p,
                        vf.repeat_interleave(rep, dim=1))


@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (16, 8)])
@pytest.mark.parametrize("Lq", [128, 256])
def test_matches_float_causal_reference(Lq, Hq, Hkv):
    fvk = _kernel_or_skip()
    hd = 128
    scale = 1.0 / math.sqrt(hd)
    torch.manual_seed(0)
    q = (torch.randn(Lq, Hq, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    out = torch.empty(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)

    rc = fvk.fmha_fp8_causal_gqa_nhd_d128(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, torch.cuda.current_stream().cuda_stream)
    assert rc == 0
    torch.cuda.synchronize()

    ref = _reference(q, k, v, scale)
    a, b = out.float().flatten(), ref.flatten()
    cos = float(a @ b / (a.norm() * b.norm()))
    assert cos > 0.9999, cos
    # The kernel requantizes P to e4m3 and rounds O to bf16, so the output
    # tracks the float reference at rounding scale (~1e-2 absolute here).
    max_abs = float((a - b).abs().max())
    assert max_abs < 0.25, max_abs


@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (16, 8)])
def test_fp4out_matches_standalone_quantizer(Hq, Hkv):
    fvk = _kernel_or_skip()
    if not hasattr(fvk, "fmha_fp8_causal_gqa_nhd_d128_fp4out"):
        pytest.skip("fp4out FMHA entry not built")
    if not hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled"):
        pytest.skip("NVFP4 quantizer not built")
    Lq, hd = 256, 128
    width = Hq * hd
    scale = 1.0 / math.sqrt(hd)
    torch.manual_seed(0)
    q = (torch.randn(Lq, Hq, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    out_bf16 = torch.empty(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream

    rc = fvk.fmha_fp8_causal_gqa_nhd_d128(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out_bf16.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, stream)
    assert rc == 0

    sf_bytes = fvk.nvfp4_sf_swizzled_bytes(Lq, width)
    fp4_ref = torch.empty(Lq * width // 2, device="cuda", dtype=torch.uint8)
    sf_ref = torch.zeros(sf_bytes, device="cuda", dtype=torch.uint8)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        out_bf16.data_ptr(), fp4_ref.data_ptr(), sf_ref.data_ptr(),
        Lq, width, stream)

    fp4_out = torch.empty_like(fp4_ref)
    sf_out = torch.zeros_like(sf_ref)
    rc = fvk.fmha_fp8_causal_gqa_nhd_d128_fp4out(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        fp4_out.data_ptr(), sf_out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, stream)
    assert rc == 0
    torch.cuda.synchronize()

    assert torch.equal(fp4_out, fp4_ref), "packed FP4 operand differs"
    assert torch.equal(sf_out, sf_ref), "swizzled SF differs"


def test_vla_q64_16_8_matches_float_reference():
    """Additive 16/8 64-row entry: same math as the 128-row 16/8 entry."""
    fvk = _kernel_or_skip()
    if not hasattr(fvk, "fmha_fp8_causal_gqa_q64_nhd_d128_16_8"):
        pytest.skip("VLA 16/8 q64 entry not built "
                    "(FLASHRT_ENABLE_VLA_JEPA=OFF)")
    Lq, Hq, Hkv, hd = 256, 16, 8, 128
    scale = 1.0 / math.sqrt(hd)
    torch.manual_seed(0)
    q = (torch.randn(Lq, Hq, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    out = torch.empty(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    rc = fvk.fmha_fp8_causal_gqa_q64_nhd_d128_16_8(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, torch.cuda.current_stream().cuda_stream)
    assert rc == 0
    torch.cuda.synchronize()
    ref = _reference(q, k, v, scale)
    a, b = out.float().flatten(), ref.flatten()
    cos = float(a @ b / (a.norm() * b.norm()))
    assert cos > 0.9999, cos


def test_vla_q64_16_8_fp4out_matches_standalone_quantizer():
    fvk = _kernel_or_skip()
    if not hasattr(fvk, "fmha_fp8_causal_gqa_q64_nhd_d128_fp4out_16_8"):
        pytest.skip("VLA 16/8 q64 fp4out entry not built")
    if not hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled"):
        pytest.skip("NVFP4 quantizer not built")
    Lq, Hq, Hkv, hd = 256, 16, 8, 128
    width = Hq * hd
    scale = 1.0 / math.sqrt(hd)
    torch.manual_seed(0)
    q = (torch.randn(Lq, Hq, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    out = torch.empty(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    assert fvk.fmha_fp8_causal_gqa_q64_nhd_d128_16_8(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, stream) == 0
    sf_bytes = fvk.nvfp4_sf_swizzled_bytes(Lq, width)
    fp4_ref = torch.empty(Lq * width // 2, device="cuda", dtype=torch.uint8)
    sf_ref = torch.zeros(sf_bytes, device="cuda", dtype=torch.uint8)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        out.data_ptr(), fp4_ref.data_ptr(), sf_ref.data_ptr(),
        Lq, width, stream)
    fp4_out = torch.empty_like(fp4_ref)
    sf_out = torch.zeros_like(sf_ref)
    assert fvk.fmha_fp8_causal_gqa_q64_nhd_d128_fp4out_16_8(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        fp4_out.data_ptr(), sf_out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, stream) == 0
    torch.cuda.synchronize()
    assert torch.equal(fp4_out, fp4_ref), "packed FP4 operand differs"
    assert torch.equal(sf_out, sf_ref), "swizzled SF differs"


def test_vla_target_q64_fp4out_matches_standalone_quantizer():
    """Target-ported 16/8 q64 fp4out entry must equal the q64 bf16 output
    followed by the standalone NVFP4 quantizer."""
    fvk = _kernel_or_skip()
    if not hasattr(fvk, "fmha_fp8_causal_gqa_q64_fp4out_vla"):
        pytest.skip("target-ported VLA q64 entry not built")
    if not hasattr(fvk, "fmha_fp8_causal_gqa_nhd_d128"):
        pytest.skip("128-row bf16 entry not built")
    if not hasattr(fvk, "quantize_bf16_to_nvfp4_swizzled"):
        pytest.skip("NVFP4 quantizer not built")
    Lq, Hq, Hkv, hd = 256, 16, 8, 128
    width = Hq * hd
    scale = 1.0 / math.sqrt(hd)
    torch.manual_seed(0)
    q = (torch.randn(Lq, Hq, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    v = (torch.randn(Lq, Hkv, hd, device="cuda") * 2).to(torch.float8_e4m3fn)
    out = torch.empty(Lq, Hq, hd, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    # The target q64 kernel is the bit-identical 64-row schedule of the
    # 128-row entry, so the reference is the 128-row bf16 output.
    assert fvk.fmha_fp8_causal_gqa_nhd_d128(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, stream) == 0
    sf_bytes = fvk.nvfp4_sf_swizzled_bytes(Lq, width)
    fp4_ref = torch.empty(Lq * width // 2, device="cuda", dtype=torch.uint8)
    sf_ref = torch.zeros(sf_bytes, device="cuda", dtype=torch.uint8)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        out.data_ptr(), fp4_ref.data_ptr(), sf_ref.data_ptr(),
        Lq, width, stream)
    fp4_out = torch.empty_like(fp4_ref)
    sf_out = torch.zeros_like(sf_ref)
    assert fvk.fmha_fp8_causal_gqa_q64_fp4out_vla(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        fp4_out.data_ptr(), sf_out.data_ptr(),
        Lq, Lq, Hq, Hkv, scale, stream) == 0
    torch.cuda.synchronize()
    assert torch.equal(fp4_out, fp4_ref), "target q64 packed FP4 differs"
    assert torch.equal(sf_out, sf_ref), "target q64 swizzled SF differs"
