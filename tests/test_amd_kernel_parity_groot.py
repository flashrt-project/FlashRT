"""AMD GROOT N1.7 kernel port — per-kernel numerical parity gates.

Companion to ``tests/test_amd_kernel_parity.py`` (which owns the pi05
kernel surface and the whole ``GemmRunner`` / hipBLASLt GEMM surface).
This module owns the two families the GROOT N1.7 branch adds on top:

    csrc/amd/gemm/bindings_smallm_bf16.inc   4 entry points
        the hand-written packed-weight MFMA small-M BF16 GEMM
        (smallm_mfma_bf16_nn_bias / _bias_gelu / _bias_res) and its
        variant enumerator.

    csrc/amd/kernels/bindings_fp16_port.inc  29 entry points
        the FP16/BF16 backbone port: elementwise, activation, casts,
        layout ops, norms, RoPE, the fused norm->FP8 pair, the seven
        ``*_vec`` fast paths and the AdaLN (LayerNorm-style) family.

Every bound name in those two files gets at least one gate here; the
inventory is asserted mechanically by ``test_entry_point_inventory``
so a name added to the .inc without a gate fails this file.

What the gates protect (why they are not loose cosines):

  * FP8 outputs are compared BYTE-for-byte against a torch
    ``float8_e4m3fn`` cast of the same fp32 reference through the SAME
    device scale, replicating the kernel's exact arithmetic (multiply by
    the fp32 reciprocal of the scale, clamp to +-448, RNE convert).
    Comparing an FP8 output against an unquantized fp32 reference would
    measure the e4m3 format (cos ~0.9996 on randn) instead of the
    kernel, so it would pass with a broken scale path.
  * Residual / accumulate variants run against a PRE-SEEDED destination
    and additionally assert the result is NOT the non-accumulated form,
    so a dropped ``+=`` cannot pass.
  * GELU references pin the tanh approximation (``approximate="tanh"``),
    the hipBLASLt GELU_BIAS / activation.hip semantics; erf-GELU would
    sail through a loose cosine on symmetric data.
  * Pure-copy / cast / layout kernels are gated BITWISE, not by
    tolerance — they have no arithmetic to round.
  * The packed GEMM is fed through the documented torch view/permute
    one-liner from ``csrc/amd/gemm/smallm_mfma_bf16.h``, and a companion
    gate feeds the SAME weight unpacked and requires the answer to be
    wrong — proving the pack layout is load-bearing rather than
    incidental.
  * The ``*_vec`` entries return ``int``: 0 when their alignment /
    divisibility preconditions hold, -1 when they do not. Both branches
    are gated, and the reject branch also asserts the destination was
    left untouched (a silent partial write is the dangerous failure).

Numeric thresholds, justified once (see the ``_TOL`` block below).

Binding signatures were read from ``csrc/amd/kernels/bindings_fp16_port.inc``
and ``csrc/amd/gemm/bindings_smallm_bf16.inc`` before writing any call
here; every tensor is bound to a named variable before ``.data_ptr()``
(an inline temporary would be GC'd before the launch reads it).

Skip conditions: ROCm torch + a visible device + the built extension.
Every AMD-tree import happens inside a fixture, so this module collects
cleanly on a CUDA-only box and every test skips with a reason.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ext():
    """The compiled AMD module, or a skip with the reason.

    Importing does not need a GPU, so the inventory test can run on any
    box that has the .so built.
    """
    try:
        return importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")


@pytest.fixture(scope="module")
def env(ext):
    """(torch, extension) on a ROCm box with a visible device."""
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None):
        pytest.skip("torch is not a ROCm build")
    if not torch.cuda.is_available():
        pytest.skip("no ROCm device visible to torch")
    return torch, ext


def _stream(torch) -> int:
    """Launch on torch's current stream so torch-side tensor prep and the
    raw-pointer launches are ordered without extra events."""
    return torch.cuda.current_stream().cuda_stream


def _cos(torch, a, b) -> float:
    af = a.detach().double().reshape(-1)
    bf = b.detach().double().reshape(-1)
    denom = af.norm() * bf.norm()
    if float(denom) == 0.0:
        return 1.0 if float((af - bf).abs().max()) == 0.0 else 0.0
    return float((af @ bf) / denom)


# ---------------------------------------------------------------------------
# Numeric gates, justified once
# ---------------------------------------------------------------------------
#
#  COS_TIGHT 0.9999  — kernel and reference share the same fp32 math and
#      differ only in reduction order plus the final dtype round; a real
#      semantic bug (wrong weight fold, wrong activation, shifted row)
#      lands orders of magnitude below this.
#  ATOL_FP16 1e-3    — one fp16 ULP at unit scale is 2^-11 ~ 4.9e-4;
#      1e-3 covers two chained roundings and still fails on O(1) errors.
#  ATOL_FP16_ACT 2e-3 — activations (GELU/SiLU/RoPE) evaluate a
#      transcendental on device vs torch; allow one extra ULP of spread.
#  ATOL_FP16_NORM 5e-3 — norms carry a block reduction over 2048 terms
#      whose wave64 order differs from torch's; the spread is a few fp16
#      ULPs of the normalized value.
#  ATOL_BF16 2e-2    — one bf16 ULP at unit scale is 2^-8 ~ 3.9e-3.
#  FP8_MISMATCH_RNE 1e-3 — for FP8 kernels whose pre-quantization value
#      comes out of a device transcendental (expf / tanhf), a last-ULP
#      fp32 difference can flip an RNE tie and change one byte. 0.1% of
#      elements is far above the observed rate and far below anything a
#      scale-path bug produces (those miss on ~100% of bytes).
#  FP8_MISMATCH_REDUCE 5e-3 — same idea for the norm-fused FP8 kernels,
#      which additionally differ in block-reduction order.
#  Pure elementwise FP8 (multiply-clamp-convert, no reduction, no
#      transcendental) is gated at ZERO mismatched bytes.
COS_TIGHT = 0.9999
COS_FP8_DEQ = 0.99999
ATOL_FP16 = 1e-3
ATOL_FP16_ACT = 2e-3
ATOL_FP16_NORM = 5e-3
ATOL_BF16 = 2e-2
FP8_MISMATCH_RNE = 1e-3
FP8_MISMATCH_REDUCE = 5e-3

E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# FP8 helpers
# ---------------------------------------------------------------------------


def _static_scale(torch, ref_f32):
    """Per-tensor symmetric static scale (device fp32 scalar), computed the
    way production calibration does: amax / 448."""
    amax = ref_f32.detach().float().abs().max().clamp(min=1e-8)
    return (amax / E4M3_MAX).float().reshape(1).contiguous()


def _fp8_ref_bytes(torch, ref_f32, scale):
    """torch float8_e4m3fn cast of ``ref_f32`` through the SAME device
    scale, replicating the kernel arithmetic exactly: the kernels compute
    ``inv = 1.0f / (*d_scale)`` once and then ``value * inv``, clamp to
    +-448 and RNE-convert. ``torch.reciprocal`` is the same fp32 op."""
    inv = torch.reciprocal(scale)
    q = (ref_f32.float() * inv).clamp(-E4M3_MAX, E4M3_MAX)
    return q.to(torch.float8_e4m3fn).view(torch.uint8)


def _assert_fp8(torch, name, out_u8, ref_f32, scale, max_mismatch):
    """Byte gate + a dequantized-magnitude cross-check.

    The byte comparison is the contract (the encoding must match torch's
    e4m3 with the same scale). The dequantized cosine is a second, coarser
    net: it catches wholesale corruption (wrong row stride, wrong operand)
    that a permissive mismatch budget might otherwise absorb.
    """
    ref_u8 = _fp8_ref_bytes(torch, ref_f32, scale).reshape(out_u8.shape)
    mismatch = float((out_u8 != ref_u8).float().mean())
    assert mismatch <= max_mismatch, (
        f"{name}: {mismatch:.5f} of FP8 bytes differ from torch e4m3 with "
        f"the same scale (budget {max_mismatch}) — encoding or scale-path "
        "drift silently poisons every downstream FP8 GEMM")
    got = out_u8.view(torch.float8_e4m3fn).float()
    ref = ref_u8.view(torch.float8_e4m3fn).float()
    assert _cos(torch, got, ref) > COS_FP8_DEQ, f"{name}: dequantized cos"


def _gelu_tanh(torch, x_f32):
    return torch.nn.functional.gelu(x_f32, approximate="tanh")


# ---------------------------------------------------------------------------
# Entry-point inventory
# ---------------------------------------------------------------------------

# csrc/amd/kernels/bindings_fp16_port.inc — every m.def in the file.
FP16_PORT_ENTRY_POINTS = [
    # elementwise / activation
    "add_bias_fp16", "gelu_inplace_fp16", "silu_inplace_fp16",
    "relu_inplace_bf16", "mul_fp16", "residual_add_fp16",
    "gpu_fill_neginf_fp16", "gpu_strided_copy_fp16",
    "gpu_repeat_interleave_heads", "cast_fp16_to_bf16", "cast_bf16_to_fp16",
    "concat2_bf16",
    # quantize / FP8 fusion
    "quantize_fp8_static_fp16", "silu_mul_split_fp8_fp16",
    # norm / RoPE / norm->FP8
    "layer_norm_fp16", "rope_rotate_half_fp16", "rms_norm_fp8_fp16",
    "residual_add_rms_norm_fp8_fp16",
    # vectorized fast paths (int status returns)
    "rms_norm_fp16_vec", "layer_norm_fp16_vec",
    "layer_norm_fp8_static_fp16_vec", "rope_rotate_half_fp16_vec",
    "quantize_fp8_static_fp16_vec", "residual_add_fp16_vec",
    "gpu_repeat_interleave_heads_vec",
    # AdaLN family
    "layer_norm_no_affine_bf16", "ada_layer_norm_bf16", "ada_layer_norm_fp8",
    "bias_gelu_quantize_fp8_static_bf16",
]

# csrc/amd/gemm/bindings_smallm_bf16.inc — every m.def in the file.
SMALLM_BF16_ENTRY_POINTS = [
    "smallm_mfma_bf16_nn_bias",
    "smallm_mfma_bf16_nn_bias_gelu",
    "smallm_mfma_bf16_nn_bias_res",
    "smallm_mfma_bf16_variants",
]


def test_entry_point_inventory(ext):
    """Every name bound by the two .inc files this module gates must exist
    on the compiled module.

    This is the fail-fast half of the contract: the numerical tests below
    only fire on a ROCm box, but a name dropped from (or renamed in) the
    bindings must fail anywhere the .so imports. It is also the tripwire
    that keeps this file honest — a NEW m.def added to either .inc without
    a gate here has to be added to these lists in the same commit.
    """
    expected = FP16_PORT_ENTRY_POINTS + SMALLM_BF16_ENTRY_POINTS
    missing = [n for n in expected if not hasattr(ext, n)]
    assert not missing, (
        "flash_rt_amd_kernels is missing bound entry points from "
        "bindings_fp16_port.inc / bindings_smallm_bf16.inc "
        f"(bindings drift or partial build): {missing}")
    not_callable = [n for n in expected if not callable(getattr(ext, n))]
    assert not not_callable, f"bound but not callable: {not_callable}"


# ---------------------------------------------------------------------------
# FP16 elementwise / activation
# ---------------------------------------------------------------------------


def test_add_bias_fp16_broadcasts_along_the_row(env):
    """``x[i] += b[i % D]`` in place: the bias is a length-D vector
    broadcast over rows. A transposed broadcast (per-row instead of
    per-column) is the classic bug here, so the reference uses a bias whose
    every element differs and the gate is elementwise, not aggregate."""
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(0)
    x = torch.randn(S, D, device="cuda").half()
    b = (0.1 * torch.randn(D, device="cuda")).half()
    ref = (x.float() + b.float()).half()

    ext.add_bias_fp16(x.data_ptr(), b.data_ptr(), S, D, _stream(torch))
    torch.cuda.synchronize()

    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_FP16, rtol=0.0)


def test_gelu_inplace_fp16_is_tanh_approx(env):
    """Must be the tanh approximation, not erf-GELU and not SiLU. The
    tolerance is tight enough that SiLU (which differs by ~0.02 near
    x = -1) fails, and the explicit not-relu / not-identity checks pin the
    shape of the activation independently of the reference."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(1)
    x = torch.randn(n, device="cuda").half()
    x0 = x.clone()
    ref = _gelu_tanh(torch, x.float()).half()

    ext.gelu_inplace_fp16(x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(torch, x, ref) > COS_TIGHT
    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_FP16_ACT, rtol=0.0)
    neg = x0.float() < -1.0
    assert float(x.float()[neg].min()) < -1e-3, (
        "GELU must pass a small negative through; ReLU would clamp to 0")
    assert not torch.allclose(x.float(), x0.float(), atol=1e-2), "identity"


def test_silu_inplace_fp16_matches_torch(env):
    """SiLU x*sigmoid(x), in place. Same not-GELU discriminator in reverse:
    the tolerance is ~1/10th of the peak GELU-vs-SiLU gap."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(2)
    x = torch.randn(n, device="cuda").half()
    ref = torch.nn.functional.silu(x.float()).half()

    ext.silu_inplace_fp16(x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(torch, x, ref) > COS_TIGHT
    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_FP16_ACT, rtol=0.0)


def test_relu_inplace_bf16_is_bit_exact(env):
    """ReLU is a select, not arithmetic: max(x, 0) on bf16 rounds nothing,
    so the gate is bitwise equality. Anything else means the kernel is
    doing math it should not (e.g. computing in fp32 and re-rounding)."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(3)
    x = torch.randn(n, device="cuda").bfloat16()
    ref = torch.nn.functional.relu(x.float()).bfloat16()

    ext.relu_inplace_bf16(x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(x.view(torch.uint16), ref.view(torch.uint16)), (
        "relu_inplace_bf16 is not bit-exact vs max(x, 0)")


def test_mul_fp16_covers_the_scalar_tail(env):
    """Elementwise a*b into a separate output. n is deliberately NOT a
    multiple of 8 so the packed vector body AND the scalar tail both run —
    a tail bug leaves the last few elements stale, which an n%8==0 test
    would never see."""
    torch, ext = env
    n = 64 * 2048 + 5
    torch.manual_seed(4)
    a = torch.randn(n, device="cuda").half()
    b = torch.randn(n, device="cuda").half()
    out = torch.full((n,), float("nan"), device="cuda", dtype=torch.half)
    ref = (a.float() * b.float()).half()

    ext.mul_fp16(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert torch.isfinite(out.float()).all(), "tail elements left unwritten"
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_FP16, rtol=0.0)


def test_residual_add_fp16_accumulates_in_place(env):
    """``residual += x``, in place on the residual buffer. Gated against a
    pre-seeded destination and additionally required NOT to equal x — a
    kernel that overwrote instead of accumulating would pass a
    cos-vs-anything gate on correlated data."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(5)
    residual = torch.randn(n, device="cuda").half()
    seed = residual.clone()
    x = torch.randn(n, device="cuda").half()
    ref = (seed.float() + x.float()).half()

    ext.residual_add_fp16(residual.data_ptr(), x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    torch.testing.assert_close(residual.float(), ref.float(),
                               atol=ATOL_FP16, rtol=0.0)
    assert not torch.allclose(residual.float(), x.float(), atol=1e-2), (
        "residual_add overwrote the destination instead of accumulating")


def test_gpu_fill_neginf_fp16_saturates_to_minus_inf(env):
    """The kernel writes ``__float2half(-1e30f)``, which saturates to fp16
    -inf. Attention masks depend on that saturation (a finite -65504 would
    still leak probability mass after a large positive score), so the gate
    asserts -inf exactly, on every element."""
    torch, ext = env
    n = 64 * 2048
    dst = torch.zeros(n, device="cuda", dtype=torch.half)

    ext.gpu_fill_neginf_fp16(dst.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert bool(torch.isinf(dst).all()) and bool((dst < 0).all()), (
        "gpu_fill_neginf_fp16 must saturate to fp16 -inf on every element")


def test_gpu_strided_copy_fp16_is_bit_exact_slice(env):
    """``dst[r, :] = src[r, off : off + dst_cols]`` — a pure copy out of a
    wider row stride (the fused-QKV column slice). Bitwise gate: a copy
    that rounds is a copy that is doing something wrong. The offset and
    stride are deliberately unequal to dst_cols so a stride/offset swap
    cannot alias into a passing result."""
    torch, ext = env
    rows, dst_cols, src_stride, off = 64, 512, 1536, 512
    torch.manual_seed(6)
    src = torch.randn(rows, src_stride, device="cuda").half()
    dst = torch.full((rows, dst_cols), float("nan"),
                     device="cuda", dtype=torch.half)
    ref = src[:, off:off + dst_cols].contiguous()

    ext.gpu_strided_copy_fp16(src.data_ptr(), dst.data_ptr(),
                              rows, dst_cols, src_stride, off, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(dst.view(torch.uint16), ref.view(torch.uint16))


def test_gpu_repeat_interleave_heads_is_bit_exact(env):
    """GQA head expansion: each of NH_src heads is repeated ``repeat``
    times CONSECUTIVELY (repeat_interleave), not tiled (repeat). The two
    differ only in ordering, so the gate is bitwise against
    ``torch.repeat_interleave`` and a tile would fail loudly."""
    torch, ext = env
    S, NH_src, HD, repeat = 32, 8, 128, 2
    torch.manual_seed(7)
    src = torch.randn(S, NH_src, HD, device="cuda").half()
    dst = torch.full((S, NH_src * repeat, HD), float("nan"),
                     device="cuda", dtype=torch.half)
    ref = torch.repeat_interleave(src, repeat, dim=1)

    ext.gpu_repeat_interleave_heads(src.data_ptr(), dst.data_ptr(),
                                    S, NH_src, HD, repeat, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(dst.view(torch.uint16), ref.contiguous().view(torch.uint16))


def test_cast_fp16_to_bf16_is_bit_exact(env):
    """fp16 -> bf16 is a narrowing round; the gate is bitwise against
    torch's own round so a truncating cast (drop the low mantissa bits
    instead of RNE) fails on ~half the elements."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(8)
    src = torch.randn(n, device="cuda").half()
    out = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    ref = src.float().bfloat16()

    ext.cast_fp16_to_bf16(src.data_ptr(), out.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(out.view(torch.uint16), ref.view(torch.uint16))


def test_cast_bf16_to_fp16_is_bit_exact(env):
    """bf16 -> fp16 is exact for every finite bf16 in fp16 range (bf16 has
    fewer mantissa bits), so the gate is bitwise."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(9)
    src = torch.randn(n, device="cuda").bfloat16()
    out = torch.empty(n, device="cuda", dtype=torch.half)
    ref = src.float().half()

    ext.cast_bf16_to_fp16(src.data_ptr(), out.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(out.view(torch.uint16), ref.view(torch.uint16))


def test_concat2_bf16_is_bit_exact(env):
    """Row-wise concat of two (rows, cols_*) bf16 blocks. cols_a != cols_b
    so a swapped-width bug cannot produce the right total layout; bitwise
    gate because concat is a copy."""
    torch, ext = env
    rows, cols_a, cols_b = 64, 1536, 512
    torch.manual_seed(10)
    a = torch.randn(rows, cols_a, device="cuda").bfloat16()
    b = torch.randn(rows, cols_b, device="cuda").bfloat16()
    out = torch.empty(rows, cols_a + cols_b, device="cuda",
                      dtype=torch.bfloat16)
    ref = torch.cat([a, b], dim=1)

    ext.concat2_bf16(a.data_ptr(), b.data_ptr(), out.data_ptr(),
                     rows, cols_a, cols_b, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(out.view(torch.uint16), ref.contiguous().view(torch.uint16))


# ---------------------------------------------------------------------------
# FP16 quantize / FP8 fusion
# ---------------------------------------------------------------------------


def test_quantize_fp8_static_fp16_is_byte_exact(env):
    """Pure multiply-by-reciprocal, clamp, RNE-convert — no reduction and
    no transcendental, so the kernel's arithmetic is exactly reproducible
    in torch and the budget is ZERO mismatched bytes."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(11)
    x = (2.0 * torch.randn(n, device="cuda")).half()
    scale = _static_scale(torch, x.float()).cuda()
    out = torch.empty(n, device="cuda", dtype=torch.uint8)

    ext.quantize_fp8_static_fp16(x.data_ptr(), out.data_ptr(),
                                 scale.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "quantize_fp8_static_fp16", out, x.float(), scale,
                max_mismatch=0.0)


def test_quantize_fp8_static_fp16_vec_matches_the_scalar_path(env):
    """The vectorized entry must be a pure speed variant: same bytes as the
    scalar kernel on the same input, not merely 'close'. Gated bitwise
    against the scalar path AND against the torch reference, plus rc == 0
    for a geometry that satisfies n%16 and 16-byte alignment."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(11)
    x = (2.0 * torch.randn(n, device="cuda")).half()
    scale = _static_scale(torch, x.float()).cuda()
    out_scalar = torch.empty(n, device="cuda", dtype=torch.uint8)
    out_vec = torch.empty(n, device="cuda", dtype=torch.uint8)

    ext.quantize_fp8_static_fp16(x.data_ptr(), out_scalar.data_ptr(),
                                 scale.data_ptr(), n, _stream(torch))
    rc = ext.quantize_fp8_static_fp16_vec(x.data_ptr(), out_vec.data_ptr(),
                                          scale.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected an aligned n={n} (rc={rc})"
    assert torch.equal(out_vec, out_scalar), (
        "quantize_fp8_static_fp16_vec differs from the scalar path")
    _assert_fp8(torch, "quantize_fp8_static_fp16_vec", out_vec, x.float(),
                scale, max_mismatch=0.0)


def test_silu_mul_split_fp8_fp16_fuses_silu_mul_and_quantize(env):
    """``out_fp8 = quant(silu(gate) * up)``. The reference uses the
    kernel's own SiLU form ``g / (1 + exp(-g))`` (not torch's
    ``g * sigmoid(g)``) so the only remaining difference is one ULP of
    ``expf``; the byte budget is sized for that RNE-tie flip. The
    discriminator gates check the multiply is really by ``up`` and the
    activation is really on ``gate``, which a cos-only gate would blur."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(12)
    gate = torch.randn(n, device="cuda").half()
    up = torch.randn(n, device="cuda").half()
    g = gate.float()
    ref_f32 = (g / (1.0 + torch.exp(-g))) * up.float()
    scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(n, device="cuda", dtype=torch.uint8)

    ext.silu_mul_split_fp8_fp16(gate.data_ptr(), up.data_ptr(),
                                out.data_ptr(), n, scale.data_ptr(),
                                _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "silu_mul_split_fp8_fp16", out, ref_f32, scale,
                max_mismatch=FP8_MISMATCH_RNE)
    got = out.view(torch.float8_e4m3fn).float() * scale
    # Operand-order discriminators: silu(up)*gate and gate*up (no
    # activation) are both plausible mis-wirings and both decorrelate.
    swapped = (up.float() / (1.0 + torch.exp(-up.float()))) * g
    assert _cos(torch, got, swapped) < 0.99, "gate/up operands swapped"
    assert _cos(torch, got, g * up.float()) < 0.99, "SiLU not applied"


# ---------------------------------------------------------------------------
# FP16 norm / RoPE / fused norm -> FP8
# ---------------------------------------------------------------------------


def test_layer_norm_fp16_matches_torch(env):
    """LayerNorm with plain affine ``w`` and ``b`` (NOT the Gemma ``1+w``
    fold — that fold happens at weight-conversion time, so a kernel
    secretly applying 1+w would double it). Weights are drawn around 1.0
    and bias around 0 so a dropped bias or a dropped weight both move the
    output well outside the tolerance."""
    torch, ext = env
    S, D, eps = 64, 2048, 1e-6
    torch.manual_seed(13)
    x = torch.randn(S, D, device="cuda").half()
    w = (1.0 + 0.1 * torch.randn(D, device="cuda")).half()
    b = (0.1 * torch.randn(D, device="cuda")).half()
    out = torch.empty(S, D, device="cuda", dtype=torch.half)
    ref = torch.nn.functional.layer_norm(
        x.float(), (D,), w.float(), b.float(), eps).half()

    ext.layer_norm_fp16(x.data_ptr(), w.data_ptr(), b.data_ptr(),
                        out.data_ptr(), S, D, eps, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(torch, out, ref) > COS_TIGHT
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_FP16_NORM, rtol=0.0)


def test_layer_norm_fp16_vec_matches_the_scalar_path(env):
    """Same math as the scalar entry; gated against the same torch
    reference AND required to return 0 on a dim%8==0, 16-byte-aligned
    geometry (the documented precondition)."""
    torch, ext = env
    S, D, eps = 64, 2048, 1e-6
    torch.manual_seed(13)
    x = torch.randn(S, D, device="cuda").half()
    w = (1.0 + 0.1 * torch.randn(D, device="cuda")).half()
    b = (0.1 * torch.randn(D, device="cuda")).half()
    out = torch.empty(S, D, device="cuda", dtype=torch.half)
    ref = torch.nn.functional.layer_norm(
        x.float(), (D,), w.float(), b.float(), eps).half()

    rc = ext.layer_norm_fp16_vec(x.data_ptr(), w.data_ptr(), b.data_ptr(),
                                 out.data_ptr(), S, D, eps, _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected an aligned dim={D} (rc={rc})"
    assert _cos(torch, out, ref) > COS_TIGHT
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_FP16_NORM, rtol=0.0)


def test_rms_norm_fp16_vec_uses_plain_weight_semantics(env):
    """RMSNorm with plain ``w`` multiplication and mean-of-squares (not
    sum-of-squares, not variance-about-the-mean). The reference spells the
    math out rather than calling a torch module so all three choices are
    pinned; rc == 0 checks the vector precondition contract."""
    torch, ext = env
    S, D, eps = 64, 2048, 1e-6
    torch.manual_seed(14)
    x = torch.randn(S, D, device="cuda").half()
    w = (1.0 + 0.1 * torch.randn(D, device="cuda")).half()
    out = torch.empty(S, D, device="cuda", dtype=torch.half)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
           * w.float()).half()

    rc = ext.rms_norm_fp16_vec(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                               S, D, eps, _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected an aligned dim={D} (rc={rc})"
    assert _cos(torch, out, ref) > COS_TIGHT
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_FP16_NORM, rtol=0.0)


def _rope_tables(torch, S, HD):
    """Rotate-half tables: (S, HD) with cos/sin duplicated across the two
    halves, the layout ``rope_rotate_half_fp16`` indexes."""
    half = HD // 2
    pos = torch.arange(S, device="cuda").float()
    inv = 1.0 / (10000.0 ** (torch.arange(half, device="cuda").float() / half))
    ang = pos[:, None] * inv[None, :]
    cos_t = torch.cat([ang.cos(), ang.cos()], dim=1).half()
    sin_t = torch.cat([ang.sin(), ang.sin()], dim=1).half()
    return cos_t, sin_t


def _rope_ref(torch, x, cos_t, sin_t, S, NH, HD):
    """ROTATE-HALF reference: pair element i with i + HD/2. (The pi05
    kernel `qkv_split_rope` uses the INTERLEAVED convention instead — the
    two are different kernels and each is gated against its own
    convention, which is the point of spelling this out.)"""
    half = HD // 2
    xv = x.float().view(S, NH, HD)
    lo, hi = xv[..., :half], xv[..., half:]
    c = cos_t.float()[:, None, :half]
    s = sin_t.float()[:, None, :half]
    return torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1) \
        .view(S, NH * HD).half()


def test_rope_rotate_half_fp16_uses_the_half_split_convention(env):
    """In-place rotate-half RoPE over (S, NH*HD). The gate additionally
    requires the output to DISAGREE with the interleaved-pair convention,
    because both conventions preserve norms and would pass a magnitude
    check."""
    torch, ext = env
    S, NH, HD = 32, 16, 128
    torch.manual_seed(15)
    x = torch.randn(S, NH * HD, device="cuda").half()
    x0 = x.clone()
    cos_t, sin_t = _rope_tables(torch, S, HD)
    ref = _rope_ref(torch, x0, cos_t, sin_t, S, NH, HD)

    ext.rope_rotate_half_fp16(x.data_ptr(), cos_t.data_ptr(),
                              sin_t.data_ptr(), S, NH, HD, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(torch, x, ref) > COS_TIGHT
    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_FP16_ACT, rtol=0.0)
    assert _cos(torch, x, x0) < 0.999, "RoPE did not rotate anything"


def test_rope_rotate_half_fp16_vec_matches_the_scalar_path(env):
    """Vector fast path, same convention, plus the rc == 0 precondition
    contract for (HD/2) % 8 == 0 and 16-byte-aligned tables."""
    torch, ext = env
    S, NH, HD = 32, 16, 128
    torch.manual_seed(15)
    x = torch.randn(S, NH * HD, device="cuda").half()
    x0 = x.clone()
    cos_t, sin_t = _rope_tables(torch, S, HD)
    ref = _rope_ref(torch, x0, cos_t, sin_t, S, NH, HD)

    rc = ext.rope_rotate_half_fp16_vec(x.data_ptr(), cos_t.data_ptr(),
                                       sin_t.data_ptr(), S, NH, HD,
                                       _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected HD={HD} (rc={rc})"
    assert _cos(torch, x, ref) > COS_TIGHT
    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_FP16_ACT, rtol=0.0)


def test_rms_norm_fp8_fp16_fuses_norm_and_quantize(env):
    """RMSNorm straight to FP8 with a device scale. The reference is the
    fp32 norm result put through torch e4m3 with the SAME scale, so this
    gates the fused path's encoding, not e4m3's error. Budget covers the
    wave64 reduction-order difference only."""
    torch, ext = env
    S, D, eps = 64, 2048, 1e-6
    torch.manual_seed(16)
    x = torch.randn(S, D, device="cuda").half()
    w = (1.0 + 0.1 * torch.randn(D, device="cuda")).half()
    xf = x.float()
    ref_f32 = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
               * w.float())
    scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(S, D, device="cuda", dtype=torch.uint8)

    ext.rms_norm_fp8_fp16(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                          S, D, eps, scale.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "rms_norm_fp8_fp16", out, ref_f32, scale,
                max_mismatch=FP8_MISMATCH_REDUCE)


def test_residual_add_rms_norm_fp8_fp16_updates_residual_and_quantizes(env):
    """Three-in-one: ``residual += x`` in place, RMSNorm the sum, emit FP8.

    Two separate contracts are gated:
      1. the residual buffer really is updated in place (checked against a
         pre-seeded copy, and required NOT to equal the seed);
      2. the FP8 output matches the kernel's exact intermediate — which is
         NOT ``rmsnorm(round_fp16(r + x))``: the kernel accumulates the
         sum-of-squares from the UNROUNDED fp32 sums but normalizes the
         value it read back from the fp16 residual it just stored
         (csrc/amd/kernels/norm_fp16.hip). The reference reproduces that
         asymmetry deliberately; using the naive form would need a much
         looser budget and would stop catching real drift.
    """
    torch, ext = env
    S, D, eps = 64, 2048, 1e-6
    torch.manual_seed(17)
    residual = torch.randn(S, D, device="cuda").half()
    seed = residual.clone()
    x = torch.randn(S, D, device="cuda").half()
    w = (1.0 + 0.1 * torch.randn(D, device="cuda")).half()

    unrounded = seed.float() + x.float()
    residual_ref = unrounded.half()
    rms = torch.rsqrt(unrounded.pow(2).mean(-1, keepdim=True) + eps)
    ref_f32 = residual_ref.float() * rms * w.float()
    scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(S, D, device="cuda", dtype=torch.uint8)

    ext.residual_add_rms_norm_fp8_fp16(
        residual.data_ptr(), x.data_ptr(), w.data_ptr(), out.data_ptr(),
        S, D, eps, scale.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    torch.testing.assert_close(residual.float(), residual_ref.float(),
                               atol=ATOL_FP16, rtol=0.0)
    assert not torch.allclose(residual.float(), seed.float(), atol=1e-2), (
        "residual buffer was not updated in place")
    _assert_fp8(torch, "residual_add_rms_norm_fp8_fp16", out, ref_f32, scale,
                max_mismatch=FP8_MISMATCH_REDUCE)


def test_layer_norm_fp8_static_fp16_vec_rounds_through_fp16(env):
    """LayerNorm -> FP8 in one launch. The kernel rounds the normalized
    value through fp16 before the e4m3 convert (it is the fused form of
    ``layer_norm_fp16`` followed by ``quantize_fp8_static_fp16``), so the
    reference rounds through fp16 too — otherwise the byte comparison
    would drift on the elements where the two roundings disagree."""
    torch, ext = env
    S, D, eps = 64, 2048, 1e-6
    torch.manual_seed(18)
    x = torch.randn(S, D, device="cuda").half()
    w = (1.0 + 0.1 * torch.randn(D, device="cuda")).half()
    b = (0.1 * torch.randn(D, device="cuda")).half()
    ref_f32 = torch.nn.functional.layer_norm(
        x.float(), (D,), w.float(), b.float(), eps).half().float()
    scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(S, D, device="cuda", dtype=torch.uint8)

    rc = ext.layer_norm_fp8_static_fp16_vec(
        x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
        scale.data_ptr(), S, D, eps, _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected an aligned dim={D} (rc={rc})"
    _assert_fp8(torch, "layer_norm_fp8_static_fp16_vec", out, ref_f32, scale,
                max_mismatch=FP8_MISMATCH_REDUCE)


def test_residual_add_fp16_vec_matches_the_scalar_path(env):
    """Vector residual accumulate: same pre-seeded-destination contract as
    the scalar entry, plus rc == 0 on an n%8==0, 16-byte-aligned buffer."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(19)
    residual = torch.randn(n, device="cuda").half()
    seed = residual.clone()
    x = torch.randn(n, device="cuda").half()
    ref = (seed.float() + x.float()).half()

    rc = ext.residual_add_fp16_vec(residual.data_ptr(), x.data_ptr(), n,
                                   _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected an aligned n={n} (rc={rc})"
    torch.testing.assert_close(residual.float(), ref.float(),
                               atol=ATOL_FP16, rtol=0.0)
    assert not torch.allclose(residual.float(), x.float(), atol=1e-2), (
        "residual_add_fp16_vec overwrote instead of accumulating")


def test_gpu_repeat_interleave_heads_vec_matches_the_scalar_path(env):
    """Vector GQA head expansion: bitwise identical to the scalar entry's
    reference, plus rc == 0 on an HD%8==0, 16-byte-aligned geometry."""
    torch, ext = env
    S, NH_src, HD, repeat = 32, 8, 128, 2
    torch.manual_seed(20)
    src = torch.randn(S, NH_src, HD, device="cuda").half()
    dst = torch.full((S, NH_src * repeat, HD), float("nan"),
                     device="cuda", dtype=torch.half)
    ref = torch.repeat_interleave(src, repeat, dim=1).contiguous()

    rc = ext.gpu_repeat_interleave_heads_vec(src.data_ptr(), dst.data_ptr(),
                                             S, NH_src, HD, repeat,
                                             _stream(torch))
    torch.cuda.synchronize()

    assert rc == 0, f"vec path rejected HD={HD} (rc={rc})"
    assert torch.equal(dst.view(torch.uint16), ref.view(torch.uint16))


# ---------------------------------------------------------------------------
# *_vec precondition contract
# ---------------------------------------------------------------------------


def test_vec_entries_reject_unsupported_geometry(env):
    """Every ``*_vec`` entry returns -1 (not 0, not a crash) when its
    divisibility / alignment precondition fails, and MUST NOT touch the
    destination when it does.

    This is the half of the contract the callers depend on: the pipeline
    calls the vector entry, checks the status, and falls back to the
    scalar kernel on -1. A vec entry that returned 0 while writing a
    partial result — or that wrote a partial result before returning -1 —
    would corrupt silently, because the fallback would then run on top of
    half-written data. Preconditions are read from
    csrc/amd/kernels/{norm_fp16,elementwise_fp16}.hip:
    dim%8 / n%8 / n%16 / (HD/2)%8 / HD%8.
    """
    torch, ext = env
    dev = "cuda"
    torch.manual_seed(21)

    # dim % 8 != 0 for the three norm entries (dim stays even so the
    # non-vector packed path would still be legal — the rejection has to
    # come from the vector precondition, not from a shape being absurd).
    rows, bad_dim, eps = 4, 12, 1e-6
    x = torch.randn(rows, bad_dim, device=dev).half()
    w = (1.0 + 0.1 * torch.randn(bad_dim, device=dev)).half()
    b = (0.1 * torch.randn(bad_dim, device=dev)).half()
    scale = torch.tensor([0.01], dtype=torch.float32, device=dev)

    out_f16 = torch.full((rows, bad_dim), float("nan"),
                         device=dev, dtype=torch.half)
    out_u8 = torch.full((rows, bad_dim), 0xAB, device=dev, dtype=torch.uint8)

    rc = ext.rms_norm_fp16_vec(x.data_ptr(), w.data_ptr(), out_f16.data_ptr(),
                               rows, bad_dim, eps, _stream(torch))
    assert rc == -1, f"rms_norm_fp16_vec accepted dim={bad_dim} (rc={rc})"

    rc = ext.layer_norm_fp16_vec(x.data_ptr(), w.data_ptr(), b.data_ptr(),
                                 out_f16.data_ptr(), rows, bad_dim, eps,
                                 _stream(torch))
    assert rc == -1, f"layer_norm_fp16_vec accepted dim={bad_dim} (rc={rc})"

    rc = ext.layer_norm_fp8_static_fp16_vec(
        x.data_ptr(), w.data_ptr(), b.data_ptr(), out_u8.data_ptr(),
        scale.data_ptr(), rows, bad_dim, eps, _stream(torch))
    assert rc == -1, (
        f"layer_norm_fp8_static_fp16_vec accepted dim={bad_dim} (rc={rc})")

    # (HD/2) % 8 != 0
    S, NH, bad_hd = 4, 2, 20
    xr = torch.randn(S, NH * bad_hd, device=dev).half()
    xr0 = xr.clone()
    cos_t, sin_t = _rope_tables(torch, S, bad_hd)
    rc = ext.rope_rotate_half_fp16_vec(xr.data_ptr(), cos_t.data_ptr(),
                                       sin_t.data_ptr(), S, NH, bad_hd,
                                       _stream(torch))
    assert rc == -1, f"rope_rotate_half_fp16_vec accepted HD={bad_hd} (rc={rc})"

    # n % 16 != 0 (quantize) and n % 8 != 0 (residual add)
    bad_n_q, bad_n_r = 24, 12
    xq = torch.randn(bad_n_q, device=dev).half()
    oq = torch.full((bad_n_q,), 0xAB, device=dev, dtype=torch.uint8)
    rc = ext.quantize_fp8_static_fp16_vec(xq.data_ptr(), oq.data_ptr(),
                                          scale.data_ptr(), bad_n_q,
                                          _stream(torch))
    assert rc == -1, (
        f"quantize_fp8_static_fp16_vec accepted n={bad_n_q} (rc={rc})")

    res = torch.randn(bad_n_r, device=dev).half()
    res0 = res.clone()
    xr2 = torch.randn(bad_n_r, device=dev).half()
    rc = ext.residual_add_fp16_vec(res.data_ptr(), xr2.data_ptr(), bad_n_r,
                                   _stream(torch))
    assert rc == -1, f"residual_add_fp16_vec accepted n={bad_n_r} (rc={rc})"

    # HD % 8 != 0
    bad_hd_r = 12
    src = torch.randn(S, NH, bad_hd_r, device=dev).half()
    dst = torch.full((S, NH * 2, bad_hd_r), float("nan"),
                     device=dev, dtype=torch.half)
    rc = ext.gpu_repeat_interleave_heads_vec(src.data_ptr(), dst.data_ptr(),
                                             S, NH, bad_hd_r, 2,
                                             _stream(torch))
    assert rc == -1, (
        f"gpu_repeat_interleave_heads_vec accepted HD={bad_hd_r} (rc={rc})")

    torch.cuda.synchronize()

    # Destinations untouched by every rejected call.
    assert bool(torch.isnan(out_f16).all()), "rejected norm vec wrote output"
    assert bool((out_u8 == 0xAB).all()), "rejected fp8 norm vec wrote output"
    assert bool((oq == 0xAB).all()), "rejected quantize vec wrote output"
    assert torch.equal(xr.view(torch.uint16), xr0.view(torch.uint16)), (
        "rejected rope vec mutated its in-place buffer")
    assert torch.equal(res.view(torch.uint16), res0.view(torch.uint16)), (
        "rejected residual vec mutated its in-place buffer")
    assert bool(torch.isnan(dst).all()), "rejected repeat vec wrote output"


# ---------------------------------------------------------------------------
# AdaLN family (LayerNorm math, BF16 + FP8 out)
# ---------------------------------------------------------------------------

# The AdaLN family defaults to eps 1e-5 (bf16 entries) and 1e-6
# (ada_layer_norm_fp8); the tests pass eps explicitly so a changed default
# shows up as a parity failure here rather than as DiT drift.
_ADALN_EPS_BF16 = 1e-5
_ADALN_EPS_FP8 = 1e-6


def test_layer_norm_no_affine_bf16_has_no_weight(env):
    """LayerNorm with NO affine at all: mean/variance normalize only. The
    reference passes weight=None/bias=None explicitly, so a kernel that
    quietly folded in a unit weight-and-bias would still pass, but one
    that read a weight pointer it should not have would not."""
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(22)
    x = torch.randn(S, D, device="cuda").bfloat16()
    out = torch.empty(S, D, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(
        x.float(), (D,), None, None, _ADALN_EPS_BF16).bfloat16()

    ext.layer_norm_no_affine_bf16(x.data_ptr(), out.data_ptr(), S, D,
                                  _ADALN_EPS_BF16, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(torch, out, ref) > COS_TIGHT
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


def test_ada_layer_norm_bf16_applies_one_plus_scale(env):
    """AdaLN modulation is ``LN(x) * (1 + scale) + shift`` — the ``1 +``
    is the whole point (scale is a delta produced by the DiT conditioning
    MLP and is near zero at init). The gate additionally requires the
    output to DISAGREE with the ``LN(x) * scale + shift`` form, which is
    the exact bug the ``1 +`` exists to prevent and which a cosine against
    correlated data would not separate."""
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(23)
    x = torch.randn(S, D, device="cuda").bfloat16()
    scale = (0.2 * torch.randn(D, device="cuda")).bfloat16()
    shift = (0.2 * torch.randn(D, device="cuda")).bfloat16()
    out = torch.empty(S, D, device="cuda", dtype=torch.bfloat16)
    ln = torch.nn.functional.layer_norm(x.float(), (D,), None, None,
                                        _ADALN_EPS_BF16)
    ref = (ln * (1.0 + scale.float()) + shift.float()).bfloat16()

    ext.ada_layer_norm_bf16(x.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                            out.data_ptr(), S, D, _ADALN_EPS_BF16,
                            _stream(torch))
    torch.cuda.synchronize()

    assert _cos(torch, out, ref) > COS_TIGHT
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)
    no_one = (ln * scale.float() + shift.float())
    assert _cos(torch, out.float(), no_one) < 0.99, (
        "ada_layer_norm_bf16 dropped the '1 +' in (1 + scale)")


def test_ada_layer_norm_fp8_rounds_through_bf16_before_quantizing(env):
    """AdaLN straight to FP8 (reviewer-named entry point).

    Contract from csrc/amd/kernels/adaln_layer_norm.hip: the modulated
    value is rounded through bf16 IN REGISTERS before the e4m3 convert,
    which is what makes this single launch bit-equivalent to the two-launch
    ``ada_layer_norm_bf16`` -> ``quantize_fp8_static`` chain it replaces.
    The reference therefore rounds through bf16 too, and the comparison is
    byte-level against torch e4m3 with the SAME device scale — an
    unquantized fp32 reference here would pass with the bf16 round-through
    silently missing, which is precisely the regression that would break
    equivalence with the two-launch form.
    """
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(24)
    x = torch.randn(S, D, device="cuda").bfloat16()
    scale = (0.2 * torch.randn(D, device="cuda")).bfloat16()
    shift = (0.2 * torch.randn(D, device="cuda")).bfloat16()
    ln = torch.nn.functional.layer_norm(x.float(), (D,), None, None,
                                        _ADALN_EPS_FP8)
    modulated = ln * (1.0 + scale.float()) + shift.float()
    ref_f32 = modulated.bfloat16().float()          # the bf16 round-through
    act_scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(S, D, device="cuda", dtype=torch.uint8)

    ext.ada_layer_norm_fp8(x.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                           out.data_ptr(), act_scale.data_ptr(), S, D,
                           _ADALN_EPS_FP8, _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "ada_layer_norm_fp8", out, ref_f32, act_scale,
                max_mismatch=FP8_MISMATCH_REDUCE)


def test_ada_layer_norm_fp8_equals_the_two_launch_chain(env):
    """The reason ada_layer_norm_fp8 exists: it must produce the same FP8
    bytes as ``ada_layer_norm_bf16`` followed by ``quantize_fp8_static_fp16``
    would, on the same input and scale. Gated against the FUSED kernel's
    own two-launch equivalent (bf16 AdaLN, then the bf16 result quantized
    by torch with the same scale) rather than against a fresh fp32
    computation, so it isolates the fusion from the norm math already
    gated above."""
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(25)
    x = torch.randn(S, D, device="cuda").bfloat16()
    scale = (0.2 * torch.randn(D, device="cuda")).bfloat16()
    shift = (0.2 * torch.randn(D, device="cuda")).bfloat16()

    two_launch = torch.empty(S, D, device="cuda", dtype=torch.bfloat16)
    ext.ada_layer_norm_bf16(x.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                            two_launch.data_ptr(), S, D, _ADALN_EPS_FP8,
                            _stream(torch))
    torch.cuda.synchronize()

    act_scale = _static_scale(torch, two_launch.float()).cuda()
    fused = torch.empty(S, D, device="cuda", dtype=torch.uint8)
    ext.ada_layer_norm_fp8(x.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                           fused.data_ptr(), act_scale.data_ptr(), S, D,
                           _ADALN_EPS_FP8, _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "ada_layer_norm_fp8 vs two-launch chain", fused,
                two_launch.float(), act_scale,
                max_mismatch=FP8_MISMATCH_REDUCE)


def test_bias_gelu_quantize_fp8_static_bf16_with_bias(env):
    """``out_fp8 = quant(gelu_tanh(in + bias))``, bias broadcast over rows.
    The GELU reference pins the tanh approximation (the kernel implements
    0.5x(1+tanh(0.7978845608(x + 0.044715 x^3))) verbatim); the byte budget
    covers one ULP of device ``tanhf``."""
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(26)
    x = torch.randn(S, D, device="cuda").bfloat16()
    bias = (0.1 * torch.randn(D, device="cuda")).bfloat16()
    ref_f32 = _gelu_tanh(torch, x.float() + bias.float())
    act_scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(S, D, device="cuda", dtype=torch.uint8)

    ext.bias_gelu_quantize_fp8_static_bf16(
        x.data_ptr(), bias.data_ptr(), out.data_ptr(), act_scale.data_ptr(),
        S, D, _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "bias_gelu_quantize_fp8_static_bf16", out, ref_f32,
                act_scale, max_mismatch=FP8_MISMATCH_RNE)
    # The bias must actually land. A dequantized cosine cannot say so
    # (gelu(x+b) and gelu(x) are ~0.995 correlated at this bias scale), so
    # the discriminator is at the byte level: dropping the bias moves a
    # large majority of the FP8 codes, hundreds of times the RNE budget.
    no_bias_u8 = _fp8_ref_bytes(torch, _gelu_tanh(torch, x.float()),
                                act_scale).reshape(out.shape)
    changed = float((out != no_bias_u8).float().mean())
    assert changed > 0.1, (
        f"only {changed:.4f} of FP8 bytes differ from the un-biased GELU — "
        "the bias was not applied")


def test_bias_gelu_quantize_fp8_static_bf16_accepts_null_bias(env):
    """The binding forwards a 0 bias pointer as ``nullptr`` (the C++ kernel
    branches on it). That null path is a distinct code path in the kernel
    and is what the no-bias production sites use, so it gets its own gate:
    the result must be the un-biased GELU, not a read of address 0 and not
    the biased form."""
    torch, ext = env
    S, D = 64, 2048
    torch.manual_seed(26)
    x = torch.randn(S, D, device="cuda").bfloat16()
    ref_f32 = _gelu_tanh(torch, x.float())
    act_scale = _static_scale(torch, ref_f32).cuda()
    out = torch.empty(S, D, device="cuda", dtype=torch.uint8)

    ext.bias_gelu_quantize_fp8_static_bf16(
        x.data_ptr(), 0, out.data_ptr(), act_scale.data_ptr(),
        S, D, _stream(torch))
    torch.cuda.synchronize()

    _assert_fp8(torch, "bias_gelu_quantize_fp8_static_bf16 (null bias)", out,
                ref_f32, act_scale, max_mismatch=FP8_MISMATCH_RNE)


# ---------------------------------------------------------------------------
# smallm_mfma_bf16 — packed-weight small-M BF16 GEMM
# ---------------------------------------------------------------------------

# Real GROOT N1.7 DiT projection shapes at M=41 (csrc/amd/gemm/
# smallm_mfma_bf16.h): q/k/v/o, ffn up, ffn down.
SMALLM_BF16_SHAPES = [
    ("dit_qkvo", 41, 1536, 1536),
    ("dit_ff1", 41, 6144, 1536),
    ("dit_ff2", 41, 1536, 6144),
]

SMALLM_BF16_EPILOGUES = ("bias", "bias_gelu", "bias_res")

_SMALLM_BF16_FN = {
    "bias": "smallm_mfma_bf16_nn_bias",
    "bias_gelu": "smallm_mfma_bf16_nn_bias_gelu",
    "bias_res": "smallm_mfma_bf16_nn_bias_res",
}

# cos gate for the packed GEMM: fp32 MFMA accumulation vs a torch fp32
# matmul over the SAME bf16 operands — only the K-reduction order and the
# final bf16 store differ, so 0.9999 is the same bar the hipBLASLt GEMM
# entries are held to in tests/test_amd_kernel_parity.py.
COS_SMALLM = 0.9999


def _pack_bf16_weight(torch, W):
    """The documented pack one-liner from csrc/amd/gemm/smallm_mfma_bf16.h.

    W is the (K, N) row-major bf16 weight; the kernel consumes a linear
    stream of 16-byte chunks in per-lane order, chunk index inside each
    16-column n-tile being ``step*64 + lane`` with
    ``lane = (k_group << 4) | n_lane``:

        Wp = W.view(K//32, 4, 8, N//16, 16).permute(3, 0, 1, 4, 2)

    Exercising the pack through this exact expression is the point: the
    header documents it as the supported way to produce Wp, so if the
    kernel's layout ever changes without the header, this test fails.
    """
    K, N = W.shape
    return W.view(K // 32, 4, 8, N // 16, 16) \
            .permute(3, 0, 1, 4, 2).contiguous()


def _smallm_bf16_case(torch, M, N, K, seed):
    """Real-distribution operands: unit-normal activations with heavy
    post-residual rows, 0.02-sigma weights, 0.1-sigma bias. Constant or
    ramp inputs would hide an accumulator or fragment-mapping bug."""
    torch.manual_seed(seed)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    A[::max(M // 4, 1)] *= 8.0
    W = (0.02 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    bias = (0.1 * torch.randn(N, device="cuda")).to(torch.bfloat16)
    return A, W, _pack_bf16_weight(torch, W), bias


def _valid_variants(ext, K):
    """Variant ids whose wave depth divides this K into a supported number
    of MFMA steps (header: K/(32*waves) must be in {6, 12, 24, 48})."""
    names = list(ext.smallm_mfma_bf16_variants())
    ok = []
    for vid, name in enumerate(names):
        if name == "auto":
            ok.append((vid, name))
            continue
        waves = 4 if name.startswith("w4") else 8
        if K % (32 * waves) == 0 and K // (32 * waves) in (6, 12, 24, 48):
            ok.append((vid, name))
    return ok


def test_smallm_mfma_bf16_variants_enumerator(env):
    """The variant enumerator is pure host introspection and is what the
    parity/bench harnesses iterate; it must answer with at least the
    documented ``auto`` entry plus the w4/w8 x fused/split forms, and the
    ids must be contiguous from 0 (the binding passes the index straight
    through as the ``variant`` argument)."""
    torch, ext = env
    names = list(ext.smallm_mfma_bf16_variants())
    assert len(names) >= 5, f"expected auto + 4 launch forms, got {names}"
    assert names[0] == "auto", f"variant 0 must be the auto heuristic: {names}"
    assert all(isinstance(n, str) and n for n in names)
    assert len(set(names)) == len(names), f"duplicate variant names: {names}"


@pytest.mark.parametrize("label,M,N,K", SMALLM_BF16_SHAPES,
                         ids=[s[0] for s in SMALLM_BF16_SHAPES])
@pytest.mark.parametrize("epilogue", SMALLM_BF16_EPILOGUES)
def test_smallm_mfma_bf16_epilogue_parity(env, label, M, N, K, epilogue):
    """Every epilogue x every valid launch variant x every production
    shape, against a torch fp32 matmul over the SAME bf16 operands.

    Semantics gated per epilogue (csrc/amd/gemm/smallm_mfma_bf16.h):
      bias      : D = bf16(A@W + bias)             — bias broadcast over N
      bias_gelu : D = bf16(gelu_tanh(A@W + bias))  — tanh approx, and the
                  result is required to differ from the plain-bias form so
                  a dropped activation cannot pass
      bias_res  : D = bf16(float(D) + A@W + bias)  — ACCUMULATE into a
                  pre-seeded destination, and the result is required to
                  differ from the non-accumulated form so a beta=0
                  regression cannot pass

    Running every variant matters because they are different kernels
    (fused vs split m-tile placement, 4 vs 8 waves): a fragment-mapping or
    LDS-reduction bug can be present in one and absent in the others.
    """
    torch, ext = env
    A, W, Wp, bias = _smallm_bf16_case(torch, M, N, K, seed=M * 7919 + N + K)
    base = A.float() @ W.float() + bias.float()
    if epilogue == "bias_gelu":
        expected_no_res = _gelu_tanh(torch, base)
    else:
        expected_no_res = base

    torch.manual_seed(1234)
    seed_buf = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    fn = getattr(ext, _SMALLM_BF16_FN[epilogue])

    variants = _valid_variants(ext, K)
    assert variants, f"no valid launch variant for K={K}"
    for vid, vname in variants:
        D = torch.full((M, N), float("nan"), device="cuda",
                       dtype=torch.bfloat16)
        if epilogue == "bias_res":
            D.copy_(seed_buf)
            ref = seed_buf.float() + expected_no_res
        else:
            ref = expected_no_res

        fn(A.data_ptr(), Wp.data_ptr(), bias.data_ptr(), D.data_ptr(),
           M, N, K, vid, _stream(torch))
        torch.cuda.synchronize()

        assert bool(torch.isfinite(D.float()).all()), (
            f"{label}/{epilogue}/{vname}: non-finite output (unwritten tile)")
        c = _cos(torch, D, ref)
        assert c > COS_SMALLM, (
            f"{label}/{epilogue}/{vname} M={M} N={N} K={K}: cos={c:.7f}")

        if epilogue == "bias_res":
            assert _cos(torch, D.float(), expected_no_res) < 0.999, (
                f"{vname}: bias_res overwrote D instead of accumulating")
        if epilogue == "bias_gelu":
            assert _cos(torch, D.float(), base) < 0.999, (
                f"{vname}: bias_gelu did not apply the activation")


def test_smallm_mfma_bf16_requires_the_documented_pack_layout(env):
    """Feeding the SAME weight bytes UNPACKED must give the wrong answer.

    The packed layout is the kernel's whole contract with the caller (a
    (K,N) row-major weight is reinterpreted as a per-lane consumption-order
    stream of 16-byte chunks). An unpacked weight has identical size and
    alignment, so the kernel runs happily and returns garbage — which is
    exactly why a parity test that only ever feeds the packed form cannot
    tell whether the pack one-liner in the header is still correct. This
    gate pins that the permutation is load-bearing: if the kernel ever
    started accepting the plain layout, the header's documented one-liner
    (and every caller following it) would be silently wrong.
    """
    torch, ext = env
    M, N, K = 41, 1536, 1536
    A, W, Wp, bias = _smallm_bf16_case(torch, M, N, K, seed=999)
    ref = A.float() @ W.float() + bias.float()

    D_packed = torch.full((M, N), float("nan"), device="cuda",
                          dtype=torch.bfloat16)
    ext.smallm_mfma_bf16_nn_bias(A.data_ptr(), Wp.data_ptr(), bias.data_ptr(),
                                 D_packed.data_ptr(), M, N, K, 0,
                                 _stream(torch))
    torch.cuda.synchronize()
    assert _cos(torch, D_packed, ref) > COS_SMALLM, "packed control arm"

    W_plain = W.contiguous()
    assert W_plain.numel() == Wp.numel(), "unpacked arm must read the same bytes"
    D_plain = torch.full((M, N), float("nan"), device="cuda",
                         dtype=torch.bfloat16)
    ext.smallm_mfma_bf16_nn_bias(A.data_ptr(), W_plain.data_ptr(),
                                 bias.data_ptr(), D_plain.data_ptr(),
                                 M, N, K, 0, _stream(torch))
    torch.cuda.synchronize()
    assert _cos(torch, D_plain, ref) < 0.99, (
        "the unpacked weight produced a correct result — the documented "
        "pack one-liner is no longer the layout the kernel consumes")
