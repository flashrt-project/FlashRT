"""AMD pi05 kernel surface — numerical parity vs torch references.

Standalone parity gates for the representative kernels the Pi0.5 CDNA4
pipeline is assembled from, each checked against an independent torch
reference on real-distribution inputs (randn-scaled — constant or integer
ramps would hide calibration- and rounding-path bugs):

    rms_norm / layer_norm     — norm math + weight semantics (plain ``w``,
                                NOT the Gemma ``1+w`` fold: the fold happens
                                at weight-conversion time, so a kernel that
                                secretly applied 1+w would double it)
    residual_add              — bit-exact fp32-add/bf16-round contract
    gelu_inplace              — tanh-approx GELU (not SiLU, not erf-GELU)
    qkv_split_rope            — interleaved-pair rotation + V passthrough
    quantize_fp8_static       — BYTE-exact vs torch float8_e4m3fn with the
                                SAME scale (never compared to raw fp32)
    attention_decoder_gqa     — split-KV flash decoder vs exact softmax
                                reference, exact and seqused (fixed-shape
                                graph) modes

plus the COMPLETE GemmRunner / FvkContext surface bound by
csrc/amd/gemm/bindings_gemm.inc — every layout convention, every
epilogue, both scale-passing conventions, and every autotune entry:

    FvkContext                — live hipBLASLt handle
    bf16_run                  — NT layout (B stored (N,K))
    bf16_nn                   — NN layout (B stored (K,N))
    bf16_nn_res               — beta=1 accumulate into a seeded D
    bf16_nn_bias              — BIAS epilogue, broadcast over N
    bf16_nn_bias_gelu         — BIAS + tanh-approx GELU epilogue
    bf16_nn_bias_res          — BIAS epilogue combined with beta=1
    fp8_nn_dev / fp8_nt_dev   — FP8 layouts + DEVICE-scale semantics
    fp16_nn                   — FP16 in/out NN GEMM
    fp8_nn_bias               — HOST alpha + BIAS -> FP16
    fp8_nn_gelu_bias          — HOST alpha + BIAS + GELU -> FP16
    fp8_descale_fp16          — DEVICE descale pointers -> FP16
    mxfp4_nt_dev              — OCP MX FP4 (E2M1 + per-1x32 UE8M0) NT
    enable_lazy_autotune      — first-call timed selection
    autotune_bf16_nn / _fp16_nn / _fp8_nn_dev / _fp8_nt_dev /
    autotune_fp8_descale_fp16 / autotune_mxfp4_nt_dev
                              — tuned algorithm must not change semantics

The two other kernel families this branch adds — the packed-weight
``smallm_mfma_bf16_*`` GEMM (csrc/amd/gemm/bindings_smallm_bf16.inc) and
the 29-entry FP16/BF16/AdaLN backbone port
(csrc/amd/kernels/bindings_fp16_port.inc) — are gated in the sibling
module ``tests/test_amd_kernel_parity_groot.py``, which keeps this file
to the pi05 kernel surface plus the hipBLASLt GEMM surface.

Binding signatures were read from csrc/amd/bindings.cpp and
csrc/amd/gemm/bindings_gemm.inc before writing any call here; every tensor
is bound to a named variable before ``.data_ptr()`` (an inline temporary
would be GC'd mid-launch).

Skip conditions: ROCm torch + a visible device + the built extension. On
NVIDIA/no-ROCm CI the module collects and every test skips with a reason.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def env():
    """(torch, extension) on a ROCm box with the extension built."""
    try:
        ext = importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None):
        pytest.skip("torch is not a ROCm build")
    if not torch.cuda.is_available():
        pytest.skip("no ROCm device visible to torch")
    return torch, ext


def _stream(torch) -> int:
    """Launch kernels on torch's current stream so torch-side tensor prep
    and the raw-pointer launches are ordered without extra events."""
    return torch.cuda.current_stream().cuda_stream


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# Numerical gates, justified once:
#  - COS_ELEMENTWISE 0.9999: elementwise/norm kernels share the reference's
#    fp32 math except reduction order → agreement is a few bf16 ULPs; any
#    real bug (wrong weight semantics, shifted row, wrong pair) lands far
#    below this.
#  - ATOL_BF16 2e-2: one bf16 ULP at unit scale is ~2^-8 ≈ 4e-3; 2e-2 gives
#    headroom for a couple of chained roundings while still failing hard on
#    O(1) math errors.
#  - COS_GEMM 0.9999: hipBLASLt accumulates fp32 like the torch reference;
#    only split-K order and the bf16 store differ.
#  - COS_ATTN 0.999: attention output is bf16 with fp32 softmax; matches
#    the gate used for this kernel's bring-up validation.
COS_ELEMENTWISE = 0.9999
ATOL_BF16 = 2e-2
COS_GEMM = 0.9999
COS_ATTN = 0.999


# ---------------------------------------------------------------------------
# Norm kernels
# ---------------------------------------------------------------------------


def test_rms_norm_matches_torch(env):
    torch, ext = env
    seq, dim = 64, 2048  # encoder width; dim must be even (packed pairs)
    torch.manual_seed(0)
    x = (0.5 * torch.randn(seq, dim, device="cuda")).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(dim, device="cuda")).to(torch.bfloat16)
    out = torch.empty_like(x)

    ext.rms_norm(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                 seq, dim, 1e-6, _stream(torch))
    torch.cuda.synchronize()

    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
           * w.float()).to(torch.bfloat16)
    assert _cos(out.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


def test_layer_norm_matches_torch(env):
    torch, ext = env
    seq, dim = 64, 1152  # SigLIP width
    torch.manual_seed(1)
    x = (0.5 * torch.randn(seq, dim, device="cuda")).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(dim, device="cuda")).to(torch.bfloat16)
    b = (0.1 * torch.randn(dim, device="cuda")).to(torch.bfloat16)
    out = torch.empty_like(x)

    ext.layer_norm(x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
                   seq, dim, 1e-6, _stream(torch))
    torch.cuda.synchronize()

    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    ref = ((xf - mean) * torch.rsqrt(var + 1e-6) * w.float()
           + b.float()).to(torch.bfloat16)
    assert _cos(out.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------


def test_residual_add_is_bit_exact(env):
    """residual += x is one fp32 add + one RNE bf16 round per element in
    both kernel and reference — nothing order-dependent, so the gate is
    bitwise equality, not a tolerance."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(2)
    residual = torch.randn(n, device="cuda").to(torch.bfloat16)
    x = torch.randn(n, device="cuda").to(torch.bfloat16)
    ref = (residual.float() + x.float()).to(torch.bfloat16)

    ext.residual_add(residual.data_ptr(), x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(residual.view(torch.uint16), ref.view(torch.uint16)), (
        "residual_add is not bit-exact vs fp32-add + bf16-RNE")


def test_gelu_inplace_is_tanh_approx(env):
    """Must be the tanh-approx GELU — erf-GELU or SiLU here would pass a
    loose cos gate on symmetric data, so the reference pins the exact
    approximate= variant."""
    torch, ext = env
    n = 64 * 1024
    torch.manual_seed(3)
    x = (0.7 * torch.randn(n, device="cuda")).to(torch.bfloat16)
    ref = torch.nn.functional.gelu(
        x.float(), approximate="tanh").to(torch.bfloat16)

    ext.gelu_inplace(x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(x.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE
    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


# ---------------------------------------------------------------------------
# QKV split + RoPE
# ---------------------------------------------------------------------------


def test_qkv_split_rope_matches_reference(env):
    """Interleaved-pair rotation (verified in csrc/amd/kernels/rope.hip):
    within each head, elements (2p, 2p+1) rotate with rope_weights[row]
    laid out cos@even / sin@odd over head_dim. Q: all heads rotated; K:
    single head, same per-row table; V: pure copy (bit-exact gate)."""
    torch, ext = env
    seq, hd = 16, 256
    q_dim, k_dim, v_dim = 8 * hd, hd, hd
    torch.manual_seed(4)
    qkv = (0.5 * torch.randn(seq, q_dim + k_dim + v_dim,
                             device="cuda")).to(torch.bfloat16)
    # Real rotary table (positions x standard inv-freq), interleaved.
    pos = torch.arange(seq, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, hd, 2, dtype=torch.float32, device="cuda") / hd))
    ang = pos[:, None] * inv_freq[None, :]           # (seq, hd/2)
    rope = torch.empty(seq, hd, device="cuda")
    rope[:, 0::2] = torch.cos(ang)
    rope[:, 1::2] = torch.sin(ang)
    rope = rope.to(torch.bfloat16)

    Q = torch.empty(seq, q_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.empty(seq, k_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.empty(seq, v_dim, dtype=torch.bfloat16, device="cuda")

    ext.qkv_split_rope(qkv.data_ptr(), rope.data_ptr(),
                       Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                       seq, q_dim, k_dim, v_dim, hd, _stream(torch))
    torch.cuda.synchronize()

    c = rope.float()[:, 0::2]
    s = rope.float()[:, 1::2]

    def rot(part):
        xf = part.float().reshape(seq, -1, hd // 2, 2)
        x0, x1 = xf[..., 0], xf[..., 1]
        cc, ss = c[:, None, :], s[:, None, :]
        even = x0 * cc - x1 * ss
        odd = x1 * cc + x0 * ss
        return torch.stack([even, odd], dim=-1).reshape(seq, -1)

    q_ref = rot(qkv[:, :q_dim]).to(torch.bfloat16)
    k_ref = rot(qkv[:, q_dim:q_dim + k_dim]).to(torch.bfloat16)
    v_ref = qkv[:, q_dim + k_dim:]

    assert torch.equal(V.view(torch.uint16), v_ref.contiguous().view(torch.uint16)), (
        "V must be a bit-exact passthrough")
    for got, ref, name in [(Q, q_ref, "Q"), (K, k_ref, "K")]:
        assert _cos(got.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE, name
        torch.testing.assert_close(got.float(), ref.float(),
                                   atol=ATOL_BF16, rtol=0.0)


# ---------------------------------------------------------------------------
# FP8 quantize
# ---------------------------------------------------------------------------


def test_quantize_fp8_static_byte_exact_vs_torch(env):
    """BYTE comparison against torch's float8_e4m3fn cast with the SAME
    scale, replicating the kernel's exact arithmetic (multiply by the fp32
    reciprocal of the scale, clamp to ±448, RNE convert — verified in
    csrc/amd/kernels/quantize_fp8.hip). Comparing against unquantized fp32
    would be a category error: the contract is the encoding, not
    closeness."""
    torch, ext = env
    n = 32768
    torch.manual_seed(5)
    x = (2.0 * torch.randn(n, device="cuda")).to(torch.bfloat16)
    # Production-style scale: amax/448 (per-tensor symmetric).
    amax = x.float().abs().max().item()
    scale = torch.tensor([max(amax / 448.0, 1e-12)],
                         dtype=torch.float32, device="cuda")
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")

    ext.quantize_fp8_static(x.data_ptr(), out.data_ptr(), scale.data_ptr(),
                            n, _stream(torch))
    torch.cuda.synchronize()

    inv_s = torch.reciprocal(scale)          # fp32, same op as the kernel
    ref = (x.float() * inv_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    mismatched = int((out.view(torch.uint8) != ref.view(torch.uint8))
                     .sum().item())
    assert mismatched == 0, (
        f"{mismatched}/{n} FP8 bytes differ from torch e4m3 with the same "
        "scale — encoding drift would silently poison every FP8 GEMM")


# ---------------------------------------------------------------------------
# GemmRunner (hipBLASLt)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemm(env):
    _, ext = env
    return ext.GemmRunner()


def test_gemm_bf16_nn_matches_torch(env, gemm):
    """bf16_nn contract: D(M,N) = A(M,K) @ B(K,N), all row-major, no
    transpose (the col-major swap inside the runner must be invisible)."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(6)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.bf16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                 _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ B.float())
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    # A transposed-layout bug produces garbage, not near-misses; the loose
    # allclose is a shape/layout tripwire on top of the cos gate.
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)


def test_gemm_fp8_nn_dev_matches_quantized_reference(env, gemm):
    """fp8_nn_dev: D_bf16 = (A_fp8(M,K) @ B_fp8(K,N)) * sa * sb, with the
    scales read from DEVICE pointers (A/B_SCALE_POINTER semantics). The
    reference multiplies the same fp8-decoded operands — never the
    pre-quantization fp32 originals."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(7)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.float8_e4m3fn)
    sa = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    sb = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.fp8_nn_dev(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                    sa.data_ptr(), sb.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()) * (sa * sb)
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.05, rtol=0.05)


def test_gemm_fp8_nt_dev_matches_quantized_reference(env, gemm):
    """fp8_nt_dev: B stored (N,K) row-major, D = A @ B^T * sa * sb. This is
    the production pi05 layout (fp8_layout='nk', -2.8 ms E2E vs kn), so its
    transpose convention gets its own gate."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(8)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    Bt = (0.3 * torch.randn(N, K, device="cuda")).to(torch.float8_e4m3fn)
    sa = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    sb = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.fp8_nt_dev(A.data_ptr(), Bt.data_ptr(), D.data_ptr(), M, N, K,
                    sa.data_ptr(), sb.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ Bt.float().t()) * (sa * sb)
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# GemmRunner — BF16 layouts and epilogues
# ---------------------------------------------------------------------------

# Extra gates, justified once:
#  - COS_GEMM_FP4 0.9999: the MXFP4 path is compared against a reference
#    that dequantizes the SAME E2M1/UE8M0 operands, so only the fp32
#    accumulation order differs — the same bar as the other GEMMs. (A cos
#    against the UNQUANTIZED matmul would measure MXFP4's quantization
#    loss, ~0.99 on randn, and would pass with a broken block-scale path.)
#  - COS_DECORRELATED 0.99: the "must NOT match the wrong interpretation"
#    discriminators. Two independent random matmuls of the same shape have
#    cos ~0 with overwhelming probability; 0.99 is a wide safety margin
#    that still fails instantly on a genuine layout/epilogue regression.
COS_GEMM_FP4 = 0.9999
COS_DECORRELATED = 0.99

E4M3_MAX = 448.0


def _quant_fp8(torch, x):
    """Per-tensor symmetric e4m3 quantize, the way production calibration
    does it: scale = amax/448. Returns (fp8 tensor, fp32 scale scalar)."""
    amax = x.float().abs().max().clamp(min=1e-8)
    scale = (amax / E4M3_MAX).float()
    q = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def test_fvk_context_creates_a_live_hipblaslt_handle(env):
    """FvkContext wraps a hipblasLtHandle_t that raw-ABI call sites take by
    address. A failed hipblasLtCreate would leave it null, and every
    downstream GEMM would fault far from the cause — so the constructed
    handle is asserted non-null here, on device, at import time of the
    suite rather than mid-capture."""
    torch, ext = env
    ctx = ext.FvkContext()
    assert isinstance(ctx.handle_ptr, int)
    assert ctx.handle_ptr != 0, "hipblasLtCreate returned a null handle"


def test_gemm_bf16_run_is_the_nt_layout(env, gemm):
    """bf16_run is the NT path: D = A(M,K) @ B(N,K)^T, B stored row-major
    as (N,K). N == K here on purpose, so the NN reading of the same bytes
    is equally well-formed — the gate then discriminates between the two
    conventions instead of relying on one of them being a shape error."""
    torch, _ = env
    M, N, K = 64, 256, 256
    torch.manual_seed(30)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    Bt = (0.3 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.bf16_run(A.data_ptr(), Bt.data_ptr(), D.data_ptr(), M, N, K,
                  _stream(torch))
    torch.cuda.synchronize()

    ref_nt = A.float() @ Bt.float().t()
    ref_nn = A.float() @ Bt.float()
    assert _cos(D.float().cpu(), ref_nt.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref_nt, atol=0.5, rtol=0.05)
    assert _cos(D.float().cpu(), ref_nn.cpu()) < COS_DECORRELATED, (
        "bf16_run produced the NN result — the NT transpose was dropped")


def test_gemm_bf16_nn_res_accumulates_into_the_destination(env, gemm):
    """bf16_nn_res is beta=1: D += A @ B, with the residual landing on the
    FP32 accumulator before the single bf16 round (that fused rounding is
    the reason the entry exists instead of a separate residual_add).

    Gated against a PRE-SEEDED D so the accumulate is actually exercised,
    and additionally required not to match plain A@B — a beta=0 regression
    would otherwise pass any gate that compared against a zeroed D.
    """
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(31)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    # Seed magnitude ~ the matmul's own, so a dropped accumulate is a large
    # relative error rather than a rounding-scale one.
    seed = (2.5 * torch.randn(M, N, device="cuda")).to(torch.bfloat16)
    D = seed.clone()

    gemm.bf16_nn_res(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                     _stream(torch))
    torch.cuda.synchronize()

    prod = A.float() @ B.float()
    ref = seed.float() + prod
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)
    assert _cos(D.float().cpu(), prod.cpu()) < COS_DECORRELATED, (
        "bf16_nn_res overwrote D (beta=0) instead of accumulating")


def test_gemm_bf16_nn_bias_broadcasts_over_columns(env, gemm):
    """BIAS epilogue: D = A@B + bias(N), one bias element per OUTPUT COLUMN
    broadcast down the rows. A row-broadcast (length-M) mix-up is the
    classic epilogue bug; the elementwise reference pins the direction."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(32)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    bias = (1.0 * torch.randn(N, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.bf16_nn_bias(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                      bias.data_ptr(), M, N, K, _stream(torch))
    torch.cuda.synchronize()

    prod = A.float() @ B.float()
    ref = prod + bias.float()
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)
    # Column-wise bias means every row shifts by the SAME vector; the
    # residual after subtracting the matmul must be that vector, repeated.
    delta = D.float() - prod
    torch.testing.assert_close(delta, bias.float().expand(M, N).contiguous(),
                               atol=0.5, rtol=0.05)


def test_gemm_bf16_nn_bias_gelu_is_tanh_approx(env, gemm):
    """BIAS + GELU epilogue: D = GELU(A@B + bias), tanh approximation (the
    hipBLASLt GELU_BIAS semantics, matching activation.hip and the CUDA
    class). The reference pins ``approximate="tanh"``; the extra gate
    requires the output to differ from the un-activated bias form so a
    dropped epilogue cannot pass."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(33)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    bias = (0.5 * torch.randn(N, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.bf16_nn_bias_gelu(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                           bias.data_ptr(), M, N, K, _stream(torch))
    torch.cuda.synchronize()

    pre = A.float() @ B.float() + bias.float()
    ref = torch.nn.functional.gelu(pre, approximate="tanh")
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)
    assert _cos(D.float().cpu(), pre.cpu()) < 0.999, "GELU was not applied"


def test_gemm_bf16_nn_bias_res_adds_bias_and_accumulates(env, gemm):
    """BIAS epilogue combined with beta=1: D += A@B + bias, both landing on
    the FP32 accumulator before one bf16 round.

    Two independent regressions are possible here and both get their own
    discriminator: dropping the residual (would match A@B + bias) and
    dropping the bias (would match D + A@B). The destination is pre-seeded
    so the accumulate is genuinely exercised.
    """
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(34)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    bias = (2.5 * torch.randn(N, device="cuda")).to(torch.bfloat16)
    seed = (2.5 * torch.randn(M, N, device="cuda")).to(torch.bfloat16)
    D = seed.clone()

    gemm.bf16_nn_bias_res(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                          bias.data_ptr(), M, N, K, _stream(torch))
    torch.cuda.synchronize()

    prod = A.float() @ B.float()
    ref = seed.float() + prod + bias.float()
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)
    assert _cos(D.float().cpu(), (prod + bias.float()).cpu()) < COS_DECORRELATED, (
        "bf16_nn_bias_res dropped the residual (beta=0)")
    assert _cos(D.float().cpu(), (seed.float() + prod).cpu()) < COS_DECORRELATED, (
        "bf16_nn_bias_res dropped the bias epilogue")


# ---------------------------------------------------------------------------
# GemmRunner — GROOT N1.7 FP16 / FP8-epilogue surface
# ---------------------------------------------------------------------------

# Real GROOT N1.7 GEMM shapes; the FP16/FP8-epilogue entries are dispatched
# on exactly these (M,N,K) in production, and hipBLASLt picks a different
# algorithm per shape — so parity is gated per shape, not on one token
# shape. (label, M, N, K)
GROOT_GEMM_SHAPES = [
    ("vit_qkv", 1024, 3072, 1024),
    ("vit_ff1", 1024, 4096, 1024),
    ("vit_ff2", 1024, 1024, 4096),
    ("vit_out", 1024, 2048, 1024),
    ("llm_qkvproj", 968, 4096, 2048),
    ("llm_ff", 968, 6144, 2048),
    ("vlsa_proj", 968, 2048, 2048),
    ("dit_ff1", 41, 6144, 1536),
    ("dit_ff2", 41, 1536, 6144),
    ("dit_qkv", 41, 1536, 1536),
]

_GROOT_IDS = [s[0] for s in GROOT_GEMM_SHAPES]

# FP16 output tolerance: one fp16 ULP at the reference's ~0.8 magnitude is
# ~5e-4; 2e-2 absolute / 5e-2 relative is the same loose layout tripwire
# the bf16 entries use on top of the cos gate, not the real gate.
ATOL_FP16_GEMM = 2e-2
RTOL_FP16_GEMM = 5e-2


def _groot_operands(torch, M, N, K, seed):
    """Real-distribution operands at GROOT magnitudes: 0.5-sigma
    activations, 0.02-sigma weights, 0.1-sigma bias. Random uniform or
    constant fills would keep every value in one exponent bucket and hide
    FP8 scale-path bugs."""
    torch.manual_seed(seed)
    A = (0.5 * torch.randn(M, K, device="cuda")).half()
    B = (0.02 * torch.randn(K, N, device="cuda")).half()
    bias = (0.1 * torch.randn(N, device="cuda")).half()
    return A, B, bias


@pytest.mark.parametrize("label,M,N,K", GROOT_GEMM_SHAPES, ids=_GROOT_IDS)
def test_gemm_fp16_nn_matches_torch(env, gemm, label, M, N, K):
    """fp16_nn contract: D_fp16(M,N) = A_fp16(M,K) @ B_fp16(K,N), all
    row-major, no transpose, FP16 in AND out (the entry exists because the
    ViT/LLM backbone runs fp16, not bf16 — a bf16 store would halve the
    mantissa, which the tight cos gate catches)."""
    torch, _ = env
    A, B, _bias = _groot_operands(torch, M, N, K, seed=40 + M + N + K)
    D = torch.empty(M, N, dtype=torch.half, device="cuda")

    gemm.fp16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                 _stream(torch))
    torch.cuda.synchronize()

    assert D.dtype is torch.half
    ref = (A.float() @ B.float()).half()
    assert _cos(D.float().cpu(), ref.float().cpu()) > COS_GEMM, label
    torch.testing.assert_close(D.float(), ref.float(),
                               atol=ATOL_FP16_GEMM, rtol=RTOL_FP16_GEMM)


@pytest.mark.parametrize("label,M,N,K", GROOT_GEMM_SHAPES, ids=_GROOT_IDS)
def test_gemm_fp8_nn_bias_applies_host_alpha_then_bias(env, gemm, label,
                                                       M, N, K):
    """fp8_nn_bias: D_fp16 = alpha * (A_fp8 @ B_fp8) + bias_fp16, with
    alpha a HOST float (not a device pointer — that is fp8_descale_fp16's
    convention, and mixing the two up is the failure this pair of gates
    separates).

    The reference multiplies the SAME fp8-decoded operands, never the
    pre-quantization fp16 originals: comparing against those would measure
    e4m3's quantization error instead of the GEMM. The bias is added AFTER
    alpha scaling — the ordering matters and is pinned by the reference.
    """
    torch, _ = env
    A, B, bias = _groot_operands(torch, M, N, K, seed=50 + M + N + K)
    A8, sa = _quant_fp8(torch, A)
    B8, sb = _quant_fp8(torch, B)
    alpha = float(sa * sb)
    D = torch.empty(M, N, dtype=torch.half, device="cuda")

    gemm.fp8_nn_bias(A8.data_ptr(), B8.data_ptr(), D.data_ptr(),
                     bias.data_ptr(), M, N, K, alpha, _stream(torch))
    torch.cuda.synchronize()

    assert D.dtype is torch.half
    prod = A8.float() @ B8.float()
    ref = (alpha * prod + bias.float()).half()
    assert _cos(D.float().cpu(), ref.float().cpu()) > COS_GEMM, label
    torch.testing.assert_close(D.float(), ref.float(),
                               atol=ATOL_FP16_GEMM, rtol=RTOL_FP16_GEMM)
    # The bias must actually land. A cosine cannot say so here — alpha*prod
    # alone is 0.95-0.99 correlated with the reference on these shapes —
    # so the discriminator is elementwise: the un-biased form must fall
    # OUTSIDE the same tolerance the parity check just passed inside.
    assert not torch.allclose(D.float(), (alpha * prod).float(),
                              atol=ATOL_FP16_GEMM, rtol=RTOL_FP16_GEMM), (
        "fp8_nn_bias did not apply the bias epilogue")


@pytest.mark.parametrize("label,M,N,K", GROOT_GEMM_SHAPES, ids=_GROOT_IDS)
def test_gemm_fp8_nn_gelu_bias_is_tanh_approx(env, gemm, label, M, N, K):
    """fp8_nn_gelu_bias: D_fp16 = GELU(alpha * A_fp8 @ B_fp8 + bias).
    The GELU is the tanh approximation (hipBLASLt GELU_BIAS); erf-GELU or
    a missing activation both fail against this reference, and the extra
    discriminator pins that the activation ran at all."""
    torch, _ = env
    A, B, bias = _groot_operands(torch, M, N, K, seed=60 + M + N + K)
    A8, sa = _quant_fp8(torch, A)
    B8, sb = _quant_fp8(torch, B)
    alpha = float(sa * sb)
    D = torch.empty(M, N, dtype=torch.half, device="cuda")

    gemm.fp8_nn_gelu_bias(A8.data_ptr(), B8.data_ptr(), D.data_ptr(),
                          bias.data_ptr(), M, N, K, alpha, _stream(torch))
    torch.cuda.synchronize()

    assert D.dtype is torch.half
    pre = alpha * (A8.float() @ B8.float()) + bias.float()
    ref = torch.nn.functional.gelu(pre, approximate="tanh").half()
    assert _cos(D.float().cpu(), ref.float().cpu()) > COS_GEMM, label
    torch.testing.assert_close(D.float(), ref.float(),
                               atol=ATOL_FP16_GEMM, rtol=RTOL_FP16_GEMM)
    assert _cos(D.float().cpu(), pre.half().float().cpu()) < 0.999, (
        "GELU epilogue was not applied")


@pytest.mark.parametrize("label,M,N,K", GROOT_GEMM_SHAPES, ids=_GROOT_IDS)
def test_gemm_fp8_descale_fp16_matches_quantized_reference(env, gemm, label,
                                                           M, N, K):
    """fp8_descale_fp16: D_fp16 = s_a * s_b * (A_fp8 @ B_fp8) with the
    descales read from DEVICE float pointers (A/B_SCALE_POINTER semantics)
    — same scale contract as fp8_nn_dev, only the output dtype differs.
    Reference over the same fp8-decoded operands."""
    torch, _ = env
    A, B, _bias = _groot_operands(torch, M, N, K, seed=70 + M + N + K)
    A8, sa = _quant_fp8(torch, A)
    B8, sb = _quant_fp8(torch, B)
    dsa = sa.reshape(1).contiguous().cuda()
    dsb = sb.reshape(1).contiguous().cuda()
    D = torch.empty(M, N, dtype=torch.half, device="cuda")

    gemm.fp8_descale_fp16(A8.data_ptr(), B8.data_ptr(), D.data_ptr(),
                          M, N, K, dsa.data_ptr(), dsb.data_ptr(),
                          _stream(torch))
    torch.cuda.synchronize()

    assert D.dtype is torch.half
    ref = (float(sa) * float(sb) * (A8.float() @ B8.float())).half()
    assert _cos(D.float().cpu(), ref.float().cpu()) > COS_GEMM, label
    torch.testing.assert_close(D.float(), ref.float(),
                               atol=ATOL_FP16_GEMM, rtol=RTOL_FP16_GEMM)


def test_gemm_fp8_descale_fp16_reads_the_scales_at_launch(env, gemm):
    """The descales must be READ FROM THE DEVICE POINTER when the kernel
    runs, not captured into the cached descriptor at first use.

    This is the property HIP-Graph replay depends on: a captured graph
    replays the same pointers, and the pipeline updates the calibrated
    scale IN PLACE between replays. The gate mutates the scale tensor in
    place (never rebinding the pointer) and requires the output to track
    it — a descriptor that baked in the first value would return the old
    result and pass every static parity test above.
    """
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(80)
    A8 = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    B8 = (0.3 * torch.randn(K, N, device="cuda")).to(torch.float8_e4m3fn)
    # Magnitudes chosen so both arms stay well inside fp16 normal range.
    dsa = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    dsb = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    D1 = torch.empty(M, N, dtype=torch.half, device="cuda")
    D2 = torch.empty(M, N, dtype=torch.half, device="cuda")

    gemm.fp8_descale_fp16(A8.data_ptr(), B8.data_ptr(), D1.data_ptr(),
                          M, N, K, dsa.data_ptr(), dsb.data_ptr(),
                          _stream(torch))
    torch.cuda.synchronize()

    dsa.fill_(0.2)                      # in place: same device pointer
    gemm.fp8_descale_fp16(A8.data_ptr(), B8.data_ptr(), D2.data_ptr(),
                          M, N, K, dsa.data_ptr(), dsb.data_ptr(),
                          _stream(torch))
    torch.cuda.synchronize()

    torch.testing.assert_close(D2.float(), 2.0 * D1.float(),
                               atol=1e-3, rtol=1e-2)
    assert not torch.allclose(D2.float(), D1.float(), atol=1e-4), (
        "output did not change when the device scale did — the descale is "
        "baked into the descriptor instead of read at launch")


# ---------------------------------------------------------------------------
# GemmRunner — MXFP4 (OCP MX: E2M1 elements + per-1x32-block UE8M0 scales)
# ---------------------------------------------------------------------------
#
# Host-side pack reference, matching what mxfp4_nt_dev consumes directly
# (csrc/amd/gemm/hipblaslt_runner.h):
#     packed : uint8 (rows, K/2)   element 2i low nibble, 2i+1 high nibble
#     scales : uint8 (rows, K/32)  UE8M0, value = 2^(byte - 127)
# Scale rule is the OCP MX v1.0 one: E = floor(log2(amax)) - emax(E2M1),
# emax(E2M1) = 2, computed exactly via frexp (no log2 rounding hazard).

_E2M1_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_MAX = 6.0
_E2M1_EMAX = 2
_MX_BLOCK = 32
_UE8M0_BIAS = 127


def _mxfp4_quantize(torch, x):
    """(rows, K) -> (packed uint8 (rows, K/2), scales uint8 (rows, K/32))."""
    rows, K = x.shape
    assert K % _MX_BLOCK == 0
    xf = x.reshape(rows, K // _MX_BLOCK, _MX_BLOCK).float()
    amax = xf.abs().amax(dim=-1)
    _, e = torch.frexp(amax)                  # amax = m * 2^e, m in [0.5, 1)
    E = e.to(torch.int32) - 1 - _E2M1_EMAX
    E = torch.where(amax == 0, torch.zeros_like(E), E)
    E = E.clamp(-_UE8M0_BIAS, _UE8M0_BIAS)
    scales_u8 = (E + _UE8M0_BIAS).to(torch.uint8)
    scale = torch.ldexp(torch.ones_like(E, dtype=torch.float32), E)

    q = xf / scale.unsqueeze(-1)
    aq = q.abs().clamp(max=_E2M1_MAX)         # spec-sanctioned saturation
    mids = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    idx = torch.bucketize(aq, mids, right=True)
    tie = (idx > 0) & (aq == mids[(idx - 1).clamp(min=0)])
    idx = torch.where(tie & (idx % 2 == 1), idx - 1, idx)   # ties-to-even
    sign = torch.signbit(q).to(torch.uint8)
    nib = (idx.to(torch.uint8) | (sign << 3)).reshape(rows, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous(), scales_u8.contiguous()


def _mxfp4_dequantize(torch, packed, scales):
    """Exact inverse of _mxfp4_quantize -> float32 (rows, K)."""
    rows, half_k = packed.shape
    K = half_k * 2
    nib = torch.empty(rows, K, dtype=torch.uint8, device=packed.device)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = packed >> 4
    codes = torch.tensor(_E2M1_CODES, device=packed.device,
                         dtype=torch.float32)
    mag = codes[(nib & 0x7).long()]
    val = torch.where((nib & 0x8) != 0, -mag, mag)
    E = scales.to(torch.int32) - _UE8M0_BIAS
    scale = torch.ldexp(torch.ones_like(E, dtype=torch.float32), E)
    val = val.reshape(rows, K // _MX_BLOCK, _MX_BLOCK) * scale.unsqueeze(-1)
    return val.reshape(rows, K)


def test_mxfp4_pack_reference_is_self_consistent(env):
    """The MXFP4 GEMM gate below is only as good as this host packer, so it
    is validated first, without touching the GPU kernel: the packed nibble
    stream must round-trip, dequantize(quantize(x)) must be an idempotent
    fixed point (proving the RNE-with-ties-to-even codebook search has no
    off-by-one), and the UE8M0 NaN byte 255 must never be emitted."""
    torch, _ = env
    torch.manual_seed(90)
    for rows, K in ((4, 64), (41, 1536), (64, 1024)):
        x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
        x[::max(rows // 4, 1)] *= 20.0        # force distinct block scales
        packed, scales = _mxfp4_quantize(torch, x)
        assert packed.shape == (rows, K // 2)
        assert scales.shape == (rows, K // _MX_BLOCK)
        assert not bool((scales == 255).any()), "emitted the e8m0 NaN byte"
        assert scales.unique().numel() > 1, "all blocks got the same scale"
        dq = _mxfp4_dequantize(torch, packed, scales)
        p2, s2 = _mxfp4_quantize(torch, dq)
        assert torch.equal(packed, p2) and torch.equal(scales, s2), (
            "MXFP4 quantize/dequantize is not idempotent")


@pytest.mark.parametrize("M,N,K", [(64, 512, 1024), (41, 1536, 1536)])
def test_gemm_mxfp4_nt_dev_matches_dequantized_reference(env, gemm, M, N, K):
    """mxfp4_nt_dev: D_bf16 = A_fp4(M,K) @ B_fp4(N,K)^T with per-1x32-block
    UE8M0 scales installed via A/B_SCALE_MODE = VEC32_UE8M0.

    The reference matmuls the DEQUANTIZED operands — the exact values the
    hardware sees — so this gates the GEMM and the block-scale plumbing,
    not MXFP4's quantization loss. (cos against the unquantized fp32
    matmul sits around 0.99 on randn data: a gate at that level would pass
    even with the block scales wired wrong.) B is (N,K): the NT layout is
    part of the contract and a swapped layout decorrelates.
    """
    torch, _ = env
    torch.manual_seed(91)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    Bt = (0.5 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
    A_p, A_s = _mxfp4_quantize(torch, A)
    B_p, B_s = _mxfp4_quantize(torch, Bt)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    try:
        gemm.mxfp4_nt_dev(A_p.data_ptr(), A_s.data_ptr(),
                          B_p.data_ptr(), B_s.data_ptr(), D.data_ptr(),
                          M, N, K, _stream(torch))
        torch.cuda.synchronize()
    except RuntimeError as exc:
        # hipBLASLt without MXFP4 a4w4 support answers "no algorithm found";
        # that is an environment capability, not a kernel regression.
        if "no algorithm" not in str(exc):
            raise
        pytest.skip(f"hipBLASLt has no MXFP4 algorithm here: {exc}")

    ref = (_mxfp4_dequantize(torch, A_p, A_s)
           @ _mxfp4_dequantize(torch, B_p, B_s).t())
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM_FP4
    assert bool(torch.isfinite(D.float()).all())


# ---------------------------------------------------------------------------
# GemmRunner — autotune entry points
# ---------------------------------------------------------------------------
#
# Autotuning selects a different hipBLASLt algorithm (different split-K,
# tile, and epilogue fusion) and caches it for the shape. That MUST NOT
# change what the entry computes — a tuned pick that reorders K into a
# different split-K count is fine, one that silently changes layout or
# drops an epilogue is not. Each gate therefore tunes on a FRESH runner
# (so it cannot inherit or poison the module-scoped runner's cache) and
# then re-checks parity through the normal inference entry.


def _fresh_runner(ext):
    return ext.GemmRunner()


def test_autotune_bf16_nn_preserves_semantics(env):
    """autotune_bf16_nn times num_algos candidates and caches the winner
    for (BF16_NN, M, N, K). The tuned pick must still compute A @ B in the
    NN layout — a candidate with a different split-K count is fine, one
    that changes the layout or the accumulate type is not."""
    torch, ext = env
    runner = _fresh_runner(ext)
    M, N, K = 64, 512, 1024
    torch.manual_seed(100)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    runner.autotune_bf16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                            M, N, K, 4)
    torch.cuda.synchronize()
    runner.bf16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                   _stream(torch))
    torch.cuda.synchronize()

    ref = A.float() @ B.float()
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM


def test_autotune_fp16_nn_preserves_semantics(env):
    """autotune_fp16_nn on a real GROOT DiT shape: the tuned FP16 algorithm
    must keep both the NN layout and the FP16 output dtype (a candidate
    that promoted the store to bf16 would halve the mantissa and slip past
    any gate that only looked at magnitudes)."""
    torch, ext = env
    runner = _fresh_runner(ext)
    M, N, K = 41, 1536, 1536
    A, B, _bias = _groot_operands(torch, M, N, K, seed=101)
    D = torch.empty(M, N, dtype=torch.half, device="cuda")

    runner.autotune_fp16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                            M, N, K, 4)
    torch.cuda.synchronize()
    runner.fp16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                   _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).half()
    assert _cos(D.float().cpu(), ref.float().cpu()) > COS_GEMM


def test_autotune_fp8_nn_dev_preserves_semantics(env):
    """autotune_fp8_nn_dev tunes with the DEVICE scale pointers in place;
    the cached algorithm must still apply scale_a * scale_b to the FP32
    accumulator afterwards, so the gate re-checks the scaled reference
    through the normal inference entry."""
    torch, ext = env
    runner = _fresh_runner(ext)
    M, N, K = 64, 512, 1024
    torch.manual_seed(102)
    A8 = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    B8 = (0.3 * torch.randn(K, N, device="cuda")).to(torch.float8_e4m3fn)
    dsa = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    dsb = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    runner.autotune_fp8_nn_dev(A8.data_ptr(), B8.data_ptr(), D.data_ptr(),
                               M, N, K, dsa.data_ptr(), dsb.data_ptr(), 4)
    torch.cuda.synchronize()
    runner.fp8_nn_dev(A8.data_ptr(), B8.data_ptr(), D.data_ptr(), M, N, K,
                      dsa.data_ptr(), dsb.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    ref = (A8.float() @ B8.float()) * (dsa * dsb)
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM


def test_autotune_fp8_nt_dev_preserves_semantics(env):
    """autotune_fp8_nt_dev, same contract as the NN tune plus the NT
    transpose: B stays (N,K) row-major and the tuned pick must keep
    D = A @ B^T. This is the production pi05 FP8 layout, so a tuned
    candidate that silently transposed would corrupt every decoder GEMM."""
    torch, ext = env
    runner = _fresh_runner(ext)
    M, N, K = 64, 512, 1024
    torch.manual_seed(103)
    A8 = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    Bt8 = (0.3 * torch.randn(N, K, device="cuda")).to(torch.float8_e4m3fn)
    dsa = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    dsb = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    runner.autotune_fp8_nt_dev(A8.data_ptr(), Bt8.data_ptr(), D.data_ptr(),
                               M, N, K, dsa.data_ptr(), dsb.data_ptr(), 4)
    torch.cuda.synchronize()
    runner.fp8_nt_dev(A8.data_ptr(), Bt8.data_ptr(), D.data_ptr(), M, N, K,
                      dsa.data_ptr(), dsb.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    ref = (A8.float() @ Bt8.float().t()) * (dsa * dsb)
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM


def test_autotune_fp8_descale_fp16_preserves_semantics(env):
    """autotune_fp8_descale_fp16 on the widest GROOT DiT shape (K=6144,
    where split-K candidates differ most): the tuned algorithm must keep
    both the device-descale semantics and the FP16 output."""
    torch, ext = env
    runner = _fresh_runner(ext)
    M, N, K = 41, 1536, 6144
    A, B, _bias = _groot_operands(torch, M, N, K, seed=104)
    A8, sa = _quant_fp8(torch, A)
    B8, sb = _quant_fp8(torch, B)
    dsa = sa.reshape(1).contiguous().cuda()
    dsb = sb.reshape(1).contiguous().cuda()
    D = torch.empty(M, N, dtype=torch.half, device="cuda")

    runner.autotune_fp8_descale_fp16(A8.data_ptr(), B8.data_ptr(),
                                     D.data_ptr(), M, N, K,
                                     dsa.data_ptr(), dsb.data_ptr(), 4)
    torch.cuda.synchronize()
    runner.fp8_descale_fp16(A8.data_ptr(), B8.data_ptr(), D.data_ptr(),
                            M, N, K, dsa.data_ptr(), dsb.data_ptr(),
                            _stream(torch))
    torch.cuda.synchronize()

    ref = (float(sa) * float(sb) * (A8.float() @ B8.float())).half()
    assert _cos(D.float().cpu(), ref.float().cpu()) > COS_GEMM


def test_autotune_mxfp4_nt_dev_preserves_semantics(env):
    """autotune_mxfp4_nt_dev: the tuned MXFP4 algorithm must keep the
    per-1x32-block UE8M0 scale plumbing wired to the right operand. Skips
    when hipBLASLt exposes no MXFP4 algorithm at all (an environment
    capability, not a kernel regression)."""
    torch, ext = env
    runner = _fresh_runner(ext)
    M, N, K = 64, 512, 1024
    torch.manual_seed(105)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    Bt = (0.5 * torch.randn(N, K, device="cuda")).to(torch.bfloat16)
    A_p, A_s = _mxfp4_quantize(torch, A)
    B_p, B_s = _mxfp4_quantize(torch, Bt)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    try:
        runner.autotune_mxfp4_nt_dev(A_p.data_ptr(), A_s.data_ptr(),
                                     B_p.data_ptr(), B_s.data_ptr(),
                                     D.data_ptr(), M, N, K, 4)
        torch.cuda.synchronize()
        runner.mxfp4_nt_dev(A_p.data_ptr(), A_s.data_ptr(),
                            B_p.data_ptr(), B_s.data_ptr(), D.data_ptr(),
                            M, N, K, _stream(torch))
        torch.cuda.synchronize()
    except RuntimeError as exc:
        if "no algorithm" not in str(exc):
            raise
        pytest.skip(f"hipBLASLt has no MXFP4 algorithm here: {exc}")

    ref = (_mxfp4_dequantize(torch, A_p, A_s)
           @ _mxfp4_dequantize(torch, B_p, B_s).t())
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM_FP4


def test_enable_lazy_autotune_preserves_semantics(env):
    """enable_lazy_autotune arms a timed selection on the FIRST call of
    each (type, M, N, K). The header warns that the timed loop re-runs the
    op with the caller's real pointers, so a lazily tuned FIRST call must
    not be trusted for numerics — but every call after it must be exact.
    That is the contract the pipeline relies on (warm every shape eagerly,
    then capture), so the gate checks the SECOND call, on a fresh runner
    so the module-scoped one keeps its default configuration."""
    torch, ext = env
    runner = _fresh_runner(ext)
    runner.enable_lazy_autotune(4)
    M, N, K = 64, 512, 1024
    torch.manual_seed(106)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    runner.bf16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                   _stream(torch))          # warmup: triggers the timed pick
    torch.cuda.synchronize()
    D.zero_()
    runner.bf16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                   _stream(torch))          # the gated call
    torch.cuda.synchronize()

    ref = A.float() @ B.float()
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)


# ---------------------------------------------------------------------------
# Decoder GQA attention
# ---------------------------------------------------------------------------

# The one production site (csrc/amd/attention/decoder_flash.hip): Sq ~10
# action tokens, Hq=8 query heads, single KV head, D=256, Skv ~590-876.
_SQ, _HQ, _D = 10, 8, 256
_SCALE = 1.0 / math.sqrt(_D)  # 0.0625, the binding default


def _attn_ref(torch, Q, K, V, skv, scale):
    """Exact softmax reference in fp64 (bidirectional GQA Hq:1)."""
    q = Q.double().permute(1, 0, 2)          # (Hq, Sq, D)
    k = K[:skv].double()                     # (Skv, D)
    v = V[:skv].double()
    scores = q @ k.t() * scale               # (Hq, Sq, Skv)
    probs = torch.softmax(scores, dim=-1)
    o = probs @ v                            # (Hq, Sq, D)
    return o.permute(1, 0, 2).contiguous()   # (Sq, Hq, D)


def test_attention_decoder_gqa_exact_mode(env):
    torch, ext = env
    skv = 590
    torch.manual_seed(9)
    Q = (0.5 * torch.randn(_SQ, _HQ, _D, device="cuda")).to(torch.bfloat16)
    K = (0.5 * torch.randn(skv, _D, device="cuda")).to(torch.bfloat16)
    V = (0.5 * torch.randn(skv, _D, device="cuda")).to(torch.bfloat16)
    O = torch.empty_like(Q)
    # Caller-owned fp32 scratch, >= 32*Hq*Sq*(D+2) floats (binding contract;
    # the fused v3 path ignores it but the v2 split path requires it).
    ws = torch.empty(32 * _HQ * _SQ * (_D + 2),
                     dtype=torch.float32, device="cuda")

    ext.attention_decoder_gqa(Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                              O.data_ptr(), ws.data_ptr(),
                              _SQ, skv, _HQ, _D, 0, _SCALE, _stream(torch))
    torch.cuda.synchronize()

    ref = _attn_ref(torch, Q, K, V, skv, _SCALE)
    assert _cos(O.float().cpu(), ref.cpu()) > COS_ATTN
    torch.testing.assert_close(O.double(), ref, atol=0.05, rtol=0.05)


def test_attention_decoder_gqa_seqused_masks_padding(env):
    """Fixed-shape graph mode: seqused is a DEVICE int32 whose [0] is the
    runtime KV length; the host Skv argument is ignored. The padded tail is
    filled with large garbage so any masking failure moves the output far
    outside the gate instead of hiding inside it."""
    torch, ext = env
    skv_pad, skv_eff = 640, 590
    torch.manual_seed(10)
    Q = (0.5 * torch.randn(_SQ, _HQ, _D, device="cuda")).to(torch.bfloat16)
    K = (0.5 * torch.randn(skv_pad, _D, device="cuda")).to(torch.bfloat16)
    V = (0.5 * torch.randn(skv_pad, _D, device="cuda")).to(torch.bfloat16)
    K[skv_eff:] = 30.0   # poison rows: huge scores if the mask leaks
    V[skv_eff:] = -30.0
    O = torch.empty_like(Q)
    ws = torch.empty(32 * _HQ * _SQ * (_D + 2),
                     dtype=torch.float32, device="cuda")
    seqused = torch.tensor([skv_eff], dtype=torch.int32, device="cuda")

    ext.attention_decoder_gqa(Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                              O.data_ptr(), ws.data_ptr(),
                              _SQ, skv_pad, _HQ, _D,
                              seqused.data_ptr(), _SCALE, _stream(torch))
    torch.cuda.synchronize()

    ref = _attn_ref(torch, Q, K, V, skv_eff, _SCALE)
    assert _cos(O.float().cpu(), ref.cpu()) > COS_ATTN
    torch.testing.assert_close(O.double(), ref, atol=0.05, rtol=0.05)
