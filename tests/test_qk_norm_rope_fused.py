"""Validation of the fused QK-LayerNorm + rotate-half RoPE FP16 kernel."""

import pytest

fvk_torch = pytest.importorskip("torch")

import torch  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"flash_rt_kernels is not built: {exc}", allow_module_level=True)

if not hasattr(fvk, "qk_norm_rope_fused_fp16"):
    pytest.skip(
        "qk_norm_rope_fused_fp16 requires FLASHRT_ENABLE_CHAMELEON",
        allow_module_level=True,
    )


HD = 128  # the only head_dim the RoPE writeback currently supports


def _layer_norm_rows(x, w, b, eps, num_heads):
    """Per-head LayerNorm in fp32: (x - mean) * rsqrt(var + eps) * w + b.

    The kernel treats q/k as [Se * num_heads, HD] rows and normalizes each
    HD-length row independently (params shared across heads). Matches the
    kernel's biased variance (divide by N) and returns the result rounded to
    fp16, because the fused kernel stores the normalized values back to half
    precision before applying RoPE.
    """
    se, width = x.shape
    xf = x.float().view(se * num_heads, HD)
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    normed = (xf - mean) * inv_std * w.float() + b.float()
    return normed.view(se, width).half()


def _rotate_half_rope(x_normed, cos_table, sin_table, num_heads):
    """rotate_half RoPE with [Se, HD] cos/sin tiled over both halves.

    x_normed: [Se, NH*HD] fp16. Returns [Se, NH*HD] fp16 computed in fp32.
    cos/sin: [Se, HD] fp16, where the second half duplicates the first
    (cat([c, c], dim=-1)) to match the kernel's tiled tables.
    """
    se = x_normed.shape[0]
    x = x_normed.float().view(se, num_heads, HD)
    half = HD // 2
    x1, x2 = x[..., :half], x[..., half:]

    # Broadcast the per-seq-position tables over all heads: [Se, 1, HD].
    cos = cos_table.float().unsqueeze(1)
    sin = sin_table.float().unsqueeze(1)
    c1, c2_ = cos[..., :half], cos[..., half:]
    s1, s2_ = sin[..., :half], sin[..., half:]

    # Kernel math:
    #   out[d]      = x[d]      * cos[d]      - x[d+HD/2] * sin[d]
    #   out[d+HD/2] = x[d+HD/2] * cos[d+HD/2] + x[d]      * sin[d+HD/2]
    out_lo = x1 * c1 - x2 * s1
    out_hi = x2 * c2_ + x1 * s2_
    out = torch.cat([out_lo, out_hi], dim=-1)
    return out.view(se, num_heads * HD).half()


def _reference(q, k, q_w, q_b, k_w, k_b, cos_t, sin_t, num_heads, eps):
    qn = _layer_norm_rows(q, q_w, q_b, eps, num_heads)
    kn = _layer_norm_rows(k, k_w, k_b, eps, num_heads)
    qr = _rotate_half_rope(qn, cos_t, sin_t, num_heads)
    kr = _rotate_half_rope(kn, cos_t, sin_t, num_heads)
    return qr, kr


def _cosine(a, b):
    a = a.float().flatten().double()
    b = b.float().flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def _run_case(seq_len, num_heads, eps=1e-5):
    torch.manual_seed(0)
    width = num_heads * HD
    q = (torch.randn(seq_len, width, device="cuda") * 2.0).half()
    k = (torch.randn(seq_len, width, device="cuda") * 2.0).half()
    q_w = (torch.randn(HD, device="cuda") * 0.1 + 1.0).half()
    q_b = (torch.randn(HD, device="cuda") * 0.1).half()
    k_w = (torch.randn(HD, device="cuda") * 0.1 + 1.0).half()
    k_b = (torch.randn(HD, device="cuda") * 0.1).half()

    # rotate_half-tiled tables: cat([c, c], dim=-1) over the half-rotation.
    pos = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (
        torch.arange(0, HD // 2, device="cuda", dtype=torch.float32) / (HD // 2)))
    ang = torch.outer(pos, freqs)  # [Se, HD/2]
    cos_half = torch.cos(ang)
    sin_half = torch.sin(ang)
    cos_t = torch.cat([cos_half, cos_half], dim=-1).half()
    sin_t = torch.cat([sin_half, sin_half], dim=-1).half()

    q_ref, k_ref = _reference(q, k, q_w, q_b, k_w, k_b, cos_t, sin_t,
                              num_heads, eps)

    q_in, k_in = q.clone(), k.clone()
    fvk.qk_norm_rope_fused_fp16(
        q_in.data_ptr(), k_in.data_ptr(),
        q_w.data_ptr(), q_b.data_ptr(), k_w.data_ptr(), k_b.data_ptr(),
        cos_t.data_ptr(), sin_t.data_ptr(),
        seq_len, num_heads, HD, eps, 0)
    torch.cuda.synchronize()

    return q_in, k_in, q_ref, k_ref


@pytest.mark.parametrize("seq_len", [1, 7, 64])
@pytest.mark.parametrize("num_heads", [1, 4])
def test_qk_norm_rope_matches_reference(seq_len, num_heads):
    q_out, k_out, q_ref, k_ref = _run_case(seq_len, num_heads)

    cos_q = _cosine(q_out, q_ref)
    cos_k = _cosine(k_out, k_ref)
    assert cos_q >= 0.999, f"Q cosine {cos_q:.6f} below 0.999 " \
        f"(seq={seq_len}, heads={num_heads})"
    assert cos_k >= 0.999, f"K cosine {cos_k:.6f} below 0.999 " \
        f"(seq={seq_len}, heads={num_heads})"

    assert torch.allclose(q_out.float(), q_ref.float(), atol=2e-2, rtol=1e-2), \
        "Q mismatch beyond fp16 tolerance"
    assert torch.allclose(k_out.float(), k_ref.float(), atol=2e-2, rtol=1e-2), \
        "K mismatch beyond fp16 tolerance"


def _make_args(seq_len, num_heads, dim, eps=1e-5):
    """Build a valid argument tuple so contract checks only trip their
    targeted validation clause. Data pointers may reference dummies because
    py::value_error is thrown before any CUDA work is launched."""
    width = num_heads * max(dim, HD)
    q = torch.zeros(seq_len, width, device="cuda", dtype=torch.float16)
    k = torch.zeros(seq_len, width, device="cuda", dtype=torch.float16)
    w = torch.zeros(dim, device="cuda", dtype=torch.float16)
    b = torch.zeros(dim, device="cuda", dtype=torch.float16)
    cos_t = torch.zeros(seq_len, dim, device="cuda", dtype=torch.float16)
    sin_t = torch.zeros(seq_len, dim, device="cuda", dtype=torch.float16)
    return (q.data_ptr(), k.data_ptr(), w.data_ptr(), b.data_ptr(),
            w.data_ptr(), b.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
            seq_len, num_heads, dim, eps, 0)


def test_contract_rejects_unsupported_dim():
    args = list(_make_args(4, 2, HD))
    args[10] = 256  # dim slot -> unsupported head_dim
    with pytest.raises(ValueError):
        fvk.qk_norm_rope_fused_fp16(*args)


def test_contract_rejects_zero_seq_len():
    with pytest.raises(ValueError):
        fvk.qk_norm_rope_fused_fp16(*_make_args(0, 2, HD))


def test_contract_rejects_zero_eps():
    with pytest.raises(ValueError):
        fvk.qk_norm_rope_fused_fp16(*_make_args(4, 2, HD, eps=0.0))
