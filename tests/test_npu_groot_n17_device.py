"""Numerical smokes for the three GR00T N1.7 Ascend kernels, on the part.

These need an Ascend 910B4 and the shared objects built with
``FLASHRT_ENABLE_NPU_GROOT_N17=ON bash scripts/npu/build.sh``; they skip
otherwise. Everything that can be judged without the part is in
``test_npu_groot_n17_units.py`` and ``test_npu_groot_n17_actions.py`` — this file
is only the arithmetic that has to run on the cube and the vector cores.

Each kernel is compared against an independent reference at the exact shapes the
model runs, not at a convenient one: the DiT's three attention geometries, the
action head's 41-row norm, and the evaluation transform's two resizes.
"""
import numpy
import pytest
import torch

INT_MAX = 2147483647


def _device_or_skip():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("torch_npu is not installed")
    if not torch.npu.is_available():
        pytest.skip("no Ascend device")
    return "npu:0"


def _built_or_skip(loader):
    try:
        loader()
    except ImportError as exc:
        pytest.skip(f"the unit is not built: {exc}")


def _cosine(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float(a @ b / (a.norm() * b.norm()))


# ── the DiT's attention ───────────────────────────────────────────────

#: heads, queries, keys, head_dim -- the DiT's self, text-cross and image-cross
#: sites, which are the only geometries the kernel accepts.
GEOMETRIES = [(32, 41, 41, 48), (32, 41, 13, 48), (32, 41, 448, 48)]


@pytest.mark.parametrize("heads,queries,keys,head_dim", GEOMETRIES)
def test_the_attention_agrees_with_the_vendor_operator(heads, queries, keys, head_dim):
    device = _device_or_skip()
    import torch_npu
    from flash_rt.npu.models.groot_n17.attention import DitAttention, DitAttentionLibrary

    _built_or_skip(DitAttentionLibrary)
    torch.manual_seed(0)
    width = heads * head_dim
    q = torch.randn(queries, width, dtype=torch.bfloat16, device=device)
    k = torch.randn(keys, width, dtype=torch.bfloat16, device=device)
    v = torch.randn(keys, width, dtype=torch.bfloat16, device=device)
    want = torch_npu.npu_prompt_flash_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), num_heads=heads,
        num_key_value_heads=heads, scale_value=head_dim ** -0.5,
        input_layout="BSH", pre_tokens=INT_MAX, next_tokens=INT_MAX,
        sparse_mode=0).reshape(queries, width)

    attention = DitAttention(heads, queries, keys, head_dim, device=device)
    (padded_q, padded_k, padded_vt), (live_q, live_k, live_vt) = attention.buffers()
    live_q.copy_(q)
    live_k.copy_(k)
    live_vt.copy_(v.t())
    got = attention(padded_q, padded_k, padded_vt)[:queries]
    assert _cosine(want.float(), got.float()) > 0.9999


@pytest.mark.parametrize("heads,queries,keys,head_dim", GEOMETRIES)
def test_the_value_transposed_on_the_way_into_l0b_is_bit_identical(
        heads, queries, keys, head_dim):
    """The self-attention sites read query, key and value as three column slices
    of one projection and let the kernel transpose the value. That has to be the
    same result as handing it in transposed, not merely a close one."""
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17.attention import DitAttention, DitAttentionLibrary

    _built_or_skip(DitAttentionLibrary)
    torch.manual_seed(0)
    width = heads * head_dim
    attention = DitAttention(heads, queries, keys, head_dim, device=device)
    q = torch.randn(queries, width, dtype=torch.bfloat16, device=device)
    k = torch.randn(keys, width, dtype=torch.bfloat16, device=device)
    v = torch.randn(keys, width, dtype=torch.bfloat16, device=device)

    (padded_q, padded_k, padded_vt), (live_q, live_k, live_vt) = attention.buffers()
    live_q.copy_(q)
    live_k.copy_(k)
    live_vt.copy_(v.t())
    transposed = attention(padded_q, padded_k, padded_vt)[:queries].clone()

    rows = max(attention.rows, attention.columns)
    fused = torch.zeros(rows, 3 * width, dtype=torch.bfloat16, device=device)
    fused[:queries, :width] = q
    fused[:keys, width:2 * width] = k
    fused[:keys, 2 * width:] = v
    pitch = 3 * width
    in_place = attention(fused[:attention.rows, :width],
                         fused[:attention.columns, width:2 * width],
                         fused[:attention.columns, 2 * width:],
                         pitch, pitch)[:queries]
    assert torch.equal(transposed, in_place)


def test_the_attention_refuses_a_geometry_it_cannot_hold_in_l0():
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17.attention import DitAttention, DitAttentionLibrary

    _built_or_skip(DitAttentionLibrary)
    with pytest.raises(ValueError, match="16-aligned head width"):
        DitAttention(4, 41, 41, 40, device=device)
    attention = DitAttention(4, 41, 1024, 128, device=device)
    (q, k, vt), _ = attention.buffers()
    with pytest.raises(RuntimeError, match="rejected arguments"):
        attention(q, k, vt)


# ── the fused add-and-normalise ───────────────────────────────────────

def test_the_norm_matches_a_reference_that_rounds_where_it_rounds():
    """The sum is rounded to BF16 before it is normalised, because that value is
    what the next block carries forward; the normalisation then runs in FP32 from
    it. A reference that keeps FP32 throughout disagrees, and that difference is
    the whole reason this kernel exists."""
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17 import norm

    _built_or_skip(norm.library)
    torch.manual_seed(0)
    rows, cols = 41, 1536
    residual = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    branch = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(cols, dtype=torch.bfloat16, device=device)
    beta = torch.randn(cols, dtype=torch.bfloat16, device=device)

    got_norm, got_total = norm.add_layer_norm(residual, branch, gamma, beta, 1e-5)
    total = (residual.float() + branch.float()).to(torch.bfloat16)
    assert torch.equal(got_total, total), "the residual sum has to be exact"
    want = torch.nn.functional.layer_norm(
        total.float(), (cols,), gamma.float(), beta.float(), 1e-5).to(torch.bfloat16)
    assert _cosine(want.float(), got_norm.float()) > 0.9999


def test_the_norm_adds_the_branchs_bias_where_the_biased_matmul_would_have():
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17 import norm

    _built_or_skip(norm.library)
    torch.manual_seed(0)
    rows, cols = 41, 1536
    residual = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    branch = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    bias = torch.randn(cols, dtype=torch.bfloat16, device=device)
    unit = torch.ones(cols, dtype=torch.bfloat16, device=device)
    zero = torch.zeros(cols, dtype=torch.bfloat16, device=device)

    _, folded = norm.add_layer_norm(residual, branch, unit, zero, 1e-5,
                                    branch_bias=bias)
    biased = (branch.float() + bias.float()).to(torch.bfloat16)
    _, separate = norm.add_layer_norm(residual, biased, unit, zero, 1e-5)
    assert torch.equal(folded, separate), (
        "the bias has to be added to the branch and rounded there, which is what "
        "the biased matmul it replaced did")


def test_the_norm_writes_the_row_at_a_pitch_and_leaves_the_pad_alone():
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17 import norm

    _built_or_skip(norm.library)
    torch.manual_seed(0)
    rows, cols, pad = 41, 1536, 16
    residual = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    branch = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    unit = torch.ones(cols, dtype=torch.bfloat16, device=device)
    zero = torch.zeros(cols, dtype=torch.bfloat16, device=device)
    out = torch.zeros(1, rows, cols + pad, dtype=torch.bfloat16, device=device)
    out[..., cols] = 1.0                      # the constant the next projection reads

    wide, _ = norm.add_layer_norm(residual, branch, unit, zero, 1e-5, out=out)
    tight, _ = norm.add_layer_norm(residual, branch, unit, zero, 1e-5)
    assert torch.equal(wide[..., :cols], tight)
    assert torch.all(wide[..., cols] == 1.0), "the pad is the caller's"
    assert torch.all(wide[..., cols + 1:] == 0.0)


# ── the evaluation image transform ────────────────────────────────────

def test_the_image_transform_is_bit_exact_against_opencv():
    """The whole transform, not just the taps: a smallest-edge resize, a centre
    crop to 95 percent, and the same resize again, all INTER_AREA on uint8."""
    cv2 = pytest.importorskip("cv2")
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17 import preprocess as pre

    _built_or_skip(pre.AreaResizeLibrary)
    height, width, images = 180, 320, 2
    rng = numpy.random.default_rng(0)
    frames = rng.integers(0, 256, (images, height, width, 3), dtype=numpy.uint8)

    transform = pre.EvalImageTransform(height, width, device=device, images=images)
    got = transform(torch.from_numpy(frames).to(device))

    want = []
    for frame in frames:
        mid_h, mid_w = transform.first.target_h, transform.first.samples // 3
        step = cv2.resize(frame, (mid_w, mid_h), interpolation=cv2.INTER_AREA)
        crop_h = max(1, int(mid_h * 0.95))
        crop_w = max(1, int(mid_w * 0.95))
        top, left = (mid_h - crop_h) // 2, (mid_w - crop_w) // 2
        cropped = step[top:top + crop_h, left:left + crop_w]
        target_h, target_w = transform.target
        want.append(cv2.resize(cropped, (target_w, target_h),
                               interpolation=cv2.INTER_AREA))
    expected = torch.from_numpy(numpy.stack(want)).permute(0, 3, 1, 2)
    assert torch.equal(got.cpu(), expected), (
        f"{int((got.cpu() != expected).sum())} of {expected.numel()} samples differ")


def test_the_image_transform_refuses_a_host_tensor():
    device = _device_or_skip()
    from flash_rt.npu.models.groot_n17 import preprocess as pre

    _built_or_skip(pre.AreaResizeLibrary)
    transform = pre.EvalImageTransform(180, 320, device=device, images=1)
    with pytest.raises(ValueError, match="must be on"):
        transform(torch.zeros(1, 180, 320, 3, dtype=torch.uint8))
