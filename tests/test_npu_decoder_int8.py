"""INT8 decoder projections.

Every decoder projection runs at ten action rows, where the cost is the weight
stream, so the weights are frozen to INT8 at setup and the dequant vector folds
the activation scale into the per-output-channel weight scale. These tests pin
the entry points, the numeric floor of the GEMM at the four shipped shapes, and
which of them write their result in fractal NZ.
"""

import pytest


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch
        return bool(torch.npu.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _npu_available(), reason="no usable Ascend NPU on this machine")

# (K, N) of the four projections, and whether the result is written in NZ.
_SHAPES = (("gate_up", 1024, 8192, False),
           ("qkv", 1024, 2560, False),
           ("out_proj", 2048, 1024, True),
           ("down_proj", 4096, 1024, True))
_ROWS = 10
_STEPS = 3


def test_the_native_entry_points_are_present():
    """A missing symbol has to fail here rather than inside a captured graph."""
    from flash_rt.npu.core.decoder_int8 import DecoderGemmLibrary, DecoderQuantKernels

    gemm = DecoderGemmLibrary()
    assert gemm.launch is not None
    kernels = DecoderQuantKernels()
    for name in ("ada", "geglu", "rope"):
        assert getattr(kernels, name) is not None


def _unpack_nz(flat, rows, columns):
    """Fractal NZ back to row major: (r, c) sits at (c//16)*256 + r*16 + c%16."""
    import torch

    c = torch.arange(columns, device=flat.device)
    index = (c // 16) * 256 + (c % 16)
    out = torch.empty(rows, columns, dtype=flat.dtype, device=flat.device)
    for r in range(rows):
        out[r] = flat[index + r * 16]
    return out


@pytest.mark.parametrize("name,k,n,nz", _SHAPES)
def test_the_gemm_reaches_the_fp16_output_floor(name, k, n, nz):
    import torch
    import torch_npu  # noqa: F401
    from flash_rt.npu.core.decoder_int8 import DecoderInt8Projection

    torch.manual_seed(hash(name) % 2 ** 31)
    weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
    activation = torch.randn(_ROWS, k, device="npu", dtype=torch.float32)
    amax = [float(activation.abs().amax()) * (1.0 + 0.1 * s) for s in range(_STEPS)]
    projection = DecoderInt8Projection.bind(weight, amax, _ROWS, nz)

    step = 1
    scale = amax[step] / 127.0
    frozen = (activation / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    result = projection(frozen, step)
    got = _unpack_nz(result, _ROWS, n) if nz else result

    weight_scale = (weight.float().abs().amax(dim=1) / 127.0).clamp_min(1e-8)
    packed = (weight.float() / weight_scale[:, None]).round().clamp(-127, 127)
    want = (frozen.float() @ packed.t()) * (weight_scale * scale)[None, :]

    a, b = got.float().reshape(-1), want.reshape(-1)
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm()))
    assert cosine > 0.99999, f"{name}: cosine {cosine}"
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("name,k,n,nz", _SHAPES)
def test_the_result_buffer_matches_the_declared_layout(name, k, n, nz):
    """An NZ result is one flat run of 16-row blocks, not a matrix."""
    import torch
    from flash_rt.npu.core.decoder_int8 import DecoderInt8Projection

    weight = torch.zeros(n, k, device="npu", dtype=torch.bfloat16)
    weight[:, 0] = 1.0
    projection = DecoderInt8Projection.bind(weight, [1.0] * _STEPS, _ROWS, nz)
    assert projection.nz is nz
    assert projection.out.shape == ((16 * n,) if nz else (_ROWS, n))


def test_only_the_narrow_projections_write_nz():
    """At 1024 columns a row is four 512-byte regions, so handing whole regions
    to twenty cores is impossible; at 2560 and 8192 it costs nothing."""
    import inspect
    from flash_rt.npu.core import decoder_int8

    source = inspect.getsource(decoder_int8.DecoderInt8Pack.build)
    assert '"qkv": False' in source and '"gu": False' in source
    assert '"self_attn.o_proj": True' in source and '"mlp.down_proj": True' in source


def test_the_nz_write_covers_the_whole_fractal_block():
    """Writing only the live rows of a 16-row block threads gaps through it, and
    that partial write is not safe between cores."""
    import inspect
    import pathlib
    from flash_rt.npu.core import decoder_int8

    root = pathlib.Path(inspect.getfile(decoder_int8)).parents[3]
    source = (root / "csrc/npu/kernels/decoder_gemm_910b.cpp").read_text()
    branch = source.index("if (nzout) {")
    fixpipe = source.index("CFG_NZ", branch)
    assert "fp.mSize = (uint16_t)M16;" in source[branch:fixpipe]


def test_the_column_tile_keeps_the_l0b_slot_at_half_the_buffer():
    from flash_rt.npu.core.decoder_int8 import column_tile

    for _, k, _, _ in _SHAPES:
        columns, depth = column_tile(k)
        assert columns == 32
        assert k % depth == 0
        assert columns * depth <= 32768
        if k // depth > 1:
            assert columns * depth == 32768
    with pytest.raises(ValueError):
        column_tile(48)
