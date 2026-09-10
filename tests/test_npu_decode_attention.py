"""Decode attention over a transposed value cache.

A raw ``Mmad`` B operand wants its GM source in ``(N, K)`` form, so ``O = P * V``
needs V as ``(HD, KV)``. The cache keeps that transpose in fractal NZ, which
makes the value tile for a key block a plain contiguous copy and makes one
sixteen-position block 8 KB of contiguous bytes to write. These tests pin the
entry points, the cache geometry that keeps every write to a whole fractal
block, the transposes against their reference layout, and the numeric floor of
the kernel at the shipped shape.
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

_HD = 256
_PB = 16          # positions in a fractal block
_PLEN = 572       # the shipped prefix: deliberately not a multiple of 16
_CHUNK = 10


def _reference_nz(v, kvp, column):
    """Fractal NZ of ``v.t()`` placed at columns ``[column, column + rows)``."""
    import torch

    out = torch.zeros(kvp // _PB, _HD, _PB, dtype=v.dtype, device=v.device)
    for i in range(v.shape[0]):
        p = column + i
        out[p // _PB, :, p % _PB] = v[i]
    return out.reshape(-1)


def test_the_native_entry_points_are_present():
    """A missing symbol has to fail here rather than inside a captured graph."""
    from flash_rt.npu.core.decode_attention import DecodeAttentionLibrary
    from flash_rt.npu.core.decoder_int8 import DecoderQuantKernels

    assert DecodeAttentionLibrary().launch is not None
    kernels = DecoderQuantKernels()
    for name in ("rope_vt", "vt_prefix"):
        assert getattr(kernels, name) is not None


def test_the_suffix_leads_the_cache_in_whole_fractal_blocks():
    """The action rows are rewritten on every layer of every denoise step, and a
    fractal block holds sixteen positions interleaved at 32-byte granularity, so
    a suffix that straddled two blocks would have cores threading partial writes
    through blocks the encoder prefix also lives in. Placed first it owns block
    zero outright. The prefix length is not a multiple of sixteen, which is what
    makes the order matter rather than a detail.
    """
    from flash_rt.npu.models.pi05.attention import TransposedDecoderAttention

    assert _PLEN % _PB
    geometry = TransposedDecoderAttention.create.__func__
    self = geometry(TransposedDecoderAttention, _PLEN, _CHUNK, device="cpu")
    assert self.column % _PB == 0
    assert self.column >= _CHUNK
    assert self.end == self.column + _PLEN
    assert self.kvp % 64 == 0 and self.kvp >= self.end
    assert self.mq % 16 == 0


def test_the_prefix_transpose_matches_its_reference_layout():
    """Bit for bit: a transpose moves bytes, so anything but equality is a bug
    in the addressing rather than a rounding difference."""
    import torch
    from flash_rt.npu.core.decoder_int8 import DecoderQuantKernels
    from flash_rt.npu.models.pi05.attention import TransposedDecoderAttention

    attention = TransposedDecoderAttention.create(_PLEN, _CHUNK)
    source = (torch.randn(_PLEN, _HD, device="npu") * 0.5).to(torch.bfloat16)
    values = torch.zeros(attention.kvp * _HD, dtype=torch.bfloat16, device="npu")
    DecoderQuantKernels().transpose_prefix(source, values, _PLEN, attention.column)
    torch.npu.synchronize()
    assert torch.equal(values, _reference_nz(source, attention.kvp, attention.column))


def test_the_rotary_writes_the_value_block_whole_and_zeroes_its_tail():
    """The padded positions at the end of block zero are read by the attention
    kernel as ordinary columns, so they have to hold zero rather than whatever
    the buffer had."""
    import torch
    from flash_rt.npu.core.decoder_int8 import DecoderQuantKernels
    from flash_rt.npu.models.pi05.attention import TransposedDecoderAttention

    attention = TransposedDecoderAttention.create(_PLEN, _CHUNK)
    kernels = DecoderQuantKernels()
    slab = (torch.randn(_CHUNK, 10 * _HD, device="npu") * 0.5).to(torch.float16)
    cos = torch.randn(_PLEN + _CHUNK + _PB, _HD, device="npu")
    sin = torch.randn(_PLEN + _CHUNK + _PB, _HD, device="npu")
    query = torch.zeros(_CHUNK, 8 * _HD, dtype=torch.bfloat16, device="npu")
    keys = torch.ones(attention.kvp, _HD, dtype=torch.bfloat16, device="npu")
    values = torch.ones(attention.kvp * _HD, dtype=torch.bfloat16, device="npu")
    kernels.decoder_rope_transposed(slab, cos, sin, query, keys, values, _PLEN)
    torch.npu.synchronize()
    expected = slab.float().reshape(_CHUNK, 10, _HD)[:, 9].to(torch.bfloat16)
    assert torch.equal(values[:_PB * _HD], _reference_nz(expected, _PB, 0))


def test_the_kernel_matches_torch_attention_at_the_shipped_shape():
    """The whole point is a different arrangement of the same arithmetic."""
    import torch
    from flash_rt.npu.core.decoder_int8 import DecoderQuantKernels
    from flash_rt.npu.models.pi05.attention import TransposedDecoderAttention

    torch.manual_seed(0)
    attention = TransposedDecoderAttention.create(_PLEN, _CHUNK)
    kernels = DecoderQuantKernels()
    keys = torch.zeros(attention.kvp, _HD, dtype=torch.bfloat16, device="npu")
    values = torch.zeros(attention.kvp * _HD, dtype=torch.bfloat16, device="npu")
    prefix_k = (torch.randn(_PLEN, _HD, device="npu") * 0.5).to(torch.bfloat16)
    prefix_v = (torch.randn(_PLEN, _HD, device="npu") * 0.5).to(torch.bfloat16)
    keys[attention.column:attention.end].copy_(prefix_k)
    kernels.transpose_prefix(prefix_v, values, _PLEN, attention.column)
    suffix_k = (torch.randn(_CHUNK, _HD, device="npu") * 0.5).to(torch.bfloat16)
    suffix_v = (torch.randn(_CHUNK, _HD, device="npu") * 0.5).to(torch.bfloat16)
    keys[:_CHUNK].copy_(suffix_k)
    values[:_PB * _HD].copy_(_reference_nz(suffix_v, _PB, 0))
    query = (torch.randn(_CHUNK, 8 * _HD, device="npu") * 0.5).to(torch.bfloat16)
    out = attention(query, keys, values)
    torch.npu.synchronize()

    rows = query.reshape(attention.mq, _HD).float()
    key_all = torch.cat([suffix_k, prefix_k], 0).float()
    value_all = torch.cat([suffix_v, prefix_v], 0).float()
    scores = rows @ key_all.t() * (_HD ** -0.5)
    reference = (torch.softmax(scores, dim=-1) @ value_all).reshape(-1)
    got = out.reshape(-1).float()
    assert torch.isfinite(got).all()
    cosine = float(got @ reference / (got.norm() * reference.norm()))
    assert cosine > 0.9999, cosine


def test_the_value_tile_is_read_as_contiguous_bytes():
    """The transposed cache exists so the B operand needs no conversion. An
    Nd2Nz on that copy would mean the layout had been given up and the kernel
    was paying for the transpose twice."""
    import inspect
    import pathlib
    from flash_rt.npu.core import decode_attention

    root = pathlib.Path(inspect.getfile(decode_attention)).parents[3]
    source = (root / "csrc/npu/kernels/decode_attn_910b.cpp").read_text()
    body = source[source.index("WaitEvent(5);"):]
    assert "DataCopy(b1, vg[(uint32_t)t * NT * HD], NT * HD);" in body
    # The value pass loads two operands. The probabilities come from a row-major
    # scratch and convert; the values must not, so exactly one conversion may
    # appear between the softmax barrier and the output fixpipe.
    assert body[:body.index("Fixpipe")].count("Nd2NzParams") == 1


def test_the_padded_columns_of_a_fresh_cache_are_zero():
    """A padded column scores zero and then multiplies a zero value column, so
    the cache cannot be handed out uninitialised."""
    import torch
    from flash_rt.npu.models.pi05 import fast as npu_fast
    from flash_rt.npu.models.pi05.attention import TransposedDecoderAttention

    attention = TransposedDecoderAttention.create(_PLEN, _CHUNK)
    keys, values = npu_fast.make_kv_buffers(_PLEN, _CHUNK, cache=attention)
    assert keys[0].shape == (attention.kvp, _HD)
    assert values[0].shape == (attention.kvp * _HD,)
    assert bool(keys[0].eq(0).all()) and bool(values[0].eq(0).all())


@pytest.mark.parametrize("bad", ("rows", "column", "cache"))
def test_the_prefix_transpose_rejects_a_layout_it_cannot_write(bad):
    """A 16-aligned column and a cache long enough to hold the prefix are what
    make the write land on whole blocks; neither can be assumed."""
    import torch
    from flash_rt.npu.core.decoder_int8 import DecoderQuantKernels

    kernels = DecoderQuantKernels()
    source = torch.zeros(_PLEN, _HD, dtype=torch.bfloat16, device="npu")
    values = torch.zeros(1024 * _HD, dtype=torch.bfloat16, device="npu")
    rows, column = _PLEN, 16
    if bad == "rows":
        rows = 0
    elif bad == "column":
        column = 8
    else:
        values = torch.zeros(_HD, dtype=torch.bfloat16, device="npu")
    with pytest.raises(ValueError):
        kernels.transpose_prefix(source, values, rows, column)
