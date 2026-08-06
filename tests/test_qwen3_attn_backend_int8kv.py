"""Tests for the Qwen3 attention backend's opt-in INT8 KV cache.

Covers the fail-fast contract (missing kernels, unsupported attention
geometry) and proves the feature is inert when left off, which is what keeps
the existing BF16 decode path byte-identical.

CPU-runnable: needs torch but no GPU (device='cpu', num_sms mocked).

Run:
    PYTHONPATH=. python -m pytest tests/test_qwen3_attn_backend_int8kv.py -v
"""
from __future__ import annotations

import importlib
import unittest.mock as mock

import pytest

torch = pytest.importorskip('torch')


BACKEND_MOD = 'flash_rt.hardware.rtx.attn_backend_qwen3'
INT8KV_FNS = ('qwen3_kv_rows_quant_int8',
              'qwen3_attn_decode_int8kv_partial',
              'qwen3_attn_decode_int8kv_combine')
# The geometry the int8-KV decode kernel is specialized for (Qwen3-VL-2B).
DIMS_2B = dict(num_layers=2, num_q_heads=16, num_kv_heads=8, head_dim=128)
# Qwen3-VL-8B: 32Q/8KV — outside the kernel's specialization.
DIMS_8B = dict(num_layers=2, num_q_heads=32, num_kv_heads=8, head_dim=128)


class _Fa2:
    """Minimal stand-in for a fully built flash_rt_fa2."""

    def __init__(self):
        self.calls: list[str] = []
        self.fwd_bf16 = self._rec('fwd_bf16')
        self.fwd_bf16_causal = self._rec('fwd_bf16_causal')

    def _rec(self, name):
        def _fn(**kw):
            self.calls.append(name)
        return _fn


class _Vlk:
    """Stand-in for flash_rt_qwen3_vl_kernels recording kernel calls."""

    def __init__(self, *, drop: str | None = None):
        self.calls: list[tuple[str, tuple]] = []
        for fn in INT8KV_FNS:
            if fn == drop:
                continue
            setattr(self, fn, self._rec(fn))

    def _rec(self, name):
        def _fn(*a, **kw):
            self.calls.append((name, a))
        return _fn


def _make_backend(*, fa2=None, vlk=None, dims=None, **kwargs):
    mod = importlib.import_module(BACKEND_MOD)
    props = mock.Mock()
    props.multi_processor_count = 8
    modules = {'flash_rt.flash_rt_fa2': fa2 or _Fa2()}
    if vlk is not None:
        modules['flash_rt.flash_rt_qwen3_vl_kernels'] = vlk
    with mock.patch.dict('sys.modules', modules), \
            mock.patch.object(torch.cuda, 'get_device_properties',
                              return_value=props):
        return mod.RtxFlashAttnBackendQwen3(
            max_seq=256, max_q_seq=1, device='cpu',
            **(dims or DIMS_2B), **kwargs)


# ── old path: the feature is inert when off ──

def test_disabled_by_default_allocates_no_int8_mirrors():
    backend = _make_backend()
    assert backend._use_int8_kv is False
    for attr in ('K8', 'V8', 'KS', 'VS'):
        assert not hasattr(backend, attr), f'unexpected {attr} allocation'


def test_quantize_helpers_are_no_ops_when_disabled():
    """Frontends call these unconditionally, so they must be safe when off."""
    backend = _make_backend()
    backend.quantize_kv_rows(0, 0)
    backend.quantize_kv_prefix(8)


def test_decode_still_uses_fa2_when_disabled():
    fa2 = _Fa2()
    backend = _make_backend(fa2=fa2)
    backend.run('full', layer_idx=0, q_seq=1, kv_seq=8)
    assert fa2.calls == ['fwd_bf16']


# ── new path ──

def test_enabling_allocates_mirrors_scales_and_partials():
    backend = _make_backend(vlk=_Vlk(), use_int8_kv=True)
    assert backend.K8.dtype is torch.int8 and backend.V8.dtype is torch.int8
    assert backend.K8.shape == backend.K_cache.shape
    # One scale per (layer, position, kv-head).
    assert backend.KS.shape == backend.K_cache.shape[:3]
    assert backend.KS.dtype is torch.bfloat16
    n_chunks_max = (256 + 127) // 128
    assert backend._i8_part_o.shape == (n_chunks_max, 16, 128)
    assert backend._i8_part_m.shape == (n_chunks_max, 16)


def test_decode_routes_to_int8_kernels_instead_of_fa2():
    fa2, vlk = _Fa2(), _Vlk()
    backend = _make_backend(fa2=fa2, vlk=vlk, use_int8_kv=True)
    backend.run('full', layer_idx=0, q_seq=1, kv_seq=200)

    assert fa2.calls == [], 'FA2 must not run when int8 KV is active at q=1'
    assert [name for name, _ in vlk.calls] == [
        'qwen3_attn_decode_int8kv_partial',
        'qwen3_attn_decode_int8kv_combine',
    ]
    # kv_len=200 spans two 128-position chunks.
    partial_args = vlk.calls[0][1]
    assert partial_args[8] == 200 and partial_args[9] == 2


def test_prefill_still_uses_fa2_when_int8_kv_enabled():
    """Only decode reads the int8 mirrors; prefill stays on the bf16 cache."""
    fa2, vlk = _Fa2(), _Vlk()
    mod = importlib.import_module(BACKEND_MOD)
    props = mock.Mock()
    props.multi_processor_count = 8
    with mock.patch.dict('sys.modules', {
            'flash_rt.flash_rt_fa2': fa2,
            'flash_rt.flash_rt_qwen3_vl_kernels': vlk}), \
            mock.patch.object(torch.cuda, 'get_device_properties',
                              return_value=props):
        backend = mod.RtxFlashAttnBackendQwen3(
            max_seq=256, max_q_seq=8, device='cpu', use_int8_kv=True,
            **DIMS_2B)
    backend.run('full', layer_idx=0, q_seq=8, kv_seq=8, causal=True)
    assert fa2.calls == ['fwd_bf16_causal']
    assert vlk.calls == []


def test_quantize_kv_rows_mirrors_both_k_and_v():
    vlk = _Vlk()
    backend = _make_backend(vlk=vlk, use_int8_kv=True)
    backend.quantize_kv_rows(1, 5)
    assert [name for name, _ in vlk.calls] == [
        'qwen3_kv_rows_quant_int8'] * 2
    # n_rows == kv_heads for a single position.
    assert all(args[3] == 8 for _, args in vlk.calls)


def test_quantize_kv_prefix_covers_every_layer():
    vlk = _Vlk()
    backend = _make_backend(vlk=vlk, use_int8_kv=True)
    backend.quantize_kv_prefix(16)
    # Two layers x (K, V).
    assert len(vlk.calls) == 4
    assert all(args[3] == 16 * 8 for _, args in vlk.calls)


# ── fail-fast ──

@pytest.mark.parametrize('missing', INT8KV_FNS)
def test_missing_kernel_symbol_raises_naming_it(missing):
    with pytest.raises(RuntimeError) as ei:
        _make_backend(vlk=_Vlk(drop=missing), use_int8_kv=True)
    assert missing in str(ei.value)


def test_unsupported_attention_geometry_raises():
    """8B's 32Q/8KV is outside the kernel's specialization — fail at
    construction rather than at the first decode launch."""
    with pytest.raises(RuntimeError) as ei:
        _make_backend(vlk=_Vlk(), dims=DIMS_8B, use_int8_kv=True)
    msg = str(ei.value)
    assert '16Q/8KV' in msg and '32Q' in msg
