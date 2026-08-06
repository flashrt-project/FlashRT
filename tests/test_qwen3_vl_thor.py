"""Smoke tests for the Qwen3-VL Thor (SM110) BF16 frontend and its backend.

CI-friendly: no checkpoint, no GPU. Covers import wiring, the fail-fast kernel
check, required-kernel-list hygiene, the wq-override validator, and the
_PIPELINE_MAP routing for both Jetson targets.

Run:
    PYTHONPATH=. python -m pytest tests/test_qwen3_vl_thor.py -v
"""
from __future__ import annotations

import importlib
import unittest.mock as mock

import pytest


FRONTEND_MOD = 'flash_rt.frontends.torch.qwen3_vl_thor'
BACKEND_MOD = 'flash_rt.hardware.thor.attn_backend_qwen3'


# ── wiring ──

def test_thor_frontend_imports():
    m = importlib.import_module(FRONTEND_MOD)
    assert hasattr(m, 'Qwen3VlTorchFrontendThor')
    assert hasattr(m, '_require_thor_kernels')


def test_thor_attn_backend_imports():
    m = importlib.import_module(BACKEND_MOD)
    assert hasattr(m, 'ThorAttnBackendQwen3')
    assert hasattr(m, 'make_qwen3_thor_attention_spec')


def test_thor_attention_spec_reports_sdpa():
    m = importlib.import_module(BACKEND_MOD)
    spec = m.make_qwen3_thor_attention_spec(
        num_layers=28, num_q_heads=16, num_kv_heads=8, head_dim=128,
        max_seq=4096)
    site = spec['sites'][0]
    assert site['name'] == 'full' and site['kernel'] == 'sdpa_bf16'
    assert site['layer_count'] == 28 and site['max_kv_seq'] == 4096


# ── required-kernel lists ──

def test_kernel_lists_are_non_empty_unique_and_bf16_only():
    m = importlib.import_module(FRONTEND_MOD)
    for names in (m._THOR_CORE_FNS, m._THOR_VL_FNS):
        assert len(names) > 0
        assert len(set(names)) == len(names)
    joined = set(m._THOR_CORE_FNS) | set(m._THOR_VL_FNS)
    assert not any('fp8' in n or 'nvfp4' in n for n in joined)


def test_neutral_kernels_are_expected_from_flash_rt_kernels():
    """embedding_lookup_bf16 / bf16_matmul_bf16 are model-neutral and built for
    every arch, so they must be required from fvk, not the Qwen3-VL module."""
    m = importlib.import_module(FRONTEND_MOD)
    for name in ('embedding_lookup_bf16', 'bf16_matmul_bf16'):
        assert name in m._THOR_CORE_FNS
        assert name not in m._THOR_VL_FNS


def test_thor_requires_the_batched_qkv_prefill_kernel():
    """Thor prefill drives qwen3_qk_norm_rope_kvwrite_batched_bf16, which only
    exists when the module is built with ENABLE_QWEN3_VL_QKV_POSTPROC."""
    m = importlib.import_module(FRONTEND_MOD)
    assert 'qwen3_qk_norm_rope_kvwrite_batched_bf16' in m._THOR_VL_FNS


# ── fail-fast on a missing kernel symbol ──

def _fake_modules(m, *, drop=None):
    fvk = type('Fvk', (), {})
    vlk = type('Vlk', (), {})
    for fn in m._THOR_CORE_FNS:
        if fn != drop:
            setattr(fvk, fn, lambda *a, **k: None)
    for fn in m._THOR_VL_FNS:
        if fn != drop:
            setattr(vlk, fn, lambda *a, **k: None)
    return {'flash_rt.flash_rt_kernels': fvk,
            'flash_rt.flash_rt_qwen3_vl_kernels': vlk}


def test_require_thor_kernels_passes_when_complete():
    m = importlib.import_module(FRONTEND_MOD)
    with mock.patch.dict('sys.modules', _fake_modules(m)):
        fvk, vlk = m._require_thor_kernels()
    assert fvk is not None and vlk is not None


@pytest.mark.parametrize('which', ['core', 'vl'])
def test_require_thor_kernels_raises_naming_the_missing_symbol(which):
    m = importlib.import_module(FRONTEND_MOD)
    target = (m._THOR_CORE_FNS if which == 'core' else m._THOR_VL_FNS)[0]
    with mock.patch.dict('sys.modules', _fake_modules(m, drop=target)):
        with pytest.raises(RuntimeError) as ei:
            m._require_thor_kernels()
    assert target in str(ei.value)


# ── constructor surface ──

def test_constructor_signature():
    import inspect
    m = importlib.import_module(FRONTEND_MOD)
    sig = inspect.signature(m.Qwen3VlTorchFrontendThor.__init__)
    params = set(sig.parameters) - {'self'}
    for required in ('checkpoint_path', 'device', 'max_seq', 'weight_mode',
                     'wq_overrides'):
        assert required in params, f'missing parameter: {required}'
    # Deferred features must not be part of the v1 surface.
    for absent in ('awq', 'awq_alpha', 'vit_backend', 'vit_mode'):
        assert absent not in params, f'unexpected parameter: {absent}'


def test_weight_mode_defaults_to_bf16():
    import inspect
    m = importlib.import_module(FRONTEND_MOD)
    sig = inspect.signature(m.Qwen3VlTorchFrontendThor.__init__)
    assert sig.parameters['weight_mode'].default == 'bf16'


@pytest.mark.parametrize('mode', ['int8', 'fp8', 'bogus', ''])
def test_invalid_weight_mode_rejected(mode):
    m = importlib.import_module(FRONTEND_MOD)
    with pytest.raises(ValueError) as ei:
        m.Qwen3VlTorchFrontendThor('/nonexistent', weight_mode=mode)
    assert 'weight_mode' in str(ei.value)


# ── wq override validator ──

@pytest.mark.parametrize('ov', [
    None,
    {},
    {'gate_up': 'w8'},
    {'lm_head': 'bf16'},
    {'L12.gate_up': 'w4', 'mlp_down': 'w8'},
])
def test_valid_wq_overrides_accepted(ov):
    m = importlib.import_module(FRONTEND_MOD)
    assert m._validate_wq_overrides(ov) == dict(ov or {})


@pytest.mark.parametrize('ov', [
    {'gate_up': 'int8'},          # not a Thor mode
    {'gate_up': 'fp8'},
    {'nonsense': 'w4'},           # unknown projection
    {'Lx.gate_up': 'w4'},         # malformed layer prefix
    {'L1.nonsense': 'w4'},
    {'L3.lm_head': 'w8'},         # lm_head is not per-layer
])
def test_invalid_wq_overrides_rejected(ov):
    m = importlib.import_module(FRONTEND_MOD)
    with pytest.raises(ValueError):
        m._validate_wq_overrides(ov)


def test_wq_overrides_returns_a_copy():
    """The frontend must not alias the caller's dict."""
    m = importlib.import_module(FRONTEND_MOD)
    src = {'gate_up': 'w8'}
    out = m._validate_wq_overrides(src)
    out['mlp_down'] = 'w4'
    assert src == {'gate_up': 'w8'}


def test_wq_active_covers_global_mode_and_overrides():
    """A non-bf16 override must activate quantization even when the global
    weight_mode is bf16 (otherwise the override is silently dropped)."""
    m = importlib.import_module(FRONTEND_MOD)
    assert m._wq_active('w8', {})
    assert m._wq_active('w4', {})
    assert not m._wq_active('bf16', {})
    assert m._wq_active('bf16', {'gate_up': 'w8'})
    assert not m._wq_active('bf16', {'gate_up': 'bf16'})


# ── routing ──

@pytest.mark.parametrize('arch,expected_cls', [
    ('thor', 'Qwen3VlTorchFrontendThor'),
    ('rtx_sm87', 'Qwen3VlTorchFrontendRtxBF16'),
])
def test_qwen3_vl_jetson_routing_resolves(arch, expected_cls):
    hw = importlib.import_module('flash_rt.hardware')
    key = ('qwen3_vl', 'torch', arch)
    assert key in hw._PIPELINE_MAP
    module_path, cls_name = hw._PIPELINE_MAP[key]
    assert cls_name == expected_cls
    mod = importlib.import_module(module_path)
    assert hasattr(mod, cls_name)


def test_sm87_allowlist_still_rejects_unlisted_configs():
    """Adding qwen3_vl must not open SM87 up to every config."""
    hw = importlib.import_module('flash_rt.hardware')
    assert ('pi05', 'torch', 'rtx_sm87') in hw._SM87_ALLOWED
    assert ('qwen3_vl', 'torch', 'rtx_sm87') in hw._SM87_ALLOWED
    with pytest.raises(RuntimeError) as ei:
        hw.resolve_pipeline_class('pi0', 'torch', 'rtx_sm87')
    assert 'qwen3_vl' in str(ei.value) and 'pi05' in str(ei.value)


def test_load_model_redirect_mentions_the_jetson_frontends():
    """config='qwen3_vl' is a chat VLM; the redirect should name every
    frontend, including the two Jetson ones."""
    api = importlib.import_module('flash_rt.api')
    with pytest.raises(NotImplementedError) as ei:
        api.load_model('/nonexistent', config='qwen3_vl')
    msg = str(ei.value)
    assert 'qwen3_vl_thor' in msg and 'qwen3_vl_rtx_bf16' in msg


# ── attention backend behaviour (CPU: needs torch, no GPU) ──

torch = pytest.importorskip('torch')

DIMS = dict(num_layers=2, num_q_heads=4, num_kv_heads=2, head_dim=8)


def _backend(max_q_seq=1, **kw):
    m = importlib.import_module(BACKEND_MOD)
    return m.ThorAttnBackendQwen3(
        max_seq=16, max_q_seq=max_q_seq, device='cpu', **DIMS, **kw)


def _gqa_reference(q, k, v, *, n_q, n_kv, causal):
    """Explicit GQA attention over (heads, seq, dim) float32 tensors."""
    ratio = n_q // n_kv
    out = torch.empty_like(q)
    for h in range(n_q):
        scores = q[h] @ k[h // ratio].transpose(0, 1)
        scores = scores / (q.shape[-1] ** 0.5)
        if causal:
            sq, sk = scores.shape
            mask = torch.ones(sq, sk, dtype=torch.bool).tril(diagonal=sk - sq)
            scores = scores.masked_fill(~mask, float('-inf'))
        out[h] = scores.softmax(dim=-1) @ v[h // ratio]
    return out


def test_backend_buffer_surface_matches_the_frontend_contract():
    b = _backend(max_q_seq=4)
    assert b.K_cache.shape == (2, 16, 2, 8) == b.V_cache.shape
    assert b.Q_buf.shape == (1, 4, 4, 8) == b.O_buf.shape
    assert b.sites() == ('full',)
    assert (b.head_dim('full'), b.num_q_heads('full'),
            b.num_kv_heads('full')) == (8, 4, 2)
    # Strides the frontend uses for its KV-write pointer math (bf16 = 2 bytes).
    assert b.kv_row_stride_bytes == 2 * 8 * 2
    assert b.kv_layer_stride_bytes == 16 * 2 * 8 * 2
    ptrs = b.get_slot_ptrs('full', 1)
    assert ptrs['K'] == b.K_cache.data_ptr() + b.kv_layer_stride_bytes
    assert ptrs['V'] == b.V_cache.data_ptr() + b.kv_layer_stride_bytes


def test_decode_output_matches_a_gqa_reference():
    torch.manual_seed(0)
    b = _backend()
    kv_seq = 6
    b.K_cache.normal_()
    b.V_cache.normal_()
    b.Q_buf.normal_()

    out_ptr = b.run('full', layer_idx=1, q_seq=1, kv_seq=kv_seq)
    assert out_ptr == b.O_buf.data_ptr()

    q = b.Q_buf[0, :1].transpose(0, 1).float()                 # (Hq, 1, hd)
    k = b.K_cache[1, :kv_seq].transpose(0, 1).float()          # (Hkv, kv, hd)
    v = b.V_cache[1, :kv_seq].transpose(0, 1).float()
    ref = _gqa_reference(q, k, v, n_q=4, n_kv=2, causal=False)
    got = b.O_buf[0, :1].transpose(0, 1).float()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_causal_prefill_masks_future_positions():
    torch.manual_seed(0)
    b = _backend(max_q_seq=5)
    S = 5
    b.K_cache.normal_()
    b.V_cache.normal_()
    b.Q_buf.normal_()
    b.run('full', layer_idx=0, q_seq=S, kv_seq=S, causal=True)
    first = b.O_buf[0, 0].clone()

    # Row 0 may only attend to position 0, so perturbing later KV must not
    # change it. A non-causal kernel would fail this.
    b.K_cache[0, 1:S].normal_()
    b.V_cache[0, 1:S].normal_()
    b.run('full', layer_idx=0, q_seq=S, kv_seq=S, causal=True)
    torch.testing.assert_close(b.O_buf[0, 0], first, rtol=0, atol=0)

    q = b.Q_buf[0, :S].transpose(0, 1).float()
    k = b.K_cache[0, :S].transpose(0, 1).float()
    v = b.V_cache[0, :S].transpose(0, 1).float()
    ref = _gqa_reference(q, k, v, n_q=4, n_kv=2, causal=True)
    got = b.O_buf[0, :S].transpose(0, 1).float()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_gqa_expand_path_agrees_with_native_gqa():
    """The fallback for torch builds without enable_gqa must match."""
    torch.manual_seed(0)
    b = _backend()
    b.K_cache.normal_()
    b.V_cache.normal_()
    b.Q_buf.normal_()
    b.run('full', layer_idx=0, q_seq=1, kv_seq=6)
    native = b.O_buf.clone()

    b._sdpa_gqa = False          # force the repeat_interleave path
    b.O_buf.zero_()
    b.run('full', layer_idx=0, q_seq=1, kv_seq=6)
    torch.testing.assert_close(b.O_buf, native, rtol=1e-3, atol=1e-3)


def test_mid_sequence_causal_block_raises():
    """SDPA's is_causal is top-left aligned, so a q-block ending mid-sequence
    would attend to the wrong positions. It must raise, not guess."""
    b = _backend(max_q_seq=4)
    with pytest.raises(NotImplementedError) as ei:
        b.run('full', layer_idx=0, q_seq=2, kv_seq=6, causal=True)
    assert 'kv_seq' in str(ei.value)
    # The equivalent non-causal call is fine (and q=1 is causal-invariant).
    b.run('full', layer_idx=0, q_seq=1, kv_seq=6, causal=True)


def test_out_of_range_and_unknown_site_rejected():
    b = _backend()
    with pytest.raises(KeyError):
        b.run('window', layer_idx=0, q_seq=1, kv_seq=1)
    with pytest.raises(ValueError):
        b.run('full', layer_idx=0, q_seq=1, kv_seq=999)
    with pytest.raises(ValueError):
        b.run('full', layer_idx=0, q_seq=99, kv_seq=1)
    with pytest.raises(KeyError):
        b.head_dim('window')


def test_uneven_gqa_ratio_rejected():
    m = importlib.import_module(BACKEND_MOD)
    with pytest.raises(ValueError) as ei:
        m.ThorAttnBackendQwen3(max_seq=8, device='cpu', num_layers=1,
                               num_q_heads=6, num_kv_heads=4, head_dim=8)
    assert 'multiple' in str(ei.value)


# ── ViT patch-attention SDPA backend probe (CPU, fake kernel modules) ──
#
# On the SDPA fallback (Thor: no flash_rt_fa2), the ViT probes the cuDNN
# SDPA backend once at lazy kernel init — never on the first forward, which
# may run inside a CUDA Graph capture — and must degrade to the default
# dispatcher when the probe fails. FA2 arches must not run the probe at all.

VISION_MOD = 'flash_rt.frontends.torch._qwen3_vl_vision_rtx'


def _vision_stub(monkeypatch, *, with_fa2: bool,
                 attention_backend: str | None = None):
    """Bare Qwen3VlVisionRtx whose lazy _kernels() sees fake kernel modules."""
    import sys
    import types

    import flash_rt

    vmod = importlib.import_module(VISION_MOD)
    monkeypatch.setattr(flash_rt, 'flash_rt_kernels',
                        types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(flash_rt, 'flash_rt_qwen3_vl_kernels',
                        types.SimpleNamespace(), raising=False)
    if with_fa2:
        monkeypatch.setattr(
            flash_rt, 'flash_rt_fa2',
            types.SimpleNamespace(fwd_bf16=lambda **kw: None), raising=False)
    else:
        monkeypatch.delattr(flash_rt, 'flash_rt_fa2', raising=False)
        # A None sys.modules entry makes the import raise ImportError.
        monkeypatch.setitem(sys.modules, 'flash_rt.flash_rt_fa2', None)
    monkeypatch.setattr(
        torch.cuda, 'get_device_properties',
        lambda _dev: types.SimpleNamespace(multi_processor_count=1),
        raising=False)
    v = vmod.Qwen3VlVisionRtx.__new__(vmod.Qwen3VlVisionRtx)
    v.device = 'cpu'
    v._fvk = v._fa2 = v._vlk = None
    v._vit_sdpa_ctx = None
    v._attention_backend = attention_backend or ('fa2' if with_fa2 else 'sdpa')
    v.num_heads = 16
    v.head_dim = 64
    return v


def test_vit_default_fa2_backend_fails_fast_when_module_missing(monkeypatch):
    v = _vision_stub(
        monkeypatch, with_fa2=False, attention_backend='fa2')
    with pytest.raises(ImportError):
        v._kernels()


def test_thor_explicitly_selects_sdpa_and_requires_checkpoint():
    import inspect
    import pathlib

    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_thor')
    source = inspect.getsource(m.Qwen3VlTorchFrontendThor._ensure_native_vision)
    assert "attention_backend='sdpa'" in source

    quickstart = (pathlib.Path(__file__).parents[1]
                  / 'examples/thor/qwen3_vl_quickstart.py').read_text()
    assert "add_argument('--checkpoint', required=True)" in quickstart


def _record_sdpa_kernel(monkeypatch):
    import contextlib

    import torch.nn.attention as tna

    picked = []

    def fake_sdpa_kernel(backend):
        picked.append(backend)
        return contextlib.nullcontext()

    monkeypatch.setattr(tna, 'sdpa_kernel', fake_sdpa_kernel)
    return picked


def test_vit_cudnn_probe_skipped_when_fa2_present(monkeypatch):
    v = _vision_stub(monkeypatch, with_fa2=True)

    def _boom(*a, **k):
        raise AssertionError('probe must not run when FA2 owns ViT attention')

    monkeypatch.setattr(
        torch.nn.functional, 'scaled_dot_product_attention', _boom)
    v._kernels()
    assert v._vit_use_fa2 is True
    assert v._vit_sdpa_ctx is None


def test_vit_cudnn_probe_sets_ctx_on_the_sdpa_fallback(monkeypatch):
    import torch.nn.attention as tna

    v = _vision_stub(monkeypatch, with_fa2=False)
    picked = _record_sdpa_kernel(monkeypatch)
    monkeypatch.setattr(
        torch.nn.functional, 'scaled_dot_product_attention',
        lambda *a, **k: torch.empty(0))
    v._kernels()
    assert v._vit_use_fa2 is False
    assert picked == [tna.SDPBackend.CUDNN_ATTENTION]
    assert v._vit_sdpa_ctx is not None
    with v._vit_sdpa_ctx():
        pass
    assert picked == [tna.SDPBackend.CUDNN_ATTENTION] * 2


def test_vit_cudnn_probe_failure_degrades_to_default_dispatcher(monkeypatch):
    import torch.nn.attention as tna

    v = _vision_stub(monkeypatch, with_fa2=False)

    def _no_backend(_b):
        raise RuntimeError('cuDNN attention unavailable')

    monkeypatch.setattr(tna, 'sdpa_kernel', _no_backend)
    v._kernels()  # must not raise
    assert v._vit_use_fa2 is False
    assert v._vit_sdpa_ctx is None
