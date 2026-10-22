"""CPU checks for the optional Flint path; no checkpoint or network required."""
from __future__ import annotations

import importlib
import json
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flash_rt.models.flint.config import FlintConfigError, _validate
from flash_rt.models.flint.gguf import GGUFFile, GGUFFormatError, _Cursor, _tensor_nbytes
from flash_rt.models.flint.weights import dequantize, split_nvfp4_blocks


def test_core_import_keeps_flint_and_serving_optional():
    code = '''
import json, sys
import flash_rt
import flash_rt.models.flint
from flash_rt.models.flint.pipeline_rtx import FlintPipeline
print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in
    {'torch', 'fastapi', 'uvicorn', 'tokenizers', 'jinja2'})))
'''
    result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_explicit_sm120_routing_only():
    from flash_rt.hardware import _PIPELINE_MAP
    routes = {k: v for k, v in _PIPELINE_MAP.items() if k[0] == 'flint'}
    assert routes == {('flint', 'torch', 'rtx_sm120'):
                      ('flash_rt.frontends.torch.flint_rtx', 'FlintTorchFrontendRtx')}


def test_gguf_readonly_container(tmp_path):
    def string(s):
        data = s.encode()
        return struct.pack('<Q', len(data)) + data
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)
    header += string('general.architecture') + struct.pack('<I', 8) + string('flint')
    header += string('weight') + struct.pack('<IQQIQ', 2, 64, 2, 40, 0)
    header += b'\0' * (-len(header) % 32)
    payload = bytes(range(72))
    path = tmp_path / 'synthetic.gguf'
    path.write_bytes(header + payload)
    with GGUFFile(str(path)) as gguf:
        assert gguf.require('general.architecture') == 'flint'
        assert gguf.tensor('weight').shape == (64, 2)
        assert gguf.raw_bytes('weight').tobytes() == payload
        assert gguf.nvfp4_blocks('weight').shape == (2, 1, 36)
        assert not gguf.raw_bytes('weight').flags.writeable
        with pytest.raises(GGUFFormatError, match='missing tensor'):
            gguf.tensor('missing')


@pytest.mark.parametrize('kind,shape,message', [(40, (63, 2), 'block size'),
                                               (99, (64, 2), 'unsupported ggml type')])
def test_gguf_storage_guards(kind, shape, message):
    with pytest.raises(GGUFFormatError, match=message):
        _tensor_nbytes('weight', kind, shape)


def test_truncated_metadata_fails_clearly():
    with pytest.raises(GGUFFormatError, match='truncated GGUF header'):
        _Cursor(memoryview(b'\0')).scalar('<I')


def test_nvfp4_repack_preserves_split_half_order():
    blocks = np.zeros((1, 1, 36), dtype=np.uint8)
    blocks[..., :4] = 0x38  # unit UE4M3 scales
    blocks[..., 4:] = 0x21  # lower half = 0.5; upper half = 1.0
    codes, scales = split_nvfp4_blocks(blocks)
    want = np.tile(np.r_[np.full(8, 0.5), np.full(8, 1.0)], 4)[None, :]
    np.testing.assert_array_equal(dequantize(codes, scales), want)


@pytest.mark.parametrize('field,value', [('hidden_size', 1024), ('num_layers', 48),
                                       ('num_dense_layers', 3), ('dense_intermediate_size', 4096),
                                       ('sliding_window', 512)])
def test_checkpoint_geometry_guards(field, value):
    cfg = dict(hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=128,
               num_experts=256, num_experts_per_tok=6, expert_intermediate_size=512,
               shared_intermediate_size=1024, num_layers=49, num_dense_layers=2,
               dense_intermediate_size=6144, sliding_window=513, sliding_layers=(False,) * 49)
    cfg[field] = value
    with pytest.raises(FlintConfigError, match=field):
        _validate(SimpleNamespace(**cfg))


def _runtime():
    pytest.importorskip('torch')
    return importlib.import_module('flash_rt.frontends.torch._flint_rtx_runtime')


def test_missing_module_fails_before_checkpoint_io(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime, '_fl', None)
    with pytest.raises(RuntimeError, match='FLASHRT_ENABLE_FLINT=ON'):
        runtime.FlintRuntime('unused.gguf')


def test_incomplete_module_fails_before_checkpoint_io(monkeypatch):
    runtime = _runtime()
    fake = SimpleNamespace(**{n: object() for n in runtime._REQUIRED_KERNELS})
    del fake.gemm_w4a16_qkrope
    monkeypatch.setattr(runtime, '_fl', fake)
    with pytest.raises(RuntimeError, match='missing required symbols: gemm_w4a16_qkrope'):
        runtime.FlintRuntime('unused.gguf')


@pytest.mark.parametrize('kwargs,message', [({'max_seq': 0}, 'max_seq'),
                                           ({'prefill_chunk': 0}, 'prefill_chunk'),
                                           ({'prefill_chunk': 8193}, 'prefill_chunk'),
                                           ({'attn_splits': 0}, 'attn_splits'),
                                           ({'sliding_ring': 0}, 'sliding_ring')])
def test_setup_guards_before_checkpoint_io(monkeypatch, kwargs, message):
    runtime = _runtime()
    monkeypatch.setattr(runtime, '_require_kernels', lambda: object())
    with pytest.raises(ValueError, match=message):
        runtime.FlintRuntime('unused.gguf', **kwargs)


@pytest.mark.parametrize('prompt,budget,message', [([], 1, 'non-empty'),
                                                  ([1], 0, 'max_new_tokens'),
                                                  ([1], -1, 'max_new_tokens'),
                                                  ([-1], 1, 'vocabulary'),
                                                  ([10], 1, 'vocabulary')])
def test_generation_input_guards(prompt, budget, message):
    pytest.importorskip('torch')
    from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx
    frontend = FlintTorchFrontendRtx.__new__(FlintTorchFrontendRtx)
    frontend.cfg = SimpleNamespace(vocab_size=10)
    with pytest.raises(ValueError, match=message):
        next(frontend.stream(prompt, budget))


def test_decode_cannot_run_beyond_allocated_kv():
    runtime = _runtime().FlintRuntime.__new__(_runtime().FlintRuntime)
    runtime.host_pos, runtime.max_seq = 10, 10
    with pytest.raises(ValueError, match='decode range'):
        runtime.replay(1)


def test_non_sm120_device_is_rejected_before_weight_upload(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime, '_require_kernels', lambda: object())
    monkeypatch.setattr(runtime, 'GGUFFile', lambda path: object())
    monkeypatch.setattr(runtime, 'load_config', lambda gguf:
                        SimpleNamespace(sliding_window=513, context_length=131072))
    monkeypatch.setattr(runtime.torch.cuda, 'get_device_capability', lambda device: (8, 9))
    with pytest.raises(RuntimeError, match='Flint requires SM120'):
        runtime.FlintRuntime('unused.gguf')
