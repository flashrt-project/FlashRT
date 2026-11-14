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


def test_explicit_routing_only():
    from flash_rt.hardware import _PIPELINE_MAP
    routes = {k: v for k, v in _PIPELINE_MAP.items() if k[0] == 'flint'}
    assert routes == {('flint', 'torch', 'rtx_sm120'):
                      ('flash_rt.frontends.torch.flint_rtx', 'FlintTorchFrontendRtx'),
                      ('flint', 'torch', 'apple_mps'):
                      ('flash_rt.frontends.torch.flint_mps', 'FlintTorchFrontendMps')}


def test_apple_runtime_import_is_framework_free():
    code = '''
import json, sys
from flash_rt.frontends.torch._flint_mps_runtime import FlintRuntime
from flash_rt.models.flint.pipeline_mps import FlintPipelineMps, FlintPrefillMps
from flash_rt.models.flint.mlx_checkpoint import MLXCheckpoint
print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in
    {'torch', 'mlx', 'fastapi', 'uvicorn', 'tokenizers', 'jinja2'})))
'''
    result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def _safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {'dtype': dtype, 'shape': list(array.shape), 'data_offsets': [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    meta = json.dumps(header).encode()
    meta += b' ' * (-len(meta) % 8)
    path.write_bytes(struct.pack('<Q', len(meta)) + meta + b''.join(blobs))


def test_mlx_affine_reader_low_nibble_first(tmp_path):
    from flash_rt.models.flint.mlx_checkpoint import (AffineMatrix, SafetensorsFile, bf16_to_f32,
                                                      dequantize, f32_to_bf16, unpack_codes)
    rng = np.random.default_rng(0)
    q = rng.integers(0, 16, size=(3, 128), dtype=np.uint8)
    codes = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    scales = f32_to_bf16(rng.standard_normal((3, 2)).astype(np.float32))
    biases = f32_to_bf16(rng.standard_normal((3, 2)).astype(np.float32))
    path = tmp_path / 'm.safetensors'
    _safetensors(path, {'w.weight': ('U8', codes), 'w.scales': ('BF16', scales), 'w.biases': ('BF16', biases)})
    f = SafetensorsFile(path)
    m = AffineMatrix(f.tensor('w.weight'), f.tensor('w.scales'), f.tensor('w.biases'), 4)
    assert np.array_equal(unpack_codes(m), q)
    expect = bf16_to_f32(scales).repeat(64, 1) * q + bf16_to_f32(biases).repeat(64, 1)
    assert np.allclose(dequantize(m), expect, rtol=0, atol=1e-6)


def test_apple_vector_path_bounds_follow_mlx():
    from flash_rt.models.flint.pipeline_mps import qmv_limit
    assert [qmv_limit(2048, 512), qmv_limit(2048, 4096), qmv_limit(4096, 2048), qmv_limit(2048, 6144),
            qmv_limit(6144, 2048), qmv_limit(1024, 2048)] == [18, 12, 12, 10, 10, 18]


def test_apple_first_token_selection_rule():
    from flash_rt.frontends.torch._flint_mps_runtime import FlintRuntime
    from flash_rt.models.flint.mlx_checkpoint import f32_to_bf16
    logits = np.full(64, -10.0, np.float32)
    logits[[5, 9, 12]] = [3.0, 3.0, 2.5]
    noise = np.zeros(64, np.float32)
    fake = SimpleNamespace(logits=lambda step: f32_to_bf16(logits), buf={'noise': 0},
                           plan=SimpleNamespace(read=lambda b, o, dt, n: noise[:n].copy()))
    assert FlintRuntime.select(fake, 'greedy') == 5          # lowest id among equal maxima
    assert FlintRuntime.select(fake, 'sampled', top_k=1) == 9  # top-k ties rank the larger id first
    noise[12] = 5.0
    assert FlintRuntime.select(fake, 'sampled', top_k=3, top_p=1.0) == 12
    assert FlintRuntime.select(fake, 'sampled', top_k=3, top_p=0.3) == 9   # nucleus keeps the first only


def _apple_config():
    """config.json fields of the Flint MLX int4 export that the reader consults."""
    full = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
    return {
        'model_type': 'flint', 'num_hidden_layers': 49,
        'layer_types': ['full_attention' if i in full else 'sliding_attention' for i in range(49)],
        'no_rope_layer_indices': full, 'rope_parameters': {'rope_theta': 10000.0, 'rope_type': 'default'},
        'scoring_func': 'sigmoid', 'norm_topk_prob': True, 'attention_sink': True, 'tie_word_embeddings': False,
        'mlp_only_layers': [0, 1], 'hidden_size': 2048, 'num_attention_heads': 32, 'num_key_value_heads': 4,
        'head_dim': 128, 'vocab_size': 150144, 'intermediate_size': 6144, 'num_experts': 256,
        'num_experts_per_tok': 6, 'moe_intermediate_size': 512, 'shared_expert_intermediate_size': 1024,
        'rms_norm_eps': 1e-6, 'sliding_window': 513, 'max_position_embeddings': 262144,
        'bos_token_id': 150000, 'eos_token_id': [150001, 150003, 150005, 150018],
        'quantization': {'group_size': 64, 'bits': 4},
        'mlx_export': {'bits': 4, 'routed_expert_bits': 4, 'embedding_bits': 8},
    }


def test_mlx_config_reader_and_refusals():
    from flash_rt.models.flint.config import FlintConfigError, config_from_json
    cfg = config_from_json(_apple_config())
    assert cfg.eog_token_ids == (150001, 150003, 150005, 150018)
    assert cfg.sliding_layers[2] and not cfg.sliding_layers[5] and cfg.is_moe(2) and not cfg.is_moe(1)
    for key, value in [('scoring_func', 'softmax'), ('norm_topk_prob', False), ('attention_sink', False),
                       ('tie_word_embeddings', True), ('rope_scaling', {'type': 'yarn'}), ('num_experts', 128),
                       ('no_rope_layer_indices', [0, 1]), ('model_type', 'qwen3')]:
        with pytest.raises(FlintConfigError):
            config_from_json({**_apple_config(), key: value})


def test_apple_kernel_geometry_guard():
    from flash_rt.models.flint.config import FlintConfigError, config_from_json
    from flash_rt.models.flint.pipeline_mps import check_geometry
    check_geometry(config_from_json(_apple_config()))
    # Valid Flint geometry that the Apple kernels (which hard-code these) must refuse.
    for key, value in [('vocab_size', 150016), ('rms_norm_eps', 1e-5),
                       ('rope_parameters', {'rope_theta': 1e6, 'rope_type': 'default'})]:
        cfg = config_from_json({**_apple_config(), key: value})
        with pytest.raises(FlintConfigError, match='Apple Flint kernels'):
            check_geometry(cfg)


@pytest.mark.parametrize('change', [{'quantization': {'group_size': 32, 'bits': 4}},
                                    {'quantization': {'group_size': 64, 'bits': 8}},
                                    {'mlx_export': {'bits': 4, 'routed_expert_bits': 4, 'embedding_bits': 4}}])
def test_mlx_checkpoint_refuses_other_quantization(tmp_path, change):
    from flash_rt.models.flint.config import FlintConfigError
    from flash_rt.models.flint.mlx_checkpoint import MLXCheckpoint
    (tmp_path / 'config.json').write_text(json.dumps({**_apple_config(), **change}))
    with pytest.raises(FlintConfigError):
        MLXCheckpoint(tmp_path)


def test_mlx_checkpoint_shape_guards(tmp_path):
    from flash_rt.models.flint.mlx_checkpoint import MLXCheckpoint
    (tmp_path / 'config.json').write_text(json.dumps(_apple_config()))
    with pytest.raises(FileNotFoundError):
        MLXCheckpoint(tmp_path)
    codes = np.zeros((4, 64), np.uint8)          # 4 rows x 128 inputs
    groups = np.zeros((4, 2), np.uint16)
    _safetensors(tmp_path / 'model-00000.safetensors',
                 {'m.weight': ('U8', codes), 'm.scales': ('BF16', groups), 'm.biases': ('BF16', groups),
                  'bad.weight': ('U8', codes), 'bad.scales': ('BF16', groups), 'bad.biases': ('BF16', groups[:, :1])})
    ck = MLXCheckpoint(tmp_path)
    assert ck.affine('m', k=128).k == 128
    with pytest.raises(ValueError):
        ck.affine('m', k=256)                    # input width the caller expects
    with pytest.raises(ValueError):
        ck.affine('m', bits=8)                   # 64 code bytes per row hold 128 4-bit inputs, not 8-bit ones
    with pytest.raises(ValueError):
        ck.affine('bad')                         # scales and biases disagree
    with pytest.raises(KeyError):
        ck.tensor('missing')
    ck.close()


def test_apple_sampling_arguments():
    from flash_rt.frontends.torch.flint_mps import FlintTorchFrontendMps
    config = FlintTorchFrontendMps._sample_config
    assert config(0.0, 1.0, 0) == ('greedy', 0)
    assert config(0.7, 0.95, 20) == ('sampled', 20)
    for args in [(-0.5, 1.0, 20), (1.0, 1.0, 0), (1.0, 1.0, 33), (1.0, 0.0, 20), (1.0, 1.5, 20)]:
        with pytest.raises(ValueError):
            config(*args)


@pytest.mark.parametrize('logical_end,fork', [
    (2000, 1000),   # a regenerate after the ring wrapped: the history before the fork is gone
    (2000, 1999),   # needs [1487, 1999); position 1999 overwrote 1486 only
    (2000, 1998),   # needs [1486, 1998); 1486's slot holds 1999
    (2000, 2000),
    (400, 300),     # fewer positions than the ring: everything is still resident
    (1000, 63),     # minimum reusable prefix
    (1000, 64),
])
def test_apple_stream_reuses_only_resident_history(logical_end, fork):
    from flash_rt.frontends.torch.flint_mps import FlintTorchFrontendMps
    window = 513
    # Oracle: the ring holds the last `window` written positions (writes are contiguous from 0).
    resident = all(p >= logical_end - window for p in range(max(0, fork - window + 1), fork))
    expected = fork if resident and fork >= 64 else 0
    starts = []
    fe = FlintTorchFrontendMps.__new__(FlintTorchFrontendMps)
    fe.cfg = SimpleNamespace(vocab_size=10, sliding_window=window)
    fe.rt = SimpleNamespace(max_seq=32768, swa_kv_start=max(0, logical_end - window),
                            prefill=lambda ids, start, **sampling: starts.append(start) or 3)
    fe._kv_ids = [1] * logical_end
    fe.min_reuse, fe.last_reused, fe.eog = 64, 0, set()
    assert list(fe.stream([1] * fork + [2] * 32, 1, reuse_prefix=True)) == [[3]]
    assert starts == [expected]
    assert fe.last_reused == expected


def test_apple_runtime_tracks_resident_sliding_history():
    from flash_rt.frontends.torch._flint_mps_runtime import FlintRuntime
    rt = FlintRuntime.__new__(FlintRuntime)
    rt.cfg = SimpleNamespace(sliding_window=513)
    rt.max_seq, rt.prefill_chunk, rt.initial, rt.swa_kv_start = 8192, 2048, 0, 0
    rt.buf = {'tokens': 0}
    rt.programs = {name: name for name in ('prefill', 'prefill_small', 'prefill_head', 'greedy')}
    rt._state = np.zeros(4, np.uint32)
    runs = []
    rt.plan = SimpleNamespace(write=lambda *args: None, run=lambda p: runs.append((p, tuple(rt._state))) or {})
    rt.select = lambda *args, **kwargs: 7
    assert rt.prefill(list(range(2300))) == 7
    assert runs == [('prefill', (0, 2048, 0, 2048)), ('prefill_small', (2048, 252, 2048, 2300)),
                    ('prefill_head', (2048, 252, 2048, 2300))]
    assert (rt.initial, rt.swa_kv_start) == (2300, 2300 - 513)
    rt._params, rt._control, rt._diagnostic = np.zeros(16, np.uint32), np.zeros(4, np.uint32), np.zeros(4, np.uint32)
    rt._packet = np.zeros(17, np.uint32)
    rt._packet[:6], rt._packet[16] = [7, 8, 8, 9, 9, 10], 3
    assert rt.decode(7, 2300, 3) == [8, 9, 10]
    assert rt._params[11] == 2300 and rt.swa_kv_start == 2303 - 513
    runs.clear()
    rt.prefill(list(range(100)))   # a new sequence starts over
    assert (rt.initial, rt.swa_kv_start) == (100, 0)
    assert runs[0] == ('prefill_small', (0, 100, 0, 100))

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
