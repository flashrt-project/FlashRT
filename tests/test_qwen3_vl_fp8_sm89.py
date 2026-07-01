"""Smoke tests for the Qwen3-VL official-FP8 SM89 text path.

These are CPU/CI-friendly import and schema tests. Real checkpoint and CUDA
coverage is provided by the local development benchmark/script because the
official 8B FP8 weights are too large for CI.
"""
from __future__ import annotations

import importlib
import inspect


def test_frontend_imports():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_fp8_sm89')
    text_cls = m.Qwen3VlFp8Sm89TextFrontend
    assert hasattr(m, 'Qwen3VlFp8Sm89TextFrontend')
    assert 'max_prefill_seq' in inspect.signature(text_cls).parameters
    assert 'max_decode_graphs' in inspect.signature(text_cls).parameters
    assert 'run_lm_head' in inspect.signature(
        text_cls.forward_hidden_prefill_fp8_blockscaled).parameters
    mm = importlib.import_module(
        'flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal')
    cls = mm.Qwen3VlFp8Sm89Frontend
    assert hasattr(mm, 'Qwen3VlFp8Sm89Frontend')
    assert 'max_prefill_seq' in inspect.signature(
        cls).parameters
    for name in ('fuse_gate_up', 'fuse_qk_postproc', 'use_fp8_lm_head',
                 'vision_bf16_first_blocks', 'max_prefill_graphs',
                 'max_decode_graphs'):
        assert name in inspect.signature(cls).parameters
    for name in (
        'prefill_graph', 'decode_step_with_graph',
        'warmup_decode_graphs', 'clear_graphs', 'graph_cache_stats',
        'generate',
    ):
        assert hasattr(cls, name)


def test_default_prefill_limit_matches_max_seq_without_cuda():
    """SM89 VL should behave like SM120: default prefill capacity is max_seq."""
    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89 import (
        _resolve_max_prefill_seq,
    )

    assert _resolve_max_prefill_seq(1234, None) == 1234
    assert _resolve_max_prefill_seq(1234, 256) == 256


def test_multimodal_prefill_graph_falls_back_to_eager_without_pg_key():
    """Match SM120: graph prefill falls back for multi-image/video prompts."""
    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal import (
        Qwen3VlFp8Sm89Frontend,
    )

    fe = Qwen3VlFp8Sm89Frontend.__new__(Qwen3VlFp8Sm89Frontend)
    fe._prompt = {'S': 7}
    called = []

    def fake_prefill():
        called.append(True)
        return 'eager-logits'

    fe.prefill = fake_prefill
    assert fe.prefill_graph() == 'eager-logits'
    assert called == [True]


def test_weight_loader_invariants_reject_missing_layer_fields():
    from flash_rt.frontends.torch._qwen3_vl_fp8_weights import (
        WeightHandles,
        assert_extraction_invariants_qwen3_vl_fp8,
    )

    h = WeightHandles()
    h.ptrs.update({
        'quant_format': 'fp8_block128',
        'num_layers': 1,
        'lm_head_quantized': False,
        'layers': [{'input_norm_w': 1}],
    })
    try:
        assert_extraction_invariants_qwen3_vl_fp8(h)
    except AssertionError as e:
        assert 'missing' in str(e)
        assert 'lm_head_w' in str(e)
    else:
        raise AssertionError('expected invariant failure')

    h.ptrs.update({
        'lm_head_w': 1,
    })
    try:
        assert_extraction_invariants_qwen3_vl_fp8(h)
    except AssertionError as e:
        assert 'missing' in str(e)
        assert 'qkv_proj_w' in str(e)
    else:
        raise AssertionError('expected layer invariant failure')
