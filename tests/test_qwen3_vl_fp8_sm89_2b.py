"""Smoke tests for the Qwen3-VL-2B block-128 FP8 SM89 path.

CPU/CI-friendly: the config-driven frontend guard and the offline quantizer's
block-128 round-trip are exercised without the (multi-GB) checkpoints or CUDA.
Real checkpoint + tps coverage lives in scripts/smoke_qwen3_vl_fp8_sm89.py and
docs/qwen3_vl_fp8_sm89_2b.md.
"""
from __future__ import annotations

import importlib

import torch


def test_quantizer_block128_roundtrip():
    q = importlib.import_module(
        'scripts.quantize_qwen3_vl_to_fp8_block128')
    torch.manual_seed(0)
    # A 2B-shaped fused gate/up tile: N=6144, K=2048, both multiples of 128.
    w = torch.randn(6144, 2048, dtype=torch.bfloat16) * 0.02
    fp8, scale = q._quantize_block128(w)
    assert fp8.dtype == torch.float8_e4m3fn
    assert fp8.shape == (6144, 2048)
    assert scale.dtype == torch.float32
    assert scale.shape == (6144 // 128, 2048 // 128)
    # Dequant = q * weight_scale_inv (multiply-to-dequant), reconstructed by
    # broadcasting each 128x128 block's scale. Error must stay within e4m3's
    # ~6% per-block relative resolution.
    deq = (fp8.float().view(48, 128, 16, 128)
           * scale[:, None, :, None]).view(6144, 2048)
    rel = (deq - w.float()).abs().mean() / w.float().abs().mean()
    assert rel < 0.05, f'block128 dequant rel-err too high: {rel}'


def test_quantizer_targets_only_language_linears():
    q = importlib.import_module(
        'scripts.quantize_qwen3_vl_to_fp8_block128')
    assert q._is_quant_target(
        'model.language_model.layers.0.self_attn.q_proj.weight')
    assert q._is_quant_target(
        'model.language_model.layers.27.mlp.down_proj.weight')
    # Norms, embeddings, vision tower, and lm_head are NOT quantized.
    assert not q._is_quant_target(
        'model.language_model.layers.0.input_layernorm.weight')
    assert not q._is_quant_target(
        'model.language_model.embed_tokens.weight')
    assert not q._is_quant_target(
        'model.visual.blocks.0.attn.qkv.weight')
    assert not q._is_quant_target('lm_head.weight')


def test_quantizer_expected_count_from_config(tmp_path):
    q = importlib.import_module(
        'scripts.quantize_qwen3_vl_to_fp8_block128')
    (tmp_path / 'config.json').write_text(
        '{"text_config": {"num_hidden_layers": 28}}')
    assert q._expected_quant_linears(str(tmp_path)) == 196


def test_loader_handles_single_file_and_tied_embeddings(tmp_path):
    """A single-file (no index.json) tied-embedding ckpt must load via the
    synthetic weight_map + embed_tokens lm_head fallback."""
    from safetensors.torch import save_file

    from flash_rt.frontends.torch import _qwen3_vl_fp8_weights as wl

    save_file(
        {'model.language_model.embed_tokens.weight': torch.zeros(256, 128)},
        str(tmp_path / 'model.safetensors'), metadata={'format': 'pt'})
    handles_d, wmap = wl._open_shards(str(tmp_path))
    assert wmap['model.language_model.embed_tokens.weight'] == \
        'model.safetensors'
    assert 'model.safetensors' in handles_d
