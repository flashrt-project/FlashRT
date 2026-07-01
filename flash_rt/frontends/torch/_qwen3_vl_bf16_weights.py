"""Loader for official BF16 Qwen3-VL language weights.

The Orin bring-up path keeps the checkpoint in its original BF16 layout and
records raw device pointers for the generic FlashRT Qwen3 BF16 kernels. It is
intentionally separate from the SM89 FP8 loader so the precision contracts do
not get mixed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch


@dataclass
class WeightHandles:
    ptrs: dict = field(default_factory=dict)
    anchors: list = field(default_factory=list)


def _anchor(handles: WeightHandles, t: torch.Tensor) -> int:
    handles.anchors.append(t)
    return int(t.data_ptr())


def _open_shards(ckpt_dir: str):
    from safetensors import safe_open

    idx_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            wmap = json.load(f)['weight_map']
        handles_d = {
            shard: safe_open(os.path.join(ckpt_dir, shard), framework='pt',
                             device='cpu')
            for shard in set(wmap.values())
        }
        return handles_d, wmap
    single = os.path.join(ckpt_dir, 'model.safetensors')
    if not os.path.isfile(single):
        raise RuntimeError(
            f'Qwen3-VL BF16 ckpt missing both index and model.safetensors: '
            f'{ckpt_dir}')
    h = safe_open(single, framework='pt', device='cpu')
    wmap = {key: 'model.safetensors' for key in h.keys()}
    return {'model.safetensors': h}, wmap


def _get_tensor(handles_d, wmap, key: str) -> torch.Tensor:
    if key not in wmap:
        raise KeyError(f'tensor {key!r} not in weight_map')
    return handles_d[wmap[key]].get_tensor(key)


def _to_bf16(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(torch.bfloat16).to(device).contiguous()


def _load_bf16_linear(handles: WeightHandles, out: dict, short: str,
                      base: str, handles_d, wmap, device: str) -> None:
    out[short + '_w'] = _anchor(
        handles, _to_bf16(_get_tensor(handles_d, wmap, base + '.weight'),
                          device))


def extract_weights_qwen3_vl_bf16(ckpt_dir: str,
                                  device: str = 'cuda:0') -> WeightHandles:
    cfg_path = os.path.join(ckpt_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f'Qwen3-VL BF16 ckpt missing config: {cfg_path}')
    with open(cfg_path) as f:
        cfg = json.load(f)
    text_cfg = cfg['text_config']

    num_layers = int(text_cfg['num_hidden_layers'])
    hidden = int(text_cfg['hidden_size'])
    vocab = int(text_cfg['vocab_size'])
    head_dim = int(text_cfg.get('head_dim') or
                   (hidden // int(text_cfg['num_attention_heads'])))
    n_q = int(text_cfg['num_attention_heads'])
    n_kv = int(text_cfg['num_key_value_heads'])
    inter = int(text_cfg['intermediate_size'])
    rms_eps = float(text_cfg.get('rms_norm_eps', 1e-6))
    rope_scaling = text_cfg.get('rope_scaling') or cfg.get('rope_scaling') or {}
    rope_theta = float(text_cfg.get('rope_theta') or cfg.get('rope_theta') or
                       rope_scaling.get('rope_theta') or 1_000_000.0)

    handles = WeightHandles()
    handles_d, wmap = _open_shards(ckpt_dir)

    embed_key = 'model.language_model.embed_tokens.weight'
    embed = _to_bf16(_get_tensor(handles_d, wmap, embed_key), device)
    handles.ptrs['embed_w'] = _anchor(handles, embed)
    final_norm = _to_bf16(
        _get_tensor(handles_d, wmap, 'model.language_model.norm.weight'),
        device)
    handles.ptrs['final_norm_w'] = _anchor(handles, final_norm)

    tied = bool(cfg.get('tie_word_embeddings',
                        text_cfg.get('tie_word_embeddings', False)))
    if 'lm_head.weight' in wmap:
        lm_head_cpu = _get_tensor(handles_d, wmap, 'lm_head.weight')
    elif tied:
        lm_head_cpu = _get_tensor(handles_d, wmap, embed_key)
    else:
        raise RuntimeError(
            'Qwen3-VL BF16 ckpt has neither lm_head.weight nor '
            'tie_word_embeddings=true')
    handles.ptrs['lm_head_w'] = _anchor(handles, _to_bf16(lm_head_cpu, device))

    per_layer: list[dict] = [None] * num_layers   # type: ignore[list-item]
    for L in range(num_layers):
        base = f'model.language_model.layers.{L}.'
        ld: dict = {'type': 'full_attention', 'quant_format': 'bf16'}
        ld['input_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap, base + 'input_layernorm.weight'),
            device))
        ld['post_attn_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap,
                        base + 'post_attention_layernorm.weight'),
            device))
        sa = base + 'self_attn.'
        q = _to_bf16(_get_tensor(handles_d, wmap, sa + 'q_proj.weight'),
                     device)
        k = _to_bf16(_get_tensor(handles_d, wmap, sa + 'k_proj.weight'),
                     device)
        v = _to_bf16(_get_tensor(handles_d, wmap, sa + 'v_proj.weight'),
                     device)
        qkv = torch.cat([q, k, v], dim=0).contiguous()
        ld['qkv_proj_w'] = _anchor(handles, qkv)
        ld['qkv_proj_N'] = int(qkv.shape[0])
        _load_bf16_linear(handles, ld, 'o_proj', sa + 'o_proj',
                          handles_d, wmap, device)
        ld['q_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap, sa + 'q_norm.weight'), device))
        ld['k_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap, sa + 'k_norm.weight'), device))

        mp = base + 'mlp.'
        gate = _to_bf16(_get_tensor(handles_d, wmap, mp + 'gate_proj.weight'),
                        device)
        up = _to_bf16(_get_tensor(handles_d, wmap, mp + 'up_proj.weight'),
                      device)
        gate_up = torch.cat([gate, up], dim=0).contiguous()
        ld['gate_up_w'] = _anchor(handles, gate_up)
        ld['gate_up_N'] = int(gate_up.shape[0])
        _load_bf16_linear(handles, ld, 'mlp_down', mp + 'down_proj',
                          handles_d, wmap, device)
        per_layer[L] = ld

    handles.ptrs.update({
        'vocab_size': vocab,
        'hidden': hidden,
        'head_dim': head_dim,
        'num_q_heads': n_q,
        'num_kv_heads': n_kv,
        'intermediate': inter,
        'num_layers': num_layers,
        'layer_types': ['full_attention'] * num_layers,
        'rms_norm_eps': rms_eps,
        'rope_theta': rope_theta,
        'rope_scaling': rope_scaling,
        'quant_format': 'bf16',
        'ckpt_dir': ckpt_dir,
        'layers': per_layer,
    })
    return handles


def assert_extraction_invariants_qwen3_vl_bf16(
        handles: WeightHandles) -> None:
    p = handles.ptrs
    if p.get('quant_format') != 'bf16':
        raise AssertionError('expected quant_format=bf16')
    if int(p['head_dim']) != 128:
        raise AssertionError('Qwen3 fused q/k norm RoPE kernels require head_dim=128')
    if int(p['num_q_heads']) % int(p['num_kv_heads']) != 0:
        raise AssertionError('num_q_heads must be a multiple of num_kv_heads')
    if len(p['layers']) != int(p['num_layers']):
        raise AssertionError('layer count mismatch')
