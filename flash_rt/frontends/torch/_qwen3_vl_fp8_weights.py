"""Loader for official Qwen3-VL FP8 language weights.

The official Qwen3-VL FP8 checkpoint stores the text stack under
``model.language_model.*`` with FP8 e4m3 weights plus 128x128 block scales
(``.weight_scale_inv``). This loader intentionally keeps the original FP8
layout and records raw device pointers for the SM89 block-scaled GEMV path.
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
        wmap = json.load(open(idx_path))['weight_map']
        handles_d = {
            shard: safe_open(os.path.join(ckpt_dir, shard), framework='pt',
                             device='cpu')
            for shard in set(wmap.values())
        }
        return handles_d, wmap
    # Single-file FP8 checkpoints (for example a quantized 2B ckpt written as
    # one shard) have no index.json. Build a synthetic weight_map from the lone
    # model.safetensors so the rest of the loader is identical to the sharded
    # path. Raw BF16 2B checkpoints still need the offline block-128 quantizer.
    single = os.path.join(ckpt_dir, 'model.safetensors')
    if not os.path.isfile(single):
        raise RuntimeError(
            f'Qwen3-VL FP8 ckpt missing both index and model.safetensors: '
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


def _to_fp8(t: torch.Tensor, device: str) -> torch.Tensor:
    if t.dtype != torch.float8_e4m3fn:
        raise TypeError(f'expected torch.float8_e4m3fn weight, got {t.dtype}')
    return t.to(device).contiguous()


def _load_fp8_linear(handles: WeightHandles, out: dict, short: str,
                     base: str, handles_d, wmap, device: str
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    w = _to_fp8(_get_tensor(handles_d, wmap, base + '.weight'), device)
    s = _get_tensor(handles_d, wmap, base + '.weight_scale_inv').to(
        torch.float32).to(device).contiguous()
    out[short + '_w'] = _anchor(handles, w)
    out[short + '_s'] = _anchor(handles, s)
    return w, s


def extract_weights_qwen3_vl_fp8(ckpt_dir: str, device: str = 'cuda:0',
                                 quantize_lm_head: bool = True
                                 ) -> WeightHandles:
    cfg_path = os.path.join(ckpt_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f'Qwen3-VL FP8 ckpt missing config: {cfg_path}')
    cfg = json.load(open(cfg_path))
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

    embed = _to_bf16(
        _get_tensor(handles_d, wmap, 'model.language_model.embed_tokens.weight'),
        device)
    handles.ptrs['embed_w'] = _anchor(handles, embed)
    final_norm = _to_bf16(
        _get_tensor(handles_d, wmap, 'model.language_model.norm.weight'),
        device)
    handles.ptrs['final_norm_w'] = _anchor(handles, final_norm)
    # tie_word_embeddings: the 2B release ties lm_head to embed_tokens and
    # ships no lm_head.weight. Fall back to the embedding matrix in that case
    # (identical math: logits = h @ embed_tokens^T).
    tied = bool(cfg.get('tie_word_embeddings',
                        text_cfg.get('tie_word_embeddings', False)))
    if 'lm_head.weight' in wmap:
        lm_head_cpu = _get_tensor(handles_d, wmap, 'lm_head.weight')
    elif tied:
        lm_head_cpu = _get_tensor(
            handles_d, wmap, 'model.language_model.embed_tokens.weight')
    else:
        raise RuntimeError(
            'Qwen3-VL FP8 ckpt has neither lm_head.weight nor '
            'tie_word_embeddings=true')
    N_lm, K_lm = lm_head_cpu.shape
    handles.ptrs['lm_head_quantized'] = bool(quantize_lm_head)
    if quantize_lm_head:
        if N_lm % 128 != 0 or K_lm % 128 != 0:
            raise RuntimeError(
                f'lm_head shape must be block128-compatible, got {lm_head_cpu.shape}')
        lm_head_fp8 = torch.empty(
            N_lm, K_lm, dtype=torch.float8_e4m3fn, device=device)
        lm_scale = torch.empty(
            N_lm // 128, K_lm // 128, dtype=torch.float32, device=device)
        for row0 in range(0, N_lm, 8192):
            row1 = min(row0 + 8192, N_lm)
            chunk = lm_head_cpu[row0:row1].to(torch.bfloat16).to(
                device).contiguous()
            rows = row1 - row0
            view = chunk.float().view(rows // 128, 128, K_lm // 128, 128)
            amax = view.abs().amax(dim=(1, 3))
            scale = (amax / 448.0).clamp_min(1e-12)
            q = (view / scale[:, None, :, None]).clamp(
                -448.0, 448.0).to(torch.float8_e4m3fn).view(
                    rows, K_lm).contiguous()
            lm_head_fp8[row0:row1].copy_(q)
            lm_scale[row0 // 128:row1 // 128].copy_(scale.float())
        handles.ptrs['lm_head_fp8_w'] = _anchor(handles, lm_head_fp8)
        handles.ptrs['lm_head_fp8_s'] = _anchor(handles, lm_scale)
    else:
        lm_head = _to_bf16(lm_head_cpu, device)
        handles.ptrs['lm_head_w'] = _anchor(handles, lm_head)

    per_layer: list[dict] = [None] * num_layers   # type: ignore[list-item]
    for L in range(num_layers):
        base = f'model.language_model.layers.{L}.'
        ld: dict = {'type': 'full_attention', 'quant_format': 'fp8_block128'}
        ld['input_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap, base + 'input_layernorm.weight'),
            device))
        ld['post_attn_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap,
                        base + 'post_attention_layernorm.weight'),
            device))
        sa = base + 'self_attn.'
        q_w, q_s = _load_fp8_linear(handles, ld, 'q_proj', sa + 'q_proj',
                                    handles_d, wmap, device)
        k_w, k_s = _load_fp8_linear(handles, ld, 'k_proj', sa + 'k_proj',
                                    handles_d, wmap, device)
        v_w, v_s = _load_fp8_linear(handles, ld, 'v_proj', sa + 'v_proj',
                                    handles_d, wmap, device)
        _load_fp8_linear(handles, ld, 'o_proj', sa + 'o_proj',
                         handles_d, wmap, device)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
        qkv_s = torch.cat([q_s, k_s, v_s], dim=0).contiguous()
        ld['qkv_proj_w'] = _anchor(handles, qkv_w)
        ld['qkv_proj_s'] = _anchor(handles, qkv_s)
        ld['qkv_proj_N'] = int(qkv_w.shape[0])

        ld['q_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap, sa + 'q_norm.weight'), device))
        ld['k_norm_w'] = _anchor(handles, _to_bf16(
            _get_tensor(handles_d, wmap, sa + 'k_norm.weight'), device))

        mp = base + 'mlp.'
        gate_w, gate_s = _load_fp8_linear(
            handles, ld, 'mlp_gate', mp + 'gate_proj',
            handles_d, wmap, device)
        up_w, up_s = _load_fp8_linear(
            handles, ld, 'mlp_up', mp + 'up_proj',
            handles_d, wmap, device)
        gate_up_w = torch.cat([gate_w, up_w], dim=0).contiguous()
        gate_up_s = torch.cat([gate_s, up_s], dim=0).contiguous()
        ld['gate_up_w'] = _anchor(handles, gate_up_w)
        ld['gate_up_s'] = _anchor(handles, gate_up_s)
        ld['gate_up_N'] = int(gate_up_w.shape[0])
        _load_fp8_linear(handles, ld, 'mlp_down', mp + 'down_proj',
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
        'quant_format': 'fp8_block128',
        'ckpt_dir': ckpt_dir,
        'layers': per_layer,
    })
    return handles


def assert_extraction_invariants_qwen3_vl_fp8(handles: WeightHandles) -> None:
    p = handles.ptrs
    assert p.get('quant_format') == 'fp8_block128'
    if p.get('lm_head_quantized', True):
        for key in ('lm_head_fp8_w', 'lm_head_fp8_s'):
            if key not in p:
                raise AssertionError(f'missing {key}')
    elif 'lm_head_w' not in p:
        raise AssertionError('missing lm_head_w')
    layers = p.get('layers')
    assert isinstance(layers, list) and len(layers) == p['num_layers']
    required = {
        'input_norm_w', 'post_attn_norm_w', 'q_proj_w', 'q_proj_s',
        'k_proj_w', 'k_proj_s', 'v_proj_w', 'v_proj_s',
        'o_proj_w', 'o_proj_s', 'qkv_proj_w', 'qkv_proj_s',
        'qkv_proj_N', 'q_norm_w', 'k_norm_w',
        'mlp_gate_w', 'mlp_gate_s', 'mlp_up_w', 'mlp_up_s',
        'gate_up_w', 'gate_up_s', 'gate_up_N', 'mlp_down_w', 'mlp_down_s',
    }
    for i, layer in enumerate(layers):
        missing = required.difference(layer)
        if missing:
            raise AssertionError(f'layer {i} missing {sorted(missing)}')
