#!/usr/bin/env python3
"""Quantize a BF16 Qwen3-VL checkpoint to the official block-128 FP8 layout.

The official Qwen3-VL-8B release ships an FP8 checkpoint (e4m3 ``weight`` +
128x128 ``weight_scale_inv`` block scales). The 2B release ships only BF16, so
this tool produces a byte-compatible FP8 checkpoint for the SM89 block-128 FP8
language path.

It quantizes exactly the language-stack linear weights
(``model.language_model.layers.*`` q/k/v/o + gate/up/down ``.weight``) to
``torch.float8_e4m3fn`` plus a fp32 ``.weight_scale_inv`` of shape
``(N/128, K/128)`` where each entry is ``amax / 448`` over its 128x128 block
(multiply-to-dequant) -- the same convention the SM89 GEMM/GEMV kernels and the
``_qwen3_vl_fp8_weights`` loader expect. Every other tensor (norms,
``embed_tokens``, the whole BF16 vision tower) is copied through unchanged.

Tied embeddings: the 2B ties ``lm_head`` to ``embed_tokens`` and ships no
``lm_head.weight``; the loader synthesises it from the embedding matrix, so this
tool does not materialise a separate (0.6 GB) lm_head.

Usage::

    python scripts/quantize_qwen3_vl_to_fp8_block128.py \
        --src /path/to/Qwen3-VL-2B-Instruct \
        --dst /path/to/Qwen3-VL-2B-Instruct-FP8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_BLOCK = 128
_FP8_MAX = 448.0

# Suffixes of language-stack linear weights that get FP8-quantized.
_QUANT_SUFFIXES = (
    '.self_attn.q_proj.weight',
    '.self_attn.k_proj.weight',
    '.self_attn.v_proj.weight',
    '.self_attn.o_proj.weight',
    '.mlp.gate_proj.weight',
    '.mlp.up_proj.weight',
    '.mlp.down_proj.weight',
)


def _is_quant_target(key: str) -> bool:
    if not key.startswith('model.language_model.layers.'):
        return False
    return any(key.endswith(suf) for suf in _QUANT_SUFFIXES)


def _quantize_block128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16/FP32 [N, K] -> (e4m3 [N, K], fp32 scale [N/128, K/128]).

    scale = amax/448 per 128x128 block; dequant is ``q * scale`` (the loader's
    ``weight_scale_inv`` semantics).
    """
    N, K = w.shape
    if N % _BLOCK != 0 or K % _BLOCK != 0:
        raise ValueError(f'weight {tuple(w.shape)} not divisible by {_BLOCK}')
    wf = w.detach().to(torch.float32)
    view = wf.view(N // _BLOCK, _BLOCK, K // _BLOCK, _BLOCK)
    amax = view.abs().amax(dim=(1, 3))
    scale = (amax / _FP8_MAX).clamp_min(1e-12)
    q = (view / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    q = q.to(torch.float8_e4m3fn).view(N, K).contiguous()
    return q, scale.to(torch.float32).contiguous()


def _iter_src_tensors(src: str):
    """Yield (key, tensor) over a sharded or single-file safetensors ckpt."""
    idx_path = os.path.join(src, 'model.safetensors.index.json')
    if os.path.isfile(idx_path):
        wmap = json.load(open(idx_path))['weight_map']
        shards = sorted(set(wmap.values()))
        for shard in shards:
            with safe_open(os.path.join(src, shard), framework='pt',
                           device='cpu') as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
    else:
        single = os.path.join(src, 'model.safetensors')
        if not os.path.isfile(single):
            raise RuntimeError(f'no safetensors found under {src}')
        with safe_open(single, framework='pt', device='cpu') as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _copy_aux_files(src: str, dst: str) -> None:
    """Copy tokenizer / processor / template config alongside the weights."""
    names = (
        'config.json', 'generation_config.json', 'tokenizer.json',
        'tokenizer_config.json', 'vocab.json', 'merges.txt',
        'chat_template.json', 'preprocessor_config.json',
        'video_preprocessor_config.json', 'configuration.json',
    )
    for name in names:
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, name))


def _write_quant_config(src: str, dst: str) -> None:
    """Add an fp8 block-128 quantization_config so the ckpt self-describes."""
    cfg = json.load(open(os.path.join(src, 'config.json')))
    cfg['quantization_config'] = {
        'activation_scheme': 'dynamic',
        'fmt': 'e4m3',
        'quant_method': 'fp8',
        'weight_block_size': [_BLOCK, _BLOCK],
        'modules_to_not_convert': ['lm_head', 'model.visual'],
    }
    with open(os.path.join(dst, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def _expected_quant_linears(src: str) -> int:
    cfg = json.load(open(os.path.join(src, 'config.json')))
    return int(cfg['text_config']['num_hidden_layers']) * len(_QUANT_SUFFIXES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, help='BF16 source checkpoint dir')
    ap.add_argument('--dst', required=True, help='output FP8 checkpoint dir')
    ap.add_argument('--shard-bytes', type=int, default=4_000_000_000,
                    help='target max bytes per output shard')
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)
    os.makedirs(dst, exist_ok=True)

    out: dict[str, torch.Tensor] = {}
    n_quant = 0
    n_copy = 0
    expected_quant = _expected_quant_linears(src)
    for key, t in _iter_src_tensors(src):
        if _is_quant_target(key):
            q, scale = _quantize_block128(t)
            out[key] = q
            out[key[:-len('.weight')] + '.weight_scale_inv'] = scale
            n_quant += 1
        else:
            # Keep norms / embeddings / vision tower in their native dtype
            # (BF16 for this checkpoint family).
            out[key] = t.detach().contiguous()
            n_copy += 1

    if n_quant != expected_quant:
        raise RuntimeError(
            f'quantized {n_quant} language linears, expected '
            f'{expected_quant}; check checkpoint layout under {src}')

    # Shard by accumulated byte size, mirroring the HF safetensors layout.
    keys = list(out.keys())
    shards: list[list[str]] = [[]]
    cur = 0
    for key in keys:
        nbytes = out[key].numel() * out[key].element_size()
        if cur > 0 and cur + nbytes > args.shard_bytes:
            shards.append([])
            cur = 0
        shards[-1].append(key)
        cur += nbytes

    total = len(shards)
    weight_map: dict[str, str] = {}
    if total == 1:
        shard_name = 'model.safetensors'
        save_file({k: out[k] for k in shards[0]}, os.path.join(dst, shard_name),
                  metadata={'format': 'pt'})
        for k in shards[0]:
            weight_map[k] = shard_name
    else:
        for i, shard_keys in enumerate(shards):
            shard_name = f'model-{i + 1:05d}-of-{total:05d}.safetensors'
            save_file({k: out[k] for k in shard_keys},
                      os.path.join(dst, shard_name),
                      metadata={'format': 'pt'})
            for k in shard_keys:
                weight_map[k] = shard_name

    total_bytes = sum(out[k].numel() * out[k].element_size() for k in keys)
    with open(os.path.join(dst, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {'total_size': total_bytes},
                   'weight_map': weight_map}, f, indent=2)

    _copy_aux_files(src, dst)
    _write_quant_config(src, dst)
    print(f'quantized {n_quant} linears, copied {n_copy} tensors '
          f'-> {total} shard(s), {total_bytes / 1e9:.2f} GB at {dst}')


if __name__ == '__main__':
    main()
