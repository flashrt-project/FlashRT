"""Build a FlashRT Qwen3-VL checkpoint (NVFP4 LLM + BF16 vision tower).

Qwen ships Qwen3-VL only in BF16 / FP8; the FlashRT RTX path expects the
language stack as a ``compressed-tensors`` ``nvfp4-pack-quantized``
checkpoint (the same on-disk schema as the Qwen3-8B path,
docs/qwen3_8b_nvfp4.md §4). This tool produces a single self-contained
checkpoint directory that the ``qwen3_vl`` RTX frontend consumes:

  * language linears  -> NVFP4 (``model.layers.<i>.<lin>.weight_packed`` /
    ``.weight_scale`` fp8_e4m3 / ``.weight_global_scale`` fp32), with the
    ``model.language_model.`` prefix stripped to ``model.`` so the shared
    ``_qwen3_rtx_nvfp4_weights`` loader resolves it unchanged;
  * embed_tokens / norm / lm_head, per-layer norms and q/k norms -> BF16;
  * the whole vision tower (``model.visual.*``) -> BF16, copied verbatim;
  * a ``config.json`` with the text fields hoisted to top level (for the
    language loader) plus the original ``vision_config`` and the
    multimodal token ids (for the vision tower and the frontend);
  * tokenizer / processor side files.

The dequant convention matches the loader exactly:

    w ~= e2m1(weight_packed) * e4m3(weight_scale) / weight_global_scale

{q,k,v} and {gate,up} share a per-layer global scale so the loader's
fused-QKV / fused-gate_up GEMM stays homogeneous.

Usage:
    python tools/quantize_qwen3_vl_nvfp4.py \
        --src /path/to/Qwen3-VL-8B-Instruct \
        --dst /path/to/Qwen3-VL-8B-FlashRT-NVFP4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_MAX = 6.0
E4M3_MAX = 448.0

_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_E2M1_THRESH = (_E2M1_LEVELS[1:] + _E2M1_LEVELS[:-1]) / 2.0

QUANT_LINEARS = (
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
)
# Linears that must share one per-layer global scale (homogeneous fused GEMM).
_SHARE_GROUPS = (
    ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'),
    ('mlp.gate_proj', 'mlp.up_proj'),
)
_BF16_NORMS = (
    'input_layernorm.weight', 'post_attention_layernorm.weight',
    'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
)
_SIDE_FILES = (
    'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
    'generation_config.json', 'preprocessor_config.json',
    'video_preprocessor_config.json', 'chat_template.json',
)


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    sign = (x < 0).to(torch.uint8)
    code = torch.bucketize(x.abs(), _E2M1_THRESH.to(x.device)).to(torch.uint8)
    nib = (sign << 3) | code
    return torch.where(code == 0, torch.zeros_like(nib), nib)


def _quant_linear(w: torch.Tensor, global_scale: float):
    """NVFP4-pack a (out, in) weight with a given fp32 global scale."""
    w = w.to(torch.float32)
    out, cin = w.shape
    assert cin % 16 == 0, f'in_features {cin} not a multiple of 16'
    wb = w.view(out, cin // 16, 16)
    block_amax = wb.abs().amax(dim=-1)
    sf = (block_amax / FP4_MAX * global_scale).to(torch.float8_e4m3fn)
    eff = (sf.to(torch.float32) / global_scale).clamp_min(1e-12)
    nib = _e2m1_codes((wb / eff[..., None]).clamp(-FP4_MAX, FP4_MAX))
    nib = nib.view(out, cin)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8).contiguous()
    return packed, sf.contiguous(), torch.tensor(
        [global_scale], dtype=torch.float32)


def _global_scale(amax: float) -> float:
    return (E4M3_MAX * FP4_MAX) / max(amax, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True, help='source Qwen3-VL ckpt dir')
    ap.add_argument('--dst', required=True, help='output FlashRT ckpt dir')
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.src, 'config.json')))
    idx = json.load(
        open(os.path.join(args.src, 'model.safetensors.index.json')))
    wmap = idx['weight_map']
    n_layers = int(cfg['text_config']['num_hidden_layers'])

    shards: dict = {}

    def get(key: str) -> torch.Tensor:
        fn = wmap[key]
        if fn not in shards:
            shards[fn] = safe_open(
                os.path.join(args.src, fn), framework='pt', device='cpu')
        return shards[fn].get_tensor(key)

    os.makedirs(args.dst, exist_ok=True)
    out: dict[str, torch.Tensor] = {}
    lm = 'model.language_model.'

    out['model.embed_tokens.weight'] = get(
        lm + 'embed_tokens.weight').to(torch.bfloat16)
    out['model.norm.weight'] = get(lm + 'norm.weight').to(torch.bfloat16)
    out['lm_head.weight'] = get('lm_head.weight').to(torch.bfloat16)

    for layer in range(n_layers):
        src = f'{lm}layers.{layer}.'
        dst = f'model.layers.{layer}.'
        for nm in _BF16_NORMS:
            out[dst + nm] = get(src + nm).to(torch.bfloat16)

        gscale: dict[str, float] = {}
        for grp in _SHARE_GROUPS:
            amax = max(float(get(src + lin + '.weight').abs().max())
                       for lin in grp)
            for lin in grp:
                gscale[lin] = _global_scale(amax)
        for lin in ('self_attn.o_proj', 'mlp.down_proj'):
            gscale[lin] = _global_scale(
                float(get(src + lin + '.weight').abs().max()))

        for lin in QUANT_LINEARS:
            packed, sf, gs = _quant_linear(get(src + lin + '.weight'),
                                           gscale[lin])
            out[dst + lin + '.weight_packed'] = packed
            out[dst + lin + '.weight_scale'] = sf
            out[dst + lin + '.weight_global_scale'] = gs

    # Vision tower: copied verbatim in BF16.
    for key in wmap:
        if key.startswith('model.visual.'):
            out[key] = get(key).to(torch.bfloat16)

    save_file(out, os.path.join(args.dst, 'model.safetensors'),
              metadata={'format': 'pt'})
    json.dump(
        {'metadata': {},
         'weight_map': {k: 'model.safetensors' for k in out}},
        open(os.path.join(args.dst, 'model.safetensors.index.json'), 'w'))

    # config.json: text fields at top level (language loader) + vision_config
    # + multimodal token ids (vision tower / frontend).
    flat = dict(cfg['text_config'])
    flat['model_type'] = 'qwen3'
    flat['tie_word_embeddings'] = cfg.get('tie_word_embeddings', False)
    flat['vision_config'] = cfg['vision_config']
    for tok in ('image_token_id', 'video_token_id',
                'vision_start_token_id', 'vision_end_token_id'):
        if tok in cfg:
            flat[tok] = cfg[tok]
    flat['quantization_config'] = {
        'quant_method': 'compressed-tensors',
        'format': 'nvfp4-pack-quantized',
    }
    json.dump(flat, open(os.path.join(args.dst, 'config.json'), 'w'), indent=2)

    for fn in _SIDE_FILES:
        path = os.path.join(args.src, fn)
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(args.dst, fn))

    print(f'wrote {len(out)} tensors -> {args.dst}')


if __name__ == '__main__':
    main()
