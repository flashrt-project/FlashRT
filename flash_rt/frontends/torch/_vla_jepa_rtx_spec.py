"""FlashRT — declarative checkpoint and precision spec for VLA-JEPA (RTX).

The frontend owns checkpoint IO: this module maps the LeRobot VLA-JEPA
``model.safetensors`` keys onto the tensors the RTX pipeline consumes
(bf16, on the target device, with the model's fixed weight folds already
applied). The pipeline receives the loaded weights and never opens a
checkpoint file itself.

The module also holds the delivered precision policy (NVFP4 sites, tiles,
attention route) so the frontend and the pipeline read one description of
the model instead of duplicating constants.
"""
from __future__ import annotations

import os

import torch

CHUNK = 7

#: Delivered decoder attention route. ``fp8_fp4out`` runs the repository
#: all-FP8 FMHA prefill kernel whose NVFP4 epilogue is the o-projection
#: operand; ``fp8`` is the same kernel with a bf16 O output, ``sage2`` is
#: the fp8-QK/fp16-PV kernel, and ``fa2`` is the bf16 FlashAttention-2
#: route. All four are served by the RTX Qwen3 attention backend.
DEFAULT_LLM_ATTENTION = "fp8_fp4out"
LLM_ATTENTION_ROUTES = ("fa2", "fp8", "fp8_fp4out", "sage2")

#: Prefill tile for the fp4-out attention route. The VLA port of the
#: production target's 64-row tile (16 Q / 8 KV) is selected by default;
#: other RTX Qwen3 callers always keep the 128-row entry.
DEFAULT_LLM_ATTENTION_TILE = "64"

#: ViT blocks kept in BF16. The all-FP8 tower is both faster and closer to
#: the reference chunk on this model (the tower feeds the same quantized
#: decoder, so its rounding does not add an independent error term).
VIT_BF16_BLOCKS = 0

#: Tile configurations (index into the skinny NVFP4 kernel's config table)
#: for the DiT action head's GEMM sites, keyed by (N, K). The head runs at
#: M = 40 rows and is weight-streaming bound: 16 blocks x 4 Euler steps
#: re-read every weight, so the K chunk is kept short (fewer partials for
#: the fused consumers) and the N tiles sized to cover the SM count.
DIT_FP4_CFG = {
    (2304, 768): 25,     # fused self-attention QKV    BN=256 KC=768  8 warps
    (768, 768): 0,       # cross-attention Q, out proj  BN=64  KC=256  4 warps
    (3072, 768): 23,     # FFN up                       BN=128 KC=768  8 warps
    (768, 3072): 11,     # FFN down                     BN=256 KC=512  8 warps
}

#: DiT-B attention geometry from the checkpoint config
#: (``action_num_heads`` / ``action_attention_head_dim``).
DIT_NUM_HEADS = 12
DIT_HEAD_DIM = 64

#: DiT blocks whose q/k/v attention projections stay BF16. Empty on the
#: delivered set: with attention running, W4A4 on all sixteen blocks passes
#: the 0.99 gate at min cosine 0.995712 on the 16-frame delivery set.
DIT_ATTN_BF16_BLOCKS = frozenset()

#: Decoder layers whose ``down`` projection keeps FP8. Empty: layer 2's
#: extreme SiLU tail is handled by the exact power-of-two operand
#: pre-scale in :data:`NVFP4_DOWN_ACT_SHIFT`.
NVFP4_DOWN_FP8_LAYERS = frozenset()
NVFP4_DOWN_ACT_SHIFT = {2: 4}

_LLM_PREFIX = "model.language_model."
_DIT_PREFIX = "model.action_model."


def _read_source(path: str):
    from safetensors import safe_open

    fname = os.path.join(path, "model.safetensors")
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            f"VLA-JEPA checkpoint is missing {fname}; pass the directory "
            "that holds the LeRobot policy's model.safetensors")
    handle = safe_open(fname, framework="pt", device="cpu")
    keys = set(handle.keys())

    def get(name: str) -> torch.Tensor:
        if name not in keys:
            raise KeyError(f"missing tensor {name!r} in {fname}")
        return handle.get_tensor(name)

    return get


def _bf16(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(torch.bfloat16).to(device).contiguous()


def load_llm_weights(ckpt: str, *, device: str = "cuda:0",
                     num_layers: int = 28) -> dict:
    """Load the Qwen3-VL-2B decoder weights in the pipeline's bf16 layout.

    Returns ``{"embed_w", "final_norm_w", "layers": [...]}``; each layer
    carries the Q/K/V and gate/up projections already concatenated into
    the wide-N GEMM operands the pipeline uses.
    """
    get = _read_source(ckpt)

    embed_w = _bf16(get(_LLM_PREFIX + "embed_tokens.weight"), device)
    final_norm_w = _bf16(get(_LLM_PREFIX + "norm.weight"), device)
    layers = []
    for layer in range(num_layers):
        base = f"{_LLM_PREFIX}layers.{layer}."
        attn = base + "self_attn."
        q = _bf16(get(attn + "q_proj.weight"), device)
        k = _bf16(get(attn + "k_proj.weight"), device)
        v = _bf16(get(attn + "v_proj.weight"), device)
        gate = _bf16(get(base + "mlp.gate_proj.weight"), device)
        up = _bf16(get(base + "mlp.up_proj.weight"), device)
        layers.append({
            "input_norm_w": _bf16(get(base + "input_layernorm.weight"), device),
            "post_norm_w": _bf16(get(base + "post_attention_layernorm.weight"), device),
            "qkv_w": torch.cat([q, k, v], dim=0).contiguous(),
            "q_norm_w": _bf16(get(attn + "q_norm.weight"), device),
            "k_norm_w": _bf16(get(attn + "k_norm.weight"), device),
            "o_w": _bf16(get(attn + "o_proj.weight"), device),
            "gate_up_w": torch.cat([gate, up], dim=0).contiguous(),
            "down_w": _bf16(get(base + "mlp.down_proj.weight"), device),
        })
    return {"embed_w": embed_w, "final_norm_w": final_norm_w, "layers": layers}


def load_weights(qwen_dir: str, action_ckpt: str, *, device: str = "cuda:0",
                 num_llm_layers: int = 28, num_dit_layers: int = 16,
                 num_inference_steps: int = 4) -> dict:
    """Load everything VlaJepaRtxPipeline consumes, keyed by sub-model."""
    return {
        "llm": load_llm_weights(qwen_dir, device=device,
                                num_layers=num_llm_layers),
        "dit": load_dit_weights(action_ckpt, device=device,
                                num_layers=num_dit_layers,
                                num_inference_steps=num_inference_steps),
    }


def load_dit_weights(ckpt: str, *, device: str = "cuda:0",
                     num_layers: int = 16, num_inference_steps: int = 4) -> dict:
    """Load the DiT-B flow head in the exact folds the pipeline consumes.

    The action decoder's last layer carries the Euler step size
    (``1 / num_inference_steps``) folded into its weights, and the
    self-attention blocks carry one fused q/k/v projection. Cross-attention
    K/V projections are fused across blocks into one wide-N operand.
    """
    get = _read_source(ckpt)

    def g(name: str) -> torch.Tensor:
        return get(_DIT_PREFIX + name)

    out = {
        "d_act_enc": {n: {t: _bf16(g(f"action_encoder.{n}.{t}"), device)
                          for t in ("weight", "bias")}
                      for n in ("layer1", "layer2", "layer3")},
        "d_state_enc": {n: {t: _bf16(g(f"state_encoder.{n}.{t}"), device)
                            for t in ("weight", "bias")}
                        for n in ("layer1", "layer2")},
        "d_future": _bf16(g("future_tokens.weight"), device),
        "d_pos_emb": _bf16(g("position_embedding.weight"), device),
        "d_proj_out1": {t: _bf16(g(f"model.proj_out_1.{t}"), device)
                        for t in ("weight", "bias")},
        "d_proj_out2": {t: _bf16(g(f"model.proj_out_2.{t}"), device)
                        for t in ("weight", "bias")},
    }
    time_emb = "model.timestep_encoder.timestep_embedder."
    out["d_time_emb"] = {
        n: {t: g(f"{time_emb}{n}.{t}").float().to(device).contiguous()
            for t in ("weight", "bias")}
        for n in ("linear_1", "linear_2")}
    out["d_act_dec"] = {n: {t: _bf16(g(f"action_decoder.{n}.{t}"), device)
                            for t in ("weight", "bias")}
                        for n in ("layer1", "layer2")}
    dt = 1.0 / float(num_inference_steps)
    out["d_act_dec"]["layer2"] = {
        t: _bf16(g(f"action_decoder.layer2.{t}").float() * dt, device)
        for t in ("weight", "bias")}

    blocks = []
    for i in range(num_layers):
        base = f"model.transformer_blocks.{i}."
        block = {
            "cross": (i % 2 == 0),
            "norm1": {t: _bf16(g(base + f"norm1.linear.{t}"), device)
                      for t in ("weight", "bias")},
            "q": {t: _bf16(g(base + f"attn1.to_q.{t}"), device)
                  for t in ("weight", "bias")},
            "k": {t: _bf16(g(base + f"attn1.to_k.{t}"), device)
                  for t in ("weight", "bias")},
            "v": {t: _bf16(g(base + f"attn1.to_v.{t}"), device)
                  for t in ("weight", "bias")},
            "o": {"weight": _bf16(g(base + "attn1.to_out.0.weight"), device),
                  "bias": _bf16(g(base + "attn1.to_out.0.bias"), device)},
            "ff1": {t: _bf16(g(base + f"ff.net.0.proj.{t}"), device)
                    for t in ("weight", "bias")},
            "ff2": {t: _bf16(g(base + f"ff.net.2.{t}"), device)
                    for t in ("weight", "bias")},
        }
        if not block["cross"]:
            block["qkv"] = {
                "weight": torch.cat([block["q"]["weight"], block["k"]["weight"],
                                     block["v"]["weight"]], dim=0).contiguous(),
                "bias": torch.cat([block["q"]["bias"], block["k"]["bias"],
                                   block["v"]["bias"]], dim=0).contiguous(),
            }
        blocks.append(block)
    out["blocks"] = blocks
    cross = [block for block in blocks if block["cross"]]
    out["d_cond_kv"] = {
        "weight": torch.cat([block["k"]["weight"] for block in cross]
                            + [block["v"]["weight"] for block in cross],
                            dim=0).contiguous(),
        "bias": torch.cat([block["k"]["bias"] for block in cross]
                          + [block["v"]["bias"] for block in cross],
                          dim=0).contiguous(),
    }
    return out
