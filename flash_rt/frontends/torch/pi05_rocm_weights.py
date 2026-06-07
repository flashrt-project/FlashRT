"""ROCm Pi0.5 torch weight extraction helpers.

The ROCm BF16 pipeline uses PyTorch/hipBLASLt linear ABI:

    out = x @ weight.T + bias

so all Linear weights in this helper are kept in PyTorch's native
``(out_features, in_features)`` layout. This is intentionally different from
the RTX pipeline's cuBLASLt-oriented converted layout.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.models.pi05.pipeline_rocm import (
    ACTION_DIM,
    DEC_D,
    DEC_H,
    DEC_L,
    DEC_NH,
    DEC_NKV,
    DEC_HD,
    NUM_STEPS_DEFAULT,
)


def _interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
        .reshape(num_heads, 2, head_dim // 2, in_dim)
        .permute(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
        .contiguous()
    )


def _bf16_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()


def _linear_weight(module) -> torch.Tensor:
    return _bf16_tensor(module.weight)


def _linear_bias(module) -> torch.Tensor:
    return _bf16_tensor(module.bias)


def _fp8_weight_tensor(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "float8_e4m3fnuz"):
        raise RuntimeError("ROCm FP8 weight packing requires torch.float8_e4m3fnuz")
    src = t.detach().to(device="cuda", dtype=torch.float32).contiguous()
    scale = torch.clamp(src.abs().max() / 240.0, min=1.0e-8).reshape(1)
    q = (src / scale).to(torch.float8_e4m3fnuz).contiguous()
    return q, scale.contiguous()


def _fp8_key(name: str, layer: int | None = None) -> str:
    return name if layer is None else f"{name}_{int(layer)}"


def _add_fp8_weight(
    weights: dict[str, Any],
    name: str,
    layer: int | None = None,
) -> None:
    value = weights[name] if layer is None else weights[name][layer]
    weights["fp8"][_fp8_key(name, layer)] = _fp8_weight_tensor(value)


def _add_fp8_weight_tensor(
    weights: dict[str, Any],
    name: str,
    value: torch.Tensor,
    layer: int | None = None,
) -> None:
    weights["fp8"][_fp8_key(name, layer)] = _fp8_weight_tensor(value)


def _to_np_u16(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _precompute_decoder_styles_from_openpi_model(
    model,
    *,
    chunk_size: int,
    num_steps: int = NUM_STEPS_DEFAULT,
) -> dict[str, np.ndarray]:
    """Precompute Pi0.5 decoder time/style BF16 buffers for ROCm upload."""
    bf16 = torch.bfloat16
    device = "cuda"
    dt = -1.0 / int(num_steps)
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float32, device=device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    rows = []
    for _ in range(int(num_steps)):
        sinusoid = t * (1.0 / period) * 2 * math.pi
        rows.append(torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1))
        t = t + dt
    time_schedule = torch.stack(rows, dim=0).to(bf16)

    time_in = model.time_mlp_in
    time_out = model.time_mlp_out
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    final_norm = model.paligemma_with_expert.gemma_expert.model.norm

    time_emb = torch.empty(num_steps, chunk_size, DEC_D, dtype=bf16, device=device)
    style_attn = torch.empty(
        num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device=device
    )
    style_ffn = torch.empty(
        num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device=device
    )
    style_final = torch.empty(num_steps, chunk_size, 3 * DEC_D, dtype=bf16, device=device)

    for step in range(int(num_steps)):
        te = time_schedule[step : step + 1]
        tmp = F.linear(te, _bf16_tensor(time_in.weight), _bf16_tensor(time_in.bias))
        tmp = F.silu(tmp.float()).to(bf16)
        tmp = F.linear(tmp, _bf16_tensor(time_out.weight), _bf16_tensor(time_out.bias))
        tmp = F.silu(tmp.float()).to(bf16)
        expanded = tmp.expand(chunk_size, -1).contiguous()
        time_emb[step] = expanded

        for i, layer in enumerate(layers):
            style_attn[step, i] = F.linear(
                expanded,
                _bf16_tensor(layer.input_layernorm.dense.weight),
                _bf16_tensor(layer.input_layernorm.dense.bias),
            )
            style_ffn[step, i] = F.linear(
                expanded,
                _bf16_tensor(layer.post_attention_layernorm.dense.weight),
                _bf16_tensor(layer.post_attention_layernorm.dense.bias),
            )
        style_final[step] = F.linear(
            expanded,
            _bf16_tensor(final_norm.dense.weight),
            _bf16_tensor(final_norm.dense.bias),
        )

    return {
        "time_emb": _to_np_u16(time_emb),
        "style_attn": _to_np_u16(style_attn),
        "style_ffn": _to_np_u16(style_ffn),
        "style_final": _to_np_u16(style_final),
    }


def build_rocm_vision_weights_from_openpi_model(
    model,
    *,
    chunk_size: int = 10,
    num_steps: int = NUM_STEPS_DEFAULT,
    include_fp8: bool = False,
) -> dict[str, Any]:
    """Build the BF16 Pi0.5 ROCm weight table for ``Pi05PipelineRocm``.

    The returned dict owns tensor references, so its lifetime must cover every
    pipeline call that consumes its raw pointers.
    """
    vision = (
        model.paligemma_with_expert
        .paligemma
        .model
        .vision_tower
        .vision_model
    )
    emb = vision.embeddings

    # Conv2d weight is (out, in, h, w); patch_im2col emits rows in (h, w, c).
    patch_w = (
        emb.patch_embedding.weight.detach()
        .to(device="cuda", dtype=torch.bfloat16)
        .permute(0, 2, 3, 1)
        .reshape(emb.patch_embedding.out_channels, -1)
        .contiguous()
    )

    weights: dict[str, Any] = {
        "vision_patch_embedding_w": patch_w,
        "vision_patch_embedding_b": _bf16_tensor(emb.patch_embedding.bias),
        "vision_position_embedding": _bf16_tensor(emb.position_embedding.weight),
        "vision_attn_qkv_w": [],
        "vision_attn_qkv_b": [],
        "vision_attn_o_w": [],
        "vision_attn_o_b": [],
        "vision_ffn_up_w": [],
        "vision_ffn_up_b": [],
        "vision_ffn_down_w": [],
        "vision_ffn_down_b": [],
        "vision_pre_attn_norm_w": [],
        "vision_pre_attn_norm_b": [],
        "vision_pre_ffn_norm_w": [],
        "vision_pre_ffn_norm_b": [],
        "vision_final_norm_w": _bf16_tensor(vision.post_layernorm.weight),
        "vision_final_norm_b": _bf16_tensor(vision.post_layernorm.bias),
    }

    projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear
    weights["encoder_multi_modal_projector_w"] = _linear_weight(projector)
    weights["encoder_multi_modal_projector_b"] = _linear_bias(projector)
    lang_layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    weights["encoder_attn_qkv_w"] = []
    weights["encoder_attn_o_w"] = []
    weights["encoder_input_norm_w"] = []
    weights["encoder_post_attn_norm_w"] = []
    weights["encoder_ffn_gate_w"] = []
    weights["encoder_ffn_up_w"] = []
    weights["encoder_ffn_down_w"] = []
    weights["decoder_action_in_proj_w"] = _linear_weight(model.action_in_proj)
    weights["decoder_action_in_proj_b"] = _linear_bias(model.action_in_proj)
    out_scale = -1.0 / float(num_steps)
    weights["decoder_action_out_proj_w"] = _bf16_tensor(
        model.action_out_proj.weight.detach() * out_scale
    )
    weights["decoder_action_out_proj_b"] = _bf16_tensor(
        model.action_out_proj.bias.detach() * out_scale
    )
    weights["decoder_attn_qkv_w"] = []
    weights["decoder_attn_o_w"] = []
    weights["decoder_ffn_gate_w"] = []
    weights["decoder_ffn_up_w"] = []
    weights["decoder_ffn_down_w"] = []

    for layer in vision.encoder.layers:
        attn = layer.self_attn
        weights["vision_attn_qkv_w"].append(
            torch.cat(
                [
                    _linear_weight(attn.q_proj),
                    _linear_weight(attn.k_proj),
                    _linear_weight(attn.v_proj),
                ],
                dim=0,
            ).contiguous()
        )
        weights["vision_attn_qkv_b"].append(
            torch.cat(
                [
                    _linear_bias(attn.q_proj),
                    _linear_bias(attn.k_proj),
                    _linear_bias(attn.v_proj),
                ],
                dim=0,
            ).contiguous()
        )
        weights["vision_attn_o_w"].append(_linear_weight(attn.out_proj))
        weights["vision_attn_o_b"].append(_linear_bias(attn.out_proj))

        weights["vision_ffn_up_w"].append(_linear_weight(layer.mlp.fc1))
        weights["vision_ffn_up_b"].append(_linear_bias(layer.mlp.fc1))
        weights["vision_ffn_down_w"].append(_linear_weight(layer.mlp.fc2))
        weights["vision_ffn_down_b"].append(_linear_bias(layer.mlp.fc2))

        weights["vision_pre_attn_norm_w"].append(_bf16_tensor(layer.layer_norm1.weight))
        weights["vision_pre_attn_norm_b"].append(_bf16_tensor(layer.layer_norm1.bias))
        weights["vision_pre_ffn_norm_w"].append(_bf16_tensor(layer.layer_norm2.weight))
        weights["vision_pre_ffn_norm_b"].append(_bf16_tensor(layer.layer_norm2.bias))

    for layer in lang_layers:
        attn = layer.self_attn
        weights["encoder_attn_qkv_w"].append(
            torch.cat(
                [
                    _interleave_qk(_linear_weight(attn.q_proj), 8),
                    _interleave_qk(_linear_weight(attn.k_proj), 1),
                    _linear_weight(attn.v_proj),
                ],
                dim=0,
            ).contiguous()
        )
        weights["encoder_attn_o_w"].append(_linear_weight(attn.o_proj))
        weights["encoder_input_norm_w"].append(_bf16_tensor(layer.input_layernorm.weight))
        weights["encoder_post_attn_norm_w"].append(
            _bf16_tensor(layer.post_attention_layernorm.weight)
        )
        weights["encoder_ffn_gate_w"].append(_linear_weight(layer.mlp.gate_proj))
        weights["encoder_ffn_up_w"].append(_linear_weight(layer.mlp.up_proj))
        weights["encoder_ffn_down_w"].append(_linear_weight(layer.mlp.down_proj))

    expert_layers = model.paligemma_with_expert.gemma_expert.model.layers
    for layer in expert_layers:
        attn = layer.self_attn
        weights["decoder_attn_qkv_w"].append(
            torch.cat(
                [
                    _interleave_qk(_linear_weight(attn.q_proj), DEC_NH),
                    _interleave_qk(_linear_weight(attn.k_proj), DEC_NKV),
                    _linear_weight(attn.v_proj),
                ],
                dim=0,
            ).contiguous()
        )
        weights["decoder_attn_o_w"].append(_linear_weight(attn.o_proj))
        weights["decoder_ffn_gate_w"].append(_linear_weight(layer.mlp.gate_proj))
        weights["decoder_ffn_up_w"].append(_linear_weight(layer.mlp.up_proj))
        weights["decoder_ffn_down_w"].append(_linear_weight(layer.mlp.down_proj))

    weights["precomputed"] = _precompute_decoder_styles_from_openpi_model(
        model,
        chunk_size=int(chunk_size),
        num_steps=int(num_steps),
    )
    if include_fp8:
        weights["fp8"] = {}
        for name in (
            "vision_patch_embedding_w",
            "encoder_multi_modal_projector_w",
            "decoder_action_in_proj_w",
            "decoder_action_out_proj_w",
        ):
            _add_fp8_weight(weights, name)
        for i in range(len(weights["vision_attn_qkv_w"])):
            for name in (
                "vision_attn_qkv_w",
                "vision_attn_o_w",
                "vision_ffn_up_w",
                "vision_ffn_down_w",
            ):
                _add_fp8_weight(weights, name, i)
        for i in range(len(weights["encoder_attn_qkv_w"])):
            for name in (
                "encoder_attn_qkv_w",
                "encoder_attn_o_w",
                "encoder_ffn_down_w",
            ):
                _add_fp8_weight(weights, name, i)
            gate_up = torch.cat(
                (weights["encoder_ffn_gate_w"][i], weights["encoder_ffn_up_w"][i]),
                dim=0,
            ).contiguous()
            _add_fp8_weight_tensor(weights, "encoder_ffn_gate_up_w", gate_up, i)
        for i in range(len(weights["decoder_attn_qkv_w"])):
            for name in (
                "decoder_attn_qkv_w",
                "decoder_attn_o_w",
                "decoder_ffn_down_w",
            ):
                _add_fp8_weight(weights, name, i)
            gate_up = torch.cat(
                (weights["decoder_ffn_gate_w"][i], weights["decoder_ffn_up_w"][i]),
                dim=0,
            ).contiguous()
            _add_fp8_weight_tensor(weights, "decoder_ffn_gate_up_w", gate_up, i)

    return weights


def weight_ptr(weights: dict[str, Any], name: str, layer: int | None = None) -> int:
    value = weights[name] if layer is None else weights[name][layer]
    return int(value.data_ptr() if hasattr(value, "data_ptr") else value)


__all__ = ["build_rocm_vision_weights_from_openpi_model", "weight_ptr"]
