"""FlashRT -- Qwen3.6-27B official FP8 raw loader for ROCm.

This loader establishes the AMD ABI for Qwen3.6 without going through
Transformers modules. It reads the official ModelScope/HF FP8 checkpoint
directly from safetensors and returns a pointer schema that mirrors the RTX
Qwen3.6 path:

* 48 linear-attention layers keep their Gated-DeltaNet state tensors.
* 16 full-attention layers expose Q/K/V/O FP8 block-scaled projections.
* All common MLP projections are official FP8 block-scaled weights.
* Norms, embeddings, conv/state projections stay BF16.
* lm_head stays BF16 for compatibility and may expose a cached FP8-fnuz
  block-scaled copy for full graph replay.

Important ROCm detail: the official checkpoint stores FP8 weights as
``torch.float8_e4m3fn`` with 128x128 ``weight_scale_inv`` tensors. gfx942's
current hipBLASLt path in this repo uses e4m3fnuz scalar scales, so the first
forward implementation must either route block-scale GEMMs through AITER/CK or
dequantize these weights to BF16 for the ROCm path. This file keeps the official
FP8 tensors intact and records that decision in metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from flash_rt.models.qwen36.pipeline_rocm import Qwen36RocmDims


@dataclass
class Qwen36RocmWeightHandles:
    ptrs: dict[str, Any] = field(default_factory=dict)
    anchors: list[torch.Tensor] = field(default_factory=list)


def _anchor(handles: Qwen36RocmWeightHandles, t: torch.Tensor) -> int:
    handles.anchors.append(t)
    return int(t.data_ptr())


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("text_config", cfg)


def _open_shards(ckpt_dir: str):
    from safetensors import safe_open

    idx = _read_json(os.path.join(ckpt_dir, "model.safetensors.index.json"))
    weight_map = idx["weight_map"]
    shards = {
        shard: safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    return shards, weight_map


def _get(shards, weight_map: dict[str, str], key: str) -> torch.Tensor:
    if key not in weight_map:
        raise KeyError(f"missing tensor {key!r}")
    return shards[weight_map[key]].get_tensor(key)


def _to_bf16(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()


def _eff_rmsnorm_weight(t: torch.Tensor, device: str) -> torch.Tensor:
    # Qwen3.5/Qwen3.6 RMSNorm stores delta weight; runtime uses (1 + w).
    return (1.0 + t.float()).to(device=device, dtype=torch.bfloat16).contiguous()


def _to_fp8_weight(t: torch.Tensor, device: str) -> torch.Tensor:
    # Preserve official e4m3fn on load. A future AITER path consumes this
    # directly; hipBLASLt e4m3fnuz conversion is intentionally not hidden here.
    return t.to(device=device, non_blocking=True).contiguous()


def _scale(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()


def dequant_fp8_block128_to_bf16(
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Dequantize official Qwen3.6 FP8 block-128 weights to BF16.

    The checkpoint stores weight as e4m3fn and ``weight_scale_inv`` as a
    2-D block table over (out_features / 128, in_features / 128). Empirically
    and per the RTX path's use of the scale tensor, the BF16 ROCm weight is
    ``weight.float() * weight_scale_inv`` per 128x128 block.
    """

    rows, cols = (int(w_fp8.shape[0]), int(w_fp8.shape[1]))
    w = w_fp8.to(device=device, non_blocking=True).contiguous()
    s = scale_inv.to(device=device, dtype=torch.float32, non_blocking=True)
    s = s.repeat_interleave(128, dim=0)[:rows]
    s = s.repeat_interleave(128, dim=1)[:, :cols].contiguous()
    return (w.float() * s).to(torch.bfloat16).contiguous()


def quant_bf16_block128_to_fp8(
    w_bf16: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight matrix into 128x128 block FP8 plus descales."""

    rows, cols = (int(w_bf16.shape[0]), int(w_bf16.shape[1]))
    if rows % 128 != 0 or cols % 128 != 0:
        raise ValueError(f"FP8 block quant needs multiples of 128, got {(rows, cols)}")
    max_fp8 = 448.0 if dtype is torch.float8_e4m3fn else 240.0
    blocks = (
        w_bf16.float()
        .view(rows // 128, 128, cols // 128, 128)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    scale = blocks.abs().amax(dim=(2, 3)).clamp_min(1.0e-8) / max_fp8
    q = (blocks / scale[:, :, None, None]).permute(0, 2, 1, 3)
    q = q.reshape(rows, cols).to(dtype).contiguous()
    return q, scale.to(torch.float32).contiguous()


def requant_official_fp8_block128_to_fnuz(
    w_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static gfx942-friendly requant from official e4m3fn to e4m3fnuz."""

    w_bf16 = dequant_fp8_block128_to_bf16(w_fp8, scale_inv, device)
    return quant_bf16_block128_to_fp8(w_bf16, torch.float8_e4m3fnuz)


def _assert_shape(key: str, t: torch.Tensor, shape: tuple[int, ...]) -> None:
    got = tuple(int(v) for v in t.shape)
    if got != shape:
        raise ValueError(f"{key} shape mismatch: got {got}, expected {shape}")


def _validate_config(cfg: dict[str, Any], dims: Qwen36RocmDims) -> list[str]:
    text = _text_config(cfg)
    layer_types = list(text.get("layer_types") or [])
    expected = {
        "hidden_size": dims.hidden,
        "num_hidden_layers": dims.num_layers,
        "intermediate_size": dims.intermediate,
        "num_attention_heads": dims.full_q_heads,
        "num_key_value_heads": dims.full_kv_heads,
        "head_dim": dims.full_head_dim,
        "vocab_size": dims.vocab_size,
    }
    for key, value in expected.items():
        got = int(text.get(key, -1))
        if got != value:
            raise ValueError(f"config {key} mismatch: got {got}, expected {value}")
    if len(layer_types) != dims.num_layers:
        raise ValueError(f"layer_types length {len(layer_types)} != {dims.num_layers}")
    if layer_types.count("linear_attention") != 48:
        raise ValueError("expected 48 linear_attention layers")
    if layer_types.count("full_attention") != 16:
        raise ValueError("expected 16 full_attention layers")
    return layer_types


def _load_linear(
    handles: Qwen36RocmWeightHandles,
    out: dict[str, Any],
    prefix: str,
    shards,
    weight_map: dict[str, str],
    key: str,
    device: str,
    shape: tuple[int, int],
    weight_mode: str,
) -> None:
    w_cpu = _get(shards, weight_map, key + ".weight")
    s_cpu = _get(shards, weight_map, key + ".weight_scale_inv")
    _assert_shape(key + ".weight", w_cpu, shape)
    _assert_shape(
        key + ".weight_scale_inv",
        s_cpu,
        (shape[0] // 128, shape[1] // 128),
    )
    if weight_mode in {"official_fp8", "fp8_fnuz_cached"}:
        if weight_mode == "fp8_fnuz_cached" and w_cpu.dtype is torch.uint8:
            w_cpu = w_cpu.contiguous().view(torch.float8_e4m3fnuz)
        w = _to_fp8_weight(w_cpu, device)
        s = _scale(s_cpu, device)
        out[prefix + "_w"] = _anchor(handles, w)
        out[prefix + "_s"] = _anchor(handles, s)
        out[prefix + "_dtype"] = str(w.dtype)
        out[prefix + "_scale_shape"] = tuple(int(v) for v in s.shape)
    elif weight_mode == "bf16_dequant":
        w = dequant_fp8_block128_to_bf16(w_cpu, s_cpu, device)
        out[prefix + "_w"] = _anchor(handles, w)
        out[prefix + "_dtype"] = str(w.dtype)
        out[prefix + "_scale_shape"] = tuple(int(v) for v in s_cpu.shape)
    elif weight_mode == "fp8_fnuz_requant":
        w, s = requant_official_fp8_block128_to_fnuz(w_cpu, s_cpu, device)
        out[prefix + "_w"] = _anchor(handles, w)
        out[prefix + "_s"] = _anchor(handles, s)
        out[prefix + "_dtype"] = str(w.dtype)
        out[prefix + "_scale_shape"] = tuple(int(v) for v in s.shape)
    else:
        raise ValueError(
            "weight_mode must be 'official_fp8', 'bf16_dequant', "
            "'fp8_fnuz_requant', or 'fp8_fnuz_cached', "
            f"got {weight_mode!r}"
        )


def extract_weights_qwen36_fp8_rocm(
    ckpt_dir: str,
    *,
    device: str = "cuda",
    max_layers: int | None = None,
    load_weights: bool = True,
    weight_mode: str = "official_fp8",
) -> Qwen36RocmWeightHandles:
    """Load Qwen3.6 official FP8 weights or just validate their schema.

    `max_layers` is a verification convenience for small layer subsets. The
    production pipeline passes ``None`` and loads all 64 layers.
    """

    dims = Qwen36RocmDims()
    if weight_mode not in {
        "official_fp8",
        "bf16_dequant",
        "fp8_fnuz_requant",
        "fp8_fnuz_cached",
    }:
        raise ValueError(
            "weight_mode must be 'official_fp8', 'bf16_dequant', "
            f"'fp8_fnuz_requant', or 'fp8_fnuz_cached', got {weight_mode!r}"
        )
    cfg = _read_json(os.path.join(ckpt_dir, "config.json"))
    layer_types = _validate_config(cfg, dims)
    qcfg = cfg.get("quantization_config") or {}
    if qcfg.get("weight_block_size") != [128, 128]:
        raise ValueError(f"expected FP8 weight_block_size [128, 128], got {qcfg}")

    handles = Qwen36RocmWeightHandles()
    ptrs = handles.ptrs
    ptrs.update(
        {
            "backend": "rocm",
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "quant_format": {
                "official_fp8": "official_fp8_block128",
                "bf16_dequant": "bf16_dequant_from_official_fp8",
                "fp8_fnuz_requant": "fp8_fnuz_block128_requant_from_official_fp8",
                "fp8_fnuz_cached": "fp8_fnuz_block128_cached",
            }.get(weight_mode),
            "fp8_weight_dtype": (
                "torch.float8_e4m3fnuz"
                if weight_mode in {"fp8_fnuz_requant", "fp8_fnuz_cached"}
                else "torch.float8_e4m3fn"
            ),
            "fp8_compute_plan": {
                "official_fp8": "aiter_blockscale_or_bf16_bringup",
                "bf16_dequant": "bf16_dequant_bringup",
                "fp8_fnuz_requant": "aiter_blockscale_fnuz_static_requant",
                "fp8_fnuz_cached": "aiter_blockscale_fnuz_cached",
            }.get(weight_mode),
            "weight_mode": weight_mode,
            "num_layers": dims.num_layers,
            "hidden": dims.hidden,
            "intermediate": dims.intermediate,
            "vocab_size": dims.vocab_size,
            "rms_norm_eps": dims.rms_norm_eps,
            "rope_theta": dims.rope_theta,
            "layer_types": layer_types,
            "layers": [],
        }
    )
    if not load_weights:
        return handles

    shards, weight_map = _open_shards(ckpt_dir)
    prefix_root = "model.language_model"
    embed = _to_bf16(_get(shards, weight_map, prefix_root + ".embed_tokens.weight"), device)
    final_norm = _eff_rmsnorm_weight(_get(shards, weight_map, prefix_root + ".norm.weight"), device)
    lm_head = _to_bf16(_get(shards, weight_map, "lm_head.weight"), device)
    _assert_shape("embed_tokens.weight", embed, (dims.vocab_size, dims.hidden))
    _assert_shape("norm.weight", final_norm, (dims.hidden,))
    _assert_shape("lm_head.weight", lm_head, (dims.vocab_size, dims.hidden))
    ptrs["embed_w"] = _anchor(handles, embed)
    ptrs["final_norm_eff_w"] = _anchor(handles, final_norm)
    ptrs["lm_head_w"] = _anchor(handles, lm_head)
    lm_head_fp8_key = "lm_head.weight_fp8_fnuz"
    lm_head_fp8_scale_key = "lm_head.weight_fp8_fnuz_scale_inv"
    if lm_head_fp8_key in weight_map and lm_head_fp8_scale_key in weight_map:
        lm_head_fp8 = _get(shards, weight_map, lm_head_fp8_key)
        if lm_head_fp8.dtype is torch.uint8:
            lm_head_fp8 = lm_head_fp8.contiguous().view(torch.float8_e4m3fnuz)
        lm_head_fp8_scale = _get(shards, weight_map, lm_head_fp8_scale_key)
        _assert_shape(lm_head_fp8_key, lm_head_fp8, (dims.vocab_size, dims.hidden))
        _assert_shape(
            lm_head_fp8_scale_key,
            lm_head_fp8_scale,
            (dims.vocab_size // 128, dims.hidden // 128),
        )
        ptrs["lm_head_fp8_w"] = _anchor(handles, _to_fp8_weight(lm_head_fp8, device))
        ptrs["lm_head_fp8_s"] = _anchor(handles, _scale(lm_head_fp8_scale, device))
        ptrs["lm_head_fp8_dtype"] = "torch.float8_e4m3fnuz"

    limit = dims.num_layers if max_layers is None else min(int(max_layers), dims.num_layers)
    for layer_idx in range(limit):
        layer_prefix = f"{prefix_root}.layers.{layer_idx}"
        layer_type = layer_types[layer_idx]
        ld: dict[str, Any] = {
            "type": layer_type,
            "quant_format": ptrs["quant_format"],
            "weight_mode": weight_mode,
        }
        ld["input_norm_eff_w"] = _anchor(
            handles,
            _eff_rmsnorm_weight(_get(shards, weight_map, layer_prefix + ".input_layernorm.weight"), device),
        )
        ld["post_attn_norm_eff_w"] = _anchor(
            handles,
            _eff_rmsnorm_weight(_get(shards, weight_map, layer_prefix + ".post_attention_layernorm.weight"), device),
        )
        mlp = layer_prefix + ".mlp"
        _load_linear(handles, ld, "mlp_gate", shards, weight_map, mlp + ".gate_proj", device, (dims.intermediate, dims.hidden), weight_mode)
        _load_linear(handles, ld, "mlp_up", shards, weight_map, mlp + ".up_proj", device, (dims.intermediate, dims.hidden), weight_mode)
        _load_linear(handles, ld, "mlp_down", shards, weight_map, mlp + ".down_proj", device, (dims.hidden, dims.intermediate), weight_mode)

        if layer_type == "linear_attention":
            la = layer_prefix + ".linear_attn"
            _load_linear(handles, ld, "in_proj_qkv", shards, weight_map, la + ".in_proj_qkv", device, (dims.lin_qkv_dim, dims.hidden), weight_mode)
            _load_linear(handles, ld, "in_proj_z", shards, weight_map, la + ".in_proj_z", device, (dims.lin_z_dim, dims.hidden), weight_mode)
            _load_linear(handles, ld, "out_proj", shards, weight_map, la + ".out_proj", device, (dims.hidden, dims.lin_z_dim), weight_mode)
            ld["in_proj_a_w"] = _anchor(handles, _to_bf16(_get(shards, weight_map, la + ".in_proj_a.weight"), device))
            ld["in_proj_b_w"] = _anchor(handles, _to_bf16(_get(shards, weight_map, la + ".in_proj_b.weight"), device))
            ld["conv1d_w"] = _anchor(handles, _to_bf16(_get(shards, weight_map, la + ".conv1d.weight").squeeze(1), device))
            ld["head_norm_w"] = _anchor(handles, _to_bf16(_get(shards, weight_map, la + ".norm.weight"), device))
            ld["A_log"] = _anchor(handles, _to_bf16(_get(shards, weight_map, la + ".A_log"), device))
            ld["dt_bias"] = _anchor(handles, _to_bf16(_get(shards, weight_map, la + ".dt_bias"), device))
        elif layer_type == "full_attention":
            sa = layer_prefix + ".self_attn"
            _load_linear(handles, ld, "q_proj", shards, weight_map, sa + ".q_proj", device, (dims.full_q_proj_dim, dims.hidden), weight_mode)
            _load_linear(handles, ld, "k_proj", shards, weight_map, sa + ".k_proj", device, (dims.full_kv_dim, dims.hidden), weight_mode)
            _load_linear(handles, ld, "v_proj", shards, weight_map, sa + ".v_proj", device, (dims.full_kv_dim, dims.hidden), weight_mode)
            _load_linear(handles, ld, "o_proj", shards, weight_map, sa + ".o_proj", device, (dims.hidden, dims.full_q_dim), weight_mode)
            ld["q_norm_eff_w"] = _anchor(handles, _eff_rmsnorm_weight(_get(shards, weight_map, sa + ".q_norm.weight"), device))
            ld["k_norm_eff_w"] = _anchor(handles, _eff_rmsnorm_weight(_get(shards, weight_map, sa + ".k_norm.weight"), device))
        else:
            raise ValueError(f"unknown layer type {layer_type!r}")
        ptrs["layers"].append(ld)

    return handles


def summarize_qwen36_rocm_weights(handles: Qwen36RocmWeightHandles) -> dict[str, Any]:
    ptrs = handles.ptrs
    layers = ptrs.get("layers", [])
    return {
        "backend": ptrs.get("backend"),
        "quant_format": ptrs.get("quant_format"),
        "fp8_weight_dtype": ptrs.get("fp8_weight_dtype"),
        "fp8_compute_plan": ptrs.get("fp8_compute_plan"),
        "weight_mode": ptrs.get("weight_mode"),
        "num_layers": ptrs.get("num_layers"),
        "loaded_layers": len(layers),
        "layer_type_counts": {
            "linear_attention": ptrs.get("layer_types", []).count("linear_attention"),
            "full_attention": ptrs.get("layer_types", []).count("full_attention"),
        },
        "anchors": len(handles.anchors),
        "layer0_keys": sorted(layers[0].keys()) if layers else [],
    }
