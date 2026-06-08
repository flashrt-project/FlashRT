"""FlashRT -- Qwen3 BF16 raw safetensors loader for ROCm.

This is the first owned-buffer step for the AMD Qwen3 pipeline. It reads the
official BF16 Hugging Face safetensors checkpoint directly, keeps tensors in
PyTorch's native Linear layout ``(out_features, in_features)``, and exposes raw
device pointers for the future hipBLASLt/kernel path.

Unlike the RTX Qwen3 path, this loader does not consume NVFP4 packed weights.
The hot-path fusion choices are still aligned with that pipeline:

* q/k/v projections are concatenated into one static ``qkv_w`` matrix.
* gate/up projections are concatenated into one static ``gate_up_w`` matrix.
* RMSNorm, q_norm/k_norm, output/down weights, embeddings, and lm_head stay BF16.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from flash_rt.models.qwen3.pipeline_rocm import Qwen3RocmDims


@dataclass
class WeightHandles:
    """Device tensor owner plus raw pointer table.

    ``ptrs`` is intentionally plain Python data so the frontend and tests can
    inspect the contract without depending on a C++ binding. ``anchors`` owns
    every device tensor whose pointer appears in ``ptrs``.
    """

    ptrs: dict[str, Any] = field(default_factory=dict)
    anchors: list[torch.Tensor] = field(default_factory=list)


def _anchor(handles: WeightHandles, t: torch.Tensor) -> int:
    handles.anchors.append(t)
    return int(t.data_ptr())


def _bf16_to_dev(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()


def _fp8_weight_tensor(t: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(torch, "float8_e4m3fnuz"):
        raise RuntimeError("ROCm FP8 weight packing requires torch.float8_e4m3fnuz")
    src = t.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    scale = torch.clamp(src.abs().max() / 240.0, min=1.0e-8).reshape(1).contiguous()
    q = (src / scale).to(torch.float8_e4m3fnuz).contiguous()
    return q, scale


def _open_shards(ckpt_dir: str):
    from safetensors import safe_open

    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError(
                f"missing model.safetensors.index.json and model.safetensors in {ckpt_dir!r}"
            )
        weight_map = {}
        with safe_open(single, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = "model.safetensors"

    shards = {
        shard: safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    return shards, weight_map


def _get_tensor(shards, weight_map: dict[str, str], key: str) -> torch.Tensor:
    try:
        shard = weight_map[key]
    except KeyError as exc:
        raise KeyError(f"tensor {key!r} not found in checkpoint") from exc
    return shards[shard].get_tensor(key)


def _load_bf16(shards, weight_map: dict[str, str], key: str, device: str) -> torch.Tensor:
    return _bf16_to_dev(_get_tensor(shards, weight_map, key), device)


def _assert_shape(key: str, t: torch.Tensor, shape: tuple[int, ...]) -> None:
    got = tuple(int(v) for v in t.shape)
    if got != tuple(shape):
        raise ValueError(f"{key} shape mismatch: got {got}, expected {shape}")


def _read_config(ckpt_dir: str) -> dict[str, Any]:
    path = os.path.join(ckpt_dir, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def _assert_config(cfg: dict[str, Any], dims: Qwen3RocmDims) -> None:
    expected = {
        "num_hidden_layers": dims.num_layers,
        "hidden_size": dims.hidden,
        "intermediate_size": dims.intermediate,
        "num_attention_heads": dims.num_q_heads,
        "num_key_value_heads": dims.num_kv_heads,
        "head_dim": dims.head_dim,
        "vocab_size": dims.vocab_size,
    }
    for key, value in expected.items():
        got = int(cfg.get(key, -1))
        if got != int(value):
            raise ValueError(f"config {key} mismatch: got {got}, expected {value}")
    rope_theta = float(cfg.get("rope_theta", dims.rope_theta))
    if abs(rope_theta - dims.rope_theta) > 1.0e-3:
        raise ValueError(
            f"config rope_theta mismatch: got {rope_theta}, expected {dims.rope_theta}"
        )
    rms_eps = float(cfg.get("rms_norm_eps", dims.rms_norm_eps))
    if abs(rms_eps - dims.rms_norm_eps) > 1.0e-12:
        raise ValueError(
            f"config rms_norm_eps mismatch: got {rms_eps}, expected {dims.rms_norm_eps}"
        )


def extract_weights_qwen3_bf16_rocm(
    ckpt_dir: str,
    *,
    device: str = "cuda",
    validate_config: bool = True,
    include_fp8: bool = False,
    fp8_modules: tuple[str, ...] = ("lm_head",),
) -> WeightHandles:
    """Load official Qwen3-8B BF16 weights onto a ROCm device.

    The returned pointer layout is the ABI seed for the future owned BF16
    pipeline. Every layer stores fused QKV and gate/up matrices, matching the
    intended compute graph instead of the Hugging Face module decomposition.
    """

    dims = Qwen3RocmDims()
    cfg = _read_config(ckpt_dir)
    if validate_config:
        _assert_config(cfg, dims)

    shards, weight_map = _open_shards(ckpt_dir)
    handles = WeightHandles()
    ptrs = handles.ptrs
    ptrs.update(
        {
            "quant_format": "bf16",
            "model_type": str(cfg.get("model_type", "qwen3")),
            "num_layers": dims.num_layers,
            "hidden": dims.hidden,
            "intermediate": dims.intermediate,
            "vocab_size": dims.vocab_size,
            "num_q_heads": dims.num_q_heads,
            "num_kv_heads": dims.num_kv_heads,
            "head_dim": dims.head_dim,
            "rotary_dim": dims.rotary_dim,
            "rope_theta": dims.rope_theta,
            "max_pos": int(cfg.get("max_position_embeddings", dims.max_pos)),
            "rms_norm_eps": dims.rms_norm_eps,
            "layer_types": ["full_attention"] * dims.num_layers,
            "layers": [],
        }
    )
    if include_fp8:
        ptrs["fp8"] = {"layers": []}

    embed = _load_bf16(shards, weight_map, "model.embed_tokens.weight", device)
    final_norm = _load_bf16(shards, weight_map, "model.norm.weight", device)
    lm_head = _load_bf16(shards, weight_map, "lm_head.weight", device)
    _assert_shape("model.embed_tokens.weight", embed, (dims.vocab_size, dims.hidden))
    _assert_shape("model.norm.weight", final_norm, (dims.hidden,))
    _assert_shape("lm_head.weight", lm_head, (dims.vocab_size, dims.hidden))

    ptrs["embed_w"] = _anchor(handles, embed)
    ptrs["final_norm_w"] = _anchor(handles, final_norm)
    ptrs["lm_head_w"] = _anchor(handles, lm_head)
    if include_fp8 and "lm_head" in fp8_modules:
        lm_head_fp8, lm_head_scale = _fp8_weight_tensor(lm_head, device)
        ptrs["fp8"]["lm_head_w"] = _anchor(handles, lm_head_fp8)
        ptrs["fp8"]["lm_head_scale"] = _anchor(handles, lm_head_scale)

    q_out = dims.num_q_heads * dims.head_dim
    kv_out = dims.num_kv_heads * dims.head_dim
    qkv_out = q_out + 2 * kv_out
    gate_up_out = 2 * dims.intermediate

    for layer_idx in range(dims.num_layers):
        prefix = f"model.layers.{layer_idx}"
        attn = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"

        input_norm = _load_bf16(
            shards, weight_map, f"{prefix}.input_layernorm.weight", device
        )
        post_attn_norm = _load_bf16(
            shards, weight_map, f"{prefix}.post_attention_layernorm.weight", device
        )
        q_norm = _load_bf16(shards, weight_map, f"{attn}.q_norm.weight", device)
        k_norm = _load_bf16(shards, weight_map, f"{attn}.k_norm.weight", device)
        q_w = _load_bf16(shards, weight_map, f"{attn}.q_proj.weight", device)
        k_w = _load_bf16(shards, weight_map, f"{attn}.k_proj.weight", device)
        v_w = _load_bf16(shards, weight_map, f"{attn}.v_proj.weight", device)
        o_w = _load_bf16(shards, weight_map, f"{attn}.o_proj.weight", device)
        gate_w = _load_bf16(shards, weight_map, f"{mlp}.gate_proj.weight", device)
        up_w = _load_bf16(shards, weight_map, f"{mlp}.up_proj.weight", device)
        down_w = _load_bf16(shards, weight_map, f"{mlp}.down_proj.weight", device)

        _assert_shape(f"{prefix}.input_layernorm.weight", input_norm, (dims.hidden,))
        _assert_shape(
            f"{prefix}.post_attention_layernorm.weight", post_attn_norm, (dims.hidden,)
        )
        _assert_shape(f"{attn}.q_norm.weight", q_norm, (dims.head_dim,))
        _assert_shape(f"{attn}.k_norm.weight", k_norm, (dims.head_dim,))
        _assert_shape(f"{attn}.q_proj.weight", q_w, (q_out, dims.hidden))
        _assert_shape(f"{attn}.k_proj.weight", k_w, (kv_out, dims.hidden))
        _assert_shape(f"{attn}.v_proj.weight", v_w, (kv_out, dims.hidden))
        _assert_shape(f"{attn}.o_proj.weight", o_w, (dims.hidden, q_out))
        _assert_shape(f"{mlp}.gate_proj.weight", gate_w, (dims.intermediate, dims.hidden))
        _assert_shape(f"{mlp}.up_proj.weight", up_w, (dims.intermediate, dims.hidden))
        _assert_shape(f"{mlp}.down_proj.weight", down_w, (dims.hidden, dims.intermediate))

        qkv_w = torch.cat((q_w, k_w, v_w), dim=0).contiguous()
        gate_up_w = torch.cat((gate_w, up_w), dim=0).contiguous()
        _assert_shape(f"{attn}.qkv_fused.weight", qkv_w, (qkv_out, dims.hidden))
        _assert_shape(f"{mlp}.gate_up_fused.weight", gate_up_w, (gate_up_out, dims.hidden))

        layer = {
            "type": "full_attention",
            "quant_format": "bf16",
            "input_norm_w": _anchor(handles, input_norm),
            "post_attn_norm_w": _anchor(handles, post_attn_norm),
            "q_norm_w": _anchor(handles, q_norm),
            "k_norm_w": _anchor(handles, k_norm),
            "qkv_w": _anchor(handles, qkv_w),
            "qkv_out": qkv_out,
            "q_out": q_out,
            "kv_out": kv_out,
            "o_w": _anchor(handles, o_w),
            "gate_up_w": _anchor(handles, gate_up_w),
            "gate_up_out": gate_up_out,
            "down_w": _anchor(handles, down_w),
        }
        ptrs["layers"].append(layer)

        if include_fp8 and "layers" in fp8_modules:
            fp8_layer: dict[str, Any] = {}
            for name, value in (
                ("qkv_w", qkv_w),
                ("o_w", o_w),
                ("gate_up_w", gate_up_w),
                ("down_w", down_w),
            ):
                q, scale = _fp8_weight_tensor(value, device)
                fp8_layer[name] = _anchor(handles, q)
                fp8_layer[name + "_scale"] = _anchor(handles, scale)
            ptrs["fp8"]["layers"].append(fp8_layer)

        del q_w, k_w, v_w, gate_w, up_w

    return handles


def summarize_qwen3_bf16_rocm_weights(handles: WeightHandles) -> dict[str, Any]:
    ptrs = handles.ptrs
    return {
        "quant_format": ptrs.get("quant_format"),
        "num_layers": ptrs.get("num_layers"),
        "hidden": ptrs.get("hidden"),
        "intermediate": ptrs.get("intermediate"),
        "vocab_size": ptrs.get("vocab_size"),
        "num_q_heads": ptrs.get("num_q_heads"),
        "num_kv_heads": ptrs.get("num_kv_heads"),
        "head_dim": ptrs.get("head_dim"),
        "anchors": len(handles.anchors),
        "layer0_keys": sorted(ptrs["layers"][0].keys()) if ptrs.get("layers") else [],
    }
