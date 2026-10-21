"""FlashRT — Qwen3-4B BF16 weight loader (HF CausalLM layout).

Loads HuggingFace-style Qwen3 dense weights::

    model.embed_tokens.weight
    model.layers.<i>.{input_layernorm,post_attention_layernorm}.weight
    model.layers.<i>.self_attn.{q,k,v,o}_proj.weight
    model.layers.<i>.self_attn.{q,k}_norm.weight
    model.layers.<i>.mlp.{gate,up,down}_proj.weight
    model.norm.weight
    lm_head.weight   (may be tied to embed_tokens)

Supports:
  * checkpoint directory (config.json + safetensors / pytorch_model.bin)
  * in-memory ``state_dict`` injection for random-weight HF cosine tests
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from flash_rt.models.qwen3.pipeline_rtx import Qwen3_4BDims

BF16 = torch.bfloat16
_DIMS = Qwen3_4BDims()


@dataclass
class WeightHandles:
    """BF16 weight tensors kept alive for the frontend forward path."""

    ptrs: dict = field(default_factory=dict)
    anchors: list = field(default_factory=list)


def _cfg_from_hf_dict(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a HF Qwen3 config.json / PretrainedConfig.__dict__."""
    rope_theta = cfg.get("rope_theta", _DIMS.rope_theta)
    if isinstance(cfg.get("rope_parameters"), dict):
        rope_theta = cfg["rope_parameters"].get("rope_theta", rope_theta)
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        head_dim = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
    return {
        "hidden": int(cfg["hidden_size"]),
        "num_layers": int(cfg["num_hidden_layers"]),
        "num_q_heads": int(cfg["num_attention_heads"]),
        "num_kv_heads": int(cfg["num_key_value_heads"]),
        "head_dim": int(head_dim),
        "intermediate": int(cfg["intermediate_size"]),
        "vocab_size": int(cfg["vocab_size"]),
        "rms_norm_eps": float(cfg.get("rms_norm_eps", _DIMS.rms_norm_eps)),
        "rope_theta": float(rope_theta),
        "max_pos": int(cfg.get("max_position_embeddings", _DIMS.max_pos)),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", True)),
    }


def qwen3_4b_config_dict() -> dict[str, Any]:
    """Authoritative Qwen/Qwen3-4B config values (no Hub download)."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": _DIMS.vocab_size,
        "hidden_size": _DIMS.hidden,
        "intermediate_size": _DIMS.intermediate,
        "num_hidden_layers": _DIMS.num_layers,
        "num_attention_heads": _DIMS.num_q_heads,
        "num_key_value_heads": _DIMS.num_kv_heads,
        "head_dim": _DIMS.head_dim,
        "max_position_embeddings": _DIMS.max_pos,
        "rms_norm_eps": _DIMS.rms_norm_eps,
        "rope_theta": _DIMS.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_cache": True,
    }


def assert_qwen3_4b_dims(cfg: Mapping[str, Any]) -> None:
    d = _DIMS
    checks = (
        ("hidden", d.hidden, cfg["hidden"]),
        ("num_layers", d.num_layers, cfg["num_layers"]),
        ("num_q_heads", d.num_q_heads, cfg["num_q_heads"]),
        ("num_kv_heads", d.num_kv_heads, cfg["num_kv_heads"]),
        ("head_dim", d.head_dim, cfg["head_dim"]),
        ("intermediate", d.intermediate, cfg["intermediate"]),
        ("vocab_size", d.vocab_size, cfg["vocab_size"]),
    )
    for name, want, got in checks:
        if int(want) != int(got):
            raise ValueError(
                f"Qwen3-4B dim mismatch: {name}={got} != expected {want}")


def _to_dev_bf16(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.detach().to(device=device, dtype=BF16).contiguous()


def _load_state_dict_from_dir(ckpt_dir: str) -> dict[str, torch.Tensor]:
    idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    single = os.path.join(ckpt_dir, "model.safetensors")
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.isfile(idx):
        from safetensors import safe_open
        wmap = json.load(open(idx))["weight_map"]
        out: dict[str, torch.Tensor] = {}
        by_shard: dict[str, list[str]] = {}
        for k, shard in wmap.items():
            by_shard.setdefault(shard, []).append(k)
        for shard, keys in by_shard.items():
            with safe_open(os.path.join(ckpt_dir, shard),
                           framework="pt", device="cpu") as f:
                for k in keys:
                    out[k] = f.get_tensor(k)
        return out
    if os.path.isfile(single):
        from safetensors.torch import load_file
        return load_file(single)
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"No BF16 weights found under {ckpt_dir!r} "
        f"(expected model.safetensors[.index.json] or pytorch_model.bin)")


def extract_weights_qwen3_4b_bf16(
    *,
    device: str = "cuda:0",
    checkpoint_path: str | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    config: Mapping[str, Any] | None = None,
    assert_dims: bool = True,
) -> tuple[WeightHandles, dict[str, Any]]:
    """Build WeightHandles from a ckpt dir and/or an in-memory state_dict.

    Exactly one of ``checkpoint_path`` / ``state_dict`` must provide weights.
    ``config`` may come from a HF config object, a config.json dict, or
    :func:`qwen3_4b_config_dict`.
    """
    if config is None:
        if checkpoint_path is None:
            config = qwen3_4b_config_dict()
        else:
            cfg_path = os.path.join(checkpoint_path, "config.json")
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError(f"missing config.json in {checkpoint_path}")
            config = json.load(open(cfg_path))
    # Accept both raw HF JSON and already-normalized cfg dicts.
    if "hidden_size" in config:
        cfg = _cfg_from_hf_dict(config)
    else:
        cfg = dict(config)
        if "hidden" not in cfg:
            raise ValueError("config missing hidden / hidden_size")
    if assert_dims:
        assert_qwen3_4b_dims(cfg)

    if state_dict is None:
        if checkpoint_path is None:
            raise ValueError("need checkpoint_path or state_dict")
        state_dict = _load_state_dict_from_dir(checkpoint_path)
    sd = state_dict

    def g(key: str) -> torch.Tensor:
        if key not in sd:
            raise KeyError(f"missing weight {key!r}")
        return _to_dev_bf16(sd[key], device)

    nl = cfg["num_layers"]
    layers = []
    for i in range(nl):
        p = f"model.layers.{i}"
        layers.append({
            "in_norm": g(f"{p}.input_layernorm.weight"),
            "q": g(f"{p}.self_attn.q_proj.weight"),
            "k": g(f"{p}.self_attn.k_proj.weight"),
            "v": g(f"{p}.self_attn.v_proj.weight"),
            "o": g(f"{p}.self_attn.o_proj.weight"),
            "qn": g(f"{p}.self_attn.q_norm.weight"),
            "kn": g(f"{p}.self_attn.k_norm.weight"),
            "post_norm": g(f"{p}.post_attention_layernorm.weight"),
            "gate": g(f"{p}.mlp.gate_proj.weight"),
            "up": g(f"{p}.mlp.up_proj.weight"),
            "down": g(f"{p}.mlp.down_proj.weight"),
        })

    embed = g("model.embed_tokens.weight")
    if "lm_head.weight" in sd:
        lm_head = g("lm_head.weight")
    else:
        # Tied embeddings: reuse the same storage (HF default for Qwen3-4B).
        lm_head = embed

    handles = WeightHandles()
    handles.ptrs = {
        "embed": embed,
        "lm_head": lm_head,
        "final_norm": g("model.norm.weight"),
        "layers": layers,
        "hidden": cfg["hidden"],
        "num_layers": nl,
        "vocab_size": cfg["vocab_size"],
        "quant_format": "bf16",
    }
    # Keep every tensor reachable for the GC.
    handles.anchors = [embed, lm_head, handles.ptrs["final_norm"]]
    for L in layers:
        handles.anchors.extend(L.values())
    return handles, cfg


def inject_hf_state_dict(
    frontend: Any,
    state_dict: Mapping[str, torch.Tensor],
    *,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Replace ``frontend`` weights from an HF ``state_dict`` (random-init OK)."""
    device = getattr(frontend, "device", "cuda:0")
    handles, cfg = extract_weights_qwen3_4b_bf16(
        device=device, state_dict=state_dict, config=config, assert_dims=True)
    frontend._weights = handles
    frontend._cfg = cfg
    # Rebuild packed views used by the decode/prefill engine.
    if hasattr(frontend, "_rebuild_packed_weights"):
        frontend._rebuild_packed_weights()
