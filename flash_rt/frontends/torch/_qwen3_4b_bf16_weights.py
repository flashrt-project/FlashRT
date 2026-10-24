"""FlashRT -- Qwen3-4B BF16 weight loader (HF CausalLM layout).

Loads HuggingFace-style Qwen3 dense weights::

    model.embed_tokens.weight
    model.layers.<i>.{input_layernorm,post_attention_layernorm}.weight
    model.layers.<i>.self_attn.{q,k,v,o}_proj.weight
    model.layers.<i>.self_attn.{q,k}_norm.weight
    model.layers.<i>.mlp.{gate,up,down}_proj.weight
    model.norm.weight
    lm_head.weight   (absent when tied to embed_tokens)

from a checkpoint directory (safetensors, read straight onto the device) or
an in-memory ``state_dict`` (random-weight parity tests). Every tensor ends
up BF16, contiguous and on one device; projections keep HF's separate
[out, in] matrices because the prefill runs them as separate GEMMs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from flash_rt.models.qwen3.prefill_bf16 import Qwen3_4BDims

BF16 = torch.bfloat16
_DIMS = Qwen3_4BDims()

#: Per-layer tensor names -> HF key suffix under ``model.layers.<i>``.
LAYER_KEYS = {
    "in_norm": "input_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "o": "self_attn.o_proj.weight",
    "post_norm": "post_attention_layernorm.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


@dataclass
class Qwen3_4BWeights:
    """BF16 device tensors for the prefill; ``lm_head`` may alias ``embed``."""

    layers: list[dict[str, torch.Tensor]] = field(default_factory=list)
    embed: torch.Tensor | None = None
    final_norm: torch.Tensor | None = None
    lm_head: torch.Tensor | None = None


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def parse_qwen3_config(cfg: Any) -> dict[str, Any]:
    """Normalize a HF Qwen3 config (config.json dict or ``Qwen3Config``).

    Raises on features this BF16 path does not implement rather than
    silently computing something else: RoPE scaling, attention bias,
    sliding-window attention, an activation other than SiLU.
    """
    rope = _get(cfg, "rope_parameters") or {}
    rope_type = rope.get("rope_type", "default") if isinstance(rope, Mapping) else "default"
    scaling = _get(cfg, "rope_scaling")
    if isinstance(scaling, Mapping):
        rope_type = scaling.get("rope_type", scaling.get("type", rope_type))
    if rope_type not in (None, "default"):
        raise ValueError(f"Qwen3-4B BF16 prefill: rope type {rope_type!r} is not supported")
    if _get(cfg, "attention_bias", False):
        raise ValueError("Qwen3-4B BF16 prefill: attention_bias=True is not supported")
    if _get(cfg, "use_sliding_window", False):
        raise ValueError("Qwen3-4B BF16 prefill: sliding-window attention is not supported")
    act = _get(cfg, "hidden_act", "silu")
    if act != "silu":
        raise ValueError(f"Qwen3-4B BF16 prefill: hidden_act={act!r} is not supported")
    theta = rope.get("rope_theta") if isinstance(rope, Mapping) else None
    if theta is None:
        theta = _get(cfg, "rope_theta", _DIMS.rope_theta)
    hidden = int(_get(cfg, "hidden_size"))
    heads = int(_get(cfg, "num_attention_heads"))
    head_dim = _get(cfg, "head_dim") or hidden // heads
    return {
        "hidden": hidden,
        "num_layers": int(_get(cfg, "num_hidden_layers")),
        "num_q_heads": heads,
        "num_kv_heads": int(_get(cfg, "num_key_value_heads")),
        "head_dim": int(head_dim),
        "intermediate": int(_get(cfg, "intermediate_size")),
        "vocab_size": int(_get(cfg, "vocab_size")),
        "rms_norm_eps": float(_get(cfg, "rms_norm_eps", _DIMS.rms_norm_eps)),
        "rope_theta": float(theta),
        "max_pos": int(_get(cfg, "max_position_embeddings", _DIMS.max_pos)),
        # Qwen3Config's default; Qwen/Qwen3-4B's config.json sets it true.
        "tie_word_embeddings": bool(_get(cfg, "tie_word_embeddings", False)),
    }


def qwen3_4b_dims(cfg: Mapping[str, Any]) -> Qwen3_4BDims:
    """Check a parsed config against Qwen/Qwen3-4B and return its dims."""
    want = _DIMS
    for name in ("hidden", "num_layers", "num_q_heads", "num_kv_heads",
                 "head_dim", "intermediate", "vocab_size"):
        if int(cfg[name]) != int(getattr(want, name)):
            raise ValueError(
                f"not a Qwen3-4B config: {name}={cfg[name]}, expected "
                f"{getattr(want, name)}")
    return Qwen3_4BDims(rope_theta=cfg["rope_theta"], max_pos=cfg["max_pos"],
                        rms_norm_eps=cfg["rms_norm_eps"])


def qwen3_4b_config_dict() -> dict[str, Any]:
    """Qwen/Qwen3-4B config values, for random-weight tests (no download)."""
    d = _DIMS
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": d.vocab_size,
        "hidden_size": d.hidden,
        "intermediate_size": d.intermediate,
        "num_hidden_layers": d.num_layers,
        "num_attention_heads": d.num_q_heads,
        "num_key_value_heads": d.num_kv_heads,
        "head_dim": d.head_dim,
        "max_position_embeddings": d.max_pos,
        "rms_norm_eps": d.rms_norm_eps,
        "rope_theta": d.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_cache": True,
    }


def read_config(checkpoint_path: str) -> dict[str, Any]:
    path = os.path.join(checkpoint_path, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing config.json in {checkpoint_path!r}")
    with open(path) as f:
        return json.load(f)


def _checkpoint_tensors(checkpoint_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = os.path.join(checkpoint_path, "model.safetensors.index.json")
    single = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    elif os.path.isfile(single):
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(
            f"no safetensors weights under {checkpoint_path!r} (expected "
            "model.safetensors or model.safetensors.index.json)")
    out: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(os.path.join(checkpoint_path, shard), framework="pt",
                       device=str(device)) as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
    return out


def load_qwen3_4b_weights(
    *,
    device: torch.device,
    checkpoint_path: str | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    config: Any = None,
) -> tuple[Qwen3_4BWeights, dict[str, Any]]:
    """Weights and parsed config from a checkpoint dir or a ``state_dict``.

    ``config`` defaults to the checkpoint's config.json. A ``state_dict``
    tensor already BF16 and on ``device`` is used in place (no copy).
    """
    if config is None:
        if checkpoint_path is None:
            raise ValueError("config is required when no checkpoint_path is given")
        config = read_config(checkpoint_path)
    cfg = parse_qwen3_config(config)
    qwen3_4b_dims(cfg)
    if state_dict is None:
        if checkpoint_path is None:
            raise ValueError("need checkpoint_path or state_dict")
        state_dict = _checkpoint_tensors(checkpoint_path, device)

    def g(key: str) -> torch.Tensor:
        if key not in state_dict:
            raise KeyError(f"missing weight {key!r}")
        return state_dict[key].detach().to(device=device, dtype=BF16).contiguous()

    w = Qwen3_4BWeights()
    for i in range(cfg["num_layers"]):
        prefix = f"model.layers.{i}."
        w.layers.append({name: g(prefix + key) for name, key in LAYER_KEYS.items()})
    w.embed = g("model.embed_tokens.weight")
    w.final_norm = g("model.norm.weight")
    # HF ties lm_head to the embedding whenever the config says so, even if a
    # checkpoint also carries an lm_head tensor.
    w.lm_head = w.embed if cfg["tie_word_embeddings"] else g("lm_head.weight")
    return w, cfg
