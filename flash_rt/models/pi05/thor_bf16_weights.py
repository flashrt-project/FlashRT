"""Checkpoint layout and OpenPI precision boundaries for the Thor BF16 pipeline."""

import torch
from safetensors import safe_open

VISION = {"num_hidden_layers": 27, "num_attention_heads": 16, "layer_norm_eps": 1e-6}
GEMMA = {
    "num_hidden_layers": 18,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
}
VISION_ROOT = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
LANGUAGE_ROOT = "paligemma_with_expert.paligemma.model.language_model"
EXPERT_ROOT = "paligemma_with_expert.gemma_expert.model"
PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"


def _weight_specs():
    """Yield required tensor names, shapes and execution dtypes."""
    bf16, fp32 = torch.bfloat16, torch.float32
    yield f"{VISION_ROOT}.embeddings.patch_embedding.weight", (1152, 3, 14, 14), fp32
    yield f"{VISION_ROOT}.embeddings.patch_embedding.bias", (1152,), fp32
    yield f"{VISION_ROOT}.embeddings.position_embedding.weight", (256, 1152), fp32
    yield f"{VISION_ROOT}.post_layernorm.weight", (1152,), bf16
    yield f"{VISION_ROOT}.post_layernorm.bias", (1152,), bf16
    yield f"{PROJECTOR}.weight", (2048, 1152), bf16
    yield f"{PROJECTOR}.bias", (2048,), bf16
    for index in range(VISION["num_hidden_layers"]):
        layer = f"{VISION_ROOT}.encoder.layers.{index}"
        for norm in ("layer_norm1", "layer_norm2"):
            yield f"{layer}.{norm}.weight", (1152,), bf16
            yield f"{layer}.{norm}.bias", (1152,), bf16
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            yield f"{layer}.self_attn.{name}.weight", (1152, 1152), bf16
            yield f"{layer}.self_attn.{name}.bias", (1152,), bf16
        for name, out_dim, in_dim in (("fc1", 4304, 1152), ("fc2", 1152, 4304)):
            yield f"{layer}.mlp.{name}.weight", (out_dim, in_dim), bf16
            yield f"{layer}.mlp.{name}.bias", (out_dim,), bf16

    yield f"{LANGUAGE_ROOT}.embed_tokens.weight", (257152, 2048), bf16
    for root, width, intermediate in ((LANGUAGE_ROOT, 2048, 16384), (EXPERT_ROOT, 1024, 4096)):
        for index in range(GEMMA["num_hidden_layers"]):
            layer = f"{root}.layers.{index}"
            for name, shape in (
                ("q_proj", (2048, width)),
                ("k_proj", (256, width)),
                ("v_proj", (256, width)),
                ("o_proj", (width, 2048)),
            ):
                yield f"{layer}.self_attn.{name}.weight", shape, bf16
            for name, shape in (
                ("gate_proj", (intermediate, width)),
                ("up_proj", (intermediate, width)),
                ("down_proj", (width, intermediate)),
            ):
                yield f"{layer}.mlp.{name}.weight", shape, bf16
            for name in ("input_layernorm", "post_attention_layernorm"):
                yield from _gemma_norm_specs(f"{layer}.{name}", width, root == EXPERT_ROOT)
        yield from _gemma_norm_specs(f"{root}.norm", width, root == EXPERT_ROOT)

    for name, out_dim, in_dim in (
        ("action_in_proj", 1024, 32),
        ("action_out_proj", 32, 1024),
        ("time_mlp_in", 1024, 1024),
        ("time_mlp_out", 1024, 1024),
    ):
        yield f"{name}.weight", (out_dim, in_dim), fp32
        yield f"{name}.bias", (out_dim,), fp32


def _gemma_norm_specs(name, width, adaptive):
    if adaptive:
        yield f"{name}.dense.weight", (3 * width, width), torch.float32
        yield f"{name}.dense.bias", (3 * width,), torch.float32
    else:
        yield f"{name}.weight", (width,), torch.float32


def load_thor_bf16_weights(checkpoint):
    weights = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        keys, aliases = set(file.keys()), file.metadata() or {}
        specs = []
        for name, shape, dtype in _weight_specs():
            stored_name = name if name in keys else aliases.get(name)
            if stored_name not in keys or tuple(file.get_slice(stored_name).get_shape()) != shape:
                raise ValueError(f"Incompatible Pi0.5 checkpoint tensor {name}; expected {shape}")
            specs.append((name, stored_name, dtype))
        for name, stored_name, dtype in specs:
            tensor = file.get_tensor(stored_name)
            if name.startswith("paligemma_with_expert."):
                # OpenPI first casts Gemma/SigLIP to BF16, then restores selected
                # weights to FP32. Direct FP32 loading would skip that rounding.
                tensor = tensor.to(device="cuda", dtype=torch.bfloat16)
            weights[name] = tensor.to(device="cuda", dtype=dtype).contiguous()
    return weights
