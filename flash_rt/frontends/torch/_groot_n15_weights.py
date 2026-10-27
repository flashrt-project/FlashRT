"""FlashRT -- GR00T N1.5 checkpoint adapter (torch).

Reads a ``GR00T_N1_5`` checkpoint directory (``config.json`` + sharded
safetensors) into device tensors laid out for
:class:`flash_rt.models.groot_n15.GrootN15Forward`.

The reference loads the policy with ``torch_dtype=bfloat16``, so every
parameter -- including the fp32-stored action head -- is held in bf16; this
adapter casts the same way. The embodiment-specific MLP weights are sliced to
the one embodiment the frontend serves. Configurations the forward does not
implement are refused here, with the field that differs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from flash_rt.models.groot_n15.forward import GrootN15Dims

BF16 = torch.bfloat16

_VISION = "backbone.eagle_model.vision_model.vision_model."
_LLM = "backbone.eagle_model.language_model.model."
_HEAD = "action_head."


def _require(cond: bool, what: str) -> None:
    if not cond:
        raise ValueError(f"unsupported GR00T N1.5 checkpoint: {what}")


def parse_dims(config: Mapping[str, Any], eagle_config: Mapping[str, Any],
               action_horizon: int) -> GrootN15Dims:
    """Dims of a checkpoint; raises on anything the forward does not implement.

    ``eagle_config`` is the Eagle2.5 backbone config the upstream package
    builds the backbone from; ``action_horizon`` is the data config's.
    """
    _require(config.get("model_type") == "gr00t_n1_5",
             f"model_type {config.get('model_type')!r} (expected 'gr00t_n1_5')")
    bb = config["backbone_cfg"]
    _require(not bb.get("reproject_vision", False), "reproject_vision")
    _require(bb.get("project_to_dim") is None,
             f"project_to_dim={bb.get('project_to_dim')} (only the identity eagle_linear)")
    head = config["action_head_cfg"]
    dit = head["diffusion_model_cfg"]
    vl = head["vl_self_attention_cfg"]
    _require(head.get("use_vlln", True), "use_vlln=False")
    _require(head.get("add_pos_embed", True), "add_pos_embed=False")
    _require(dit.get("interleave_self_attention", False),
             "interleave_self_attention=False")
    _require(dit.get("norm_type", "ada_norm") == "ada_norm", f"DiT norm_type {dit.get('norm_type')}")
    _require(dit.get("positional_embeddings") is None, "DiT positional_embeddings")
    _require(vl.get("positional_embeddings") is None, "VL positional_embeddings")
    _require(dit.get("cross_attention_dim") == head["backbone_embedding_dim"],
             "DiT cross_attention_dim != backbone_embedding_dim")
    _require(dit.get("activation_fn", "gelu-approximate") == "gelu-approximate",
             "DiT activation_fn")
    _require(not dit.get("upcast_attention", False), "DiT upcast_attention")
    _require(not vl.get("upcast_attention", False), "VL upcast_attention")
    _require(int(head["action_horizon"]) == int(action_horizon),
             f"the data config's action horizon {action_horizon} differs from the "
             f"checkpoint's {head['action_horizon']}")
    _require(vl.get("activation_fn", "gelu-approximate") == "gelu-approximate",
             "VL activation_fn")
    _require(int(dit["num_attention_heads"]) * int(dit["attention_head_dim"])
             == int(head["input_embedding_dim"]), "DiT width != input_embedding_dim")
    _require(int(vl["num_attention_heads"]) * int(vl["attention_head_dim"])
             == int(head["backbone_embedding_dim"]), "VL width != backbone_embedding_dim")

    _require(eagle_config.get("use_pixel_shuffle") is False, "Eagle use_pixel_shuffle")
    _require(int(eagle_config.get("mlp_connector_layers", 0)) == 1,
             f"Eagle mlp_connector_layers={eagle_config.get('mlp_connector_layers')}")
    _require(int(eagle_config.get("select_layer", 0)) == -1, "Eagle vision select_layer")
    vc, tc = eagle_config["vision_config"], eagle_config["text_config"]
    _require(vc.get("model_type") == "siglip_vision_model",
             f"vision model {vc.get('model_type')}")
    _require(vc.get("hidden_act") == "gelu_pytorch_tanh", f"vision act {vc.get('hidden_act')}")
    _require(tc.get("architectures") == ["Qwen3ForCausalLM"],
             f"language model {tc.get('architectures')}")
    _require(not tc.get("attention_bias", False), "Qwen3 attention_bias")
    _require(tc.get("rope_scaling") is None, "Qwen3 rope_scaling")
    _require(tc.get("hidden_act") == "silu", "Qwen3 hidden_act")
    _require(not tc.get("use_sliding_window", False), "Qwen3 use_sliding_window")
    head_dim = int(tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"])
    _require(head_dim == 128, f"Qwen3 head_dim={head_dim} (the q/k norm + RoPE kernel takes 128)")
    for name, width in (("vision hidden_size", vc["hidden_size"]),
                        ("VL width", head["backbone_embedding_dim"]),
                        ("DiT width", head["input_embedding_dim"])):
        _require(int(width) <= 2048, f"{name}={width} (the LayerNorm kernels take <= 2048)")
    image_size = int(eagle_config.get("force_image_size") or vc["image_size"])
    llm_layers = int(bb["select_layer"])
    _require(0 < llm_layers <= int(tc["num_hidden_layers"]),
             f"select_layer={llm_layers}")
    _require(int(head["backbone_embedding_dim"]) == int(tc["hidden_size"]),
             "backbone_embedding_dim != language hidden size")
    return GrootN15Dims(
        image_size=image_size,
        patch_size=int(vc["patch_size"]),
        vision_hidden=int(vc["hidden_size"]),
        vision_layers=int(vc["num_hidden_layers"]),
        vision_heads=int(vc["num_attention_heads"]),
        vision_intermediate=int(vc["intermediate_size"]),
        vision_eps=float(vc.get("layer_norm_eps", 1e-6)),
        llm_hidden=int(tc["hidden_size"]),
        llm_layers=llm_layers,
        llm_intermediate=int(tc["intermediate_size"]),
        llm_q_heads=int(tc["num_attention_heads"]),
        llm_kv_heads=int(tc["num_key_value_heads"]),
        llm_head_dim=head_dim,
        llm_vocab=int(tc["vocab_size"]),
        rope_theta=float(tc["rope_theta"]),
        rms_norm_eps=float(tc["rms_norm_eps"]),
        vl_layers=int(vl["num_layers"]),
        vl_heads=int(vl["num_attention_heads"]),
        vl_head_dim=int(vl["attention_head_dim"]),
        dit_layers=int(dit["num_layers"]),
        dit_heads=int(dit["num_attention_heads"]),
        dit_head_dim=int(dit["attention_head_dim"]),
        dit_output=int(dit["output_dim"]),
        dit_eps=float(dit.get("norm_eps", 1e-5)),
        state_dim=int(head["max_state_dim"]),
        action_dim=int(head["action_dim"]),
        head_hidden=int(head["hidden_size"]),
        future_tokens=int(head["num_target_vision_tokens"]),
        action_horizon=int(action_horizon),
        denoising_steps=int(head["num_inference_timesteps"]),
        timestep_buckets=int(head["num_timestep_buckets"]),
    )


def fp8_per_channel(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 copy of a weight ``[n, k]`` with one fp32 scale per output channel:
    ``W[i] ~= W8[i] * scale[i]``, each row's largest magnitude mapped to 448.

    Returns the bytes as uint8 (the GEMM's operand) and the scales ``[n]``.
    """
    amax = W.float().abs().amax(dim=1).clamp_min(1e-12)
    scale = amax / 448.0
    W8 = (W.float() / scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return W8, scale.contiguous()


def read_config(checkpoint_dir: str) -> dict:
    path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no config.json in GR00T N1.5 checkpoint {checkpoint_dir}")
    with open(path) as f:
        return json.load(f)


def _load_state(checkpoint_dir: str, device: torch.device) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    state: dict[str, torch.Tensor] = {}
    for name in files:
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing checkpoint shard {path}")
        with safe_open(path, framework="pt", device=str(device)) as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(BF16)
    return state


@dataclass
class GrootN15Weights:
    """Device tensors of one checkpoint, bf16, embodiment sliced.

    Projections are ``(W [n, k], b [n])`` pairs; norms ``(gamma, beta)``;
    embodiment MLP layers ``(W [k, n], b [n])`` as torch.bmm reads them.
    The language model is the per-layer key set of the shared Qwen3 sequence.
    """

    vision: dict[str, Any] = field(default_factory=dict)
    vision_layers: list[dict[str, Any]] = field(default_factory=list)
    llm_embed: torch.Tensor | None = None
    llm_layers: list[dict[str, torch.Tensor]] = field(default_factory=list)
    llm_final_norm: torch.Tensor | None = None
    vl: dict[str, Any] = field(default_factory=dict)
    vl_layers: list[dict[str, Any]] = field(default_factory=list)
    dit_layers: list[dict[str, Any]] = field(default_factory=list)
    head: dict[str, Any] = field(default_factory=dict)


def load_weights(checkpoint_dir: str, dims: GrootN15Dims, embodiment_id: int,
                 device: torch.device) -> GrootN15Weights:
    """Load and lay out a checkpoint for one embodiment."""
    st = _load_state(checkpoint_dir, device)
    used: set[str] = set()

    def get(key: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        if key not in st:
            raise KeyError(f"GR00T N1.5 checkpoint has no tensor {key!r}")
        t = st[key]
        if shape is not None and tuple(t.shape) != tuple(shape):
            raise ValueError(f"{key}: shape {tuple(t.shape)}, expected {tuple(shape)}")
        used.add(key)
        return t.contiguous()

    def lin(prefix: str, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        return get(prefix + ".weight", (n, k)), get(prefix + ".bias", (n,))

    def norm(prefix: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return get(prefix + ".weight", (n,)), get(prefix + ".bias", (n,))

    d = dims
    w = GrootN15Weights()
    # ── vision tower ──
    D, F = d.vision_hidden, d.vision_intermediate
    pw = get(_VISION + "embeddings.patch_embedding.weight", (D, 3, d.patch_size, d.patch_size))
    w.vision["patch"] = (pw.reshape(D, d.patch_features).contiguous(),
                         get(_VISION + "embeddings.patch_embedding.bias", (D,)))
    w.vision["position"] = get(_VISION + "embeddings.position_embedding.weight",
                               (d.patches_per_tile, D))
    for i in range(d.vision_layers):
        p = f"{_VISION}encoder.layers.{i}."
        w.vision_layers.append({
            "ln1": norm(p + "layer_norm1", D),
            "q": lin(p + "self_attn.q_proj", D, D),
            "k": lin(p + "self_attn.k_proj", D, D),
            "v": lin(p + "self_attn.v_proj", D, D),
            "o": lin(p + "self_attn.out_proj", D, D),
            "ln2": norm(p + "layer_norm2", D),
            "fc1": lin(p + "mlp.fc1", F, D),
            "fc2": lin(p + "mlp.fc2", D, F),
        })
    w.vision["post_ln"] = norm(_VISION + "post_layernorm", D)
    w.vision["mlp1"] = lin("backbone.eagle_model.mlp1.0", d.llm_hidden, D)
    # The vision head (attention pooling) feeds nothing the policy reads.
    used.update(k for k in st if k.startswith(_VISION + "head."))

    # ── language model ──
    H, I = d.llm_hidden, d.llm_intermediate
    QD, KD = d.llm_q_heads * d.llm_head_dim, d.llm_kv_heads * d.llm_head_dim
    w.llm_embed = get(_LLM + "embed_tokens.weight", (d.llm_vocab, H))
    for i in range(d.llm_layers):
        p = f"{_LLM}layers.{i}."
        w.llm_layers.append({
            "in_norm": get(p + "input_layernorm.weight", (H,)),
            "q": get(p + "self_attn.q_proj.weight", (QD, H)),
            "k": get(p + "self_attn.k_proj.weight", (KD, H)),
            "v": get(p + "self_attn.v_proj.weight", (KD, H)),
            "q_norm": get(p + "self_attn.q_norm.weight", (d.llm_head_dim,)),
            "k_norm": get(p + "self_attn.k_norm.weight", (d.llm_head_dim,)),
            "o": get(p + "self_attn.o_proj.weight", (H, QD)),
            "post_norm": get(p + "post_attention_layernorm.weight", (H,)),
            "gate": get(p + "mlp.gate_proj.weight", (I, H)),
            "up": get(p + "mlp.up_proj.weight", (I, H)),
            "down": get(p + "mlp.down_proj.weight", (H, I)),
        })
    w.llm_final_norm = get(_LLM + "norm.weight", (H,))
    # The language head's logits are computed by the reference and discarded.
    used.add("backbone.eagle_model.language_model.lm_head.weight")

    # ── vlln + VL self-attention ──
    V, VF = d.vl_hidden, d.vl_intermediate
    w.vl["vlln"] = norm(_HEAD + "vlln", V)
    for i in range(d.vl_layers):
        p = f"{_HEAD}vl_self_attention.transformer_blocks.{i}."
        w.vl_layers.append({
            "norm1": norm(p + "norm1", V),
            "q": lin(p + "attn1.to_q", V, V),
            "k": lin(p + "attn1.to_k", V, V),
            "v": lin(p + "attn1.to_v", V, V),
            "o": lin(p + "attn1.to_out.0", V, V),
            "norm3": norm(p + "norm3", V),
            "ff1": lin(p + "ff.net.0.proj", VF, V),
            "ff2": lin(p + "ff.net.2", V, VF),
        })

    # ── DiT ──
    De, DF = d.dit_hidden, d.dit_intermediate
    for i in range(d.dit_layers):
        p = f"{_HEAD}model.transformer_blocks.{i}."
        cross = i % 2 == 0
        kin = d.llm_hidden if cross else De
        w.dit_layers.append({
            "cross": cross,
            "ada": lin(p + "norm1.linear", 2 * De, De),
            "q": lin(p + "attn1.to_q", De, De),
            "k": lin(p + "attn1.to_k", De, kin),
            "v": lin(p + "attn1.to_v", De, kin),
            "o": lin(p + "attn1.to_out.0", De, De),
            "ff1": lin(p + "ff.net.0.proj", DF, De),
            "ff2": lin(p + "ff.net.2", De, DF),
        })
    m = _HEAD + "model."
    w.head["t_lin1"] = lin(m + "timestep_encoder.timestep_embedder.linear_1", De, 256)
    w.head["t_lin2"] = lin(m + "timestep_encoder.timestep_embedder.linear_2", De, De)
    w.head["proj_out1"] = lin(m + "proj_out_1", 2 * De, De)
    w.head["proj_out"] = lin(m + "proj_out_2", d.dit_output, De)

    # ── embodiment MLPs (one embodiment) ──
    def cat(prefix: str, k: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        W = get(prefix + ".W")
        b = get(prefix + ".b")
        if W.dim() != 3 or tuple(W.shape[1:]) != (k, n) or tuple(b.shape[1:]) != (n,):
            raise ValueError(f"{prefix}: shape {tuple(W.shape)}, expected [E, {k}, {n}]")
        if not 0 <= embodiment_id < W.shape[0]:
            raise ValueError(f"embodiment id {embodiment_id} outside the checkpoint's "
                             f"{W.shape[0]} embodiments")
        return W[embodiment_id].contiguous(), b[embodiment_id].contiguous()

    Hh = d.head_hidden
    w.head["state1"] = cat(_HEAD + "state_encoder.layer1", d.state_dim, Hh)
    w.head["state2"] = cat(_HEAD + "state_encoder.layer2", Hh, De)
    w.head["act1"] = cat(_HEAD + "action_encoder.W1", d.action_dim, De)
    w.head["act2"] = cat(_HEAD + "action_encoder.W2", 2 * De, De)
    w.head["act3"] = cat(_HEAD + "action_encoder.W3", De, De)
    w.head["dec1"] = cat(_HEAD + "action_decoder.layer1", d.dit_output, Hh)
    w.head["dec2"] = cat(_HEAD + "action_decoder.layer2", Hh, d.action_dim)
    w.head["future"] = get(_HEAD + "future_tokens.weight", (d.future_tokens, De))
    pos = get(_HEAD + "position_embedding.weight")
    if pos.dim() != 2 or pos.shape[1] != De or pos.shape[0] < d.action_horizon:
        raise ValueError(f"position_embedding shape {tuple(pos.shape)} cannot cover "
                         f"an action horizon of {d.action_horizon}")
    w.head["position"] = pos[:d.action_horizon].contiguous()

    unused = sorted(set(st) - used)
    if unused:
        raise ValueError(f"GR00T N1.5 checkpoint has tensors this forward does not read: "
                         f"{unused[:6]}{' ...' if len(unused) > 6 else ''}")
    return w
