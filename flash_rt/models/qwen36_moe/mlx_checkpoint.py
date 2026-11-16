"""Reader for qwen3_5_moe checkpoints in MLX affine-quantized safetensors form
(e.g. ``mlx-community/Qwen3.6-35B-A3B-4bit``), text decoder only.

The export stores every projection, the routed experts, the token embedding and the LM head as affine 4-bit
codes with one BF16 scale and one BF16 bias per group of 64 inputs (``w = scale * q + bias``), the router
(``mlp.gate``) and the shared-expert gate as affine 8-bit codes with the same group size, and the norms, the
depthwise conv and the DeltaNet gate parameters in BF16. Codes are stored as little-endian uint32 words along
the input dimension: for 4-bit, element ``2i`` is the low nibble of byte ``i``. This module only maps and
validates the files (numpy, no framework); the hardware runtimes copy the bytes verbatim into their weight
buffers, so the numerics stay those of the export. Vision-tower tensors are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from flash_rt.models.flint.mlx_checkpoint import GROUP_SIZE, AffineMatrix, SafetensorsFile

PREFIX = "language_model."


class Qwen35MoeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Qwen35MoeConfig:
    """Text-decoder geometry of a qwen3_5_moe checkpoint."""
    num_layers: int
    full_attention_interval: int
    hidden_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dims: int
    rope_theta: float
    linear_key_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    conv_kernel: int
    num_experts: int
    num_experts_per_tok: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    rms_norm_eps: float
    context_length: int
    eos_token_ids: Tuple[int, ...]
    temperature: float
    top_k: int
    top_p: float

    def is_full_attention(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_interval == 0

    @property
    def conv_dim(self) -> int:
        return 2 * self.linear_key_heads * self.linear_key_dim + self.linear_value_heads * self.linear_value_dim

    @property
    def value_dim(self) -> int:
        return self.linear_value_heads * self.linear_value_dim

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def full_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.num_layers) if self.is_full_attention(i))

    @property
    def linear_layers(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.num_layers) if not self.is_full_attention(i))


def config_from_json(raw: dict, generation: Optional[dict] = None) -> Qwen35MoeConfig:
    if raw.get("model_type") != "qwen3_5_moe":
        raise Qwen35MoeConfigError(f"model_type {raw.get('model_type')!r}, expected 'qwen3_5_moe'")
    t = raw.get("text_config") or {}
    try:
        layer_types = t["layer_types"]
        interval = int(t.get("full_attention_interval", 4))
        expected = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(int(t["num_hidden_layers"]))]
        if list(layer_types) != expected:
            raise Qwen35MoeConfigError("layer_types is not (interval - 1) linear layers followed by one full layer")
        rope = t.get("rope_parameters") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise Qwen35MoeConfigError(f"unsupported rope type {rope}")
        if t.get("tie_word_embeddings") or not t.get("attn_output_gate", False) or t.get("attention_bias"):
            raise Qwen35MoeConfigError("expected untied embeddings, an attention output gate and no attention bias")
        if t.get("hidden_act", "silu") != "silu" or not t.get("norm_topk_prob", True):
            raise Qwen35MoeConfigError("expected SiLU activations and renormalized top-k router weights")
        gen = generation or {}
        eos = gen.get("eos_token_id", raw.get("eos_token_id", t.get("eos_token_id")))
        eos = tuple(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
        head_dim = int(t["head_dim"])
        return Qwen35MoeConfig(
            num_layers=int(t["num_hidden_layers"]), full_attention_interval=interval,
            hidden_size=int(t["hidden_size"]), vocab_size=int(t["vocab_size"]),
            num_heads=int(t["num_attention_heads"]), num_kv_heads=int(t["num_key_value_heads"]), head_dim=head_dim,
            rotary_dims=int(head_dim * float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))),
            rope_theta=float(rope.get("rope_theta", 100000.0)),
            linear_key_heads=int(t["linear_num_key_heads"]), linear_value_heads=int(t["linear_num_value_heads"]),
            linear_key_dim=int(t["linear_key_head_dim"]), linear_value_dim=int(t["linear_value_head_dim"]),
            conv_kernel=int(t["linear_conv_kernel_dim"]),
            num_experts=int(t["num_experts"]), num_experts_per_tok=int(t["num_experts_per_tok"]),
            expert_intermediate_size=int(t["moe_intermediate_size"]),
            shared_intermediate_size=int(t["shared_expert_intermediate_size"]),
            rms_norm_eps=float(t["rms_norm_eps"]), context_length=int(t["max_position_embeddings"]),
            eos_token_ids=eos, temperature=float(gen.get("temperature", 1.0)), top_k=int(gen.get("top_k", 20)),
            top_p=float(gen.get("top_p", 1.0)))
    except KeyError as e:
        raise Qwen35MoeConfigError(f"config.json text_config lacks {e}") from None


def check_quantization(raw: dict, cfg: Qwen35MoeConfig) -> None:
    """4-bit group-64 affine everywhere, 8-bit for every layer's router and shared-expert gate."""
    q = raw.get("quantization") or {}
    base = {k: v for k, v in q.items() if not isinstance(v, dict)}
    if base != {"group_size": GROUP_SIZE, "bits": 4, "mode": "affine"}:
        raise Qwen35MoeConfigError(f"expected group-64 4-bit affine quantization, found {base}")
    special = {k: v for k, v in q.items() if isinstance(v, dict)}
    want = {f"{PREFIX}model.layers.{i}.mlp.{name}": {"group_size": GROUP_SIZE, "bits": 8}
            for i in range(cfg.num_layers) for name in ("gate", "shared_expert_gate")}
    if special != want:
        raise Qwen35MoeConfigError("expected 8-bit router and shared-expert gate overrides for every layer")


class Qwen35MoeMLXCheckpoint:
    """A directory with ``config.json``, ``generation_config.json``, tokenizer files and ``model-*.safetensors``
    shards of an MLX affine export of a qwen3_5_moe model."""

    def __init__(self, directory):
        self.directory = Path(directory)
        config_path = self.directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"{config_path} not found")
        self.raw_config = json.loads(config_path.read_text())
        gen_path = self.directory / "generation_config.json"
        self.generation = json.loads(gen_path.read_text()) if gen_path.is_file() else None
        self.config = config_from_json(self.raw_config, self.generation)
        check_quantization(self.raw_config, self.config)
        shards = sorted(self.directory.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no model-*.safetensors in {self.directory}")
        self.files = [SafetensorsFile(p) for p in shards]
        self._index: Dict[str, SafetensorsFile] = {}
        for f in self.files:
            for name in f.entries:
                if name in self._index:
                    raise ValueError(f"tensor {name} appears in more than one shard")
                self._index[name] = f

    def tensor(self, name: str) -> np.ndarray:
        """Text-decoder tensor ``name`` (without the ``language_model.`` prefix); BF16 as raw uint16."""
        try:
            return self._index[PREFIX + name].tensor(PREFIX + name)
        except KeyError:
            raise KeyError(f"checkpoint has no tensor {PREFIX}{name}") from None

    def affine(self, prefix: str, bits: int = 4, k: Optional[int] = None) -> AffineMatrix:
        words = self.tensor(prefix + ".weight")
        scales = self.tensor(prefix + ".scales")
        biases = self.tensor(prefix + ".biases")
        if words.dtype != np.uint32 or scales.dtype != np.uint16 or biases.dtype != np.uint16:
            raise ValueError(f"{prefix}: expected uint32 code words and BF16 scales/biases")
        codes = words.view(np.uint8)
        if scales.shape != biases.shape or codes.shape[:-1] != scales.shape[:-1]:
            raise ValueError(f"{prefix}: codes/scales/biases shapes disagree")
        kk = scales.shape[-1] * GROUP_SIZE
        if codes.shape[-1] * 8 != kk * bits or (k is not None and kk != k):
            raise ValueError(f"{prefix}: {codes.shape} code bytes do not hold {bits}-bit x {kk} inputs")
        return AffineMatrix(codes, scales, biases, bits)

    def close(self) -> None:
        for f in self.files:
            f.close()
