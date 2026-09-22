"""Static geometry of Spark-X2.5-4B.

Everything here is fixed at load time from the checkpoint's config.json. The
runtime has no dynamic shapes, so these constants become buffer sizes and kernel
launch parameters that are identical on every decode step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SparkX25Config:
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 10240
    vocab_size: int = 131072
    max_position_embeddings: int = 1048576
    sliding_window: int = 512
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    layer_types: tuple = ()

    # RoPE is per layer type: full-attention layers rotate only the first
    # quarter of each head, sliding layers rotate all of it.
    rope_theta_full: float = 5000000.0
    rope_partial_full: float = 0.25
    rope_theta_sliding: float = 10000.0
    rope_partial_sliding: float = 1.0

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def rope_dim(self, layer_type: str) -> int:
        f = (self.rope_partial_full if layer_type == "full_attention"
             else self.rope_partial_sliding)
        return int(self.head_dim * f)

    def rope_theta(self, layer_type: str) -> float:
        return (self.rope_theta_full if layer_type == "full_attention"
                else self.rope_theta_sliding)

    @property
    def full_layer_idx(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def sliding_layer_idx(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]


def load_config(ckpt_dir: str) -> SparkX25Config:
    with open(f"{ckpt_dir}/config.json") as f:
        raw = json.load(f)
    rp = raw["rope_parameters"]
    return SparkX25Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        sliding_window=raw["sliding_window"],
        rms_norm_eps=raw["rms_norm_eps"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        layer_types=tuple(raw["layer_types"]),
        rope_theta_full=rp["full_attention"]["rope_theta"],
        rope_partial_full=rp["full_attention"]["partial_rotary_factor"],
        rope_theta_sliding=rp["sliding_attention"]["rope_theta"],
        rope_partial_sliding=rp["sliding_attention"]["partial_rotary_factor"],
    )
