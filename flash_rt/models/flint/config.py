"""Static geometry of Flint (``FlintForCausalLM``), read from GGUF metadata.

Flint is a 49-layer MoE decoder: two leading dense SwiGLU layers, then 47 MoE
layers (256 routed experts, top-6 sigmoid routing with renormalization, plus
one ungated shared expert). Attention is GQA 32/4 with head_dim 128, per-head
Q/K RMSNorm and a learned per-query-head sink logit in every layer. Layers
alternate between a 513-token sliding window with NeoX RoPE and global causal
attention without positional encoding.

Everything here is fixed at load time; the runtime has no dynamic shapes, so
these values become buffer sizes and kernel launch parameters. Unsupported
variants are rejected here rather than silently mis-executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from flash_rt.models.flint.gguf import GGUFFile

ARCH = "flint"

# Expert gating function ids used by GGUF ``<arch>.expert_gating_func``.
GATING_SOFTMAX = 1
GATING_SIGMOID = 2


class FlintConfigError(ValueError):
    """The checkpoint describes a Flint variant this pipeline does not run."""


@dataclass(frozen=True)
class FlintConfig:
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    dense_intermediate_size: int
    num_dense_layers: int
    num_experts: int
    num_experts_per_tok: int
    expert_intermediate_size: int
    shared_intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int  # number of keys a sliding query sees, current token included
    sliding_layers: Tuple[bool, ...]  # True: sliding window + RoPE; False: global, NoPE
    context_length: int
    bos_token_id: int
    eos_token_id: int
    eog_token_ids: Tuple[int, ...]

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim

    @property
    def gqa_group(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def full_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.sliding_layers) if not s)

    @property
    def sliding_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.sliding_layers) if s)

    def is_moe(self, layer: int) -> bool:
        return layer >= self.num_dense_layers


def _key(arch: str, name: str) -> str:
    return f"{arch}.{name}"


def load_config(gguf: GGUFFile) -> FlintConfig:
    arch = gguf.require("general.architecture")
    if arch != ARCH:
        raise FlintConfigError(f"general.architecture is {arch!r}, expected {ARCH!r}")
    g = lambda name: gguf.require(_key(arch, name))  # noqa: E731

    n_layer = int(g("block_count"))
    swa = np.asarray(g("attention.sliding_window_pattern"), dtype=bool)
    rope = np.asarray(g("attention.rope_pattern"), dtype=bool)
    if swa.shape != (n_layer,) or rope.shape != (n_layer,):
        raise FlintConfigError("layer pattern arrays do not match block_count")
    if not np.array_equal(swa, rope):
        raise FlintConfigError(
            "rope_pattern differs from sliding_window_pattern; this pipeline "
            "implements RoPE exactly on the sliding-window layers")

    head_dim = int(g("attention.key_length"))
    if int(g("attention.value_length")) != head_dim:
        raise FlintConfigError("key_length != value_length")
    if int(gguf.get(_key(arch, "rope.dimension_count"), head_dim)) != head_dim:
        raise FlintConfigError("partial rotary is not supported")
    for k in ("rope.scaling.type", "rope.scaling.factor", "attn_logit_softcapping",
              "final_logit_softcapping"):
        if _key(arch, k) in gguf.metadata:
            raise FlintConfigError(f"unsupported metadata {_key(arch, k)!r}")
    if int(g("expert_gating_func")) != GATING_SIGMOID:
        raise FlintConfigError("only sigmoid expert gating is supported")
    if not bool(g("expert_weights_norm")):
        raise FlintConfigError("top-k weight renormalization is required")
    if float(g("expert_weights_scale")) != 1.0:
        raise FlintConfigError("routed scaling factor != 1 is not supported")

    eog = gguf.get("tokenizer.ggml.eog_token_ids")
    eos = int(gguf.require("tokenizer.ggml.eos_token_id"))
    cfg = FlintConfig(
        num_layers=n_layer,
        hidden_size=int(g("embedding_length")),
        num_heads=int(g("attention.head_count")),
        num_kv_heads=int(g("attention.head_count_kv")),
        head_dim=head_dim,
        vocab_size=int(len(gguf.require("tokenizer.ggml.token_type"))),
        dense_intermediate_size=int(g("feed_forward_length")),
        num_dense_layers=int(g("leading_dense_block_count")),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        expert_intermediate_size=int(g("expert_feed_forward_length")),
        shared_intermediate_size=int(g("expert_shared_feed_forward_length")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        rope_theta=float(g("rope.freq_base")),
        sliding_window=int(g("attention.sliding_window")),
        sliding_layers=tuple(bool(x) for x in swa),
        context_length=int(g("context_length")),
        bos_token_id=int(gguf.require("tokenizer.ggml.bos_token_id")),
        eos_token_id=eos,
        eog_token_ids=tuple(int(x) for x in eog) if eog is not None else (eos,),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: FlintConfig) -> None:
    """Shape assumptions the Flint kernels are specialized for."""
    expect = {
        "num_layers": 49, "num_dense_layers": 2, "dense_intermediate_size": 6144,
        "sliding_window": 513,
        "hidden_size": 2048, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128,
        "num_experts": 256, "num_experts_per_tok": 6, "expert_intermediate_size": 512,
        "shared_intermediate_size": 1024,
    }
    for k, v in expect.items():
        if getattr(cfg, k) != v:
            raise FlintConfigError(
                f"{k}={getattr(cfg, k)}; the Flint kernels are specialized for {v}")
    if any(cfg.sliding_layers[:cfg.num_dense_layers]):
        raise FlintConfigError("dense leading layers are expected to use global attention")
    if cfg.sliding_window < 2:
        raise FlintConfigError(f"sliding_window={cfg.sliding_window} is invalid")
