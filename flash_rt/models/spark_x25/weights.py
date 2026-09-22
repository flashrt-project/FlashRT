"""Checkpoint loading and NVFP4 weight packing for Spark-X2.5.

The checkpoint ships BF16. The decode hot path wants block-scaled NVFP4
(E2M1 values, one UE4M3 scale per 16 elements along K) so the per-token weight
traffic drops from 8.2 GB to ~2.2 GB, which is the single largest lever on a
bandwidth-bound batch-1 decode.

Quantization uses FlashRT's own `bf16_weight_to_nvfp4_swizzled`, which is the
repository-native weight quantizer (the SM120 frontends use it for the lm_head).
It emits packed values plus *swizzled* scale factors and a per-tensor global
scale; the matching GEMM scalar is `alpha = global_scale`.

All packed weights and scale factors are allocated once, at load time, and never
move afterwards, so the kernels see stable raw pointers and the whole steady
state is capturable in a CUDA Graph.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

import flash_rt.flash_rt_kernels as fvk

from flash_rt.models.spark_x25.config import SparkX25Config


@dataclass
class Nvfp4Linear:
    """One NVFP4-quantized linear layer, ready for `fp4_w4a4_mma_sm120_full_n_bf16out`."""

    n: int            # output features (GEMM N)
    k: int            # input features (GEMM K)
    packed: torch.Tensor   # uint8 (n, k//2), row-major
    sf: torch.Tensor       # uint8, swizzled, nvfp4_sf_swizzled_bytes(n, k)
    alpha: float

    @property
    def packed_ptr(self) -> int:
        return self.packed.data_ptr()

    @property
    def sf_ptr(self) -> int:
        return self.sf.data_ptr()


@dataclass
class LayerWeights:
    qkv: Nvfp4Linear
    g_proj: Nvfp4Linear      # tiny 2560 -> 16; kept NVFP4 for uniformity
    o_proj: Nvfp4Linear
    gate_up: Nvfp4Linear     # gate_proj and up_proj fused along N (prefill)
    down: Nvfp4Linear
    input_norm: torch.Tensor
    post_attn_norm: torch.Tensor
    # Same weights with gate/up INTERLEAVED column-wise, for the decode kernel
    # that applies gelu(gate)*up in its epilogue: the gated multiply needs both
    # projections of one intermediate index coresident in a warp's N-tile.
    gate_up_il: Nvfp4Linear = None


@dataclass
class ModelWeights:
    cfg: SparkX25Config
    embed: torch.Tensor                       # bf16 (vocab, hidden), also the lm_head
    layers: list = field(default_factory=list)
    final_norm: Optional[torch.Tensor] = None
    # The attention output gate is 2560 -> 16 per layer, 0.04% of the model.
    # Quantizing it would cost a 4-bit block-scale round trip on a tensor whose
    # whole point is a coarse per-head scalar, so it stays bf16 and runs as a
    # matvec.
    g_proj_bf16: list = field(default_factory=list)
    # The lm_head is tied to the embedding, but the two uses have opposite
    # traffic profiles: the lookup reads a handful of rows, while the projection
    # streams the whole 131072x2560 matrix once per token. Keeping a second,
    # NVFP4 copy for the projection cuts that read from 671 MB to ~180 MB while
    # the bf16 table stays byte-exact for the lookup.
    lm_head: Optional[Nvfp4Linear] = None
    # BF16 copies kept only for parity debugging
    bf16: Dict[str, torch.Tensor] = field(default_factory=dict, repr=False)


def _load_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open
    idx = json.load(open(f"{ckpt_dir}/model.safetensors.index.json"))
    weight_map = idx["weight_map"]
    shards = sorted(set(weight_map.values()))
    out: Dict[str, torch.Tensor] = {}
    for sh in shards:
        with safe_open(os.path.join(ckpt_dir, sh), framework="pt", device="cpu") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


def _split_gate_up(sd, prefix):
    """Build the fused gate+up matrix, N-major: rows [gate (interm) ; up (interm)]."""
    return torch.cat([sd[f"{prefix}.mlp.gate_proj.weight"],
                      sd[f"{prefix}.mlp.up_proj.weight"]], dim=0)


def _interleave_gate_up(sd, prefix):
    """Same two matrices with gate/up alternating: row 2j = gate_j, row 2j+1 = up_j.

    The gated decode kernel consumes ADJACENT (gate, up) accumulator columns, so
    a warp's 8-column tile holds four complete pairs.
    """
    g = sd[f"{prefix}.mlp.gate_proj.weight"]
    u = sd[f"{prefix}.mlp.up_proj.weight"]
    out = torch.empty(2 * g.shape[0], g.shape[1], dtype=g.dtype)
    out[0::2] = g
    out[1::2] = u
    return out


def quantize_nvfp4(w_bf16: torch.Tensor, stream) -> Nvfp4Linear:
    """Quantize a BF16 (N, K) weight with FlashRT's SW120 weight quantizer.

    Returns packed values, swizzled UE4M3 block scales, and the GEMM scalar
    (`alpha = global_scale`, because the kernel stores scales pre-divided by it).
    """
    n, k = w_bf16.shape
    assert k % 16 == 0, f"K={k} must be a multiple of 16 for 16-element SF blocks"
    w = w_bf16.contiguous().to("cuda", torch.bfloat16)
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device="cuda")
    sf_bytes = fvk.nvfp4_sf_swizzled_bytes(n, k)
    sf = torch.zeros(sf_bytes, dtype=torch.uint8, device="cuda")
    scratch = torch.zeros(1, dtype=torch.float32, device="cuda")
    gscale = torch.zeros(1, dtype=torch.float32, device="cuda")
    fvk.bf16_weight_to_nvfp4_swizzled(
        w.data_ptr(), packed.data_ptr(), sf.data_ptr(),
        scratch.data_ptr(), gscale.data_ptr(), n, k, stream)
    torch.cuda.synchronize()
    return Nvfp4Linear(n=n, k=k, packed=packed, sf=sf, alpha=float(gscale.item()))


def load_weights(ckpt_dir: str, cfg: SparkX25Config, device="cuda",
                 dtype=torch.bfloat16, stream=None) -> ModelWeights:
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    sd = _load_state_dict(ckpt_dir)

    embed = sd["model.embedding.weight"].to(device, dtype).contiguous()
    mw = ModelWeights(cfg=cfg, embed=embed)
    mw.lm_head = quantize_nvfp4(embed, stream)
    mw.final_norm = sd["model.norm.weight"].to(device, dtype).contiguous()

    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        qkv = quantize_nvfp4(sd[f"{p}.self_attn.q_k_v_proj.weight"], stream)
        g = quantize_nvfp4(sd[f"{p}.self_attn.g_proj.weight"], stream)
        o = quantize_nvfp4(sd[f"{p}.self_attn.out_proj.weight"], stream)
        gu = quantize_nvfp4(_split_gate_up(sd, p), stream)
        gu_il = quantize_nvfp4(_interleave_gate_up(sd, p), stream)
        dn = quantize_nvfp4(sd[f"{p}.mlp.down_proj.weight"], stream)
        lw = LayerWeights(
            qkv=qkv, g_proj=g, o_proj=o, gate_up=gu, gate_up_il=gu_il, down=dn,
            input_norm=sd[f"{p}.input_layernorm.weight"].to(device, dtype).contiguous(),
            post_attn_norm=sd[f"{p}.post_attention_layernorm.weight"].to(device, dtype).contiguous(),
        )
        mw.layers.append(lw)
        mw.g_proj_bf16.append(
            sd[f"{p}.self_attn.g_proj.weight"].to(device, dtype).contiguous())
        # the BF16 sources are dead once quantized; free as we go
        for key in (f"{p}.self_attn.q_k_v_proj.weight", f"{p}.self_attn.g_proj.weight",
                    f"{p}.self_attn.out_proj.weight", f"{p}.mlp.gate_proj.weight",
                    f"{p}.mlp.up_proj.weight", f"{p}.mlp.down_proj.weight",
                    f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"):
            sd.pop(key, None)
    del sd
    torch.cuda.empty_cache()
    return mw
