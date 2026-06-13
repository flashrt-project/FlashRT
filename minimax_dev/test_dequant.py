#!/usr/bin/env python3
"""Validate nvfp4_dequant_swizzled_to_bf16 against the original BF16 weights.

For a sampled set of quantized projections in /models/MiniMax-M3-NVFP4, run
the dequant kernel on the packed+swizzled artifact and compare to the ORIGINAL
BF16 weight from the source checkpoint:
  - cos (per-tensor quant fidelity; expect ~0.99)
  - magnitude ratio mean|deq| / mean|orig| (expect ~1.0 — confirms alpha)
This validates both the dequant math and the stored alpha before the runtime
adopts the W4A16 path.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/FlashRT/flash_rt")
import flash_rt_kernels as fvk  # noqa: E402
from raw_st_reader import RawShardReader  # noqa: E402

QDIR = "/models/MiniMax-M3-NVFP4"
SRC = "/models/MiniMax-M3"
PFX = "language_model.model."
DEV = "cuda:0"


def dequant(packed, sf, alpha, N, K):
    d = torch.empty(N, K, dtype=torch.bfloat16, device=DEV)
    fvk.nvfp4_dequant_swizzled_to_bf16(
        int(packed.data_ptr()), int(sf.data_ptr()), int(d.data_ptr()),
        N, K, alpha, 0)
    torch.cuda.synchronize()
    return d


def check(name, packed, sf, alpha, orig):
    N, K2 = packed.shape
    K = K2 * 2
    deq = dequant(packed.to(DEV), sf.to(DEV), alpha, N, K)
    o = orig.to(DEV).to(torch.float32)
    d = deq.to(torch.float32)
    cos = F.cosine_similarity(d.flatten(), o.flatten(), dim=0)
    ratio = d.abs().mean() / o.abs().mean().clamp(min=1e-9)
    print(f"  {name:28s} N={N:5d} K={K:5d} cos {float(cos):.5f} "
          f"mag {float(ratio):.4f} alpha {alpha:.4e}", flush=True)
    return float(cos)


def main():
    rd = RawShardReader(SRC, DEV)
    coss = []
    # layer 0 dense + layer 5 sparse: q_proj, o_proj, (dense) mlp_down,
    # (sparse) shared down + expert 0 w1/w2
    for li in [0, 5, 30, 59]:
        res = torch.load(os.path.join(QDIR, f"resident_layer_{li:02d}.pt"),
                         map_location="cpu", weights_only=False)
        p = f"{PFX}layers.{li}."
        for pref, src in [("q_proj", "self_attn.q_proj.weight"),
                          ("o_proj", "self_attn.o_proj.weight")]:
            orig = rd.get(p + src, "cpu")
            coss.append(check(f"L{li}.{pref}", res[pref + "_packed"],
                              res[pref + "_sf"], res[pref + "_alpha"], orig))
        if li in (0, 1, 2):
            orig = rd.get(p + "mlp.down_proj.weight", "cpu")
            coss.append(check(f"L{li}.mlp_down", res["mlp_down_packed"],
                              res["mlp_down_sf"], res["mlp_down_alpha"], orig))
        else:
            orig = rd.get(p + "block_sparse_moe.shared_experts.down_proj.weight",
                          "cpu")
            coss.append(check(f"L{li}.sh_down", res["sh_down_packed"],
                              res["sh_down_sf"], res["sh_down_alpha"], orig))
        rd.drop_all()

    # one routed expert from the packed bin
    from m3_quant_nvfp4 import (BLOCK_BYTES, INTER, W1_PACKED, W1_SF)
    from m3_nvfp4_layerwise_check import OFF_W1P, OFF_W1S
    HIDDEN = 6144
    li = 30
    with open(os.path.join(QDIR, f"experts_layer_{li:02d}.bin"), "rb") as f:
        blk = f.read(BLOCK_BYTES)
    res = torch.load(os.path.join(QDIR, f"resident_layer_{li:02d}.pt"),
                     map_location="cpu", weights_only=False)
    al = res["expert_alphas"]
    w1p = torch.frombuffer(bytearray(blk[OFF_W1P:OFF_W1P + W1_PACKED]),
                           dtype=torch.uint8).view(INTER, HIDDEN // 2)
    w1s = torch.frombuffer(bytearray(blk[OFF_W1S:OFF_W1S + W1_SF]),
                           dtype=torch.uint8)
    orig = rd.get(f"{PFX}layers.{li}.block_sparse_moe.experts.0.w1.weight",
                  "cpu")
    coss.append(check(f"L{li}.expert0.w1", w1p, w1s, float(al[0, 0]), orig))

    t = torch.tensor(coss)
    print(f"\nsummary: min {t.min():.5f} mean {t.mean():.5f} "
          f"max {t.max():.5f} over {len(coss)} tensors")


if __name__ == "__main__":
    main()
