#!/usr/bin/env python3
"""Validate the FP8 expert bin layout + fp8_block128_dequantize_to_bf16 kernel
against the original BF16 expert weights (uses already-quantized layers)."""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/FlashRT/flash_rt")
import flash_rt_kernels as fvk  # noqa: E402
from raw_st_reader import RawShardReader  # noqa: E402
from m3_quant_fp8_experts import (  # noqa: E402
    BLOCK_BYTES, HIDDEN, INTER, W1_FP8, W1_SCALE, W2_FP8, W2_SCALE, W3_FP8,
    W3_SCALE)
from m3_runtime_fp8 import (  # noqa: E402
    OFF_W1_FP8, OFF_W1_SCALE, OFF_W2_FP8, OFF_W2_SCALE, OFF_W3_FP8,
    OFF_W3_SCALE)

QDIR = "/models/MiniMax-M3-FP8E"
SRC = "/models/MiniMax-M3"
PFX = "language_model.model."
DEV = "cuda:0"


def dequant(fp8_bytes, scale_bytes, N, K):
    fp8 = torch.frombuffer(bytearray(fp8_bytes), dtype=torch.uint8).to(DEV)
    scale = torch.frombuffer(bytearray(scale_bytes),
                             dtype=torch.float32).to(DEV)
    out = torch.empty(N, K, dtype=torch.bfloat16, device=DEV)
    fvk.fp8_block128_dequantize_to_bf16(
        int(fp8.data_ptr()), int(scale.data_ptr()), int(out.data_ptr()),
        N, K, 0)
    torch.cuda.synchronize()
    return out


def main():
    rd = RawShardReader(SRC, DEV)
    coss = []
    for li in [3, 5, 7]:
        path = os.path.join(QDIR, f"experts_fp8_layer_{li:02d}.bin")
        if not os.path.exists(path):
            print(f"layer {li}: not quantized yet, skip")
            continue
        for e in [0, 17, 99]:
            with open(path, "rb") as f:
                f.seek(e * BLOCK_BYTES)
                blk = f.read(BLOCK_BYTES)
            specs = [
                ("w1", OFF_W1_FP8, W1_FP8, OFF_W1_SCALE, W1_SCALE, INTER, HIDDEN),
                ("w3", OFF_W3_FP8, W3_FP8, OFF_W3_SCALE, W3_SCALE, INTER, HIDDEN),
                ("w2", OFF_W2_FP8, W2_FP8, OFF_W2_SCALE, W2_SCALE, HIDDEN, INTER),
            ]
            b = f"{PFX}layers.{li}.block_sparse_moe.experts.{e}."
            for wn, of, fz, os_, sz, N, K in specs:
                deq = dequant(blk[of:of + fz], blk[os_:os_ + sz], N, K)
                orig = rd.get(b + f"{wn}.weight", DEV).to(torch.bfloat16)
                c = float(F.cosine_similarity(deq.float().flatten(),
                                              orig.float().flatten(), dim=0))
                coss.append(c)
                print(f"  L{li}.e{e}.{wn} N={N} K={K} cos {c:.5f}", flush=True)
        rd.drop_all()
    t = torch.tensor(coss)
    print(f"\nFP8 CUDA dequant cos: min {t.min():.5f} mean {t.mean():.5f} "
          f"over {len(coss)} tensors")


if __name__ == "__main__":
    main()
