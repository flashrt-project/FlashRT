#!/usr/bin/env python3
"""Route B: re-quantize MiniMax-M3 routed experts to FP8 block-128.

User chose experts-FP8 (cos ~0.99) over single-Spark speed. Resident weights
stay BF16 (loaded from the original checkpoint at runtime — small). Only the
413B routed-expert weights are re-quantized here, from the ORIGINAL BF16
checkpoint (FP4->FP8 would not recover precision).

Per-128x128-block e4m3 scaling, matching FlashRT's fp8_block128_dequantize_to_bf16
(scale layout [N/128, K/128] row-major, out[i,j]=fp8[i,j]*scale[(i/128)*kb+j/128]).

Output /models/MiniMax-M3-FP8E/:
  manifest.json
  experts_fp8_layer_NN.bin   (sparse layers) 128 fixed blocks, each:
    w1_fp8 | w1_scale | w3_fp8 | w3_scale | w2_fp8 | w2_scale
  expert_meta_NN.pt          per-layer: nothing extra (scales are inline)

Resident weights are NOT re-saved; the FP8 runtime loads them as BF16 from the
original checkpoint. Resumable; early selfcheck aborts if FP8 cos is bad.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_st_reader import RawShardReader  # noqa: E402

HIDDEN = 6144
INTER = 3072
N_EXPERTS = 128
LAYERS = 60
DENSE_LAYERS = {0, 1, 2}
PFX = "language_model.model."
BLK = 128
E4M3_MAX = 448.0

# fixed FP8 block layout (bytes)
W1_FP8 = INTER * HIDDEN                      # 18,874,368
W1_SCALE = (INTER // BLK) * (HIDDEN // BLK) * 4   # 24*48*4 = 4608
W3_FP8, W3_SCALE = W1_FP8, W1_SCALE
W2_FP8 = HIDDEN * INTER                      # 18,874,368
W2_SCALE = (HIDDEN // BLK) * (INTER // BLK) * 4   # 48*24*4 = 4608
BLOCK_BYTES = W1_FP8 + W1_SCALE + W3_FP8 + W3_SCALE + W2_FP8 + W2_SCALE


def quant_fp8_block128(w_bf16, device):
    """w [N,K] bf16 -> (fp8 u8 bytes [N*K], scale fp32 [N/128*K/128])."""
    N, K = w_bf16.shape
    nb, kb = N // BLK, K // BLK
    wb = w_bf16.float().view(nb, BLK, kb, BLK)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-9)
    scale = amax / E4M3_MAX
    q = (wb / scale).to(torch.float8_e4m3fn).view(N, K)
    scale_t = scale.view(nb, kb).contiguous()
    return q, scale_t


def selfcheck(w_bf16, q, scale):
    """cos(dequant, original) — should be ~0.999 for FP8."""
    N, K = w_bf16.shape
    nb, kb = N // BLK, K // BLK
    deq = (q.float().view(nb, BLK, kb, BLK)
           * scale.view(nb, 1, kb, 1)).view(N, K)
    return float(F.cosine_similarity(deq.flatten(),
                                     w_bf16.float().flatten(), dim=0))


def do_layer(i, rd, out_dir, device, check):
    bin_path = os.path.join(out_dir, f"experts_fp8_layer_{i:02d}.bin")
    if os.path.exists(bin_path) and \
            os.path.getsize(bin_path) == N_EXPERTS * BLOCK_BYTES:
        print(f"layer {i}: exists, skip", flush=True)
        return None
    b = f"{PFX}layers.{i}.block_sparse_moe.experts."
    names = [f"{b}{e}.{wn}.weight"
             for e in range(N_EXPERTS) for wn in ("w1", "w3", "w2")]
    t0 = time.time()
    cpu = rd.get_many(names, device="cpu")
    cos0 = None
    tmp = bin_path + ".tmp"
    with open(tmp, "wb") as f:
        for e in range(N_EXPERTS):
            blk = bytearray()
            for j, wn in enumerate(("w1", "w3", "w2")):
                w = cpu[3 * e + j].to(device).to(torch.bfloat16)
                q, scale = quant_fp8_block128(w, device)
                if check and e == 0 and j == 0:
                    cos0 = selfcheck(w, q, scale)
                blk += q.view(torch.uint8).cpu().numpy().tobytes()
                blk += scale.cpu().numpy().tobytes()
                del w, q, scale
            assert len(blk) == BLOCK_BYTES, (len(blk), BLOCK_BYTES)
            f.write(blk)
    os.replace(tmp, bin_path)
    rd.drop_all()
    print(f"layer {i}: done {time.time() - t0:.1f}s selfcheck_cos {cos0}",
          flush=True)
    return cos0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/MiniMax-M3")
    ap.add_argument("--out", default="/models/MiniMax-M3-FP8E")
    ap.add_argument("--layers", default="3:60")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rd = RawShardReader(args.model, args.device)

    manifest = {
        "format": "flashrt-m3-fp8-experts-v1",
        "block_bytes": BLOCK_BYTES,
        "block_layout": ["w1_fp8", "w1_scale", "w3_fp8", "w3_scale",
                         "w2_fp8", "w2_scale"],
        "sizes": {"w1_fp8": W1_FP8, "w1_scale": W1_SCALE,
                  "w3_fp8": W3_FP8, "w3_scale": W3_SCALE,
                  "w2_fp8": W2_FP8, "w2_scale": W2_SCALE},
        "blk": BLK, "n_experts": N_EXPERTS,
        "dense_layers": sorted(DENSE_LAYERS),
        "resident": "BF16 from original checkpoint (not re-saved)",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    a, bnd = (int(x) for x in args.layers.split(":"))
    first = True
    for i in range(a, bnd):
        if i in DENSE_LAYERS:
            continue
        c = do_layer(i, rd, args.out, args.device, check=True)
        if first and c is not None:
            if c < 0.99:
                print(f"ABORT: FP8 selfcheck cos {c} < 0.99 — scheme wrong",
                      flush=True)
                return
            print(f"FP8 quality confirmed (cos {c:.5f}), continuing.",
                  flush=True)
            first = False
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
