#!/usr/bin/env python3
"""Cosmos3-Nano text2video FP8 denoise quickstart.

Runs the kernelized 10-step UniPC denoise (FP8 GEMMs, static text-KV cache,
optional TeaCache step caching) through the standard FlashRT API and reports the
denoise latency + the latent cosine vs the official reference.

Conditioning (text / VAE encode) is upstream and consumed from the official
reference dump; this is the denoise policy, so infer() returns the denoised vision
latent (Wan VAE decode to frames is the downstream step).

Build the model-local kernels once on the target GPU:
  cd flash_rt/models/cosmos3_av/kernels && python3 setup.py build_ext --inplace

Run (paths via args; no host paths baked in):
  python3 examples/cosmos3_video_quickstart.py \
      --checkpoint <cosmos3 flat weights .safetensors> \
      --ref <.../tensors.safetensors> [--quant fp8] [--teacache-skip 3,5,7]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flash_rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Cosmos3 flat-format weights (.safetensors)")
    parser.add_argument("--ref", default=os.environ.get("COSMOS_R"),
                        help="Official reference dump tensors.safetensors")
    parser.add_argument("--quant", choices=("fp8", "bf16", "fp4"), default="fp8",
                        help="denoise precision (fp8 is near-lossless for video)")
    parser.add_argument("--teacache-skip", default="",
                        help="TeaCache skip steps, e.g. 3,5,7 (safe) or 2,4,6,8")
    parser.add_argument("--shift", type=float, default=10.0)
    args = parser.parse_args()
    if not args.ref:
        raise SystemExit("--ref (or COSMOS_R) is required")

    os.environ["COSMOS_R"] = args.ref
    os.environ["COSMOS_VIDEO_QUANT"] = args.quant
    os.environ["COSMOS_VIDEO_TEACACHE_SKIP"] = args.teacache_skip
    os.environ["COSMOS_SHIFT"] = str(args.shift)

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        config="cosmos3_video",
        hardware="rtx_sm120",
    )
    print(f"[cosmos3_video] load_model wall={time.perf_counter() - t0:.2f}s")

    out = model.infer({"compare_ref": True})
    print(f"[cosmos3_video] denoise {out['latency_ms']:.1f} ms  quant={args.quant}"
          f"  teacache_skip=[{args.teacache_skip}]")
    print(f"[cosmos3_video] latent {tuple(out['latent'].shape)}")
    print(f"[cosmos3_video] latent cos {out['cos']:.5f}  "
          f"rel_l2 {out['rel_l2'] * 100:.3f}%  (vs official reference)")


if __name__ == "__main__":
    main()
