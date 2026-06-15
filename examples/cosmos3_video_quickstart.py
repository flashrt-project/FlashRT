#!/usr/bin/env python3
"""Cosmos3-Nano text2video FP8 denoise quickstart.

Runs the kernelized UniPC denoise (FP8 GEMMs, static text-KV cache, optional
TeaCache step caching) through the standard FlashRT API and reports the denoise
latency + the latent cosine vs the official reference.

Conditioning (text / VAE encode) is consumed from the official reference dump;
this is the denoise policy, so infer() returns the denoised vision latent (Wan
VAE decode to frames is the downstream step).

Build the model-local kernels once on the target GPU:
  cd flash_rt/models/cosmos3_video/kernels && python3 setup.py build_ext --inplace

Run (all config via typed parameters; no environment knobs):
  python3 examples/cosmos3_video_quickstart.py \
      --checkpoint <cosmos3 flat weights .safetensors> \
      --ref <.../tensors.safetensors> [--teacache-skip 3,5,7] [--shift 10.0]
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--ref", required=True,
                        help="Official reference dump tensors.safetensors")
    parser.add_argument("--bf16", action="store_true",
                        help="reference-accuracy bf16 path (default: fp8)")
    parser.add_argument("--teacache-skip", default="",
                        help="TeaCache skip steps, e.g. 3,5,7 (safe) or 2,4,6,8")
    parser.add_argument("--shift", type=float, default=10.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        config="cosmos3_video",
        hardware="rtx_sm120",
        use_fp8=not args.bf16,
    )
    model.set_prompt(ref=args.ref)
    print(f"[cosmos3_video] load_model + set_prompt "
          f"wall={time.perf_counter() - t0:.2f}s")

    out = model.infer(
        teacache_skip=args.teacache_skip,
        shift=args.shift,
        compare_ref=True,
        return_metadata=True,
    )
    quant = "bf16" if args.bf16 else "fp8"
    print(f"[cosmos3_video] denoise {out['latency_ms']:.1f} ms  quant={quant}"
          f"  teacache_skip=[{args.teacache_skip}]")
    print(f"[cosmos3_video] latent {tuple(out['latent'].shape)}")
    print(f"[cosmos3_video] latent cos {out['cos']:.5f}  "
          f"rel_l2 {out['rel_l2'] * 100:.3f}%  (vs official reference)")


if __name__ == "__main__":
    main()
