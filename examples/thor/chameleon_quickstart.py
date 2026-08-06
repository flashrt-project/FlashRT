#!/usr/bin/env python
"""Chameleon-7B (Thor sm_110) quickstart.

Runs the standalone Chameleon-7B image+text frontend on a real image and
reports prefill latency. The frontend is a direct-instantiation class
(``ChameleonTorchFrontendThor``, registered in ``_PIPELINE_MAP`` but not
dispatched by ``flash_rt.load_model`` — same pattern as Qwen3-VL / Nex-N2
/ LingBot).

Build first (one shared module — the Chameleon kernels live inside
flash_rt_kernels; FA4 is optional):

    cmake -B build -S . -DGPU_ARCH=110
    cmake --build build -j
    pip install -e ".[torch]"            # add ,thor-fa4 for the FA4 fast path

Run:

    python examples/thor/chameleon_quickstart.py \
        --checkpoint /path/to/Chameleon_7B_mGPT \
        --image /path/to/hand_1.jpg \
        --prompt "Describe the image."

Expected on Thor (dynamic FP8, CUDA graph, target_size=512, eager VQGAN):
~190 ms/prefill E2E; with --use-trt-vqgan (engines present) ~120 ms;
with FA4 enabled (FLASHRT_CHAMELEON_FA4_ATTN=1) the transformer part drops
to ~104 ms. The script prints the actual VQGAN backend and FA4 status.
"""
import argparse
import time

import torch

from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor
from flash_rt.hardware.thor import fa4_backend


def main() -> None:
    ap = argparse.ArgumentParser(description="Chameleon-7B Thor quickstart")
    ap.add_argument("--checkpoint", required=True,
                    help="Chameleon-7B dir (model-*-of-*.safetensors + config.json)")
    ap.add_argument("--image", required=True, help="real image path (jpg/png)")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--use-trt-vqgan", action="store_true",
                    help="Use TensorRT VQGAN if compatible engines exist "
                         "(recommended when engines are available; default is eager VQGAN)")
    ap.add_argument("--use-fp16", action="store_true",
                    help="Use the FP16 reference path instead of dynamic FP8")
    ap.add_argument("--no-graph", action="store_true", help="Disable CUDA Graph")
    ap.add_argument("--iters", type=int, default=10, help="timed replays")
    ap.add_argument("--benchmark", action="store_true",
                    help="report wall-clock prefill latency (P50)")
    args = ap.parse_args()

    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    fa4 = fa4_backend.is_available()
    print(f"[chameleon] FA4 attention available: {fa4} ({fa4_backend.status()})"
          f"{'' if fa4 else '  <-- CUTLASS FMHA will be used; pip install .[thor-fa4]'}")

    fe = ChameleonTorchFrontendThor(
        args.checkpoint,
        use_fp8=not args.use_fp16,
        use_cuda_graph=not args.no_graph,
        target_size=args.target_size,
        use_trt_vqgan=args.use_trt_vqgan,
    )
    print(f"[chameleon] VQGAN backend: {fe.vqgan_backend}  "
          f"FA4 active: {fe.fa4_attn_active}")

    out = fe.prefill(args.prompt, [image])
    print(f"[chameleon] Se={out['Se']} (real_len={len([i for i in out['input_ids'] if i != 1])}) "
          f"logits={tuple(out['logits'].shape)}")
    top = int(torch.argmax(out["logits"]).item())
    print(f"[chameleon] greedy next-token id: {top}")

    if args.benchmark:
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fe.prefill(args.prompt, [image])
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        p50 = ts[len(ts) // 2]
        print(f"[chameleon] prefill P50: {p50:.1f} ms over {args.iters} iters "
              f"(wall-clock, includes VQGAN; fp8={not args.use_fp16}, "
              f"graph={not args.no_graph}, vqgan={fe.vqgan_backend})")


if __name__ == "__main__":
    main()
