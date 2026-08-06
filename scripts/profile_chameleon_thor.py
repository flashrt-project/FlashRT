#!/usr/bin/env python3
"""Nsight Systems profiling helper for standalone Chameleon Thor.

Use with CUDA profiler capture range, for example:

    nsys profile --force-overwrite=true \
      -o /tmp/chameleon_prefill_only_512 \
      --capture-range=cudaProfilerApi -t cuda,nvtx \
      env PYTHONPATH=. python scripts/profile_chameleon_thor.py \
        --checkpoint /path/to/Chameleon_7B_mGPT \
        --image-dir /path/to/images \
        --target-size 512 --reuse-input-ids --iters 5
"""

from __future__ import annotations

import argparse
import pathlib


def _load_real_images(image_dir: pathlib.Path, max_images: int):
    from PIL import Image
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No real images found under {image_dir}")
    paths = paths[:max_images]
    return [Image.open(p).convert("RGB") for p in paths]


def _pad_ids(input_ids: list[int], pad_id: int = 1) -> tuple[list[int], int]:
    real_len = len(input_ids)
    padded = list(input_ids)
    rem = len(padded) % 16
    if rem:
        padded.extend([pad_id] * (16 - rem))
    return padded, real_len


def _run_prefill_body(fe, prompt: str, images, cached_ids, *, use_graph: bool):
    import torch

    if cached_ids is None:
        ids = fe.encode_prompt(prompt, images)
    else:
        ids = cached_ids
    padded, real_len = _pad_ids(ids)
    fe._real_len = real_len
    fe.Se = len(padded)
    fe._last_input_ids = padded
    if fe._use_autotune:
        fe._autotune_gemms(fe.Se)
    fe._embed_ids(padded)
    if use_graph:
        fe._capture_graph(fe.Se)
        fe._infer_graph.replay()
    else:
        fe._run_backbone(fe.Se)
    fe._project_last()
    torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-dir", required=True,
                    help="Directory of real input images")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--max-images", type=int, default=1)
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--use-trt-vqgan", action="store_true")
    ap.add_argument("--trt-vqgan-engine-dir", default=None)
    ap.add_argument("--use-fp16", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--reuse-input-ids", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    import torch
    from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor

    images = _load_real_images(pathlib.Path(args.image_dir), args.max_images)
    fe = ChameleonTorchFrontendThor(
        args.checkpoint,
        use_fp8=not args.use_fp16,
        use_cuda_graph=not args.no_graph,
        target_size=args.target_size,
        use_trt_vqgan=args.use_trt_vqgan,
        trt_vqgan_engine_dir=args.trt_vqgan_engine_dir,
    )
    cached_ids = fe.encode_prompt(args.prompt, images) if args.reuse_input_ids else None

    for _ in range(args.warmup):
        _run_prefill_body(fe, args.prompt, images, cached_ids, use_graph=not args.no_graph)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        _run_prefill_body(fe, args.prompt, images, cached_ids, use_graph=not args.no_graph)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print({
        "Se": fe.Se,
        "real_len": fe._real_len,
        "vqgan_backend": fe.vqgan_backend,
        "use_fp8": not args.use_fp16,
        "graph": not args.no_graph,
        "reuse_input_ids": args.reuse_input_ids,
        "iters": args.iters,
    })


if __name__ == "__main__":
    main()
