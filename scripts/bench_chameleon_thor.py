#!/usr/bin/env python3
"""Real-image latency benchmark for standalone Chameleon-7B on Thor.

Measures HF BF16, FlashRT FP16, and FlashRT dynamic FP8 prefill
latency on the same real-image prompt. Inputs are always real images
(from a user-supplied directory), never synthetic token ids.

Usage
-----
    PYTHONPATH=. python scripts/bench_chameleon_thor.py \\
        --checkpoint /path/to/Chameleon_7B_mGPT \\
        --image-dir /path/to/images \\
        --prompt "Describe the image." \\
        --iters 10 --warmup 2 \\
        --output /tmp/chameleon_thor_bench.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import numpy as np


def _stats(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _load_real_images(image_dir: pathlib.Path, max_images: int):
    from PIL import Image
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No real images found under {image_dir}")
    paths = paths[:max_images]
    return [Image.open(p).convert("RGB") for p in paths], [str(p) for p in paths]


def _pad_ids(input_ids: List[int], pad_id: int = 1) -> tuple[List[int], int]:
    real_len = len(input_ids)
    padded = list(input_ids)
    rem = len(padded) % 16
    if rem:
        padded.extend([pad_id] * (16 - rem))
    return padded, real_len


def _estimate_prefill_tflops(Se: int) -> float:
    D, Dff, L, vocab = 4096, 11008, 32, 65536

    def gemm_flops(M: int, N: int, K: int) -> int:
        return 2 * M * N * K

    per_layer_gemm = (
        gemm_flops(Se, 3 * D, D)
        + gemm_flops(Se, D, D)
        + gemm_flops(Se, 2 * Dff, D)
        + gemm_flops(Se, D, Dff)
    )
    per_layer_attn = 4 * Se * Se * D
    lm_head = gemm_flops(1, vocab, D)
    return (L * (per_layer_gemm + per_layer_attn) + lm_head) / 1e12


def _roofline(Se: int, prefill_ms: float, peak_tflops: float) -> Dict[str, float]:
    tflops = _estimate_prefill_tflops(Se)
    achieved = tflops / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0
    floor_ms = tflops / peak_tflops * 1000.0 if peak_tflops > 0 else 0.0
    return {
        "estimated_tflops": float(tflops),
        "assumed_peak_tflops": float(peak_tflops),
        "achieved_tflops": float(achieved),
        "efficiency_vs_peak": float(achieved / peak_tflops) if peak_tflops > 0 else 0.0,
        "optimistic_compute_floor_ms": float(floor_ms),
        "measured_over_floor": float(prefill_ms / floor_ms) if floor_ms > 0 else 0.0,
    }


def _run_flashrt_prefill_once(fe, prompt: str, images, cached_ids,
                              *, use_cuda_graph: bool):
    import torch

    times = {}
    t0 = time.perf_counter()
    if cached_ids is None:
        ids = fe.encode_prompt(prompt, images)
    else:
        ids = cached_ids
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    padded, real_len = _pad_ids(ids)
    fe._real_len = real_len
    fe.Se = len(padded)
    fe._last_input_ids = padded
    if fe._use_autotune:
        fe._autotune_gemms(fe.Se)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    fe._embed_ids(padded)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    if use_cuda_graph:
        fe._capture_graph(fe.Se)
        fe._infer_graph.replay()
    else:
        fe._run_backbone(fe.Se)
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    fe._project_last()
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    times["encode_ms"] = (t1 - t0) * 1000.0
    times["prepare_ms"] = (t2 - t1) * 1000.0
    times["embed_ms"] = (t3 - t2) * 1000.0
    times["backbone_ms"] = (t4 - t3) * 1000.0
    times["lm_head_ms"] = (t5 - t4) * 1000.0
    times["transformer_prefill_ms"] = times["embed_ms"] + times["backbone_ms"] + times["lm_head_ms"]
    times["total_ms"] = (t5 - t0) * 1000.0
    return times, fe.Se, real_len, fe.vqgan_backend


def _bench_flashrt(checkpoint_dir: pathlib.Path, prompt: str, images,
                    *, use_fp8: bool, use_cuda_graph: bool,
                    target_size: int, use_trt_vqgan: bool,
                    trt_vqgan_engine_dir: str | None,
                    iters: int, warmup: int,
                    reuse_input_ids: bool,
                    generate_greedy: int,
                    peak_tflops: float) -> Dict:
    import torch
    from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor

    fe = ChameleonTorchFrontendThor(
        str(checkpoint_dir), use_fp8=use_fp8, use_cuda_graph=use_cuda_graph,
        target_size=target_size, use_trt_vqgan=use_trt_vqgan,
        trt_vqgan_engine_dir=trt_vqgan_engine_dir)

    cached_ids = None
    input_build_ms = None
    if reuse_input_ids:
        t0 = time.perf_counter()
        cached_ids = fe.encode_prompt(prompt, images)
        torch.cuda.synchronize()
        input_build_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        _run_flashrt_prefill_once(
            fe, prompt, images, cached_ids, use_cuda_graph=use_cuda_graph)

    stage_values: Dict[str, List[float]] = {}
    Se = real_len = None
    backend = fe.vqgan_backend
    for _ in range(iters):
        times, Se, real_len, backend = _run_flashrt_prefill_once(
            fe, prompt, images, cached_ids, use_cuda_graph=use_cuda_graph)
        for k, v in times.items():
            stage_values.setdefault(k, []).append(v)

    stage_stats = {k: _stats(v) for k, v in stage_values.items()}
    prefill_ms = stage_stats["transformer_prefill_ms"]["p50"]
    result = {
        "Se": int(Se),
        "real_len": int(real_len),
        "vqgan_backend": backend,
        "fa4_attn": fe.fa4_attn_active,
        "reuse_input_ids": bool(reuse_input_ids),
        "one_time_input_build_ms": input_build_ms,
        "latency_ms": stage_stats["total_ms"],
        "stage_breakdown_ms": stage_stats,
        "roofline": _roofline(int(Se), prefill_ms, peak_tflops),
    }

    if generate_greedy > 0:
        for _ in range(max(1, min(warmup, 2))):
            fe.generate_greedy(prompt, images, max_new_tokens=generate_greedy)
        gen_lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fe.generate_greedy(prompt, images, max_new_tokens=generate_greedy)
            torch.cuda.synchronize()
            gen_lat.append((time.perf_counter() - t0) * 1000.0)
        result["generate_greedy"] = {
            "max_new_tokens": int(generate_greedy),
            "latency_ms": _stats(gen_lat),
            "ms_per_token": _stats([x / generate_greedy for x in gen_lat]),
            "output_token_count": len(out["input_ids"]),
        }

    del fe
    torch.cuda.empty_cache()
    return result


def _bench_hf(checkpoint_dir: pathlib.Path, prompt: str, images,
                   *, target_size: int, use_trt_vqgan: bool,
                   trt_vqgan_engine_dir: str | None,
                   iters: int, warmup: int) -> Dict:
    import torch
    from transformers import AutoConfig
    from safetensors.torch import load_file

    try:
        from transformers import ChameleonForConditionalGeneration as _Cls
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[bench] HF BF16 reference unavailable ({e}); skipping")
        return None

    cfg = AutoConfig.from_pretrained(str(checkpoint_dir))
    cfg.rope_scaling = None
    if not hasattr(cfg, "rope_theta") or cfg.rope_theta is None:
        cfg.rope_theta = 10000.0

    model = _Cls(cfg)
    sd = {}
    for shard in sorted(checkpoint_dir.glob("model-*-of-*.safetensors")):
        sd.update(load_file(str(shard)))
    model.load_state_dict(sd, strict=False, assign=False)
    model = model.to(torch.bfloat16).cuda().eval()

    from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor
    fe = ChameleonTorchFrontendThor(
        str(checkpoint_dir), use_fp8=False, use_cuda_graph=False,
        target_size=target_size, use_trt_vqgan=use_trt_vqgan,
        trt_vqgan_engine_dir=trt_vqgan_engine_dir)
    ids = fe.encode_prompt(prompt, images)
    backend = fe.vqgan_backend
    del fe
    torch.cuda.empty_cache()

    ids_t = torch.tensor([ids], dtype=torch.long, device="cuda")

    def _fwd():
        with torch.no_grad():
            model(input_ids=ids_t, use_cache=False)
        torch.cuda.synchronize()

    for _ in range(warmup):
        _fwd()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _fwd()
        lat.append((time.perf_counter() - t0) * 1000.0)
    del model
    torch.cuda.empty_cache()
    return {"Se": len(ids), "vqgan_backend": backend, "latency_ms": _stats(lat)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-dir", required=True,
                    help="Directory of real input images")
    ap.add_argument("--prompt", default="Describe the image.")
    ap.add_argument("--max-images", type=int, default=1)
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--use-trt-vqgan", action="store_true",
                    help="Use TensorRT VQGAN if compatible engines exist "
                         "(recommended when available; default is eager VQGAN)")
    ap.add_argument("--trt-vqgan-engine-dir", default=None)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-graph", action="store_true",
                     help="Disable CUDA Graph capture for FlashRT paths")
    ap.add_argument("--reuse-input-ids", action="store_true",
                    help="Build real-image input ids once and benchmark transformer prefill only")
    ap.add_argument("--stage-breakdown", action="store_true",
                    help="Include per-stage timing in JSON output (currently always collected)")
    ap.add_argument("--generate-greedy", type=int, default=0,
                    help="Also benchmark full-prefix greedy generation for N new tokens")
    ap.add_argument("--peak-tflops", type=float, default=240.0,
                    help="Measured Thor FP8 GEMM plateau used for roofline efficiency")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--output", default="/tmp/chameleon_thor_bench.json")
    args = ap.parse_args()

    import torch

    checkpoint_dir = pathlib.Path(args.checkpoint)
    image_dir = pathlib.Path(args.image_dir)
    images, image_paths = _load_real_images(image_dir, args.max_images)
    device_name = torch.cuda.get_device_name(0)

    result: Dict = {
        "checkpoint": str(checkpoint_dir),
        "image_dir": str(image_dir),
        "image_paths": image_paths,
        "prompt": args.prompt,
        "num_images": len(images),
        "target_size": args.target_size,
        "use_trt_vqgan": bool(args.use_trt_vqgan),
        "trt_vqgan_engine_dir": args.trt_vqgan_engine_dir,
        "vqgan_backend_requested": "trt" if args.use_trt_vqgan else "eager",
        "device": device_name,
        "iters": args.iters,
        "warmup": args.warmup,
        "graph": not args.no_graph,
        "reuse_input_ids": bool(args.reuse_input_ids),
        "stage_breakdown": bool(args.stage_breakdown),
        "generate_greedy": int(args.generate_greedy),
        "peak_tflops": float(args.peak_tflops),
    }

    print("[bench] FlashRT FP16...")
    result["flashrt_fp16"] = _bench_flashrt(
        checkpoint_dir, args.prompt, images, use_fp8=False,
        use_cuda_graph=not args.no_graph, target_size=args.target_size,
        use_trt_vqgan=args.use_trt_vqgan,
        trt_vqgan_engine_dir=args.trt_vqgan_engine_dir,
        iters=args.iters, warmup=args.warmup,
        reuse_input_ids=args.reuse_input_ids,
        generate_greedy=args.generate_greedy,
        peak_tflops=args.peak_tflops)
    print(f"[bench] FlashRT FP16: {result['flashrt_fp16']}")

    print("[bench] FlashRT dynamic FP8...")
    result["flashrt_fp8"] = _bench_flashrt(
        checkpoint_dir, args.prompt, images, use_fp8=True,
        use_cuda_graph=not args.no_graph, target_size=args.target_size,
        use_trt_vqgan=args.use_trt_vqgan,
        trt_vqgan_engine_dir=args.trt_vqgan_engine_dir,
        iters=args.iters, warmup=args.warmup,
        reuse_input_ids=args.reuse_input_ids,
        generate_greedy=args.generate_greedy,
        peak_tflops=args.peak_tflops)
    print(f"[bench] FlashRT FP8: {result['flashrt_fp8']}")

    if not args.skip_hf:
        print("[bench] HF BF16 (eager)...")
        result["hf_bf16"] = _bench_hf(
            checkpoint_dir, args.prompt, images,
            target_size=args.target_size,
            use_trt_vqgan=args.use_trt_vqgan,
            trt_vqgan_engine_dir=args.trt_vqgan_engine_dir,
            iters=args.iters, warmup=args.warmup)
        print(f"[bench] HF BF16: {result['hf_bf16']}")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
