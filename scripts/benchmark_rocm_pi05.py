#!/usr/bin/env python3
"""Synthetic Pi0.5 ROCm benchmark helpers."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def _time_ms(fn: Callable[[], Any], *, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    _sync()

    samples = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    count = len(samples)
    return {
        "count": float(count),
        "min_ms": samples[0],
        "p50_ms": samples[count // 2],
        "max_ms": samples[-1],
        "mean_ms": sum(samples) / count,
    }


def _require_rocm() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("benchmark_rocm_pi05.py requires ROCm PyTorch")
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "capability": tuple(int(x) for x in torch.cuda.get_device_capability()),
    }


def _benchmark_bf16_linear(args) -> dict[str, Any]:
    import torch

    from flash_rt import flash_rt_rocm_kernels as rocm

    x = torch.randn(args.linear_m, args.linear_k, device="cuda").to(torch.bfloat16)
    weight = torch.randn(args.linear_n, args.linear_k, device="cuda").to(
        torch.bfloat16
    )
    bias = torch.randn(args.linear_n, device="cuda").to(torch.bfloat16)

    def run():
        rocm.hipblaslt_linear_bf16(x, weight, bias)

    return _time_ms(run, warmup=args.warmup, repeat=args.repeat)


def _benchmark_attention(args) -> dict[str, Any]:
    import torch

    from flash_rt.hardware.rocm.attn_backend import RocmSdpaAttnBackend

    backend = RocmSdpaAttnBackend(
        num_views=args.num_views,
        encoder_seq_max=args.encoder_seq,
        chunk_size=args.chunk_size,
        preferred_backend=args.attn_backend,
        decoder_preferred_backend=args.attn_backend,
    )
    backend.vis_Q.normal_()
    backend.vis_K.normal_()
    backend.vis_V.normal_()
    backend.enc_Q[: args.encoder_seq].normal_()
    backend.enc_K[0, : args.encoder_seq + args.chunk_size].normal_()
    backend.enc_V[0, : args.encoder_seq + args.chunk_size].normal_()
    backend.dec_Q[: args.chunk_size].normal_()
    torch.cuda.synchronize()

    return {
        "backend": backend.active_backend_name,
        "decoder_backend": backend.decoder_backend_name,
        "siglip": _time_ms(
            lambda: backend.run("siglip", 0, 256),
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "encoder": _time_ms(
            lambda: backend.run("encoder", 0, args.encoder_seq),
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "decoder": _time_ms(
            lambda: backend.run(
                "decoder",
                0,
                args.chunk_size,
                kv_seq=args.encoder_seq + args.chunk_size,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        ),
    }


def _probe_fp8() -> dict[str, Any]:
    from flash_rt.hardware.rocm.backend import get_backend_info

    info = get_backend_info(probe_fp8_gemm=True)
    return {
        "supports_fp8_dtype": info.supports_fp8_dtype,
        "supports_fp8_gemm": info.supports_fp8_gemm,
    }


def _benchmark_pipeline_bake(args) -> dict[str, Any]:
    from flash_rt import flash_rt_rocm_kernels as rocm
    from flash_rt.models.pi05.pipeline_rocm import Pi05PipelineRocm

    pipe = Pi05PipelineRocm(
        num_views=args.num_views,
        max_prompt_len=args.max_prompt_len,
        chunk_size=args.chunk_size,
        num_steps=args.num_steps,
    )
    start = time.perf_counter()
    result = pipe.bake_bf16_gemms(rocm)
    rocm.hip_sync()
    result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--encoder-seq", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--linear-m", type=int, default=512)
    parser.add_argument("--linear-n", type=int, default=2048)
    parser.add_argument("--linear-k", type=int, default=2048)
    parser.add_argument(
        "--attn-backend",
        choices=("ck_wmma",),
        default="ck_wmma",
    )
    parser.add_argument("--fp8", action="store_true", help="probe FP8 GEMM support")
    parser.add_argument(
        "--include-pipeline-bake",
        action="store_true",
        help="include the larger Pi0.5 BF16 GEMM bake path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "runtime": _require_rocm(),
        "bf16_linear": _benchmark_bf16_linear(args),
        "attention": _benchmark_attention(args),
    }
    if args.fp8:
        result["fp8"] = _probe_fp8()
    if args.include_pipeline_bake:
        result["pipeline_bf16_bake"] = _benchmark_pipeline_bake(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
