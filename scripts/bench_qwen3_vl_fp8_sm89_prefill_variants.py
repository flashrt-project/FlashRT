#!/usr/bin/env python3
"""Sweep SM89 FP8 blockscaled GEMM variants on runtime prefill shapes."""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _time_cuda(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times), statistics.mean(times), min(times)


def _fmt_triplet(t: tuple[float, float, float]) -> str:
    return f"{t[0]:8.4f} / {t[1]:8.4f} / {t[2]:8.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--S", type=int, default=1581)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(torch.device(args.device))

    from flash_rt import flash_rt_kernels as fvk
    from flash_rt.frontends.torch.qwen3_vl_fp8_sm89 import (
        Qwen3VlFp8Sm89TextFrontend,
    )

    fe = Qwen3VlFp8Sm89TextFrontend(
        args.checkpoint, device=args.device, max_seq=max(args.S, 2048),
        max_prefill_seq=args.S)
    cfg = fe._cfg
    assert cfg is not None
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate"])
    lw = fe._weights.ptrs["layers"][args.layer]
    stream = torch.cuda.current_stream().cuda_stream

    variants = [
        ("16x64_w4", fvk.fp8_block128_gemm_bs_sm89_16x64x128_w4),
        ("16x128_w4", fvk.fp8_block128_gemm_bs_sm89_16x128x128_w4),
        ("32x64_w4", fvk.fp8_block128_gemm_bs_sm89_32x64x128_w4),
        ("32x128_w4", fvk.fp8_block128_gemm_bs_sm89_32x128x128_w4),
        ("64x64_w4", fvk.fp8_block128_gemm_bs_sm89_64x64x128_w4),
        ("64x128_w4", fvk.fp8_block128_gemm_bs_sm89_64x128x128_w4),
        ("64x128_w8", fvk.fp8_block128_gemm_bs_sm89_64x128x128_w8),
        ("128x64_w4", fvk.fp8_block128_gemm_bs_sm89_128x64x128_w4),
        ("128x128_w4", fvk.fp8_block128_gemm_bs_sm89_128x128x128_w4),
        ("128x128_w8", fvk.fp8_block128_gemm_bs_sm89_128x128x128_w8),
        ("32x128_w4_s1", fvk.fp8_block128_gemm_bs_sm89_32x128x128_w4_s1),
        ("64x64_w4_s1", fvk.fp8_block128_gemm_bs_sm89_64x64x128_w4_s1),
        ("128x128_w8_s1", fvk.fp8_block128_gemm_bs_sm89_128x128x128_w8_s1),
        ("auto", fvk.fp8_block128_gemm_blockscaled_sm89_bf16out),
    ]
    shapes = [
        ("qkv", int(lw["qkv_proj_w"]), int(lw["qkv_proj_s"]),
         int(lw["qkv_proj_N"]), hidden),
        ("o_proj", int(lw["o_proj_w"]), int(lw["o_proj_s"]), hidden,
         hidden),
        ("gate", int(lw["mlp_gate_w"]), int(lw["mlp_gate_s"]), inter,
         hidden),
        ("up", int(lw["mlp_up_w"]), int(lw["mlp_up_s"]), inter, hidden),
        ("down", int(lw["mlp_down_w"]), int(lw["mlp_down_s"]), hidden,
         inter),
    ]

    print(f"device={torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"S={args.S} layer={args.layer} warmup={args.warmup} "
          f"iters={args.iters}")
    total_auto = 0.0
    total_best = 0.0
    for name, w_ptr, ws_ptr, N, K in shapes:
        x = torch.randn(args.S, K, device=args.device,
                        dtype=torch.float32).to(torch.bfloat16)
        x_fp8 = torch.empty(args.S, K, device=args.device,
                            dtype=torch.float8_e4m3fn)
        x_scale = torch.empty(args.S, K // 128, device=args.device,
                              dtype=torch.float32)
        out = torch.empty(args.S, N, device=args.device, dtype=torch.bfloat16)
        fvk.fp8_per_token_block128_quant_bf16(
            x.data_ptr(), x_fp8.data_ptr(), x_scale.data_ptr(),
            args.S, K, stream)
        rows = []
        for variant_name, fn in variants:
            def call(fn=fn) -> None:
                fn(x_fp8.data_ptr(), w_ptr, out.data_ptr(), args.S, N, K,
                   x_scale.data_ptr(), ws_ptr, stream)
            t = _time_cuda(call, args.warmup, args.iters)
            rows.append((t[0], variant_name, t))
        rows.sort()
        best = rows[0]
        auto = next(row for row in rows if row[1] == "auto")
        total_best += best[0]
        total_auto += auto[0]
        print(f"--- {name} M={args.S} N={N} K={K} ---")
        for _, variant_name, t in rows:
            marker = " *" if variant_name == "auto" else ""
            print(f"{variant_name:12s} {_fmt_triplet(t)}{marker}")
        print(f"best={best[1]} auto_over_best={auto[0] / best[0]:.4f}x")

    print("--- one-layer runtime-shape total ---")
    print(f"auto_total_ms={total_auto:.4f}")
    print(f"best_per_shape_total_ms={total_best:.4f}")
    print(f"auto_over_best={total_auto / total_best:.4f}x")
    print(f"36_layer_auto_total_ms={total_auto * 36:.2f}")
    print(f"36_layer_best_per_shape_total_ms={total_best * 36:.2f}")


if __name__ == "__main__":
    main()
