#!/usr/bin/env python3
"""Microbench SM89 Qwen3-VL official-FP8 M=S prefill GEMM candidates."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass

import torch
from safetensors import safe_open


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class LinearSpec:
    name: str
    key: str


def _open_shards(ckpt_dir: str):
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx_path, "r") as f:
        weight_map = json.load(f)["weight_map"]
    shards = {
        shard: safe_open(os.path.join(ckpt_dir, shard), framework="pt",
                         device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    return shards, weight_map


def _tensor(shards, weight_map, key: str) -> torch.Tensor:
    return shards[weight_map[key]].get_tensor(key)


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
    p.add_argument("--S", type=int, default=79)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(torch.device(args.device))

    from flash_rt import flash_rt_kernels as fvk

    shards, weight_map = _open_shards(args.checkpoint)
    if "model.language_model.layers.0.self_attn.q_proj.weight" in weight_map:
        layer_prefix = "model.language_model.layers"
    else:
        layer_prefix = "model.layers"

    specs = [
        LinearSpec("q_proj", "self_attn.q_proj"),
        LinearSpec("k_proj", "self_attn.k_proj"),
        LinearSpec("v_proj", "self_attn.v_proj"),
        LinearSpec("o_proj", "self_attn.o_proj"),
        LinearSpec("gate_proj", "mlp.gate_proj"),
        LinearSpec("up_proj", "mlp.up_proj"),
        LinearSpec("down_proj", "mlp.down_proj"),
    ]
    stream = torch.cuda.current_stream().cuda_stream
    print(f"device={torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"S={args.S} layer={args.layer} warmup={args.warmup} iters={args.iters}")
    print("GEMM-only rows match the SM120a blockscaled GEMM timing boundary: "
          "pre-quantized A + FP8 B + block scales -> BF16 D.")
    print("e2e rows add activation quantization before the GEMM.")
    total_descale_gemm = 0.0
    total_bs_gemm = 0.0
    total_descale_e2e = 0.0
    total_bs_e2e = 0.0
    total_m1_best = 0.0
    for spec in specs:
        base = f"{layer_prefix}.{args.layer}.{spec.key}"
        w = _tensor(shards, weight_map, base + ".weight").to(
            args.device).contiguous()
        ws = _tensor(shards, weight_map, base + ".weight_scale_inv").to(
            torch.float32).to(args.device).contiguous()
        N, K = w.shape
        x = torch.randn((args.S, K), dtype=torch.bfloat16, device=args.device)
        x_fp8 = torch.empty((args.S, K), dtype=torch.float8_e4m3fn,
                            device=args.device)
        x_scale = torch.empty((args.S, K // 128), dtype=torch.float32,
                              device=args.device)
        out = torch.empty((args.S, N), dtype=torch.bfloat16,
                          device=args.device)
        scratch_a = torch.empty((args.S, K), dtype=torch.bfloat16,
                                device=args.device)
        scratch_b = torch.empty((N, K), dtype=torch.bfloat16,
                                device=args.device)
        fvk.fp8_per_token_block128_quant_bf16(
            x.data_ptr(), x_fp8.data_ptr(), x_scale.data_ptr(),
            args.S, K, stream)
        torch.cuda.synchronize()

        def descale_gemm_call() -> None:
            fvk.fp8_block128_gemm_descale_bf16out(
                x_fp8.data_ptr(), w.data_ptr(), out.data_ptr(),
                args.S, N, K, x_scale.data_ptr(), ws.data_ptr(),
                scratch_a.data_ptr(), scratch_b.data_ptr(), stream)

        def descale_e2e_call() -> None:
            fvk.fp8_per_token_block128_quant_bf16(
                x.data_ptr(), x_fp8.data_ptr(), x_scale.data_ptr(),
                args.S, K, stream)
            fvk.fp8_block128_gemm_descale_bf16out(
                x_fp8.data_ptr(), w.data_ptr(), out.data_ptr(),
                args.S, N, K, x_scale.data_ptr(), ws.data_ptr(),
                scratch_a.data_ptr(), scratch_b.data_ptr(), stream)

        descale_gemm = _time_cuda(descale_gemm_call, args.warmup, args.iters)
        descale_e2e = _time_cuda(descale_e2e_call, args.warmup, args.iters)
        total_descale_gemm += descale_gemm[0]
        total_descale_e2e += descale_e2e[0]

        # Native Ada FP8 block-128 GEMM (reads FP8 weight directly, in-module).
        out_bs = torch.empty((args.S, N), dtype=torch.bfloat16,
                             device=args.device)

        def bs_gemm_call() -> None:
            fvk.fp8_block128_gemm_blockscaled_sm89_bf16out(
                x_fp8.data_ptr(), w.data_ptr(), out_bs.data_ptr(),
                args.S, N, K, x_scale.data_ptr(), ws.data_ptr(), stream)

        def bs_e2e_call() -> None:
            fvk.fp8_per_token_block128_quant_bf16(
                x.data_ptr(), x_fp8.data_ptr(), x_scale.data_ptr(),
                args.S, K, stream)
            fvk.fp8_block128_gemm_blockscaled_sm89_bf16out(
                x_fp8.data_ptr(), w.data_ptr(), out_bs.data_ptr(),
                args.S, N, K, x_scale.data_ptr(), ws.data_ptr(), stream)

        bs_gemm = _time_cuda(bs_gemm_call, args.warmup, args.iters)
        bs_e2e = _time_cuda(bs_e2e_call, args.warmup, args.iters)
        total_bs_gemm += bs_gemm[0]
        total_bs_e2e += bs_e2e[0]
        descale_gemm_call()
        bs_gemm_call()
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), out_bs.float().flatten(), dim=0).item()

        x1 = x[:1].contiguous()
        x1_fp8 = torch.empty((1, K), dtype=torch.float8_e4m3fn,
                             device=args.device)
        x1_scale = torch.empty((1, K // 128), dtype=torch.float32,
                               device=args.device)
        out1 = torch.empty((1, N), dtype=torch.bfloat16, device=args.device)
        fvk.fp8_per_token_block128_quant_bf16(
            x1.data_ptr(), x1_fp8.data_ptr(), x1_scale.data_ptr(),
            1, K, stream)
        torch.cuda.synchronize()
        m1_results = []
        for fn in (fvk.ht_gemv_fp8_block128_m1_w4,
                   fvk.ht_gemv_fp8_block128_m1_w8,
                   fvk.ht_gemv_fp8_block128_m1_w16):
            def m1_call(fn=fn) -> None:
                fn(x1_fp8.data_ptr(), w.data_ptr(), out1.data_ptr(),
                   1, N, K, x1_scale.data_ptr(), ws.data_ptr(), 1.0,
                   stream)
            m1_results.append(_time_cuda(m1_call, args.warmup, args.iters)[0])
        m1_best = min(m1_results)
        total_m1_best += m1_best * args.S
        print(f"{spec.name:10s} N,K={(N, K)} "
              f"descale_gemm {_fmt_triplet(descale_gemm)} "
              f"bs_sm89_gemm {_fmt_triplet(bs_gemm)} "
              f"gemm_speedup {descale_gemm[0] / bs_gemm[0]:5.2f}x "
              f"e2e_speedup {descale_e2e[0] / bs_e2e[0]:5.2f}x "
              f"cos {cos:.5f} "
              f"m1_best_xS {m1_best * args.S:8.4f}")

    print("--- one-layer linear total ---")
    print(f"descale_gemm_total_ms={total_descale_gemm:.4f}")
    print(f"bs_sm89_gemm_total_ms={total_bs_gemm:.4f}")
    print(f"gemm_speedup={total_descale_gemm / total_bs_gemm:.2f}x")
    print(f"descale_e2e_total_ms={total_descale_e2e:.4f}")
    print(f"bs_sm89_e2e_total_ms={total_bs_e2e:.4f}")
    print(f"e2e_speedup={total_descale_e2e / total_bs_e2e:.2f}x")
    print(f"m1_token_loop_estimate_ms={total_m1_best:.4f}")
    print(f"36_layer_descale_gemm_linear_ms={total_descale_gemm * 36:.2f}")
    print(f"36_layer_bs_sm89_gemm_linear_ms={total_bs_gemm * 36:.2f}")
    print(f"36_layer_descale_e2e_linear_ms={total_descale_e2e * 36:.2f}")
    print(f"36_layer_bs_sm89_e2e_linear_ms={total_bs_e2e * 36:.2f}")


if __name__ == "__main__":
    main()
