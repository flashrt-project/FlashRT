#!/usr/bin/env python3
"""Verify FP4 Gate+Up substitution at Chameleon layer-16 FFN shape.

Compares three paths driven by identical fp16 weights/activation:

    REF : pure fp16 matmul + silu*mul + matmul                (fp32 accumulate)
    FP8 : full FP8 path (used by current chameleon_forward)
    MIX : FP4 Gate+Up + (existing) silu_mul_split_fp8_fp16 + FP8 Down
    ALL4: FP4 Gate+Up + fp16 silu*mul + FP4 Down              (upper bound)

For each path: cosine similarity vs REF + microbenchmark latency.
"""
import pytest

torch = pytest.importorskip("torch")
fp4 = pytest.importorskip(
    "flash_rt.flash_rt_fp4",
    reason="flash_rt_fp4 requires an NVFP4 (sm_120+) build")
import numpy as np
import flash_rt.flash_rt_kernels as fvk
from flash_rt.executors.fp4_utils import (
    quant_weight_nvfp4, FP4ActScratch, quant_act_nvfp4, fp4_gemm, pick_variant,
)


def fp16_t(*shape, scale=1.0):
    return (torch.randn(*shape, dtype=torch.float16, device='cuda') * scale).contiguous()


def cuda_time(fn, iters=100, warmup=20):
    s = torch.cuda.current_stream()
    for _ in range(warmup): fn()
    s.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); s.synchronize()
    return e0.elapsed_time(e1) / iters * 1000   # μs


def amax_scale(t: torch.Tensor) -> float:
    return max(t.abs().max().item() / 448.0, 1e-9)


def make_scale_buf(scale: float) -> torch.Tensor:
    return torch.tensor([scale], dtype=torch.float32, device='cuda')


def quant_fp8(W: torch.Tensor, scale: float):
    out = torch.empty_like(W, dtype=torch.uint8)
    sb = make_scale_buf(scale)
    fvk.quantize_fp8_static_fp16(W.data_ptr(), out.data_ptr(),
                                  sb.data_ptr(), W.numel(), 0)
    return out, sb


def cos_vs(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0)).item()


def main():
    print(f"FP4 enabled: {fp4.has_nvfp4()}; variants: {fp4.cutlass_fp4_gemm_num_variants()}")

    Se, D, Dff = 1216, 4096, 11008

    torch.manual_seed(0)
    W_g = fp16_t(Dff, D, scale=0.02)
    W_u = fp16_t(Dff, D, scale=0.02)
    W_d = fp16_t(D,   Dff, scale=0.02)
    X   = fp16_t(Se, D, scale=1.0)

    # ---- REF ----
    gate_ref = (X.float() @ W_g.float().T).half()
    up_ref   = (X.float() @ W_u.float().T).half()
    h_ref    = (torch.nn.functional.silu(gate_ref.float()) * up_ref.float()).half()
    out_ref  = (h_ref.float() @ W_d.float().T).half()
    print(f"REF: |gate|max={gate_ref.abs().max():.2f}  |up|max={up_ref.abs().max():.2f}"
          f"  |h|max={h_ref.abs().max():.2f}  |out|max={out_ref.abs().max():.2f}")

    # ---- Pre-compute calibrated scales (per-tensor amax/448) ----
    s_x   = amax_scale(X)
    s_wg  = amax_scale(W_g)
    s_wu  = amax_scale(W_u)
    s_wd  = amax_scale(W_d)
    s_h   = amax_scale(h_ref)        # post-silu*up → fp8 input to Down
    print(f"scales:  x={s_x:.3e}  w_g={s_wg:.3e}  w_u={s_wu:.3e}  w_d={s_wd:.3e}  h={s_h:.3e}")

    gemm = fvk.GemmRunner()

    # FP8 weights + activation
    # NB: fp8_nn_dev is NN (no transpose), so B must be [K, N] row-major.
    # We store HF-style W as [N, K]; transpose before fp8 quant.
    Wg_fp8, sg = quant_fp8(W_g.t().contiguous(), s_wg)   # [D, Dff]
    Wu_fp8, su = quant_fp8(W_u.t().contiguous(), s_wu)   # [D, Dff]
    Wd_fp8, sd = quant_fp8(W_d.t().contiguous(), s_wd)   # [Dff, D]
    sx_buf = make_scale_buf(s_x);  sh_buf = make_scale_buf(s_h)
    X_fp8 = torch.empty(Se, D, dtype=torch.uint8, device='cuda')
    fvk.quantize_fp8_static_fp16(X.data_ptr(), X_fp8.data_ptr(),
                                  sx_buf.data_ptr(), Se*D, 0)

    gate_out = torch.empty(Se, Dff, dtype=torch.float16, device='cuda')
    up_out   = torch.empty(Se, Dff, dtype=torch.float16, device='cuda')
    gu_fp8   = torch.empty(Se, Dff, dtype=torch.uint8,   device='cuda')
    out_fp8  = torch.empty(Se, D,   dtype=torch.float16, device='cuda')

    def run_fp8():
        gemm.fp8_nn_dev(X_fp8.data_ptr(), Wg_fp8.data_ptr(), gate_out.data_ptr(),
                        Se, Dff, D, sx_buf.data_ptr(), sg.data_ptr(), 0)
        gemm.fp8_nn_dev(X_fp8.data_ptr(), Wu_fp8.data_ptr(), up_out.data_ptr(),
                        Se, Dff, D, sx_buf.data_ptr(), su.data_ptr(), 0)
        fvk.silu_mul_split_fp8_fp16(gate_out.data_ptr(), up_out.data_ptr(),
                                     gu_fp8.data_ptr(), Se*Dff,
                                     sh_buf.data_ptr(), 0)
        gemm.fp8_nn_dev(gu_fp8.data_ptr(), Wd_fp8.data_ptr(), out_fp8.data_ptr(),
                        Se, D, Dff, sh_buf.data_ptr(), sd.data_ptr(), 0)

    run_fp8(); torch.cuda.synchronize()
    cos_fp8 = cos_vs(out_fp8, out_ref)
    fp8_us = cuda_time(run_fp8)

    # ---- MIX (FP4 Gate+Up, FP8 Down) ----
    qg = quant_weight_nvfp4(W_g)
    qu = quant_weight_nvfp4(W_u)
    sc_x = FP4ActScratch(max_M=Se, K=D)
    var_gu = pick_variant(Dff, D)
    out_mix = torch.empty(Se, D, dtype=torch.float16, device='cuda')

    def run_mix():
        quant_act_nvfp4(X, sc_x, Se, stream=0)
        fp4_gemm(sc_x, qg, gate_out, Se, Dff, D, variant_idx=var_gu, stream=0)
        fp4_gemm(sc_x, qu, up_out,   Se, Dff, D, variant_idx=var_gu, stream=0)
        fvk.silu_mul_split_fp8_fp16(gate_out.data_ptr(), up_out.data_ptr(),
                                     gu_fp8.data_ptr(), Se*Dff,
                                     sh_buf.data_ptr(), 0)
        gemm.fp8_nn_dev(gu_fp8.data_ptr(), Wd_fp8.data_ptr(), out_mix.data_ptr(),
                        Se, D, Dff, sh_buf.data_ptr(), sd.data_ptr(), 0)

    run_mix(); torch.cuda.synchronize()
    cos_mix = cos_vs(out_mix, out_ref)
    mix_us = cuda_time(run_mix)

    # ---- ALL-FP4 (Gate+Up+Down all FP4, fp16 silu*mul) ----
    qd = quant_weight_nvfp4(W_d)
    sc_h = FP4ActScratch(max_M=Se, K=Dff)
    var_dn = pick_variant(D, Dff)
    h_buf = torch.empty(Se, Dff, dtype=torch.float16, device='cuda')
    out_all4 = torch.empty(Se, D, dtype=torch.float16, device='cuda')

    def run_all4():
        quant_act_nvfp4(X, sc_x, Se, stream=0)
        fp4_gemm(sc_x, qg, gate_out, Se, Dff, D, variant_idx=var_gu, stream=0)
        fp4_gemm(sc_x, qu, up_out,   Se, Dff, D, variant_idx=var_gu, stream=0)
        # fp16 silu*up via torch (bench-only)
        torch.mul(torch.nn.functional.silu(gate_out), up_out, out=h_buf)
        quant_act_nvfp4(h_buf, sc_h, Se, stream=0)
        fp4_gemm(sc_h, qd, out_all4, Se, D, Dff, variant_idx=var_dn, stream=0)

    run_all4(); torch.cuda.synchronize()
    cos_all4 = cos_vs(out_all4, out_ref)
    all4_us = cuda_time(run_all4)

    print()
    print("="*72)
    print("Chameleon layer-16 FFN block (Se=1216, D=4096, Dff=11008)")
    print("="*72)
    fmt = "  {:14s}  cos_vs_ref = {:.6f}    {:7.1f} μs    speedup={}"
    print(fmt.format("FP8 baseline",   cos_fp8,  fp8_us,  "1.00x"))
    print(fmt.format("MIX (FP4 GU)",   cos_mix,  mix_us,  f"{fp8_us/mix_us:.2f}x"))
    print(fmt.format("ALL-FP4",        cos_all4, all4_us, f"{fp8_us/all4_us:.2f}x"))
    delta = fp8_us - mix_us
    print(f"\n  Per-layer MIX saves {delta:6.1f} μs → 32 layers ≈ {delta*32/1000:5.2f} ms")


if __name__ == '__main__':
    main()
