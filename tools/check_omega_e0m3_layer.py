#!/usr/bin/env python3
"""Single-layer cosine harness: Omega fake-quant vs. FlashRT E0M3 GEMM.

Quantifies the fidelity cost of migrating one Omega-QVLA dit_svdquant_v1
record (docs/omega_pack_e0m3.md) onto the FlashRT SM110 E0M3 path.

References (torch, fp32 accumulate):
  y_fp    = x2 @ W^T                    — no activation quant (ceiling;
                                          W is already fake-quant dequant)
  y_omega = fakequant(x2, s_t) @ W^T    — exact Omega consumer semantics
                                          (per-channel table, int [-8,7])

Variants:
  S0: A = e0m3(x2),        B = e0m3(W)              — drops the scale table
  S1: A = e0m3(x2 / s_t),  B = e0m3(W * diag(s_mean)) — mean-fold strategy

x2 is the rotated activation (perm + 64x64 block rotation), computed in
fp16 exactly like the Omega runtime. The output rotation is an exact
orthonormal transform applied to BOTH compared vectors, so it is skipped
(cosines are invariant to it).

Modes:
  --mode emulate : pure-torch E0M3 emulation (per-16 amax/7, UE4M3-rounded
                   scales, int [-7,7]). Runs anywhere; approximates the
                   tcgen05 result up to accumulation order and UE4M3
                   rounding corner cases. Use for local pre-checks.
  --mode kernel  : real kernels via flash_rt_fp4 (quantize + GEMM,
                   a_format=0). Requires CUDA + built extension (Thor).

Usage:
  python tools/check_omega_e0m3_layer.py \
      --pack packs_hf/pi05_long/quantized.pt --mode emulate
  python tools/check_omega_e0m3_layer.py \
      --pack packs_hf/pi05_long/quantized.pt --mode kernel \
      --layer paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj
"""

from __future__ import annotations

import argparse
import sys

import torch

DEFAULT_LAYER = ("paligemma_with_expert.gemma_expert.model.layers.0."
                 "self_attn.q_proj")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pack", required=True)
    p.add_argument("--layer", default=DEFAULT_LAYER)
    p.add_argument("--mode", choices=("emulate", "kernel"), default="emulate")
    p.add_argument("--tokens", type=int, default=256, help="M (rows)")
    p.add_argument("--step", type=int, default=0,
                   help="denoise step index into act_scale_table")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="",
                   help="torch device for reference/emulation math "
                        "(default: cuda in kernel mode, cpu in emulate mode)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# Omega consumer semantics (mirror gr00t/quantization/gptq_layers.py)
# ────────────────────────────────────────────────────────────────────
def apply_input_rotation(x: torch.Tensor, perm: torch.Tensor,
                         blocks: torch.Tensor) -> torch.Tensor:
    """x[N, in] fp16 -> bmm(x[:, perm].view(N, nb, B), blocks). fp16 in/out."""
    nb, b, _ = blocks.shape
    dev = x.device
    x2 = x.index_select(dim=-1, index=perm.to(dev))
    x2 = x2.reshape(-1, nb, b)
    x2 = torch.bmm(x2.transpose(0, 1).contiguous(),
                   blocks.to(device=dev, dtype=x.dtype))
    return x2.transpose(0, 1).contiguous().reshape(x.shape[0], nb * b)


def fake_quant_omega(x: torch.Tensor, scale: torch.Tensor,
                     bits: int = 4) -> torch.Tensor:
    """clamp(round(x/s), -2^(b-1), 2^(b-1)-1) * s — Omega's asymmetric grid."""
    qmax = 2 ** (bits - 1) - 1
    return (torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
            * scale).float()


# ────────────────────────────────────────────────────────────────────
# E0M3 emulation (approximates quantize_e0m3_dynamic_sfa_fp16)
# ────────────────────────────────────────────────────────────────────
def ue4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round positive tensor to the nearest UE4M3 value (E4M3 without sign,
    exp bias 7, 3 mantissa bits, subnormals below 2^-6). Approximation for
    emulation mode; the kernel's exact rounding may differ at bin edges."""
    x = x.clamp_min(1e-12)
    e = torch.floor(torch.log2(x))
    # normals: value = 2^E * (1 + M/8), M in 0..7
    base = torch.pow(2.0, e)
    m = torch.round(x / base - 1.0).clamp(0, 8)
    overflow = m == 8
    e = e + overflow.float()
    m = m * (~overflow).float()
    normal = torch.pow(2.0, e) * (1.0 + m / 8.0)
    # subnormals: step 2^-9
    sub = torch.round(x / 2 ** -9) * 2 ** -9
    return torch.where(x < 2 ** -6, sub, normal).clamp(max=480.0)


def e0m3_emulate(t: torch.Tensor) -> torch.Tensor:
    """Per-16 dynamic E0M3 fake-quant along the last dim. Returns fp32
    dequantized tensor with the same shape."""
    *lead, d = t.shape
    assert d % 16 == 0
    v = t.float().reshape(-1, d // 16, 16)
    scale = ue4m3_round(v.abs().amax(dim=-1, keepdim=True) / 7.0)
    # all-zero blocks round to scale 0; clamp to the smallest UE4M3
    # subnormal so emulation never divides by zero (kernel writes a real
    # scale here, exact corner behavior is hardware-specific)
    scale = scale.clamp_min(2 ** -9)
    q = torch.clamp(torch.round(v / scale), -7, 7)
    return (q * scale).reshape(*lead, d)


# ────────────────────────────────────────────────────────────────────
# Kernel path (requires flash_rt_fp4, i.e. Thor)
# ────────────────────────────────────────────────────────────────────
def e0m3_kernel_gemm(a_fp16: torch.Tensor, b_fp16: torch.Tensor,
                     fvk_fp4) -> torch.Tensor:
    """Quantize A[M,K] and B[N,K] with the real E0M3 kernels and run the
    tcgen05 block-scaled GEMM (a_format=0). Returns fp16 D[M, N]."""
    a_fp16 = a_fp16.contiguous()
    b_fp16 = b_fp16.contiguous()
    m, k = a_fp16.shape
    n, kb = b_fp16.shape
    assert k == kb and k % 16 == 0

    a_packed = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
    a_sfa = torch.zeros(fvk_fp4.sfa_size_bytes(m, k, False),
                        dtype=torch.uint8, device="cuda")
    rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
        a_fp16.data_ptr(), a_packed.data_ptr(), a_sfa.data_ptr(),
        m, k, False, 0)
    if rc != 0:
        raise RuntimeError(f"A quantize failed rc={rc}")

    b_packed = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
    b_sfb = torch.zeros(fvk_fp4.sfa_size_bytes(n, k, True),
                        dtype=torch.uint8, device="cuda")
    rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
        b_fp16.data_ptr(), b_packed.data_ptr(), b_sfb.data_ptr(),
        n, k, True, 0)
    if rc != 0:
        raise RuntimeError(f"B quantize failed rc={rc}")

    d = torch.empty(m, n, dtype=torch.float16, device="cuda")
    rc = fvk_fp4.cutlass_fp4_gemm_e0m3w(
        a_packed.data_ptr(), a_sfa.data_ptr(),
        b_packed.data_ptr(), b_sfb.data_ptr(), d.data_ptr(),
        m, n, k, 1.0, 0.0, 0, 0)
    if rc != 0:
        raise RuntimeError(f"cutlass_fp4_gemm_e0m3w failed rc={rc:#x}")
    torch.cuda.synchronize()
    return d


# ────────────────────────────────────────────────────────────────────
def cosine_stats(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """(global cos, per-row cos mean, per-row cos min), fp32 inputs."""
    a = a.float()
    b = b.float()
    glob = torch.dot(a.flatten(), b.flatten()) / (
        a.norm() * b.norm()).item()
    per = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return glob, per.mean().item(), per.min().item()


def report(tag: str, a: torch.Tensor, b: torch.Tensor) -> None:
    g, mean, mn = cosine_stats(a, b)
    print(f"  {tag:<28} global {g:.6f}   per-token mean {mean:.6f}   "
          f"min {mn:.6f}")


def main() -> int:
    args = parse_args()
    pack = torch.load(args.pack, map_location="cpu", weights_only=True)
    if args.layer not in pack:
        print(f"error: layer '{args.layer}' not in pack", file=sys.stderr)
        return 2
    rec = pack[args.layer]
    table = rec["act_scale_table"].float()
    if not 0 <= args.step < table.shape[0]:
        print(f"error: --step {args.step} out of range "
              f"(table has {table.shape[0]} steps)", file=sys.stderr)
        return 2
    s_t = table[args.step]
    s_mean = table.mean(dim=0)

    w = rec["weight_res_q"].float()  # (out, in), rotated+permuted domain
    out_f, in_f = w.shape
    print(f"layer: {args.layer}  N(out)={out_f} K(in)={in_f}  "
          f"table=({table.shape[0]},{table.shape[1]}) step={args.step}")

    # Synthetic activations with realistic per-channel heterogeneity:
    # lognormal gains plus a few strong outlier channels (DuQuant's target).
    g = torch.Generator().manual_seed(args.seed)
    gains = torch.exp(torch.randn(in_f, generator=g))
    outlier_idx = torch.randperm(in_f, generator=g)[: in_f // 128 + 1]
    gains[outlier_idx] *= 10.0
    x = (torch.randn(args.tokens, in_f, generator=g) * gains).half()

    dev = args.device or ("cuda" if args.mode == "kernel" else "cpu")
    x = x.to(dev)
    x2 = apply_input_rotation(
        x, rec["duquant_rotation_perm"],
        rec["duquant_rotation_blocks"].to(dev))
    w = w.to(dev)
    s_t = s_t.to(dev)
    s_mean = s_mean.to(dev)

    # Calibrate synthetic activations to the pack's scale table: the table
    # is q99.9(|x2|)/7 on real (rotated) activations, so rescale synthetic
    # x2 per channel to match. Without this the fake-quant clips almost
    # everything and the comparison measures clipping, not format migration.
    q999 = torch.quantile(x2.abs().float(), 0.999, dim=0).clamp_min(1e-8)
    x2 = (x2.float() * (7.0 * s_t / q999)).half()

    # References (fp32 accumulate).
    y_fp = x2.float() @ w.t()
    y_omega = fake_quant_omega(x2.float(), s_t) @ w.t()
    print("\nreferences:")
    report("omega vs fp (own quant cost)", y_omega, y_fp)

    if args.mode == "emulate":
        y_s0 = e0m3_emulate(x2) @ e0m3_emulate(w).t()
        y_s1 = (e0m3_emulate((x2.float() / s_t).half().float())
                @ e0m3_emulate(w * s_mean).t())
    else:
        if not torch.cuda.is_available():
            print("error: --mode kernel requires CUDA", file=sys.stderr)
            return 2
        try:
            import flash_rt.flash_rt_fp4 as fvk_fp4
        except ImportError:
            print("error: flash_rt_fp4 not importable — run on Thor",
                  file=sys.stderr)
            return 2
        y_s0 = e0m3_kernel_gemm(x2, w.half(), fvk_fp4).float()
        y_s1 = e0m3_kernel_gemm((x2.float() / s_t).half(),
                                (w * s_mean).half(), fvk_fp4).float()

    print(f"\nvariants vs references (mode={args.mode}):")
    report("S0 (drop table) vs omega", y_s0, y_omega)
    report("S1 (mean fold) vs omega", y_s1, y_omega)
    report("S0 vs fp", y_s0, y_fp)
    report("S1 vs fp", y_s1, y_fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
