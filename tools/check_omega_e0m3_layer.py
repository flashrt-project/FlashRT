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
  python tools/check_omega_e0m3_layer.py \
      --pack /tmp/fixture_pack.pt --artifact /tmp/fixture_e0m3.pt \
      --mode kernel --min-artifact-cos 0.98
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

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
    p.add_argument("--artifact",
                   help="converted omega_e0m3_v1 artifact; in kernel mode, "
                        "use its packed/SFB weight directly instead of re-quantizing")
    p.add_argument("--min-artifact-cos", type=float, default=0.98,
                   help="minimum artifact-vs-fp16 global cosine (default: 0.98)")
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


def validate_artifact(artifact: Mapping, layer: str, n: int, k: int,
                      fvk_fp4) -> tuple[Mapping, Mapping]:
    if not isinstance(artifact, Mapping) or artifact.get("format") != "omega_e0m3_v1":
        raise ValueError("artifact format must be 'omega_e0m3_v1'")
    if artifact.get("schema_version") != 1:
        raise ValueError(
            f"artifact schema_version must be 1, got {artifact.get('schema_version')!r}")
    weights = artifact.get("weights")
    aux = artifact.get("aux")
    if not isinstance(weights, Mapping) or not isinstance(aux, Mapping):
        raise ValueError("artifact weights and aux must be mappings")
    selected = artifact.get("selected_layers")
    count = artifact.get("selected_record_count")
    if not isinstance(selected, list) or count != len(selected):
        raise ValueError("artifact selected layer metadata is inconsistent")
    if set(selected) != set(weights) or set(selected) != set(aux):
        raise ValueError("artifact weights/aux do not cover every selected layer")
    if layer not in weights or layer not in aux:
        raise ValueError(f"artifact does not contain layer {layer!r}")
    entry = weights[layer]
    aux_entry = aux[layer]
    if not isinstance(entry, Mapping) or not isinstance(aux_entry, Mapping):
        raise ValueError(f"artifact layer {layer!r} entries must be mappings")
    if entry.get("N") != n or entry.get("K") != k:
        raise ValueError(
            f"artifact layer shape mismatch: expected N={n}, K={k}, got "
            f"N={entry.get('N')!r}, K={entry.get('K')!r}")
    packed = entry.get("packed")
    sfb = entry.get("sfb")
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8 \
            or tuple(packed.shape) != (n, k // 2):
        raise ValueError(
            f"artifact packed must be uint8 shape ({n},{k // 2})")
    expected_sfb = fvk_fp4.sfa_size_bytes(n, k, True)
    if not isinstance(sfb, torch.Tensor) or sfb.dtype != torch.uint8 \
            or sfb.numel() != expected_sfb:
        raise ValueError(
            f"artifact sfb must be uint8 with {expected_sfb} bytes")
    fold = artifact.get("fold")
    if fold not in {"none", "mean", "actnorm"} or aux_entry.get("fold") != fold:
        raise ValueError("artifact fold metadata is missing or inconsistent")
    return entry, aux_entry


def e0m3_artifact_gemm(a_fp16: torch.Tensor, entry: Mapping,
                       fvk_fp4, *, alpha: float = 1.0) -> torch.Tensor:
    """Quantize A, then consume artifact packed/SFB without re-quantizing B."""
    a_fp16 = a_fp16.contiguous()
    m, k = a_fp16.shape
    n = int(entry["N"])
    a_packed = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
    a_sfa = torch.zeros(fvk_fp4.sfa_size_bytes(m, k, False),
                        dtype=torch.uint8, device="cuda")
    rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
        a_fp16.data_ptr(), a_packed.data_ptr(), a_sfa.data_ptr(),
        m, k, False, 0)
    if rc != 0:
        raise RuntimeError(f"A quantize failed rc={rc}")
    b_packed = entry["packed"].to(device="cuda", non_blocking=False).contiguous()
    b_sfb = entry["sfb"].to(device="cuda", non_blocking=False).contiguous()
    d = torch.empty(m, n, dtype=torch.float16, device="cuda")
    rc = fvk_fp4.cutlass_fp4_gemm_e0m3w(
        a_packed.data_ptr(), a_sfa.data_ptr(),
        b_packed.data_ptr(), b_sfb.data_ptr(), d.data_ptr(),
        m, n, k, alpha, 0.0, 0, 0)
    if rc != 0:
        raise RuntimeError(f"artifact cutlass_fp4_gemm_e0m3w failed rc={rc:#x}")
    torch.cuda.synchronize()
    return d


# ────────────────────────────────────────────────────────────────────
def cosine_stats(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """(global cos, per-row cos mean, per-row cos min), fp32 inputs."""
    a = a.float()
    b = b.float()
    glob = (torch.dot(a.flatten(), b.flatten()) / (
        a.norm() * b.norm())).item()
    per = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return glob, per.mean().item(), per.min().item()


def report(tag: str, a: torch.Tensor, b: torch.Tensor) -> None:
    g, mean, mn = cosine_stats(a, b)
    print(f"  {tag:<28} global {g:.6f}   per-token mean {mean:.6f}   "
          f"min {mn:.6f}")


def main() -> int:
    args = parse_args()
    if args.artifact and args.mode != "kernel":
        print("error: --artifact requires --mode kernel", file=sys.stderr)
        return 2
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
        if args.artifact:
            artifact = torch.load(
                args.artifact, map_location="cpu", weights_only=True)
            try:
                entry, aux_entry = validate_artifact(
                    artifact, args.layer, out_f, in_f, fvk_fp4)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            fold = artifact["fold"]
            if fold == "none":
                artifact_input, alpha = x2, 1.0
            elif fold == "mean":
                artifact_input, alpha = (x2.float() / s_t).half(), 1.0
            else:
                static = aux_entry.get("act_scale_static")
                alpha = aux_entry.get("act_out_scale")
                if not isinstance(static, torch.Tensor) \
                        or tuple(static.shape) != (in_f,) or not isinstance(alpha, float):
                    print("error: invalid actnorm metadata", file=sys.stderr)
                    return 2
                artifact_input = (x2.float() / static.to(x2.device)).half()
            y_artifact = e0m3_artifact_gemm(
                artifact_input, entry, fvk_fp4, alpha=alpha).float()
            artifact_cos = cosine_stats(y_artifact, y_fp)[0]
            print("\nconverted artifact round-trip:")
            report(f"artifact ({fold}) vs omega", y_artifact, y_omega)
            report(f"artifact ({fold}) vs fp", y_artifact, y_fp)
            if artifact_cos < args.min_artifact_cos:
                print(
                    f"error: artifact cosine {artifact_cos:.6f} is below "
                    f"{args.min_artifact_cos:.6f}", file=sys.stderr)
                return 1
            return 0
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
