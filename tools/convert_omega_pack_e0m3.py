#!/usr/bin/env python3
r"""Offline converter: Omega-QVLA dit_svdquant_v1 pack -> FlashRT E0M3 weights.

Reads an Omega-QVLA quantized pack (see docs/omega_pack_e0m3.md for the
record schema) and re-quantizes every `weight_res_q` tensor into the
FlashRT SM110 E0M3 operand format: packed 4-bit elements [N, K/2] plus
tile-interleaved UE4M3 SFB scales (per-16, amax/7), via the
`quantize_e0m3_dynamic_sfa_fp16` kernel. No GPTQ bitstream decoding is
needed — the pack stores weights as dequantized fp16.

Scale-fold strategies (--fold):
  none : B = e0m3(W). The activation-side per-channel calibration table is
         not represented anywhere (strategy S0 in the doc).
  mean : B = e0m3(W * diag(s_mean)), s_mean = act_scale_table.mean(dim=0).
         The runtime must then divide activations by s_t per step before
         quantization (strategy S1). Exact for the mean step; residual is
         the table's step-to-step spread. BROKEN on hardware: raw s_mean
         (~1e-2) shrinks weight columns, pressing per-16 block scales
         below the UE4M3 subnormal floor (2^-9). Kept for reference.
  actnorm : floor-safe S1. Decompose s_mean = c * r with c = geomean
         (per-layer scalar) and r = s_mean / c (geomean 1, O(1) entries):
         weights fold r (magnitudes preserved, no floor issue),
         activations are divided by s_mean at runtime (static — no
         per-step dispatch), and c is absorbed into the GEMM alpha.
         Identity: (x/s_mean) @ (W*r)^T * c == x @ W^T.

Auxiliary tensors needed by a runtime consumer (input/output rotations,
permutation, scale tables) are copied through unchanged into the output.

Requires: CUDA + the compiled flash_rt_fp4 extension (i.e. run on Thor;
the quantize kernels are plain CUDA but the GEMM they feed is SM110).

Usage:
  python tools/convert_omega_pack_e0m3.py \
      --pack /path/to/Omega-QVLA/packs_hf/pi05_long/quantized.pt \
      --out pi05_long_e0m3.pt --fold none
  # subset for bring-up:
  python tools/convert_omega_pack_e0m3.py --pack ... --out /tmp/one.pt \
      --fold none --layer-regex 'layers\.0\.self_attn\.q_proj'
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import torch

OUTPUT_FORMAT = "omega_e0m3_v1"

# Record fields copied verbatim into the output's per-layer aux entry.
AUX_TENSORS = (
    "duquant_rotation_blocks",
    "duquant_rotation_perm",
    "duquant_rotation_out_blocks",
    "act_scale_table",
)
AUX_SCALARS = ("weight_bits", "a_bits", "in_features", "out_features", "rank")
EXPECTED_FULL_PACK_RECORDS = 252


def _tensor(rec: Mapping, name: str, field: str) -> torch.Tensor:
    value = rec.get(field)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name}: {field} must be a tensor")
    return value


def validate_record(name: str, rec: Mapping) -> tuple[int, int]:
    """Validate the complete rank-0 ``dit_svdquant_v1`` record contract."""
    if not isinstance(rec, Mapping):
        raise ValueError(f"{name}: record must be a mapping")
    if rec.get("format") != "dit_svdquant_v1":
        raise ValueError(
            f"{name}: unsupported format {rec.get('format')!r}; "
            "expected 'dit_svdquant_v1'")
    if rec.get("rank") != 0:
        raise ValueError(f"{name}: only rank=0 is supported, got {rec.get('rank')!r}")
    if rec.get("weight_bits") != 4 or rec.get("a_bits") != 4:
        raise ValueError(
            f"{name}: expected weight_bits=a_bits=4, got "
            f"{rec.get('weight_bits')!r}/{rec.get('a_bits')!r}")

    weight = _tensor(rec, name, "weight_res_q")
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError(f"{name}: weight_res_q must be a floating [N,K] tensor")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"{name}: weight_res_q contains NaN or Inf")
    n, k = weight.shape
    if n <= 0 or k <= 0 or n % 64 or k % 64:
        raise ValueError(
            f"{name}: N and K must be positive multiples of 64, got N={n}, K={k}")
    if rec.get("out_features") != n or rec.get("in_features") != k:
        raise ValueError(
            f"{name}: metadata shape mismatch: tensor=({n},{k}), "
            f"out_features/in_features={rec.get('out_features')!r}/"
            f"{rec.get('in_features')!r}")

    lowrank_a = _tensor(rec, name, "lowrank_A")
    lowrank_b = _tensor(rec, name, "lowrank_B")
    if tuple(lowrank_a.shape) != (n, 0) or tuple(lowrank_b.shape) != (k, 0):
        raise ValueError(
            f"{name}: rank-0 lowrank shapes must be ({n},0)/({k},0), got "
            f"{tuple(lowrank_a.shape)}/{tuple(lowrank_b.shape)}")

    table = _tensor(rec, name, "act_scale_table")
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != k:
        raise ValueError(
            f"{name}: act_scale_table must have shape [steps,{k}], got "
            f"{tuple(table.shape)}")
    if not table.is_floating_point() or not bool(torch.isfinite(table).all()) \
            or not bool((table > 0).all()):
        raise ValueError(f"{name}: act_scale_table must contain finite positive values")

    rotation_in = _tensor(rec, name, "duquant_rotation_blocks")
    rotation_out = _tensor(rec, name, "duquant_rotation_out_blocks")
    if tuple(rotation_in.shape) != (k // 64, 64, 64):
        raise ValueError(
            f"{name}: input rotation shape must be ({k // 64},64,64), got "
            f"{tuple(rotation_in.shape)}")
    if tuple(rotation_out.shape) != (n // 64, 64, 64):
        raise ValueError(
            f"{name}: output rotation shape must be ({n // 64},64,64), got "
            f"{tuple(rotation_out.shape)}")
    if not bool(torch.isfinite(rotation_in).all()) \
            or not bool(torch.isfinite(rotation_out).all()):
        raise ValueError(f"{name}: rotation tensors contain NaN or Inf")

    perm = _tensor(rec, name, "duquant_rotation_perm")
    if perm.dtype != torch.int64 or tuple(perm.shape) != (k,):
        raise ValueError(
            f"{name}: permutation must be int64 shape ({k},), got "
            f"{perm.dtype} {tuple(perm.shape)}")
    if not torch.equal(torch.sort(perm).values, torch.arange(k, dtype=torch.int64)):
        raise ValueError(f"{name}: duquant_rotation_perm is not a permutation of [0,{k})")
    return n, k


def expected_record_count(meta: Mapping, *, layer_regex: str,
                          selected_count: int,
                          explicit: int | None) -> int:
    if explicit is not None:
        if explicit < 1:
            raise ValueError("--expected-records must be positive")
        return explicit
    recipe = meta.get("recipe", "")
    if layer_regex or (isinstance(recipe, str)
                       and recipe.startswith("synthetic fixture")):
        return selected_count
    return EXPECTED_FULL_PACK_RECORDS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pack", required=True, help="input Omega quantized.pt")
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--fold", choices=("none", "mean", "actnorm"),
                   default="none",
                   help="scale-table fold strategy (default: none = S0; "
                        "mean/actnorm are ablation-only, see docstring)")
    p.add_argument("--layer-regex", default="",
                   help="only convert layers matching this regex")
    p.add_argument("--keep-fp16", action="store_true",
                   help="also store the (possibly folded) fp16 weight, "
                        "for offline reference checks")
    p.add_argument("--expected-records", type=int,
                   help="required converted record count; defaults to 252 for "
                        "a full pack and to the selected count for fixtures/subsets")
    p.add_argument("--validate-only", action="store_true",
                   help="validate schema and record coverage without CUDA")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.validate_only \
            and Path(args.pack).resolve() == Path(args.out).resolve():
        print("error: --out must not overwrite the source pack", file=sys.stderr)
        return 2

    pack = torch.load(args.pack, map_location="cpu", weights_only=True)
    if not isinstance(pack, Mapping):
        print("error: pack must be a mapping", file=sys.stderr)
        return 2
    meta = pack.get("__meta__", {})
    if not isinstance(meta, Mapping):
        print("error: __meta__ must be a mapping", file=sys.stderr)
        return 2
    names = sorted(k for k in pack if k != "__meta__")
    if args.layer_regex:
        rx = re.compile(args.layer_regex)
        names = [n for n in names if rx.search(n)]
    if not names:
        print("error: no layers matched", file=sys.stderr)
        return 2

    try:
        expected = expected_record_count(
            meta, layer_regex=args.layer_regex, selected_count=len(names),
            explicit=args.expected_records)
        if len(names) != expected:
            raise ValueError(
                f"record coverage mismatch: selected {len(names)}, expected {expected}")
        shapes = {name: validate_record(name, pack[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(names)}/{expected} records")
    if args.validate_only:
        return 0

    if not torch.cuda.is_available():
        print("error: CUDA is required (quantize kernels run on GPU)",
              file=sys.stderr)
        return 2
    try:
        import flash_rt.flash_rt_fp4 as fvk_fp4
    except ImportError:
        print("error: flash_rt_fp4 extension not importable — run this on a "
              "machine with FlashRT built (Thor)", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    weights: dict = {}
    aux: dict = {}
    t0 = time.time()
    for i, name in enumerate(names):
        rec = pack[name]
        w = rec["weight_res_q"].to(device=device, dtype=torch.float16,
                                   non_blocking=False).contiguous()
        table = rec["act_scale_table"].float()
        act_out_scale = None
        if args.fold == "mean":
            s_mean = table.mean(dim=0)  # (in,)
            w = (w * s_mean.to(device=device, dtype=torch.float16)
                   .unsqueeze(0)).contiguous()
        elif args.fold == "actnorm":
            s_mean = table.mean(dim=0).clamp_min(1e-12)  # (in,)
            c = float(torch.exp(torch.log(s_mean).mean()))
            r = s_mean / c                      # geomean 1, O(1) entries
            w = (w * r.to(device=device, dtype=torch.float16)
                   .unsqueeze(0)).contiguous()
            act_out_scale = c
        n, k = shapes[name]

        packed = torch.empty(n, k // 2, dtype=torch.uint8, device=device)
        # Zero-init: tile-interleaved SFB pads K to 64-element atoms and the
        # kernel never writes padding entries (see fp4_utils.py).
        sfb = torch.zeros(fvk_fp4.sfa_size_bytes(n, k, True),
                          dtype=torch.uint8, device=device)
        rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
            w.data_ptr(), packed.data_ptr(), sfb.data_ptr(), n, k, True, 0)
        if rc != 0:
            raise RuntimeError(f"quantize_e0m3_dynamic_sfa_fp16 failed on "
                               f"{name}: rc={rc}")

        entry = {"packed": packed.cpu(), "sfb": sfb.cpu(), "N": n, "K": k}
        if args.keep_fp16:
            entry["weight_fp16_folded"] = w.cpu()
        weights[name] = entry

        aux_entry = {f: rec[f].clone() for f in AUX_TENSORS if f in rec}
        aux_entry.update({f: rec[f] for f in AUX_SCALARS if f in rec})
        aux_entry["fold"] = args.fold
        if args.fold == "actnorm":
            # Consumer contract: divide activations by act_scale_static
            # (post-rotation, pre-quantize) and pass act_out_scale as the
            # GEMM alpha. See the --fold actnorm note in the docstring.
            aux_entry["act_scale_static"] = s_mean.clone()
            aux_entry["act_out_scale"] = act_out_scale
        aux[name] = aux_entry

        if (i + 1) % 21 == 0 or i + 1 == len(names):
            print(f"[{i + 1}/{len(names)}] {name}  N={n} K={k}  "
                  f"({time.time() - t0:.1f}s)")

    torch.cuda.synchronize()
    if set(weights) != set(names) or set(aux) != set(names):
        raise RuntimeError(
            "internal coverage error: not all validated records were converted")
    out = {
        "format": OUTPUT_FORMAT,
        "schema_version": 1,
        "source_pack_meta": meta,
        "source_record_count": len(pack) - int("__meta__" in pack),
        "selected_record_count": len(names),
        "selected_layers": names,
        "fold": args.fold,
        "weights": weights,
        "aux": aux,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=out_path.name + ".", suffix=".tmp",
            dir=out_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(out, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"wrote {args.out}: {len(weights)} layers, fold={args.fold}, "
          f"{time.time() - t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
