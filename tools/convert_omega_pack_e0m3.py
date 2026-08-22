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
import re
import sys
import time

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
    return p.parse_args()


def main() -> int:
    args = parse_args()

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

    pack = torch.load(args.pack, map_location="cpu", weights_only=True)
    meta = pack.get("__meta__", {})
    names = sorted(k for k in pack if k != "__meta__")
    if args.layer_regex:
        rx = re.compile(args.layer_regex)
        names = [n for n in names if rx.search(n)]
    if not names:
        print("error: no layers matched", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    weights: dict = {}
    aux: dict = {}
    t0 = time.time()
    for i, name in enumerate(names):
        rec = pack[name]
        if rec.get("format") != "dit_svdquant_v1":
            print(f"skip {name}: format={rec.get('format')!r}")
            continue
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
        n, k = w.shape
        if k % 16 != 0:
            print(f"skip {name}: K={k} not divisible by 16")
            continue

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
    out = {
        "format": OUTPUT_FORMAT,
        "source_pack_meta": meta,
        "fold": args.fold,
        "weights": weights,
        "aux": aux,
    }
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(weights)} layers, fold={args.fold}, "
          f"{time.time() - t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
