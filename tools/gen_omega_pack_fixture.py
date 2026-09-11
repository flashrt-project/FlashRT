#!/usr/bin/env python3
"""Generate a synthetic miniature dit_svdquant_v1 pack (fixture).

Lets anyone exercise the converter + consumer round-trip WITHOUT the
real 4.8GB Omega-QVLA pack:

    python tools/gen_omega_pack_fixture.py --out /tmp/fixture_pack.pt
    python tools/convert_omega_pack_e0m3.py --pack /tmp/fixture_pack.pt \
        --out /tmp/fixture_e0m3.pt --fold none --keep-fp16        # Thor
    python tools/check_omega_e0m3_layer.py --pack /tmp/fixture_pack.pt \
        --artifact /tmp/fixture_e0m3.pt --mode kernel             # Thor

Design notes (what makes the fixture a real test and not a toy):

- Schema is field-for-field identical to the real pack (checked against
  packs_hf/pi05_long/quantized.pt): format tag, fp16 `weight_res_q`,
  empty rank-0 lowrank tensors, fp16 64x64 rotation blocks, int64 perm,
  float32 act_scale_table (expert style: T=10 rows; paligemma style:
  T=1), plus the calibration metadata scalars.
- Rotations are RANDOM ORTHOGONAL (QR of gaussian), not identity —
  otherwise the consumer's bmm rotate path would pass trivially.
- Weights carry per-input-channel lognormal scale outliers, the regime
  that stresses E0M3 per-16 quantization and the UE4M3 subnormal floor
  (the S1 failure mode in docs/omega_pack_e0m3.md).
- Shapes mix one full-size realistic layer (16384x2048) with small ones
  (256x1024) to cover kernel edge sizes; all N,K are multiples of 64 as
  the 64x64 rotation blocks require.

Pure CPU, deterministic under --seed. Output ~85MB with defaults.
"""

from __future__ import annotations

import argparse

import torch

# (name, N=out, K=in, table_rows) — names/shapes mirror the real pack.
LAYERS = [
    ("paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
     2048, 1024, 10),
    ("paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj",
     256, 1024, 10),
    ("paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj",
     1024, 4096, 10),
    ("paligemma_with_expert.paligemma.model.language_model.layers.0."
     "mlp.gate_proj", 16384, 2048, 1),
]


def _orthogonal_blocks(n_blocks: int, g: torch.Generator) -> torch.Tensor:
    """(nb, 64, 64) random orthogonal blocks, fp16 (as the real pack)."""
    a = torch.randn(n_blocks, 64, 64, generator=g)
    q, r = torch.linalg.qr(a)
    # QR sign convention: make diag(R) positive so Q is uniform-ish.
    sign = torch.sign(torch.diagonal(r, dim1=1, dim2=2))
    q = q * sign[:, None, :]
    return q.to(torch.float16)


def make_record(name: str, n: int, k: int, t_rows: int,
                g: torch.Generator) -> dict:
    assert k % 64 == 0 and n % 64 == 0, "64x64 rotation blocks need it"
    # Weight: gaussian body x per-input-channel lognormal outliers.
    chan_scale = torch.exp(torch.randn(k, generator=g) * 0.8)
    w = (torch.randn(n, k, generator=g)
         * chan_scale[None, :] * 0.02).to(torch.float16)

    # Act scale table: positive, ~1.0, mild per-step + per-channel jitter.
    table = torch.exp(torch.randn(t_rows, k, generator=g) * 0.10
                      + torch.randn(k, generator=g)[None, :] * 0.05)

    return {
        "format": "dit_svdquant_v1",
        "weight_res_q": w,
        "lowrank_A": torch.empty(n, 0, dtype=torch.float16),
        "lowrank_B": torch.empty(k, 0, dtype=torch.float16),
        "act_scale_table": table.float(),
        "duquant_rotation_blocks": _orthogonal_blocks(k // 64, g),
        "duquant_rotation_perm": torch.randperm(k, generator=g),
        "duquant_rotation_out_blocks": _orthogonal_blocks(n // 64, g),
        "weight_bits": 4,
        "a_bits": 4,
        "rank": 0,
        "in_features": k,
        "out_features": n,
        "n_calib_total": 100 * t_rows,
        "n_calib_per_step": [100] * t_rows,
        "act_percentile": 99.9,
        "gptq_damp_percent": 0.05,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    pack = {"__meta__": {
        "recipe": "synthetic fixture (tools/gen_omega_pack_fixture.py)",
        "suite": "fixture",
        "fresh": True,
        "seed": args.seed,
    }}
    for name, n, k, t in LAYERS:
        pack[name] = make_record(name, n, k, t, g)
        print(f"  {name}  N={n} K={k} table=({t},{k})")

    torch.save(pack, args.out)
    print(f"wrote {args.out}: {len(LAYERS)} layers, seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
