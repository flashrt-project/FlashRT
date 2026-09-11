# Omega-QVLA pack to FlashRT E0M3 format

This document describes the **Milestone 1 offline toolchain** shipped in this
repository. It converts rank-0 Omega-QVLA `dit_svdquant_v1` records into the
packed E0M3/UE4M3 weight format consumed by FlashRT's existing SM110 GEMM.

This change does not connect the artifact to a frontend or pipeline. It adds no
runtime route, server, CUDA graph, CMake source, binding, or public Python API.
Runtime-consumer results mentioned in development discussions were produced on
an external experimental branch and are not capabilities of this repository.

## Included tools

- `tools/convert_omega_pack_e0m3.py`: validates and converts a pack.
- `tools/check_omega_e0m3_layer.py`: CPU emulation diagnostics and a Thor
  kernel round-trip that directly consumes converted `packed` and `sfb` data.
- `tools/gen_omega_pack_fixture.py`: creates a four-record synthetic fixture.

## Input container

The input is a plain `torch.save` mapping loadable with
`torch.load(..., weights_only=True)`. A full inspected pi0.5 pack contains 252
records plus `__meta__`: 126 expert records and 126 PaliGemma records.

Each selected record must have `format == "dit_svdquant_v1"` and satisfy:

| Field | Required contract |
|---|---|
| `weight_res_q` | finite floating tensor `[out_features, in_features]` |
| `rank` | exactly `0` |
| `lowrank_A`, `lowrank_B` | `[out_features, 0]`, `[in_features, 0]` |
| `weight_bits`, `a_bits` | both `4` |
| `act_scale_table` | finite positive floating tensor `[steps, in_features]` |
| `duquant_rotation_blocks` | finite `[in_features / 64, 64, 64]` |
| `duquant_rotation_perm` | int64 permutation of `[0, in_features)` |
| `duquant_rotation_out_blocks` | finite `[out_features / 64, 64, 64]` |
| `in_features`, `out_features` | exact match for the weight tensor |

Both feature dimensions must be positive multiples of 64. The converter
validates every selected record before importing the CUDA extension or writing
an artifact. Unsupported records are errors; they are never skipped.

For an unfiltered non-fixture pack, the default coverage gate is 252 records.
`--layer-regex` and fixture packs expect every selected record. Use
`--expected-records` when intentionally converting a different complete pack.

## Output container

The output format is `omega_e0m3_v1`, schema version 1:

```text
{
  "format": "omega_e0m3_v1",
  "schema_version": 1,
  "source_pack_meta": {...},
  "source_record_count": int,
  "selected_record_count": int,
  "selected_layers": [str, ...],
  "fold": "none" | "mean" | "actnorm",
  "weights": {
    layer: {
      "packed": uint8[N, K / 2],
      "sfb": uint8[sfa_size_bytes(N, K, True)],
      "N": int,
      "K": int,
    }
  },
  "aux": {layer: {...}}
}
```

The converter writes through a sibling temporary file and atomically replaces
the destination only after all selected records convert successfully. The
`weights`, `aux`, and `selected_layers` sets must be identical.

## Conversion math

Omega stores `weight_res_q` as a dequantized fp16 tensor already in the
rotated and permuted domain. No GPTQ bitstream decoding is required.

FlashRT converts each `[N, K]` weight with:

```text
quantize_e0m3_dynamic_sfa_fp16(weight, packed, sfb, N, K, is_sfb=True)
```

This produces packed 4-bit E0M3 elements and tile-interleaved UE4M3 per-16
scales. SFB storage is allocated with
`flash_rt_fp4.sfa_size_bytes(N, K, True)` and zero-initialized because the
tile-interleaved layout contains padding.

The default `--fold none` strategy intentionally does not fold the
per-channel activation table into the weight. Thor experiments on four real
layers measured E0M3-vs-fp16 cosine around 0.9924-0.9932, comparable to the
source fake-quant path. `mean` and `actnorm` remain ablation modes; they are not
the recommended artifact format.

## Reproduction

### CPU fixture and emulation

```bash
python tools/gen_omega_pack_fixture.py --out /tmp/fixture_pack.pt

python tools/convert_omega_pack_e0m3.py \
    --pack /tmp/fixture_pack.pt \
    --out /tmp/unused.pt \
    --validate-only

python tools/check_omega_e0m3_layer.py \
    --pack /tmp/fixture_pack.pt \
    --mode emulate
```

The validation command performs no CUDA work and does not create its output.

### Thor artifact round-trip

```bash
python tools/convert_omega_pack_e0m3.py \
    --pack /tmp/fixture_pack.pt \
    --out /tmp/fixture_e0m3.pt \
    --fold none

python tools/check_omega_e0m3_layer.py \
    --pack /tmp/fixture_pack.pt \
    --artifact /tmp/fixture_e0m3.pt \
    --mode kernel \
    --min-artifact-cos 0.98
```

With `--artifact`, the checker quantizes only the activation. It loads the
weight's `packed` and `sfb` tensors directly from the converted artifact,
validates their shape and byte count, runs the GEMM, and fails if cosine is
below the requested threshold. This is the converter round-trip gate.

### Full pack

```bash
python tools/convert_omega_pack_e0m3.py \
    --pack /path/to/quantized.pt \
    --out /path/to/pi05_e0m3.pt \
    --fold none
```

Without a layer filter, this command fails unless all 252 records validate and
convert. A representative artifact layer should then be checked with the same
`--artifact --mode kernel` command above. Release evidence must include both
the direct packed/SFB GEMM result and `252/252` coverage.

## Scope and roadmap

Milestone 1 is limited to conversion, schema validation, a synthetic fixture,
CPU emulation, and direct artifact GEMM verification.

A future runtime PR may load these artifacts in a pi0.5 Thor frontend, apply
the DuQuant input/output rotations, and establish end-to-end accuracy and
latency gates. Such a PR must independently add and test its frontend,
pipeline, graph-lifecycle, and serving contracts. None of that runtime surface
is provided here.
