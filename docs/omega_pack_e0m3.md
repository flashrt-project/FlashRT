# Omega-QVLA pack format and the E0M3 consumption contract

## 0. Background concepts (90 seconds)

- **4-bit quantization**: store `round(x / s)` (a small integer) plus the
  "ruler" `s` (the scale) instead of `x`. Compute happens as
  `integer × s`. Fewer bits = less memory bandwidth, more rounding error.
- **fake-quant**: quantize then *immediately dequantize*, staying in float.
  Simulates quantization error without needing integer hardware — Omega's
  whole runtime is fake-quant emulation on plain PyTorch matmuls.
- **scale granularity**: how many elements share one ruler. *Per-channel* =
  each of the K input channels gets its own (Omega's choice). *Per-16
  block* = 16 adjacent elements share one (the hardware format's choice).
  Coarser granularity = fewer scales to store, but elements of different
  magnitudes get crushed under a shared ruler.
- **static vs. dynamic scale**: *static* = measured offline on calibration
  data, stored in the pack (Omega's `act_scale_table`). *Dynamic* =
  computed per token at runtime from the actual data (`amax / 7`). Dynamic
  is fresher but constrains the layout to what hardware computes cheaply.
- **E0M3**: the 4-bit element format here — sign + 3 mantissa-ish bits
  decoding the uniform integer grid −7..+7. "Uniform" = evenly spaced
  levels, unlike E2M1 (NVFP4) whose levels bunch near zero.
- **UE4M3**: an unsigned 4-exponent/3-mantissa mini-float used *only for
  scales* (the ruler itself is quantized too). Per-16 scales on both
  operands are UE4M3.
- **packed + SFA/SFB**: 4-bit elements are stored two per byte ("packed");
  the per-16 scales live in a separate buffer in CUTLASS's tile-interleaved
  layout (SFA for the activation operand, SFB for the weight operand).
- **DuQuant rotation / permutation**: a learned orthogonal transform
  (64×64 blocks + a channel shuffle) applied to activations before
  quantization. Its job: even out channel magnitudes so no single outlier
  channel dominates a shared scale. Orthogonal = length- and
  angle-preserving, so it is mathematically free.
- **tcgen05 MMA**: the SM100/SM110 tensor-core instruction that consumes
  packed 4-bit operands + UE4M3 scales directly in hardware. This is the
  payoff: Omega's math runs as emulation today; this instruction makes it
  native.
- **cosine similarity**: the fidelity metric. 1.0 = identical direction;
  per-token cos 0.98 means the quantized output vector points in nearly
  the same direction as the reference, with ~2% orthogonal noise.

With those nine, every section below should read top to bottom without
external references.

Status: recon complete, converter/harness in `tools/` (Milestone 1).
Scope: `packs_hf/pi05_long/quantized.pt` (4.8 GB, pi0.5 LIBERO-10 recipe
`paligemma=svdh+gptq, expert=svdh+rtn+perstep`). Other Omega packs share the
`dit_svdquant_v1` record format but were not inspected.

## 1. Container

Plain `torch.save` dict, loadable with `weights_only=True` (no custom
classes). 253 top-level keys:

- 126 expert records:
  `paligemma_with_expert.gemma_expert.model.layers.{0..17}.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}`
- 126 PaliGemma records:
  `paligemma_with_expert.paligemma.model.language_model.layers.{0..17}.<same>`
- `__meta__`: `{"recipe": str, "suite": "10", "fresh": bool}`

Small projections (state_proj, action_in/out_proj, time_mlp) are
deliberately absent — they break under A4 and stay BF16 at runtime.

## 2. Record schema (`format == "dit_svdquant_v1"`)

| field | shape / dtype | meaning |
|---|---|---|
| `weight_res_q` | `(out, in)` fp16 | fake-quantized-then-dequantized weight, **already in the rotated + permuted domain** |
| `lowrank_A` / `lowrank_B` | `(out, 0)` / `(in, 0)` fp16 | SVDQuant low-rank branch; rank = 0 in this pack (INT4-only path) |
| `act_scale_table` | `(num_steps, in)` fp32 | per-denoise-step, per-channel activation scales. Expert: `num_steps = 10`; PaliGemma: `1` |
| `duquant_rotation_blocks` | `(in/64, 64, 64)` fp16 | block-diagonal input rotation R_in |
| `duquant_rotation_perm` | `(in,)` int64 | input-channel permutation (applied before R_in) |
| `duquant_rotation_out_blocks` | `(out/64, 64, 64)` fp16 | block-diagonal output rotation (restore) |
| `weight_bits` / `a_bits` | int | 4 / 4 (both sides). The *runtime* `DuQuantLinear` path defaults activations to A8 (`GR00T_DUQUANT_ABITS`), which is where the "W4A8 PaliGemma" label comes from; pack records consumed through `GptqLinear` use their own `a_bits` (4) |
| `in_features` / `out_features` | int | redundant with tensor shapes |
| `n_calib_*`, `act_percentile`, `gptq_damp_percent` | scalars | calibration provenance |

Notably absent (vs. a classic GPTQ pack): no packed int4 bitstream, no
`qweight`/`qzeros`/group scales — the weight survives only as dequantized
fp16 on the 4-bit grid (~8k unique values per tensor). No `smooth_scale`.

## 3. Consumer math (Omega `gr00t/quantization/gptq_layers.py`, verified)

```
x2 = bmm(x[..., perm].view(N, in/64, 64), R_in_blocks)   # input rotation, runtime
x_q = clamp(round(x2 / s_t), -8, 7) * s_t                # s_t = act_scale_table[step]
y'  = x_q @ W_res_q^T                                    # bf16-promoted accumulate
y   = bmm(y'.view(N, out/64, 64), R_out_blocks) + bias   # output rotation restore
```

PaliGemma (`duquant_layers.py`) is identical in structure with A8
activations and a single-row scale table.

Consequences for a FlashRT consumer:

- The input rotation **cannot** be folded into `weight_res_q`: fake-quant
  sits between rotation and GEMM. It must run on activations (torch bmm, or
  a prologue kernel). Same for the output restore.
- The rotation is an exact orthonormal transform, so it does not by itself
  affect GEMM fidelity; fidelity questions live entirely in the quantizers.
- `weight_res_q` being plain fp16 means the converter re-quantizes from
  fp16 — no GPTQ bitstream decoding needed.

## 4. Mapping to the FlashRT E0M3 contract

FlashRT SM110 path (`csrc/gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cuh`,
bindings in `csrc/fp4_bindings.cpp`):

- Weights: fp16 `[N, K]` → `quantize_e0m3_dynamic_sfa_fp16(..., is_sfb=True)`
  → packed E0M3 `[N, K/2]` + SFB tile-interleaved UE4M3 (per-16, amax/7).
- Activations: same kernel with `is_sfb=False` → packed + SFA.
- GEMM: `cutlass_fp4_gemm_e0m3w(A, SFA, B, SFB, D, M, N, K, α, β, stream,
  a_format)` with `a_format=0` for E0M3 activations (1 = E2M1).
- Buffer sizing: `flash_rt_fp4.sfa_size_bytes(N, K, is_sfb)`; scale buffers
  must be zero-initialized (tile-interleave pads K to 64-element atoms;
  garbage padding decodes as UE4M3 NaN).

Grid differences vs. Omega fake-quant:

| | Omega A4 | FlashRT E0M3 |
|---|---|---|
| element grid | int `[-8, 7]` (asymmetric clamp) | sign-magnitude uniform `[-7, 7]` |
| scale | static calibrated, **per-channel** fp32 | dynamic amax/7, **per-16** UE4M3 |
| weight grid | int4 per-channel-group (already dequantized) | per-16 UE4M3 |

The scale-granularity mismatch (per-channel static table vs. per-16 dynamic)
is the one real fidelity risk. Two candidate strategies, both implemented in
`tools/check_omega_e0m3_layer.py`:

- **S0 (drop the table)**: `A = e0m3(x2)`, `B = e0m3(W)`. Loses all
  calibration information.
- **S1 (fold step-mean table into W, per-step residual into A)**:
  `A = e0m3(x2 / s_t)` per step, `B = e0m3(W · diag(s̄))` once, where
  `s̄ = mean_t(s_t)`. Exact for the mean step; residual error scales with
  the table's step-to-step spread (measured: std/mean ≈ 10% on expert
  layer-0 q_proj).

  Mathematically S1 relies on `Σ_k q_k s_k W_nk = Σ_k q_k (s_k W_nk)`:
  a per-K-column scale commutes into the weight. A true per-step fold would
  need 10 weight copies (unacceptable), hence the mean fold.

RHT (per-16 Hadamard, `use_rht=1` variants) is orthogonal to the DuQuant
rotation — `(x2·H)(W·H)^T = x2·W^T` — and can be ablated on top of either
strategy if per-block distributions remain problematic.

### Measured

Emulation mode (torch, synthetic activations calibrated to q999 = 7·s_t,
M = 256 tokens), per-token cosine vs. the unquantized-activation reference:

| layer | omega vs fp | S0 vs fp | S1 vs fp |
|---|---|---|---|
| expert L0 q_proj (K=1024) | 0.9927 | 0.9834 | 0.9052 |
| expert L0 down_proj (K=4096) | 0.9928 | 0.9825 | 0.9799 |
| expert L11 o_proj (K=2048) | 0.9929 | 0.9825 | 0.9615 |
| paligemma L0 gate_proj (K=2048) | 0.9928 | 0.9810 | 0.9775 |

Kernel mode (real tcgen05 GEMM, Thor SM110, same seed, per-token mean
cosine vs. the unquantized-activation reference):

| layer | omega vs fp | S0 vs fp | S1 vs fp |
|---|---|---|---|
| expert L0 q_proj (K=1024) | 0.99247 | **0.99322** | 0.158 |
| expert L0 down_proj (K=4096) | 0.99274 | **0.99257** | 0.98999 |
| expert L11 o_proj (K=2048) | 0.99275 | **0.99303** | 0.85557 |
| paligemma L0 gate_proj (K=2048) | 0.99206 | **0.99239** | 0.98204 |

Three findings:

1. **S0 is lossless on real hardware on every layer tested** — within
   ±0.001 of Omega's own fake-quant everywhere (the tiny edges come from
   dynamic per-token amax beating a static table on data calibrated only
   at the q999 point). Error independence holds wherever checked:
   cos(S0, fp)·cos(omega, fp) ≈ measured cos(S0, omega), i.e. S0's
   residual is fresh rounding noise, not a systematic shift.
2. **S1's collapse on real hardware is scale-magnitude-dependent.**
   Mechanism: `W · diag(s̄)` shrinks weights by the mean table value,
   pushing per-16 block scales toward the UE4M3 subnormal floor (2⁻⁹),
   where scale mantissas disintegrate and whole blocks quantize to
   garbage. Layers with small s̄ die hard (q_proj 0.16, o_proj 0.86);
   layers whose table happens to be larger merely degrade (down_proj
   0.99 — still worse than S0). The emulator's lenient subnormal
   handling masked the severe cases.
3. The pure-torch references reproduce across machines to 5 decimal
   places (0.992700 Thor vs. 0.992707 x86), cross-validating the harness.

**S0 wins; S1 is dead.** The table's per-channel scale spread (~4×)
distorts weights when folded, while S0's per-token dynamic per-16 amax is
a *better* quantizer than Omega's static per-channel table — the DuQuant
rotation+perm has already whitened per-channel magnitudes, so the table
is only a second-order correction.

Decision: **the converter emits S0 (`--fold none`) as the production
format**; `--fold mean` is kept for ablation only. This also shrinks the
runtime story — no per-step scale dispatch is needed on the E0M3 path.

**Follow-up: `actnorm` (floor-safe S1) — also dead (2026-08-18).** A
reviewer-natural fix for S1's floor problem is to normalize before
folding: decompose `s̄ = c·r̄` with `c = geomean(s̄)`, fold only `r̄`
(O(1), geomean 1) into the weights, divide activations by `s̄` at
runtime, and absorb `c` into the GEMM alpha. This is exactly
`(x/s̄) @ (W·r̄)^T · c = x @ W^T`, and it does fix the floor (0% of
block scales below 2⁻⁹ vs 100% for raw S1 on q_proj). But measured on
Thor (consumer-level, fp16 reference, real pack):

| layer | S0 vs fp16 | actnorm vs fp16 |
|---|---|---|
| q_proj | 0.9935 | 0.9886 |
| down_proj | 0.9929 | 0.9869 |
| o_proj | 0.9936 | 0.9885 |

actnorm is *worse* than S0 everywhere. Mechanism: the fold is a zero-sum
redistribution — dividing activations by `s̄` whitens the activation
blocks, but multiplying weights by `r̄` (range 0.43–2.68 on q_proj)
re-opens intra-block magnitude spread on the weight side, where per-16
single-scale 4-bit pays for it. DuQuant's rotation had already whitened
both operands; any per-channel re-scaling of either side undoes that.
**Per-channel calibration tables are fundamentally incompatible with
per-16 block quantization — the information has to live on one side and
always de-whitens it.** Dynamic per-16 amax is the optimum at this
granularity; S0 is the endpoint, not a compromise. (`--fold actnorm` +
consumer support remain in the tree, `OMEGA_E0M3_ACT_TABLE=0`/artifact
driven, as the documented ablation.) The residual end-to-end gap vs.
the full Omega recipe (90.4% vs 93.2%, concentrated in task9) is not
recoverable by table injection; remaining options are mixed precision
for sensitive layers or acceptance.

Remaining caveats: synthetic activations (lognormal + outlier channels,
calibrated only at the q999 point) — real activation tails differ. Next:
captured real activations, then LIBERO paired SR (Milestone 2).

## 5. Roadmap and Milestone-1 deliverables

**Milestone 1 — offline converter + format doc + single-layer gates
(done).** Deliverables below; acceptance: 252/252 records converted, S0
per-token cosine ≥ Omega fake-quant on real hardware (4/4 layers, §4).

**Milestone 2 — runtime consumption (next).** Wire the converted pack
into the pi0.5 Thor pipeline: load `packed`/`sfb` as decoder GEMM
operands, run the DuQuant input rotation (perm + 64×64 block bmm) and
output restore around each replaced Linear, keep the small projections
(state/action/time) BF16 from the checkpoint. Acceptance: end-to-end
action cosine vs. the Omega fake-quant server, then LIBERO-10 ×500
paired SR vs. the BF16 baseline (target: no measurable loss, matching
the pack's own 93.2% vs 91.6%).

**Milestone 3 — upstream PRs (after M2 evidence).** Split per
`CONTRIBUTING.fork.md` §6, each with LIBERO paired SR + action cosine +
p50/p95 latency: ① converter + this format doc (pure additive, easiest);
② runtime wiring as a flag-gated `weight_format` branch (the S0 result
shrunk this from the originally-planned per-step scale path); ③ SVDQuant
low-rank epilogue (deferred — rank = 0 in this pack).

**Milestone 2 status (done, incl. 2c/2d landed after the original
write-up):**

- **M2a/b — consumer + serving (done).** `tools/omega_e0m3_linear.py`
  (`OmegaE0M3Linear`, drop-in for gr00t's `GptqLinear` via
  `tools/serve_omega_e0m3.py` monkeypatch) + `tools/check_omega_e0m3_consumer.py`
  gate: per-layer cosine 0.978–0.982 vs. GptqLinear, 1.2× layer latency.
  Server smoke 10/10 ≙ arm D; **LIBERO-10 ×500 paired: 90.4%** vs. BF16
  91.6% (McNemar p = 0.53) and vs. fake-quant arm D 93.2% (p = 0.070) —
  no significant loss; 58 s/episode vs. 148 s fake-quant (2.6×). Eager
  mode with `torch.compile` disabled: the pybind kernels graph-break and
  the HF KV cache recompiles per step (~25 min/episode stall) — see the
  env flags in `serve_omega_e0m3.py`.
- **M2d — hand-rolled CUDA graph over the whole denoise loop (done).**
  `tools/omega_e0m3_graph.py` captures all 10 flow-matching steps
  (unrolled, pi05_thor style) into one `torch.cuda.CUDAGraph`: static KV
  slabs behind a `DynamicCache` shell, static mask/position buffers
  filled by `copy_` per inference, adaRMS conditioning precomputed for
  the deterministic time grid. The eager blockers removed are documented
  in the module docstring (device-scalar `while`, per-step H2D mask
  upload, per-call KV allocation). Enable with `OMEGA_E0M3_CUDA_GRAPH=1`
  (see `tools/start_e0m3_server.sh`); falls back to eager permanently on
  any capture failure. Thor validation: capture succeeds
  (`prefix_len=968, layers=18, steps=10`), smoke 10/10, ~43–50
  s/episode vs. ~58 s eager.
- **M2e — PaliGemma E0M3 (route A: official all-W4A4 pack recipe).** The
  converter already emits all 252 records, PaliGemma included; serving
  with `OMEGA_E0M3_PATCH_DUQUANT=1` (the default in
  `tools/start_e0m3_server.sh`) substitutes the runtime `DuQuantLinear`
  wraps with `OmegaE0M3Linear` consumers built from the pack's PaliGemma
  records — GPTQ W4A4 weights instead of runtime RTN, single-row scale
  table dropped per the S0 decision. With `omega_e0m3_graph.py`'s prefix
  graph (default on), the prefix prefill is captured too, so the E0M3
  pybind kernels run inside a CUDA graph on this path as well — the
  capture smoke (`check_omega_e0m3_graph_smoke.py`) covers their
  capturability.
  Validation ladder on Thor: artifact coverage check (252 records) →
  per-layer consumer gate on PaliGemma layers → 10-episode smoke →
  LIBERO-10 ×500 paired SR + per-episode latency.

Deliverables:

- `tools/convert_omega_pack_e0m3.py` — offline pack → E0M3 converter
  (S0 weight emission + aux tensors: perm, R_in/R_out blocks,
  act_scale_table). Runs where `flash_rt_fp4` is built (Thor).
- `tools/check_omega_e0m3_layer.py` — single-layer cosine harness:
  Omega fake-quant reference vs. FlashRT E0M3 GEMM (S0/S1), plus a pure
  torch emulation mode that runs without the extension for pre-checks.
  Emulation results and the S0 decision are in §4.
- `tools/omega_e0m3_linear.py` — `OmegaE0M3Linear` consumer (M2a).
- `tools/check_omega_e0m3_consumer.py` — consumer-vs-GptqLinear gate
  (cosine + layer latency).
- `tools/serve_omega_e0m3.py` — openpi serving entry (monkeypatches
  gr00t's wrap classes; env-gated compile kill switch).
- `tools/omega_e0m3_graph.py` — CUDA-graph capture of the 10-step
  denoise loop (M2d), `OMEGA_E0M3_CUDA_GRAPH=1`.
- `tools/check_omega_e0m3_graph_smoke.py` — P0 capture gate for a single
  consumer layer (pybind capturability check).
- `tools/start_e0m3_server.sh` — Thor server launcher (repo-relative
  paths, env overrides).

Deferred: SVDQuant low-rank epilogue (rank = 0 everywhere in this pack),
per-step weight tables (10× memory; also refuted by the S0 result),
per-step activation scale dispatch (refuted by the S0 result).

### Reproducing

**Self-contained fixture round-trip (no Omega-QVLA pack or checkout
needed — the PR review path):**

```bash
cd third_party/flashrt
# 1. Synthetic miniature pack: schema-identical records, random
#    orthogonal rotations, outlier-channel weights (pure CPU, seconds)
python tools/gen_omega_pack_fixture.py --out /tmp/fixture_pack.pt
# 2. Convert (Thor)
python tools/convert_omega_pack_e0m3.py --pack /tmp/fixture_pack.pt \
    --out /tmp/fixture_e0m3.pt --fold none
# 3. Consumer vs fp16 reference, gr00t-free (Thor)
PYTHONPATH=$PWD/tools python tools/check_omega_e0m3_consumer.py \
    --reference fp16 --pack /tmp/fixture_pack.pt \
    --artifact /tmp/fixture_e0m3.pt
```

**Full pack (development path):**

```bash
# Point at an Omega pack (any machine for emulate, Thor for kernel/convert)
export OMEGA_PACK=/path/to/Omega-QVLA/packs_hf/pi05_long/quantized.pt
cd third_party/flashrt

# 1. Local pre-check, no extension needed (pure torch, CPU is fine)
python tools/check_omega_e0m3_layer.py --pack "$OMEGA_PACK" --mode emulate
 
# 2. Hardware check — real tcgen05 GEMM (Thor, flash_rt_fp4 built)
python tools/check_omega_e0m3_layer.py --pack "$OMEGA_PACK" --mode kernel
```

```bash
# 2. Hardware check output
layer: paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj  N(out)=2048 K(in)=1024  table=(10,1024) step=0

references:...
```
```bash
python tools/check_omega_e0m3_layer.py --pack "$OMEGA_PACK" --mode kernel \
    --layer paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj
  # --layer paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj
  # --layer paligemma_with_expert.gemma_expert.model.layers.11.self_attn.o_proj
# 3. Full conversion (252 layers, ~1.3 GB output)
python tools/convert_omega_pack_e0m3.py \
    --pack "$OMEGA_PACK" --out pi05_long_e0m3.pt --fold none
```

Gate for accepting the conversion: per-token cosine of S0 vs. fp ≥
Omega's own fake-quant (per-layer, same seed). Currently met on every
layer tested (see §4).

## 6. Accuracy context (pi0.5 LIBERO-10, 500 episodes)

The 93.2% figure was measured on the hybrid deployment: expert records from
this pack (W4A4, `GptqLinear`) + PaliGemma via the *runtime* DuQuant path
(W4 weights, A8 activations by the `GR00T_DUQUANT_ABITS` default) — vs. BF16
baseline 91.6% (McNemar p = 0.32, no significant difference) on the Omega
PyTorch fake-quant path. The pack itself is the official Omega recipe,
which is W4A4 on both sides (PaliGemma records carry `a_bits=4` and a
single-row `act_scale_table`); consuming the PaliGemma records through the
E0M3 consumer (`OMEGA_E0M3_PATCH_DUQUANT=1`) therefore *is* the official
recipe, and additionally replaces runtime RTN weights with the pack's GPTQ
weights. The E0M3 migration target
is therefore "no measurable SR loss against an already lossless baseline" —
the single-layer cosine gates are the leading indicator, LIBERO the final
one.
