> Historical experiment notes. The old local golden shared a vision attention
> layout error and a missing encoder output projection with the serving path.
> Its correctness and speedup claims do not qualify the corrected model.
> Earlier custom-vector failures also do not establish hardware limitations:
> the standalone BF16-input/FP32-compute row quantizer now executes on Ascend.
> See [current deployment contract](deployment_npu.md).

# Pi0.5 Ascend NPU L3 Fusion Programme (reuse-first + custom-kernel gaps)

> STATUS NOTE (2026-09-08): L3-P0 official-op fusions are all landed at
> frontend p50 ~86 ms (report: `docs/npu_pi05_optimization_report.md`).
> The custom-kernel part of this plan was superseded by measured facts: the
> bespoke kernel channel is limited to 16-bit GM traffic + fp16 vector math
> (no bf16 vector math; every fp32 path faults/inert), and the official fused
> RoPE/RMSNorm+cache family fails at runtime on this CANN/torch_npu build.
> See `flash_rt/npu/ops/README.md` for the device matrix, the option-c kernel
> (#1 `kv_cache_store`, A/B slower than aclnn copies) and the option-b audit.
> Sections below remain the historical plan, not current status.

> Goal: drive the captured raw ~135 ms / frontend p50 ~182 ms (after M3-L1)
> toward the ~30 ms class. Basis: `docs/pi05_npu_fusion_plan.md` (L1 results,
> M3-0 segment baseline, capture red lines);
> `docs/kernel_fusion.md` / `docs/kernel_catalog.md` / review appendix
> (CUDA/AMD fusion shapes with file:line).
> Verified local facts: CANN 8.5.2 with `ccec`, `ascendc_pack_kernel`,
> `ascendc` cmake, `msopgen`, `op_project_templates` all present; torch_npu
> already exposes many **ready-made fused ops** (see the mapping in §4).

---

## 1. Current state and gap (after M3-L1, captured medians)

| Segment | Captured ms | Remaining "glue/small-op" work (per run) |
|---|---|---|
| decoder 10 steps | ~84.5 | per layer per step: 2× AdaRMS apply chains (~7 ops), rope chain (~12 ops), 2× KV suffix writes, manual attention (bmm×2+softmax+reshape/transpose/expand), residual/gate mul, style lookup |
| encoder 18L S=552 | ~27.7 | per layer: 2× rms (~7 ops), 2× rope (~12 ops), KV contiguous, attention bmm pair, residuals |
| vision 27L | ~23.3 | per layer LN (~7 ops), gelu, bias/res memory chains |
| FULL raw | ~135 (frontend p50 ~182 incl. host/graph scheduling overhead) | — |

Bottleneck nature (proven in M3-0): **in-graph node count × per-node
scheduling/kernel cost**, not compute.
→ Fusion = compressing one op group into a single aclnn/custom node per call;
node count drops proportionally and latency follows almost linearly.
Reference: CUDA/AMD compress the encoder layer to 10/9 launches and the
decoder to 13–15 per layer by fusing "glue between GEMMs" and never the GEMMs.

---

## 2. Three iron rules set by Ascend characteristics

1. **Matmul goes through Cube = aclnn/npu_* GEMM only; no custom matrix
   kernels.** CUDA/AMD can hand-write MFMA/SIMT GEMM epilogues; on Ascend the
   Cube is scheduled by aclnn, and writing Cube+epilogue fusion inside one
   Ascend C kernel is another order of effort → not done.
2. **Vector-side memory fusion is the main battlefield and is mostly already
   in aclnn** (§4). CUDA/AMD hand-write norm/gate/rope SIMT kernels; Ascend's
   Vector unit + aclnn already ships `npu_rms_norm`, `npu_add_rms_norm`,
   `npu_kv_rmsnorm_rope_cache`, `npu_geglu`… **use first, write only what is
   missing.**
3. **Capture red lines (measured lessons)**:
   - a failed capture = capture_end 107025 + **native abort killing the
     process** (SDPA lesson) → every candidate op must be capture-verified in
     an **isolated subprocess**; try/except is useless for whole-graph
     failures.
   - input layout / head dim / sequence length change aclnn flash kernel
     selection (hd=72 and S=560 are not capturable; only the decoder small-S
     manual bmm path is confirmed capturable). New ops always pass the
     "isolated capture probe" before entering a graph.
   - aclop ops (e.g. conv2d) cannot be captured → already avoided with im2col;
     each candidate is classified aclnn/aclop individually.

---

## 3. Reuse-first (L2.5) or custom kernels (true L3)?

**Decision rule**: for each needed fusion, check the §4 table for a
semantically matching ready-made op → if present, run the isolated probe
(capturable? cos? timing?) → if it passes, adopt it in the graph (cheap,
deterministic); if absent, evaluate a custom Ascend C kernel (write only the
"one fused call" gap; K-gap list in §5).

This is not abandoning L2; it makes L2 the first phase of L3 (a reuse sweep),
avoiding re-writing wheels. The SDPA lesson does not apply wholesale: these
ops are first-party torch_npu ops; capturability must be verified per op, not
dismissed by analogy with SDPA.

---

## 4. Fusion mapping table (candidate op → coverage → semantic risk; all
pending the isolated probe)

| Needed fusion (CUDA/AMD kernel) | Ready-made torch_npu op (CANN 8.5.2, verified present via dir) | Semantic risk (probe must check) |
|---|---|---|
| encoder/decoder RMS (incl. (1+w) weight) | `npu_rms_norm`; `npu_gemma_rms_norm` | whether gamma is (1+w) or w; bf16 input/output contract; capturable |
| **residual+RMS** (= CUDA `residual_add_rms_norm_*`) | `npu_add_rms_norm` / `_cast` variants | semantics: residual order, any residual scaling; capture |
| decoder K/V **RMS+RoPE+KV cache write** (= CUDA `qkv_split_rope_kvcache_*`) | `npu_kv_rmsnorm_rope_cache_v2` (+`_functional`) | input semantics (q/k/v, positions, cache layout), GQA/kv heads, capture |
| RoPE (encoder/decoder q/k rotation) | `npu_apply_rotary_pos_emb`; `npu_rotary_mul`; `_npu_rotary_embedding` | rotation formula / half-split matching `rope_half_split` (cos/sin tables or frequency input) |
| full GQA attention | `npu_prompt_flash_attention`; `npu_fusion_attention(_v2)`; `npu_incre_flash_attention` | **SDPA precedent**: isolated capture per shape/head-dim/non-contiguous input; cross-process determinism |
| gate+SiLU/GELU(mul/up) (= CUDA `gate_geglu_merged`) | `npu_geglu`, `npu_gelu_mul`, `npu_swiglu`, `npu_fast_gelu`, `npu_clipped_swiglu` | gelu approximation type: our gate is **tanh-approx**; exact variants differ numerically → cos gate |
| vision LN / bias+res+LN | `npu_layer_norm_eval`, `npu_add_layer_norm` | eps/dtype, residual-folding semantics |
| in-attention softmax(mask) | `npu_scaled_masked_softmax` | masks unused today; low priority |
| GEMM fused epilogues (QKV/O/GateUp/down bias/gelu/res) | **N/A**: Cube already optimal via aclnn linear/matmul → skip (mirrors CUDA doing it only in fp8/CUTLASS) | — |

**Probe protocol per candidate** (standalone subprocess so an abort never
infects the main process):
1. build the op inputs/weights (pi05-real shapes, same dtype, contiguous);
   run the npu_* op once eagerly;
2. compare with the **equivalent manual chain** (cos/max_abs, expect ≥0.9999
   when semantics should match; if semantics differ — e.g. exact gelu —
   record the delta and evaluate at the acceptance gate);
3. capture the op alone in `torch.npu.graph` + replay; verify capturable and
   replay == eager;
4. median captured timing (≥30 Events); compare with the covered segment's
   current median → decide whether to adopt in the fast path;
5. after adoption: full-frame golden cos ≥0.9999 + segment A/B + pytest green.

---

## 5. Gaps the ready-made ops do not cover (custom Ascend C considered only
here)

| Gap (CUDA/AMD twin) | Why missing | Suggestion |
|---|---|---|
| decoder **AdaRMS apply** (style scale/shift/gate + norm + cast, 2 per layer per step) | needs a 3·D style input; no `npu_*ada*` semantic op found; currently ~7 ops/call | if switching the RMS side to `npu_rms_norm` still leaves a "scale+shift+gate chain", this is the **first choice for a small custom kernel** (or evaluate whether it is worth it) |
| cross-layer `gate×residual+next RMS` (= CUDA `gate_res_adarms_*`) | cross-op semantics are complex | low priority; only after the residual side has been saturated |
| rope+KV-write merge (if kv_rmsnorm_rope_cache semantics do not match) | depends on the probe | decide custom `K2` from probe results |
| vision LN/bias/gelu chain (if npu_add_layer_norm/gelu_mul do not match) | same | decide from probes |

**Custom toolchain facts (verified)**: `ccec`, `ascendc` cmake + tikcpp,
`ascendc_pack_kernel`, `msopgen`/`op_project_templates` are all present under
CANN 8.5.2.
**Two integration lanes for custom kernels (decided by spike)**:
- a) msopgen-generated aclnn-style single op → still needs bridging into
  torch_npu (heavy, no verified shortcut);
- b) if ready-made npu_* ops suffice → custom kernels are unnecessary and the
  whole toolchain risk disappears.
**Conclusion: do the P0 reuse sweep (§6) first, then a spike decides whether
to touch lane b.**

---

## 6. Staged execution plan (measurement-driven; every step passes §7)

**P0 — reuse-sweep probes (in progress)**
Progress:
- ✅ `npu_rms_norm(x, gamma=(1+w) bf16, eps)`: cos 0.999995; capturable;
  0.040 ms vs manual 0.176 ms
- ✅ `npu_add_rms_norm(x1,x2,gamma,eps) → (norm, rstd, residual)`: replaces
  `x=x+o; rms` in one call; capturable; 0.046 vs 0.195 ms
  (note: the residual output is fp32 — cast back to bf16 to continue the
  chain; bf16 gamma must pair with bf16 x)
- ✅ **encoder fused norms in the graph (`encoder_pass_opt` is the frontend
  default)**: segment 27.7→24.4 ms; **full-frame p50 182→~139 ms** (the global
  gain from fewer whole-graph nodes is far larger than the segment gain);
  cos 0.999993 (inside the ≥0.9999 gate); pytest 8 passed
- ✅ **decoder AdaRMS norm fusion** (`make_styles_opt` + `_ada_opt`: style
  pre-split into gamma=(1+scale)/shift/gate; `npu_rms_norm` replaces the
  manual var/rsqrt/mul fp32 chain in one node): decoder 84.5→**63.8 ms**;
  **full-frame p50 ~139→~110 ms**; cos 0.999992; pytest 8 passed
- ✅ **vision LayerNorm fusion** (`vision_tower_opt`: `npu_layer_norm_eval` is
  an aclop and cannot be captured → use the capturable
  `npu_add_layer_norm(x, zeros, gamma, beta)` as a plain LN): p50
  ~110→~105.5 ms; cos 0.999990; pytest 8 passed
- ✅ **RoPE table precompute + rotation minimisation** (`make_rope_tables` /
  `rope_fast`, bit-identical to the previous rope): encoder 24.4→22.6,
  decoder10 63.5→**54.6 ms**; p50 ~105.5→~95 ms; cos 0.999990; 8 passed
- ✅ **attention fusion** (`attention_flash`: `npu_prompt_flash_attention`,
  encoder S=552 and decoder S=562 both **isolated-capturable**; vision hd=72
  keeps the manual path): p50 ~95→~86.4 ms; cos 0.999993; 8 passed
- **Current segment baseline (re-measured): vision 18.4 / encoder 21.6 /
  decoder10 46.1 ≈ 86 ms**; frontend p50 ~86.4 ms (from ~202 ms, **-57%**),
  cos≥0.99999, 8 passed

Run the §4 probe protocol per op and output a "usability matrix":
`usable & gain | usable but neutral | semantics mismatch | not capturable
(aclop/abort)`. High-ROI first three: ① `npu_kv_rmsnorm_rope_cache_v2`
(decoder K/V: rms+rope+write, ~20+ nodes/layer saved)
② `npu_rms_norm` (encoder rms chain) ③ `npu_add_rms_norm` (residual+rms).
④ `npu_apply_rotary_pos_emb` (encoder q/k rope). ⑤ flash family last (abort
risk / uncertain gain). Deliverable: `docs/pi05_npu_reuse_matrix.md`.

**P1 — decoder fused onto the graph (expect 84.5 → 40–55 ms)**
Adopt the matrix winners into the `fast.py` decoder: rms/rope/KV-write via
ready-made ops or combinations; minimise the style-apply chain. Every step:
golden cos ≥0.9999 + segment median + pytest.

**P2 — encoder fusion (27.7 → ~15 ms expected)**
`npu_rms_norm`/`npu_add_rms_norm` cover residual+rms; rope switched to
`npu_apply_rotary_pos_emb` (semantics permitting); KV contiguous removal.
(Note: encoder QKV merge was proven negative → do not repeat.)

**P3 — vision fusion (23.3 → ~12 ms expected)**
`npu_add_layer_norm`/`npu_layer_norm_eval` + `npu_gelu_mul`/fast_gelu
semantics verification.

**P4 — custom-kernel gaps (only when the matrix exposes a hard need; 2–4
weeks per kernel)**
First a spike: minimal Ascend C elementwise → registered → called from torch
→ captured in NpuGraph (verifies whether lane b works). If it works, write
1–2 gap-covering kernels by ROI (ada apply chain first); if not, keep those
gaps as multiple nodes and accept the residue.

**P5 — accuracy / latency / stability closure**
Full-chain paired A/B (same-process alternating, ≥50 runs), captured-full vs
eager bit-identical, golden cos, `npu_*` cross-process determinism (flash
family, if used, needs determinism proof or a deterministic regression path),
latency tripwire tightened to ~captured expectation ×1.5.

---

## 7. Acceptance gates

1. every op/kernel: the isolated-process probe must pass first (capturability
   + semantic cos), otherwise it never enters the real graph;
2. after adoption: `replay == eager` (same order) or **golden raw cos ≥ 0.9999**
   (order/reduction changed);
3. segment timing protocol unchanged: Event medians ≥30 runs, reported against
   the current baseline;
4. full-frame paired A/B (alternating) is authoritative; single-arm
   cross-process comparisons are banned (noise ±3–5 ms, proven);
5. `FLASH_RT_NPU_PI05_REF=... pytest tests/test_npu_pi05_model.py` green;
6. flash-family adoption requires a determinism proof or a deterministic
   regression path.

---

## 8. Expected gains (honest, tiered)

- **P0–P3 reuse tier**: captured raw 135 → **60–85 ms**; frontend p50 ~182 →
  **~90–120 ms** (mostly ready-made ops, moderate risk, a probe gate at every
  step).
- **P4 custom tier**: depending on the gap size → 50–70 ms band.
- **~30 ms tier**: needs (a) the above plus more radical node reduction
  (whole-layer fusion), or (b) 910B-native **INT8 calibration** (≈2×, the
  equivalent of others' FP8, a separate programme); CUDA/AMD's ~16–30 ms is
  hand-written MFMA plus years of tuning — the NPU needs the same class of
  effort. This document only promises the 60–85 ms of P0–P3 as near-term
  achievable; further investment is decided by measurement.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| npu_* op capture failure (abort) | isolated-subprocess probes first; excluded on failure; never blocks the whole effort |
| npu_* numeric semantics ≠ reference (gelu tanh/exact, rms weight convention, rope formula) | per-op cos comparison against the manual chain in the probe; record or discard on mismatch |
| version drift (CANN 8.5.2 names/behaviour) | trust only what `dir()` reports on this box; document the version |
| flash family cross-process non-determinism | determinism requirement + deterministic regression path (precedent: AMD fp8-out handling) |
| custom-kernel integration lane unverified | P4 spike precedes any custom investment; stop if it fails |
