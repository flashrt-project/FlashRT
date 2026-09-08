# Pi0.5 Ascend NPU Fusion Implementation Plan

> Context: a plain torch_npu + `torch.npu.graph` full-frame capture already
> works (2 views, 10 steps, BF16, ~200 ms/frame, fp32-reference cos=0.999998).
> This document is the plan for driving ~200 ms toward the AMD-BF16 class
> (~30 ms). Sources: `docs/kernel_fusion.md`, `docs/kernel_catalog.md`,
> `docs/optimization-details.md`, `docs/deployment_amd{,_pi05}.md`,
> `csrc/kernels/*.cu`, `csrc/amd/kernels/*.hip` (kernel-by-kernel analysis in
> the review appendix).

## 1. How FlashRT organises fusion (the CUDA/AMD layering)

| Layer | What | CUDA | AMD twin |
|---|---|---|---|
| L0 | Library GEMM epilogues (bias/gelu/res into cuBLASLt or hand-written MFMA) | `csrc/gemm/gemm_runner.cu` `bf16_nn_bias{,_gelu,_res}` | `csrc/amd/gemm/hipblaslt_runner.hip` + `smallm_*` |
| L1 | Single-op fused kernels: norm+act, bias/residual, gate+act (hand SIMT) | `csrc/kernels/norm.cu`(14+), `activation.cu` | `csrc/amd/kernels/norm.hip`, `activation.hip` |
| L2 | Composite fusions (fusion composite, 17): residual+rms+quant, ada_rms emitting gate, gate_geglu_merged, memory-bound chains | `fusion.cu`, `norm.cu`, `activation.cu` | `fusion.hip`, `norm.hip`, etc. |
| L3 | Whole-layer / cross-layer megakernels + flash attention | `decoder_fused.cu` (gate_res folded into next layer C1), FA2/aiter | `decoder_flash.hip`, `encoder_flash.hip`, aiter |
| L4 | CUDA/HIP graphs (eliminate launches: pi05 ~2,840 nodes vs compiler ~21,000) | CUDAGraph | HipGraph |

On the BF16 (unquantised) path all the "→fp8" variants of L1/L2/L3 collapse,
leaving only the dequantised norm/gate/bias/residual/rope versions —
**this is the shape the NPU should align with**.

## 2. Real per-layer kernel composition on pi05 (launches per layer; file:line in the review appendix §C)

- CUDA Thor FP8 encoder layer = **10 launches/layer**: RMS+quant → QKV GEMM →
  qkv_split_rope+KV write → attention → quant → O GEMM → residual+RMS+quant →
  GateUp GEMM → gate_geglu → Down GEMM.
- Decoder layer FP8 = **13 launches/layer** (C1..C7, including cross-layer
  gate_res_adarms folded into the next layer); **BF16 = 14–15 launches/layer**
  with Gate/Up as two GEMMs (wide merge is fp8/CUTLASS only).
- Vision layer FP8 = 9 launches/layer (GEMM epilogues absorb bias/gelu/
  residual); AMD BF16 = 10–11 (standalone kernels).
- FlashRT-specific hand-written points that also pay off in BF16:
  `qkv_split_rope(+KV cache write)`, `residual+rms`, `ada_rms emitting gate in
  one pass`, `gate_geglu_merged`, cross-layer `gate×residual+next-layer norm`,
  seqused softmax.

## 3. Fusion rules and lessons (what not to step on on the NPU)

1. **Fuse the memory-bound glue between/before/after GEMMs, never the GEMMs
   themselves**: on Thor folding a whole attention into one SIMT kernel was
   5–7× slower than the GEMM chain (two cuBLAS tensor-core calls ≈1 µs). On
   the NPU matmul must go through Cube/aclnn — same rule.
2. Fold the norm weight into the next GEMM (`(1+w)` multiplied into
   qkv/gate/up once at load) → noweight RMS; precompute AdaRMS dense/style
   (370 calls saved per inference).
3. Every dtype cast is one full-width HBM read+write → removing casts is pure
   gain; push casts to the fewest op boundaries.
4. Never introduce transpose/concat data movement just to save one matmul.
5. Capture rules: fixed buffers, no Python branches, no `.cpu()`/sync/dynamic
   allocation, warm-up at the exact inference shapes, capture one step graph
   and replay it 10×. NPU addition: **aclop operators cannot be captured**
   (conv2d already avoided; every new op is verified case by case).
6. Variable shapes (prompt length) → rebuild the capture graph per length
   (already how the NPU frontend works).

## 4. Three-tier progression (recommended path)

**L1 plain-torch rearrangement (no C++; estimate 200 → ~90–110 ms, 3–5 person-days)**
1. Remove per-op fp32 up/down casts (rms/layer_norm/rope/attention currently
   call `.to(fp32)`/`.to(dtype)` every call).
2. Fold encoder/decoder norm weights into q/k/v/gate/up (noweight RMS, as in
   FlashRT).
3. Merge the three QKV linears into one matmul (concat `[K,(NH+2·NKV)·HD]`),
   one matmul + one split in the forward; Gate/Up the same way (`[K,2H]`, only
   when the concat adds no copies).
4. GQA without expand (per-head sliced bmm) or switch to aclnn flash attention.
5. Decoder per-step K/V **preallocated, no cat**: cache buffer `[total,256]`,
   slice-write the current step, one cross-attention.
6. Precompute style/cond tables outside the graph; look them up inside.
7. Vision: remove casts in the LN/GELU chain, merge QKV into one matmul.

**L2 aclnn native ops (survey + trial runs, 2–5 person-days)**
- `npu_prompt_flash_attention` replaces encoder/decoder/vision bmm attention
  (fall back if it does not fit);
- `npu_rms_norm` / BF16 `F.layer_norm` (pre-cast gamma/beta to honour the
  same-dtype constraint);
- verify the aclnn/aclop class of every op (capturable?) and the exact CANN
  8.5.2 names one by one.

**L3 Ascend C custom kernels (memory-bound glue; 6 kernels, 2–3 weeks; bench
each kernel standalone before a shadow-path A/B)**
- K2 `qkv_split+rope(+KV cache write)` — highest priority (replaces manual
  rope + cat/contiguous);
- K3 `residual+RMS` (encoder noweight);
- K1/K4 `ada_rms(+gate+residual)` and cross-layer `down×gate+residual+next RMS`;
- K6 vision `bias+residual+LN`, `add_bias+gelu`;
- K5 seqused softmax only if aclnn flash cannot take a mask.

**Skip**: all fp8-only fusions, FP16 specialisations, hand-written
GEMM/megakernels (matmul goes through aclnn/Cube).

## 5. M3-0 measured segment baseline (910B4, captured medians, torch.npu.Event)

| Segment | Captured ms | Eager ms | torch-op calls (eager census) |
|---|---|---|---|
| vision (SigLIP 27L) | 23.3 | 25.8 | 4563 (empty 825 / to 332 / reshape 303) |
| encoder (18L, S=552) | 27.7 | 33.7 | 5733 (empty 1141 / **aclnnCast 289×2** / as_strided 594) |
| **decoder 10 steps (18L)** | **103.3** | 365.0 | **63693 (empty 10640 / to 4751 / slice 3270 / transpose 2550)** |
| FULL | **154.3** | 439.3 | sum of segments ≈ full frame (self-consistent) |

**First cut goes to decoder** (67% of the captured frame; 6.4k ops/step;
dtype-cast/allocation explosion), then the encoder cast chain (578 aclnnCast
per run), vision is smallest.

## 6. Measured revision of the fusion rules (capture red lines, CANN 8.5.2)

- **`F.scaled_dot_product_attention` cannot be captured in an NPU graph**:
  vision hd=72, encoder S=560 and in-context decoder calls all trigger an
  internal "unjoined stream" in aclnn flash; `capture_end` reports 107025 and
  the process **aborts natively** (try/except cannot stop it). Conclusion: the
  captured path can only use manual bmm/softmax/bmm; switching to flash
  requires an op proven capturable in isolation (L2's
  `npu_prompt_flash_attention` must pass the isolated capture probe before it
  can be trusted).
- The manual bmm attention path was restored as the baseline (capture fully
  green). Speed-up must come from **capture-safe node reduction**: K/V
  preallocation without cat, QKV/FFN GEMM merges, folded norm weights, minimal
  casts — nothing that depends on flash.

## 7. Implementation progress (L1, continuously updated)

| Item | Status | Measured |
|---|---|---|
| decoder: AdaRMS style precomputed outside the graph (370 dense calls/run removed) | ✅ in `fast.py` | — |
| decoder: QKV / GateUp GEMM merge (3+2 linears/layer → 2) | ✅ in `fast.py` | — |
| decoder fast segment captured median | 103.3 → **90.5 ms** (-12%) | eager 365→323; frontend p50 ~202→~187 ms |
| decoder: K/V preallocated, no cat (`fill_kv_prefix` + `decoder_step_fast_nocat`) | 90.5 → **84.5 ms** (another -7%, cumulative -18%) | frontend p50 ~187→**~182 ms** |
| accuracy (after L1a+b) | fp32 golden raw cos=0.999998 (same as baseline) | pytest 8 passed |
| encoder QKV/GateUp merge attempt | ❌ **negative, disabled**: captured 27.7→30.1 ms (a single 2560-wide GEMM loses to split linears on alignment); `encoder_pass_fast` kept for re-evaluation with cast/norm folding | — |
| attention in native dtype bmm (drop 3×fp32 cast + fp32 bmm) | ✅ kept: CPU fp32 reference semantics unchanged (inputs already fp32); bf16 on NPU uses Cube; small gain on large-S encoder (encoder_fast 30.1→28.8), neutral on small decoder shapes; cos=0.999998, 8 passed | — |
| Conclusion: plain-torch L1 tweaks nearly exhausted | remaining single-segment gains < noise band (±3–5 ms, indistinguishable across processes); reaching ~30 ms needs L3 fused kernels | — |
| decoder: K/V preallocated no-cat | ✅ in `fast.py` (L1b) | another ~7%, cumulative -18% |
| encoder: cast reduction + norm-weight fold + QKV merge | ⏳ QKV merge proven negative; cast/norm folding to be evaluated together | — |
| vision: LN/GELU cast removal, QKV merge | ⏳ later | — |

Architecture: the reference math (`pipeline.py`, fp32 CPU `sample` = golden
source) stays untouched; the serving-side decoder optimisations live in
`flash_rt/npu/models/pi05/fast.py` (`make_fast_weights`, `make_styles`,
`decoder_step_fast`), used by the frontend `_CapturedRunner` by default.

## 8. Acceptance and guardrails (every change must pass)

1. Segment timing is fixed (this table is the baseline): re-measure the
   median with the same protocol after every change and A/B against the
   154.3 ms baseline.
2. Correctness: bit-identical to the previous BF16 output when the op order
   is unchanged; fp32 eager reference cos ≥ 0.9999 when order/reduction
   changes (currently 0.999998, enough headroom for one fusion).
3. Latency: same-process alternating medians of ≥50 runs; report -X ms with
   cos not regressing.
4. Capture safety: every new op passes an **isolated NpuGraph capture probe**
   before entering the real graph (SDPA lesson: failure = abort; validate in
   /tmp scripts, never by editing the graph directly).
