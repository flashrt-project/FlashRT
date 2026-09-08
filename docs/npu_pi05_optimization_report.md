# Pi0.5 NPU inference: optimisation report from ~200 ms to ~86 ms (M1/M2/L1 + L3-P0 closure)

Date: 2026-09-08 · Hardware: Ascend 910B4 (Atlas 800I A2) · CANN 8.5.2 ·
torch 2.7.1 + torch_npu 2.7.1.post2
Protocol: 2 views · 40-token prompt · 10 flow steps · BF16 · full-frame
`torch.npu.graph` captured replay

## 1. Results summary

| Metric | Value |
|---|---|
| **Frontend infer p50** | **~86 ms** (started ~202 ms, -57%) |
| Captured segments | vision 18.4 / encoder 21.6 / decoder10 46.1 ≈ 86 ms |
| Accuracy | fp32 CPU golden raw cos = **0.99999x** (gate ≥0.9999) |
| Determinism | same-noise captured replay is bit-identical |
| Gates | `tests/test_npu_pi05_model.py` + `test_npu_arch_gate.py` **8 passed** (tripwire 350 ms) |

Historical anchors: torch_npu eager ~0.45–0.56 s (host-bound); plain capture
(reference op order) ~0.15–0.20 s; this work ~0.086 s. References: CUDA eager
~117 ms (repo docs); AMD native BF16 ~30 ms (repo docs, hand-written MFMA +
fused kernels).

## 2. Contribution of each step (frontend p50, incremental)

| Stage | Change | p50 |
|---|---|---|
| start | captured baseline (reference op order) | ~200 ms |
| M3-L1 | decoder style precomputed outside the graph + QKV/GateUp merge + K/V no-cat + bf16 attention | ~182 ms |
| L3-P0 | encoder fused norms (npu_rms_norm / npu_add_rms_norm) | ~139 ms |
| L3-P0 | decoder AdaRMS norm fusion (style split into gamma/shift/gate + npu_rms_norm) | ~110 ms |
| L3-P0 | vision LayerNorm fusion (npu_add_layer_norm as a plain LN) | ~105 ms |
| L3-P0 | RoPE cos/sin table precompute + rotation minimisation | ~95 ms |
| L3-P0 | attention fusion (npu_prompt_flash_attention replaces manual bmm) | **~86 ms** |

Every fusion passed three gates: **isolated-subprocess capture probe**
(failure = abort; never infects the main process) → **fp32 golden cos ≥ 0.9999**
→ **pytest all green**.

## 3. Key lessons (transferable to other models)

1. **Nature of the bottleneck**: on the NPU the eager/captured cost is
   dominated by "per-op python/aclnn dispatch + in-graph node scheduling",
   not compute (large GEMMs measure 204 TFLOPS). Optimisation is therefore
   **node-count reduction**; every fused chain wins.
2. **Prefer ready-made fused ops** (CANN/torch_npu): `npu_rms_norm`,
   `npu_add_rms_norm`, `npu_add_layer_norm`, `npu_prompt_flash_attention`
   replace 7–15 manual ops with a single node.
3. **Capture red lines**: ① aclop is not capturable (conv2d,
   npu_layer_norm_eval); ② flash/SDPA-class ops are not capturable and a
   failure aborts the process → every candidate op must be verified in an
   **isolated subprocess** before entering a graph; ③ precompute outside the
   graph (style/cond, cos/sin tables, language embeddings) to eliminate
   host→device copies during capture.
4. **Numeric layering**: the reference math (`pipeline.py`, fp32) stays
   untouched as the golden source; all optimisation lives in `fast.py`
   (serving side) and is gated against the fp32 golden rather than comparing
   optimised builds with each other.
5. **The big wins come from whole-graph node count**: the encoder fusion saved
   only ~3 ms as a segment but ~43 ms of full-frame p50 — node-scheduling cost
   is amortised across the whole graph; re-measure at full-frame paired, never
   single-segment.

## 4. Files and artefacts

- Reference math: `flash_rt/npu/models/pi05/pipeline.py` (fp32, untouched)
- Optimised serving path: `flash_rt/npu/models/pi05/fast.py`
- Frontend/gates: `flash_rt/npu/frontends/torch/pi05.py`
- verify: `flash_rt/npu/verify.py` (reuses the official
  `structures.gates.parity_metrics`)
- Golden: `checkpoints/pi05_libero_pytorch/npu_fp32_reference.npz` (produced
  by `scripts/npu/make_pi05_reference.py`)
- Docs: `docs/deployment_npu.md`, `docs/pi05_npu_fusion_plan.md`,
  `docs/pi05_npu_l3_plan.md` (and this report)

## 5. Reproduction

```bash
# all-green gates (need NPU + checkpoint)
FLASH_RT_NPU_PI05_REF=<ckpt>/npu_fp32_reference.npz python -m pytest \
    tests/test_npu_arch_gate.py tests/test_npu_pi05_model.py -q

# quick latency/accuracy
python - <<'PY'
import flash_rt, numpy as np
from flash_rt.npu import verify
m = flash_rt.load_model("<ckpt>", config="pi05", hardware="npu", num_views=2)
fe = m.pipeline
fe.set_prompt(verify.CANONICAL_PROMPT)
for _ in range(15): fe.infer(verify.canonical_images())
print(fe.get_latency_stats()["p50_ms"])
PY
```

## 6. Next steps (see `docs/pi05_npu_p4_evaluation.md`)

- **P4 custom Ascend C kernels — OUTCOME (2026-09-08, see
  `flash_rt/npu/ops/README.md` "Empirical device matrix")**: the bespoke
  kernel channel was proven end-to-end (compile → torch → `NpuGraph`
  capture), but on 910B4 it is limited by law to 16-bit GM traffic and fp16
  vector math: bf16 vector math does not exist and every fp32 path (GM
  copies, count wrappers, Level-0 mask/repeat ops) faults or is inert. The
  targeted ~40–60 ms via cross-op fused bf16 kernels is therefore **not
  reachable on this stack**; the official fused RoPE/RMSNorm+cache family
  (`npu_kv_rmsnorm_rope_cache_v2` etc.) also fails at runtime on this
  CANN/torch_npu build. One validated 16-bit glue kernel landed as a library
  capability (`kv_cache_store`) and A/B showed it slower than captured aclnn
  copies on the pi05 hot path (mode 0 default). Re-open only after a
  CANN/torch_npu version change or an fp16-lane decision.
- **INT8 calibration route**: a 910B-native ~2× quantisation tier (separate
  programme).
- Reaching the CUDA/AMD ~30 ms / 16 ms tiers requires hand-written-class fusion
  investment (their years of MFMA tuning). This report's commitment stops at
  the achieved ~86 ms; further progress is driven by measurement and effort.
