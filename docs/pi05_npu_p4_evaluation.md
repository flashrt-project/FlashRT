# P4 Evaluation: Ascend C Custom Fusion Kernels (Go/No-Go analysis)

> Scope: ready-made fused-op reuse already delivers **~86 ms**
> (`docs/npu_pi05_optimization_report.md`). P4 must answer whether the
> cross-op fusions the official library does not cover are worth writing as
> **Ascend C custom kernels**, what they could achieve, and how the biggest
> risk — getting a custom kernel into torch and captured by `NpuGraph` — gets
> verified.

---

## 1. What is still worth fusing (current captured profile)

| Segment | ms | dominant nodes per layer-step after fusion (remaining) |
|---|---|---|
| vision 27L | 18.4 | manual attention hd=72 (27×~9 nodes), FFN two linears + gelu, patch/residual, LN (already fused) |
| encoder 18L | 21.6 | qkv/o/gate/up/down 5 linears (Cube), qkv rope (table-based), pfa (fused), residuals, add_rms (fused) |
| decoder 18L×10 | 46.1 | per layer-step: qkv/o/gu/down 4 linears, q/k rope ~8 nodes each, 2 slice-writes of KV, pfa (fused), gelu+mul, gate×residual×2 |

Candidate kernels (cross-referenced with CUDA/AMD hand-written glue and the
K-table in `docs/pi05_npu_l3_plan.md`):

| Kernel | Fuses | Applies to | Expected |
|---|---|---|---|
| K2 `qkv+rope+KV-write` | q/k rotation + slice-write into the KV slots (removes ~16 rope nodes + 2 copies) | decoder (180×)/encoder | ~15–20 nodes saved per decoder layer-step → estimate 8–15 ms whole-frame |
| K4 cross-layer `gate×down+residual+next norm` | layer-tail gate·residual + next layer's first norm in one kernel | decoder | 3–4 nodes/layer → estimate 3–6 ms |
| K1 Ada apply chain (RMS already npu; remaining scale/shift/gate splits and casts) | post-norm scale+shift+gate and casts merged | decoder | 2–3 nodes per layer-step → estimate 2–5 ms |
| K6 vision LN/gelu/FFN glue | glue between LN, gelu and the two linears | vision | manual attention still dominates vision; small FFN-side gain |

**Rough estimate: K2+K1+K4 all done → BF16 ~86 → roughly 55–70 ms** (this
rests on the "in-graph node scheduling is the cost" hypothesis; the larger
the fusion the closer to that upper bound. If node scheduling is not the only
cost, the real gain is lower.) Treat these as "worth-investing upper bounds",
not commitments.

**Judgment: real, but modest headroom (~15–30 ms) — and only after the
channel is proven.**

## 2. The decisive risk: how a custom kernel gets into torch_npu and captured
by NpuGraph

This is P4's Go/No-Go gate. It is **unverified** and the past lessons are
strong (aclop is not capturable; flash failure aborts the process). Known
integration lanes:

| Lane | Description | Risk / effort |
|---|---|---|
| A) msopgen aclnn-style single op → torch binding | official path: write Ascend C → msopgen produces an op package → install into OPP / provide an aclnn API → still needs registration into torch_npu | heaviest; the torch-binding step has no verified shortcut and capture in NpuGraph is unproven |
| B) torch_npu custom-op registration (`torch.library` / C++ custom op) wrapping an **aclnn/custom single op** | if the op executes as aclnn on the main stream only → plausibly capturable | requires confirming the torch_npu custom op lands on aclnn, not aclop |
| C) no new op: let the acl fusion pass merge aclnn ops | CANN ships fusion passes (e.g. `libops_fusion_pass_aicore.so`) for specific patterns | uncontrolled; must be tried per pattern; random gains |

**P4 spike's sole goal = verify lane B (or a light version of A)**:
- write a **minimal Ascend C kernel** (e.g. an elementwise `(x,a,b) -> a*x+b`,
  a few lines);
- register it via B so torch_npu can call it successfully;
- capture it in a single-node `NpuGraph` and replay successfully (the key
  gate);
- if B fails, try the minimal version of A; if both fail → **P4 custom
  kernels are No-Go**, move to INT8 / closure evaluation.

Success criteria: all three steps above pass, the kernel's replay on a real
graph matches eager, and the golden cos does not degrade.

### Stage-1 result (2026-09-08) — channel de-risked
A minimal torch C++ extension op that issues a raw `aclrtMemcpyAsync`
(device→device) on the **current** npu stream was built (g++ + ninja +
`torch.utils.cpp_extension`, linked against `libascendcl`):
- eager call equals a normal copy (`torch.equal` True);
- it is **captured by `NpuGraph` and replays bit-equal**;
- captured single-node replay median ~0.099 ms.

### Stage-2 result (2026-09-08) — real Ascend C kernel end-to-end
A real vector kernel (community silu sample, compiled with the official
compiler) now runs on the NPU through a torch extension and is captured:
- kernel build: `bisheng -fPIC -shared -xcce -O2 --cce-soc-version=Ascend910B4
  --cce-soc-core-type=VecCore -I<tikcfw> -o lib<name>_ascendc.so <name>.cpp`
  (core type is **VecCore** for 910B4);
- host: kernel `<<<block_dim, nullptr, stream>>>(...)` launch exported as
  `extern "C"`, torch extension calls it on `c10_npu::getCurrentNPUStream`;
- eager result matches the torch reference (fp16 maxdiff ~2e-3) and the op is
  **captured by `NpuGraph` and replays matching**.

Conclusion: the whole custom-kernel path (compile → torch → capture) works on
this box. K2 (and every later kernel in `flash_rt/npu/ops/`) is now a normal
kernel-implementation task, no longer a toolchain risk.

## 3. If custom kernels work: implementation plan (time-boxed)

- **Start with K2** (largest gain, cleanest boundary: inputs q/k + position/
  table + KV slots → outputs written into the slots). Implement to the FlashRT
  `qkv_split_rope_kvcache` boundary (bf16, GQA kv=1, row-write semantics
  matching our buffer layout).
- Numerics: fp32 golden cos ≥0.9999 plus bit-identical comparison against the
  current `rope_fast` output (same order).
- Per-kernel flow: isolated bench (median outside any graph) → shadow-path
  A/B → whole-frame paired ≥50 runs → pytest.
- Scope ceiling: K2 → (if clearly beneficial) K1/K4; K6 and hd=72 attention
  are not custom kernels (high cost, low return).

## 4. Alternatives if we do not write custom kernels (side-by-side)

| Route | Expected | Cost / risk |
|---|---|---|
| Accept ~86 ms as the BF16 ceiling (productise / close out) | — | 0 |
| 910B-native **INT8 calibration** (≈FP8's ~2×) | ~86 → target ~45–55 ms | quantisation framework (reuses fp32 golden + cos gates), separate 2–4 weeks |
| P4 custom kernels (only if the spike passes) | ~86 → ~55–70 ms | channel risk high; 2–4 days per kernel |
| Move to 910C and add an FP8 tier | hardware-level ~2× starting point | hardware not in hand |

**Recommended ordering: do the P4 spike first (≤2 days, sole purpose is to
answer the channel question) → if it passes, build K2 and proceed; if it
fails, promote INT8 calibration as the next route (its ceiling is above the
custom-kernel route).**

## 5. Explicit decision points

1. Invest ≤2 days in the P4 spike? (Recommended: yes — low cost, answers the
   channel question once.)
2. If the spike passes, continue with K2 (time-boxed, real gain measured
   within a week)?
3. If the spike fails, start the INT8 calibration evaluation?

Whether P4 proceeds is decided only by the spike result plus the time-boxed
K2 measurement — no unbounded custom-kernel work.
