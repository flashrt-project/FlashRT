# NPU Kernel Development Standards (Ascend C)

Purpose: make the NPU backend's custom-operator work follow the same
engineering shape as the mature CUDA and AMD implementations, so the three
backends stay mutually comprehensible, reviewable and extendable. These
standards mirror the existing repo conventions (see the references column)
and encode the lessons proven on CANN 8.5.2 / Ascend 910B4.

## 1. Mirror principle

Every NPU structure has a CUDA and an AMD twin to copy from:

| Concern | CUDA | AMD | NPU (this standard) |
|---|---|---|---|
| native kernel sources | `csrc/kernels/*.cu` | `csrc/amd/kernels/*.hip` | `csrc/npu/kernels/*.cpp` (Ascend C, `kernel_operator.h`) |
| host-side glue / tiling | `csrc/kernels/*.cu` host parts | `csrc/amd/kernels/*.hip` host parts | `csrc/npu/op_host/*.cpp` (tiling/host) |
| library GEMM runner | `csrc/gemm/gemm_runner.cu` | `csrc/amd/gemm/hipblaslt_runner.hip` | none — GEMM stays in CANN aclnn/Cube (see §3) |
| attention backends | `csrc/kernels/attention_*.cu`, fa2 | `csrc/amd/attention/*.hip` | aclnn `npu_prompt_flash_attention` first; custom only for gaps |
| pybind/bindings surface | `csrc/bindings.cpp` (528 `m.def`) | `csrc/amd/bindings.cpp` (46) | `csrc/npu/launcher.cpp` (torch C++ extension) + `flash_rt/npu/kernels/` python wrappers |
| python model pipeline | `flash_rt/models/pi05/pipeline_*.py` | `flash_rt/amd/models/pi05/pipeline.py` | `flash_rt/npu/models/pi05/{pipeline,fast}.py` |
| python backend package | — | `flash_rt/amd/{core,frontends,hardware,models}` | `flash_rt/npu/{core,frontends,hardware,models,ops}` |
| docs / catalog | `docs/kernel_catalog.md`, `docs/kernel_fusion.md` | (mirrors) | `docs/npu_kernel_development_standards.md`, `flash_rt/npu/ops/README.md` |

Naming rule: keep the **semantic role** of the CUDA/AMD kernel name where the
dataflow is identical (e.g. AMD `qkv_split_rope` / CUDA `rope.cu` family →
NPU `rope_kv_cache`), and only append NPU-specific suffixes when behaviour
differs. Do not invent a parallel taxonomy.

## 2. What to REUSE from the official operator library (reuse-first)

An official (CANN/torch_npu) op is used — not a custom kernel — when it
satisfies **all** of:

1. exists for our dtype/shape (verified by a probe, not assumed);
2. is **capturable** by `torch.npu.graph` in an isolated subprocess probe;
3. is semantically equivalent to the manual chain (cos ≥ 0.9999, or the
   documented mismatch is acceptable at the gate);
4. is not measurably slower than the manual chain inside a captured graph.

Proven reusable (already adopted for pi05):
`npu_rms_norm`, `npu_add_rms_norm` (norm + residual), `npu_add_layer_norm`,
`npu_prompt_flash_attention`. All GEMMs go through CANN aclnn/Cube.

**Never** reuse-by-assumption: every candidate is individually probed.
Known traps: aclop ops (e.g. `conv2d`, `npu_layer_norm_eval`) cannot be
captured; flash/SDPA-class ops can abort the process on a failed capture.

## 3. What is SELF-DEVELOPED (custom kernel scope)

A custom Ascend C kernel is written only for **memory-bound glue that the
official library does not ship**, i.e. a fusion across op boundaries whose
components are individually reusable but not fused anywhere. The kernel
backlog in `flash_rt/npu/ops/README.md` (K2 `rope+KV-write`, K1
`ada-norm-tail`, K4 `gate×residual+next-layer norm`, vision-hd-attention
variants) is the working list.

Red lines for custom kernels:
- **Never write a matmul/Cube kernel** — GEMM always goes through aclnn.
- **Never write an aclop-style op** — it cannot enter `NpuGraph`.
- Every custom kernel is launched on the **current capture stream**
  (`aclrtCreateBinary → aclrtBinaryLoad → aclrtBinaryGetFunction →
  aclrtLaunchKernel` or an aclnn-opapi wrapper); sub-streams break capture.
- Additive only: never modify official ops, and keep the fp32 reference math
  (`flash_rt/npu/models/*/pipeline.py`) untouched as the golden source.

## 4. Kernel spec template (one block per kernel, mirrors `kernel_catalog`)

Each kernel gets a catalog-style spec before implementation:

| Field | Meaning / example |
|---|---|
| semantic name | what the CUDA/AMD twin is called (e.g. `qkv_split_rope_kvcache`) |
| boundary | input/output tensors, slots written, dtypes (bf16) |
| shape envelope | real shapes/dtype/layout/phase it covers (e.g. GQA kv=1, hd=256, chunk rows) |
| register/entry name | kernel + opapi entry as registered |
| capture | isolated `NpuGraph` probe result |
| reference | the CUDA/AMD kernel and file:line it mirrors, plus the manual-chain gate |
| acceptance evidence | eager-vs-captured bit-equal, golden cos, paired median, pytest |

## 5. Verification and acceptance (unchanged from the rest of the backend)

1. Isolated-subprocess capture probe first (a failed capture aborts the
   process — never probe inside the real process or graph).
2. Replay == eager (same order), or fp32 golden cos ≥ 0.9999.
3. Whole-frame paired A/B (alternating, ≥50) medians; no single-arm
   cross-process comparisons (noise ±3–5 ms).
4. `FLASH_RT_NPU_PI05_REF=... pytest tests/test_npu_pi05_model.py` green;
   golden cos never regresses.
5. Latency tripwire: `p50 < 350 ms` in the model gate.

## 6. Documentation rules (public docs in English)

- Every kernel ships: a catalog-style spec block (§4) and an entry in
  `flash_rt/npu/ops/README.md`.
- Cross-reference the CUDA/AMD kernel it mirrors (file:line) and any reused
  official op.
- State the CANN/torch_npu version facts were verified against (8.5.2) and
  re-verify after upgrades.

## 7. Delivery checklist per kernel

- [ ] spec block written and reviewed (boundary/dtypes/envelope/register)
- [ ] probe script: capturable + semantics cos vs manual chain
- [ ] kernel + host + launcher in `csrc/npu/…`; python wrapper in
      `flash_rt/npu/kernels/` (mirror AMD bindings style)
- [ ] eager == captured (same order) or golden cos ≥ 0.9999
- [ ] whole-frame paired A/B median reported, cos not regressed
- [ ] README/catalog updated in English
