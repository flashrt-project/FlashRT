# AGENTS.md — producing structures for the FlashRT structures layer

You are working on the structures layer: verified, host-independent
acceleration structures assembled from Hub kernels. This document is the
complete operating procedure — how to extract a structure abstraction
from FlashRT's native implementations, how to keep it general, how to
wire Hub kernels, what the red lines are, and what acceptance looks
like. Follow it literally. When unsure, stop and ask; do not guess.

---

## 0. Required reading, in order

| Document | What you take from it |
|---|---|
| `docs/structures.md` | The norm: what a structure is, the three-layer split, calibration reuse, accuracy bands, runtime contract and ledger, how to add structures/backends/hosts/schemes, and §7 — norms that came from being wrong |
| `docs/structure_contributing.md` | The external contribution boundary and copyable PR self-review checklist |
| `docs/calibration.md` | The house calibration standard (statistics, two-level reduction, diagnostics). You must not invent a second one |
| `catalog/*/structure.yaml` + `reference.py` | Worked precedents; copy the format of `decoder_ffn` |

Code landmarks under `flash_rt/structures/`: `catalog/` (spec +
reference), `bindings/` (per-host addressing receipts), `impls/`
(executable forms), `discover.py` (structural discovery),
`autobuild.py` (assembly), `points.py` (calibration collection),
`schemes.py` (quantisation schemes), `gates.py` (accuracy judgment),
`guard.py` + `swap.py` (runtime contract, ledger, attach/detach),
`frontdoor.py` (the gated one-call door).

### 0.1 One-model closure is the only allowed execution order

Work on **one target model at a time**. Before changing code, lock the
target model, its official host repository or repositories, the
corresponding FlashRT native pipeline, and the exact end-to-end
boundary. Do not move to another model because a local seam is easier
to test. A model may be left only when it is closed or explicitly
blocked on a missing Hub artifact.

The following stages are mandatory and may not be reordered:

1. **Build the native coverage map.** Read the native `pipeline_*`,
   frontend, kernel catalog, and model configuration. List every
   performance-relevant native region and classify it as:
   `configured`, `existing_structure_not_configured`,
   `missing_structure`, `host_stage_or_state`, or
   `intentionally_retained`. The map must name the native file and
   symbol, boundary, relevant shapes/dtypes/phases, and intended
   structure. No implementation or performance claim starts before
   this table exists.
2. **Prove generality.** For every uncovered native region, locate the
   same dataflow in at least one unrelated model or host family. If the
   catalog already describes it, add only the missing addressing or
   executable form. If a genuinely recurring region is absent, add a
   catalog spec and reference. Model-only behaviour remains in a
   binding or host stage; it never becomes a model-special structure.
   Record the cross-model evidence before implementation.
3. **Audit kernel readiness.** Audit only formally delivered,
   published Hub artifacts. If the required Tensor API, executable
   form, shape/dtype/layout/phase envelope, architecture build, or
   capture behaviour is absent, mark the region
   `blocked_on_kernel`, report it immediately, and stop work on that
   region. The report must contain:
   - native file and symbol;
   - structure boundary and real shape/dtype/layout/phase/hardware;
   - nearest Hub API and exactly why it is insufficient;
   - the generic API or artifact capability required;
   - all known consuming structures, models, and hosts;
   - standalone correctness, compile, capture, and E2E acceptance
     gates.

   The structures worker must not patch `FlashRT-HF-kernels`, native
   kernel sources, bindings, build files, package metadata, fake
   registrations, or kernel-package documentation. It must not inject
   a source checkout into a host to manufacture an artifact. Kernel
   packaging and validation belong to the kernel owner. Resume only
   after that owner supplies a commit, built artifact, and its
   correctness/compile/capture evidence.
4. **Integrate the structure.** Using the verified artifact, land the
   catalog/reference if needed, generic discovery or host binding, the
   executable form, configuration, and public contract tests. No
   model-ID branch, one-off model wrapper, private source injection,
   or undeclared fallback is allowed.
5. **Validate in this order, without skipping a rung:**
   1. reference and public CPU contract;
   2. local boundary correctness, target call count, and zero
      unexpected fallback;
   3. cross-model correctness and benefit on the second family used
      to prove generality;
   4. official target-host E2E on real inputs, comparing the
      unmodified strong host baseline with identical inputs/noise and
      the same compile/capture policy used for deployment;
   5. aligned comparison with the FlashRT native pipeline at the same
      E2E boundary and assembly policy;
   6. cross-host and cross-hardware cells when available;
   7. detach/null/negative controls.

   A native result at a different boundary is a ceiling or diagnostic,
   not a direct speedup comparison. A seam benchmark is not model E2E.
6. **Close the model.** Every native hot-path region must end as
   `e2e_verified`, `blocked_on_kernel`, or
   `intentionally_retained` with evidence. Run the final all-enabled
   configuration E2E and record the native, cross-host, and
   cross-model comparisons. Only then may work move to the next model.

Use these progress states literally:
`mapped` → `generality_proven` → `configured` →
`local_verified` → `cross_model_verified` → `e2e_verified` →
`native_compared` → `cross_hardware_verified`. A region may branch
from `generality_proven` to `blocked_on_kernel` until the formal
artifact arrives.

“E2E ran once” does not mean “configuration complete”. If a planned
attention form, cache path, cadence/state path, host stage, or other
native region remains uncovered, report the model as **incomplete**.
Every progress update uses one coverage table:

| Native region | Structure | Cross-model evidence | Hub artifact | Config | Local | Cross-model | Official E2E | Native comparison | Blocker |
|---|---|---|---|---|---|---|---|---|---|

Keep work in progress bounded to one model and one uncovered region.
When a region is blocked, report it immediately; continue only with
already-ready regions of the same model unless the user explicitly
changes scope.

---

## 1. Finding a structure abstraction in the native implementations

**Where the ore is** (same repository): the native pipelines under
`flash_rt/models/*/pipeline_*.py` and `flash_rt/frontends/torch/*.py`
are a record of fusion decisions that were already measured to pay;
`docs/kernel_catalog.md` and `docs/optimization-details.md` list the
kernels and their yields.

**What qualifies as a structure**: a region of dataflow that recurs
**across host families**, defined by four things — boundary tensors,
weight slots, calibration points, and gates. It is never one host's
module name.

**Procedure**:
1. In a native pipeline, list the fusion decisions: which ops were
   merged into one kernel, who shares a scale with whom, what is cached
   at observation cadence. Each such decision is a candidate structure.
2. Take the candidate to **at least two unrelated host families** and
   find the same stretch of dataflow. Found in both → it is a
   structure. Found in one → it is a binding specialisation and does
   not enter the catalog.
3. Name calibration points by **position in the structure's own
   dataflow** — `act_after_mul` (after the gated activation) means the
   same thing in a GGML graph as in a torch module tree. Never a name
   that only exists on one host.
4. Write `catalog/<name>/structure.yaml`: boundary (symbolic dims,
   dtype may be `"@binding"`), weights (framework-neutral slot names),
   `calibration.points`, gates. **Statistics do not go in the spec** —
   they belong to the quantisation scheme (`docs/structures.md` §2).
5. Write `catalog/<name>/reference.py`, a plain-torch reference.
   A structure without a reference cannot gate anything and is not a
   structure.

**Discovery is semantic, not name-blind.** Discovery rules may use
semantic slot names (`gate_proj`/`up_proj`/`down_proj` are a gated-MLP
signature), tensor shapes, dataflow relations, and forward signatures;
a non-standard host gets a family adapter. What is forbidden: matching
on model IDs, on concrete class-name whitelists, or on one incidental
module name as the only evidence.

**Generality is proven, not claimed**: run discovery on two families
and record the results as evidence — and verify the negative case too.
A structure must *not* be discovered on hosts it does not describe;
"correctly not found" is an acceptance item with test precedent.

### 1.1 The adapter contract (both review rejections were adapters)

An adapter is the wing that touches the host's API, and it is where
work gets returned. The paradigm that survived review, as a contract:

1. **Structural predicates, never identity.** Recognition reads slots,
   shapes and parameter presence (`in_proj_qkv` + `conv1d` + `A_log` +
   the 48/16-head profile *is* a fused gated-delta layer; a
   `scale_shift_table` parameter + 4-D `temb` *is* a per-token
   modulated block). Class names and model IDs are forbidden evidence.
2. **Refuse with the reason, and let the ladder fall.** Out-of-profile
   hosts and packages predating an entry raise `ValueError("refused:
   ...")` once; registration order is the ladder (fused form before
   rule-level form), and a refusal is a routing event, not an error.
3. **Resolve at bind, freeze in a closure.** Weights are detached,
   cast, packed and smoke-tested at bind time; the forward touches
   only what the closure captured. A bind-time smoke on zeros is the
   difference between a clean bind refusal and a crash inside the
   host's forward.
4. **Masks, scales and positions are explicit.** Whatever the host
   passes (attention masks, cache positions, rope deltas) is either
   handled or named in the refusal. Two receipts to remember: a cache
   that misreports its progress sends host glue down the continuation
   branch, and a KV slot index is not a rotary position on multimodal
   hosts.
5. **Declare capabilities, don't sniff.** An adapter that wants the
   active scheme sets `scheme_aware = True` and takes it as a kwarg;
   probing call signatures (or catching TypeError) hides real errors.
6. **Host cache contracts are followed, not replaced.** Slots are
   written in place with the host's own semantics (last-K raw inputs,
   final state); repointing a slot strands every captured graph on the
   old tensors — the repeat-identical gate exists to catch exactly
   this.

The template test for a new adapter pins: the positive shape case, one
negative shape case (out of profile → clean refusal), the ladder
fallthrough (stale package → next adapter), and revert/toggle
round-trip. `tests/test_structures_gated_delta_core.py`'s fused
adapter pins are the copyable precedent.

---

## 2. Finding and wiring Hub kernels

1. **Shop before building**: check the `flashrt/*` and
   `kernels-community/*` Hub organisations for an existing package.
2. **Loading convention**: always go through the shared
   `impls.hub_kernel(repo, version)` helper. Loading the same package
   twice re-registers its fake ops; the loader must be shared.
3. **Resolve ops at bind time; the swapped forward must be a single
   custom-op call.** Calling a hub loader inside `forward` drags the
   version resolution into the compiler's trace and fragments the
   graph (a measured 26-graph-break incident).
4. **Standalone-bench the kernel before wiring it** at the host's real
   shapes, and separate kernel time from launch time — an eager
   preflight win can evaporate inside a captured graph.
5. Shapes the kernel does not cover get **declared dispatch** (the
   weight-only FFN precedent: decode band to the kernel, prefill back
   to the host, both counted separately in the ledger) — never a
   silent stretch of the kernel's envelope.
6. If the kernel you need does not exist, stop and report "a kernel is
   missing", with the shapes and the roofline. Do not emulate it with
   a chain of eager ops; a merge that is not one real kernel loses to
   the compiler's own fusion.
7. **Hardware support comes from the package, not from you.** The Hub
   package's own metadata declares the archs it was built for, and the
   shared loader enforces it with a clean refusal. Do not write arch
   tables into an impl — a second table drifts, and hardware support is
   maintained on the kernels side.
8. **A binder that loads a kernel runs a bind-time smoke**: one
   real launch through the entry point at the seam's own width before
   the seam is handed out (`w4a16_static.bind_mlp_seam` is the
   precedent). A stale build or missing symbol must surface as a bind
   refusal — in a fallback-capable system it can never be caught by
   comparing outputs, because falling back is numerically exact.
   The smoke's input must carry the host's real rank and shape class —
   a minimal 2D fixture stays green against a kernel whose contract is
   2D-only while every real host hands it 3D (the norm_fused
   incident).
9. **Device differences absorb at three tiers, and only one of them
   is your code.** When a kernel package gains per-device work, decide
   which tier it is before touching an impl:
   - *Same symbols, new arch build* (a package adds an SM110 variant
     of entries it already ships): nothing to do. The package's
     metadata unlocks the device; the impl binds unchanged.
   - *New capability entries inside the seam* (a fused-bias epilogue,
     a direct BF16 quantize — better bodies for work the impl already
     does): add a **capability probe**, once. `getattr(kern, "entry",
     None)`; prefer the entry when the installed package ships it,
     keep the existing path when it does not — absence is a fallback,
     never a refusal (`nvfp4_dynamic`'s GEMV and BF16-quantize probes
     are the precedents). One probe serves every device forever; no
     new impl variant.
   - *Fusion that crosses a seam boundary* (an epilogue that emits the
     next GEMM's quantized input): that is a new executable form —
     an impl variant or a wire-dtype negotiation, judged by the same
     gates as any variant.
   Variant count grows with executable forms, never with devices;
   the bulk of per-device tuning must stay contained in the packages'
   build-variant distribution. The kernels-side half of this bargain:
   within one package version, an entry's signature is invariant
   across archs, and capability growth extends the symbol surface
   without touching existing entries.

---

## 3. Landing one structure, step by step

1. **Read** the native implementation and the target host source. List:
   boundary tensors, weight slots, fusion decisions, and any state that
   changes with the observation — that state must be handled explicitly
   (see red lines).
2. **Spec + reference** (§1, steps 4–5).
3. **Addressing**: hosts whose slots are structurally findable go into
   `discover.py` rules; others get `bindings/<host>.yaml`. Binding
   YAMLs are the *receipt* of an addressing decision; structural
   discovery is the runtime mechanism.
4. **Impl** in `impls/<name>/<backend>.py`. It must:
   - subclass `GuardedSeam` and declare its executable form via
     `_frt_arm(dtypes/device/k/rows)`. A row-locked structure on a
     variable-length host falls back by contract — that is correct
     behaviour, and it must be documented on the impl;
   - state its weight-layout convention in the binder's docstring.
     Two binders with different layout conventions have already
     collided once, with the dimension check passing under swapped
     names — the convention must be written where the next caller
     will read it;
   - retain the original host module (fallback and `state_dict`
     delegation depend on it).
5. **Calibration**: through the `points.py` collector and the scheme
   interface only. Do not hook activations yourself; if you need a
   granularity the collector cannot measure, extend the collector —
   the loud failure you hit is intentional.
6. **Tests** — every new structure ships **public CPU contract tests**
   in `tests/` alongside the PR:
   - reference correctness against the spec;
   - positive discovery on a synthetic host of the right shape;
   - negative discovery: it must not fire on hosts it does not
     describe;
   - guard dispatch and fallback behaviour;
   - attach/detach reversibility (bit-exact restore, module count
     unchanged);
   - clean refusal when a capability is missing.
   GPU validation on real hosts follows the same ladder — per-seam
   gate at the declared boundary (residual included), end-to-end
   same-input/same-output against the unmodified host, held-out
   evaluation with a null check (rerunning the unmodified host must be
   bit-identical) and a negative control (deliberately break the
   refresh or window; the metric must visibly degrade, proving the
   test can detect the failure), a clean ledger, and **evidence that
   the kernel path actually ran** — in a fallback-capable system,
   identical output alone proves nothing, because falling back to the
   host is numerically exact.
7. **Data**: calibration and evaluation inputs must come from the
   host's real inference distribution, built through the host's own
   preprocessing chain (`docs/structures.md` §7). Out-of-distribution
   or synthetic inputs have mismeasured hosts here before.
8. **Reporting**: `plan.report()` / the plan notes are the receipt —
   discovered, activated or refused and why, band, measured speedup,
   calibration method, ledger. Latency claims come from paired
   alternating timing; single-arm wall-clock drifts several percent
   and is not accepted.

---

## 4. Red lines (any violation returns the work)

1. **Additive only.** Existing kernels, bindings, loaders and the
   catalog schema are not modified. Changing a shared helper's return
   type requires grepping every reader first.
2. **Grep before building.** Reuse the repo's existing mechanisms
   (calibration, diagnostics, sampling, receipts). Claiming something
   is "not implemented" requires grep evidence.
3. **No new calibration entry points, and no new precision entry
   points.** The calibration axis is `forward`/`samples`, once. A
   scheme declares and consumes statistics; it does not open a second
   door. Precision profiles are registered schemes selected by the
   existing `scheme=` parameter (`docs/structures.md` §4.1) — a new
   precision mode is a scheme registration, never a new parameter.
4. **Fail loudly; never degrade silently.** Unmeasurable granularity,
   unknown format variants, unlocatable points — raise, with the
   reason. No silent approximations.
5. **Identical output is not evidence on its own** — pair it with the
   ledger's fallback count and the target path's call count.
6. **The forward you swap in must be compiler-friendly**: no Python
   side effects in the hot path, no loader calls, one custom-op call.
7. Every API mentioned in docs or a PR description is checked against
   the source signature before it is written down.
8. **Kernel ownership is strict.** Structures work may inspect and
   report a kernel-package gap, but it may not implement, repair,
   package, publish, or validate that package on the kernel owner's
   behalf. A local kernel edit is not a delivered dependency.

---

## 5. Deliverables and acceptance

**You deliver**:
1. the completed native coverage table, including every blocked or
   intentionally retained region;
2. the repository diff — additions under `catalog/`, `bindings/`,
   `impls/`, `tests/`, `docs/` only;
3. the public CPU contract tests, green;
4. a one-page report: the results table (host / baseline stating eager
   or compiled / result / speedup / output match with worst case),
   the data-source statement, and the null-check / negative-control /
   ledger / detach results.

**The reviewer will**: run all tests; grep every API signature you
cite and every "not implemented" claim; check the data is the host's
real distribution; check the ledger is clean and the kernel path was
exercised; probe generality on a third host (correct discovery or
correct absence); and scan the diff for anything that does not belong
in a public repository.

## 6. Calibration discipline (hard rule, incident-tested)

Every activation statistic in this repository — static-scale amax,
AWQ channel vectors, anything a producer or GEMM consumes — flows
through the house calibration machinery, never a private copy:

- Samples enter through ``auto_swaps(observations=...)``; a region
  bind's ``probe`` carries them with per-sample boundaries exposed
  via ``probe.samples``. A chain may register its own collection
  hooks at bind, but the statistic itself is the house two-level
  reduction: max over calls within one sample, then
  ``flash_rt.core.calibration.accumulate_amax`` across samples.
- Calibration data is the host's real input distribution, with
  provenance (source, frame indices, basic statistics). Synthetic or
  random tensors are never calibration data and never a fidelity
  reference.
- Sample count is not a fidelity remedy. One real frame is a valid
  calibration; more samples change *recipe selection* (channel
  statistics drive layer subsets and therefore which form binds),
  which is a different effect and must be reasoned about as such.
- Attribution requires a controlled experiment. When a fidelity or
  latency shift spans more than one changed variable, the record says
  "correlates" — "caused by" is earned only by re-running with a
  single variable moved.
- Smoke floors are load-bearing. A band may carry its own floor only
  together with the end-to-end parity judge, and no receipt is ever
  written from a floor-relaxation experiment.
