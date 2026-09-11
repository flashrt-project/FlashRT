# Contributing to the structures layer

This is the contributor entry point and pull-request self-review standard for
`flash_rt/structures/`. It applies to external and internal contributors. Its
goal is to keep additions portable across hosts, explicit about unsupported
cases, and removable without changing the host model.

The normative design is [`structures.md`](structures.md). Release performance
claims additionally follow
[`structure_release_qualification.md`](structure_release_qualification.md).
This document does not replace either contract; it tells a contributor what a
reviewable change must contain.

## 1. Choose the layer before changing files

Put each fact in the narrowest owner below. If a change needs two layers, keep
the boundary visible in the diff rather than teaching either layer about the
other one's concerns.

| Fact or behavior | Owner | Must not contain |
|---|---|---|
| Framework-neutral dataflow boundary, slots, state, calibration points, gates | `catalog/<structure>/` | Model IDs, host class names, GPU architecture, package selection |
| Where that boundary exists in one host | `discover.py` or `bindings/*.yaml` | Kernel choice, benchmark policy, device tuning |
| Executable form for a structure | `impls/<structure>/` | Model routing, private source loading, duplicated calibration |
| Engine lifecycle integration | `adapters/` | Structure definitions or model-specific kernels |
| Precision/statistics policy | `schemes.py`, `points.py`, `gates.py` | A second calibration API or hidden defaults |
| Runtime replacement, refusal, ledger, rollback | `guard.py`, `swap.py`, `frontdoor.py` | Silent fallback or process-global policy |
| Hardware support and entry-point availability | Published kernel package metadata | A second architecture table in FlashRT |

A recurring boundary found in at least two unrelated host families may be a
structure. A model-only path is a binding or host stage. A faster kernel for an
existing boundary is an implementation. A new device build with unchanged
symbols normally requires no structures-layer change.

## 2. Required reading by change type

All structure changes start with [`structures.md`](structures.md). Then read
only the material needed for the proposed surface:

| Change | Also read |
|---|---|
| Catalog, discovery, binding, implementation, or adapter | Repository [`AGENTS.md`](../AGENTS.md), especially its ordered workflow and red lines |
| Calibration or precision | [`calibration.md`](calibration.md) |
| Engine integration | [`hosts.md`](hosts.md) and [`serving_engines.md`](serving_engines.md) |
| A release or performance claim | [`structure_release_qualification.md`](structure_release_qualification.md) |
| Public API or packaging | [`stable_api.md`](stable_api.md) and the root [`CONTRIBUTING.md`](../CONTRIBUTING.md) |

Before adding a mechanism, search the catalog, discovery rules, schemes,
adapters, diagnostics, and tests. Reuse an existing structure or executable
form when its contract fits. Keep a PR focused on one structure boundary or
one independently reviewable behavior; kernel-package work belongs in its own
repository and review.

## 3. Contribution workflow

### 3.1 State the contract first

The PR description must name:

- the structure boundary and why it belongs at this layer;
- affected hosts, models, phases, shapes, dtypes, layouts, and devices;
- the baseline behavior, including whether unsupported cases refuse or retain
  the host path;
- files and public behavior intentionally not changed.

For a new structure, record the same dataflow in two unrelated host families.
If only one host is known, keep the work in a binding or mark the catalog entry
provisional; do not encode the first host as a universal abstraction.

### 3.2 Implement through existing extension seams

- Additive changes are preferred. Do not change a shared schema or helper
  contract without enumerating and testing every reader.
- Recognition uses slots, shapes, dataflow relations, and forward contracts;
  model IDs and concrete class-name allowlists are not discovery evidence.
- Resolve, detach, cast, pack, and smoke-test at bind time. The installed
  forward must contain no resolution: no loader call, symbol lookup,
  capability probe, or plan construction. A route decided at bind and read
  per call, such as a declared shape band or an already-chosen variant, is not
  bookkeeping; it is the form that was bound.
- Use the shared Hub loader. Symbol presence is a capability hint, not proof
  that a shape, dtype, layout, or architecture is qualified.
- Route unsupported cases through an explicit refusal or declared host
  fallback. Record the reason in the ledger.
- Use the house calibration collector and scheme interface. Do not add private
  hooks, statistics, caches, or precision entry points.
- Preserve the original host module and prove detach restores it exactly.

When a required published kernel capability is absent, stop that region and
report `blocked_on_kernel`: nearest public API, missing signature or envelope,
real shapes/dtypes/layout/device, consumers, and acceptance gates. Do not copy
kernel source into this repository or load from a contributor's checkout.

### 3.3 Add evidence proportional to risk

Every behavior change needs focused tests. New structures and executable forms
normally need:

- reference or boundary correctness;
- positive discovery and a negative non-match;
- supported dispatch plus unsupported shape/capability refusal;
- target-path call count greater than zero and unexpected fallback count zero;
- attach/detach and repeated-call lifecycle checks;
- compile/capture checks when the path claims to support them.

GPU and end-to-end evidence is required only when the PR changes GPU behavior
or makes a hardware, precision, compatibility, or performance claim. Use real
host preprocessing and representative inputs; compare the same boundary and
execution assembly. Report commands, versions, device, shape band, numerical
result, latency distribution, ledger, and rollback result. A microbenchmark
does not establish end-to-end speedup, and identical output without path and
fallback counts does not establish that the implementation ran.

Documentation-only changes may report link and source-signature checks instead
of runtime tests. State that exception explicitly.

## 4. Pull-request self-review

Copy the following block into the PR description and remove inapplicable rows
with a short reason. Do not check a box when evidence is unavailable.

```markdown
### Structures self-review

Scope
- [ ] I identified the owning layer and kept unrelated model/kernel work out.
- [ ] I listed affected and intentionally unaffected hosts, shapes, dtypes,
      phases, devices, and public APIs.
- [ ] A new catalog boundary has cross-host evidence; otherwise it remains a
      binding/host concern or is explicitly provisional.

Contracts and failure modes
- [ ] Discovery is structural, with positive and negative cases.
- [ ] No catalog spec bytes changed, or the change carries a version bump with
      its coexistence window and re-qualification note.
- [ ] Hardware capability comes from the kernel package; this change adds no
      duplicate architecture table.
- [ ] Unsupported or missing capability refuses clearly or takes a declared
      host fallback, and the ledger records it.
- [ ] Calibration and precision use the existing collector/scheme entry point.
- [ ] Attach/detach, state ownership, repeated calls, and compile/capture
      behavior are preserved where applicable.

Evidence
- [ ] I added focused public tests for the changed contract.
- [ ] The intended path ran (call count > 0) and unexpected fallback count is 0.
- [ ] Numerical comparison uses the declared boundary and tolerance.
- [ ] Performance claims use paired final-form measurements, not summed
      microbenchmarks, and include the qualification cell.
- [ ] I recorded exact commands and results below, or explained each test that
      could not run.

Maintenance and hygiene
- [ ] I checked every changed shared helper/schema reader and every documented
      API against source.
- [ ] The diff contains no secrets, credentials, private/local paths,
      checkpoints, generated binaries, logs, or benchmark traces.
- [ ] Optional dependencies remain optional at import time; errors name the
      missing capability and a supported remedy.
- [ ] User-facing behavior, support limits, and invalidated performance
      receipts are documented.

Validation
- Environment: <GPU/compute capability, CUDA, framework, host/package versions>
- Commands and results: <exact commands; pass/fail/skip counts>
- Correctness/path/ledger: <metric, tolerance, call count, fallback count>
- Performance, if claimed: <baseline and candidate P50/spread, timed boundary,
  eager/compiled/captured assembly>
- Not run or not covered: <reason and resulting limitation>
```

## 5. Reviewer stop conditions

Request changes before performance discussion when any of the following is
present:

- a model ID, class allowlist, or incidental module name is the only discovery
  evidence;
- a structure definition contains hardware routing or a binding selects a
  kernel package;
- a catalog `structure.yaml` is edited without a version decision: any byte
  change alters `spec_digest` and orphans every receipt that cites it;
- unsupported behavior silently degrades, or success can be explained by host
  fallback;
- calibration, precision selection, diagnostics, or receipts are duplicated;
- a kernel is loaded or probed inside the swapped forward;
- local source injection, unpublished artifacts, private paths/data, generated
  binaries, or credentials enter the diff;
- a benchmark crosses different boundaries, inputs, stochastic windows, or
  compile/capture assemblies;
- rollback, ownership, optional-dependency import behavior, or negative
  discovery is untested for a change that can affect it.

When hardware or fixtures are unavailable but the structure is otherwise
sound, label the result as needing qualification; do not convert missing
evidence into either a performance claim or a universal rejection.

## 6. Definition of done

A structures PR is ready for review when its boundary and ownership are clear,
unsupported cases are explicit, focused tests cover the contract and rollback,
and the PR description contains reproducible evidence at the level of its
claims. Release readiness is a separate decision: only qualification cells
that satisfy [`structure_release_qualification.md`](structure_release_qualification.md)
may be published as certified fast.
