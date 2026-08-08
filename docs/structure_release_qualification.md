# Structure adaptation and release qualification

This document defines how a structure moves from a reusable dataflow
boundary to a released hardware route. It is a release process, not a
runtime autotuning design.

The central rule is:

> Adaptation proves that a structure is the right abstraction. Release
> qualification proves that one implementation is correct and fast in one
> declared hardware/workload cell. Distribution only consumes qualified
> results.

Runtime validation must not be used to compensate for missing release
tests. A production import or model load does not benchmark combinations,
search configurations, or infer performance from the current machine.

## 1. Ownership boundaries

| Layer | Owns | Does not own |
|---|---|---|
| Structure catalog | framework-neutral boundary, weight slots, state semantics, calibration points, gates | GPU architecture, kernel package, launch policy |
| Host binding | where catalog regions and pipeline stages exist on one host | hardware tuning or kernel selection |
| Implementation/kernel package | executable format, shape qualification, architecture capability, dispatch inside its declared forms | model pipeline composition |
| Release qualification | measured evidence for one model/hardware/precision/workload cell | runtime search |
| Distribution | deterministic selection of a qualified cell | benchmarking, automatic ablation, speculative fallback |

The kernel package metadata remains the source of architecture capability.
A release receipt records where that capability was actually tested; it is
evidence, not a second architecture-support declaration.

## 2. Qualification states

Every implementation route is in one of three states for a concrete
qualification cell:

- `supported_correct`: the implementation passes correctness and lifecycle
  tests, but no release-grade performance claim is attached.
- `certified_fast`: correctness passes and the implementation meets the
  declared net-win gate on the target hardware and workload band.
- `unknown`: the cell has not been qualified, or its receipt was invalidated.

`unknown` is not treated as probably fast. Distribution either refuses it or
keeps the host path under an explicit caller policy. It must not silently
select an unmeasured low-precision or slower implementation.

A structure is not required to be `certified_fast` on every hardware target.
For example, a decode projection may qualify on one GPU and remain
`supported_correct` or `unknown` on another. That is a matrix result, not a
request to force performance parity between devices.

## 3. Qualification cell

Performance evidence is valid only for the cell it measured. A receipt must
identify at least:

- structure name and version;
- host binding and model/checkpoint family;
- implementation form and kernel package version;
- hardware architecture and device class;
- precision and relevant layout;
- phase and shape band, such as prefill/decode or denoise row counts;
- software/runtime versions that affect dispatch;
- calibration method and data receipt when calibration is required;
- correctness, path-execution, latency, and fallback results.

Changing any item that can alter arithmetic or dispatch invalidates the
performance certification for that cell. It does not invalidate unrelated
hardware or shape cells.

## 4. Adaptation workflow

### 4.1 Prove the structure

Before hardware performance work:

1. locate the same dataflow boundary in at least two host families, or
   explicitly record why the first binding is provisional;
2. define framework-neutral inputs, outputs, weight slots, state, cadence,
   and calibration points;
3. provide the catalog reference and correctness gates for region
   structures;
4. add the host binding without embedding hardware dispatch in it;
5. verify a host that does not contain the structure is correctly rejected.

Pipeline bindings additionally classify the complete declared hot path as
`structure`, `state_region`, `host_stage`, or `control`. A `host_stage`
classification is an explicit gap, not a performance claim.

### 4.2 Qualify the kernel implementation

For each target hardware/precision/shape cell:

1. confirm the kernel package declares the target architecture;
2. run a standalone preflight at representative real shapes;
3. compare against the actual host/vendor form used by the pipeline;
4. run boundary parity against the structure reference or host boundary;
5. prove the intended kernel path ran and fallback count is zero;
6. record qualification limits and refusal reasons.

A kernel that loses in a cell is not forced into the release. Select another
qualified implementation or leave the host path in place.

### 4.3 Qualify the pipeline integration

Before publishing a fast route:

1. use real inference-distribution inputs through the host preprocessing
   chain;
2. assemble every selected structure, cadence/static region, host lowering,
   and backend route before making the release decision;
3. capture or compile both arms in the same final execution form used by
   deployment, then compare that configured pipeline with the unmodified
   host in the same process;
4. use paired alternating timing and report the measurement spread;
5. verify parity at the deployment output boundary with identical
   stochastic windows;
6. require target-path calls greater than zero and fallback count zero;
7. verify detach/rollback restores the host;
8. report stage costs when cadence changes where work is paid.

A cadence region may be hoisted only when captures from every invocation in
the fast loop agree under its declared tolerance. Do not prove a cadence by
calling the candidate module once on a guessed upstream tensor: that can
freeze a changing self/cross-attention projection, improve the reported
latency, and silently change the model. If the observed outputs vary, leave
the region in the host loop until the actual slower-cadence producer boundary
is represented.

The final configured pipeline, not a sum of kernel microbenchmarks, owns the
release performance claim. Kernel timings explain a result but do not replace
the end-to-end gate.

Cross-version or cross-model comparisons add one more constraint: both cells
must use the same upstream host family, the official model implementation and
processor for each checkpoint, the same timed boundary, and the same execution
assembly. A port in another repository qualifies that port only; its latency
must not be compared directly with a sibling version measured in the upstream
host.

“CUDA Graph” alone does not identify an execution assembly. Direct eager graph
capture, compile-then-capture, segmented graphs, and one complete compiled graph
are different programs for performance purposes. Receipts must record the
sequence explicitly. When an existing sibling result uses compile-then-capture
plus cadence-static refreshes, a comparison cell must include the same sequence
and cadence ownership before attributing a gap to structures or kernels.

An eager per-family gate is a preflight only. It must not prune the plan that
will later run under CUDA Graph or compilation, and its surviving seams must
not be reported as the final pipeline result. Launch overhead, upstream dtype
changes, cadence materialization, and backend composition can reverse the
decision after assembly. The release decision therefore belongs to the final
captured/compiled form as one unit; if that unit loses or fails parity, use
ablation to diagnose it and rebuild, then repeat the final-form gate.

Optional ablation is a debugging tool when the final integration regresses
or the source of a loss is unclear. It is not required runtime machinery and
does not belong in a pipeline binding.

## 5. Distribution rules

Distribution selects a previously qualified route from stable facts:

```text
model/binding + hardware + precision + phase/shape band
    -> qualified implementation and configuration
```

It must not:

- benchmark during import, model load, or the first production request;
- search combinations of structures;
- auto-prune structures based on live traffic;
- write performance results back into package configuration;
- assume a nearby architecture or shape band has the same result;
- hide an unknown cell behind an unreported fallback.

`structures.attach` and recipe audits remain useful qualification and
development doors. Their measurements may produce release receipts, but a
released deterministic route consumes the receipt outcome rather than
repeating the audit for every user.

Runtime checks are deliberately narrower:

- verify device, dtype, layout, shape, and lifecycle contracts;
- verify the selected implementation is available;
- count calls and fallbacks in the ledger;
- refuse or follow the caller's explicit host-fallback policy when the
  contract is not met.

## 6. Cross-hardware release sequence

For a new batch of structures:

1. finish catalog/reference/binding tests and prove structure properties;
2. qualify representative kernel cells on each target hardware separately;
3. run the selected final pipeline configuration on each target hardware;
4. publish only the cells with complete receipts;
5. leave other cells correct-but-unclaimed or unknown.

Do not block one hardware release on making every optimization win on every
other device. Do not fork the logical structure or pipeline binding merely to
carry a different kernel choice. Hardware compute producers may remain
separate when their capture, buffer, or layout contracts genuinely differ.

## 7. Release checklist

### Structure and binding

- [ ] Catalog version and boundary are explicit.
- [ ] Cross-host evidence or provisional status is recorded.
- [ ] Binding stages and hot-path classifications validate.
- [ ] Hardware dispatch is absent from the structure definition.
- [ ] Negative discovery/refusal behavior is tested.

### Per hardware cell

- [ ] Kernel architecture metadata accepts the target.
- [ ] Representative phase/shape bands are covered.
- [ ] Boundary parity passes.
- [ ] Intended path call count is positive.
- [ ] Fallback count is zero.
- [ ] Standalone preflight and final pipeline timing are both recorded.

### Pipeline and publication

- [ ] Real-distribution input and host preprocessing are documented.
- [ ] The unmodified-host empty control and stochastic-window parity pass.
- [ ] All structures, cadence regions, host lowerings, and backend routes are
      assembled before the performance decision.
- [ ] Both arms use the deployment's final capture/compile form; no eager
      per-family result is presented as the pipeline result.
- [ ] Cross-version comparisons use the same official host family, timed
      boundary, compile/capture sequence, and cadence ownership.
- [ ] Paired end-to-end timing clears the release gate.
- [ ] Detach/rollback restores the host.
- [ ] Receipt contains no local paths, private data, or transient logs.
- [ ] Distribution has a deterministic qualified route.
- [ ] Unknown cells refuse or use an explicit caller-selected host policy.
