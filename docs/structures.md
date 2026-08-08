# Structures

> **Target audience**: engineers adding a structure, adding a backend for
> an existing structure, or working out why an attachment refused.
>
> **TL;DR**
> - A structure is a **spec** (boundary, weight slots, calibration points,
>   gates) plus a plain-torch **reference** that is the gate's ground
>   truth. Implementations are separate and plural.
> - Three layers, split by what varies: the **spec** names positions, a
>   **binding** says where they sit on one host, an **impl** decides what
>   to do there. A backend change touches only the third.
> - Calibration is **not this layer's**. Points come from the spec, the
>   reduction is `flash_rt.core.calibration`, the receipt is
>   `ModelPrecisionSpec`, the argument names are
>   `flash_rt.api.FlashRT.calibrate`'s. See [`calibration.md`](calibration.md).
> - Every swapped-in structure carries a **runtime contract and a ledger**.
>   Falling back to the host is numerically exact, so a seam that quietly
>   reverted is invisible to parity — the ledger is how you see it.
> - Adaptation, hardware qualification, and distribution are separate.
>   See [`structure_release_qualification.md`](structure_release_qualification.md):
>   production consumes measured release cells; it does not benchmark or
>   search configurations at runtime.

---

## 1. What a structure is

One model region, versioned, with four parts declared in
`flash_rt/structures/catalog/<name>/structure.yaml`:

| Part | What it fixes |
|---|---|
| `boundary` | the tensors in and out, in symbolic dims. `dtype: "@binding"` defers dtype to the host binding on purpose |
| `weights` | framework-neutral slots and their dims, not checkpoint key names |
| `calibration.points` | what has to be observed to calibrate it, **named by position in this structure's own dataflow** |
| `gates` | parity metrics and the latency rule that qualify an implementation |

plus `reference:` — a plain-torch implementation in the catalog, which is
what a gate compares against. A structure with no reference cannot gate
anything, so it is not a structure.

## 2. The three layers, and why the split is where it is

```
catalog/<name>/structure.yaml   the definition. Positions, slots, gates.
                                Changes ~never; a change is a version bump
                                and moves spec_digest.

bindings/<host>.yaml            how this host realises it: which module
                                path holds which weight slot, which
                                submodule a calibration point sits on,
                                what the boundary dtype actually is.
                                Changes once per host family.

impls/<name>/<backend>.py       the executable form: kernels, quantisation
                                format, what statistic to take at a point
                                and how to reduce it. Changes with every
                                new format.
```

The rule is **what varies at what rate**, not who needs it. A worked
example, because this is the part that goes wrong:

`decoder_ffn` declares `calibration.points: [x_after_norm, act_after_mul]`.
`act_after_mul` is a position — the gated activation — and it means the
same thing in a GGML graph as in a torch module tree. *Where* it sits is
per host (`...layers.{i}.mlp.down_proj`'s input, on a transformers-shaped
host). *What to measure there* is per backend: a per-tensor amax for FP8,
a per-column second moment for an importance-matrix flow, a per-block
statistic for a k-quant, nothing at all for a backend that quantises
activations dynamically.

Put the statistic in the spec and every new format needs a schema change.
Put the position in the impl and the same dataflow knowledge gets written
once per backend and drifts. Neither is recoverable later, so the split is
load-bearing.

### 2.1 Pipeline coverage bindings

A native pipeline is bound at two levels. Region bindings map weight slots
and calibration points as above. A pipeline binding maps the larger stage
seams and classifies every declared hot-path segment as one of:

| Classification | Owner |
|---|---|
| `structure` | one or more catalog regions own the composition |
| `state_region` | an explicit buffer/cache/window owns state and cadence |
| `host_stage` | host preprocessing, embedding, or other retained glue |
| `control` | loop, branch, or scheduling logic rather than a kernel region |

`structures.load_binding(name, require_pipeline_coverage=True)` validates
the binding against the catalog: stage names must exist, every hot-path name
must resolve to one segment, and every referenced region structure must
exist. Unknown classifications and unclassified hot-path names fail at load
time. The normalized `BindingSpec.manifest()` is JSON-serialisable for
runtime exporters and native consumers.

This is a composition contract, not a compiler IR. It does not encode a
kernel, target architecture, launch policy, or tensor lowering. Those stay
in each referenced structure's implementation and Hub package, so adding a
hardware target does not fork the pipeline declaration.

The catalog currently uses two schedule families:

- `autoregressive_decode_pipeline`: optional modality encoding, causal
  prefill/KV materialization, token decode, and token selection. Qwen3 and
  Qwen3-VL share this family; the latter binds the optional modality stage.
- `vla_tick_pipeline`: observation-cadence condition preparation, a
  fixed-step iterative update, and an optional output readout. Pi0.5 and
  Motus bind this family with different state and readout regions.
- `video_generation_pipeline`: request-level condition preparation, explicit
  latent initialization, a fixed-step generation loop, and optional VAE
  decode. Cosmos3 and Wan2.2 share this family without inheriting VLA
  action-chunk or observation-cadence semantics.

A `host_stage` entry is an explicit coverage result, not a claim that the
region is finished. For example, a fused Q/K norm plus RoPE path remains a
host stage until a catalog structure owns that boundary; landing the
structure changes the classification without changing the pipeline family.

### 2.2 Catalog evolution: what a version bump means for your binding

Three facts anchor the process. A spec's ``version`` is an integer in
``structure.yaml``; every qualification record carries a
``spec_digest`` — the sha256 of that file's bytes — so any edit to a
spec, compatible or not, is visible in receipts; and bindings resolve
specs by name through the registry, never by digest.

The rules:

1. **Byte changes without meaning changes do not exist.** Reordering
   keys or rewording a comment changes the digest, and a changed
   digest orphans every receipt that cites the old one. Treat the yaml
   as frozen bytes between versions; editorial changes ride along with
   the next real bump.
2. **A version bump is a new contract, not a patch.** ``v1 -> v2``
   means at least one of: a point moved, a dimension constraint
   changed, the reference changed. Old receipts stay valid *for v1*;
   nothing re-validates automatically.
3. **Coexistence window.** When v2 lands, v1 stays in the catalog
   until every in-tree impl and binding has re-qualified against v2 —
   receipts named with the new digest — and the release notes say so.
   Third-party bindings pin the version they qualified against; the
   registry loads what the yaml says, so an unmigrated binding keeps
   working against v1 until v1 is retired in a *major* release.
4. **Retirement is loud.** Removing a spec version is a release-notes
   event with the migration note inline; the registry failing to find
   a name is a refusal with the reason, not a KeyError.

If you write bindings out of tree: record the ``spec_digest`` your
qualification ran against (the receipt already does), and re-run your
own gate when the digest you depend on disappears from the catalog.

## 3. Calibration: reuse, do not redefine

**The standard is `docs/calibration.md`. This layer adds no second
vocabulary for it.** Concretely:

| Concern | Where it comes from |
|---|---|
| what to observe | the spec's `calibration.points` |
| where it is on this host | `flash_rt/structures/points.py` + discovery |
| reduction across samples | `flash_rt.core.calibration.accumulate_amax` |
| dispersion diagnostics | `summarize_amax_dispersion` / `format_summary` |
| outlier-scale warning | `check_scale_ceiling` |
| picking calibration frames | `stratified_sample_indices`, `flash_rt.datasets.libero` |
| the receipt | `flash_rt.core.precision_spec.ModelPrecisionSpec` |
| argument names and defaults | `flash_rt.api.FlashRT.calibrate` |

```python
structures.auto_swaps(model, forward)                          # one sample
structures.auto_swaps(model, feed, observations=frames)         # N samples
structures.auto_swaps(model, feed, observations=frames,
                      percentile=95.0, max_samples=64)
plan.precision_spec        # ModelPrecisionSpec, same as rt.precision_spec
plan.notes["calibration"]  # method, samples, percentile, dispersion
```

### 3.1 The reduction is two-level, and both levels are the house's

**Within one sample: max over every call.** Required, not chosen —
`calibration.md` §4.2 records that per-step scales on a flow-matching host
gave the compiler inconsistent shapes and crashed it. One forward already
covers every step.

**Across samples: `accumulate_amax(per_sample, percentile)`.** Each sample
contributes one `[num_points]` vector, kept host-side. This ordering is
not a detail: a running max *across* samples destroys the per-sample
values as it produces them, which makes a percentile impossible rather
than merely unused. If you find yourself accumulating in place across
samples, that is the bug.

### 3.2 A point is measured where it is, never recomputed

Every activation scale an implementation needs is the amax at some host
GEMM's input, so it is one hook and one float. `decoder_ffn`'s hidden
scale is the amax at `down_proj`'s input; `vision_ffn`'s is at `fc2`'s.
Keeping the seam's input alive to re-run gate/up over it arrives at the
same number the host already produced, and costs GiB to do it.

Weight scales are derived from weights at bind time and are not
calibration — same division as `calibration.md` §2.1/§2.2.

### 3.3 Three kinds of capture, and only one is calibration

| Kind | Examples | Reduced by a percentile? |
|---|---|---|
| **statistic** | `x_after_norm`, `act_after_mul`, the shared q/k/v input | yes — this is calibration |
| **content** | an adaptive norm's step table, an attention prefix KV, a cadence buffer | no. The artefact *is* the output; a percentile over it is meaningless |
| **observation** | row counts, observed dtypes, a return convention | no. One scalar |

Content and observation are plan-time captures, bounded by construction,
and they get **no public calibration surface of their own**. Giving them
one would be exactly the extra entry point this layer must not add.

### 3.4 Reporting parity, and what the bands mean

Report **held-out** parity, and say when a figure is not. A parity measured
on the frame its scales were fitted to is a fit residual: on Pi0.5 that gap
is about 0.0029, measured three times. Report cosine **and** max-abs
against the reference — multi-sample calibration's benefit shows up mostly
in the worst case, so cosine alone can miss it entirely
(`calibration.md` §10).

Bands, from `gates.py`, are per output kind. Value outputs (a policy's
action tensor): `pass` at cosine 0.999 and above, `warn` from 0.995, and
`low` below that. Distribution outputs (a language model's logits) are
judged on **token agreement**, where cosine-grade edges do not transfer:
a clean static W8A8 with every per-seam gate passing sits at 0.95–0.98
agreement on real text — that is the grade of the quantisation, not
damage, which instead looks like agreement collapsing while seam-level
parity stays fine. So `pass` from 0.95, `warn` from 0.85, `low` below.
**None of the three refuses.** Low-precision execution is
increasingly the intent rather than a defect — a four-bit host belongs in
the bottom band by design — so a `low` band warns with the number and the
calibration method attached, and whether it is acceptable is the
deployment's call. Pass `floors={...}` to turn a number into a hard
requirement; that is the caller stating a requirement, which is the only
place such a number can honestly come from.

## 4. Doors

```python
plan = structures.attach(model, forward)      # gated: discover→gate→activate
plan = structures.auto_swaps(model, forward)  # build only, you own the gate
mod  = structures.get("decoder_ffn").bind(module, calibration=[x])
stage = structures.capture(hot, windows={...})
```

The serving door builds the whole-loop decode form over whatever swaps
are attached — a duck-typed static hybrid cache, the decoder stack
found by its slots, argmax and the position increment in-graph, the
step optionally compiled before capture:

```python
loop = structures.decode_loop(model, max_len=4096)
out = loop.generate(input_ids, max_new_tokens=256)     # greedy, exact

loop.enable_mtp(ckpt_dir, default_k=6)                 # draft head
out = loop.generate_speculative(input_ids, 256)        # bit-identical
```

Speculative decode is exact by construction (the verify pass recomputes
every draft token), so its gate is token identity; draft precision is a
scheme decision (``mtp_projection_format``) judged by acceptance
length, and BF16 is the arm until a measured table says otherwise.

Fixed-iteration hosts use the same doors. A graph-safe fixed ``for`` follows
the ordinary capture path. A registered host-family adapter may normalize a
recognized tensor-controlled ``while`` to the catalog's canonical
``init -> K * step -> readout`` schedule when ``model=`` is supplied:

```python
from flash_rt.structures import swap

plan = structures.auto_swaps(model, [calibration_0, calibration_1])
handle = swap.attach(model, plan.swaps, on_guard_fail="raise")
stage = structures.capture(forward, model=model)
```

Normalization is not source rewriting. The adapter matches semantic host
capabilities, records one real invocation, requires an explicit noise/init
tensor, and runs the original and canonical schedules before calibration or
capture. A fixed schedule must be bit-exact or it is refused. Its observation
and noise tensors become declared replay windows automatically. Data-dependent
or unbounded loops remain on the existing per-step/bucketed host path; they are
never silently forced into a fixed graph.

`attach` gates unit by unit — a unit is a structure, except that a
negotiated FP8 chain is one unit, because the producer emits under a scale
the consumer was bound for. It judges accuracy with the metric the host's
output type deserves, checks the ledger, and settles latency by timing
both arms in every round.

`modnorm_qkv_chain` is the data-flow qualification for the common DiT/VLA
case behind that negotiation. It admits a conditional LayerNorm only when
its output feeds attention projections directly: self-attention binds the
shared Q/K/V pack, while cross-attention binds only the query projection
because K/V consume encoder features on another cadence. An intervening
positional module refuses the chain. Diffusers-style `to_out[0]` projections
are discovered as ordinary `linear_proj` seams, but retain the host path when
their measured small-M bias form is outside the profitable work band.

Processor-preflattened vision patches use the narrower `patch_projection`
structure. It admits a Conv3D wrapper only when the module declares one
complete temporal/spatial patch, `kernel == stride`, zero padding, unit
dilation and one group, and calibration proves the host actually supplies
that full patch as the input's final dimension. The executable form reshapes
the checkpoint Conv3D weight once and calls the BF16 Hub projection API.
Ordinary image/video volumes, overlapping patches and grouped convolutions
remain on the host. This BF16 lowering is independent of `scheme=`; the final
accuracy/latency gate may still refuse it when its sub-millisecond model-level
gain is lost inside a larger request boundary.

`cadence_static` remains an explicit host-stage structure rather than part of
the automatic front door. Its updater needs the host's observation boundary:
the host must capture the repeated-loop outputs, bind the static K/V buffers,
and call `refresh_cross_attention_kv` when encoder inputs change. Replacement
projections may be reused for refresh only when they are independently
callable; sibling-ordered readers such as a packed QKV stash are rejected as
refresh producers so stale data cannot be copied into the cadence buffer.

Capability-compatible Diffusers attention-processor sites use the same
`attention_core` structure:
the family adapter preserves the host projections and output contract, while
the stateless core consumes complete Q/K/V on every call. An unmasked dense
call is passed directly; a mask with one or two contiguous allowed key ranges
is packed explicitly. More fragmented masks are refused instead of being
approximated.

Qwen-style sequence attention composes `qkv_pack` with the per-head GQA
variant of `qk_norm_rope` when the host exposes compatible Q/K/V projections,
128-wide per-head Q/K RMSNorm, a valid Q-to-KV head ratio, and pre-expanded
BF16 rotate-half tables. The adapter consumes the pack's joint output and
writes contiguous Q/K/V workspaces through
`flashrt-qkv-cache-rope`; cache mutation remains on the host after this
stateless boundary. Qualification is capability-based, so Qwen3 and Qwen3-VL
hosts share the same structure implementation while retaining their own
attention dispatch function and cache object.

Factored two-way attention uses the same region without introducing a
model-labelled form.  A compatible host exposes two complete sibling-QKV
groups over causal and full-only sequence packs, one pair of 128-wide
per-head Q/K RMSNorm weights for each group, and one shared factored
attention dispatcher.  The adapter requires both groups and composes two
`qk_norm_rope` entries with the existing factored `attention_core`; an
incomplete pair, context-parallel layout, neighborhood-attention metadata,
or mutating KV-cache call is refused.  The output projections remain dynamic
module calls so independently qualified `linear_proj` swaps still compose
with this region.

Packed vision attention without Q/K normalization uses the separate
`qkv_rope` boundary: one already-packed projection result plus its bias and
pre-expanded FP32 rotate-half table become attention-ready Q/K/V through one
Hub custom op. Keeping this distinct from `qk_norm_rope` prevents a vision
tower from acquiring normalization it does not have. The capability adapter
qualifies packed equal-head QKV, an observed fixed token capacity, an even
head dimension no larger than 256, and the host's own attention dispatcher;
Qwen3-VL's D=72 vision form is one consumer, not a model-special branch.

Transformers-style hybrid decoders expose their Gated Delta recurrence through
callable recurrent and chunk slots. `gated_delta_core` owns that stateful
Q/K/V, decay, update-strength and explicit-final-state boundary; projections,
causal convolution and gated output normalization remain neighbouring
structures. The Hub v3 decode implementation qualifies the D=128, H=32/H=48
BF16 profiles used by two independent hybrid-decoder families. Sequence
prefill is retained by the host until the formal artifact exposes a
non-mutating explicit state output. A host that exposes non-contiguous split
views is likewise refused until the recurrence artifact can consume their
strides; the adapter never inserts hidden `.contiguous()` or state-copy
kernels into the hot path.

`auto_swaps` builds and does not judge. If you use it, you own the gate —
and read the ledger, or you have not checked that what you measured was on.

### 4.1 Selecting a precision profile

`scheme=` is the one precision entry, on both doors. A profile is a
registered quantisation scheme (§6) selected by name:

| name | what it does |
|---|---|
| `"auto"` (default) | resolves to the fastest profile this device can execute: `fp8_static` on FP8-capable hardware (bit-identical to the pre-profile default), `"none"` elsewhere. The resolution table is one function, so a profile that measures faster is promoted by editing one line |
| `"fp8_static"` | static per-tensor FP8, the shipped behaviour; `"fp8_static_keep_outliers"` keeps outlier seams at host precision by the house scale-ceiling criterion |
| `"bf16_structural"` | no quantisation; binds numerically conservative structural forms (shared-input QKV packing and qualified full-patch projection) and keeps dense FFNs/projections at host precision |
| `"w8a16_decode"` | weight-only INT8 on `decoder_ffn` and on `linear_proj` (the attention Q/K/V/O family), decode band only, everything else at host precision. Each impl mirrors its kernel's own auto-dispatch qualification; prefill dispatches to the retained host module and is counted |
| `"w4a16_decode"` | the NVFP4 twin, `decoder_ffn` only — its linear auto band is too narrow to route projections blind, so they stay at host precision until a measured table says otherwise |
| `"none"` | quantisation off. An explicit choice, not a degraded mode: fusion structures never consult a scheme decision and attach as usual, so a BF16/FP16 host under `"none"` still gets every fusion structure |

Quantisation happens **at attach time, from the host's own weights** —
the same discipline as this repo's native pipelines. The checkpoint is
loaded at host precision; weight scales and packed formats are derived
from the floating weights at bind, activation statistics come from
running the host's own forward. Nothing is destroyed: the original
module is retained, and detach restores it bit-exactly.

A checkpoint that arrives *already* quantized in a packed layout goes
through the other door, `structures.adopt_prequantized(model, fmt)`
(first supported: compressed-tensors NVFP4 in its `run_compressed`
form). Each packed projection is unpacked by the compressor registered
for the checkpoint's own format and converted once, at load time, into
the `linear_proj/nvfp4_dynamic` impl; the per-layer conversion error is
recorded in the returned report. Unlike a scheme attachment this is a
load-time transform with no detach — the packed source cannot execute,
and undoing an adoption is reloading the checkpoint.

Hardware support is not declared here at all. A Hub kernel package
ships the archs it was built for in its own metadata; the shared loader
reads that declaration and refuses a device outside it with the package
name and both arch strings in the message. The structures layer keeps
no second table — hardware support is maintained where the kernels are.
The loader interprets the CUDA arch notation rather than comparing it
as an opaque string: a plain cubin serves the same major compute family
at or above its minor capability, generic ``+PTX`` serves equal or
higher compute capabilities, and architecture-specific ``a`` targets
serve only the exact capability. Missing declarations retain the legacy
load path; an incompatible declaration refuses before binding.

The tested release matrix is also not a second capability table. It is a
set of receipts saying which model/hardware/precision/shape cells passed
correctness and net-win qualification. The adaptation and publication
workflow, including what must never move into runtime distribution, is
defined in
[`structure_release_qualification.md`](structure_release_qualification.md).

## 5. Runtime contract and the ledger

Every swapped-in structure declares the form it was calibrated for
(device, input dtype, width, and either an exact row count or a maximum
row capacity where buffers were preallocated). Called outside it, a seam
runs the retained host module and records that it did.

```python
handle.report()             # per seam: calls, fallbacks, last_reason, form
handle.summary()["clean"]
handle.raise_on_fallback()
structures.swap.attach(..., on_guard_fail="raise")   # refuse instead
```

The first fallback per seam warns; 32 consecutive fallbacks restore the
host module for good. Counts are eager-only — inside a compiled or
captured region the kernel runs without re-entering Python, which is also
why the check costs nothing there.

Refused rather than approximated: training mode, device/dtype migration
while attached, `load_state_dict` while attached, and a second thread in
one seam. `state_dict()` delegates to the retained host module, so saving
while attached yields the unattached schema and bytes.

## 6. Adding things

**A backend for an existing structure**: add `impls/<name>/<backend>.py`.
Read the spec's points, take whatever statistic that format needs, declare
your own qualification band per executable form. Do not touch the spec —
if you need a position it does not name, that is a signal the boundary is
drawn wrong, and it should be raised rather than absorbed.

**A host family**: add `bindings/<host>.yaml` with the weight map and the
point addressing. Discovery covers hosts whose slots are findable
structurally; the binding is the receipt, and the only source for hosts
where they are not.

**A structure**: spec + reference + gates first, in the catalog, with an
implementation second. A spec whose points nothing can locate fails loudly
at plan time, which is the intent.

**A per-device kernel update** (a package grows an arch build, a tuned
body, a fused epilogue): decide the tier before writing any code.
*Same symbols, new arch build* — no code; the package's metadata
unlocks the device and the impl binds unchanged. *New capability
entries inside the seam* (fused-bias epilogue, direct BF16 quantize) —
one **capability probe** in the impl: `getattr(kern, "entry", None)`,
prefer it when the installed package ships it, keep the existing path
when it does not; absence is a fallback, never a refusal
(`nvfp4_dynamic`'s GEMV and BF16-quantize probes are the precedents),
and one probe serves every device forever. *Fusion that crosses a seam
boundary* (an epilogue emitting the next GEMM's quantized input) —
that is a new executable form: an impl variant or a wire-dtype
negotiation, judged like any variant. Variant count grows with
executable forms, never with devices; per-device tuning stays
contained in the packages' build-variant distribution, and within one
package version an entry's signature is invariant across archs.

**A region family** (`regions.py`): when hardware disagrees about the
*shape* of a span larger than one seat — a fused launch chain on one
device, the seat-by-seat composition on another — the difference is a
region family, never a device branch and never adapter registration
order. Declare a `RegionFamily` (a structural identifier over the
module graph) and its `RegionCandidate`s; each candidate states the
factual prerequisites it needs (hub symbols, shape band, memory plan)
and binds through structure primitives — seats, producers, workspace
leases, guards. A form that swaps in a hand-written forward has no
seat here: nothing for the ledger, the fallback contract, or revert
to certify. The tier discipline is fixed: the automatic tier consumes
receipts only (author pin > decision cache > `seated` floor) and
never experiments at bind — a cold box runs seated, correct but
possibly not full speed, until a production-form measurement run
records the winner per `(device, region)` through `regions.record`
(which refuses undeclared names — a typo dies at the writer, not at
every reader). A receipt naming a form this box cannot qualify falls
through to seated with the reason on the trail. The explicit tier is
maximum host replacement: it pins winners and claims regions
discovery refuses, over the same candidate set — so anything it
proves, the automatic tier inherits through the cache.

Inside a candidate, an element the hosts disagree about — the
attention entry is the recurring case — is a *ladder*, not a device
branch: rungs in preference order, each resolved by loading its
package and running one functional probe at the bound shapes during
bind. The first rung that executes serves; the ones that fell through
land on the guard's notes. The same candidate then binds its best
form on every host without naming any device, and the receipts stay
comparable because the measured thing is still the one candidate.
Both shipped families follow this file layout: `region.py` holds the
structural identifier and the `RegionFamily`, the chain module holds
the candidate — `dit_stack/` (a DiT block stack, NVFP4 chain) and
`adarms_stack/` (a conditioned-norm decoder tower over a cached
prefix, static-FP8 chain with the FA4/FA2 attention ladder).

**A quantisation scheme**: register an instance in `schemes.py` — two
methods and nothing else. `statistics` declares what each calibration
point needs (statistic and granularity: per-tensor, per-channel,
per-16-block; `None` is legal and means the format quantises that point
dynamically at runtime). `decide` turns the reduced statistics into
per-seam outcomes: bind with these values, or keep the host at host
precision — a first-class decision recorded in the receipt with its
reason, not a refusal. Bytes are not the scheme's: scale-factor layouts,
sub-normal handling, packing and kernel choice live in the impl variant
that executes the decision, which is what lets one decision serve
different kernels. A statistic the collector cannot measure yet fails
loudly at plan time; extending the collector is the supported path, and
silently substituting per-tensor is not. Schemes add no calibration
entry point — the calibration axis is fixed, a scheme only declares what
to measure along it.

## 7. Norms that came from being wrong

- **Calibrate and judge on the host's real inference distribution.**
  Training-domain data through the host's own preprocessing chain, or
  deployment inputs — never a neighbouring dataset with a hand-assembled
  mapping, and never synthetic text repeated and padded to length. Two
  hosts here measured 0.02 of cosine and 13 points of token agreement
  worse on such inputs than on their real data, with the mechanism
  unchanged; the dirty figures nearly became a quantisation-scheme
  decision. Before attaching, prove the measurement itself: rerun the
  host unmodified and require bit-identical output.
- **Grep the repo before writing a mechanism.** Percentile reduction,
  stratified sampling, dispersion diagnostics, scale-ceiling warnings and
  the precision receipt all existed before this layer reimplemented worse
  versions of the first two and skipped the rest.
- **Grep the repo before saying something is not implemented.** Asserting
  a missing capability is worse than missing it, because the assertion
  becomes a documented design boundary.
- **A declaration nobody checks is a comment.** The spec's point names are
  safe upstream only because they resolve against something — the
  reference and the loud failure when they do not.
- **Report the increment, not the peak.** A resource number that includes
  the model weights and the bound plan is not the cost of the thing being
  measured.
- **The final execution form owns the decision.** An eager per-family gate is
  preflight evidence, not permission to prune a plan that will be captured or
  compiled. Assemble structures, cadence/static regions, host lowerings, and
  backend routes first; then capture both host and treated arms in the
  deployment form and make one paired end-to-end decision. Reporting the
  eager survivors as the pipeline result measures a different program.
- **Refusals record the form and the shape.** "refused" must never read as
  "this cannot be bound", only as "not in that form, at that size".

- **The race qualifies; the receipt activates.** A bind-time A/B
  micro-race is a qualification signal, never an activation: the
  norm→FP8 pairing won its race at 28 of 28 sites on a device where
  the same flip cost 0.6ms end-to-end in the captured form, in the
  same day that a kernel-form swap winning its standalone numbers
  cost another 1.0ms inside a fused chain. Anything that changes a
  production form activates only on a production-form receipt for
  this box (`decisions` — the same cache, transport and manifest
  column the bands and regions use); the race result goes to the
  trail so the refusal is auditable. This is the seat-level
  micro-timing refutation, promoted from a finding to a gate.
