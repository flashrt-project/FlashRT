# The explicit structure pipeline — GR00T N1.7

This folder is the **explicit** way into the structures layer: the
author declares every seat, runs their own calibration hooks, and calls
each structure family's binder directly. It is the hand-written
counterpart of the three-line automatic path, and both are measured
here on two different GR00T N1.7 hosts — the official Isaac-GR00T
repository and the LeRobot port — same checkpoint, same prepared input
tensors, same timing protocol.

```
groot_n17.py    the explicit assembly: seat tables, calibration hooks,
                per-family binder calls, attach, parity, eager+compiled
                timing with both baselines
full_graph.py   the same assembly at full speed: fixed-shape host
                lowering + whole-graph CUDA capture (the deployed form)
```

## Explicit versus the three-line automatic path

The automatic path is:

```python
plan = structures.auto_swaps(model, run_once)
handle = swap.attach(model, plan.swaps, observe=plan.observed,
                     revert=plan.revert)
```

Both paths produce the same kind of object — a set of bound structure
implementations attached onto the host's module tree, hot-pluggable,
guarded per seam. They differ in **who makes each decision**, not in
what can be reached:

| decision | automatic | explicit (this folder) |
|---|---|---|
| which modules are seats | discovery walks the model | the author's seat tables, path by path |
| calibration | one instrumented pass, library-owned points | the author's own hooks |
| qualification | work bands, shape envelopes, library judgment | the author's judgment — plus any runtime check they write |
| producer negotiation (e.g. FP8 norm → packed QKV) | resolved internally | written out as code: the norm binder and the pack binder share one scale tensor |
| a seat the library would refuse | stays refused | the author may claim it anyway and answer to the parity gate |
| safety net | per-seam guards; refusals recorded; production mode falls back to the host per call | identical — the same guards arm at bind time |

The practical meaning of the last rows showed up in this very
measurement. On the LeRobot host the automatic path binds the vision
`packed_qkv_rope` family whose rope tables can never satisfy the FP32
contract there; the guards refuse every one of those calls before any
work is done and the ledger counts them — while parity holds at
0.9999. The explicit book never declared that seat: zero run-time
refusals. One path is armor, the other is aim.

Neither path is faster by construction: they bind the same
implementation layer, and with a matched seat book the captured form
lands on the same number on both hosts. Explicit control matters
where automatic qualification is conservative — a specialist seat the
discovery cannot prove safe, a cadence the author knows
(observation-rate cross-attention K/V versus denoise-rate compute), a
scheme choice per seat.

## Measured — RTX 5090, one checkpoint, one input set, one protocol

GR00T N1.7 has two hosts — the official Isaac-GR00T repository and the
LeRobot port. Both are measured on the same prepared model-level
inputs (exported once, loaded by both), pinned noise, median of
interleaved rounds. Parity is the treated output against that host's
own untouched eager run; each arm re-measures the stock graph in its
own process (±3% across runs — read speedups, not cross-run
milliseconds). The **same `build()` — every seat path, every binder
call — ran on both hosts without a single edit**: the LeRobot port
vendors the official module layout, so the seat tables transfer
verbatim. Every number is the production form: the cross-attention
K/V banks refresh inside the hot path (wired to their producer), so
each call carries its own observation and pays its own refresh.

**The captured form (deployed), both hosts, both arms:**

| captured | official host | LeRobot host |
|---|---|---|
| stock graph | ~23.3 ms | ~23.5 ms |
| automatic (3 lines, 285 seats) | 15.16 ms (1.541×) p=0.99945 | 16.20 ms (1.482×) p=0.99990, **96 guarded refusals** |
| explicit (this folder, 333 seats) | **14.90 ms (1.566×)** p=0.99949 | **14.85 ms (1.565×)** p=0.99975, **0 refusals at run time** |

At a matched 285-seat book the two arms land within 0.01 ms of each
other on either host — the measured proof that the paths differ in
**who writes the seat book**, not in what the seats can do; the
kernel profile agrees bucket by bucket within 0.02 ms. The explicit
book then goes past the automatic one — 48 vision projection seats
automatic qualification does not claim — and lands **flat across
hosts**: 14.90 versus 14.85, a 0.05 ms spread from one unchanged
`build()`.

The cross-host story deserves its own line, because it was never
about semantics. With matched books the gap traced to two
compiler-fortune artifacts in the *unclaimed* territory: an 86-GFLOP
projection into the full vocabulary that a feature-extraction
pipeline never reads (one host's wrapper form let the compiler
dead-code-eliminate it, the other's kept it alive at 0.86 ms per
call), and one large matrix product whose template partitioning
differed threefold between transformers generations. The family
lowering now pins the dead head under a bit-exact proof — skip the
head, re-run the recorded request, keep the pin only when the
pipeline output is bit-identical — and the vision seats take the
partitioning question away from the compiler. What remains between
hosts is 0.05 ms.

**The explicit ladder on the official host** (same assembly, cheaper
forms):

| form | host | explicit |
|---|---|---|
| eager | 46.1 ms | 57.1 ms (0.81× — the per-call guard admission is real and printed) |
| compiled | 33.9 ms | 27.7 ms (1.22×) |
| captured | 23.3 ms | 14.90 ms (1.57×) |

What the tables say:

1. **A matched seat book is the whole game — and the explicit book
   may keep going.** The first explicit assembly stopped at 216 seats
   and lost 9% to the automatic path; the kernel-bucket profile
   located the missing milliseconds, the tables grew to the matched
   285-seat book and the arms met within 0.01 ms; then 48 vision
   projection seats the automatic qualification does not claim took
   another 0.3 ms on each host and flattened the cross-host spread to
   0.05 ms.
2. **Where they differ is the run-time story, not the speed.** On the
   LeRobot host the automatic path also engages the vision
   `packed_qkv_rope` family, whose rope tables can never satisfy the
   FP32 contract on this host (see the loading note below); the guards
   refuse every one of those calls before any work is done — 96
   refusals per run, parity held, ledger says so. The explicit book
   never declared that seat: zero run-time refusals. Armor versus
   aim, measured.
3. **Speedups may only be read within a row.** Eager pays a per-call
   Python admission check at every guarded seam and is *slower* than
   the host — printed, not hidden. Captured is the deployed form,
   where guards and glue are paid once at capture time.
4. **A host loading note that costs real precision**: the LeRobot
   recipe `from_pretrained(...).to(dtype=bf16)` casts *every* floating
   buffer — including the rotary `inv_freq` that transformers'
   dtype-aware loading deliberately keeps in FP32. On this host the
   vision rope tables are therefore born BF16; loading with
   `from_pretrained(..., dtype=bf16)` instead keeps them FP32. This is
   why the vision `qkv_rope` family refuses here, and it is a
   precision loss the host pays with or without structures.
5. Host coupling lives in the capture lowering, and the lowering
   lives in the library: the registered Qwen3-VL family adapter
   carries the probed branches for both transformers vision-contract
   generations and the LeRobot wrapper glue. Porting the measurement
   between the two hosts changed none of the assembly and none of the
   capture code — the same door served both.
6. **Teardown is a gate, not a hope**: every adapter's undo rides
   `extras["revert"]` into `attach`, and `handle.detach()` restores
   the host bit-for-bit — measured by re-running the untouched eager
   pass after detach and comparing exactly.

## Running

```bash
# explicit assembly, eager + compiled ladder
python groot_n17.py \
  --host /path/to/Isaac-GR00T \
  --checkpoint /path/to/GR00T-N1.7-3B \
  --backbone-assets /path/to/backbone-config-assets \
  --fixture /path/to/observation_fixture.pt \
  --compile --report report.json

# the same assembly, captured (the deployed form)
python full_graph.py --host ... --checkpoint ... \
  --backbone-assets ... --fixture ...
```

`--fixture` is a saved observation dict (`{"inputs": {...}}`) from the
host's own preprocessing. Kernel packages resolve from the Hugging Face
Hub per host; offline, stage them under any directory laid out as
`<org>/<name>/build/<variant>/` and point the kernels resolver at it.

## Reading the report

- `seats_bound` / `refused` — every declared seat lands in exactly one;
  a seat served by the host is an outcome, a silently skipped seat is
  a bug.
- `attention_variants` — which family member serves this host and the
  recorded reasons the preferred members stepped aside (on an RTX 5090
  FA2 binds; on a host without the FA2 build the family resolves to
  FA4 from the same line of code).
- `kernel_unavailable` — packages this host asked for and could not
  get, original error preserved: "never shipped here" and "broken
  here" must stay distinguishable.
- `ledger.fallbacks` — nonzero means a calibration assumption did not
  hold at run time and the guards routed those calls back to the host.
