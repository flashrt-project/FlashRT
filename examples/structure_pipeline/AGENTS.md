# Working in this folder — the usage model, the footprint, the recipe

Every number in the README comes from the scripts in this folder run
against **unmodified host checkouts**. Read this before touching or
reproducing anything: it states what each tier of the library actually
requires from a user, what each script touches, and exactly how large
the footprint is.

## The usage model, tier by tier

**Automatic** — install once, then two calls, nothing else:

```bash
pip install flash-rt kernels   # hub kernel packages fetch per host at first bind
```

```python
plan = structures.auto_swaps(model, run_once)   # run_once: the host's hot path, once
handle = swap.attach(model, plan.swaps, observe=plan.observed,
                     revert=plan.revert)
```

Zero host-source changes, zero files edited, nothing forked. The user
supplies the loaded model and one callable that runs its hot path on a
representative observation (that single pass is the calibration).
Everything is in-process and `handle.detach()` restores the host
bit-for-bit. This buys the eager/compiled band with per-seam guards
and a ledger.

**Captured (full speed)** — one more mechanism call, not a harness:

```python
stage = structures.capture(torch.compile(hot), model=model)
stage.replay()
```

When `capture` is handed the model, the registered host-family
lowering adapter recognizes it, records one real request, pins that
family's shape glue (both transformers vision-contract generations,
plus wrapper glue via capability probe), and writes the family and its
pins into the stage certification. `stage.restore_host()` takes every
pin back off. The automatic and explicit assemblies reach their
fastest numbers through this same door; a host no family recognizes is
captured as-is, and a family that cannot pin safely refuses with a
reason instead of leaving the host half-pinned.

**Weight residency (a lifecycle, not a mode)** — attach → validate →
`handle.consume()` → optionally `handle.finalize()`. There is no
resident tier: after your parity gate passes, `consume()` moves every
replaced original's truth off the device — to the checkpoint file when
provenance verifies (a sampled-block match against the live tensor),
to pinned host RAM otherwise — and frees its device storage. The
receipt names bytes freed and the tier split. Fallback and `detach`
survive as restore-from-store: a seam called outside its contract
restores its host once (the ledger notes `restored_for_fallback`), and
`detach` reloads before it puts the host back, still bit-exact. Seats
that actively serve through their retained host (a cadence bank
refreshing through the host projection) declare `_frt_host_serving`
and are kept whole — the receipt counts them. `handle.finalize()`
then drops the restore tickets: fallback flips to refusal, `detach`
is forbidden, irreversible and recorded as such. Consumption comes
after validation because until then the attached model still owes the
host schema (state_dict, A/B reference arms, and any capture that
aliases host weights dies the moment its pointers are freed).
Seat scratch (sibling stashes, producer workspaces, the packed-output
and quantize scratch, wire buffers) is pooled by shape: sequential
layers share one buffer, so the memory bill is one layer's worth, not
layers x tokens (`structures.workspace.report()` is the receipt's
memory column). Binding itself is budgeted: under 512 MiB free VRAM a
seat refuses with `insufficient_vram(...)` instead of eating the
remainder. The pool's safety argument is single-stream sequential
execution — leasing from a non-default CUDA stream refuses loudly.

**Receipts (one document)** — `handle.manifest()` answers "why does
this box run this form" in one serializable dict: device fingerprint,
every seam with kind/calls/fallbacks/notes, the band decisions this
device consumed, the workspace ledger, and the weight-residency
receipt. Captured windows are strict at the door: `stage.write(name,
value)` requires the exact shape/dtype/device the graph was captured
with — a broadcastable-but-wrong write refuses instead of replaying
over coerced data.

**Transaction boundary** — `attach` is the commit point. A bind that
dies midway rolls the whole plan back (routes disabled, host
mutations reverted, streamed weights restored); a caller that decides
not to commit calls `plan.abort()`. There is no half-routed state to
clean up by hand.

**Explicit** — the only tier with real orchestration in user code:
seat tables, the author's own calibration hooks, direct binder calls —
`build()`, 215 lines. It exists for control the automatic
qualification will not exercise: claiming seats discovery refuses,
writing a producer negotiation out as code, choosing a scheme per
seat.

The explicit DiT band form (FP8 chains versus the FP4 wire) is an
author pin backed by captured-form receipts, selected by
`FRT_DIT_BAND` (`fp4` default — the Thor-measured winner; `fp8` — the
5090-measured winner) until the band-level adjudicator with its
decision cache lands. Seat-level micro-timing was refuted in both
directions by production-form measurement; the design doc in the
records repo carries the receipts.

## The footprint

| where | what changes | size |
|---|---|---|
| official Isaac-GR00T source | **nothing** | 0 lines |
| LeRobot source | **nothing** | 0 lines |
| host process at run time | structure swaps attached onto the module tree | revertible; `detach()` restores the host bit-for-bit |
| host process at run time (captured form only) | fixed-shape lowering: function pins applied by the registered family adapter inside `structures.capture(model=...)`, undone by `stage.restore_host()` | in-process only, never written to disk; the family and its pins are listed in the stage certification |
| host process at run time (cadence seats) | the cross-attention K/V bank refresh is wired onto the producing module's own forward (`wire_refresh_to_producer`), so eager, compiled and captured forms all carry the current observation | removed with the wire handle; `detach()` restores the host bit-for-bit |

The author-owned code, by role:

| file | role | lines |
|---|---|---|
| `groot_n17.py::build` | **the explicit assembly itself** — seat tables, calibration hooks, binder calls, family-adapter entries | 345 |
| `groot_n17.py` (rest) | host loading (34), input capture (~30), timing/report harness (~150) | 282 |
| `full_graph.py` | captured-form harness (both arms): `structures.capture` per arm, interleaved replay timing, noise pin | 227 |
| `lerobot_host.py`, `lerobot_full_graph.py` | the same measurement on the LeRobot host; `build()` is imported, not rewritten | ~300 |
| `make_model_inputs.py` | exports one set of prepared model-level inputs both hosts consume | ~60 |

## What you need

1. **Hosts** — either or both, unmodified:
   - official: `github.com/NVIDIA/Isaac-GR00T`
   - LeRobot: a checkout with the GR00T N1.7 policy
     (`lerobot/policies/groot/groot_n1_7.py`)
2. **Checkpoint** — the public GR00T N1.7 3B release (one copy serves
   both hosts).
3. **Backbone config assets** — the backbone's config/processor files
   (`nvidia/Cosmos-Reason2-2B`); the official constructor's redundant
   base-weight download is redirected to these, construction-I/O only.
4. **This repository** on `PYTHONPATH`, plus this folder for the
   cross-host runners (`build()` is imported from `groot_n17.py`).
5. **Kernel packages** resolve from the Hugging Face Hub at first
   bind; offline, stage them as `<org>/<name>/build/<variant>/` and
   point the kernels resolver at the directory.

Environment notes, learned the hard way and probed in code rather than
pinned: the capture lowering supports both transformers vision-contract
generations (tuple-return and output-class); the LeRobot loading recipe
`from_pretrained(...).to(dtype=bf16)` casts the rotary `inv_freq`
buffer to BF16 — a host-side precision loss the README documents, and
the reason the `qkv_rope` family refuses on that host.

## Step by step

```bash
# 0) one observation fixture from the host's own preprocessing:
#    run the host policy once on any observation and save it —
#    torch.save({"inputs": observation_dict}, "obs_fixture.pt")

# 1) official host — explicit assembly, eager + compiled ladder
python groot_n17.py --host <Isaac-GR00T> --checkpoint <ckpt> \
  --backbone-assets <cosmos-reason2-assets> --fixture obs_fixture.pt \
  --compile --report official_explicit.json

# 2) official host — the same assembly, captured (deployed form)
python full_graph.py --host <Isaac-GR00T> --checkpoint <ckpt> \
  --backbone-assets <cosmos-reason2-assets> --fixture obs_fixture.pt

# 3) export the prepared model-level inputs both hosts consume
python make_model_inputs.py --host <Isaac-GR00T> --checkpoint <ckpt> \
  --backbone-assets <assets> --fixture obs_fixture.pt \
  --out model_inputs.pt

# 4) LeRobot host — baseline / auto, eager + compiled
python lerobot_host.py --lerobot-src <lerobot>/src \
  --checkpoint <ckpt> --inputs model_inputs.pt \
  --arm auto --compile --report lerobot_auto.json

# 5) LeRobot host — explicit and auto at the captured form
python lerobot_full_graph.py --lerobot-src <lerobot>/src \
  --checkpoint <ckpt> --inputs model_inputs.pt --arm explicit
python lerobot_full_graph.py --lerobot-src <lerobot>/src \
  --checkpoint <ckpt> --inputs model_inputs.pt --arm auto
```

Step 3 is what makes the two hosts comparable: both consume the same
prepared tensors, so host code is the only variable.

## What to expect

The README carries the measured matrix (RTX 5090). Judge a
reproduction by, in order:

1. `ledger.fallbacks` and `refused` match in *kind* (a refusal is an
   outcome; a silently missing seat is a bug);
2. parity ≥ 0.999 on every arm, against that host's own eager run;
3. speedups within a form row land in the same band — clocks move a
   few percent between runs, so read ratios, not milliseconds across
   processes.

Region adjudication in reports: the automatic arm's `seats` block
carries `regions` (the resolution trail: winner, source, fall-through
reasons), `regions_bound` (root, claimed seam count, smoke cosine) and
`regions_refused`. A cold box shows `seated (default)` and unchanged
numbers; a box with a recorded receipt grows the winning form from
`source=cache`. The receipts themselves live in the decision cache and
travel with `import_decisions`, and `handle.manifest()` lists every
entry for this device — bands, regions, and pairing receipts alike.
