# Attaching structures to a serving engine

A model library is a friendly host: you own the module tree and can put
anything anywhere. A serving engine is not. It owns its own model
implementation, its own memory planner and its own graph capture, and it
compiles the model before you get a chance to touch it.

This page is how to attach anyway — on vLLM and on SGLang — what the
engine facts are that shape the call, and what the refusals mean when it
does not work.

Nothing here forks or patches the engine. The engine stays in charge; the
layer replaces sub-blocks of the model it executes.

---

## 1. Install

```bash
pip install flash-rt              # the layer: catalog, adapters, specs
pip install 'flash-rt[hub]'       # + the kernel-hub client
```

The distribution is pure Python and carries **no compiled artifact**. Two
things follow:

- Kernels arrive at bind time from the kernel hub. Without the hub client
  every bind refuses — legibly, naming the package it wanted.
- The native `flash_rt.load_model` path is a different thing entirely and
  needs a local build. See the README's Build section.

Bring your own torch. The extra deliberately does not name it, because
**which kernels exist is decided by your torch version** — the published
face is thickest at torch 2.11 and thin at the newest release. A bind
that refuses on a fresh install is usually saying that, not saying the
package is broken.

Qualified on Python 3.10 / 3.11 / 3.12 / 3.13, Linux x86-64 and aarch64.

---

## 2. vLLM

### The four lines

```python
from flash_rt.structures.adapters import vllm_engine
vllm_engine.install_load_hook()

from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B", max_model_len=1024)
```

That is the whole integration. `install_load_hook()` must run **before**
the engine is constructed.

### Why a hook and not "attach the model"

Three engine facts, each of which cost a debugging session:

- **Seats go in after weights load and before the first trace.** vLLM's
  compiled artifact resolves parameters by tree path, so a swap made
  after compilation either raises `KeyError` or is silently bypassed.
  `install_load_hook` patches the model runner's `load_model` for exactly
  that window.
- **The compile cache does not see the module tree.** A stale artifact
  would resolve parameters the seats replaced, so the hook sets
  `VLLM_DISABLE_COMPILE_CACHE=1` for you.
- **A Python shape branch dies under tracing.** vLLM traces with guard
  evaluation off, so the MoE band (decode rows to the packed bank,
  prefill rows back to the host) lives inside a `torch.library.custom_op`
  registered at import. A plain `if rows > N` freezes at trace time and
  the arm that looks enabled is dead code.

### Call it before anything touches CUDA

This is the one trap that costs a debugging session, because it fails
without raising anything.

The engine **forks** its worker by default, which is what carries the
patch into the process that loads the model. But it switches to
**spawning** one the moment CUDA is already initialized in your process —
and a spawned worker re-imports the engine from scratch, so the patch
simply is not there. Checking free memory, calling
`torch.cuda.is_available()`, setting a device, a warm-up, or importing
any library that initializes CUDA is enough to flip it.

```python
import torch
torch.cuda.mem_get_info()          # ← now the worker will be spawned
vllm_engine.install_load_hook()    # patches this process, which is not
llm = LLM(model=...)               #   the one that loads the model
```

Nothing raises. `install_load_hook()` really did patch, so it does not
report "no vLLM model runner found"; the callback just never fires, and
the run comes out at baseline speed. It reads exactly like "this layer
does nothing".

`install_load_hook()` refuses this state rather than let it happen
silently. If you have a reason to proceed anyway, pass
`allow_spawn=True`. Two ways out:

- call `install_load_hook()` before anything touches CUDA, or
- run the engine in this process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
  which is independent of initialization order.

**Verify rather than assume.** The seats are in if and only if this line
appeared:

```
[structures.vllm] 144 seats (5 head slabs), 0 refused
```

For a test or a service that should not start unaccelerated, assert it:

```python
llm = LLM(model=...)
assert vllm_engine.attached(), "the hook never fired"
```

`strict=True` does not help here — it governs what happens when seats
refuse, and in this failure attach never runs at all.

### Reading what happened

```
[structures.vllm] 144 seats (5 head slabs), 0 refused
```

`on_attached` hands you the handle if you want the detail:

```python
def note(handle):
    print(handle.summary())          # seams, guarded_calls, fallbacks, clean
    print(handle.notes["refused"])   # [(site, reason), ...]

vllm_engine.install_load_hook(on_attached=note)
```

`handle.detach()` restores the module tree, the expert modules and the
head's quant method — same objects, bit-identical outputs.

### Options

```python
vllm_engine.install_load_hook(
    seats=vllm_engine.DENSE_SEAT_SUFFIXES,  # dense projection positions
    experts=True,                           # routed expert banks
    head=True,                              # LM head, row-sliced
    strict=False,                           # see below
)
```

`strict=False` (the default) means a host where **every** seat refuses
still starts, unaccelerated, with every refusal reported. A server that
fails to boot because its accelerator was absent is worse than one that
boots slow. In CI use `strict=True`: a silent zero-seat run is a fallback
nobody reads.

---

## 3. SGLang

SGLang runs its scheduler in a **spawned subprocess**, so patching in the
parent does not reach the process that owns the model. The door writes a
`sitecustomize` onto `PYTHONPATH` instead, and the child picks it up:

```python
from flash_rt.structures.adapters import sglang_engine
sglang_engine.install()

import sglang as sgl
llm = sgl.Engine(model_path="Qwen/Qwen3-8B", mem_fraction_static=0.75)
```

`install()` must run before the engine is constructed, and it sets
`FRT_SGLANG_ATTACH=1` so other processes that happen to inherit the path
see a dormant flag rather than an attach.

Two things that bite:

- **Your driver script needs an `if __name__ == "__main__":` guard.** The
  spawned child re-imports the main module by path; without the guard it
  re-enters `Engine(...)` and rank 0 dies during init.
- `mem_fraction_static` needs headroom for the bind. Binding packs weights
  on the device, and the transient peak is what fails under a tight budget.

For a host where the weights barely fit, `install(release=True)` attaches
in slabs and releases each original as its replacement lands, instead of
holding both for the whole pass.

---

## 4. Batch size, and the knob that decides it

The MoE seat declares a **band**: at or below a row count it serves the
batch from the packed bank; above it, the host's own fused-MoE module
serves. Prefill always goes back to the host.

```bash
FRT_MOE_BAND_T=16   # default; the largest batch measured to pay
```

The default is measured, not assumed. On a routed-MoE host the gain does
**not** thin out with batch the way it does on a dense host — each token
picks its own experts, so expert weight traffic grows with the batch
instead of being shared, and a 4-bit bank keeps paying well past batch 1.
An earlier value of 8 handed batch-16 traffic back to the host and threw
that away.

Sweep it on your own host before changing it. `tests/thor_parasite_kit/`
carries a concurrency probe that runs paired arms at a given concurrency:

```bash
MODEL=<your-model> ARM=base   CONC=8 OUT=base.json  python 04_concurrency_probe.py
MODEL=<your-model> ARM=attach CONC=8 OUT=attach.json python 04_concurrency_probe.py
```

Use distinct prompts (the probe does): identical ones share prefix-cache
blocks and quietly turn N requests into one request's worth of work.

Two measurement notes, both learned the hard way:

- **Report prefill and steady decode separately.** An end-to-end average
  hides which one moved.
- **Some hosts run per-process bimodal.** A single pair can land in the
  slow mode and misstate the result by 10%. Run repeats and read the fast
  mode; a spread wider than the effect means you have not measured yet.

---

## 5. Kernels: where they come from, and air-gapped hosts

Binds pull packages from the kernel hub through the `kernels` client.

**`HF_HUB_OFFLINE=1` does not work as an air-gap strategy**: a version
specifier has to resolve refs online, so offline mode makes every package
unavailable even against a fully warm cache. Stage the packages and point
at them instead:

```bash
export LOCAL_KERNELS="<repo>=<path>:<repo>=<path>"
```

To pin one repository to an exact artifact — for bisecting a performance
or correctness drift that arrived with a rebuild:

```bash
export FRT_KERNEL_REV_<REPO_NAME_UPPERCASED_WITH_UNDERSCORES>=<revision>
```

### Coverage is per package *and per version*, which is where it bites

A package having a build for your architecture is not the same as the
version being resolved having one. A repository whose newest revision
covers aarch64 can have an older one that does not, and a request that
resolves to the older revision refuses on that architecture with

```
CPU (x86_64) does not match system CPU (aarch64)
```

This has a specific consequence worth knowing before you read a
disappointing number: on a routed-MoE host it is the **expert bank**
seats that carry the concurrency gain, so if that one package resolves
to a build without your architecture, every expert seat refuses while
the dense seats attach normally. The run is correct and the log says
`N seats, 40 refused` — but the batch-16 gain is most of what is gone.

So read the refusal count, not just the speedup:

```python
handle.notes["refused"]        # (site, reason) per seat that declined
impls.unavailable_report()     # per package: repo, version, error, detail
```

A large refused count concentrated on one seam family is a coverage
report, not a mystery. Pin a revision that has your architecture with
`FRT_KERNEL_REV_*`, or stage it and point at it with `LOCAL_KERNELS`.

---

## 6. Refusals

A refusal is an outcome, not a crash. `KernelUnavailable` is a
`ValueError` on purpose, so layers that record a refusal and keep going
can catch it.

| what you see | what it means | what to do |
|---|---|---|
| `the kernel client is not installed` | no hub client in this environment | `pip install 'flash-rt[hub]'` |
| `unavailable on this host: OfflineModeIsEnabled` | offline mode defeats version resolution | use `LOCAL_KERNELS`, see §5 |
| `no build variant for this host` | the package has no build for your torch/CUDA/arch | check torch version first, §1 |
| `quantize ... rc=-1` on a large head | shape outside the entry's support | the head binds as row slabs; this is the refusal that strategy exists for |
| `refused: no vLLM model runner found to hook` | the engine layout is outside this adapter's profile | a host version this adapter has not been fitted to |
| `0 seats, N refused — host runs unmodified` | nothing could be seated | read the reasons; the engine is up and correct, just unaccelerated |

Everything the process could not get is on the ledger:

```python
from flash_rt.structures import impls
impls.unavailable_report()   # [{repo, version, error, detail}, ...]
```

The original failure is kept verbatim, because a package that is *broken*
here and one that was simply never *shipped* here both look like "skipped"
and only the first is somebody's bug.

---

## 7. What has actually been measured

Numbers below are single-stream steady decode unless stated. They are
receipts from specific hosts, not promises about yours — the point of the
band and the gates is that your host decides.

| host | engine | model | base → attached |
|---|---|---|---|
| consumer GPU (sm_120) | vLLM 0.22 | Qwen3-8B | 99.4 → 157.1 tok/s (144 seats + 5 head slabs) |
| consumer GPU (sm_120) | SGLang 0.5.13 | Qwen3-8B | 101.0 → 201.5 tok/s (144 seats) |
| edge module (sm_110) | vLLM 0.26 | 35B-A3B MoE | 37.2 → 76.0 tok/s (200 seats + 4 head slabs) |

Concurrency, on the edge module with the MoE model — the shape is the
point, not the absolutes:

| concurrency | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| gain | 2.05× | 2.31× | 2.46× | 1.65× |

It **peaks at 8 and is still positive at 16**, which is not what a dense
model would do. Above the band the MoE seat stands down by design, so
"out of band" means "no worse than the host", not "broken".

Verified interpreter and platform coverage: Python 3.10–3.13, Linux
x86-64 and aarch64, with and without a GPU present.
