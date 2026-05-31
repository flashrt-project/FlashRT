# serving/robot_handoff — hierarchical planner→actor buffer hand-off

A serving host for a **multi-model robot hierarchy** built on the FlashRT
execution contract: a low-rate **planner** hands a subtask to a high-rate
**actor** through a shared `Buffer`, optionally via a **world model** that turns
the subtask into subgoal images. It is the *buffer hand-off* counterpart to
[`../robot_recap/`](../robot_recap/) (the concurrent policy‖critic rollout
pattern) — together they cover the two multi-model shapes the contract is built
for.

The hierarchy shape is **inspired by** π0.7's runtime (High-Level Policy →
World Model → action VLA); this host does **not** run π0.7 — it co-hosts models
we actually have to verify the *contract orchestration mechanism*, not VLA
semantics. The directory is named for the mechanism (`robot_handoff`), not for any
external model.

## The pattern

```
  PLANNER (low rate) --subtask--> [WORLD MODEL] --subgoal imgs--> ACTOR (high rate) --> actions
       (Pi05)        shared Buffer    (Wan2.2)    shared Buffer       (Pi05)
                              ▲
        interrupt / verbal coaching: overwrite the subtask buffer (no recapture)
```

- **PLANNER / ACTOR**: two Pi05 instances (stand-ins for the two roles; in a real
  hierarchy they differ in role/size).
- **WORLD MODEL (optional)**: a video diffusion model (Wan2.2-TI2V-5B) stands in
  for π0.7's BAGEL world model — subtask → subgoal frames. This is the honest
  substitute for "drop the world model": use a model we have rather than claim one
  we don't.

## What `verify_handoff.py` checks (multi-model hot-path mechanism)

Co-hosts the models through **ONE** `frt_ctx` and verifies, byte-exact where
noted:

- two (or three) adopted graphs driven from one host on separate streams;
- **hand-off through a shared buffer** (`frt_buffer_copy`), verified byte-equal
  (producer output == shared buffer == consumer input);
- **multi-rate**: the planner runs once every N actor ticks (1:4 measured);
- **interrupt**: overwrite the subtask buffer mid-run (verbal coaching) — the next
  actor tick consumes the new subtask, **no recapture**.

## Usage (reproducible)

**Prerequisites**

- A CUDA GPU; the FlashRT runtime built with the Pi0.5 path (FP8 frontend), and
  the execution-contract module `_flashrt_exec` built
  (`cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release && cmake --build exec/build -j`).
- A Pi0.5 checkpoint directory (and, for the world-model stage, a Wan2.2-TI2V-5B
  checkpoint).

**Run (two-stage planner→actor)**

```bash
PYTHONPATH=.:./exec/build \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python serving/robot_handoff/verify_handoff.py --checkpoint /path/to/pi05_libero_pytorch
```

**Arguments**

| flag | default | meaning |
| --- | --- | --- |
| `--checkpoint` | (required) | Pi0.5 checkpoint directory |
| `--num-views` | `3` | camera views per observation |
| `--ticks` | `8` | total actor ticks to run |
| `--planner-every` | `4` | run the planner once every N actor ticks (multi-rate) |

**Expected output**: per-tick lines showing the actor acting and the planner
running every 4th tick, a mid-run `INTERRUPT` line where the subtask buffer is
overwritten (planner→subtask and subtask→actor hand-offs asserted byte-equal),
and a final `PASS` summary. Actions are checked finite; the hand-off and interrupt
are checked byte-exact.

## Honest scope

Two Pi05 stand in for planner + actor (in a real hierarchy they differ in
role/size); the optional world-model stage uses Wan2.2 in place of π0.7's BAGEL.
The subtask hand-off is plumbing (producer output → shared buffer → consumer
input), not a semantic planner→language mapping. We verify the **contract
orchestration** (co-host, hand-off, multi-rate, interrupt, no recapture), not VLA
semantics. Setup (capture) is done once by the in-process Python frontend; the
host then drives replay via the contract.
