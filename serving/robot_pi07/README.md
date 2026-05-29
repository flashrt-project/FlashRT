# serving/robot_pi07 — hierarchical two-VLA host (π0.7-style)

A serving host for the π0.7 multi-model hierarchy, built on the FlashRT
execution contract.

## What π0.7 does (and what we simplify)
π0.7's runtime is a multi-model hierarchy (paper Fig. 2):
- **High-Level Policy** (SigLIP+Gemma) → emits a **subtask** instruction;
- **World Model** (BAGEL 14B) → emits **subgoal images**;
- **π0.7 action VLA** → consumes subtask + subgoal images → **actions**.

We **drop the BAGEL world model** and model the two-stage hierarchy:

```
  PLANNER (low rate) --subtask (shared Buffer)--> ACTOR (high rate) --> actions
                                ▲
        interrupt / verbal coaching: overwrite the subtask buffer (no recapture)
```

## What it verifies (multi-model hot-path mechanism)
`verify_pi07.py` co-hosts **two Pi05 instances** through **ONE** `frt_ctx`:
- two adopted graphs (planner + actor) driven from one host on two streams;
- **PLANNER → ACTOR hand-off** through a shared buffer (`frt_buffer_copy`),
  verified byte-equal (planner output == subtask buffer == actor input);
- **multi-rate**: PLANNER runs once every N ACTOR ticks (1:4 measured);
- **interrupt**: overwrite the subtask buffer mid-run (verbal coaching) — the
  next ACTOR tick consumes the new subtask, **no recapture**.

This is the sequential-hand-off counterpart to `serving/robot_recap/` (which is
the concurrent policy‖critic rollout pattern) — together they cover the two
multi-model shapes the contract is built for.

## Usage (reproducible)

**Prerequisites**

- A CUDA GPU; the FlashRT runtime built with the Pi0.5 path (FP8 frontend used
  here), and the execution-contract module `_flashrt_exec` built
  (`cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release && cmake --build exec/build -j`).
- A Pi0.5 checkpoint directory.

**Run**

```bash
PYTHONPATH=.:./exec/build \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python serving/robot_pi07/verify_pi07.py --checkpoint /path/to/pi05_libero_pytorch
```

**Arguments**

| flag | default | meaning |
| --- | --- | --- |
| `--checkpoint` | (required) | Pi0.5 checkpoint directory |
| `--num-views` | `3` | camera views per observation |
| `--ticks` | `8` | total ACTOR ticks to run |
| `--planner-every` | `4` | run the PLANNER once every N ACTOR ticks (multi-rate) |

**Expected output**: per-tick lines showing the actor acting and the planner
running every 4th tick, a mid-run `INTERRUPT` line where the subtask buffer is
overwritten (planner→subtask and subtask→actor hand-offs asserted byte-equal),
and a final `PASS — pi0.7-sim hierarchy ...` summary. Actions are checked finite;
the hand-off and interrupt are checked byte-exact.

## Honest scope
Two Pi05 stand in for planner + actor (in real π0.7 they differ in role/size);
the subtask hand-off is plumbing (planner output → subtask buffer → actor
input), not a semantic planner→language mapping. We verify the **contract
orchestration** (co-host, hand-off, multi-rate, interrupt), not VLA semantics.
Setup (capture) is done once by the in-process Python frontend; the host then
drives replay via the contract.
