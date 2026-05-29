# serving/robot_recap — RL Rollout Host (π*0.6 / RECAP-style)

A serving host for **RL rollout / data collection** with an advantage-conditioned
VLA, built on the FlashRT execution contract. Models the π*0.6 (RECAP) pattern:
an advantage-conditioned policy + a value-function critic, driven by a host-side
**episode state machine**.

## The real problem it solves
From a community RL user:
> "inference keeps running between episodes — I can't stop to reset the robot
>  or record with a keyboard. Do I need to code my own rollout strategy?"

Root cause: a monolithic `while True: act(obs)` loop with no episode-boundary
control. The fix is **not** a smarter policy — it is a host-driven episode state
machine on top of the contract's **interruptible, per-chunk replay**:

```
  RESET -> RUNNING --(value<thr / keyboard / timeout)--> STOP_INFER
    ^                                                        |
    +------ RESET(buffers) <- RECORD <- AWAIT_RESET <--------+
```
Because each action chunk is one short replay and the **host** fires them one at
a time, inference halts cleanly at an episode boundary (interrupt granularity =
one chunk). Episode reset = reinit state buffers, **no recapture**.

## Contract mechanism vs serving policy
- Contract (mechanism): per-chunk replay, multi-model concurrency (policy ‖
  critic on separate streams via ONE `frt_ctx`), buffer reset.
- This host (policy): the episode state machine, keyboard/intervention handling,
  termination conditions, recording, reset. **Never in the contract.**

So the community user does not write a rollout engine from scratch — they reuse
this host and plug in termination conditions + reset hooks. ("Dedicated rollout
model" setups like AgiBot's are exactly this: a dedicated rollout-serving stack,
separate from training.)

## Files
- `verify_capsule.py` — the robot side of "one capsule, two scenarios": the
  episode-boundary snapshot/restore done through the execution contract's Buffer
  copy, verified **bit-identical** (cosine 1.0). Episode reset *is* a capsule
  restore — the same mechanism as the LLM agent capsule
  (`serving/qwen36_agent/capsules.md`), see `docs/serving_design.md`.
- `verify_recap.py` — the advantage-conditioned RL/CFG inference
  (`set_rl_mode`, `Pi05CFGPipeline`) driven by the exec contract, **bit-identical
  to ctypes replay (cosine 1.0)**. Verifies the RL inference path on the contract.
- `rollout_host.py` — the full rollout host: policy (Pi05 CFG) + a real
  lightweight value critic (`StandaloneValueFunction`) co-hosted via ONE exec
  ctx; episode state machine with per-chunk interruptible replay, keyboard/auto/
  timeout stop, and buffer reset between episodes. Verifies the hot-path
  mechanism (clean STOP at episode boundary, multi-model concurrency, reset).

## Usage (reproducible)

**Prerequisites**

- A CUDA GPU; the FlashRT runtime built with the Pi0.5 path (FP8 frontend used
  here), and the execution-contract module `_flashrt_exec` built
  (`cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release && cmake --build exec/build -j`).
- A Pi0.5 checkpoint directory.

All three scripts take `--checkpoint` and run inside the CUDA container with:

```bash
PYTHONPATH=.:./exec/build PYTORCH_ALLOC_CONF=expandable_segments:True \
python serving/robot_recap/<script>.py --checkpoint /path/to/pi05_libero_pytorch
```

| script | extra flags (default) | what it prints |
| --- | --- | --- |
| `verify_recap.py` | `--num-views 3` `--steps 10` `--cfg-beta 1.5` | RL/CFG inference driven by the contract, cosine vs ctypes replay; `PASS` at cos ≥ 0.999 |
| `verify_capsule.py` | `--num-views 3` `--steps 10` | episode-boundary snapshot/restore via the contract; `PASS` when restore is bit-identical (cos 1.0) |
| `rollout_host.py` | `--episodes 3` `--max-chunks 8` `--value-stop-threshold 0.0` `--record-dir DIR` `--num-views 3` | one line per episode (chunks run, STOP reason: keyboard / value / timeout, recorded chunks), then a `PASS` summary |

`rollout_host.py --record-dir DIR` writes one `episode_*.npz` per episode (actions,
values, stop reason). The keyboard start/end and robot reset hooks are scripted
no-ops by default — swap in `pynput`/`termios` and your robot driver for teleop.

## Notes (honest scope)
- Mechanism demo: it reuses the captured policy chunk with a restored noise
  buffer; production writes fresh observations each chunk. The value critic is
  random-initialized (swap in a trained `StandaloneValueFunction`); we verify the
  rollout *mechanism*, not RL semantics.
- Real RECAP inference at deployment is a single advantage-conditioned policy
  (CFG = batched cond+uncond in one graph); the value function is the second
  model used during rollout/data-collection (and here as a runtime critic).
