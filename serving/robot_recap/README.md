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
- `verify_recap.py` — the advantage-conditioned RL/CFG inference
  (`set_rl_mode`, `Pi05CFGPipeline`) driven by the exec contract, **bit-identical
  to ctypes replay (cosine 1.0)**. Verifies the RL inference path on the contract.
- `rollout_host.py` — the full rollout host: policy (Pi05 CFG) + a real
  lightweight value critic (`StandaloneValueFunction`) co-hosted via ONE exec
  ctx; episode state machine with per-chunk interruptible replay, keyboard/auto/
  timeout stop, and buffer reset between episodes. Verifies the hot-path
  mechanism (clean STOP at episode boundary, multi-model concurrency, reset).

## Notes (honest scope)
- Mechanism demo: it reuses the captured policy chunk with a restored noise
  buffer; production writes fresh observations each chunk. The value critic is
  random-initialized (swap in a trained `StandaloneValueFunction`); we verify the
  rollout *mechanism*, not RL semantics.
- Real RECAP inference at deployment is a single advantage-conditioned policy
  (CFG = batched cond+uncond in one graph); the value function is the second
  model used during rollout/data-collection (and here as a runtime critic).
