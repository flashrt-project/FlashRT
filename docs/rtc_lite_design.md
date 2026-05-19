# RTC-lite Runtime Design

RTC-lite is an execution-layer wrapper for action chunk policies. It does not
change model weights, denoising math, CUDA kernels, or calibration. The goal is
to keep a foreground controller supplied with actions while the model generates
the next action chunk in a background worker.

## Scope

RTC-lite is intended for policies with this contract:

```text
observation -> action_chunk[horizon, action_dim]
```

Examples:

- Motus-style models: `infer(...) -> (frames, actions)`
- Pi0/Pi0.5-style models: `infer(observation) -> {"actions": actions}`

The runtime does not know about image preprocessing, prompt setup, CUDA Graph
capture, or robot transport. Those remain in the model frontend and deployment
layer.

## Components

`flash_rt.runtime.rtc.ActionChunkAdapter`

Minimal model adapter with one method:

```python
infer_actions(observation) -> np.ndarray  # [horizon, action_dim]
```

`flash_rt.runtime.rtc.CallablePolicyAdapter`

Small helper for existing frontends. It can extract actions from either a dict
return value or a tuple return value.

`flash_rt.runtime.rtc.AsyncChunkRunner`

Owns one model worker thread. The foreground loop calls `next_action()` once per
controller tick. The first call blocks to initialize the first chunk. Later
calls consume actions from the current chunk and submit the next chunk before
the current chunk is exhausted.

`flash_rt.runtime.rtc.RTCStats`

Tracks chunk starts/completions, action count, deadline misses, held actions,
and model latency.

## Execution Model

```text
foreground control loop:
  obs = latest_robot_observation()
  action = runner.next_action(obs)
  robot.step(action)

background worker:
  actions = adapter.infer_actions(latest_obs)
```

The foreground loop owns timing. RTC-lite does not sleep, run robot IO, or
change the frontend's CUDA stream policy.

## Policies

`miss_policy="hold_last"`

If the next chunk is not ready when the current chunk is exhausted, repeat the
last action and count a deadline miss. This is conservative and keeps the
controller loop non-blocking.

`miss_policy="block"`

Block until the model returns the next chunk. This is useful for offline
diagnostics, but it reintroduces pauses and is not the default deployment mode.

`blend_steps`

Optional small boundary smoothing. This is deliberately simple; it does not run
gradient guidance through the policy.

## What This Is Not

RTC-lite does not implement a new policy or train-time chunking method. It is an
inference scheduling layer. It validates action supply at a fixed controller
rate; robot task success still has to be measured in the target environment.
