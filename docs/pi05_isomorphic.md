# Pi0.5 on the arithmetic of its trainer

`Pi05TorchFrontendIsomorphic` serves Pi0.5 rollouts that the trainer
reproduces bit for bit, and carries that trainer. It runs on
[FlashRT-Train](https://github.com/LiangSu8899/FlashRT-Train)'s policy
schedule rather than on FlashRT's own kernels. SM80 (A100 / A800), BF16.

## Why a second Pi0.5 path

An on-policy update (PPO and its relatives on a flow policy) recomputes the
velocity of every recorded denoising transition and weighs the transition by
the ratio of its probability under the trainer's forward to its probability
under the rollout's. Before the first update those are the same policy and
the ratio is 1 by definition. It is 1 in practice only if the trainer's
forward returns the rollout's velocity.

The native pipelines do not aim at that. They own their arithmetic: a BF16
denoising state, fused projections, time conditioning precomputed per step,
an output projection pre-scaled by the step size. A trainer lands near their
rollouts, not on them, and a flow policy makes the distance count: one
transition of a 50 × 32 action chunk is 1 600 Gaussian dimensions, and its
log-probability sums their shifts.

`tools/check_pi05_isomorphism.py --native` reads the distance on the A800
(the π0.5 base checkpoint, three views, horizon 50, ten steps, transition
noise 0.1; every recorded state of a rollout handed to FlashRT-Train's
training forward in shuffled minibatches of 3):

| rollout by | batch | step elements that differ from the trainer's | relative rms | log-ratio of a recorded transition, rms | ratio range |
|---|---|---|---|---|---|
| native BF16 pipeline | 1 | 15 999–16 000 of 16 000 | 5.6e-3 | 0.11–0.24 (by run) | 0.63–1.54 |
| native BF16 pipeline | 8 | 127 998–128 000 of 128 000 | 5.9e-3 | 0.17 | 0.63–1.47 |
| native BF16 pipeline | 32 | 511 997–512 000 of 512 000 | 6.0e-3 | 0.20 | 0.49–1.83 |
| isomorphic | 1, 8, 32 | 0 of 16 000, 128 000, 512 000 | 0 | 0 | 1–1 |

(The native rows use one prompt for every sample; see
[pi05_sm80.md](pi05_sm80.md) for mixed prompt lengths.) A ratio that spreads
from 0.5 to 1.8 before anything has been learned is wider than a PPO clip
range of 0.2. The native pipeline remains the right one for serving a
policy; this frontend is for rollouts that feed an on-policy trainer.

## What it guarantees

- **Replay.** The trainer's forward at a recorded `(x, t)` returns the
  rollout's velocity bit for bit, in any minibatch size and order, whether
  it takes the observation again (`policy_forward`) or the kept prefix
  (`policy_forward_kept`).
- **Width.** A sample's rollout does not depend on how many samples share
  its batch: `infer` and any slot of `infer_batch` at any width agree in
  every bit.
- **Process.** The same inputs give the same rollout in another process on
  the same device model and library (four processes, one digest per width).
- **No weight synchronization.** The rollout and the update read the same
  parameter buffers; an optimizer step is visible to the next rollout with
  nothing copied or reloaded.

The check's exit status is the first two; `tests/test_sm80_routing.py`
covers the routing.

## Use

```bash
# FlashRT-Train, branch policy-loop (the policy schedule, C ABI 11): build its native library for the device, then
export PYTHONPATH=<FlashRT>:<FlashRT-Train>
export FRT_TRAIN_LIB=<FlashRT-Train>/build/libflashrt_train.so
```

FlashRT's own kernel module is not needed for this frontend.

```python
from flash_rt.frontends.torch.pi05_isomorphic import Pi05TorchFrontendIsomorphic

fe = Pi05TorchFrontendIsomorphic(checkpoint, max_prompt_len=48,
                                 denoise_trace=True, sde=True, trainable="expert")
fe.set_prompt_batch(prompts)                       # one per environment, any width
results = fe.infer_batch(observations, sde_sigma=sigma)

roll = fe.last_rollout                             # device tensors: trace_x, trace_v, images, tokens, mask, ...
records = fe.export_prefix()                       # per observation, for the replay buffer

trainer = fe.trainer                               # flashrt_train.native.Trainer, objective "policy"
trainer.policy_import(records[idx])                # any subset, any order
trainer.policy_forward_kept(x, t, v)               # v == the rollout's velocity, bit for bit
# ... the caller's surrogate from v (PPO, weighted regression, anything) ...
trainer.policy_backward(d_loss_d_v)
trainer.optimizer_step(lr)                         # the next infer_batch reads the new weights
```

`examples/pi05_isomorphic_loop.py` is this loop end to end with a stand-in
advantage: rollout, replay (0 of 115 200 elements differ at width 8 in
minibatches of 4, and the first update's ratios are exactly 1), four
updates of the action expert over the kept prefix, rollout again.

| argument | meaning |
|---|---|
| `num_views` | three; the trainer's forward takes three cameras |
| `chunk_size` | the checkpoint's action horizon (50 for the base model) |
| `max_prompt_len` | text columns of the forward. Every prompt is padded to it under a mask, so the layout does not depend on the other prompts of the batch; shorter is faster (see below) |
| `denoise_trace`, `sde` | as on the native frontend; the trace holds `x`, `v` and `delta = dt * v` in fp32 |
| `trainable` | `"none"` serves only; `"expert"` trains the action expert over a frozen prefix; `"lora"` trains the PEFT adapter in `adapter` |
| `sampler_graph` | the denoising steps as one captured device graph (default on): the same bits; one layout (batch, steps, sampler, trace) is held at a time |
| `arena_gib` | device memory of the trainer's activation arena |
| `trainer_kwargs` | further `Trainer` arguments, e.g. `gemm_reference_width` |

`infer` / `infer_batch` take `noise`, `generator`, `step_noise`,
`sde_sigma` and `return_noise` with the native frontend's meaning, and
return `actions`, `raw_actions` and the trace per sample.
`load_model(config="pi05_isomorphic")` redirects here: the surface is a
rollout and an update, not `predict(images, prompt)`.

## Readings

A800 80 GB, same setup as [pi05_sm80.md](pi05_sm80.md), medians of nine
calls; the whole call including host staging.

| batch | native BF16, per sample | isomorphic (48 text columns), per sample |
|---|---|---|
| 1 | 38.8 ms | 90.1 ms |
| 8 | 26.1 ms | 29.2 ms |
| 32 | 26.2 ms | 24.9 ms |

At batch 32 the isomorphic path is 5% ahead; at batch 8 it costs 12%; for a
single sample it is 2.3× the native pipeline. The full 200-column text
layout adds 3% at batch 1 and 12% at 8.

The denoising steps run as one captured device graph (`sampler_graph=True`,
the default; the same bits as without it), which is worth 6% of a
single-sample rollout and less above. The single-sample cost is not
dispatch and not the bit-exactness itself: a velocity step is about 480
small kernels where the native pipeline has fused them and precomputed the
time conditioning, and its 50-row GEMMs run the algorithm cuBLASLt picks
for eight samples' rows (`gemm_reference_width`, the width that costs
nothing; 8 by default). Both are open work in FlashRT-Train.

## Scope

- SM80, one device model and one cuBLAS release: the algorithms are pinned
  per operator on that library. Across device models nothing is claimed.
- Three views, the plain vision tower, no state other than through the
  prompt, the BF16 profile of the openpi PyTorch model.
- The precision is the trainer's (openpi's BF16 profile), not a deployment
  tier: there is no FP8 or NVFP4 form of this path, and none on SM80.
- The checkpoint's own norm stats are required, as on the native frontend.
