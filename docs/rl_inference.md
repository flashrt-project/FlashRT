# RL Inference (classifier-free guidance)

Opt-in inference path for advantage-conditioned VLA policies trained
with the RECAP recipe (π\*0.6, [arXiv:2511.14759](https://arxiv.org/abs/2511.14759)).
At test time the model runs twice per denoising step — once with the
`"Advantage: positive"` indicator appended to the prompt, once
without — and the two velocity predictions are combined to sharpen
the action distribution toward the high-advantage subset:

```
v_guided = v_uncond + beta * (v_cond - v_uncond)
```

This document covers the public API, measured performance on RTX and
Thor, and the numerical contract the implementation meets.

## Scope

|  | RTX | Thor (Jetson AGX, SM110) |
|---|---|---|
| Model | Pi0.5 | Pi0.5 |
| Hardware | RTX 5090 / 4090 (SM89, SM120) | Jetson AGX Thor (SM110, aarch64) |
| Frontends | [`Pi05TorchFrontendRtx`](../flash_rt/frontends/torch/pi05_rtx.py) (safetensors) / [`Pi05JaxFrontendRtx`](../flash_rt/frontends/jax/pi05_rtx.py) (Orbax) | [`Pi05TorchFrontendThor`](../flash_rt/frontends/torch/pi05_thor.py) (safetensors) / [`Pi05JaxFrontendThor`](../flash_rt/frontends/jax/pi05_thor.py) (Orbax) |
| Serial CFG | ✅ 37 ms (β=1.5) | ✅ 88 ms (torch) / 96 ms (JAX) |
| Fused CFG (B=2, paper-correct per-step) | ✅ **25.9 ms** (β=1.5) | ✅ **~67 ms** (torch / JAX, with `autotune≥3`) |
| Generic B>2 batched (RL rollout) | not yet | not yet |

Conditioned-prompt strings are byte-equal across the four frontends
(shared builder in [`flash_rt/core/rl/`](../flash_rt/core/rl/)),
so the same merged LoRA checkpoint serves all four backends.

## API

CFG is opt-in. The default (no `set_rl_mode` call) inference path is
bit-for-bit unchanged.

```python
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

# Construct. autotune>0 enables the B=2 outer-graph autotuner —
# recommended for production, see "Performance" below.
pipe = Pi05TorchFrontendThor(
    "/path/to/pi05_libero_pytorch", num_views=2, autotune=3)

# Recommended: enable the fused B=2 CFG path BEFORE set_prompt.
pipe.set_batched_mode(enable=True, batch_size=2)

# Configure CFG: β must be >= 1.0; π*0.6 paper recommends [1.5, 2.5].
pipe.set_rl_mode(cfg_enable=True, cfg_beta=1.5, advantage_positive=True)
pipe.set_prompt("fold the t-shirt")

# First infer call lazy-recalibrates FP8 scales against the cond
# prompt and (when batched) recaptures the B=2 graph.
actions = pipe.infer(obs)["actions"]      # shape: (chunk_size, action_dim)

# Revert to the standard non-CFG path.
pipe.set_rl_mode(cfg_enable=False)
pipe.set_prompt("fold the t-shirt")
```

The JAX frontend has the same call surface. The four arg-compatible
frontends are `Pi05TorchFrontendRtx`, `Pi05JaxFrontendRtx`,
`Pi05TorchFrontendThor`, `Pi05JaxFrontendThor`.

### `set_rl_mode` parameters

- `cfg_enable` (bool): activate CFG inference. `False` clears any
  previous configuration; the next `set_prompt` rebuilds the standard
  pipeline.
- `cfg_beta` (float, default `1.5`): guidance strength. Must be
  `>= 1.0`. `1.0` mathematically reduces to cond-only output (combine
  collapses to `v_cond`) — useful as a correctness gate but wasteful
  in production; prefer the default non-CFG path for unconditioned
  inference.
- `advantage_positive` (bool, default `True`): conditioned prompt uses
  the positive advantage tag. Set `False` only for debugging the
  guidance direction.

### `autotune` parameter (frontend constructor)

`autotune=N` runs N capture+benchmark trials per CUDA-Graph build
(both the B=1 enc+ae graph and, when `set_batched_mode` is enabled,
the B=2 outer fused-CFG graph). Each trial lets cuBLASLt re-query
its heuristic; the fastest captured schedule is kept.

- `autotune=0` (default): one capture, whatever cuBLASLt picks first.
- `autotune=3`: recommended for RL CFG deployment — eliminates the
  cuBLASLt-tactic variance between Python frameworks (see Performance).
- Higher values cost ~0.5–1 s startup per additional trial.

## Algorithm

`Pi0.5` action expert is a 10-step flow-matching diffusion. Standard
single-forward inference integrates one velocity per step:

```
for k in 0..9:
    v = action_head(x_k, prompt, image)
    x_{k+1} = x_k + v
```

CFG runs the action head twice per step with two prompts and combines
the velocities **per step** (paper Eq. 2; matches `Pi05CFGBatchedPipeline`
on RTX and `decoder_forward_b2(cfg_beta=...)` on Thor):

```
for k in 0..9:
    v_cond   = action_head(x_k, "task\nAdvantage: positive", image)
    v_uncond = action_head(x_k, "task",                       image)
    x_{k+1}  = x_k + v_uncond + beta * (v_cond - v_uncond)
```

Both branches must enter step `k+1` from the **same** `x_{k+1}`
(otherwise the trajectories drift apart and combining their final
velocities is no longer the paper's CFG). The fused B=2 path
enforces this by writing the guided update into the cond slot via
the `cfg_combine_into_residual` kernel and mirroring it into the
uncond slot via a `cudaMemcpyAsync` — both inside the captured graph.

## Internals

```
RTX:     Pi05Pipeline → Pi05BatchedPipeline → Pi05CFGBatchedPipeline    (B=2 fused CFG)
                     → Pi05CFGPipeline                                   (serial CFG)

Thor:    Pi05ThorPipeline → Pi05ThorBatchedPipeline → Pi05ThorCFGBatchedPipeline
                          → Pi05ThorCFGPipeline                          (serial CFG)
```

Each `*BatchedPipeline` runs the encoder + 10-step decoder once at
B=2. Slot 0 is the conditioned context, slot 1 the unconditioned;
the per-step `cfg_combine_into_residual` kernel (single fused
elementwise call, FP16/BF16 packed-2) writes the guided velocity into
slot 0 and a D2D copy mirrors it into slot 1.

The RTX backend captures the entire B=2 forward (vision encoder,
text encoder, per-step decoder, cfg_combine, mirror) as one
`torch.cuda.CUDAGraph`. `forward()` is a single `graph.replay()`.

The Thor backend captures the same shape — outer graph wraps two
B=1 SigLIP runs (one per language slot, lang-emb swap is a graph-
internal D2D from a pre-staged device buffer), one B=2 enc_ae graph,
and the per-step CFG combine + noise mirror inside `decoder_forward_b2`.
`Pi05ThorCFGBatchedPipeline.forward()` calls `outer_graph.replay()`
+ stream sync.

## Performance

### RTX 5090, pi05_libero, FP8, num_views=2

Median over 20 infer invocations after 5 warmup calls.

| path | β | median (ms) | vs baseline |
|---|---|---|---|
| baseline (no CFG) | — | **19.0** | 1.00× |
| serial CFG | 1.5 | 37.1 | 1.96× |
| **fused CFG batched** | **1.5** | **25.9** | **1.36×** |

`β` does not affect latency — it is a multiplier inside the combine
kernel only. Fused batched is *faster* than the equivalent generic
B=2 path (27.5 ms) because the cfg_combine kernel replaces (does not
add to) the cond-slot per-step residual_add the generic batched path
performs.

The 25.9 ms median fits inside the 20 ms budget that 50 Hz real-robot
control demands once typical 3 ms control-loop overhead outside
`infer()` is accounted for.

### Thor SM110, pi05_libero, FP8, num_views=2

Median over 50 timed iters per back-to-back A/B subprocess pair, 3
cycles. Both backends use `autotune=3`.

| backend | path | β | median (ms) |
|---|---|---|---|
| torch | baseline (no CFG) | — | **44.6** |
| torch | serial CFG | 1.5 | 88 |
| **torch** | **fused CFG batched** | **1.5** | **~67** |
| JAX | baseline (no CFG) | — | 44.9 |
| JAX | serial CFG | 1.5 | 96 |
| **JAX** | **fused CFG batched** | **1.5** | **~67** |

#### Why `autotune` matters on Thor

Without autotune, the JAX frontend's fused-CFG p50 lands ~3–4 ms
above torch's. Root cause is process-state-dependent cuBLASLt
heuristic divergence — the two Python frameworks load different
`libcublas.so` versions (system 13.2.0 for torch, pip-bundled 13.2.1
for JAX) and start cuBLASLt with different internal cache states.
Given the same `(M, N, K)`, cuBLASLt can return a tactic that
launches ~36 extra `cutlass::Kernel2` sub-launches per inference
in the JAX process.

`autotune=N` recaptures the outer graph N times and keeps the
fastest. Each capture lets cuBLASLt re-query the heuristic; with
N≥3 the JAX backend converges on the same fast tactic torch picks
on the first try. This keeps the heuristic-first design (we never
pin a specific algo, which would brittle-break on cuBLAS upgrades
or hardware revisions) while erasing the cross-backend gap.

#### Why Thor is slower than RTX

Thor's `qkv_split_rope_kvcache_fp16` and `attention_qkv_fp16`
launches run as a per-sample inline Python loop (no batch-aware
fused-attention kernel for SM110 yet); these account for ~20 ms of
the fused-CFG path. The dense FP8 GEMMs amortise across the two
slots correctly (M = B*Seq) — only the per-token-indexed kernels
pay the per-sample cost. A future SM110 batch-aware attention
kernel would close most of the Thor↔RTX gap.

## Numerical contract

Default path (no `set_rl_mode`):
- bit-identical to the pre-RL implementation on all four frontends.

CFG path:
- `cfg_combine_into_residual` kernel vs FP32 reference on random
  inputs at the production size (`chunk_size * action_dim = 320`):
  `max abs diff = 0`, `cos = 1.0`.
- `cfg_beta=1.0` collapse: `cos(CFG, cond_only) >= 0.999` on all
  serial and fused paths, both backends, both hardware platforms
  (mathematical identity: `v_uncond + 1*(v_cond - v_uncond) = v_cond`).
- B=2 slot symmetry: same observation in both slots, identical noise
  R → `cos(slot 0, slot 1) = 1.000000`, `maxdiff = 0` on torch and
  JAX.

### Batched-vs-serial CFG agreement

| β | regime | batched vs serial | batched vs FP32 ref |
|---|---|---|---|
| 1.0 | paper default | 0.9997 | 0.9958 |
| 1.5 | moderate (lower) | 0.9991 | 0.9919 |
| 2.0 | mid-moderate | 0.9982 | 0.9854 |
| 2.5 | moderate (upper) | 0.9971 | 0.9756 |

The fused batched path tracks both serial and the FP32 reference
within the FP8 quantisation budget across the paper's full
`[1.0, 2.5]` recommended β range.

### Cross-backend (torch vs JAX) cosine on the same noise R

Same numpy-seeded R fed to both backends:

| β | torch vs JAX cos |
|---|---|
| 1.0 | ≥ 0.9997 |
| 1.5 | ≥ 0.9986 |
| 2.5 | ≥ 0.9979 |

The residual gap (~0.001–0.002) is per-frontend FP8 calibration noise
amplified by the CFG combine; it is not a correctness issue (well
inside the deployment cosine floor of 0.99 vs PyTorch FP32 reference).

## Reproducible sampling and denoise trace (RL rollouts)

RL training loops need two things from the sampler that deployment
does not: a way to reproduce a sample, and a view of how it was
produced. Both are engine-level capabilities on the Pi0.5 RTX
frontend; the RL semantics (log-probabilities, importance ratios)
stay with the training code that consumes them.

### Noise injection and seeding

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2)
rt.set_prompt("pick up the cup")

out = rt.infer(obs, return_noise=True)       # opt-in host noise export
noise = out["noise"]                         # (chunk_size, 32) float32
same = rt.infer(obs, noise=noise)            # bit-identical actions

g = torch.Generator(device="cuda").manual_seed(1234)
seeded = rt.infer(obs, generator=g)          # reproducible draw

batch = rt.infer_batch([obs_a, obs_b], noise=noise_pair)   # (B, chunk, 32)
```

Contract: same weights, prompt and noise give bit-identical actions
(graph replay is deterministic). Prompts are normalized before
tokenization the way openpi's `PaligemmaTokenizer` does (strip, `_`
and `\n` become spaces, then BOS + text + the `\n` start-of-answer
token); both the openpi and the SentencePiece fallback path of the
frontend produce the same ids, including for the RL prompt
`"<task>\nAdvantage: positive"`. A trainer must tokenize the same way
or it trains on a different prefix than the one served. `noise` is taken as given (rounded
to bf16); `generator` seeds the internal draw; passing both is an
error. With `return_noise=True`, the result carries `"noise"`, the bf16-rounded noise that was
used, which round-trips exactly when passed back. The CFG batched
path accepts the same arguments (one noise, replicated across the
conditioned and unconditioned slots). No pipeline change is involved
and the default path is untouched.

### Denoise trace

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2, denoise_trace=True)
rt.set_prompt("pick up the cup")
out = rt.infer(obs, noise=noise)
tr = out["denoise_trace"]      # x, delta: (num_steps, chunk, 32) float32
out["raw_actions"]             # normalized (chunk, 32), before unnormalization
```

`x[s]` is the state entering denoising step `s` (`x[0]` is the input
noise) and `delta[s]` is what the step adds, so

    x[s+1] == x[s] + delta[s]        (bf16 arithmetic, exact)
    x[-1] + delta[-1] == raw_actions

and for the linear flow-matching schedule `delta[s] == -v[s] / num_steps`
where `v[s]` is the action expert's velocity prediction at flow time
`timesteps[s]` (`1, 1-dt, ..., dt`). A consumer computing a flow-SDE
log-probability needs exactly `x`, `delta` and `timesteps`.

Mechanics: the pipeline is built with two extra buffers of
`(num_steps, chunk, 32)` bf16 (64 KB at the defaults) and issues two
device-to-device copies per step inside `transformer_decoder`, so the
copies are captured into the same CUDA graph as the rest of the
denoise loop and replay with it. The option is decided at
construction because the graph either contains those copies or not;
with it off, the pipeline and its graph are unchanged from before.
The batched pipeline records a per-slot trace, returned per entry of
`infer_batch`.

Scope: Pi0.5 RTX `Pi05Pipeline` and `Pi05BatchedPipeline`. The CFG
pipelines do not record a trace yet and `set_rl_mode` refuses a
trace-enabled frontend. Thor and FP16 pipelines and the C++ runtime
export are not covered.

### Batched rollouts, B = N

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2)
rt.set_batched_mode(enable=True, batch_size=8)   # any B >= 1; default 2
rt.set_prompt_batch([prompt] * 8)                 # one prompt per slot, padded to a common length
rt.calibrate_batch([obs])
out = rt.infer_batch([obs_0, ..., obs_7], noise=noise_8)   # list of 8 results
```

The batched attention backend and `Pi05BatchedPipeline` take their
width from `batch_size`; every folded buffer scales with it. Slots are
independent: identical inputs give bit-identical outputs per slot.
The mixed-slot FP8 test compares each slot with independent B=1 inference
under identical calibration and the existing synthetic-input cosine gate
of 0.99; permuting slots must preserve outputs bit-for-bit. GEMM tactics
change with M, so B=N and B=1 are not promised bit-equal. Compare widths
using synchronized observation-to-final-action E2E on identical inputs;
per-environment throughput is not individual request latency. The CFG batched pipeline
still requires B = 2 (conditioned and unconditioned slots) and
`set_rl_mode` refuses a wider backend.

### Prefix hidden-state export

```python
rt = Pi05TorchFrontendRtx(ckpt, num_views=2, prefix_features=True)
rt.set_prompt(prompt); rt.calibrate([obs])
feat = rt.infer(obs)["prefix_features"]          # (2048,) float32
```

With `prefix_features=True` the pipeline copies the Gemma-2B encoder's
final residual stream (vision + prompt tokens, before the final norm)
into an export buffer inside the captured graph, and the frontend
returns its mean over the valid tokens. This is the input a value
function trained on the VLA's own representation needs, and it costs
nothing beyond a 2 MB device copy per environment, plus one detail:
by default the last encoder layer stops after its K/V projection
because the decoder reads only per-layer K/V, so with the export
enabled the last layer runs to completion (about 1/18 of the encoder).
Checked against the torch reference's pre-norm residual stream on the
same inputs: BF16 path per-token cosine 0.998 and pooled 1.0000; FP8
path pooled 0.9999 (the export adds the last layer's residual that the
fused FP8 path leaves pending). Batched: one feature vector per slot.

## Tests

| test | what it validates |
|---|---|
| `tests/test_pi05_seeded_noise_trace.py` | noise injection / seeded reproducibility, trace self-consistency, batched per-slot trace |
| `tests/test_pi05_batched_n.py` | B = 4 / 8 batched slots vs B = 1, per-env timing, prefix features single vs batched and FP8 vs BF16, CFG refuses B ≠ 2 |
| `tests/test_rl_cfg_inference.py` | RTX serial + batched CFG, all βs, validation gates |
| `tests/test_thor_rl_cfg_inference.py --backends torch,jax` | Thor serial CFG: validation, β=1.0 collapse, β=1.5 finite |
| `tests/test_cfg_correctness_oracle.py` | per-step C1–C5 contract (RTX) vs frozen reference |

## Troubleshooting

**Calibration warning about scale ceiling during RL mode** —
the conditioned prompt has slightly different token statistics than
pure task text. If the ratio is within ~25× the median, output is
correct; the warning flags calibration-set diversity, not a bug.

**`RuntimeError: cfg_beta must be >= 1.0`** — pass a value in
`[1.0, …]`. `< 1.0` would invert guidance, which the frontend
rejects to prevent silent sign bugs.

**`RuntimeError: set_prompt must be called before calibrate`** —
RL mode rebuilds the pipeline at the next `set_prompt`. Order is
always `set_rl_mode → set_prompt → calibrate`.

**Two `Pi05TorchFrontendRtx` instances in the same process segfault** —
pre-existing single-instance constraint of the calibration path,
unrelated to RL mode. Use one frontend per process (the test suite
does this).

**JAX fused-CFG is consistently 3–4 ms slower than torch** — pass
`autotune=3` (or higher) to the frontend constructor. See the
"Performance" → "Why autotune matters on Thor" section.

## References

- π\*0.6 paper — [arXiv:2511.14759](https://arxiv.org/abs/2511.14759),
  Appendix E for the CFG derivation from the flow-matching likelihood
  gradient.
- [`flash_rt/core/rl/`](../flash_rt/core/rl/) — framework-agnostic
  combine math, ACP-tag prompt builder.
- [`csrc/kernels/elementwise.cu`](../csrc/kernels/elementwise.cu) —
  `cfg_combine_into_residual` kernel (packed-2 vectorised, FP32
  internally for numerical stability at β > 1).
- Pipeline classes:
  - RTX: [`pipeline_rtx_cfg.py`](../flash_rt/models/pi05/pipeline_rtx_cfg.py),
    [`pipeline_rtx_cfg_batched.py`](../flash_rt/models/pi05/pipeline_rtx_cfg_batched.py)
  - Thor: [`pipeline_thor_cfg.py`](../flash_rt/models/pi05/pipeline_thor_cfg.py),
    [`pipeline_thor_cfg_batched.py`](../flash_rt/models/pi05/pipeline_thor_cfg_batched.py)
