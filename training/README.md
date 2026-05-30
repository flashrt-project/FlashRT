# FlashRT Training (BETA)

Status: **v0.1.0 BETA — API not stable**. Maintained alongside the
inference framework `flash_rt/`; will be released as a separate package
once the API stabilises.

This subtree is the **FP8 + LoRA + RECAP training infrastructure**
for VLA models. It ships paper-aligned (π\*0.6,
[arXiv:2511.14759](https://arxiv.org/abs/2511.14759)) on both
PyTorch and JAX, dataset-agnostic at the driver layer, with LIBERO
shipped as the first concrete dataset adapter.

## What this stack supports

| Capability | Status | Where |
|---|---|---|
| **PyTorch FP8 + LoRA** training (frozen FP8 base, BF16 LoRA backward) | ✅ shipping | this README, see [Train on your own dataset](#train-on-your-own-dataset) below |
| **JAX FP8 + LoRA** training (XLA FFI patch over openpi's JAX trainer) | ✅ shipping | **[`training/jax/README.md`](jax/README.md)** — quick start, status, 30k bench |
| **RECAP / ACP**-conditioned policy training (advantage-conditioned, paper §V-B + Appendix E–F) | ✅ shipping | [What is the RECAP / ACP pipeline?](#what-is-the-recap--acp-pipeline) |
| **Plain LoRA fine-tune** (same driver with `use_acp=False`) | ✅ shipping | [Plain LoRA fine-tune](#plain-lora-fine-tune-no-rl--no-acp) |
| **Custom datasets** (any [`RecapPolicyDataset`](rl/dataset_protocol.py) Protocol implementer) | ✅ supported | [Train on your own dataset](#train-on-your-own-dataset) |
| **LIBERO** dataset adapter (first concrete reference) | ✅ shipping | [LIBERO concrete example](#libero-concrete-example) |
| **Adapting to a new VLA** (pi0.5 today; new-model checklist) | ✅ ~2 files of model-specific code, rest reused | [Adapting to a new VLA](#adapting-to-a-new-vla) |
| **Train → serve (PyTorch)** — merge LoRA → safetensors → inference engine | ✅ shipping | [Train → serve](#train--serve) |
| **Train → serve (JAX)** — merge Orbax → Orbax → JAX inference frontend | ✅ shipping | [`training/jax/README.md` § Train → serve](jax/README.md) |
| **RL inference (CFG-guided)** — both PyTorch + JAX frontends serve the merged LoRA via classifier-free guidance | ✅ shipping | [`docs/rl_inference.md`](../docs/rl_inference.md) |
| **FP8 backward** (E5M2) | 🚫 intentionally not implemented | [Why FP8 backward is not implemented](#why-fp8-backward-is-not-implemented) |

The PyTorch and JAX lines are **independent end-to-end paths** —
JAX trains/serves JAX, PyTorch trains/serves PyTorch; they share
algorithm primitives (ACP tag strings, advantage thresholds,
reward-target math) under
[`flash_rt/core/rl/`](../flash_rt/core/rl/) and the same FP8
cuBLASLt kernel under [`csrc/gemm/`](../csrc/gemm/), but the
checkpoints don't cross over and you don't need to think about
the other framework while you use one.

## Why this exists

FlashRT's training stack is one of several PyTorch options the
community has for fine-tuning pi0.5; it sits alongside the upstream
[openpi](https://github.com/Physical-Intelligence/openpi) JAX path
(the canonical reference) and the [lerobot](https://github.com/huggingface/lerobot)
PyTorch port (which provides a PEFT-based BF16 LoRA training loop
out of the box). What this stack adds is a **FP8-forward training
path on Ada/Blackwell consumer GPUs**: the same kernels FlashRT's
inference path uses for FP8 GEMM, fused RMSNorm, and SwiGLU are
reused for the training forward pass, while backward stays in BF16
through small LoRA adapters. The aim is to let users who already
have those kernels available pick a different point on the
memory / step-time curve than the BF16 path:

- Single-GPU 24 GB fine-tuning of pi0.5 with two recommended configs:
  speed-priority (peak 14.65 GB, 16.14 samples/s at B=4) or
  memory-priority (peak 10.05 GB, 15.80 samples/s at B=4) — pick the
  one that fits your card and iteration budget.
- Trained LoRA checkpoints load directly into the FlashRT inference
  engine via `merge_lora_into_base`.

Public reference numbers from the upstream projects, for context:

| Source | Mode | Memory | Notes |
|---|---|---|---|
| [openpi README](https://github.com/Physical-Intelligence/openpi#gpu-requirements) | LoRA fine-tune | > 22.5 GB | JAX path on RTX 4090 |
| [openpi README](https://github.com/Physical-Intelligence/openpi#gpu-requirements) | Full fine-tune | > 70 GB | JAX path on A100 (80 GB) / H100 |
| lerobot (this README's bench, openpi-aligned LoRA target set) | LoRA fine-tune | 9.08 GB | PyTorch BF16, B=4, RTX 5090 |

## Layout

```
training/
├── _vendor/openpi_pi0_pytorch/  # JAX-free snapshot of openpi PyTorch model
├── trainers/                    # FP8-fwd model wrappers
│   └── pi05_torch_trainer.py
├── lora/                        # FP8 + LoRA adapter primitives
│   ├── fp8_autograd.py          #   torch.autograd.Function over our kernel
│   ├── fp8_linear.py            #   nn.Linear replacement (frozen FP8 + BF16 LoRA)
│   └── inject.py                #   target-module walker + calibration
├── rl/                          # RECAP-style RL pipeline
│   ├── acp_hook.py              #   prompt injection (30 % dropout)
│   ├── advantage.py / reward.py / value_function.py — see flash_rt/core/rl/
│   ├── checkpoint.py            #   pi0.5 base loader + LoRA save/load
│   ├── dataset_stats.py         #   ACP indicator audit (vendored from Evo-RL)
│   ├── jax_baseline_compare.py  #   head-to-head with openpi JAX script
│   ├── lerobot_libero.py        #   LeRobot v3 loader (action chunking, ACP read)
│   ├── dataset_protocol.py      #   RecapPolicyDataset / RecapMetadataDataset Protocols
│   ├── merge_lora.py            #   merge LoRA → standalone pi0.5 safetensors
│   ├── observation.py           #   frame → pi0.5 Observation (LIBERO_CAMERA_MAP default)
│   ├── pi05_vf.py               #   ValueFunctionHead over pi0.5 prefix embedding
│   ├── recap_iter.py            #   one full RECAP iter (dataset-agnostic)
│   ├── tokenizer.py             #   local PaliGemma SentencePiece wrapper
│   ├── train_recap.py           #   generic RECAP / ACP policy driver (dataset-agnostic)
│   ├── train_libero_recap.py    #   thin LIBERO entry point — wraps train_recap
│   ├── train_policy.py          #   single ACP-aware step
│   ├── train_value.py           #   distributional VF training
│   └── value_infer.py           #   VF → acp_indicator annotation
└── tests/                       # 22 cases, all green
```

Algorithm primitives shared with the inference path live under
[`flash_rt/core/rl/`](../flash_rt/core/rl/) (`acp_tags.py`,
`cfg_sampler.py`, `reward.py`, `advantage.py`, `value_function.py`).

## Quick start

The dataset, base ckpt, tokenizer, and JAX baseline all need to live
on the local filesystem — no downloads. Configure paths with env
vars (preferred) or pass them explicitly to each constructor:

| Resource | Env var | Required by |
|---|---|---|
| pi0.5 PyTorch base ckpt | `FLASHVLA_PI05_CKPT_PYTORCH` | `load_pi05_pretrained`, `run_pytorch_pipeline` |
| pi0.5 JAX (Orbax) base ckpt | `FLASHVLA_PI05_CKPT_JAX` | `run_jax_baseline` |
| LIBERO RECAP-annotated dataset | `FLASHVLA_RECAP_DATASET` | `LeRobotLiberoDataset`, `run_pytorch_pipeline`, `run_jax_baseline` |
| LIBERO unannotated dataset | `FLASHVLA_LIBERO_ROOT` | `LeRobotLiberoDataset` (default fallback) |
| PaliGemma SentencePiece tokenizer | `FLASHVLA_TOKENIZER_PATH` | `PaligemmaTokenizer` |
| openpi JAX baseline script | `FLASHVLA_JAX_BASELINE_SCRIPT` | `run_jax_baseline` |

A LeRobot v3 dataset is expected to follow this layout::

    <root>/
        meta/info.json
        meta/tasks.parquet
        meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet
        data/chunk-{c:03d}/file-{f:03d}.parquet

The PaliGemma tokenizer directory must contain ``tokenizer.model``,
``tokenizer_config.json``, and ``special_tokens_map.json`` — i.e.,
the standard HuggingFace local layout.

### Single-step (install smoke)

Confirms the FP8 + LoRA injection compiles end to end on your
hardware before you wire in any dataset:

```python
from training.lora.inject import InjectionConfig
from training.rl.checkpoint import load_pi05_pretrained
from training.trainers.pi05_torch_trainer import Pi05Trainer

model = load_pi05_pretrained(
    "<your-pi05-pytorch-ckpt-dir>", action_horizon=10,
)
trainer = Pi05Trainer(model, device="cuda")
trainer.compile(InjectionConfig(encoder_rank=16, decoder_rank=16))
print(f"trainable params: {trainer.num_trainable_parameters():,}")
```

### Train on your own dataset

The training driver
[`train_recap_policy`](rl/train_recap.py) is **dataset-agnostic**.
To plug in your own data, implement the
[`RecapPolicyDataset`](rl/dataset_protocol.py) Protocol — five
methods, no inheritance — and provide an observation builder that
turns one mini-batch's decoded images + padded states into a pi0.5
`Observation`:

```python
import numpy as np

from training.lora.inject import InjectionConfig
from training.rl.checkpoint import load_pi05_pretrained
from training.rl.observation import decoded_to_observation
from training.rl.tokenizer import PaligemmaTokenizer
from training.rl.train_recap import RecapTrainConfig, train_recap_policy
from training.trainers.pi05_torch_trainer import Pi05Trainer


# 1. Implement the dataset Protocol — no base class to inherit;
# duck-typing against ``training.rl.dataset_protocol.RecapPolicyDataset``.
class MyDataset:
    def build_chunk_starts(self, action_horizon: int) -> np.ndarray:
        """Valid frame indices ``s`` where [s, s+action_horizon) is in one episode."""
        ...

    def has_acp_column(self) -> bool:
        """True iff per-frame ``acp_indicator`` is pre-annotated."""
        ...

    def ensure_acp_indicators(self) -> np.ndarray:
        """``(num_frames,) int64`` indicator array."""
        ...

    def get_frame(self, idx: int):
        """Returns an object with ``.state``, ``.action``, ``.image_bytes``,
        ``.task_name``. Called from DataLoader workers — must be picklable."""
        ...

    def get_action_chunk(self, idx: int, action_horizon: int) -> np.ndarray:
        """``(action_horizon, action_dim) float32`` ground-truth chunk."""
        ...


# 2. Pick a camera mapping. pi0.5 has three image slots
# (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb). Map each one
# to the corresponding key in your ``get_frame().image_bytes``,
# or pass ``None`` to leave a slot empty (filled with zeros +
# ``image_masks[<slot>] = False`` so the encoder treats it as
# padded). Both 2-cam and 3-cam datasets are supported — fill in
# whichever slots you have.
MY_CAMERA_MAP = {
    "base_0_rgb":         "front_cam",        # base view (typically required)
    "left_wrist_0_rgb":   "left_wrist_cam",   # left wrist if present, else None
    "right_wrist_0_rgb":  "right_wrist_cam",  # right wrist if present, else None
}


# 3. Wire your camera map into a tiny observation_builder. The
# decoded-images dict you receive is already keyed by the pi0.5
# slot names because the async loader applied your map for you —
# the builder just stacks into the Observation dataclass.
def my_obs_builder(decoded_images, states_padded, *,
                   tokenized_prompt, tokenized_prompt_mask, device):
    return decoded_to_observation(
        decoded_images, states_padded,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        device=device,
    )


# 4. Build the trainer (same recipe regardless of dataset).
model = load_pi05_pretrained("<your-pi05-pytorch-ckpt-dir>", action_horizon=10)
trainer = Pi05Trainer(model, device="cuda")
trainer.compile(InjectionConfig(encoder_rank=16, decoder_rank=16))
tokenizer = PaligemmaTokenizer(
    max_token_len=trainer.model.config.max_token_len, device=trainer.device,
)

# 5. Run the generic driver. Identical to the LIBERO call below;
# the difference is just *what dataset class* you hand in.
cfg = RecapTrainConfig(
    num_steps=1000,
    batch_size=4,
    lr=2.5e-5,
    acp_dropout=0.30,
    camera_map=MY_CAMERA_MAP,
)
result = train_recap_policy(
    trainer, MyDataset(), tokenizer, my_obs_builder,
    config=cfg, output_dir="<your-run-output-dir>",
)
```

Put your `MY_CAMERA_MAP` on `RecapTrainConfig.camera_map`; the
generic driver forwards it to the async step loader before images
reach your `observation_builder`. If you bypass the driver, pass the
same map to `make_step_dataloader(..., camera_map=MY_CAMERA_MAP)` or
`frames_to_observation(..., camera_map=...)` directly. The default is
`LIBERO_CAMERA_MAP`, which is what the LIBERO example below uses
implicitly.

The driver matches the openpi JAX baseline's optimiser + LR
schedule 1-for-1: AdamW (`weight_decay=1e-10`) + warmup-cosine
decay (`init=0, peak=lr, warmup=min(100, steps//30), end=peak*0.1`),
global-norm grad clip 1.0. None of these are dataset-specific.

### LIBERO concrete example

LIBERO ships as the first reference adapter
([`LeRobotLiberoDataset`](rl/lerobot_libero.py)). The
[`train_libero_recap`](rl/train_libero_recap.py) entry point is a
30-line shim that wires `LeRobotLiberoDataset` +
`decoded_to_observation` (using the default `LIBERO_CAMERA_MAP`)
into the same `train_recap_policy` driver shown above:

```python
import os

# Resolve everything from env vars in one place:
os.environ.setdefault("FLASHVLA_PI05_CKPT_PYTORCH", "<your-pi05-pytorch-ckpt-dir>")
os.environ.setdefault("FLASHVLA_RECAP_DATASET",     "<your-libero-recap-dataset>")
os.environ.setdefault("FLASHVLA_TOKENIZER_PATH",    "<your-paligemma-tokenizer-dir>")

from training.lora.inject import InjectionConfig
from training.rl.checkpoint import load_pi05_pretrained
from training.rl.lerobot_libero import LeRobotLiberoDataset
from training.rl.tokenizer import PaligemmaTokenizer
from training.rl.train_recap import RecapTrainConfig
from training.rl.train_libero_recap import train_libero_recap
from training.trainers.pi05_torch_trainer import Pi05Trainer

dataset = LeRobotLiberoDataset(os.environ["FLASHVLA_RECAP_DATASET"])
model = load_pi05_pretrained(
    os.environ["FLASHVLA_PI05_CKPT_PYTORCH"], action_horizon=10,
)
trainer = Pi05Trainer(model, device="cuda")
trainer.compile(InjectionConfig(encoder_rank=16, decoder_rank=16))
tokenizer = PaligemmaTokenizer(
    max_token_len=trainer.model.config.max_token_len, device=trainer.device,
)

cfg = RecapTrainConfig(num_steps=1000, batch_size=4, lr=2.5e-5, acp_dropout=0.30)
result = train_libero_recap(
    trainer, dataset, tokenizer, config=cfg, output_dir="<your-run-output-dir>",
)
print(
    f"final loss: {result.loss_history[-1]:.4f}, "
    f"peak GPU memory: {result.peak_memory_bytes / 1e9:.2f} GB, "
    f"wall: {result.seconds_total:.1f} s, "
    f"LoRA saved to {result.final_lora_dir}",
)
```

Internally that's just::

    train_libero_recap(...) ≡ train_recap_policy(
        trainer, dataset, tokenizer,
        observation_builder=decoded_to_observation,  # uses LIBERO_CAMERA_MAP
        ...,
    )

so any improvement to the generic driver applies to LIBERO and to
your own dataset identically.

### Plain LoRA fine-tune (no RL / no ACP)

The same driver runs as a vanilla supervised LoRA fine-tune when
``use_acp=False``: no ``acp_indicator`` lookup, no
``ACPPromptHook``, no per-step prompt re-tokenisation. The flow-
matching loss + AdamW + warmup-cosine LR schedule + LoRA adapters
(through the FP8 base GEMM kernel) are exactly the same — only the
prompt-conditioning side is bypassed. Useful when you just want to
domain-adapt pi0.5 to your own dataset.

```python
cfg = RecapTrainConfig(
    num_steps=1000,
    batch_size=4,
    lr=2.5e-5,
    use_acp=False,         # <-- plain LoRA fine-tune
)
result = train_libero_recap(trainer, dataset, tokenizer, config=cfg)
```

Any dataset satisfying
[`RecapPolicyDataset`](rl/dataset_protocol.py) plugs in via the
generic [`train_recap_policy`](rl/train_recap.py) entry point;
LIBERO is the first concrete implementation. For a dataset with
the same parquet layout but a different camera / state schema,
pass a custom ``camera_map`` through ``RecapTrainConfig``. The async
loader applies the map before the decoded pi0.5 camera slots reach
the observation builder. The default is ``LIBERO_CAMERA_MAP``.

### Async data prep + ``torch.compile`` (production fast path)

The driver exposes two opt-in performance knobs on
``RecapTrainConfig``. Both default off so the baseline path is the
one the numerics-correctness gates were measured on; flipping either
is purely a throughput choice.

```python
cfg = RecapTrainConfig(
    num_steps=30_000,
    batch_size=4,
    lr=2.5e-5,
    # Async data pipeline: parquet read + JPEG decode + np.stack run
    # in 2 DataLoader worker processes, overlapped with GPU compute.
    # Per-step chunk-start indices are precomputed from the seeded
    # rng so loss curves stay byte-identical for any worker count at
    # the same seed. ``0`` keeps everything on the main thread.
    dataloader_workers=2,
    # ``torch.compile`` of trainer.model.forward. ``"default"`` is
    # Inductor fusion only; ``"reduce-overhead"`` adds PyTorch's
    # CUDA-Graph cache. Pays a one-time ~25 s compile on the first
    # step, then holds the steady speedup for the rest of the run.
    compile_mode="reduce-overhead",
)
result = train_libero_recap(trainer, dataset, tokenizer, config=cfg)
```

Per-step loss values are *not* byte-equal to eager once
``compile_mode`` is set — the model samples ``noise`` and ``time``
inside ``forward()`` via the global torch RNG and Inductor's op-fusion
shifts the consumption order, so each step sees different noise.
Trajectory shape (mean over a window) stays in the same band.

The driver restores ``trainer.model`` to the eager ``nn.Module`` on
exit so downstream callers (``save_lora_state``, ``merge_lora_into_base``,
``sample_actions``) still see the original handle.

### What is the RECAP / ACP pipeline?

The driver supports two training modes that share the same FP8+LoRA
plumbing but differ in how the prompt is conditioned each step:

* `use_acp=False` (plain LoRA): standard supervised fine-tune on
  `(observation, action_chunk)` pairs from the dataset. The flow-
  matching loss + AdamW + warmup-cosine schedule are exactly what
  most users expect. No advantage information is consumed.
* `use_acp=True` (RECAP): the driver appends an `Advantage: positive`
  / `Advantage: negative` tag to each sample's task prompt, derived
  from a per-frame `acp_indicator` carried by the dataset. The
  policy learns to condition on the tag, and at inference time
  classifier-free guidance combines the conditional + unconditional
  predictions to push generated actions toward the high-advantage
  side of the offline distribution. This is the recipe from
  Physical Intelligence's π\*0.6 paper
  ([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)), distilled
  to a single per-step prompt-injection hook.

The RECAP loop in this stack has three components, each in its own
file under [`training/rl/`](rl/):

* **Value function** ([`pi05_vf.py`](rl/pi05_vf.py),
  [`train_value.py`](rl/train_value.py)) — a small head trained on
  top of a frozen pi0.5 prefix embedding to predict the long-horizon
  return of an `(obs, language)` pair. Provides the "advantage"
  signal used to label data.
* **Value inference + ACP indicator annotation**
  ([`value_infer.py`](rl/value_infer.py),
  [`recap_iter.py`](rl/recap_iter.py)) — runs the trained value
  function over the dataset, computes per-frame advantage, and
  writes back a 0/1 `acp_indicator` (positive vs negative) into a
  parquet column the policy loop later reads. The
  `libero10_recap_lerobot` dataset that ships with this stack is
  pre-annotated; brand-new datasets need this pass once before
  RECAP training begins.
* **ACP-aware policy training** ([`acp_hook.py`](rl/acp_hook.py),
  [`train_policy.py`](rl/train_policy.py),
  [`train_libero_recap.py`](rl/train_libero_recap.py)) — the full
  driver. Each step it samples a mini-batch, looks up the indicator
  per sample, randomly drops the tag with probability
  `acp_dropout=0.30` (Appendix E of the paper — keeps an
  unconditional distribution alive for CFG at inference), and
  feeds the modified prompt through the FP8+LoRA forward.

The shared inference-side primitives (advantage tag string format,
CFG sampler, value-function reward shaping) live under
[`flash_rt/core/rl/`](../flash_rt/core/rl/) so train and serve
agree byte-for-byte on what `Advantage: positive/negative` means.

If you only want a vanilla supervised LoRA fine-tune, set
`use_acp=False` and ignore the rest of this section. The value
function + indicator annotation only matter if you want the
classifier-free-guidance reward shaping at inference.

### Train → serve

The ACP injection re-tokenises prompts each step; the LoRA delta
must be merged back into a standalone safetensors before the
inference engine can consume it.

```python
from training.rl.merge_lora import merge_lora_into_base

merge_lora_into_base(
    base_dir="<your-pi05-pytorch-ckpt-dir>",
    lora_dir="<your-run-output-dir>/final",
    output_dir="<your-merged-ckpt-output-dir>",
)
# Result is a drop-in replacement directory.
```

The merged directory mirrors the base layout (`config.json`,
`policy_postprocessor.json`, `policy_preprocessor.json`, `assets/`,
`model.safetensors`) so any pi0.5 PyTorch consumer — including
FlashRT's `Pi05TorchFrontendRtx` — can load it directly.

For **classifier-free-guidance inference** on the merged LoRA
checkpoint (the RECAP / ACP test-time recipe — runs the model
twice per denoising step and combines `v_uncond + β·(v_cond −
v_uncond)`), see [`docs/rl_inference.md`](../docs/rl_inference.md).
Both `Pi05TorchFrontendRtx` (safetensors) and `Pi05JaxFrontendRtx`
(Orbax) serve the same CFG pipeline through inheritance — the
inference engine doesn't care which training stack produced the
LoRA, only what format the checkpoint is in.

## Adapting to a new VLA

This stack is the **pi0.5 instantiation** of a model-agnostic
training infrastructure. Most of the machinery is reusable; the
parts that touch the specific model architecture are isolated to
two files. Concretely:

| Component | pi0.5-specific? | Reusable across VLAs |
|---|---|---|
| FP8 cuBLASLt GEMM kernel ([`csrc/gemm/`](../csrc/gemm/)) | no | ✅ any model with M ≥ 64 LoRA-bearing GEMMs benefits |
| RECAP / ACP algorithm primitives ([`flash_rt/core/rl/`](../flash_rt/core/rl/)) | no | ✅ pure paper-aligned algorithm — tag strings, advantage, soft-bin loss, CFG combine |
| Dataset Protocol ([`rl/dataset_protocol.py`](rl/dataset_protocol.py)) | no | ✅ 5-method contract; no model assumptions |
| Optimizer + LR + grad-clip + LoRA injection ([`train_recap.py`](rl/train_recap.py), [`lora/inject.py`](lora/inject.py)) | no | ✅ `InjectionConfig.target_modules` defaults to HF naming (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`); pass your own list if your model uses different names |
| RECAP iter / VF training ([`recap_iter.py`](rl/recap_iter.py), [`train_value.py`](rl/train_value.py), [`pi05_vf.py`](rl/pi05_vf.py)) | mostly no | ✅ `StandaloneValueFunction` is state-only; `Pi05ValueFunction` swaps the backbone for any VLA exposing pooled prefix embeddings |
| Train → serve LoRA merge math ([`merge_lora.py`](rl/merge_lora.py)) | no | ✅ generic LoRA fold (the JAX-side merge is similarly model-agnostic) |
| Trainer wrapper ([`trainers/pi05_torch_trainer.py`](trainers/pi05_torch_trainer.py)) | **yes** | ❌ write a parallel `<YourModel>Trainer` exposing `compile()` + `model.compute_loss()` |
| Observation adapter ([`rl/observation.py`](rl/observation.py) — pi0.5's 3-cam + 32-dim state) | **yes** | ❌ write a parallel observation builder for your camera + state schema (or pass a different `camera_map` if your model also uses pi0.5's 3-cam layout) |
| Vendored model code ([`_vendor/openpi_pi0_pytorch/`](_vendor/openpi_pi0_pytorch/)) | **yes** | ❌ vendor your model's PyTorch implementation |

**The minimum work to adapt is ~two files**: a `<YourModel>Trainer`
wrapper and an observation builder for your camera/state schema.
Everything else — FP8 kernel + LoRA injection + RECAP/ACP +
optimizer recipe + dataset Protocol + train→serve merge — keeps
working unchanged.

The JAX path's adaptation cost is the same shape: a JAX-side
trainer entry point + a JAX observation adapter; the FP8 patch
(`training.jax.fp8.lora_patch`), the cross-language algorithm
primitives, and `training.jax.merge_lora` apply to any VLA whose
LoRA-bearing layers go through openpi-style
`lora.Einsum` / `lora.FeedForward`.

What this stack does **not** abstract over:

* **Model architecture novelty** — if your model has fundamentally
  new layer types (state-space blocks, mixture-of-experts, etc.)
  that the FP8 + LoRA path doesn't recognise, you'll need to
  extend `InjectionConfig` accordingly.
* **Action representation** — pi0.5 uses flow-matching on a
  fixed-horizon action chunk. Discrete-action / autoregressive
  models would need a different trainer body inside
  `<YourModel>Trainer.compute_loss()`.

## Compare against the openpi JAX baseline

`training/rl/jax_baseline_compare.py` shells out to the openpi
JAX script, parses the `Step N/M | loss=L` log lines into a CSV,
and produces a side-by-side trajectory comparison. The intent is
not to match absolute final loss; it is to detect a structurally
broken pipeline (e.g., loss diverging by orders of magnitude).

```python
from training.rl.jax_baseline_compare import (
    run_pytorch_pipeline, run_jax_baseline, compare_curves,
)

run_pytorch_pipeline(output_csv="run/pytorch.csv", num_steps=200)
run_jax_baseline(
    output_csv="run/jax.csv", output_dir="run/jax_outputs", num_steps=200,
)
cmp = compare_curves("run/pytorch.csv", "run/jax.csv")
print(f"pytorch final {cmp.pytorch_final:.4f}, jax final {cmp.jax_final:.4f}, "
      f"order-of-magnitude ratio {cmp.final_ratio:.2f}×")
```

The `final_ratio_max=5.0` default (configurable up to ~10× for short
runs at the converged-ckpt MSE noise floor) is the "no egregious
bug" gate.

## Status

| Component | State | Notes |
| --- | --- | --- |
| Vendored pi0.5 PyTorch model | ✅ | `_vendor/openpi_pi0_pytorch` |
| FP8 + LoRA trainer | ✅ | `Pi05Trainer.compile()` + calibration |
| Forward FP8 kernel correctness | ✅ | cos ≥ 0.999 on all six pi0.5 GEMM shapes |
| LoRA grad correctness | ✅ | A/B grad cos = 1.000 vs all-BF16 reference |
| RECAP value function training | ✅ | StandaloneVF + Pi05ValueFunction (frozen-backbone head) |
| RECAP policy training | ✅ | ACP hook + flow-matching loss + LoRA |
| End-to-end RECAP iter on real LIBERO | ✅ | LeRobot v3 loader + PaliGemma tokenizer + Observation adapter |
| Long-form policy training driver | ✅ | 30k step on real ckpt, AdamW + warmup-cosine |
| LoRA save / load + merge → base safetensors | ✅ | byte-equal round-trip; merged ckpt drives sample_actions |
| JAX baseline trajectory comparison | ✅ | order-of-magnitude shape match on real ckpt + dataset |
| Async DataLoader for LIBERO step prep | ✅ | `RecapTrainConfig.dataloader_workers`, byte-equal at fixed seed |
| `torch.compile` of model.forward + CUDA-Graph capture | ✅ | `RecapTrainConfig.compile_mode`; FP8 GEMM as registered `custom_op` |
| FP8 backward (E5M2) | 🚫 intentionally not implemented | see "Why FP8 backward is not implemented" below |

## Per-shape FP8 forward throughput (RTX 5090 SM120, BF16-backward path)

Per-shape forward + fwd+bwd speedup vs an all-BF16 LoRA reference,
M = 512, LoRA rank = 16:

| Shape | dims | fwd-only | fwd+bwd |
|---|---|---|---|
| enc.attn.qkv | 2048 → 2048 | 0.96× | 0.68× |
| **enc.ffn.gate_up** | 2048 → 16384 | **2.00×** | **1.67×** |
| **enc.ffn.down** | 16384 → 2048 | **1.65×** | **1.53×** |
| dec.attn.qkv | 1024 → 1024 | 0.61× | 0.69× |
| dec.ffn.gate_up | 1024 → 4096 | 1.08× | 0.69× |
| dec.ffn.down | 4096 → 1024 | 1.07× | 0.69× |

## End-to-end training-step throughput (RTX 5090 SM120, B=4, lora_rank=16)

To make the numbers below easy to read against the upstream
PyTorch reference, every row uses the same pi0.5 architecture, the
same LoRA target set (q/k/v/o + gate/up/down on both encoder and
action expert, matching openpi `gemma_2b_lora` / `gemma_300m_lora`),
the same trainable parameter count (~26.5M ≈ 0.7% of the 3.6B base),
and the same `gradient_checkpointing=True`. The lerobot row uses
`peft.LoraConfig` on `lerobot/pi05` with that target regex at
rank 16; the FlashRT rows use `Pi05Trainer.compile(InjectionConfig(
encoder_rank=16, decoder_rank=16))`. Everything ran on the same
RTX 5090 SM120 / 32 GB card.

| Stack | `cache_bf16_weight` | `compile_mode` | step/s | samples/s | peak GPU |
|---|---|---|---:|---:|---:|
| lerobot pi05 BF16+LoRA (PyTorch, PEFT, openpi-aligned target) | n/a (BF16 base) | n/a | 1.83 | 7.33 | 9.08 GB |
| FlashRT FP8+LoRA, sync data, eager | True | None | 1.46 | 5.84 | 13.88 GB |
| + `dataloader_workers=2` | True | None | 2.68 | 10.72 | 13.88 GB |
| + `compile_mode="default"` (Inductor fusion) | True | default | 3.89 | 15.56 | 14.46 GB |
| **+ `compile_mode="reduce-overhead"`** (speed-priority recipe) | **True** | **reduce-overhead** | **4.04** | **16.14** | **14.65 GB** |
| **`cache_bf16_weight=False`** (memory-priority recipe) | **False** | **reduce-overhead** | **3.95** | **15.80** | **10.05 GB** |

The two FlashRT bottom rows are the recommended starting points,
sitting at different points of the same memory ↔ step-time curve.
Pick the one that matches your card and iteration budget.

**30k-step wall-clock at B=4** (the same step count as the
`training/_runs/` archive), measured directly for the FlashRT
recipes and extrapolated from the steady step/s above for the
lerobot row:

| Stack | wall (30k steps) |
|---|---:|
| lerobot pi05 BF16+LoRA (PyTorch, openpi-aligned LoRA target) | ~273 min (~4 h 33 min, extrapolated from 1.83 step/s) |
| FlashRT, speed-priority recipe | **~124 min** (≈2 h 4 min, measured) |
| FlashRT, memory-priority recipe | **~127 min** (≈2 h 7 min, extrapolated) |

The FlashRT speed-priority number is from a real 30k-step LIBERO
RECAP run (`B=4`, `dataloader_workers=2`, `compile_mode="reduce-overhead"`,
`cache_bf16_weight=True`, `seed=42`): plain LoRA wall 126.6 min,
RL+LoRA wall 128.7 min, loss mean dropped 0.042 → 0.012 / 0.044 →
0.012, 0 NaN over 30 000 steps, peak memory 14.65 GB held flat
throughout. The lerobot row has not been replicated end-to-end at
30k steps in this repo; the wall figure is just `30000 / 1.83 step/s`.

### Memory vs step-time trade-off (`cache_bf16_weight` ablation, `compile_mode="reduce-overhead"`, B=4)

| `cache_bf16_weight` | step/s | samples/s | peak GPU |
|---|---:|---:|---:|
| True (default, speed-priority) | 4.04 | 16.14 | 14.65 GB |
| False (memory-priority)        | 3.95 | 15.80 | 10.05 GB |

`cache_bf16_weight=True` (the default) pre-dequantises the frozen
FP8 base into a BF16 buffer that the backward GEMM consumes
directly — costs ~4.6 GB of weight cache, saves the per-step
dequant kernel launches. With CUDA-Graph capture
(`compile_mode="reduce-overhead"`) this trade-off shrinks to
~2 % step-time (the per-layer dequant kernels are amortised inside
the captured graph), so users on tighter cards can flip to
`cache_bf16_weight=False` and get effectively the same FlashRT
step-time at a peak memory close to the lerobot BF16+LoRA path.

The eager equivalent of the same ablation costs more
(2.68 → 2.36 step/s, −12 %) because every layer's dequant becomes
its own kernel launch. CUDA-Graph capture is what brings the
memory-priority recipe close to free.

### Batch-size sweep (`compile_mode="reduce-overhead"`, plain LoRA)

100-step quick scan, RTX 5090 SM120 / 32 GB cap, ``dataloader_workers=2``,
``lora_rank=16``. Loss finite throughout in every cell.

| batch_size | step/s | samples/s | ms/step | peak GPU |
|---:|---:|---:|---:|---:|
|  1 | 10.87 | 10.87 |  92 ms | 12.90 GB |
|  2 |  6.94 | 13.87 | 144 ms | 13.51 GB |
|  4 |  4.01 | **16.03** | 250 ms | 14.65 GB |
|  6 |  2.61 | 15.69 | 382 ms | 15.76 GB |
|  8 |  2.05 | **16.42** | 487 ms | 16.88 GB |

samples/s plateaus around B=4–8 — beyond B=4 the GEMMs are
compute-bound rather than launch-bound, so each extra sample costs
real ms. B=8 is the maximum safe batch on a 24 GB cap (peak
16.88 GB leaves ~7 GB headroom for activation spikes during ACP
prompt re-tokenisation and the optimiser merge step). B=6 sits in a
mild wave-quantisation dip; either B=4 or B=8 is preferable.

30k-step stability runs at B=4 (RTX 5090, ``compile_mode="reduce-overhead"``,
``dataloader_workers=2``, seed=42) confirm the path is production-grade:

| | plain LoRA | RL+LoRA |
|---|---:|---:|
| wall | 126.6 min | 128.7 min |
| steady throughput | 3.95 step/s | 3.89 step/s |
| **peak GPU** (flat over 30k) | **14.65 GB** | **14.65 GB** |
| loss mean first 500 / last 500 | 0.042 → **0.012** | 0.044 → **0.012** |
| convergence ratio | **3.66× ↓** | **3.56× ↓** |

Steady throughput excludes the first ~25 s Inductor compile on the
first call (amortised on a 1k+ step run); peak GPU memory grows by
+0.77 GB to 14.65 GB, well under the 24 GB cap. Per-step loss values
are *not* byte-equal to eager once `compile_mode` is set — the model
samples `noise` and `time` inside `forward()` via the global torch
RNG and Inductor's op-fusion shifts the consumption order, so each
step sees different noise. Trajectory shape (mean over a window)
stays in the same band; the JAX-baseline ratio gate still applies.

Three pieces of plumbing make the compile / CUDA-Graph path safe:

1. **Async data prep** moves PIL JPEG decode + parquet reads off the
   training thread (`training/rl/async_loader.py`). At `workers=2`
   the LIBERO prep window (~325 ms / step at B=4) overlaps with GPU
   compute, removing the largest bubble in the eager loop.
2. **`Fp8MatmulFunction` exposes the FP8 GEMM as a registered
   `torch.library.custom_op`** (`flashvla::fp8_lora_gemm_bf16` and
   `flashvla::fp8_lora_gemm_bf16_into`). Without this, Dynamo /
   Inductor cannot infer the output shape / dtype of the raw-pointer
   `runner.fp8_nn_dev` call and downstream consumers read garbage.
3. **The kernel calls pass `torch.cuda.current_stream(device).cuda_stream`**
   — both `quantize_fp8_static` and `fp8_nn_dev` previously defaulted
   to the legacy default stream, escaping cudagraph_trees capture and
   producing NaN at step 0 in `reduce-overhead`.

### Why FP8 backward is not implemented

A natural follow-up to FP8 forward is FP8 E5M2 backward via
``torch._scaled_mm`` (the path NVTE uses on H100). After working
through the design, this stack does not ship it. The reasoning:

* **Memory savings are already covered.** The reason an FP8 backward
  would help peak GPU is by removing the cached BF16 weight buffer
  (~4.6 GB at lora_rank=16). The
  ``cache_bf16_weight=False`` config already does this via on-the-fly
  FP8 → BF16 dequant, which CUDA-Graph capture amortises to a ~2 %
  step-time cost (see the trade-off table above). FP8 backward would
  reduce that further by maybe ~1–2 GB in scratch, not by another
  base-weight-sized chunk.
* **Speed envelope is small.** Per-step ``aten::mm`` self-CUDA time
  in our profile is ~89 ms / step at B=4, of which only the
  ``M ≥ 512`` half is in a regime where FP8 GEMM beats BF16 GEMM on
  SM120 cuBLASLt. Re-paying the activation quantize for backward
  shrinks the envelope further; the realistic gain is ~2–4 % step
  time on top of the current recipe.
* **Numerical risk is real.** Flow-matching MSE gradients at our
  scale fall in the ~1e-3 to ~1e-1 range; FP8 E5M2 (5-bit exponent)
  has 4-bit mantissa precision and quantising at this magnitude can
  underflow. Validating that LoRA grads stay within the 0.85 cosine
  gate requires a separate numerics study.
* **Integration cost is real.** A backward GEMM that uses FP8 base +
  FP8 quantised grad_y has the same raw-pointer / current-stream /
  ``custom_op`` registration concerns as the forward path (see commit
  ``stream`` fix in ``training/lora/fp8_autograd.py``). Each is a
  potential NaN-at-step-0 debug round.
* **SM120 maturity.** PyTorch 2.9's ``torch._scaled_mm`` SM120 path
  is relatively new; failure modes may need upstream coordination.

Net: 1–2 weeks of work for a ~2–4 % step-time improvement at a
~1–2 GB additional memory saving, with non-trivial numerics risk.
The same engineering budget spent on data-pipeline polish or on
moving to H100 + NVTE produces a better return. Re-evaluate only
if (a) a substantially larger batch / sequence makes the backward
GEMMs become compute-bound rather than launch-bound, or (b) the
``cache_bf16_weight=False`` peak GPU still does not fit the
deployment card.

## Tests

`training/tests/` is `.gitignore`d (per-developer local validation
scripts; not part of the shipped package). To run locally against
this checkout, set the env vars from the table above and execute::

    PYTHONPATH=. python training/tests/test_*.py

`test_rl_jax_baseline_compare.py` spawns a subprocess and runs
~3 minutes (XLA compile dominates short runs); the rest finish in
seconds-to-tens-of-seconds each.
