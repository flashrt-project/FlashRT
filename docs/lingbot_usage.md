# LingBot-VLA on Thor (sm_110)

LingBot-VLA is a Qwen2.5-VL backbone + flow-matching action expert. This doc
covers building, the FA4 fast path, accuracy/latency, and running it on Jetson
AGX Thor (sm_110).

> **Status — low-level path only.** LingBot runs through the **low-level
> `graph_runner` path** (`graph_runner.sample_actions_graph`: weight spec +
> CUDA-graph capture). It is **not** registered in `_PIPELINE_MAP` and
> `flash_rt.load_model` does **not** dispatch a `lingbot` config; this is **not**
> the stable `load_model()` API. `LingbotTorchFrontendThor`
> (`flash_rt/frontends/torch/lingbot_thor.py`) is a **G1 scaffold** whose methods
> raise `NotImplementedError`. Use `examples/lingbot_quickstart.py` /
> `benchmarks/lingbot_thor_latency.py`, not `flash_rt.load_model("lingbot")`.

## 1. Architecture / shapes

| stage | layers | notes |
|-------|--------|-------|
| ViT (SigLIP-style) | 32 | 3 camera views, 224² |
| VLM prefix (Qwen2.5-VL) | 36 | FP8; FA4 for the prefix self-attention |
| Action expert (flow-matching) | 36 | per-step denoise loop, FP8 + FP4 gate_up; FA4 denoise attention |

Action chunk: `[1, 50, 75]` (horizon 50, action dim 75). Denoise step count is
configurable (10 / 25 / 50).

## 2. Build (one shared module)

LingBot's model-specific kernels (fused AdaRMSNorm, SwiGLU tail, QKV+RoPE) are
compiled **into `flash_rt_kernels`** — same pattern as the qwen36 kernels in
`csrc/kernels/`. There is **no separate `flash_rt_lingbot.so`**. The kernels are
`lingbot_`-prefixed and gated behind `ENABLE_LINGBOT`, built only when
`FLASHRT_ENABLE_LINGBOT=ON` (default) **and** `GPU_ARCH=110` (Thor). RTX
(sm_120) / Orin / L40 builds compile neither the sources nor the bindings.

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -B build -S . -DGPU_ARCH=110            # -DFLASHRT_ENABLE_LINGBOT=OFF to skip
cmake --build build -j --target flash_rt_kernels flash_rt_fp4 fmha_fp16_strided
pip install -e ".[torch,thor-fa4]"
```

Sanity-check that the LingBot kernels and FA4 are present:

```bash
python - <<'PY'
import flash_rt.flash_rt_kernels as k
print("lingbot kernels:", sum(x.startswith("lingbot_") for x in dir(k)))   # 15
from flash_rt.hardware.thor import fa4_backend
print("FA4:", fa4_backend.is_available(), "-", fa4_backend.status())        # True - active
PY
```

## 3. FlashAttention-4 (the Thor fast path)

FA4 (CuTe-DSL) gives the denoise + prefix attention ~17% over the fmha path
(`pack_gqa`, cosine preserved). On Thor it must be compiled for **sm_101a** (the
sm_110 Blackwell alias; `fa4_backend` sets `CUTE_DSL_ARCH=sm_101a` for you).

- The FA4 forward source is **vendored, trimmed, and privately namespaced** at
  `csrc/attention/flash_attn_4_src/` (package `flashrt_fa4`, a forward /
  SM100-only subset of `flash_attn/cute`; see its `VENDOR.md`). No `flash-attn`
  wheel is needed, and it never shadows a pip-installed `flash_attn`.
  > **Install note:** the vendor lives under `csrc/` and is loaded from the
  > source tree, so it is **not** bundled into a built wheel. Use an editable /
  > source-tree install (`pip install -e .`); a plain `pip install .` wheel
  > would not ship it (FA4 would silently fall back to fmha). If wheel
  > packaging is needed later, move the vendor under `flash_rt/` or add it to
  > `package-data`.
- The import is isolated in `flash_rt/hardware/thor/fa4_backend.py` — the only
  place that touches FA4. It returns `None` (→ fmha fallback) if unavailable.
- Its runtime deps (`nvidia-cutlass-dsl`, `quack-kernels`) come from the
  **`thor-fa4`** extra: `pip install ".[thor-fa4]"`. They are **not** in `all`.
- FA4 is an **optional fast path**: if its deps are missing it silently falls
  back to the fmha kernel (correct, ~+18 ms@25).

**A/B the attention path:**

```bash
FLASHRT_THOR_FA4=1 python benchmarks/lingbot_thor_latency.py ...   # FA4
FLASHRT_THOR_FA4=0 python benchmarks/lingbot_thor_latency.py ...   # force fmha
```

The benchmark prints `FA4 status: active` or the failure reason. To debug an
unexpected fallback, set `LINGBOT_FA4_DEBUG=1` to print the import traceback,
and check (1) `pip install .[thor-fa4]`, (2) the vendored
`csrc/attention/flash_attn_4_src` exists, (3) `FLASHRT_THOR_FA4` is not `0`.

## 4. Run

```bash
python examples/lingbot_quickstart.py \
    --checkpoint /path/to/lingbot-vla-4b \
    --calibration /path/to/lingbot_thor_static.json \
    --inputs /path/to/baseline_artifacts_10/inputs \
    --steps 50 25 10
```

`--checkpoint` is the `lingbot-vla-4b/` dir (`model.safetensors` + `config.json`;
`modelscope download --model Robbyant/lingbot-vla-4b`). `--inputs` is a dir of
`images/img_masks/lang_tokens/lang_masks/state/noise` `.pt` tensors.

## 5. Accuracy + latency (Thor sm_110, CUDA-graph replay)

Measured back-to-back A/B (fixed `noise` from `baseline_artifacts_10`). Cosine is
the action chunk `[1,50,75]` vs the upstream LingBot BF16 PyTorch reference
(`baseline_artifacts_10/outputs/actions.pt`, available for the 10-step run):

| attention path | steps | cosine vs 10-step ref | P50 |
|----------------|------:|----------------------:|----:|
| upstream LingBot BF16 (reference) | 10 | 1.000000 | — |
| FlashRT (FA4)           | 10 | 0.996245 |  64.1 ms |
| FlashRT (fmha fallback) | 10 | 0.996067 |  73.0 ms |
| FlashRT (FA4)           | 25 | 0.995721 |  97.5 ms |
| FlashRT (fmha fallback) | 25 | 0.994890 | 118.1 ms |
| FlashRT (FA4)           | 50 | 0.995455 | 155.8 ms |
| FlashRT (fmha fallback) | 50 | 0.994928 | 193.8 ms |

- Reference: upstream LingBot BF16, fixed-noise action chunk, 10 denoise steps
  (`baseline_artifacts_10/outputs/actions.pt`). The 25/50-step rows are compared
  against the same 10-step reference, hence the slightly lower cosine — they are
  not a step-matched comparison.
- Acceptance: cosine ≥ 0.995 vs the step-matched BF16 reference (FP8 bring-up
  threshold) — met by both paths at 10 steps.
- FA4 vs fmha are numerically equivalent paths (the FP8/FP4 GEMMs are identical;
  only the attention kernel differs); FA4 is ~15–20% faster (e.g. 155.8 vs
  193.8 ms @50, 97.5 vs 118.1 ms @25), measured back-to-back with
  `FLASHRT_THOR_FA4=1` vs `=0`.

Thor has CUDA-graph tactic jitter (±2–3 ms); always A/B back-to-back and don't
compare runs taken at different times.

## 6. Notes / known limitations

- **Thor (sm_110) only.** The kernels and FA4 are **additive** — other hardware
  builds neither compile the `lingbot_*` sources (gated) nor inherit the FA4
  deps. There is no RTX / JAX LingBot path.
- All intermediate buffers are pre-allocated; the denoise loop is captured into
  a CUDA Graph (no dynamic allocation on the hot path). Shapes (action dim 75,
  horizon 50) and step counts are fixed per captured graph.
- FP8 static scales come from the calibration JSON (`docs/calibration.md`
  contract); calibration is required.
