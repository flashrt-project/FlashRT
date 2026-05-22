# LingBot-VLA on Thor (sm_110a)

LingBot-VLA is a Qwen2.5-VL backbone + flow-matching action expert. This doc
covers building, the FA4 fast path, and running it on Jetson AGX Thor (SM110).

> **Status.** LingBot runs through the **low-level `graph_runner` path** (weight
> spec + CUDA-graph capture). It is **not yet registered in `load_model` /
> `_PIPELINE_MAP`**, and `LingbotTorchFrontendThor` is a **G1 scaffold** (its
> methods raise `NotImplementedError`). Use `examples/lingbot_quickstart.py`,
> not `flash_rt.load_model("lingbot")`.

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
`lingbot_`-prefixed and gated behind `ENABLE_LINGBOT` (auto-on for SM100-class).

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j --target flash_rt_kernels flash_rt_fp4 fmha_fp16_strided
pip install -e ".[torch,thor-fa4]"
```

Sanity-check that the LingBot kernels and FA4 are present:

```bash
python - <<'PY'
import flash_rt.flash_rt_kernels as k
print("lingbot kernels:", sum(x.startswith("lingbot_") for x in dir(k)))   # 15
from flash_rt.models.lingbot import kernel_ops as ko
print("FA4 active:", ko._get_fa4() is not None)                            # True
PY
```

## 3. FlashAttention-4 (the Thor fast path)

FA4 (CuTe-DSL) gives the denoise + prefix attention ~17% over the fmha path
(`pack_gqa`, cos preserved). On Thor it must be compiled for **sm_101a** (the
sm_110 Blackwell alias; the loader sets `CUTE_DSL_ARCH=sm_101a` for you).

- The `flash_attn.cute` **source is vendored** at `third_party/fa4/` — no
  `flash-attn` wheel needed. `_get_fa4()` finds it automatically.
- Its runtime deps (`nvidia-cutlass-dsl`, `quack-kernels`) come from the
  **`thor-fa4`** extra: `pip install ".[thor-fa4]"`.
- FA4 is an **optional fast path**: if its deps are missing it silently falls
  back to the fmha kernel (correct, ~+18 ms@25). Confirm it loaded with the
  snippet above, or set `LINGBOT_FA4_DEBUG=1` to print the import error.

**Common gotcha (the "120 ms" symptom):** if `_get_fa4()` returns `None` and
latency is ~120 ms@25, FA4 is falling back. Check (1) `pip install .[thor-fa4]`,
(2) `CUTE_DSL_ARCH=sm_101a` (the loader defaults it), (3) the vendored
`third_party/fa4` exists, (4) `kernel_ops.py` imports `sys`.

## 4. Run

```bash
CUTE_DSL_ARCH=sm_101a python examples/lingbot_quickstart.py \
    --checkpoint /path/to/lingbot-vla-4b \
    --calibration /path/to/lingbot_thor_static.json \
    --inputs /path/to/baseline_artifacts/inputs \
    --steps 50 25 10
```

`--checkpoint` is the `lingbot-vla-4b/` dir (`model.safetensors` + `config.json`;
`modelscope download --model Robbyant/lingbot-vla-4b`). `--inputs` is a dir of
`images/img_masks/lang_tokens/lang_masks/state/noise` `.pt` tensors.

## 5. Reference latency (Thor sm_110a, CUDA-graph replay, FA4 active)

| denoising steps | P50 |
|-----------------|-----|
| 50 | ~158 ms |
| 25 | ~100 ms |
| 10 | ~64 ms |

Thor has CUDA-graph tactic jitter (±2–3 ms); always A/B back-to-back and don't
compare runs taken at different times.

## 6. Notes

- All intermediate buffers are pre-allocated; the denoise loop is captured into
  a CUDA Graph (no dynamic allocation on the hot path).
- FP8 static scales come from the calibration JSON (`calibration.md` contract).
- LingBot is Thor-only today. The kernels and FA4 are **additive** — other
  hardware builds neither compile the `lingbot_*` sources (gated) nor inherit
  the FA4 deps.
