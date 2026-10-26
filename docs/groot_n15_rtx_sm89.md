# GR00T N1.5 on RTX 4090 (SM89) — torch CUDA-Graph tier

`config="groot_n15"`, `framework="torch"`, `hardware="rtx_sm89"` serves
**GR00T N1.5** (`GR00T_N1_5`: Eagle2.5 = SigLIP2 + 12-layer Qwen3, 4-step
flow-matching DiT head), e.g. the RoboCasa-365 checkpoint
`huangmz21/gr00t-n1.5-robocasa-atomic-seen-180k`.

> **Scope.** This is a *torch-level* bf16 tier. It does **not** use FlashRT's
> FP8 kernels, needs no calibration and does not require the compiled
> `flash_rt_kernels` extension (`load_model` skips that gate for this config).
> It is the N1.5 entry point until a kernel-level N1.5 pipeline exists.

## How it works

The stock Isaac-GR00T `Gr00tPolicy` is kept for Eagle pre-processing and
(un)normalisation. Its model call is replaced by
`Gr00tN15GraphRunner` (`flash_rt/frontends/torch/_groot_n15_graph.py`), which
captures the whole `backbone → vlln/vl_self_attention → 4× DiT` forward as one
CUDA Graph per `(prompt tokens, image tiles)` shape:

- SDPA attention in the vision tower and Qwen3 (`flash_attn` is still needed to
  *construct* Eagle, whose remote code imports it at module level);
- the Eagle image-token scatter becomes cumsum/gather/where (capturable);
- initial noise is a static buffer refilled with `normal_()` before each replay;
- DiT cross-attention `to_k`/`to_v` are computed once and reused across the
  4 denoise steps (bit-identical, ~4 ms saved);
- LRU cache of `max_graphs` graphs; capture failure or batch > 1 falls back to
  the eager reference call (counted in `runner.eager_fallbacks`).

## Usage

```bash
export GROOT_N15_ROOT=/path/to/Isaac-GR00T            # providing `gr00t`
export GROOT_N15_DATA_CONFIG=panda_omron               # DATA_CONFIG_MAP key
```

```python
import flash_rt
model = flash_rt.load_model(
    "/path/to/gr00t-n1.5-checkpoint",
    config="groot_n15", hardware="rtx_sm89",
    num_views=3,                       # must equal the data config's video keys
    embodiment_tag="new_embodiment")
actions = model.predict(
    images=[main, extra, wrist],       # data-config video-key order, uint8 HxWx3
    prompt="open the microwave",
    state={"base_position": ..., "gripper_qpos": ...})  # dict by state key
# actions: {action_key: np.ndarray[horizon, dim]}  (GR00T per-key action dict)
```

Native `video.*` / `state.*` keys in the `images=` dict are accepted too.
`panda_omron` (RoboCasa) comes from the `robocasa-benchmark/Isaac-GR00T` fork;
the upstream release does not ship it.

## Results (RTX 4090, RoboCasa-365 N1.5 checkpoint, 3 views, batch 1)

| Path | `get_action` | Note |
|---|---:|---|
| Eager reference (autocast bf16) | ~70–80 ms | machine-load dependent |
| CUDA-Graph tier | ~46–50 ms | of which graph replay ≈ 33 ms; the rest is CPU pre-processing |

Action difference vs eager (same seed): max 0.005 (≈2 % of the action scale);
discrete outputs (`gripper_close`, `control_mode`) identical. FP8 (torchao
row-wise) was tried and hung under graph capture, so it is not shipped.

## Tests

```bash
pytest tests/test_groot_n15_rtx_sm89.py                       # CPU: routing, observation, K/V cache
GROOT_N15_CHECKPOINT=... GROOT_N15_ROOT=... GROOT_N15_DATA_CONFIG=panda_omron \
  pytest tests/test_groot_n15_rtx_sm89.py                     # + RTX 4090 graph-vs-eager e2e
```
