# GROOT N1.7 on AMD Instinct (MI350X / CDNA4)

Model-specific guide. For hardware support, the build, routing, shared
environment knobs and the test suite, see
[deployment_amd.md](deployment_amd.md) first.

## Performance

GROOT N1.7 (3.1 B: Qwen3-VL ViT + truncated Cosmos-Reason2 LLM + VL
adapter + 32-layer DiT action head), 2 camera views, 4-step flow
matching, action horizon 40. Median **full-frame** end-to-end latency —
a fresh observation through the backbone graph plus the action chain,
i.e. what a serving loop pays per frame:

| Configuration | Latency | vs eager |
|---|---|---|
| PyTorch eager (official policy) | 67.9 ms | 1.00× |
| torch.compile (max-autotune) | 77.9 ms | 0.87× |
| **FlashRT AMD, FP8 backbone + bf16 DiT** | **16.0 ms** | **4.24×** |

`torch.compile` is slower than eager for this model on this stack; the
judged baseline is therefore eager.

Scope note: the frame cost splits into roughly 7 ms of backbone graph
replay and 9 ms of action chain (the 4 denoise steps over the 32-layer
DiT). Deployments that reuse backbone features across frames pay only
the action chain.

Accuracy: combined denormalized-action cosine against the reference is
**0.9995** with the initial noise pinned (per-modality: end-effector and
joint targets above 0.999; the 1-D gripper signal is near-constant over
a trajectory so its cosine is naturally lower and is not a useful gate).

## Usage

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="groot_n17",
                            framework="torch", num_views=2)
fe = model.pipeline

# aux is the official processor's output bundle for the observation
# (input_ids, visual/text embeds, rope tables, pixel features, grid_thw).
fe.set_prompt(aux=aux, prompt=instruction_text)

state_normed = fe.normalize_state(state_dict)
out = fe.infer(state_normed, aux=aux)          # fresh observation
actions = fe.denormalize_action(out, state_dict=state_dict)
```

Omitting `aux` from `infer` reuses the backbone features computed at
`set_prompt` and runs the action chain only.

## Precision

The tier here is an **FP8 backbone with an unquantized bf16 DiT action
head**. Note this is deliberately *less* quantized than the Thor and RTX
FP8 frontends, which additionally run the DiT FFN and fused
self-attention QKV in FP8 (`_DIT_USE_FP8 = True`); that DiT calibration
path is not ported to CDNA4 yet, so it remains available headroom rather
than a shipped capability.

- ViT, DeepStack mergers, the truncated LLM and the VL self-attention
  adapter run FP8 GEMMs with per-tensor activation scales.
- The DiT action head, the state/action encoders and the decoder stay
  bf16; they are never quantized.
- Activation scales are calibrated once and cached to disk. A warm
  `set_prompt` loads the cache and runs kernels only; a cold cache runs
  a one-time reference pass purely to extract activation maxima.

There is no BF16-only tier: `use_fp8=False` without `use_fp16=True` is
rejected rather than silently ignored, and the non-quantized full-FP16
reference tier is not ported to CDNA4 (`use_fp16=True` raises
`NotImplementedError`).

## Attention

One backend serves all five attention sites — ViT self-attention, the
causal LLM self-attention, the VL adapter, and the DiT self- and
cross-attention. On the measured GROOT geometries aiter beats torch SDPA
at every site (1.19×–1.83×), including head_dim 48 and the GQA causal
site, so aiter is the default and `FVK_AMD_ATTN=sdpa` is the fallback.

The LLM K/V slots hold the model's native 8 KV heads: aiter performs
GQA internally, so the head-expansion step the CUDA path needs is not
run here.

## Fusion and kernel routing

Two AMD-specific paths carry most of the speedup over a direct port.
Both are controlled by environment knobs so either can be A/B'd against
the decomposed form in one process.

**Fused GEMM epilogues** (`FVK_AMD_FUSED_EPILOGUE`, default on) —
hipBLASLt on gfx950 supports fused FP8 bias and bias+GELU epilogues that
the CUDA SM120 path cannot use, so the backbone runs
`fp8_nn_bias`/`fp8_nn_gelu_bias` instead of a descale GEMM followed by
separate bias and activation kernels. The DiT and the frontend's own
graph closures likewise use `bf16_nn_bias`, `bf16_nn_bias_gelu` and
`bf16_nn_bias_res` (residual accumulation in the epilogue), and the
norm→quantize chains collapse into the fused norm kernels. Together this
removes on the order of a thousand kernel launches per frame.

Numerics note: a fused epilogue adds the bias on the FP32 accumulator
before the output rounding, where the decomposed form adds it after.
The difference is last-ULP and is judged by the end-to-end cosine gate,
not by bit equality.

**Packed-weight MFMA GEMMs** (`FVK_AMD_DIT_GEMM`, default `smallm`) —
the DiT projections are a small-M, weight-bandwidth-bound shape
(M = 41 action+state tokens). A hand-written MFMA kernel with weights
pre-packed at setup into per-lane consumption order beats the autotuned
hipBLASLt kernel by 1.41× on that shape (951 GB/s versus 673 GB/s
effective weight bandwidth) and is routed for the Q/K/V/O projections
only. The FFN shapes measured slower than the library and deliberately
stay on hipBLASLt — routing is by measured shape list, not blanket
substitution. Set `FVK_AMD_DIT_GEMM=hipblaslt` to disable the routing;
the packing costs roughly 430 MB of additional weight storage.

## Environment knobs

Shared knobs are in [deployment_amd.md](deployment_amd.md). GROOT-specific:

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_FUSED_EPILOGUE` | `1` | fused GEMM epilogues and fused norm→quantize chains; `0` selects the decomposed form (also switches the DiT driver back to the hardware-independent forward) |
| `FVK_AMD_DIT_GEMM` | `smallm` | DiT projection GEMMs: `smallm` (packed-weight MFMA on the measured-faster shape) or `hipblaslt` |

Both are read at setup and graph-build time, never on the inference
path; changing them after the graphs are built has no effect.

## Tests

```bash
export FLASH_RT_GROOT_N17_AMD_CKPT=<groot_n17_checkpoint_dir>
python -m pytest tests/test_amd_groot_routing.py \
                 tests/test_amd_groot_model.py -v
```

`test_amd_groot_routing.py` covers the pipeline-map entry, the
`load_model` precision-tier contracts and the attention backend's site
and layer-index validation. `test_amd_groot_model.py` covers the
end-to-end path: frontend construction, `set_prompt` through the FP8
kernel backbone, finite actions of the expected shape, bit-identical
results for a pinned initial noise, and — when a reference fixture is
provided — a cosine gate against it. Both skip with a stated reason
without ROCm, the extension, or the checkpoint.

## Reproducing the numbers

1. Build on a gfx950 machine (see the general guide) with aiter
   importable — without it attention falls back to torch SDPA and every
   number here shifts.
2. Run a full-frame loop: `set_prompt` once, then `infer(state, aux=aux,
   initial_noise=pinned)` in a timed loop after warmup, reporting the
   median.
3. Judge accuracy on the same call with the initial noise pinned to the
   array the reference was generated with, comparing denormalized
   actions per modality.
4. To attribute a change, flip one knob at a time in a single process:
   `FVK_AMD_FUSED_EPILOGUE=0/1` for the fusion tier and
   `FVK_AMD_DIT_GEMM=hipblaslt/smallm` for the projection routing.
