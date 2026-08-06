# Qwen3-VL official BF16 on Jetson Thor (SM110)

This path brings up Qwen3-VL on Jetson AGX Thor using the official BF16
checkpoint weights. It is the SM110 counterpart to the
[Orin BF16 path](./qwen3_vl_rtx_bf16.md) and to the optimized
[SM89 FP8](./qwen3_vl_fp8_sm89.md) and [SM120 NVFP4](./qwen3_vl_nvfp4.md)
Qwen3-VL paths: the model stays in BF16, while the runtime uses FlashRT's
fixed-shape CUDA Graph decode replay and a small set of Thor-friendly BF16
kernels, with opt-in weight-only decode quantization on top.

All language dimensions are read from `config.json`, so the 2B/4B/8B variants
load unchanged. `Qwen3-VL-2B-Instruct` is the primary bring-up target.

## Supported configuration

| | |
|---|---|
| Config name | `qwen3_vl` |
| Framework | `torch` |
| Hardware | `thor` (SM110, compute capability 11.0) |
| Frontend | `flash_rt.frontends.torch.qwen3_vl_thor.Qwen3VlTorchFrontendThor` |
| Attention | `flash_rt.hardware.thor.attn_backend_qwen3.ThorAttnBackendQwen3` |
| Precision | BF16 weights; opt-in W8A16 / W4A16 decode |
| Inputs | text-only, or a single image plus text |

## Checkpoint

Use an official BF16 checkpoint, for example:

```text
Qwen3-VL-2B-Instruct
Qwen3-VL-8B-Instruct
```

Only **2B is validated** on Thor. Larger variants load through the same
config-driven path, but they are untested here, and their decode projections
fall outside the fixed-K M=1 GEMV (K in {2048, 6144}) onto the cuBLASLt path, so
decode throughput will differ materially.

No offline quantization step is required. When `weight_mode` is not `bf16`, the
decode weights are quantized once at load time from the resident BF16 tensors.

## Build

Thor needs `GPU_ARCH=110` and the gated Qwen3-VL kernel module:

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
pip install -e ".[torch]"
cmake -B build -S . -DGPU_ARCH=110 -DFLASHRT_BUILD_QWEN3_VL=ON
cmake --build build -j$(nproc) \
  --target flash_rt_kernels flash_rt_qwen3_vl_kernels
```

Note there is **no `flash_rt_fa2` target here.** The vendored FA2 is not built
for sm_110 (see the `ENABLE_FA2` gate in `CMakeLists.txt`), which is the main
reason Thor has its own frontend and attention backend.

For SM110 the Qwen3-VL module compiles the BF16 ViT helpers, the cuBLASLt BF16
matmul, the `bf16`/`w8`/`w4` M=1 decode GEMVs, and the batched QK norm-rope
kernel used by prefill. It does **not** compile the FP8 block-128 path, so the
ViT tower runs BF16.

Two Orin-only pieces are deliberately absent here: the `int8`/`int4` decode
GEMVs and the INT8 KV cache (`csrc/kernels/qwen3_int8_kv.cu`,
`ENABLE_QWEN3_VL_INT_DECODE`). Thor has hardware FP8 conversion, so `w8`/`w4`
are the right weight tiers, and `ThorAttnBackendQwen3` has no INT8 KV path —
building those would export unused bindings. There is no `kv_mode` knob on Thor.

## Runtime architecture

**Attention.** `ThorAttnBackendQwen3` keeps exactly the buffer surface the RTX
backend exposes (`Q_buf` / `K_cache` / `V_cache` / `O_buf`, `get_slot_ptrs`,
`kv_layer_stride_bytes` / `kv_row_stride_bytes`, `reset_cache`, `run`), so the
frontend's KV-write pointer math is arch-independent. The kernel underneath is a
GQA `scaled_dot_product_attention`: numerically faithful (BF16 math, FP32
softmax) and the correctness baseline for this bring-up. Swapping in the Thor
CUTLASS causal FMHA later requires no frontend change.

Native-GQA SDPA support is probed once at construction, not on the first
attention call — the probe itself launches an SDPA, which must not happen inside
a CUDA Graph capture. When unsupported, `run` expands K/V instead.

**RoPE and KV writes.** Prefill drives the batched
`qwen3_qk_norm_rope_kvwrite_batched_bf16` in a single launch per layer, which is
bit-identical to the per-row `qwen3_q_norm_rope_qstage_bf16` /
`qwen3_k_norm_rope_kvwrite_bf16` kernels decode uses. Prefill and decode math
therefore agree without re-implementing RoPE in torch, and prefill avoids
roughly 2·S kernel launches per layer.

**Vision.** The BF16 `Qwen3VlVisionRtx` tower is reused with its FP8 path
explicitly disabled. Autodetecting on compute capability is not sufficient here:
Thor reports sm_110, which satisfies the `>= 89` FP8 test, but its Qwen3-VL
module deliberately omits the FP8 block-128 ViT kernels. The Thor frontend
therefore explicitly selects non-causal SDPA for patch attention; existing RTX
frontends retain the required FA2 backend and still fail fast if
`flash_rt_fa2` is missing. Patch attention is bidirectional, so SDPA is
equivalent rather than an approximation. On Thor the tower prefers the
**cuDNN backend**, probed once at kernel init (never inside a capture): at the
ViT shape (head_dim 64, non-causal, 6256 patches) it measures 1.6 ms per block
vs 3.8 ms for torch's default flash backend — 2.3×, worth ~52 ms of
full-resolution prefill across the 24 blocks. When the probe fails the code
falls back to the default SDPA dispatcher.

**Linears.** M=1 decode uses the dedicated warp-per-row BF16 GEMV, which beats
cuBLASLt's weak M=1 tactics. Larger M (prefill) goes through the Thor cuBLASLt
BF16 matmul.

## Quickstart

```bash
python examples/thor/qwen3_vl_quickstart.py \
  --checkpoint /path/to/Qwen3-VL-2B-Instruct \
  --image FlashRT.png \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 32
```

Omit `--image` for a text-only prompt. Use `--no-graph` for the eager decode
correctness reference. `--benchmark N` times eager prefill and warm graph decode
separately.

Constructor arguments:

| Argument | Default | Meaning |
|---|---|---|
| `checkpoint_path` | — | BF16 checkpoint directory |
| `device` | `'cuda:0'` | target device |
| `max_seq` | `4096` | KV cache capacity, in prompt + generated tokens |
| `max_pixels` | `None` | cap on visual tokens, forwarded to the processor's smart-resize |
| `weight_mode` | `'bf16'` | decode weight tier (see below) |
| `wq_overrides` | `None` | per-projection tier overrides (see below) |

`max_seq` sizes the KV cache and must cover prompt plus generated tokens; a
full-resolution image prompt already costs ~1.6 K tokens, so the `4096` default
leaves modest headroom. `set_prompt_text()` takes a plain string (plus optional
`system=`) instead of a `messages` list; it wraps the string in a user message
and applies the chat template, for text-only harnesses.

Decode graphs are cached per `(cache_pos, rope_pos)` and the cache is capped at
`max_seq` entries, so a graph is never evicted within a single generation.
Prompts with different image geometries key different `(cache_pos, rope_pos)`
pairs and may evict older entries across a session. Each cached graph holds a
private CUDA memory pool, so very long generations trade GPU memory for
capture-free decode. Thor therefore has no graph-cache environment variables;
the `FLASHRT_QWEN3_VL_*_GRAPH_CACHE_MAX` knobs belong to the other Qwen3-VL
frontends (Orin/SM89/SM120) and are not read here.

The frontend is registered for `('qwen3_vl', 'torch', 'thor')`, so
`flash_rt.hardware.resolve_pipeline_class` finds it. `load_model()` raises a
redirect, because Qwen3-VL exposes a chat surface (`generate(messages) -> str`)
rather than the VLA `predict()` surface.

## Decode weight quantization (`weight_mode`)

Decode runs at M=1 and is bound by weight bandwidth, so shrinking the weights
converts almost directly into tokens/s. The GEMV dequantizes to BF16 in-kernel.
Prefill always keeps using the BF16 weights.

| `weight_mode` | format | weight bytes/element | decode vs BF16 | logit cosine vs BF16 |
|---|---|---:|---:|---:|
| `bf16` (default) | BF16 | 2.0 | 1.00× | — |
| `w8` | FP8 e4m3, scale = amax/448 | 1.125 | 1.61× | 0.99923 (effectively lossless) |
| `w4` | NVFP4 e2m1, scale = amax/6 | 0.625 | 2.38× | 0.986 (chat-grade) |

Scales are per 16 elements, stored BF16. The measured columns come from the
workload in [Jetson Thor validation](#jetson-thor-validation) below; treat the
ratios as the portable result and re-measure on your own prompt lengths.

Unlike Orin, Thor *does* have hardware FP8 conversion, so the e4m3/e2m1 tiers
are the right choice here and stay bandwidth-bound. (The Orin doc explains why
the integer tiers win on Ampere instead.)

`wq_overrides` sets the tier per projection for sensitivity work, keyed
`'{proj}'` or `'L{layer}.{proj}'` over
`qkv_proj` / `o_proj` / `gate_up` / `mlp_down` / `lm_head`:

```python
fe = Qwen3VlTorchFrontendThor(ckpt, weight_mode='w4',
                              wq_overrides={'lm_head': 'w8',
                                            'L0.gate_up': 'bf16'})
```

Unknown keys are rejected rather than ignored, because a silently dropped
override reads as "this projection does not matter" in a sweep. Overrides
apply independently of the global mode: `weight_mode='bf16'` with a single
non-bf16 override quantizes just that projection.
`set_wq_overrides()` requantizes in place and drops the captured decode graphs,
which bake in the quant-buffer pointers.

Measured on the validation workload below, `w8` matched the BF16 greedy top-1
on 31 of 32 teacher-forced steps and `w4` on 30 of 32; every divergence was a
near-tie wording change ("an orange lightning bolt symbol" vs "a stylized
orange lightning bolt icon") and the captions stayed accurate. Still, a single
early divergence can reword a whole continuation, so the low-bit tiers are a
quality/throughput trade rather than a free win. `w4` at full strength wants an
outlier-channel (AWQ-style) fold to hold precision; that fold is not part of
this path, so validate `w4` output against `bf16` on your own workload before
relying on it.

## Jetson Thor validation

Environment:

- Device: Jetson AGX Thor, SM110 (compute capability 11.0)
- L4T: R38.2.0 (Ubuntu 24.04)
- CUDA Toolkit: 13.0.88, driver 580.00
- PyTorch: 2.9.0 + CUDA 13.0
- Checkpoint: `Qwen/Qwen3-VL-2B-Instruct`, official BF16 weights
- Workload: `FlashRT.png`, prompt `Describe this image in one sentence.`
  → 6256 vision patches, 1581 prompt tokens

Build and symbol checks passed as described in [Local checks](#local-checks);
`tests/test_qwen3_vl_thor.py` reports 42 passed on device.

**Correctness.** Text-only and single-image greedy prompts both produce coherent
answers. The eager decode path (`--no-graph`) and CUDA-Graph replay returned
character-identical text on the same prompt, which is the intended property:
graph replay changes launch overhead, not math.

Full-resolution BF16 result, `generate(max_new_tokens=64)` (greedy hits EOS
after ~33 generated tokens on this prompt; cold includes decode-graph capture
and cuDNN/cuBLASLt plan building in a fresh process):

```text
vision patches: 6256
prompt tokens: 1581
generate latency: 2097.0 ms cold / 736.3 ms warm
prefill P50 (eager): 230 ms (run-to-run range 218-233 ms)
decode throughput (warm graph): 65.5 tok/s
```

### Where prefill time goes

Profiled at the workload above, prefill splits roughly 60% ViT tower / 40%
language stack, and the single largest kernel is the ViT's bidirectional patch
attention over all 6256 patches (24 blocks; O(P²) work). Two results from
chasing that number are worth recording:

- **A prefill CUDA Graph is deliberately not shipped.** A working prototype
  (language stack captured, staged inputs, LRU-bucketed on `(S, span)`,
  bit-identical logits) measured **0.3%** at full resolution and 1.6% on a
  text-only prompt. Prefill is GPU-bound on Thor: the host enqueues the whole
  prefill in ~11 ms against ~280 ms of GPU work, so the enqueue cost a graph
  removes was already fully hidden. The decode graphs stay — decode's per-step
  GPU work is small enough that launch overhead matters there.
- **The ViT patch-attention backend switch is what moved prefill.** Routing the
  SDPA fallback through the probed cuDNN backend (see
  [Runtime architecture](#runtime-architecture)) took full-resolution prefill
  P50 from a repeatable ~282 ms to ~230 ms (−18%, run-to-run range 218–233 ms)
  with the generated text unchanged — consistent with the per-block kernel
  delta (2.1 ms × 24 blocks ≈ 51 ms). The CUTLASS SM100 FMHA
  (`libfmha_fp16_strided.so`) was also measured here and **lost** to cuDNN at
  this shape (6.2 ms vs 1.6 ms per block) — its tiles are sized for head_dim
  128, and the ViT runs head_dim 64.

### Measured tiers

Single process, retiering in place via `set_wq_overrides()` so all three rows
share one weight load; decode throughput is the best of 5 × 64-token warm graph
replays at prompt = 1581 tokens. Logit fidelity is teacher-forced against the
BF16 run over 32 decode steps, so it isolates weight-quantization error from
divergence in the sampled sequence:

```bash
python examples/thor/qwen3_vl_quickstart.py --checkpoint <ckpt> \
  --image FlashRT.png --prompt "Describe this image in one sentence." \
  --max-new-tokens 64 --benchmark 10 --weight-mode w4
```

| `weight_mode` | decode | vs BF16 | logit cosine (mean / min) | greedy top-1 agreement |
|---|---:|---:|---:|---:|
| `bf16` | 66.4 tok/s | 1.00× | — | — |
| `w8` | 106.6 tok/s | 1.61× | 0.99923 / 0.99837 | 31/32 |
| `w4` | 158.1 tok/s | 2.38× | 0.98617 / 0.96577 | 30/32 |

The tiers scale close to the bytes they move (2.0 → 1.125 → 0.625 bytes per
weight predicts 1.78× and 3.2×), which confirms decode is bandwidth-bound here
and that Thor's hardware FP8 conversion keeps it that way. The shortfall is the
per-step fixed cost that does not shrink with the weights: attention over a
1581-token KV cache, the norm/RoPE kernels, and graph replay overhead.

### Resolution knob

Capping visual resolution reduces both vision patches and LLM prefill tokens.
Measured at `weight_mode='bf16'`; warm generate is `generate(max_new_tokens=64)`
(greedy hits EOS after ~30–35 tokens depending on resolution), while the decode
column forces all 64 graph-replay steps:

| `max_pixels` | vision patches | prompt tokens | warm generate | prefill P50 (eager) | decode |
|---|---:|---:|---:|---:|---:|
| none | 6256 | 1581 | 736.3 ms | 230 ms | 65.5 tok/s |
| 1.00 M | 3888 | 989 | 622.3 ms | 127.5 ms | 66.5 tok/s |
| 0.50 M | 1824 | 473 | 570.8 ms | 62.3 ms | 68.1 tok/s |
| 0.25 M | 972 | 260 | 518.9 ms | 42.0 ms | 69.1 tok/s |

Every setting produced an accurate caption on this image, though the wording
varied between resolutions; `max_pixels` discards visual detail and should be
validated per task, not assumed free. The knob is a prefill lever: prefill
drops ~5.5× from full resolution to 0.25 M while decode moves only ~5%, since
decode cost is dominated by weight traffic rather than by KV length at these
sequence lengths.

**cuBLASLt autotune.** `FLASHRT_BF16_CUBLASLT_AUTOTUNE_ALGOS` (default `8`,
set `1` to effectively disable) measured under 2% on prefill P50 and no
measurable decode effect. Thor's default cuBLASLt heuristics are already close
for these prefill shapes, so this knob matters much less here than the M=1 GEMV
and `weight_mode` choices.

**Versus Orin.** On the identical workload and prompt length, the
[Orin BF16 path](./qwen3_vl_rtx_bf16.md) records 36.8 tok/s BF16 decode and
927.5 ms prefill P50. Thor is ~1.8× on decode and ~4× on prefill at BF16
(230 ms eager vs 927.5 ms graph replay — modes differ, but Thor measured that
a prefill graph is worth 0.3% here, so the comparison stands), and reaches
158 tok/s with `w4`. The clearest remaining prefill target is the ViT tower
itself — its O(P²) patch attention and FP8-less GEMMs — not the launch path.

## Validation status

| Item | Status |
|---|---|
| Text-only prompt, greedy | supported |
| Single-image prompt, greedy | supported |
| Eager prefill | supported |
| Graph-replay decode | supported (bit-identical to eager) |
| BF16 weights | supported |
| W8A16 / W4A16 decode | opt-in |
| Multi-image, video | **not supported** — raises |
| FP8 ViT tower | **not supported** on Thor |
| Prefill CUDA Graph capture | **deliberately omitted** — prototyped, measured 0.3% (prefill is GPU-bound) |
| INT8 KV cache (`kv_mode`) | **not available** on Thor — Orin-only kernels |
| Speculative decoding | **not supported** |
| Mid-sequence causal prefill block | **not supported** — raises |
| Server integration / `load_model()` | not included |

The last item deserves a note: a causal prefill block ending mid-sequence needs
a bottom-right-aligned mask, which SDPA's `is_causal` does not provide.
Constructing one would allocate on the attention hot path, so the backend raises
instead of silently attending to the wrong positions.

## Local checks

No GPU or checkpoint required:

```bash
python -m py_compile \
  flash_rt/frontends/torch/qwen3_vl_thor.py \
  flash_rt/hardware/thor/attn_backend_qwen3.py \
  examples/thor/qwen3_vl_quickstart.py
PYTHONPATH=. python -m pytest tests/test_qwen3_vl_thor.py -q
```

On the device, confirm the expected symbols resolved and that FA2 is absent:

```bash
PYTHONPATH=. python - <<'PY'
from flash_rt import flash_rt_kernels as fvk
from flash_rt import flash_rt_qwen3_vl_kernels as vlk
for n in ('bf16_matmul_cublaslt_bf16', 'qwen3_vl_bf16_gemv_m1',
          'qwen3_qk_norm_rope_kvwrite_batched_bf16',
          'qwen3_vl_w8_gemv_m1', 'qwen3_vl_w4_gemv_m1'):
    assert hasattr(vlk, n), n
for n in ('embedding_lookup_bf16', 'qwen3_q_norm_rope_qstage_bf16',
          'qwen3_k_norm_rope_kvwrite_bf16'):
    assert hasattr(fvk, n), n
try:
    from flash_rt import flash_rt_fa2
    raise SystemExit('unexpected: flash_rt_fa2 built on sm_110')
except ImportError:
    print('ok: fa2 absent as expected')
PY
```
