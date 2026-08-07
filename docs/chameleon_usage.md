# Chameleon-7B on Thor (sm_110)

Standalone Chameleon-7B (image + text) is a direct-instantiation Thor
frontend. This doc covers building, running, the VQGAN backend policy,
the optional FA4 attention fast path, precision results, and measured
latency on Jetson AGX Thor (sm_110).

> **Status — direct-instantiation frontend.** `ChameleonTorchFrontendThor`
> (`flash_rt/frontends/torch/chameleon_thor.py`) is registered in
> `_PIPELINE_MAP` but is **not** dispatched by `flash_rt.load_model` (same
> pattern as Qwen3-VL). Use
> `examples/thor/chameleon_quickstart.py` / `benchmarks/chameleon_thor_latency.py`,
> not `flash_rt.load_model("chameleon")`.

## 1. Architecture / shapes

| stage | detail |
|---|---|
| VQGAN image tokenizer | Chameleon VQGAN (`vqgan.ckpt`), eager PyTorch default; TensorRT opt-in |
| LLM backbone | 32 layers, MHA, D=4096, 32 heads, HD=128, SwiGLU Dff=11008, per-head QK LayerNorm + RoPE, `attention_bias=false` |
| Output | lm_head over 65536 vocab, `mask_image_logits` applied (image-codebook ids suppressed) |

Compute runs in the Chameleon backbone forward in
`flash_rt/models/chameleon/pipeline_thor.py` (dynamic per-tensor FP8 with
fused quantization kernels, cuBLASLt per-shape autotune, L31 selective
clamp; see `docs/chameleon_thor_sm110.md` for the full engineering notes).

## 2. Build (one shared module)

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j$(nproc)
pip install -e ".[torch]"        # add ,thor-fa4 for the FA4 fast path
```

Sanity-check the optional FA4 attention backend:

```bash
python - <<'PY'
from flash_rt.hardware.thor import fa4_backend
print("FA4:", fa4_backend.is_available(), "-", fa4_backend.status())
PY
```

## 3. VQGAN backend policy

The default VQGAN path is the **eager Chameleon tokenizer**
(`use_trt_vqgan=False`) and does not depend on external TensorRT engines,
so the framework can run standalone.

- **If compatible TensorRT engines exist in the deployment, it is
  recommended to opt in explicitly** (`use_trt_vqgan=True` / script flag
  `--use-trt-vqgan`; build them with `scripts/build_vqgan_trt.py`):
  VQGAN encode drops from ~75 ms to ~17 ms and E2E from ~190 ms to ~121 ms.
- The actual backend is recorded in `prefill()` output and benchmark JSON
  (`vqgan_backend: "eager" | "trt"`).

## 4. Run

```bash
python examples/thor/chameleon_quickstart.py \
    --checkpoint /path/to/Chameleon_7B_mGPT \
    --image /path/to/hand_1.jpg \
    --prompt "Describe the image." \
    --benchmark

# explicit TensorRT VQGAN opt-in (recommended when engines exist)
python examples/thor/chameleon_quickstart.py ... --use-trt-vqgan

# latency benchmark with stage separation
python benchmarks/chameleon_thor_latency.py \
    --checkpoint /path/to/Chameleon_7B_mGPT \
    --image-dir /path/to/images \
    --reuse-input-ids --iters 50 --warmup 10
```

## 5. Precision (real images)

All validation uses real photographs, never synthetic token ids:

- FlashRT FP16 vs HF BF16 (last-token logits cosine after
  `mask_image_logits`): **0.9999997**, greedy next-token identical.
- FlashRT dynamic FP8 vs FlashRT FP16: **0.99999999**, greedy identical.
- FA4 attention vs CUTLASS FMHA: output cosine **0.99999994**; E2E FP8-vs-
  FP16 logits cosine stays 0.99999999.
- Incremental KV-cache decode vs full-recompute: greedy tokens identical
  per position (32-token text generation); the bottom-right causal FMHA
  kernel (`fmha_fp16_causal_br`) matches a PyTorch SDPA reference within
  fp16 rounding (SQ ∈ {1, 2, 128, 144}, SK up to 256).

Known limitation: the pure-FP16 path's full-sequence `hidden` output can
contain NaN on image-token rows for long sequences (residual overflow at
L31); last-token logits are unaffected. The dynamic FP8 path (the
recommended default) does not exhibit this.

## 6. Optional fast paths

| Path | Opt-in | Effect (Se≈1056, real image) |
|---|---|---|
| TensorRT VQGAN | `use_trt_vqgan=True` / `--use-trt-vqgan` | encode 74.9→17.3 ms |
| FA4 attention | `use_fa4_attn=True` / `FLASHRT_CHAMELEON_FA4_ATTN=1` | transformer 111→104 ms |
| FP4 FFN | `fp4_ffn_layers=[...]` | measured ~1-2 ms at single image — not worth it, default off |

FA4 (FlashAttention-4, CuTe-DSL) is the vendored/namespaced fast path in
`csrc/attention/flash_attn_4_src` (`thor-fa4` pip extra, `sm_101a` arch
alias). If its deps are missing it silently falls back to the CUTLASS
FMHA kernel. FA4 is **not** enabled by default — the generic path must
keep the framework's own kernels as the default.

## 7. Latency (Jetson AGX Thor, `hand_1.jpg`, target_size=512)

| Metric | FlashRT FP8 | vs HF BF16 (403 ms transformer-only) |
|---|---:|---:|
| E2E, eager VQGAN | ~190 ms | — |
| E2E, TRT VQGAN opt-in + FA4 | **120.2 ms** | **3.4×** |
| transformer-prefill-only (FA4) | **104.2 ms** | **3.9×** |
| incremental decode, steady state | **30.4 tok/s** (32.9 ms/token) | — |

Roofline: per-shape FP8 GEMM microbenchmarks (193-260 TFLOPS) show the
GEMM tactics are already at the Thor ceiling for these shapes; the
remaining gap to the 240 TFLOP/s-based floor is non-GEMM elementwise /
attention / norm work. The optimization ladder: dynamic per-tensor FP8
(baseline FP16 → FP8) was the largest single lever, followed by three
fused dynamic-quantize kernels (residual+norm+quantize, gate+GELU+
quantize, norm+quantize) and the FA4 attention fast path.

## 8. Notes

- The frontend fail-fasts on non-Thor hardware: `ChameleonTorchFrontendThor`
  checks `torch.cuda.get_device_capability()` before checkpoint loading and
  raises on anything other than SM110. The documented development override is
  `FLASHRT_CHAMELEON_THOR_FORCE=1` (skips the probe only; kernels still need
  the real hardware at runtime).
- Prompt capacity is floored to a multiple of 16 at allocation, and
  `set_prompt` validates the **padded** length against that capacity, so a
  non-aligned `max_seq` can never let pad-to-16 overshoot the buffers or the
  KV cache. `generate_greedy` rejects negative `max_new_tokens` and returns
  the prompt-only result for zero.
- `generate_greedy(...)` runs **incremental KV-cache decode**: one prefill
  over the prompt, then M=1 decode steps (`chameleon_decode_step` +
  bottom-right causal FMHA `fmha_fp16_causal_br`). Measured ~30 tok/s
  steady state and ~2.8× wall-clock over the full-prefix recompute path
  for short prompts; per-token cost grows with KV length. Stops on EOS
  (default `</s>`=2, override via `eos_token_id`) or `max_seq`. Requires
  the dynamic-FP8 path (`use_fp8=True`); the eager full-recompute path is
  retained as `_generate_greedy_recompute` for oracle comparisons.
  Benchmark via `scripts/bench_chameleon_thor.py --generate-greedy N`.
- The TRT VQGAN path uses a square `target_size×target_size` bicubic
  resize while eager uses aspect-preserving `var_center_crop` — token
  counts can differ slightly between backends (expected behavior
  difference, not a bug).

## 9. Third-party license (VQ-GAN)

The VQ-GAN module under `flash_rt/models/chameleon/vqgan/` is vendored from
Meta Chameleon and is governed by the Chameleon Research License — not by this
repository's Apache-2.0 license. See `flash_rt/models/chameleon/vqgan/LICENSE`
and `NOTICE` for the full text, provenance (including the upstream CompVis
MIT attribution), and the modification record. The license is noncommercial-
research-only; treat that subdirectory as a separately-licensed component.
