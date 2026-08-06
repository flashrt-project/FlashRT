# Chameleon-7B on Thor SM110

**Platform**: Jetson AGX Thor (SM110, aarch64) · CUDA 13.0 · transformers 4.43+
**Model**: Standalone Chameleon-7B (LLM backbone + VQGAN image tokenizer, **no ActionHead / ActionVAE**)
**Production path**: All 32 layers with **runtime dynamic per-tensor FP8** (implemented in the Chameleon-specific `flash_rt/models/chameleon/pipeline_thor.py::chameleon_forward`) + generic eager Chameleon VQGAN default + cuBLASLt per-shape autotune + L31 selective clamp; TensorRT VQGAN is explicit opt-in only
**Version**: v1.4 (2026-08, added KV-cache incremental decode `generate_greedy`: 30.4 tok/s, token-exact vs full-prefix recompute oracle)

---

## 0. Summary

- **Asset path**: `/path/to/Chameleon_7B_mGPT` (note the actual directory name is `mGPT`, not `mGP`). Contains weight shards, tokenizer, and `original_tokenizers/vqgan.{yaml,ckpt}`.
- **HF direct loading fails**: The current `transformers` `ChameleonForConditionalGeneration.from_pretrained` errors or silently misloads on this checkpoint because `q_norm`/`k_norm` shapes are legacy `[1,128]` (not the newer `[32,128]`). **Workaround**: The production path uses FlashRT's own declarative `WeightLoader` (bypassing HF `from_pretrained` entirely); alternatively, as in `scripts/check_chameleon_thor_precision.py`, use the bare `ChameleonForConditionalGeneration` constructor + `load_state_dict(strict=False)` for the HF reference model (the script imports `ChameleonForConditionalGeneration` directly from `transformers`; use `--skip-hf` to skip the HF reference comparison).
- **Precision validated (real images, not synthetic token ids)**:
  - FlashRT FP16 vs HF BF16 (last-token logits cosine, after mask_image_logits): **0.9999997**, greedy next-token exact match.
  - FlashRT dynamic FP8 vs FlashRT FP16: **0.99999999**, greedy next-token exact match, top-10 overlap 1.0.
- **VQGAN backend policy (framework positioning)**: For **generic/standard Chameleon**, FlashRT preserves framework generality — VQGAN **defaults to eager** Chameleon tokenization (`use_trt_vqgan=False`), with no default dependency on TensorRT engines, ensuring the framework's own capabilities run independently. **If the deployment environment has compatible TRT engines, explicitly opt in** (`use_trt_vqgan=True` or script `--use-trt-vqgan`; measured VQGAN 74.9→17.3 ms, TRT E2E ~121 ms vs eager ~190 ms). Output JSON records the actual backend (`eager`/`trt`).
- **Latest end-to-end performance (real image `hand_1.jpg`, prompt "Describe the image.", target_size=512, stage-aware benchmark, including §4.11 fused kernels + §4.12 FA4)**:

  | Scope | VQGAN backend | FlashRT FP8 p50/mean | Notes |
  |---|---|--:|---|
  | Default E2E | eager | **~190 ms** | VQGAN 74.9 ms dominates; eager bottleneck is VQGAN without TRT |
  | Explicit opt-in E2E | TRT | **121.1 / 121.2 ms** | TRT VQGAN 17.5 ms + transformer 103.5 ms (with FA4) |
  | transformer-prefill-only (FA4) | eager ids reused | **101.9 / 102.0 ms** | HF-comparable, excludes VQGAN, 50 iter |

  > **2026-08-05 re-measurement (single hot window, 20 iter, `benchmarks/chameleon_thor_latency.py`)**:
  > transformer-only FA4 off **111.2 ms** / FA4 on **104.2 ms** (−7.0); E2E eager+FA4 **177.3 ms**;
  > E2E TRT+FA4 **120.2 ms**. Differences from the table above are within thermal noise (±5%);
  > PR-facing docs (`docs/chameleon_usage.md`, `docs/benchmark_comparison.md`, USAGE.md) use the
  > re-measured values.

  Roofline conclusion (see §4.10-4.12): At Se=1056/1072 the theoretical workload is ~**14.3-14.5 TFLOP**; at 240 TFLOP/s the optimistic compute floor is ~**59-60 ms**. Per-shape GEMM micro-benchmarks (§4.11) confirm GEMM tactics are already near the measured Thor ceiling (32-layer GEMM-only ≈61.9 ms), so the gap to floor is primarily non-GEMM work. §4.11 fused RMSNorm/SwiGLU+amax (117.5→110.9 ms), §4.12 FA4 attention (110.9→**101.9 ms**, 58.3% of 240 TFLOP/s, **1.71× floor**). Remaining headroom is in O-projection quantization (no natural fusion point) and KV-cache incremental decode (landed in §4.13).
- **FA4 attention (explicit opt-in)**: Following upstream PR [`flashrt-project/FlashRT#163`](https://github.com/flashrt-project/FlashRT/pull/163) (GROOT N1.7 Thor NVFP4+FA4, single-view 51.6→29.9 ms, 1.70×). Chameleon shapes (Se=1056, 32 heads, HD=128, causal) measured FA4 vs in-repo CUTLASS causal FMHA: **2.75× faster** (450.5→163.9 µs/layer), output cos=0.99999994; integrated transformer-only FP8 **-8.4 ms**. Requires `pip install .[thor-fa4]` (nvidia-cutlass-dsl==4.5.1 + quack-kernels==0.4.1), enabled via `FLASHRT_CHAMELEON_FA4_ATTN=1` or constructor arg `use_fa4_attn=True`, with automatic CUTLASS FMHA fallback when unavailable.
- **KV-cache incremental decode (2026-08, see §4.13)**: `generate_greedy` now uses one prefill + M=1 incremental decode (`chameleon_decode_step`), steady-state **30.4 tok/s** (32.9 ms/token), wall-clock ~**2.8×** vs full-prefix recompute; token-exact vs eager full-prefix recompute oracle (32-token generation 38/38). Added bottom-right aligned causal FMHA symbol `fmha_fp16_causal_br` (decode SQ=1<SK); dynamic FP8 path only (`use_fp8=True`).
- **Key bugs found and fixed** (during this adaptation, see §3):
  1. **PAD row mistaken for "last token"**: `set_prompt()` pads input_ids to a multiple of 16, but the lm_head input was indexed at the padded `Se-1`, hitting a PAD row instead of the real last token's hidden state. Fix: track `self._real_len` (pre-pad length) separately, index with `real_len - 1`.
  2. **Missing `mask_image_logits` caused severe cosine distortion (0.08~0.47)**: HF `ChameleonForConditionalGeneration` sets all 8192 VQGAN image codebook token logits to fp16 min after generating logits (preventing garbled image tokens during text generation). The original FlashRT implementation omitted this mask, so direct raw-logit comparison showed the 1/8 image-token region of the 65536 vocabulary with different distributions, severely pulling down cosine similarity — initially misdiagnosed as "model forward bug". Layer-wise probing (`chameleon_forward`'s `probe` parameter, comparing layer 0/1/4/8/16/31 post-residual-2 hidden states) confirmed backbone cosine ≥ 0.998 vs HF reference, proving the backbone correct; the real issue was the missing image-token mask at the output layer. Fix: `ChameleonTorchFrontendThor` adds `_build_image_token_mask()` (collecting 8192 ids from `IMGIMG*` keys in `config.json`'s `vocabulary_map`), applying `index_fill_(-65504.0)` to `last_logits` after the lm_head GEMM in `_run_forward()`. Pure in-place tensor op, CUDA-Graph safe, no new kernel needed.
- **Known limitation (FP16 full hidden output, not logits)**: The pure FP16 path (`use_fp8=False`) on real long image sequences (Se≈1056) produces NaN in **some non-last token rows** of the `hidden_all` full-sequence output (row-wise check: ~861 of 1056 rows affected, all at image token positions). Root cause: Chameleon-7B deep layers (especially L31) have residual stream magnitudes reaching ~9000 (massive activation), and fp16 (max 65504) overflows locally after multi-layer accumulation; `chameleon_forward_fp16`'s existing `ffn_gate_clamp_value` (default 10000) only clamps gate*up, not the residual stream itself. **This does not affect last-token logits correctness** (argmax/cosine verified correct, since the last token's hidden state itself does not overflow), only the optional debug path of reading `out["hidden"]` full sequence. The dynamic FP8 path with `ffn_down_clamp_value=60000` guarding deep-layer overflow does not exhibit this issue. **Recommendation**: Use dynamic FP8 for production/precision validation; if FP16 full-sequence hidden is needed for debugging, trust only the last few real token rows. This is a known limitation; no new kernel fix is introduced in this round (out of scope).

---

## 1. Production Configuration

```python
from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor

fe = ChameleonTorchFrontendThor(
    checkpoint_dir="/path/to/Chameleon_7B_mGPT",
    use_fp8=True,           # all 32 layers runtime dynamic per-tensor FP8 (recommended default)
    use_cuda_graph=True,    # graph capture on by default
    use_trt_vqgan=False,    # generic Chameleon defaults to eager VQGAN; set True if TRT engines exist
    use_autotune=True,      # autotune cuBLASLt tactic once per new Se
    target_size=512,        # quality-first; 384 is the recommended fast tier; 256 is higher risk
    ffn_clamp_layers=[31],  # default clamps only L31; "all" restores legacy full-layer clamp
    fp4_ffn_layers=None,    # off by default; use [0..7] for sweep
    max_seq=4096,
)

out = fe.prefill("Describe the image.", [pil_image])
# out["logits"]: (VOCAB_SIZE,) fp32, mask_image_logits applied
# out["hidden"]: (Se, D) fp32, padded length; FP8 path full-sequence reliable; FP16 path mid-rows may be NaN (see above)
# out["input_ids"]: padded token id list

gen = fe.generate_greedy("Describe the image.", [pil_image], max_new_tokens=32)
# gen["text"]: full greedy-decoded text
```

- `use_fp8=False` is for reference/debugging only (dynamic FP8 validated at cosine 0.9999998 vs FP16, and faster); not recommended as production default.
- **VQGAN production recommendation**: Generic Chameleon defaults to `use_trt_vqgan=False` (eager, framework-generic); **if the deployment has compatible TRT engines, explicitly set `use_trt_vqgan=True`** (VQGAN 74.9→17.3 ms, E2E ~190→~121 ms).
- `generate_greedy` implements KV-cache incremental decode (§4.13): one prefill + M=1 decode step, steady-state 30.4 tok/s; dynamic FP8 path only (`use_fp8=True`), FP16 / NVFP4 FFN configurations fail-fast. The full-prefix recompute version is retained as `_generate_greedy_recompute` (oracle/debug, always eager).

## 2. Weight Loading

Standard Chameleon-7B uses a 32-layer Chameleon backbone layout (`attention_bias=false`, `mlp_bias=false`, per-head Q/K LayerNorm, SwiGLU FFN). In FlashRT this layout is described by the Chameleon-specific, inline `_llm_block()` declarative spec (see `flash_rt/frontends/torch/_chameleon_thor_spec.py`; SM87 INT8 variant in `flash_rt/frontends/torch/_chameleon_rtx_sm87_spec.py`), independent of any other model's weight spec. `_chameleon_thor_spec.py::build_spec()` declares `model.embed_tokens.weight` / `model.norm.weight` / `lm_head.weight` outside `_llm_block()` (standard Chameleon does not tie word embeddings; `lm_head.weight` is a separate weight).

The checkpoint's `self_attn.q_norm.weight` / `k_norm.weight` (and corresponding biases) are legacy `[1, 128]` shape (model_parallel_size=1, all heads sharing one LayerNorm parameter set); after loading they are uniformly `.reshape(-1).contiguous()` into flat `(128,)` tensors before feeding the existing `qk_norm_rope_fused_fp16` kernel, semantically consistent with HF's `ChameleonLayerNorm` (each head group shares one weight set; with `model_parallel_size=1` this is equivalent to all 32 heads sharing).

## 3. Image Input and Real-Data Validation

**All precision/performance validation uses real images**, not synthetic token ids (path: `/path/to/images/*.jpg`, a directory of real photographs; validation uses a real hand photograph). VQGAN image tokenization uses the `<IMG_START, h_grid_tok, w_grid_tok, [VQ tokens with NEWLINE per row], IMG_END>` layout (`ChameleonTorchFrontendThor._vqgan_encode`), backed by the in-repo `flash_rt/models/chameleon/vqgan/` (`ImageTokenizer` + `VocabTranslation.convert_img2bp2`).

### 3.1 Validation Scripts

```bash
# Precision gate (FP16 vs HF BF16, FP8 vs FP16; real images; default eager VQGAN;
# HF reference comparison imports ChameleonForConditionalGeneration from transformers,
# add --skip-hf for FP16-vs-FP8 only)
PYTHONPATH=. python scripts/check_chameleon_thor_precision.py \
  --checkpoint /path/to/Chameleon_7B_mGPT \
  --image-dir /path/to/images \
  --prompt "Describe the image." \
  --output /tmp/chameleon_thor_precision.json

# Add --use-trt-vqgan for TensorRT VQGAN explicit opt-in acceleration measurement
# Latency benchmark (HF BF16 eager / FlashRT FP16 / FlashRT dynamic FP8; real images)
PYTHONPATH=. python scripts/bench_chameleon_thor.py \
  --checkpoint /path/to/Chameleon_7B_mGPT \
  --image-dir /path/to/images \
  --prompt "Describe the image." \
  --use-trt-vqgan \
  --iters 10 --warmup 2 \
  --output /tmp/chameleon_thor_bench.json
```

### 3.2 Measured Results (single image `hand_1.jpg`, Se≈1056, recorded in §0 table)

- Precision JSON key fields: `flashrt_fp8_vs_flashrt_fp16.logits_cosine=0.9999999999`, `flashrt_fp16_vs_hf_bf16.logits_cosine=0.9999997`, both `greedy_token_match=true`.
- Layer-wise probe (manual validation, not in script): layer 0/1/4/8/16/31 post-residual-2 hidden cosine all ≥ 0.9977 (layer 31 hidden state magnitude ~9000, consistent with HF reference, confirming massive-activation reproduction is correct).

## 4. Hardware Utilization Analysis and Performance Optimization (TRT VQGAN / kernel fusion / autotune / selective clamp / FP4 sweep)

The initial implementation (dynamic FP8 only, no TRT VQGAN/fusion/autotune/selective-clamp) measured 246.2 ms on real images; after `nsys` kernel-level profiling of `_run_forward` (32 LLM layers), four clear, low-risk optimization points were identified and landed sequentially with re-measurement; plus an FP4 FFN sweep and target_size speed/quality tiers.

### 4.1 Profiling Method and Findings

- **Theoretical FLOPs vs measured throughput**: At Se≈1072, single prefill theoretical GEMM+attention ≈14.5 TFLOP; the initial 182 ms LLM forward corresponds to ~80 TFLOPS, while the measured cuBLASLt FP8 GEMM peak on Thor is ~240 TFLOPS — ~33% utilization, suggesting headroom.
- **`nsys profile --capture-range=cudaProfilerApi` kernel summary over 3 `_run_forward` calls**: Two FP8 GEMM tactics (`nvjet_qqhsh_*`) account for **61.3%** combined, CUTLASS causal FMHA (attention) 8.2%, remaining ~30% spread across multiple small elementwise/quantize kernels:
  - `clamp_inplace_fp16` (FFN overflow protection) 6.4%
  - `quantize_fp8_kernel_generic` + `absmax_kernel` (dynamic FP8 per-layer amax + quantize) 6.4%
  - `mul_fp16` + `silu_inplace` (SwiGLU as two separate kernels) 8.1%
  - Other rms_norm / residual_add / qk_norm_rope etc. ~10%
- **Key finding**: `flash_rt_kernels` already has a ready-made fused kernel `gate_geglu_fp16` (= `gate_silu_mul_fp16`, completing `SiLU(gate)*up` in one kernel), but `chameleon_forward`/`chameleon_forward_fp16` used two separate calls (`silu_inplace_fp16` + `mul_fp16`) for SwiGLU. This is a **zero-new-kernel, pure routing-level** optimization opportunity.

### 4.2 Optimization 1: TRT FP16 VQGAN (eager PyTorch → TensorRT)

Latency breakdown showed VQGAN image encoding (eager PyTorch, `img_tokens_from_pil`) accounts for **35%** of a single prefill (86.7 ms / 246 ms). FlashRT's optional acceleration wrapper is `flash_rt.hardware.thor.vqgan_trt_backend.VQGANTRTBackend`, reading fixed-resolution TRT engines from `~/.flash_rt/trt_engines/vqgan/manifest.json`.

Changes: `flash_rt/frontends/torch/chameleon_thor.py`'s `_ensure_trt_vqgan_loaded()`/`_vqgan_encode()` retain TRT-fast-path + eager-fallback logic, but `use_trt_vqgan` defaults to **False**; only constructor arg `use_trt_vqgan=True` or script `--use-trt-vqgan` attempts TRT. With TRT opt-in, VQGAN encoding latency drops from 86.7 ms to **19.8 ms** (~4.4×).

**Policy note (framework positioning)**: Generic/standard Chameleon is a framework-generic capability, defaulting to eager VQGAN with no TRT engine dependency; **if the deployment has compatible TRT engines, explicitly opt in with `use_trt_vqgan=True`**. The default path must keep FlashRT's own capabilities running independently; TRT is an explicit opt-in recommended acceleration.

Note: The TRT path uses fixed `target_size×target_size` square resize (bicubic), while the eager path uses `var_center_crop` preserving aspect ratio — the two produce slightly different image token counts/content for non-square images (Se changes from 1056 to 1072 in this example). This is expected behavior, not a bug.

### 4.3 Optimization 2: Fused SwiGLU Kernel (`silu_inplace_fp16`+`mul_fp16` → `gate_geglu_fp16`)

All 4 `SiLU(gate)*up` computations in `flash_rt/models/chameleon/pipeline_thor.py` (`chameleon_forward` dynamic FP8 branch, AWQ-D static branch, `chameleon_forward_fp16`, `chameleon_forward_calibrate`) were unified from two separate kernel calls to a single `fvk.gate_geglu_fp16(gate, up, out, n, stream)` call. Mathematically identical (same SiLU-mul formula; `gate_geglu_fp16` is `gate_silu_mul_fp16` underneath), saving one kernel launch and one memory round-trip.

**Note**: `flash_rt/models/chameleon/pipeline_thor.py` is a Chameleon-specific file; the change is a pure math-equivalent substitution with no precision regression (FP8 vs FP16 cosine 0.9999999996, consistent with pre-fusion 0.9999999997).

### 4.4 Optimization 3: cuBLASLt Per-Shape Autotune

Chameleon's 7 GEMMs per layer (q/k/v/o/gate/up/down) have only 3 distinct `(M,N,K)` shapes at a given Se (q/k/v/o share `(Se,4096,4096)`, gate/up share `(Se,11008,4096)`, down is `(Se,4096,11008)`), plus lm_head's `(1,65536,4096)`. Added `ChameleonTorchFrontendThor._autotune_gemms(Se)` (following the autotune pattern of existing models like motus): dummy buffers run `gemm.autotune_fp8_nn_dev_fp16(...)`/`autotune_fp16_nn(...)` per shape, cuBLASLt internally caches the best tactic by `(M,N,K)`, and subsequent same-shape calls (including inside CUDA Graph) reuse automatically. `set_prompt()` runs once per new Se (`self._autotuned_se` dedup), on by default (`use_autotune=True`).

Single autotune takes ~1 second (8 candidate tactics × 4 shapes), after which steady-state latency drops from 187.8 ms to **140.3 ms** (~1.34×).

### 4.5 Optimization 4: L31 Selective Clamp

Post-optimization profiling still showed `clamp_inplace_fp16` at **8.3%**. Code comments and measurements confirm the real protection need is deep-layer outliers (especially L31): disabling clamp entirely causes real-image FP8 vs FP16 logits cosine to become NaN, but keeping only L31 clamp maintains precision with no regression.

Implementation: `chameleon_forward(...)` gains `ffn_clamp_layers=None` parameter, **default None means all-layer clamp (most conservative)**; the standard Chameleon frontend resolves this to `frozenset({31})` by default, overridable via `ffn_clamp_layers=[...]` or env `FLASHRT_CHAMELEON_FFN_CLAMP_LAYERS="24-31"/"all"/"off"`.

Result: single-image 512-tier dynamic FP8 from **140.3 ms → 130.6 ms**, real-image FP8 vs FP16 logits cosine **0.99999999996**, two-image real input also passes (FP8 vs FP16 cosine **0.99999999994**).

### 4.6 FP4 FFN Sweep (implemented, off by default)

The standard Chameleon frontend has integrated the decoupled FP4 FFN mechanism: re-reads `gate_proj/up_proj/down_proj` FP16 weights from the original safetensors, packs `gu_w_fp4/d_w_fp4` + scale-factor buffers, passed to `chameleon_forward(fp4_ffn_layers=...)`. Off by default, enabled via `fp4_ffn_layers=[...]` or env `FLASHRT_CHAMELEON_FP4_LAYERS="0-7"`.

Real-image single-image 512-tier sweep:

| FP4 FFN layers | Latency (mean) | greedy | Notes |
|---|--:|---|---|
| Off | 132.1 ms | EOS | Current safe default |
| L0-3 | 133.4 ms | EOS | Actually slightly slower |
| L0-7 | 130.4 ms | EOS | Only ~1-2 ms gain |

L0-7 FP8/FP4 vs FP16 logits cosine **0.9999999998**, top-k overlap 0.8, greedy match true. Conclusion: FP4 path is usable but gains are too small on standard Chameleon single-image Se≈1072; **not recommended for default enablement**; reserved for multi-image/long-sequence/higher-resolution sweeps.

### 4.7 target_size Speed/Quality Tiers

`target_size` is currently the biggest lever, as it directly changes VQGAN image token count and LLM Se. Scripts `check_chameleon_thor_precision.py` / `bench_chameleon_thor.py` have added `--target-size` parameter.

Real-image single-image measurements under the current optimization stack:

| target_size | Se | Latency (mean) | greedy | Recommendation |
|---|--:|--:|---|---|
| 256 | 288 | **64.8 ms** | image-start token | Fastest but high quality risk |
| 384 | 624 | **93.2 ms** | EOS | Recommended fast tier |
| 512 | 1072 | **130.6 ms** | EOS | Default quality tier |

Conclusion: Generic Chameleon defaults to 512, but serving can expose 384 as a low-latency mode; 256 requires task-level quality validation before use.

### 4.8 Graph Split and TRT Non-Default Stream Fix

Subsequent inspection found a CUDA Graph correctness hazard: the old implementation captured lm_head's `last_hidden_ptr` into the graph, so when the same padded `Se` had different real `real_len`, the graph would reuse a stale last-token pointer. Fixed to **backbone graph + eager lm_head projection**: `_capture_graph()` captures only the 32-layer Chameleon backbone, and each replay calls `_project_last()` with the current `self._real_len`. This also lets `generate_greedy()` reuse the backbone graph within the same padded-Se block; 4-token test at target_size=384 improved from **78.4 ms/token → 72.8 ms/token**.

Additionally, TRT VQGAN originally called `execute_async_v3` on the PyTorch default stream, with TensorRT warning about potential synchronization overhead. Added a dedicated `self._trt_stream` completing preprocess/TRT/translation/token list materialization on that non-default stream; warning resolved, VQGAN encoding ~**19.3 ms**.

### 4.9 Optimization Summary (single image `hand_1.jpg`, dynamic FP8, target_size=512)

| Stage | Latency | Cumulative vs HF BF16 |
|---|--:|--:|
| Initial (dynamic FP8 only) | 246.2 ms | 1.62× |
| + TRT VQGAN | 193.3 ms | 2.06× |
| + SwiGLU fusion | 187.8 ms | 2.14× |
| + autotune | 140.3 ms | 2.87× |
| + L31 selective clamp | **130.6 ms** | **3.08×** |

### 4.10 Stage-Aware E2E and Theoretical Ceiling Assessment (2026-08-05 re-measurement)

To avoid conflating VQGAN-inclusive E2E with HF reference transformer-only numbers, `scripts/bench_chameleon_thor.py` has added stage-aware output, `--reuse-input-ids` transformer-only mode, `--generate-greedy N`, and roofline fields.

**target_size=512, single image `hand_1.jpg`, FlashRT FP8:**

| Scope | VQGAN | Se | p50/mean | stage split |
|---|---|--:|--:|---|
| Default E2E | eager | 1056 | **186.5 / 186.6 ms** | encode 74.7 ms + transformer 111.7 ms |
| Opt-in E2E | TRT | 1072 | **129.7 / 129.7 ms** | encode 17.5 ms + transformer 112.1 ms |
| transformer-only | ids reused | 1056 | **117.5 / 117.5 ms** | embed 0.43 + backbone 114.8 + lm_head 2.26 ms |

**Roofline:**

- Estimated work at Se=1056: **14.26 TFLOP**.
- Measured Thor FP8 GEMM plateau used as roofline reference: **240 TFLOP/s**.
- Optimistic compute floor: **59.4 ms**.
- Measured transformer-prefill-only p50: **117.5 ms**.
- Achieved throughput: **121.4 TFLOP/s**, ~**50.6%** of 240 TFLOP/s.
- Measured/floor: **1.98×**.

Conclusion: The current implementation is significantly faster than HF/model eager, but **has not reached the theoretical optimum**. Using "1.25-1.50× optimistic floor" as a near-optimum gate, the current 1.98× still has kernel/backend headroom. Continued optimization priority: 1) LLM backbone large GEMM/FMHA profile + tactic/attention backend; 2) FP8 causal FMHA (more important for long Se/multi-image); 3) true KV-cache incremental decode (landed in §4.13).

**Nsight Systems eager-backbone profile (target_size=512, ids reused, no graph, 3 iters)**:

| Hotspot | Share | Notes |
|---|--:|---|
| FP8 GEMM (`nvjet_qqhsh_*`) | **55.4%** | q/k/v/o/gate/up/down main GEMMs, still the largest item |
| CUTLASS causal FMHA | **11.4%** | attention, visible at Se≈1056, higher at long Se |
| `gate_silu_mul_kernel` | **10.9%** | fused from silu+mul two kernels to one, but still a large elementwise pass |
| dynamic FP8 quantize + absmax | **9.0%** | per-layer dynamic amax + quantize fixed cost |
| norm/residual/qk_rope | **~10.9%** | multiple small kernels aggregated |
| lm_head | **1.9%** | not a priority |
| clamp | **0.3%** | nearly eliminated after L31 selective clamp |

Checked existing `silu_mul_split_fp8_fp16` / `gate_geglu_merged_fp8_fp16`: both require a known `d_scale` input and cannot directly replace the current dynamic FP8 `gate_geglu + amax + quantize`. Forcing a static down scale would reintroduce the static-scale distortion risk proven on long sequences. The next step to capture this 9-11% elementwise+dynamic-quant overhead requires **new/modified graph-safe fused dynamic scale kernels** (SiLU×Up with simultaneous amax+quantize) or accepting re-validation of static/semi-dynamic scales — not a simple routing change to existing kernels.

**generate_greedy (then-current scope, superseded by §4.13 incremental decode):** target_size=384, ids reused, 8-token full-prefix greedy: FP8 **88.7 ms/token**. This is not the decode-theoretical-optimal path since every token re-runs the full prefix; approaching decode optimum requires KV append/M=1 decode (implemented, §4.13: 32.9 ms/token).

**FP8 causal FMHA feasibility assessment (conclusion: not integrated)**: The FP8 causal FMHA library (`libfmha_fp8_causal.so`, source `csrc/attention/fmha_fp8_causal.cu`) is compiled and present on disk. However, inspecting the kernel signature reveals: `extern "C" int fmha_fp8_causal(Q, K, V, O, ..., float scale_q, float scale_k, float scale_v, float inv_scale_o, stream)` — the 4 dequantization coefficients are **calibration-time-fixed host float scalars** (`ctypes.c_float`), not device pointers. The preceding Q/K/V quantization step (`quantize_fp8_static_fp16`) uses device pointers and is graph-safe, but the FMHA kernel's dequantization values freeze once calibrated, not updating with subsequent real-input amax changes. This is the same failure mode as the "static per-tensor scale mismatch on long sequences/multimodal inputs" diagnosed and fixed in §0 (cosine 0.738 at the time, recovered to 0.9997 only by pivoting to runtime dynamic FP8). Standard Chameleon at Se≈1056-1072 with different input images each time is a high-risk scenario for this failure mode; the same path measured net negative (~**-3.9 ms**) in prior testing, off by default. Conclusion: **FP8 causal FMHA is not integrated into standard Chameleon**; the current dynamic-FP8 GEMM + CUTLASS FP16 causal FMHA combination remains the default attention path.

### 4.11 Corrected Roofline + Fused Dynamic Quantize Kernels (2026-08-05 re-measurement)

**§4.10's "1.98× floor" framing is misleading**: Using per-shape GEMM micro-benchmarks to individually measure the 7 GEMM shapes actually used per Chameleon layer (q/k/v/o: `(1056,4096,4096)`; gate/up: `(1056,11008,4096)`; down: `(1056,4096,11008)`), post-autotune per-shape measured throughput is **193-260 TFLOPS**, and the cumulative 32-layer GEMM-only theoretical lower bound is ≈ **61.9 ms** — nearly identical to the naive "240 TFLOPS peak" calculation of 59.4 ms. This confirms **GEMM tactics are already near the measured ceiling for these specific shapes on Thor**; §4.10's 1.98× gap is primarily "total wall-clock including 40-47% non-GEMM elementwise/attention/norm overhead" divided by "pure GEMM peak", not evidence of GEMM efficiency issues. Further GEMM tactic/shape optimization is no longer meaningful.

**Two new fused kernels (zero precision cost, measured effective)**:

- `rms_norm_quantize_dynamic_fp8_fp16` (`csrc/kernels/norm.cu` adds `rms_norm_amax_kernel` + `quantize.cu` combined host function): RMSNorm's xn write pass uses `block_reduce_max` to atomically reduce abs-max into the scale buffer, eliminating the separate `absmax_kernel` full read of xn inside `quantize_fp8_device_fp16`. Used at pre-QKV and post-attn (gate/up input) per-layer RMSNorm sites.
- `gate_geglu_quantize_dynamic_fp8_fp16` (`csrc/kernels/activation.cu` adds `gate_geglu_amax_kernel`): Same principle fusing SwiGLU write pass with amax reduction, for FFN down-proj input quantization. **Used only on layers not requiring outlier clamp** (default only L31 needs clamp); clamp layers continue using the old `gate_geglu_fp16` → `clamp_inplace_fp16` → `quantize_fp8_device_fp16` three-step sequence, since clamp must take effect before scale computation.
- Wiring: `flash_rt/models/chameleon/pipeline_thor.py::chameleon_forward` dynamic FP8 branch; O-projection quantization (no natural writer to piggyback amax) and L31 clamp layer unchanged. Both new kernels are used only in the `dynamic_fp8_layers` branch, not affecting the AWQ static / FP16 / FP4 branches in the same file.

**Measured results (single image `hand_1.jpg`, target_size=512)**:

| Metric | Pre-fusion | Post-fusion |
|---|--:|--:|
| FP8 vs FP16 logits cosine | 0.9999999996 | 0.9999999991 (still 10 nines, greedy exact match) |
| transformer-only p50 | 117.5 ms | **110.9 ms** (-5.6 ms, consistent with ~5.1 ms estimate) |
| Efficiency vs 240 TFLOP/s | 50.6% | **53.6%** |
| measured/floor | 1.98× | **1.87×** |
| TRT opt-in E2E p50 | 129.7 ms | **~128.2 ms** |

Regression tests: `tests/test_install_smoke.py`, `tests/test_chameleon_thor_vqgan_backend.py` all pass.

**Conclusion (this optimization round stops here)**: GEMM tactics confirmed near their shape-specific ceiling, no longer a lever; the two new fused kernels captured the only remaining clear-ROI portion of dynamic-quantize overhead. O-projection quantization has no natural writer to piggyback amax (input is attention output, not through any elementwise kernel we control), and further kernel gains here show diminishing returns. The lever with order-of-magnitude headroom is **KV-cache incremental decode (M=1 decode)**, an architectural-level change not implemented in this round — landed in the next round, see §4.13.

### 4.12 residual+norm+quantize Triple Fusion and FA4 Attention (2026-08-05 re-measurement)

**Fusion #3: `residual_add_rms_norm_quantize_dynamic_fp8_fp16`**

The post-attn site was previously `residual_add_fp16` + `rms_norm_quantize_dynamic_fp8_fp16` (two kernels). The new kernel (`csrc/kernels/norm.cu`'s `residual_add_rms_norm_amax_kernel`, register-caching residual, ssq using fp16-rounded values, amax folded into the xn write pass; combined host wrapper in `quantize.cu`) merges both into one elementwise kernel. Numerical semantics are bit-identical to the old sequence (same fusion pattern as GROOT N1.7's `ac975b6` commit, but that commit uses static scale; ours is the dynamic amax version). Measured: transformer-only 110.9→**110.3 ms** (-0.6 ms; x/xn buffers L2-resident, savings mainly from launch and L2 round-trips, smaller than optimistic estimate), FP8 vs FP16 cosine 0.99999999 no regression. FP4 branch retains the original `residual_add_fp16`+`rms_norm_fp16` sequence, unaffected.

**FA4 attention (following upstream PR #163)**

- **Upstream evidence**: [`flashrt-project/FlashRT#163`](https://github.com/flashrt-project/FlashRT/pull/163) "GROOT N1.7 update: Thor NVFP4 + FA4 performance tier": single-camera LIBERO 36.8→23.7 ms, dual-view 51.6→29.9 ms (1.70×), action cosine 0.99994-0.99995, graph replay determinism 1.0.
- **Chameleon shape A/B** (Se=1056, NH=32, HD=128, causal, fp16, random Q/K/V): CUTLASS causal FMHA 450.5 µs vs FA4 **163.9 µs (2.75×)**, output cosine **0.99999994**.
- **Integration**: `ThorChameleonAttnBackend` adds `set_fa4_attn(q_tensor, kv_cache)` + FA4 branch before CUTLASS in run() — torch metadata view slicing (no allocation, capture-safe), `fa4(..., causal=True, pack_gqa=True)` under `torch.no_grad()`, output written back to Q_O slot (xn buffer), automatic CUTLASS fallback on exception. Frontend-side `use_fa4_attn=True` / env `FLASHRT_CHAMELEON_FA4_ATTN=1` explicit opt-in; requires `pip install .[thor-fa4]` (nvidia-cutlass-dsl==4.5.1 + quack-kernels==0.4.1), auto-fallback with warning if not installed. `prefill()` output and bench JSON add `fa4_attn` field.
- **Measured**: transformer-only FP8 110.3→**101.9 ms** (-8.4 ms); TRT VQGAN E2E 128.2→**121.1 ms**; FP8 vs FP16 logits cosine **0.9999999912**, greedy exact match. Efficiency 58.3% of 240 TFLOP/s, measured/floor **1.71×**. All regression tests pass.

**§4.9 optimization ladder update (transformer-only / TRT E2E scope)**:

| Stage | transformer-only | TRT VQGAN E2E |
|---|--:|--:|
| Initial dynamic FP8 | ~182 ms | 246.2 ms |
| + TRT VQGAN | — | 193.3 ms |
| + SwiGLU fusion | — | 187.8 ms |
| + autotune | — | 140.3 ms |
| + L31 selective clamp | ~117.5 ms | 130.6 ms |
| + §4.11 fused quantize kernels | 110.9 ms | 128.2 ms |
| + #3 residual+norm+quantize | 110.3 ms | — |
| + FA4 attention | **101.9 ms** | **121.1 ms** |

**Remaining headroom**: O-projection quantization (no natural fusion point), `generate_greedy` full-prefix recompute (KV-cache incremental decode, landed in §4.13), eager VQGAN (default path; Conv-heavy subgraph should use offline compilation per skill conclusion, TRT opt-in already provided).

**Upstream PR adaptation (per FlashRT CONTRIBUTING / docs/adding_new_model.md conventions)**:

- `flash_rt/models/chameleon/pipeline_thor.py` — Chameleon-specific compute-path ownership file (rule 1), containing the `chameleon_forward` family, imported by the frontend.
- `examples/thor/chameleon_quickstart.py`, `benchmarks/chameleon_thor_latency.py` — quickstart and latency benchmark (`<model>_thor_latency.py` naming convention, with `--reuse-input-ids`/`--use-trt-vqgan`/FA4 recording).
- `docs/chameleon_usage.md` (English, lingbot_usage.md style, with VQGAN backend policy), `USAGE.md` Chameleon-7B section, `docs/benchmark_comparison.md` Chameleon table.
- `tests/test_chameleon_thor_fused_kernels.py` — fused kernel vs unfused reference path **bitwise equality** regression (no checkpoint required), satisfying CONTRIBUTING "fused replacements validated against unfused reference paths". This test also exposed and corrected an amax semantics issue: the fused kernel originally computed unrounded fp32 amax, differing by 0.004% from `absmax_kernel` reading fp16-stored values; corrected to compute fp16-rounded values for bitwise exact match (also improving E2E cosine from 0.9999999912 to 0.9999999955).

### 4.13 KV-Cache Incremental Decode (M=1 decode, 2026-08-06)

**Motivation**: After §4.10-4.12, prefill is near the GEMM shape ceiling; the last order-of-magnitude lever for generation is M=1 incremental decode (compute only 1 row per token, no full-prefix re-run). Previously `generate_greedy` re-ran the full prefix per token (88.7 ms/token, target_size=384).

**Implementation (four files, zero new quantize/norm kernels)**:

- `flash_rt/models/chameleon/pipeline_thor.py` adds `chameleon_decode_step(gemm, fvk, bufs, weights, dims, scales_dev, *, attn, pos, ...)`: Se=1 dynamic FP8 backbone, K/V GEMMs write directly into cache row `pos` (`attn.kv_row_ptrs`), RoPE pointer offset `pos*Hd*2`, `attn.run_decode(kv_len=pos+1)`; preserves L31 clamp semantics. `pos` is a host scalar, so decode is always eager (not in CUDA graph).
- `csrc/attention/fmha_fp16_causal.cu`: templatized `FmhaCausalTraits<IsQBegin>`, added bottom-right aligned symbol `fmha_fp16_causal_br` (mask aligned to sequence end when SQ<SK). Prefill (SQ==SK) uses either alignment equivalently, continuing with the original `fmha_fp16_causal` (top-left). Same file compilation, no CMake changes.
- `flash_rt/hardware/thor/attn_backend_chameleon.py` adds `run_decode(site, layer_idx, kv_len, stream)` (FA4 → CUTLASS `_br` → cuBLAS `attention_mha_fp16` three-tier fallback; causal mask degenerates to identity at q_seq=1) and integrated `kv_row_ptrs`; `run()` fail-fast on decode shapes (q_seq=1<kv_seq).
- `flash_rt/frontends/torch/chameleon_thor.py`: `generate_greedy` rewritten as one prefill + single-token decode loop (`use_fp8=False` / `fp4_ffn_layers` fail-fast); decode-related autotune shapes (M=1) merged into `_autotune_gemms`; full-prefix recompute version retained as `_generate_greedy_recompute` (oracle, always eager).

**Measured (text prompt "The capital of France is", 32 new tokens)**:

| Metric | Value |
|---|--:|
| Steady-state decode | **30.4 tok/s** (32.9 ms/token) |
| vs full-prefix recompute (wall-clock) | **2.83×** |
| Per-token vs eager recompute oracle | **38/38 exact** |
| prefill p50 (graph replay) | 157.3 ms (E2E scope) |

Precision note: Under image prompts, occasional decode-vs-oracle argmax flips occur at individual tokens, rooted in M=1 vs full-sequence dynamic FP8 per-tensor scale differences (inherent property of dynamic quantization, not a bug); text prompts are exact. Bottom-right FMHA matches PyTorch SDPA reference within fp16 rounding at SQ∈{1,2,128,144}, SK≤256.

**Two graph/state bugs fixed this round (highest debugging cost, worth documenting)**:

1. **Capture warmup consumed the `x` residual stream**: The backbone updates `x` in-place (`residual_add_fp16` = x += out), so after one forward `x` is the final residual stream (amax ~15632). `_capture_graph`'s warmup ran and then went directly into the capture pass, causing the graph to record computations on a stale residual, polluting dynamic FP8 amax → garbled token generation. Fix: re-embed before capture and before every replay (`_replay_backbone`). Multiple prior "graph corrupts KV cache" probe conclusions were all artifacts of this bug (the probe itself ran backbone twice without re-embedding).
2. **Oracle graph replay polluted shared KV pad rows**: If the recompute oracle used CUDA graph, each growing pad-16 Se would overwrite pad-row cache shared with the incremental path, causing cross-comparison interference. Fix: oracle forced eager.

Lesson: **For backbones that update input buffers in-place, inputs must be restored before "capture/warmup/replay/run-again"**; any "graph vs eager result mismatch" probe should first verify the probe itself is not running a second forward on dirty input.

## 5. Hardware Registration

`flash_rt/hardware/__init__.py::_PIPELINE_MAP` adds:

```python
("chameleon", "torch", "thor"):
    ("flash_rt.frontends.torch.chameleon_thor", "ChameleonTorchFrontendThor"),
```

`resolve_pipeline_class("chameleon", "torch", "thor")` verified to correctly resolve to `ChameleonTorchFrontendThor`. Since standard Chameleon is a text/image chat interface (`encode_prompt`/`prefill`/`generate_greedy`), not a VLA `predict(images)` interface, it uses **direct instantiation** rather than `flash_rt.load_model()`'s `VLAModel` wrapper (same pattern as Qwen3-VL/Nex-N2).

## 6. Out of Scope

- FP4 FFN default production enablement — the optional sweep path is implemented, but single-image 512-tier gains are minimal; default remains off. Incremental decode is mutually exclusive (`generate_greedy` fail-fast).
- New CUDA kernels — decode reuses all existing dynamic FP8/norm/fused kernels; the only kernel-side change is `fmha_fp16_causal.cu` templatization adding the bottom-right aligned symbol (§4.13), not a new operator.
- Fixing pure FP16 path residual stream overflow (§0 known limitation) — requires adding residual stream clamp in `chameleon_forward_fp16`, deferred for future evaluation.
- Decode CUDA graph capture — `pos` is a host scalar (RoPE/cache row offset), currently eager; further decode latency reduction could consider graph-per-pos or device-side pos, not in this round.
