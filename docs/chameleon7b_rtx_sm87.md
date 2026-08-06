# Chameleon-7B VLM on Jetson AGX Orin (SM87) — FlashRT adaptation

> **Status: Phase 1 complete — Gate 1 PASSES.** Production tier is
> **INT8 W8A8 + Hadamard (QuaRot at 8 bits)**: greedy output is **bit-identical
> to the HF bf16 reference for 16/16 tokens**, worst layer cosine 0.9986,
> last-row logit cosine 0.999968, at **21.07 tok/s decode and 273.8 ms warm
> prefill** (ISL=1032) — both at their measured ceilings (§4.2). Both defects found during bring-up were fixed by
> **quantization-method changes, not precision fallbacks**: a basis rotation for
> the massive-activation outliers (§4.5) and a ported clamp for the L31 FP16
> overflow (§4.6).
>
> This is the authoritative document for **upstream Chameleon-7B as an image+text
> → text VLM** on Orin SM87.

---

## 0. Conclusion first

**Shipped configuration** — `ChameleonTorchFrontendRtxSm87`, all defaults:

| | |
|---|---|
| LLM GEMMs (Q/K/V/O, gate, up) | **INT8 W8A8 + Hadamard rotation** (QuaRot at 8 bits), per-row dynamic activation scales |
| FFN down | INT8 W8A8, per-row dynamic (K=11008 is not a power of two) |
| lm_head | INT8 W8A8 |
| residual / QK-LayerNorm / RoPE / attention / KV cache | FP16, with `ffn_down_clamp=60000` on the last 4 layers |
| attention | FA2 fp16 causal, `split_kv_bias=4` on decode |
| VQ-GAN encoder | FP16 convs, **fp32** codebook distance/argmin |

**Result** (ISL=1032 = one 512² image + prompt, OSL=16, warm):

| metric | value |
|---|---|
| greedy output vs HF bf16 reference | **bit-identical, 16/16 tokens** |
| worst per-layer residual cosine (L0..L31) | **0.9986** |
| last-row logit cosine | **0.999968** |
| decode | **21.07 tok/s** (47.5 ms/token) |
| prefill (LLM) | **273.8 ms** — GEMM at 91 % of the achievable CUTLASS ceiling |
| steady GPU memory | **7.6 GB** |

**Why it works, in one line:** the Chameleon backbone's massive-activation
channels are fixed by a **basis rotation** rather than by more bits, finer
granularity, smoothing, or a per-layer precision fallback — and because the
rotation preserves per-row scales it reuses the stock CUTLASS INT8 GEMMs, so it
costs nothing (§4.5).

## 1. Platform

| Field | Value |
|---|---|
| Device | NVIDIA Jetson AGX Orin 64 GB (`torch.cuda.get_device_properties`: `Orin`, 61.4 GB) |
| GPU | SM **8.7** (Ampere), **16 SMs**, L2 4 MB |
| Memory | LPDDR5X unified, 204.8 GB/s spec |
| FP8 / FP4 | **not native** (Ada sm89+/Hopper/Blackwell only) → INT8/INT4 is the only low-bit route |
| CUDA / torch | 12.2 / 2.3.0 |
| Build | `cmake -B build -S . -DGPU_ARCH=87 -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS='128;256' -DFA2_DTYPES='fp16;bf16'` |

**Measured bandwidth (this is the load-bearing calibration).** Three different numbers, and
picking the wrong one produces >100 % "efficiency" nonsense:

| probe | result | use for |
|---|---|---|
| D2D copy, 512 MB, read+write | **124.5–126.0 GB/s** | copy-bound ops |
| single-stream vectorized reduce (`roofline.py --measure-bw`) | 99 GB/s | nothing — undersaturated |
| **best achieved by a real weight-streaming kernel** (int8 gate GEMM @ M=1) | **173.3 GB/s** = 85 % of spec | **the decode roofline denominator** |

A D2D copy measures 124.5 GB/s in this container. Treat 173 GB/s
(kernel-achieved, read-dominated) as the decode ceiling.

> ⚠️ `/sys/devices/gpu.0/devfreq/*/cur_freq` is **not readable in this container**, so clocks
> cannot be locked or even observed. Every number below is warm (≥30 warmup iters) and a median
> over ≥50 iters. Cross-config *ratios* are trustworthy; absolute values carry DVFS uncertainty
> (idle 306 MHz vs 1300.5 MHz loaded — principle #16).

## 2. Phase 0 — probe verdicts

### 2.1 R1 — FA2 split-KV is a silent no-op at 32 Q heads ⚠️ **and the fix is one Python argument**

`csrc/attention/fa2_wrapper_causal.cu:41-43,152-158`:

```
num_splits = fa2_num_splits_heuristic_causal(batch*num_heads_q*num_m_blocks, num_sms*2, ...)
  → if (batch_nheads_mblocks >= 0.8f * num_SMs) return 1;
```

Chameleon decode: `1*32*1 = 32` vs `0.8 * (16*2) = 25.6` → **`num_splits = 1`**. Passing the
accumulators does nothing. A split-KV win works *only* when the model has few
enough Q heads (e.g. **16**) for the heuristic to engage splitting.

`num_sms` is a pure heuristic knob in this wrapper, so biasing it selects the split count.
Measured (q=1, kv=1040, 32 Q heads, head_dim 128, fp16):

| `num_sms` passed | latency | speedup | max abs diff vs no-split |
|---|---|---|---|
| 0 (baseline, no accum) | 204.9 µs | 1.00× | — |
| 16 (**real SM count**) | 195.2 µs | 1.05× | **0.000e+00** ← proves `num_splits=1` |
| 32 | 149.8 µs | 1.37× | 1.221e-04 |
| **64** | **141.8 µs** | **1.44×** | 1.221e-04 |
| 128 | 154.8 µs | 1.32× | 1.221e-04 (over-split) |

**Verdict: ship a `split_kv_bias` backend parameter, default 4× (`num_sms=64`).** 1.44× on
decode attention, pure Python, graph-safe. The 1.221e-04 delta is fp16 accumulation-order noise
(fp16 eps ≈ 9.8e-4 at magnitude 1), not an error.

### 2.2 R2/R3 — M=1 GEMM: INT8 needs no GEMV; INT4 needs a small-M tile

All shapes at M=1, achieved GB/s = weight bytes / time, normalized to the **173.3 GB/s**
kernel-achieved read ceiling.

| shape | INT8 variant | µs | GB/s | %ceil | INT4 variant | µs | GB/s | %ceil |
|---|---|---|---|---|---|---|---|---|
| Q/K/V/O 4096×4096 | fp16out | 144.2 | 116.4 | 67 % | fp16out | 121.4 | 69.1 | **40 %** ⚠️ |
| gate 11008×4096 | bf16out | 260.2 | 173.3 | **100 %** | bf16out | 178.5 | 126.3 | 73 % |
| up+silu 11008×4096 | silu_gated | 273.3 | 165.0 | 95 % | silu_gated | 165.7 | 136.1 | 79 % |
| down 4096×11008 | fp16out | 267.3 | 168.7 | 97 % | fp16out | 170.7 | 132.1 | 76 % |
| lm_head 65536×4096 | bf16out | 1743.8 | 153.9 | 89 % | *(stays int8)* | — | — | — |
| **per token (GEMM only)** | | **47.70 ms** | | | | **33.45 ms** | | |
| **→ tok/s (GEMM only)** | | **21.0** | | | | **29.9** | | |

**Verdict 1 — INT8: ship CUTLASS as-is, do not write a GEMV.** Four of five shapes are at
89–100 % of the achieved read ceiling. Measured 47.70 ms vs the predicted 44 ms weight floor —
**measured ≈ predicted, so the bottleneck model is correct** (principle #15).

**Verdict 2 — INT4 delivers only 1.43×, not 2×.** Root cause confirmed in source:
`csrc/gemm/cutlass_sm80_int4_rowwise.cu:61` defines exactly one tile (`GemmShape<128,128,128>`)
with **no `M<=64` dispatcher**, whereas INT8 dispatches `M<=64 → 64×128`
(`cutlass_sm80_int8_rowwise_fp16out.cu:330-333`). At M=1 INT4 therefore wastes a 128-row tile —
visible as Q/K/V/O at **40 %** of ceiling (69.1 GB/s) versus INT8's 67 % (116.4 GB/s) on the
same shape with half the bytes. → **`cutlass_sm80_int4_rowwise_t64x128.cu` is justified**;
predicted recovery 33.45 → ~27 ms/token (~37 tok/s GEMM-only).

**Verdict 3 — the planned "M=1 up-projection split" lever is DEAD. Dropped before writing any
code.** The hypothesis was that `cutlass_int8_silu_gated_bf16out` (128×128 tile only,
`cutlass_sm80_int8_silu_gated.cu:54`) would lose ~93 µs/layer at M=1 versus a
`bf16out` (64-tile) + `silu_mul_qwen36_bf16` split. Measured: **273.3 µs vs 260.3 µs = 13 µs**,
i.e. 0.4 ms/token ≈ 0.9 % — and the split adds a `silu_mul` launch plus an 11008-element bf16
round trip that roughly cancels it. The prediction was **7× too optimistic**; both GEMMs are
already bandwidth-bound. (Principle #13: microbenchmark before writing the kernel.)

### 2.3 R4 — HF reference: **PASS**

Stock transformers 4.57.1 has `ChameleonForConditionalGeneration`, but its `ChameleonLayerNorm`
builds weights of shape `(num_heads, head_dim) = (32,128)`
(`transformers/models/chameleon/modeling_chameleon.py:187-202, 281-282`) while this Lumina-mGPT
checkpoint stores `(1,128)` — so it **cannot be loaded directly**.

Working recipe (also: 4.57 *rejects* `state_dict=` together with a checkpoint path, so a
naked-constructor pattern is required — exactly the one used by the HF reference builder in
`scripts/chameleon_orin_check.py`):

1. read both shards, `repeat_interleave(32, dim=0)` the **128** `self_attn.{q,k}_norm.{weight,bias}` tensors;
2. `torch.set_default_dtype(torch.bfloat16)`; `ChameleonForConditionalGeneration(cfg)` with `cfg._attn_implementation = "eager"`;
3. `load_state_dict(sd, strict=False)` → **0 missing, 0 unexpected** (548 tensors); `.eval().cuda()` → 13.1 GB.

Verified in the same run: `mask_image_logits` is live — logits over ids **4..8195** come back at
`-3.390e+38` = `finfo(bf16).min`, while the text range max is `-13.375`. Since
`model_parallel_size == 1`, the expansion is a pure broadcast, so this reference is numerically
equivalent to official `facebook/chameleon-7b`.

### 2.4 R5 — VQ-GAN codebook argmin precision

`ChameleonVQVAE._from_config` + the 129 `model.vqmodel.*` tensors load with **0 missing / 0
unexpected** (confirming the checkpoint is encoder-only and so is the HF module). Codebook index
match on a deterministic 512×512 input, versus a full-fp32 reference:

| configuration | index match |
|---|---|
| fp32 convs + fp32 argmin | 100.00 % |
| **fp16 convs + fp16 argmin** | 98.14 % |
| **fp16 convs + fp32 distance/argmin** | 99.02 % |

The fp32-argmin fix helps (`z²+e²−2ez` is cancellation-prone in fp16;
`modeling_chameleon.py:850-861`) and costs <0.1 ms. This probe used random noise;
divergence on **real images** can be much higher (~92 % has been observed), so
re-measure on real content at Phase 5 before declaring the fix sufficient.

### 2.5 R8 — decode-graph primitives are graph-safe: **PASS**

Captured `index_select(emb, 0, tok, out=x)` → lm_head stand-in → `mask_view.fill_(bf16_min)` →
`argmax(out=)` → `tok.copy_(out_tok)`: capture succeeded (no `code=13`), the **stale-value test
passed** (changing the seed token changed the embedding output, maxdiff 4.85 — i.e. the graph
re-reads `tok` rather than baking it), and the logit mask survived replay. So no fp16
embedding-lookup kernel is needed.

### 2.6 R6 — deferred

Original `original_tokenizers/vqgan.{yaml,ckpt}` vs HF `model.vqmodel.*` equivalence only gates
the **TRT engine** track (a TRT engine built from `vqgan.ckpt` must produce the same tokens as
the safetensors weights). Deferred to Phase 5.

## 3. Design

### 3.1 Zero new CUDA kernels for the QK-Norm / RoPE / KV path

Three source facts combine to make the existing prefill kernel cover decode too:

1. CUTLASS int8/int4 GEMM output row stride is hard-wired to `N`
   (`cutlass_sm80_int8_rowwise_fp16out.cu:169-171`), so a `[32, max_seq, 32, 128]` fp16 KV cache —
   whose per-layer slab is a contiguous `[max_seq, 4096]` with row stride exactly 4096 == N — is
   a **legal GEMM destination**. `AlignmentC=8` (16 B) is satisfied by both the layer and row offsets.
2. `qk_norm_rope_fused_fp16` is in-place with implicit row stride `dim=128` and derives position
   as `seq_pos = row / num_heads` (`qk_norm_rope_fused.cu:56-57, 65-66`) — so at `seq_len=1` every
   row maps to row 0 of whatever cos/sin pointer it is handed.
3. The RoPE tables are C-contiguous `[max_seq, 128]` fp16 (`ChameleonTorchFrontendRtxSm87`
   builds them that way), so position `pos` is `data_ptr() + pos*128*2` bytes.

Therefore:

* **prefill** — point the K and V GEMMs at `Kcache + li*layer_stride` / `Vcache + li*layer_stride`,
  then call `qk_norm_rope_fused_fp16` unchanged (V needs no transform);
* **decode** — point them at `+ pos*4096*2` and call the same kernel with `seq_len=1` and cos/sin
  pre-offset by `pos*128*2`.

Also required: **`Se` must not be even-padded** (e.g. for FP8 GEMM alignment) —
with a real KV cache the pad row is junk that decode *will* attend to, and
CUTLASS constrains only `K`.

Attention correctness: FA2 causal is **bottom-right aligned**
(`fa2_wrapper_causal.cu:126-138`), so `q=1, kv=N` attends all N keys. The cuBLAS fallback
`attention_mha_causal_fp16` is **top-left aligned** (`softmax.cu:182-191` masks with
`q = row % S_q`) and is therefore *silently wrong* at q=1 — the Chameleon backend must **raise**
rather than degrade to it.

### 3.2 Precision policy

| Component | Default (lossless tier) | Opt-in tier |
|---|---|---|
| Q/K/V/O, gate/up, down | INT8 W8A8, per-output-row weight scale, dynamic per-row act | QuaRot W4A4 (`use_int4`), down via block-diagonal `H_128` (`use_int4_down`) |
| lm_head (65536×4096) | INT8 (268 MB/token = 1.74 ms = 3.7 % of budget) | stays INT8 — never int4 |
| residual / RMSNorm / QK-LayerNorm / RoPE / attention / KV cache | FP16 | unchanged |
| VQ-GAN encoder | FP16 convs + **fp32 distance/argmin** | TRT FP16 (Phase 5) |

Decode always uses **dynamic per-row** activation quant — never the prefill static calibration,
which was fitted at M=Se and does not describe a single decode row.

### 3.3 Token contract

`[BOS 0] + n_img × ([8197 <racm3:break>] + [8711 <image>]×1024 + [8196 <eoss>]) + text + [8710 sep]`,
so `S = 1 + n_img*1026 + n_text + 1`. Image token id = **VQ codebook index + 4**, exactly, for
all 8192 codes; the 1024 tokens are a raster scan of the 32×32 latent grid.

> ⚠️ **Trap:** do not hardcode `<racm3:break>: 8710, <eoss>: 8720`. Both are
> **wrong** for upstream Chameleon (8710 is the `sep_token`). Likewise, special
> ids 65536-65539 are out of range for `vocab_size=65536`. This is one of three
> reasons the Chameleon frontend is standalone rather than a subclass — see §4.
>
> ⚠️ `config.json` says `bos_token_id: 1`, which is **stale** (`<pad>`); `tokenizer.json` gives
> `<s> = 0` and that is what the processor emits.

### 3.4 Why the frontend is standalone (not a subclass)

Three of the most attractive inheritable helpers from a VLA-style frontend are
*actively wrong* for upstream Chameleon: its `_preprocess_image` is bicubic/384/`x*2-1` where
Chameleon needs PIL **LANCZOS**/512/`u8*0.0078-1.0` → `[-1, +0.989]`; its `_vqgan_encode` emits a
grid+newline token layout instead of a bare 1024 raster; its `_load_tokenizer` /
`_init_special_token_ids` produce the wrong ids above. Genuinely reusable are the quantizers and
`_split_fused_llm_weights` — extracted to `flash_rt/frontends/torch/_chameleon_quant.py`.

## 4. Phase 1 — implementation and Gate 1

### 4.1 Shipped

| file | role |
|---|---|
| `flash_rt/frontends/torch/_chameleon_quant.py` | checkpoint-agnostic INT8 / QuaRot-INT4 weight quantizers + the fused-projection split |
| `flash_rt/frontends/torch/_chameleon_spec.py` | weight spec — an inlined, Chameleon-specific `_llm_block()` (no dependency on any other model's spec) + `embed`/`norm`/`lm_head` singletons |
| `flash_rt/hardware/rtx/attn_backend_chameleon.py` | `ChameleonAttnBackend` — real per-layer FP16 KV cache, prefill + decode, `split_kv_bias` |
| `flash_rt/models/chameleon/pipeline_rtx.py` | one `chameleon_forward` serving both prefill (`pos=None`) and decode (`S=1`, `pos` set) |
| `flash_rt/frontends/torch/chameleon_rtx_sm87.py` | `ChameleonTorchFrontendRtxSm87` — `set_prompt` / `prefill` / `decode_step` / `generate` |
| `scripts/chameleon_orin_check.py` | Gate-1 harness (HF reference + graph safety + overflow guard) |
| `flash_rt/hardware/__init__.py`, `flash_rt/api.py` | dispatch entry + `_SM87_ALLOWED` + chat-VLM redirect |

**Zero new CUDA kernels**, as predicted in §3.1.

### 4.2 Measured performance (warm p50 over 10 iters after 2 discarded)

| quantity | measured | predicted (§5) | verdict |
|---|---|---|---|
| **decode, ISL=1032** | **47.5 ms/token = 21.07 tok/s** | 53.7 ms = 18.6 tok/s | **beats prediction** -> bottleneck model correct |
| **prefill LLM, ISL=1032** | **273.8 ms** (min 270.6) | ~255 ms | **within 7 %** |
| prefill, plain INT8 (no rotation) | 282.2 ms | — | rotation is free at prefill too |
| prefill, **first call** | 460 ms | — | **1.68x cold-start penalty** — CUTLASS workspace `cudaMalloc` + JIT |
| load / steady memory | 29 s / **7.6 GB** | — | fp16 originals freed after quantization |

> WARNING: an earlier revision of this doc claimed "prefill 496 ms, 1.9x worse
> than predicted — unexplained". That was **our measurement error**: the Gate-1
> harness calls `prefill()` exactly once, so the number included first-call
> CUTLASS workspace allocation and JIT. **There is no prefill gap.** Principle
> #16 exists for exactly this reason — warm before every measurement.

### 4.2.1 Per-kernel breakdown (torch.profiler, S=1032, post-warmup)

Total GPU 281.1 ms (measured before the clamp restriction in §4.2.3):

| kernel | ms | % | calls |
|---|---|---|---|
| CUTLASS INT8 GEMM — Q/K/V/O `(1032,4096,4096)` | 72.71 | 25.9 % | 128 |
| CUTLASS INT8 GEMM — up + fused SiLU-gate | 57.81 | 20.6 % | 32 |
| CUTLASS INT8 GEMM — gate (bf16 out) | 49.36 | 17.6 % | 32 |
| CUTLASS INT8 GEMM — down (t256x128) | 47.61 | 16.9 % | 32 |
| FA2 fp16 causal | 14.07 | 5.0 % | 32 |
| `residual_add_rms_norm_fht` (rotation fused into the norm) | 11.52 | 4.1 % | 63 |
| `qk_norm_rope_fused_fp16` | 9.36 | 3.3 % | 32 |
| `quantize_int8_rowwise_vec8` (bf16, pre-down) | 6.38 | 2.3 % | 32 |
| `clamp_inplace_fp16` | 6.23 | 2.2 % | 32 |
| `fht_int8_quant` (pre-O) | 3.90 | 1.4 % | 32 |
| lm_head + tail | 2.17 | 0.8 % | 4 |

=> **GEMM 229.2 ms (81.5 %)**, elementwise tail 37.8 ms (13.5 %), attention 14.07 ms (5.0 %).

### 4.2.2 The real GEMM ceiling is 64.4 TOPS, not 84.8 — GEMM tuning is spent

The often-quoted 84.8 TOPS figure is the **raw `mma.s8` issue rate** from a
register-only probe. What CUTLASS actually achieves on its best-case shape is
lower: big-square probes measure **58.7 TOPS at 4096^3 and 64.4 TOPS at
8192^3**. Against that realistic ceiling:

| shape | ms/call | TOPS | vs 64.4 ceiling |
|---|---|---|---|
| Q/K/V/O `(1032,4096,4096)` | 0.567 | 61.1 | **95 %** |
| gate / up `(1032,11008,4096)` | 1.694 | 54.9 | 85 % |
| down `(1032,4096,11008)` | 1.540 | 60.4 | 94 % |
| **whole LLM** | **229.2** | **58.7** | **91 %** |

The prefill GEMMs are at **91 % of the achievable CUTLASS ceiling**, and the
isolated probe reproduces the in-pipeline time to within 0.5 % (0.567 vs
0.568 ms on Q/K/V/O) — so there is no pipeline overhead left to recover. This
confirms that the GEMM ladder (swizzle Id4 / stages-5 / t256x128) is spent.
Only `gate/up` at 85 % shows slack, and tile sweeps there measured 256x128 as
"only ~2 % better, not worth a 4th instantiation".

> WARNING: **the roofline probe itself had a DVFS bug**, found here. Whichever
> shape was measured *first* was penalised by clock ramp: Q/K/V/O reported
> **28.3 TOPS** measured first versus **61.1** for the identical shape after
> adding a 3-second saturating pre-ramp, and the per-shape TOPS ascended purely
> in measurement order (28.3 -> 53.8 -> 60.5). `_ramp_clocks()` now runs before
> any timing in the roofline script
> (`scripts/bench/orin_int8_roofline.py`). Any earlier per-shape number from that
> script is suspect.

### 4.2.3 Clamp restricted to the last 4 layers: -6.3 ms

`clamp_inplace_fp16` cost 6.23 ms (2.2 %) across all 32 layers, but the measured
magnitudes (§4.6) grow monotonically with depth and L28 is 1616 — **37x below the
60000 clamp** — so early layers can never reach it. Restricting it to the last
`ffn_down_clamp_last_n` layers (default 4):

| | before | after |
|---|---|---|
| prefill warm p50 | 280.1 ms | **273.8 ms** |
| decode | 20.96 tok/s | **21.07 tok/s** |
| L31 / final / logit cosine, greedy text | 0.999722 / 0.999447 / 0.999968 / 16-of-16 | **bit-identical** |

The Gate-1 harness reports per-layer clamp saturation, so a checkpoint that
violates the monotonicity assumption is detectable; set
`FLASHRT_CHAMELEON_DOWN_CLAMP_LAST_N=32` if that ever happens.


### 4.3 Gate 1 — PASS

Real image (`FlashRT.png`), `"<image>Describe this image."`, ISL=1032, OSL=16,
tier **INT8+Hadamard**:

| check | result | gate | verdict |
|---|---|---|---|
| **greedy text identical to HF** | **16/16 tokens** | 16/16 | **PASS** |
| worst layer cosine (L0..L31) | **0.9986** | ≥0.97 | PASS |
| L31 / final-norm cosine | **0.999722 / 0.999447** | — | PASS |
| last-row logit cosine | **0.999968** | ≥0.999 | PASS |
| graph safety (capture + stale-value) | cos 0.9884 between two seed tokens | not frozen | PASS |
| FP16 residual finiteness | no inf/nan | finite | PASS |
| argmax, image positions (1026) | 89.6 % exact / **95.0 % tie-adjusted** | — | informational (§4.7) |
| argmax, text positions (6) | 5/6 | — | **not binding** — n=6 is too small; one near-tie flip moves it 17 points |

Both engines produce `"The image is a logo for the company Flexsteel. The logo is a"`.

### 4.4 Root cause: the row-0 massive activation

Probing every layer 20-31 against the reference localizes the failure precisely.
It is **not** spread across the tensor — it is **row 0, the BOS/attention-sink
token**, in the last four layers:

| layer | cosine (all rows) | worst-row cosine | worst row | FlashRT max\|x\| | ref max\|x\| |
|---|---|---|---|---|---|
| L24 | 0.9966 | 0.982 | row4 | 1993 | 2512 |
| L27 | 0.9922 | 0.994 | row4 | 1990 | 2512 |
| **L28** | 0.850 | **0.688** | **row0** | 240 | 1632 |
| **L31** | 0.691 | **−0.384** | **row0** | 3502 | 23936 |

(ISL=20 text-only.) The reference's L31 row-0 norm is **42971 vs a median of
10456**, concentrated in a few channels — **d632 = 23936**, then d808, d1282,
d2669. FlashRT has 1225 in d632.

Mechanism: per-row INT8 activation quantization sets `scale = amax/127` from that
outlier, so the other ~4090 channels of row 0 round to zero and the row's
direction is destroyed (cosine goes *negative*). This is the documented
Chameleon massive-activation zone (the skill's backbone profile names d671/d579
for the L15→L19 band; here it is d632 in the L28→L31 band), and per principle
#3/#17 the fix is **basis rotation, not smoothing**.

### 4.5 The tier ladder: rotation × bit-width (W8A8+Hadamard wins)

Two independent error sources act here, and each shipped tier only fixed one:

* **outlier conditioning** — a row whose amax is set by a massive-activation
  channel loses its other ~4090 channels to rounding. Fixed by a **basis
  rotation**, not by finer granularity or smoothing (principle #17).
* **quantization noise** — the resolution left for the other 1031 ordinary rows.
  Fixed by **more bits**.

Measured on the same prompt, all three tiers:

| ISL | tier | L24 | L28 | L31 | final | last-row logit cos | greedy prefix |
|---|---|---|---|---|---|---|---|
| 7 | INT8 (per-row) | 0.9987 | 0.699 | 0.508 | 0.794 | 0.998850 | 8/12 |
| 7 | INT4 (QuaRot) | 0.9985 | **0.981** | **0.960** | 0.955 | 0.999484 | 8/12 |
| 1032 | INT8 (per-row) | 0.9952 | 0.9953 | 0.9983 | 0.9969 | 0.999916 | 8/16 |
| 1032 | INT4+down | 0.9334 | 0.9313 | — | — | 0.998881 | **0/16** |
| **1032** | **INT8+Hadamard** | **0.9989** | **0.9989** | **0.99972** | **0.99945** | **0.999968** | **16/16** |

> This may appear to **contradict an earlier prefill-only conclusion** ("both
> INT4 tiers beat INT8 at every layer probe on every frame"). That measurement
> is not wrong — it was taken on a *prefill-only workload at fixed short Se*;
> the verdict is ISL-dependent, and a VLM's production ISL sits in the opposite
> regime.

So the INT8-vs-INT4 verdict *inverts with sequence length* — at short ISL the
sink row is 1/7 of the tensor and rotation dominates; at long ISL it is 1/1032
and 4-bit noise dominates. That inversion is the tell that the two tiers were
each solving half the problem. **Rotating at 8 bits solves both and strictly
dominates**, which is why it is the default.

Cost: **one new device-side pack function** (`quant_int8`) inside the existing
`csrc/kernels/fht_int4.cu`, plus templating its three kernels on the output
width — the norm and the radix-16 register FHT are shared verbatim with the INT4
path. **No new GEMM**: because the rotation keeps plain per-row scales, the
unmodified `cutlass_int8_rowwise_*` kernels consume the rotated activations
directly. Measured decode **20.96 tok/s vs 19.9** for plain INT8, i.e. no
throughput cost (the FHT rides inside an already-optimized fused norm kernel;
the difference is within the ±3 % process-to-process variance this platform
shows).

The weight side folds offline (`W_rot = H·W/√K`, `quantize_int8_hadamard`); the
activation side is fused into the norm (`rms_norm_fht_int8_fp16`,
`residual_add_rms_norm_fht_int8_fp16`, `fht_int8_quant_fp16`). The FFN **down**
projection stays plain INT8: K=11008 is not a power of two and its input is the
un-rotated BF16 SiLU output.

**Why not the alternatives** (principle #17's measured ladder on this backbone):
SmoothQuant reached only 0.641 and outlier-splitting 0.970 on the A4 variant of
this problem, while group-128 / block-scaled schemes need a bespoke GEMM whose
hand-written ceiling on 16-SM Orin measured just 41 TOPS. A per-layer FP16
fallback would also have worked, but it is checkpoint-specific tuning that
permanently costs throughput — the rotation is free and generalizes.

### 4.6 SOLVED — the FP16 overflow, via `ffn_down_clamp`

With a real image the reference's L31 residual reaches **max|x| = 89088**, above
FP16's 65504, so FlashRT stored `inf` and the final RMSNorm turned that row's
logits into `nan`. It affects **both** precision tiers — it is a property of the
residual *dtype*, not of the quantization.

The first instinct (a BF16 residual stream, ~1 new kernel) was **wrong** — the
answer is a clamp. This port empirically confirmed that a clamp is sufficient.

**Why a clamp is sufficient** — measured per-layer magnitudes in the bf16
reference (ISL=1032). The explosion is confined to **exactly one layer**:

| quantity | L28 | L29 | L30 | **L31** |
|---|---|---|---|---|
| residual | 1616 | 1720 | 2032 | **266240** |
| o_proj output | 76 | 78 | 80 | 1056 |
| down **input** (gu) | 1128 | 1528 | 6080 | **151552** |
| down **output** | 1120 | 1032 | 1880 | **264192** |

Because the pre-L31 residual is only ~2032, clamping the down **output** at
60000 leaves the residual at ~62000 < 65504. So one `clamp_inplace_fp16`
(already in `flash_rt_kernels`, CUDA-Graph safe) removes the overflow with
**zero new kernels and no dtype change**.

We do **not** need to clamp the down *input*: ours is BF16
(`cutlass_int8_silu_gated_bf16out`), whose range absorbs 151552 without issue.
The clamp is applied on every layer because L0-L30 are three
orders of magnitude below it and therefore untouched; cost is 32 extra
elementwise launches (<0.3 % of the decode budget, ~0.7 % of prefill).

**Result** (ISL=1032, INT8, real image):

| | before | after |
|---|---|---|
| L31 cosine | `nan` (inf) | **0.998266** |
| final-norm cosine | `nan` | **0.996923** |
| L31 max\|x\| | `inf` | 60160 (saturating at the clamp, as intended) |
| last-row logit cosine | 0.999916 | 0.999916 |
| greedy prefix vs HF | 8/16 | 8/16 |

Exposed as `ffn_down_clamp` (default 60000, env `FLASHRT_CHAMELEON_DOWN_CLAMP`).

⚠️ **The clamp did not change the text divergence** (still 8/16). That confirms
the overflow was confined to the sink row's post-L31 residual, which feeds only
that row's final norm — so it was never the cause of the divergence. The
remaining gap is ordinary INT8 error at high-confidence text decisions and is
still open; see §6 for the ranked options.

Also note the **gate itself was wrong** at first: an absolute
"max|x| < 30000" threshold fails by construction on a backbone whose reference
legitimately runs at 2.6e5. The correct gate is **finiteness**, with clamp
saturation reported as information.

### 4.7 Measurement-hygiene finding: argmax-over-all-positions is meaningless here

1024 of the 1032 teacher-forced positions are **image** positions. At those the
model predicts a next token while all 8192 image ids are masked out of the
logits (§3.3), so the winner is an arbitrary low-confidence text token — median
reference top1−top2 gap **0.250** on a logit scale of ~20, versus **0.895** at
the 6 text positions. An unsplit "argmax match = 87.21 %" therefore says almost
nothing about generation quality. The harness now reports image and text
positions separately and gates only on text positions, and additionally
classifies a mismatch as a **BF16 tie** when the reference's top-2 gap is within
one BF16 ULP (58 of the 132 mismatches were ties).

### 4.8 Two harness traps worth remembering

* **A forward hook that returns a value replaces the module output.** Using
  `dict.setdefault(...)` inside a `register_forward_hook` lambda returns the
  stored tensor, which silently substituted a *CPU* tensor for
  `model.model.norm`'s output and crashed `lm_head` with a device mismatch.
  Always `return None`.
* **Launching on stream 0 while another stream is capturing silently drops the
  kernels from the graph.** The first graph-safety run reported
  `stale-value: FAIL (frozen)` with cos exactly 1.0000 — not because anything
  was baked, but because only the torch ops got captured and none of the `fvk`
  kernels did. `decode_step` now takes an explicit `stream` argument.

## 5. Roofline ladder — predicted vs finally measured (principle #15)

Decode reads 6.745 G params/token (32 layers 6.476 G + lm_head 0.268 G). **MHA
with 32 KV heads makes the KV cache 4x heavier than a GQA model** — 0.524 MB per
token of context, so 0.55 GB at S=1040 and **2.15 GB at S=4096, where KV would
dominate an int4 tier.**

| tier | weight floor @173 GB/s | + KV @S~1040 | GEMM-only µbench | predicted total | **finally measured** |
|---|---|---|---|---|---|
| **int8 (+Hadamard, shipped)** | 39.0 ms | +3.2 ms | 47.70 ms | 53.7 ms = 18.6 tok/s | **47.5 ms = 21.07 tok/s** |
| int4 (as built) | 19.5 ms | +3.2 ms | 33.45 ms | 39.5 ms = 25.3 tok/s | not shipped (loses on precision, §4.5) |

Decode came in **13 % better than predicted** — the prediction charged full price
for attention and the elementwise tail, but `split_kv_bias` (§2.1) and the fused
FHT norm absorb part of it. A prediction that is close *and* slightly pessimistic
is the sign the bottleneck model is right (a large gap in either direction would
mean the model of the bottleneck is wrong, not that there is tuning left).

**Prefill**: predicted ~255 ms by scaling a measured Se=1214 prefill to
Se=1032; **measured 273.8 ms warm** (within 7 %), of which GEMM is 229.2 ms at
**91 % of the achievable CUTLASS ceiling** (§4.2.2). Image tokenize adds ~53 ms
(PyTorch VQ-GAN; ~27 ms with a 512x512 TRT engine, not built).

> Superseded numbers, kept so they don't re-mislead: an earlier revision of this
> section predicted **18.6 tok/s** decode and this doc once reported **496 ms**
> prefill and an **84.8 TOPS** GEMM target. Current values: **21.07 tok/s**,
> **273.8 ms**, and a **64.4 TOPS** achievable ceiling. See §4.2 for why the
> 496 ms was a cold-start artifact and §4.2.2 for the ceiling correction.

## 6. Ranked lever menu

### Precision — status: closed

| # | lever | outcome |
|---|---|---|
| 1 | **W8A8 + Hadamard (QuaRot at 8 bits)** | **DONE — this closed it.** greedy 8/16 → **16/16**, worst layer 0.9946 → 0.9986, last-row logit 0.999916 → 0.999968, at no throughput cost. Default tier. |
| 2 | `ffn_down_clamp` | **DONE** — removed the L31 FP16 `inf` (§4.6) |
| ~~3~~ | ~~Tier-3 FP16 fallback for L31~~ | **not needed** — the rotation fixed the same layer at 8 bits. A per-layer precision fallback is checkpoint-specific tuning and costs throughput permanently; prefer the quantization method. |
| ~~4~~ | ~~AWQ / SmoothQuant per-K smoothing~~ | **not needed** — and principle #17's measured ladder on this backbone puts smoothing (0.641) far below rotation (0.9914). Kept only as a fallback if a future checkpoint defeats rotation. |
| 5 | ISL-adaptive tier selection | **obsolete** — W8A8+Hadamard wins at both short and long ISL, so there is nothing to switch between |
| ~~6~~ | ~~BF16 residual stream~~ | **superseded by the clamp** — would have cost a new `qk_norm_rope_fused_bf16` kernel to fix what one existing elementwise kernel already fixes |

**Decode (M=1, weight-bandwidth-bound)**

| # | lever | predicted | effort | status |
|---|---|---|---|---|
| 1 | `use_int4` / `use_int4_down` | **1.43×** (measured, not 2× — §2.2) | trivial, already built | Phase 2 |
| 2 | `cutlass_sm80_int4_rowwise_t64x128.cu` + `M<=64` dispatcher | int4 33.45 → ~27 ms = **+20 %** | ~150 lines + 1 CMake line | Phase 3 — **justified by §2.2** |
| 3 | `split_kv_bias = 4` (`num_sms=64`) | attention **1.44×** = +1.7 ms/token (+3–5 %), more at long S | 1 Python arg | Phase 1 — **measured (§2.1)** |
| 4 | Per-position decode CUDA graph | 0–15 % throughput, −6 s startup | medium | Phase 4 |
| 5 | INT8 Q/K/V/O at 67 % of ceiling (small-N tail: 4096/128 = 32 tiles on 16 SMs) | up to +12 % if it reached 100 % | high (hand GEMV) | open |
| 6 | Devpos kernel + fp16 seqused-splitkv FA2 → *one* decode graph | 0 % throughput; removes capture cost + `max_new_tokens` cap | high (1 `.cu` + FA2 rebuild) | deferred |
| 7 | Reduced lm_head (drop rows 4..8195) | +0.5–1 % | low | deferred |
| 8 | INT8 KV cache | +3.8 % @1040, **+12 % @4096** | high (needs a 32Q/32KV variant; break-even measured) | S≥4096 only |
| ~~9~~ | ~~M=1 up-projection split~~ | ~~+7 %~~ → **measured 0.9 %, net ≈0** | — | **DEAD (§2.2)** |
| 10 | Speculative decode | 1.5–2× | N/A — no draft model | — |

**Prefill / TTFT (FLOPs-bound — levers do not transfer)**

| # | lever | predicted | status |
|---|---|---|---|
| 1 | `use_int4` / `use_int4_down` | LLM ~265 → ~165 ms (**−38 % TTFT**) | already built |
| 2 | 512×512 TRT VQ-GAN engine | 53 → 27 ms (−9 % TTFT) | `scripts/build_vqgan_trt.py`, inputs present in `original_tokenizers/` |
| 3 | GEMM-util / elementwise fusion | **≈0** — measured spent at 74 % of the 84.8 TOPS mma peak | closed |
| 4 | Per-Se prefill CUDA graph | 0–5 %, and a full capture per new prompt length; `Se` cannot be bucketed (padding poisons the KV cache) | **reject for a VLM** |

## 7. Dead-ends (measured — do not re-walk)

| direction | result | one-line reason |
|---|---|---|
| FA2 split-KV with the real `num_sms=16` | **bit-identical, 1.05×** | `32 >= 0.8*32` → `num_splits=1`; the heuristic disables itself at 32 Q heads |
| M=1 up-projection split (`bf16out` + `silu_mul`) | 13 µs/layer ≈ 0.9 %, net ≈0 | both GEMMs already bandwidth-bound; the extra launch + bf16 round trip cancels it |
| INT4 at M=1 expecting 2× | **1.43×** | single 128×128 tile, no `M<=64` dispatcher |
| **`use_int4` / `use_int4_down` as the VLM default** | **L24/L28 cosine 0.933 vs INT8's 0.995; greedy prefix 0/16 vs 8/16** | at production ISL the sink row is 1/1032 of the sequence, so INT8's per-row damage is diluted and 4-bit noise on the other 1031 rows dominates (§4.5). INT4 still wins at short ISL — the verdict is ISL-dependent |
| **unsplit argmax match as a precision metric** | "87.21 %" says nothing | 1024/1032 teacher-forced positions are image positions whose logits are fully masked → arbitrary low-confidence winners (§4.7) |
| forward hook capturing via `dict.setdefault` | CPU/CUDA device crash in `lm_head` | a hook returning non-None **replaces** the module output |
| launching pipeline kernels on `stream=0` during graph capture | `stale-value: FAIL (frozen)`, cos exactly 1.0000 | kernels on the default stream are silently *not* recorded; only the torch ops were captured |
| `attention_mha_causal_fp16` for decode | silently wrong | top-left-aligned causal mask; at `S_q=1` only column 0 survives |
| stock `from_pretrained(..., state_dict=...)` | `ValueError` in 4.57 | use naked ctor + `load_state_dict` |
| stock transformers loading this ckpt unmodified | shape mismatch on 128 tensors | `ChameleonLayerNorm` wants `(32,128)`, ckpt has `(1,128)` |
| **measuring prefill on the first call** | 460-496 ms vs 273.8 ms warm | 1.68x cold-start penalty from CUTLASS workspace `cudaMalloc` + JIT; produced a phantom "1.9x prefill gap" |
| **roofline probe without a clock pre-ramp** | first shape measured reads 28.3 TOPS vs 61.1 warm | Orin DVFS ramps 306 -> 1300 MHz; per-shape TOPS ascend in measurement order |
| clamping the down output on all 32 layers | 6.23 ms (2.2 %) for no effect on 28 of them | magnitudes grow monotonically with depth; L28 is 37x below the clamp |
| trusting 84.8 TOPS as the GEMM target | it is the **raw mma issue rate**, not achievable | CUTLASS peaks at 64.4 TOPS big-square on this part; we are at 91 % of *that* |
| single-stream reduce as a bandwidth probe | 99 GB/s | undersaturated; use a real weight-streaming kernel (173 GB/s) |
| int8 `sum(dtype=int64)` as a read-BW probe | 9.3 GB/s | ALU-bound reduction, not a bandwidth measurement |

## 8. Reproduction

```bash
# Gate 1: correctness vs the HF bf16 reference + graph safety + fp16 health
PYTHONPATH=. python3 scripts/chameleon_orin_check.py \
    --checkpoint /path/to/Chameleon_7B_mGPT \
    --image FlashRT.png --prompt "Describe this image." --steps 16
#   ... --text-only          fast smoke, no image
#   ... --int4 / --int4-down  alternative tiers
#   ... --vq-fp16-argmin      measure VQ index drift instead of avoiding it

# M=1 decode roofline (no checkpoint) — the "do we need a GEMV?" gate
# (roofline probe script: scripts/bench/orin_int8_roofline.py)
python3 scripts/bench/orin_int8_roofline.py --decode

# large-M prefill roofline at the production shape
python3 scripts/bench/orin_int8_roofline.py --M 1032
```

Minimal use:

```python
from flash_rt.frontends.torch.chameleon_rtx_sm87 import ChameleonTorchFrontendRtxSm87
f = ChameleonTorchFrontendRtxSm87("/path/to/Chameleon_7B_mGPT")   # int8+hadamard
f.set_prompt("<image>Describe this image.", images=[pil_img])
print(f.generate(max_new_tokens=32))
```

`load_model(config="chameleon")` deliberately raises with this snippet: it is a
chat VLM (`set_prompt` + `generate`), not the VLA `predict()` surface.

**Build** (the INT8 FHT kernels ride in the existing `ENABLE_SM80_INT8_CUTLASS`
source list, so no CMake change is needed):

```bash
cmake -B build -S . -DGPU_ARCH=87 -DFA2_ARCH_NATIVE_ONLY=ON \
      -DFA2_HDIMS='128;256' -DFA2_DTYPES='fp16;bf16'
cmake --build build --target flash_rt_kernels -j6
```

## 9. Constructor parameters (classified: required / recommended / experimental / informational)

`ChameleonTorchFrontendRtxSm87(checkpoint_dir, **kwargs)`. There is no CLI/server
entry yet (out of scope this round), so these are the frontend kwargs.

**Required**

| param | notes |
|---|---|
| `checkpoint_dir` | path to the Chameleon-7B checkpoint. Backbone dims and the special token ids are **hard-asserted** against its `config.json` |

**Recommended (safe defaults; tune for deployment)**

| param | default | notes |
|---|---|---|
| `max_seq` | 2048 | sizes the KV cache (`2 × 32 × max_seq × 4096 × 2 B` = 2.15 GB at 2048) and the RoPE tables. Hard-checked against `max_position_embeddings=4096`. `S = 1 + n_img*1026 + n_text + 1` |
| `use_hadamard` | **`True`** | the W8A8+QuaRot tier. **Do not disable in production** — plain per-row INT8 reproduces only 8/16 reference tokens (§4.5). Kept switchable for A/B only |
| `split_kv_bias` | `4` | multiplies the `num_sms` passed to FA2 so split-KV actually engages at 32 Q heads (§2.1). `1` disables |
| `ffn_down_clamp` | `60000` | **correctness requirement**, not a knob (§4.6). Env: `FLASHRT_CHAMELEON_DOWN_CLAMP` |
| `ffn_down_clamp_last_n` | `4` | layers to clamp, counted from the end. `32` = all layers (safe but costs 2.2 % of prefill). Env: `FLASHRT_CHAMELEON_DOWN_CLAMP_LAST_N` |
| `vq_argmin_fp32` | `True` | fp32 codebook distance/argmin; costs <0.1 ms and lifts index match 98.1 % → 99.0 % (§2.4) |
| `free_fp16_weights` | `True` | drop the 13 GB fp16 originals after quantization |

**Experimental (off by default)**

| param | default | notes |
|---|---|---|
| `use_int4` | `False` | QuaRot W4A4 on the six K=4096 projections. Wins only at very short ISL; **below `use_hadamard` at production ISL** (§4.5) |
| `use_int4_down` | `False` | additionally int4 the FFN down via block-H128. **Not recommended** — worst tier measured at production ISL (0/16 greedy) |
| `probe_layers` | `None` | list of layer indices to snapshot post-residual hidden states for `snapshot_probe()`; zero cost when `None` |

**Informational**

| param | notes |
|---|---|
| `precision_tier` / `precision_spec()` / `get_model_info()` | report the resolved configuration; `timing` carries `prompt_ms` / `prefill_ms` / `decode_tok_s` |

**Not accepted**: `use_fp8`, `use_fp4`,
`use_fp8_attn`, `use_awq_v_proj`, `num_views`, `action_dim`,
`action_chunk_size`, `state_dim` — SM87 has no FP8/FP4 tensor cores and this is
not a VLA. Unknown kwargs are swallowed by `**_ignored`.

## 10. Tier status (all four verified to run and produce finite logits)

| tier | flag | verdict |
|---|---|---|
| **int8+hadamard** | default | **production** — Gate 1 PASS, 16/16 greedy match, 21.07 tok/s |
| int8 plain | `use_hadamard=False` | works; loses the outlier conditioning (8/16 greedy) — kept for A/B |
| int4 (QuaRot) | `use_int4=True` | works; best at very short ISL, below int8+hadamard at production ISL |
| int4+down | `use_int4_down=True` | works but **not recommended** — worst at production ISL (§4.5) |
