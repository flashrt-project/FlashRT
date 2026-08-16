# LTX-2.5 on the structures layer — design

Scope: promote the LTX-2.5 integration (attention swap, W4A4 FFN chain,
quantize-on-adopt) from model-private modules into catalog structures, so the
same regions attach to any host that binds them — the native frontend here,
and diffusers-hosted checkpoints through a binding, with no model-specific
code in any impl.

Everything below follows `docs/structures.md`: specs name positions, bindings
place them on a host, impls decide what runs there. Nothing in this design
adds a second vocabulary for calibration or qualification.

## 1. What generalizes, and to which structure

| Region (measured on LTX-2.5) | Catalog structure | Status |
|---|---|---|
| unmasked self/cross attention, head_dim 128 | `attention_core` | new backend impl |
| GELU FFN pair (proj → tanh-GELU → down) | `vision_ffn` | new backend impl |
| per-linear NVFP4 weight adoption | `quantize_on_adopt` | new scheme + binding attribute |
| adaLN (rms · (1+scale) + shift, per-token tables) | host stage | future `adaln_producer` backend |
| q/k RMSNorm + RoPE | host stage | future `qk_norm_rope` backend |
| two-stage denoise pipeline | `video_generation_pipeline` | new pipeline binding |

The audio branch (head_dim 64, short sequences) stays on the host attention
path by measurement: quantized attention loses to SDPA at those shapes, so
the binding simply does not claim those sites.

## 2. `attention_core` backend: `sage2_qk_int8_pv_fp8`

Executable form: per-warp INT8 quantization of Q, per-block INT8 of K,
per-channel FP8 of V, one fused attention kernel, bf16 out. The kernel and
its quantizers ship as one Hub artifact; the impl reads the supported head
dims and layouts from the artifact instead of duplicating capability
knowledge, exactly as the FA2 backend does.

Qualification (all decided from real captures, refusal is legible):

- head_dim must be advertised by the artifact (128 today); other dims return
  no binding so the host keeps its own attention,
- masked sites are not claimed — a mask that packs to a dense run can ride
  the existing packed-KV plan later; today the masked path stays host,
- scratch (int8/fp8 staging + output) is allocated per shape and shared
  across all same-shaped sites; call sequences are pointer-stable, so the
  region is CUDA-graph capturable.

Parity gate: the spec's `real_distribution` rule. Measured on the target
model this backend holds ~0.9992 cosine per call against an fp32 reference,
and matched-input single-forward parity sits inside the noise floor of any
same-precision kernel substitution; the latency rule is satisfied with
2.0-2.4x over the strongest SDPA backend at the model's sequence lengths.

## 3. `vision_ffn` backend: `w4a4_nvfp4_cutlass`

Executable form, three launches replacing six:

    activation quantize (bf16 -> NVFP4 + block scales)
    up GEMM with bias + tanh-GELU + NVFP4 output epilogue
    down GEMM (bf16 out; bias added when the slot carries one)

Weight slots come from the spec; the impl accepts either origin:

- **prequantized hosts** (checkpoint ships NVFP4): dequantize with the
  reference kernel, requantize into the executable layout at adopt,
- **bf16 hosts**: direct quantize at adopt (~seconds for a 22B model).

Qualification:

- both dims divisible by 16; rows padded to 128 through a staging buffer when
  the host batches oddly — the GEMM rejects unaligned M *without writing
  output*, so the impl owns the pad rather than trusting a return code,
- adopt is layer-by-layer so peak memory stays near the fp4 footprint,
- parity is gated against the plain-torch reference on real captures; on the
  target model the chain holds the same distance from a bf16 golden as the
  host's own W4A4 path while being 1.25-1.3x faster.

## 4. `quantize_on_adopt`: the site list is a binding attribute

The measured result that shapes this design: blanket adoption of every large
linear visibly damages output, while adopting exactly the checkpoint
author's calibrated selection (per-block attention/FFN linears, minus the
final blocks; never adaLN producers, connectors, or patch/readout
projections) matches bf16 quality. That selection is knowledge about the
*host*, not about any impl — so it lives in the host binding as an explicit
site list, and the scheme refuses to adopt outside it unless the caller
overrides deliberately. A prequantized checkpoint is itself the receipt for
that list.

## 5. Pipeline binding

`video_generation_pipeline`, same family as the existing video hosts:
condition encoding (text tower + connector stack, slower cadence, embeddings
cacheable per prompt), latent preparation, the fixed-step denoise loop, an
optional latent upsample stage, and VAE decode. Hot-path segments classify
per the coverage contract; attention and FFN regions point at the structures
above, adaLN/RoPE stay declared host stages until their structures land, and
the denoise loop is the graph-capture boundary.

Two facts from bring-up that the binding must carry as attributes rather
than rediscover:

- the distilled checkpoint generation wants single-pass denoising — guidance
  and modality-isolation scales at 1.0 — and defaults that re-enable extra
  passes triple the step cost silently,
- with the transformer resident, decode tiling must be budgeted against the
  memory decode will actually see, not a pre-build snapshot.

## 6. Measured context (RTX 5090, 1536x1024x121f unless noted)

- native frontend: denoise 23.9s -> 11.7s (2.04x) with attention + FFN +
  compile + whole-loop capture; per-step 1068.6 -> 491.6ms (stage 1),
  5111.7 -> 2596ms (stage 2)
- diffusers-hosted bf16 checkpoint, single-pass distilled schedule:
  254 -> 54s end-to-end (3.4x; per-step 3.85x) with adopt + attention swap +
  per-block compile, quality matched to the bf16 baseline by frame
  inspection; at 768x512x49f the gap to the offload baseline is >10x
- adopt cost: ~6s for 1176 linears of a 22B transformer

## 7. Sequencing

1. attention backend impl + gate records
2. vision_ffn backend impl + gate records
3. quantize_on_adopt site-list attribute + host binding
4. pipeline binding with coverage classification
5. adaLN / qk-norm-RoPE structures (removes the two biggest remaining host
   stages; profiled at ~35% of a denoise step on the diffusers host)
