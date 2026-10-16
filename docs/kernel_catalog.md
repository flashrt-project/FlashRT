# FlashRT Kernel Catalog

> **Audience**: you are evaluating whether FlashRT can accelerate
> *your* model (not necessarily a VLA). This doc lists every CUDA
> kernel shipped by FlashRT and the fusion patterns they enable,
> so you can judge re-use against your model's compute graph
> before adopting the framework.
>
> **TL;DR**: FlashRT exports pybind entries across three modules:
> **`flash_rt_kernels`** (~98 entries; memory-bound ops, fp8
> quant/dequant, cuBLASLt wrappers, Thor FMHA — always built),
> **`flash_rt_fp4`** (~23 entries; NVFP4 weight prep and SM120
> block-scaled GEMM wrappers — built on SM100+/SM120),
> and **`flash_rt_fa2`** (2 entries; vendored Flash-Attention 2
> fwd for fp16 + bf16 — RTX/SM80-family only, skipped on Thor).
> Counts are approximate and can shift by a handful between
> releases; this doc lists them by group, not by individual entry.
> Hand-written kernels cover the memory-bound ops (norm / activation
> / fusion / quantize / pointer marshalling); compute-bound ops
> (GEMM, attention) delegate to cuBLASLt, CUTLASS, and the vendored
> FA2.

---

## Why this matters beyond VLA

The same kernel set that powers Pi0 / Pi0.5 / GROOT is a natural fit
for any small-batch transformer-flavored model:

- **Gemma / PaliGemma / Qwen backbones** — re-use the full encoder
  FFN fusion stack (RMSNorm + QKV + RoPE + GQA + gate-geglu + FP8
  GEMM; Qwen3's true-SiLU FFN uses the `silu_mul_split_*` path
  instead of the merged GEGLU kernels).
- **DiT / diffusion transformers** — re-use the MHA attention entry
  (`attention_mha_fp16`), residual+norm fusion, FP8 GEMM, and the
  diffusion-step helper (`gpu_euler_step`).
- **Audio / video diffusion** — internal teams have already
  validated a ~4× speedup on audio/video transformer generators by
  reusing this exact kernel set. The abstraction is "small-batch
  low-latency inference", not "VLA-specific".
- **Anything with CUDA Graph potential** — every kernel here is
  graph-capture-safe: no dynamic allocation, deterministic launch
  footprint, pointer-only interface.

If your model has any of {RMSNorm, SwiGLU/GeGLU, GQA, FP8 tensors,
CUDA Graph, diffusion Euler step} in its forward, you will likely
find fused kernels here that save 30–60% latency vs a naive PyTorch
implementation.

---

## Module layout

```python
from flash_rt import flash_rt_kernels as fvk       # 87 kernels, always built (pybind, PyTorch)
from flash_rt import flash_rt_fa2     as fa2       # 2 entries, RTX only (SM80/86/89/120)
# JAX side (training + inference): consumed via ctypes + jax.ffi
#   <repo>/flash_rt/flash_rt_jax_ffi.so             # 2 XLA FFI handlers, optional
```

`flash_rt_fa2` is skipped on Thor SM110 (Thor uses `fvk.attention_qkv_fp16`
cuBLAS-decomposed attention — hand-tuned for Thor's unified LPDDR).
`flash_rt_jax_ffi.so` is built only when JAX + the XLA FFI header are
importable at configure time; it carries **zero new compute kernels**
— each handler delegates to an existing entry in `flash_rt_kernels`
(see § 11).

---

## 1. Norm (14 kernels)

RMSNorm and LayerNorm variants, parameterized on compute dtype, output
dtype (float vs FP8-quantized), weight presence, and Ada / style
modulation.

| Kernel | Compute in / out | Notes |
|---|---|---|
| `layer_norm` | fp32 | Reference |
| `layer_norm_fp16` | fp16 in/out | |
| `layer_norm_fp8` | fp16 → FP8 | Fused quantize-out; pairs with cuBLASLt FP8 GEMM |
| `layer_norm_no_affine_fp16` | fp16, no γ/β | Used by some Qwen3 blocks |
| `rms_norm` | fp32 | Reference |
| `rms_norm_fp16` | fp16 in/out | |
| `rms_norm_fp8` | fp16 → FP8 | |
| `rms_norm_fp8_fp16` | fp16 → FP8 + fp16 passthrough | Dual-output |
| `rms_norm_fp8_noweight_bf16` / `_fp16` | Normalize without external weight | Paligemma RMS has folded (1+w) into QKV; this variant handles it |
| `rms_norm_inplace` | In-place | |
| `ada_layer_norm_fp16` | fp16 + style tensor | Pi0.5 decoder AdaLN |
| `ada_rms_norm_style` / `_fp8` | AdaRMSNorm with style modulation | Used by DiT blocks |
| `adarms_fp16` | AdaRMSNorm, explicit style weight | |

## 2. Activation (4 kernels)

| Kernel | Op | dtype |
|---|---|---|
| `gelu_inplace` / `_fp16` | GELU | fp32 / fp16 |
| `silu_inplace_fp16` | SiLU | fp16 |
| `relu_inplace_fp16` | ReLU | fp16 |
| `fused_add_silu_bf16` / `_fp16` | `silu(x + bias)` fused | bf16 / fp16 |

## 3. Fusion — composite (17 kernels)

This is where most of FlashRT's latency wins come from. Each kernel
collapses 2–4 PyTorch ops into a single launch, removing memory
round-trips.

| Kernel | What it fuses | Typical site |
|---|---|---|
| `gate_geglu` / `_fp16` | `gelu_tanh(gate) * up` (GEGLU — **not** SwiGLU) | Paligemma / Gemma FFN |
| `gate_geglu_merged` / `_fp16` / `_fp8` / `_fp8_fp16` | Same, but reads from a merged `[gate \| up]` buffer; one load for both halves | When `gate_up_proj` is a single GEMM output |
| `silu_mul_split_fp8_fp16` | SwiGLU from a pre-split buffer, FP8 quantize out | Encoder FFN boundary into FP8 down_proj |
| `geglu_fp8_static_fp16` | GeGLU (`gelu(gate) * up`) + FP8 quantize | Alternative to SwiGLU |
| `gate_mul_residual` | `gate * up + residual` | |
| `gate_res_fp16` | Same, fp16 | |
| `gate_res_adarms_fp8_static_fp16` | `x + gate*up → AdaRMSNorm → FP8` in one launch | Pi0.5 decoder; saves 3 ops |
| `gate_residual_ada_norm_fp8` | Variant | |
| `fused_adarms_fp8_static_fp16` | `AdaRMSNorm → FP8` | |
| `residual_add` / `_fp16` | Plain residual | |
| `residual_add_rms_norm` | `x + residual → RMSNorm` | |
| `residual_add_rms_norm_fp8` / `_fp8_fp16` / `_fp8_noweight_bf16` / `_fp8_noweight_fp16` | Same but FP8-quantized out, with/without explicit weight | The single biggest latency-saving fusion in encoder blocks |
| `plain_res_rms_fp8_fp16` | Non-Ada variant | |
| `plain_rms_fp8_fp16` | RMSNorm only → FP8 | |
| `bias_residual` / `_fp16` | `x + residual + bias` | |

## 4. Quantize (8 kernels)

| Kernel | In → Out | Notes |
|---|---|---|
| `quantize_fp8` | fp32 → FP8 E4M3 | Per-tensor scale (static) |
| `quantize_fp8_device` / `_fp16` | Scale pointer on device | Enables fused pipelines |
| `quantize_fp8_static` / `_fp16` | Scale as host scalar | |
| `cast_fp16_fp8` | fp16 → FP8, no scale | Matches PyTorch `to(torch.float8_e4m3fn)` |
| `quantize_bf16_to_nvfp4` | bf16 → NVFP4 | SM120 only, requires `has_nvfp4() == True` |
| `quantize_bf16_to_nvfp4_swizzled` | Swizzled NVFP4 layout | For CUTLASS block-scaled GEMM input |

## 5. GEMM (3 wrappers + 1 class)

| Symbol | Under the hood | Use when |
|---|---|---|
| `GemmRunner` (class) | cuBLASLt handle + workspace + heuristic cache | You want a persistent handle with heuristic autotune |
| `gmm_fp16` | cuBLAS HGEMM | Small / non-FP8 matmuls (noise/state projection etc.) |
| `fp8_gemm_descale_fp16` / `_bf16out` / `_f32out` | cuBLASLt FP8 GEMM with per-tensor `alpha` descale | Main workhorse for all FP8 projections |

Add-on NVFP4 / block-scaled GEMM lives in the separate
`flash_rt_fp4.so` module (SM120 only).

## 6. Attention (10 entries, fvk + fa2 combined)

### Thor / decomposed path (`flash_rt_kernels`)

| Kernel | Shape it serves | Notes |
|---|---|---|
| `attention_qkv_fp16` | GQA 8Q/1KV, HD=256 | Paligemma encoder/decoder on Thor |
| `attention_qkv_fp16_padded` | Same, S_kv rounded | For odd-sequence prompts |
| `attention_qkv_fp16_state_masked` | Pi0 decoder row-0 sees only `[:state_nk]` | Single-call state-masked cross-attn |
| `attention_mha_fp16` | MHA (H_q == H_kv), arbitrary HD | Used by DiT per-head, or SigLIP loop-per-view |
| `fmha_strided_full` | FA-style strided QKV buffer | SigLIP on Thor |
| `fmha_forward` / `fmha_strided_forward` | Low-level loaders | |
| `load_fmha_library` / `_strided_library` | Dynamic .so load for Thor's FMHA lib | |
| `softmax_fp16` | Standalone softmax | Useful as a building block |

### RTX / Flash-Attention 2 path (`flash_rt_fa2`)

| Kernel | Notes |
|---|---|
| `fa2.fwd_fp16` | FA2 forward, fp16, GQA + cross-attn + splitkv |
| `fa2.fwd_bf16` | FA2 forward, bf16, same feature set |

The RTX backend (`RtxFlashAttnBackend`) auto-dispatches between
`fa2.fwd_fp16` and `fa2.fwd_bf16` based on buffer dtype. The Thor
backend (`ThorFlashAttnBackend`) uses the decomposed path above.

## 7. RoPE / QKV split (6 kernels)

These fuse the projection output → head split → optional RoPE →
optional KV-cache write into a single kernel, saving one memory
traversal per layer.

| Kernel | Action |
|---|---|
| `qkv_split` / `_fp16` | Split a merged `[Q \| K \| V]` buffer into three output pointers |
| `qkv_split_rope` | Same + apply RoPE to Q/K |
| `qkv_split_rope_kvcache_fp16` | Same + write K/V into a layered KV cache slot in one pass |
| `rope_apply` | Standalone RoPE (for cases without qkv_split) |
| `rope_rotate_half_fp16` | Rotate half the last-dim axis; also reusable for DiT cross-attn |

## 8. Vision patch embedding (2 kernels)

| Kernel | Action |
|---|---|
| `patch_im2col` | Image → `(num_patches, patch_dim)` |
| `patch_embed_bias_pos` | Patch → `linear + bias + pos_embed` fused |

## 9. Memory / utility (7 kernels)

| Kernel | Action |
|---|---|
| `add_bias_fp16` | `x += b` broadcast |
| `gpu_cast_fp32_to_fp16` | Element-wise cast |
| `gpu_copy` | Aligned device-to-device memcpy (graph-safe) |
| `gpu_strided_copy_fp16` | Strided copy (for gather/scatter patterns) |
| `gpu_fill_neginf_fp16` | Attention mask init |
| `gpu_repeat_interleave_heads` | GQA K/V broadcast when an attention backend needs full H_q × HD tensors |
| `gpu_euler_step` | `x_{t+1} = x_t + dt * v(x_t, t)` for flow-matching / diffusion decoders |

## 10. Context & capability flags (5)

| Symbol | What it does |
|---|---|
| `FvkContext` (class) | Holds cuBLAS handle + workspace; required by GEMM & attention kernels |
| `GemmRunner` (class) | High-level GEMM dispatch with autotune cache |
| `get_sm_version()` | Returns `(major, minor)` of the current device |
| `has_cutlass_fmha()` | Thor / SM100+ FMHA template availability |
| `has_cutlass_sm100()` | SM100-family CUTLASS FP8 GEMMs built in |
| `has_nvfp4()` | SM120 NVFP4 quant + block-scaled GEMM available (separate `flash_rt_fp4.so`) |

## 11. JAX FFI bindings (2 entries, optional `flash_rt_jax_ffi.so`)

XLA Foreign Function Interface handlers that expose the existing
FP8 GEMM + activation-quantize entries to JAX, used by the
**training** path under [`training/jax/`](../training/jax/).
Both handlers carry **zero new compute logic** — each delegates
to a kernel already documented in §4 / §5 — and accept the
caller's CUDA stream via XLA's `PlatformStream<cudaStream_t>` so
the FP8 path stays inside an XLA GraphTrees capture (mirrors the
PyTorch fix for `cudaErrorInvalidResourceHandle` under
`torch.compile(reduce-overhead)`). Source under
[`csrc/training/jax_ffi/`](../csrc/training/jax_ffi/); built by
CMake gated on `python3 -c "import jax.ffi; print(jax.ffi.include_dir())"`.

| FFI target name | Source file | Delegates to | Use site |
|---|---|---|---|
| `flashvla::quantize_fp8_static` | [`activation_quantize_ffi.cu`](../csrc/training/jax_ffi/activation_quantize_ffi.cu) | `quantize_fp8_static` (§4) — bf16 → FP8 E4M3 byte tensor with pre-computed device scale | JAX-side activation quantize before each FP8 GEMM in [`training/jax/fp8/fp8_jax.py`](../training/jax/fp8/fp8_jax.py) |
| `flashvla::fp8_gemm_bf16_out` | [`fp8_gemm_ffi.cu`](../csrc/training/jax_ffi/fp8_gemm_ffi.cu) | `GemmRunner::fp8_nn_dev` (§5) — `D_bf16 = scale_a · scale_b · A_fp8 @ B_fp8` (no transpose, K-N layout) | JAX-side FP8 LoRA GEMM (training) and openpi LoRA Einsum / FeedForward patch in [`training/jax/fp8/lora_patch.py`](../training/jax/fp8/lora_patch.py) |

**Why no new compute kernels:** the FP8 cuBLASLt path
(`cublasLtMatmul` with `CUDA_R_8F_E4M3` operands +
device-scale-pointer descale) was already there for PyTorch
inference + training; the JAX path's only requirement was a
non-pybind binding XLA can call. The performance characteristics
are identical to the PyTorch FP8 path (the underlying kernel is
the same one); see
[`training/jax/README.md` § Per-shape FP8 forward microbench](../training/jax/README.md)
for measured numbers on RTX 5090 SM120.

## 12. SM120 small-row NVFP4 and fused prefill families (`GPU_ARCH=120/121`)

Built into `flash_rt_kernels` on SM120 builds only; the bindings are compiled
under the same gate, so other architectures do not expose them and the
frontends keep their FP8 paths there.

| Group | Symbols | What it is |
|---|---|---|
| Skinny W4A4 GEMM | `nvfp4_skinny_gemm`, `nvfp4_skinny_gemm_bf16_act`, `nvfp4_skinny_pack_weight`, `nvfp4_skinny_quantize_weight`, `nvfp4_skinny_config_*` | `mma.sync` block-scaled e2m1 GEMM for <= 48 rows per CTA (action decoders, DiT heads); weights prefetched into registers before the programmatic-dependent-launch wait; FP32 K-split partials |
| Skinny consumers | `nvfp4_skinny_layer_norm`, `nvfp4_skinny_sum_bias_residual_norm`, `nvfp4_skinny_sum_bias_gelu`, `nvfp4_skinny_partials_attention` | Sum the partials and run the next step (bias, residual, LayerNorm / AdaLN, GELU, attention) while emitting the next NVFP4 operand |
| Prefill NVFP4 | `nvfp4_norm_quant`, `nvfp4_quant`, `nvfp4_qk_norm_rope`, `nvfp4_scatter_add_rows`, `nvfp4_sf_bytes` | Norm / quantize producers writing e2m1 + swizzled UE4M3 scales for the CUTLASS block-scaled GEMMs |
| CUTLASS NVFP4 epilogues | `nvfp4_gemm_bias_res_bf16out`, `nvfp4_gemm_bias_gelu_fp4out`, `nvfp4_gemm_geglu_fold_fp4out` | W4A4 GEMMs with bias + residual (BF16 out) or GELU / GeGLU straight to the next NVFP4 operand |
| FP8 prefill (FP16 stream) | `fp8_residual_norm_quant`, `fp8_qkv_split`, `fp8_bias_gelu_quant` | Residual + norm -> FP8, Q/K/V split with bias / q-k RMSNorm / RoPE, bias + GELU -> FP8 around per-tensor FP8 GEMMs |
| Prefill attention | `fused_qkv_attention` | Non-causal FP16 / BF16 attention reading Q/K/V straight from projection rows (bias / RoPE on load), head_dim 64 / 72 |

Pi0.5's decoder-specific consumers (`pi05_dec_fp4_*`) build only with
`-DFLASHRT_ENABLE_PI05_NVFP4=ON`.

---

## Re-use decision tree

When evaluating FlashRT for a new model, walk this checklist:

1. **Does your model use RMSNorm + SwiGLU/GeGLU + GQA?**
   → `residual_add_rms_norm_fp8_noweight_fp16` + `qkv_split_rope_kvcache_fp16`
   + `gate_geglu_merged_fp8_fp16` cover the Gemma FFN pattern (for true-SiLU Qwen3, use `silu_mul_split_fp8_fp16`)
   end-to-end.
2. **Is your attention standard MHA / GQA / cross-attn?**
   → `fa2.fwd_fp16` / `fa2.fwd_bf16` on RTX, `fvk.attention_qkv_fp16`
   on Thor. Batch=1, arbitrary seqlen, HD ≤ 256.
3. **Do you need FP8 anywhere?**
   → `fp8_gemm_descale_*` (cuBLASLt) for the matmul + the full
   calibrate-once-use-forever machinery in `flash_rt.core.quant`.
4. **Is your decoder a flow-matching / diffusion loop?**
   → `gpu_euler_step` + CUDA Graph capture of a single denoise step
   replayed N times. Pi0 / Pi0.5 / GROOT DiT all work this way; your
   generative model likely does too.
5. **Is attention-masking the blocker?** (state token, causal, window)
   → `attention_qkv_fp16_state_masked` for row-0 masks; FA2 also
   supports causal + `attn_mask` via the pybind args.
6. **Are you on JAX (training or inference)?**
   → Same FP8 GEMM + activation-quantize, exposed as XLA FFI
   handlers in `flash_rt_jax_ffi.so` (§ 11). The training path
   (FP8 + LoRA + RECAP) is documented in
   [`training/jax/README.md`](../training/jax/README.md); the
   inference path uses
   [`flash_rt.frontends.jax.pi05_rtx`](../flash_rt/frontends/jax/pi05_rtx.py),
   which inherits the CFG pipeline machinery from the PyTorch
   frontend so RL inference works on JAX-loaded checkpoints
   without extra wiring.

If the answer to ≥2 of these is yes, FlashRT saves you the 3-6
months of hand-written CUDA + cuBLASLt wrangling it took to build
this. See [`adding_new_model.md`](adding_new_model.md) for the
step-by-step model integration guide and
[`plugin_model_template.md`](plugin_model_template.md) for an
out-of-tree plugin example.

---

## Not yet supported

Shape features that currently have no fused kernels in FlashRT:

- **MLA (DeepSeek-style)**. No GQA alternative that splits
  latent K/V differently. Would need a dedicated `attention_mla_*`
  family.
- **Sliding-window attention** (SWA). The vendored FA2 has the
  feature flag disabled; enabling it is ~50 LoC but the path has
  not been exercised end-to-end.
- **Tree-attention / speculative decoding**. Out of scope —
  FlashRT targets batch=1, not batch-N speculation.
- **RWKV / Mamba / state-space models**. Different primitive set
  entirely; no overlap with the kernel library here.
- **8-bit weight + 8-bit activation in pure INT8** (BitsAndBytes
  style). The FP8 path is E4M3 with per-tensor scale; INT8
  symmetric would need new GEMM bindings.

File a request (or a PR adding the instantiation) if your model
needs one of the above.
