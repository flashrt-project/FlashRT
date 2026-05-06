# Qwen3-8B-Instruct-2512-SFT-NVFP4 — HF Baseline Reference

> Branch: `feat/qwen3-8b-nvfp4` (cut from `main` @ `ed7b69e`).
> Captured: 2026-05-06, RTX 5090 (sm_120, 32 GB), driver 580.82.07.
> Container: `pi0-stablehlo-test` (image `pi0-fp8-x86`).
> Venv: `/opt/venv_qwen3_8b` (torch 2.9 / CUDA 13 / transformers 5.3 /
> compressed-tensors 0.12.2 / accelerate 1.13).
>
> Capture script: `internal-tests/qwen3_8b_baseline_hf.py` (gitignored
> dev-local; reruns are reproducible).
> Frozen fixtures (host-mounted, persistent, gitignored npy/npz):
> `~/suliang/PI/checkpoints/qwen3_8b_baseline/` →
> `/workspace/PI/checkpoints/qwen3_8b_baseline/` inside the container.

This doc is the correctness + perf oracle every later FlashRT-side
optimization is gated against. **Do not regenerate without flagging
that the previous oracle is being replaced.**

---

## 1. Checkpoint identity

Source: `JunHowie/Qwen3-8B-Instruct-2512-SFT-NVFP4` (modelscope mirror).

```
arch                  : Qwen3ForCausalLM (pure dense, NO mixed lin-attn, NO MTP)
num_hidden_layers     : 36     (all full_attention)
hidden_size           : 4096
num_attention_heads   : 32     (Q heads)
num_key_value_heads   : 8      (GQA 4:1)
head_dim              : 128
intermediate_size     : 12288
vocab_size            : 151_936
hidden_act            : silu   (SwiGLU = silu(gate)*up)
rms_norm_eps          : 1e-6
rope_parameters       : {rope_theta: 1_000_000, rope_type: "default"}
max_position_embeds   : 40_960
tie_word_embeddings   : False
```

### Quantization (compressed-tensors `nvfp4-pack-quantized`)

```
weights:           NVFP4 group_size=16, symmetric, static
input_activations: NVFP4 group_size=16, symmetric, dynamic="local"
ignore:            ["lm_head"]   (lm_head stays BF16)
```

### Per-linear ckpt schema (verified layer 0)

| Field | dtype | shape (q_proj example) | role |
|---|---|---|---|
| `weight_packed`        | u8         | `(out, in/2)`     | NVFP4 e2m1, 2 elts/byte |
| `weight_scale`         | fp8_e4m3   | `(out, in/16)`    | per-block-16 weight SF (linear layout) |
| `weight_global_scale`  | fp32       | `(1,)`            | GS_w (`448 / amax`) |
| `input_global_scale`   | fp32       | `(1,)`            | **GS_a precomputed** — bake into GEMM `alpha` |

**Note on `input_global_scale`**: Qwen3.6-NVFP4 (the 27B path) does
**not** have this — there activations are dequantized BF16 throughout.
Here, activations are quantized at runtime to NVFP4 with the per-block
SF computed dynamically from the input batch, while the global activation
scale is pre-computed during the SFT/calibration of the ckpt and shipped
as `input_global_scale`. So at runtime the FlashRT path:

1. Computes per-block-16 act SF on-the-fly from BF16 input
   (`quantize_bf16_to_nvfp4_swizzled` already does this).
2. Bakes `alpha = 1 / (input_global_scale * weight_global_scale)`
   into the `fp4_w4a16_gemm_sm120_bf16out` `alpha` parameter.

This is a **frontend wiring detail, NOT a kernel change** — the
existing FP4 GEMM signature already accepts `alpha`, so no kernel work.

### Confirmed weight shapes (layer 0)

```
self_attn.q_proj.weight_packed     (4096,  2048)  u8        Q proj OUT=4096 (32 heads × 128)
self_attn.k_proj.weight_packed     (1024,  2048)  u8        K proj OUT=1024 (8 heads × 128)
self_attn.v_proj.weight_packed     (1024,  2048)  u8        V proj OUT=1024
self_attn.o_proj.weight_packed     (4096,  2048)  u8        O proj OUT=4096
mlp.gate_proj.weight_packed        (12288, 2048)  u8
mlp.up_proj.weight_packed          (12288, 2048)  u8
mlp.down_proj.weight_packed        (4096,  6144)  u8        IN=12288 → packed/2
self_attn.q_norm.weight            (128,)         bf16      per-head RMSNorm
self_attn.k_norm.weight            (128,)         bf16
input_layernorm.weight             (4096,)        bf16
post_attention_layernorm.weight    (4096,)        bf16

embed_tokens.weight                (151936, 4096) bf16
model.norm.weight                  (4096,)        bf16
lm_head.weight                     (151936, 4096) bf16      ← UNQUANTIZED (in ignore list)
```

### Tokenizer + tools template

`Qwen2TokenizerFast` (vocab 151,936). `apply_chat_template(tools=[...],
add_generation_prompt=True, tokenize=False)` rendered without error
for `{type:"function", function:{name:"get_weather", ...}}`-shaped
tools. The chat template emits `<|im_start|>assistant\n` after the
tool block — first-class native tool-calling support is in the ckpt.

---

## 2. Correctness oracle (frozen fixtures)

```
fixture_token_ids.npy        int64  (32,)         greedy 32-tok ids
fixture_first_logits.npy     fp32   (151936,)     logits at last prompt pos
                                                  (no-cache forward,
                                                   prompt = §3 prompt)
fixture_layer_hidden.npz     fp32   37 arrays     embed + 36 layer outputs
                                                  at last prompt pos
                                                  each (4096,)
```

Greedy 32-tok decode (prompt = §3):

```
ids[0:8] = [55313, 1197, 524, 986, 374, 264, 24844, 304]
text     = " Quantum entanglement is a phenomenon in which two or more
            particles become correlated in such a way that the state
            of one particle is instantly connected to the state of"
```

First-pass last-token logits argmax: `id=55313` ` Quantum` (i.e. the
greedy-decode invariant is "next token after the prompt is ' Quantum'").

Layer-0 hidden state norm at last prompt pos: `‖h‖₂ ≈ 11.53`.
Final hidden norm at the same pos: shows monotonic-ish growth across
the 36 layers (saved fully in `fixture_layer_hidden.npz`).

These are the gates G1/G2/G3 (single-layer cosine, full-logits
cosine, greedy 32/32 token match — see PLAN §7).

---

## 3. Performance baseline

Single prompt for decode bench (used in §6.1 throughout):

```
"Explain quantum entanglement in one short paragraph."   (11 tokens)
```

### TTFT vs prompt_len (HF SDPA + compressed-tensors NVFP4 dequant-in-loop)

| prompt_len | TTFT min (ms) | per-token (ms) |
|---:|---:|---:|
|   64 |  279.8 | 4.37 |
|  256 |  294.7 | 1.15 |
|  512 |  315.3 | 0.62 |
| 1024 |  366.4 | 0.36 |

The per-token rate drops sharply past 64 because the NVFP4 dequant cost
is amortized over more tokens; at S=1024 the kernel is closer to its
BW-bound ceiling. The flat ~280-370ms band tells us **the
compressed-tensors runtime is already non-trivial overhead at S=64**.

### Decode tok/s (greedy, batch=1, KV cache on, prompt = 11 tokens)

| max_new | wall (s) | end-to-end tok/s | TTFT (ms) | TPOT (ms/tok) | decode-only tok/s |
|---:|---:|---:|---:|---:|---:|
|  64 | 17.70 | 3.6 | 281 | 276.4 | 3.6 |
| 128 | 35.40 | 3.6 | 281 | 276.5 | 3.6 |
| 160 | 44.26 | 3.6 | 281 | 276.6 | 3.6 |

**Decode is heinously slow** — 3.6 tok/s, 276 ms/tok. Cause:
compressed-tensors path performs full NVFP4 → BF16 weight dequant per
forward step (no fused FP4 GEMM, no graph capture, no KV-aware path).
This is exactly the headroom FlashRT has to exploit.

Memory: peak VRAM allocated = **5.99 GiB** (matches expected ~4 GiB
NVFP4 weights + scratch + KV + SDPA workspace).

---

## 4. Headroom = what FlashRT must beat

The PLAN's gate targets compared to this baseline:

| Metric | HF baseline | FlashRT target (gate G5/G6) | Speedup |
|---|---:|---:|---:|
| TTFT @ P=64    | 280 ms | <  30 ms |  **9.3×** |
| TTFT @ P=256   | 295 ms | <  70 ms |  **4.2×** |
| TTFT @ P=512   | 315 ms | < 130 ms |  **2.4×** |
| TTFT @ P=1024  | 366 ms | < 240 ms |  **1.5×** |
| decode tok/s   | 3.6    |  ≥ 130   | **36×**   |
| VRAM peak      | 5.99 GiB | < 14 GiB at P=1024+N=256 | (constraint) |

The decode delta (36×) is the headline number — driven entirely by
graph-captured single-token decode + native NVFP4 GEMM (no per-step
dequant). The TTFT delta tightens at large prompts because at S=1024
even HF's SDPA path starts approaching the FA2 wall.

---

## 5. What this baseline does NOT cover (deliberate)

- **No FA2 baseline** — HF baseline uses SDPA. FA2 in HF 5.x was
  attempted; SDPA is more cross-version-stable for archival numbers.
  FlashRT will use the vendored `flash_rt_fa2.fwd_bf16` path; the
  comparison there is internal (cosine vs HF SDPA).
- **No long-prompt (≥4K)** — gate G9 (1024+256 in <14 GiB) is enough
  to validate VRAM budget; longer-prompt characterization will live
  in the close report once a FlashRT path is actually competitive.
- **No tool-call output capture** — the chat-template renders
  correctly, but the actual JSON-emission behavior of greedy decode
  on a `tools=[...]` prompt belongs in gate G8 once the FlashRT
  decode path is up; until then it's unfalsifiable.

End of baseline reference.
