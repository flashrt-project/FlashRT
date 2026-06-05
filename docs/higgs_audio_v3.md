# Higgs Audio v3 TTS-4B on FlashRT (RTX 5090 / SM120)

> Single-stream, zero-shot text-to-speech: an **FP8 W8A8 Qwen3-4B** backbone
> drives a fused 8-codebook head under a delay pattern, decoded autoregressively
> and synthesised by the bundled neural codec — **text → 24 kHz waveform in one
> process, no server required.** Per-frame decode is fully kernelised (no torch
> in the math path) behind a clean `generate(text) -> waveform` API.

Model: [`bosonai/higgs-audio-v3-tts-4b`](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
— a dense Qwen3-4B backbone (36 layers, hidden 2560, GQA 32q/8kv, head_dim 128,
SwiGLU, RoPE θ=1e6) + a fused multi-codebook acoustic head (8 codebooks × 1026)
+ a DAC-style convolutional codec (25 Hz / 24 kHz, delay pattern). Research /
non-commercial license — check the model card.

---

## Performance

Measured on a single **RTX 5090** (SM120), single stream, warm (numbers vary
with text and clocks). Two precisions, **both SM120-only** (the GEMV kernels are
`sm_120a`): **FP8 W8A8** (default) and **BF16 W16A16** (`fp8=False`) — the
unquantised precision path for the same hardware (see below):

| Metric | **FP8** (default) | **BF16** (`fp8=False`) |
|---|---|---|
| Real-time factor (RTF) | **0.095 – 0.11** (≈ 9–10× real time) | **0.15** (≈ 6.5× real time) |
| Time to first audio (TTFA) | **≈ 94 ms** | **≈ 138 ms** |
| Autoregressive decode | **≈ 3.2 ms/frame** | **≈ 6.1 ms/frame** (at the BF16 bandwidth wall) |
| Prompt prefill | **≈ 1.0 ms/token** | **≈ 0.9 ms/token** (cuBLAS, weight-once) |
| Peak VRAM | **6.6 GB** | **9.6 GB** |
| Fidelity | teacher-forced logits **cos 1.0**; codec **cos 0.99993**; streamed == one-shot **cos 1.0** | same (bit-exact vs eager) |
| Prefix reuse | shared `system` preamble cuts prefill **~64 %**, output bit-identical | same |

Both precisions run the **same** fully-kernelised, zero-torch decode path
(position-agnostic CUDA graph, batched prefill, prefix reuse). BF16 reads 2× the
weight bytes of FP8, so its per-frame is ~1.7× — the bandwidth floor, not
overhead. **Want unquantised BF16 numerics (e.g. to skip quantisation)?
construct with `fp8=False` or pass `--bf16`** — everything else is identical.

**Architecture**: as shipped both paths are built for **SM120** (`-gencode
sm_120a`), so a default build runs on SM120 only. The BF16 decode kernels use no
SM120-specific instruction (the M=1 GEMV is plain CUDA, the rest is shared
kernels) and compile for Ada (sm89) — a 4090 BF16 build is a CMake-gating change
away, but is not built or validated here. The FP8 path additionally needs the
SM120 FP8 GEMM, so it stays SM120-only. See [§8](#8-notes--limitations).

**Speedup over the unoptimised PyTorch reference** (same model + GPU, transformers
eager backbone):

| Stage | PyTorch eager | FlashRT FP8 | |
|---|---|---|---|
| Autoregressive decode (no codec) | 10.8 ms/frame | **3.2 ms/frame** | **3.3× faster** |
| Prompt prefill | 3.7 ms/token | **1.0 ms/token** | **3.7× faster** |
| Backbone weight VRAM | 7.3 GB (bf16) | **3.6 GB (FP8)** | **2× smaller** |

The decode math path is fully kernelised (RMSNorm + quant, dedicated M=1 GEMV
with fused residual, fused q/k-norm+RoPE+KV-write, FlashAttention-2) and replayed
from a single position-agnostic CUDA graph. The prompt is prefilled in one
batched M=P pass; a shared `system` preamble's KV is reused across requests (only
the new text is prefilled — see [§4](#4-python-api)). The dedicated M=1 GEMV runs
at **86 % HBM bandwidth** in both precisions; per-frame is bandwidth-bound, so
the only lever below the BF16 floor is the FP8 quantisation (half the bytes).

---

## 1. Requirements

| | |
|---|---|
| **GPU** | RTX 5090 (SM120). Other SM120 Blackwell parts should work; the FP8 GEMV/GEMM kernels are `sm_120a`. |
| **FlashRT** | Built with `GPU_ARCH=120` — see [Build & install](../README.md#build--install) (`cmake .. && make -j` produces `flash_rt/flash_rt_kernels*.so` and `flash_rt/flash_rt_fa2.so`). |
| **Python** | 3.12, CUDA 13, torch ≥ 2.9. |
| **Packages** | `transformers` (≥ 4.53; tokenizer + codec config classes), `safetensors`, `numpy`. `torchaudio` is **optional** (only the codec *encode* path uses it; decode does not — it is auto-stubbed if absent). |

> **The `kernels` package.** Some `transformers` installs pull the optional
> `kernels` accelerator, whose `huggingface_hub` strict-dataclass usage can
> raise on import. The codec never needs it; FlashRT neutralises it at import
> time (`flash_rt/models/higgs_audio_v3/_codec/env_guard.py`), so **no manual
> step is required**. If you prefer, `pip uninstall kernels` has the same effect.

---

## 2. Get the checkpoint

Download the checkpoint to a directory of your choice and point an environment
variable at it (used by the quickstart and the examples below):

```bash
huggingface-cli download bosonai/higgs-audio-v3-tts-4b --local-dir /path/to/higgs-audio-v3-tts-4b
export HIGGS_CHECKPOINT=/path/to/higgs-audio-v3-tts-4b
```

The directory must contain `config.json`, `model.safetensors`, and
`tokenizer.json`. The codec weights are bundled inside `model.safetensors`
(prefix `tied.embedding.modality_embeddings.0.model.`) — no separate download.

---

## 3. Quickstart

```bash
python examples/higgs_audio_v3_quickstart.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --out fox.wav --benchmark 3
```

Expected output on a 5090 (numbers vary with text length and clocks):

```
[FP8 W8A8] 'The quick brown fox jumps over the lazy dog.'
  -> fox.wav  (~3.0s audio, ~4.5s wall incl 1st-call setup)
  bench 1: AR decode ~290 ms (~3.8 ms/frame)
  bench 2: AR decode ~290 ms (~3.8 ms/frame)
  ...
```

First call pays a one-time cost: FP8 activation-scale **calibration** (a short
BF16 free-run) and **codec load**. Subsequent calls are warm. Add `--bf16` to
run the BF16 backbone instead of FP8.

---

## 4. Python API

```python
from flash_rt.frontends.torch.higgs_audio_v3_rtx import HiggsAudioV3TorchFrontendRtx

fe = HiggsAudioV3TorchFrontendRtx(CHECKPOINT_DIR, fp8=True)   # fp8=False -> BF16 backbone

wav = fe.generate("Hello from FlashRT.")     # text -> 24 kHz mono waveform [L] (cpu f32)

# or split the stages:
codes = fe.predict("Hello from FlashRT.")    # [T, 8] acoustic codes (int64, cpu)
wav   = fe.synthesize(codes)                 # codes -> waveform

# streaming (low TTFA): yields 24 kHz chunks as frames decode
for chunk in fe.generate_stream("A longer sentence to speak aloud."):
    ...                                      # chunk: waveform [n] (cpu f32)
```

Save with any WAV writer (the quickstart uses the stdlib `wave` module at 24 kHz).

**Shared preamble + prefix reuse.** Pass a `system` preamble (a fixed
voice/style instruction) to reuse its KV across requests — when successive calls
share the same preamble, only the new text is prefilled, cutting prefill cost
while producing bit-identical audio:

```python
SYSTEM = "Narrate in a calm, warm storyteller voice with clear diction."
for line in lines:                           # same voice, many utterances
    for chunk in fe.generate_stream(line, system=SYSTEM):
        ...                                  # preamble KV reused after the 1st call
```

The reuse is a frontend mechanism (the KV cache is owned by the frontend); the
caching/eviction policy belongs in the serving layer
(`serving/higgs_audio_agent` forwards the request `instructions` as `system`).

---

## 5. What runs under the hood

Per acoustic frame, the FP8 decode step is fully kernelised — **no torch in the
math path**:

```
rms_norm_fp8 (norm + quant)
  -> M=1 FP8 GEMV  (qkv)                    # warp-per-output-row, no MMA padding tax
  -> fused q/k-norm + RoPE  -> FA2
  -> quantize_fp8_static (attn-out)
  -> M=1 FP8 GEMV  (o_proj, fused residual epilogue: h += o)
  -> rms_norm_fp8  -> GEMV (gate/up) -> silu_mul -> quantize_fp8_static
  -> M=1 FP8 GEMV  (down_proj, fused residual epilogue: h += down)
  -> rms_norm + quantize_fp8_static -> GEMV (fused 8-codebook head)
```

The M=1 GEMV (`csrc/gemm/fp8_gemv_m1_sm120.cu`) is the key kernel: the hand-tuned
MMA GEMMs pad M=1 to BLOCK_M=16 and starve the SMs on the N=2560 projections;
the GEMV assigns one warp per output row and folds the residual add into the
epilogue. Greedy generation applies the delay pattern (BOC/EOC) and un-delays
the codes before the codec.

---

## 6. Faithfulness & validation

| check | metric | result |
|---|---|---|
| FP8 backbone vs eager BF16 | teacher-forced logits cosine | **1.0** |
| codec (authoritative codes → wave) vs reference | waveform cosine | **0.99993** |

**On free-run vs other implementations.** Greedy decoding over discrete audio
codes is numerically chaotic: even teacher-forced, two faithful BF16
implementations agree on only ~84–92 % of tokens (codebook logits ≈ 86, bf16
ULP ≈ 0.5 — near-ties resolve differently). In free-run, the first near-tie
difference feeds back and compounds, so FlashRT, the BF16 reference, and the
upstream engine each produce a **different but valid** realisation of the same
text (frame 0 agrees; full divergence by frame ~2). This is intrinsic to the
task, **not** a quantisation error — faithfulness is established by the
teacher-forced cosine above, and the codec is bit-faithful on identical codes.

---

## 7. Measurement notes

Headline numbers are in [Performance](#performance) above. Methodology:

- **Full pipeline, warm.** RTF / TTFA are end-to-end text→waveform through the
  standardized `generate` / `generate_stream` frontend, FP8 backbone, after a
  warm-up call (lazy FP8 calibration + codec load + CUDA-graph capture happen
  once on the first call and are excluded).
- **Decode floor 3.2 ms/frame** is the clean single-position CUDA-graph replay;
  the per-token GEMMs read 3.6 GB of distinct FP8 weights, so this is genuinely
  HBM-bound (micro-benchmarks that reuse weights report L2-cached fiction). The
  full-pipeline per-frame is slightly higher because attention cost grows with KV
  length over a long generation.
- **Prefill** is one batched M=P forward (≈ 1 ms/token); a shared `system`
  preamble reuses its resident KV across requests (only the new text suffix is
  prefilled — bit-identical to a cold prefill).
- **Codec** runs in fp32 (ConvTranspose is unstable in low precision) as one
  small pass at the end (≤ 50 ms for 40 s of audio); in streaming it is decoded
  in overlapping windows so the streamed waveform matches the one-shot output.
- **VRAM** is the peak process working set, not a reservation: FP8 backbone
  weights ≈ 3.6 GB (bf16 would be 7.3 GB) + bf16 embed/head + fp32 codec.

---

## 8. Notes & limitations

- **FP8 calibration** is per-tensor static (activation `amax/448`), measured
  once from a short BF16 free-run of the first prompt and reused. Activation
  ranges are stable across prompts; re-instantiate the frontend to recalibrate.
- The BF16 projection weights are freed after calibration (the FP8 backbone is
  the active path); pass `fp8=False` for the BF16 backbone, which keeps them.
- **GPU support.** A default build targets **SM120** (`GPU_ARCH=120`,
  `-gencode sm_120a`), so the frontend's decode kernels are only compiled there;
  a non-SM120 build (e.g. `GPU_ARCH=89` for a 4090) builds the rest of FlashRT
  but **not** these kernels, so the frontend raises at runtime (the `ht_gemv_*`
  symbols are absent). The BF16 GEMV itself is plain CUDA (no SM120 intrinsics)
  and *compiles* for Ada (sm89) — and all the support kernels (RMSNorm, SwiGLU,
  q/k-norm+RoPE, FlashAttention-2) are arch-generic — so a 4090 **BF16** build is
  a build-gating change away, but it is not enabled or hardware-validated here.
  The FP8 path additionally needs the SM120 FP8 GEMM and stays SM120-only.
- Synthesis here is **non-streaming** (the codec decodes the whole clip at the
  end). Streaming / chunked synthesis is not yet wired.
- Codec source: the `bosonai/higgs-audio` v2 tokenizer (decode path only),
  vendored under `flash_rt/models/higgs_audio_v3/_codec/`.
