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
  -> fox.wav  (2.76s audio, ~4.5s wall incl 1st-call setup)
  bench 1: AR decode ~334 ms (~4.8 ms/frame)
  bench 2: AR decode ~333 ms (~4.8 ms/frame)
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
```

Save with any WAV writer (the quickstart uses the stdlib `wave` module at 24 kHz).

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

## 7. Measured comparison (RTX 5090, single stream, greedy)

Full text→waveform pipeline. FlashRT numbers are the standardized frontend
(`generate`), warm, FP8 backbone:

| | per-frame | RTF | first-token / TTFA | VRAM | precision |
|---|---|---|---|---|---|
| PyTorch eager (transformers Qwen3, AR backbone only) | 10.8 ms | 0.27 | — | 9.0 GB | bf16 |
| sglang-omni (full pipeline) | ~6.4 ms | 0.16–0.19 | 0.36–0.63 s | 28.3 GB¹ | bf16 |
| **FlashRT (this frontend, FP8 + codec)** | **4.6–5.1 ms** | **0.12** | **~40 ms²** | **6.3 GB** | FP8 W8A8 |

¹ sglang reserves ~85 % of the GPU for its KV pool (`mem_fraction_static`); its
  actual working set is ≈ 10 GB. FlashRT does not over-reserve.
² FlashRT prefill latency (time to first frame logits) for a short prompt; the
  codec adds 3–49 ms once for the whole clip (3–40 s of audio).

Notes:
- **RTF 0.12 across short/medium/long** (≈ 8× real time), ~1.4× faster than
  sglang's full pipeline; **VRAM ~4.5× smaller** than sglang's reservation.
- The kernel-level decode floor is **3.2 ms/frame** (clean CUDA-graph replay at
  a single position, short KV). The frontend's eager per-frame is higher because
  it includes Python dispatch and the attention cost growing with KV length over
  a full generation; a per-position CUDA-graph capture closes most of that gap.
- The codec runs in fp32 (ConvTranspose is unstable in low precision) and costs
  a small one-shot pass at the end (≤ 50 ms for 40 s of audio).

---

## 8. Notes & limitations

- **FP8 calibration** is per-tensor static (activation `amax/448`), measured
  once from a short BF16 free-run of the first prompt and reused. Activation
  ranges are stable across prompts; re-instantiate the frontend to recalibrate.
- The BF16 projection weights are freed after calibration (the FP8 backbone is
  the active path); pass `fp8=False` for the BF16 backbone, which keeps them.
- Synthesis here is **non-streaming** (the codec decodes the whole clip at the
  end). Streaming / chunked synthesis is not yet wired.
- Codec source: the `bosonai/higgs-audio` v2 tokenizer (decode path only),
  vendored under `flash_rt/models/higgs_audio_v3/_codec/`.
