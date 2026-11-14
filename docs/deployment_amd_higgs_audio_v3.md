# Higgs Audio v3 on AMD RDNA

FlashRT provides native single-stream text-to-speech on W7900D, Radeon 8060S /
Strix Halo and Radeon AI PRO R9700. It preserves the CUDA frontend's greedy
default, temperature/seed sampling, delayed codebooks, EOC winddown and streaming
packet/tail protocol. Output is CPU float32 mono audio at 24 kHz.

## Hardware and precision

| Target | Auto (`fp8=None`) | `fp8=True` | `fp8=False` |
|---|---|---|---|
| gfx1100 / W7900D | BF16 W16A16 | Warning, BF16 fallback | BF16 W16A16 |
| gfx1151 / Radeon 8060S, Strix Halo | BF16 W16A16 | Warning, BF16 fallback | BF16 W16A16 |
| gfx1201 / Radeon AI PRO R9700 | FP8 W8A8 | FP8 W8A8 | BF16 W16A16 |

FP8 uses RDNA4's native E4M3 path. An unsupported explicit request emits a
`RuntimeWarning` before loading weights. `frontend.fp8` and `precision_spec`
report the effective precision. CUDA usage remains in the [RTX guide](higgs_audio_v3.md).

## Setup and usage

Use ROCm PyTorch with its matching AMD Triton package:

```bash
python -m pip install --no-deps -e .
python -m pip install 'transformers==4.57.1' tokenizers safetensors sentencepiece
hf download bosonai/higgs-tts-3-4b \
  --revision 239f63fb7b02b1aa085f98d9efae5e35cc5523e8 \
  --local-dir /path/to/higgs-tts-3-4b
export HIGGS_CHECKPOINT=/path/to/higgs-tts-3-4b
python examples/amd/higgs_audio_v3_quickstart.py --out hello.wav
# Explicit BF16 and native full-vocabulary temperature sampling:
python examples/amd/higgs_audio_v3_quickstart.py --bf16 \
  --temperature 1.0 --seed 1234 --text 'Hello from FlashRT.' --out sampled.wav
```

The RDNA implementation uses ROCm libraries and lazy-compiled AMD Triton kernels;
no separate native GEMM extension is required. Consult the checkpoint's model
card for license terms. PyTorch spells a ROCm device `cuda:N`.

```python
import os
from flash_rt.models.higgs_audio_v3 import create_frontend

fe = create_frontend(os.environ["HIGGS_CHECKPOINT"], device="cuda:0")
# Add fp8=False for explicit BF16.
audio = fe.generate("Hello from FlashRT.")  # Greedy default.
for packet in fe.generate_stream(
    "你好，欢迎使用本地语音助手。", temperature=1.0, seed=1234,
):
    consume(packet)  # CPU float32 mono at 24 kHz.
```

The factory detects the selected device; explicit `hardware="amd_rdna3"`,
`"amd_rdna35"` or `"amd_rdna4"` validates that target before loading weights.
The existing [HTTP host](../serving/higgs_audio_agent/README.md) uses this factory.
See the [API contract](stable_api.md#higgs-audio-v3-tts) for code-generation and
waveform methods.

The codec API preserves float values, which can exceed PCM's `[-1, 1]` range.
The quickstart applies one gain to the complete file before PCM16 export; it
does not normalize individual streaming packets. Provide equivalent headroom
when playing the float output directly.

## Execution and streaming

`temperature <= 1e-5` selects greedy decoding; a larger temperature samples the
full codebook vocabulary using native SplitMix64/CDF sampling. A seed fixes
the request's random stream, but numerical differences between devices or
algorithm selections can still change the generated speech.

One frontend owns its weights, KV cache, scratch and position-independent
decode graph. Serialize requests to each instance. Prefill batches prompt
tokens and chunks prompts beyond `max_prefill_tokens`; validated resident
prefixes can be reused. `max_seq` bounds prompt plus delayed audio positions.
`last_finish_reason` reports `eoc`, `repeated_code`, `max_new_frames` or `max_seq`.
EOC termination alone does not establish transcript completeness.

Streaming defaults to `first_chunk=8, chunk=25, ctx=8, holdback=8`, with 40 ms
per acoustic frame. It waits for the first packet's right context, decodes
left/right codec context, and trims each packet and the final tail to committed
frames. Concatenated streaming audio approximates one-shot codec output.

`FLASHRT_HIGGS_GRAPH=0` disables the decode graph;
`FLASHRT_HIGGS_BATCHED_PREFILL=0` selects token-wise prefill. Packing, compilation,
graph capture and codec loading are setup costs. Batched/concurrent execution
and reference-audio voice cloning are outside this frontend's interface.

## Numerical implementation

R9700 uses OCP E4M3 weights with one FP32 `amax/448` scale per tensor, including
fused Q/K/V and gate/up weights. Activations use dynamic per-row
`max(amax/448, 1e-12)` scales. Multi-token prefill uses native FP8 WMMA;
single-token prefill and decode use quantized GEMV with FP32 accumulation.
CUDA's existing FP8 path uses static activation calibration and an FP8 audio
head. RDNA keeps the audio head in BF16; cross-device comparisons must identify
these policies rather than assume identical FP8 math.

Embeddings, audio-head weights, normalization and KV remain BF16; the codec is
FP32. Head logits round to BF16 before sampling, matching CUDA native. RDNA
rounds SiLU before multiplication and
uses an FP32 attention denominator. Q/K normalization rounds before rotary
embedding; residual normalization computes statistics before storing the BF16
residual. `precision_spec` describes the effective numerical path.

Model traversal, sampling and codec adaptation live under
`flash_rt/amd/models/higgs_audio_v3/`; reusable operations are under
`flash_rt/amd/hardware/rdna/`. Codec dilated convolutions use equivalent dense
polyphase batches through ROCm with autocast disabled.

## Lightweight regressions

These tests require no checkpoint or reference download:

```bash
# Routing, API compatibility, precision selection and native generation policy.
python -m pytest -q tests/test_amd_higgs_audio_v3.py
# RDNA-only operator, quantization, cache and native sampling/CDF contracts.
python -m pytest -q tests/test_amd_higgs_audio_v3_ops.py
```

CUDA sampling tests remain separate in `tests/test_higgs_audio_v3_sampling_kernel.py`.

## Measuring latency

Use a single available GPU and sequential requests. Match text, precision,
`temperature/seed`, generation limits, packet settings, warmup and cache policy
when comparing implementations. End-to-end timing includes text preparation,
generation, codec and CPU delivery; excludes loading, warmup and file/player IO.
Different generated durations are not equal-work kernel comparisons.

```bash
python benchmarks/higgs_audio_v3_rdna_latency.py \
  --checkpoint "$HIGGS_CHECKPOINT" --precision bf16 \
  --text 'The quick brown fox jumps over the lazy dog.' \
  --temperature 0 --seed 0 --warmup 3 --iters 10 --stage-iters 30 \
  --output latency.json
```

The report records effective precision, finish reasons, every packet arrival,
TTFA, RTF, variation and optional prefill/decode stages. Stage timings exclude
sampling and codec. A run that stops on the frame limit or repeated-code guard
is retained as such, not counted as a successful EOC request. Packet timing
also records how much playback buffering the measured run would require.
