# OmniVoice via flash_rt.structures — whole-LLM NVFP4 structure + MaskGIT schedule

The OmniVoice TTS host (Qwen3-1.5B backbone + 8-codebook MaskGIT head)
accelerated through the structures layer: a **`decoder_llm`** catalog
structure binds the whole backbone to the native NVFP4 engines (fp4 GEMMs
+ fused qk-norm+RoPE + FA2 + fused residual/norm/quant), and a
**`maskgit_gen`** schedule drives the two-phase MaskGIT loop (BF16 CFG
step, then FP4 noCFG single-stream graph replays).

Kernel resolution is the PR-175 tiering: hub artifact first, the local
native build second, the retained host stack always the floor. The hub's
fp4 packages do not yet ship a torch-2.13 variant, so the local build is
the active tier here.

## Measured (RTX 5060 Ti, torch 2.13+cu130, omnivoice 0.2.1, design mode,
32 MaskGIT steps, gs=2.0, seed=42, median of 3)

| text | baseline RTF | structures RTF | speedup |
|---|---|---|---|
| short | 0.2292 | 0.0438 | 5.24x |
| medium | 0.1187 | 0.0277 | 4.29x |
| long | 0.0791 | 0.0185 | 4.28x |
| **median** | 0.1187 | 0.0277 | **4.29x** |

The native inject path measures 0.0284 median on the same box — the
structures route is identical within noise (and slightly ahead once the
codec is included). Per-seam-only adoption (fp4 FFN, fp4 FFN+projections)
measured ~1.0x: the win lives in the whole-LLM boundary and the schedule,
not the individual kernels.

### Rejected: CFG steps on the FP4 engine (measured, 32 steps fixed)

The CFG phase (B=2, ~5% of steps at cfg_ratio=0.05) rides the BF16
engine by design — no quantization drift on the guidance subtraction.
Moving it onto the FP4 engine via two B=1 graph replays is 2.1x faster
per step (10.95 ms vs 2x 2.60 ms) but **breaks generation**: same-seed
tokens match only 0.8% and the output collapses to 0.68 s of audio vs
3.14 s on the same schedule. The FP4 engine's hidden states differ from
BF16's by ~2x relative (W4A4 quantization is a different distribution,
not a perturbation), which the cond-uncond difference amplifies. Rejected
per AGENTS.md: the 6.3% win is not worth a broken CFG path. A real fix
needs a B=2 FP4 engine (kernel-side, FlashRT-HF-kernels delivery).

### The codec (the remaining 12%)

The neural codec decode is compute-bound on fp32 cuDNN convolutions; the
only measured lever is fp16 autocast (1.22x, waveform cosine 1.00000 vs
the fp32 host — torch.compile and CUDA graph both measure ~1.0x, and the
pydub postprocess is only ~3.7 ms, not worth replacing). It is adopted
as an `audio_codec` structure (`impls/audio_codec/fp16.py`): codes ->
waveform, fp16 decode with host fallback. The host family
(HiggsAudioV2TokenizerModel) is shared by OmniVoice and Higgs-Audio-v3,
so the structure recurs beyond one model.

## Usage

```python
import torch
from omnivoice import OmniVoice
from omnivoice import OmniVoiceGenerationConfig
from flash_rt import structures
from flash_rt.structures.impls.decoder_llm import nvfp4 as llm_impl

model = OmniVoice.from_pretrained("/path/to/OmniVoice",
                                  dtype=torch.bfloat16).to("cuda:0")
model.eval()

def calibration():                       # one short real forward
    with torch.no_grad():
        g = OmniVoiceGenerationConfig(num_step=2, guidance_scale=2.0,
                                      denoise=True, preprocess_prompt=True,
                                      postprocess_output=False,
                                      position_temperature=5.0)
        model.generate(text="Calibration sentence.", generation_config=g)

plan = structures.auto_swaps(model, calibration,
                             structures=("decoder_llm", "audio_codec"),
                             scheme="nvfp4_static")
print(structures.explain(plan))          # receipt: llm + audio_tokenizer
handle = plan.attach()                   # the one-call door

g = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0,
                              denoise=True, preprocess_prompt=True,
                              postprocess_output=True,
                              position_temperature=5.0)
task = model._preprocess_all(text="Hello.", language=None, ref_text=None,
                             ref_audio=None, voice_clone_prompt=None,
                             instruct=None, preprocess_prompt=True,
                             speed=None, duration=None,
                             normalize_text=False)
st = task.slice_task(task.get_indices(
    g, model.audio_tokenizer.config.frame_rate)[0])
loop = structures.maskgit_loop(model)               # the schedule door
with torch.no_grad():
    tokens = loop.generate(st, g)                   # two-phase schedule
    audio = model._decode_and_post_process(tokens[0], None, g)
```

Requirements: `flash_rt_kernels` + `flash_rt_omnivoice` + `flash_rt_fa2`
built with `-DFLASHRT_ENABLE_OMNIVOICE=ON -DGPU_ARCH=120` (sm_120a), the
`omnivoice` pip package, and a checkpoint.

## What was added

- `flash_rt/structures/catalog/decoder_llm/` — structure spec + torch
  reference (the host's eager stack forward is the parity ground truth).
- `flash_rt/structures/impls/decoder_llm/nvfp4.py` — the whole-LLM seam:
  BF16 fused engine for the CFG batch (B=2), FP4 CUDA-graph engine for
  single stream (B=1); profile-envelope refusal outside the native
  engine's v1 contract (D=1024/L=28/NH=16/NKV=8/HD=128/FFN=3072); and the
  `maskgit_gen` schedule door.
- `flash_rt/structures/impls/decoder_ffn/nvfp4_static.py` and
  `flash_rt/structures/impls/linear_proj/nvfp4_static.py` — per-seam fp4
  backends (kept; useful on other hosts).
- `flash_rt/structures/schemes.py` — `nvfp4_static` scheme (weight-only,
  no calibration data).
- `flash_rt/structures/discover.py` — decoder-stack discovery rule
  (layers/embed_tokens/norm/rotary_emb slots, not model names).
- `flash_rt/structures/autobuild.py` — bind dispatch for the new formats.
- `flash_rt/structures/bindings/omnivoice_llm.yaml` — host addressing
  receipt.
- `tests/test_structures_decoder_llm.py` — CPU contract pins.

## Notes

- The noCFG FP4 phase collapses tiny prompts into a repeated-code
  attractor (silent audio) in both the host and this path — a host
  characteristic, not a structure bug. Keep `guidance_scale > 0` and use
  real sentences.
- Attaching over a service thread pool: the seam guard pins one thread
  per attachment; reset `model.llm._frt_guard.thread = None` per request
  when uvicorn-style pooling is in play.
