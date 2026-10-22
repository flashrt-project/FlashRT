# Flint official NVFP4 GGUF on RTX 5090 (SM120)

FlashRT runs the official NVFP4 GGUF of Flint (`RecursiveSI/flint-nvfp4-gguf`)
natively on SM120: an independent GGUF reader and a lossless NVFP4 repack feed
FlashRT's own W4A16 kernels, the decode step is one captured CUDA graph, and
prefill runs in chunks whose launch sequences are captured as CUDA graphs too.
Python only builds buffers, captures the graphs and replays them; no tensor work
runs through PyTorch on the hot path.
The GGUF is a weight container here -- nothing from llama.cpp is linked or
called at run time.

The arithmetic follows the official vLLM implementation of the same GGUF
(W4A16): exact NVFP4 dequantization, BF16 activations and KV cache, FP32
accumulation (see *Numerics*).

## Requirements

| | |
|---|---|
| Checkpoint | `RecursiveSI/flint-nvfp4-gguf`, the single NVFP4 GGUF (22,863,084,160 bytes) |
| Hardware | SM120 (RTX 5090 class), 32 GB |
| Framework | PyTorch (control plane only); `fastapi` + `uvicorn` for the example server |
| Build flag | `FLASHRT_ENABLE_FLINT=ON` (default OFF; requires `GPU_ARCH=120`) |

```bash
cmake -S . -B build -DGPU_ARCH=120 -DFLASHRT_ENABLE_FLINT=ON
cmake --build build -j --target flash_rt_flint
pip install -e ".[torch]"
```

The kernels build into their own module, `flash_rt_flint`; no other target
changes when the flag is on. The CMake option refuses any `GPU_ARCH` other than
120, and the runtime raises a clear error when the module is missing. The
module is compiled without fast-math because the numerics contract uses IEEE
division and precise `expf`.

## Usage

```python
from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx

fe = FlintTorchFrontendRtx("/models/flint-nvfp4.gguf", max_seq=32768)
print(fe.chat([{"role": "user", "content": "Write a CUDA reduction checklist."}], max_new_tokens=512))
```

`chat()` renders the checkpoint's own chat template (tools included) and runs
greedy decoding until an end-of-generation token. Lower-level entry points:

- `fe.tokenizer.apply_chat_template(messages, tools=...)`, `fe.tokenizer.encode/decode`
- `fe.generate(prompt_ids, max_new_tokens)` -- the full continuation
- `fe.stream(prompt_ids, max_new_tokens, reuse_prefix=False)` -- yields the first
  token right after prefill, then blocks of decoded tokens. The next decode
  block is queued before the current one is read back, so the GPU never waits
  for the host. `reuse_prefix=True` keeps the KV cache of the longest common
  prefix with the previous sequence (its prompt and output) and prefills only
  the rest -- the follow-up-turn case.

`max_seq` sizes the full-attention KV cache and is allocated up front; the 35
sliding-window layers use a 16 Ki-slot ring regardless of `max_seq`.

`flash_rt.load_model(config="flint")` is refused with a pointer to this class:
the VLA-style `predict()` API does not fit a text LLM.

### OpenAI-compatible server

```bash
pip install fastapi uvicorn
python examples/flint_openai_server.py --gguf /models/flint-nvfp4.gguf --port 8000 --max-seq 131072
```

`/v1/chat/completions` (stream and non-stream), `/v1/models`, `/health`.
Greedy only (`temperature` must be 0), batch size 1 (requests are serialised).
Extensions: `ignore_eos` (generate exactly `max_tokens`), `cache_prompt`
(default true: prefix reuse across requests), and a `timings` object
(`prompt_n`, `cache_n`, `ttft_ms`, `predicted_per_second`).

## Checkpoint contract

The GGUF reader (`flash_rt/models/flint/gguf.py`) is independent of llama.cpp
and maps the file read-only. Before allocating GPU memory the runtime checks the
exact Flint geometry: 49 layers, hidden 2048, 32 query / 4 KV heads of 128,
two dense SwiGLU layers (6144), 47 MoE layers (256 experts, top-6, expert 512,
one ungated shared expert of 1024), the 14 full-attention NoPE layers and 35
sliding-window RoPE layers (window 513), per-head sinks and Q/K RMSNorm, and the
end-of-generation ids. A mismatch raises `FlintConfigError` naming the field.

NVFP4 tensors (GGML type 40: blocks of 64 values = 4 UE4M3 scale bytes + 32 code
bytes, split-half nibble order) are repacked on the GPU into a row-major
code/scale layout. The repack is lossless: codes and scale bytes are copied
bit for bit (the official file's scale bytes all lie in 0x01..0x7E), and each
tensor's FP32 `.scale` becomes the global scale applied to the FP32 accumulator.
Q/K/V and gate/up are fused only where their global scales are identical, which
holds for every layer of the official file (checked at load).

## Numerics

Matches the official vLLM W4A16 path for this GGUF:

- weights dequantized exactly (`e2m1 * e4m3`), BF16 activations, FP32
  accumulation, global scale on the FP32 accumulator, one rounding to BF16;
- fused-add RMSNorm computed from the unrounded sum;
- BF16 KV cache; attention in FP32 with `P` rounded to BF16 for `P.V` while the
  denominator accumulates the FP32 `P`; sink logits enlarge the denominator;
- router: BF16 logits, top-6 with ties to the lower id, sigmoid, renormalize,
  BF16 weights; routed experts use the approximate SiLU on FP32 gate/up, the
  shared and dense experts the precise SiLU on BF16 gate/up; MoE output
  `bf16(bf16(routed) + shared)`;
- FP32 LM-head logits, on-device argmax.

Decode, prefill and split-KV merges differ only in FP32 summation order.

## Runtime controls

Defaults are the measured best on RTX 5090. These variables are diagnostics and
tuning hooks for other SM120 parts; none changes the arithmetic.

- `FLINT_DECODE=basic` -- unfused reference decode step (default `fused`).
- `FLINT_EAGER=1` -- launch decode steps without the CUDA graph.
- `FLINT_PF_GRAPH=0` -- launch prefill chunks eagerly instead of replaying
  their CUDA graphs.
- `FLINT_PDL=0` -- no programmatic dependent launch between decode kernels.
- `FLINT_LM_GREEDY=0` -- final norm, LM head and argmax as three kernels
  instead of one.
- `FLINT_ATTN_CLUSTER=0` -- merge decode-attention splits through global
  memory instead of a thread-block cluster.
- `FLINT_ATTN_BUCKETS`, `FLINT_SWA_ATTN`, `FLINT_GEMV_CFG`, `FLINT_MOE_CFG`,
  `FLINT_LM_ROWS` -- decode launch shapes (context buckets, split counts, warps,
  rows per warp).
- `FLINT_PF_ATTN_VARIANT` (0-4), `FLINT_PF_SPLIT_MAX` -- prefill attention tile
  and KV-split limits.

`FlintRuntime(prefill_chunk=...)` overrides the prefill chunk; by default it is
the largest multiple of `lcm(16, 4 x SMs)` not above 8192 tokens (8160 on a
170-SM RTX 5090), so a chunk's attention work fills whole waves, and a prompt
that fits 8192 tokens is one chunk.

## Validation

### Tests

```bash
PYTHONPATH=. pytest -q tests/test_flint_kernels.py                     # synthetic, no checkpoint
FLASHRT_FLINT_GGUF=/models/flint-nvfp4.gguf PYTHONPATH=. pytest -q tests/test_flint_pipeline.py
```

The kernel tests compare every kernel against FP64 references built from the
repacked weights (and configurations of one kernel against each other where they
are bit-identical by construction). The checkpoint tests check that programmatic
dependent launch does not change tokens, that the fused and unfused decode steps
agree, that chunked prefill agrees with stepping the prompt through the decode
graph, that prefix reuse continues exactly like a full prefill, and that prefill
chunks replayed from CUDA graphs match eager launches token for token.

### Accuracy

Teacher-forced next-token distributions on eight real coding/agent prompts
(1753 positions) against an FP32 reference of the same weights (exact dequant,
FP32 activations, attention and KV, TF32 off):

| executor | KL to FP32 | top-1 | PPL (FP32 1.2070) |
|---|---:|---:|---:|
| FlashRT decode path | 0.00192 | 98.6% | 1.2072 |
| FlashRT prefill path | 0.00201 | 98.9% | 1.2085 |
| llama.cpp (official executor), decode | 0.00253 | | |
| llama.cpp (official executor), prefill | 0.00234 | 98.4% | 1.2073 |

At 100K tokens of context (last 1024 positions of a real 100K-token prompt):

| executor | KL to FP32 (mean / median) | top-1 | NLL (FP32 1.1486) |
|---|---:|---:|---:|
| FlashRT decode path | 0.123 / 0.00053 | 96.8% | 1.1507 |
| FlashRT prefill path | 0.173 / 0.00048 | 96.4% | 1.1847 |
| llama.cpp prefill path | 0.317 / 0.00169 | 92.8% | 1.3840 |

The FP32 reference itself scores 1.149 at 100K against 0.965 with 4K of context
for the same positions: the rise is the model's, not the kernels'. At this length
small rounding differences move individual distributions a lot (two FlashRT
prefill configurations differ by as much as from FP32), so the mean KL is a
sensitivity band; the median and NLL are the stable figures.

HumanEval (greedy, 8192-token budget): FlashRT 158/164, llama.cpp 155/164; exact
McNemar p = 0.375.

## RTX 5090 numbers

One RTX 5090 (170 SMs, driver 580), the official GGUF on both sides. llama.cpp
is the official executor at the commit that added Flint, run with its serving
flags (f16 KV cache, flash attention) through `llama-bench`: prompt processing
at ubatch 2048, generation of 128 tokens at a populated KV depth. FlashRT
figures are the forward path on real prompts of the stated length: prefill is
GPU time from the first token to the first generated token; decode is the
steady CUDA-graph step including the LM head and on-device argmax, over 256
generated tokens after that prompt. FlashRT figures are medians of three
processes run interleaved with the previous FlashRT build in one session; the
llama.cpp figures come from a separate `llama-bench` sweep on the same machine.

| context | prefill | llama.cpp | | decode | llama.cpp | |
|---:|---:|---:|---:|---:|---:|---:|
| 58 | **11.0 ms** | | | **680 tok/s** | 245.5 | 2.8x |
| 1 K | **41.0 ms** | 102.8 ms | 2.5x | **681 tok/s** | 238.5 | 2.9x |
| 4 K | **132 ms** | 341 ms | 2.6x | **627 tok/s** | 228.4 | 2.7x |
| 8 K | **271 ms** | 690 ms | 2.5x | **604 tok/s** | 226.8 | 2.7x |
| 16 K | **643 ms** | 1445 ms | 2.2x | **554 tok/s** | 218.5 | 2.5x |
| 32 K | **1.62 s** | 3.18 s | 2.0x | **479 tok/s** | 205.1 | 2.3x |
| 64 K | **4.57 s** | 7.59 s | 1.7x | **374 tok/s** | 182.3 | 2.1x |
| 100 K | **9.15 s** | 14.04 s | 1.5x | **308 tok/s** | 162.7 | 1.9x |

Decode at 4 K and 8 K settles at one of two levels about 3% apart from one
process to the next (identical output tokens; e.g. 1.58 or 1.62 ms per token at
4 K); other depths vary by under 1%. The table is the interleaved median. Where
the time goes:

- decode at short context is the per-layer dependency chain (QKV, attention,
  o_proj, router, gate/up, down) with each kernel's weight loads issued before
  its programmatic-dependent-launch wait (a dependent-launch boundary costs about
  0.8 us on this card against 1.3 us for an all-to-all grid barrier inside a
  persistent kernel, so merging the stages behind such barriers would not
  help); at 100 K the 14 full-attention layers take more than
  half of the step (129 us per layer, ~93% of the measured streaming peak);
- prefill GEMMs run at 190-215 TFLOPS on large chunks (the bf16 `mma.sync`
  peak measured on this card is 259 TFLOPS); attention at ~190 TFLOPS, which is
  70% of a 100 K prefill; a short prompt is bound by streaming the routed
  experts' weights (~1.6 TB/s).

GPU memory with a 110,592-token context (device total from `nvidia-smi`; the KV
cache is preallocated): 26.9 GiB idle and after a 100 K request (+30 MiB of
cached prefill graphs, at most ~150 MiB for 64 shapes), against 24.6 GiB for
llama-server with the same context. The
difference is mostly the 16 Ki-slot rings of the 35 sliding-window layers
(1.1 GiB; llama.cpp keeps window-sized caches there) and the preallocated
prefill workspace for 8192-token chunks.

Served through `examples/flint_openai_server.py` and `llama-server` (official
flags, `-np 1`), the same OpenAI streaming client for both: chat messages with
the prompts' tool definitions, greedy, `ignore_eos`, 128 tokens, prompt caching
off. TTFT is measured by the client (request sent to first streamed token,
including templating and tokenization); decode is each server's own rate.

| prompt tokens | TTFT | llama-server | decode | llama-server |
|---:|---:|---:|---:|---:|
| 58 | **12.2 ms** | 57.7 ms | **686 tok/s** | 232 tok/s |
| 1,022 | **45.2 ms** | 227 ms | **683 tok/s** | 225 tok/s |
| 4,089 | **137 ms** | 686 ms | **617 tok/s** | 217 tok/s |
| 8,188 | **280 ms** | 1,317 ms | **596 tok/s** | 216 tok/s |
| 16,371 | **655 ms** | 2,597 ms | **555 tok/s** | 210 tok/s |
| 32,765 | **1.63 s** | 5.42 s | **475 tok/s** | 197 tok/s |
| 65,515 | **4.59 s** | 12.18 s | **374 tok/s** | 176 tok/s |
| 99,986 | **9.17 s** | 21.46 s | **308 tok/s** | 159 tok/s |

Other shapes, same setup:

| scenario | FlashRT | llama-server |
|---|---:|---:|
| 8,192 new tokens on 91,794 tokens of existing KV (forward; llama-bench `pp8192 @ d91808`) | **1.25 s** | 1.72 s |
| follow-up turn on an 8 K conversation, TTFT with prefix reuse / full prefill | **21 / 303 ms** | 128 / 1,359 ms |
| two concurrent 8 K requests (batch 1, queued), TTFT first / second | **289 / 779 ms** | 1,323 / 3,213 ms |
| 1,024-token reply to the 8 K prompt, request time | **2.02 s** | 6.01 s |

## Limitations

- Text only; batch size 1; greedy decoding only (no sampling, no beams).
- SM120 only; the kernels assume the Flint geometry and the checks above refuse
  anything else.
- A prefill chunk graph is captured the first time its shape (rows rounded up
  to 16, attention split counts) occurs, about 1 ms once; up to 64 shapes are
  kept.
- One cached sequence for prefix reuse.
