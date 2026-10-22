# Flint official NVFP4 GGUF on RTX 5090 (SM120)

FlashRT runs the official NVFP4 GGUF of Flint (`RecursiveSI/flint-nvfp4-gguf`)
natively on SM120: an independent GGUF reader and a lossless NVFP4 repack feed
FlashRT's own W4A16 kernels, the decode step is one captured CUDA graph, and
prefill runs in chunks whose launch sequences are captured as CUDA graphs too.
The captured compute paths call native kernels. The torch frontend owns weight
loading, buffers, graph capture and replay; prompt upload/reset and token
readback remain outside the graphs.
The GGUF is a weight container here -- nothing from llama.cpp is linked or
called at run time.

The W4A16 arithmetic contract is: exact NVFP4 dequantization, BF16 activations and KV cache, FP32
accumulation (see *Numerics*).

## Requirements

| | |
|---|---|
| Checkpoint | `RecursiveSI/flint-nvfp4-gguf`, the single NVFP4 GGUF (22,863,084,160 bytes) |
| Hardware | SM120 (RTX 5090 class), 32 GB |
| Framework | PyTorch for buffer/graph ownership; `tokenizers` + `jinja2` for chat; `fastapi` + `uvicorn` only for the example server |
| Validated toolchain | Linux, Python 3.10.12, PyTorch 2.14.1+cu130, CUDA 13.0.88, driver 580.178.04, GCC 11.4 |
| Build flag | `FLASHRT_ENABLE_FLINT=ON` (default OFF; requires `GPU_ARCH=120`) |

```bash
# Follow docs/INSTALL.md for the CUDA toolchain and CUTLASS setup first.
pip install -e ".[flint]"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=120 -DFLASHRT_ENABLE_FLINT=ON
cmake --build build -j --target flash_rt_flint
```

The kernels build into their own module, `flash_rt_flint`; no other target
changes when the flag is on. The CMake option refuses any `GPU_ARCH` other than
120, and the runtime raises a clear error when the module is missing. The
module is compiled without fast-math because the numerics contract uses IEEE
division and precise `expf`.

## Usage

```python
from flash_rt.frontends.torch.flint_rtx import FlintTorchFrontendRtx

fe = FlintTorchFrontendRtx("<path-to-flint-nvfp4.gguf>", max_seq=32768)
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
Prefix reuse requires the entire historical sliding window at the continuation
position to remain resident. If an earlier request has overwritten it, the
frontend automatically recomputes the full prompt (`last_reused == 0`). Cache
lifetime tracking includes padded prefill rows and queued decode lookahead,
even when those tokens are discarded at an early stop.

`flash_rt.load_model(config="flint")` is refused with a pointer to this class:
the VLA-style `predict()` API does not fit a text LLM.

### OpenAI-compatible server

```bash
pip install fastapi uvicorn
python examples/flint_openai_server.py --gguf "<path-to-flint-nvfp4.gguf>" --port 8000 --max-seq 131072
```

`/v1/chat/completions` (stream and non-stream), `/v1/models`, `/health`.
Greedy only (`temperature` must be 0), batch size 1 (requests are serialised).
Extensions: `ignore_eos` (generate exactly `max_tokens`), `cache_prompt`
(default true: prefix reuse across requests), and a `timings` object
(`prompt_n`, `cache_n`, `ttft_ms`, `predicted_per_second`).

## Checkpoint contract

The GGUF reader (`flash_rt/models/flint/gguf.py`) is independent of llama.cpp
and maps the file read-only. Before allocating GPU memory the runtime checks the
supported Flint geometry: 49 layers, hidden 2048, 32 query / 4 KV heads of 128,
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

W4A16 contract for this GGUF:

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
- `FLINT_PF_FUSE=0` -- separate prefill kernels for the Q/K norm + RoPE + KV
  write and for the norm after the MoE combine (default: fused into the QKV
  GEMM epilogue and the combine).
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
PYTHONPATH=. pytest -q tests/test_flint_contracts.py                   # CPU contracts, no checkpoint
PYTHONPATH=. pytest -q tests/test_flint_kernels.py                     # synthetic, no checkpoint
FLASHRT_FLINT_GGUF="<path-to-flint-nvfp4.gguf>" PYTHONPATH=. pytest -q tests/test_flint_pipeline.py
```

The kernel tests compare every kernel against FP64 references built from the
repacked weights (and configurations of one kernel against each other where they
are bit-identical by construction). The checkpoint tests check that programmatic
dependent launch does not change tokens, that the fused and unfused decode steps
agree, that chunked prefill agrees with stepping the prompt through the decode
graph, that prefix reuse continues exactly like a full prefill, and that prefill
chunks replayed from CUDA graphs match eager launches token for token.

### Reproduce forward timings

After building FlashRT above, create the public token fixtures once. The model
tokenizer encodes this exact text: a fixed instruction followed by the pinned
README; the token sequence is repeated and truncated. Both drivers consume
these same files, and the native JSON reports their SHA256 fingerprints.

```bash
git show 1fbe995089af4148bacd10ead61764d0aeb80f7a:README.md > build/flint-corpus.txt
PYTHONPATH=. python examples/flint_benchmark.py \
    --gguf "<path-to-flint-nvfp4.gguf>" \
    --make-tokens build/flint-tokens --corpus build/flint-corpus.txt
```

Build the optional reference driver against the pinned upstream executor.
It is used only for comparison and is not linked into FlashRT.

```bash
LLAMA_DIR=build/llama.cpp-reference
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
git -C "$LLAMA_DIR" checkout --detach 1b44e30775dcb2f93952116b05f89e5bd797fb00
git -C "$LLAMA_DIR" fetch origin pull/30929/head
git -C "$LLAMA_DIR" cherry-pick --no-commit FETCH_HEAD
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_OPENSSL=OFF
cmake --build "$LLAMA_DIR/build" -j --target llama-bench
g++ -O3 -std=c++17 examples/flint_llama_benchmark.cpp \
    -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
    -L "$LLAMA_DIR/build/bin" -Wl,-rpath,"$(realpath "$LLAMA_DIR/build/bin")" \
    -lllama -lggml -o build/flint-llama-benchmark
```

Run the following commands in four fresh processes, in reference / native /
native / reference order. Keep all fixture files in the same order so the
requested context allocation is identical. The drivers emit every measured
repetition; pool the two runs of each executor by prompt length, then take the
median of the four repetitions. Use the `*_wall_*` fields for this comparison;
the native driver's CUDA-event fields are separate diagnostics.

```bash
prompts=()
for n in 58 1024 4096 8192 16384 32768 65536 102400; do
    prompts+=("build/flint-tokens/prompt-$n.i32")
done
# Reference: 256 decode steps, one warmup, two measured repetitions.
CUDA_VISIBLE_DEVICES=0 build/flint-llama-benchmark \
    "<path-to-flint-nvfp4.gguf>" 256 1 2 "${prompts[@]}"
# Native: the same protocol.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python examples/flint_benchmark.py \
    --gguf "<path-to-flint-nvfp4.gguf>" --tokens "${prompts[@]}" \
    --n-gen 256 --warmup 1 --reps 2
```

Corpus SHA256: `3c91b20122fd59d895dfd4493572e005026d92ef04c911012d39f01000c3a605`.
The reference driver explicitly enables backend greedy sampling and window-sized
SWA caches, and reserves one output. The low-level llama.cpp API otherwise
defaults to full-length SWA allocation and batch-sized output capacity, which
is not the single-sequence CLI configuration used here.

### Accuracy

The full quality figures below come from the frozen numerical core, before
publication cleanup. Kernel arithmetic and launch order are unchanged; this
cleanup reran 150 numerical kernel tests, 26 CPU/interface contracts, the
serving ownership test and all five checkpoint pipeline tests. A fresh
teacher-forced FP32 probe has KL **0.000670**, top-1 **99.22%**.

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

One RTX 5090 (170 SMs, 32 GB), CUDA 13.0.88, driver 580.178.04,
Python 3.10.12, PyTorch 2.14.1+cu130, tokenizers 0.23.3 and Jinja2 3.1.6.
Both executors read the same official NVFP4 GGUF and identical int32 token
files generated from the public README at FlashRT `1fbe995`. This is a fixed
text workload, repeated/truncated to the exact lengths below; accuracy is
evaluated separately on coding prompts.

The reference is stock llama.cpp CUDA at
`1b44e30775dcb2f93952116b05f89e5bd797fb00` with the Jinja-only fix from
PR #30929, without a FlashRT overlay. Both runs are single-sequence greedy
generation, flash attention, default CUDA graphs and requested context capacity
`102720` (longest prompt + 256 + 64). llama.cpp uses 8 CPU threads,
`n_batch=n_ubatch=2048`, one output, sliding-window KV allocation and backend
greedy sampling. FlashRT uses its default 8192-token prefill capacity / 8160-token
full chunk. Runtime precision differs: FlashRT is W4A16 with BF16 activations/KV;
llama.cpp uses its stock NVFP4 CUDA path with F16 KV. These are executor results,
with separate correctness evidence, rather than an identical-arithmetic kernel A/B.

Fresh process order: llama.cpp / FlashRT / FlashRT / llama.cpp. Each process
warms each prompt once, then measures it twice. Tables pool all four measured
repetitions per executor and report the median. Prefill is wall time from
prompt reset/upload through the first greedy token. Decode is wall time for
256 additional steps, including the LM head and greedy selection; throughput
is `1000 / median_ms_per_step`. Initial loading/capture, tokenization, HTTP and
text rendering are excluded from both. Results measured 2026-10-23 (UTC+8).

### Decode

| Context tokens | llama.cpp tok/s | FlashRT tok/s | Speedup | Throughput increase |
|---:|---:|---:|---:|---:|
| 58 | 241.8 | **677.0** | **2.80×** | +180.0% |
| 1,024 | 234.8 | **683.6** | **2.91×** | +191.1% |
| 4,096 | 219.3 | **632.1** | **2.88×** | +188.3% |
| 8,192 | 217.9 | **603.0** | **2.77×** | +176.8% |
| 16,384 | 210.7 | **554.0** | **2.63×** | +163.0% |
| 32,768 | 197.7 | **478.9** | **2.42×** | +142.3% |
| 65,536 | 176.2 | **374.3** | **2.12×** | +112.4% |
| 102,400 | 157.8 | **303.8** | **1.93×** | +92.5% |

### Prefill

| Prompt tokens | llama.cpp ms | FlashRT ms | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 58 | 36.8 | **10.8** | **3.40×** | 70.6% |
| 1,024 | 105.0 | **41.4** | **2.54×** | 60.6% |
| 4,096 | 342.3 | **131.2** | **2.61×** | 61.7% |
| 8,192 | 681.4 | **267.9** | **2.54×** | 60.7% |
| 16,384 | 1,425.7 | **637.3** | **2.24×** | 55.3% |
| 32,768 | 3,126.0 | **1,603.2** | **1.95×** | 48.7% |
| 65,536 | 7,517.5 | **4,551.5** | **1.65×** | 39.5% |
| 102,400 | 14,410.5 | **9,476.2** | **1.52×** | 34.2% |

Fresh native decode samples span under 0.3% at every length. The widest
reference prefill range is 334.3–349.8 ms at 4096 tokens (median 342.3 ms).
Earlier sessions observed two native decode levels about 3% apart at 4K/8K;
the tables use all samples of the new matched run and do not pool older data.

## Limitations

- Text only; batch size 1; greedy decoding only (no sampling, no beams).
- SM120 only; the kernels assume the Flint geometry and the checks above refuse
  anything else.
- A prefill chunk graph is captured the first time its shape (rows rounded up
  to 16, attention split counts) occurs, about 1 ms once; up to 64 shapes are
  kept.
- One cached sequence for prefix reuse.

## Ownership and references

- [Flint model](https://huggingface.co/RecursiveSI/flint) and
  [official NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf):
  checkpoint, architecture, tokenizer and embedded chat template. The evaluated
  GGUF revision is `6ec43e19b565fc8d560e2d04985d7724f20673bf`, SHA256
  `97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) supplies the external
  executor reference and GGUF format conventions; its
  [Jinja fix](https://github.com/ggml-org/llama.cpp/pull/30929) is included in
  the reference build. This module does not link or execute llama.cpp.
- FlashRT owns `flash_rt_flint` / `flash_rt::flint` kernels and the model-local
  launch plan. `_flint_rtx_runtime.py` in the torch frontend owns GGUF loading,
  persistent allocations and CUDA Graph lifecycle. `pipeline_rtx.py` composes
  already-owned raw pointers and handles. Python kernel names are private to
  the dedicated model module; they do not modify `flash_rt_kernels`.
- W4A16 uses the checkpoint's stored block and global scales, without
  calibration or an on-disk repack cache. Each runtime instance owns its
  weights and graphs; loading another checkpoint creates a new instance.
