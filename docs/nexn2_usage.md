# Nex-N2-mini (qwen3_5_moe) — Usage

Nex-N2-mini is a 35B-A3B Mixture-of-Experts **text LLM** in the `qwen3_5_moe`
family (shared architecture with Qwen3.6-35B-A3B):

* 40 decoder layers — Gated DeltaNet linear attention on 3 of every 4 layers,
  full GQA causal attention (16 Q / 2 KV heads, head_dim 256, partial RoPE) on
  every 4th,
* fine-grained MoE FFN: 256 experts, top-8 routed + 1 shared expert,
* hidden 2048, vocab 248320.

It is a text LLM, not a VLA — the frontend exposes `infer()` (logits) and
`generate()` (greedy decode), not the `predict(images, ...)` VLA API.
`flash_rt.load_model(config="nexn2")` raises with a redirect; construct the
frontend directly.

## Support matrix

| | |
|---|---|
| Hardware | RTX 5090 / Blackwell **SM120** only |
| GPU memory | ≥ 32 GB (NVFP4 weights ~22 GB resident) |
| Framework | PyTorch (`torch`) |
| Weight quant | `nvfp4` (default) or `fp8` |
| Build flag | `-DFLASHRT_ENABLE_QWEN35MOE=ON` (required) |
| Prefill | ~12.5k tok/s @ 4k ctx (see table) |
| Decode | ~250 tok/s, token-exact CUDA-graph |
| Precision | cos vs BF16 reference 0.9924, deterministic |

## 1. Build

The model's CUDA kernels are gated behind a build flag so SM89 / SM87 / SM110
and non-Nex-N2 builds never compile them. On an SM120 toolchain NVFP4 and the
vendored FA2 kernel auto-enable; only the gate flag is needed:

```bash
cmake -S . -B build -DFLASHRT_ENABLE_QWEN35MOE=ON
cmake --build build -j
```

This produces `flash_rt/flash_rt_kernels*.so` (with the qwen3_5_moe kernels) and
`flash_rt/flash_rt_fa2*.so` (the attention kernel). Omitting the flag still
builds `flash_rt_kernels`, but the qwen3_5_moe symbols are absent and the
kernelized path raises at first use.

## 2. Install

```bash
pip install -e ".[torch]"
```

The kernelized path uses native FlashRT CUDA kernels only — no Triton / FLA
Python kernels.

## 3. Quickstart

```python
from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx

fe = Nexn2TorchFrontendRtx(
    "<checkpoint_path>",        # HF-style checkpoint directory
    device="cuda",
    kernelized=True,            # NVFP4 kernel path (production)
    quant_scope="experts",
)

fe.set_prompt("The history of artificial intelligence")
ids = fe.generate(max_new_tokens=64)        # greedy decode -> list[int]
print(fe.tokenizer.decode(ids))

logits = fe.infer()                          # (1, S, vocab) prefill logits
```

## 4. Constructor reference

```python
Nexn2TorchFrontendRtx(
    checkpoint_path: str,       # required, HF-style checkpoint directory
    *,
    device: str = "cuda:0",
    max_seq: int = 2048,        # KV cache + scratch sized to this
    quant: str = "nvfp4",       # "nvfp4" or "fp8"
    kernelized: bool = False,   # see below
    quant_scope: str = "full",  # see below
)
```

| Parameter | Values | Meaning |
|---|---|---|
| `checkpoint_path` | path | HF checkpoint directory (config + safetensors + tokenizer). |
| `device` | `"cuda"`, `"cuda:N"` | Target GPU. SM120 required for `kernelized=True`. |
| `max_seq` | int | Max prompt+generation length; KV cache and decode scratch are sized to it. |
| `quant` | `"nvfp4"`, `"fp8"` | Weight quantization for the kernel path. |
| `kernelized` | `False` (default) | BF16 HF reference model (needs the full BF16 weights, >32 GB). For correctness/golden only. |
| | `True` | NVFP4 kernel forward/decode — the production path, fits 32 GB. |
| `quant_scope` | `"experts"` (recommended) | Only routed experts are NVFP4; dense projections run the deterministic BF16-weight w16a16 GEMM → prefill cos ~0.99, bit-reproducible. |
| | `"full"` | Additionally NVFP4-quantises the non-red-line dense projections (q/k/v/o / out_proj / shared) for a smaller footprint at lower cos. |

Methods: `set_prompt(text)`, `infer() -> (1, S, vocab)`, `generate(max_new_tokens) -> list[int]`, `tokenizer`, `latency_records` (list[float], per `infer()`).

## 5. Performance

RTX 5090, `kernelized=True`, `quant_scope="experts"`, greedy decode. TTFT is the
prefill latency (prompt → first-token logits); prefill tok/s = S / TTFT; decode
tok/s is the warm CUDA-graph steady-state rate (KV grows with context).

| Context S | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|---:|---:|---:|---:|
| 128  | 44.2  | 2,897  | 258.5 |
| 256  | 51.3  | 4,993  | 258.5 |
| 512  | 67.7  | 7,563  | 258.5 |
| 1024 | 102.3 | 10,014 | 254.5 |
| 2048 | 176.1 | 11,631 | 249.6 |
| 4096 | 327.8 | 12,494 | 240.8 |

Reference llama.cpp NVFP4 GGUF on the same class of card: prefill 9.5–10.1k,
decode 193–259 tok/s. FlashRT crosses the prefill target from ~1k context and
holds decode at the top of that band.

Reproduce (needs the checkpoint):

```python
import time, torch
from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx
from flash_rt.frontends.torch._nexn2_rtx_decode import (
    Nexn2DecodeState, seed_prefill, generate_greedy_graph)

S, GEN = 2048, 64
fe = Nexn2TorchFrontendRtx("<checkpoint_path>", device="cuda",
                           kernelized=True, quant_scope="experts",
                           max_seq=S + GEN + 8)
ids = fe.tokenizer("Hello " * S, return_tensors="pt")["input_ids"][:, :S].cuda()
st = Nexn2DecodeState(fe._weights, S + GEN + 8, "cuda")

for _ in range(2):
    st.reset(); seed_prefill(st, ids, fe._fvk, "cuda")
torch.cuda.synchronize(); t0 = time.perf_counter()
st.reset(); seed_prefill(st, ids, fe._fvk, "cuda")
torch.cuda.synchronize(); ttft = time.perf_counter() - t0
print(f"TTFT {ttft*1e3:.1f} ms  prefill {S/ttft:.0f} tok/s")

generate_greedy_graph(st, ids, GEN, fe._fvk, "cuda")            # capture
torch.cuda.synchronize(); t0 = time.perf_counter()
generate_greedy_graph(st, ids, GEN, fe._fvk, "cuda")            # warm replay
torch.cuda.synchronize()
print(f"decode {GEN/(time.perf_counter()-t0-ttft):.0f} tok/s")
```

## 6. Precision

`quant_scope="experts"` is deterministic and bit-reproducible run-to-run
(deterministic w16a16 GEMM + deterministic MoE unpermute), so the last-token
logits that seed decode are stable:

* prefill cos vs the BF16 reference **0.9924**, argmax match 438/441,
* decode token-exact: the CUDA-graph replay matches eager decode exactly.

## 7. Limitations

* SM120 (RTX 5090 / Blackwell) only for `kernelized=True`.
* Requires the `-DFLASHRT_ENABLE_QWEN35MOE=ON` build.
* `infer()` materialises all-position logits `(S, 248320)`; on a 32 GB card the
  full-logits prefill ceiling is ~4096 tokens (8192 OOMs on the logits alone).
  Decode is unaffected (single-token logits).
* Text LLM only — not wired into the `load_model` / VLA `predict()` API.
