# Qwen3-4B E2E Prefill Latency: HuggingFace vs FlashRT

## Setup

| Item | Value |
|---|---|
| Model | Qwen3-4B (dense CausalLM, BF16) |
| GPU | NVIDIA GeForce RTX 4090 (SM89) |
| Weights | Random init, shared `state_dict` (no Hub download) |
| Metric | **e2e** = embed + decoder layers + final RMSNorm + `lm_head` (full-seq logits) |
| Timing | CUDA events, mean over runs (warmup excluded) |
| FlashRT path | `Qwen3_4BTorchFrontendRtx` default (FA2 + fvk norm/SwiGLU; `use_hf_qkn_rope` + `use_sep_qkv`) |
| HF path | `transformers.Qwen3ForCausalLM` BF16 |
| Bench script | `benchmarks/qwen3_4b_latency.py` |

Note: Qwen3-4B is text-only; vision is N/A. Numbers below are **e2e only**.
Cosine for this RTX default path is in
[`qwen3_4b_bf16_precision_fix.md`](qwen3_4b_bf16_precision_fix.md)
(full-seq cosine ≥ 0.999 from S=8 through S=2048). Thor cosine and latency
are in [`qwen3_4b_thor.md`](qwen3_4b_thor.md).

## How to run

```bash
cd flashrt
export PYTHONPATH=$PWD:$PYTHONPATH
python benchmarks/qwen3_4b_latency.py --seq 1024 --warmup 3 --runs 10
python benchmarks/qwen3_4b_latency.py --seq 512 --warmup 3 --runs 10 --all-fvk
python benchmarks/qwen3_4b_latency.py --seq 1024 --warmup 3 --runs 10 --compare-paths
```

## E2E latency (ms) — streamlined default (after leave-one-out)

Measured with defaults `use_hf_qkn_rope=True`, `use_sep_qkv=True`, other HF
toggles off (FA2 / fvk norm / fused SwiGLU):

| seq_len | HF e2e (ms) | FlashRT e2e (ms) | Speedup (HF / FlashRT) |
|---:|---:|---:|---:|
| 512 | 36.388 | 34.869 | 1.04× |
| 1024 | 67.379 | 66.983 | 1.01× |

## Earlier all-fvk snapshot (pre precision defaults)

For reference, an earlier eager all-fvk path (before HF-parity defaults):

| seq_len | HF e2e (ms) | FlashRT e2e (ms) | Speedup (HF / FlashRT) |
|---:|---:|---:|---:|
| 8 | 14.843 | 10.156 | 1.46× |
| 64 | 15.941 | 11.627 | 1.37× |
| 128 | 17.272 | 12.457 | 1.39× |
| 256 | 23.189 | 18.225 | 1.27× |
| 512 | 36.371 | 31.497 | 1.15× |
| 1024 | 67.374 | 61.659 | 1.09× |
| 2048 | 137.190 | 124.680 | 1.10× |

## Notes

- Precision write-up: `docs/qwen3_4b_bf16_precision_fix.md`.
- Enabling all HF-semantic toggles (SDPA + HF norm + sep gateup) previously
  made FlashRT slightly slower than HF (~70 ms e2e at S=1024).
- seq ≥ 4096 was not recorded: loading HF + FlashRT together OOM'd on 24 GB
  when materializing full `[S, vocab]` logits.
