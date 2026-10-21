# Qwen3-4B BF16 on Jetson Thor

Thor (sm_110) bring-up of the dense HuggingFace **Qwen/Qwen3-4B** path.
Attention uses torch SDPA because the vendored FA2 build is not available on
sm_110. Weights in the numbers below are a **shared random** `state_dict`
(no Hub download). The selected flag set is `default_hf_parity`.

Geometry: H=2560, 36 layers, GQA 32q/8kv, head_dim=128, intermediate=9728, BF16.

## Differences vs RTX (`docs/qwen3_4b_bf16.md`)

| | RTX 4090 (SM89) | Jetson AGX Thor (SM110) |
|---|---|---|
| Attention | FA2 `fwd_bf16_causal` | torch SDPA (`use_sdpa=True`, auto when FA2 is missing) |
| Attention buffers | `RtxFlashAttnBackendQwen3` | `ThorAttnBackendQwen3` |
| Pipeline map | `rtx_sm89` / `rtx_sm120` | `thor` |
| Build | `GPU_ARCH=89` + FA2 | `GPU_ARCH=110` (no FA2) |

## Flags

`use_sdpa` is omitted from the constructor so the frontend turns it on when
`flash_rt_fa2` is missing.

| flag | value |
|---|---|
| `use_sdpa` | `True` (auto on Thor) |
| `use_hf_qkn_rope` | `True` |
| `use_sep_qkv` | `True` |
| `use_torch_qkv` | `True` |
| `use_hf_norm` | `False` |
| `use_sep_gateup` | `False` |

Relative to HuggingFace eager BF16, FlashRT keeps RMSNorm/residual, fused
SwiGLU, embedding, and most linear layers on CUDA / cuBLASLt BF16 kernels,
with an explicit KV cache. q/k-Norm+RoPE and split QKV stay on the HF
algorithm. No FP8 or NVFP4 on this path.

## Cosine similarity (default path, vs HuggingFace)

Gate: full-sequence logits cosine ≥ 0.999. Same random shared weights on both
sides. Source: `artifacts/qwen3_4b_thor_kernel_flags.json` (`default_summary`).

| seq | cos_full | cos_last | max‖Δ‖ | mean‖Δ‖ | gate |
|---:|---:|---:|---:|---:|:---:|
| 8 | 0.999561 | 0.999544 | 0.147 | 0.0238 | PASS |
| 128 | 0.999518 | 0.999439 | 0.191 | 0.0250 | PASS |
| 800 | 0.999386 | 0.999243 | 0.234 | 0.0282 | PASS |

`all_fvk` (fused QKV, fused q/k-Norm+RoPE) passes the 0.999 floor at S=128
(cos_full 0.999217) and fails it at S=800, so the checked-in default stays
`default_hf_parity`.

## E2E prefill latency (default path)

CUDA-event timing, warmup excluded. Metric is embed + decoder + final RMSNorm
+ `lm_head` (full-sequence logits).

| seq | HF | FlashRT default | speedup | cos_full |
|---:|---:|---:|---:|---:|
| 128 | 69.8 ms | 49.6 ms | 1.41× | 0.999518 |
| 800 | 138.0 ms | 118.1 ms | 1.17× | 0.999386 |

Candidate sweep at S=128 (same artifact; HF e2e 82.2 ms on that run):

| candidate | e2e | vs HF | cos_full | selected |
|---|---:|---:|---:|:---:|
| `default_hf_parity` | 49.4 ms | 1.67× | 0.999518 | yes |
| `fused_qkv` | 48.3 ms | 1.70× | 0.999443 | no |
| `fvk_qkn_rope` | 41.7 ms | 1.97× | 0.999218 | no |
| `all_fvk` | 40.3 ms | 2.04× | 0.999217 | no (fails S=800) |

The 1.67× row is the adaptation script's S=128 sweep. The 1.41× / 1.17×
table is the later default-path measurement at S=128 and S=800.

## How to reproduce

From the repo root, with a Thor build (`GPU_ARCH=110`) and
`PYTHONPATH=$PWD`:

```bash
# Cosine gate (random weights). Needs transformers + a CUDA build.
python -m pytest tests/test_qwen3_4b_hf_cosine.py -q -s

# Latency vs HuggingFace
python benchmarks/qwen3_4b_latency.py --seq 128 --warmup 2 --runs 5 --compare-paths

# Flag sweep + kernel_flags JSON (random weights, SDPA)
python scripts/adapt_qwen3_4b_thor.py --seq 128 --cosine-seq 8 --warmup 2 --runs 5
```

RTX 4090 cosine and latency live in
[`qwen3_4b_bf16_precision_fix.md`](qwen3_4b_bf16_precision_fix.md) and
[`qwen3_4b_e2e_latency_hf_vs_flashrt.md`](qwen3_4b_e2e_latency_hf_vs_flashrt.md).
