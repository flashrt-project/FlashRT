# Nex-N2-mini (qwen3_5_moe) — Usage

Nex-N2-mini is a 35B-A3B Mixture-of-Experts **text LLM** in the
`qwen3_5_moe` family (shared architecture with Qwen3.6-35B-A3B):

* 40 decoder layers — Gated DeltaNet linear attention on 3 of every 4
  layers, full GQA causal attention (16 Q / 2 KV heads, head_dim 256,
  partial RoPE) on every 4th,
* fine-grained MoE FFN: 256 experts, top-8 routed + 1 shared expert,
* hidden 2048, vocab 248320.

It is **not a VLA** — the frontend exposes an LLM surface (`infer()`
returns logits, `generate()` greedy-decodes) rather than the
`predict(images, ...)` VLA API. It is registered in
`flash_rt.hardware._PIPELINE_MAP` for discoverability
(`resolve_pipeline_class("nexn2", "torch", "rtx_sm120")`) but is used by
**direct instantiation** of the frontend, not `load_model`.

## Hardware & build

RTX 5090 / Blackwell **SM120 only** (NVFP4 + the vendored FA2 kernel).
The model's CUDA kernels are gated behind a build flag so that SM89 /
SM87 / SM110 and non-Nex-N2 builds never compile them:

```bash
cmake -S . -B build -DFLASHRT_ENABLE_QWEN35MOE=ON
cmake --build build --target flash_rt_kernels -j
```

Without `-DFLASHRT_ENABLE_QWEN35MOE=ON` the `flash_rt_kernels` module
still builds, but the qwen3_5_moe symbols are absent and the kernelized
path raises at first use. The decode/prefill attention additionally
requires the vendored `flash_rt_fa2.so` (`ENABLE_FA2`, already a hard
dependency of every RTX LLM path).

## Installation

```bash
pip install -e ".[torch]"          # frontend + tokenizer
```

The kernelized path uses native FlashRT CUDA kernels only — no
Triton / FLA Python kernels.

## Constructor

```python
from flash_rt.frontends.torch.nexn2_rtx import Nexn2TorchFrontendRtx

fe = Nexn2TorchFrontendRtx(
    checkpoint_path,            # required, HF-style checkpoint dir
    *,                          # keyword-only below
    device='cuda:0',
    max_seq=2048,               # KV cache + scratch sized to this
    quant='nvfp4',              # 'nvfp4' (default) or 'fp8'
    kernelized=True,            # True = NVFP4 kernel path (production);
                                # False = BF16 HF reference (needs >32 GB)
    quant_scope='experts',      # 'experts' = routed experts NVFP4, dense
                                # GEMMs BF16 (w16a16) -> cos ~0.99;
                                # 'full' = also NVFP4 the dense GEMMs
)
```

The 35B-A3B checkpoint does not fit a 32 GB RTX 5090 in BF16, so the
kernelized loader streams each shard, quantizes the large GEMMs to NVFP4
(GDN in_proj / norms / router kept BF16), and frees the BF16 source as it
goes (~22 GB resident).

## Inference

```python
fe.set_prompt("The history of artificial intelligence")
logits = fe.infer()                       # (1, S, vocab) — prefill
tokens = fe.generate(max_new_tokens=128)  # greedy decode (KV + GDN state)
```

## Compute path (kernelized)

All heavy compute runs on FlashRT SM120 kernels — no `torch` matmuls,
no `F.scaled_dot_product_attention`, no host-side sync in the hot path:

| Stage | Kernel |
|---|---|
| Dense projections (q/k/v/o, GDN in/out, router, shared, lm_head) | `w16a16_gemm_sm120` (BF16 weight, FP32 accumulate, deterministic) |
| Full-attn (prefill) | vendored FA2 causal (`flash_rt_fa2.fwd_bf16_causal`), native GQA |
| Full-attn (decode) | vendored FA2 (KV cache) |
| GDN linear attn | WY chunked delta-rule (`linear_attn_gdn_wy_*`) / recurrent decode |
| MoE routed experts | block-tile NVFP4 mma (`moe_blocktile_mma_sm120`, sync-free tiles) |
| RoPE / gating / norms / conv1d / silu·gate | fused `qwen36_*` / `*_sm120` kernels |

The prefill attention backend was selected by a head-to-head meta-test
(`nexn2_dev/tests/phase_attn_metatest.py`): over the Nex-N2 full-attn
shape (S, 16Q/2KV, HD=256, causal, bf16), vendored FA2 causal ranks
first at every S (cos 1.0), beating `flash_attn` pip (~4%); SageAttention
rejects HD=256.

## Precision & performance

Measured on RTX 5090, prompt S=2048, `quant_scope='experts'`:

* prefill **~11.3k tok/s**, deterministic cos vs the BF16 HF golden
  **0.992** (red line ≥ 0.984),
* decode **~250 tok/s**, token-exact (CUDA-graph replay == eager),
* beats the llama.cpp NVFP4 GGUF baseline (prefill 9.5–10.1k, decode
  193–259 tok/s).

The prefill is bit-reproducible run-to-run (deterministic w16a16 GEMM +
deterministic MoE unpermute), so the last-token logits that seed decode
are stable.

## End-to-end validation

The cos-vs-golden / token-exact harness lives in the dev tree
(`nexn2_dev/tests/`, requires the checkpoint + `refs/nexn2_golden_long2.pt`):

* `phase_attn_e2e_check.py` — prefill cos vs golden + argmax,
* `phase4m_ondevice_argmax.py` — decode graph == eager + tok/s,
* `phase8t_attn_glue.py` — prefill lever stack cos + tok/s.

The in-repo `tests/test_nexn2_smoke.py` covers registry/dims/kernel-symbol
availability without the checkpoint.
