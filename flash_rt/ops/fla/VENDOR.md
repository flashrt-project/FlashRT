# FlashRT FLA Subset Vendor Notes

`flash_rt/ops/fla/` is a minimal Python/Triton subset used by the
Qwen3.6 Gated DeltaNet long-prefill path. It is vendored to avoid a
runtime dependency on the full FLA or SGLang packages.

## Source Baselines

| Upstream | Commit | License |
|---|---|---|
| [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) | `abfa403de2146b9a2ab762a603f8fdb61cc3c166` | MIT |
| [`sgl-project/sglang`](https://github.com/sgl-project/sglang) | `93fa577bb95a37699e7f1f56a486e436d5792b71` | Apache-2.0 |

See [`LICENSE.flash-linear-attention`](LICENSE.flash-linear-attention)
and [`NOTICE`](NOTICE) in this directory.

## Imported Subset

| Local file | Upstream role |
|---|---|
| `chunk.py` | `fla/ops/gated_delta_rule/chunk.py` plus SGLang Qwen integration shape |
| `wy_fast.py` | `fla/ops/gated_delta_rule/wy_fast.py` |
| `chunk_delta_h.py` | `fla/ops/common/chunk_delta_h.py` |
| `chunk_o.py` | `fla/ops/common/chunk_o.py` |
| `chunk_scaled_dot_kkt.py` | `fla/ops/common/chunk_scaled_dot_kkt.py` |
| `cumsum.py` | `fla/ops/utils/cumsum.py` |
| `index.py` | `fla/ops/utils/index.py` |
| `l2norm.py` | `fla/modules/l2norm.py` |
| `op.py` | `fla/ops/utils/op.py` |
| `solve_tril.py` | `fla/ops/utils/solve_tril.py` |
| `utils.py` | `fla/utils.py` |

Every source file keeps an `Adapted from ...` header with the original
upstream path.

## FlashRT Changes

- Namespace imports are redirected from upstream package names to
  `flash_rt.ops.fla`.
- Only inference-forward pieces needed by Qwen3.6 Gated DeltaNet are
  retained; unrelated training/autograd surface is not vendored.
- Several entry points accept caller-owned output buffers so long
  prefill can reuse FlashRT-managed memory and avoid per-call allocation.
- Optional final-state and diagnostic outputs are suppressed on the hot
  path unless explicitly requested by the Qwen frontend.
- The subset is intentionally Python/Triton-only. CUDA/CUTLASS kernels
  that are production hot paths live under `csrc/`.

## Runtime Dependencies

This subset imports `torch`, `triton`, `einops`, and `packaging`.
They are declared by the `flash-rt[torch]` extra in `pyproject.toml`.
