# SPDX-License-Identifier: Apache-2.0
# Framework-decoupling shim for the vendored MiniMax-M3 sparse-attention Triton
# kernels. The upstream SGLang modules pull two tiny things from the framework:
#
#   from sglang.srt.utils import is_hip        -> HIP/ROCm detection (fp8 KV path)
#   from sglang.srt.environ import envs        -> a feature flag for an optional
#                                                 JIT-CUDA radix top-k fast path
#
# Both are replaced here so the kernels import with only torch + triton. The
# defaults select the pure-Triton CUDA path (no HIP fp8 widening, no jit_kernel
# radix select), which is what we want on sm_121 (GB10).


def is_hip() -> bool:
    """Return True on AMD/ROCm builds. Always False here (we target CUDA sm_121).

    Used by the decode/prefill topk_sparse kernels to gate the `IS_FP8` paged-KV
    widening branch (HIP-only). With this False, the CUDA dtype contract
    (K/V cache dtype == Q dtype) is enforced exactly as upstream on CUDA.
    """
    try:
        import torch

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


class _Flag:
    """Minimal stand-in for an sglang.srt.environ boolean flag (always off)."""

    def __init__(self, default: bool = False):
        self._default = default

    def get(self) -> bool:
        return self._default


class _Envs:
    """Stub for `sglang.srt.environ.envs`.

    Only `SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX` is referenced by the decode
    indexer. Keeping it False routes top-k selection through the 2-stage Triton
    fallback (no `sglang.jit_kernel.minimax_decode_topk` CUDA import), so the
    decode indexer stays pure-Triton.
    """

    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = _Flag(False)


envs = _Envs()
