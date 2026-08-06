"""Flash Attention CUTE (CUDA Template Engine) implementation.

FlashRT vendors a forward / SM100-only subset of FlashAttention-4 for Thor
(sm_110). The public entry point lives in ``interface_fwd_sm100`` instead of
the upstream ``interface`` module so that importing this package does NOT pull
in backward, SM80/SM90/SM120, or MLA kernels. The HD256 2CTA forward kernel is
kept for Pi0.5 encoder attention; see ``VENDOR.md``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface_fwd_sm100 import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
