"""Flash Attention CUTE (CUDA Template Engine) implementation.

FlashRT vendors a forward / SM100-only subset of FlashAttention-4 for Thor
(sm_110). The public entry point lives in ``interface_fwd_sm100`` instead of
the upstream ``interface`` module so that importing this package does NOT pull
in the backward, SM80/SM90/SM120, MLA, or head_dim=256 2CTA kernels (all
removed -- see ``VENDOR.md``).
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
