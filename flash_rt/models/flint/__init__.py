"""Flint (RecursiveSI/flint) official NVFP4 GGUF on RTX SM120.

Native W4A16 pipeline: an independent GGUF reader and a lossless NVFP4 repack
feed FlashRT kernels; the steady-state decode step is one captured CUDA graph.
See ``docs/flint_usage.md``.
"""

from flash_rt.models.flint.config import FlintConfig, FlintConfigError, load_config
from flash_rt.models.flint.gguf import GGUFFile

__all__ = ["FlintConfig", "FlintConfigError", "GGUFFile", "load_config", "FlintRuntime"]


def __getattr__(name: str):
    """Import ``FlintRuntime`` on first use (PEP 562).

    The runtime module needs the SM120-gated ``flash_rt_flint`` extension;
    binding it lazily keeps config parsing and GGUF inspection importable on
    machines without the build. The runtime constructor raises the clear
    refusal instead.
    """
    if name == "FlintRuntime":
        from flash_rt.models.flint.pipeline_rtx import FlintRuntime
        return FlintRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
