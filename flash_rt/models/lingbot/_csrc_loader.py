"""Access the LingBot-VLA model-specific CUDA kernels.

The LingBot pipeline uses a handful of model-specific kernels (fused
AdaRMSNorm, SwiGLU tail, QKV+RoPE). They are compiled INTO the shared
``flash_rt_kernels`` module (``csrc/kernels/lingbot_*.cu``, gated by
``ENABLE_LINGBOT`` for SM100-class GPUs) with ``lingbot_``-prefixed pybind
names — there is no separate ``flash_rt_lingbot`` .so anymore. This matches
how the qwen36 model kernels live inside ``flash_rt_kernels``.

``get_lingbot_ext()`` returns a thin view over ``flash_rt_kernels`` that maps
the un-prefixed call names used by ``kernel_ops.py`` (e.g.
``silu_mul_merged_fp8_mpad_fp16in``) onto the prefixed symbols
(``lingbot_silu_mul_merged_fp8_mpad_fp16in``), so callers need no changes.
"""

from __future__ import annotations

import threading

_EXT = None
_LOCK = threading.Lock()


class _LingbotKernels:
    """Maps ``ext.<name>`` -> ``flash_rt_kernels.lingbot_<name>``."""

    def __init__(self, module):
        self._m = module

    def __getattr__(self, name):
        try:
            return getattr(self._m, "lingbot_" + name)
        except AttributeError as e:
            raise AttributeError(
                f"flash_rt_kernels has no LingBot symbol 'lingbot_{name}'. "
                "Rebuild flash_rt_kernels for an SM100-class GPU (Thor sm_110a):\n"
                "    cmake -B build -S . -DGPU_ARCH=110\n"
                "    cmake --build build -j --target flash_rt_kernels\n"
                "(LingBot kernels are gated behind ENABLE_LINGBOT.)"
            ) from e


def get_lingbot_ext():
    """Return a view over ``flash_rt_kernels`` exposing the LingBot kernels."""
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        try:
            import flash_rt.flash_rt_kernels as k
        except ImportError as e:
            raise ImportError(
                "flash_rt_kernels is not built. Build it for an SM100-class GPU "
                "(Thor sm_110a):\n"
                "    cmake -B build -S . -DGPU_ARCH=110\n"
                "    cmake --build build -j --target flash_rt_kernels"
            ) from e
        if not hasattr(k, "lingbot_silu_mul_merged_fp8_mpad_fp16in"):
            raise ImportError(
                "flash_rt_kernels was built without the LingBot kernels "
                "(ENABLE_LINGBOT). Configure with -DGPU_ARCH=110 (SM100-class) "
                "and rebuild the flash_rt_kernels target."
            )
        _EXT = _LingbotKernels(k)
    return _EXT


__all__ = ["get_lingbot_ext"]
