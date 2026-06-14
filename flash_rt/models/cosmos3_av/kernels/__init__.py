"""Loader for the Cosmos3-AV model-local kernel extension.

Imports the precompiled `cosmos3_av_kernels` .so built by setup.py (build_ext
--inplace). These 4 kernels are model-specific and intentionally NOT in the shared
flash_rt_kernels.so (docs/adding_new_model.md §4.3/§4.5).

Exposes the same callable surface the cosmos pipeline uses:
    fp4_silu_aux, sage_gqa_d128, sage_gqa_f8_d128, qk_norm_rope
"""
import os

_BUILD_HINT = (
    "cosmos3_av_kernels extension not built. Build it once on the target GPU:\n"
    "  cd flash_rt/models/cosmos3_av/kernels && python3 setup.py build_ext --inplace"
)

try:
    from . import cosmos3_av_kernels as _ext  # the built .so sits in this package
except ImportError as e:  # pragma: no cover - surfaced to the user with a fix
    raise ImportError(f"{_BUILD_HINT}\n(original error: {e})") from e

fp4_silu_aux = _ext.fp4_silu_aux
sage_gqa_d128 = _ext.sage_gqa_d128
sage_gqa_f8_d128 = _ext.sage_gqa_f8_d128
qk_norm_rope = _ext.qk_norm_rope

__all__ = ["fp4_silu_aux", "sage_gqa_d128", "sage_gqa_f8_d128", "qk_norm_rope"]
