"""Loader for the Cosmos3-video model-local kernel extension.

Imports the precompiled `cosmos3_video_kernels` .so built by setup.py (build_ext
--inplace). This kernel is model-specific and intentionally NOT in the shared
flash_rt_kernels.so (docs/adding_new_model.md §4.3/§4.5).

Exposes the callable the cosmos3_video pipeline uses:
    qk_norm_rope
"""
_BUILD_HINT = (
    "cosmos3_video_kernels extension not built. Build it once on the target GPU:\n"
    "  cd flash_rt/models/cosmos3_video/kernels && python3 setup.py build_ext --inplace"
)

try:
    from . import cosmos3_video_kernels as _ext  # the built .so sits in this package
except ImportError as e:  # pragma: no cover - surfaced to the user with a fix
    raise ImportError(f"{_BUILD_HINT}\n(original error: {e})") from e

qk_norm_rope = _ext.qk_norm_rope

__all__ = ["qk_norm_rope"]
