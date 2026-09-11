"""
FlashRT — High-performance VLA inference engine.

Public exports (stable API — see ``docs/stable_api.md``):

    flash_rt.load_model(...)   → VLAModel
    flash_rt.VLAModel          — unified inference wrapper

Supported models: Pi0.5, Pi0, Pi0-FAST, GROOT N1.6, GROOT N1.7.
Supported hardware: Jetson Thor (SM110), RTX 5090 (SM120), RTX 4090
(SM89), AMD Instinct MI350 series (ROCm gfx950, pi05).

Extending with new models: see ``docs/plugin_model_template.md``.

Usage::

    import flash_rt

    model = flash_rt.load_model(
        checkpoint="/path/to/checkpoint",
        framework="torch",
        autotune=3,
    )

    actions = model.predict(images=[base_img, wrist_img],
                            prompt="pick up the red block")
"""

__version__ = "0.2.0"

# ── Windows: register CUDA / cuDNN DLL search paths ──
# Python 3.8+ on Windows ignores PATH for C-extension dependencies
# (security hardening). The compiled .pyd needs cudart64_*.dll,
# cublas64_*.dll, cublasLt, cudnn — we add their canonical install
# directories to the secure DLL loader so `import flash_rt` works
# without the user pre-loading them. Linux is unaffected: this whole
# block is skipped via the sys.platform guard.
import os as _os
import sys as _sys
if _sys.platform == 'win32':
    _cuda_roots = [
        _os.environ.get('CUDA_PATH'),
        _os.environ.get('CUDA_PATH_V13_0'),
        _os.environ.get('CUDA_PATH_V12_9'),
        _os.environ.get('CUDA_PATH_V12_8'),
        _os.environ.get('CUDA_PATH_V12_4'),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
        _os.environ.get('CUDNN_PATH'),
    ]
    _seen = set()
    for _root in filter(None, _cuda_roots):
        for _sub in ('bin', r'extras\CUPTI\lib64', ''):
            _p = _os.path.join(_root, _sub) if _sub else _root
            if _p in _seen:
                continue
            _seen.add(_p)
            if _os.path.isdir(_p):
                try:
                    _os.add_dll_directory(_p)
                except (OSError, ValueError):
                    pass
    del _root, _sub, _p, _seen, _cuda_roots
del _os, _sys

from flash_rt import _extensions as _ext  # noqa: E402

__all__ = ["load_model", "VLAModel", "catalog"]


def __getattr__(name):
    """PEP 562. ``import flash_rt`` stays free of torch and of the
    compiled extensions.

    The structure catalog is usable without either, and a consumer that
    only wants ``flash_rt.catalog`` should not pay for the VLA API to
    reach it. Naming ``load_model`` still loads everything it needs.

    ``flash_rt.structures`` moved to its own distribution
    (``flashrt-structures``); asking for it here answers with that
    pointer rather than an AttributeError.

    An extension name reaching here means the import machinery did not
    find it beside the package — this distribution ships no ``.so`` — so
    answer with the build instructions rather than an AttributeError.
    """
    if name in ("load_model", "VLAModel"):
        from flash_rt import api
        return getattr(api, name)
    if name == "catalog":
        import flash_rt.catalog as mod
        return mod
    if name == "structures":
        raise ImportError(
            "flash_rt.structures moved to the flashrt-structures "
            "distribution: pip install flashrt-structures, then "
            "`import flashrt_structures as structures`. The structure "
            "catalog itself stayed here as flash_rt.catalog. "
            "See https://github.com/flashrt-project/FlashRT-Structures")
    if name in _ext.EXTENSIONS:
        return _ext.require(name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
