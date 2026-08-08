"""Runtime checks for CUDA libraries with model-specific minimum versions."""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys


def _format_cublas_version(version: int) -> str:
    return (
        f"{version // 10000}."
        f"{(version // 100) % 100}."
        f"{version % 100}"
    )


def _packaged_cuda13_dirs() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    override = os.environ.get("FLASH_RT_CUDA13_LIB_DIR")
    if override:
        candidates.append(pathlib.Path(override).expanduser())
    for entry in sys.path:
        if not entry:
            continue
        candidates.append(pathlib.Path(entry) / "nvidia" / "cu13" / "lib")
    found: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_dir():
            found.append(path)
    return found


def _load_cublas13(directory: pathlib.Path | None):
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    if directory is None:
        lt_name = "libcublasLt.so.13"
        blas_name = "libcublas.so.13"
    else:
        lt_name = str(directory / "libcublasLt.so.13")
        blas_name = str(directory / "libcublas.so.13")
        if not (pathlib.Path(lt_name).is_file()
                and pathlib.Path(blas_name).is_file()):
            raise OSError(f"incomplete cuBLAS directory: {directory}")
    ctypes.CDLL(lt_name, mode=mode)
    return ctypes.CDLL(blas_name, mode=mode)


def _cublas_version(library) -> int:
    handle = ctypes.c_void_p()
    status = library.cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
    try:
        version = ctypes.c_int()
        status = library.cublasGetVersion_v2(
            handle, ctypes.byref(version))
        if status != 0:
            raise RuntimeError(
                f"cublasGetVersion_v2 failed with status {status}")
        return int(version.value)
    finally:
        library.cublasDestroy_v2(handle)


def require_cublas13(
    minimum: int,
    *,
    feature: str,
) -> int:
    """Preload packaged cuBLAS 13 and enforce a model-specific minimum.

    Loading happens before ``flash_rt_kernels`` so an installed
    ``nvidia-cublas`` wheel can override an older CUDA-toolkit RUNPATH.
    """
    errors: list[str] = []
    choices: list[pathlib.Path | None] = [
        *_packaged_cuda13_dirs(), None]
    for directory in choices:
        try:
            library = _load_cublas13(directory)
            version = _cublas_version(library)
        except (OSError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        if version >= minimum:
            return version
        found = _format_cublas_version(version)
        needed = _format_cublas_version(minimum)
        raise RuntimeError(
            f"{feature} requires cuBLAS >= {needed}; found {found}. "
            "Install the Motus runtime extra with "
            "`pip install 'flash-rt[motus]'`, or point "
            "FLASH_RT_CUDA13_LIB_DIR at a compatible CUDA 13 library "
            "directory before importing FlashRT."
        )
    raise RuntimeError(
        f"{feature} requires cuBLAS 13, but it could not be loaded: "
        + "; ".join(errors)
    )


__all__ = ["require_cublas13"]
