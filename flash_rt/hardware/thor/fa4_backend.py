"""Isolated FlashAttention-4 (CuTe-DSL) backend for Thor (sm_110).

FA4's SM100 Blackwell forward kernel runs on Thor when compiled for ``sm_101a``
(the legacy alias of sm_110; the ``sm_110a`` path hits a cutlass-dsl chip bug).
At the LingBot denoise shape (Sq=51, Skv~891, GQA 16/2, HD=128) with
``pack_gqa`` it is ~17% faster than the vendored fmha kernel, cos=1.0, and is
CUDA-graph capture-safe.

This module is the **only** place that imports FA4. It exists to keep the FA4
import isolated from the rest of FlashRT:

- FA4 is vendored under the **private** package name ``flashrt_fa4`` (a trimmed
  ``flash_attn/cute`` forward-only subset, see
  ``csrc/attention/flash_attn_4_src/VENDOR.md``). It is loaded by transiently
  adding the vendor dir to ``sys.path`` and importing ``flashrt_fa4.cute`` —
  the vendor path is removed from ``sys.path`` again right after import, and the
  private name guarantees it never shadows a pip-installed ``flash_attn`` (the
  RTX backends use ``from flash_attn import flash_attn_func``).
- It is an **optional fast path**. If the FA4 runtime deps
  (``nvidia-cutlass-dsl`` + ``quack-kernels``, the ``thor-fa4`` pip extra) are
  missing, every accessor returns ``None`` and the caller falls back to the
  fmha kernel. Set ``LINGBOT_FA4_DEBUG=1`` to print the import error.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# csrc/attention/flash_attn_4_src holds the vendored `flashrt_fa4` package.
# parents: [0]=thor [1]=hardware [2]=flash_rt [3]=<repo root>
_VENDOR_DIR = (
    Path(__file__).resolve().parents[3]
    / "csrc" / "attention" / "flash_attn_4_src"
)

_FUNC = None          # flashrt_fa4.cute.flash_attn_func (autograd wrapper)
_FWD = None           # flashrt_fa4.cute.interface_fwd_sm100._flash_attn_fwd
_TRIED = False
_REASON = "not attempted"


def _load() -> None:
    """Import the vendored FA4 once; cache the entry points and the reason."""
    global _FUNC, _FWD, _TRIED, _REASON
    if _TRIED:
        return
    _TRIED = True

    # Explicit A/B / opt-out switch: FLASHRT_THOR_FA4=0 forces the fmha path.
    if os.environ.get("FLASHRT_THOR_FA4", "1") == "0":
        _REASON = "disabled (FLASHRT_THOR_FA4=0)"
        return

    # Thor target: sm_101a (sm_110's Blackwell alias). Don't clobber a user value.
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_101a")
    os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100a")

    # Allow an override dir (e.g. for local FA4 development), else the vendor.
    src = os.environ.get("LINGBOT_FA4_SRC") or (
        str(_VENDOR_DIR) if _VENDOR_DIR.is_dir() else None
    )
    if src is None:
        _REASON = f"FA4 vendor dir not found at {_VENDOR_DIR}"
        return

    added = src not in sys.path
    if added:
        sys.path.insert(0, src)
    try:
        from flashrt_fa4.cute import flash_attn_func
        from flashrt_fa4.cute.interface_fwd_sm100 import _flash_attn_fwd
        _FUNC = flash_attn_func
        _FWD = _flash_attn_fwd
        _REASON = "active"
    except Exception as exc:  # missing deps / arch issue -> fmha fallback
        if os.environ.get("LINGBOT_FA4_DEBUG"):
            import traceback
            traceback.print_exc()
        _REASON = f"{type(exc).__name__}: {exc}"
        _FUNC = None
        _FWD = None
    finally:
        # Do NOT leave the vendor path lingering at sys.path[0]; the module is
        # already cached in sys.modules under `flashrt_fa4`.
        if added:
            try:
                sys.path.remove(src)
            except ValueError:
                pass


def is_available() -> bool:
    """True if the FA4 fast path loaded (deps present, import succeeded)."""
    _load()
    return _FUNC is not None


def status() -> str:
    """One-line human-readable status (``active`` or the failure reason)."""
    _load()
    return _REASON


def fa4_func():
    """The autograd ``flash_attn_func`` (forward inference), or ``None``."""
    _load()
    return _FUNC


def fa4_fwd():
    """The internal ``_flash_attn_fwd`` (exposes ``seqused_k``), or ``None``."""
    _load()
    return _FWD
