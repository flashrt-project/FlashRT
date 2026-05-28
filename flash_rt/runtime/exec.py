"""Thin Python wrapper over the FlashRT execution-contract C ABI.

Dev / research / migration only — the deployment hot path is the C ABI linked
directly by C++/Rust/robot hosts. See docs/exec_contract.md.

Build the native module first:
    cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release
    cmake --build exec/build -j
then make `_flashrt_exec` importable (it is emitted under exec/build/).
"""

from __future__ import annotations

import os
import sys


def _import_native():
    try:
        import _flashrt_exec as _c  # noqa: F401
        return _c
    except ImportError:
        pass
    # Fall back to the standalone build directory next to the repo's exec/.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    candidate = os.path.join(repo, "exec", "build")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
    try:
        import _flashrt_exec as _c  # noqa: F401
        return _c
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Could not import _flashrt_exec. Build it first:\n"
            "  cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build exec/build -j"
        ) from e


_c = _import_native()

Ctx = _c.Ctx
Buffer = _c.Buffer
Graph = _c.Graph
Plan = _c.Plan
Event = _c.Event

# dev/test helpers: allocation-free, capture-safe ops on a raw stream integer.
memset_async = _c.memset_async
memcpy_async = _c.memcpy_async

__all__ = ["Ctx", "Buffer", "Graph", "Plan", "Event", "memset_async", "memcpy_async"]
