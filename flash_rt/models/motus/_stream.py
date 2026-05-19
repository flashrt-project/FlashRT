"""Tiny helper for fvk wrappers to use the current CUDA stream.

In normal eager execution, ``torch.cuda.current_stream().cuda_stream`` is
the default stream's handle (= 0 on the legacy default stream). Inside
``torch.cuda.graph(...)`` capture, it's the capture stream — passing 0
to fvk would emit work on the wrong stream and the graph would capture
nothing (encountered at G5 smoke test). Always route fvk via this
helper so the same wrapper code works inside and outside capture.

The helper is a function call rather than a cached attribute because
the current stream IS context-dependent: torch.cuda.graph() swaps the
thread's current stream on entry and restores on exit. A closure that
captured the install-time stream would still emit on the original
stream during capture.
"""

import torch


def cs() -> int:
    """Current cuda stream handle as an int (suitable for fvk stream=)."""
    return torch.cuda.current_stream().cuda_stream
