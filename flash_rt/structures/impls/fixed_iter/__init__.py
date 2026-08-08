"""Executable fixed-iteration schedule normalization.

The stage catalog describes iterative pipelines semantically.  This package
is the executable bridge for hosts whose Python spelling is not graph-safe:
host-family adapters expose the same ``init -> K * step -> readout`` schedule
as a fixed callable that :func:`flash_rt.structures.capture` can compile and
capture without changing the host repository.
"""

from .protocol import (
    FixedIterationLowering,
    FixedIterationRefused,
    normalize_fixed_iteration,
    register_fixed_iteration_adapter,
)

# Built-ins register at import time.  They use semantic capabilities and
# signatures, never model IDs.
from .openpi import OpenPIFixedIterationAdapter

register_fixed_iteration_adapter(OpenPIFixedIterationAdapter())

__all__ = [
    "FixedIterationLowering",
    "FixedIterationRefused",
    "OpenPIFixedIterationAdapter",
    "normalize_fixed_iteration",
    "register_fixed_iteration_adapter",
]
