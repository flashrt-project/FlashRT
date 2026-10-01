"""FlashRT structure catalog — the contract data every host reads.

A *structure* is a versioned specification of one model region: boundary
tensors, framework-neutral weight slots, calibration points, gates, and a
plain-torch reference that is the gate's ground truth. A *binding* says
where those positions sit on one concrete host, and classifies every
hot-path segment of a pipeline.

This package holds only the specifications and their loaders. It carries
no kernels, no host adapters and no runtime machinery, and is readable
without torch. Its consumers are:

- the native ggml host adapter (FlashRT-llama.cpp), whose qualification
  gates check a binding against the catalog;
- the runtime exporters, which serialise ``BindingSpec.manifest()``;
- the FlashRT-Structures package, which attaches structures onto an
  unmodified PyTorch host (https://github.com/flashrt-project/FlashRT-Structures).

Layout: ``structures/<name>/structure.yaml`` (+ ``reference.py``) and
``bindings/<host>.yaml``. ``structures/README.md`` is the catalog charter.
"""

from flash_rt.catalog.binding import (
    BindingSpec,
    CoverageSegment,
    list_bindings,
    load_binding,
)
from flash_rt.catalog.registry import StructureSpec, list_structures, load

__all__ = [
    "BindingSpec",
    "CoverageSegment",
    "StructureSpec",
    "list_bindings",
    "list_structures",
    "load",
    "load_binding",
]
