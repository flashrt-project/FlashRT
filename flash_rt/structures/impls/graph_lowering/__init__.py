"""Whole-graph shape-lowering adapters, one per host family.

``capture`` consults this registry when it is handed a model: every
adapter that recognizes the host pins that family's shape glue for the
fixed request, and hands back an ``undo``. A host no family recognizes
is captured as-is — correct for hosts that are already graph-safe.
"""

from .protocol import (GraphLowering, GraphLoweringRefused,
                       lower_for_capture,
                       register_graph_lowering_adapter)
from .qwen3_vl import Qwen3VLGraphLoweringAdapter

# Built-ins register at import time; they recognize by capability, not
# by class name or version string.
register_graph_lowering_adapter(Qwen3VLGraphLoweringAdapter())

from .pi052_denoise import Pi05DenoiseGraphLoweringAdapter  # noqa: E402

register_graph_lowering_adapter(Pi05DenoiseGraphLoweringAdapter())

__all__ = [
    "GraphLowering",
    "GraphLoweringRefused",
    "lower_for_capture",
    "register_graph_lowering_adapter",
]
