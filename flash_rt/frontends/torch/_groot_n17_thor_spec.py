"""The GR00T N1.7 checkpoint description, under the name it was first given.

The spec describes the checkpoint rather than any one backend, and Thor, RTX,
SM89 and Ascend all read it, so it now lives at
``flash_rt.models.groot_n17.weight_spec``. This module stays so that every
existing import keeps working unchanged; new code should import the model layer
directly.
"""

from __future__ import annotations

from flash_rt.models.groot_n17.weight_spec import WEIGHT_SPEC, build_spec

__all__ = ["WEIGHT_SPEC", "build_spec"]
