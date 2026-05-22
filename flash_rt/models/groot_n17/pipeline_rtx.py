"""FlashRT -- GROOT N1.7 RTX pointer forwards.

The RTX frontend uses this module as its model-side import boundary while
sharing the N1.7 DiT pointer implementation.
"""

from __future__ import annotations

from flash_rt.models.groot_n17.pipeline_thor import dit_forward

__all__ = ["dit_forward"]
