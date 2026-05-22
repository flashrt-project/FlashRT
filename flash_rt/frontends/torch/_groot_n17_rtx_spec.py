"""FlashRT -- GROOT N1.7 weight spec for the RTX torch frontend.

The N1.7 checkpoint tensor layout is hardware-independent.  Keep an RTX
module entry point so the frontend can depend on an RTX-named spec without
importing Thor frontend code, while reusing the single declarative spec that
already covers all N1.7 safetensors keys.
"""

from __future__ import annotations

from flash_rt.frontends.torch._groot_n17_thor_spec import WEIGHT_SPEC, build_spec

__all__ = ["WEIGHT_SPEC", "build_spec"]
