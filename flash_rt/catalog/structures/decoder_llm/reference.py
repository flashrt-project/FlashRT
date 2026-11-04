"""Plain-torch reference for ``decoder_llm``.

The parity ground truth is the host's own eager stack forward: the
reference is the module itself, called with its native interface
(``inputs_embeds=...``). No standalone torch re-implementation exists
because the structure's boundary is the whole stack — decomposing it
here would be a second host, not a reference.
"""

from __future__ import annotations

import torch


def decoder_llm_ref(module: torch.nn.Module,
                    inputs_embeds: torch.Tensor, **kwargs) -> torch.Tensor:
    """Run the host stack eagerly; returns ``hidden_states``."""
    out = module(inputs_embeds=inputs_embeds, **kwargs)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out
