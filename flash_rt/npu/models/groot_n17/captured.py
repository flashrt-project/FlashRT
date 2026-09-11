"""One captured action chain at a fixed backbone-token count.

Every tensor the graph reads is allocated here and refilled with ``copy_``
between frames, so the captured sequence always sees the same addresses. The
two cross-attention key masks are frame-independent -- they are a function of
where the prompt's image tokens sit in the sequence -- so they are built when
the prompt is set and then read, not rebuilt.
"""

from __future__ import annotations

import torch

from flash_rt.npu.core.npu_graph import NpuGraph
from flash_rt.npu.models.groot_n17 import pipeline as pl


class CapturedChain:
    """Backbone features and a noise draw in, an action trajectory out."""

    def __init__(self, bound: pl.BoundChain, tokens: int, *, batch: int = 1,
                 state_history: int = 1):
        if batch != 1:
            raise NotImplementedError(
                "the Ascend GR00T N1.7 chain serves one observation per call")
        self.bound = bound
        self.tokens = int(tokens)
        device = bound.device
        self.features = torch.zeros(batch, self.tokens, pl.BACKBONE_DIM,
                                    dtype=torch.bfloat16, device=device)
        self.state = torch.zeros(batch, state_history, pl.STATE_DIM,
                                 dtype=torch.bfloat16, device=device)
        self.noise = torch.zeros(batch, bound.horizon, pl.ACTION_DIM,
                                 dtype=torch.bfloat16, device=device)
        self.text_mask = torch.zeros(bound.horizon + state_history, self.tokens,
                                     dtype=torch.bool, device=device)
        self.image_mask = torch.zeros_like(self.text_mask)
        self.actions = torch.zeros(batch, bound.horizon, pl.ACTION_DIM,
                                   dtype=torch.bfloat16, device=device)
        self._graph = None

    # ------------------------------------------------------------------
    def set_prompt(self, image_mask: torch.Tensor, attention_mask: torch.Tensor):
        """Freeze the two key masks for a prompt's token layout."""
        text, image = pl.attention_masks(
            image_mask.reshape(-1).to(self.text_mask.device),
            attention_mask.reshape(-1).to(self.text_mask.device).bool(),
            self.text_mask.shape[0])
        self.text_mask.copy_(text)
        self.image_mask.copy_(image)

    def fill(self, features: torch.Tensor, state: torch.Tensor,
             noise: torch.Tensor) -> None:
        self.features.copy_(features.reshape(self.features.shape))
        self.state.copy_(state.reshape(self.state.shape))
        self.noise.copy_(noise.reshape(self.noise.shape))

    def _run(self) -> torch.Tensor:
        vl = pl.encode_backbone_features(self.bound, self.features)
        state_features = pl.encode_state(self.bound, self.state)
        return pl.denoise(self.bound, vl, state_features,
                          (self.text_mask, self.image_mask), self.noise)

    def capture(self) -> None:
        """Warm up on the default stream, then capture on a dedicated one.

        The order matters: warming up inside the capture stream leaves a stream
        the capture never joins, and CANN refuses to end such a capture.
        """
        with torch.no_grad():
            for _ in range(3):
                self._run()
        torch.npu.synchronize()
        graph = NpuGraph(stream=torch.npu.Stream())
        with torch.no_grad():
            with graph:
                self.actions = self._run()
        torch.npu.synchronize()
        self._graph = graph

    def replay(self) -> torch.Tensor:
        if self._graph is None:
            raise RuntimeError("the action chain graph has not been captured")
        self._graph.replay()
        return self.actions

    def run_eager(self) -> torch.Tensor:
        with torch.no_grad():
            return self._run()
