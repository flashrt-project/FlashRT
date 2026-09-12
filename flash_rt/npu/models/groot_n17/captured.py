"""One captured action chain, at a fixed prompt token layout.

Every tensor the graph reads is allocated here and refilled with ``copy_``
between frames, so the captured sequence always sees the same addresses.

The token layout is part of the shape the graph is built for. A prompt fixes
how many of the backbone's tokens are language and how many are image, and the
DiT's cross-attention layers read one class each, so those two counts appear in
the buffer shapes exactly the way the token count does. A different prompt
length builds a different runner, which is the same rule the Pi0.5 port applies
to its prompt buckets.
"""

from __future__ import annotations

import torch

from flash_rt.npu.core.npu_graph import NpuGraph
from flash_rt.npu.models.groot_n17 import backbone as bb
from flash_rt.npu.models.groot_n17 import pipeline as pl


class CapturedChain:
    """Backbone features and a noise draw in, an action trajectory out."""

    def __init__(self, bound: pl.BoundChain, image_mask: torch.Tensor,
                 attention_mask: torch.Tensor, *, batch: int = 1,
                 state_history: int = 1):
        if batch != 1:
            raise NotImplementedError(
                "the Ascend GR00T N1.7 chain serves one observation per call")
        self.bound = bound
        device = bound.device
        text_index, image_index = pl.token_partition(
            image_mask.to(device), attention_mask.to(device))
        self.text_index = text_index.to(torch.int32).to(device)
        self.image_index = image_index.to(torch.int32).to(device)
        self.tokens = int(image_mask.reshape(-1).shape[0])
        self.text_tokens = int(self.text_index.numel())
        self.image_tokens = int(self.image_index.numel())

        self.features = torch.zeros(batch, self.tokens, pl.BACKBONE_DIM,
                                    dtype=torch.bfloat16, device=device)
        self.state = torch.zeros(batch, state_history, pl.STATE_DIM,
                                 dtype=torch.bfloat16, device=device)
        self.noise = torch.zeros(batch, bound.horizon, pl.ACTION_DIM,
                                 dtype=torch.bfloat16, device=device)
        self.actions = torch.zeros(batch, bound.horizon, pl.ACTION_DIM,
                                   dtype=torch.bfloat16, device=device)
        self._graph = None

    # ------------------------------------------------------------------
    def fill(self, features: torch.Tensor, state: torch.Tensor,
             noise: torch.Tensor) -> None:
        self.features.copy_(features.reshape(self.features.shape))
        self.state.copy_(state.reshape(self.state.shape))
        self.noise.copy_(noise.reshape(self.noise.shape))

    def _run(self) -> torch.Tensor:
        vl = pl.encode_backbone_features(self.bound, self.features)
        text = torch.index_select(vl, 1, self.text_index)
        image = torch.index_select(vl, 1, self.image_index)
        state_features = pl.encode_state(self.bound, self.state)
        return pl.denoise(self.bound, text, image, state_features, self.noise)

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


class CapturedFrame:
    """The whole served frame in one captured graph.

    Everything that depends only on the prompt is resolved when the graph is
    built and then only read: the rope tables, the causal extent, where the
    image tokens sit in the sequence, and the language embeddings of the
    instruction. What is left varying per frame is the camera -- the patch
    features -- plus the robot state and the noise draw, and those are the only
    buffers a frame refills.

    That is the whole point of building it this way. The eager frame spends
    about 70% of its wall on the host; a graph with three inputs has no host in
    it at all.
    """

    def __init__(self, backbone: bb.BoundBackbone, chain: pl.BoundChain, aux: dict,
                 *, state_history: int = 1):
        device = chain.device
        self.backbone = backbone
        self.chain = chain
        self.views = int(aux["views"])

        image_mask = aux["image_mask"].reshape(-1).to(device)
        attention_mask = aux["attention_mask"].reshape(-1).to(device)
        text_index, image_index = pl.token_partition(image_mask, attention_mask)
        self.text_index = text_index.to(torch.int32)
        self.image_index = image_index.to(torch.int32)
        self.visual_index = torch.nonzero(image_mask, as_tuple=False).reshape(-1).to(
            torch.int32)
        self.tokens = int(image_mask.numel())

        # Prompt constants.
        self.vit_cos = aux["vit_cos"].to(device).to(torch.bfloat16).contiguous()
        self.vit_sin = aux["vit_sin"].to(device).to(torch.bfloat16).contiguous()
        self.llm_cos = aux["llm_cos"].to(device).to(torch.bfloat16).contiguous()
        self.llm_sin = aux["llm_sin"].to(device).to(torch.bfloat16).contiguous()
        self.causal = bb.causal_mask(self.tokens, device)
        # The language embeddings of the instruction, zero where an image token
        # will be written. The image half is refilled per frame; this half never
        # changes, so the buffer is the sum by construction.
        self.embeds = aux["text_embeds"].to(device).to(torch.bfloat16).contiguous()

        patch_tokens, patch_dim = aux["patch_shape"]
        if patch_dim != bb.PATCH_DIM:
            raise ValueError(
                f"the patch projection consumes {bb.PATCH_DIM}-wide voxels, "
                f"got {patch_dim}")
        self.patches = torch.zeros(patch_tokens, patch_dim, dtype=torch.bfloat16,
                                   device=device)
        # The interpolated position table depends only on the image grid.
        self.positions = aux["patch_positions"].to(device).to(torch.bfloat16).contiguous()
        self.state = torch.zeros(1, state_history, pl.STATE_DIM,
                                 dtype=torch.bfloat16, device=device)
        self.noise = torch.zeros(1, chain.horizon, pl.ACTION_DIM,
                                 dtype=torch.bfloat16, device=device)
        self.actions = torch.zeros(1, chain.horizon, pl.ACTION_DIM,
                                   dtype=torch.bfloat16, device=device)
        self._graph = None

    # ------------------------------------------------------------------
    def fill(self, patches: torch.Tensor, state: torch.Tensor,
             noise: torch.Tensor) -> None:
        self.patches.copy_(patches.reshape(self.patches.shape))
        self.state.copy_(state.reshape(self.state.shape))
        self.noise.copy_(noise.reshape(self.noise.shape))

    def _run(self) -> torch.Tensor:
        features = bb.patch_project(self.backbone, self.patches, self.positions)
        merged, taps = bb.vision(self.backbone, features, self.vit_cos,
                                 self.vit_sin, self.views)
        self.embeds.reshape(-1, bb.LLM_DIM).index_copy_(0, self.visual_index, merged)
        features = bb.language(self.backbone, self.embeds, self.llm_cos, self.llm_sin,
                               self.visual_index, taps, self.causal)
        vl = pl.encode_backbone_features(self.chain, features)
        text = torch.index_select(vl, 1, self.text_index)
        image = torch.index_select(vl, 1, self.image_index)
        state_features = pl.encode_state(self.chain, self.state)
        return pl.denoise(self.chain, text, image, state_features, self.noise)

    def capture(self) -> None:
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
            raise RuntimeError("the frame graph has not been captured")
        self._graph.replay()
        return self.actions

    def run_eager(self) -> torch.Tensor:
        with torch.no_grad():
            return self._run()
