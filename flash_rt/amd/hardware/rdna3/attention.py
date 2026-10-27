"""gfx1100 attention policy over the shared wave32 implementation."""

from ..rdna35.attention import Rdna35AttentionBackend


class Rdna3AttentionBackend(Rdna35AttentionBackend):
    def __init__(self, kernels):
        super().__init__(kernels)
        # Dense prefixes use SDPA; padded prefixes retain native attention.
        self.hip_dense_encoder_gqa = False
