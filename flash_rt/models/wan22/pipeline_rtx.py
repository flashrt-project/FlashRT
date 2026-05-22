"""Wan2.2 RTX shape constants.

The official Wan2.2 route exposes the upstream Python pipeline through
``Wan22TorchFrontendRtx``. These constants document the DiT attention
surface used by the RTX frontend:

* ``wan_self``: 30 layers, MHA, 24 heads, head_dim 128.
* ``wan_cross``: 30 layers, MHA, 24 heads, head_dim 128, KV text length 512.
"""

WAN22_NUM_LAYERS = 30
WAN22_NUM_HEADS = 24
WAN22_HEAD_DIM = 128
WAN22_TEXT_LEN = 512
WAN22_VAE_STRIDE = (4, 16, 16)
WAN22_PATCH_SIZE = (1, 2, 2)


def latent_token_count(width: int, height: int, frames: int) -> int:
    """Return Wan2.2 DiT token count for a generated video shape."""
    latent_t = (int(frames) - 1) // WAN22_VAE_STRIDE[0] + 1
    latent_h = int(height) // WAN22_VAE_STRIDE[1]
    latent_w = int(width) // WAN22_VAE_STRIDE[2]
    return (
        latent_t
        * (latent_h // WAN22_PATCH_SIZE[1])
        * (latent_w // WAN22_PATCH_SIZE[2])
    )
