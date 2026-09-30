"""CDNA3 name for the AITER attention backend."""

from flash_rt.amd.hardware.cdna4.attn_backend_aiter import Cdna4AiterAttnBackend


class Cdna3AiterAttnBackend(Cdna4AiterAttnBackend):
    """AITER attention backend used on gfx942."""
