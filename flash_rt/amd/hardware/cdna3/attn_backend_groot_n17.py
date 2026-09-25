"""CDNA3 name for the GROOT N1.7 AITER attention backend."""

from flash_rt.amd.hardware.cdna4.attn_backend_groot_n17 import Cdna4GrootN17AttnBackend


class Cdna3GrootN17AttnBackend(Cdna4GrootN17AttnBackend):
    """GROOT N1.7 AITER attention backend used on gfx942."""
