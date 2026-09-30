"""CDNA3 name for the architecture-neutral torch SDPA backend."""

from flash_rt.amd.hardware.cdna4.attn_backend import Cdna4AttnBackend


class Cdna3AttnBackend(Cdna4AttnBackend):
    """Torch SDPA attention storage/backend used on gfx942."""
