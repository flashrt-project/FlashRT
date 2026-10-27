"""gfx1201 bindings, including the shared providers' gfx12 dispatch policy."""

from ..rdna35.attention import Rdna35AttentionBackend as Rdna4AttentionBackend
from ..rdna35.gemm import Rdna35GemmBackend as Rdna4GemmBackend

__all__ = ["Rdna4AttentionBackend", "Rdna4GemmBackend"]
