"""Static dim contract for Higgs Audio v3 TTS-4B on RTX SM120.

The backbone is dense Qwen3-4B (all full-attention). The audio head/embedding
is a fused multi-codebook table tied to the embedding weight: ``num_codebooks``
codebooks of ``codebook_vocab`` entries each, predicted per acoustic frame.
"""

from dataclasses import dataclass


@dataclass
class HiggsAudioV3Dims:
    """Authoritative record of the dims the frontend hard-codes.

    Constants are read from the checkpoint's ``config.json`` at load time and
    asserted against these values.
    """

    # Backbone (text_config of HiggsMultimodalQwen3)
    hidden: int = 2560
    num_layers: int = 36
    text_vocab: int = 151_936
    intermediate: int = 9728

    # Attention
    num_q_heads: int = 32
    num_kv_heads: int = 8           # GQA 4:1
    head_dim: int = 128
    rotary_dim: int = 128           # full RoPE (rotary_dim == head_dim)
    rope_theta: float = 1_000_000.0
    max_pos: int = 32_768

    # Norm
    rms_norm_eps: float = 1e-6

    # Audio codec head/embedding
    num_codebooks: int = 8
    codebook_vocab: int = 1026      # includes BOC=1024 / EOC=1025
    sample_rate: int = 24_000
    frame_rate: int = 25            # acoustic frames per second
    boc_id: int = 1024
    eoc_id: int = 1025


class HiggsAudioV3Pipeline:
    """Framework-agnostic placeholder holding the frontend's WeightHandles.

    The actual forward path is hand-written in
    ``flash_rt.frontends.torch.higgs_audio_v3_rtx`` against the
    flash_rt_kernels / flash_rt_fa2 entry points.
    """

    DIMS = HiggsAudioV3Dims()

    def __init__(self, weights) -> None:
        self.weights = weights

    @property
    def num_layers(self) -> int:
        return int(self.weights.ptrs.get("num_layers", self.DIMS.num_layers))

    @property
    def hidden(self) -> int:
        return int(self.weights.ptrs.get("hidden", self.DIMS.hidden))
