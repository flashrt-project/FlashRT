"""FlashRT -- ROCm Qwen3 dense pipeline contract.

This is the AMD counterpart to :mod:`flash_rt.models.qwen3.pipeline_rtx`.
The current ROCm frontend uses the official BF16 Hugging Face model as a
correctness baseline; this module fixes the owned-buffer/kernel pipeline shape
that subsequent ROCm stages should implement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3RocmDims:
    """Static dimension contract for official Qwen3-8B BF16 on ROCm."""

    hidden: int = 4096
    num_layers: int = 36
    vocab_size: int = 151_936
    intermediate: int = 12_288

    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rotary_dim: int = 128
    rope_theta: float = 1_000_000.0
    max_pos: int = 40_960

    rms_norm_eps: float = 1.0e-6


class Qwen3PipelineRocm:
    """Placeholder for the Qwen3-8B ROCm owned-buffer pipeline.

    Stage-1 ROCm serving is implemented by
    :class:`flash_rt.frontends.torch.qwen3_rocm.Qwen3TorchFrontendRocm`, which
    loads the official BF16 checkpoint through Hugging Face and establishes the
    correctness baseline.

    This class intentionally contains only the static contract until the BF16
    hipBLASLt/attention/kernel path is ported. Keeping the contract separate
    mirrors the RTX layout and gives tests/tools a stable import target.
    """

    DIMS = Qwen3RocmDims()

    def __init__(self, weights=None) -> None:
        self.weights = weights

    @property
    def num_layers(self) -> int:
        if self.weights is not None and hasattr(self.weights, "ptrs"):
            return int(self.weights.ptrs.get("num_layers", self.DIMS.num_layers))
        return self.DIMS.num_layers

    @property
    def hidden(self) -> int:
        if self.weights is not None and hasattr(self.weights, "ptrs"):
            return int(self.weights.ptrs.get("hidden", self.DIMS.hidden))
        return self.DIMS.hidden

    @property
    def supports_owned_bf16(self) -> bool:
        if self.weights is None or not hasattr(self.weights, "ptrs"):
            return False
        return self.weights.ptrs.get("quant_format") == "bf16"

    @property
    def supports_static_fp8(self) -> bool:
        if self.weights is None or not hasattr(self.weights, "ptrs"):
            return False
        return "fp8" in self.weights.ptrs
