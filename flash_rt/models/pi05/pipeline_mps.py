"""Bind Apple MPS providers to the shared Pi0.5 BF16 model traversal."""

from flash_rt.hardware.mps.attention import MpsAttentionBackend
from flash_rt.hardware.torch_reference import TorchGemmBackend, TorchTensorOps
from flash_rt.models.pi05.torch_pipeline import Pi05TorchPipeline


class Pi05PipelineMps(Pi05TorchPipeline):
    def __init__(self, weights, *, num_views=2, max_prompt_len=200,
                 chunk_size=10, num_steps=10):
        super().__init__(
            weights, TorchTensorOps(), TorchGemmBackend(), MpsAttentionBackend(),
            num_views=num_views, max_prompt_len=max_prompt_len,
            chunk_size=chunk_size, num_steps=num_steps, device="mps",
        )
