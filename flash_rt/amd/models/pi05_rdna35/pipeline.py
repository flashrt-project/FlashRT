"""Compatibility binding of the shared Pi0.5 tensor pipeline to RDNA."""
from __future__ import annotations
import os
import torch
from flash_rt.models.pi05.torch_pipeline import (
    Pi05TorchPipeline, VIS_L, VIS_D, VIS_H, VIS_NH, VIS_HD,
    ENC_L, ENC_D, ENC_H, ENC_NH, ENC_HD,
    DEC_L, DEC_D, DEC_H, DEC_NH, DEC_HD, PATCHES_PER_VIEW, ACTION_DIM,
)
from flash_rt.amd.hardware.rdna35.ops import Rdna35TensorOps


class Pi05PipelineRdna35(Pi05TorchPipeline):
    def __init__(
        self, weights, kernels, gemm, attn, *, num_views=3,
        max_prompt_len=200, chunk_size=10, num_steps=10,
        dtype=torch.bfloat16,
    ):
        super().__init__(
            weights, Rdna35TensorOps(kernels, dtype), gemm, attn,
            num_views=num_views, max_prompt_len=max_prompt_len,
            chunk_size=chunk_size, num_steps=num_steps, dtype=dtype,
            precompute_modulation=os.getenv("FLASHRT_RDNA35_PRECOMPUTE_MODULATION", "1") == "1",
            compact_encoder=os.getenv("FLASHRT_RDNA35_COMPACT_ENCODER", "1") == "1",
        )
