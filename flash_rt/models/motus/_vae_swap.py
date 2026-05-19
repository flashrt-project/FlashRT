"""G6.2 — Cast Wan2_2_VAE to BF16.

Profiling at G5 (g6.1_artifacts/profile_top.json) revealed the dominant
kernels in the captured graph are TF32 3D-convolution implicit_gemm
ops driven by the VAE encoder/decoder, not the FP8 Wan DiT GEMMs we
optimized at G4. The handoff doc estimated VAE at ~5-10 ms; the
profile says ~245 ms (≈42% of wall time at G5).

Root cause: ``wan.modules.vae2_2.Wan2_2_VAE.__init__`` defaults
``dtype=torch.float`` and Motus's ``models/wan_model.py:62`` does not
override it. The internal ``with amp.autocast(dtype=self.dtype)``
therefore autocasts to fp32, and cudnn dispatches TF32 conv kernels
on sm_120. Casting the VAE module to bf16 lets cudnn pick BF16
tensorop conv kernels (≈1.7-2× faster on Ada/Blackwell for these
shapes).

We do NOT modify upstream Motus or Wan source (K2/K3). The cast is
applied to the live module post-load:

    vae.model.to(bf16)        # weights + conv buffers
    vae.scale = [s.to(bf16) for s in vae.scale]   # normalization
    vae.dtype = bf16           # controls the autocast region

The pipeline already feeds bf16 inputs to ``encode_video`` (see
``encode_first_frame`` casting first_frame to ``self.dtype`` = bf16),
so input handling is unchanged. ``decode`` ends with ``.float()`` which
is an explicit upcast independent of weight dtype.

Cos risk:
    Wan VAE was trained in fp32 but supports the autocast path; the
    only practical drift is when an internal accumulator overflows
    bf16 mantissa. Wan VAE's tiled decoder uses small spatial windows
    keeping intermediate magnitudes modest; published Wan inference
    code already runs autocast(bf16) without quality loss on similar
    geometries. Validate empirically (tests/test_motus_g6_2_cosine.py).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def install_vae_bf16(model, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Cast ``model.video_model.vae`` to ``dtype`` in place.

    Returns a stats dict for logging.
    """
    vae = model.video_model.vae
    prev_dtype = vae.dtype

    # 1) Cast the inner WanVAE_ module (all conv weights + biases).
    #    .to() is idempotent for matching dtype.
    vae.model = vae.model.to(dtype)

    # 2) Cast the per-channel mean / inv_std tensors used at the very
    #    edges of encode/decode. These are 48-element vectors; cheap.
    new_scale = []
    for s in vae.scale:
        if torch.is_tensor(s):
            new_scale.append(s.to(dtype))
        else:
            new_scale.append(s)
    vae.scale = new_scale

    # 3) Tell ``Wan2_2_VAE`` itself that it now lives in bf16, so its
    #    ``with amp.autocast(dtype=self.dtype):`` region matches the
    #    weight dtype (no per-call casts inside autocast).
    vae.dtype = dtype

    n_params = sum(p.numel() for p in vae.model.parameters())
    logger.info(
        f"[g6.2] VAE cast {prev_dtype} -> {dtype}; "
        f"{n_params/1e6:.1f}M params, scale tensors casted")

    return {"prev_dtype": str(prev_dtype), "new_dtype": str(dtype),
            "n_params": int(n_params)}
