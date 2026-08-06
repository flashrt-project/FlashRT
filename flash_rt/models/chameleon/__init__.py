"""FlashRT Chameleon-7B (VLM/LLM) model namespace.

Chameleon-7B is a 32-layer early-fusion multimodal LLM (MHA 32x128 with
per-head QK LayerNorm + RoPE, attention_bias=false, mlp_bias=false,
SwiGLU FFN, 8192-token VQ-GAN image vocabulary).

Two hardware paths:

- Thor SM110 (``pipeline_thor.py``): dynamic per-tensor FP8 backbone with
  fused norm/activation+quantize kernels, optional NVFP4 FFN layers,
  causal CUTLASS SM100 FMHA (``libfmha_fp16_causal.so`` /
  ``libfmha_fp8_causal.so``) and CUDA-graph decode.
- Orin SM87 (``pipeline_rtx.py``): INT8 W8A8 + INT4 W4A4 QuaRot-Hadamard
  rotated weights, SM80 CUTLASS rowwise GEMMs, FA2 fp16 causal attention.

Vendored code notice
--------------------
``flash_rt/models/chameleon/vqgan`` contains the Chameleon VQ-GAN image
tokenizer reference implementation, vendored from Meta's Chameleon
repository (``chameleon.vae_ori``). Those files are Copyright (c) Meta
Platforms, Inc. and affiliates, licensed under the Chameleon License (see
the copyright headers in ``vqgan/*.py``). They are inference-only and are
used solely to decode generated image tokens; no training code is
included.
"""
