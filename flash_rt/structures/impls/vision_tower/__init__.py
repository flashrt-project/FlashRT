"""The biased-LayerNorm vision tower region: the patch encoder pass.

A ViT-style tower — LayerNorm pairs with affine bias, biased QKV/out
projections, a biased tanh-GELU MLP — whose whole per-layer loop the
chain candidate re-expresses in static-FP8 hub primitives with the
bias and residual folded into the GEMM epilogues.
"""
