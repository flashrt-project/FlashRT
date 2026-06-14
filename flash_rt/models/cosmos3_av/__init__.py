"""Cosmos3-Nano AV FP4 inverse-dynamics denoise model (RTX SM120).

10-step UniPC denoise over a two-tower (und/causal + gen/full) MoT transformer with
NVFP4 weights, cutlass fp4-direct FFN, and int8-sage attention on the late layers.
See docs/cosmos3_official_av_fp4_baseline.md. Model-local kernels live in ./kernels/.
"""
