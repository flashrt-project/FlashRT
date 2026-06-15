"""Cosmos3-Nano text2video denoise model (RTX SM120).

Fully self-contained (docs/adding_new_model.md §0): the two-tower MoT compute
path is pipeline_rtx.py, the UniPC scheduler is fm_solvers_unipc.py, and the
model-local CUDA kernels live in kernels/. No dependency on any other model.
"""
