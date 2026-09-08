"""FlashRT Ascend NPU backend (torch_npu / CANN).

Self-contained NPU tree. Unlike the CUDA/AMD backends it does not build a
raw-pointer C++ kernel extension: the 910/A2-generation NPU is driven
through ``torch_npu`` ops (which execute aclnn kernels under the hood)
and captured with ``torch.npu.graph`` NPU graphs. The NVIDIA tree never
imports from here and vice versa; ``detect_arch`` routes an Ascend box to
the ``"npu"`` arch string before it ever reaches the CUDA checks.

Chip generation is carried in filenames, not directories: a file with a
``_910b``-style suffix holds behaviour that is specific to that part
family, and un-suffixed files hold the common behaviour.
"""
