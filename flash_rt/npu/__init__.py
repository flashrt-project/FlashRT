"""Self-contained Ascend backend.

Torch-NPU constructs graphs during setup. AscendCL moves fixed buffers and
replays them without tensor dispatch. Custom Ascend C kernels build through
scripts/npu/build.sh, independently of the CUDA and AMD build trees.
"""
