"""FlashRT LTX-2.5 RTX SM120 integration.

Wraps the official LTX-2 ``ltx-pipelines`` distilled two-stage pipeline and
progressively swaps compute onto FlashRT kernels:

    * attention: sage2 qk-int8/pv-fp8 raw kernels (video sites, d128) with
      per-shape preallocated buffers, graph-capture safe
    * NVFP4 linear path: checkpoint ships prequantized weights + static
      activation scales; GEMMs run through the upstream cuBLASLt entry with
      FlashRT fused quantize epilogues arriving in later stages
    * CUDA graph capture over the transformer block loop per stage

The official model source is discovered through ``FLASH_RT_LTX2_ROOT`` (a
checkout of the LTX-2 monorepo with ``ltx-core`` / ``ltx-pipelines``
importable), mirroring the Wan2.2 integration contract.
"""
