"""Lazy native RDNA surface; importing providers never loads HIP or Triton."""
def kernels():
    from flash_rt.amd import flash_rt_amd_kernels as module
    required = ("attention_bshd_bf16_rdna", "normalize_bf16_rdna", "rope_bf16_rdna")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"RDNA extension is missing {missing}; rebuild csrc/amd for the selected gfx target")
    return module
