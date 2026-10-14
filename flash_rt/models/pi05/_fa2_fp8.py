"""FA2 causal attention with an FP8 attention-output epilogue.

The prefix o_proj reads a static FP8 copy of the attention output. Stock FA2 writes
only bf16, so the pipeline pays a separate quantize pass over every layer's output.
This extension compiles FlashRT's own vendored FA2 (same sources, same flags) with
two patched headers in ``csrc/fa2_fp8`` that add the FP8 store to the three sites
that write the final output, and lets the bf16 store be skipped where the FP8 copy
is the only consumer.

It needs a FlashRT *source* checkout for the unpatched FA2 sources and the CUTLASS
3.x tree FA2 is pinned against; a wheel-only install has neither, so the loader
returns None and the caller keeps the stock FA2 + quantize path.
"""
from __future__ import annotations

import os
import pathlib

_EXT = False          # False = not attempted, None = unavailable
_PATCHED = pathlib.Path(__file__).resolve().parent / "csrc" / "fa2_fp8"

# Build flags are FlashRT's FA2 set (CMakeLists.txt fa2_vendor_obj). The four -U
# flags and --use_fast_math are load-bearing: without them the SASS drifts from
# FlashRT's by ~1 ULP per layer, which this path must not do.
_CUDA_FLAGS = [
    "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3", "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-DFLASH_NAMESPACE=dm05_fa2_fp8", "-DFA2_HAS_BF16=1", "-DFA2_HAS_HDIM_256=1",
    "-gencode", "arch=compute_120,code=sm_120", "-std=c++17",
]


def _flashrt_src() -> pathlib.Path | None:
    """Locate FlashRT's csrc/attention tree (env override, then the package parent)."""
    env = os.environ.get("FLASHRT_SRC")
    cands = [pathlib.Path(env)] if env else []
    try:
        import flash_rt
        cands.append(pathlib.Path(flash_rt.__file__).resolve().parent.parent)
    except ImportError:
        return None
    for root in cands:
        att = root / "csrc" / "attention"
        if (att / "fa2_causal_inst" / "flash_fwd_hdim256_bf16_sm80_causal.cu").is_file():
            return att
    return None


def load():
    """Return the compiled extension, or None when FlashRT's sources are absent."""
    global _EXT
    if _EXT is not False:
        return _EXT
    att = _flashrt_src()          # only the CUTLASS 3.x tree is still needed from there
    if att is None:
        _EXT = None
        return None
    from torch.utils.cpp_extension import load as _load

    build_dir = os.environ.get("DM05_NATIVE_BUILD_DIR")
    if build_dir:
        build_dir = os.path.join(build_dir, "fa2_fp8")
        os.makedirs(build_dir, exist_ok=True)
    try:
        _EXT = _load(
            name="dm05_fa2_fp8",
            sources=[str(_PATCHED / "fa2_fp8_binding.cu"),
                     str(_PATCHED / "fa2_fp8_wrapper.cu"),
                     str(_PATCHED / "fa2_fp8_wrapper_noncausal.cu"),
                     # the instantiations are vendored too: compiled from FlashRT's tree
                     # they would resolve their quoted includes against their own directory
                     # and pick up the unpatched Flash_fwd_params
                     str(_PATCHED / "flash_attn" / "flash_fwd_hdim256_bf16_sm80_causal.cu"),
                     str(_PATCHED / "flash_attn" / "flash_fwd_split_hdim256_bf16_sm80_causal.cu"),
                     str(_PATCHED / "flash_attn" / "flash_fwd_hdim256_bf16_sm80.cu"),
                     str(_PATCHED / "flash_attn" / "flash_fwd_split_hdim256_bf16_sm80.cu")],
            # the patched headers must precede FlashRT's originals
            # the vendored headers must precede FlashRT's originals: a quoted
            # include resolves against the including file's directory first, so the
            # whole header set has to sit together (see csrc/fa2_fp8/flash_attn/VENDOR.md)
            extra_include_paths=[str(_PATCHED / "flash_attn"), str(_PATCHED), str(_PATCHED.parent),
                                 str(att / "flash_attn_2_src" / "flash_attn"),
                                 str(att / "flash_attn_2_src"),
                                 str(att / "flash_attn_2_src" / "cutlass" / "include"),
                                 str(att), str(att.parent)],
            extra_cuda_cflags=list(_CUDA_FLAGS), extra_cflags=["-O3", "-std=c++17"],
            build_directory=build_dir, verbose=False,
        )
    except Exception:
        _EXT = None
    return _EXT
