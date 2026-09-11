"""AMD extension (flash_rt_amd_kernels) — import surface + required symbols.

Sync-fail-fast net for the compiled AMD module: the pi05 pipeline layer
resolves these bindings lazily by name at run time, so a symbol dropped by
a bindings.cpp / .inc edit (or by a partial build) would otherwise only
surface as an AttributeError mid-graph-capture on an MI350X box. This file
makes the whole bound surface a single loud collect-time-cheap assertion.

The expected-name lists are written out explicitly (not derived from the
module) and mirror, one to one:

    csrc/amd/bindings.cpp                 (kernels, probes, memory ops)
    csrc/amd/gemm/bindings_gemm.inc       (FvkContext + GemmRunner)
    csrc/amd/gemm/bindings_smallm.inc     (hand-tuned small-M FP8 GEMM)
    csrc/amd/gemm/bindings_ffn_fused.inc  (fused decoder-FFN pair)

If a name is deliberately removed from the bindings, remove it here in the
same commit — that is the point.

Skip conditions: every test skips unless ``flash_rt.amd.flash_rt_amd_kernels``
imports (the .so is built and libamdhip64 is loadable). Importing the module
does NOT require a GPU; the on-ROCm coherence assertions additionally gate
on a visible gfx device.
"""

from __future__ import annotations

import importlib

import pytest


def _import_ext():
    """Import the compiled AMD module or skip with the reason."""
    try:
        return importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")


# ---------------------------------------------------------------------------
# build_info() / device_arch()
# ---------------------------------------------------------------------------


def test_build_info_reports_hip_platform():
    """build_info is the first thing a bring-up session inspects; its
    platform tag is the proof the AMD module (not a CUDA build) was
    imported."""
    m = _import_ext()
    info = m.build_info()
    assert info["platform"] == "hip"
    # hipRuntimeGetVersion result; an int by binding contract. May be 0 if
    # the runtime query failed, so only the type is asserted here.
    assert isinstance(info["hip_runtime_version"], int)


def test_build_info_gpu_arch_is_gfx950_when_stamped():
    """FLASHRT_AMD_GPU_ARCH is stamped by csrc/amd/CMakeLists.txt at build
    time; when present it must be the CDNA4 target this backend supports.
    (Absent key = older build without the stamp; tolerated.)"""
    m = _import_ext()
    info = m.build_info()
    if "gpu_arch" not in info:
        pytest.skip("build lacks the FLASHRT_AMD_GPU_ARCH stamp")
    assert str(info["gpu_arch"]).startswith("gfx950"), (
        f"AMD module built for {info['gpu_arch']!r}, expected gfx950*")


def test_device_arch_coherent_with_build():
    """device_arch() queries the live device. On a gfx950 box it must agree
    with the build target; with no visible device it answers the documented
    sentinels instead of raising."""
    m = _import_ext()
    arch = m.device_arch()
    assert isinstance(arch, str)
    if arch in ("none", "unknown"):
        pytest.skip(f"no usable HIP device (device_arch()={arch!r})")
    assert arch.split(":")[0] == "gfx950", (
        f"extension is gfx950-only but the visible device is {arch!r}")


# ---------------------------------------------------------------------------
# Required symbol surface
# ---------------------------------------------------------------------------

# csrc/amd/bindings.cpp — the pi05 kernel surface + probes.
_REQUIRED_FUNCS = [
    # introspection
    "build_info", "device_arch",
    # norm
    "rms_norm", "rms_norm_fp16", "rms_norm_inplace", "layer_norm",
    "ada_rms_norm_style",
    # fused norm -> fp8
    "rms_norm_fp8", "ada_rms_norm_style_fp8", "residual_add_rms_norm_fp8",
    "bias_residual_layer_norm_bf16",
    # vision helpers
    "avg_pool_vision_tokens", "patch_im2col", "patch_embed_bias_pos",
    # activation
    "gate_geglu", "gelu_inplace", "bias_gelu_bf16_strict",
    "gate_geglu_merged", "gate_geglu_merged_fp8",
    # elementwise
    "gate_mul_residual", "bias_residual", "residual_add", "add_bias_bf16",
    # qkv split / rope
    "qkv_split", "qkv_split_rope", "qkv_split_rope_devpos",
    # attention
    "attention_decoder_gqa", "attention_decoder_gqa_fp8out",
    "encoder_attention_flash", "attn_partial_probe",
    # fusion
    "gate_residual_ada_norm_fp8", "gate_residual_ada_norm_fp8_ksum",
    # quantize
    "quantize_fp8_static", "quantize_fp8_device", "fp8_accumulate_scale_max",
    # memory ops + probes
    "gpu_copy", "stream_probe", "stream_probe_variants",
    "ew_tune_quant", "ew_tune_norm", "ew_tune_rope", "ew_tune_variants",
    # MFMA small-M FP8 GEMM (csrc/amd/gemm/smallm_mfma.h surface)
    "smallm_mfma_nt", "smallm_mfma_nt_partial", "smallm_mfma_nt_packed",
    "smallm_mfma_variants",
]

# csrc/amd/gemm/bindings_smallm.inc — weight-streaming small-M FP8 GEMM.
_REQUIRED_FUNCS += [
    "smallm_fp8_nn_ws_bytes",
    "smallm_fp8_nn_dev", "smallm_fp8_nn_dev_alt",
    "smallm_fp8_nt_dev", "smallm_fp8_nt_lds_dev",
    "smallm_fp8_nt_lds_async_available", "smallm_fp8_nt_dev_alt",
]

# csrc/amd/gemm/bindings_ffn_fused.inc — fused decoder-FFN pair.
_REQUIRED_FUNCS += [
    "smallm_fp8_gateup_geglu", "smallm_fp8_gateup_geglu_alt",
    "smallm_fp8_down_gateres", "smallm_fp8_down_gateres_alt",
]

# csrc/amd/gemm/bindings_gemm.inc — classes.
_REQUIRED_CLASSES = ["FvkContext", "GemmRunner"]

# GemmRunner methods the pi05 pipeline dispatches on by name.
_REQUIRED_GEMM_METHODS = [
    "bf16_run", "bf16_nn", "bf16_nn_res", "bf16_nn_bias",
    "bf16_nn_bias_gelu", "bf16_nn_bias_res",
    "fp8_nn_dev", "fp8_nt_dev", "mxfp4_nt_dev",
    # GROOT N1.7 surface: FP16 GEMM + FP8 epilogue variants
    "fp16_nn", "fp8_nn_bias", "fp8_nn_gelu_bias", "fp8_descale_fp16",
    "enable_lazy_autotune",
    "autotune_bf16_nn", "autotune_fp8_nn_dev", "autotune_fp8_nt_dev",
    "autotune_mxfp4_nt_dev", "autotune_fp16_nn",
    "autotune_fp8_descale_fp16",
]


def test_all_required_symbols_present():
    """One shot, full inventory: report EVERY missing name, not the first."""
    m = _import_ext()
    missing = [n for n in _REQUIRED_FUNCS + _REQUIRED_CLASSES
               if not hasattr(m, n)]
    assert not missing, (
        "flash_rt_amd_kernels is missing bound symbols "
        f"(bindings.cpp / .inc drift or partial build): {missing}")


def test_required_symbols_are_callable_or_types():
    m = _import_ext()
    for name in _REQUIRED_FUNCS:
        if hasattr(m, name):  # missing ones already reported above
            assert callable(getattr(m, name)), f"{name} bound but not callable"
    for name in _REQUIRED_CLASSES:
        if hasattr(m, name):
            assert isinstance(getattr(m, name), type), f"{name} is not a class"


def test_gemm_runner_method_surface():
    """The pipeline calls these methods on a GemmRunner instance; a method
    dropped from bindings_gemm.inc must fail here, not at capture time."""
    m = _import_ext()
    runner_cls = getattr(m, "GemmRunner", None)
    if runner_cls is None:
        pytest.skip("GemmRunner missing (reported by the inventory test)")
    missing = [n for n in _REQUIRED_GEMM_METHODS if not hasattr(runner_cls, n)]
    assert not missing, f"GemmRunner missing methods: {missing}"


def test_fvk_context_exposes_handle_ptr():
    """FvkContext.handle_ptr hands the hipBLASLt handle address to raw-ABI
    call sites; it is a readonly property on the class."""
    m = _import_ext()
    ctx_cls = getattr(m, "FvkContext", None)
    if ctx_cls is None:
        pytest.skip("FvkContext missing (reported by the inventory test)")
    assert hasattr(ctx_cls, "handle_ptr")


def test_variant_enumerators_answer():
    """The *_variants() enumerators are pure host-side introspection (no
    kernel launch, no device memory): they must answer on any box that can
    import the module, and their answers feed the tuning probes."""
    m = _import_ext()
    v = m.stream_probe_variants()
    assert len(v) > 0 and all("name" in d for d in v)
    fams = m.ew_tune_variants()
    assert set(fams.keys()) == {"quant", "norm", "rope"}
    assert all(len(fams[k]) > 0 for k in fams)
    assert len(m.smallm_mfma_variants()) > 0
    # smallm split-K workspace sizing is also pure host arithmetic.
    assert m.smallm_fp8_nn_ws_bytes(10, 2048) > 0
