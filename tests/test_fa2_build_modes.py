"""Build-graph and ELF boundary checks for the opt-in FA2 native C API."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("FLASHRT_BUILD_DIR", REPO_ROOT / "build"))
EXPECTED_RAW_EXPORTS = {
    "fvk_attention_fa2_fwd_fp16",
    "fvk_attention_fa2_fwd_bf16",
    "fvk_attention_fa2_fwd_bf16_window",
    "fvk_attention_fa2_fwd_bf16_seqused",
    "fvk_attention_fa2_fwd_bf16_seqused_splitkv",
    "fvk_attention_fa2_fwd_bf16_causal",
}


def _cache_value(name: str) -> str | None:
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        pytest.skip(f"no configured build directory at {BUILD_DIR}")
    prefix = f"{name}:"
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return line.rsplit("=", 1)[-1].strip()
    return None


def _cache_bool(name: str) -> bool:
    return (_cache_value(name) or "").upper() in {"1", "ON", "TRUE", "YES"}


def _link_manifest(target: str) -> str | None:
    link = BUILD_DIR / "CMakeFiles" / f"{target}.dir" / "link.txt"
    if link.is_file():
        manifest = link.read_text(errors="replace")
        for response_file in re.findall(r"@(\S+\.rsp)", manifest):
            response_path = BUILD_DIR / response_file
            if response_path.is_file():
                manifest += "\n" + response_path.read_text(errors="replace")
        return manifest
    ninja = BUILD_DIR / "build.ninja"
    if ninja.is_file():
        # Keep per-edge variables (including LINK_FLAGS), not just the
        # build line; exclude unrelated phony targets mentioning this name.
        blocks = re.split(r"\n(?=build )", ninja.read_text(errors="replace"))
        lines = [block for block in blocks
                 if re.match(r"build .*: .*(?:SHARED|MODULE)_LIBRARY_LINKER__" +
                             re.escape(target) + r"_", block)]
        return "\n".join(lines) if lines else None
    return None


def _fa2_supported() -> bool:
    return (_cache_value("GPU_ARCH") or "") in {
        "80", "86", "87", "89", "120", "121",
    }


def _readelf_dynamic(path: Path) -> str:
    return subprocess.check_output(
        ["readelf", "-d", str(path)], text=True, errors="replace")


def test_configured_fa2_target_matrix():
    native = _cache_bool("FLASHRT_ENABLE_NATIVE_CPP")
    adapter = _cache_bool("FLASHRT_BUILD_FA2_PYTHON_ADAPTER")
    enabled = _fa2_supported() and (native or adapter)

    adapter_link = _link_manifest("flash_rt_fa2")
    raw_link = _link_manifest("flashrt_fa2_raw")
    object_dir = BUILD_DIR / "CMakeFiles" / "fa2_vendor_obj.dir"

    assert (adapter_link is not None) == (_fa2_supported() and adapter)
    assert (raw_link is not None) == (_fa2_supported() and native)
    assert object_dir.is_dir() == enabled


def test_fa2_objects_have_one_owner():
    if not _fa2_supported():
        pytest.skip("FA2 is not supported by this configured architecture")

    native = _cache_bool("FLASHRT_ENABLE_NATIVE_CPP")
    adapter = _cache_bool("FLASHRT_BUILD_FA2_PYTHON_ADAPTER")
    adapter_link = _link_manifest("flash_rt_fa2")
    raw_link = _link_manifest("flashrt_fa2_raw")

    if adapter:
        assert adapter_link is not None
        if native:
            assert "flashrt_fa2_raw" in adapter_link
            assert "fa2_vendor_obj.dir" not in adapter_link
        else:
            assert "flashrt_fa2_raw" not in adapter_link
            assert "fa2_vendor_obj.dir" in adapter_link

    if native:
        assert raw_link is not None
        assert "fa2_vendor_obj.dir" in raw_link
        assert "--version-script" in raw_link
        assert "-static-libstdc++" not in raw_link
        assert "-static-libgcc" not in raw_link


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="ELF export check is Linux-specific")
def test_built_raw_library_has_exact_export_surface():
    raw_value = os.environ.get("FLASHRT_FA2_RAW_LIBRARY")
    if not raw_value:
        pytest.skip("set FLASHRT_FA2_RAW_LIBRARY to validate a built raw library")
    raw = Path(raw_value)
    assert raw.is_file()

    output = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(raw)],
        text=True,
        errors="replace",
    )
    exports = {line.split()[-1] for line in output.splitlines() if line.split()}
    assert exports == EXPECTED_RAW_EXPORTS

    dynamic = _readelf_dynamic(raw)
    assert "libpython" not in dynamic
    assert "RUNPATH" in dynamic and "$ORIGIN" in dynamic

    undefined = subprocess.check_output(
        ["nm", "-D", "--undefined-only", str(raw)],
        text=True,
        errors="replace",
    )
    assert " Py" not in undefined
    assert "fvk_attention_fa2_" not in undefined


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="ELF dependency check is Linux-specific")
def test_built_adapter_dependency_matches_mode():
    adapter_value = os.environ.get("FLASHRT_FA2_ADAPTER_LIBRARY")
    if not adapter_value:
        pytest.skip("set FLASHRT_FA2_ADAPTER_LIBRARY to validate an adapter")
    adapter = Path(adapter_value)
    assert adapter.is_file()

    dynamic = _readelf_dynamic(adapter)
    if _cache_bool("FLASHRT_ENABLE_NATIVE_CPP"):
        assert "libflashrt_fa2_raw.so" in dynamic
        assert "RUNPATH" in dynamic and "$ORIGIN" in dynamic
    else:
        assert "libflashrt_fa2_raw.so" not in dynamic


def test_built_adapter_chameleon_symbol_matches_build_mode():
    adapter_value = os.environ.get("FLASHRT_FA2_ADAPTER_LIBRARY")
    if not adapter_value:
        pytest.skip("set FLASHRT_FA2_ADAPTER_LIBRARY to validate an adapter")
    adapter = Path(adapter_value)
    assert adapter.is_file()

    spec = importlib.util.spec_from_file_location("flash_rt_fa2", adapter)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "fwd_fp16_causal") == _cache_bool(
        "FLASHRT_ENABLE_CHAMELEON")


def test_chameleon_fa2_compile_contract_is_consistent():
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    wrapper = (
        REPO_ROOT / "csrc" / "attention" / "fa2_wrapper_causal.cu"
    ).read_text()
    header = (
        REPO_ROOT / "csrc" / "attention" / "fa2_wrapper.h"
    ).read_text()
    bindings = (REPO_ROOT / "csrc" / "fa2_bindings.cpp").read_text()

    assert "target_compile_definitions(fa2_vendor_obj PRIVATE\n      FLASHRT_ENABLE_CHAMELEON=1)" in cmake
    assert "target_compile_definitions(flash_rt_fa2 PRIVATE\n        FLASHRT_ENABLE_CHAMELEON=1)" in cmake
    assert "defined(FLASHRT_ENABLE_CHAMELEON) && defined(FA2_HAS_FP16)" in wrapper
    assert "#ifdef FLASHRT_ENABLE_CHAMELEON\nFLASHRT_FA2_NATIVE_API void fvk_attention_fa2_fwd_fp16_causal" in header
    assert "#ifdef FLASHRT_ENABLE_CHAMELEON\n    // FP16 causal sibling" in bindings


def test_unused_fp8_bias_autotune_api_is_absent():
    for relative in (
        "csrc/bindings.cpp",
        "csrc/gemm/gemm_runner.h",
        "csrc/gemm/gemm_runner.cu",
    ):
        assert "autotune_fp8_nn_bias" not in (REPO_ROOT / relative).read_text()
