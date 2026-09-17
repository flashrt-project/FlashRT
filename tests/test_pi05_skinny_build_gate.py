"""Skinny's declaration and binding must share a model-owned opt-in gate."""
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_skinny_gate_is_default_off_and_independent_of_qwen():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    option = next(line for line in cmake.splitlines()
                  if line.startswith("option(FLASHRT_ENABLE_PI05_SKINNY "))
    assert option.endswith(" OFF)")
    stack = []
    matches = 0
    for line in (ROOT / "csrc/bindings.cpp").read_text().splitlines():
        stripped = line.strip()
        if ('#include "kernels/pi05/pi05_decoder_skinny_fp8_sm120.cuh"' in line
                or 'm.def("pi05_dec_skinny_available"' in line):
            assert "#ifdef FLASHRT_PI05_DECODER_SKINNY_SM120" in stack
            assert not any("QWEN" in entry for entry in stack)
            matches += 1
        if stripped.startswith(("#if ", "#ifdef ", "#ifndef ")):
            stack.append(stripped)
        elif stripped.startswith("#endif"):
            stack.pop()
    assert matches == 2


def test_skinny_ownership_and_configure_status():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "add_library(pi05_decoder_skinny_sm120_obj OBJECT" in cmake
    assert "csrc/kernels/pi05/pi05_decoder_skinny_fp8_sm120.cu" in cmake
    assert "Pi0.5 skinny decoder kernels: ENABLED" in cmake
    assert "Pi0.5 skinny decoder kernels: DISABLED" in cmake
    assert not (ROOT / "csrc/kernels/decoder_skinny_fp8_sm120.cu").exists()
    assert not (ROOT / "csrc/kernels/decoder_skinny_fp8_sm120.cuh").exists()
    assert "sm120_decoder_skinny_obj" not in cmake
