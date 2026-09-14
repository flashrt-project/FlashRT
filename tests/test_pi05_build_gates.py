"""Pi0.5 optional kernels must not inherit another model's compile gates."""
from pathlib import Path

ROOT = Path(__file__).parents[1]


def guards_at(text, needle):
    stack = []
    for line in text.splitlines():
        if needle in line:
            return tuple(stack)
        stripped = line.strip()
        if stripped.startswith(("#if ", "#ifdef ", "#ifndef ")):
            stack.append(stripped)
        elif stripped.startswith("#endif"):
            stack.pop()
    raise AssertionError(f"Missing {needle}")


def test_headers_and_bindings_share_pi05_gates():
    text = (ROOT / "csrc/bindings.cpp").read_text()
    for header, symbol, guard in (
        ("sde_step.cuh", "pi05_sde_residual_add", "ENABLE_PI05_SDE"),
        ("geglu_nvfp4_quant.cuh", "pi05_geglu_merged_to_nvfp4_swizzled", "ENABLE_PI05_NVFP4"),
        ("decoder_skinny_fp8_sm120.cuh", "pi05_dec_skinny_available", "FLASHRT_DECODER_SKINNY_SM120"),
    ):
        for needle in (f'#include "kernels/{header}"', f'm.def("{symbol}"'):
            guards = guards_at(text, needle)
            assert f"#ifdef {guard}" in guards
            assert not any("QWEN" in item for item in guards)


def test_pi05_features_default_off():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    for option in ("SKINNY", "SDE", "NVFP4"):
        line = next(line for line in cmake.splitlines()
                    if line.startswith(f"option(FLASHRT_ENABLE_PI05_{option} "))
        assert line.endswith(" OFF)")
