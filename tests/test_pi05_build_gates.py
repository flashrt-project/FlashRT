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
        ("pi05/pi05_sde_step.cuh", "pi05_sde_residual_add", "ENABLE_PI05_SDE"),
        ("pi05/pi05_geglu_nvfp4_quant.cuh", "pi05_geglu_merged_to_nvfp4_swizzled", "ENABLE_PI05_NVFP4"),
        ("pi05/pi05_decoder_skinny_fp8_sm120.cuh", "pi05_dec_skinny_available", "FLASHRT_PI05_DECODER_SKINNY_SM120"),
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


def test_fused_sde_obeys_both_feature_gates():
    binding = (ROOT / "csrc/bindings.cpp").read_text()
    guards = guards_at(binding, 'm.def("pi05_dec_skinny_action_out_residual_sde"')
    assert "#ifdef ENABLE_PI05_SDE" in guards
    assert "#ifdef FLASHRT_PI05_DECODER_SKINNY_SM120" in guards
    for suffix in ("cu", "cuh"):
        source = (ROOT / f"csrc/kernels/pi05/pi05_decoder_skinny_fp8_sm120.{suffix}").read_text()
        assert "#ifdef ENABLE_PI05_SDE" in guards_at(source, "int action_out_residual_sde(")
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "target_compile_definitions(pi05_decoder_skinny_sm120_obj PRIVATE ENABLE_PI05_SDE=1)" in cmake


def test_pi05_kernel_ownership_and_status():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    for name, status in (("geglu_nvfp4_quant", "NVFP4 prefix"), ("sde_step", "SDE")):
        for suffix in ("cu", "cuh"):
            assert (ROOT / f"csrc/kernels/pi05/pi05_{name}.{suffix}").is_file()
            assert not (ROOT / f"csrc/kernels/{name}.{suffix}").exists()
        assert f"csrc/kernels/pi05/pi05_{name}.cu" in cmake
        assert f"Pi0.5 {status} kernels: ENABLED" in cmake
        assert f"Pi0.5 {status} kernels: DISABLED" in cmake
