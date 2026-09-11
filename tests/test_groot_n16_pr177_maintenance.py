from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_thor_qk_norm_source_and_binding_share_one_gate():
    cmake = _read("CMakeLists.txt")
    bindings = _read("csrc/bindings.cpp")

    gate = cmake.index("if(ENABLE_SM100_CUTLASS)", cmake.index("Thor-class VLA"))
    source = cmake.index("csrc/kernels/qk_norm_rope_rotate_half_bf16.cu", gate)
    gate_end = cmake.index("endif()", gate)
    assert gate < source < gate_end
    assert '#ifdef FLASHRT_HAVE_THOR_VLA_KERNELS\n#include "kernels/qk_norm_rope_rotate_half_bf16.cuh"' in bindings
    assert '#ifdef FLASHRT_HAVE_THOR_VLA_KERNELS\n    m.def("qk_norm_rope_rotate_half_bf16"' in bindings


def test_fa4_compilation_alias_does_not_override_runtime_arch():
    backend = _read("flash_rt/hardware/thor/fa4_backend.py")
    interface = _read(
        "csrc/attention/flash_attn_4_src/flashrt_fa4/cute/"
        "interface_fwd_sm100.py"
    )

    assert 'setdefault("CUTE_DSL_ARCH", "sm_101a")' in backend
    assert 'setdefault("FLASH_ATTENTION_ARCH"' not in backend
    assert "torch.cuda.get_device_capability()" in interface
    assert "use_dedicated_hd256_kernel = arch // 10 == 10" in interface


def test_n16_new_precision_tiers_are_opt_in():
    source = _read("flash_rt/frontends/torch/groot_thor.py")

    assert 'embodiment_tag="new_embodiment", use_fp8=True,' in source
    assert "image_size=224, parity=False" in source
    assert "parity=True requires use_fp8=False" in source
    assert "parity=True requires image_size=252" in source
    for name in (
        "FLASHRT_N16_FA4",
        "FLASHRT_N16_QWEN3_FP4",
        "FLASHRT_N16_SIGLIP_FP4",
        "FLASHRT_N16_DIT_FP4",
    ):
        assert f'os.environ.get("{name}", "0")' in source

    capture_start = source.index("    def _capture_all_graphs(")
    capture = source[capture_start:]
    assert "if self.parity:\n            self._setup_torch_dit()" in capture
    assert 'if self.parity and not hasattr(self, "_torch_qwen3")' in capture
    assert 'if self.parity and not hasattr(self, "_torch_siglip")' in capture


def test_prompt_switch_and_calibration_follow_runtime_layout_contracts():
    source = _read("flash_rt/frontends/torch/groot_thor.py")
    prompt_start = source.index("    def set_prompt(")
    prompt = source[prompt_start:source.index("    def infer_action_head(", prompt_start)]

    assert "self.reset_graph_runtime()" in prompt
    assert prompt.index("self.reset_graph_runtime()") < prompt.index(
        "self._input_ids = torch.tensor"
    )
    reset_start = source.index("    def reset_graph_runtime(")
    reset = source[reset_start:source.index("    def set_prompt(", reset_start)]
    assert '"_qwen3_fp4_done", "_siglip_fp4_done"' in reset

    collect_start = source.index("    def _collect_dit_amax(")
    collect = source[collect_start:source.index("    def _calibrate_dit(", collect_start)]
    assert "fvk.concat2_bf16(a_emb_out.data_ptr()," in collect
    assert "ae_concat.data_ptr()+T*D*2" not in collect


def test_eagle_remote_code_requires_checkpoint_local_or_explicit_revision():
    source = _read("flash_rt/frontends/torch/groot_thor.py")
    resolver_start = source.index("    def _resolve_eagle_dir(")
    resolver = source[resolver_start:source.index(
        "    def _setup_torch_siglip(", resolver_start
    )]

    assert "FLASHRT_N16_EAGLE_DIR" in resolver
    assert "self._checkpoint_path" in resolver
    assert "transformers_modules" not in resolver
    assert ".glob(" not in resolver
