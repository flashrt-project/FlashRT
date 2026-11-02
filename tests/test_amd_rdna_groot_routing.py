"""RDNA GR00T registration and setup contracts, without a checkpoint."""

import pytest


@pytest.mark.parametrize("hardware", ["amd_rdna3", "amd_rdna35", "amd_rdna4"])
def test_rdna_groot_resolves_to_bf16_frontend(hardware):
    pytest.importorskip("torch")
    from flash_rt.hardware import resolve_pipeline_class
    from flash_rt.amd.frontends.torch.groot_n17_rdna import GrootN17TorchFrontendAmdRdna

    assert resolve_pipeline_class("groot_n17", "torch", hardware) is GrootN17TorchFrontendAmdRdna
    with pytest.raises(ValueError, match="BF16 only"):
        GrootN17TorchFrontendAmdRdna("/nonexistent", hardware=hardware, use_fp8=True)


def test_rdna_groot_does_not_replace_existing_routes():
    from flash_rt.hardware import _PIPELINE_MAP

    for hardware in ("amd_cdna3", "amd_cdna4"):
        assert _PIPELINE_MAP[("groot_n17", "torch", hardware)] == (
            "flash_rt.amd.frontends.torch.groot_n17", "GrootN17TorchFrontendAmd")
    for hardware in ("amd_rdna3", "amd_rdna35", "amd_rdna4"):
        assert _PIPELINE_MAP[("pi05", "torch", hardware)] == (
            "flash_rt.amd.frontends.torch.pi05_rdna", "Pi05TorchFrontendAmdRdna")


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151", "gfx1201"])
def test_groot_gemm_policy_preserves_existing_callers(monkeypatch, arch):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from flash_rt.amd.frontends.torch.groot_n17_rdna import _make_transformer_ops
    from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend

    class Runner:
        pool = None

        def enable_lazy_autotune(self, pool):
            self.pool = pool

    kernels = SimpleNamespace(RdnaGemmRunner=Runner)
    monkeypatch.setenv("FLASHRT_RDNA35_GEMM_AUTOTUNE", "1")
    monkeypatch.setenv("FLASHRT_RDNA35_GEMM_ALGOS", "7")
    ops = _make_transformer_ops(kernels, SimpleNamespace(gpu_arch=arch))
    assert ops.gemm.autotune is False
    assert ops.gemm._runner.pool is None
    # Pi0.5 and other existing callers still honor the original environment.
    assert Rdna35GemmBackend(kernels)._runner.pool == 7
    monkeypatch.setenv("FLASHRT_RDNA35_GEMM_AUTOTUNE", "0")
    assert Rdna35GemmBackend(kernels)._runner.pool is None
