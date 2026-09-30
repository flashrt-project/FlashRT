"""Host-only contracts for the CDNA3 architecture layer."""

from types import SimpleNamespace

import pytest

from flash_rt.amd.hardware.capabilities import load_capabilities


def _module(build="gfx942", device="gfx942:sramecc+:xnack-", **overrides):
    info = {
        "gpu_arch": build,
        "hardware": "amd_cdna3" if build == "gfx942" else "amd_cdna4",
        "fp8_format": "e4m3fnuz" if build == "gfx942" else "e4m3fn",
        "fp8_max_finite": 240.0 if build == "gfx942" else 448.0,
        "supports_mxfp4": build == "gfx950",
        "supports_packed_fp8_mfma": build in ("gfx942", "gfx950"),
        "supports_packed_bf16_mfma": True,
        "packed_bf16_layout": "mfma_k16_lane4" if build == "gfx942" else "mfma_k32_lane8",
        "packed_bf16_shapes": ([(10, 2560, 1024), (10, 1024, 2048), (10, 8192, 1024)]
                               if build == "gfx942" else
                               [(m, 1536, 1536) for m in range(1, 49)]),
        "supports_fused_attention_fp8_output": True,
        "supports_aiter": True,
    }
    info.update(overrides)
    return SimpleNamespace(build_info=lambda: info, device_arch=lambda: device)


def test_cdna3_declares_fnuz_and_its_validated_packed_fp8_form():
    caps = load_capabilities(_module())
    assert caps.hardware == "amd_cdna3"
    assert caps.fp8_format == "e4m3fnuz"
    assert caps.fp8_max_finite == 240.0
    assert not caps.supports_mxfp4
    assert caps.supports_packed_fp8_mfma
    assert caps.supports_packed_bf16_mfma


def test_cdna4_declarations_remain_ocp():
    caps = load_capabilities(_module(build="gfx950", device="gfx950"))
    assert caps.hardware == "amd_cdna4"
    assert caps.fp8_format == "e4m3fn"
    assert caps.fp8_max_finite == 448.0
    assert caps.supports_mxfp4


@pytest.mark.parametrize("build,device", [
    ("gfx942", "gfx950"),
    ("gfx950", "gfx942"),
    ("gfx942", "gfx9420"),
])
def test_extension_device_mismatch_is_refused(build, device):
    with pytest.raises(RuntimeError, match="mismatch|unsupported AMD device"):
        load_capabilities(_module(build=build, device=device))


def test_missing_capability_metadata_is_refused():
    mod = _module()
    info = mod.build_info()
    del info["supports_mxfp4"]
    mod.build_info = lambda: info
    with pytest.raises(RuntimeError, match="supports_mxfp4"):
        load_capabilities(mod)


def test_weight_loader_quant_override_uses_fnuz_bytes():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU for torch FP8 conversion")
    from flash_rt.executors.torch_weights import DictSource, Quant
    from flash_rt.executors.weight_loader import LoaderContext

    target = SimpleNamespace(
        _weight_fp8_dtype=torch.float8_e4m3fnuz,
        _weight_fp8_max_finite=240.0,
    )
    ctx = LoaderContext(source=DictSource({}), target=target)
    weight = torch.linspace(-3, 3, 1024, device="cuda", dtype=torch.bfloat16)
    actual = Quant().apply(weight, ctx)
    scale = max(weight.float().abs().max().item() / 240.0, 1e-12)
    expected = (weight.float() / scale).clamp(-240, 240).to(
        torch.float8_e4m3fnuz)
    assert ctx.scratch["_pending_scale"] == scale
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("n,k", [(2560, 1024), (1024, 2048), (8192, 1024)])
def test_cdna3_accepts_pi05_bf16_forms_only_with_matching_layout(n, k):
    caps = load_capabilities(_module())
    assert caps.supports_packed_bf16(10, n, k, layout="mfma_k16_lane4")
    assert not caps.supports_packed_bf16(10, n, k, layout="mfma_k32_lane8")
    assert not caps.supports_packed_bf16(49, n, k, layout="mfma_k16_lane4")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_groot_bf16_route_requires_declared_shape_and_layout(arch):
    caps = load_capabilities(_module(build=arch, device=arch))
    assert caps.supports_packed_bf16(41, 1536, 1536, layout="mfma_k32_lane8") == (arch == "gfx950")
    assert not caps.supports_packed_bf16(49, 1536, 1536, layout="mfma_k32_lane8")
    assert not caps.supports_packed_bf16(41, 6144, 1536, layout="mfma_k32_lane8")


def test_groot_cdna3_skips_packing_and_refuses_forced_smallm(monkeypatch):
    from flash_rt.amd.frontends.torch.groot_n17 import GrootN17TorchFrontendAmd

    model = object.__new__(GrootN17TorchFrontendAmd)
    model._amd_caps = load_capabilities(_module())
    monkeypatch.delenv("FVK_AMD_DIT_GEMM", raising=False)
    model._pack_smallm_dit_weights()
    assert model._dit_smallm_packed == {}
    model._dit_smallm_packed = None
    monkeypatch.setenv("FVK_AMD_DIT_GEMM", "smallm")
    with pytest.raises(ValueError, match="shape/layout"):
        model._pack_smallm_dit_weights()
