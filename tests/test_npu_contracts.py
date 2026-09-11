"""Ascend contracts that hold without an Ascend device.

The rest of the NPU suite skips itself when no NPU is present, which means a
CI machine proves nothing about the parts that are pure policy: how the arch is
detected, how a missing runtime is reported, which shared objects are refused,
and whether the INT8 tier is reachable from the public API. Those are exactly
the things a reviewer on a laptop has to take on trust otherwise, so they are
tested here with fakes instead.
"""
import builtins
import ctypes
import sys

import pytest


# ── arch routing ──────────────────────────────────────────────────────

class _FakeNpu:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


def _detect_with(monkeypatch, *, torch_npu_importable, npu_available,
                 cuda_available=False):
    """Run detect_arch against a torch whose NPU support we control."""
    import torch
    from flash_rt import hardware

    monkeypatch.setattr(torch, "npu", _FakeNpu(npu_available), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch_npu" and not torch_npu_importable:
            raise ImportError("no torch_npu on this machine")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "torch_npu",
                        sys.modules.get("torch_npu") or object())
    if not torch_npu_importable:
        monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    return hardware.detect_arch


def test_npu_is_detected_before_cuda(monkeypatch):
    """On a CANN box ``torch.cuda.is_available()`` is False while the part is
    perfectly usable, so the NPU probe has to come first."""
    pytest.importorskip("torch")
    detect = _detect_with(monkeypatch, torch_npu_importable=True,
                          npu_available=True, cuda_available=True)
    assert detect() == "npu"


def test_absent_torch_npu_falls_through_rather_than_raising(monkeypatch):
    """A machine without torch_npu must reach the CUDA branch, not die in the
    NPU probe. Here there is no CUDA either, so the CUDA message is the proof
    that we got that far."""
    pytest.importorskip("torch")
    detect = _detect_with(monkeypatch, torch_npu_importable=False,
                          npu_available=False, cuda_available=False)
    with pytest.raises(RuntimeError, match="CUDA- or ROCm-capable"):
        detect()


def test_installed_torch_npu_with_no_device_does_not_claim_the_box(monkeypatch):
    pytest.importorskip("torch")
    detect = _detect_with(monkeypatch, torch_npu_importable=True,
                          npu_available=False, cuda_available=False)
    with pytest.raises(RuntimeError, match="CUDA- or ROCm-capable"):
        detect()


def test_the_pi05_npu_frontend_is_registered():
    from flash_rt.hardware import _PIPELINE_MAP

    assert _PIPELINE_MAP[("pi05", "torch", "npu")] == (
        "flash_rt.npu.frontends.torch.pi05", "Pi05TorchFrontendNpu")


# ── shared object identity ────────────────────────────────────────────

class _FakeSymbol:
    def __init__(self, value):
        self._value = value
        self.restype = None
        self.argtypes = None

    def __call__(self):
        return self._value


class _FakeLibrary:
    """Stands in for a ctypes CDLL with a chosen identity."""

    def __init__(self, abi=None, soc=None):
        self._name = "libfake.so"
        self._symbols = {}
        if abi is not None:
            self._symbols["flashrt_npu_abi_version"] = _FakeSymbol(abi)
        if soc is not None:
            self._symbols["flashrt_npu_soc_version"] = _FakeSymbol(soc.encode())

    def __getattr__(self, name):
        try:
            return self.__dict__["_symbols"][name]
        except KeyError:
            raise AttributeError(name)


def test_a_matching_library_verifies():
    from flash_rt.npu.core import abi

    assert abi.verify(_FakeLibrary(abi.ABI_VERSION, abi.SOC_VERSION),
                      "dispatch") == abi.SOC_VERSION


def test_a_stale_library_is_refused_by_abi():
    """The four shared objects are built and loaded separately, so a partial
    rebuild leaves one behind with the same symbol names and a different
    contract. That is what the version number is for."""
    from flash_rt.npu.core import abi

    stale = _FakeLibrary(abi.ABI_VERSION - 1, abi.SOC_VERSION)
    with pytest.raises(ImportError, match="ABI"):
        abi.verify(stale, "decoder GEMM")


def test_a_library_for_another_part_is_refused():
    from flash_rt.npu.core import abi

    other = _FakeLibrary(abi.ABI_VERSION, "Ascend910B2")
    with pytest.raises(ImportError, match="Ascend910B2"):
        abi.verify(other, "gate/up cube")


def test_a_library_without_the_identity_symbols_is_refused():
    from flash_rt.npu.core import abi

    with pytest.raises(ImportError, match="rebuild"):
        abi.verify(_FakeLibrary(), "decode attention")


def test_every_loader_checks_its_own_library():
    """Each shared object has its own environment override, so checking only
    the dispatch unit would leave three unchecked."""
    import inspect

    from flash_rt.npu.core import decode_attention, decoder_int8, gu_int8
    from flash_rt.npu.core import native_kernels

    sources = (inspect.getsource(native_kernels._NativeLibrary),
               inspect.getsource(gu_int8.GuLibrary),
               inspect.getsource(decoder_int8.DecoderGemmLibrary),
               inspect.getsource(decode_attention.DecodeAttentionLibrary))
    for source in sources:
        assert "abi.verify" in source


def test_a_failing_device_query_is_a_fault_not_a_skip(monkeypatch):
    """The SoC half of the check used to swallow every exception and return
    None, which skipped the hardware comparison on exactly the machine the
    comparison exists for. Absent runtime and absent device still return None;
    a device that is present and cannot be named must raise."""
    import sys
    import types

    from flash_rt.npu.core import abi

    torch_npu = types.ModuleType("torch_npu")
    fake_torch = types.ModuleType("torch")
    fake_torch.npu = types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda _i: (_ for _ in ()).throw(RuntimeError("driver busy")))
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="driver busy"):
        abi._running_soc()


def test_no_device_is_still_a_skip(monkeypatch):
    import sys
    import types

    from flash_rt.npu.core import abi

    fake_torch = types.ModuleType("torch")
    fake_torch.npu = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert abi._running_soc() is None


def test_the_build_script_targets_only_the_validated_part():
    """The host tiling names Ascend910B4 and several kernels divide work by
    its twenty cube cores, so accepting the rest of the B-series would build
    libraries whose tiling is wrong for the target."""
    import pathlib

    from flash_rt.npu.core import abi

    root = pathlib.Path(abi.__file__).parents[3]
    script = (root / "scripts/npu/build.sh").read_text()
    assert 'npu_soc_version" != "Ascend910B4"' in script
    assert "Ascend910B[1-4]" not in script


def test_the_build_script_emits_each_library_once():
    """Two identical compiles of one source into one output shipped in an
    earlier revision of this script; nothing failed, it just built twice."""
    import pathlib

    from flash_rt.npu.core import abi

    root = pathlib.Path(abi.__file__).parents[3]
    script = (root / "scripts/npu/build.sh").read_text()
    for library in ("libflashrt_npu.so", "libflashrt_npu_cube.so",
                    "libflashrt_npu_decoder.so", "libflashrt_npu_attn.so"):
        assert script.count(f"/{library}\"") == 1, library


# ── the public API reaches the optimised tier ─────────────────────────

def test_load_model_rejects_an_unknown_precision():
    from flash_rt.api import load_model

    with pytest.raises(ValueError, match="precision must be one of"):
        load_model("unused", hardware="npu", precision="int4")


def test_fp8_is_refused_on_a_part_that_has_no_fp8(monkeypatch):
    """The frontend used to warn and run BF16, which is the failure this
    parameter exists to prevent: a tier that was asked for, not delivered, and
    not reported. It is refused before any backend import, so it fails the same
    way on a machine with no Ascend runtime installed."""
    from flash_rt.api import load_model

    with pytest.raises(ValueError, match="no FP8 tensor hardware"):
        load_model("unused", hardware="npu", precision="fp8")


@pytest.mark.parametrize("arch", ["rtx_sm120", "thor", "amd_cdna4"])
@pytest.mark.parametrize("tier", ["bf16", "int8", "fp8"])
def test_a_named_tier_is_refused_on_the_backends_that_do_not_read_it(arch, tier):
    """Those backends pick their frontend class from use_fp8/use_fp4/use_fp16
    before a generic tier could be consulted, so honouring precision= there
    would silently mean something else — precision='bf16' would still land on
    an FP4 frontend. Refused rather than ignored until that routing is
    reworked."""
    from flash_rt.api import load_model

    with pytest.raises(ValueError, match="Ascend NPU backend only"):
        load_model("unused", hardware=arch, precision=tier)


def test_auto_leaves_every_other_backend_exactly_as_it_was():
    """The parameter has to be inert at its default, or adding it changed the
    other platforms."""
    from flash_rt.api import load_model

    try:
        load_model("unused", hardware="rtx_sm120", precision="auto")
    except Exception as exc:  # something further along; never the tier check
        assert "precision" not in str(exc), exc


def _capture_routing(monkeypatch, **options):
    """Drive load_model to the point of constructing a frontend and return the
    keyword arguments it would have been built with."""
    import sys
    import types

    import flash_rt.api as api
    import flash_rt.hardware as hardware

    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    seen = {}

    class _Fake:
        def __init__(self, checkpoint, num_views=2, use_fp8=True, use_int8=False,
                     **rest):
            seen.update(checkpoint=checkpoint, num_views=num_views,
                        use_fp8=use_fp8, use_int8=use_int8, **rest)

    monkeypatch.setattr(hardware, "resolve_pipeline_class", lambda *a, **k: _Fake)
    monkeypatch.setattr(api, "VLAModel", lambda pipe, framework: pipe)
    api.load_model("unused", hardware="npu", **options)
    return seen


def test_int8_actually_reaches_the_frontend(monkeypatch):
    assert _capture_routing(monkeypatch, precision="int8")["use_int8"] is True


def test_bf16_actually_turns_the_int8_tier_off(monkeypatch):
    seen = _capture_routing(monkeypatch, precision="bf16")
    assert seen["use_int8"] is False


def test_the_npu_never_receives_use_fp8_true(monkeypatch):
    """load_model defaults use_fp8=True, and the part has no FP8 tensor
    hardware; the coercion has to survive every tier."""
    for tier in ("auto", "bf16", "int8"):
        assert _capture_routing(monkeypatch, precision=tier)["use_fp8"] is False


def test_the_npu_frontend_accepts_the_int8_tier():
    """``precision='int8'`` is forwarded as ``use_int8`` by feature detection,
    so the frontend has to keep that parameter name."""
    import importlib
    import inspect

    module = importlib.import_module("flash_rt.npu.frontends.torch.pi05")
    sig = inspect.signature(module.Pi05TorchFrontendNpu)
    assert "use_int8" in sig.parameters


def test_load_model_exposes_precision():
    import inspect

    from flash_rt.api import load_model

    assert inspect.signature(load_model).parameters["precision"].default == "auto"


def test_the_public_model_can_freeze_a_calibrated_tier():
    from flash_rt.api import VLAModel

    assert callable(VLAModel.calibrate)


def test_the_npu_frontend_takes_the_unified_calibration_signature():
    """``VLAModel.calibrate`` forwards percentile, max_samples and verbose to
    the frontend. A frontend that omits any of them is reachable only by
    holding it directly, which is what the public INT8 path has to avoid."""
    import importlib
    import inspect

    module = importlib.import_module("flash_rt.npu.frontends.torch.pi05")
    params = inspect.signature(module.Pi05TorchFrontendNpu.calibrate).parameters
    for name in ("observations", "percentile", "max_samples", "verbose"):
        assert name in params, name


def test_calibrate_is_refused_where_there_is_no_calibrated_tier():
    from flash_rt.api import VLAModel

    class _Plain:
        pass

    model = VLAModel.__new__(VLAModel)
    model._pipe = _Plain()
    with pytest.raises(NotImplementedError):
        model.calibrate([{}])
