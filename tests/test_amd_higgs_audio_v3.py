"""API and generation policy checks that do not require a checkpoint."""
import inspect
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import HiggsAudioV3TorchFrontendAmdRdna
from flash_rt.frontends.torch.higgs_audio_v3_rtx import HiggsAudioV3TorchFrontendRtx
from flash_rt.models.higgs_audio_v3.pipeline_rtx import HiggsAudioV3Dims


def test_rdna_frontend_import_does_not_load_gpu_or_model_dependencies():
    subprocess.run([sys.executable, "-c", """
import importlib.abc
import sys
import torch

class NoGpuDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("triton", "transformers", "flash_rt.flash_rt_kernels",
                                "flash_rt.amd.hardware.rdna.",
                                "flash_rt.amd.models.higgs_audio_v3.")):
            raise AssertionError(f"frontend import loaded {fullname}")

sys.meta_path.insert(0, NoGpuDependencies())
from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import HiggsAudioV3TorchFrontendAmdRdna
assert not torch.cuda.is_initialized()
"""], check=True)


def test_rdna_constructor_and_public_methods_match_cuda():
    for name in ("build_prompt", "set_prompt", "prefill", "decode_stream", "predict",
                 "synthesize", "generate", "generate_stream", "resident_prefix_len"):
        cuda = inspect.signature(getattr(HiggsAudioV3TorchFrontendRtx, name))
        rdna = inspect.signature(getattr(HiggsAudioV3TorchFrontendAmdRdna, name))
        assert list(cuda.parameters) == list(rdna.parameters)
        for key in cuda.parameters:
            assert cuda.parameters[key].default == rdna.parameters[key].default
    cuda = inspect.signature(HiggsAudioV3TorchFrontendRtx)
    rdna = inspect.signature(HiggsAudioV3TorchFrontendAmdRdna)
    for name, parameter in cuda.parameters.items():
        assert name in rdna.parameters
        assert rdna.parameters[name].default == parameter.default


def test_explicit_rdna_routing_and_precision_metadata():
    from flash_rt.hardware import resolve_pipeline_class
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _TARGETS
    for arch, gfx in (("amd_rdna3", "gfx1100"), ("amd_rdna35", "gfx1151"),
                      ("amd_rdna4", "gfx1201")):
        frontend = resolve_pipeline_class("higgs_audio_v3", "torch", arch)
        assert frontend.__module__.startswith("flash_rt.amd.frontends.")
        assert _TARGETS[gfx].precision_spec(True)["weights"] == "float8_e4m3fn"
        assert _TARGETS[gfx].precision_spec(True)["scales"] == "per_tensor_float32"
        assert _TARGETS[gfx].precision_spec(True)["logits"] == "bfloat16"
        assert _TARGETS[gfx].precision_spec(False)["logits"] == "bfloat16"
        assert _TARGETS[gfx].precision_spec(True)["decode"] == "fp8_w8a8_dynamic"
        assert _TARGETS[gfx].precision_spec(False)["decode"] == "bf16_w16a16"


@pytest.mark.parametrize("gfx,native", [("gfx1100", False), ("gfx1151", False), ("gfx1201", True)])
def test_precision_auto_selection_and_warned_fallback(gfx, native):
    import warnings
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _TARGETS

    target = _TARGETS[gfx]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert target.select_fp8(None, gfx) is native
        assert target.select_fp8(False, gfx) is False
    assert not caught
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = target.select_fp8(True, gfx)
    assert actual is native
    assert len(caught) == (0 if native else 1)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert gfx in str(caught[0].message) and "falling back to BF16" in str(caught[0].message)
    spec = target.precision_spec(actual)
    assert spec["prefill"] == spec["single_token_prefill"] == spec["decode"]
    assert spec["weights"] == ("float8_e4m3fn" if native else "bfloat16")


def test_triton_device_capabilities_are_exact_and_need_no_hip_bindings():
    from flash_rt.amd.hardware.capabilities_rdna import device_capabilities
    for gfx, hardware in (("gfx1100", "amd_rdna3"), ("gfx1151", "amd_rdna35"),
                           ("gfx1201", "amd_rdna4")):
        caps = device_capabilities(gfx + ":xnack-", hardware)
        assert caps.gpu_arch == gfx and caps.hardware == hardware
    with pytest.raises(RuntimeError, match="unsupported"):
        device_capabilities("gfx11000")
    with pytest.raises(RuntimeError, match="requires"):
        device_capabilities("gfx1151", "amd_rdna4")


@pytest.mark.parametrize("arch", ["rtx_sm120", "amd_rdna35"])
def test_factory_preserves_precision_and_binds_the_requested_target(monkeypatch, arch):
    from flash_rt.models.higgs_audio_v3 import create_frontend
    import flash_rt.hardware as hardware
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(hardware, "resolve_pipeline_class", lambda *args: lambda path, **kw: (path, kw))
    path, kwargs = create_frontend("checkpoint", hardware=arch, fp8=True, device="cuda:1")
    assert path == "checkpoint" and kwargs["fp8"] is True and kwargs["device"] == "cuda:1"
    assert kwargs.get("hardware") == (arch if arch.startswith("amd_rdna") else None)


def test_cuda_auto_factory_defers_capability_checks_to_original_frontend(monkeypatch):
    from flash_rt.models.higgs_audio_v3 import create_frontend
    import flash_rt.hardware as hardware
    import flash_rt.frontends.torch.higgs_audio_v3_rtx as cuda
    monkeypatch.setattr(torch.version, "hip", None)

    def unexpected(*args):
        raise AssertionError("CUDA auto must retain the direct constructor without a device guard")

    monkeypatch.setattr(torch.cuda, "device", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    monkeypatch.setattr(hardware, "detect_arch", unexpected)
    monkeypatch.setattr(hardware, "resolve_pipeline_class", unexpected)
    monkeypatch.setattr(cuda, "HiggsAudioV3TorchFrontendRtx", lambda path, **kw: (path, kw))
    assert create_frontend("checkpoint", device="cuda:1", fp8=None, max_seq=128) == (
        "checkpoint", {"device": "cuda:1", "fp8": None, "max_seq": 128})


def test_auto_factory_detects_the_selected_rocm_device(monkeypatch):
    from flash_rt.models.higgs_audio_v3 import create_frontend
    import flash_rt.hardware as hardware
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index=None: SimpleNamespace(
        gcnArchName="gfx1151" if index is None else "gfx1100"))
    monkeypatch.setattr(hardware, "resolve_pipeline_class", lambda *args: lambda path, **kw: kw)
    assert create_frontend("checkpoint", device="cuda:1")["hardware"] == "amd_rdna35"


class _Policy(HiggsAudioV3TorchFrontendAmdRdna):
    DIMS = HiggsAudioV3Dims()

    def __init__(self):
        self._cfg = {"num_codebooks": 8, "codebook_vocab": 1026, "hidden": 1}
        self._gen_pos = 1
        self._gen_logits = torch.zeros(8, 1026)
        self._codes_dev = torch.zeros(8, dtype=torch.long)
        self._embed_buf = torch.zeros(1, 1)
        self.max_seq = 9
        self.max_new_frames = 100
        self.positions = []
        self._step = 0

    def _sample_logits(self, *args):
        self._codes_dev.fill_(self._step)
        self._step += 1

    def _decode_logits(self, embed, position):
        self.positions.append(position)
        return self._gen_logits

    def _ensure_codec(self):
        raise AssertionError("invalid options must fail before loading the codec")


def test_generation_capacity_stops_before_an_out_of_bounds_kv_write():
    frontend = _Policy()
    frames = list(frontend.decode_stream())
    assert len(frames) == 1
    assert frames[0].tolist() == list(range(8))
    assert max(frontend.positions) < frontend.max_seq
    assert frontend.last_finish_reason == "max_seq"


def test_decode_requires_prefill():
    frontend = _Policy()
    frontend._gen_logits = None
    with pytest.raises(RuntimeError, match="prefill"):
        list(frontend.decode_stream())


@pytest.mark.parametrize("kwargs", [{"first_chunk": 0}, {"chunk": -1},
                                    {"ctx": -1}, {"holdback": True},
                                    {"temperature": float("nan")}, {"seed": -1}])
def test_invalid_streaming_options_fail_before_codec_load(kwargs):
    with pytest.raises((ValueError, TypeError)):
        list(_Policy().generate_stream("Hello", **kwargs))


@pytest.mark.parametrize("dilation,length,groups", [(3, 18, 1), (3, 17, 1), (9, 1, 3)])
def test_polyphase_convolution_matches_independent_double_reference(dilation, length, groups):
    from flash_rt.amd.hardware.rdna.convolution import dilated_conv1d
    torch.manual_seed(71)
    x = torch.randn(2, 6, length)
    weight = torch.randn(9, 6 // groups, 7)
    bias = torch.randn(9)
    actual = dilated_conv1d(x, weight, bias, dilation=dilation, padding=3*dilation, groups=groups)
    expected = torch.nn.functional.conv1d(x.double(), weight.double(), bias.double(),
        dilation=dilation, padding=3*dilation, groups=groups).float()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_codec_adapter_retains_parameters_and_is_idempotent():
    pytest.importorskip("transformers")
    from flash_rt.amd.models.higgs_audio_v3.codec_rdna import _prepare_decoder, _DilatedConv1d
    module = torch.nn.Sequential(torch.nn.Conv1d(2, 2, 3, padding=1),
        torch.nn.Conv1d(2, 2, 7, dilation=3, padding=9))
    state = module.state_dict()
    _prepare_decoder(module)
    assert isinstance(module[0], torch.nn.Conv1d)
    assert isinstance(module[1], _DilatedConv1d)
    assert module.state_dict().keys() == state.keys()
    assert all(module.state_dict()[key].data_ptr() == value.data_ptr() for key, value in state.items())
    first = module[1]
    _prepare_decoder(module)
    assert module[1] is first


@pytest.mark.parametrize("codebook", [0, 3, 7])
def test_native_eoc_winddown_matches_cuda(codebook):
    from flash_rt.amd.frontends.torch.higgs_audio_v3_rdna import _delayed_eoc_countdown
    from flash_rt.frontends.torch.higgs_audio_v3_rtx import _delayed_eoc_countdown as cuda_countdown

    codes = torch.arange(8)
    codes[codebook] = 1025
    assert _delayed_eoc_countdown(codes, 8, 1025) == cuda_countdown(codes, 8, 1025)
    assert _delayed_eoc_countdown(torch.arange(8), 8, 1025) is None


@pytest.mark.parametrize("frames", [0, 4, 8, 16, 33, 58, 65])
def test_native_stream_windows_and_tail_match_cuda_without_model(monkeypatch, frames):
    """Run the real packet policies on the same synthetic acoustic codes."""
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    outputs = []
    for cls in (HiggsAudioV3TorchFrontendRtx, HiggsAudioV3TorchFrontendAmdRdna):
        frontend = object.__new__(cls)
        frontend.device = "cpu"
        frontend.set_prompt = lambda text: None
        frontend.prefill = lambda: None
        windows = []
        def codec(codes):
            windows.append(codes[:, 0].tolist())
            # Native trims to committed frames, including on the final flush.
            return torch.arange(len(codes) * 960 + 17, dtype=torch.float32)
        frontend._codec = SimpleNamespace(decode=codec)
        frontend._ensure_codec = lambda: None
        frontend.decode_stream = lambda **kwargs: (
            torch.full((8,), i, dtype=torch.long) for i in range(frames))
        chunks = list(frontend.generate_stream("Hello.", temperature=1.0, seed=1234))
        outputs.append((windows, chunks))
    assert outputs[0][0] == outputs[1][0]
    assert [len(x) for x in outputs[0][1]] == [len(x) for x in outputs[1][1]]
    assert sum(len(x) for x in outputs[1][1]) == frames * 960
    for cuda, rdna in zip(outputs[0][1], outputs[1][1]):
        assert torch.equal(cuda, rdna)
