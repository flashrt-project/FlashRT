"""Exercise shared GEMM dispatch without ROCm kernels or a checkpoint.

Fake CUDA tensors retain the production shape/layout validation. Only native
launches are replaced; GR00T and LingBot execute their real linear methods.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flash_rt.amd.hardware.rdna35.gemm import Rdna35GemmBackend
from flash_rt.amd.models.lingbot_vla_v2.pipeline_rdna import _TransformerOps


class Runner:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("bf16_"):
            return lambda *args: self.calls.append((name, args))
        raise AttributeError(name)


@pytest.fixture
def dispatch(monkeypatch):
    calls = []
    for name in ("mm", "addmm"):
        original = getattr(torch, name)

        def record(*args, _name=name, _original=original, **kwargs):
            # FakeTensor may decompose addmm into an internal mm. Record
            # the provider entry point, which always supplies its output.
            if "out" in kwargs:
                calls.append((_name, args, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, record)
    monkeypatch.setattr(FakeTensor, "data_ptr", lambda self: id(self))
    monkeypatch.setattr(Rdna35GemmBackend, "_stream", staticmethod(lambda tensor: 123))
    return SimpleNamespace(kernels=SimpleNamespace(RdnaGemmRunner=Runner), calls=calls)


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
@pytest.mark.parametrize("biased", [False, True])
def test_groot_actual_linear_keeps_torch_dispatch(dispatch, arch, biased):
    from flash_rt.amd.frontends.torch.groot_n17_rdna import _make_transformer_ops

    ops = _make_transformer_ops(dispatch.kernels, SimpleNamespace(gpu_arch=arch))
    # Another model opting in must not change GR00T's instance policy.
    native = Rdna35GemmBackend(dispatch.kernels, autotune=False, native_nt=True)
    with FakeTensorMode():
        x = torch.empty((2, 3, 32), device="cuda", dtype=torch.bfloat16)
        weight = ops.prepare_weight(torch.empty((32, 48), device=x.device, dtype=x.dtype))
        bias = torch.empty(48, device=x.device, dtype=x.dtype) if biased else None
        out = ops.linear(x, weight, bias)
        assert out.shape == (2, 3, 48)
    assert [call[0] for call in dispatch.calls] == ["addmm" if biased else "mm"]
    _, args, kwargs = dispatch.calls[0]
    assert args[-2].shape == (6, 32) and args[-1].shape == (32, 48)
    assert kwargs["out"].shape == (6, 48)
    assert ops.gemm._runner.calls == []
    assert not ops.gemm.native_nt and native.native_nt


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("reference_rows", [None, 72])
def test_lingbot_actual_linear_uses_explicit_native_nt(dispatch, biased, reference_rows):
    backend = Rdna35GemmBackend(dispatch.kernels, autotune=False, native_nt=True)
    ops = _TransformerOps(gemm=backend)
    with FakeTensorMode():
        x = torch.empty((1, 7, 32), device="cuda", dtype=torch.bfloat16)
        weight = ops.prepare_weight(torch.empty((32, 48), device=x.device, dtype=x.dtype))
        bias = torch.empty(48, device=x.device, dtype=x.dtype) if biased else None
        out = ops.linear(x, weight, bias, reference_rows=reference_rows)
        assert out.shape == (1, 7, 48)
    assert dispatch.calls == []
    name, args = backend._runner.calls.pop()
    assert name == ("bf16_nt" if reference_rows is None else "bf16_nt_rows")
    assert bool(args[3]) == biased
    assert args[4:] == ((7, 48, 32, 32, 48, 123) if reference_rows is None
                       else (7, 72, 48, 32, 32, 48, 123))
    assert not backend._runner.calls


@pytest.mark.parametrize("biased", [False, True])
def test_default_kn_dispatch_is_unchanged(dispatch, monkeypatch, biased):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123))
    backend = Rdna35GemmBackend(dispatch.kernels, autotune=False)
    with FakeTensorMode():
        x = torch.empty((7, 32), device="cuda", dtype=torch.bfloat16)
        weight = torch.empty((32, 48), device=x.device, dtype=x.dtype)
        out = torch.empty((7, 48), device=x.device, dtype=x.dtype)
        bias = torch.empty(48, device=x.device, dtype=x.dtype) if biased else None
        backend.linear(out, x, weight, bias)
    assert dispatch.calls == []
    assert [call[0] for call in backend._runner.calls] == ["bf16_nn_bias" if biased else "bf16_nn"]


def test_reference_rows_requires_explicit_native_nt(dispatch):
    backend = Rdna35GemmBackend(dispatch.kernels, autotune=False)
    with pytest.raises(ValueError, match="reference_rows requires"):
        backend.linear(None, None, None, weight_transposed=True, reference_rows=72)
    assert dispatch.calls == [] and backend._runner.calls == []
