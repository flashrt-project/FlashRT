"""CPU-only device refusals before the Flint Tensor API can launch CUDA."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_rt.native_kernels import nvfp4 as api


def _arguments(operation, rows=8):
    def tensor(shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device="cuda:0")

    width = 2048 if operation == "routed_shared_ffn" else 128
    x = tensor((rows, width), torch.bfloat16)
    if operation == "dense_ffn":
        args = {
            "gu_codes": tensor((256, 64)),
            "gu_scales": tensor((256, 8)),
            "gu_global": 1.0,
            "down_codes": tensor((128, 64)),
            "down_scales": tensor((128, 8)),
            "down_global": 1.0,
        }
    elif operation == "routed_shared_ffn":
        args = {
            "router": tensor((256, 2048), torch.bfloat16),
            "egc": tensor((256 * 1024, 1024)),
            "egs": tensor((256 * 1024, 128)),
            "egg": tensor((256,), torch.float32),
            "edc": tensor((256 * 2048, 256)),
            "eds": tensor((256 * 2048, 32)),
            "edg": tensor((256,), torch.float32),
            "sgc": tensor((2048, 1024)),
            "sgs": tensor((2048, 128)),
            "sgg": 1.0,
            "sdc": tensor((2048, 512)),
            "sds": tensor((2048, 64)),
            "sdg": 1.0,
            "scratch": [
                tensor(shape, dtype) for shape, dtype in api.workspace_shapes(rows)
            ],
            "capacity": rows,
        }
    else:
        args = {"codes": tensor((128, 64)), "scales": tensor((128, 8))}
        if operation == "linear":
            args["global_scale"] = 1.0
    return x, args


def _call_implementation(monkeypatch, operation, x, args, cuda_entry=None):
    def unexpected_cuda(*args, **kwargs):
        pytest.fail("entered CUDA execution before rejecting mismatched devices")

    monkeypatch.setattr(api, "_EXTENSION", object())
    monkeypatch.setattr(torch.cuda, "device", cuda_entry or unexpected_cuda)
    # Exercise the real implementation: FakeTensor dispatch normally selects
    # the shape-only fake registration and would bypass runtime validation.
    return getattr(api, "_" + operation)._init_fn(x, **args)


@pytest.mark.parametrize(
    "operation", ["linear", "dense_ffn", "lm_head", "routed_shared_ffn"]
)
@pytest.mark.parametrize("rows", [1, 8])
def test_tensor_api_rejects_activation_on_another_device(monkeypatch, operation, rows):
    with FakeTensorMode():
        x, args = _arguments(operation, rows)
        x = torch.empty(x.shape, dtype=x.dtype, device="cuda:1")
        with pytest.raises(ValueError, match="share a device"):
            _call_implementation(monkeypatch, operation, x, args)


_DEVICE_CASES = (
    [(op, name) for op in ("linear", "lm_head") for name in ("codes", "scales")]
    + [
        ("dense_ffn", name)
        for name in ("gu_codes", "gu_scales", "down_codes", "down_scales")
    ]
    + [
        ("routed_shared_ffn", name)
        for name in (
            "router",
            "egc",
            "egs",
            "egg",
            "edc",
            "eds",
            "edg",
            "sgc",
            "sgs",
            "sdc",
            "sds",
        )
    ]
    + [("routed_shared_ffn", f"scratch:{i}") for i in range(14)]
)


@pytest.mark.parametrize("operation,field", _DEVICE_CASES)
def test_tensor_api_rejects_each_foreign_weight_or_scratch(
    monkeypatch, operation, field
):
    with FakeTensorMode():
        x, args = _arguments(operation)
        if field.startswith("scratch:"):
            index = int(field.split(":")[1])
            original = args["scratch"][index]
            args["scratch"][index] = torch.empty(
                original.shape, dtype=original.dtype, device="cuda:1"
            )
        else:
            original = args[field]
            args[field] = torch.empty(
                original.shape, dtype=original.dtype, device="cuda:1"
            )
        with pytest.raises(ValueError, match="share a device"):
            _call_implementation(monkeypatch, operation, x, args)


class _CudaBoundaryReached(Exception):
    pass


@pytest.mark.parametrize(
    "operation", ["linear", "dense_ffn", "lm_head", "routed_shared_ffn"]
)
def test_tensor_api_accepts_colocated_inputs_before_cuda(monkeypatch, operation):
    def accepted(*args, **kwargs):
        raise _CudaBoundaryReached

    with FakeTensorMode():
        x, args = _arguments(operation)
        with pytest.raises(_CudaBoundaryReached):
            _call_implementation(monkeypatch, operation, x, args, accepted)
