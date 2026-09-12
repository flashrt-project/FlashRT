"""The operand checks that stand in front of every Ascend raw pointer.

Each native entry point on that backend is reached through ``ctypes`` with
``data_ptr()``, so a host tensor, a mismatched width or a strided row is a device
fault rather than a wrong number. These checks inspect tensor attributes only,
which is why they run on a machine with no Ascend device.
"""
import pytest
import torch

from flash_rt.npu.core import operands


def test_a_host_tensor_is_refused_and_the_message_says_why():
    with pytest.raises(ValueError, match="must be on an Ascend device"):
        operands.require_npu(torch.zeros(4), "frames")


def test_require_npu_returns_the_device_the_rest_are_held_to():
    """Documents the contract the wrappers rely on; the assertion runs only
    where there is a device to return."""
    if not hasattr(torch, "npu") or not getattr(torch.npu, "is_available", bool)():
        pytest.skip("no Ascend device")
    x = torch.zeros(4, device="npu:0")
    assert operands.require_npu(x, "x") == x.device


def test_a_tensor_on_the_wrong_device_is_refused_index_included():
    """The index is part of the comparison: a kernel launched on one die cannot
    read a buffer allocated on another. Spelled with a device type this host can
    parse, because ``npu`` is only a registered device once torch_npu is
    imported."""
    x = torch.zeros(4)
    with pytest.raises(ValueError, match="must be on cuda:1"):
        operands.require(x, "residual", device="cuda:1")


def test_the_dtype_is_checked():
    x = torch.zeros(4)
    with pytest.raises(ValueError, match="must be torch.bfloat16"):
        operands.require(x, "residual", device="cpu", dtype=torch.bfloat16)


def test_the_shape_is_checked_and_a_free_axis_is_allowed():
    x = torch.zeros(3, 8)
    operands.require(x, "plane", device="cpu", shape=(None, 8))
    with pytest.raises(ValueError, match=r"must be \(\*, 9\)"):
        operands.require(x, "plane", device="cpu", shape=(None, 9))
    with pytest.raises(ValueError, match="must be"):
        operands.require(x, "plane", device="cpu", shape=(3, 8, 1))


def test_a_row_pitch_wider_than_the_row_is_accepted():
    """A column slice of a wider projection is the normal case: the kernel is
    told the pitch and reads in place."""
    fused = torch.zeros(6, 12)
    operands.require(fused[:, :4], "query", device="cpu", shape=(6, 4), row_pitch=12)
    with pytest.raises(ValueError, match="row pitch of 4"):
        operands.require(fused[:, :4], "query", device="cpu", shape=(6, 4), row_pitch=4)


def test_a_transposed_operand_is_refused_where_unit_element_stride_is_required():
    x = torch.zeros(4, 6).t()
    with pytest.raises(ValueError, match="unit element stride"):
        operands.require(x, "value", device="cpu", row_pitch=4)
    with pytest.raises(ValueError, match="must be contiguous"):
        operands.require(x, "value", device="cpu", contiguous=True)


def test_a_non_tensor_is_refused_before_anything_is_read_off_it():
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        operands.require(object(), "residual", device="cpu")
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        operands.require_npu(None, "residual")


def test_mixing_devices_inside_one_launch_names_the_operand_that_broke_it():
    a = torch.zeros(4)
    b = torch.zeros(4)
    assert operands.same_device(("residual", a), ("branch", b)) == a.device
    with pytest.raises(ValueError, match="a launch needs at least one operand"):
        operands.same_device()
