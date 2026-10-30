"""Build-time Metal descriptor contracts; no GPU or compiled dylib required."""

from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flash_rt.hardware.apple_native.plan import Attention, Plan, Tensor


def recording_plan():
    # Exercise the real descriptor/binding checks without opening a Metal device.
    plan = Plan.__new__(Plan)
    plan.handle = object()
    plan.dispatch = Mock()
    plan.copy_buffer = Mock()
    plan.allocate = Mock(side_effect=AssertionError("Unexpected scratch allocation"))
    return plan


def tensor(plan, shape, *, itemsize=2, stride=None):
    return Tensor(plan, shape, buffer=0, itemsize=itemsize, stride=stride)


@pytest.mark.parametrize("strided", [False, True])
def test_binding_rejects_foreign_plan_even_with_same_buffer_id(strided):
    owner, other = recording_plan(), recording_plan()
    foreign = tensor(other, (2, 4), stride=(8, 1) if strided else None)
    with pytest.raises(ValueError, match="another plan"):
        owner.bind(foreign, allow_strided=strided)


def test_copy_elides_exact_aliases_created_as_separate_views():
    plan = recording_plan()
    storage = tensor(plan, (4, 4))
    source, target = storage[1:3], storage[1:3].view(2, 4)
    assert source is not target
    assert target.copy_(source) is target
    plan.dispatch.assert_not_called()
    plan.copy_buffer.assert_not_called()


@pytest.mark.parametrize("case,exception,match", [
    ("foreign", ValueError, "another plan"),
    ("closed", RuntimeError, "closed"),
    ("shape", ValueError, "matching contiguous"),
    ("dtype", ValueError, "matching contiguous"),
    ("strided", ValueError, "Noncontiguous|matching contiguous"),
])
def test_copy_validates_descriptors_before_alias_elision(case, exception, match):
    plan = recording_plan()
    source, target = tensor(plan, (2, 4)), tensor(plan, (2, 4))
    if case == "foreign":
        source = tensor(recording_plan(), source.shape)
    elif case == "closed":
        plan.handle = None
    elif case == "shape":
        target = tensor(plan, (4, 2))
    elif case == "dtype":
        source.itemsize = target.itemsize = 4
    elif case == "strided":
        source.stride = target.stride = (5, 1)
    with pytest.raises(exception, match=match):
        target.copy_(source)
    plan.dispatch.assert_not_called()
    plan.copy_buffer.assert_not_called()


@pytest.mark.parametrize("different", ["offset", "buffer"])
def test_copy_preserves_distinct_storage(different):
    plan = recording_plan()
    source, target = tensor(plan, (4,)), tensor(plan, (4,))
    if different == "offset":
        target.offset = 8
    else:
        target.buffer = 1
    target.copy_(source)
    plan.copy_buffer.assert_called_once()
    plan.dispatch.assert_not_called()


@pytest.mark.parametrize("case", ["channels", "sequence", "rank", "output", "missing_output",
                                 "qkv_dtype", "output_dtype", "layout", "foreign_output"])
def test_vision_attention_rejects_invalid_descriptors_before_dispatch(case):
    plan = recording_plan()
    qkv = tensor(plan, (1, 256, 3456))
    out = tensor(plan, (1, 256, 1152))
    if case == "channels":
        qkv = tensor(plan, (1, 256, 3457))
    elif case == "sequence":
        qkv = tensor(plan, (1, 255, 3456))
    elif case == "rank":
        qkv = tensor(plan, (256, 3456))
    elif case == "output":
        out = tensor(plan, (1, 128, 1152))
    elif case == "missing_output":
        out = None
    elif case == "qkv_dtype":
        qkv = tensor(plan, qkv.shape, itemsize=4)
    elif case == "output_dtype":
        out = tensor(plan, out.shape, itemsize=4)
    elif case == "layout":
        qkv = tensor(plan, qkv.shape, stride=(256 * 3458, 3458, 1))
    elif case == "foreign_output":
        out = tensor(recording_plan(), out.shape)
    with pytest.raises(ValueError, match="vision attention|another plan|Noncontiguous"):
        Attention(plan).vision(qkv, out=out)
    plan.dispatch.assert_not_called()
    plan.allocate.assert_not_called()


def test_cpu_reference_uses_current_observation_and_prompt_without_gpu_staging():
    from examples.apple.pi05 import _cpu_reference_inputs

    frontend = SimpleNamespace(
        dtype=torch.bfloat16,
        _prompt_len=3,
        _prompt_cpu=torch.arange(20, dtype=torch.bfloat16).reshape(5, 4),
        # Native inference leaves these legacy MPS staging buffers untouched.
        _image_buf=torch.full((2, 2, 2, 3), 42.0),
        _prompt_buf=torch.full((5, 4), 42.0),
    )
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3) * 11
    for current in (pixels, 255 - pixels):
        images, prompt = _cpu_reference_inputs(
            frontend, {"image": current[0], "wrist_image": current[1]}
        )
        expected = (torch.tensor(current, dtype=torch.float32) / 127.5 - 1).bfloat16()
        torch.testing.assert_close(images, expected, rtol=0, atol=0)
        torch.testing.assert_close(prompt, frontend._prompt_cpu[:3], rtol=0, atol=0)
        assert images.device.type == prompt.device.type == "cpu"
        assert prompt.data_ptr() != frontend._prompt_cpu.data_ptr()


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("case", ["output", "missing_output", "rank", "dim", "kv_heads",
                                 "kv_shape", "prefix", "capacity", "layout",
                                 "q_dtype", "k_dtype", "v_dtype", "output_dtype", "foreign_output"])
def test_gqa_rejects_invalid_descriptors_before_scratch_or_dispatch(split, case):
    plan = recording_plan()
    q = tensor(plan, (2, 1, 256))
    k = tensor(plan, (130, 1, 256))
    v = tensor(plan, k.shape)
    out = tensor(plan, (2, 256))
    prefix, capacity = 128, 128
    if case == "output":
        out = tensor(plan, (1, 256))
    elif case == "missing_output":
        out = None
    elif case == "rank":
        q = tensor(plan, (2, 256))
    elif case == "dim":
        q = tensor(plan, (2, 1, 128))
    elif case == "kv_heads":
        k = v = tensor(plan, (130, 2, 256))
    elif case == "kv_shape":
        v = tensor(plan, (129, 1, 256))
    elif case == "prefix":
        prefix = 127
    elif case == "capacity":
        capacity = 127
    elif case == "layout":
        q = tensor(plan, q.shape, stride=(512, 256, 1))
    elif case == "q_dtype":
        q = tensor(plan, q.shape, itemsize=4)
    elif case == "k_dtype":
        k = tensor(plan, k.shape, itemsize=4)
    elif case == "v_dtype":
        v = tensor(plan, v.shape, itemsize=4)
    elif case == "output_dtype":
        out = tensor(plan, out.shape, itemsize=4)
    elif case == "foreign_output":
        out = tensor(recording_plan(), out.shape)
    with pytest.raises(ValueError, match="GQA|another plan|Noncontiguous"):
        Attention(plan, split_decode=split).gqa(q, k, v, valid_prefix=prefix,
                                               prefix_capacity=capacity, out=out)
    plan.dispatch.assert_not_called()
    plan.allocate.assert_not_called()
