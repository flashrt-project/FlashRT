"""Dynamic INT8 byte, integer-dot and unchanged BF16 producer oracles."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from flash_rt.hardware.apple_native.int8 import INT8Plan
from flash_rt.hardware.apple_native.plan import Plan, Weight

LIBRARY = os.environ.get("FLASHRT_APPLE_INT8_LIBRARY")
gpu = pytest.mark.skipif(
    not LIBRARY or not Path(LIBRARY).is_file(),
    reason="Set FLASHRT_APPLE_INT8_LIBRARY to an INT8-enabled plan library",
)


def raw_read(plan, tensor, dtype):
    result = np.empty(tensor.shape, dtype)
    plan.check(
        plan.lib.frt_apple_plan_read(
            plan.handle, tensor.buffer, tensor.offset, result.ctypes.data, result.nbytes
        )
    )
    return result


@pytest.mark.parametrize(
    "m,k,n",
    [
        (10, 257, 259),
        (17, 256, 320),
        (10, 4304, 259),
        (17, 16384, 259),
        (512, 1152, 256),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@gpu
def test_int8_row_channel_gemm_and_replay(m, k, n, bias):
    torch.manual_seed(2026)
    x = torch.randn(m, k).bfloat16()
    w = (torch.randn(k, n) / k**0.5).bfloat16()
    # Very different row/channel ranges expose accidental per-tensor scales.
    x[0] = 0
    x[1] *= 100
    x[2] = 127
    w[:, 0] = 0
    w[:, 1] *= 10
    w[:, 2] = -127
    b = torch.randn(n).bfloat16() if bias else None
    plan = INT8Plan(LIBRARY)
    try:
        plan.stage = "decode"
        a, out = plan.allocate((m, k)), plan.allocate((m, n))
        plan.linear(out, a, Weight(w, "w"), Weight(b, "b") if bias else None)
        plan.finish()
        plan.set_resident(True)
        for factor in (1, 0.125):
            value = ((x if factor == 1 else x.flip(-1)) * factor).bfloat16()
            plan.write(a, value)
            plan.run("direct")
            actual = plan.read(out)
            plan.run("icb")
            np.testing.assert_array_equal(actual, plan.read(out))
            af, wf = value.float().numpy(), w.float().numpy()
            sa = np.maximum(
                np.max(np.abs(af), axis=1) / np.float32(127), np.float32(1e-12)
            )
            sw = np.maximum(
                np.max(np.abs(wf), axis=0) / np.float32(127), np.float32(1e-12)
            )
            qa = (
                np.rint(af * (np.float32(1) / sa[:, None]))
                .clip(-127, 127)
                .astype(np.int8)
            )
            qw = np.rint(wf / sw[None, :]).clip(-127, 127).astype(np.int8)
            stride = (k + 127) // 128 * 128
            stored_a, stored_sa = plan.scratch[m, stride]
            stored_w, stored_sw = next(iter(plan.packed_weights.values()))
            got_a, got_w = (
                raw_read(plan, stored_a, np.int8),
                raw_read(plan, stored_w, np.int8),
            )
            np.testing.assert_array_equal(got_a[:, :k], qa)
            np.testing.assert_array_equal(got_w[:, :k], qw.T)
            assert not got_a[:, k:].any() and not got_w[:, k:].any()
            np.testing.assert_array_equal(raw_read(plan, stored_sa, np.float32), sa)
            np.testing.assert_array_equal(raw_read(plan, stored_sw, np.float32), sw)
            dot = qa.astype(np.int64) @ qw.astype(np.int64)
            expected = torch.from_numpy(
                dot.astype(np.float32) * (sa[:, None] * sw[None, :])
            ).bfloat16()
            if bias:
                expected = expected + b
            np.testing.assert_array_equal(actual, expected.float().numpy())
    finally:
        plan.close()


@gpu
def test_int8_rounding_nonfinite_and_identity():
    plan = INT8Plan(LIBRARY)
    try:
        values = np.zeros((4, 256), np.float32)
        values[0, :7] = [127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
        values[1, 0], values[2, 0] = np.nan, np.inf
        a = plan.allocate(values.shape)
        q = plan.raw(np.ones(values.shape, np.int8))
        scales = plan.raw(np.zeros(4, np.float32))
        plan.write(a, values)
        plan.dispatch("int8_pack", [a, q, scales], {1, 2}, [4, 256, 256], (1,), 128)
        plan.finish()
        plan.run()
        np.testing.assert_array_equal(
            raw_read(plan, q, np.int8)[0, :7], [127, 0, 2, 2, 0, -2, -2]
        )
        assert np.isnan(raw_read(plan, scales, np.float32)[1:3]).all()
        assert not raw_read(plan, q, np.int8)[1:].any()
    finally:
        plan.close()
    plan = INT8Plan(LIBRARY)
    try:
        plan.stage = "decode"
        a, out = plan.allocate((1, 256)), plan.allocate((1, 256))
        w = Weight(torch.ones(256, 256, dtype=torch.bfloat16), "w")
        plan.linear(out, a, w)
        with pytest.raises(ValueError, match="Ambiguous INT8"):
            plan.linear(out, a, Weight(torch.zeros_like(w.tensor), "w"))
        with pytest.raises(ValueError, match="Nonfinite INT8 weight"):
            plan.linear(out, a, Weight(torch.full_like(w.tensor, float("nan")), "bad"))
    finally:
        plan.close()


@gpu
def test_unselected_stage_is_exact_bf16():
    torch.manual_seed(19)
    x, w = torch.randn(3, 256).bfloat16(), torch.randn(256, 256).bfloat16()
    results = []
    for cls in (Plan, INT8Plan):
        plan = cls(LIBRARY)
        try:
            plan.stage = "vision"
            a, out = plan.allocate(x.shape), plan.allocate((3, 256))
            plan.write(a, x)
            plan.linear(out, a, Weight(w, "w"))
            plan.finish()
            plan.run()
            results.append(plan.read(out))
        finally:
            plan.close()
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("stages", [(), ("unknown",), ("vision", "typo")])
def test_invalid_stage_fails_before_loading_library(stages):
    with pytest.raises(ValueError, match="INT8 stages"):
        INT8Plan("not-a-library", stages=stages)
