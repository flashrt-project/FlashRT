"""SmoothQuant observer, channel transform and independent integer-dot checks."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from flash_rt.hardware.apple_native.int8 import INT8Plan, smoothquant_factors
from flash_rt.hardware.apple_native.plan import Plan, Weight

LIBRARY = os.environ.get("FLASHRT_APPLE_INT8_LIBRARY")
gpu = pytest.mark.skipif(
    not LIBRARY or not Path(LIBRARY).is_file(), reason="Requires INT8 plan library"
)


def test_channel_transform_preserves_float_matmul():
    torch.manual_seed(43)
    x, w = torch.randn(11, 257), torch.randn(259, 257)
    x[:, 0] *= 500
    w[:, 0] *= 0.001
    a = x.abs().amax(0).numpy()
    s = smoothquant_factors(a, w)
    np.testing.assert_allclose(
        ((x / s) @ (w * s).T).numpy(), (x @ w.T).numpy(), rtol=2e-4, atol=5e-5
    )
    independent = (
        np.maximum(a, 1e-5) ** 0.5 / np.maximum(w.abs().amax(0).numpy(), 1e-5) ** 0.5
    ).clip(1e-5)
    np.testing.assert_allclose(s.numpy(), independent, rtol=2e-6)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_alpha_guard(alpha):
    with pytest.raises(ValueError, match="alpha"):
        INT8Plan("missing", smooth_alpha=alpha)


@pytest.mark.parametrize("a", [[1, 2], [float("nan")] * 256, [-1] * 256])
def test_channel_guards(a):
    with pytest.raises(ValueError, match="channel maxima"):
        smoothquant_factors(a, torch.ones(259, 256))


@gpu
@pytest.mark.parametrize("m", [10, 17])
def test_observed_bf16_and_smoothed_gpu_match_cpu(m):
    torch.manual_seed(43)
    k, n = 257, 259
    x, w = torch.randn(m, k), torch.randn(k, n)
    x[:, 0] *= 500
    w[0] *= 0.001
    x[:, -1], w[-1] = 0, 0
    x, w = x.bfloat16(), w.bfloat16()
    bias = torch.randn(n).bfloat16()

    def build(p):
        p.stage = "prefix"
        a, out = p.allocate((m, k)), p.allocate((m, n))
        p.linear(out, a, Weight(w, "w"), Weight(bias, "bias"))
        p.finish()
        p.set_resident(True)
        return a, out

    reference, observer = (
        Plan(LIBRARY),
        INT8Plan(LIBRARY, stages=("prefix",), observe=True),
    )
    smoothed = None
    try:
        ra, ro = build(reference)
        oa, oo = build(observer)
        for value in (x, x.flip(-1) * 0.25):
            for p, a in ((reference, ra), (observer, oa)):
                p.write(a, value)
                p.run()
            np.testing.assert_array_equal(reference.read(ro), observer.read(oo))
        maxima = observer.activation_maxima()
        expected_max = np.maximum(
            x.float().abs().amax(0).numpy(),
            (x.flip(-1) * 0.25).float().abs().amax(0).numpy(),
        )
        np.testing.assert_array_equal(next(iter(maxima.values())), expected_max)
        smoothed = INT8Plan(LIBRARY, stages=("prefix",), smooth_amax=maxima)
        a, out = build(smoothed)
        # Independent NumPy scale equation, integer-dot oracle and BF16 bias boundary.
        wf = w.float().numpy()
        s = np.maximum(
            np.maximum(expected_max, np.float32(1e-5)) ** np.float32(0.5)
            / np.maximum(np.abs(wf).max(1), np.float32(1e-5)) ** np.float32(0.5),
            np.float32(1e-5),
        )
        transformed_w = wf * s[:, None]
        ws = np.maximum(
            np.abs(transformed_w).max(0) / np.float32(127), np.float32(1e-12)
        )
        qw = np.rint(transformed_w / ws).clip(-127, 127).astype(np.int64)
        for value in (x, x.flip(0) * 0.125):
            smoothed.write(a, value)
            smoothed.run("direct")
            actual = smoothed.read(out)
            smoothed.run("icb")
            np.testing.assert_array_equal(actual, smoothed.read(out))
            xf = value.float().numpy() * (np.float32(1) / s)
            sa = np.maximum(np.abs(xf).max(1) / np.float32(127), np.float32(1e-12))
            qa = (
                np.rint(xf * (np.float32(1) / sa[:, None]))
                .clip(-127, 127)
                .astype(np.int64)
            )
            expected = (
                torch.from_numpy(
                    (qa @ qw).astype(np.float32) * (sa[:, None] * ws)
                ).bfloat16()
                + bias
            )
            np.testing.assert_array_equal(actual, expected.float().numpy())
    finally:
        reference.close()
        observer.close()
        if smoothed is not None:
            smoothed.close()


@gpu
def test_missing_scales_and_nonfinite_observations_fail():
    p = INT8Plan(LIBRARY, stages=("prefix",), smooth_amax={})
    try:
        p.stage = "prefix"
        with pytest.raises(ValueError, match="Missing SmoothQuant"):
            p.linear(
                p.allocate((1, 256)),
                p.allocate((1, 256)),
                Weight(torch.ones(256, 256, dtype=torch.bfloat16), "w"),
            )
    finally:
        p.close()
    p = INT8Plan(LIBRARY, stages=("prefix",), observe=True)
    try:
        p.stage = "prefix"
        a, out = p.allocate((1, 256)), p.allocate((1, 256))
        p.write(a, np.full((1, 256), np.nan, np.float32))
        p.linear(out, a, Weight(torch.ones(256, 256, dtype=torch.bfloat16), "w"))
        p.finish()
        p.run()
        with pytest.raises(ValueError, match="Nonfinite SmoothQuant"):
            p.activation_maxima()
    finally:
        p.close()
