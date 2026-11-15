"""Gated FFN split between the GPU and a Core ML program inside one command buffer."""

import importlib.util
from pathlib import Path
import platform

import numpy as np
import pytest
import torch

from flash_rt.hardware.apple_native.fused import FusedPlan
from flash_rt.hardware.apple_native.plan import Weight

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or not torch.backends.mps.is_available()
    or importlib.util.find_spec("coremltools") is None
    or not (
        Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib"
    ).is_file(),
    reason="Built native plan library, Apple host GPU and coremltools required",
)


def record(plan, m, channels, hidden, weights, x, residual):
    gate_up, down = weights
    plan.stage = "prefix"
    t = dict(x=plan.allocate((m, channels)), res=plan.allocate((m, channels)),
             merged=plan.allocate((m, 2 * hidden)), hidden=plan.allocate((m, hidden)),
             out=plan.allocate((m, channels)))
    plan.ops.rms_norm(t["x"], t["x"])
    plan.linear(t["merged"], t["x"], Weight(gate_up, "gate_up"))
    plan.ops.gelu_mul_merged(t["hidden"], t["merged"])
    plan.linear(t["out"], t["hidden"], Weight(down, "down"))
    plan.ops.residual(t["res"], t["out"], t["res"])
    plan.finish()
    plan.set_resident(True)

    def run(mode):
        plan.write(t["x"], x)
        plan.write(t["res"], residual)
        plan.run(mode)
        return plan.read(t["res"])
    return run


@pytest.mark.parametrize("m", [37, 64])
def test_coprocessor_ffn_matches_gpu_lowering(tmp_path, m):
    torch.manual_seed(4)
    channels, hidden = 256, 2048
    weights = ((torch.randn(channels, 2 * hidden) / channels ** 0.5).bfloat16(),
               (torch.randn(hidden, channels) / hidden ** 0.5).bfloat16())
    x = torch.randn(m, channels).bfloat16()
    residual = torch.randn(m, channels).bfloat16()
    gpu = FusedPlan()
    split = FusedPlan(coprocessor=dict(hidden=1024, cache_dir=tmp_path))
    try:
        reference = record(gpu, m, channels, hidden, weights, x, residual)("icb")
        run = record(split, m, channels, hidden, weights, x, residual)
        assert [f[0] for f in split.fused] == ["coprocessor_ffn"]
        assert split.coprocessor_tasks[0]["gpu_hidden"] == split.coprocessor_tasks[0]["coprocessor_hidden"] == 1024
        assert sum(op["kind"] == "external" for op in split.operations) == 2
        outputs = [run(mode) for mode in ("direct", "icb", "icb")]
        for later in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], later)
        actual = outputs[0]
        assert np.isfinite(actual).all()
        # FP16 coprocessor share versus BF16 GPU lowering of the same FFN.
        error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        assert error < 0.01, error
        # Diagnostic profile: the Core ML task runs synchronously on the host.
        profile = split.profile()
        assert len(profile) == len(split.operations)
        signal = [row for row in profile if row["kind"] == "external" and not row["wait"]]
        assert len(signal) == 1 and signal[0]["wall_us"] > 0
    finally:
        gpu.close()
        split.close()


def test_coprocessor_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown coprocessor options"):
        FusedPlan(coprocessor=dict(width=1))


def test_coprocessor_ungated_ffn_with_biases(tmp_path):
    torch.manual_seed(8)
    m, channels, hidden = 48, 192, 1536
    up, up_bias = (torch.randn(channels, hidden) / channels ** 0.5).bfloat16(), (torch.randn(hidden) * 0.1).bfloat16()
    down, down_bias = (torch.randn(hidden, channels) / hidden ** 0.5).bfloat16(), (torch.randn(channels) * 0.1).bfloat16()
    x, residual = torch.randn(m, channels).bfloat16(), torch.randn(m, channels).bfloat16()

    def build(plan):
        plan.stage = "vision"
        t = dict(x=plan.allocate((m, channels)), res=plan.allocate((m, channels)),
                 hidden=plan.allocate((m, hidden)), out=plan.allocate((m, channels)))
        plan.linear(t["hidden"], t["x"], Weight(up, "up"), Weight(up_bias, "up_b"))
        plan.ops.gelu(t["hidden"], t["hidden"])
        plan.linear(t["out"], t["hidden"], Weight(down, "down"), Weight(down_bias, "down_b"))
        plan.ops.residual(t["res"], t["out"], t["res"])
        plan.finish()
        plan.set_resident(True)

        def run(mode):
            plan.write(t["x"], x)
            plan.write(t["res"], residual)
            plan.run(mode)
            return plan.read(t["res"])
        return run

    gpu = FusedPlan()
    split = FusedPlan(coprocessor=dict(hidden=512, stages=("vision",), cache_dir=tmp_path))
    try:
        reference = build(gpu)("icb")
        run = build(split)
        assert [f[0] for f in split.fused] == ["coprocessor_ffn"]
        outputs = [run(mode) for mode in ("direct", "icb")]
        np.testing.assert_array_equal(outputs[0], outputs[1])
        error = np.linalg.norm(outputs[0] - reference) / np.linalg.norm(reference)
        assert error < 0.01, error
    finally:
        gpu.close()
        split.close()
