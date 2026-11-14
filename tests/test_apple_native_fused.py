"""Fused BF16 native lowering: parity with the one-dispatch-per-operation lowering.

Large-row epilogues keep every BF16 boundary of the separate dispatches and must be
bit-identical for the same tile. Few-row split-K sums FP32 partials in a fixed order,
so it is compared with the unfused lowering within BF16 rounding and must replay
deterministically.
"""

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
    or not (
        Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib"
    ).is_file(),
    reason="Built native plan library and Apple host GPU required",
)


def replay(plan, outputs):
    # In-place residual streams are rewritten before every run.
    plan.finish()
    plan.set_resident(True)
    runs = []
    for mode in ("direct", "icb", "icb"):
        plan.reset()
        plan.run(mode)
        runs.append([plan.read(t) for t in outputs])
    for later in runs[1:]:
        for a, b in zip(runs[0], later):
            np.testing.assert_array_equal(a, b)
    return runs[0]


def build(fuse, tune, record, inputs):
    p = FusedPlan(fuse_epilogues=fuse, tune_tiles=tune)
    p.stage = "prefix"
    tensors = {k: p.allocate(v.shape) for k, v in inputs.items()}

    def reset():
        for k, v in inputs.items():
            p.write(tensors[k], v)
    p.reset = reset
    outputs = record(p, tensors)
    return p, outputs


@pytest.mark.parametrize("pattern", ["gelu_mul", "residual", "bias_gelu", "bias_residual", "bias"])
@pytest.mark.parametrize("m,k,n", [(37, 192, 256), (70, 8192, 256), (65, 256, 8192)])
@pytest.mark.parametrize("tune", [False, True])
def test_large_row_epilogues_match_unfused_bits(pattern, m, k, n, tune):
    torch.manual_seed(7)
    weight = (torch.randn(k, n) / k ** 0.5).bfloat16()
    bias = (torch.randn(n) * 0.1).bfloat16()
    inputs = dict(x=torch.randn(m, k).bfloat16(), y=torch.randn(m, n).bfloat16(),
                  hidden=torch.zeros(m, n // 2).bfloat16())
    inputs["x"][0, :7] = 0

    def record(p, t):
        b = Weight(bias, "b") if pattern.startswith("bias") else None
        if pattern == "gelu_mul":
            p.linear(t["y"], t["x"], Weight(weight, "w"))
            p.ops.gelu_mul_merged(t["hidden"], t["y"])
            return [t["hidden"]]
        if pattern in ("residual", "bias_residual"):
            out = p.allocate((m, n))
            p.linear(out, t["x"], Weight(weight, "w"), b)
            p.ops.residual(t["y"], out, t["y"])
            return [t["y"]]
        p.linear(t["y"], t["x"], Weight(weight, "w"), b)
        if pattern == "bias_gelu":
            p.ops.gelu(t["y"], t["y"])
        return [t["y"]]

    results = []
    for fuse in (False, True):
        p, outputs = build(fuse, tune, record, inputs)
        try:
            if fuse and pattern != "bias":
                assert p.fused, "expected a fused epilogue"
            results.append(replay(p, outputs))
        finally:
            p.close()
    for unfused, fused in zip(*results):
        np.testing.assert_array_equal(unfused, fused)


@pytest.mark.parametrize("boundary", ["finish", "copy", "dispatch", "other_output"])
def test_large_row_lookahead_flushes_in_semantic_order(boundary):
    torch.manual_seed(3)
    m, k, n = 40, 128, 256
    weight = (torch.randn(k, n) / k ** 0.5).bfloat16()
    inputs = dict(x=torch.randn(m, k).bfloat16(), y=torch.zeros(m, n).bfloat16(),
                  hidden=torch.zeros(m, n // 2).bfloat16(), other=torch.randn(m, n).bfloat16())

    def record(p, t):
        p.linear(t["y"], t["x"], Weight(weight, "w"))
        extra = []
        if boundary == "copy":
            copy = p.allocate((m, n))
            p.copy_buffer(t["y"], copy)
            extra.append(copy)
        elif boundary == "dispatch":
            p.ops.residual(t["other"], t["other"], t["other"])
        elif boundary == "other_output":
            p.ops.gelu_mul_merged(t["hidden"], t["other"])
        return [t["y"], t["hidden"], t["other"], *extra]

    results = []
    for fuse in (False, True):
        p, outputs = build(fuse, True, record, inputs)
        try:
            assert not p.fused
            results.append(replay(p, outputs))
        finally:
            p.close()
    for unfused, fused in zip(*results):
        np.testing.assert_array_equal(unfused, fused)


def bf16_close(actual, expected):
    # Split-K changes only the FP32 summation order before the BF16 rounding.
    tolerance = np.abs(expected) * 2.0 ** -7 + 1e-6
    bad = np.abs(actual - expected) > tolerance
    assert bad.mean() < 0.002 and np.abs(actual - expected).max() < 0.05


@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("k", [1024, 4096])
def test_few_row_splitk_residual_adarms(norm, k):
    torch.manual_seed(11)
    rows, d = 10, 1024
    weight = (torch.randn(k, d) / k ** 0.5).bfloat16()
    gate = (torch.rand(1, d) + 0.5).bfloat16()
    mod = (torch.randn(1, 3 * d) * 0.1).bfloat16()
    inputs = dict(x=torch.randn(rows, k).bfloat16(), res=torch.randn(rows, d).bfloat16(),
                  norm=torch.zeros(rows, d).bfloat16())

    def record(p, t):
        out = p.allocate((rows, d))
        p.linear(out, t["x"], Weight(weight, "w"))
        p.ops.residual(t["res"], out, t["res"], Weight(gate, "g"))
        if norm:
            p.ops.adarms(t["norm"], t["res"], None, None, None, modulation=Weight(mod, "m"))
        return [t["res"], t["norm"]]

    results = []
    for fuse in (False, True):
        p, outputs = build(fuse, True, record, inputs)
        try:
            results.append(replay(p, outputs))
            if fuse:
                # Without a following AdaRMS the held reduction is emitted at finish.
                kinds = [f[0] for f in p.fused]
                assert kinds == ["splitk_residual" + ("_adarms" if norm else "")]
        finally:
            p.close()
    for unfused, fused in zip(*results):
        bf16_close(fused, unfused)


@pytest.mark.parametrize("position", [0, 661])
def test_few_row_splitk_rope_places_kv(position):
    torch.manual_seed(5)
    rows, heads, dim, k = 10, 8, 256, 1024
    n = (heads + 2) * dim
    weight = (torch.randn(k, n) / k ** 0.5).bfloat16()
    table = torch.arange(position + rows)[:, None] * (1.0 / 10000 ** (torch.arange(0, dim, 2) / dim))
    inputs = dict(x=torch.randn(rows, k).bfloat16(), q=torch.zeros(rows, heads, dim).bfloat16(),
                  kcache=torch.zeros(position + rows + 4, 1, dim).bfloat16(),
                  vcache=torch.zeros(position + rows + 4, 1, dim).bfloat16())

    def record(p, t):
        qkv = p.allocate((rows, n))
        p.linear(qkv, t["x"], Weight(weight, "w"))
        suffix = slice(position, position + rows)
        p.ops.qkv_rope(t["q"], t["kcache"][suffix], t["vcache"][suffix], qkv,
                       Weight(table.cos().bfloat16(), "cos"), Weight(table.sin().bfloat16(), "sin"), position)
        return [t["q"], t["kcache"], t["vcache"]]

    results = []
    for fuse in (False, True):
        p, outputs = build(fuse, True, record, inputs)
        try:
            if fuse:
                assert [f[0] for f in p.fused] == ["splitk_rope"]
            results.append(replay(p, outputs))
        finally:
            p.close()
    for unfused, fused in zip(*results):
        bf16_close(fused, unfused)


def test_few_row_gate_up_gelu_mul_matches_unfused_bits():
    torch.manual_seed(9)
    rows, k, h = 10, 1024, 4096
    weight = (torch.randn(k, 2 * h) / k ** 0.5).bfloat16()
    inputs = dict(x=torch.randn(rows, k).bfloat16(), hidden=torch.zeros(rows, h).bfloat16())

    def record(p, t):
        merged = p.allocate((rows, 2 * h))
        p.linear(merged, t["x"], Weight(weight, "w"))
        p.ops.gelu_mul_merged(t["hidden"], merged)
        return [t["hidden"]]

    results = []
    for fuse in (False, True):
        p, outputs = build(fuse, True, record, inputs)
        try:
            results.append(replay(p, outputs))
        finally:
            p.close()
    np.testing.assert_array_equal(results[0][0], results[1][0])
