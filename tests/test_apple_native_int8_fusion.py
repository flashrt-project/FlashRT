"""Strict BF16 producer/quantizer parity and setup lookahead boundaries."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from test_apple_native_int8 import raw_read

from flash_rt.hardware.apple_native.int8 import INT8Plan
from flash_rt.hardware.apple_native.plan import Weight

LIBRARY = os.environ.get("FLASHRT_APPLE_INT8_LIBRARY")
pytestmark = pytest.mark.skipif(
    not LIBRARY or not Path(LIBRARY).is_file(), reason="Requires INT8 plan library"
)


@pytest.mark.parametrize("m,k", [(17, 4097), (662, 16384)])
def test_fused_gelu_preserves_hidden_quantized_bytes_scales_and_replay(m, k):
    torch.manual_seed(2026)
    n = 259
    weight = torch.randn(k, n).bfloat16()
    key = repr(("prefix", "w", 0, (k, n), tuple(weight.stride())))
    maxima = {key: np.linspace(0.01, 16, k, dtype=np.float32)}
    plans = []
    try:
        for fuse in (False, True):
            p = INT8Plan(
                LIBRARY, stages=("prefix",), smooth_amax=maxima, fuse_gelu=fuse
            )
            plans.append(p)
            p.stage = "prefix"
            src, hidden, out = (
                p.allocate((m, 2 * k)),
                p.allocate((m, k)),
                p.allocate((m, n)),
            )
            p.ops.gelu_mul_merged(hidden, src)
            p.linear(out, hidden, Weight(weight, "w"))
            # Materialized BF16 hidden remains observable by later consumers.
            copy = p.allocate((m, k))
            p.copy_buffer(hidden, copy)
            p.finish()
            p.set_resident(True)
            p.test_tensors = src, hidden, out, copy
            assert p.linears[-1]["fused_gelu"] == fuse
        for nonfinite in (False, True):
            x = (torch.randn(m, 2 * k) * 4).bfloat16()
            x[0] = 0
            if nonfinite:
                x[1, 3], x[2, k + 7] = float("nan"), float("inf")
            outputs = []
            for p in plans:
                src, hidden, out, copy = p.test_tensors
                p.write(src, x)
                p.run("direct")
                direct = p.read(out)
                p.run("icb")
                np.testing.assert_array_equal(p.read(out), direct)
                np.testing.assert_array_equal(p.read(hidden), p.read(copy))
                qa, sa = p.scratch[m, (k + 127) // 128 * 128]
                outputs.append(
                    (
                        p.read(hidden),
                        p.read(out),
                        raw_read(p, qa, np.int8),
                        raw_read(p, sa, np.float32),
                    )
                )
            for old, new in zip(*outputs):
                np.testing.assert_array_equal(old, new)
    finally:
        for p in plans:
            p.close()


@pytest.mark.parametrize(
    "boundary", ["finish", "copy", "dispatch", "stage", "other_input"]
)
def test_pending_producer_flushes_before_other_operations(boundary):
    torch.manual_seed(2026)
    m, k, n = 17, 4096, 259
    w = torch.randn(k, n).bfloat16()
    key = repr(("prefix", "w", 0, (k, n), tuple(w.stride())))
    plans = []
    try:
        results = []
        value = torch.randn(m, k * 2).bfloat16()
        for fuse in (False, True):
            p = INT8Plan(
                LIBRARY,
                stages=("prefix", "decode"),
                smooth_amax={key: np.ones(k)},
                fuse_gelu=fuse,
            )
            plans.append(p)
            p.stage = "prefix"
            src, hidden = p.allocate((m, k * 2)), p.allocate((m, k))
            p.write(src, value)
            p.ops.gelu_mul_merged(hidden, src)
            if boundary == "copy":
                target = p.allocate((m, k))
                p.copy_buffer(hidden, target)
            elif boundary == "dispatch":
                p.ops.gelu(hidden, hidden)
            elif boundary in ("stage", "other_input"):
                if boundary == "stage":
                    p.stage = "decode"
                other = hidden if boundary == "stage" else p.allocate((m, k))
                if boundary == "other_input":
                    p.write(other, torch.ones(m, k))
                p.linear(p.allocate((m, n)), other, Weight(w, "w"))
                assert not p.linears[-1]["fused_gelu"]
            p.finish()
            p.run()
            results.append(p.read(target if boundary == "copy" else hidden))
            assert p.dispatches[0]["stage"] == "prefix"
        np.testing.assert_array_equal(*results)
    finally:
        for p in plans:
            p.close()


@pytest.mark.parametrize("tile", ["64x64", "64x128"])
@pytest.mark.parametrize("bias", [False, True])
def test_tiled_gemm_matches_original_integer_kernel(tile, bias):
    torch.manual_seed(2026)
    m, k, n = 65, 257, 259  # Tails in all dimensions, including padded K.
    w = Weight(torch.randn(k, n).bfloat16(), "w")
    b = Weight(torch.randn(n).bfloat16(), "b") if bias else None
    value = torch.randn(m, k).bfloat16()
    plans = []
    results = []
    try:
        for tiled in (False, True):
            p = INT8Plan(LIBRARY)
            plans.append(p)
            p.stage = "decode"
            original = p.dispatch

            def dispatch(
                name,
                tensors,
                writes,
                params,
                groups,
                threads=64,
                *,
                tiled=tiled,
                original=original,
                **kwargs,
            ):
                if tiled and name in ("int8_gemm", "int8_gemm_bias"):
                    tn = 64 if tile == "64x64" else 128
                    name = "int8_gemm_" + tile + ("_bias" if bias else "")
                    groups, threads = (
                        ((n + tn - 1) // tn, (m + 63) // 64),
                        (64 if tn == 64 else 128),
                    )
                return original(
                    name, tensors, writes, params, groups, threads, **kwargs
                )

            p.dispatch = dispatch
            x, out = p.allocate((m, k)), p.allocate((m, n))
            p.linear(out, x, w, b)
            p.finish()
            p.write(x, value)
            p.run()
            results.append(p.read(out))
        np.testing.assert_array_equal(*results)
    finally:
        for p in plans:
            p.close()
