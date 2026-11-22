"""Concurrent dispatches of the native plan (``NativePlan.dispatch(..., barrier=False)``): a dispatch without a
barrier may overlap the ones since the last barrier, the next barrier waits for all of them.

Skipped unless the host is an Apple Silicon Mac with the plan library built (scripts/apple/build_native.sh).
"""
from __future__ import annotations

import os
from pathlib import Path
import platform

import numpy as np
import pytest

LIBRARY = Path(os.environ.get("FLASHRT_APPLE_PLAN_LIBRARY") or
               Path(__file__).parents[1] / ".cache/apple-native/libflashrt_apple_plan.dylib")
pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64"
                                or not LIBRARY.is_file(), reason="needs an Apple Silicon Mac and the plan library")

N = 1 << 20


def _bf16(values):
    # values are multiples of 1/4 below 64 in magnitude: exact in BF16, and so are their sums here
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("execution", ["icb", "direct"])
def test_barrier_waits_for_every_concurrent_dispatch(execution):
    from flash_rt.hardware.apple_native.native import READ, WRITE, NativePlan
    plan = NativePlan(LIBRARY)
    x0, x1, u, y0, y1, z0, z1 = (plan.buffer(N * 2) for _ in range(7))
    gate, prm = plan.buffer(64), plan.buffer(64)
    groups = (N // 256, 1, 1)

    def add(a, b, out, barrier=True):
        plan.dispatch("tr_residual", [(a, 0, READ), (b, 0, READ), (out, 0, WRITE), (gate, 0, READ), (prm, 0, READ)],
                      groups, 256, barrier=barrier)

    pid = plan.program()
    add(x0, u, y0)
    add(x1, u, y1, barrier=False)        # overlaps the first
    add(y0, y1, z0)                      # the barrier: both inputs complete
    add(y1, u, z1, barrier=False)        # overlaps the third, reads what the barrier waited for
    plan.finish()
    i = np.arange(N)
    a, b, c = (i % 61 - 30) * 0.25, (i % 53 - 26) * 0.5, (i % 7 - 3) * 0.25
    plan.write(prm, 0, np.array([N, 1, 0], np.uint32))
    plan.write(x0, 0, _bf16(a))
    plan.write(x1, 0, _bf16(b))
    plan.write(u, 0, _bf16(c))
    for _ in range(3):
        for buf in (y0, y1, z0, z1):
            plan.write(buf, 0, np.full(N, 0xFFFF, np.uint16))
        plan.run(pid, execution=execution)
        assert np.array_equal(_f32(plan.read(z0, 0, np.uint16, N)), (a + c) + (b + c))
        assert np.array_equal(_f32(plan.read(z1, 0, np.uint16, N)), (b + c) + c)


def test_default_dispatch_records_barriers():
    """barrier=True (the default) records the same program as before the option: one barrier per dispatch."""
    from flash_rt.hardware.apple_native.native import READ, WRITE, NativePlan
    plan = NativePlan(LIBRARY)
    x, y, gate, prm = plan.buffer(N * 2), plan.buffer(N * 2), plan.buffer(64), plan.buffer(64)
    pid = plan.program()
    for _ in range(8):     # y += x eight times: wrong unless every dispatch waits for the previous one
        plan.dispatch("tr_residual", [(y, 0, READ), (x, 0, READ), (y, 0, WRITE), (gate, 0, READ), (prm, 0, READ)],
                      (N // 256, 1, 1), 256)
    plan.finish()
    plan.write(prm, 0, np.array([N, 1, 0], np.uint32))
    plan.write(x, 0, _bf16(np.full(N, 0.5)))
    for execution in ("icb", "direct"):
        plan.write(y, 0, _bf16(np.zeros(N)))
        plan.run(pid, execution=execution)
        assert np.array_equal(_f32(plan.read(y, 0, np.uint16, N)), np.full(N, 4.0, np.float32))
