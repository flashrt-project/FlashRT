"""Affine int4 (group 64) kernels and the multi-program plan API on Metal, against a NumPy reference.

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


def _affine(rng, rows, k):
    from flash_rt.models.flint.mlx_checkpoint import AffineMatrix, f32_to_bf16
    q = rng.integers(0, 16, size=(rows, k), dtype=np.uint8)
    codes = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    scales = f32_to_bf16(rng.uniform(0.005, 0.02, (rows, k // 64)).astype(np.float32))
    biases = f32_to_bf16(rng.uniform(-0.1, 0.1, (rows, k // 64)).astype(np.float32))
    return AffineMatrix(codes, scales, biases, 4)


def _block(uints, ulongs):
    """A parameter struct: uint32 fields, then 8-byte aligned uint64 fields (MSL struct order)."""
    raw = np.array(uints, np.uint32).tobytes()
    raw += b"\0" * (-len(raw) % 8)
    return raw + np.array(ulongs, np.uint64).tobytes()


def test_affine_products_and_programs():
    from flash_rt.hardware.apple_native.native import READ, READ_WRITE, WRITE, NativePlan
    from flash_rt.models.flint.mlx_checkpoint import bf16_to_f32, dequantize, f32_to_bf16
    rng = np.random.default_rng(7)
    M, N, K = 544, 512, 512          # one partition (no split-K) for the dense product
    m = _affine(rng, N, K)
    x = f32_to_bf16(rng.standard_normal((M, K)).astype(np.float32))
    plan = NativePlan(LIBRARY)
    offsets = [0, m.codes.nbytes, m.codes.nbytes + m.scales.nbytes]
    weights = plan.buffer(offsets[2] + m.biases.nbytes)
    for offset, array in zip(offsets, (m.codes, m.scales, m.biases)):
        plan.write(weights, offset, array)
    xb, yv, ym = plan.buffer(x.nbytes), plan.buffer(N * 2), plan.buffer(M * N * 2)
    state, empty, args = plan.buffer(16), plan.buffer(1024), plan.buffer(4096)
    plan.write(xb, 0, x)
    plan.write(state, 0, np.array([0, M, 0, M], np.uint32))
    # AffineQmv {inner, columns, x_stride, y_stride, limit, experts, per_row, pad; w, s, b, w2, s2, b2, x, y}
    # and AffineGemm {rows, columns, inner, x_stride, y_stride, experts, limit, per_row; w, s, b, w2, s2, b2}.
    vec, mat = 0, 256
    plan.write(args, vec, np.frombuffer(_block([K, N, 0, 0, 0, 0, 0, 0], offsets * 2 + [0, 0]), np.uint8))
    plan.write(args, mat, np.frombuffer(_block([M, N, K, K, N, 0, 0, 0], offsets * 2), np.uint8))

    first = plan.program()
    plan.dispatch("qmv_w4a16_affine", [(weights, 0, READ), (xb, 0, READ), (yv, 0, WRITE), (args, vec, READ)],
                  (1, N // 8, 1), 64)
    second = plan.program()
    plan.dispatch("gemm_w4a16_affine", [(xb, 0, READ), (weights, 0, READ), (ym, 0, READ_WRITE), (args, mat, READ),
                                        (empty, 0, READ), (empty, 0, READ), (state, 0, READ)],
                  (N // 64, (M + 63) // 64, 1), 64)
    plan.finish(resident=True)
    assert (first, second) == (1, 2)
    with pytest.raises(RuntimeError, match="sealed"):
        plan.buffer(64)

    w = dequantize(m).astype(np.float64)
    xf = bf16_to_f32(x).astype(np.float64)
    plan.run(first)
    ref = w @ xf[0]
    got = bf16_to_f32(plan.read(yv, 0, np.uint16, N))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-2 * np.abs(ref).max())
    plan.run(second)
    ref = xf @ w.T
    got = bf16_to_f32(plan.read(ym, 0, np.uint16, M * N)).reshape(M, N)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-2 * np.abs(ref).max())
    # The direct (per-dispatch encoding) and ICB executions of a program agree bit for bit.
    plan.run(second, execution="direct")
    np.testing.assert_array_equal(bf16_to_f32(plan.read(ym, 0, np.uint16, M * N)).reshape(M, N), got)

    profile = plan.profile(second)
    assert [label for label, _ in profile] == ["gemm_w4a16_affine"] and profile[0][1]["gpu_us"] > 0
    with pytest.raises(RuntimeError, match="unknown program"):
        plan.run(3)
    with pytest.raises(RuntimeError, match="unknown program"):
        plan.run(0)                  # nothing was recorded before the first program
    plan.close()
