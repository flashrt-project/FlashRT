"""Affine int4 (group 64) kernels (row-vector, dense, split-K and expert-grouped products) and the multi-program
plan API on Metal, against a NumPy reference.

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
    from flash_rt.models.flint.pipeline_mps import GEMM_THREADS
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
                  (N // 64, (M + 63) // 64, 1), GEMM_THREADS)
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


def test_split_k_product():
    """A product of 40 rows, N = 512, K = 1024 in 16 K partitions: MLX's split-K path for a short chunk's
    K/V projections."""
    from flash_rt.hardware.apple_native.native import READ, READ_WRITE, NativePlan
    from flash_rt.models.flint.mlx_checkpoint import bf16_to_f32, dequantize, f32_to_bf16
    from flash_rt.models.flint.pipeline_mps import SPLITK_ROW_TILES, SPLITK_THREADS, TM, TN
    rng = np.random.default_rng(11)
    M, N, K = 40, 512, 1024
    m = _affine(rng, N, K)
    x = f32_to_bf16(rng.standard_normal((M, K)).astype(np.float32))
    plan = NativePlan(LIBRARY)
    offsets = [0, m.codes.nbytes, m.codes.nbytes + m.scales.nbytes]
    weights = plan.buffer(offsets[2] + m.biases.nbytes)
    for offset, array in zip(offsets, (m.codes, m.scales, m.biases)):
        plan.write(weights, offset, array)
    xb, y = plan.buffer(x.nbytes), plan.buffer(M * N * 2)
    state, empty, args = plan.buffer(16), plan.buffer(1024), plan.buffer(256)
    plan.write(xb, 0, x)
    plan.write(state, 0, np.array([0, M, 0, M], np.uint32))
    plan.write(args, 0, np.frombuffer(_block([M, N, K, K, N, 0, 0, 0], offsets * 2), np.uint8))
    program = plan.program()
    plan.dispatch("gemm_w4a16_affine_splitk", [(xb, 0, READ), (weights, 0, READ), (y, 0, READ_WRITE),
                                               (args, 0, READ), (empty, 0, READ), (empty, 0, READ), (state, 0, READ)],
                  (N // TN, min(-(-M // TM), SPLITK_ROW_TILES), 1), SPLITK_THREADS)
    plan.finish(resident=True)
    plan.run(program)
    ref = bf16_to_f32(x).astype(np.float64) @ dequantize(m).astype(np.float64).T
    got = bf16_to_f32(plan.read(y, 0, np.uint16, M * N)).reshape(M, N)
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-2 * np.abs(ref).max())
    plan.close()


@pytest.mark.parametrize("swiglu", [False, True])
def test_grouped_product(swiglu):
    """Expert-grouped product with per-expert row counts that reach every block shape (8-row steps of the
    64-row expert blocks, full blocks, empty experts) and a partial last column tile."""
    from flash_rt.hardware.apple_native.native import READ, READ_WRITE, NativePlan
    from flash_rt.models.flint.mlx_checkpoint import AffineMatrix, bf16_to_f32, dequantize, f32_to_bf16
    from flash_rt.models.flint.pipeline_mps import GEMM_THREADS, GROUPED_TM, TN
    rng = np.random.default_rng(13)
    counts = np.array([0, 1, 7, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 63, 64, 65, 100, 130, 0, 3, 200])
    E, N, K = counts.size, 96, 512
    total = int(counts.sum())
    assert total >= 1024           # the grouped kernels serve products from 1024 assignments
    mats = [_affine(rng, E * N, K) for _ in range(2 if swiglu else 1)]
    x = f32_to_bf16(rng.standard_normal((total, K)).astype(np.float32))
    plan = NativePlan(LIBRARY)
    offsets, size = [], 0
    for m in mats:
        for array in (m.codes, m.scales, m.biases):
            offsets.append(size)
            size += -(-array.nbytes // 256) * 256
    weights = plan.buffer(size)
    for offset, array in zip(offsets, [a for m in mats for a in (m.codes, m.scales, m.biases)]):
        plan.write(weights, offset, array)
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.uint32)
    tiles = np.concatenate(([0], np.cumsum(-(-counts // GROUPED_TM)))).astype(np.uint32)
    xb, y, sb, tb = plan.buffer(x.nbytes), plan.buffer(total * N * 2), plan.buffer(starts.nbytes), plan.buffer(tiles.nbytes)
    state, args = plan.buffer(16), plan.buffer(256)
    plan.write(xb, 0, x)
    plan.write(sb, 0, starts)
    plan.write(tb, 0, tiles)
    plan.write(state, 0, np.array([0, total, 0, total], np.uint32))
    second = offsets[3:6] if swiglu else offsets[0:3]
    plan.write(args, 0, np.frombuffer(_block([total, N, K, K, N, E, 0, 1], offsets[0:3] + second), np.uint8))
    name = "gemm_w4a16_affine_grouped" + ("_swiglu" if swiglu else "")
    program = plan.program()
    plan.dispatch(name, [(xb, 0, READ), (weights, 0, READ), (y, 0, READ_WRITE), (args, 0, READ),
                         (sb, 0, READ), (tb, 0, READ), (state, 0, READ)],
                  (-(-N // TN), -(-total // GROUPED_TM) + E, 1), GEMM_THREADS)
    plan.finish(resident=True)
    plan.run(program)
    got = bf16_to_f32(plan.read(y, 0, np.uint16, total * N)).reshape(total, N)
    xf = bf16_to_f32(x).astype(np.float64)
    ws = [dequantize(m).astype(np.float64).reshape(E, N, K) for m in mats]
    for e in range(E):
        rows = slice(int(starts[e]), int(starts[e + 1]))
        ref = xf[rows] @ ws[0][e].T
        if swiglu:
            u = xf[rows] @ ws[1][e].T
            ref = ref / (1.0 + np.exp(-ref)) * u
        if ref.size:
            np.testing.assert_allclose(got[rows], ref, rtol=0, atol=3e-2 * np.abs(ref).max() + 1e-3,
                                       err_msg=f"expert {e} ({counts[e]} rows)")
    plan.close()
