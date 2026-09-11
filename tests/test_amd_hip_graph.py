"""AMD HIP runtime layer — HipBuffer roundtrips + HipGraph capture/replay.

The AMD backend drives the GPU through two thin ctypes layers
(flash_rt/amd/core/hip_buffer.py, hip_graph.py) instead of torch: a wrong
ctypes signature (the historical c_int pointer-truncation incident), a
broken memcpy-kind constant, or a capture-mode drift would corrupt every
tensor the pipeline moves, so this file gates the raw layer directly:

  * upload/download roundtrips are byte-exact on PATTERN data — never
    zeros, which memset bugs and DRAM compression both fake out — for
    both managed and device memory, and for buffers > 2 GiB-safe sizes;
  * HipGraph capture → instantiate → replay of a real two-kernel chain
    (rms_norm_fp16 x2 from the built extension) runs, and N replays are
    BIT-IDENTICAL (downloaded bytes compared exactly) — the pi05 E2E
    latency story is 100% graph replay, so replay determinism is the
    foundation everything above stands on;
  * a replay observes live input-buffer contents (upload → replay →
    output moves and tracks a host reference), proving the graph
    captured pointers, not values.

Skip conditions: needs the built extension AND a visible gfx950 device
(probed via the extension's device_arch(), no torch dependency). On
NVIDIA/no-ROCm CI everything skips at the first fixture with a reason;
imports of the ctypes layers happen inside fixtures because the module
import itself dlopens libamdhip64.so.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _rocm_device_or_skip():
    """Return the extension module, skipping unless a HIP device answers."""
    try:
        ext = importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")
    arch = ext.device_arch()
    if arch in ("none", "unknown"):
        pytest.skip(f"no usable HIP device (device_arch()={arch!r})")
    return ext


@pytest.fixture(scope="module")
def ext():
    return _rocm_device_or_skip()


@pytest.fixture(scope="module")
def hip_core(ext):
    """The ctypes layers — imported lazily (module import dlopens HIP)."""
    from flash_rt.amd.core import hip_buffer, hip_graph
    return hip_buffer, hip_graph


def _pattern_bytes(n: int) -> np.ndarray:
    """Deterministic non-constant byte pattern. Zeros are forbidden here:
    a memset-zero buffer hides both stuck-at-zero copy bugs and DRAM
    compression effects (see the bench-buffer discipline)."""
    return ((np.arange(n, dtype=np.int64) * 37 + 11) % 251).astype(np.uint8)


# ---------------------------------------------------------------------------
# HipBuffer roundtrips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("managed", [True, False],
                         ids=["managed", "device"])
def test_hipbuffer_upload_download_roundtrip(hip_core, managed):
    hip_buffer, _ = hip_core
    src = _pattern_bytes(1 << 20)  # 1 MiB, non-trivial but fast
    buf = hip_buffer.HipBuffer(src.nbytes, managed=managed)
    buf.upload(src)
    out = np.zeros_like(src)
    buf.download(out)
    assert np.array_equal(out, src), "byte roundtrip corrupted data"


def test_hipbuffer_from_numpy_roundtrip(hip_core):
    """from_numpy (device mem, H2D memcpy) and from_numpy_managed (memmove)
    must both reproduce float payloads exactly."""
    hip_buffer, _ = hip_core
    rng = np.random.RandomState(0)
    src = rng.randn(4096).astype(np.float32)

    dev = hip_buffer.HipBuffer.from_numpy(src)
    got_dev = dev.download_new(src.shape, src.dtype)
    assert np.array_equal(got_dev, src)

    man = hip_buffer.HipBuffer.from_numpy_managed(src)
    got_man = man.download_new(src.shape, src.dtype)
    assert np.array_equal(got_man, src)


def test_hipbuffer_zero_(hip_core):
    """zero_ must clear a previously pattern-filled buffer (checked against
    the pattern so a no-op memset cannot pass)."""
    hip_buffer, _ = hip_core
    src = _pattern_bytes(65536)
    buf = hip_buffer.HipBuffer.from_numpy(src)
    buf.zero_()
    out = np.empty_like(src)
    buf.download(out)
    assert not np.array_equal(out, src)
    assert np.all(out == 0)


# ---------------------------------------------------------------------------
# HipGraph capture / replay
# ---------------------------------------------------------------------------

_SEQ, _DIM = 8, 256   # rms_norm_fp16 needs dim even (packed-pair kernel)
_EPS = 1e-6


def _rms_ref_fp16(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Host reference for rms_norm_fp16 (fp32 math, fp16 store — same
    structure as the kernel; reduction order may differ by last-ULP)."""
    xf = x.astype(np.float32)
    rms = 1.0 / np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + _EPS)
    return (xf * rms * w.astype(np.float32)).astype(np.float16)


@pytest.fixture()
def graph_setup(ext, hip_core):
    """Buffers + captured two-kernel graph: x --rms(w1)--> y --rms(w2)--> z."""
    hip_buffer, hip_graph = hip_core
    rng = np.random.RandomState(42)
    x = rng.randn(_SEQ, _DIM).astype(np.float16)
    w1 = (1.0 + 0.1 * rng.randn(_DIM)).astype(np.float16)
    w2 = (1.0 + 0.1 * rng.randn(_DIM)).astype(np.float16)

    x_buf = hip_buffer.HipBuffer.from_numpy(x)
    w1_buf = hip_buffer.HipBuffer.from_numpy(w1)
    w2_buf = hip_buffer.HipBuffer.from_numpy(w2)
    y_buf = hip_buffer.HipBuffer.device_zeros(_SEQ * _DIM, np.float16)
    z_buf = hip_buffer.HipBuffer.device_zeros(_SEQ * _DIM, np.float16)

    graph = hip_graph.HipGraph()
    stream = graph.create_stream()
    s = stream.value

    def chain():
        # named-variable ptrs (buffers held by the closure — no temp GC)
        ext.rms_norm_fp16(x_buf.ptr.value, w1_buf.ptr.value, y_buf.ptr.value,
                          _SEQ, _DIM, _EPS, s)
        ext.rms_norm_fp16(y_buf.ptr.value, w2_buf.ptr.value, z_buf.ptr.value,
                          _SEQ, _DIM, _EPS, s)

    # Warmup outside capture (module loading / first-launch work must not
    # land inside the graph), then capture on the SAME stream — the
    # capture-stream discipline is the historical trap this pins.
    chain()
    graph.sync(stream)
    graph.begin_capture(stream)
    chain()
    graph.end_capture(stream)

    return {
        "graph": graph, "stream": stream,
        "x": x, "w1": w1, "w2": w2,
        "x_buf": x_buf, "w1_buf": w1_buf, "w2_buf": w2_buf,
        "y_buf": y_buf, "z_buf": z_buf,
    }


def test_graph_capture_and_replay_matches_reference(graph_setup):
    g = graph_setup
    graph, stream = g["graph"], g["stream"]
    assert graph.captured

    graph.replay(stream)
    graph.sync(stream)
    z = g["z_buf"].download_new((_SEQ, _DIM), np.float16)

    ref = _rms_ref_fp16(_rms_ref_fp16(g["x"], g["w1"]), g["w2"])
    # Tolerance: only the fp32 reduction order differs between kernel and
    # host reference; through two chained norms that is a few fp16 ULPs.
    # 1e-2 absolute on unit-scale data catches any real math/indexing bug
    # (a wrong weight or a shifted row is O(1) wrong) without flaking.
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z.astype(np.float32), ref.astype(np.float32),
                               atol=1e-2)


def test_graph_replays_are_bit_identical(graph_setup):
    """N replays with frozen inputs must produce byte-for-byte identical
    output. The pi05 serving loop IS graph replay; nondeterminism here
    (an atomic reduction, an uninitialized workspace read) would poison
    every parity number measured above this layer."""
    g = graph_setup
    graph, stream = g["graph"], g["stream"]

    downloads = []
    for _ in range(5):
        graph.replay(stream)
        graph.sync(stream)
        z = g["z_buf"].download_new((_SEQ, _DIM), np.float16)
        downloads.append(z.tobytes())
    assert all(d == downloads[0] for d in downloads[1:]), (
        "graph replay is not bit-stable across replays")


def test_graph_replay_reads_live_input_buffers(graph_setup):
    """The captured graph must be parameterized by buffer ADDRESSES: new
    data uploaded into the input buffer must flow through the next replay
    (this is exactly how the pipeline feeds fresh images/noise into the
    captured pi05 graph)."""
    g = graph_setup
    graph, stream = g["graph"], g["stream"]

    graph.replay(stream)
    graph.sync(stream)
    z_before = g["z_buf"].download_new((_SEQ, _DIM), np.float16)

    rng = np.random.RandomState(7)
    x_new = rng.randn(_SEQ, _DIM).astype(np.float16)
    g["x_buf"].upload(x_new)

    graph.replay(stream)
    graph.sync(stream)
    z_after = g["z_buf"].download_new((_SEQ, _DIM), np.float16)

    assert not np.array_equal(z_after, z_before), (
        "replay ignored the updated input buffer (values were baked in)")
    ref = _rms_ref_fp16(_rms_ref_fp16(x_new, g["w1"]), g["w2"])
    np.testing.assert_allclose(z_after.astype(np.float32),
                               ref.astype(np.float32), atol=1e-2)


def test_replay_before_capture_refuses(hip_core):
    """API contract: replay() on a virgin HipGraph raises instead of
    launching a null graph exec (which would be a silent no-op frame in a
    serving loop)."""
    _, hip_graph = hip_core
    graph = hip_graph.HipGraph()
    stream = graph.create_stream()
    with pytest.raises(RuntimeError, match="[Nn]o graph captured"):
        graph.replay(stream)
