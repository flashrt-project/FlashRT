"""Phase-A toy validation of the FlashRT execution contract.

No real model — uses allocation-free, capture-safe memset/memcpy as the
"kernel" inside captured graphs, to exercise the mechanism only:
  1. capture -> replay
  2. zero-copy hand-off via a shared bound Buffer
  3. imperative multi-stream + event cross-stream sync
  4. frt_buffer_copy (device-to-device, for spec-decode-style snapshot/restore)
  5. ShapeKey variant table LRU eviction
  6. missing-variant replay returns NO_VARIANT (never a silent no-op)

Run inside the container after building exec/:
    cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release && cmake --build exec/build -j
    python exec/tests/test_exec.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "build"))

import torch  # noqa: E402
import _flashrt_exec as ex  # noqa: E402

FRT_ERR_NO_VARIANT = -2


def _buf(ctx, name, nbytes):
    """A device byte buffer wrapped from a torch tensor (so we can read it back)."""
    t = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    b = ctx.wrap(name, t.data_ptr(), nbytes)
    return t, b


def test_capture_replay():
    ctx = ex.Ctx()
    t, b = _buf(ctx, "b", 256)
    dptr = b.dptr()
    g = ctx.graph("memset7")
    g.capture(0, lambda stream: ex.memset_async(dptr, 7, 256, stream))
    t.zero_()
    assert g.replay(0, 0) == 0
    torch.cuda.synchronize()
    assert torch.all(t == 7), "capture/replay did not run the captured memset"
    print("PASS  1. capture -> replay")


def test_bind_handoff():
    ctx = ex.Ctx()
    tx, bx = _buf(ctx, "x", 256)   # shared hand-off buffer
    tb, bb = _buf(ctx, "out", 256)
    px, pb = bx.dptr(), bb.dptr()

    g1 = ctx.graph("producer")     # writes X = 5
    g1.capture(0, lambda s: ex.memset_async(px, 5, 256, s))
    g1.bind("out", bx)

    g2 = ctx.graph("consumer")     # copies X -> OUT
    g2.capture(0, lambda s: ex.memcpy_async(pb, px, 256, s))
    g2.bind("in", bx)              # SAME buffer => zero-copy hand-off
    g2.bind("out", bb)

    tb.zero_()
    assert g1.replay(0, 0) == 0
    assert g2.replay(0, 0) == 0
    torch.cuda.synchronize()
    assert torch.all(tb == 5), "hand-off via shared bound buffer failed"
    print("PASS  2. zero-copy hand-off via shared bound Buffer")


def test_multistream_event():
    ctx = ex.Ctx()
    tx, bx = _buf(ctx, "x", 256)
    tb, bb = _buf(ctx, "out", 256)
    px, pb = bx.dptr(), bb.dptr()

    s1 = ctx.stream(0)             # a second stream
    evt = ctx.event()

    gp = ctx.graph("prod")
    gp.capture(0, lambda s: ex.memset_async(px, 9, 256, s))
    gc = ctx.graph("cons")
    gc.capture(0, lambda s: ex.memcpy_async(pb, px, 256, s))

    tb.zero_()
    assert gp.replay(0, 0) == 0    # producer on stream 0
    evt.record(0)                  # record completion on stream 0
    ctx.stream_wait(s1, evt)       # stream s1 waits for it
    assert gc.replay(0, s1) == 0   # consumer on s1, ordered after producer
    torch.cuda.synchronize()
    assert torch.all(tb == 9), "imperative cross-stream event ordering failed"
    print("PASS  3. imperative multi-stream + event")


def test_buffer_copy():
    ctx = ex.Ctx()
    ts, bs = _buf(ctx, "src", 256)
    td, bd = _buf(ctx, "dst", 256)
    ps = bs.dptr()

    g = ctx.graph("fill3")
    g.capture(0, lambda s: ex.memset_async(ps, 3, 256, s))
    assert g.replay(0, 0) == 0
    ctx.copy(bd, 0, bs, 0, 256, 0)  # device-to-device snapshot/restore primitive
    torch.cuda.synchronize()
    assert torch.all(td == 3), "frt_buffer_copy did not copy device-to-device"
    print("PASS  4. frt_buffer_copy")


def test_lru_eviction():
    ctx = ex.Ctx()
    t, b = _buf(ctx, "b", 64)
    p = b.dptr()
    g = ctx.graph("lru", max_variants=2)
    for key in (1, 2, 3):           # capacity 2 -> capturing 3 evicts key 1
        g.capture(key, lambda s: ex.memset_async(p, 1, 64, s))
    assert not g.has_variant(1), "oldest variant should have been LRU-evicted"
    assert g.has_variant(2) and g.has_variant(3)
    assert g.replay(1, 0) == FRT_ERR_NO_VARIANT, "evicted key must report NO_VARIANT"
    assert g.replay(2, 0) == 0
    print("PASS  5. ShapeKey variant LRU eviction + NO_VARIANT")


def main():
    assert torch.cuda.is_available(), "CUDA device required"
    test_capture_replay()
    test_bind_handoff()
    test_multistream_event()
    test_buffer_copy()
    test_lru_eviction()
    print("\nALL PHASE-A CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
