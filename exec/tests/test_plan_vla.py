"""Phase-C — frt_Plan over a VLA-shaped multi-subgraph chain.

Validates the last unexercised contract primitive (the Plan DAG executor) on
the exact shape a VLA pipeline has — vision -> encoder -> action -> ... —
where each stage hands off to the next through a SHARED bound buffer (zero
copy), and stages run across streams with explicit cross-stream dependencies.

Stand-in "kernels" are allocation-free memset/memcpy (mechanism test, no
model needed). The chain:
    vision : write  ENC_X = 7              (stream 0)
    encoder: copy   ENC_X -> KV            (stream 1, after vision)
    action : copy   KV    -> OUT           (stream 0, after encoder)
If the Plan's shared-buffer hand-off and cross-stream event ordering are
correct, OUT ends up == 7.

Run (inside the container, after building exec/):
    PYTHONPATH=.../exec/build python exec/tests/test_plan_vla.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import torch
import _flashrt_exec as ex


def _buf(ctx, name, nbytes):
    t = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    return t, ctx.wrap(name, t.data_ptr(), nbytes)


def test_plan_vla_chain():
    ctx = ex.Ctx()
    s_vis = 0                 # default stream
    s_enc = ctx.stream(0)     # a second stream (id 1)

    encx_t, encx = _buf(ctx, "enc_x", 256)   # vision -> encoder hand-off buffer
    kv_t, kv = _buf(ctx, "kv", 256)          # encoder -> action hand-off buffer
    out_t, out = _buf(ctx, "out", 256)
    p_encx, p_kv, p_out = encx.dptr(), kv.dptr(), out.dptr()

    g_vis = ctx.graph("vision")
    g_vis.capture(0, lambda s: ex.memset_async(p_encx, 7, 256, s))
    g_vis.bind("enc_x", encx)

    g_enc = ctx.graph("encoder")
    g_enc.capture(0, lambda s: ex.memcpy_async(p_kv, p_encx, 256, s))
    g_enc.bind("in", encx)    # SAME buffer vision wrote — zero-copy hand-off
    g_enc.bind("kv", kv)

    g_act = ctx.graph("action")
    g_act.capture(0, lambda s: ex.memcpy_async(p_out, p_kv, 256, s))
    g_act.bind("kv", kv)
    g_act.bind("out", out)

    out_t.zero_(); kv_t.zero_(); encx_t.zero_()
    torch.cuda.synchronize()

    plan = ctx.plan()
    n_vis = plan.add(g_vis, 0, s_vis)
    n_enc = plan.add(g_enc, 0, s_enc)
    n_act = plan.add(g_act, 0, s_vis)
    plan.after(n_enc, n_vis)              # encoder (stream1) waits vision (stream0)
    plan.after(n_act, n_enc)              # action (stream0) waits encoder (stream1)
    plan.execute(0)
    plan.sync()

    assert torch.all(out_t == 7), (
        f"VLA-shaped Plan hand-off failed: out unique={torch.unique(out_t).tolist()}")
    assert torch.all(kv_t == 7), "encoder stage did not receive vision output"
    print("PASS  frt_Plan VLA chain: vision -> encoder -> action across streams, "
          "zero-copy hand-off correct (OUT==7)")


def main():
    assert torch.cuda.is_available()
    test_plan_vla_chain()
    print("\nPHASE-C PLAN DEMO PASSED")


if __name__ == "__main__":
    main()
