"""serving/robot_recap — the robot side of "one capsule, two scenarios".

The LLM agent capsule (snapshot/restore a committed execution boundary) has a
model-specific frontend API because its boundary is a large hybrid state (KV +
recurrent + conv + MTP; see serving/qwen36_agent/capsules.md). The robot rollout
boundary is just a small set of buffers (the diffusion seed; in production also
the observation), so here the **same capsule mechanism is expressed directly
through the execution contract**: a capsule is a `frt` Buffer, snapshot/restore is
`frt` device-to-device copy. No new mechanism, no model-specific API.

This verifies the robot-side capsule end to end:
  1. snapshot the episode boundary into a contract Buffer (capsule);
  2. restore it and replay the captured policy graph -> bit-identical actions
     (cosine 1.0), even after the live boundary buffer was overwritten.

That is exactly what `rollout_host.py`'s per-episode reset needs: restore to the
episode-initial boundary with no recapture. RECAP rollout reset == capsule
restore, the same verb the coding agent uses.

Run (inside the CUDA container):
  PYTHONPATH=.:./exec/build \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python serving/robot_recap/verify_capsule.py --checkpoint checkpoints/pi05_libero_pytorch
"""

import argparse
import numpy as np
import torch
import _flashrt_exec as ex

import flash_rt


def _read(buf):
    return buf.download_new((buf.nbytes,), np.uint8).copy()


def _cos(a_u8, b_u8):
    a = torch.frombuffer(a_u8.tobytes(), dtype=torch.bfloat16).float()
    b = torch.frombuffer(b_u8.tobytes(), dtype=torch.bfloat16).float()
    na, nb = a.norm(), b.norm()
    return float(torch.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
              for _ in range(args.num_views)]

    model = flash_rt.load_model(
        args.checkpoint, framework="torch", config="pi05", hardware="auto",
        num_views=args.num_views, num_steps=args.steps, cache_frames=1,
        use_fp8=True, use_fp16=False)
    fe = model._pipe
    fe.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    model.predict(images, prompt="pick up the red block")   # capture policy graph
    pl = fe.pipeline
    assert getattr(pl, "_graph", None) is not None, "policy graph not captured"

    out_buf = pl.bufs["diffusion_noise"]          # the rollout boundary buffer
    n = out_buf.nbytes

    # The capsule mechanism, expressed through the execution contract.
    ctx = ex.Ctx()
    gs = ctx.wrap_stream(int(pl._graph_stream.value))
    g = ctx.graph("recap_policy", 1)
    g.adopt(0, pl._graph._graph_exec.value)
    b_live = ctx.wrap("rollout_boundary", out_buf.ptr.value, n)
    b_capsule = ctx.buffer("rollout_capsule", n)

    def sync():
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)

    # SNAPSHOT: copy the live episode boundary into the capsule (contract D2D).
    ctx.copy(b_capsule, 0, b_live, 0, n, gs)
    sync()
    capsule_bytes = n

    def restore_and_act():
        # RESTORE: capsule -> live boundary, then replay the policy graph.
        ctx.copy(b_live, 0, b_capsule, 0, n, gs)
        rc = g.replay(0, gs)
        assert rc == 0, f"frt replay rc={rc}"
        sync()
        return _read(out_buf)

    a1 = restore_and_act()

    # Dirty the live boundary with an unrelated state (a different episode).
    out_buf.upload(np.frombuffer(
        rng.randint(0, 256, n, dtype=np.uint8).tobytes(), dtype=np.uint8).copy())
    sync()

    a2 = restore_and_act()      # restore from capsule -> must reproduce a1
    a3 = restore_and_act()      # idempotent reuse

    cos12 = _cos(a1, a2)
    cos13 = _cos(a1, a3)
    print("\n===== ROBOT CAPSULE THROUGH THE EXEC CONTRACT =====")
    print(f"capsule bytes        : {capsule_bytes}")
    print(f"restore reproduces   : exact={np.array_equal(a1, a2)} cos={cos12:.6f}")
    print(f"capsule reuse stable : exact={np.array_equal(a1, a3)} cos={cos13:.6f}")
    assert np.array_equal(a1, a2) and np.array_equal(a1, a3), \
        "capsule restore is not bit-identical"
    print("\nPASS — robot rollout boundary snapshot/restore via the exec contract "
          "is bit-identical (episode reset == capsule restore, same mechanism as "
          "the LLM agent capsule).")


if __name__ == "__main__":
    main()
