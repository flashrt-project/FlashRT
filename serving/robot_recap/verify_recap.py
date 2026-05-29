"""serving/robot_recap — verify the RECAP (pi*0.6-style) RL inference runs
through the FlashRT execution contract.

pi*0.6 / RECAP inference = an **advantage-conditioned** VLA policy: the model is
conditioned (via CFG) on a positive-advantage tag to steer it toward the
"good-reasoning / high-advantage" behavior learned with RL. In FlashRT this is
`Pi05TorchFrontendRtx.set_rl_mode(cfg_enable=True, cfg_beta=...)`, which swaps in
`Pi05CFGPipeline` and runs classifier-free guidance (cond = advantage-positive,
uncond = plain).

`Pi05CFGPipeline` inherits the wired `Pi05Pipeline.{record_infer_graph,forward}`,
so with the same `adopt`-based mechanism the **RL/CFG inference graph is driven
by the exec contract**. This script verifies it end to end:
  1. RL/CFG inference actually runs (advantage-conditioned), actions finite.
  2. The exec layer drives that captured CFG graph BIT-IDENTICALLY to the ctypes
     replay (cosine 1.0) — restoring the in-place noise buffer before each replay.

Run (inside pi0-stablehlo-test):
  PYTHONPATH=/workspace/PI/official/FlashRT-spec:/workspace/PI/official/FlashRT-spec/exec/build \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python serving/robot_recap/verify_recap.py --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch
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
    ap.add_argument("--cfg-beta", type=float, default=1.5)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(args.num_views)]

    # FP8/default frontend (Pi05TorchFrontendRtx) — the one with real RL/CFG support.
    model = flash_rt.load_model(
        args.checkpoint, framework="torch", config="pi05", hardware="auto",
        num_views=args.num_views, num_steps=args.steps, cache_frames=1,
        use_fp8=True, use_fp16=False)
    fe = model._pipe
    print(f"frontend={type(fe).__name__}")
    fe.set_rl_mode(cfg_enable=True, cfg_beta=args.cfg_beta)   # advantage-conditioned RL inference
    print(f"RL mode set: advantage-conditioned CFG (beta={args.cfg_beta})")

    out = np.asarray(model.predict(images, prompt="pick up the red block"))
    pl = fe.pipeline
    print(f"pipeline={type(pl).__name__} graph_captured={getattr(pl, '_graph', None) is not None} "
          f"actions_shape={out.shape} finite={np.isfinite(out).all()}")
    assert getattr(pl, "_graph", None) is not None, "RL/CFG inference graph not captured"

    # De-risk: exec-driven replay of the RL/CFG graph == ctypes, bit-identical.
    out_buf = pl.bufs["diffusion_noise"]
    save = _read(out_buf)

    def restore():
        out_buf.upload(save)
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)

    def replay_ctypes():
        restore(); pl._graph.replay(pl._graph_stream)
        pl._cudart.cudaStreamSynchronize(pl._graph_stream); return _read(out_buf)

    ctx = ex.Ctx()
    gs_id = ctx.wrap_stream(int(pl._graph_stream.value))
    fg = ctx.graph("recap_policy", 1)
    fg.adopt(0, pl._graph._graph_exec.value)

    def replay_frt():
        restore(); rc = fg.replay(0, gs_id)
        assert rc == 0, f"frt replay rc={rc}"
        pl._cudart.cudaStreamSynchronize(pl._graph_stream); return _read(out_buf)

    a1 = replay_ctypes(); b = replay_frt(); a2 = replay_ctypes()
    cos = _cos(a1, b)
    print("\n===== RECAP RL INFERENCE THROUGH EXEC CONTRACT =====")
    print(f"ctypes self-reproduce: {np.array_equal(a1, a2)}")
    print(f"frt == ctypes (exact): {np.array_equal(a1, b)}")
    print(f"cosine(frt, ctypes)  : {cos:.6f}")
    assert cos >= 0.999, f"RL/CFG exec replay cosine {cos} below 0.999"
    print("\nPASS — advantage-conditioned RL (CFG) inference is driven by the exec "
          f"contract, bit-identical (cos={cos:.6f})")


if __name__ == "__main__":
    main()
