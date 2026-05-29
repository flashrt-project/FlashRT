"""serving/robot_pi07 — hierarchical two-VLA host (pi0.7-style, simplified).

pi0.7 runtime is a multi-model hierarchy: a High-Level Policy emits a subtask, a
World Model (BAGEL) emits subgoal images, and the action VLA consumes them.
Simplified (BAGEL/world-model dropped) it is a two-stage hierarchy:

    PLANNER (low rate) --subtask (shared Buffer)--> ACTOR (high rate) --> actions

This host co-hosts TWO Pi05 instances through ONE exec ctx and verifies the
multi-model hot-path mechanism:
  - two adopted graphs driven from one host;
  - PLANNER -> ACTOR hand-off through a SHARED buffer (frt_buffer_copy);
  - multi-rate: PLANNER runs once every N ACTOR ticks;
  - interrupt / verbal coaching: overwrite the subtask buffer mid-run -> the
    next ACTOR tick consumes the new subtask, NO recapture.

Mechanism demo (honest scope): two Pi05 stand in for planner+actor; the subtask
hand-off is plumbing (planner output -> subtask buffer -> actor input), not a
semantic planner->language mapping. We verify the contract orchestration.

Run (inside pi0-stablehlo-test):
  PYTHONPATH=/workspace/PI/official/FlashRT-spec:/workspace/PI/official/FlashRT-spec/exec/build \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python serving/robot_pi07/verify_pi07.py --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch
"""

import argparse
import numpy as np
import torch
import _flashrt_exec as ex

import flash_rt


def _load(ckpt, num_views):
    return flash_rt.load_model(
        ckpt, framework="torch", config="pi05", hardware="auto",
        num_views=num_views, num_steps=10, cache_frames=1,
        use_fp8=True, use_fp16=False)


def _bytes(cuda_buffer):
    return cuda_buffer.download_new((cuda_buffer.nbytes,), np.uint8).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--planner-every", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(args.num_views)]

    planner = _load(args.checkpoint, args.num_views)   # two independent Pi05 instances
    actor = _load(args.checkpoint, args.num_views)
    planner.predict(images, prompt="clean the kitchen")     # capture planner graph
    actor.predict(images, prompt="pick up the knife")       # capture actor graph
    plP, plA = planner._pipe.pipeline, actor._pipe.pipeline
    n = plA.bufs["diffusion_noise"].nbytes
    assert plP.bufs["diffusion_noise"].nbytes == n

    # ONE exec ctx co-hosts BOTH VLAs.
    ctx = ex.Ctx()
    sP = ctx.wrap_stream(int(plP._graph_stream.value))
    sA = ctx.wrap_stream(int(plA._graph_stream.value))
    gP = ctx.graph("planner", 1); gP.adopt(0, plP._graph._graph_exec.value)
    gA = ctx.graph("actor", 1);   gA.adopt(0, plA._graph._graph_exec.value)

    # buffers wired into the contract: planner out, actor in, shared subtask.
    b_planner_out = ctx.wrap("planner_out", plP.bufs["diffusion_noise"].ptr.value, n)
    b_actor_in = ctx.wrap("actor_in", plA.bufs["diffusion_noise"].ptr.value, n)
    subtask_t = torch.zeros(n, dtype=torch.uint8, device="cuda")   # torch-backed => readable
    b_subtask = ctx.wrap("subtask", subtask_t.data_ptr(), n)
    new_goal = torch.full((n,), 7, dtype=torch.uint8, device="cuda")
    b_new_goal = ctx.wrap("new_goal", new_goal.data_ptr(), n)

    planner_runs, interrupt_tick = 0, 5
    print(f"co-hosted via ONE exec ctx: planner(stream {sP}) + actor(stream {sA})\n")
    for t in range(args.ticks):
        if t == interrupt_tick:                                # interrupt / verbal coaching
            ctx.copy(b_subtask, 0, b_new_goal, 0, n, sA)
            torch.cuda.synchronize()
            assert np.array_equal(subtask_t.cpu().numpy(), new_goal.cpu().numpy())
            print(f"  tick {t}: INTERRUPT — subtask overwritten (no recapture); subtask==new_goal OK")

        ran_planner = (t % args.planner_every == 0) and (t != interrupt_tick)
        if ran_planner:                                        # low-rate PLANNER -> subtask
            assert gP.replay(0, sP) == 0
            plP._cudart.cudaStreamSynchronize(plP._graph_stream)
            ctx.copy(b_subtask, 0, b_planner_out, 0, n, sP)    # planner -> subtask (hand-off)
            torch.cuda.synchronize()
            assert np.array_equal(subtask_t.cpu().numpy(), _bytes(plP.bufs["diffusion_noise"])), \
                "planner->subtask hand-off mismatch"
            planner_runs += 1

        ctx.copy(b_actor_in, 0, b_subtask, 0, n, sA)           # subtask -> actor input
        assert gA.replay(0, sA) == 0                           # high-rate ACTOR
        plA._cudart.cudaStreamSynchronize(plA._graph_stream)
        actions = torch.frombuffer(
            _bytes(plA.bufs["diffusion_noise"]).tobytes(), dtype=torch.bfloat16).float()
        assert np.isfinite(actions.numpy()).all()
        print(f"  tick {t}: actor acted (planner_run={ran_planner})")

    print(f"\nPASS — pi0.7-sim hierarchy: {args.ticks} actor ticks, {planner_runs} planner runs "
          f"(1:{args.planner_every} multi-rate), planner->subtask->actor hand-off via shared "
          "buffer (verified), mid-run subtask interrupt (no recapture), two VLAs co-hosted via "
          "ONE exec ctx.")


if __name__ == "__main__":
    main()
