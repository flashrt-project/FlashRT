"""serving/robot_recap — RL Rollout Host (pi*0.6 / RECAP-style).

Systematic answer to the real community rollout pain ("inference keeps running
between episodes; I can't stop to reset the robot / record with a keyboard").
The fix is NOT a smarter policy — it is a host-driven EPISODE STATE MACHINE on
top of the exec contract's interruptible, per-chunk replay:

  RESET -> RUNNING --(value low / keyboard / timeout)--> STOP_INFER
    ^                                                        |
    +------ RESET(buffers) <- RECORD <- AWAIT_RESET <--------+

What the exec contract provides (mechanism) and this host uses (policy):
  - per-CHUNK replay: the host fires one action-chunk replay at a time and
    decides between chunks whether to continue or STOP — so inference halts
    cleanly at an episode boundary (interrupt granularity = one short replay).
  - multi-model concurrency: the advantage-conditioned policy (Pi05 CFG) and a
    value-function critic run on separate streams via ONE exec ctx; the critic
    drives AUTO episode termination (less manual keyboarding).
  - buffer reset: episode reset = reinit state buffers, NO recapture.

This verifies the hot-path MECHANISM (it reuses the captured policy chunk with a
restored noise buffer; production writes fresh observations each chunk). The
episode state machine / keyboard / reset policy live HERE in serving, never in
the contract.

Run (inside the CUDA container):
  PYTHONPATH=.:./exec/build \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python serving/robot_recap/rollout_host.py --checkpoint checkpoints/pi05_libero_pytorch
"""

import argparse
import numpy as np
import torch
import _flashrt_exec as ex

import flash_rt
from flash_rt.core.rl.value_function import StandaloneValueFunction

ACTION_DIM = 7
STATE_DIM = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--value-stop-threshold", type=float, default=0.0)
    ap.add_argument("--record-dir", default="", help="dir to write episode_*.npz (empty = none)")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    images = [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(args.num_views)]

    # ── policy: advantage-conditioned RL (CFG) Pi05 ──
    model = flash_rt.load_model(
        args.checkpoint, framework="torch", config="pi05", hardware="auto",
        num_views=args.num_views, num_steps=10, cache_frames=1,
        use_fp8=True, use_fp16=False)
    fe = model._pipe
    fe.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    model.predict(images, prompt="pick up the red block")   # capture policy graph
    pl = fe.pipeline
    assert getattr(pl, "_graph", None) is not None
    out_buf = pl.bufs["diffusion_noise"]
    noise0 = out_buf.download_new((out_buf.nbytes,), np.uint8).copy()

    # ── critic: a real (lightweight) value function, captured as a CUDA graph ──
    critic = StandaloneValueFunction(state_dim=STATE_DIM, use_images=False).cuda().eval().half()
    state_buf = torch.zeros(1, STATE_DIM, device="cuda", dtype=torch.float16)
    val_out = torch.zeros(1, device="cuda", dtype=torch.float16)
    critic_stream = torch.cuda.Stream()
    with torch.cuda.stream(critic_stream):
        for _ in range(3):
            val_out.copy_(critic.predict_value(state_buf).view(1).half())
    torch.cuda.current_stream().wait_stream(critic_stream)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg, stream=critic_stream):
        val_out.copy_(critic.predict_value(state_buf).view(1).half())

    # ── ONE exec ctx co-hosts BOTH models ──
    ctx = ex.Ctx()
    s_policy = ctx.wrap_stream(int(pl._graph_stream.value))
    s_critic = ctx.wrap_stream(int(critic_stream.cuda_stream))
    g_policy = ctx.graph("recap_policy", 1)
    g_policy.adopt(0, pl._graph._graph_exec.value)
    g_critic = ctx.graph("recap_value", 1)
    g_critic.adopt(0, cg.raw_cuda_graph_exec())

    def reset_state():
        # episode reset = reinit state buffers, NO recapture.
        out_buf.upload(noise0)
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)

    def run_chunk(c):
        # policy (stream P) + critic (stream C) concurrently via the contract.
        out_buf.upload(noise0)  # fresh "noise" for the chunk (mechanism: reuse captured graph)
        state_buf.fill_(float(c) * 0.05)   # vary critic input per chunk (progress proxy)
        torch.cuda.synchronize()
        assert g_policy.replay(0, s_policy) == 0
        assert g_critic.replay(0, s_critic) == 0
        pl._cudart.cudaStreamSynchronize(pl._graph_stream)
        torch.cuda.synchronize()
        actions = torch.frombuffer(
            out_buf.download_new((out_buf.nbytes,), np.uint8).tobytes(),
            dtype=torch.bfloat16).float()          # raw action-chunk buffer (flat)
        value = float(val_out.float().item())
        return actions, value

    # ── pluggable host hooks (the parts the community user asks about) ──
    # Keyboard START/END: a swappable event source. Default is SCRIPTED so the
    # demo runs headless; swap in a real listener (pynput/termios) with the same
    # interface for teleop. Returns "END" to stop the current episode.
    def keyboard_event(ep, chunk):
        end_at = {0: 3}.get(ep)            # episode 0 ends on a "keyboard" press at chunk 3
        return "END" if (end_at is not None and chunk + 1 >= end_at) else None

    # Robot reset-to-initial: a host hook (hardware-specific). No-op here (no
    # robot); in teleop you call your driver, e.g. robot.move_to_home().
    def robot_reset_to_initial():
        pass                               # robot.move_to_home(blocking=True)

    import os
    rec_dir = args.record_dir
    if rec_dir:
        os.makedirs(rec_dir, exist_ok=True)

    # ── episode state machine: START -> RUNNING -> STOP_INFER -> reset -> RECORD ──
    print(f"frontend={type(fe).__name__} pipeline={type(pl).__name__}")
    print(f"co-hosted via ONE exec ctx: policy(stream {s_policy}) + value critic(stream {s_critic})\n")
    total_chunks = 0
    for ep in range(args.episodes):
        reset_state()                                  # RESET model state (buffers, no recapture)
        traj, chunks, stop_reason, value = [], 0, None, 0.0
        for c in range(args.max_chunks):               # RUNNING (one action chunk per replay)
            actions, value = run_chunk(c)
            chunks += 1; total_chunks += 1
            assert np.isfinite(actions.numpy()).all()
            traj.append({"chunk": c, "action": actions.numpy(), "value": value})  # RECORD buffer
            if keyboard_event(ep, c) == "END":
                stop_reason = "keyboard(END)"; break     # human end -> clean STOP between chunks
            if value < args.value_stop_threshold:
                stop_reason = "auto(value<thr)"; break   # critic-driven termination
        else:
            stop_reason = "timeout(max_chunks)"
        # STOP_INFER: no further replay this episode. AWAIT_RESET -> reset robot ->
        # serialize the recorded episode -> next episode.
        robot_reset_to_initial()                       # reset robot to initial position (hook)
        saved = ""
        if rec_dir:
            path = os.path.join(rec_dir, f"episode_{ep:03d}.npz")
            np.savez(path, actions=np.stack([t["action"] for t in traj]),
                     values=np.array([t["value"] for t in traj], dtype=np.float32),
                     stop_reason=stop_reason)
            saved = f" -> {path}"
        print(f"episode {ep}: ran {chunks} chunks, STOP={stop_reason}, "
              f"last_value={value:+.3f}, recorded {len(traj)} chunks{saved}")

    print(f"\nPASS — RL rollout host: {args.episodes} episodes, {total_chunks} chunks total, "
          "policy+critic co-hosted via ONE exec ctx, per-chunk interruptible (clean STOP at "
          "episode boundary via keyboard/value/timeout), robot-reset + episode recording hooks, "
          "model-state buffer reset between episodes (no recapture).")


if __name__ == "__main__":
    main()
