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

Run (inside pi0-stablehlo-test):
  PYTHONPATH=/workspace/PI/official/FlashRT-spec:/workspace/PI/official/FlashRT-spec/exec/build \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python serving/robot_recap/rollout_host.py --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch
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

    # ── episode state machine ──
    print(f"frontend={type(fe).__name__} pipeline={type(pl).__name__}")
    print(f"co-hosted via ONE exec ctx: policy(stream {s_policy}) + value critic(stream {s_critic})\n")
    total_chunks = 0
    for ep in range(args.episodes):
        reset_state()                                  # RESET (buffers, no recapture)
        chunks, stop_reason, value = 0, None, 0.0
        # simulate a keyboard "stop" pressed at a per-episode chunk (None = never)
        keyboard_stop_at = {0: 3, 1: None}.get(ep, None)
        for c in range(args.max_chunks):               # RUNNING
            actions, value = run_chunk(c)
            chunks += 1; total_chunks += 1
            assert np.isfinite(actions.numpy()).all()
            if value < args.value_stop_threshold:
                stop_reason = "auto(value<thr)"; break  # critic-driven termination
            if keyboard_stop_at is not None and c + 1 >= keyboard_stop_at:
                stop_reason = "keyboard"; break          # human stop -> STOP_INFER
        else:
            stop_reason = "timeout(max_chunks)"
        # STOP_INFER reached: NO further replay happens this episode (verified by `chunks`).
        # AWAIT_RESET (human resets robot) -> RECORD -> next episode.
        print(f"episode {ep}: ran {chunks} chunks, STOP={stop_reason}, "
              f"last_value={value:+.3f}, recorded.")

    print(f"\nPASS — RL rollout host: {args.episodes} episodes, {total_chunks} chunks total, "
          "policy+critic co-hosted via ONE exec ctx, per-chunk interruptible (clean STOP at "
          "episode boundary), buffer reset between episodes (no recapture).")


if __name__ == "__main__":
    main()
