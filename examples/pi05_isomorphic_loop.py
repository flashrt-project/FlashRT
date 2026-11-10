"""Rollout, replay, update, rollout: Pi0.5 on one arithmetic.

The isomorphic frontend serves rollouts and carries the trainer that updates the same weights. This script walks one
turn of an on-policy loop with a stand-in reward:

  1. roll W observations out with the stochastic sampler and keep the trace and the prefix records;
  2. replay: the trainer's forward on shuffled minibatches of the recorded transitions returns the rollout's
     velocities, so every transition's probability ratio is exactly 1 before the first update;
  3. a few PPO-style updates of the action expert (or, with --adapter, of a LoRA adapter) over the kept prefix (the
     vision tower and the language model are not run again);
  4. roll out again: the new weights are already the ones the rollout reads.

    PYTHONPATH=<FlashRT>:<FlashRT-Train> FRT_TRAIN_LIB=<libflashrt_train.so> \
        python examples/pi05_isomorphic_loop.py --checkpoint-dir <pi05 checkpoint> --width 8 --minibatch 4

The advantage here is random: the script shows the mechanics and the gate, not a learning curve.
"""

import argparse

import numpy as np
import torch

from flash_rt.frontends.torch.pi05_isomorphic import Pi05TorchFrontendIsomorphic


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--width", type=int, default=8, help="observations per rollout")
    ap.add_argument("--minibatch", type=int, default=4, help="transitions per update")
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-prompt-len", type=int, default=48)
    ap.add_argument("--adapter", default=None, help="a PEFT LoRA adapter directory: train it instead of the action expert")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def log_prob(v, x, x_next, sigma, dt):
    """log N(x_next; x + dt v, sigma^2) of a transition, up to its constant."""
    return -0.5 * ((x_next - (x + dt * v)) / sigma).pow(2).sum(dim=(1, 2))


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    W = args.width
    form = {"trainable": "lora", "adapter": args.adapter} if args.adapter else {"trainable": "expert"}
    fe = Pi05TorchFrontendIsomorphic(args.checkpoint_dir, max_prompt_len=args.max_prompt_len, denoise_trace=True, sde=True, **form)
    trainer = fe.trainer
    steps = len(fe.denoise_timesteps())
    ts = np.asarray(fe.denoise_timesteps(), dtype=np.float32)
    dt = float(np.float32(-1.0) / np.float32(steps))
    sigma = [args.sigma] * (steps - 1) + [0.0]          # the last step is deterministic
    observations = [{"images": [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(3)]} for _ in range(W)]
    fe.set_prompt_batch(["pick up the black bowl and place it on the plate"] * W)

    # 1. rollout
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    first = fe.infer_batch(observations, generator=gen, sde_sigma=sigma)
    roll = fe.last_rollout
    trace_x, trace_v, actions = roll["trace_x"].clone(), roll["trace_v"].clone(), roll["actions"].clone()
    records = fe.export_prefix()                         # the replay buffer's share of each observation
    advantage = torch.randn(W, device="cuda")            # stand-in for the environment's return

    # the stochastic transitions (sample, step), shuffled into minibatches
    pairs = [(b, s) for b in range(W) for s in range(steps - 1)]
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed)).tolist()
    batches = [[pairs[j] for j in order[lo:lo + args.minibatch]] for lo in range(0, len(order) - args.minibatch + 1, args.minibatch)]

    def gather(chunk):
        idx = torch.tensor([b for b, _ in chunk], device="cuda")
        x = torch.stack([trace_x[s, b] for b, s in chunk]).contiguous()
        v_old = torch.stack([trace_v[s, b] for b, s in chunk]).contiguous()
        x_next = torch.stack([trace_x[s + 1, b] for b, s in chunk]).contiguous()
        t = np.asarray([ts[s] for _, s in chunk], dtype=np.float32)
        return idx, x, v_old, x_next, t

    # 2. replay: the trainer's forward against the kept prefix, before any update
    differing = total = 0
    for chunk in batches:
        idx, x, v_old, _, t = gather(chunk)
        kept = records[idx].contiguous()
        trainer.policy_import(kept)
        v = torch.empty_like(x)
        trainer.policy_forward_kept(x, t, v)
        trainer.synchronize()
        differing += int((v != v_old).sum())
        total += v.numel()
    print(f"replay: {differing} of {total} velocity elements differ from the rollout (every ratio is {'exactly 1' if differing == 0 else 'not 1'})")

    # 3. updates on the kept prefix
    for chunk in batches[: args.updates]:
        idx, x, v_old, x_next, t = gather(chunk)
        kept = records[idx].contiguous()
        trainer.policy_import(kept)
        v = torch.empty_like(x)
        trainer.policy_forward_kept(x, t, v)
        leaf = v.clone().requires_grad_(True)             # the surrogate is the caller's: autograd from the velocity up
        ratio = torch.exp(log_prob(leaf, x, x_next, args.sigma, dt) - log_prob(v_old, x, x_next, args.sigma, dt))
        adv = advantage[idx]
        loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
        loss.backward()
        cotangent = leaf.grad.contiguous()
        trainer.policy_backward(cotangent)                # the engine's backward from d loss / d v
        trainer.optimizer_step(args.lr)
        print(f"update: surrogate {float(loss):+.4f}, ratio in [{float(ratio.min()):.4f}, {float(ratio.max()):.4f}]")

    # 4. the next rollout reads the updated weights; nothing was copied
    second = fe.infer_batch(observations, noise=roll["noise"], step_noise=roll["step_noise"], sde_sigma=sigma)
    moved = float(np.abs(np.stack([r["raw_actions"] for r in second]) - actions.cpu().numpy()).max())
    print(f"rollout after {min(args.updates, len(batches))} updates: actions moved by up to {moved:.3e} on the same noise")
    print(f"rollout latency: {fe.latency_records[-1]:.1f} ms at width {W}; first actions {first[0]['actions'][0, :3]}")
    return 0 if differing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
