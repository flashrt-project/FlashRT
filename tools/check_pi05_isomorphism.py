"""Are Pi0.5 rollouts the trainer's arithmetic? A gate for the isomorphic frontend, a reading for the native one.

An on-policy update recomputes the velocity of every recorded denoising transition and weighs the transition by
exp(log p_trainer - log p_rollout). Before any update that weight is 1 by definition; it is 1 in practice only if the
trainer's forward returns the rollout's velocity. This tool rolls a checkpoint out through a FlashRT frontend with the
stochastic sampler and a recorded trace, hands every recorded (x, t) to FlashRT-Train's training forward in shuffled
minibatches of another size, and counts the elements that differ.

    isomorphic   Pi05TorchFrontendIsomorphic: the gate. No element may differ, at any width, and a sample's rollout
                 may not depend on the width it ran at. The exit status is the gate.
    native       Pi05TorchFrontendRtx (--native): a reading. The native pipeline owns its arithmetic (BF16 state,
                 fused projections, a pre-scaled output projection); the tool reports how far the trainer's forward
                 lands from it and what that does to a transition's weight, next to the two frontends' latency.

    PYTHONPATH=<FlashRT>:<FlashRT-Train> FRT_TRAIN_LIB=<libflashrt_train.so> \
        python tools/check_pi05_isomorphism.py --checkpoint-dir <pi05 checkpoint> --widths 1,4,8 [--native] [--json out.jsonl]
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time

import numpy as np
import torch

ACTION_DIM = 32
PROMPTS = (
    "pick up the black bowl and place it on the plate",
    "open the top drawer of the cabinet",
    "put the red mug on the left shelf",
    "turn on the stove",
    "push the plate to the front of the stove and put the cream cheese in the bowl",
    "close the microwave",
    "stack the blue block on the green block",
    "pick up the book and place it in the back compartment of the caddy",
)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--widths", default="1,4,8", help="rollout widths")
    ap.add_argument("--minibatch", type=int, default=3, help="transitions per recomputation (deliberately not a width)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--max-prompt-len", type=int, default=48, help="text columns of the isomorphic forward")
    ap.add_argument("--sigma", type=float, default=0.1, help="transition noise of every step but the last")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeats", type=int, default=9, help="timed calls per arm (the median is read)")
    ap.add_argument("--arena-gib", type=float, default=16.0)
    ap.add_argument("--native", action="store_true", help="also read the native BF16 pipeline against the trainer")
    ap.add_argument("--same-prompt", action="store_true",
                    help="one prompt for every sample (the native batched pipeline pads shorter prompts with unmasked zero "
                         "embeddings, so mixed lengths add its own approximation to the reading)")
    ap.add_argument("--json", default=None)
    return ap.parse_args()


def observations(width, seed):
    g = np.random.default_rng(seed)
    return [{"images": [g.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(3)]} for _ in range(width)]


def noises(width, steps, chunk, seed):
    """Initial and per-step noise, bf16-representable so both frontends take the same values."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(width, chunk, ACTION_DIM, generator=g).to(torch.bfloat16).float()
    eps = torch.randn(steps, width, chunk, ACTION_DIM, generator=g).to(torch.bfloat16).float()
    return x0, eps


def log_ratio(mean_new, mean_old, x_next, sigma):
    """log N(x_next; mean_new, sigma^2) - log N(x_next; mean_old, sigma^2) per transition, in float64."""
    a = ((x_next.double() - mean_new.double()) / sigma).pow(2).sum(dim=(-1, -2))
    b = ((x_next.double() - mean_old.double()) / sigma).pow(2).sum(dim=(-1, -2))
    return -0.5 * (a - b)


def replay(trainer, roll, timesteps, minibatch, seed):
    """Recompute every recorded velocity through the trainer's forward in shuffled minibatches; return the velocities
    laid out as the trace."""
    steps, width = roll["trace_x"].shape[:2]
    pairs = [(s, b) for s in range(steps) for b in range(width)]
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    out = torch.empty_like(roll["trace_x"])
    mask = roll["mask"]
    for lo in range(0, len(order), minibatch):
        chunk = [pairs[j] for j in order[lo:lo + minibatch]]
        idx = torch.tensor([b for _, b in chunk], device=out.device)
        x = torch.stack([roll["trace_x"][s, b] for s, b in chunk]).contiguous()
        t = np.asarray([timesteps[s] for s, _ in chunk], dtype=np.float32)
        images = [img[idx].contiguous() for img in roll["images"]]
        tokens = roll["tokens"][idx].contiguous()
        v = torch.empty_like(x)
        trainer.policy_forward(images, tokens, mask[idx.cpu().numpy()], x, t, v)
        trainer.synchronize()
        for k, (s, b) in enumerate(chunk):
            out[s, b] = v[k]
    return out


def digest(t):
    """A short digest of a tensor's bytes: the same value in two processes is the same rollout."""
    return hashlib.md5(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:12]


def median_ms(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main():
    args = parse()
    widths = [int(w) for w in args.widths.split(",")]
    wmax = max(widths)
    obs = observations(wmax, args.seed)
    prompts = [PROMPTS[0 if args.same_prompt else b % len(PROMPTS)] for b in range(wmax)]
    x0, eps = noises(wmax, args.steps, args.chunk, args.seed)
    sigma = [args.sigma] * (args.steps - 1) + [0.0]
    rows, ok = [], True

    from flash_rt.frontends.torch.pi05_isomorphic import Pi05TorchFrontendIsomorphic
    iso = Pi05TorchFrontendIsomorphic(args.checkpoint_dir, chunk_size=args.chunk, max_prompt_len=args.max_prompt_len,
                                      num_steps=args.steps, denoise_trace=True, sde=True, arena_gib=args.arena_gib)
    ts = np.asarray(iso.denoise_timesteps(), dtype=np.float32)
    dt = float(np.float32(-1.0) / np.float32(args.steps))

    def iso_roll(width):
        iso.set_prompt_batch(prompts[:width])
        return iso.infer_batch(obs[:width], noise=x0[:width], step_noise=eps[:, :width], sde_sigma=sigma)

    print(f"== isomorphic frontend: rollouts recomputed by the training forward in minibatches of {args.minibatch}", flush=True)
    alone = None
    for width in widths:
        iso_roll(width)
        roll = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in iso.last_rollout.items()}
        roll["images"] = [img.clone() for img in iso.last_rollout["images"]]
        v = replay(iso.trainer, roll, ts, args.minibatch, args.seed + width)
        differing = int((v != roll["trace_v"]).sum())
        total = v.numel()
        # the same samples at another width: the first min(width, wmin) samples against the narrowest rollout
        if alone is None:
            alone = {"actions": roll["actions"].clone(), "trace_v": roll["trace_v"].clone(), "width": width}
            across = 0
        else:
            n = alone["width"]
            across = int((roll["actions"][:n] != alone["actions"]).sum()) + int((roll["trace_v"][:, :n] != alone["trace_v"]).sum())
        ok = ok and differing == 0 and across == 0
        print(f"  width {width}: {differing} of {total} velocity elements differ from the rollout; "
              f"{across} elements differ from the same samples at width {alone['width']}; actions digest {digest(roll['actions'])}", flush=True)
        rows.append({"arm": "isomorphic", "width": width, "minibatch": args.minibatch, "differing": differing, "elements": total,
                     "across_width_differing": across, "actions_digest": digest(roll["actions"])})
    for width in widths:
        ms = median_ms(lambda: iso_roll(width), args.repeats)
        print(f"  width {width}: {ms:.1f} ms per call, {ms / width:.1f} ms per sample", flush=True)
        rows.append({"arm": "isomorphic", "width": width, "ms_per_call": ms, "ms_per_sample": ms / width})

    if args.native:
        import gc
        from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
        print("== native frontend (its own BF16 pipeline) against the training forward", flush=True)
        for width in widths:
            # one frontend per width: a parked pipeline of another width keeps its buffers
            native = Pi05TorchFrontendRtx(args.checkpoint_dir, num_views=3, chunk_size=args.chunk, num_steps=args.steps,
                                          denoise_trace=True, sde=True)
            if width == 1:
                native.set_prompt(prompts[0])
                native.calibrate([obs[0]])
                call = lambda: [native.infer(obs[0], noise=x0[0], step_noise=eps[:, 0], sde_sigma=sigma)]  # noqa: E731
            else:
                native.set_batched_mode(enable=True, batch_size=width)
                native.set_prompt_batch(prompts[:width])
                native.calibrate_batch([obs[0]])
                call = lambda width=width: native.infer_batch(obs[:width], noise=x0[:width], step_noise=eps[:, :width], sde_sigma=sigma)  # noqa: E731
            res = call()
            again = call()
            repeat = sum(int((np.asarray(a["raw_actions"]) != np.asarray(b["raw_actions"])).sum()) for a, b in zip(res, again))
            tx = torch.from_numpy(np.stack([r["denoise_trace"]["x"] for r in res], axis=1)).cuda()      # [steps, W, chunk, 32]
            td = torch.from_numpy(np.stack([r["denoise_trace"]["delta"] for r in res], axis=1)).cuda()
            raw = torch.from_numpy(np.stack([r["raw_actions"] for r in res])).cuda()
            # the trainer's forward at the native rollout's recorded states
            iso.set_prompt_batch(prompts[:width])
            roll = {"images": iso._camera_tensors(obs[:width]), "tokens": iso._tokens, "mask": iso._mask, "trace_x": tx}
            v = replay(iso.trainer, roll, ts, args.minibatch, args.seed + width)
            delta = dt * v
            differing = int((delta != td).sum())
            rel = float((delta - td).double().pow(2).mean().sqrt() / td.double().pow(2).mean().sqrt())
            # the recorded stochastic transitions: x[s + 1] as the rollout produced it, under both means
            lr = []
            for s in range(args.steps - 1):
                lr.append(log_ratio(tx[s] + delta[s], tx[s] + td[s], tx[s + 1], args.sigma))
            lr = torch.cat(lr)
            # the same samples through the isomorphic frontend
            iso_res = iso_roll(width)
            iso_raw = torch.from_numpy(np.stack([r["raw_actions"] for r in iso_res])).cuda()
            act_rel = float((iso_raw - raw).double().pow(2).mean().sqrt() / raw.double().pow(2).mean().sqrt())
            cos = float(torch.nn.functional.cosine_similarity(iso_raw.double().flatten(), raw.double().flatten(), dim=0))
            ms = median_ms(call, args.repeats)
            print(f"  width {width}: {differing} of {td.numel()} step elements differ from the trainer's (relative rms {rel:.2e}); "
                  f"log-ratio of a recorded transition at sigma {args.sigma}: rms {float(lr.pow(2).mean().sqrt()):.3e}, "
                  f"ratio within [{math.exp(float(lr.min())):.4f}, {math.exp(float(lr.max())):.4f}]; "
                  f"actions vs the isomorphic frontend: relative rms {act_rel:.2e}, cosine {cos:.6f}; "
                  f"the same call twice: {repeat} elements differ; actions digest {digest(raw)}; "
                  f"{ms:.1f} ms per call, {ms / width:.1f} ms per sample", flush=True)
            rows.append({"arm": "native", "width": width, "minibatch": args.minibatch, "differing": differing, "elements": td.numel(),
                         "relative_rms": rel, "log_ratio_rms": float(lr.pow(2).mean().sqrt()), "ratio_min": math.exp(float(lr.min())),
                         "ratio_max": math.exp(float(lr.max())), "actions_relative_rms": act_rel, "actions_cosine": cos,
                         "repeat_differing": repeat, "actions_digest": digest(raw), "ms_per_call": ms, "ms_per_sample": ms / width})
            del native, call
            gc.collect()
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "a") as f:
            for r in rows:
                f.write(json.dumps({**r, "steps": args.steps, "chunk": args.chunk, "sigma": args.sigma,
                                    "max_prompt_len": args.max_prompt_len, "prompts": "same" if args.same_prompt else "mixed"}) + "\n")
    print("isomorphic gate:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
