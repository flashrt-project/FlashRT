"""Run OpenPI pi05_libero and check FlashRT against the same weights and noise."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from _validation import metrics, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("reference", "fp8", "fp4"), default="fp4")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "assets/pi05_libero.npz")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/pi05.npz"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("warmup must be nonnegative and iters must be positive")
    if args.mode == "reference":
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch
    prompt = "pick up the black bowl and place it on the plate"
    data = np.load(args.input, allow_pickle=False)
    observations = [dict(image=data[f"img_{i}"], wrist_image=data[f"wrist_{i}"],
                         state=data[f"state_{i}"]) for i in range(int(data["n"]))]
    if args.mode == "reference":
        from openpi.training import config
        from openpi.policies import policy_config
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        cfg = config.get_config("pi05_libero")
        policy = policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device="cuda")

        def run(index):
            obs = observations[index]
            noise = np.random.randn(cfg.model.action_horizon, 32).astype(np.float16).astype(np.float32)
            return policy.infer({"observation/image": obs["image"],
                                 "observation/wrist_image": obs["wrist_image"],
                                 "observation/state": obs["state"].astype(np.float32),
                                 "prompt": prompt}, noise=noise)["actions"]
    else:
        import flash_rt
        options = dict(use_fp4=True, use_fp4_decoder=True, awq_alpha=0.5,
                       encoder_p1_combiner="epilogue_hw_nod", encoder_down_variant=8) if args.mode == "fp4" else {}
        policy = flash_rt.load_model(args.checkpoint, config="pi05", hardware="thor",
                                     framework="torch", num_views=2, use_fa4=True, **options)
        policy.set_prompt(prompt)
        policy.calibrate(observations, percentile=99.9, verbose=False)

        def run(index):
            return policy.infer(observations[index])["actions"]

    for index in range(args.warmup):
        run(index % len(observations))
    actions = []
    for index in range(len(observations)):
        np.random.seed(20261007 + index)
        actions.append(np.asarray(run(index)))
    actions = np.stack(actions)
    if actions.shape != (8, 10, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Unexpected actions: {actions.shape}")
    samples = []
    for index in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run(index % len(observations))
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    report = dict(mode=args.mode, shape=list(actions.shape), warmup=args.warmup,
                  iters=args.iters, latency_ms=float(np.median(samples)),
                  p95_ms=float(np.percentile(samples, 95)))
    passed = True
    if args.reference:
        reference = np.load(args.reference, allow_pickle=False)["actions"]
        report.update(metrics(actions, reference))
        report["gripper_sign_disagreement"] = float((np.sign(actions[..., 6]) != np.sign(reference[..., 6])).mean())
        passed = report["mean_cosine"] >= .999 and report["worst_cosine"] >= .995 and report["gripper_sign_disagreement"] == 0
        report["passed"] = bool(passed)
    elif args.mode != "reference":
        report["accuracy_checked"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, actions=actions, samples_ms=samples)
    save_report(args.output.with_suffix(".json"), report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
