"""Measure every Pi0.5 precision tier against a common FP16 reference.

The strict E2E suite (``bench_pi05_decoder_fp4_e2e.py``) reports each
quantized tier's cosine against FP8, which leaves FP8's own deviation
unmeasured. This harness runs FP16, FP8 and the three quantized tiers
through the identical observation / prompt / seed protocol and reports
every tier against FP16, so the tiers sit on one yardstick.

One model per subprocess: Thor cannot hold two of these pipelines in a
single process.

    python tests/bench_pi05_precision_vs_fp16.py \
        --checkpoint <CHECKPOINT_DIR> \
        --fixture <FIXTURE_DIR>/libero_obs3v_n8.npz \
        --num-views 3 --output-dir <OUT_DIR>
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

PROMPT_TOKENS = [
    2, 18075, 908, 573, 3118, 3963, 578,
    2040, 665, 575, 573, 24655, 108,
]

TIERS = {
    # name: extra kwargs for the FP4 frontend (None => FP8/FP16 base class)
    "fp16": None,
    "fp8": None,
    "nvfp4": {},
    "int4": {"decoder_weight_format": "e0m3", "decoder_act_format": "e0m3"},
    "int4rht": {"decoder_weight_format": "e0m3", "decoder_act_format": "e0m3",
                "decoder_rht": True},
}


def build_pipe(mode, checkpoint, num_views):
    if mode in ("fp16", "fp8"):
        from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
        return Pi05TorchFrontendThor(
            checkpoint, num_views=num_views, autotune=3, use_fa4=True,
            use_fp8=(mode == "fp8"))
    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4
    return Pi05TorchFrontendThorFP4(
        checkpoint, num_views=num_views, autotune=3,
        use_fp4_encoder_ffn=True, fp4_layers=tuple(range(17)),
        use_awq=True, awq_alpha=0.8, use_p1_split_gu=True,
        use_fp4_decoder=True, use_fa4=True,
        use_fp4_encoder_attn=True, use_fp4_siglip_ffn=True,
        **TIERS[mode])


def load_observations(fixture, num_views):
    data = np.load(fixture)
    count = int(data["n"])
    observations = []
    for index in range(count):
        obs = {"image": data[f"img_{index}"], "state": data[f"state_{index}"]}
        if num_views >= 2:
            obs["wrist_image"] = data[f"wrist_{index}"]
        if num_views == 3:
            obs["wrist_image_right"] = data[f"wrist_right_{index}"]
        observations.append(obs)
    return observations


def run_child(args):
    import torch  # noqa: F401  (import side effects before pipeline import)

    pipe = build_pipe(args.child_mode, args.checkpoint, args.num_views)
    observations = load_observations(args.fixture, args.num_views)

    pipe.set_prompt(PROMPT_TOKENS)
    pipe.calibrate(observations, percentile=99.9, verbose=False)
    for index in range(args.warmup):
        pipe.infer(observations[index % len(observations)])

    raw_outputs, action_outputs = [], []
    for index, observation in enumerate(observations):
        np.random.seed(args.seed + index)
        output = pipe.infer(observation)
        raw_outputs.append(pipe._g_noise.float().cpu().numpy())
        action_outputs.append(output["actions"])

    if args.cuda_profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        pipe.infer(observations[0])
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    latencies = []
    for index in range(args.iters):
        start = time.perf_counter()
        pipe.infer(observations[index % len(observations)])
        latencies.append((time.perf_counter() - start) * 1000.0)
    torch.cuda.synchronize()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.child_mode}_actions.npz"
    np.savez(out, raw=np.stack(raw_outputs), actions=np.stack(action_outputs))
    result = {
        "tier": args.child_mode,
        "num_views": args.num_views,
        "iters": args.iters,
        "p50_ms": statistics.median(latencies),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    print("__CHILD_RESULT__ " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def cosines(lhs_all, rhs_all):
    per_sample = []
    for index in range(lhs_all.shape[0]):
        lhs = lhs_all[index].reshape(-1)
        rhs = rhs_all[index].reshape(-1)
        per_sample.append(float(
            lhs @ rhs / (np.linalg.norm(lhs) * np.linalg.norm(rhs) + 1e-12)))
    lhs = lhs_all.reshape(-1)
    rhs = rhs_all.reshape(-1)
    aggregate = float(
        lhs @ rhs / (np.linalg.norm(lhs) * np.linalg.norm(rhs) + 1e-12))
    return aggregate, min(per_sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-mode", choices=sorted(TIERS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--num-views", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--cuda-profile", action="store_true",
        help="capture one stable-state infer between "
             "cudaProfilerStart/Stop")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--modes", default="fp16,fp8,nvfp4,int4,int4rht")
    args = parser.parse_args()

    # Latency is only meaningful under the same locked-clock discipline the
    # strict suite enforces; reuse its verifier rather than duplicating it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_pi05_decoder_fp4_e2e import machine_state
    machine_state()

    if args.child_mode is not None:
        return run_child(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = args.modes.split(",")
    latency = {}
    for mode in modes:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--child-mode", mode,
            "--checkpoint", args.checkpoint,
            "--fixture", args.fixture,
            "--num-views", str(args.num_views),
            "--output-dir", str(output_dir),
            "--warmup", str(args.warmup),
            "--iters", str(args.iters),
            "--seed", str(args.seed),
        ]
        child = subprocess.run(command, check=False, capture_output=True,
                               text=True, env=os.environ.copy())
        if child.returncode != 0:
            raise RuntimeError(
                f"{mode} child failed rc={child.returncode}\n"
                f"{child.stdout[-2000:]}\n{child.stderr[-4000:]}")
        lines = [line for line in child.stdout.splitlines()
                 if line.startswith("__CHILD_RESULT__")]
        if len(lines) != 1:
            raise RuntimeError(
                f"expected one result line from {mode}, got {len(lines)}")
        latency[mode] = json.loads(lines[0].split(" ", 1)[1])
        print(f"  {mode} done  p50={latency[mode]['p50_ms']:.3f} ms",
              flush=True)

    ref = np.load(output_dir / "fp16_actions.npz")
    raw_ref = ref["raw"].astype(np.float64)
    act_ref = ref["actions"].astype(np.float64)

    rows = []
    for mode in modes:
        if mode == "fp16":
            continue
        cur = np.load(output_dir / f"{mode}_actions.npz")
        raw_cos, raw_min = cosines(cur["raw"].astype(np.float64), raw_ref)
        act_cos, act_min = cosines(cur["actions"].astype(np.float64), act_ref)
        rows.append((mode, raw_cos, raw_min, act_cos, act_min))

    ref_p50 = latency["fp16"]["p50_ms"] if "fp16" in latency else None
    print(f"\n=== precision matrix, {args.num_views} view(s) "
          "(latency measured, cosine vs FP16) ===")
    header = (f"{'tier':10s} {'p50 ms':>8s} {'p95 ms':>8s} {'vs FP16':>8s} "
              f"{'raw cos':>9s} {'raw min':>9s} {'act cos':>9s} "
              f"{'act min':>9s}")
    print(header)
    if "fp16" in latency:
        f16 = latency["fp16"]
        print(f"{'fp16':10s} {f16['p50_ms']:8.3f} {f16['p95_ms']:8.3f} "
              f"{1.0:8.3f} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s}")
    for mode, raw_cos, raw_min, act_cos, act_min in rows:
        lat = latency.get(mode, {})
        p50 = lat.get("p50_ms", float("nan"))
        p95 = lat.get("p95_ms", float("nan"))
        speedup = (ref_p50 / p50) if (ref_p50 and p50 == p50) else float("nan")
        print(f"{mode:10s} {p50:8.3f} {p95:8.3f} {speedup:8.3f} "
              f"{raw_cos:9.5f} {raw_min:9.5f} {act_cos:9.5f} {act_min:9.5f}")

    payload = {
        "num_views": args.num_views,
        "latency": latency,
        "accuracy_vs_fp16": [
            dict(zip(("tier", "raw_cosine", "raw_min_sample_cosine",
                      "action_cosine", "action_min_sample_cosine"), r))
            for r in rows],
    }
    json.dump(payload, open(output_dir / "precision_matrix.json", "w"),
              indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
