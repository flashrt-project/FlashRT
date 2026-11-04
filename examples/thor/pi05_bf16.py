"""Compare single-view, text-only Pi0.5 actions on the public Thor fixture."""

import argparse
import json
import os
import time
from pathlib import Path
from types import MethodType

import numpy as np

PROMPT = "pick up the black bowl and place it on the plate"


def reference_runner(checkpoint):
    import torch
    from openpi import transforms
    from openpi.models.model import Observation
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.preprocessing_pytorch import (
        preprocess_observation_pytorch,
    )
    from openpi.training import checkpoints, config

    cfg = config.get_config("pi05_libero")
    if cfg.model.action_horizon != 10 or cfg.model.action_dim != 32:
        raise ValueError("Expected OpenPI Pi0.5 LIBERO model geometry")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_float32_matmul_precision("high")
    model = cfg.model.load_pytorch(cfg, str(checkpoint / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.eval().cuda()
    stats = checkpoints.load_norm_stats(checkpoint / "assets", "physical-intelligence/libero")
    unnormalize = transforms.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)
    ids, mask = PaligemmaTokenizer(max_len=200).tokenize(PROMPT, state=None)
    camera = "base_0_rgb"

    def preprocess(self, obs, *, train=False):
        if train:
            raise ValueError("This reference uses inference preprocessing")
        result = preprocess_observation_pytorch(obs, train=False, image_keys=(camera,))
        return (
            list(result.images.values()),
            list(result.image_masks.values()),
            result.tokenized_prompt,
            result.tokenized_prompt_mask,
            result.state,
        )

    model._preprocess_observation = MethodType(preprocess, model)
    tokens = torch.from_numpy(ids)[None].cuda()
    token_mask = torch.from_numpy(mask)[None].cuda()

    @torch.no_grad()
    def run(image, noise):
        obs = Observation.from_dict(
            {
                "image": {camera: torch.from_numpy(image)[None].cuda()},
                "image_mask": {camera: torch.ones(1, dtype=torch.bool, device="cuda")},
                "state": torch.zeros((1, 32), device="cuda"),
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
            }
        )
        raw = (
            model.sample_actions(
                "cuda", obs, noise=torch.from_numpy(noise)[None].cuda(), num_steps=10
            )[0]
            .cpu()
            .numpy()
        )
        return {"raw_actions": raw, "actions": unnormalize({"actions": raw[:, :7]})["actions"]}

    return run


def flashrt_runner(checkpoint, mode):
    if mode == "eager":
        from flash_rt.frontends.torch.pi05_thor_bf16 import Pi05TorchFrontendThorBf16

        frontend = Pi05TorchFrontendThorBf16(checkpoint, use_cuda_graph=False)
    else:
        import flash_rt

        frontend = flash_rt.load_model(
            checkpoint, hardware="thor", num_views=1, precision="bf16", use_fp8=False
        ).pipeline
    frontend.set_prompt(PROMPT)
    return lambda image, noise: frontend.infer({"image": image, "noise": noise})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference", "eager", "graph"), required=True)
    parser.add_argument(
        "--input", type=Path, default=Path(__file__).parent / "assets/pi05_libero.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or (args.mode != "reference" and args.reference is None):
        parser.error("Use nonnegative warmup, positive iters, and --reference in FlashRT modes")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import torch

    with np.load(args.input, allow_pickle=False) as data:
        images = [data[f"img_{index}"] for index in range(int(data["n"]))]
    rng = np.random.default_rng(20261007)
    cases = [
        (image, rng.standard_normal((10, 32), dtype=np.float32))
        for image in images
        for _ in range(3)
    ]
    run = (
        reference_runner(args.checkpoint)
        if args.mode == "reference"
        else flashrt_runner(args.checkpoint, args.mode)
    )
    outputs = [run(*case) for case in cases]
    results = {key: np.stack([out[key] for out in outputs]) for key in ("raw_actions", "actions")}
    if not all(np.isfinite(array).all() for array in results.values()):
        raise ValueError("Nonfinite action output")
    for i in range(args.warmup):
        run(*cases[i % len(cases)])
    timings = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run(*cases[i % len(cases)])
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    report = {
        "mode": args.mode,
        "pairs": len(cases),
        "p50_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }
    if args.reference is not None:
        with np.load(args.reference, allow_pickle=False) as expected:
            for key, actual in results.items():
                if actual.shape != expected[key].shape:
                    raise ValueError(f"Shape mismatch for {key}")
                delta = actual.astype(np.float64) - expected[key].astype(np.float64)
                report[key] = {
                    "max_abs": float(np.max(np.abs(delta))),
                    "rmse": float(np.sqrt(np.mean(delta**2))),
                    "bytes_equal": actual.dtype == expected[key].dtype
                    and actual.tobytes() == expected[key].tobytes(),
                }
            report["gripper_sign_disagreements"] = int(
                np.count_nonzero(
                    np.sign(results["actions"][..., 6]) != np.sign(expected["actions"][..., 6])
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **results, samples_ms=timings)
    report_json = json.dumps(report, indent=2)
    args.output.with_suffix(".json").write_text(report_json + "\n")
    print(report_json)


if __name__ == "__main__":
    main()
