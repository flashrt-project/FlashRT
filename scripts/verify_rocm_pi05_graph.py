#!/usr/bin/env python3
"""Verify the Pi0.5 ROCm graph pipeline against the openpi reference."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np


def _require_rocm() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("verify_rocm_pi05_graph.py requires ROCm PyTorch")
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "capability": tuple(int(x) for x in torch.cuda.get_device_capability()),
    }


def _make_observation(num_views: int, action_dim: int) -> dict[str, Any]:
    images = [
        np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(max(1, int(num_views)))
    ]
    return {
        "images": images,
        "image": images[0],
        "wrist_image": images[1] if len(images) > 1 else images[0],
        "wrist_image_right": images[2] if len(images) > 2 else images[-1],
        "state": np.zeros((int(action_dim),), dtype=np.float32),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float32).reshape(-1)
    bf = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom == 0.0:
        return 1.0 if np.array_equal(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def _reference_actions(model, observation: dict[str, Any], num_steps: int) -> np.ndarray:
    import torch

    obs = model._make_observation(observation)
    noise = torch.zeros(
        (1, model._model_cfg.action_horizon, model._model_cfg.action_dim),
        device="cuda",
        dtype=torch.float32,
    )
    with torch.inference_mode():
        ref = model._model.sample_actions(
            "cuda",
            obs,
            noise=noise,
            num_steps=int(num_steps),
        )[0]
    torch.cuda.synchronize()
    return ref.float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--use-fp8", action="store_true")
    parser.add_argument("--compare-openpi", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=None)
    parser.add_argument("--expect-path", default=None)
    parser.add_argument("--expect-attn-backend", default="ck_wmma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = _require_rocm()

    import torch

    from flash_rt.frontends.torch.pi05_rocm import Pi05TorchFrontendRocm

    load_t0 = time.perf_counter()
    model = Pi05TorchFrontendRocm(
        args.checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        num_steps=args.num_steps,
        use_fp8=args.use_fp8,
    )
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    observation = _make_observation(args.num_views, model._model_cfg.action_dim)
    model.set_prompt(args.prompt, state=observation["state"])

    capture_t0 = time.perf_counter()
    model.calibrate_with_real_data([observation])
    torch.cuda.synchronize()
    graph_capture_ms = (time.perf_counter() - capture_t0) * 1000.0

    samples = []
    last = None
    for _ in range(max(1, int(args.repeat))):
        last = model.infer(observation, debug=True)
        samples.append(float(last["debug"]["infer_ms"]))
    assert last is not None

    actions = np.asarray(last["actions"], dtype=np.float32)
    if not np.isfinite(actions).all():
        raise AssertionError("Pi0.5 ROCm graph pipeline returned non-finite actions")

    debug = dict(last["debug"])
    expected_path = args.expect_path or ("fp8_graph" if args.use_fp8 else "bf16_graph")
    if debug.get("pipeline_path") != expected_path:
        raise AssertionError(
            f"unexpected pipeline_path={debug.get('pipeline_path')!r}; "
            f"expected {expected_path!r}"
        )
    if not debug.get("pipeline_graph_recorded"):
        raise AssertionError("Pi0.5 ROCm graph was not recorded")
    if debug.get("attn_backend") != args.expect_attn_backend:
        raise AssertionError(
            f"unexpected attn_backend={debug.get('attn_backend')!r}; "
            f"expected {args.expect_attn_backend!r}"
        )
    if debug.get("decoder_attn_backend") != args.expect_attn_backend:
        raise AssertionError(
            "unexpected decoder_attn_backend="
            f"{debug.get('decoder_attn_backend')!r}; "
            f"expected {args.expect_attn_backend!r}"
        )

    result: dict[str, Any] = {
        "runtime": runtime,
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "num_views": int(args.num_views),
        "num_steps": int(args.num_steps),
        "chunk_size": int(args.chunk_size),
        "load_ms": float(load_ms),
        "graph_capture_ms": float(graph_capture_ms),
        "latency_ms": {
            "count": len(samples),
            "last": samples[-1],
            "mean": float(np.mean(samples)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
        },
        "actions_shape": tuple(int(x) for x in actions.shape),
        "debug": debug,
    }

    if args.compare_openpi:
        ref_t0 = time.perf_counter()
        ref = _reference_actions(model, observation, args.num_steps)
        torch.cuda.synchronize()
        ref_ms = (time.perf_counter() - ref_t0) * 1000.0
        cos = _cosine(actions, ref)
        mean_abs = float(np.mean(np.abs(actions - ref)))
        result["openpi_reference"] = {
            "latency_ms": float(ref_ms),
            "cosine": cos,
            "mean_abs": mean_abs,
        }
        if cos < float(args.min_cosine):
            raise AssertionError(
                f"cosine {cos:.6f} is below threshold {args.min_cosine:.6f}"
            )
        if args.max_mean_abs is not None and mean_abs > float(args.max_mean_abs):
            raise AssertionError(
                f"mean_abs {mean_abs:.6f} exceeds threshold {args.max_mean_abs:.6f}"
            )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
