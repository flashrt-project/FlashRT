#!/usr/bin/env python3
"""Motus RTC-lite execution demo.

This script keeps the Motus model path unchanged and wraps it with
``AsyncChunkRunner``. It is an offline timing harness for deployment-style
foreground action consumption plus background chunk generation.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import motus_quickstart as mq
from flash_rt.runtime.rtc import ActionChunkAdapter, AsyncChunkRunner, RTCConfig


class MotusBundleAdapter:
    """Adapter for MotusTorchFrontendRtx using one prepared input bundle."""

    def __init__(self, pipe, first_frame, state, seed: int):
        self.pipe = pipe
        self.first_frame = first_frame
        self.state = state
        self.seed = seed

    def infer_actions(self, observation) -> np.ndarray:
        _, actions, _ = mq._infer_once(
            self.pipe,
            observation.get("first_frame", self.first_frame),
            observation.get("state", self.state),
            self.seed,
        )
        actions = actions.detach().float().cpu().numpy()
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


def _build_motus(args):
    import torch

    assert torch.cuda.is_available(), "Motus RTC-lite requires CUDA"
    mq._install_deepspeed_stub()
    mq._install_optional_import_stubs(pathlib.Path(args.motus_root))
    mq._install_wan_config_filter()
    mq._set_motus_runtime_defaults()
    os.environ["FLASH_RT_MOTUS_ROOT"] = str(pathlib.Path(args.motus_root))
    os.environ["FLASH_RT_MOTUS_FP4_PROFILE"] = args.fp4_profile

    bundle = pathlib.Path(args.input_bundle)
    first_frame, state, instruction, t5_embeds, vlm_inputs = mq._load_inputs(bundle)

    from flash_rt.frontends.torch.motus_rtx import MotusTorchFrontendRtx

    pipe = MotusTorchFrontendRtx(
        checkpoint_dir=args.checkpoint,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        num_inference_steps=args.num_inference_steps,
        autotune=args.autotune,
    )
    mq._patch_qwen3vl_image_features(pipe)
    pipe.set_prompt(instruction, t5_embeds=t5_embeds, vlm_inputs=vlm_inputs)
    return pipe, first_frame, state


def _default_start_next_at(horizon: int, target_hz: float, latency_ms: float) -> int:
    delay_steps = int(np.ceil((latency_ms / 1000.0) * target_hz))
    return max(1, min(horizon - 1, horizon - delay_steps - 1))


def main() -> None:
    env_motus_root = os.environ.get("FLASH_RT_MOTUS_ROOT") or os.environ.get("MOTUS_ROOT")
    env_motus_root_path = pathlib.Path(env_motus_root) if env_motus_root else None

    def _default_from_root(*parts: str) -> str | None:
        if env_motus_root_path is None:
            return None
        return str(env_motus_root_path.joinpath(*parts))

    parser = argparse.ArgumentParser(description="Motus RTC-lite timing harness")
    parser.add_argument("--checkpoint", default=os.environ.get("MOTUS_CHECKPOINT")
                        or _default_from_root("pretrained_models", "Motus_robotwin2"))
    parser.add_argument("--motus-root", default=env_motus_root)
    parser.add_argument("--wan-path", default=os.environ.get("MOTUS_WAN_PATH")
                        or _default_from_root("pretrained_models", "Wan2.2-TI2V-5B"))
    parser.add_argument("--vlm-path", default=os.environ.get("MOTUS_VLM_PATH")
                        or _default_from_root("pretrained_models", "Qwen3-VL-2B-Instruct"))
    parser.add_argument("--input-bundle", default=os.environ.get("MOTUS_INPUT_BUNDLE")
                        or _default_from_root("baseline_artifacts"))
    parser.add_argument("--fp4-profile", default="fast",
                        choices=["fast", "fast-cache", "fast-tiny",
                                 "on", "off"])
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--autotune", type=int, default=0)
    parser.add_argument("--target-hz", type=float, default=20.0)
    parser.add_argument("--ticks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-next-at", type=int, default=None,
                        help="Action index that triggers next background chunk. "
                             "Defaults to a latency-derived value after warmup.")
    parser.add_argument("--miss-policy", choices=["hold_last", "block"],
                        default="hold_last")
    parser.add_argument("--blend-steps", type=int, default=0)
    parser.add_argument("--no-sleep", action="store_true",
                        help="Run ticks as fast as possible for smoke tests.")
    args = parser.parse_args()

    required = {
        "--motus-root or FLASH_RT_MOTUS_ROOT/MOTUS_ROOT": args.motus_root,
        "--checkpoint or MOTUS_CHECKPOINT": args.checkpoint,
        "--wan-path or MOTUS_WAN_PATH": args.wan_path,
        "--vlm-path or MOTUS_VLM_PATH": args.vlm_path,
        "--input-bundle or MOTUS_INPUT_BUNDLE": args.input_bundle,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing required Motus paths: " + ", ".join(missing))

    pipe, first_frame, state = _build_motus(args)
    adapter = MotusBundleAdapter(pipe, first_frame, state, args.seed)

    print("[motus.rtc] warmup: calibration + graph capture")
    _ = adapter.infer_actions({"first_frame": first_frame, "state": state})
    print("[motus.rtc] warmup: graph replay latency probe")
    t0 = time.perf_counter()
    first_actions = adapter.infer_actions({"first_frame": first_frame, "state": state})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    horizon = int(first_actions.shape[0])
    start_next_at = args.start_next_at
    if start_next_at is None:
        start_next_at = _default_start_next_at(horizon, args.target_hz, latency_ms)
    print(f"[motus.rtc] horizon={horizon} action_dim={first_actions.shape[1]} "
          f"target_hz={args.target_hz:.2f} latency_probe={latency_ms:.3f} ms "
          f"start_next_at={start_next_at}")

    runner = AsyncChunkRunner(
        adapter,
        RTCConfig(
            target_hz=args.target_hz,
            action_horizon=horizon,
            start_next_at=start_next_at,
            miss_policy=args.miss_policy,
            blend_steps=args.blend_steps,
        ),
    )

    period_s = 1.0 / args.target_hz
    actions = []
    tick_times = []
    obs = {"first_frame": first_frame, "state": state}
    print("[motus.rtc] prefill initial chunk")
    runner.reset(obs)
    t_start = time.perf_counter()
    next_tick = t_start + period_s
    try:
        for _ in range(args.ticks):
            if not args.no_sleep:
                now = time.perf_counter()
                if next_tick > now:
                    time.sleep(next_tick - now)
            tick_t = time.perf_counter()
            action = runner.next_action(obs)
            actions.append(action)
            tick_times.append(tick_t)
            next_tick += period_s
    finally:
        runner.close()

    elapsed = time.perf_counter() - t_start
    effective_hz = len(actions) / elapsed if elapsed > 0 else 0.0
    action_arr = np.stack(actions, axis=0)
    diffs = np.diff(action_arr, axis=0)
    l2_jump = np.linalg.norm(diffs, axis=-1) if diffs.size else np.array([])
    max_jump = float(l2_jump.max()) if l2_jump.size else 0.0
    mean_jump = float(l2_jump.mean()) if l2_jump.size else 0.0

    print(f"[motus.rtc] served={len(actions)} elapsed={elapsed:.3f}s "
          f"effective_hz={effective_hz:.2f}")
    print(f"[motus.rtc] stats={runner.stats}")
    print(f"[motus.rtc] action_jump_l2 mean={mean_jump:.6f} max={max_jump:.6f}")


if __name__ == "__main__":
    main()
