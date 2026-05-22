#!/usr/bin/env python
"""LingBot-VLA (Thor sm_110a) quickstart.

Runs the LingBot-VLA flow-matching action expert end-to-end on Thor and
reports per-step-count latency. LingBot is wired through the low-level
``graph_runner.sample_actions_graph`` path (the ``LingbotTorchFrontendThor``
class is a G1 scaffold and is NOT used here; LingBot is not yet registered in
``load_model`` / ``_PIPELINE_MAP``).

Build first (one shared module — LingBot kernels live inside flash_rt_kernels):

    cmake -B build -S . -DGPU_ARCH=110
    cmake --build build -j --target flash_rt_kernels flash_rt_fp4 fmha_fp16_strided
    pip install ".[torch,thor-fa4]"      # thor-fa4 = FA4 deps (cutlass-dsl + quack)

Run (FA4 source is vendored at csrc/attention/flash_attn_4_src; the loader sets
the Thor arch alias CUTE_DSL_ARCH=sm_101a for you):

    python examples/lingbot_quickstart.py \
        --checkpoint /path/to/lingbot-vla-4b \
        --calibration /path/to/lingbot_thor_static.json \
        --inputs /path/to/baseline_artifacts_10/inputs \
        --steps 50 25 10

Expected on Thor (FA4 active): ~158ms@50 / ~100ms@25 / ~64ms@10. If you see
~176/120/73ms, FA4 fell back to fmha — check `pip install .[thor-fa4]` and the
printed FA4 status (``flash_rt.hardware.thor.fa4_backend.status()``).
"""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from flash_rt.executors.torch_weights import SafetensorsSource
from flash_rt.executors.weight_loader import WeightLoader
from flash_rt.frontends.torch._lingbot_thor_spec import build_spec
from flash_rt.models.lingbot.buffer_binder import bind_target_to_device
from flash_rt.models.lingbot.kernel_ops import clear_fp8_weight_cache
from flash_rt.models.lingbot.graph_runner import sample_actions_graph
from flash_rt.models.lingbot import calibration as calib
from flash_rt.hardware.thor import fa4_backend

_INPUT_KEYS = ["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"]


def _load_inputs(inputs_dir: Path) -> dict:
    return {k: torch.load(inputs_dir / f"{k}.pt").cuda() for k in _INPUT_KEYS}


def main():
    ap = argparse.ArgumentParser(description="LingBot-VLA Thor quickstart")
    ap.add_argument("--checkpoint", required=True,
                    help="lingbot-vla-4b dir (contains model.safetensors)")
    ap.add_argument("--calibration", required=True,
                    help="FP8 static-scale calibration JSON")
    ap.add_argument("--inputs", required=True,
                    help="dir with images/img_masks/lang_tokens/lang_masks/state/noise .pt")
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 25, 10],
                    help="denoising step counts to benchmark")
    ap.add_argument("--iters", type=int, default=20, help="timed replays per step count")
    args = ap.parse_args()

    fa4 = fa4_backend.is_available()
    print(f"[lingbot] FA4 (denoise/prefix attention) active: {fa4} ({fa4_backend.status()})"
          f"{'' if fa4 else '  <-- falling back to fmha (+~18ms@25); pip install .[thor-fa4]'}")

    ckpt = Path(args.checkpoint) / "model.safetensors"
    clear_fp8_weight_cache()
    src = SafetensorsSource(str(ckpt), device="cpu", strip_prefix="")
    target = SimpleNamespace()
    WeightLoader(src, target=target, spec=build_spec()).run()
    bind_target_to_device(target, dtype=torch.bfloat16, device="cuda")
    calib.set_static_scales(calib.load_calibration(
        args.calibration, device=torch.device("cuda")))
    inp = _load_inputs(Path(args.inputs))

    for ns in args.steps:
        replay, out, g = sample_actions_graph(
            target=target, num_steps=ns, warmup_iters=3, **inp)
        replay(); torch.cuda.synchronize()
        if ns == args.steps[0]:
            print(f"[lingbot] action chunk shape={tuple(out.shape)} "
                  f"finite={bool(torch.isfinite(out).all())}")
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            replay(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        p50 = sorted(ts)[len(ts) // 2] * 1000
        print(f"[lingbot] num_denoising_steps={ns:3d}  P50={p50:6.1f} ms (CUDA-graph replay)")
        del replay, out, g
        torch.cuda.empty_cache(); clear_fp8_weight_cache()


if __name__ == "__main__":
    main()
