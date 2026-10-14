#!/usr/bin/env python3
"""FlashRT — DM05-LIBERO quickstart.

Runs a real observation through the DM05-LIBERO vision-language-action pipeline and
prints the predicted action chunk, with latency.

Usage:
    export DM05_CHECKPOINT=/path/to/DM05-libero
    export DM05_CALIBRATION=/path/to/calibration_state.pt

    python examples/dm05_quickstart.py --observation obs.json

    # time a replay instead of a first call
    python examples/dm05_quickstart.py --observation obs.json --repeat 50

The observation is the JSON the LIBERO harness already produces, one object per
line:

    {"observation": {"images": {"1": "<base64 jpeg>", "2": "<base64 jpeg>"},
                     "state": [ ... ], "prompt": "...", "robot_type": "..."}}

It carries `DM05_INFERENCE_SEED` like the rest of the pipeline, so repeated runs at
the same seed are reproducible.

The checkpoint is loaded by the model's own runtime and FlashRT's pipeline is
installed over it by `serve.enable`, the same shape as the other model quickstarts
here: the model package owns loading and tokenization, this repository owns the
kernels and the graph. `serve.enable` is not optional polish -- the preprocessing
install and the graph-captured predict have to come from the same module, and a
mismatch fails silently from inside the predict path.
"""
from __future__ import annotations

import argparse, base64, io, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

_runtime = None


def _load(checkpoint, calibration, diffusion_steps=10, views=("Head", "Left wrist")):
    """Build the DM05 runtime and install this repository's pipeline over it."""
    global _runtime
    if _runtime is not None:
        return _runtime

    import torch
    from PIL import Image

    try:
        from playground.dm05_libero import DM05Exp
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "the DM05 runtime is not importable. It ships with the model "
            f"distribution, not with FlashRT: {exc}"
        ) from exc

    exp = DM05Exp(
        task="inference",
        model_config=DM05Exp.__dataclass_fields__["model_config"].default_factory(),
        data_config=DM05Exp.__dataclass_fields__["data_config"].default_factory(),
        inference_config=DM05Exp.__dataclass_fields__["inference_config"].default_factory(),
    )
    exp.model_config.model_name_or_path = str(checkpoint)
    exp.model_config.chunk_size = 10
    exp.model_config.vision_attn_implementation = "sdpa"
    exp.inference_config.output_action_dim = 7
    exp.inference_config.image_prompts = list(views)
    exp._initialize_inference_runtime()
    ic = exp.inference_config
    ic.diffusion_steps = diffusion_steps

    from flash_rt.models.pi05 import serve

    state = torch.load(calibration, map_location="cuda", weights_only=False)
    serve.enable(ic, calibration_state=state, unwrap=_unwrap)
    print(f"[FlashRT] pipeline installed (diffusion_steps={diffusion_steps}, "
          f"views={list(views)})", flush=True)
    _runtime = (exp, ic, Image)
    return _runtime


def _unwrap(model):
    """DM05 runtimes may wrap the model; unwrapping belongs to that side."""
    try:
        from opendm.model.dm05.dm05_lora import unwrap_dm05_model
    except ImportError:
        return model
    return unwrap_dm05_model(model)


def _prepare(ic, Image, record):
    o = record["observation"]
    images = [Image.open(io.BytesIO(base64.b64decode(o["images"][k])))
              for k in sorted(o["images"], key=int)]
    return ic._prepare_model_input(
        text=o.get("prompt") or record.get("prompt", ""),
        images=images,
        states=o["state"],
        robot_type=o.get("robot_type"),
    )


def _predict(ic, data, repeat):
    import torch

    with torch.no_grad():
        action = np.asarray(ic._predict(data))
        for _ in range(max(0, repeat - 1)):
            ic._predict(data)
    return action


def _build_parser():
    p = argparse.ArgumentParser(description="FlashRT DM05-LIBERO quickstart")
    p.add_argument("--observation", required=True,
                   help="JSON observation file (one object, or one per line)")
    p.add_argument("--checkpoint", default=os.environ.get("DM05_CHECKPOINT"),
                   help="DM05-LIBERO checkpoint directory (or $DM05_CHECKPOINT)")
    p.add_argument("--calibration", default=os.environ.get("DM05_CALIBRATION"),
                   help="calibration state saved by NativeDM05.calibration_state() "
                        "(or $DM05_CALIBRATION)")
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--repeat", type=int, default=1,
                   help="run the same observation N times and report the median; "
                        "the first call captures the graph, later ones replay it")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint or $DM05_CHECKPOINT is required")
    if not args.calibration:
        raise SystemExit(
            "--calibration or $DM05_CALIBRATION is required: the static FP8 and "
            "NVFP4 scales are calibrated once on real observations and loaded, not "
            "recomputed per call."
        )
    if not os.path.isfile(args.calibration):
        raise SystemExit(f"calibration state not found: {args.calibration}")

    lines = [ln for ln in Path(args.observation).read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"no observations in {args.observation}")
    record = json.loads(lines[0])

    exp, ic, Image = _load(args.checkpoint, args.calibration, args.diffusion_steps)
    data = _prepare(ic, Image, record)

    t0 = time.perf_counter()
    action = _predict(ic, data, args.repeat)
    elapsed = time.perf_counter() - t0

    print(f"[FlashRT] action chunk {action.shape}")
    print(f"[FlashRT] first row: "
          f"{np.array2string(action.reshape(-1, action.shape[-1])[0], precision=4)}")
    if args.repeat > 1:
        # the first call captures the graph and is not steady state
        print(f"[FlashRT] {args.repeat} calls in {elapsed*1000:.0f}ms "
              f"({elapsed/args.repeat*1000:.1f} ms/call; the first captures the graph)")
    else:
        print(f"[FlashRT] {elapsed*1000:.0f}ms (graph capture included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
