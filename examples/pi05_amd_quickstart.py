#!/usr/bin/env python
"""Pi0.5 on AMD Instinct (ROCm / CDNA4) quickstart.

Runs the pi05 AMD backend end-to-end — HIP-graph capture, real-data FP8
calibration, N timed inferences — and prints the latency stats plus the
precision spec. Works through the stable ``flash_rt.load_model`` door
(``hardware="amd_cdna4"``, auto-detected on ROCm builds for gfx950).

Build first (self-contained HIP module; the CUDA tree is not involved):

    bash scripts/amd/build_amd.sh gfx950
    # or: cmake -B build-amd -S csrc/amd -DGPU_ARCH=gfx950
    #     cmake --build build-amd -j 8

Run:

    python examples/pi05_amd_quickstart.py \
        --checkpoint /path/to/pi05_libero_pytorch \
        --prompt "pick up the black bowl and place it on the plate"

Expected on an MI350-series part (gfx950, ROCm 7.x, FP8 default):
~16-17 ms median per inference after warmup. ``--bf16`` selects the
unquantized baseline (~30 ms). See docs/deployment_amd.md for the
support matrix, environment knobs and measured numbers.
"""
import argparse
import statistics
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="Pi0.5 PyTorch checkpoint dir (model.safetensors)")
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--num-views", type=int, default=2, choices=[2, 3],
                    help="camera views: 2 = base + wrist (LIBERO), "
                         "3 = + right wrist")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20,
                    help="replays before the timed loop")
    ap.add_argument("--bf16", action="store_true",
                    help="disable FP8 (BF16 baseline)")
    ap.add_argument("--state-prompt-mode", default="exact",
                    choices=["exact", "fixed"],
                    help="'exact' = one graph per prompt length; "
                         "'fixed' = one padded graph for every length")
    args = ap.parse_args()

    import flash_rt
    model = flash_rt.load_model(
        args.checkpoint,
        config="pi05",
        framework="torch",
        hardware="amd_cdna4",
        num_views=args.num_views,
        use_fp8=not args.bf16,
        state_prompt_mode=args.state_prompt_mode,
    )
    fe = model.pipeline   # Pi05TorchFrontendAmd

    rng = np.random.default_rng(0)
    # Synthetic observation with one image per configured view, using the
    # frontend's per-view key names (2 views: image + wrist_image;
    # 3 views adds wrist_image_right).
    view_keys = ("image", "wrist_image", "wrist_image_right")
    obs = {
        key: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        for key in view_keys[:args.num_views]
    }
    state = rng.standard_normal(8).astype(np.float32)

    fe.set_prompt(args.prompt, state=state)   # builds + captures the HIP graph
    # Pi0.5 renders the robot state into the prompt, so the token count --
    # and with it the encoder sequence length and the latency -- depends on
    # both the prompt text and the state values.
    print(f"prompt+state tokens: {fe.current_prompt_len}  "
          f"encoder seq: {fe.pipeline.encoder_seq_len}")
    fe.calibrate(obs)                         # real-data FP8 calibration
    for _ in range(args.warmup):
        fe.infer(obs)                     # warmup replays

    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        result = fe.infer(obs)
        lat.append((time.perf_counter() - t0) * 1e3)

    print(f"actions: {result['actions'].shape}  "
          f"median {statistics.median(lat):.2f} ms  min {min(lat):.2f} ms "
          f"over {args.iters} iters")
    print("precision:", fe.precision_spec)
    print("stats:", fe.get_latency_stats())


if __name__ == "__main__":
    main()
