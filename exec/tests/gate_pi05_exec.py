"""Phase-C gate — Pi0.5 RTX infer graph replay routed through the exec contract.

Pi0.5 (VLA) uses the framework's ctypes CUDAGraph (one fused infer graph:
vision + encoder + decoder + diffusion). Under FLASHRT_PI05_USE_EXEC=1 the
pipeline adopts that graph's instantiated exec into an frt_graph and replays it
through the exec layer. This run prints the action-output hash + latency under
one flag value (run twice, off vs on, to compare):
  - action hash MUST be identical  (exec-driven replay is bit-identical)
  - latency MUST NOT regress

A within-process determinism check (same seed -> same actions twice) guards
that the hash comparison is meaningful.

Run twice (inside pi0-stablehlo-test, after building fp16 fa2):
  for v in 0 1; do
    PYTHONPATH=/workspace/PI/official/FlashRT-spec \
    FLASHRT_PI05_USE_EXEC=$v PYTORCH_ALLOC_CONF=expandable_segments:True \
    python exec/tests/gate_pi05_exec.py --checkpoint /workspace/PI/checkpoints/pi05_libero_pytorch
  done
"""

import argparse
import hashlib
import os
import time

import numpy as np
import torch

import flash_rt

USE_EXEC = os.environ.get("FLASHRT_PI05_USE_EXEC", "0")
SEED = 1234
PROMPT = "pick up the red block"


def _fixed_images(num_views):
    rng = np.random.RandomState(SEED)
    return [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(num_views)]


def _seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--fp8", action="store_true")
    args = ap.parse_args()

    images = _fixed_images(args.num_views)
    print(f"precision={'fp8' if args.fp8 else 'fp16'}")
    model = flash_rt.load_model(
        args.checkpoint, framework="torch", config="pi05", hardware="auto",
        num_views=args.num_views, num_steps=args.steps, cache_frames=1,
        use_fp8=bool(args.fp8), use_fp16=not args.fp8)
    pipe = model._pipe

    _seed(); out0 = np.asarray(model.predict(images, prompt=PROMPT))  # builds graph
    # _use_exec is set during the first predict's graph capture, so check now.
    print(f"pipe={type(pipe).__name__} use_exec_active="
          f"{getattr(getattr(pipe, 'pipeline', None), '_use_exec', False)}")
    _seed(); out_a = np.asarray(model.predict(images))
    _seed(); out_b = np.asarray(model.predict(images))
    deterministic = np.array_equal(out_a, out_b)
    h = hashlib.sha256(out_a.tobytes()).hexdigest()[:16]

    # warm + timed
    for _ in range(10):
        model.predict(images)
    torch.cuda.synchronize()
    wall = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        model.predict(images)
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000.0)
    wall.sort()
    p50 = wall[len(wall) // 2]

    print(f"RESULT USE_EXEC={USE_EXEC} actions_shape={out_a.shape} "
          f"deterministic={deterministic} action_sha={h} "
          f"p50_ms={p50:.3f} min_ms={wall[0]:.3f} finite={np.isfinite(out_a).all()}")


if __name__ == "__main__":
    main()
