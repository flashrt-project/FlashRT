#!/usr/bin/env python3
"""LIBERO eval for the Pi0.5 encoder-FP4 + decoder-FP4 + FA4 path.

Usage:
    python tests/eval_libero_fp4.py \\
        --checkpoint /workspace/pytorch_checkpoints/pi05_libero_converted \\
        --task_suite libero_spatial [--full]
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import flash_rt
from examples.thor import eval_libero

_LAT = []


def _hook(m):
    orig = m.predict
    def timed(*a, **kw):
        t0 = time.perf_counter()
        r = orig(*a, **kw)
        _LAT.append((time.perf_counter() - t0) * 1000.0)
        return r
    m.predict = timed
    return m


def run(args):
    eval_libero._patch_egl_cleanup()
    fp4_layers = tuple(int(x) for x in args.fp4_layers.split(","))
    model = flash_rt.load_model(
        checkpoint=args.checkpoint, framework="torch",
        num_views=2, autotune=args.autotune, config="pi05", hardware="auto",
        use_fp4=True, fp4_layers=fp4_layers,
        use_awq=args.use_awq, awq_alpha=args.awq_alpha,
        use_p1_split_gu=args.use_p1_split_gu,
        use_fp4_decoder=True, use_fa4=True)
    _hook(model)

    pipe_obj = model._pipe
    if not pipe_obj.use_fp4_encoder_ffn or not pipe_obj.use_fp4_decoder:
        raise RuntimeError("strict LIBERO eval requires encoder and decoder FP4")
    if not pipe_obj.use_fa4:
        raise RuntimeError("strict LIBERO eval requires FA4")
    if pipe_obj._fp4_layers != frozenset(fp4_layers):
        raise RuntimeError(
            f"requested FP4 layers {fp4_layers}, got "
            f"{sorted(pipe_obj._fp4_layers)}")
    print(
        f"[FP4+FP4+FA4 active] layers={sorted(pipe_obj._fp4_layers)} "
        f"AWQ={pipe_obj.use_awq} alpha={pipe_obj.awq_alpha} "
        f"P1={pipe_obj.use_p1_split_gu}", flush=True)

    from libero.libero import benchmark as _lb
    import tqdm
    task_suite = _lb.get_benchmark_dict()[args.task_suite]()
    max_steps = eval_libero.MAX_STEPS_DICT[args.task_suite]

    if args.full:
        task_ids = list(range(task_suite.n_tasks))
        num_trials = args.trials_per_task
    else:
        task_ids = [0, 1, 2]
        num_trials = 3

    print(f"tasks={len(task_ids)}  trials/task={num_trials}  max_steps={max_steps}", flush=True)

    total_s = 0; total_e = 0
    task_results = []

    for tid in task_ids:
        task = task_suite.get_task(tid)
        init = task_suite.get_task_init_states(tid)
        env, desc = eval_libero.get_libero_env(
            task, eval_libero.LIBERO_ENV_RESOLUTION, args.seed)

        succ = 0
        bar = tqdm.tqdm(range(num_trials), desc=f"Task {tid}")
        for i in bar:
            env.reset()
            obs = env.set_init_state(init[i % len(init)])
            ok = eval_libero.run_episode(
                env, model, desc, max_steps=max_steps,
                replan_steps=args.replan_steps, num_steps_wait=10, obs=obs)
            succ += int(ok)
            bar.set_postfix(success=f"{succ}/{i+1}")
        env.close()
        total_s += succ; total_e += num_trials
        task_results.append((tid, succ, num_trials, desc))
        print(f"Task {tid}: {succ}/{num_trials} — {desc}", flush=True)

    print(f"\n{'='*60}\nFP4 Overall: {total_s}/{total_e} = {total_s/total_e:.1%}\n{'='*60}")

    if _LAT:
        lat = np.array(_LAT[3:])  # drop first 3 recalibration
        lat_s = np.sort(lat)
        n = len(lat_s)
        print(f"\nE2E predict() latency ({n} calls, steady-state):")
        print(f"  P50={lat_s[n//2]:.2f}  P90={lat_s[int(n*0.9)]:.2f}  "
              f"P99={lat_s[int(n*0.99)]:.2f}  max={lat.max():.2f}  mean={lat.mean():.2f}")

    return total_s, total_e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", default="libero_spatial")
    p.add_argument("--autotune", type=int, default=3)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--full", action="store_true", help="run all tasks, else 3x3 quick")
    p.add_argument("--trials_per_task", type=int, default=10)
    p.add_argument(
        "--fp4_layers", default=",".join(str(i) for i in range(17)),
        help="comma-separated live encoder FFN layers")
    p.add_argument(
        "--use_awq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--awq_alpha", type=float, default=0.8)
    p.add_argument(
        "--use_p1_split_gu", action=argparse.BooleanOptionalAction,
        default=True, help="P1 split-GU 2-GEMM path")
    args = p.parse_args()
    t0 = time.perf_counter()
    run(args)
    print(f"\nTotal wall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
