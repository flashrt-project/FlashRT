#!/usr/bin/env python3
"""Time openpi policy.infer with a TensorRT engine on one observation.

  --plugin given: FlashRT engine through openpi_flashrt.py
  --plugin omitted: the openpi Thor tutorial engine through its own hook

Reports the median model time (policy_timing.infer_ms) and total call time.
Run inside the openpi Thor container.

usage: benchmark_openpi.py <observations.npz> <checkpoint> <engine> [--plugin libflashrt_trt_pi05.so] [--iters 100]
"""
import argparse
import os
import sys
import time

import numpy as np
from openpi.policies import policy_config
from openpi.training import config as _config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("observations")
p.add_argument("checkpoint")
p.add_argument("engine")
p.add_argument("--plugin")
p.add_argument("--config", default="pi05_libero")
p.add_argument("--index", type=int, default=0)
p.add_argument("--iters", type=int, default=100)
args = p.parse_args()

data = np.load(args.observations)
i = args.index
example = {"observation/image": data[f"img_{i}"], "observation/wrist_image": data[f"wrist_{i}"],
           "observation/state": data[f"state_{i}"], "prompt": str(data[f"prompt_{i}"])}
policy = policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
if args.plugin:
    from openpi_flashrt import setup_pi0_flashrt_engine
    policy = setup_pi0_flashrt_engine(policy, args.engine, args.plugin)
else:
    from deployment_scripts.trt_model_forward import setup_pi0_tensorrt_engine
    policy = setup_pi0_tensorrt_engine(policy, args.engine)
noise = np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32)
for _ in range(10):
    policy.infer(example, noise=noise)
model, total = [], []
for _ in range(args.iters):
    t = time.perf_counter()
    out = policy.infer(example, noise=noise)
    total.append((time.perf_counter() - t) * 1e3)
    model.append(out["policy_timing"]["infer_ms"])
name = "FlashRT engine" if args.plugin else "tutorial engine"
print(f"{name}: prompt {example['prompt']!r} | model median {np.median(model):.2f} ms | "
      f"total median {np.median(total):.2f} ms | actions {out['actions'].shape}")
