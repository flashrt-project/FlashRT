#!/usr/bin/env python3
"""Action accuracy of TensorRT engines against openpi PyTorch, per observation.

For every observation in a calibration-style .npz (tools/make_libero_fixture.py)
it runs openpi's pi05 PyTorch model and each engine through openpi's policy
transforms with the same pinned noise, and reports the cosine similarity of
the raw actions over the robot's action dimensions. Engines are the FlashRT
engine (through openpi_flashrt.py) and, optionally, the openpi Thor tutorial
engine (through its own hook). Run inside the openpi Thor container.

usage: compare_openpi_accuracy.py <observations.npz> <checkpoint> <flashrt.engine> <plugin.so>
                                  [--tutorial-engine model_fp8_nvfp4.engine] [--action-dim 7]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
from openpi.policies import policy_config
from openpi.training import config as _config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openpi_flashrt import setup_pi0_flashrt_engine  # noqa: E402
from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix, setup_pi0_tensorrt_engine  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("observations")
p.add_argument("checkpoint")
p.add_argument("engine")
p.add_argument("plugin")
p.add_argument("--tutorial-engine")
p.add_argument("--config", default="pi05_libero")
p.add_argument("--action-dim", type=int, default=7)
args = p.parse_args()

data = np.load(args.observations)
n = int(data["n"])
config = _config.get_config(args.config)


def observation(policy, i):
    example = {"observation/image": data[f"img_{i}"], "observation/wrist_image": data[f"wrist_{i}"],
               "observation/state": data[f"state_{i}"], "prompt": str(data[f"prompt_{i}"])}
    x = policy._input_transform(dict(example))
    to = lambda v: torch.from_numpy(np.array(v)).to("cuda")[None, ...]  # noqa: E731
    x = {k: ({kk: to(vv) for kk, vv in v.items()} if isinstance(v, dict) else to(v)) for k, v in x.items()}
    for k, img in x["image"].items():
        if img.dtype == torch.uint8:
            x["image"][k] = img.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
    return types.SimpleNamespace(images=x["image"], image_masks=x["image_mask"], state=x["state"],
                                 tokenized_prompt=x["tokenized_prompt"],
                                 tokenized_prompt_mask=x["tokenized_prompt_mask"],
                                 token_ar_mask=x.get("token_ar_mask"), token_loss_mask=x.get("token_loss_mask"))


def run_all(policy):
    out = []
    for i in range(n):
        noise = np.random.default_rng(i).standard_normal((1, 10, 32)).astype(np.float32)
        with torch.no_grad():
            a = policy._model.sample_actions("cuda", observation(policy, i), noise=torch.from_numpy(noise).cuda(),
                                             num_steps=10)
        out.append(a[0].float().cpu().numpy().astype(np.float64))
    return out


policy = policy_config.create_trained_policy(config, args.checkpoint)
install_attention_mask_dtype_fix(policy._model)
reference = run_all(policy)
del policy
torch.cuda.empty_cache()

engines = [("FlashRT engine", lambda pol: setup_pi0_flashrt_engine(pol, args.engine, args.plugin))]
if args.tutorial_engine:
    engines.append(("tutorial engine", lambda pol: setup_pi0_tensorrt_engine(pol, args.tutorial_engine)))


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


d = args.action_dim
print(f"{n} observations, prompts: {[str(data[f'prompt_{i}']) for i in range(n)]}")
for name, setup in engines:
    policy = setup(policy_config.create_trained_policy(config, args.checkpoint))
    got = run_all(policy)
    c = [cos(g[:, :d], r[:, :d]) for g, r in zip(got, reference)]
    print(f"{name:16s} vs openpi PyTorch, cosine over {d} action dims: mean {np.mean(c):.5f} min {np.min(c):.5f} "
          f"per observation {[round(v, 5) for v in c]}")
    del policy
    torch.cuda.empty_cache()
