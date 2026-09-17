#!/usr/bin/env python3
"""Write FlashRT calibration observations from the LIBERO LeRobot dataset.

Samples N frames evenly across the whole dataset (the openpi Thor tutorial's
calibration sampling), loading only the episodes that hold them, and stores
the base and wrist camera images as uint8 HWC resized to 224x224 the way
openpi's input transform does (resize_with_pad), the state and the task prompt
in the .npz layout FlashRT's Pi0.5 calibration reads. Runs wherever openpi
and lerobot are installed, e.g. the openpi Thor container.

usage: make_libero_fixture.py <out.npz> [--repo-id physical-intelligence/libero] [--num 8]
"""
import argparse

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi.policies.libero_policy import _parse_image
from openpi.shared.image_tools import resize_with_pad

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--repo-id", default="physical-intelligence/libero")
p.add_argument("--num", type=int, default=8)
args = p.parse_args()

meta = LeRobotDatasetMetadata(args.repo_id)
step = max(meta.total_frames // args.num, 1)
starts, total = {}, 0
for ep in sorted(meta.episodes):
    starts[ep] = total
    total += meta.episodes[ep]["length"]
located = []
for g in range(0, step * args.num, step):
    ep = max(e for e, s in starts.items() if s <= g)
    located.append((ep, g - starts[ep]))
episodes = sorted({ep for ep, _ in located})
ds = LeRobotDataset(args.repo_id, episodes=episodes)
subset_start, s = {}, 0
for ep in episodes:
    subset_start[ep] = s
    s += meta.episodes[ep]["length"]

out = {"n": np.array(args.num)}
for i, (ep, off) in enumerate(located):
    d = ds[subset_start[ep] + off]
    img, wrist = (np.asarray(resize_with_pad(_parse_image(d[k]), 224, 224)) for k in ("image", "wrist_image"))
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8, img.shape
    out[f"img_{i}"] = img
    out[f"wrist_{i}"] = wrist
    out[f"wrist_right_{i}"] = np.zeros_like(img)
    out[f"state_{i}"] = np.asarray(d["state"], dtype=np.float32)
    out[f"prompt_{i}"] = np.array(d["task"])
np.savez(args.out, **out)
print(f"wrote {args.out}: {args.num} observations from {len(episodes)} episodes of {args.repo_id}")
