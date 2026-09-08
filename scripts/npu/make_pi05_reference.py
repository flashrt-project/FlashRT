#!/usr/bin/env python3
"""Generate the golden fp32 reference for NPU Pi0.5 verification.

Usage:
    python scripts/npu/make_pi05_reference.py <checkpoint_dir> <out.npz>

Runs the canonical recipe (image seed 0, noise seed 1, prompt
"pick up the cup", 2 views, 10 flow steps) through the **fp32 CPU eager**
math of ``flash_rt/npu/models/pi05/pipeline.py`` and stores the normalized
raw actions (10, 32). This file is the numeric floor that every BF16
captured-graph change is cosine-gated against:

    FLASH_RT_NPU_PI05_REF=<out.npz> python -m pytest \
        tests/test_npu_pi05_model.py -q

The fp32 pass is slow (one full model on CPU); run once and commit the
artifact to a stable store, not per-CI.
"""
import argparse
import pathlib
import sys

from flash_rt.npu import verify
from flash_rt.npu.frontends.torch.pi05 import Pi05TorchFrontendNpu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    args = ap.parse_args()

    fe = Pi05TorchFrontendNpu(args.checkpoint, num_views=2)
    payload = verify.make_reference_actions(fe, args.checkpoint)
    verify.save_reference(pathlib.Path(args.out), payload)
    print("reference raw[:3,:3]:", payload["raw"][:3, :3])


if __name__ == "__main__":
    sys.exit(main())
