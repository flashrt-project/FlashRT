#!/usr/bin/env python3
"""Compare matched fixed-input/fixed-noise final-action records, including gripper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_rt.models.dm05.validation import compare_actions


def compare_records(reference, candidate, **thresholds):
    # Required provenance prevents a plausible score on reordered frames or
    # different initial diffusion noise. Record values are hashes/IDs, not paths.
    for key in ("sample_ids", "input_sha256", "noise_sha256", "action_space"):
        if key not in reference or key not in candidate:
            raise ValueError(f"both records require {key}")
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"record mismatch: {key}")
    if np.asarray(reference["sample_ids"]).shape != (len(reference["actions"]),):
        raise ValueError("sample_ids must contain one ID per observation")
    return compare_actions(reference["actions"], candidate["actions"], **thresholds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--gripper-threshold", type=float, required=True,
                        help="actual downstream open/close threshold; decision is value > threshold")
    parser.add_argument("--arm-cosine-min", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with np.load(args.reference, allow_pickle=False) as reference, np.load(args.candidate, allow_pickle=False) as candidate:
        report = compare_records(reference, candidate, atol=args.atol, rtol=args.rtol,
                     gripper_threshold=args.gripper_threshold, arm_cosine_min=args.arm_cosine_min)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
