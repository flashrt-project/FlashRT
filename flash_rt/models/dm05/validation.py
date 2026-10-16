"""Tolerance and gripper-decision checks on final, unnormalized DM05 actions."""
from __future__ import annotations

import numpy as np


def compare_actions(reference, candidate, *, atol, gripper_threshold, rtol=0.0,
                    arm_cosine_min=0.99):
    """Check every action component and every gripper decision.

    Arrays are (observations, chunk, 7), in the same physical/output action
    space, ordered as six arm components followed by the gripper. The caller
    supplies tolerances and the robot consumer's gripper threshold explicitly.
    A high arm cosine never exempts a gripper mismatch.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[-1] != 7:
        raise ValueError("DM05 action arrays must have equal shape (observations, chunk, 7)")
    if not reference.shape[0] or not reference.shape[1]:
        raise ValueError("DM05 action arrays must not be empty")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("DM05 action arrays must be finite")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(gripper_threshold) or not -1 <= arm_cosine_min <= 1:
        raise ValueError("invalid gripper threshold or arm cosine threshold")
    delta = np.abs(candidate - reference)
    close = delta <= atol + rtol * np.abs(reference)
    a = reference[..., :6].reshape(reference.shape[0], -1)
    b = candidate[..., :6].reshape(candidate.shape[0], -1)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    denom = an * bn
    cosine = np.divide(np.sum(a*b, axis=-1), denom, out=np.zeros_like(denom), where=denom > 0)
    # Cosine is undefined for a zero vector. Treat two sufficiently close vectors
    # as agreeing; one zero and one nonzero still fails unless within tolerance.
    cosine = np.where((denom == 0) & close[..., :6].all(axis=(1, 2)), 1.0, cosine)
    cosine = np.clip(cosine, -1.0, 1.0)
    flips = (reference[..., 6] > gripper_threshold) != (candidate[..., 6] > gripper_threshold)
    bad_numeric = np.argwhere(~close)
    bad_decisions = np.argwhere(flips)
    return {
        "passed": bool(close.all() and (cosine >= arm_cosine_min).all() and not flips.any()),
        "observations": int(reference.shape[0]), "chunk": int(reference.shape[1]),
        "atol": float(atol), "rtol": float(rtol), "gripper_threshold": float(gripper_threshold),
        "arm_cosine_min": float(cosine.min()), "arm_cosine_required": float(arm_cosine_min),
        "max_abs_by_component": delta.max(axis=(0, 1)).tolist(),
        "mean_abs_by_component": delta.mean(axis=(0, 1)).tolist(),
        "numeric_mismatch_count": int(len(bad_numeric)),
        "numeric_mismatch_indices": bad_numeric.tolist(),
        "gripper_flip_count": int(flips.sum()),
        "gripper_flip_indices": bad_decisions.tolist(),
        "gripper_decisions": [{"observation": int(i), "step": int(j),
            "reference": float(reference[i,j,6]), "candidate": float(candidate[i,j,6])}
            for i, j in bad_decisions],
    }
