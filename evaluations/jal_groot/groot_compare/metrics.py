"""Metrics on physical actions, with explicit dimensions and no hidden pass threshold."""
import numpy as np

ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def action_array(actions):
    """Convert official batch-one action dict to (time, seven), in fixed order."""
    columns = []
    for key in ACTION_KEYS:
        value = np.asarray(actions[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 1:
            raise ValueError(f"Expected {key} shape (1,T,1), got {value.shape}")
        columns.append(value[0])
    result = np.concatenate(columns, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite physical actions")
    return result


def error_metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7 or len(a) == 0:
        raise ValueError(f"Expected matching nonempty (T,7) actions: {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite values in action comparison")
    d = a - b
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    # Zero-norm vectors have no defined direction; do not invent a cosine score.
    cosine = float(np.sum(a * b) / norm) if norm else None
    return {
        "count": len(a), "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt(np.square(d).mean())),
        "mse": float(np.square(d).mean()), "max_abs": float(np.abs(d).max()),
        "cosine": cosine,
        "gripper_disagreement_at_0_5": float(np.mean((a[:, 6] >= .5) != (b[:, 6] >= .5))),
        "per_dimension": {
            key: {"mae": float(np.abs(d[:, i]).mean()),
                  "rmse": float(np.sqrt(np.square(d[:, i]).mean())),
                  "max_abs": float(np.abs(d[:, i]).max())}
            for i, key in enumerate(ACTION_KEYS)
        },
    }
