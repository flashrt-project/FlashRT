"""Numerical checks shared by the Thor examples."""
from pathlib import Path
import json
import numpy as np


def metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"Action shapes differ: {actual.shape} / {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Actions contain NaN or Inf")
    a, b = actual.reshape(len(actual), -1), reference.reshape(len(reference), -1)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if (norm == 0).any():
        raise ValueError("Cosine is undefined for a zero-norm sample")
    cosine = (a * b).sum(axis=1) / norm
    error = actual - reference
    return dict(mean_cosine=float(cosine.mean()), worst_cosine=float(cosine.min()),
                max_abs=float(np.abs(error).max()), rmse=float(np.sqrt((error * error).mean())))


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def load_groot_input(path):
    with np.load(path, allow_pickle=False) as data:
        schema = json.loads(str(data["schema"]))
        return {group: {key: (data[f"{group}/{key}"].tolist() if group == "language"
                      else data[f"{group}/{key}"]) for key in keys}
                for group, keys in schema.items()}
