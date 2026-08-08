"""Qualification gates — parity judgment against a structure's reference.

The harness is structure-agnostic. Implementations follow the structure
calling convention: required boundary inputs in declared order, then
weight tensors in slot order, then variant selections and any optional
boundary inputs as keyword arguments — the same signature the reference
implementation exposes.

A qualification produces a machine-readable record whose ``plan_digest``
binds the spec content, variant, resolved workload dims, thresholds,
implementation identity, and environment. A record certifies exactly one
execution plan; change any component and the record no longer applies.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from flash_rt.structures.registry import StructureSpec, _CATALOG_DIR


@dataclass(frozen=True)
class QualificationCase:
    """One workload to qualify: boundary inputs, weights, and variant."""

    inputs: Mapping[str, torch.Tensor]
    weights: Mapping[str, torch.Tensor]
    variant: Mapping[str, str] = field(default_factory=dict)


def solve_dims(
    spec: StructureSpec,
    inputs: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Resolve symbolic dims from actual tensors, rejecting inconsistency."""
    dims: dict[str, int] = {}

    def bind(declared: list[str], tensor: torch.Tensor, what: str) -> None:
        if tensor.ndim != len(declared):
            raise ValueError(
                f"{what}: expected rank {len(declared)} {declared}, "
                f"got shape {tuple(tensor.shape)}"
            )
        for name, size in zip(declared, tensor.shape):
            if dims.setdefault(name, int(size)) != int(size):
                raise ValueError(
                    f"{what}: dim {name}={int(size)} conflicts with "
                    f"{name}={dims[name]} resolved earlier"
                )

    for entry in spec.boundary["inputs"]:
        tensor = inputs.get(entry["name"])
        if tensor is None:
            if entry.get("optional", False):
                continue
            raise ValueError(f"missing required input: {entry['name']!r}")
        bind(entry["dims"], tensor, f"input {entry['name']!r}")
    for entry in spec.weights:
        slot = entry["slot"]
        if slot not in weights:
            raise ValueError(f"missing weight slot: {slot!r}")
        bind(entry["dims"], weights[slot], f"weight {slot!r}")
    return dims


def _call(spec: StructureSpec, fn: Callable[..., Any],
          case: QualificationCase, *, bound: bool = False) -> torch.Tensor:
    """Invoke ``fn`` per the structure calling convention.

    ``bound=False`` targets full-signature callables (the reference):
    required inputs, then weight slots, with variants and optional inputs
    as keywords. ``bound=True`` targets bound implementations whose
    weights and variant were baked in at bind time: inputs only.
    """
    args: list[torch.Tensor] = []
    kwargs: dict[str, Any] = {} if bound else dict(case.variant)
    for entry in spec.boundary["inputs"]:
        tensor = case.inputs.get(entry["name"])
        if entry.get("optional", False):
            if tensor is not None:
                kwargs[entry["name"]] = tensor
        else:
            args.append(tensor)
    if not bound:
        args.extend(case.weights[slot] for slot in spec.weight_slots)
    return fn(*args, **kwargs)


def parity_metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    """Cosine / max-abs / p99-abs between an implementation and a truth."""
    return _parity_metrics(got, want)


# ---- which metric a host's output deserves ----------------------------
#
# A cosine over the whole output tensor is the right measure for a host
# whose output *is* the answer — an action chunk, a hidden state, a
# feature map. It is the wrong measure for a host whose output is a
# distribution over a vocabulary, and measuring both is what showed why.
# Same bindings, same weights, only the prompt length changed:
#
#     tokens   cosine over all logits   top-1 agreement
#     15       0.9991                   93.3%
#     360      0.9450                   99.4%
#
# The two move in opposite directions with length. Aggregated over every
# position, the cosine is dominated by positions that never drive a
# decision, so it tracks sequence length more than output fidelity, while
# the quantities generation actually depends on — the last position, top-1
# agreement, per-token KL — are stable across both lengths.
#
# So the metric is not a library-wide constant. It belongs to the host's
# output type, and the gate has to select it rather than assume one.

OUTPUT_KINDS = ("values", "distribution")

#: Band edges for the headline accuracy metric. ``>= BAND_PASS`` is clean,
#: ``>= BAND_WARN`` is the recorded WARN band, and below that is ``low``.
#:
#: ``low`` warns; it does not refuse. Low-precision execution is
#: increasingly the intent rather than a defect — a W4A4 or MXFP4 host sits
#: here by design, and a layer that hard-refused at a fixed cosine would be
#: deciding something only the caller can. So the band, the number and the
#: calibration method are reported and said out loud, and whether that is
#: acceptable belongs to the deployment.
BAND_PASS, BAND_WARN = 0.999, 0.995

#: Band edges for distribution outputs (language hosts), judged on token
#: agreement. Cosine-grade edges do not transfer: a language model's
#: headline is "does it pick the same next token", and a clean static
#: W8A8 measured on real text sits at 0.95-0.98 agreement with the
#: per-seam structure gates all passing — that is the honest level of the
#: quantisation, not damage. Damage looks like agreement falling through
#: the floor while seam-level parity stays fine. So: >= 0.95 is ``pass``
#: (the quantisation grade measured when every structure qualifies),
#: >= 0.85 is ``warn``, and below that is ``low``.
DIST_BAND_PASS, DIST_BAND_WARN = 0.95, 0.85

#: no hard accuracy floor by default, for the reason above. Pass ``floors=``
#: to impose one — that is the caller stating a requirement, which is the
#: only place such a number can honestly come from.
DEFAULT_FLOORS: dict[str, dict[str, float]] = {
    "values": {},
    "distribution": {},
}


def infer_output_kind(output: Any) -> str:
    """``"distribution"`` if the host returns logits, else ``"values"``.

    Read off the object the host actually returned rather than guessed
    from its class name: a forward that hands back something carrying
    ``logits`` is scoring a vocabulary, whatever the model is called.
    Callers who know better pass the kind explicitly.
    """
    if output is None:
        return "values"
    if hasattr(output, "logits") and torch.is_tensor(
            getattr(output, "logits")):
        return "distribution"
    if isinstance(output, Mapping) and torch.is_tensor(
            output.get("logits")):
        return "distribution"
    return "values"


def distribution_metrics(got: torch.Tensor,
                         want: torch.Tensor) -> dict[str, float]:
    """Agreement metrics for logits over a vocabulary.

    ``top1_agreement`` is the fraction of positions choosing the same
    token, ``last_position_cosine`` scores the position generation reads,
    and ``kl_per_token`` is summed over the vocabulary and averaged over
    positions — nats per token, not per batch. (``reduction="batchmean"``
    divides by the batch dimension, which for a single sequence is one,
    and reports the whole sequence's KL as if it were one token's.)
    """
    if got.shape != want.shape:
        raise ValueError(
            f"output shape mismatch: impl {tuple(got.shape)} vs "
            f"reference {tuple(want.shape)}")
    got_f, want_f = got.double(), want.double()
    flat_got = got_f.reshape(-1, got_f.shape[-1])
    flat_want = want_f.reshape(-1, want_f.shape[-1])
    top1 = (flat_got.argmax(-1) == flat_want.argmax(-1)).double().mean()
    last = torch.nn.functional.cosine_similarity(
        flat_got[-1], flat_want[-1], dim=0)
    kl = torch.nn.functional.kl_div(
        flat_got.log_softmax(-1), flat_want.log_softmax(-1),
        log_target=True, reduction="none").sum(-1).mean()
    return {
        "top1_agreement": float(top1),
        "last_position_cosine": float(last),
        "kl_per_token": float(kl),
        "max_abs": float((got_f - want_f).abs().max()),
        "positions": int(flat_got.shape[0]),
        # kept as evidence, deliberately not thresholded: this is the
        # number whose length dependence is the reason for this function
        "cosine_all_positions": float(torch.nn.functional.cosine_similarity(
            got_f.flatten(), want_f.flatten(), dim=0)),
    }


def metrics_for(kind: str, got: torch.Tensor,
                want: torch.Tensor) -> dict[str, float]:
    """Score ``got`` against ``want`` the way ``kind`` should be scored."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"unknown output kind: {kind!r} "
                         f"(expected one of {OUTPUT_KINDS})")
    if kind == "distribution":
        return distribution_metrics(got, want)
    return _parity_metrics(got, want)


def passes(metrics: Mapping[str, float],
           floors: Mapping[str, float]) -> tuple[bool, str]:
    """Judge scored metrics against floors; report the first shortfall.

    ``cosine``-like names and agreement fractions are floors; anything
    named for an error or a divergence is a ceiling. Naming the metric
    that failed is the point — "refused" with no number attached is how a
    refusal turns into folklore.
    """
    for name, floor in floors.items():
        if name not in metrics:
            raise ValueError(
                f"floor on a metric that was not measured: {name!r} "
                f"(have {sorted(metrics)})")
        value = metrics[name]
        ceiling = any(tok in name for tok in ("abs", "kl", "err"))
        if (value > floor) if ceiling else (value < floor):
            return False, (f"{name}={value:.6f} "
                           f"{'above' if ceiling else 'below'} {floor}")
    return True, ""


def headline(kind: str) -> str:
    """The metric a band is read off for this output kind."""
    return "top1_agreement" if kind == "distribution" else "cosine"


def band_of(metrics: Mapping[str, float], kind: str) -> str:
    """``pass`` / ``warn`` / ``low`` for the headline metric of ``kind``.

    None of the three is a refusal. A static per-tensor scale calibrated
    from a handful of frames gives a workload's parity, not a host's, and
    at four-bit weights the honest number is simply lower — so the band is
    recorded next to the calibration method and the sample count rather
    than collapsed into a yes or no. ``low`` is the band a caller should
    look at before deploying, not one this layer rejects for them.
    """
    value = metrics.get(headline(kind))
    if value is None:
        return "unknown"
    hi, lo = ((DIST_BAND_PASS, DIST_BAND_WARN)
              if kind == "distribution" else (BAND_PASS, BAND_WARN))
    if value >= hi:
        return "pass"
    return "warn" if value >= lo else "low"


def band_note(metrics: Mapping[str, float], kind: str,
              calibration: Mapping[str, Any] | None = None) -> str:
    """One line a caller can act on: the band, the number, how it was got."""
    key = headline(kind)
    value = metrics.get(key)
    worst = metrics.get("max_abs")
    parts = [f"band {band_of(metrics, kind)}",
             f"{key}={value:.6f}" if value is not None else f"{key}=n/a"]
    if worst is not None:
        parts.append(f"max_abs={worst:.4g}")
    if calibration:
        parts.append(
            f"from {calibration.get('samples')} sample(s), "
            f"{calibration.get('method')}")
    return ", ".join(parts)


def _parity_metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    if got.shape != want.shape:
        raise ValueError(
            f"output shape mismatch: impl {tuple(got.shape)} vs "
            f"reference {tuple(want.shape)}"
        )
    diff = (got.double() - want.double()).abs().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        got.double().flatten(), want.double().flatten(), dim=0
    )
    # kthvalue instead of quantile: exact and free of quantile's input
    # size limit (qualification outputs can exceed it, e.g. LLM logits)
    k = max(1, int(0.99 * diff.numel()))
    return {
        "cosine": float(cosine),
        "max_abs": float(diff.max()),
        "p99_abs": float(diff.kthvalue(k).values),
    }


def _spec_digest(spec: StructureSpec) -> str:
    path = _CATALOG_DIR / spec.name / "structure.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, str]:
    env = {"torch": torch.__version__}
    if torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        env["arch"] = f"sm_{major}{minor}"
    else:
        env["device"] = "cpu"
    return env


def qualify_parity(
    spec: StructureSpec,
    impl: Callable[..., Any],
    case: QualificationCase,
    *,
    impl_id: str,
    thresholds: Mapping[str, float],
    bound: bool = False,
) -> dict[str, Any]:
    """Judge ``impl`` against the structure reference on one workload.

    ``thresholds`` maps metric name to its passing bound (``cosine`` is a
    floor, absolute-error metrics are ceilings). Every thresholded metric
    must pass for a PASS verdict; metrics without thresholds are recorded
    as evidence only. Set ``bound=True`` when ``impl`` was produced by an
    implementation's ``bind`` and takes boundary inputs only.
    """
    for key in case.variant:
        if key not in spec.variants:
            raise ValueError(f"unknown variant key: {key!r}")
    workload = solve_dims(spec, case.inputs, case.weights)
    reference = spec.reference()

    want = _call(spec, reference, case)
    got = _call(spec, impl, case, bound=bound)
    metrics = _parity_metrics(got, want)

    passed = True
    for name, bound in thresholds.items():
        if name not in metrics:
            raise ValueError(f"threshold on unknown metric: {name!r}")
        ok = metrics[name] >= bound if name == "cosine" else metrics[name] <= bound
        passed = passed and ok

    x_dtype = next(iter(case.inputs.values())).dtype
    record = {
        "structure": f"{spec.name}@{spec.version}",
        "spec_digest": _spec_digest(spec),
        "impl": impl_id,
        "variant": dict(case.variant),
        "workload": {**workload, "dtype": str(x_dtype)},
        "env": _environment(),
        "gate": "parity",
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "verdict": "PASS" if passed else "FAIL",
    }
    record["plan_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            {k: record[k] for k in
             ("structure", "spec_digest", "impl", "variant", "workload",
              "env", "thresholds")},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return record


def env_lock() -> dict[str, Any]:
    """The environment a receipt was earned in, reconstructible.

    A receipt without its environment cannot be re-run: a silent torch
    downgrade broke two GROOT receipts before this existed. The lock
    carries the exact versions of the packages that decide numerics,
    plus a digest over the full installed set — enough to detect any
    drift, small enough to live in every record.
    """
    import hashlib as _hashlib
    import importlib.metadata as _md
    import platform as _platform

    key = {}
    for pkg in ("torch", "transformers", "diffusers", "kernels",
                "compressed-tensors", "safetensors", "numpy"):
        try:
            key[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:
            pass
    frozen = "\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in _md.distributions() if d.metadata["Name"]))
    lock = {
        "python": _platform.python_version(),
        "packages": key,
        "pip_freeze_sha256": _hashlib.sha256(
            frozen.encode("utf-8")).hexdigest(),
    }
    try:
        import torch as _torch

        lock["cuda"] = _torch.version.cuda
        if _torch.cuda.is_available():
            lock["device"] = _torch.cuda.get_device_name(0)
    except Exception:
        pass
    return lock


def verify_record(record: Mapping[str, Any]) -> bool:
    """Recompute a record's digest; ``False`` means tampered or torn.

    Two digest recipes exist in the wild: qualification records digest
    a fixed key subset, probe records digest the whole record as it
    stood before the digest (and before the env lock) was added. A
    record verifying under either recipe is intact."""
    stated = str(record.get("plan_digest", ""))
    if not stated.startswith("sha256:"):
        return False
    body = {k: v for k, v in record.items()
            if k not in ("plan_digest", "env_lock")}
    whole = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    if whole == stated:
        return True
    subset_keys = ("structure", "spec_digest", "impl", "variant",
                   "workload", "env", "thresholds")
    if all(k in record for k in subset_keys):
        subset = "sha256:" + hashlib.sha256(
            json.dumps({k: record[k] for k in subset_keys},
                       sort_keys=True).encode("utf-8")).hexdigest()
        if subset == stated:
            return True
    return False


def check_env(record: Mapping[str, Any]) -> list[str]:
    """Name every way the current environment drifts from a receipt's.

    Empty list = re-runnable as-is. A receipt without a lock is itself
    a finding."""
    lock = record.get("env_lock")
    if not lock:
        return ["record carries no env_lock"]
    now = env_lock()
    drift = []
    for pkg, ver in (lock.get("packages") or {}).items():
        cur = now["packages"].get(pkg)
        if cur != ver:
            drift.append(f"{pkg}: receipt {ver}, current {cur}")
    if lock.get("python") != now["python"]:
        drift.append(f"python: receipt {lock.get('python')}, "
                     f"current {now['python']}")
    if (lock.get("pip_freeze_sha256") != now["pip_freeze_sha256"]
            and not drift):
        drift.append("installed set differs (freeze digest mismatch)")
    return drift


def save_record(record: Mapping[str, Any], directory: str | pathlib.Path) -> pathlib.Path:
    """Write one qualification record as JSON, named by its plan digest.

    Every record is stamped with the environment lock unless the caller
    already supplied one."""
    if "env_lock" not in record:
        record = {**record, "env_lock": env_lock()}
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = record["plan_digest"].split(":", 1)[1][:16]
    path = directory / f"{record['gate']}_{digest}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
