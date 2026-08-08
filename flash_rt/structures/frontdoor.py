"""One-call front door: ``structures.attach(model, forward, ...)``.

The consumption contract mirrors ``kernels.get_kernel``: one import, one
call. Everything the structure layer needs — seam discovery, real
distribution calibration, accuracy and net-win gates, transactional swap,
receipt — runs inside the call. An attachment that does not both stay
accurate and win latency is refused; the model is left untouched and the
refusal is reported, never silently absorbed.

Three things about this gate are deliberate, and each replaced something
that used to be assumed:

**It gates what ``auto_swaps`` binds, not a subset of it.** Binding lives
in one place (:mod:`.autobuild`) and this module only judges. The earlier
arrangement had its own binding path covering three structures, so the
one call a caller was told to make was the one call that judged the least.

**The accuracy metric follows the host's output type.** A cosine over a
whole logits tensor falls as the sequence grows while token agreement
rises (see :mod:`.gates`); scoring a language host that way measures
sequence length. Distribution outputs are judged on top-1 agreement and
the last position, value outputs on cosine.

**Latency is judged by paired alternating timing.** Timing one arm and
then the other attributes machine drift to whichever arm it landed on;
this stack has produced an 8% spread between runs of the same
configuration, which is four times the margin the gate is asked to
resolve. Both arms are timed in every round and the decision is the
median of the per-round ratios.

And one thing it checks that nothing used to: after each scoring forward
it reads the attachment's ledger. A family whose seams fell back to the
host module did not run, however good its parity looks — that parity is
the host's own.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .autobuild import AutoPlan, _layer_of, auto_swaps
from .gates import (DEFAULT_FLOORS, band_note, band_of, infer_output_kind,
                    metrics_for, passes)
from .guard import GuardRefused
from .swap import AttachHandle as _AttachHandle, attach as _swap_attach

#: the full catalog, which is what "one call" has to mean
ALL_STRUCTURES = ("decoder_ffn", "vision_ffn", "qkv_pack", "adaln_producer",
                  "linear_proj", "patch_projection", "norm_fused",
                  "attention_core",
                  "decoder_block", "modnorm_qkv_chain", "qk_norm_rope",
                  "qkv_rope", "gated_delta_core")

#: structure name per implementation class, for swaps whose path is not
#: itself a discovered seam (a pack's sibling readers, a composed block's
#: absorbed children). The type of the module that was bound is a local
#: fact; inferring it from the path would be a naming guess.
_STRUCTURE_BY_IMPL = {
    "FusedGeGluMlp": "decoder_ffn", "FusedGluMlpW8A16": "decoder_ffn",
    "FusedGeluMlp": "vision_ffn", "FusedLinearProj": "linear_proj",
    "FlatPatchProjection": "patch_projection",
    "PackedLinear": "qkv_pack", "StashReader": "qkv_pack",
    "AttnBlockPacked": "qkv_pack", "AdaLNProducer": "adaln_producer",
    "StyleTable": "adaln_producer", "FusedNorm": "norm_fused",
    "FusedDecoderBlock": "decoder_block", "StaticOutput": "cadence_static",
}

#: a negotiated fp8 chain is one gate unit. The producer emits fp8 under a
#: scale the consumer was bound for, so attaching one without the other
#: hands the consumer a dtype it refuses — which the ledger would report
#: as a family that fell back, from a split this gate created itself.
_CHAIN = "negotiated_fp8_chain"
_ROUTED = "attention_core_routed"


def _cuda_time_ms(fn: Callable[[], Any], warmup: int = 3,
                  iters: int = 10) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if not torch.cuda.is_available():
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) * 1e3 / iters
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters


def _paired_ab(thunk: Callable[[], Any], on: Callable[[], Any],
               off: Callable[[], Any], *, rounds: int = 5,
               iters: int = 10) -> dict[str, float]:
    """Time both arms in every round; decide on the median paired ratio.

    Drift on this machine lands on whichever arm happens to be running,
    so an A-then-B measurement attributes it to that arm. Alternating and
    pairing cancels it: every ratio comes from two measurements taken
    seconds apart, and the spread across rounds says how much to trust
    the answer rather than leaving it to be assumed.
    """
    rows: list[tuple[float, float]] = []
    for _ in range(max(1, rounds)):
        on()
        treated = _cuda_time_ms(thunk, warmup=1, iters=iters)
        off()
        base = _cuda_time_ms(thunk, warmup=1, iters=iters)
        rows.append((treated, base))
    ratios = sorted(b / t for t, b in rows)
    return {
        "ms": round(statistics.median(t for t, _ in rows), 3),
        "base_ms": round(statistics.median(b for _, b in rows), 3),
        "speedup": round(statistics.median(ratios), 4),
        "speedup_min": round(ratios[0], 4),
        "speedup_max": round(ratios[-1], 4),
        "spread": round(ratios[-1] - ratios[0], 4),
        "rounds": len(rows),
    }


def _score_tensor(value: Any) -> torch.Tensor | None:
    """The tensor a host's output should be judged on.

    Logits when the host produces them, otherwise the first tensor in a
    deterministic walk. Mappings are walked in sorted key order so the two
    arms of a comparison never pick different leaves.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    logits = getattr(value, "logits", None)
    if torch.is_tensor(logits):
        return logits
    if isinstance(value, Mapping):
        if torch.is_tensor(value.get("logits")):
            return value["logits"]
        for key in sorted(value):
            found = _score_tensor(value[key])
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _score_tensor(item)
            if found is not None:
                return found
    return None


def _gate_groups(plan: AutoPlan) -> dict[str, dict[str, torch.nn.Module]]:
    """Split the plan into units that can be judged independently."""
    structure_of = {s.path: s.structure for s in plan.seams}
    negotiated = set(plan.notes.get("negotiated_layers", ()))
    groups: dict[str, dict[str, torch.nn.Module]] = {}
    for path, module in plan.swaps.items():
        if negotiated and _layer_of(path) in negotiated:
            key = _CHAIN
        else:
            key = (structure_of.get(path)
                   or _STRUCTURE_BY_IMPL.get(type(module).__name__)
                   or "other")
        groups.setdefault(key, {})[path] = module
    return groups


class _Arm:
    """One gate unit, switchable on and off for the paired timing loop.

    Holds a single handle rather than one per switch: attaching twice over
    the same paths would record the first replacement as the "original",
    and detaching would then restore a structure instead of the host.
    Idempotent on both sides so the timing loop can call them freely.
    """

    def __init__(self, model: torch.nn.Module,
                 swaps: Mapping[str, torch.nn.Module], plan: AutoPlan,
                 mode: str, *, routed: bool = False,
                 revert: Any = None) -> None:
        self.model = model
        self.swaps = dict(swaps)
        self.plan = plan
        self.mode = mode
        self.routed = routed
        self.revert = revert
        self.handle: _AttachHandle | None = None

    def on(self) -> None:
        if self.handle is None:
            self.handle = _swap_attach(
                self.model, self.swaps,
                observe=self.plan.observed if self.routed else None,
                on_guard_fail=self.mode, revert=self.revert)
        if self.routed:
            self.plan.enable_routed()

    def off(self) -> None:
        if self.handle is not None:
            self.handle.detach()
            self.handle = None
        if self.routed:
            self.plan.disable_routed()


@dataclass
class Plan:
    """Result of one ``attach`` call: what was activated, why, evidence."""

    activated: dict[str, torch.nn.Module]
    families: dict[str, dict[str, Any]]
    receipt: dict[str, Any]
    _handle: _AttachHandle | None = None
    _plan: AutoPlan | None = None
    active: bool = field(init=False)

    def __post_init__(self) -> None:
        self.active = self._handle is not None

    def report(self) -> str:
        lines = [f"structures.attach: {len(self.activated)} seam(s) active"]
        for name, stat in self.families.items():
            line = (f"  {name}: {stat['seams']} seam(s) -> "
                    f"{stat['outcome']}")
            if stat.get("metrics"):
                line += f" [{stat.get('band', '?')}]"
            if stat["outcome"] == "refused":
                line += f" ({stat.get('reason', '')})"
            lines.append(line)
        e2e = self.receipt.get("e2e")
        if e2e:
            lines.append(
                f"  e2e: {e2e['base_ms']:.2f} -> {e2e['ms']:.2f} ms "
                f"({e2e['speedup']:.3f}x, spread {e2e['spread']:.3f})")
        led = self.receipt.get("ledger")
        if led:
            lines.append(f"  ledger: {led['fallbacks']} fallback(s) over "
                         f"{led['guarded_calls']} guarded call(s)"
                         + ("" if led["clean"] else
                            f" — {led['seams_fell_back']}"))
        return "\n".join(lines)

    def ledger(self) -> dict[str, Any]:
        """The live per-seam ledger of the committed attachment."""
        return {} if self._handle is None else self._handle.report()

    def detach(self) -> None:
        """Restore the host, including seams that are not modules."""
        if self._handle is not None:
            self._handle.detach()
        elif self._plan is not None:
            self._plan.revert_all()
        self.active = False

    def save_receipt(self, directory: str | pathlib.Path) -> pathlib.Path:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"attach_{self.receipt['digest'][:12]}.json"
        path.write_text(json.dumps(self.receipt, indent=2, default=str))
        return path


def attach(
    model: torch.nn.Module,
    forward: Callable[[], Any] | Sequence[Callable[[], Any]],
    *,
    structures: tuple[str, ...] = ALL_STRUCTURES,
    observations: Iterable[Any] | None = None,
    prefix_cadence: bool = False,
    percentile: float = 99.9,
    max_samples: int | None = None,
    output: Callable[[], Any] | None = None,
    output_kind: str = "auto",
    floors: Mapping[str, float] | None = None,
    min_speedup: float = 1.02,
    rounds: int = 5,
    iters: int = 10,
    on_guard_fail: str = "fallback",
    scheme: str | Any = "auto",
    verbose: bool = True,
) -> Plan:
    """Discover, calibrate, gate and activate structures in one call.

    ``forward`` runs the host once; ``observations`` / ``percentile`` /
    ``max_samples`` are the repo's calibration arguments and mean exactly
    what they mean in ``flash_rt.api.FlashRT.calibrate``.

    ``scheme`` selects the precision profile by registered name
    (:mod:`.schemes`); the default ``"auto"`` resolves to the fastest
    profile the device can execute, and ``"none"`` is the explicit
    quantisation off-switch.

    ``output_kind`` picks how accuracy is measured — ``"values"`` for a
    host whose output is the answer, ``"distribution"`` for one whose
    output scores a vocabulary, ``"auto"`` to read it off what the host
    returns. ``floors`` overrides the per-kind parity floors.

    Refusal is per gate unit and per form: a unit that is refused is
    refused *in the form it was measured, at the shape it was measured
    at*, and the receipt records both.
    """

    def say(msg: str) -> None:
        if verbose:
            print(f"[structures] {msg}", flush=True)

    thunks = (list(forward) if isinstance(forward, (list, tuple))
              else [forward])
    eval_thunk = thunks[-1]

    # ---- reference: the host as it shipped, before anything is built ---
    with torch.no_grad():
        base_out = eval_thunk()
    kind = (infer_output_kind(base_out) if output_kind == "auto"
            else output_kind)
    get_out = output or eval_thunk
    want = _score_tensor(base_out)
    want = None if want is None else want.detach().float().cpu()
    # Some generation-style model outputs retain a full KV cache alongside
    # logits. Only the scored CPU tensor is needed after this point; keeping
    # the host output alive through candidate binding can consume the memory
    # needed by a reversible FP8 arm on near-capacity models.
    del base_out
    say(f"host output scored as {kind!r}"
        + ("" if want is not None else " (no tensor found — accuracy gate "
           "cannot run, latency gate only)"))
    band_floors = dict(floors or DEFAULT_FLOORS[kind])

    # ---- bind: one path, the same one the plain call uses --------------
    plan = auto_swaps(model, forward, structures=structures,
                      observations=observations, percentile=percentile,
                      max_samples=max_samples, prefix_cadence=prefix_cadence,
                      scheme=scheme, verbose=verbose)
    if not plan.swaps and not plan.toggles:
        plan.revert_all()
        return Plan({}, {}, {"digest": "none", "seams": 0,
                             "output_kind": kind}, _plan=plan)

    calibration = plan.notes.get("calibration") or {}
    # Adapters build their routed seam enabled so plain ``auto_swaps`` can
    # be consumed directly. The front door must start its A/B gate from the
    # untouched host and judge that routed seam as its own unit.
    plan.disable_routed()
    groups = _gate_groups(plan)
    if plan.toggles:
        groups[_ROUTED] = {}
    bound_count = len(plan.swaps) + len(plan.observed)
    say(f"{bound_count} bound seam(s) in {len(groups)} gate unit(s): "
        + ", ".join(
            f"{k}×{len(plan.observed) if k == _ROUTED else len(v)}"
            for k, v in sorted(groups.items())))

    def scored() -> torch.Tensor | None:
        with torch.no_grad():
            got = _score_tensor(get_out())
        return None if got is None else got.detach().float().cpu()

    # ---- per unit: accuracy, then that it ran, then net win -----------
    stats: dict[str, dict[str, Any]] = {}
    winners: dict[str, torch.nn.Module] = {}
    routed_winner = False
    for name, swaps in sorted(groups.items()):
        routed = name == _ROUTED
        paths = plan.observed if routed else swaps
        stat: dict[str, Any] = {"seams": len(paths),
                                "paths": sorted(paths)[:4],
                                "outcome": "pending"}
        stats[name] = stat
        arm = _Arm(model, swaps, plan, on_guard_fail, routed=routed)
        try:
            arm.on()
            if want is not None:
                metrics = metrics_for(kind, scored(), want)
                stat["metrics"] = _round(metrics)
                stat["band"] = band_of(metrics, kind)
                stat["band_note"] = band_note(metrics, kind, calibration)
                _say_band(name, stat["band"], stat["band_note"], say)
                ok, why = passes(metrics, band_floors)
                if not ok:
                    stat["outcome"] = "refused"
                    stat["reason"] = f"{why} (caller floor)"
                    continue
            # read before the timing loop: it is this unit's own scoring
            # forward that the accuracy number came from
            led = arm.handle.summary()
            stat["ledger"] = led
            if not led["clean"]:
                # the parity above looked fine because the host computed it
                stat["outcome"] = "refused"
                stat["reason"] = (
                    f"{len(led['seams_fell_back'])} seam(s) fell back to "
                    f"the host module, so this unit did not run: "
                    f"{led['seams_fell_back'][:3]}")
                continue
            timing = _paired_ab(eval_thunk, arm.on, arm.off,
                                rounds=rounds, iters=iters)
            stat["e2e"] = timing
            if timing["speedup"] < min_speedup:
                stat["outcome"] = "refused"
                stat["reason"] = (
                    f"no net win ({timing['speedup']:.3f}x, spread "
                    f"{timing['spread']:.3f}) at {_shape_note(plan)}")
                continue
            stat["outcome"] = "activated"
            if routed:
                routed_winner = True
            else:
                winners.update(swaps)
        except GuardRefused as refusal:
            stat["outcome"] = "refused"
            stat["reason"] = f"runtime form refused: {refusal}"
        finally:
            arm.off()
        say(f"{name}: {stat['outcome']}"
            + (f" ({stat.get('reason')})" if stat.get("reason") else
               f" {stat['e2e']['speedup']:.3f}x, band {stat.get('band')}"))

    # ---- union re-check, then commit ---------------------------------
    e2e_final: dict[str, Any] | None = None
    ledger_final: dict[str, Any] | None = None
    handle = None
    if winners or routed_winner:
        arm = _Arm(model, winners, plan, on_guard_fail,
                   routed=routed_winner, revert=plan.revert)
        reason = ""
        try:
            arm.on()
            metrics = (metrics_for(kind, scored(), want)
                       if want is not None else {})
            ok, why = (
                passes(metrics, band_floors) if metrics else (True, ""))
            ledger_final = arm.handle.summary()
            timing = _paired_ab(eval_thunk, arm.on, arm.off,
                                rounds=rounds, iters=iters)
            arm.on()                 # the loop ends on the off arm
            handle = arm.handle
            if not ok or not ledger_final["clean"] \
                    or timing["speedup"] < min_speedup:
                reason = (why if not ok
                          else "seams fell back" if not ledger_final["clean"]
                          else f"no net win ({timing['speedup']:.3f}x)")
        except GuardRefused as refusal:
            reason = f"runtime form refused: {refusal}"
        if reason:
            arm.off()                # also reverts the routed seams
            handle, winners, routed_winner = None, {}, False
            for stat in stats.values():
                if stat["outcome"] == "activated":
                    stat["outcome"] = "refused"
                    stat["reason"] = f"union: {reason}"
            say(f"union of activated units refused — {reason}; host "
                "restored, including the routed seams")
        else:
            e2e_final = timing
            e2e_final["metrics"] = _round(metrics)
            e2e_final["band"] = band_of(metrics, kind) if metrics else "n/a"
            if metrics:
                e2e_final["band_note"] = band_note(metrics, kind, calibration)
                _say_band("e2e", e2e_final["band"],
                          e2e_final["band_note"], say)
            say(f"active: {len(winners)} seam(s), "
                f"{timing['base_ms']:.2f} -> {timing['ms']:.2f} ms "
                f"({timing['speedup']:.3f}x, spread {timing['spread']:.3f})"
                f", band {e2e_final['band']}")
    if not winners and not routed_winner:
        plan.revert_all()
        say("outcome: whole-host refusal — model left untouched")

    activated = dict(winners)
    if routed_winner:
        activated.update(plan.observed)
    receipt = {
        "schema_version": 1,
        "environment": _environment(),
        "model": type(model).__name__,
        "output_kind": kind,
        "scheme": plan.notes.get("scheme"),
        "floors": band_floors,
        "min_speedup": min_speedup,
        "timing": {"method": "paired alternating", "rounds": rounds,
                   "iters": iters},
        "calibration": plan.notes.get("calibration"),
        "assumed": plan.notes.get("assumed", []),
        "refused_at_bind": plan.notes.get("refused", []),
        "negotiated_layers": plan.notes.get("negotiated_layers", []),
        "units": stats,
        "e2e": e2e_final,
        "ledger": ledger_final,
        "seams_active": sorted(activated),
    }
    receipt["digest"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, default=str).encode()
    ).hexdigest()
    return Plan(activated, stats, receipt, _handle=handle, _plan=plan)


def _environment() -> dict[str, str]:
    """Version fingerprint baked into every receipt.

    A parity or latency figure without the environment it was measured
    in is not comparable to anything. The concrete case: a host
    modelling contract that moved between two library versions changed
    every activation scale while both arms of an A/B stayed mutually
    consistent — two receipts differing only in this block is exactly
    how that shows up.
    """
    import platform

    env = {"python": platform.python_version(),
           "torch": torch.__version__}
    if torch.cuda.is_available():
        env["cuda"] = str(torch.version.cuda)
        env["device"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        env["sm"] = f"sm{cap[0]}{cap[1]}"
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def _say_band(where: str, band: str, note: str, say) -> None:
    """Report the band; say it out loud when it is the one to look at.

    A ``low`` band is not a refusal — see :mod:`.gates`. It is the caller's
    call, so it has to reach the caller rather than sit in a receipt nobody
    opens.
    """
    say(f"{where}: {note}")
    if band == "low":
        warnings.warn(
            f"structures: {where} is in the low accuracy band — {note}. "
            "This is reported, not refused: whether it is acceptable "
            "depends on the deployment. Pass floors={...} to make it a "
            "hard requirement, or widen the calibration set.",
            RuntimeWarning, stacklevel=3)


def _round(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 7) if isinstance(v, float) else v)
            for k, v in metrics.items()}


def _shape_note(plan: AutoPlan) -> str:
    """The workload a refusal was measured at, for the receipt.

    A refusal with no shape attached turns into "that structure does not
    work here"; the shape is what makes it "not at this size".
    """
    rows = sorted({m for s in plan.seams for m in (s.m_profile or ())})
    return f"rows={rows}" if rows else "rows unrecorded"
