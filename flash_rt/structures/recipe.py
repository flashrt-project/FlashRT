"""Declarative e2e recipes: assemble levers, audit on the graph, certify.

A recipe declares how a host pipeline eats structure yield end to end:
which levers engage (region families, cadence pieces, seam negotiations,
dtype changes), how the hot stage is built, and under which explicit
gates the result is judged. ``run_recipe`` assembles the levers
transactionally, audits baseline vs treated **in the same process**
(cross-run anchors carry sub-millisecond drift, so sub-2% verdicts are
unreliable across runs), and emits a receipt recording every switch
state, every gate number, and every refusal reason.

Switch discipline (the lever lifecycle):

  off         declared but not engaged; recorded, never built
  candidate   engaged and audited this run; a refused lever stays a
              candidate — refusal is an outcome, not an error
  certified   won a same-process audit under the current plan digest;
              any digest change (shapes, weights, environment, gates)
              demotes it back to candidate for re-audit — certification
              never outlives the plan it was earned on

Gate thresholds are switches too: they are folded into the plan digest
and written into the receipt, so a relaxed gate is always visible and
invalidates prior certifications instead of silently inheriting them.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from .gates import parity_metrics
from .swap import AttachHandle, attach as _swap_attach

_LEVER_KINDS = ("regions", "cadence", "seam_negotiation", "dtype",
                "capture_form")


@dataclass
class Gates:
    """Explicit gate thresholds. Part of the plan digest: changing one
    is a plan change, not a tweak.

    ``parity_cos`` is the hard floor — below it the recipe refuses.
    ``parity_warn`` marks the comfort line: results in
    ``[parity_cos, parity_warn)`` pass but the receipt records
    ``parity_band: "warn"`` with a note that calibration owes the
    remaining accuracy. During the performance-assembly phase the floor
    is intentionally loose (collapse-only); tighter banded gates arrive
    with the calibration/fallback design, not before.
    """

    parity_cos: float = 0.99
    parity_warn: float = 0.999
    min_speedup: float = 1.02
    drift_budget: float = 0.02

    def as_dict(self) -> dict[str, float]:
        return {"parity_cos": self.parity_cos,
                "parity_warn": self.parity_warn,
                "min_speedup": self.min_speedup,
                "drift_budget": self.drift_budget}


@dataclass
class Lever:
    """One named yield switch: a group of swaps engaged and judged
    together.

    ``build(model, ctx)`` returns either a mapping ``path -> module`` of
    swaps, or a ``(swaps, outside_update)`` tuple where
    ``outside_update`` is a callable the host must run at the lever's
    cadence outside the captured stage (e.g. refreshing static KV
    buffers on observation ticks). Outside updates are collected into
    ``ctx["outside_updates"]`` before ``build_stage`` runs.
    """

    name: str
    kind: str
    build: Callable[[torch.nn.Module, dict], Any] | None = None
    state: str = "candidate"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _LEVER_KINDS:
            raise ValueError(f"lever kind {self.kind!r} not in "
                             f"{_LEVER_KINDS}")
        if self.state not in ("off", "candidate", "certified"):
            raise ValueError(f"lever state {self.state!r}")


@dataclass
class Arm:
    """One executable form of the pipeline under audit.

    ``refs`` must pin every object whose device memory the arm's graph
    reads (input buffers, static intermediates, closures holding them).
    A captured graph keeps no Python references of its own: if a tensor
    it reads is garbage-collected, the next arm's allocations reuse that
    memory and this arm replays over foreign data — the same-process
    baseline retime is exactly where that corruption surfaces.
    """

    tick: Callable[[], Any]
    output: Callable[[], torch.Tensor]
    teardown: Callable[[], None] | None = None
    refs: Any = None
    stage: Any = None    # the CapturedStage, if the arm graphed one —
                         # lets the winning arm feed stage.export()


@dataclass
class Recipe:
    """The declaration ``run_recipe`` executes.

    ``build_stage(model, ctx)`` constructs the hot stage for whatever is
    currently attached to the model (it is called once for the untouched
    baseline and once with the levers engaged) and returns an
    :class:`Arm`. ``reference(model, ctx)`` produces the stock eager
    output that anchors parity. ``between_arms`` runs after the baseline
    arm is built and before levers attach — compile hosts pass
    ``torch._dynamo.reset`` here so the treated arm gets a fresh
    compile budget instead of inheriting cache-limit fallout.
    """

    name: str
    levers: list[Lever]
    build_stage: Callable[[torch.nn.Module, dict], Arm]
    reference: Callable[[torch.nn.Module, dict], torch.Tensor]
    gates: Gates = field(default_factory=Gates)
    between_arms: Callable[[], None] | None = None


@dataclass
class RecipeRun:
    """Outcome of one audit: verdict, evidence, and the live stage."""

    verdict: str
    receipt: dict[str, Any]
    arm: Arm | None = None
    _handle: AttachHandle | None = None

    def report(self) -> str:
        r = self.receipt
        lines = [f"recipe {r['recipe']}: {self.verdict}"
                 + (f" [{r['reason']}]" if r.get("reason") else "")]
        for name, stat in r["levers"].items():
            demote = (f" (demoted: {stat['demoted']})"
                      if stat.get("demoted") else "")
            lines.append(f"  {name} [{stat['kind']}]: "
                         f"{stat['state_in']} -> {stat['state_out']}"
                         f", {stat.get('seams', 0)} seam(s){demote}")
        base, treat = r.get("baseline"), r.get("treated")
        if base:
            lines.append(f"  baseline {base['ms']:.2f} ms "
                         f"(retime {base.get('retime_ms')}, "
                         f"drift {base.get('drift')})")
        if treat:
            lines.append(f"  treated  {treat['ms']:.2f} ms "
                         f"({treat['speedup']:.3f}x), parity "
                         f"{treat['parity_vs_reference']:.6f}")
        return "\n".join(lines)

    def detach(self) -> None:
        if self.arm is not None and self.arm.teardown is not None:
            self.arm.teardown()
        self.arm = None
        if self._handle is not None:
            self._handle.detach()
            self._handle = None

    def save_receipt(self, directory) -> pathlib.Path:
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"recipe_{self.receipt['recipe']}.json"
        path.write_text(json.dumps(self.receipt, indent=2, default=str))
        return path


def _plan_digest(recipe: Recipe, model: torch.nn.Module,
                 ctx: Mapping[str, Any]) -> str:
    ident = {
        "recipe": recipe.name,
        "levers": sorted((lv.name, lv.kind) for lv in recipe.levers),
        "gates": recipe.gates.as_dict(),
        "torch": torch.__version__,
        "device": (torch.cuda.get_device_name()
                   if torch.cuda.is_available() else "cpu"),
        "model": type(model).__name__,
        "shape_sig": str(ctx.get("shape_sig", "")),
    }
    return hashlib.sha256(json.dumps(
        ident, sort_keys=True, default=str).encode()).hexdigest()


def _time_ms(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_recipe(
    recipe: Recipe,
    model: torch.nn.Module,
    ctx: dict[str, Any] | None = None,
    *,
    receipts_dir: str | pathlib.Path | None = None,
    reaudit: str = "always",
    warmup: int = 5,
    iters: int = 30,
    verbose: bool = True,
) -> RecipeRun:
    """Assemble, audit, and certify a recipe in one call.

    ``reaudit="always"`` runs the full same-process A/B.
    ``reaudit="on_change"`` skips the timing audit when every engaged
    lever is already certified under the current plan digest in the
    stored receipt — parity is still re-checked, timing numbers carry
    over marked ``cached``. Any digest mismatch forces the full audit.
    """

    def say(msg: str) -> None:
        if verbose:
            print(f"[recipe] {msg}", flush=True)

    ctx = ctx if ctx is not None else {}
    gates = recipe.gates
    digest = _plan_digest(recipe, model, ctx)
    say(f"{recipe.name}: digest {digest[:12]}")

    prior = None
    if receipts_dir is not None:
        prior_path = (pathlib.Path(receipts_dir)
                      / f"recipe_{recipe.name}.json")
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text())

    # ---- lever lifecycle: demote certifications the digest no longer
    # covers, adopt ones the stored receipt still backs ----
    lever_stats: dict[str, dict[str, Any]] = {}
    engaged: list[Lever] = []
    for lever in recipe.levers:
        stat = {"kind": lever.kind, "state_in": lever.state,
                "notes": lever.notes}
        state = lever.state
        prior_rec = (prior or {}).get("levers", {}).get(lever.name)
        prior_certified = (prior is not None
                           and prior.get("digest") == digest
                           and prior_rec is not None
                           and prior_rec.get("state_out") == "certified")
        if state == "certified" and not prior_certified:
            state = "candidate"
            stat["demoted"] = ("plan digest changed"
                               if prior is not None else "no stored receipt")
        elif state == "candidate" and prior_certified:
            state = "certified"
        stat["state"] = state
        lever_stats[lever.name] = stat
        if state != "off":
            engaged.append(lever)
    if not engaged:
        say("no engaged levers — nothing to audit")
        receipt = {"recipe": recipe.name, "digest": digest,
                   "levers": {}, "verdict": "empty"}
        return RecipeRun("empty", receipt)

    cached_ok = (reaudit == "on_change" and prior is not None
                 and prior.get("digest") == digest
                 and prior.get("verdict") == "win"
                 and all(lever_stats[lv.name]["state"] == "certified"
                         for lv in engaged))

    # ---- stock reference + baseline arm ----
    with torch.no_grad():
        reference_out = recipe.reference(model, ctx).detach().float().cpu()

    base_ms = base_retime = drift = None
    base_arm: Arm | None = None
    if not cached_ok:
        base_arm = recipe.build_stage(model, ctx)
        base_arm.tick()   # outputs are only defined after a full tick
        base_out = base_arm.output().detach().float().cpu()
        base_parity = parity_metrics(base_out, reference_out)["cosine"]
        if base_parity < gates.parity_cos:
            if base_arm.teardown is not None:
                base_arm.teardown()
            say(f"baseline arm parity {base_parity:.6f} < "
                f"{gates.parity_cos} vs stock eager — audit invalid")
            for stat in lever_stats.values():
                stat["state_out"] = stat.pop("state")
            receipt = {"recipe": recipe.name, "digest": digest,
                       "levers": lever_stats,
                       "baseline": {"parity_vs_reference": base_parity},
                       "verdict": "invalid_baseline",
                       "reason": "baseline arm does not match stock eager"}
            return RecipeRun("invalid_baseline", receipt)
        base_ms = _time_ms(base_arm.tick, warmup, iters)
        say(f"baseline arm {base_ms:.2f} ms "
            f"(parity {base_parity:.6f})")

    if recipe.between_arms is not None:
        recipe.between_arms()

    # ---- engage levers (one transaction) ----
    swaps: dict[str, torch.nn.Module] = {}
    updates: list[Callable[[], None]] = []
    refused_build: list[Lever] = []
    for lever in engaged:
        try:
            built = lever.build(model, ctx) if lever.build else {}
        except ValueError as refusal:
            # a lever may disqualify itself at build time (calibration
            # shows its precondition does not hold on this host) —
            # that is an outcome to record, not a reason to abort the
            # other levers
            lever_stats[lever.name]["state_out"] = "refused"
            lever_stats[lever.name]["reason"] = str(refusal)[:120]
            lever_stats[lever.name]["seams"] = 0
            refused_build.append(lever)
            say(f"{lever.name}: refused at build "
                f"[{str(refusal)[:80]}]")
            continue
        if isinstance(built, tuple):
            lever_swaps, update = built
        else:
            lever_swaps, update = built, None
        overlap = set(lever_swaps) & set(swaps)
        if overlap:
            raise ValueError(f"lever {lever.name!r} overlaps prior "
                             f"levers at {sorted(overlap)[:3]}")
        swaps.update(lever_swaps)
        if update is not None:
            updates.append(update)
        lever_stats[lever.name]["seams"] = len(lever_swaps)
    ctx["outside_updates"] = updates
    handle = _swap_attach(model, swaps) if swaps else None
    say(f"engaged {len(engaged)} lever(s), {len(swaps)} seam(s)")

    # ---- treated arm: parity gate, then net-win gate ----
    arm = recipe.build_stage(model, ctx)
    arm.tick()
    treated_out = arm.output().detach().float().cpu()
    parity = parity_metrics(treated_out, reference_out)["cosine"]
    parity_ok = parity >= gates.parity_cos
    parity_band = ("ok" if parity >= gates.parity_warn
                   else "warn" if parity_ok else "fail")
    say(f"treated parity vs stock eager: {parity:.6f}"
        + (" [WARN: below comfort line — calibration owes the rest]"
           if parity_band == "warn" else ""))

    treated_ms = speedup = None
    if parity_ok and cached_ok:
        base_ms = prior["baseline"]["ms"]
        treated_ms = prior["treated"]["ms"]
        speedup = prior["treated"]["speedup"]
        win = True
        say(f"digest hit — timings carried from stored receipt "
            f"({treated_ms:.2f} ms)")
    elif parity_ok:
        treated_ms = _time_ms(arm.tick, warmup, iters)
        # baseline retime after the treated arm: same-process drift
        # bound; the win must clear the *faster* of the two baselines
        base_retime = _time_ms(base_arm.tick, warmup, iters)
        drift = round(abs(base_retime - base_ms) / base_ms, 4)
        base_floor = min(base_ms, base_retime)
        speedup = round(base_floor / treated_ms, 4)
        win = speedup >= gates.min_speedup
        say(f"treated {treated_ms:.2f} ms vs baseline floor "
            f"{base_floor:.2f} ({speedup:.3f}x, drift {drift})")
    else:
        win = False

    if base_arm is not None and base_arm.teardown is not None:
        base_arm.teardown()

    verdict = "win" if win else "refused"
    reason = None
    if not parity_ok:
        reason = f"parity {parity:.6f} < {gates.parity_cos}"
    elif not win:
        reason = f"no net win ({speedup}x < {gates.min_speedup})"
    for lever in engaged:
        if lever in refused_build:
            continue
        lever_stats[lever.name]["state_out"] = (
            "certified" if win else "candidate")
    for name, stat in lever_stats.items():
        stat.setdefault("state_out", stat["state"])
        stat.pop("state", None)

    if not win:
        if arm.teardown is not None:
            arm.teardown()
        arm = None
        if handle is not None:
            handle.detach()
            handle = None
        say(f"refused [{reason}] — model restored untouched")
    else:
        say(f"win: {len(engaged)} lever(s) certified under digest "
            f"{digest[:12]}")

    receipt = {
        "recipe": recipe.name, "digest": digest,
        "model": type(model).__name__,
        "device": (torch.cuda.get_device_name()
                   if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
        "gates": gates.as_dict(),
        "reaudit": reaudit, "cached": bool(cached_ok),
        "levers": lever_stats,
        "baseline": {"ms": None if base_ms is None else round(base_ms, 3),
                     "retime_ms": (None if base_retime is None
                                   else round(base_retime, 3)),
                     "drift": drift},
        "treated": (None if treated_ms is None else
                    {"ms": round(treated_ms, 3), "speedup": speedup,
                     "parity_vs_reference": round(parity, 7),
                     "parity_band": parity_band}),
        "verdict": verdict, "reason": reason,
    }
    run = RecipeRun(verdict, receipt, arm=arm, _handle=handle)
    if receipts_dir is not None:
        path = run.save_receipt(receipts_dir)
        say(f"receipt -> {path}")
    return run
