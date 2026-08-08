"""Auto-assembly: discover seams, calibrate them in one pass, bind them.

This is the distribution layer. Given a host model and a way to run it,
it finds every structure seam (:mod:`.discover`), captures exactly the
calibration each one needs in a single forward pass, and binds each
through its library impl. The caller gets a ``path -> module`` swap map
and any outside-cadence update functions — the same thing the hand
recipes produced, derived from the model object rather than written by
hand. A host integrates by importing and calling; it writes no
per-seam scaffolding.

The calibration each structure needs, captured structure-aware:
  linear_proj / qkv_pack : the shared input the projection(s) see, and
                           its per-tensor amax (the static act scale)
  adaln_producer         : the (cond, style) pairs the conditioning
                           projection emits across the tick, for the
                           step table and its fingerprint locator
  decoder_ffn / vision_ffn : the normed input the MLP sees

Seam negotiation is resolved here: when an adaln_producer feeds a
sibling qkv_pack under the same parent, the producer emits fp8 and the
pack takes the shared act scale, skipping its own input quantization.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

from .discover import (Seam, _resolve, discover, group_families,
                       seam_weights)
from .points import Collector, Point, resolve as resolve_points

_FP8 = torch.float8_e4m3fn
_FP8_CHAIN_MAX_ROWS = 256  # fp8 producer chain qualifies at denoise
                          # M (bandwidth-bound); large-M prefill skips

# Host-family attention adapters. Attention seams are not a static
# module pattern — where the attention math actually runs is
# host-specific (a function in one host, a processor in another), so
# auto-discovery of the attention_core structure is delegated to
# registered adapters. Each adapter, given the model and a way to run
# it, returns (swaps, update) or None (this host is not its family).
_ATTENTION_ADAPTERS: list = []
_QK_NORM_ROPE_ADAPTERS: list = []
_QKV_ROPE_ADAPTERS: list = []
_GATED_DELTA_ADAPTERS: list = []


def register_attention_adapter(adapter) -> None:
    """Register a host-family attention adapter (callable)."""
    _ATTENTION_ADAPTERS.append(adapter)


def register_qk_norm_rope_adapter(adapter) -> None:
    """Register a host-family adapter for a Q/K norm + RoPE boundary."""
    _QK_NORM_ROPE_ADAPTERS.append(adapter)


def register_qkv_rope_adapter(adapter) -> None:
    """Register a host-family adapter for packed QKV + bias + RoPE."""
    _QKV_ROPE_ADAPTERS.append(adapter)


def register_gated_delta_adapter(adapter) -> None:
    """Register a host-family Gated Delta callable adapter."""
    _GATED_DELTA_ADAPTERS.append(adapter)


# Per-structure binders registered from impls, consulted before the
# built-in routing in :func:`_bind_auto`. New structures land by
# registering here from their own module instead of editing the routing
# function — parallel additions then touch disjoint files. A binder is
# ``f(model, seam, cap, *, points, fmt) -> module | None`` with the same
# refusal contract as ``_bind_auto`` (raise ``ValueError`` with the
# reason; return ``None`` for "host keeps its path").
_STRUCTURE_BINDERS: dict[str, Any] = {}


def register_structure_binder(structure: str, binder) -> None:
    """Route ``structure`` seams to ``binder`` (last write wins).

    A binder is called as ``binder(model, seam, cap, *, points, fmt,
    fmt_params)`` — ``points`` is the reduced collector, ``fmt`` the
    scheme's format routing for this seam (or ``None`` for the default),
    ``fmt_params`` the decision's recipe payload for that format (or
    ``None``).
    """
    _STRUCTURE_BINDERS[structure] = binder


@dataclass
class AutoPlan:
    """Discovered + calibrated swaps, ready to stage."""

    swaps: dict[str, torch.nn.Module] = field(default_factory=dict)
    updates: list[Callable[[], None]] = field(default_factory=list)
    seams: list[Seam] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    #: modules that carry a guard but are not swapped at a path — an
    #: adapter's routed seam. Reported by the attachment's ledger, never
    #: installed by it, so a seam that cannot be swapped can still be
    #: counted instead of being invisible.
    observed: dict[str, torch.nn.Module] = field(default_factory=dict)
    #: undo callables for host mutations that had to happen while the plan
    #: was being built rather than when it was attached (a patched
    #: module-level function). Handed to ``attach`` so ``detach`` really
    #: does give back the model that came in.
    revert: list[Callable[[], None]] = field(default_factory=list)
    #: ``(enable, disable)`` pairs for those same non-module seams, so a
    #: gate can put the host back for the baseline arm without unbinding
    #: anything. A seam that cannot be turned off cannot be measured.
    toggles: list[tuple[Callable[[], None], Callable[[], None]]] = field(
        default_factory=list)
    #: ``flash_rt.core.precision_spec.ModelPrecisionSpec`` for the scales
    #: this plan baked in — the repo's introspection format, not a private
    #: one, so ``plan.precision_spec`` reads like ``rt.precision_spec``
    precision_spec: Any = None

    def enable_routed(self) -> None:
        for on, _ in self.toggles:
            on()

    def disable_routed(self) -> None:
        for _, off in self.toggles:
            off()

    def abort(self) -> None:
        """Roll back everything this plan touched without an attach.

        The plan mutates the host as it builds (adapter routes enable,
        interface switches land, streamed originals leave the device).
        ``attach`` is the commit point; a caller that decides not to
        commit — or a bind that dies midway — calls this instead, and
        the model returns to the loaded form.
        """
        self.disable_routed()
        self.revert_all()
        store = self.notes.get("stream_store")
        if store is not None:
            store.restore_all()

    def revert_all(self) -> None:
        """Undo the plan-time host mutations. Idempotent per adapter."""
        for undo in reversed(self.revert):
            undo()
        self.revert.clear()
        self.observed.clear()


def _spec_points(seam) -> tuple[str, ...]:
    """The point names this seam's structure spec declares."""
    from .registry import load

    try:
        calibration = load(seam.structure).calibration
        if (seam.structure == "modnorm_qkv_chain"
                and seam.variant.get("modulation") == "per_token_table"):
            # the table form owns its sublayers' seams, so it carries
            # their static scales itself; the spec names them separately
            # because the scale_shift form keeps no points at all
            return tuple(calibration.get("per_token_table_points", ()))
        return tuple(calibration.get("points", ()))
    except Exception:                                   # noqa: BLE001
        return ()


def _consumer_point(seam) -> tuple[str, str]:
    """Where a negotiated consumer's input is observed.

    The producer's output *is* this tensor, so one amax serves both sides
    of the chain — which is why the pair can share a static scale at all.
    """
    if seam.structure == "qkv_pack":
        return (seam.path + "." + (seam.pack_attrs or ("q_proj",))[0], "x")
    if seam.structure == "decoder_ffn":
        return (seam.path, "x_after_norm")
    return (seam.path, "x")


def _seam_key(seam: Seam) -> str:
    """Unique receipt/decision key for a discovered structure instance.

    Most structures own one module path. A dual-path attention module may
    expose two independent sibling QKV groups under the same parent; the
    first projection is the stable, real host path that distinguishes them.
    """
    if seam.structure == "qkv_pack" and seam.pack_attrs:
        return seam.path + "." + seam.pack_attrs[0]
    if seam.structure == "modnorm_qkv_chain":
        # This catalog structure describes the composition at the same host
        # path as a possible block seam; keep both receipt identities.
        return seam.path + "::modnorm_qkv_chain"
    return seam.path


def _race_ms(fn, *, warmup: int = 3, iters: int = 10) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _pair_vision_norm_fp8(model, plan, points, stream) -> None:
    """Seat an FP8-emitting norm producer where its FFN seat consumes.

    Seat-to-seat only — the measured lesson stands: a *host* handed
    FP8 keeps going and the output is silent garbage, so the producer
    is created here, paired with the FFN seat as its direct consumer,
    and nowhere else. Both forms quantize at the same static scale, so
    the pair's smoke against the bf16 form isolates wiring breakage
    rather than calibration; the flip itself is decided by a bind-time
    measurement (the flip-only-if-faster house rule) and recorded next
    to the other format races.
    """
    import dataclasses
    from .impls import KernelUnavailable
    from .impls.vision_ffn import fp8_static as vis_impl

    for seam in plan.seams:
        if seam.structure != "vision_ffn" \
                or not getattr(seam, "norm_attr", None):
            continue
        seat = plan.swaps.get(seam.path)
        if not isinstance(seat, vis_impl.FusedGeluMlp):
            continue
        bound = seat._bound
        if bound.in_dtype != "bf16" or not bound.fc1_fp8.is_cuda:
            continue
        norm_path = (f"{seam.parent_path}.{seam.norm_attr}"
                     if seam.parent_path else seam.norm_attr)
        try:
            host_norm = model.get_submodule(norm_path)
        except AttributeError:
            continue
        try:
            from .impls.norm_fused.fp8_producer import (
                bind_norm_fp8_producer)
            producer = bind_norm_fp8_producer(host_norm,
                                              bound.input_scale)
            kern = vis_impl._kernel()
            twin = vis_impl.FusedGeluMlp(
                dataclasses.replace(
                    bound, in_dtype="fp8_static",
                    fused_mlp=(getattr(kern, "fp8_gelu_mlp_v2_bf16",
                                       None)
                               or kern.fp8_gelu_mlp_bf16)),
                original=seat._frt_host())
        except (KernelUnavailable, ValueError,
                RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (f"{norm_path}::norm_fp8_producer",
                 str(refusal)[:200]))
            continue
        rows_seen = points.row_profile(seam.path, "x_after_norm")
        rows = rows_seen[len(rows_seen) // 2] if rows_seen else 128
        dim = int(host_norm.weight.shape[0])
        dev = producer.w.device
        x = torch.randn(rows, dim, device=dev, dtype=torch.bfloat16)
        current_norm = plan.swaps.get(norm_path)
        norm_a = current_norm if current_norm is not None else host_norm
        with torch.no_grad():
            ref = seat(norm_a(x).to(torch.bfloat16))
            got = twin(producer(x))
            cos = torch.nn.functional.cosine_similarity(
                got.float().flatten(), ref.float().flatten(), dim=0)
        if float(cos) < 0.98:
            plan.notes.setdefault("refused", []).append(
                (f"{norm_path}::norm_fp8_producer",
                 f"pair smoke cos {float(cos):.6f} < 0.98"))
            continue
        ms_a = _race_ms(lambda: seat(norm_a(x).to(torch.bfloat16)))
        ms_b = _race_ms(lambda: twin(producer(x)))
        plan.notes.setdefault("format_race", []).append(
            {"layer": norm_path, "rows": rows, "dim": dim,
             "bf16_chain_ms": round(ms_a, 4),
             "fp8_norm_chain_ms": round(ms_b, 4),
             "smoke_cos": round(float(cos), 6),
             "winner": ("fp8_norm_chain" if ms_b < ms_a
                        else "bf16_chain")})
        # the bind-time race is a qualification signal, not the
        # activation: seat-level micro-timing was refuted in both
        # directions by production-form measurement (28/28 race wins
        # here measured +0.6ms end-to-end on one device). Activation
        # needs a production-form receipt for this box.
        from . import decisions as _dec
        if ms_b >= ms_a or _dec.lookup(
                "vision_norm_fp8") != "fp8_norm_chain":
            continue
        plan.swaps[seam.path] = twin
        plan.swaps[norm_path] = producer
        stream(norm_path)


def _bind_regions(model, seams, *, probe, say):
    """Resolve and bind every registered region family on this host.

    Returns ``None`` when no family identifies a region here. The
    stability contract: resolution reads receipts (pin > cache >
    seated) and a winning candidate binds *before* any seat, so its
    failure — refused symbols, a smoke miss, an exception — leaves
    every seam in place and lands on the trail. Only a successful
    bind claims the seams under its root; the seated floor is what
    remains everywhere else, always.
    """
    from . import regions
    from .impls.dit_stack import region as _dit_region
    from .impls.adarms_stack import region as _adarms_region
    from .impls.prefill_tower import region as _prefill_region
    from .impls.vision_tower import region as _vision_region
    _dit_region.register()
    _adarms_region.register()
    _prefill_region.register()
    _vision_region.register()

    notes: dict = {}
    extras = {"seams": list(seams), "observed": {}, "revert": [],
              "toggles": [], "notes": notes}
    engaged = False
    for fam in regions.registered():
        try:
            roots = list(fam.identify(model))
        except Exception:       # noqa: BLE001 — identify never kills
            continue
        if not roots:
            continue
        engaged = True
        try:
            first = (model.get_submodule(roots[0]) if roots[0]
                     else model)
            host_sig = regions.structural_signature(first)
        except Exception:   # noqa: BLE001 — scoping never kills a bind
            host_sig = None
        winner, source = regions.resolve(fam.family, host_sig=host_sig,
                                         notes=notes)
        if winner == regions.SEATED:
            say(f"region {fam.family}: seated ({source})")
            continue
        cand = fam.candidate(winner)
        for root in roots:
            label = f"{root}::{winner}"
            if probe is None:
                notes.setdefault("regions_refused", []).append(
                    (label, "no probe callable"))
                continue
            try:
                result = cand.bind(model, root, probe)
            except Exception as exc:  # noqa: BLE001 — never kills
                result = {"refused": f"{type(exc).__name__}: {exc}"}
            if not result or result.get("refused"):
                reason = (result or {}).get(
                    "refused", "bind returned nothing")
                notes.setdefault("regions_refused", []).append(
                    (label, str(reason)[:200]))
                say(f"region {fam.family}@{root}: {winner} refused "
                    "-> seated")
                continue
            prefix = root + "." if root else ""
            keep, claimed = [], 0
            for s in extras["seams"]:
                if s.path == root or s.path.startswith(prefix):
                    claimed += 1
                else:
                    keep.append(s)
            extras["seams"] = keep
            extras["observed"].update(result.get("observed", {}))
            extras["revert"].extend(result.get("revert", ()))
            if result.get("toggle") is not None:
                extras["toggles"].append(result["toggle"])
            notes.setdefault("regions_bound", []).append(
                {"family": fam.family, "root": root, "winner": winner,
                 "source": source, "claimed_seams": claimed,
                 "smoke_cos": result.get("smoke_cos")})
            say(f"region {fam.family}@{root}: {winner} bound "
                f"({claimed} seam(s) claimed, source={source})")
    if engaged:
        _wire_region_products(extras, notes, say)
    return extras if engaged else None


def _wire_region_products(extras, notes, say) -> None:
    """Cross-region wires: a bound producer feeds a bound consumer's
    chain buffers directly, and the consumer drops its own in-graph
    restaging. Structure-level, receipt-visible, and only ever armed
    when both ends are bound with agreeing facts — a missing or
    refused end simply leaves both chains in their standalone form.

    First wire: a prefill tower's per-layer K/V (already in the shared
    adjacent-pair chain layout — both chains assemble the same split
    kernel) into a conditioned decoder stack's cache prefix rows. The
    decoder's per-step prefix gather is the identity of this write, so
    it leaves the graph.
    """
    towers, stacks = [], []
    for label, bound in extras["observed"].items():
        if "::prefill_" in label and getattr(bound, "prefix_sinks",
                                             "no") != "no":
            towers.append((label, bound))
        if "::adarms_" in label and getattr(bound, "prefix_wired",
                                            "no") != "no":
            stacks.append((label, bound))
    if len(towers) != 1 or len(stacks) != 1:
        return
    (t_label, tower), (s_label, stack) = towers[0], stacks[0]
    tk = {k: tower.dims.get(k) for k in ("seq", "hd", "layers")}
    if (tk["seq"] != stack.p_used or tk["hd"] != stack.dims.get("hd")
            or tk["layers"] != stack.dims.get("layers")
            or not stack.buf.get("kc")):
        notes.setdefault("region_wires", []).append(
            {"wire": "prefix_kv", "armed": False,
             "reason": f"facts disagree: tower {tk} vs stack "
                       f"P={stack.p_used}"})
        return
    P = stack.p_used
    tower.prefix_sinks = [
        (stack.buf["kc"][l][0, :P, 0], stack.buf["vc"][l][0, :P, 0])
        for l in range(tk["layers"])]
    stack.prefix_wired = True

    def unwire() -> None:
        tower.prefix_sinks = None
        stack.prefix_wired = False

    extras["revert"].append(unwire)
    notes.setdefault("region_wires", []).append(
        {"wire": "prefix_kv", "armed": True, "producer": t_label,
         "consumer": s_label, "rows": P})
    say(f"region wire prefix_kv: {t_label} -> {s_label} ({P} rows)")


def _merge_region_extras(plan, extras) -> None:
    plan.notes.update(extras["notes"])
    plan.observed.update(extras["observed"])
    plan.revert.extend(extras["revert"])
    plan.toggles.extend(extras["toggles"])


def auto_swaps(
    model: torch.nn.Module,
    forward: Callable[..., Any] | Sequence[Callable[[], Any]],
    *,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn",
                                   "qkv_pack", "adaln_producer",
                                   "linear_proj", "patch_projection",
                                   "norm_fused",
                                   "attention_core", "decoder_block",
                                   "modnorm_qkv_chain", "qk_norm_rope",
                                   "qkv_rope",
                                   "gated_delta_core"),
    negotiate_fp8: bool = True,
    prefix_cadence: bool = False,
    observations: Iterable[Any] | None = None,
    percentile: float = 99.9,
    max_samples: int | None = None,
    scheme: str | Any = "auto",
    verbose: bool = False,
    stream_store: Any = None,
) -> AutoPlan:
    """Discover, calibrate in one pass, and bind every applicable seam.

    The calibration arguments are the repo's, not this layer's: the names,
    the defaults and the meaning of ``percentile`` / ``max_samples`` /
    ``observations`` are ``flash_rt.api.FlashRT.calibrate``'s, and the
    reduction is ``flash_rt.core.calibration.accumulate_amax``. A second
    vocabulary for the same thing is the one thing this layer must not
    add.

        auto_swaps(model, forward)                        # one sample
        auto_swaps(model, [f0, f1, f2])                    # one thunk each
        auto_swaps(model, feed, observations=dataset)      # feed(obs) per obs
        auto_swaps(model, feed, observations=ds, percentile=95.0)

    ``scheme`` names a registered quantisation scheme (:mod:`.schemes`):
    what statistic each point needs, and per seam whether to bind or keep
    the host at host precision. The default ``"auto"`` resolves to the
    highest-performing profile the device can execute — on FP8-capable
    hardware that is ``fp8_static``, bit-identical to the behaviour this
    layer shipped with; elsewhere it is ``none`` (fusion structures
    attach, quantised seams stay at host precision). Explicit selection
    (``scheme="none"``, ``scheme="w4a16_decode"``, ...) overrides. It
    adds no calibration entry.

    ``forward`` is always "run the host once"; with ``observations`` it
    takes one observation. That indirection is this layer's only
    difference from a frontend's ``calibrate``, and it exists because a
    host here is an arbitrary ``nn.Module`` with no common observation
    contract — not because the calibration standard differs.

    ``prefix_cadence`` declares that the caller will run ``plan.updates``
    whenever the observation changes. Structures that hold per-observation
    host state — the attention core keeps the prefix keys and values — are
    only offered when it is set, because without the refresh they attend to
    whatever the calibration saw. Leaving it off is the accurate default.

    On ``percentile``: it reduces *across* samples. Within one sample the
    reduction is a max over every call, which is required rather than
    chosen — see ``docs/calibration.md`` §4.2. And note §10's own caveat
    that at small N a 99.9 percentile barely clips at all (it interpolates
    between the top two ranks); with N ≤ 64 and suspected outliers, pass a
    lower one.
    """

    def say(msg: str) -> None:
        if verbose:
            print(f"[autobuild] {msg}", flush=True)

    thunks, source = _calibration_thunks(forward, None, observations)
    if max_samples is not None and len(thunks) > max_samples:
        thunks = thunks[:max_samples]
    plan_notes_calibration: dict[str, Any] = {}
    plan_refusals: list[tuple[str, str]] = []

    # A schedule adapter changes only the Python spelling of a qualified
    # fixed loop.  Calibration must observe that same canonical execution:
    # otherwise a tensor-controlled host ``while`` re-enters a compiled
    # graph break on every denoise step before the region hooks even run.
    from .impls.fixed_iter import (
        FixedIterationRefused,
        normalize_fixed_iteration,
    )

    schedule_notes: list[dict[str, Any]] = []
    normalized_thunks = []
    for thunk in thunks:
        try:
            schedule = normalize_fixed_iteration(thunk, model)
        except FixedIterationRefused as exc:
            raise ValueError(str(exc)) from exc
        if schedule is None:
            normalized_thunks.append(thunk)
            continue
        normalized_thunks.append(schedule.forward)
        schedule_notes.append({
            "family": schedule.family,
            "steps": schedule.steps,
            "exact": schedule.exact,
            **dict(schedule.details),
        })
    thunks = normalized_thunks

    seams = discover(model, structures, refused=plan_refusals)
    say(f"discovered {len(seams)} seam(s)")

    # ---- region adjudication: the structure-level winner is a receipt.
    # Resolution reads author pin > decision cache > seated and never
    # experiments here; a winning candidate binds before any seat does
    # (its failure leaves every seam in place — seated is always the
    # floor), and only a *successful* bind claims the seams it absorbs.
    region_extras = _bind_regions(
        model, seams, probe=(thunks[0] if thunks else None), say=say)
    if region_extras is not None:
        seams = region_extras["seams"]

    adapter_only = not seams and bool(
        {"attention_core", "gated_delta_core"}.intersection(structures))
    if not seams and not adapter_only:
        plan = AutoPlan()
        if region_extras is not None:
            _merge_region_extras(plan, region_extras)
        return plan

    # ---- one calibration pass, structure-aware capture ----
    # Activation scales go through the house two-level statistic: a max
    # over every call inside one sample (docs/calibration.md §4.2 — a
    # flow-matching host runs every step inside one forward and per-step
    # scales crashed the compiler), then accumulate_amax's percentile
    # across samples. Nothing holds an activation tensor for this: each
    # point is one float, measured where the spec says it is
    # (:mod:`.points`).
    caps: dict[str, dict[str, Any]] = {}
    hooks = []
    all_points: list[Point] = []
    seam_points: dict[str, list[Point]] = {}
    for seam in seams:
        try:
            pts = resolve_points(seam, _spec_points(seam))
        except ValueError as refusal:
            plan_refusals.append((seam.path, str(refusal)[:120]))
            continue
        seam_points[_seam_key(seam)] = pts
        all_points.extend(pts)

    from . import schemes as _schemes
    auto_resolved = isinstance(scheme, str) and scheme == "auto"
    if auto_resolved:
        scheme = _schemes.resolve_auto()
        say(f"scheme auto -> {scheme}")
    scheme_obj = (_schemes.get(scheme) if isinstance(scheme, str)
                  else scheme)
    # loud wall before any calibration work: a scheme asking for a
    # granularity the collector cannot measure must not silently get
    # per-tensor numbers of the wrong shape
    stat_request = scheme_obj.statistics(tuple(all_points))
    _schemes.validate_request(stat_request)
    collector = Collector(points=all_points, request=dict(stat_request))

    # observed call order across the whole calibration pass. Anything
    # that has to know which seam runs first (a stream-scoped buffer
    # needs a writer, and the writer has to be the one the host calls
    # first) reads it from here rather than assuming the module tree's
    # order matches the forward's.
    call_order = itertools.count()

    def cap_cond(path):
        def hook(module, args, out):
            cap = caps[path]
            if "order" not in cap:
                cap["order"] = next(call_order)
            cap.setdefault("pairs", []).append(
                (args[0].detach().clone(), out.detach().clone()))
            return None
        return hook

    def cap_shape(path):
        # a block seam needs no tensors of its own, only the host's
        # return convention (bare tensor or 1-tuple)
        def hook(module, args, kwargs, out):
            caps[path]["returns_tuple"] = isinstance(out, tuple)
            return None
        return hook

    def cap_pack_input(path, attr):
        """Record the executable shared-input property of a pack sibling.

        Equal K dimensions only prove that projections *could* share an
        input.  Cross attention is the counterexample: Q consumes the live
        latent while K/V consume encoder state.  The leaf implementation
        runs the first projection and turns later siblings into stash reads,
        so exact storage identity and call order are part of its contract.
        """
        def hook(module, args):
            x = args[0] if args else None
            cap = caps[path]
            cap.setdefault("pack_events", []).append(attr)
            if not torch.is_tensor(x):
                cap.setdefault("pack_inputs", {}).setdefault(attr, []).append(
                    None)
                return None
            signature = (
                int(x.data_ptr()), int(x.storage_offset()), tuple(x.shape),
                tuple(x.stride()), x.dtype, x.device,
            )
            cap.setdefault("pack_inputs", {}).setdefault(attr, []).append(
                signature)
            return None
        return hook

    def cap_patch_input(path, width):
        """Record that the host really supplies complete flattened patches.

        Matching Conv3d slots is not enough: a module with the same kernel
        may consume an ordinary 5-D volume. The lowering is legal only when
        the calibrated host input exposes K as its final dimension, exactly
        as the processor-preflattened dataflow declares.
        """
        def hook(module, args):
            x = args[0] if args else None
            form = None
            if torch.is_tensor(x) and x.numel() % width == 0:
                form = (int(x.shape[-1]), int(x.numel() // width))
            caps[path].setdefault("patch_inputs", []).append(form)
            return None
        return hook

    # the amax points are hooked by the collector; only the two
    # content/observation captures need their own hooks here, and neither
    # is a statistic: a step table is memoised host output, a return
    # convention is one boolean
    for seam in seams:
        key = _seam_key(seam)
        caps[key] = {}
        target = _resolve(model, seam.path)
        if seam.structure == "decoder_block":
            hooks.append(target.register_forward_hook(
                cap_shape(key), with_kwargs=True))
        elif seam.structure == "adaln_producer":
            hooks.append(getattr(target, seam.cond_attr)
                         .register_forward_hook(cap_cond(key)))
        elif seam.structure == "qkv_pack":
            for attr in seam.pack_attrs or ():
                hooks.append(getattr(target, attr).register_forward_pre_hook(
                    cap_pack_input(key, attr)))
        elif seam.structure == "patch_projection":
            hooks.append(target.register_forward_pre_hook(
                cap_patch_input(key, seam.dims["K"])))
    hooks.extend(collector.hooks(lambda path: _resolve(model, path)))

    if hooks:
        # the calibration pass is a transaction over the host: if a thunk
        # raises, the hooks come off and no plan is returned. Removing
        # them only on the success path leaves a failed calibration's
        # hooks on the model, where they keep firing into a dict nobody
        # reads and slow down every later forward for reasons that are
        # nowhere in sight.
        try:
            with torch.no_grad():
                for thunk in thunks:
                    thunk()
                    # one vector per sample, so the percentile across them
                    # is possible at all
                    collector.end_sample()
        finally:
            for h in hooks:
                h.remove()
        plan_notes_calibration = dict(
            collector.reduce(percentile, verbose=verbose,
                             label=f"structures_N{len(thunks)}"),
            source=source)
        say(f"calibration pass done ({len(thunks)} sample(s) from "
            f"{source}, {plan_notes_calibration['points']} point(s), "
            f"{plan_notes_calibration['method']}"
            + (f" p={percentile}" if len(thunks) > 1 else "") + ")")

    # Adapters bind after the precision scheme may remove host-precision
    # seams from ``plan.seams``. Preserve only the observed row capacity in
    # the transient cap map so a structural adapter can preallocate without
    # retaining activations or depending on a quantized sibling being bound.
    for seam in seams:
        key = _seam_key(seam)
        rows = [
            row
            for point in seam_points.get(key, ())
            for row in collector.row_profile(point.path, point.name)
        ]
        if rows:
            caps[key]["rows"] = max(rows)

    # A pack owns its sibling projections only after the calibration pass
    # proves the property its execution relies on: identical input storage
    # and q/k/v call order for every invocation.  If it does not, retain the
    # independent linear projections; they are valid structures at the
    # narrower boundary.  This is deliberately a data-flow check rather than
    # a class/path allow-list, so self- and cross-attention implementations
    # using the same module type are classified by what they actually do.
    qualified_packs = []
    for seam in (s for s in seams if s.structure == "qkv_pack"):
        attrs = tuple(seam.pack_attrs or ())
        cap = caps.get(_seam_key(seam), {})
        inputs = cap.get("pack_inputs", {})
        columns = [inputs.get(attr, []) for attr in attrs]
        count_ok = bool(columns) and len({len(col) for col in columns}) == 1
        calls = len(columns[0]) if count_ok else 0
        shared = count_ok and calls > 0 and all(
            all(col[i] is not None and col[i] == columns[0][i]
                for col in columns[1:])
            for i in range(calls)
        )
        ordered = cap.get("pack_events", []) == list(attrs) * calls
        if shared and ordered:
            qualified_packs.append(seam)
            continue
        plan_refusals.append((
            _seam_key(seam),
            "qkv_pack refused: sibling projections did not consume the "
            "same tensor in fixed order during calibration",
        ))
    qualified_ids = {id(seam) for seam in qualified_packs}
    seams = [s for s in seams
             if s.structure != "qkv_pack" or id(s) in qualified_ids]
    packed = {s.path + "." + a for s in qualified_packs
              for a in (s.pack_attrs or ())}
    seams = [s for s in seams
             if not (s.structure == "linear_proj" and s.path in packed)]

    # ---- per-token-table chains own their block's producer-fed members:
    # the self-attention pack (the chain quantizes once for all three)
    # and the FFN (the chain's second producer site feeds it fused).
    # Everything else under the block — the output projection, the whole
    # cross-attention — stays individually bindable, and the chain's
    # forward calls whatever is attached there. ----
    chain_blocks = {
        s.path for s in seams
        if s.structure == "modnorm_qkv_chain"
        and s.variant.get("modulation") == "per_token_table"}
    if chain_blocks:
        def _chain_owns(seam):
            for block in chain_blocks:
                if (seam.structure == "qkv_pack"
                        and seam.path == block + ".attn1"):
                    return True
                if (seam.structure == "vision_ffn"
                        and seam.path == block + ".ffn"):
                    return True
                if (seam.structure == "linear_proj"
                        and seam.path.startswith(block + ".attn1.")
                        and seam.path.rsplit(".", 1)[1] in
                        ("to_q", "to_k", "to_v")):
                    return True
            return False

        seams = [s for s in seams if not _chain_owns(s)]

    # ---- the scheme turns statistics into decisions. Keep-host is a
    # first-class outcome recorded in the receipt, not a refusal: the
    # seam is healthy, the scheme chose host precision for it. ----
    class _SeamStats(dict):
        def __init__(self, *args, structure: str, **kwargs):
            super().__init__(*args, **kwargs)
            self.structure = structure

    seam_by_key = {_seam_key(seam): seam for seam in seams}
    # Host-precision structural lowerings do not belong to a quantisation
    # scheme decision. They still use the collector for real row/dtype
    # qualification, but scheme="none" must not remove them.
    scheme_independent = {"patch_projection"}
    scheme_report = {
        path: _SeamStats(
            {f"{pt.path}|{pt.name}": collector.amax(pt.path, pt.name)
             for pt in pts}, structure=seam_by_key[path].structure)
        for path, pts in seam_points.items()
        if path in seam_by_key
        and seam_by_key[path].structure not in scheme_independent}
    decision = scheme_obj.decide(scheme_report)
    scheme_note: dict[str, Any] = {
        "name": getattr(scheme_obj, "name", type(scheme_obj).__name__)}
    if auto_resolved:
        scheme_note["auto"] = True
    if decision.keep_host:
        kept = set(decision.keep_host)
        seams = [s for s in seams if _seam_key(s) not in kept]
        scheme_note["keep_host"] = {
            p: decision.reasons.get(p, "") for p in sorted(kept)}
        say(f"scheme {scheme_note['name']}: {len(kept)} seam(s) kept at "
            f"host precision")
    formats: dict[str, str] = dict(decision.formats or {})
    fmt_params: dict[str, Any] = dict(getattr(decision, "params", None) or {})
    if formats:
        scheme_note["formats"] = dict(sorted(formats.items()))
        if fmt_params:
            scheme_note["params"] = {p: dict(v) for p, v
                                     in sorted(fmt_params.items())}
        say(f"scheme {scheme_note['name']}: {len(formats)} seam(s) "
            f"routed to a non-default format")

    # ---- fp8 seam negotiation: the load-bearing structure combination.
    # A single kernel need not win alone (fp8 qkv at M=50 is marginal,
    # fa2 in a bf16 stack loses); the *chain* wins — an adaln producer
    # that emits fp8 lets the qkv pack skip its own input quantization
    # and hands a clean fp8 seam down to the attention core. Bind the
    # producer→pack pair together with one shared act scale wherever a
    # producer feeds a pack under the same parent layer. ----
    act_scales: dict[str, torch.Tensor] = {}
    chain_rows: dict[str, int] = {}
    negotiated: dict[str, dict[str, Seam]] = {}
    if negotiate_fp8:
        by_parent: dict[str, dict[str, Seam]] = {}
        for seam in seams:
            if formats.get(_seam_key(seam)):
                # a chain shares one scale and one wire dtype; a member
                # routed to another format has neither, so it binds
                # standalone through its own impl instead
                continue
            layer = _layer_of(seam.path)
            if seam.structure == "adaln_producer":
                # a layer has two producer→consumer seams: the norm
                # before attention feeds the projections, the norm after
                # it feeds the MLP. Both can hand fp8 downstream.
                slot = ("producer" if _feeds_attention(seam.path)
                        else "producer_ffn")
                by_parent.setdefault(layer, {})[slot] = seam
            elif seam.structure == "qkv_pack":
                by_parent.setdefault(layer, {})["pack"] = seam
            elif (seam.structure == "linear_proj"
                  and seam.proj_attr in ("q_proj", "to_q",
                                         "add_q_proj")):
                by_parent.setdefault(layer, {})["query"] = seam
            elif seam.structure == "decoder_ffn":
                by_parent.setdefault(layer, {})["ffn"] = seam
        # the chain wins at small M (denoise): fp8 is bandwidth-bound and
        # pays there, while a large-M prefill GEMM is compute-bound and
        # fp8 buys little — and an fp8 producer feeding a big compiled
        # prefill region is where the triton fp8 codegen chokes. Qualify
        # on the calibrated row count, not on host names.
        dev = next(model.parameters()).device
        blocks = {s.path for s in seams if s.structure in (
            "decoder_block", "modnorm_qkv_chain")}
        for lay, g in by_parent.items():
            # the attention pack is always negotiated. The FFN chain is
            # negotiated only where a decoder_block owns the boundary,
            # and the reason is the boundary rather than the kernel: at
            # the norm seam the fused producer costs a kernel
            # (gate_residual, +180 launches) plus its style
            # materialization (+180) to save the FFN's own input
            # quantize (-180) — measured net +0.17ms, so it is refused
            # there. Inside a block the same kernel *replaces* the
            # host's gated residual add instead of adding to it, which
            # is the whole point of owning the block.
            # Both chains need the block boundary, and for the same
            # reason: a negotiated producer emits FP8, and only a caller
            # that owns the block consumes it. Bound at the norm boundary
            # the *host* is the consumer, and the host expects its norm to
            # return a compute dtype — handed FP8 it keeps going and the
            # output is garbage (measured 0.24 output match, and NaN on a
            # neighbouring configuration) with nothing to see, because
            # every shape and dtype is inside its contract. The FFN chain
            # was already gated this way; the attention chain was not.
            if lay not in blocks:
                continue
            pairs = [("producer", "pack"), ("producer", "query"),
                     ("producer_ffn", "ffn")]
            keep = {}
            for p_slot, c_slot in pairs:
                if p_slot not in g or c_slot not in g:
                    continue
                c_path, c_name = _consumer_point(g[c_slot])
                amax = collector.amax(c_path, c_name)
                rows_seen = collector.row_profile(c_path, c_name)
                rows = rows_seen[len(rows_seen) // 2] if rows_seen else 1 << 30
                if amax is None or rows > _FP8_CHAIN_MAX_ROWS:
                    continue
                # the consumer's input == the producer's output; its amax
                # is the one static scale both sides share
                keep[p_slot], keep[c_slot] = g[p_slot], g[c_slot]
                act_scales[f"{lay}|{c_slot}"] = torch.tensor(
                    [max(amax / 448.0, 1e-8)], device=dev)
                chain_rows[f"{lay}|{c_slot}"] = rows
            if keep:
                negotiated[lay] = keep

    # ---- the negotiated chain binds as one unit ----
    # producer and consumer must agree on the seam dtype: a pack bound
    # for fp8 input whose producer failed to bind would be handed BF16,
    # and the host would silently grow a quantize fused into whatever
    # produced it. Bind the pair together, or leave both on BF16.
    plan = AutoPlan(seams=seams)
    plan._requested_structures = frozenset(structures)
    if region_extras is not None:
        _merge_region_extras(plan, region_extras)

    # ---- per-seat streaming consumption (the bind-peak lever) ----
    # With a weight store handed in, every leaf placement immediately
    # moves the replaced original's truth off the device: the originals
    # shrink as the quantized copies grow, and the bind peak stays near
    # one model instead of two. Regions whose later stages still read
    # host originals are excluded up front — the decoder_block seams
    # compose from their children, and the cross-attention K/V
    # projections serve the cadence banks — those wait for the
    # attachment's own consume().
    _stream_exclude: set[str] = set()
    if stream_store is not None:
        for _s in seams:
            if _s.structure == "decoder_block":
                _stream_exclude.add(_s.path)
        from .impls.cadence_static.cross_attention import (
            discover_cross_attention_kv as _discover_ckv)
        for _cand in _discover_ckv(model):
            _stream_exclude.add(_cand.path)

    def _stream(placed) -> None:
        if stream_store is None:
            return
        paths = placed if isinstance(placed, (list, tuple, set)) \
            else [placed]
        from .swap import resolve_parent as _rp, _get as _sg
        for p in paths:
            if any(p == e or p.startswith(e + ".")
                   for e in _stream_exclude):
                continue
            try:
                parent, attr = _rp(model, p)
                stream_store.stash_module(p, _sg(parent, attr))
            except (AttributeError, TypeError, ValueError):
                continue
        plan.notes["streamed_bytes"] = stream_store.stats["freed_bytes"]
    # ---- backbone attention interface: measured band decision. This
    # must precede every adapter that resolves the host's attention
    # registry into a closure (the rope routes, the vision pin), or a
    # switched interface arrives after the traffic already left ----
    if "attention_core" in structures:
        from .adapters.transformers_attention_interface import (
            TransformersAttentionInterfaceAdapter)
        try:
            iface = TransformersAttentionInterfaceAdapter()(model, plan)
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                ("backbone_attn", str(refusal)[:200]))
            iface = None
        if iface:
            if iface.get("refused"):
                plan.notes.setdefault("refused", []).extend(
                    iface["refused"])
            plan.revert.extend(iface.get("revert", ()))
            plan.notes.update(iface.get("notes", {}))
    plan.notes["scheme"] = scheme_note
    if schedule_notes:
        plan.notes["schedules"] = schedule_notes
    handled: set[str] = set()
    for lay, g in negotiated.items():
        for p_slot, c_slot in (("producer", "pack"),
                               ("producer", "query"),
                               ("producer_ffn", "ffn")):
            if p_slot not in g or c_slot not in g:
                continue
            p_seam, c_seam = g[p_slot], g[c_slot]
            p_cap = caps.get(_seam_key(p_seam), {})
            if not p_cap.get("pairs"):
                continue
            try:
                pair = _bind_negotiated(
                    model, p_seam, c_seam, p_cap, collector,
                    act_scales[f"{lay}|{c_slot}"],
                    chain_rows[f"{lay}|{c_slot}"], plan)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (f"{lay} [{c_slot} chain]", str(refusal)[:200]))
                continue
            plan.swaps.update(pair)
            _stream(list(pair))
            handled.update({_seam_key(p_seam), _seam_key(c_seam)})
            for chain in (
                seam for seam in seams
                if seam.structure == "modnorm_qkv_chain"
                and seam.path == lay
            ):
                handled.add(_seam_key(chain))
                plan.notes.setdefault("composed_structures", []).append(
                    _seam_key(chain))
    plan.notes["negotiated_layers"] = sorted(
        lay for lay, g in negotiated.items()
        if any(_seam_key(sm) in handled for sm in g.values()))

    # ---- bind the remaining seams individually ----
    for name, members in group_families(seams).items():
        for seam in members:
            key = _seam_key(seam)
            if key in handled:
                continue
            cap = caps.get(key, {})
            try:
                bound = _bind_auto(model, seam, cap, plan, act_scales,
                                   negotiate_fp8, points=collector,
                                   fmt=formats.get(key),
                                   fmt_params=fmt_params.get(key))
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (key, str(refusal)[:200]))
                continue
            if bound is None:
                continue
            if isinstance(bound, dict):
                plan.swaps.update(bound)
                _stream(list(bound))
            else:
                plan.swaps[seam.path] = bound
                _stream(seam.path)
    # ---- pre-FFN norm → FP8 producer pairs (seat-to-seat only) ----
    # The norm's sole consumer is the FFN seat, so the pair moves the
    # FFN's input quantize into the norm kernel. Both ends are seats
    # (the FP8_ONLY guard refuses anything else between them), the
    # smoke compares the pair against the bf16 form it would replace,
    # and the flip happens only when the measured chain is faster.
    _pair_vision_norm_fp8(model, plan, collector, _stream)

    # ---- streaming window: the later stages record through the model,
    # and the streamed originals are meta now — so the bound seats stand
    # in for them for the duration. Temporarily attached with plain
    # setattr (no guards armed), undone before returning: the caller's
    # attach() must still find the originals at every path.
    _stream_temp = []
    if stream_store is not None:
        from .swap import resolve_parent as _srp, _get as _sgt, _set as _sst
        for _p, _m in list(plan.swaps.items()):
            try:
                _par, _at = _srp(model, _p)
                _stream_temp.append((_par, _at, _sgt(_par, _at)))
                _sst(_par, _at, _m)
            except (AttributeError, TypeError, ValueError):
                continue

    if stream_store is not None:
        plan.notes["stream_store"] = stream_store

    def _stream_window_undo():
        if not _stream_temp:
            return
        from .swap import _set as _sst2
        for _par, _at, _orig in reversed(_stream_temp):
            _sst2(_par, _at, _orig)
        _stream_temp.clear()

    # every later stage may record through the seated model; whatever
    # they do the window closes, and a stage that dies rolls the whole
    # plan back — a half-routed model is worse than no plan
    try:
        # ---- qk_norm_rope: compose a packed QKV seam with host attention ----
        if "qk_norm_rope" in structures:
            from . import adapters as _adapters  # noqa: F401 (registers)
            import inspect as _inspect
            _probe0 = (forward if callable(forward)
                       else (forward[0] if forward else None))
            for adapter in _QK_NORM_ROPE_ADAPTERS:
                try:
                    if "probe" in _inspect.signature(
                            adapter.__call__).parameters:
                        result = adapter(model, plan, probe=_probe0)
                    else:
                        result = adapter(model, plan)
                except (ValueError, RuntimeError) as refusal:
                    plan.notes.setdefault("refused", []).append(
                        ("qk_norm_rope", str(refusal)[:200]))
                    continue
                if result is None:
                    continue
                extras = result
                if extras.get("refused"):
                    plan.notes.setdefault("refused", []).extend(
                        extras["refused"])
                engaged = bool(
                    extras.get("observed")
                    or extras.get("revert")
                    or extras.get("toggle")
                )
                if not engaged:
                    continue
                plan.observed.update(extras.get("observed", {}))
                plan.revert.extend(extras.get("revert", ()))
                if extras.get("toggle") is not None:
                    plan.notes["qk_norm_rope_toggle_index"] = len(plan.toggles)
                    plan.toggles.append(extras["toggle"])
                plan.notes["qk_norm_rope_adapter"] = (
                    type(adapter).__name__
                    if hasattr(adapter, "__name__")
                    else str(adapter)
                )
                break
        # ---- qkv_rope: packed biased QKV plus rotate-half RoPE ----
        if "qkv_rope" in structures:
            from . import adapters as _adapters  # noqa: F401 (registers)
            for adapter in _QKV_ROPE_ADAPTERS:
                try:
                    result = adapter(model, plan, caps)
                except (ValueError, RuntimeError) as refusal:
                    plan.notes.setdefault("refused", []).append(
                        ("qkv_rope", str(refusal)[:100]))
                    continue
                if result is None:
                    continue
                if result.get("refused"):
                    plan.notes.setdefault("refused", []).extend(result["refused"])
                engaged = bool(
                    result.get("observed")
                    or result.get("revert")
                    or result.get("toggle")
                )
                if not engaged:
                    continue
                plan.observed.update(result.get("observed", {}))
                plan.revert.extend(result.get("revert", ()))
                if result.get("toggle") is not None:
                    plan.toggles.append(result["toggle"])
                plan.notes["qkv_rope_adapter"] = type(adapter).__name__
                break
        # ---- attention_core: host-family adapters (fa2 seam) ----
        if "attention_core" in structures:
            from . import adapters as _adapters  # noqa: F401 (registers)
            for adapter in _ATTENTION_ADAPTERS:
                try:
                    # the adapter needs "run the host once", which is what a
                    # thunk is. Handing it the caller's callable breaks the
                    # sample entry, where that callable takes a sample —
                    # the whole point of normalising the three ways in was
                    # that nothing downstream should see the difference
                    result = adapter(model, thunks[0],
                                     prefix_cadence=prefix_cadence)
                except (ValueError, RuntimeError) as refusal:
                    plan.notes.setdefault("refused", []).append(
                        ("attention_core", str(refusal)[:200]))
                    continue
                if result is None:
                    continue
                # an adapter may hand back a third element for the parts of
                # its seam that are not modules at paths: how to undo them,
                # and what to report
                att_swaps, update = result[0], result[1]
                extras = result[2] if len(result) > 2 else {}
                if extras.get("refused"):
                    plan.notes.setdefault("refused", []).extend(
                        extras["refused"])
                engaged = bool(
                    att_swaps or update
                    or extras.get("observed")
                    or extras.get("revert")
                    or extras.get("toggle")
                )
                if not engaged:
                    continue
                plan.swaps.update(att_swaps)
                plan.observed.update(extras.get("observed", {}))
                if extras.get("attention_variants"):
                    plan.notes.setdefault(
                        "attention_core_variants", {}).update(
                            extras["attention_variants"])
                plan.revert.extend(extras.get("revert", ()))
                if extras.get("toggle") is not None:
                    plan.toggles.append(extras["toggle"])
                if update is not None:
                    plan.updates.append(update)
                plan.notes["attention_adapter"] = type(adapter).__name__ \
                    if hasattr(adapter, "__name__") else str(adapter)
                break


        # ---- gated_delta_core: stateful host callable adapters ----
        if "gated_delta_core" in structures:
            from . import adapters as _adapters  # noqa: F401 (registers)
            for adapter in _GATED_DELTA_ADAPTERS:
                try:
                    # adapters that declare scheme awareness receive the
                    # active scheme; the rest keep the two-argument call
                    if getattr(adapter, "scheme_aware", False):
                        result = adapter(model, thunks[0], scheme=scheme_obj)
                    else:
                        result = adapter(model, thunks[0])
                except (ValueError, RuntimeError) as refusal:
                    plan.notes.setdefault("refused", []).append(
                        ("gated_delta_core", str(refusal)[:120]))
                    continue
                if result is None:
                    continue
                plan.observed.update(result.get("observed", {}))
                plan.revert.extend(result.get("revert", ()))
                if result.get("toggle") is not None:
                    plan.toggles.append(result["toggle"])
                plan.notes["gated_delta_adapter"] = type(adapter).__name__ \
                    if hasattr(adapter, "__name__") else str(adapter)
                break

        # ---- one step-scoped style materialisation per conditioning stream
        # Every adaptive-norm producer on one stream resolves the same step,
        # so the whole stream's styles are fixed for the step's duration.
        # Materialising them once beats materialising them per call by the
        # launch count, which is what that work actually costs. Runs before
        # the block assembly: a block holds its producers directly and drops
        # them from the swap map, so afterwards they are no longer findable
        # here.
        _attach_brokers(caps, plan, say)

        # ---- decoder_block: compose the bound sublayers into one block ----
        # last, because it is assembled from what the region structures
        # produced. The swaps it absorbs are dropped from the plan: the
        # block holds those modules directly, and a swap that also targeted
        # the host child would leave two live copies of the same seam.
        for seam in (s for s in seams if s.structure == "decoder_block"):
            try:
                block = _bind_block(
                    model, seam, caps.get(_seam_key(seam), {}), plan)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (seam.path + " [block]", str(refusal)[:200]))
                continue
            if block is None:
                continue
            for child in _BLOCK_OWNED:
                plan.swaps.pop(seam.path + "." + child, None)
            plan.swaps[seam.path] = block

    except BaseException:
        _stream_window_undo()
        plan.abort()
        raise
    finally:
        _stream_window_undo()

    # what discovery took on trust, for the seams that actually bound. An
    # assumption that reaches the model without reaching the receipt is
    # indistinguishable from something that was checked.
    assumed = [(s.path, note) for s in seams if s.assumptions
               and s.path in plan.swaps for note in s.assumptions]
    if assumed:
        plan.notes["assumed"] = assumed
        say(f"{len(assumed)} seam(s) carry an assumption the parity gate "
            f"has to check (see notes['assumed'])")

    if plan_notes_calibration:
        # the calibration method is part of the result, not a detail of
        # how it was produced: a parity band means something different
        # depending on how much of the distribution it was scaled from
        plan.notes["calibration"] = plan_notes_calibration
        # and the receipt itself is the repo's, so a structures attachment
        # answers ``precision_spec`` the same way a frontend does
        from .points import precision_spec as _spec
        plan.precision_spec = _spec(collector, plan_notes_calibration)
    if plan_refusals:
        plan.notes.setdefault("refused", []).extend(plan_refusals)
    # Skipping a package this host cannot supply is what lets one plan
    # build everywhere, but it must never be silent: a package that is
    # broken here and one that was never shipped here both come out as
    # "skipped", and only the first is a defect. Carry the raw failures
    # into the plan so they reach the receipt.
    from .impls import unavailable_report
    unavailable = unavailable_report()
    if unavailable:
        plan.notes["kernel_unavailable"] = unavailable
        say(f"{len(unavailable)} kernel package(s) unavailable here: "
            + ", ".join(f"{row['repo']} ({row['error']})"
                        for row in unavailable))
    say(f"bound {len(plan.swaps)} seam(s), "
        f"{len(plan.notes.get('refused', []))} refused")
    return plan


def _attach_brokers(caps, plan, say) -> None:
    from .impls.adaln_producer import AdaLNProducer, bind_style_broker

    groups: dict[tuple, list] = {}
    for path, module in plan.swaps.items():
        if not isinstance(module, AdaLNProducer):
            continue
        cap = caps.get(path, {})
        order = cap.get("order")
        if order is None or not cap.get("pairs"):
            continue
        # one broker per (stream, style width, row count): producers
        # that differ in any of those cannot share a buffer
        key = (_stream_key(cap["pairs"]), int(module.styles.shape[-1]),
               int(module.resid.shape[0]))
        groups.setdefault(key, []).append((order, path, module))

    for key, members in groups.items():
        # the writer is the producer the host calls first, taken from the
        # observed order of the calibration pass — not from the module
        # tree's order, which need not match the forward's
        members.sort(key=lambda entry: entry[0])
        try:
            broker = bind_style_broker([m for _, _, m in members], key[2])
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (f"style_broker[{key[1]}x{key[2]}]", str(refusal)[:200]))
            continue
        if broker is None:
            continue
        plan.notes.setdefault("brokers", []).append(
            {"slots": broker.slots, "rows": key[2], "width": key[1],
             "writer": members[0][1]})
        say(f"style broker: {broker.slots} producer(s) share one "
            f"step-scoped materialisation (writer {members[0][1]})")


_BLOCK_OWNED = ("input_layernorm", "post_attention_layernorm", "mlp")


def _cond_kw(host) -> str:
    """The keyword the host threads its conditioning through."""
    import inspect
    try:
        params = list(inspect.signature(host.forward).parameters)
    except (TypeError, ValueError):
        params = []
    for name in ("adarms_cond", "cond", "temb", "emb"):
        if name in params:
            return name
    return "adarms_cond"


def _bind_block(model, seam, cap, plan):
    """Assemble one decoder_block from its already-bound sublayers."""
    from .impls.decoder_block import bind_decoder_block

    prod_in = plan.swaps.get(seam.path + ".input_layernorm")
    prod_out = plan.swaps.get(seam.path + ".post_attention_layernorm")
    ffn = plan.swaps.get(seam.path + ".mlp")
    if prod_in is None or prod_out is None or ffn is None:
        # a sublayer that did not bind leaves the host block intact:
        # the block structure adds composition, it does not substitute
        # for the region seams it is made of
        return None
    host = _resolve(model, seam.path)
    # the attention sublayer is family-specific (where the attention runs
    # and which rotary form it uses), so it comes from the same adapters
    # that bound the attention core. None keeps the host's attention
    # module, which is the pre-block behaviour.
    attn = None
    for adapter in _ATTENTION_ADAPTERS:
        builder = getattr(adapter, "sublayer", None)
        if builder is None:
            continue
        attn = builder(host)
        if attn is not None:
            break
    if attn is not None:
        _alias_kv_region(plan, seam.path, attn)
    return bind_decoder_block(
        host, prod_in, prod_out, ffn, cond_kw=_cond_kw(host),
        returns_tuple=bool(cap.get("returns_tuple")), attn=attn)


def _alias_kv_region(plan, path: str, sublayer) -> None:
    """Let the packed projections write into the core's packed KV region.

    Both sides can express this (see ``beta.joins``); the qualification
    is that nothing transforms the tensor in between. Value goes straight
    from the projection to the kernel and qualifies. Key does not on this
    family: a rotary embedding runs after the projection, so aliasing it
    would leave untransformed keys in the packed region — writing the
    transformed ones back is the copy this was meant to remove. Hosts
    without a rotary step qualify for both; the attribute is general and
    the qualification is per join.
    """
    from .impls.qkv_pack import PackedLinear

    head = plan.swaps.get(path + ".self_attn.q_proj")
    core = getattr(sublayer, "core", None)
    if not isinstance(head, PackedLinear) or core is None:
        return
    if not hasattr(core, "alias_suffix"):
        return
    _, v_region = core.alias_suffix(key=False, value=True)
    if v_region is None:
        return
    try:
        head.alias_stash(2, v_region)          # sibling order q, k, v
    except (ValueError, RuntimeError) as refusal:
        core._alias_v = False
        plan.notes.setdefault("refused", []).append(
            (path + " [kv alias]", str(refusal)[:200]))
        return
    plan.notes.setdefault("aliased_kv", []).append(path)

    # The joint q|k view is deliberately not enabled here. q and k are
    # one contiguous run of the packed output and take the same rotary
    # arithmetic, so one pass over the pair should replace two — and
    # measured, it replaces nothing: the rotary kernels keep their exact
    # launch counts (180/162/63) because the compiler splits the merged
    # pass back apart, fusing it into each consumer (q is made
    # contiguous for the kernel, k is copied into the packed region).
    # Paired timing: -0.014 ms on 23.1, which is below the margin this
    # stack ships at. Expressing "do this once" in tensor ops does not
    # survive a compiler that re-derives its fusion boundaries from the
    # consumers; the style broker only survived by being opaque, and an
    # opaque wrapper here would be worse, since the rotary would then run
    # as several eager ops instead of one fused kernel. Merging these
    # needs a rotary kernel, not a rearrangement. The capability stays
    # on the impl for a host where the arithmetic is not launch-bound.


def _bind_auto(model, seam, cap, plan, act_scales, negotiate_fp8,
               points=None, fmt=None, fmt_params=None):
    """Route one seam to its impl with the calibrated scales.

    ``points`` is the reduced collector: every scale an impl needs is one
    float looked up by (path, spec point name). No activation tensors are
    threaded through here, because none are needed — the two scales that
    used to be recomputed from held inputs are measured at the GEMM whose
    input they are (:mod:`.points`).

    ``fmt`` is the scheme's per-seam format routing. ``None`` binds the
    structure's default impl; a named format binds that variant instead,
    and a name with no variant for this structure fails loudly — the
    scheme author's error surfaces at bind time, not as accuracy.
    ``fmt_params`` is the decision's recipe payload for that format
    (algorithm parameters, never bytes), handed to the variant's binder.
    """
    dev0 = None
    if model is not None:
        dev0 = next(model.parameters(), torch.empty(0)).device
    if dev0 is not None and dev0.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev0)
        if free < (512 << 20):
            # binding is a transaction against a budget: below the
            # headroom every further seat is refused with the number,
            # instead of eating the remainder and failing later as an
            # unattributable OOM in the first treated forward
            raise ValueError(
                f"insufficient_vram(free={free >> 20}MiB, "
                "headroom=512MiB) — host keeps this seam")

    from .impls.decoder_ffn import fp8_static as ffn_impl
    from .impls.vision_ffn import fp8_static as vis_impl

    def scale(name, path=None):
        return None if points is None else points.scale(path or seam.path,
                                                        name)

    custom = _STRUCTURE_BINDERS.get(seam.structure)
    if custom is not None:
        return custom(model, seam, cap, points=points, fmt=fmt,
                      fmt_params=fmt_params)

    if fmt and not (seam.structure == "qkv_pack"
                    and fmt in ("bf16_pack", "nvfp4_balance")) \
            and not (seam.structure == "vision_ffn"
                     and fmt == "nvfp4_balance") \
            and seam.structure not in ("decoder_ffn", "linear_proj"):
        raise ValueError(f"scheme routed {seam.structure} to format "
                         f"{fmt!r}, which has no impl variant here")

    if seam.structure == "decoder_ffn":
        if fmt in ("w8a16_static", "w4a16_static"):
            if fmt == "w8a16_static":
                from .impls.decoder_ffn import w8a16_static as wq_impl
            else:
                from .impls.decoder_ffn import w4a16_static as wq_impl

            # two callers, two layout conventions: ``seam_weights``
            # serves the fp8 impl transposed ([D, F]); these binders are
            # checkpoint-native ([F, D]) and their dim check passes with
            # the names swapped, so handing them the transposed dict
            # binds a guard with k = F and every call falls back.
            # Transpose back here, at the seam between the conventions.
            w = seam_weights(model, seam)
            w = dict(w,
                     w_gate=w["w_gate"].t().contiguous(),
                     w_up=w["w_up"].t().contiguous(),
                     w_down=w["w_down"].t().contiguous())
            return wq_impl.bind_mlp_seam(
                w, variant=seam.variant,
                original=_resolve(model, seam.path))
        if fmt not in (None, "fp8_static"):
            raise ValueError(f"scheme routed decoder_ffn to format "
                             f"{fmt!r}, which has no impl variant here")
        in_s = scale("x_after_norm")
        hid_s = scale("act_after_mul", seam.path + ".down_proj")
        if in_s is None or hid_s is None:
            return None
        return ffn_impl.bind_mlp_seam(
            seam_weights(model, seam), variant=seam.variant,
            input_scale=in_s, hidden_scale=hid_s,
            original=_resolve(model, seam.path))

    if seam.structure == "vision_ffn":
        fc2 = (seam.fc_attrs or ("fc1", "fc2"))[1]
        if fmt == "nvfp4_balance":
            from .impls.vision_ffn import nvfp4_balance as vis_w4
            chan_in = points.channel_amax(seam.path, "x_after_norm")
            chan_hid = points.channel_amax(
                seam.path + "." + fc2, "hidden_after_act")
            if chan_in is None or chan_hid is None:
                return None
            return vis_w4.bind_mlp_seam(
                seam_weights(model, seam), channel_in=chan_in,
                channel_hidden=chan_hid,
                original=_resolve(model, seam.path),
                **dict(fmt_params or {}))
        in_s = scale("x_after_norm")
        hid_s = scale("hidden_after_act", seam.path + "." + fc2)
        if in_s is None or hid_s is None:
            return None
        rows_seen = points.row_profile(seam.path, "x_after_norm")
        rows_med = rows_seen[len(rows_seen) // 2] if rows_seen else 1 << 30
        if rows_med <= 64:
            # the small-M denoise band: the measured band decision for
            # this box (recorded by a band run, cached per device)
            # routes these seats — never a shape rule alone, never a
            # device name
            from .decisions import lookup as _band_lookup
            if _band_lookup("groot_dit", default="fp8") == "fp4":
                from .impls.vision_ffn import nvfp4_balance as vis_w4
                w = seam_weights(model, seam)
                dev = w["w_fc1"].device
                try:
                    # flat channel vectors: no balance folded — the W4
                    # quantizer alone, judged by the parity gate
                    return vis_w4.bind_mlp_seam(
                        w,
                        channel_in=torch.ones(
                            w["w_fc1"].shape[1], device=dev),
                        channel_hidden=torch.ones(
                            w["w_fc2"].shape[1], device=dev),
                        original=_resolve(model, seam.path),
                        fuse_wire=True)
                except (ValueError, RuntimeError):
                    pass
        return vis_impl.bind_mlp_seam(
            seam_weights(model, seam), input_scale=in_s,
            hidden_scale=hid_s, original=_resolve(model, seam.path))

    if seam.structure == "modnorm_qkv_chain":
        if seam.variant.get("modulation") == "per_token_table":
            from .impls.modnorm_qkv_chain import fp8_ptok_table as chain
            return chain.bind_block_seam(model, seam, points=points)
        # the scale_shift form composes through producer negotiation
        return None

    if seam.structure == "norm_fused":
        from .impls.norm_fused import bind_norm_fused
        return bind_norm_fused(
            _resolve(model, seam.path),
            host_dtypes=(None if points is None
                         else points.seen_dtypes(seam.path, "x")))

    if seam.structure == "linear_proj":
        if fmt == "nvfp4_balance":
            from .impls.linear_proj import nvfp4_balance as proj_w4
            chan = points.channel_amax(seam.path, "x")
            if chan is None:
                return None
            return proj_w4.bind_proj_seam(
                seam_weights(model, seam), channel_amax=chan,
                original=_resolve(model, seam.path),
                **dict(fmt_params or {}))
        if fmt == "w8a16_static":
            # weight-only decode band: no calibration scale to look up,
            # and the weight dict is already the kernel's [N, K] layout
            from .impls.linear_proj import w8a16_static as proj_w8
            return proj_w8.bind_proj_seam(
                seam_weights(model, seam),
                original=_resolve(model, seam.path))
        if fmt not in (None, "fp8_static"):
            raise ValueError(f"scheme routed linear_proj to format "
                             f"{fmt!r}, which has no impl variant here")
        in_s = scale("x")
        if in_s is None:
            return None
        from .impls.linear_proj import fp8_static as proj_impl
        return proj_impl.bind_proj_seam(
            seam_weights(model, seam), input_scale=in_s,
            row_profile=points.row_profile(seam.path, "x"),
            original=_resolve(model, seam.path))

    if seam.structure == "patch_projection":
        from .impls.patch_projection import bind_flat_patch_projection

        forms = tuple(cap.get("patch_inputs", ()))
        if not forms or any(
            form is None or form[0] != seam.dims["K"] for form in forms
        ):
            raise ValueError(
                "patch_projection: calibrated host input is not "
                f"preflattened full-patch rows with K={seam.dims['K']}"
            )
        rows = points.row_profile(seam.path, "x") if points else ()
        dtypes = points.seen_dtypes(seam.path, "x") if points else ()
        return bind_flat_patch_projection(
            seam_weights(model, seam),
            row_profile=rows,
            host_dtypes=dtypes,
            original=_resolve(model, seam.path),
        )

    if seam.structure == "qkv_pack":
        if fmt == "bf16_pack":
            if seam.variant.get("bind") == "module":
                raise ValueError(
                    "qkv_pack bf16_pack v1 supports leaf binding only")
            first = (seam.pack_attrs or ("q_proj",))[0]
            rows_seen = points.row_profile(
                seam.path + "." + first, "x")
            if not rows_seen:
                return None
            from .impls.qkv_pack import bf16 as pack_impl
            block = _resolve(model, seam.path)
            mods = [getattr(block, attr) for attr in seam.pack_attrs]
            parts = pack_impl.bind_qkv_pack(mods, rows=max(rows_seen))
            return {seam.path + "." + attr: mod
                    for attr, mod in zip(seam.pack_attrs, parts)}
        if fmt == "nvfp4_balance":
            if seam.variant.get("bind") == "module":
                raise ValueError(
                    "qkv_pack nvfp4_balance supports leaf binding only")
            from .impls.qkv_pack import nvfp4_balance as pack_w4
            first = (seam.pack_attrs or ("q_proj",))[0]
            chan = points.channel_amax(seam.path + "." + first, "x")
            rows_seen = points.row_profile(seam.path + "." + first, "x")
            if chan is None or not rows_seen:
                return None
            block = _resolve(model, seam.path)
            mods = [getattr(block, a) for a in seam.pack_attrs]
            parts = pack_w4.bind_qkv_pack(
                mods, channel_amax=chan, rows=max(rows_seen),
                **dict(fmt_params or {}))
            return {seam.path + "." + a: m
                    for a, m in zip(seam.pack_attrs, parts)}
        from .impls.qkv_pack import bind_attn_block, bind_qkv_pack
        first = (seam.pack_attrs or ("q_proj",))[0]
        amax = None if points is None else points.amax(
            seam.path + "." + first, "x")
        if amax is None:
            return None
        block = _resolve(model, seam.path)
        rows = points.row_profile(seam.path + "." + first, "x")
        # The packed implementation preallocates scratch/stash storage but
        # the Hub entry accepts any logical M covered by that storage. Use
        # the largest calibrated observation as capacity; choosing the
        # median here turns a valid variable-row call into a guard fallback
        # and can also under-allocate when calibration itself has buckets.
        cap = dict(cap or {}, rows=max(rows) if rows else 1)
        act_scale = torch.tensor(
            [max(amax / 448.0, 1e-8)],
            device=getattr(block, first).weight.device)
        if seam.variant.get("bind") == "module":
            # the whole block: packed projections *and* the attention
            # compute dtype (hosts that run SDPA in fp32 pay for it)
            return {seam.path: bind_attn_block(
                block, act_scale, rows=cap["rows"],
                sdpa_dtype=torch.bfloat16)}
        mods = [getattr(block, a) for a in seam.pack_attrs]
        parts = bind_qkv_pack(mods, act_scale, rows=cap["rows"],
                              in_dtype="bf16_fused_quant")
        return {seam.path + "." + a: m
                for a, m in zip(seam.pack_attrs, parts)}

    if seam.structure == "adaln_producer":
        from .impls.adaln_producer import (bind_adaln_producer,
                                           bind_style_table)
        if not cap.get("pairs"):
            return None
        norm = _resolve(model, seam.path)
        proj = getattr(norm, seam.cond_attr)
        key = _stream_key(cap["pairs"])
        loc = plan.notes.setdefault("_locators", {}).get(key)
        table = bind_style_table(proj, cap["pairs"], locator=loc)
        plan.notes["_locators"][key] = table.locator
        return {seam.path + "." + seam.cond_attr: table}

    return None


class _Eager(torch.nn.Module):
    """Wrap a module so its forward runs outside the compiled region.

    An fp8-emitting seam's arithmetic, if traced by inductor, gets fused
    into fp8 math (illegal on sm120 triton) — and the quantize even
    reaches back across the boundary, so inductor casts the host's own
    gated residual to fp8 to feed it. The hand recipes never hit this
    because the whole denoise block froze to eager. A swapped-in module
    does not inherit that freezing, so fp8 seams declare it. Overriding
    the instance ``forward`` is not enough (dynamo inlines the class
    forward); the disable must sit on a class method, which is what this
    wrapper provides. The kernels are opaque either way, so eager here
    is a graph break, not real work.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def _eager(module):
    return _Eager(module)


def _bind_negotiated(model, p_seam, k_seam, p_cap, points, scale, rows,
                     plan):
    """Bind an fp8 producer and the pack it feeds as one chain.

    This is the combination the structure layer exists for: neither half
    is worth much alone (a small-M fp8 projection barely beats BF16, a
    producer that only reshapes styles saves nothing), but together the
    producer's fused quantize removes the consumer's input quantization
    entirely and hands a clean fp8 seam downstream.
    """
    from .impls.adaln_producer import bind_adaln_producer
    from .impls.qkv_pack import bind_qkv_pack

    norm = _resolve(model, p_seam.path)
    consumer = _resolve(model, k_seam.path)
    key = _stream_key(p_cap["pairs"])
    loc = plan.notes.setdefault("_locators", {}).get(key)
    dim, form = _adaln_form(p_cap, points, p_seam)
    prod = bind_adaln_producer(
        norm, p_cap["pairs"], act_scale=scale, rows=rows,
        dim=dim, locator=loc, norm=form)
    plan.notes["_locators"][key] = prod.locator

    swaps = {p_seam.path: prod}
    if k_seam.structure == "decoder_ffn":
        from .impls.decoder_ffn import fp8_static as ffn_impl
        # the input scale is the one the producer upstream will quantize
        # with — the same number, because the producer's output is this
        # consumer's input; the hidden scale is measured at the down
        # projection whose input it is
        w = seam_weights(model, k_seam)
        bound = ffn_impl.bind_mlp_seam(
            w, variant={**k_seam.variant, "in_dtype": "fp8_static"},
            input_scale=float(scale.item()),
            hidden_scale=points.scale(k_seam.path + ".down_proj",
                                      "act_after_mul"),
            original=consumer)
        swaps[k_seam.path] = bound
        return swaps
    if k_seam.structure == "linear_proj":
        from .impls.linear_proj import fp8_static as proj_impl
        swaps[k_seam.path] = proj_impl.bind_proj_seam(
            seam_weights(model, k_seam),
            input_scale=float(scale.item()),
            row_profile=points.row_profile(k_seam.path, "x"),
            original=consumer,
            in_dtype="fp8_static",
        )
        return swaps
    mods = [getattr(consumer, a) for a in k_seam.pack_attrs]
    parts = bind_qkv_pack(mods, scale, rows=rows,
                          in_dtype="fp8_static")
    # ---- the FP4 wire is a second candidate for the same seats: a
    # producer that norms straight into packed NVFP4 + swizzled scale
    # factors, and a pack that consumes the wire with no quantize of
    # its own. Which chain is faster is a property of this device's
    # GEMM bands at this row count — so it is measured here, on the
    # calibrated shape with the real conditioning, and the winner takes
    # the seats. A candidate that cannot build or run loses by default.
    import os
    if dim >= 512 and not os.environ.get("FRT_DISABLE_FP4_RACE"):
        try:
            prod4 = bind_adaln_producer(
                norm, p_cap["pairs"], act_scale=None, rows=rows,
                dim=dim, locator=prod.locator, norm=form,
                out_format="nvfp4")
            from .impls.qkv_pack import nvfp4_balance as pack_w4
            parts4 = pack_w4.bind_qkv_pack(
                mods, channel_amax=None, rows=rows, wire=True)
            parts4[0].accept_wire(prod4.wire_sfa)
            cond0 = p_cap["pairs"][0][0].detach()
            dev = prod4.wire_sfa.device
            x_bench = torch.randn(rows, dim, device=dev,
                                  dtype=torch.bfloat16)

            def _chain_ms(producer, head, iters=30):
                def once():
                    y = producer(x_bench, cond0)
                    head(y)
                for _ in range(5):
                    once()
                torch.cuda.synchronize()
                start = torch.cuda.Event(True)
                end = torch.cuda.Event(True)
                start.record()
                for _ in range(iters):
                    once()
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end) / iters

            with torch.no_grad():
                a_ms = _chain_ms(prod, parts[0])
                b_ms = _chain_ms(prod4, parts4[0])
            race = {"layer": p_seam.path, "rows": int(rows),
                    "dim": int(dim), "fp8_chain_ms": round(a_ms, 4),
                    "nvfp4_wire_ms": round(b_ms, 4),
                    "winner": "nvfp4_wire" if b_ms < a_ms else
                              "fp8_chain"}
            plan.notes.setdefault("format_race", []).append(race)
            if b_ms < a_ms:
                prod, parts = prod4, parts4
                swaps[p_seam.path] = prod4
        except (ValueError, RuntimeError, KeyError, OSError) as lost:
            plan.notes.setdefault("format_race", []).append(
                {"layer": p_seam.path,
                 "winner": "fp8_chain",
                 "nvfp4_wire": f"refused: {str(lost)[:200]}"})
    swaps.update({k_seam.path + "." + a: m
                  for a, m in zip(k_seam.pack_attrs, parts)})
    return swaps


def _calibration_thunks(forward, frames, samples):
    """Turn the three ways of asking for calibration into one list.

    They differ only in where a frame's input comes from, so they end as
    the same thing: a list of callables, each of which runs the host
    once. Keeping them one axis is what stops "how much calibration" and
    "how to run the host" from becoming two interfaces that can disagree.
    """
    if samples is not None:
        if not callable(forward):
            raise ValueError(
                "auto_swaps: with samples=, forward takes one sample")
        taken = list(samples) if frames is None else [
            s for _, s in zip(range(max(1, frames)), samples)]
        if not taken:
            raise ValueError("auto_swaps: samples is empty")
        return [(lambda s=s: forward(s)) for s in taken], "samples"
    if isinstance(forward, (list, tuple)):
        if not forward:
            raise ValueError("auto_swaps: no forward thunks given")
        if frames is not None and frames != len(forward):
            raise ValueError(
                f"auto_swaps: {len(forward)} thunk(s) given but "
                "observations= and a thunk list are alternatives, "
                "not a pair; the thunks decide")
        return list(forward), "thunks"
    if not callable(forward):
        raise ValueError("auto_swaps: forward must be callable")
    return [forward] * max(1, frames or 1), "forward"


def _adaln_form(cap, points, seam) -> tuple[int, str]:
    """Read the producer's width and form off the calibration.

    Both were assumed before: the form was hard-coded to rms and the
    width taken as ``style_width // 3``. That holds only where the style
    carries three parts. A host whose adaptive norm emits (scale, shift)
    — the layer form — got bound as rms at two thirds of its real width,
    and the plan built cleanly and then could not run. It took a second
    host and an actual forward to see it, because nothing on the way
    there had to disagree.

    The norm's own input says how wide it is, and the ratio to the style
    says which form it is. Neither is a guess.
    """
    dim = points.width(seam.path, "x")
    if dim is None:
        raise ValueError(
            "adaln_producer: the norm's own input was never observed, so "
            "the form cannot be told from the style width alone")
    style_width = int(cap["pairs"][0][1].shape[-1])
    if style_width == 3 * dim:
        return dim, "rms"           # scale, shift, gate
    if style_width == 2 * dim:
        return dim, "layer"         # scale, shift
    raise ValueError(
        f"adaln_producer: style width {style_width} is neither two nor "
        f"three times the norm width {dim} — the modulation is a shape "
        "this structure does not model")


def _stream_key(pairs) -> str:
    """Identify the conditioning stream a producer was calibrated on.

    Locators were keyed by seam family, which gives every family its own
    lookup even when they all read the same conditioning — the two norms
    of one block among them. Keying by the observed conditioning instead
    shares one locator across the whole stream. It is safe by
    construction rather than by convention: the key is a digest of the
    conditioning rows themselves, so two seams share a locator only when
    they saw byte-identical inputs, and identical inputs resolve to
    identical indices whichever seam built the table.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for cond, _ in pairs:
        c = cond.detach().reshape(-1, cond.shape[-1]).to(torch.float32)
        digest.update(c.cpu().numpy().tobytes())
    return digest.hexdigest()


def _layer_of(path: str) -> str:
    """The parent layer key: a.layers.7.self_attn -> a.layers.7."""
    import re
    m = re.search(
        r"((?:.*\.)?(?:layers|transformer_blocks)\.\d+)\.", path)
    return m.group(1) if m else path.rsplit(".", 1)[0]


def _feeds_attention(path: str) -> bool:
    """An adaln producer that feeds attention (input_layernorm) rather
    than the MLP (post_attention_layernorm)."""
    leaf = path.rsplit(".", 1)[-1]
    return "input" in leaf or leaf in ("norm1", "ln1")
