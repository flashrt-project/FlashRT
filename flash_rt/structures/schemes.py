"""Quantisation schemes: how statistics become per-seam decisions.

A scheme owns exactly two questions and nothing else:

1. **What statistic does each calibration point need?**
   (:meth:`QuantScheme.statistics`) — amax for FP8-style static scales, a
   per-channel second moment for imatrix-style weight quantisation, or
   ``None`` for formats that quantise dynamically in-kernel and need no
   calibration at that point (this repo's NVFP4 activation path computes
   per-block scale factors at runtime). The statistic *discipline* is not
   the scheme's to change: per-sample reduction then a cross-sample
   percentile, one vector held per sample, never activations.

2. **Given the reduced statistics, what happens at each seam?**
   (:meth:`QuantScheme.decide`) — bind with these values, or keep the
   host module ("this layer stays at host precision" is a decision, not
   a failure).

What a scheme does **not** own: bytes. Scale-factor memory layouts,
sub-normal handling in packed formats, kernel selection, M-dispatch
tables — all execution detail, owned by the impl variant that consumes
the decision. The same decision can be executed by different kernels;
that boundary is what keeps schemes portable across backends.

Schemes are registered by name and selected at the door::

    structures.auto_swaps(model, forward, scheme="fp8_static")

Registering a scheme adds no calibration entry point: the calibration
axis (``forward`` / ``samples``) is fixed, and a scheme only declares
what to measure along it and consumes the result.
"""

from __future__ import annotations

import statistics as _stats
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = ["PointStat", "Decision", "QuantScheme", "Fp8Static",
           "NoQuant", "Bf16Structural", "W8A16Decode", "W4A16Decode",
           "Nvfp4Awq", "Nvfp4Balance",
           "register", "get", "names", "resolve_auto", "validate_request"]

#: statistics the collector can currently execute. Granularities other
#: than per-tensor (per-channel, per-block16) are part of the declared
#: interface — NVFP4 weight scale factors are per-16-block, imatrix is
#: per-channel — but the collector does not measure them yet, so a
#: scheme requesting one fails loudly at plan time instead of silently
#: getting per-tensor numbers with the wrong shape.
_EXECUTABLE = {("amax", "tensor"), (None, "tensor"),
               ("amax", "channel"), ("second_moment", "channel")}


@dataclass(frozen=True)
class PointStat:
    """What one calibration point should measure.

    ``stat`` is ``"amax"`` (this repo's static-scale statistic),
    ``"second_moment"``, ``"histogram"``, or ``None`` — ``None`` means
    the format quantises this point dynamically at runtime and wants no
    calibration data at all. ``granularity`` is ``"tensor"``,
    ``"channel"`` or ``"block16"``.
    """

    stat: str | None = "amax"
    granularity: str = "tensor"


@dataclass
class Decision:
    """What :meth:`QuantScheme.decide` hands back.

    ``keep_host`` lists seam paths that stay on the host module at host
    precision — a first-class outcome, recorded in the plan notes, not a
    refusal. ``reasons`` says why, per path, so the receipt can print it.
    ``formats`` routes a seam to a named impl variant instead of the
    structure's default (``"w8a16_static"`` on a ``decoder_ffn`` seam
    binds the weight-only path). A seam routed to a non-default format
    is excluded from FP8 seam negotiation — a chain shares one scale and
    one wire dtype, and a member in another format has neither. An
    unknown format fails loudly at bind time.

    ``params`` carries per-seam recipe parameters for the routed format
    (an algorithm's ``alpha``, clamp bounds, recipe name). They are
    *values of the decision*, handed to the impl at bind and recorded in
    the receipt — never read from environment variables, and never the
    bytes: how a parameterised algorithm is executed stays with the
    impl.
    """

    keep_host: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)
    formats: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


class QuantScheme:
    """Base scheme: amax everywhere, bind everything.

    Subclass and override the two methods; do not add entry points.
    """

    name = "base"

    #: Optional format for an MTP draft head's expert bank / projections
    #: (``decode_loop.enable_mtp`` consumes this vocabulary). ``None``
    #: keeps the draft weights BF16 — the conservative arm. The draft
    #: answers to acceptance length alone: the verify pass anchors the
    #: output stream, so both measured arms (``"bf16"``,
    #: ``"nvfp4_dynamic"`` — AL-equal on the record) are quality-safe,
    #: and the choice trades memory for nothing else. The draft's
    #: private W8 head view is part of the member's fixed form, not a
    #: scheme decision: the model's own head stays on the step/verify
    #: numeric family in every scheme.
    mtp_projection_format: str | None = None

    #: Optional format for the gated-delta layer's packed projections.
    #: The fused-layer adapter consults this; ``None`` keeps them at
    #: host precision. This is a scheme attribute, not an impl default —
    #: quantising those projections is a precision decision.
    gdn_projection_format: str | None = None

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        """Per point key (``"path|name"``): what to measure there."""
        return {f"{p.path}|{p.name}": PointStat() for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        """``report`` is per seam path: its points' reduced statistics."""
        return Decision()


class Fp8Static(QuantScheme):
    """The default: static per-tensor FP8, exactly the shipped behaviour.

    ``keep_outliers`` turns the house scale-ceiling diagnostic into a
    decision: seams owning a point whose reduced amax sits more than
    ``keep_outliers`` times above the median of all points stay at host
    precision. The criterion is the one ``check_scale_ceiling`` already
    warns with (20.0 there); this consumes it instead of only saying it.
    ``None`` (the default) keeps nothing and binds identically to the
    behaviour before schemes existed.
    """

    name = "fp8_static"

    def __init__(self, keep_outliers: float | None = None) -> None:
        self.keep_outliers = keep_outliers

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        if not self.keep_outliers or not report:
            return Decision()
        values = [v for pts in report.values() for v in pts.values()
                  if v is not None and v > 0]
        if not values:
            return Decision()
        median = _stats.median(values)
        keep, reasons = [], {}
        for seam_path, pts in report.items():
            worst = max(((k, v) for k, v in pts.items() if v is not None),
                        key=lambda kv: kv[1], default=None)
            if worst is not None and worst[1] > self.keep_outliers * median:
                keep.append(seam_path)
                reasons[seam_path] = (
                    f"{worst[0]} amax {worst[1]:.4g} > "
                    f"{self.keep_outliers:g}x median {median:.4g}; "
                    f"kept at host precision")
        return Decision(keep_host=tuple(keep), reasons=reasons)


class W8A16Decode(QuantScheme):
    """Weight-only INT8, activations untouched — the decode-band recipe.

    Needs no calibration data at all (quantisation is per-output-channel
    on weights, done at bind time), so every point declares ``None``.
    Routes ``decoder_ffn`` seams to the ``w8a16_static`` impl, whose own
    M-dispatch sends decode shapes to the kernel and prefill back to the
    host, and ``linear_proj`` seams (the attention Q/K/V/O family) to
    its projection twin under the same band contract. Other structures
    stay at host precision: this scheme is the decode recipe, not a
    whole-host FP8 replacement.

    A ``decoder_ffn`` seam is recognised by the point its spec declares
    (``act_after_mul`` — the gated activation), which is
    backend-independent by construction; a ``linear_proj`` seam by the
    structure name the report entry carries.
    """

    name = "w8a16_decode"
    _format = "w8a16_static"
    #: format for linear_proj seams, or None to keep them at host
    #: precision (the 4-bit twin: its linear auto band is too narrow to
    #: route blind — M in [1, 2] with strict N/K limits — so it stays
    #: host until a measured table says otherwise)
    _linear_format: str | None = "w8a16_static"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat(None) for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        formats, keep = {}, []
        for seam_path, pts in report.items():
            if any(k.endswith("|act_after_mul") for k in pts):
                formats[seam_path] = self._format
            elif (self._linear_format is not None
                    and getattr(pts, "structure", None) == "linear_proj"):
                formats[seam_path] = self._linear_format
            else:
                keep.append(seam_path)
        return Decision(keep_host=tuple(keep),
                        reasons={p: f"{self.name} binds decode-band "
                                    f"GEMM seams only"
                                 for p in keep},
                        formats=formats)


class W4A16Decode(W8A16Decode):
    """Weight-only NVFP4 (E2M1 packed + block scale factors) twin of
    :class:`W8A16Decode` — same decode band, same M-dispatch, half the
    weight bytes. The ``flashrt/weight-only-ffn`` package quantises
    weights per 16-element block at bind time; activations stay BF16,
    so like the INT8 twin it needs no calibration data.
    """

    name = "w4a16_decode"
    _format = "w4a16_static"
    _linear_format = None


class Nvfp4Awq(QuantScheme):
    """NVFP4 with activation-aware per-input-channel balance.

    Requests per-channel amax at every calibration point (the collector
    measures it; :func:`validate_request` admits it), and routes
    ``decoder_ffn`` seams to the ``nvfp4_awq`` impl variant with the
    recipe parameters as the decision's payload. The impl consumes the
    channel statistics and calls the one shared algorithm
    (``flash_rt.core.quantization.fit_input_channel_balance``) at bind;
    this scheme owns the *decision*, not the fold and not the bytes.

    ``recipe="balance"`` is the production formula validated on Pi0.5
    (Thor FP4) and Motus video FFN — activation-only channel balance.
    ``"smoothquant"`` names the legacy activation/weight ratio and is
    accepted for experiments; there is no environment-variable fork.
    Other structures stay at host precision until their NVFP4 variants
    land — extending the routing is a change to this method, in one
    place.
    """

    name = "nvfp4_awq"

    def __init__(self, alpha: float = 0.5,
                 clamp: tuple[float, float] = (0.25, 4.0),
                 recipe: str = "balance") -> None:
        if recipe not in ("balance", "smoothquant"):
            raise ValueError(f"unknown recipe {recipe!r}; "
                             f"known: balance, smoothquant")
        self.alpha = float(alpha)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        self.recipe = recipe

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat("amax", "channel")
                for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        formats, params, keep = {}, {}, []
        payload = {"alpha": self.alpha, "clamp": list(self.clamp),
                   "recipe": self.recipe}
        for seam_path, pts in report.items():
            if any(k.endswith("|act_after_mul") for k in pts):
                formats[seam_path] = "nvfp4_awq"
                params[seam_path] = dict(payload)
            else:
                keep.append(seam_path)
        return Decision(keep_host=tuple(keep),
                        reasons={p: "nvfp4_awq routes decoder_ffn only "
                                    "in this version"
                                 for p in keep},
                        formats=formats, params=params)


class Nvfp4Balance(QuantScheme):
    """NVFP4 W4A4 with activation-only channel balance at every GEMM.

    The recorded W4 chain recipe as a scheme decision: projection and
    FFN seams (``qkv_pack`` / ``vision_ffn`` / ``linear_proj``) route to
    their ``nvfp4_balance`` impl variants — weights folded with the
    balance fitted on calibrated per-channel amax, then packed to NVFP4;
    activations quantized dynamically per call with per-block scale
    factors. The channel statistic feeds the balance, never a scale, so
    nothing static exists to drift across a schedule. Everything else
    stays at host precision: this is the half-weight-bytes showcase
    band, not a whole-host replacement.
    """

    name = "nvfp4_balance"

    def __init__(self, alpha: float = 0.5,
                 clamp: tuple[float, float] = (0.25, 4.0),
                 fuse_ffn_wire: bool = False) -> None:
        self.alpha = float(alpha)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        # the FFN's FP4-wire chain (GEMM emits bias+GELU re-quantized,
        # the second GEMM consumes it) drops fc2's input-side balance —
        # a numerics change, so it is a scheme decision the receipt
        # records, never a silent flip on symbol presence
        self.fuse_ffn_wire = bool(fuse_ffn_wire)
        if fuse_ffn_wire:
            self.name = "nvfp4_balance_wire"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat("amax", "channel")
                for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        formats, params, keep = {}, {}, []
        payload = {"alpha": self.alpha, "clamp": list(self.clamp)}
        for seam_path, pts in report.items():
            if getattr(pts, "structure", None) in (
                    "qkv_pack", "vision_ffn", "linear_proj"):
                formats[seam_path] = "nvfp4_balance"
                params[seam_path] = dict(payload)
                if (self.fuse_ffn_wire
                        and pts.structure == "vision_ffn"):
                    params[seam_path]["fuse_wire"] = True
            else:
                keep.append(seam_path)
        return Decision(keep_host=tuple(keep),
                        reasons={p: "nvfp4_balance binds projection and "
                                    "FFN GEMM seams only"
                                 for p in keep},
                        formats=formats, params=params)


class NoQuant(QuantScheme):
    """Quantisation off: every quantised seam stays at host precision.

    This is the explicit off-switch, not a degraded mode. Structures
    that are pure fusion (the attention core, cadence buffers) never
    consult a scheme decision and attach as usual — a BF16/FP16 host
    under this scheme still gets every fusion structure, it just gets
    no quantised GEMMs. Zero calibration, zero kernel dependencies.
    """

    name = "none"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat(None) for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        keep = tuple(report)
        return Decision(keep_host=keep,
                        reasons={p: "quantisation off (scheme 'none')"
                                 for p in keep})


class W4A4Decode(NoQuant):
    """Mixed decode band for hosts built around gated-delta layers.

    Two decisions on top of the ``none`` scheme, both decode-band only.
    The fused gated-delta layer's packed input projection and output
    projection — the bandwidth-dominant GEMVs — go through the dynamic
    NVFP4 path (weights packed at bind time, activations quantised per
    call). The attention/head ``linear_proj`` seams go to the INT8
    weight-only band instead: their output feeds attention scores and
    logits, where the denser grid is the right conservatism. Prefill
    dispatches back to the host either way, and everything else stays
    at host precision.
    """

    name = "w4a4_decode"
    gdn_projection_format = "nvfp4_dynamic"
    _linear_format: str | None = "w8a16_static"
    #: one-way arm: after the FP4 band binds (and only then), the
    #: layer's BF16 projection weights are released — ~11GB on the 27B
    #: host, trading exact detach for the headroom a draft head and a
    #: W8 lm_head need to coexist. Never a default; the receipt says so.
    gdn_release_host_weights = False

    def __init__(self, release_host_weights: bool = False) -> None:
        if release_host_weights:
            self.gdn_release_host_weights = True
            self.name = "w4a4_decode_release"

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        formats, keep = {}, []
        for seam_path, pts in report.items():
            if (self._linear_format is not None
                    and getattr(pts, "structure", None) == "linear_proj"):
                formats[seam_path] = self._linear_format
            else:
                keep.append(seam_path)
        return Decision(keep_host=tuple(keep),
                        reasons={p: f"{self.name} binds decode-band "
                                    f"GEMM seams only"
                                 for p in keep},
                        formats=formats)


class Bf16Structural(QuantScheme):
    """No quantisation; retain only BF16 structural fusions."""

    name = "bf16_structural"

    def statistics(self, points: Sequence) -> dict[str, PointStat]:
        return {f"{p.path}|{p.name}": PointStat(None) for p in points}

    def decide(self, report: Mapping[str, Mapping[str, float]]) -> Decision:
        formats, keep, reasons = {}, [], {}
        for seam_path, pts in report.items():
            structure = getattr(pts, "structure", None)
            if structure == "qkv_pack":
                formats[seam_path] = "bf16_pack"
            elif structure in ("decoder_ffn", "vision_ffn", "linear_proj"):
                keep.append(seam_path)
                reasons[seam_path] = "bf16_structural introduces no quantisation"
        return Decision(keep_host=tuple(keep), reasons=reasons,
                        formats=formats)


_REGISTRY: dict[str, QuantScheme] = {}


def register(name: str, scheme: QuantScheme) -> None:
    """Register a scheme instance under ``name`` (last write wins)."""
    _REGISTRY[name] = scheme


def get(name: str) -> QuantScheme:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown quantisation scheme {name!r}; "
                       f"registered: {sorted(_REGISTRY)}") from None


def names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def validate_request(request: Mapping[str, PointStat]) -> None:
    """Refuse loudly what the collector cannot measure yet.

    A scheme asking for a per-block or per-channel statistic must not
    silently receive per-tensor numbers — wrong-shaped scales bind and
    run, and the error surfaces as accuracy nobody can trace. The wall
    stays until the collector grows that granularity.
    """
    bad = {key: ps for key, ps in request.items()
           if (ps.stat, ps.granularity) not in _EXECUTABLE}
    if bad:
        k, ps = next(iter(bad.items()))
        raise NotImplementedError(
            f"scheme requests ({ps.stat}, {ps.granularity}) at {k} "
            f"(and {len(bad) - 1} more point(s)); the collector currently "
            f"measures only per-tensor amax. Extending it is the "
            f"supported path — do not fall back to per-tensor silently.")


def resolve_auto() -> str:
    """Resolve the ``"auto"`` profile: highest performance this device
    can execute, from the registered names.

    FP8-capable hardware (SM >= 89) gets ``fp8_static`` — bit-identical
    to the behaviour before ``auto`` existed. Anything else gets
    ``none``: fusion structures still attach, quantised seams stay at
    host precision, and the receipt records why. The resolution table is
    deliberately one function so a future profile that measures faster
    (an FP4 mix, say) is promoted by editing exactly one line.
    """
    try:
        from flash_rt.core.utils.hardware import supports_fp8
        fp8 = bool(supports_fp8())
    except Exception:
        fp8 = False
    return "fp8_static" if fp8 else "none"


register("fp8_static", Fp8Static())
register("fp8_static_keep_outliers", Fp8Static(keep_outliers=20.0))
register("w8a16_decode", W8A16Decode())
register("w4a16_decode", W4A16Decode())
register("w4a4_decode", W4A4Decode())
register("w4a4_decode_release", W4A4Decode(release_host_weights=True))
register("none", NoQuant())
register("bf16_structural", Bf16Structural())
register("nvfp4_awq", Nvfp4Awq())
register("nvfp4_balance", Nvfp4Balance())
register("nvfp4_balance_wire", Nvfp4Balance(fuse_ffn_wire=True))
