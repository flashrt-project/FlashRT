"""Calibration points: the spec names them, discovery locates them.

A structure's spec already declares what has to be observed to calibrate
it — ``calibration.points`` in ``catalog/<name>/structure.yaml``, named by
position in the structure's own dataflow (``x_after_norm``,
``act_after_mul``) rather than by any host's module names. This module is
the small amount of glue that turns those names into hooks on a
particular host, and collects the statistic the house calibration path
expects.

Why the split lives where it does, for the backends still to come:

    catalog/<name>/structure.yaml   the point's *name*, i.e. its position
                                    in this structure's dataflow. Backend-
                                    independent: ``act_after_mul`` means
                                    the same thing in a GGML graph as in
                                    a torch module tree.
    here + discovery                where that position sits on this host.
    impls/<name>/<backend>.py       what statistic to take there and how
                                    to reduce it — per-tensor amax for
                                    FP8, something else entirely for a
                                    backend whose quantisation is
                                    per-block or driven by an importance
                                    matrix.

Nothing about the *statistic* is decided here, and nothing about the
*position* is decided in an implementation. That is the whole point: the
positions change when a structure's definition changes (rarely, with a
version bump), while the statistic changes with every new quantisation
format (often, per backend).

The reduction is two-level, and both levels are the house's
(``flash_rt.core.calibration``, ``docs/calibration.md``):

  within one sample   max over every call the host makes to that point.
                      Required, not chosen: §4.2 of the calibration doc
                      records that per-step scales on a flow-matching host
                      gave the compiler inconsistent shapes and crashed
                      it. One forward covers every step, and the max
                      across them is the sample's amax.
  across samples      ``accumulate_amax(per_sample, percentile)``. Kept as
                      one vector per sample so the percentile is possible
                      at all — a running max across samples destroys the
                      per-sample values as it produces them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

#: how each spec point name is reached from a discovered seam. The name on
#: the left must appear in that structure's ``calibration.points``; the
#: right-hand side is resolved from slots discovery already established,
#: never from a module-name guess.
_AT_SEAM = ("", "input")


@dataclass(frozen=True)
class Point:
    """One observation site on a host: a spec point name, resolved."""

    name: str
    path: str
    side: str = "input"          # "input" | "output"

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.name)


def _child(seam_path: str, attr: str) -> str:
    return f"{seam_path}.{attr}" if attr else seam_path


def resolve(seam: Any, spec_points: Sequence[str]) -> list[Point]:
    """Locate this seam's spec-declared points on its host.

    Raises if the structure declares a point this module cannot place, so
    a spec that grows a point is a loud failure rather than a silently
    uncalibrated seam.
    """
    structure = seam.structure
    placed: dict[str, Point] = {}

    def put(name: str, attr: str = "", side: str = "input") -> None:
        placed[name] = Point(name, _child(seam.path, attr), side)

    if structure == "decoder_ffn":
        put("x_after_norm")
        # the gated activation is the down projection's input; measuring it
        # there is what removes the need to keep the seam's input around
        # and recompute gate/up over it
        put("act_after_mul", "down_proj")
    elif structure == "vision_ffn":
        put("x_after_norm")
        put("hidden_after_act", (seam.fc_attrs or ("fc1", "fc2"))[1])
    elif structure == "qkv_pack":
        # siblings share one input; the first one the host calls sees it
        put("x", (seam.pack_attrs or ("q_proj",))[0])
    elif structure in ("linear_proj", "patch_projection", "norm_fused"):
        put("x")
    elif structure == "adaln_producer":
        # ``cond`` is not an amax point: the step table is captured content,
        # not a statistic, and is collected by the conditioning hook
        put("x")
    elif (structure == "modnorm_qkv_chain"
            and seam.variant.get("modulation") == "per_token_table"):
        # the table form owns the whole block, so the four static scales
        # its composition needs are measured at the block's own sublayer
        # inputs — the same real-distribution sites the sublayers would
        # calibrate at if bound individually
        put("attn_in", "attn1.to_q")
        put("o_in", "attn1.to_out.0")
        put("ffn_in", "ffn")
        put("ffn_hid", "ffn.net.2")
    elif structure in ("decoder_block", "modnorm_qkv_chain"):
        pass                      # composed; its sublayers carry the points

    unplaced = [p for p in spec_points if p not in placed
                and not (structure == "adaln_producer" and p == "cond")
                and not (structure == "attention_core")
                and not (structure == "cadence_static")]
    if unplaced:
        raise ValueError(
            f"{structure}: spec declares calibration point(s) {unplaced} "
            "that cannot be located on a host — either the spec grew a "
            "point or this seam's slots were not discovered")
    return [placed[p] for p in spec_points if p in placed]


@dataclass
class Collector:
    """Per-sample amax vectors for a set of points, house-shaped.

    ``sample_amax`` is reduced with a max while one sample runs, snapshot
    into ``per_sample`` when it ends, and reduced across samples by the
    house percentile. ``rows`` and ``dtypes`` ride along because they are
    the other two things a bind needs from the same forward, and they are
    observations of one scalar each rather than statistics.
    """

    points: list[Point] = field(default_factory=list)
    per_sample: list[Any] = field(default_factory=list)
    keys: list[tuple[str, str]] | None = None
    _cur: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)
    rows: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    widths: dict[tuple[str, str], int] = field(default_factory=dict)
    dtypes: dict[tuple[str, str], set] = field(default_factory=dict)
    final: dict[tuple[str, str], float] = field(default_factory=dict)
    #: per-point extra statistic requests, keyed ``"path|name"`` — objects
    #: with ``.stat`` / ``.granularity`` (a scheme's ``PointStat``). The
    #: scalar amax is always collected regardless (it keeps the per-sample
    #: vectors aligned and costs one float); a channel request adds a
    #: per-channel track beside it, it does not replace it.
    request: dict[str, Any] = field(default_factory=dict)
    _cur_chan: dict[tuple[str, str], torch.Tensor] = field(
        default_factory=dict)
    _cur_sm: dict[tuple[str, str], tuple] = field(default_factory=dict)
    chan_samples: dict[tuple[str, str], list] = field(default_factory=dict)
    sm_samples: dict[tuple[str, str], list] = field(default_factory=dict)
    chan_final: dict[tuple[str, str], Any] = field(default_factory=dict)
    sm_final: dict[tuple[str, str], Any] = field(default_factory=dict)

    # ---- capture -----------------------------------------------------

    def _record(self, point: Point, x: torch.Tensor) -> None:
        if not torch.is_tensor(x):
            return
        key = point.key
        req = self.request.get(f"{point.path}|{point.name}")
        if req is not None and getattr(req, "granularity", None) == "channel":
            flat = x.detach().float().reshape(-1, x.shape[-1])
            if req.stat == "amax":
                chan = flat.abs().amax(dim=0)
                prev = self._cur_chan.get(key)
                self._cur_chan[key] = (chan if prev is None
                                       else torch.maximum(prev, chan))
            elif req.stat == "second_moment":
                # per-sample (sum of squares, token count) so the
                # per-sample values survive to the reduction, same
                # discipline as the amax vectors
                sq = (flat * flat).sum(dim=0)
                prev = self._cur_sm.get(key)
                self._cur_sm[key] = ((sq, flat.shape[0]) if prev is None
                                     else (prev[0] + sq,
                                           prev[1] + flat.shape[0]))
        amax = x.detach().float().abs().max()
        prev = self._cur.get(key)
        self._cur[key] = amax if prev is None else torch.maximum(prev, amax)
        self.rows.setdefault(key, []).append(
            int(x.numel() // x.shape[-1]) if x.ndim else 1)
        if x.ndim:
            self.widths.setdefault(key, int(x.shape[-1]))
        self.dtypes.setdefault(key, set()).add(x.dtype)

    def hooks(self, resolve_module: Callable[[str], torch.nn.Module]) -> list:
        """Install one hook per point; caller removes them."""
        handles = []
        for point in self.points:
            target = resolve_module(point.path)
            if point.side == "output":
                handles.append(target.register_forward_hook(
                    lambda m, a, out, p=point: self._record(p, out)))
            else:
                handles.append(target.register_forward_pre_hook(
                    lambda m, a, p=point: self._record(
                        p, a[0] if a else None)))
        return handles

    def end_sample(self) -> None:
        """Freeze this sample's amax into one vector, house ordering."""
        import numpy as np

        if self.keys is None:
            self.keys = sorted(self._cur)
        missing = [k for k in self.keys if k not in self._cur]
        if missing:
            raise ValueError(
                f"calibration sample reached {len(self._cur)} of "
                f"{len(self.keys)} points; {missing[:3]} were not called. "
                "The host took a different path on this sample, so the "
                "per-sample vectors cannot be aligned")
        self.per_sample.append(
            np.array([float(self._cur[k]) for k in self.keys],
                     dtype=np.float32))
        self._cur = {}
        for k, chan in self._cur_chan.items():
            self.chan_samples.setdefault(k, []).append(
                chan.cpu().numpy().astype(np.float32))
        self._cur_chan = {}
        for k, (sq, n) in self._cur_sm.items():
            self.sm_samples.setdefault(k, []).append(
                (sq.cpu().numpy().astype(np.float64), int(n)))
        self._cur_sm = {}

    # ---- reduce ------------------------------------------------------

    def reduce(self, percentile: float, *, verbose: bool = False,
               label: str = "structures") -> dict[str, Any]:
        """Percentile-reduce across samples using the house helpers."""
        from flash_rt.core.calibration import (accumulate_amax,
                                               check_scale_ceiling,
                                               format_summary,
                                               summarize_amax_dispersion)

        if not self.per_sample:
            return {"points": 0, "samples": 0}
        final = accumulate_amax(self.per_sample, percentile=percentile)
        self.final = {k: float(v) for k, v in zip(self.keys or [], final)}
        # channel amax reduces with the same house helper, per point —
        # it is elementwise over vectors, the vector is just per-channel
        # instead of per-point
        for k, samples in self.chan_samples.items():
            self.chan_final[k] = accumulate_amax(samples,
                                                 percentile=percentile)
        # a second moment is a mean, not an amax: combine the per-sample
        # (sum, count) pairs. Per-sample values are kept so a robust
        # cross-sample reduction stays possible later.
        for k, samples in self.sm_samples.items():
            total = sum(s for s, _ in samples)
            count = sum(n for _, n in samples)
            self.sm_final[k] = total / max(count, 1)
        note: dict[str, Any] = {
            "points": len(self.final),
            "samples": len(self.per_sample),
            "percentile": percentile,
            "method": ("single_frame" if len(self.per_sample) == 1
                       else "percentile"),
        }
        if len(self.per_sample) > 1:
            summary = summarize_amax_dispersion(self.per_sample, final)
            note["dispersion"] = summary
            if verbose:
                print(f"[structures] {format_summary(summary)}", flush=True)
        # the house diagnostic, on the same scales the impls will bake in
        offenders = check_scale_ceiling(
            {f"{p}|{n}": v / 448.0 for (p, n), v in self.final.items()},
            label=label)
        if offenders:
            note["scale_ceiling_offenders"] = [n for n, _ in offenders]
        return note

    # ---- what a bind asks for ---------------------------------------

    def amax(self, path: str, name: str) -> float | None:
        return self.final.get((path, name))

    def channel_amax(self, path: str, name: str):
        """Per-channel amax vector for a point that requested it."""
        return self.chan_final.get((path, name))

    def second_moment(self, path: str, name: str):
        """Per-channel E[x^2] for a point that requested it (imatrix
        statistic: what each input column contributes, on real data)."""
        return self.sm_final.get((path, name))

    def scale(self, path: str, name: str, *,
              fp8_max: float = 448.0) -> float | None:
        """The FP8 per-tensor scale for one point, house formula."""
        value = self.final.get((path, name))
        return None if value is None else max(value / fp8_max, 1e-8)

    def row_profile(self, path: str, name: str) -> list[int]:
        return sorted(self.rows.get((path, name), []))

    def seen_dtypes(self, path: str, name: str) -> set:
        return self.dtypes.get((path, name), set())

    def width(self, path: str, name: str) -> int | None:
        """The last-dim this point was observed at — a shape, not a stat."""
        return self.widths.get((path, name))


def measure(run_once: Callable[[], Any], points: Sequence[Point],
            resolve_module: Callable[[str], torch.nn.Module], *,
            samples: int = 1, percentile: float = 99.9,
            verbose: bool = False,
            label: str = "structures") -> tuple[Collector, dict[str, Any]]:
    """Run the host ``samples`` times under point hooks and reduce.

    The one calibration pass, shaped like every other one in this repo:
    hooks on, one forward per sample with the per-sample vector snapshot
    at the end of each, hooks off in a ``finally`` so a raising thunk
    cannot leave them on the model.
    """
    collector = Collector(points=list(points))
    handles = collector.hooks(resolve_module)
    try:
        with torch.no_grad():
            for _ in range(max(1, samples)):
                run_once()
                collector.end_sample()
    finally:
        for handle in handles:
            handle.remove()
    return collector, collector.reduce(percentile, verbose=verbose,
                                       label=label)


def precision_spec(collector: Collector, note: dict[str, Any]):
    """Package the reduced scales as the house ``ModelPrecisionSpec``.

    The receipt format is the repo's, so a structures attachment answers
    ``precision_spec`` the same way a frontend does — same fields, same
    ``calibration_method`` vocabulary, same introspection.
    """
    import numpy as np
    from flash_rt.core.precision_spec import ModelPrecisionSpec, PrecisionSpec

    method = note.get("method")
    samples = note.get("samples")
    pct = note.get("percentile") if method == "percentile" else None
    specs = {
        f"{path}|{name}": PrecisionSpec(
            dtype="fp8_e4m3", granularity="per_tensor", scheme="symmetric",
            scale_source="calibration",
            scale=np.array([max(value / 448.0, 1e-8)], dtype=np.float32),
            calibration_method=method, calibration_samples=samples,
            calibration_percentile=pct)
        for (path, name), value in collector.final.items()
    }
    # these are activation scales measured at GEMM inputs; weight scales are
    # derived at bind time from the weights themselves and belong to the
    # other bucket, which this layer does not populate
    return ModelPrecisionSpec(activation_specs=specs, source="calibration")
