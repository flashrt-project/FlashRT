"""Explicit front door: ``structures.get(name)`` — pull one structure,
bind it to one host module, plug it in yourself.

This is the single-site, fully visible counterpart of ``attach``. The
consumption mirrors ``kernels.get_kernel``:

    ffn = structures.get("decoder_ffn")
    new_mlp = ffn.bind(model.model.layers[3].mlp, calibration=[x1, x2])
    model.model.layers[3].mlp = new_mlp        # your swap, your call

``bind`` extracts the weights from the module you hand it, calibrates on
the samples you provide, builds the fused replacement, and self-checks
it against the original module on those same samples. A replacement
that misses the parity gate raises ``GateRefused`` instead of returning
— the explicit path fails loudly, it never hands back a bad part.
``attach`` is this same operation, batched over every discovered site
with the net-win A/B added.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .discover import _VISION_PROJ, activation_for
from .gates import parity_metrics
from .registry import StructureSpec, load


class GateRefused(RuntimeError):
    """The bound replacement did not meet the parity gate."""


class StructureHandle:
    """One pulled structure, ready to bind to host modules."""

    def __init__(self, spec: StructureSpec):
        self.spec = spec
        self.name = spec.name

    def __repr__(self) -> str:
        return f"StructureHandle({self.name!r}, v{self.spec.version})"

    def bind(
        self,
        module: torch.nn.Module,
        calibration: torch.Tensor | Sequence[torch.Tensor],
        *,
        variant: dict[str, str] | None = None,
        norm: torch.nn.Module | None = None,
        residual: torch.Tensor | Sequence[torch.Tensor] | None = None,
        gate_cos: float = 0.999,
        percentile: float = 99.9,
    ) -> torch.nn.Module:
        """Build a gated drop-in replacement for ``module``.

        ``calibration``: one or more real inputs of ``module`` (the
        normed hidden states it actually sees). ``residual``: the
        matching pre-norm hidden states — the structure gate is declared
        at the boundary *including* the residual add, so provide these
        to gate at the declared boundary; without them the self-check
        runs on the bare seam output, which is strictly harsher.
        ``norm``: optional sibling norm module, recorded for
        full-structure use; the MLP seam itself leaves the norm in the
        host. ``gate_cos=0`` skips the self-check (not recommended).
        """
        samples = ([calibration] if torch.is_tensor(calibration)
                   else list(calibration))
        if not samples:
            raise ValueError("calibration must contain at least one sample")
        residuals = (None if residual is None else
                     [residual] if torch.is_tensor(residual)
                     else list(residual))
        if residuals is not None and len(residuals) != len(samples):
            raise ValueError("residual must align 1:1 with calibration")

        if self.name == "decoder_ffn":
            from .impls.decoder_ffn import fp8_static as impl
            gate = module.gate_proj
            device = gate.weight.device
            dims = {"D": gate.in_features, "F": gate.out_features}
            var = {"activation": activation_for(module, "silu"),
                   "norm_weight_mode": "direct"}
            weights = {
                "w_norm": (norm.weight.detach() if norm is not None
                           and getattr(norm, "weight", None) is not None
                           else torch.ones(dims["D"], device=device)),
                "w_gate": gate.weight.detach().t().contiguous(),
                "w_up": module.up_proj.weight.detach().t().contiguous(),
                "w_down": module.down_proj.weight.detach().t().contiguous(),
            }
        elif self.name == "vision_ffn":
            from .impls.vision_ffn import fp8_static as impl
            fc_attrs = next(
                (pair for pair in _VISION_PROJ
                 if all(isinstance(getattr(module, a, None), torch.nn.Linear)
                        for a in pair)), None)
            if fc_attrs is None:
                raise ValueError(
                    "module has no fc1/fc2 (or linear_fc1/linear_fc2) pair")
            fc1 = getattr(module, fc_attrs[0])
            fc2 = getattr(module, fc_attrs[1])
            device = fc1.weight.device
            dims = {"D": fc1.in_features, "F": fc1.out_features}
            var = {"activation": activation_for(module, "gelu")}
            weights = {
                "w_norm": (norm.weight.detach() if norm is not None
                           else torch.ones(dims["D"], device=device)),
                "b_norm": (norm.bias.detach() if norm is not None
                           else torch.zeros(dims["D"], device=device)),
                "w_fc1": fc1.weight.detach(), "b_fc1": fc1.bias.detach(),
                "w_fc2": fc2.weight.detach(), "b_fc2": fc2.bias.detach(),
            }
        else:
            raise ValueError(
                f"structure {self.name!r} has no module-seam bind; "
                "stage pipelines go through the capture door")
        var.update(variant or {})

        # the explicit door owns real samples, so it measures the seam's
        # points by running the host module under the same hooks the
        # distribution layer uses — one path for what a point is and where
        # it lives, whichever door asked
        from .points import Collector, Point

        points = [Point("x_after_norm", "", "input")]
        second = ("down_proj" if self.name == "decoder_ffn"
                  else (fc_attrs[1] if self.name == "vision_ffn" else None))
        second_name = ("act_after_mul" if self.name == "decoder_ffn"
                       else "hidden_after_act")
        if second is not None:
            points.append(Point(second_name, second, "input"))
        collector = Collector(points=points)
        handles = collector.hooks(
            lambda path: module if not path else getattr(module, path))
        try:
            with torch.no_grad():
                for sample in samples:
                    module(sample.to(device))
                    collector.end_sample()
        finally:
            for handle in handles:
                handle.remove()
        collector.reduce(percentile)
        kwargs = {} if self.name == "vision_ffn" else {"variant": var}
        bound = impl.bind_mlp_seam(
            weights,
            input_scale=collector.scale("", "x_after_norm"),
            hidden_scale=collector.scale(second or "", second_name),
            original=module, **kwargs)

        worst = 1.0
        with torch.no_grad():
            for k, sample in enumerate(samples):
                x = sample.to(device)
                got, want = bound(x), module(x)
                if residuals is not None:
                    res = residuals[k].to(device)
                    got, want = res + got, res + want
                worst = min(worst, parity_metrics(got, want)["cosine"])
        boundary = ("structure (incl. residual)" if residuals is not None
                    else "bare seam")
        if gate_cos and worst < gate_cos:
            raise GateRefused(
                f"{self.name} replacement worst cos {worst:.6f} < "
                f"{gate_cos} at {boundary} boundary on {len(samples)} "
                "calibration sample(s) — not handing back a part that "
                "fails its own gate"
                + ("" if residuals is not None else
                   "; pass residual= to gate at the declared boundary"))
        bound.certification = {
            "structure": self.name, "version": self.spec.version,
            "dims": dims, "variant": var, "worst_cos": round(worst, 7),
            "gate_boundary": boundary, "samples": len(samples),
            "m_profile": sorted({
                int(s.reshape(-1, s.shape[-1]).shape[0]) for s in samples}),
        }
        return bound


def get(name: str) -> StructureHandle:
    """Pull one structure from the catalog (get_kernel-style)."""
    spec = load(name)
    if spec.kind != "region":
        raise ValueError(
            f"{name!r} is a {spec.kind} structure — module-level bind "
            "does not apply; use the capture/provider route")
    return StructureHandle(spec)
