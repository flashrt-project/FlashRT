"""An affine LayerNorm that emits FP8 directly, at a consumer's scale.

The pipeline fact this serves: a vision block's pre-FFN norm output has
exactly one consumer, the FFN — and when that FFN is seated in FP8 form
its first act is to quantize its input. Emitting FP8 from the norm
itself (one fused kernel: no-affine LN + scale/shift + static quantize,
with scale=(gamma-1), shift=beta reproducing the affine norm exactly)
deletes the FFN's own input quantize and the norm's dtype round trip.

The hard precedent this respects: handing FP8 to a *host* consumer is
garbage-in-silence (measured 0.24 output match at the decoder norm
boundary). This producer is therefore only ever seated by the
negotiation pass that pairs it with an FP8-input seat as the direct
consumer — seat produces, seat consumes, and the consumer's FP8_ONLY
guard refuses loudly if anything else arrives between them. Whether
the pair actually pays is measured at bind, never assumed.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "hf",
    "repo": "flashrt/adaptive-layernorm-producers",
    "version": ">=1",
}


class FusedNormFp8Producer(GuardedSeam, torch.nn.Module):
    """Drop-in for an affine LayerNorm whose sole consumer eats FP8."""

    _frt_host_attr = "host_norm"
    _frt_can_fallback = False   # the consumer expects FP8; a BF16
    # fallback here would feed the paired seat out of contract, so an
    # out-of-form input must refuse loudly instead of degrading quietly

    def __init__(self, original: torch.nn.Module,
                 act_scale: torch.Tensor):
        super().__init__()
        self.host_norm = original
        ks = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._fn = ks.ada_layer_norm_quant_fp8_bf16
        # scale=(gamma-1), shift=beta: norm(x)*(1+scale)+shift is then
        # exactly the host's affine LayerNorm, quantized
        self.register_buffer("w", (original.weight.detach().float() - 1.0)
                             .to(torch.bfloat16).contiguous())
        self.register_buffer("b", original.bias.detach()
                             .to(torch.bfloat16).contiguous())
        self.register_buffer("act_scale",
                             act_scale.detach().reshape(1).float())
        self.eps = float(getattr(original, "eps", 1e-6))
        self._frt_arm(dtypes=CAST_OK, device=self.w.device,
                      k=int(self.w.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        y = self._fn(flat, self.w, self.b, self.act_scale, self.eps)
        return y.reshape(shape)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_norm":
                raise
            return getattr(super().__getattr__("host_norm"), name)


def bind_norm_fp8_producer(original: torch.nn.Module,
                           act_scale: torch.Tensor
                           ) -> FusedNormFp8Producer:
    if getattr(original, "weight", None) is None \
            or getattr(original, "bias", None) is None:
        raise ValueError("fp8 norm producer needs a two-sided affine "
                         "LayerNorm")
    return FusedNormFp8Producer(original, act_scale)
