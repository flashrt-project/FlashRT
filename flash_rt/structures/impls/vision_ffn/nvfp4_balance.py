"""NVFP4 (W4A4) ``vision_ffn`` with per-input-channel balance.

Both projections of the MLP slice go through the balanced W4 path:
fc1's balance is fitted on the calibrated per-channel amax at the MLP
input, fc2's on the amax at its own input (the post-activation hidden).
Each fold is exact before anything is quantized. The activation between
them stays at the host's compute dtype with the host's tanh GELU — the
kernel boundary is the two GEMMs, not the elementwise middle, which is
exactly where the recorded W4 chain drew it.

Activation quantization is dynamic per call (per-block scale factors),
so no static scale exists to drift across a denoise schedule.
"""

from __future__ import annotations

from typing import Mapping

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam
from .. import hub_kernel
from .fp8_static import SUPPORT, _check  # noqa: F401

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp4-gemm",
    "version": ">=1",
}

_VARIANT = 2


class FusedGeluMlpNvfp4(GuardedSeam, torch.nn.Module):
    """MLP-seam module: the host keeps its own norm and residual."""

    _frt_host_attr = "host_mlp"
    _frt_can_fallback = True

    def __init__(self, wp1, sfb1, inv1, b1, wp2, sfb2, inv2, b2,
                 d: int, f: int,
                 original: torch.nn.Module | None = None,
                 fuse_wire: bool = False):
        super().__init__()
        kern = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._kern = kern
        self._gemm = kern.fp4_w4a16_linear_bf16
        # the FP4-wire chain: fc1's GEMM emits bias+tanh-GELU already
        # re-quantized (packed + SFA) and fc2 consumes it with a fused
        # bias — the elementwise middle disappears entirely. Explicitly
        # opted in (scheme decision), never flipped by symbol presence:
        # on the wire fc2's input-side balance cannot be applied, which
        # is a numerics change the gates must judge as a chosen form.
        chain_fn = getattr(kern, "nvfp4_gemm_bias_gelu_nvfp4", None)
        bias_fn = getattr(kern, "nvfp4_gemm_bias_bf16", None)
        self._chain = chain_fn if (fuse_wire and chain_fn is not None
                                   and bias_fn is not None) else None
        self._gemm_bias = bias_fn
        for name, t in (("wp1", wp1), ("sfb1", sfb1), ("inv1", inv1),
                        ("b1", b1), ("wp2", wp2), ("sfb2", sfb2),
                        ("inv2", inv2), ("b2", b2)):
            self.register_buffer(name, t)
        self._d = d
        self._f = f
        if original is not None:
            self.host_mlp = original
        self._frt_arm(dtypes=CAST_OK, device=wp1.device, k=d)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_mlp":
                raise
            return getattr(super().__getattr__("host_mlp"), name)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(hidden)
        if admitted is not PROCEED:
            return admitted
        shape = hidden.shape
        flat = (hidden.reshape(-1, shape[-1]).to(torch.float16)
                * self.inv1).contiguous()
        ap, sfa = self._kern.quantize_fp4_sfa_fp16(flat)
        if self._chain is not None:
            hp, hsfa = self._chain(ap, self.wp1, sfa, self.sfb1, self.b1)
            y = self._gemm_bias(hp, self.wp2, hsfa, self.sfb2, self.b2)
            return y.reshape(*shape[:-1], self._d).to(hidden.dtype)
        h = self._gemm(ap, self.wp1, sfa, self.sfb1, variant=_VARIANT)
        h = h + self.b1
        h = torch.nn.functional.gelu(h, approximate="tanh")
        hf = (h.to(torch.float16) * self.inv2).contiguous()
        ap2, sfa2 = self._kern.quantize_fp4_sfa_fp16(hf)
        y = self._gemm(ap2, self.wp2, sfa2, self.sfb2, variant=_VARIANT)
        y = y + self.b2
        return y.reshape(*shape[:-1], self._d).to(hidden.dtype)


@torch.no_grad()
def bind_mlp_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    channel_in,
    channel_hidden,
    original: torch.nn.Module | None = None,
    alpha: float = 0.5,
    clamp=(0.25, 4.0),
    fuse_wire: bool = False,
) -> FusedGeluMlpNvfp4:
    """Bind the MLP-seam slice from two calibrated channel-amax vectors.

    ``channel_in`` (``[D]``) is measured at the MLP input,
    ``channel_hidden`` (``[F]``) at the second projection's input — the
    post-activation hidden. Each parameterises its projection's balance
    fold; neither is a scale.
    """
    from flash_rt.core.quantization import fit_input_channel_balance

    dim_d, dim_f = _check(weights)
    kern = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
    clamp = (float(clamp[0]), float(clamp[1]))

    def fold_pack(w, chan):
        amax = torch.as_tensor(chan, device=w.device, dtype=torch.float32)
        w_bal, inv = fit_input_channel_balance(
            w.detach().float(), amax, alpha=alpha, clamp=clamp,
            out_dtype=torch.float32)
        wp, sfb = kern.quantize_fp4_sfa_fp16(
            w_bal.to("cuda", torch.float16).contiguous(), is_sfb=True)
        return wp, sfb, inv.to("cuda", torch.float16)

    wp1, sfb1, inv1 = fold_pack(weights["w_fc1"], channel_in)
    if fuse_wire:
        # on the FP4 wire fc2's input arrives already quantized, so no
        # activation-side inverse can be applied — fc2 packs unbalanced
        # (a folded weight without its inverse is wrong arithmetic, not
        # a weaker recipe). inv2 stays identity so the two-step
        # fallback path remains exact if the entries are absent.
        wp2, sfb2 = kern.quantize_fp4_sfa_fp16(
            weights["w_fc2"].detach().to("cuda", torch.float16)
            .contiguous(), is_sfb=True)
        inv2 = torch.ones(dim_f, device="cuda", dtype=torch.float16)
    else:
        wp2, sfb2, inv2 = fold_pack(weights["w_fc2"], channel_hidden)
    to_bf16 = lambda t: t.detach().to("cuda", torch.bfloat16)
    bound = FusedGeluMlpNvfp4(
        wp1, sfb1, inv1, to_bf16(weights["b_fc1"]),
        wp2, sfb2, inv2, to_bf16(weights["b_fc2"]),
        dim_d, dim_f, original=original, fuse_wire=fuse_wire)
    probe = bound(torch.zeros(1, dim_d, device=wp1.device,
                              dtype=torch.bfloat16))
    if probe.shape != (1, dim_d) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: vision_ffn nvfp4_balance bind smoke produced "
            f"shape {tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound
