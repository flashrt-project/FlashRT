"""W4A16 implementation of the ``moe_experts`` structure.

The SM110 twin of :mod:`.nvfp4_dynamic`: same packed NVFP4 expert bank,
same external routing contract, but the grouped launch keeps activations
in BF16 — the ``grouped_w4a4_*`` entries require SM120/SM121
block-scaled MMA and refuse on Thor, while ``grouped_w4a16_gemv_bf16``
serves one routed slot per activation row on every arch the package
ships. The call convention therefore differs: rows are expanded to one
per routed slot ([T*k, K]) instead of the W4A4 entry's [T, k] batch.

Packing is byte-identical to the W4A4 impl (one ``quantize_weights_
nvfp4_bf16`` layout serves both entries); only the bind-time probe and
the forward launches change.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam
from .nvfp4_dynamic import _BIND_SLAB, _kernel, _sf_bytes, check_experts

__all__ = ["MoeExpertsW4A16", "bind_experts_seam"]


class MoeExpertsW4A16(GuardedSeam, torch.nn.Module):
    """Packed expert bank behind the host contract, BF16 activations."""

    _frt_can_fallback = False

    def __init__(self, gu_packed, gu_sfb, dn_packed, dn_sfb, act_fn,
                 num_experts, hidden, inter):
        super().__init__()
        self.register_buffer("_gu_packed", gu_packed)
        self.register_buffer("_gu_sfb", gu_sfb)
        self.register_buffer("_dn_packed", dn_packed)
        self.register_buffer("_dn_sfb", dn_sfb)
        self.register_buffer("_alpha", torch.ones(
            num_experts, device=gu_packed.device, dtype=torch.float32))
        self._act = act_fn
        self._e = num_experts
        self._h = hidden
        self._i = inter
        self._grouped = _kernel().grouped_w4a16_gemv_bf16
        self._frt_arm(dtypes=CAST_OK, device=gu_packed.device, k=hidden)

    def _launch(self, x, packed, sfb, ids, n):
        return self._grouped(
            x, packed, sfb, self._alpha, ids, n=n,
            w_stride=packed.shape[1] * packed.shape[2],
            sfb_stride=sfb.shape[1])

    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(hidden_states)
        if admitted is not PROCEED:
            return admitted
        t = hidden_states.shape[0]
        k = top_k_index.shape[1]
        ids = top_k_index.reshape(-1).to(torch.int32)
        x = hidden_states.to(torch.bfloat16).contiguous()
        xr = x.repeat_interleave(k, dim=0)              # [T*k, H]
        y = self._launch(xr, self._gu_packed, self._gu_sfb, ids,
                         2 * self._i)                    # [T*k, 2I]
        gate, up = y.chunk(2, dim=-1)
        inter = (self._act(gate) * up).contiguous()      # [T*k, I]
        d = self._launch(inter, self._dn_packed, self._dn_sfb, ids,
                         self._h)                        # [T*k, H]
        out = (d.view(t, k, self._h).float()
               * top_k_weights[..., None].float()).sum(dim=1)
        return out.to(hidden_states.dtype)


@torch.no_grad()
def _pack_bank(kern, bank: torch.Tensor, alpha: torch.Tensor,
               probe_gen: torch.Generator) -> tuple[
                   torch.Tensor, torch.Tensor, float]:
    """Pack one 3D stack ``[E, N, K]``; probe through the W4A16 entry."""
    e, n, kdim = bank.shape
    packed = torch.empty(e, n, kdim // 2, device="cuda", dtype=torch.uint8)
    sfb = torch.empty(e, _sf_bytes(n, kdim), device="cuda",
                      dtype=torch.uint8)
    worst = 0.0
    for lo in range(0, e, _BIND_SLAB):
        slab = bank[lo:lo + _BIND_SLAB].to("cuda", torch.bfloat16)
        for j in range(slab.shape[0]):
            kern.quantize_weights_nvfp4_bf16(
                slab[j].contiguous(), packed=packed[lo + j],
                sfb=sfb[lo + j])
        x = (torch.randn(1, kdim, device="cuda", generator=probe_gen,
                         dtype=torch.float32) * 0.05).to(torch.bfloat16)
        got = kern.grouped_w4a16_gemv_bf16(
            x, packed, sfb, alpha,
            torch.tensor([lo], device="cuda", dtype=torch.int32),
            n=n, w_stride=n * kdim // 2, sfb_stride=sfb.shape[1])[0]
        ref = x[0].float() @ slab[0].float().t()
        rel = float((got.float() - ref).norm() / ref.norm().clamp_min(1e-12))
        worst = max(worst, rel)
        del slab
    return packed, sfb, worst


@torch.no_grad()
def bind_experts_seam(
    weights: Mapping[str, torch.Tensor], act_fn,
) -> tuple[MoeExpertsW4A16, dict[str, float]]:
    """Bind one expert bank from its dense 3D stacks (W4A16 launches)."""
    e, h, i = check_experts(weights)
    kern = _kernel()
    alpha = torch.ones(e, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    gu_packed, gu_sfb, gu_rel = _pack_bank(
        kern, weights["gate_up_proj"], alpha, gen)
    dn_packed, dn_sfb, dn_rel = _pack_bank(
        kern, weights["down_proj"], alpha, gen)
    bound = MoeExpertsW4A16(gu_packed, gu_sfb, dn_packed, dn_sfb,
                            act_fn, e, h, i)
    probe = bound(torch.zeros(1, h, device="cuda", dtype=torch.bfloat16),
                  torch.zeros(1, 1, device="cuda", dtype=torch.int64),
                  torch.ones(1, 1, device="cuda", dtype=torch.float32))
    if probe.shape != (1, h) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: w4a16 experts bind smoke produced shape "
            f"{tuple(probe.shape)}, finite={bool(torch.isfinite(probe).all())}")
    return bound, {"gate_up_relL2": gu_rel, "down_relL2": dn_rel}
