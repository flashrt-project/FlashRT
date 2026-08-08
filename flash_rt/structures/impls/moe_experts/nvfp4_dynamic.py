"""NVFP4 (W4A4, dynamic activation scales) ``moe_experts`` implementation.

The expert bank of a sparse-MoE block stores every expert's projections
as stacked 3D tensors (``gate_up_proj [E, 2I, H]``, ``down_proj
[E, H, I]``). Each expert's matrices pack once, at bind time, into the
grouped kernel's stacked NVFP4 layout, and the forward runs the bank as
two grouped launches per call: one for every routed gate_up slot
(``[T, top_k]``), one for every down slot (flattened to ``[T*top_k, 1]``
because each routed pair carries its own intermediate activation). The
routing tensor stays on the device end to end — no host sync, fixed
shapes for a given ``T`` — so the step is legal inside a compiled
region or a captured graph, and the *same* kernels serve the M=1 decode
row and the M=K+1 verify pass: one numeric family across both, which
is what token-identity between a spec verify and the plain step needs.

Contributions accumulate in FP32 over the fixed top-k axis before the
single cast back to the host dtype.

Known ceiling, recorded not hidden: weight traffic is per routed slot.
A long prefill (hundreds of tokens and up) re-reads shared expert
weights once per slot where a per-expert grouping would read them once;
until a grouped entry with per-expert accumulation ships, long prompts
through this bank pay slot-linear traffic.

There is no host fallback: binding exists to retire the dense weights
whose footprint keeps the checkpoint off the card, so the guard refuses
out-of-form calls instead of falling back.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/grouped-moe-gemv",
    "version": ">=2",
}

#: mirrors the kernel's own checks (K divisible by 16, N by 8) — both
#: contraction dims of an expert bank are K once: H for gate_up, I for
#: down; both output dims are N once: 2I and H
SUPPORT = {
    "K": {"min": 16, "multiple_of": 16},
    "N": {"min": 8, "multiple_of": 8},
    "E": {"min": 1},
}

#: experts are streamed to the GPU in slabs of this many during bind so
#: the transient footprint stays at slab size, not the full bank
_BIND_SLAB = 32


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


def _sf_bytes(rows: int, dim: int) -> int:
    """The kernel's swizzled scale-factor buffer size for one [rows, dim]."""
    return ((rows + 127) // 128) * (((dim // 16) + 3) // 4) * 512


def check_experts(weights: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    """Validate an expert bank's shapes; returns ``(E, H, I)``."""
    gu, dn = weights["gate_up_proj"], weights["down_proj"]
    if gu.dim() != 3 or dn.dim() != 3:
        raise ValueError(
            f"expert bank must be 3D stacks, got gate_up "
            f"{tuple(gu.shape)}, down {tuple(dn.shape)}")
    e, two_i, h = gu.shape
    e2, h2, i = dn.shape
    if e != e2 or h != h2 or two_i != 2 * i:
        raise ValueError(
            f"inconsistent expert bank: gate_up {tuple(gu.shape)} vs "
            f"down {tuple(dn.shape)}")
    if e < SUPPORT["E"]["min"]:
        raise ValueError(f"E={e} outside support envelope")
    for name, dim in (("H", h), ("I", i)):
        if dim < SUPPORT["K"]["min"] or dim % SUPPORT["K"]["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a positive multiple of "
                f"{SUPPORT['K']['multiple_of']} (it is a contraction dim)")
        if dim % SUPPORT["N"]["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a multiple of "
                f"{SUPPORT['N']['multiple_of']} (it is an output dim)")
    return e, h, i


class MoeExpertsNvfp4Dynamic(GuardedSeam, torch.nn.Module):
    """Packed expert bank: two grouped FP4 launches behind the host
    contract, for any short token batch."""

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
        self._grouped = _kernel().grouped_w4a4_gemv_from_bf16
        self._frt_arm(dtypes=CAST_OK, device=gu_packed.device, k=hidden)

    def forward(self, hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(hidden_states)
        if admitted is not PROCEED:
            return admitted
        t = hidden_states.shape[0]
        k = top_k_index.shape[1]
        idx = top_k_index.to(torch.int32)
        y = self._grouped(hidden_states.contiguous(), self._gu_packed,
                          self._gu_sfb, self._alpha, idx)
        gate, up = y.chunk(2, dim=-1)
        inter = self._act(gate) * up                    # [T, k, I], fresh
        d = self._grouped(inter.reshape(t * k, self._i), self._dn_packed,
                          self._dn_sfb, self._alpha,
                          idx.reshape(t * k, 1))
        out = (d.view(t, k, self._h).float()
               * top_k_weights[..., None].float()).sum(dim=1)
        return out.to(hidden_states.dtype)


@torch.no_grad()
def _pack_bank(kern, bank: torch.Tensor, alpha: torch.Tensor,
               probe_gen: torch.Generator) -> tuple[
                   torch.Tensor, torch.Tensor, float]:
    """Pack one 3D stack ``[E, N, K]``; returns worst probe-row relL2.

    The kernel ships no dequantizer, so conversion is accounted at the
    output: one random row through the grouped entry against the BF16
    slab's own matmul, per slab, worst kept for the receipt.
    """
    e, n, kdim = bank.shape
    packed = torch.empty(e, n, kdim // 2, device="cuda", dtype=torch.uint8)
    sfb = torch.empty(e, _sf_bytes(n, kdim), device="cuda",
                      dtype=torch.uint8)
    worst = 0.0
    grouped = kern.grouped_w4a4_gemv_from_bf16
    for lo in range(0, e, _BIND_SLAB):
        slab = bank[lo:lo + _BIND_SLAB].to("cuda", torch.bfloat16)
        for j in range(slab.shape[0]):
            kern.quantize_weights_nvfp4_bf16(
                slab[j].contiguous(), packed=packed[lo + j],
                sfb=sfb[lo + j])
        x = (torch.randn(1, kdim, device="cuda", generator=probe_gen,
                         dtype=torch.float32) * 0.05).to(torch.bfloat16)
        got = grouped(x, packed, sfb, alpha,
                      torch.tensor([[lo]], device="cuda",
                                   dtype=torch.int32))[0, 0]
        ref = x[0].float() @ slab[0].float().t()
        rel = float((got.float() - ref).norm() / ref.norm().clamp_min(1e-12))
        worst = max(worst, rel)
        del slab
    return packed, sfb, worst


@torch.no_grad()
def bind_experts_seam(
    weights: Mapping[str, torch.Tensor], act_fn,
) -> tuple[MoeExpertsNvfp4Dynamic, dict[str, float]]:
    """Bind one expert bank from its dense 3D stacks.

    Weights stream to the GPU in expert slabs and pack there; the
    returned dict carries the worst probe-row relative L2 per stack,
    for the adoption receipt. The bound module holds only the packed
    layout — retiring the dense bank is the caller's move (and the
    point).
    """
    e, h, i = check_experts(weights)
    kern = _kernel()
    alpha = torch.ones(e, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    gu_packed, gu_sfb, gu_rel = _pack_bank(
        kern, weights["gate_up_proj"], alpha, gen)
    dn_packed, dn_sfb, dn_rel = _pack_bank(
        kern, weights["down_proj"], alpha, gen)
    bound = MoeExpertsNvfp4Dynamic(gu_packed, gu_sfb, dn_packed, dn_sfb,
                                   act_fn, e, h, i)
    # bind-time smoke: one decode-shaped call through the real entries
    probe = bound(
        torch.zeros(1, h, device=gu_packed.device, dtype=torch.bfloat16),
        torch.zeros(1, 1, device=gu_packed.device, dtype=torch.long),
        torch.ones(1, 1, device=gu_packed.device, dtype=torch.bfloat16))
    if probe.shape != (1, h) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: moe_experts nvfp4 bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound, {"gate_up_proj": gu_rel, "down_proj": dn_rel}
