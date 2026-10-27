"""FlashRT -- the input-independent tables of the GR00T N1.5 action head.

The DiT is conditioned on the denoising step only, and the step schedule is
fixed (``step / steps * buckets``), so everything derived from the step is a
constant of the checkpoint: the timestep embedding, every layer's
AdaLayerNorm modulation, the output modulation and the action encoder's
sinusoidal step encoding. They are computed here once, at load, with the
reference's torch expressions on the checkpoint's bf16 weights (diffusers
``Timesteps`` / ``TimestepEmbedding``, ``AdaLayerNorm``, ``proj_out_1``,
GR00T's ``SinusoidalPositionalEncoding``), so the tables hold the
reference's bits; :class:`~flash_rt.models.groot_n15.GrootN15Forward` reads
them by pointer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from flash_rt.models.groot_n15.forward import GrootN15Dims

BF16 = torch.bfloat16


@dataclass
class StepTables:
    """Per denoising step: ``ada[step][layer] = (1 + scale, shift)`` and
    ``ada_out[step] = (1 + scale, shift)`` (bf16 ``[dit_hidden]``), and
    ``tau[step]`` the step encoding (bf16 ``[action_horizon, dit_hidden]``)."""

    ada: list[list[tuple[torch.Tensor, torch.Tensor]]]
    ada_out: list[tuple[torch.Tensor, torch.Tensor]]
    tau: list[torch.Tensor]


def _timestep_proj(t: torch.Tensor, channels: int) -> torch.Tensor:
    # diffusers Timesteps(channels, flip_sin_to_cos=True, downscale_freq_shift=1)
    half = channels // 2
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device)
    exponent = exponent / (half - 1)
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half:], emb[:, :half]], dim=-1)


def _step_encoding(t: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    # GR00T SinusoidalPositionalEncoding over the step repeated for every action row
    ts = t.unsqueeze(1).expand(-1, rows).float()
    half = width // 2
    exponent = -torch.arange(half, dtype=torch.float, device=t.device) * (
        torch.log(torch.tensor(10000.0)) / half)
    freqs = ts.unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


@torch.no_grad()
def build_step_tables(dims: GrootN15Dims, timestep_lin1: Sequence[torch.Tensor],
                      timestep_lin2: Sequence[torch.Tensor],
                      layer_ada: Sequence[Sequence[torch.Tensor]],
                      proj_out_1: Sequence[torch.Tensor],
                      device: torch.device) -> StepTables:
    """The step tables from the bf16 weights ``(W, b)`` of the timestep MLP,
    every DiT layer's ``norm1.linear`` and ``proj_out_1``."""
    if len(layer_ada) != dims.dit_layers:
        raise ValueError(f"{len(layer_ada)} AdaLayerNorm projections for {dims.dit_layers} layers")
    De, H = dims.dit_hidden, dims.action_horizon
    ada, ada_out, tau = [], [], []
    for step in range(dims.denoising_steps):
        t = torch.full((1,), dims.step_bucket(step), device=device)
        temb = F.linear(F.silu(F.linear(_timestep_proj(t, timestep_lin1[0].shape[1]).to(BF16),
                                        *timestep_lin1)), *timestep_lin2)
        act = F.silu(temb)
        per_layer = []
        for w in layer_ada:
            scale, shift = F.linear(act, *w).chunk(2, dim=1)
            per_layer.append(((1 + scale).reshape(De).contiguous(),
                              shift.reshape(De).contiguous()))
        ada.append(per_layer)
        shift, scale = F.linear(act, *proj_out_1).chunk(2, dim=1)
        ada_out.append(((1 + scale).reshape(De).contiguous(), shift.reshape(De).contiguous()))
        tau.append(_step_encoding(t, H, De).to(BF16).reshape(H, De).contiguous())
    return StepTables(ada, ada_out, tau)


def table_pointers(tables: StepTables, position: torch.Tensor) -> Mapping[str, object]:
    """The pointer form :class:`GrootN15Forward` takes (``position`` is the
    action position-embedding rows ``[action_horizon, dit_hidden]``)."""
    p = lambda t: t.data_ptr()  # noqa: E731
    return {
        "ada": [[(p(a), p(s)) for a, s in per] for per in tables.ada],
        "ada_out": [(p(a), p(s)) for a, s in tables.ada_out],
        "tau": [p(t) for t in tables.tau],
        "position": p(position),
    }
