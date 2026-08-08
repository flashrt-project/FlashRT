"""Non-quantising BF16 implementation of ``qkv_pack``.

Sibling projections that proved shared-input fixed-order dataflow are one
larger BF16 GEMM.  No quantisation is introduced: this is the portable
structural form used when an end-to-end accuracy gate refuses FP8/FP4.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam


class PackedBf16Linear(GuardedSeam, torch.nn.Module):
    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, mods: Sequence[torch.nn.Linear], rows: int):
        super().__init__()
        if len(mods) < 2:
            raise ValueError("qkv_pack: need at least two siblings")
        kdims = {int(mod.weight.shape[1]) for mod in mods}
        if len(kdims) != 1:
            raise ValueError(f"qkv_pack: sibling K dims differ {kdims}")
        self.splits = tuple(int(mod.weight.shape[0]) for mod in mods)
        weight = torch.cat(
            [mod.weight.detach() for mod in mods], dim=0).contiguous()
        bias = None
        if any(mod.bias is not None for mod in mods):
            bias = torch.cat([
                (mod.bias.detach() if mod.bias is not None else
                 torch.zeros(mod.weight.shape[0], device=mod.weight.device,
                             dtype=mod.weight.dtype))
                for mod in mods
            ]).contiguous()
        self.register_buffer("packed_weight", weight)
        self.register_buffer("packed_bias", bias)
        self.host_linear = mods[0]
        for index, width in enumerate(self.splits[1:], 1):
            self.register_buffer(
                f"stash{index}", torch.empty(
                    rows, width, device=weight.device, dtype=weight.dtype))
        self._frt_arm(dtypes=CAST_OK, device=weight.device,
                      k=next(iter(kdims)), row_capacity=rows)

    def _run(self, flat: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.linear(
            flat.to(self.packed_weight.dtype), self.packed_weight,
            self.packed_bias)
        offset = self.splits[0]
        for index, width in enumerate(self.splits[1:], 1):
            getattr(self, f"stash{index}")[:flat.shape[0]].copy_(
                y[:, offset:offset + width])
            offset += width
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        y = self._run(x.reshape(-1, x.shape[-1]))
        out = y[:, :self.splits[0]].contiguous()
        return out.reshape(*x.shape[:-1], self.splits[0]).to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


class Bf16StashReader(GuardedSeam, torch.nn.Module):
    _frt_host_attr = "host_linear"
    _frt_can_fallback = True
    _frt_requires_sibling_order = True

    def __init__(self, original: torch.nn.Linear,
                 packed: PackedBf16Linear, index: int):
        super().__init__()
        self.host_linear = original
        self._packed = (packed,)
        self.index = int(index)
        head = packed._frt_guard
        self._frt_arm(dtypes=head.dtypes, device=head.device, k=head.k,
                      row_capacity=head.row_capacity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        rows = x.numel() // x.shape[-1]
        out = getattr(self._packed[0], f"stash{self.index}")[:rows]
        return out.reshape(*x.shape[:-1], out.shape[-1]).to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


def bind_qkv_pack(mods: Sequence[torch.nn.Linear], *, rows: int):
    packed = PackedBf16Linear(mods, rows)
    return [packed, *(
        Bf16StashReader(mod, packed, index)
        for index, mod in enumerate(mods[1:], 1)
    )]
