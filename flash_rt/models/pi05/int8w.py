"""Weight-only int8 linear (W8A16) on the DM05 tensor-core kernel.

Ported from ``opendm/model/dm05/native/int8w.py``. The only change is the kernel
module: ``dm05_native``'s ``w8a16_pack`` / ``w8a16_tc`` are FlashRT's
``dm05_w8a16_pack`` / ``dm05_w8a16_tc``, added to this repository because it had no
weight-only int8 path at this shape.

The weight is quantized once, symmetric per output channel (s = amax_row / 127),
and repacked into mma fragment order; the activation stays bf16. There is no
activation quantization, so there is no activation scale to calibrate: every
quantization parameter is fixed at load time.
"""
from __future__ import annotations

import torch

from flash_rt import flash_rt_dm05 as _dm05


class Int8WLinear:
    def __init__(self, weight: torch.Tensor, name: str = "?"):
        self.name = name
        self.N, self.K = int(weight.shape[0]), int(weight.shape[1])
        if self.N % 16 or self.K % 32:
            raise ValueError(f"{name}: N={self.N} K={self.K} not supported by w8a16_tc")
        w = weight.detach().float()
        self.scale = (w.abs().amax(1) / 127.0).clamp(min=1e-12).contiguous()
        q = torch.round(w / self.scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
        self.packed = torch.empty_like(q)
        _dm05.dm05_w8a16_pack(q.data_ptr(), self.packed.data_ptr(), self.N, self.K, 0)
        torch.cuda.synchronize()
        self.q_rows = q            # kept for fused repacks (gate|up), dropped by drop_rows()
        # split-K width measured per shape at M = 10 with DRAM-resident weights: narrow
        # outputs have few 16-row blocks, so they need more warps per block to fill the GPU.
        self.warps = 16 if self.N <= 1024 else 8

    def drop_rows(self) -> None:
        self.q_rows = None

    def run(self, x: torch.Tensor, out: torch.Tensor, m: int, stream: int) -> None:
        _dm05.dm05_w8a16_tc(x.data_ptr(), self.packed.data_ptr(), self.scale.data_ptr(),
                            out.data_ptr(), m, self.N, self.K, self.warps, stream)
