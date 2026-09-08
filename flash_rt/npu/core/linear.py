"""Setup-time static W8A8 linear bindings for captured NPU pipelines.

The calibration observer is only used by the eager setup pass. Frozen
bindings contain real INT8 operands and constant device scales; graph replay
never measures an activation or changes a quantization parameter.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CalibrationWeight:
    tensor: torch.Tensor
    name: str
    amax: torch.Tensor
    calls: int = 0

    @classmethod
    def create(cls, tensor, name):
        return cls(tensor, name, torch.zeros((), device=tensor.device))

    def reset(self):
        self.amax.zero_()
        self.calls = 0

    def __call__(self, x, bias=None):
        self.amax.copy_(torch.maximum(self.amax, x.float().abs().amax()))
        self.calls += 1
        return F.linear(x, self.tensor, bias)


@dataclass(frozen=True)
class StaticInt8Weight:
    tensor: torch.Tensor
    input_scales: torch.Tensor
    output_scales: torch.Tensor
    activation_scale: float
    weight_scale: float

    @classmethod
    def bind(cls, weight, activation_amax):
        import torch_npu
        act_scale = max(float(activation_amax) / 127.0, 1e-12)
        weight_scale = max(float(weight.float().abs().amax().cpu()) / 127.0, 1e-12)
        quantized = (weight.float() / weight_scale).round().clamp(-127, 127).to(torch.int8)
        # CANN consumes logical (K,N); packing is setup-only.
        quantized = quantized.t().contiguous()
        scales = torch.full((weight.shape[1],), 1.0 / act_scale,
                            dtype=torch.float32, device=weight.device)
        # Compute the descale product once in FP32, as the house contract requires.
        import numpy as np
        product = float(np.float32(act_scale) * np.float32(weight_scale))
        output_scales = torch.full((weight.shape[0],), product,
                                  dtype=torch.bfloat16, device=weight.device)
        return cls(quantized, scales, output_scales, act_scale, weight_scale)

    def __call__(self, x, bias=None):
        import torch_npu
        shape = x.shape[:-1] + (self.tensor.shape[-1],)
        q = torch_npu.npu_quantize(x.reshape(-1, x.shape[-1]).float(),
                                   self.input_scales, None, torch.qint8,
                                   axis=-1, div_mode=False)
        out = torch_npu.npu_quant_matmul(q, self.tensor, self.output_scales,
                                        output_dtype=torch.bfloat16)
        out = out.reshape(shape).to(x.dtype)
        return out if bias is None else out + bias


def linear(x, weight, bias=None):
    """Resolve a setup-time binding while constructing a captured graph."""
    if isinstance(weight, (CalibrationWeight, StaticInt8Weight)):
        return weight(x, bias)
    return F.linear(x, weight, bias)
