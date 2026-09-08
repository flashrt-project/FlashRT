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
        quantized = quantized.contiguous().t()
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
    if isinstance(weight, (CalibrationWeight, StaticInt8Weight, StaticRowInt8Weight)):
        return weight(x, bias)
    return F.linear(x, weight, bias)


@dataclass
class RowCalibrationWeight(CalibrationWeight):
    image_rows: int = 0

    @classmethod
    def create(cls, tensor, name, image_rows):
        return cls(tensor, name, torch.zeros(image_rows + 1, device=tensor.device),
                   image_rows=image_rows)

    def __call__(self, x, bias=None):
        rows = x.float().abs().reshape(-1, x.shape[-1]).amax(-1)
        if rows.numel() <= self.image_rows:
            raise ValueError("calibration requires image and language tokens")
        # Camera token positions have individual scales. Language tokens
        # share a conservative scale across positions and prompt lengths.
        values = torch.cat((rows[:self.image_rows], rows[self.image_rows:].amax().view(1)))
        self.amax.copy_(torch.maximum(self.amax, values))
        self.calls += 1
        return F.linear(x, self.tensor, bias)


@dataclass(frozen=True)
class StaticRowInt8Weight:
    """Static token-row activations and output-channel INT8 weights.

    Buckets are prepared during warmup before graph capture. No reduction or
    scale update occurs in captured execution. Language rows share the last
    calibration scale, so an unseen prompt length needs no invented samples.
    """
    tensor: torch.Tensor
    activation_scales: torch.Tensor
    weight_scales: torch.Tensor
    image_rows: int
    buckets: dict
    quantizer: object = None

    @classmethod
    def bind(cls, weight, activation_amax, image_rows, quantizer=None):
        import numpy as np
        amax = np.asarray(activation_amax, dtype=np.float32)
        if amax.shape != (image_rows + 1,) or not np.isfinite(amax).all() or (amax < 0).any():
            raise ValueError("invalid real-data activation statistics")
        if weight.ndim != 2:
            raise ValueError("linear weight must be a matrix")
        acts = torch.as_tensor(amax.copy(), device=weight.device).clamp_min(1e-10) / 127.0
        scales = (weight.float().abs().amax(-1) / 127.0).clamp_min(1e-10)
        if not bool(torch.isfinite(scales).all().cpu()):
            raise ValueError("weight contains nonfinite values")
        quantized = (weight.float() / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        return cls(quantized.contiguous().t(), acts, scales, image_rows, {}, quantizer)

    def quantize_input(self, x):
        import torch_npu
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        if rows <= self.image_rows:
            raise ValueError("expected image and language token rows")
        if rows not in self.buckets:
            acts = torch.cat((self.activation_scales[:self.image_rows],
                              self.activation_scales[-1:].expand(rows - self.image_rows)))
            self.buckets[rows] = (acts, acts.reciprocal().contiguous(),
                                  torch.ones(flat.shape[-1], device=x.device))
        acts, inverse, ones = self.buckets[rows]
        if self.quantizer is None:
            q = torch_npu.npu_quantize(flat.float() * inverse[:, None], ones,
                                       None, torch.qint8, axis=-1, div_mode=False)
        else:
            q = self.quantizer(flat, inverse)
        return q, acts

    def project_quantized(self, q, acts, shape, dtype, bias=None):
        import torch_npu
        out = torch_npu.npu_quant_matmul(q, self.tensor, self.weight_scales,
                                        pertoken_scale=acts, output_dtype=torch.bfloat16)
        out = out.reshape(shape[:-1] + (self.tensor.shape[-1],)).to(dtype)
        return out if bias is None else out + bias

    def __call__(self, x, bias=None):
        q, acts = self.quantize_input(x)
        return self.project_quantized(q, acts, x.shape, x.dtype, bias)


@dataclass(frozen=True)
class StaticRowInt8Group:
    """Share one frozen input quantization across separate native GEMMs."""
    weights: tuple

    @classmethod
    def bind(cls, weights):
        weights = tuple(weights)
        if not weights or not all(isinstance(w, StaticRowInt8Weight) for w in weights):
            raise ValueError("group requires static row INT8 weights")
        first = weights[0]
        for weight in weights[1:]:
            if (weight.image_rows != first.image_rows
                    or weight.tensor.shape[0] != first.tensor.shape[0]
                    or weight.tensor.device != first.tensor.device
                    or not torch.equal(weight.activation_scales, first.activation_scales)):
                raise ValueError("shared quantization requires identical input scales and layout")
        return cls(weights)

    def __call__(self, x):
        q, acts = self.weights[0].quantize_input(x)
        return tuple(w.project_quantized(q, acts, x.shape, x.dtype) for w in self.weights)
