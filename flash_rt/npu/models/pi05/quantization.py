"""Real-observation static encoder calibration, outside graph replay."""
import numpy as np
import torch
from dataclasses import dataclass, replace

from flash_rt.core.calibration import accumulate_amax
from flash_rt.npu.core.linear import RowCalibrationWeight, StaticRowInt8Weight, StaticRowInt8Group
from .pipeline import _EP, ENC_L, ENC_NH, ENC_NKV, ENC_HD


def calibrate_encoder(weights, samples, make_runner, image_rows, percentile=99.9,
                      quantizer=None, activation_producer=None, norm_producer=None,
                      attention_output_quant=False):
    """Collect one eager sample at a time; bind immutable INT8 operands.

    ``make_runner(sample, observed_weights)`` owns host preprocessing and
    returns a filled model runner. This layer has no frontend dependency.
    """
    observed = dict(weights)
    sites = {}
    for key, weight in weights.items():
        if (key.startswith(_EP) and key.endswith('.weight') and weight.ndim == 2
                and any(part in key for part in ('.self_attn.', '.mlp.'))):
            sites[key] = RowCalibrationWeight.create(weight, key, image_rows)
            observed[key] = sites[key]
    if not sites:
        raise ValueError("encoder has no calibratable linear weights")
    per_sample = []
    attention_scales = {}
    with torch.inference_mode():
        for sample in samples:
            for observer in sites.values():
                observer.reset()
            runner = make_runner(sample, observed)
            runner._run()
            torch.npu.synchronize()
            if any(observer.calls == 0 for observer in sites.values()):
                raise RuntimeError("calibration did not execute every encoder site")
            values = np.stack([observer.amax.cpu().numpy() for observer in sites.values()])
            if not np.isfinite(values).all():
                raise ValueError("nonfinite activation in calibration sample")
            per_sample.append(values.reshape(-1))
            del runner
        if not per_sample:
            raise ValueError("INT8 requires nonempty real observations")
        final = accumulate_amax(per_sample, percentile).reshape(len(sites), image_rows + 1)
        bound = dict(weights)
        for index, (key, observer) in enumerate(sites.items()):
            # The MLP down projection reduces over 16384 columns, which is where
            # a fractal-NZ operand pays; the 2048- and 256-wide projections
            # measure slower in NZ and stay ND.
            bound[key] = StaticRowInt8Weight.bind(observer.tensor, final[index],
                                                  image_rows, quantizer,
                                                  nz=key.endswith(".mlp.down_proj.weight"))
        for layer in range(ENC_L):
            prefix = f"{_EP}.{layer}"
            for name, projections in (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                                ("gu", ("mlp.gate_proj", "mlp.up_proj"))):
                bound[f"{prefix}.{name}.group"] = StaticRowInt8Group.bind(
                    bound[f"{prefix}.{site}.weight"] for site in projections)
                if norm_producer is not None:
                    bound[f"{prefix}.{name}.norm"] = RmsQuantProjection(
                        bound[f"{prefix}.{name}.group"], norm_producer)
            if activation_producer is not None:
                bound[f"{prefix}.down.fused"] = GeluMulProjection(
                    bound[f"{prefix}.mlp.down_proj.weight"], activation_producer)
            if attention_output_quant:
                key = f"{prefix}.self_attn.o_proj.weight"
                projection = AttentionQuantProjection.bind(bound[key])
                bound[key] = projection.weight
                bound[f"{prefix}.attention.quantized"] = projection
                attention_scales[key] = float(projection.weight.activation_scales[0].cpu())
    if activation_producer is not None and norm_producer is not None:
        from flash_rt.npu.core.gu_int8 import GuInt8Weights
        for layer in range(ENC_L - 1):
            prefix = f"{_EP}.{layer}"
            norm = bound[f"{prefix}.gu.norm"]
            gate, up = norm.group.weights
            packed = GuInt8Weights.create(gate.tensor, up.tensor,
                                          gate.weight_scales, up.weight_scales)
            bound[f"{prefix}.mlp.native"] = NativeEncoderMlp(
                norm, bound[f"{prefix}.mlp.down_proj.weight"], packed)
    return bound, {'samples': len(per_sample), 'percentile': percentile,
                   'method': 'sample-call max then house percentile; image-row and language-group scales',
                   'image_rows': image_rows,
                   'attention_output_scales': attention_scales,
                   'attention_scale_rule': 'maximum of frozen row scales after house aggregation',
                   'amax': {key: final[index].copy() for index, key in enumerate(sites)}}


@dataclass(frozen=True)
class GeluMulProjection:
    weight: StaticRowInt8Weight
    producer: object

    def __call__(self, gate, up):
        acts, inverse, _ = self.weight.scales_for_rows(gate.shape[0], gate.shape[1], gate.device)
        quantized = self.producer(gate, up, inverse)
        return self.weight.project_quantized(quantized, acts, gate.shape, gate.dtype)


@dataclass(frozen=True)
class RmsQuantProjection:
    group: StaticRowInt8Group
    producer: object

    def __call__(self, x, other, gamma):
        acts, inverse, _ = self.group.weights[0].scales_for_rows(x.shape[0], x.shape[1], x.device)
        quantized, residual = self.producer(x, other, gamma, inverse)
        outputs = tuple(w.project_quantized(quantized, acts, x.shape, x.dtype)
                        for w in self.group.weights)
        return outputs, residual


@dataclass(frozen=True)
class AttentionQuantProjection:
    """CANN attention INT8 output followed by its static output projection."""
    weight: StaticRowInt8Weight
    inverse_scale: torch.Tensor

    @classmethod
    def bind(cls, weight):
        # CANN attention supports a scalar output scale. Freeze the maximum
        # across the already calibrated row scales and expose that same scale
        # through the linear binding and precision specification.
        scale = weight.activation_scales.max().reshape(1)
        bound = replace(weight, activation_scales=scale.expand_as(
            weight.activation_scales).clone(), buckets={})
        return cls(bound, scale.reciprocal())

    def __call__(self, q, k, v):
        import torch_npu
        out = torch_npu.npu_prompt_flash_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            num_heads=ENC_NH, num_key_value_heads=ENC_NKV,
            input_layout="BSH", scale_value=ENC_HD ** -0.5,
            pre_tokens=2147483647, next_tokens=2147483647,
            quant_scale2=self.inverse_scale).reshape(q.shape)
        acts, _, _ = self.weight.scales_for_rows(q.shape[0], q.shape[1], q.device)
        return self.weight.project_quantized(out, acts, q.shape, q.dtype)


@dataclass(frozen=True)
class NativeEncoderMlp:
    norm: RmsQuantProjection
    down: StaticRowInt8Weight
    packed: object

    def prepare(self, rows, workspace):
        first = self.norm.group.weights[0]
        acts, inverse, _ = first.scales_for_rows(rows, self.packed.columns, first.tensor.device)
        down_acts, down_inverse, _ = self.down.scales_for_rows(rows, self.packed.hidden, first.tensor.device)
        return PreparedEncoderMlp(self.norm.producer, inverse, self.down, down_acts,
                                  self.packed.prepare(rows, acts, down_inverse, workspace))


@dataclass(frozen=True)
class PreparedEncoderMlp:
    norm: object
    inverse: object
    down: StaticRowInt8Weight
    down_acts: object
    gu: object

    def __call__(self, x, other, gamma):
        q, residual = self.norm(x, other, gamma, self.inverse)
        hidden = self.gu(q)
        out = self.down.project_quantized(hidden, self.down_acts, hidden.shape, x.dtype)
        return out, residual


def prepare_encoder_mlp(weights, rows):
    """Each runner owns one scratch buffer, shared only by its sequential layers."""
    if weights is None:
        return None
    sites = {key: value for key, value in weights.items() if isinstance(value, NativeEncoderMlp)}
    if not sites:
        return weights
    first = next(iter(sites.values()))
    workspace = torch.empty(20 * 2 * 128 * 1024, dtype=torch.int32, device=first.packed.packed.device)
    result = dict(weights)
    for key, value in sites.items():
        result[key] = value.prepare(rows, workspace)
    return result
