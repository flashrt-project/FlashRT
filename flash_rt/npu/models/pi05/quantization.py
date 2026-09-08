"""Real-observation static encoder calibration, outside graph replay."""
import numpy as np
import torch
from dataclasses import dataclass

from flash_rt.core.calibration import accumulate_amax
from flash_rt.npu.core.linear import RowCalibrationWeight, StaticRowInt8Weight, StaticRowInt8Group
from .pipeline import _EP, ENC_L


def calibrate_encoder(weights, samples, make_runner, image_rows, percentile=99.9,
                      quantizer=None, activation_producer=None):
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
            bound[key] = StaticRowInt8Weight.bind(observer.tensor, final[index],
                                                  image_rows, quantizer)
        for layer in range(ENC_L):
            prefix = f"{_EP}.{layer}"
            for name, projections in (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                                ("gu", ("mlp.gate_proj", "mlp.up_proj"))):
                bound[f"{prefix}.{name}.group"] = StaticRowInt8Group.bind(
                    bound[f"{prefix}.{site}.weight"] for site in projections)
            if activation_producer is not None:
                bound[f"{prefix}.down.fused"] = GeluMulProjection(
                    bound[f"{prefix}.mlp.down_proj.weight"], activation_producer)
    return bound, {'samples': len(per_sample), 'percentile': percentile,
                   'method': 'sample-call max then house percentile; image-row and language-group scales',
                   'image_rows': image_rows,
                   'amax': {key: final[index].copy() for index, key in enumerate(sites)}}


@dataclass(frozen=True)
class GeluMulProjection:
    weight: StaticRowInt8Weight
    producer: object

    def __call__(self, gate, up):
        acts, inverse, _ = self.weight.scales_for_rows(gate.shape[0], gate.shape[1], gate.device)
        quantized = self.producer(gate, up, inverse)
        return self.weight.project_quantized(quantized, acts, gate.shape, gate.dtype)
