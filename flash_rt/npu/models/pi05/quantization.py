"""Real-observation static encoder calibration, outside graph replay."""
import numpy as np
import torch

from flash_rt.core.calibration import accumulate_amax
from flash_rt.npu.core.linear import RowCalibrationWeight, StaticRowInt8Weight
from .pipeline import _EP


def calibrate_encoder(weights, samples, make_runner, image_rows, percentile=99.9,
                      quantizer=None):
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
    return bound, {'samples': len(per_sample), 'percentile': percentile,
                   'method': 'sample-call max then house percentile; image-row and language-group scales',
                   'image_rows': image_rows,
                   'amax': {key: final[index].copy() for index, key in enumerate(sites)}}
