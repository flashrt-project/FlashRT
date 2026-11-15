"""Core ML programs for Neural Engine shares of tanh-GELU gated FFNs (optional coremltools).

A share computes, for hidden channels [h0, h1) of an FFN, the FP16 partial product of the
down projection: down[h0:h1]^T (gelu_tanh(gate[:, h0:h1]^T x) * (up[:, h0:h1]^T x)). Inputs and
outputs use the channel-major (1, C, S, rows / S) layout; the down projection is split into
convolutions of at most 2048 input channels, which the Neural Engine executes far faster than
one convolution with a long input-channel axis. Compiled models are cached by content.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

COMPUTE_UNITS = {"cpu_and_ne": 0, "all": 1, "cpu_and_gpu": 2, "cpu": 3}
DOWN_CHUNKS = (2048, 1024, 512)   # input channels per down convolution, first that divides
FORMAT_VERSION = "gated-ffn-share-1"


def default_cache_dir():
    return Path(os.environ.get("FLASHRT_COREML_CACHE", Path.home() / ".cache" / "flash_rt" / "coreml"))


def gated_ffn_share(gate, up, down, *, rows, spatial, cache_dir=None):
    """Return (compiled model path, input name, output name, shape) for one FFN share.

    gate/up: [channels, hidden] and down: [hidden, channels] tensors (any float dtype; CPU or device).
    rows must be a multiple of spatial; the program is fixed-shape.
    """
    try:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Neural Engine FFN shares need coremltools (pip install coremltools)") from exc
    channels, hidden = gate.shape
    if up.shape != gate.shape or down.shape != (hidden, channels) or rows % spatial:
        raise ValueError(f"Invalid FFN share shapes: gate={tuple(gate.shape)}, up={tuple(up.shape)}, "
                         f"down={tuple(down.shape)}, rows={rows}, spatial={spatial}")
    arrays = [t.detach().to("cpu").float().numpy().astype(np.float16) for t in (gate, up, down)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("FFN share weights must be finite in FP16")
    shape = (1, channels, spatial, rows // spatial)
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    chunk = next((c for c in DOWN_CHUNKS if hidden % c == 0), hidden)
    digest.update(repr((FORMAT_VERSION, shape, chunk, ct.__version__)).encode())
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    key = digest.hexdigest()[:24]
    compiled, meta = root / f"{key}.mlmodelc", root / f"{key}.json"
    if compiled.is_dir() and meta.is_file():
        names = json.loads(meta.read_text())
        return str(compiled), names["input"], names["output"], shape
    wg = np.ascontiguousarray(arrays[0].T)[:, :, None, None]
    wu = np.ascontiguousarray(arrays[1].T)[:, :, None, None]
    wd = np.ascontiguousarray(arrays[2].T)[:, :, None, None]            # [channels, hidden, 1, 1]
    parts = hidden // chunk
    chunks = np.split(wd, parts, axis=1)

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)], opset_version=ct.target.macOS15)
    def program(x):
        h = mb.mul(x=mb.gelu(x=mb.conv(x=x, weight=wg), mode="TANH_APPROXIMATION"), y=mb.conv(x=x, weight=wu))
        pieces = mb.split(x=h, num_splits=parts, axis=1) if parts > 1 else [h]
        y = None
        for piece, weight in zip(pieces, chunks):
            z = mb.conv(x=piece, weight=weight)
            y = z if y is None else mb.add(x=y, y=z)
        return y

    model = ct.convert(program, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                       compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.macOS15)
    spec = model.get_spec()
    names = dict(input=spec.description.input[0].name, output=spec.description.output[0].name)
    package = root / f"{key}.mlpackage"
    model.save(str(package))
    staging = root / f"{key}.tmp.mlmodelc"
    ct.utils.compile_model(str(package), str(staging))
    shutil.rmtree(package)
    os.replace(staging, compiled)
    meta.write_text(json.dumps(names))
    return str(compiled), names["input"], names["output"], shape
