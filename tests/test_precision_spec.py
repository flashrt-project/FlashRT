"""Unit tests for flash_rt.core.precision_spec.

CPU-only, no CUDA / torch. Covers:
  * validate() accepts the v1 canonical combo
  * validate() raises NotImplementedError for every unsupported combo
  * validate() raises ValueError for malformed combos (orthogonal to capability)
  * JSON round-trip preserves all fields (including numpy arrays)
"""

import json
import os
import tempfile

import numpy as np
import pytest

from flash_rt.core.precision_spec import ModelPrecisionSpec, PrecisionSpec


def test_default_is_canonical_supported():
    s = PrecisionSpec()
    assert s.dtype == "fp8_e4m3"
    assert s.granularity == "per_tensor"
    assert s.scheme == "symmetric"
    s.validate()  # must not raise


def test_explicit_canonical_validates():
    s = PrecisionSpec(
        dtype="fp8_e4m3", granularity="per_tensor", scheme="symmetric",
        scale=np.array([0.05], dtype=np.float32),
        calibration_method="percentile",
        calibration_samples=64,
        calibration_percentile=99.9,
    )
    s.validate()


def test_unsupported_dtype_raises_notimplemented():
    for dtype in ("fp16", "bf16", "fp8_e5m2", "nvfp4", "int4"):
        with pytest.raises(NotImplementedError):
            PrecisionSpec(dtype=dtype).validate()


def test_int8_per_tensor_symmetric_validates():
    s = PrecisionSpec(
        dtype="int8",
        granularity="per_tensor",
        scheme="symmetric",
        scale_source="runtime_dynamic",
    )
    s.validate()


def test_per_channel_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        PrecisionSpec(granularity="per_channel").validate()


def test_per_group_raises_notimplemented():
    # per_group requires group_size else it's a ValueError; provide one,
    # then the capability check still fails with NotImplementedError.
    with pytest.raises(NotImplementedError):
        PrecisionSpec(granularity="per_group", group_size=128).validate()


def test_asymmetric_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        PrecisionSpec(scheme="asymmetric",
                      zero_point=np.zeros(1, dtype=np.int8)).validate()


def test_asymmetric_without_zero_point_is_valueerror():
    # This is a structural bug, not a capability gap.
    with pytest.raises(ValueError):
        PrecisionSpec(scheme="asymmetric").validate()


def test_group_size_without_per_group_is_valueerror():
    with pytest.raises(ValueError):
        PrecisionSpec(granularity="per_tensor", group_size=128).validate()


def test_per_group_without_group_size_is_valueerror():
    with pytest.raises(ValueError):
        PrecisionSpec(granularity="per_group").validate()


def test_precision_spec_roundtrip_to_dict():
    s = PrecisionSpec(
        scale=np.array([0.02, 0.05, 0.1], dtype=np.float32),
        calibration_method="percentile",
        calibration_samples=128,
        calibration_percentile=99.9,
    )
    s2 = PrecisionSpec.from_dict(s.to_dict())
    assert s2.dtype == s.dtype
    assert s2.calibration_samples == 128
    assert s2.calibration_percentile == 99.9
    np.testing.assert_array_equal(s2.scale, s.scale)


def test_model_precision_spec_json_roundtrip():
    m = ModelPrecisionSpec(source="calibration")
    m.encoder_layer_specs["layer0.qkv"] = PrecisionSpec(
        scale=np.array([0.03], dtype=np.float32),
        calibration_method="percentile", calibration_samples=64,
        calibration_percentile=99.9)
    m.decoder_layer_specs["layer3.down"] = PrecisionSpec(
        scale=np.array([0.07], dtype=np.float32),
        calibration_method="single_frame", calibration_samples=1)
    m.validate()

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "spec.json")
        m.to_json(p)
        m2 = ModelPrecisionSpec.from_json(p)
    assert set(m2.encoder_layer_specs.keys()) == {"layer0.qkv"}
    np.testing.assert_array_equal(
        m2.encoder_layer_specs["layer0.qkv"].scale,
        m.encoder_layer_specs["layer0.qkv"].scale)
    assert m2.decoder_layer_specs["layer3.down"].calibration_method == "single_frame"
    assert m2.source == "calibration"


def test_model_precision_spec_validate_propagates():
    m = ModelPrecisionSpec()
    m.encoder_layer_specs["bad"] = PrecisionSpec(dtype="int4")
    with pytest.raises(NotImplementedError):
        m.validate()
