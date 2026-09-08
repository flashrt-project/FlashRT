"""CPU setup checks for mixed-kernel geometry and per-runner scratch ownership."""
from types import SimpleNamespace
import pytest
import torch
from flash_rt.npu.core.gu_int8 import GuInt8Weights
from flash_rt.npu.models.pi05.quantization import NativeEncoderMlp, prepare_encoder_mlp


@pytest.mark.parametrize('shape', [(2048,128), (2048,768), (65536,65536)])
def test_rejects_truncated_or_overflowing_weight_geometry(shape):
    # Geometry must be rejected before allocating device storage or loading CANN.
    weight = SimpleNamespace(ndim=2, dtype=torch.int8, device=SimpleNamespace(type='npu'), shape=shape)
    with pytest.raises(ValueError, match='gate/up requires'):
        GuInt8Weights.create(weight, weight, None, None)


def test_scratch_is_shared_within_runner_and_isolated_between_runners():
    class Weight:
        tensor = torch.empty(1)
        def scales_for_rows(self, rows, columns, device):
            return torch.ones(rows), torch.ones(rows), None
    class Packed:
        packed = torch.empty(1)
        columns, hidden = 2048, 16384
        def prepare(self, rows, acts, inverse, workspace):
            return SimpleNamespace(workspace=workspace)
    weight = Weight()
    norm = SimpleNamespace(group=SimpleNamespace(weights=(weight,)), producer=object())
    site = NativeEncoderMlp(norm, weight, Packed())
    original = {'first':site, 'second':site}
    a = prepare_encoder_mlp(original, 572)
    b = prepare_encoder_mlp(original, 572)
    assert a['first'].gu.workspace is a['second'].gu.workspace
    assert a['first'].gu.workspace.data_ptr() != b['first'].gu.workspace.data_ptr()
    assert original['first'] is site


def test_no_native_sites_require_no_workspace():
    weights = {'ordinary':object()}
    assert prepare_encoder_mlp(weights, 572) is weights
    assert prepare_encoder_mlp(None, 572) is None
