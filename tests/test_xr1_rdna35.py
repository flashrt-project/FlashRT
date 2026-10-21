"""XR-1 API routing and observation contracts, without checkpoint downloads."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest


def test_load_model_xr1_requires_no_compiled_extension(monkeypatch):
    import flash_rt
    import flash_rt.hardware as hardware
    import flash_rt._extensions as extensions
    captured = {}

    class Frontend:
        def __init__(self, checkpoint, num_views=3, hardware=None, use_fp8=False,
                     num_steps=5, embodiment_tag='robocasa_mg', action_dim=7,
                     attention='sdpa', fuse=False):
            captured.update(locals())

    def forbidden(*args, **kwargs):
        raise AssertionError('XR-1 reference requested a compiled extension')

    monkeypatch.setattr(hardware, 'detect_arch', lambda: 'amd_rdna35')
    monkeypatch.setattr(hardware, 'resolve_pipeline_class', lambda *a: Frontend)
    monkeypatch.setattr(extensions, 'require', forbidden)
    monkeypatch.setitem(sys.modules, 'flash_rt.amd.flash_rt_amd_kernels', None)
    model = flash_rt.load_model('local-checkpoint', config='xr1', num_views=3,
                               num_steps=2, embodiment_tag='robot', action_dim=6,
                               attention='eager', fuse=False)
    assert isinstance(model, flash_rt.VLAModel)
    assert captured['use_fp8'] is False
    assert (captured['num_steps'], captured['embodiment_tag'], captured['action_dim'],
            captured['attention'], captured['fuse']) == (2, 'robot', 6, 'eager', False)


@pytest.mark.parametrize('kwargs', [
    {'framework': 'jax', 'hardware': 'amd_rdna35'},
    {'hardware': 'rtx_sm120'}, {'hardware': 'amd_cdna4'},
    {'hardware': 'amd_rdna35', 'use_fp4': True},
    {'hardware': 'amd_rdna35', 'use_fp16': True},
    {'hardware': 'amd_rdna35', 'action_horizon': 10},
])
def test_invalid_xr1_route_fails_before_extension_import(kwargs):
    import flash_rt
    with pytest.raises(ValueError, match='XR-1'):
        flash_rt.load_model('unused', config='xr1', **kwargs)


def test_registration_is_lazy():
    from flash_rt.hardware import _PIPELINE_MAP
    assert _PIPELINE_MAP['xr1', 'torch', 'amd_rdna35'] == (
        'flash_rt.amd.frontends.torch.xr1_rdna35', 'XR1TorchFrontendAmdRdna35')


@pytest.fixture
def frontend():
    from flash_rt.amd.frontends.torch.xr1_rdna35 import XR1TorchFrontendAmdRdna35
    x = object.__new__(XR1TorchFrontendAmdRdna35)
    x.prompt, x.num_views, x.state_dim = 'pick up object', 3, 60
    x.foundation = False
    return x


@pytest.mark.parametrize('state', [None, [], np.full(61, 0.), [np.nan], [np.inf]])
def test_invalid_state_is_rejected_before_processor(frontend, state):
    with pytest.raises(ValueError, match='state|State'):
        frontend.infer({'images': [np.zeros((256, 256, 3), np.uint8)] * 3,
                        'state': state})


def test_wrong_view_count_is_rejected(frontend):
    with pytest.raises(ValueError, match='Expected 3 images'):
        frontend.infer({'images': [], 'state': np.zeros(8)})


def test_public_wrapper_forwards_new_state_with_reused_prompt():
    from flash_rt.api import VLAModel
    seen = []
    class Pipe:
        def set_prompt(self, prompt):
            seen.append(prompt)
        def infer(self, obs):
            return {'actions': obs['state'][None]}
    model = VLAModel(Pipe(), 'torch')
    images = [np.zeros((16, 16, 3), np.uint8)]
    np.testing.assert_array_equal(model.predict(images, 'task', np.array([1.])), [[1.]])
    np.testing.assert_array_equal(model.predict(images, state=np.array([2.])), [[2.]])
    assert seen == ['task']


@pytest.fixture
def foundation_processor(tmp_path, monkeypatch):
    import json
    import torch
    import transformers
    from flash_rt.amd.models.xr1.checkpoint_rdna35 import XR1FoundationProcessor
    stats = {'mean': np.full((30, 60), .1).tolist(),
             'std': np.full((30, 60), .5).tolist(),
             'q01': np.full((1, 60), -1.).tolist(),
             'q99': np.full((1, 60), 1.).tolist(), 'embodiment': 'test_robot'}
    (tmp_path / 'normalization.json').write_text(json.dumps(stats))
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return {'input_ids': torch.tensor([[1, 2, 3]])}
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained',
                        lambda *a, **kw: Processor())
    return XR1FoundationProcessor(tmp_path)


def test_foundation_normalizes_state_and_preserves_prefix(foundation_processor):
    import torch
    p = foundation_processor
    state = np.zeros((1, 1, 60), np.float32)
    state[..., 0] = .5
    prefix = np.full((3, 60), .2, np.float32)
    batch = p.apply_chat_template([], state=state, robot_type='test_robot',
                                  action_prefix=prefix)
    assert batch['state'].shape == (1, 1, 60)
    torch.testing.assert_close(batch['state'][0, 0, 0], torch.tensor(.5, dtype=torch.bfloat16))
    assert batch['prefix_length'] == 3
    decoded = p.decode_action(batch['action'], 'test_robot')
    np.testing.assert_allclose(decoded[0, :3, 0].numpy(), .2, atol=.001)
    np.testing.assert_array_equal(decoded[0, :, 7].numpy(), 0.)
    np.testing.assert_array_equal(decoded[0, :, 20:].numpy(), 0.)
    # Upstream uses std + ACTION_EPS in both directions, including decode.
    one_sigma = p.decode_action(torch.ones_like(batch['action']), 'test_robot')
    assert float(one_sigma[0, 0, 0]) == pytest.approx(.600001, abs=1e-7, rel=0)


@pytest.mark.parametrize('prefix', [np.zeros((0, 60)), np.zeros((31, 60)),
                                   np.zeros((3, 7)), np.full((3, 60), np.nan)])
def test_foundation_rejects_invalid_prefix(foundation_processor, prefix):
    with pytest.raises(ValueError, match='action_prefix'):
        foundation_processor.apply_chat_template(
            [], state=np.zeros((1, 1, 60)), robot_type='test_robot', action_prefix=prefix)


def test_foundation_resize_matches_runtime_boundaries():
    from PIL import Image
    from flash_rt.amd.models.xr1.checkpoint_rdna35 import XR1FoundationProcessor
    sizes = [(224, 224), (256, 256), (336, 336), (640, 480), (1920, 1080)]
    expected = [(224, 224), (256, 256), (320, 320), (448, 320), (512, 288)]
    actual = XR1FoundationProcessor.prepare_images([Image.new('RGB', size) for size in sizes])
    assert [image.size for image in actual] == expected


@pytest.mark.parametrize('prompt', ['', '   ', None, 42])
def test_invalid_prompt_is_rejected(frontend, prompt):
    with pytest.raises(ValueError, match='prompt'):
        frontend.set_prompt(prompt)


def test_reference_pipeline_keeps_extensions_unloaded(monkeypatch):
    from flash_rt.amd.models.xr1.pipeline_rdna35 import XR1PipelineRdna35
    monkeypatch.setitem(sys.modules, 'flash_rt.amd.flash_rt_amd_kernels', None)
    model = object()
    pipeline = XR1PipelineRdna35(model, native=False)
    assert pipeline.model is model and pipeline.gemm is None
