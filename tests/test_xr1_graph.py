"""Graph opt-in routing, capture helper isolation and profile invalidation."""
from functools import wraps
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from flash_rt.amd.frontends.torch.xr1_graph import (
    _bind_helpers, _profile, _temporary_bindings)


@pytest.mark.parametrize('requested', [None, False, True])
def test_graph_opt_in_is_forwarded_only_when_requested(monkeypatch, requested):
    import flash_rt
    import flash_rt.hardware as hardware
    seen = {}

    class Frontend:
        def __init__(self, checkpoint, num_views=3, hardware=None, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(hardware, 'resolve_pipeline_class', lambda *a: Frontend)
    kwargs = {} if requested is None else {'use_graph': requested}
    flash_rt.load_model('unused', config='xr1', hardware='amd_rdna35', **kwargs)
    assert seen.get('use_graph', False) is (requested is True)
    assert ('use_graph' in seen) is (requested is True)


@pytest.mark.parametrize('kwargs', [
    {'use_graph': 1}, {'use_graph': 'true'}, {'use_graph': None},
    {'use_graph': True, 'config': 'pi05'},
    {'use_graph': True, 'framework': 'jax'},
    {'use_graph': True, 'hardware': 'rtx_sm120'},
])
def test_unsupported_graph_requests_fail_before_loading(monkeypatch, kwargs):
    import flash_rt
    import flash_rt.hardware as hardware

    def forbidden(*args):
        raise AssertionError('frontend must not be loaded')

    monkeypatch.setattr(hardware, 'resolve_pipeline_class', forbidden)
    options = dict(config='xr1', hardware='amd_rdna35') | kwargs
    with pytest.raises(ValueError, match='use_graph|XR-1'):
        flash_rt.load_model('unused', **options)


def _inputs():
    return dict(input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                image_grid_thw=torch.tensor([[1, 16, 16]]),
                pixel_values=torch.zeros(256, 1536), state=torch.zeros(1, 1, 60),
                action_mask=torch.ones(1, 30, 60), action=torch.zeros(1, 30, 60),
                prefix_length=3)


@pytest.mark.parametrize('key', ['input_ids', 'attention_mask', 'image_grid_thw'])
def test_metadata_values_invalidate_even_with_same_shape(key):
    inputs = _inputs()
    before = _profile(inputs, 5, 'robot')
    inputs[key].view(-1)[0] += 1
    assert _profile(inputs, 5, 'robot') != before


def test_dynamic_values_reuse_profile_but_shape_and_control_changes_rebuild():
    inputs = _inputs()
    before = _profile(inputs, 5, 'robot')
    for name in ('pixel_values', 'state', 'action_mask', 'action'):
        inputs[name].add_(.25)
    assert _profile(inputs, 5, 'robot') == before
    assert _profile(inputs, 2, 'robot') != before
    assert _profile(inputs, 5, 'another_robot') != before
    assert _profile(inputs | {'prefix_length': 4}, 5, 'robot') != before
    assert _profile(inputs | {'pixel_values': torch.zeros(512, 1536)}, 5, 'robot') != before
    assert _profile(inputs | {'position_ids': torch.zeros(3, 1, 3, dtype=torch.long)}, 5, 'robot') != before


def test_unsupported_profile_fails_explicitly():
    with pytest.raises(ValueError, match='input keys'):
        _profile(_inputs() | {'pixel_values_videos': torch.zeros(1)}, 5, 'robot')
    with pytest.raises(ValueError, match='batch_size'):
        _profile(_inputs() | {'input_ids': torch.ones(2, 3)}, 5, 'robot')


def _helper(value):
    return value + 1


def test_helper_copy_preserves_decorators_and_leaves_module_globals_untouched():
    def decorator(fn):
        @wraps(fn)
        def wrapped(value):
            return fn(value) * 2
        return wrapped

    @decorator
    @decorator
    def original(value):
        return _helper(value)

    bound = _bind_helpers(original, {'_helper': lambda value: value + 10})
    assert bound(3) == 52
    assert original(3) == 16
    assert _helper(3) == 4


@pytest.mark.parametrize('fail', [False, True])
def test_instance_bindings_are_restored_even_on_exception(fail):
    class Owner:
        def method(self):
            return 'class'

    owner = Owner()
    other = SimpleNamespace(method=lambda: 'instance')
    original = other.method
    try:
        with _temporary_bindings([(owner, 'method', lambda: 'bound'),
                                  (other, 'method', lambda: 'bound')]):
            assert owner.method() == other.method() == 'bound'
            if fail:
                raise RuntimeError('capture failed')
    except RuntimeError:
        assert fail
    assert 'method' not in vars(owner)
    assert owner.method() == 'class'
    assert other.method is original
