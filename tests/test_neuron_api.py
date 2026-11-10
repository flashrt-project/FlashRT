"""Neuron routing must neither initialize CUDA nor silently fall back to CPU."""

import sys
from types import SimpleNamespace

import pytest


def test_neuron_route_avoids_native_extensions(monkeypatch):
    from flash_rt import _extensions, load_model

    def unexpected(*args, **kwargs):
        pytest.fail('Neuron routing loaded a GPU extension')

    monkeypatch.setattr(_extensions, 'require', unexpected)
    monkeypatch.setitem(sys.modules, 'flash_rt.neuron.pi', SimpleNamespace(
        PiNeuronFrontend=lambda checkpoint, **kwargs: SimpleNamespace(
            checkpoint=checkpoint, settings=kwargs)))
    result = load_model('checkpoint', hardware='neuron', framework='jax',
                        config='pi0', precision='bf16', num_steps=10)
    assert result.pipeline.checkpoint == 'checkpoint'
    assert result.pipeline.settings['config'] == 'pi0'
    assert result.pipeline.settings['num_steps'] == 10


@pytest.mark.parametrize('override', [
    {'framework': 'torch'}, {'config': 'groot'}, {'precision': 'fp8'},
    {'use_fp4': True}, {'use_fp16': True}, {'use_fp4_decoder': True},
])
def test_unsupported_neuron_modes_fail_before_reading_weights(override):
    from flash_rt import load_model

    options = dict(hardware='neuron', framework='jax', config='pi05', precision='bf16')
    options.update(override)
    with pytest.raises(ValueError, match='Neuron'):
        load_model('/missing/checkpoint', **options)


def test_pi_frontend_rejects_cpu_device(tmp_path):
    jax = pytest.importorskip('jax')
    from flash_rt.neuron.pi import PiNeuronFrontend

    with pytest.raises(RuntimeError, match='Neuron PJRT device'):
        PiNeuronFrontend(tmp_path, device=jax.devices('cpu')[0])


def test_zero_steps_is_not_silently_replaced_by_default():
    pytest.importorskip('jax')
    from flash_rt import load_model

    with pytest.raises(ValueError, match='must be positive'):
        load_model('/missing/checkpoint', hardware='neuron', framework='jax',
                   config='pi0', num_steps=0)
