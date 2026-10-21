"""Offline checkpoint tests: exact graph parity, live inputs and bounded recapture."""
import gc
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')


@pytest.mark.parametrize('foundation', [False, True], ids=['robocasa', '5b'])
@pytest.mark.parametrize('native', [False, True], ids=['reference', 'native'])
def test_full_graph_live_inputs_and_recapture(foundation, native, monkeypatch):
    env = 'FLASHRT_XR1_5B_CKPT' if foundation else 'FLASHRT_XR1_CKPT'
    checkpoint = os.environ.get(env)
    if not checkpoint:
        pytest.skip(f'set {env} to a local checkpoint')
    if (not torch.version.hip or not torch.cuda.is_available()
            or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1151'):
        pytest.skip('requires ROCm gfx1151')
    from flash_rt import load_model

    model = load_model(checkpoint, config='xr1', hardware='amd_rdna35',
                       num_views=3, fuse=native, use_graph=True, autotune=0)
    frontend = model.pipeline
    captured_raw = []
    decode = frontend.processor.decode_action

    def record_raw(actions, **kwargs):
        captured_raw[:] = [actions.float().cpu().numpy().copy()]
        return decode(actions, **kwargs)

    monkeypatch.setattr(frontend.processor, 'decode_action', record_raw)
    vlm = frontend.model.vlm.model
    owners = [(frontend.model, 'forward'), (vlm, 'forward'),
              (vlm, 'get_placeholder_mask'), (vlm.visual, 'rot_pos_emb'),
              (vlm.visual, 'fast_pos_embed_interpolate'),
              (vlm.language_model, 'forward'), (vlm.language_model, '_deepstack_process')]
    before = [(owner, name, name in vars(owner), vars(owner).get(name))
              for owner, name in owners]
    rng = np.random.default_rng(235)

    def images(hw=256):
        return [rng.integers(0, 256, (hw, hw, 3), dtype=np.uint8) for _ in range(3)]

    checks = 0

    def compare(obs, prompt='pick up the red block', *, rebuild=False):
        nonlocal checks
        runner = frontend.graph_runner
        count = runner.capture_count if runner else 0
        cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
        frontend.use_graph = False
        expected = model.predict(obs, prompt)
        expected_raw = captured_raw[0]
        frontend.use_graph = True
        actual = model.predict(obs, prompt)
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(captured_raw[0], expected_raw)
        assert torch.equal(cpu, torch.get_rng_state())
        assert torch.equal(gpu, torch.cuda.get_rng_state())
        assert frontend.graph_runner.capture_count == count + int(rebuild)
        for owner, name, existed, value in before:
            assert (name in vars(owner)) is existed
            assert vars(owner).get(name) is value
        checks += 1
        return actual

    try:
        obs = dict(images=images(), state=np.zeros(60 if foundation else 8, np.float32), seed=42)
        baseline = compare(obs, rebuild=True)
        np.testing.assert_array_equal(compare(obs), baseline)
        for seed in (0, 123):
            assert not np.array_equal(compare(obs | {'seed': seed}), baseline)
        assert not np.array_equal(compare(obs | {'state': obs['state'] + .3}), baseline)
        assert not np.array_equal(compare(obs | {'images': images()}), baseline)
        # Both same-length token changes and longer prompts must invalidate.
        compare(obs, 'pick up the blue block', rebuild=True)
        compare(obs, 'place the red object carefully on the kitchen table', rebuild=True)
        obs['images'] = images(336)
        compare(obs, rebuild=True)
        frontend.num_steps = 2
        compare(obs, rebuild=True)
        frontend.num_steps = 5
        baseline = compare(obs, rebuild=True)
        if foundation:
            prefix = baseline[:3].copy()
            with_prefix = compare(obs | {'action_prefix': prefix}, rebuild=True)
            np.testing.assert_allclose(with_prefix[:3], prefix, atol=.015, rtol=.02)
            changed_prefix = prefix.copy()
            # Padded output channels are zeroed by decode_action; change only
            # active Cartesian channels when checking prefix preservation.
            changed_prefix[:, :3] += .1
            result = compare(obs | {'action_prefix': changed_prefix})
            assert not np.array_equal(result, with_prefix)
            np.testing.assert_allclose(result[:3], changed_prefix, atol=.015, rtol=.02)
            compare(obs | {'action_prefix': baseline[:4].copy()}, rebuild=True)
            observations = os.environ.get('FLASHRT_XR1_OBSERVATIONS')
            if observations:
                with np.load(observations, allow_pickle=False) as data:
                    for i in range(len(data['frames'])):
                        real = dict(images=[data[f'image_{i}_{j}'] for j in range(3)],
                                    state=data[f'state_{i}'], seed=42)
                        compare(real, 'Load washer.', rebuild=(i == 0))

        runner = frontend.graph_runner
        print(json.dumps(dict(checkpoint='5b' if foundation else 'robocasa',
                              native=native, exact_raw_and_decoded_checks=checks,
                              captures=runner.capture_count, max_abs=0.0)), flush=True)
        # A failed capture must restore all bindings, surface the error and
        # leave the eager path usable. This uses an injected warmup failure.
        runner.clear()
        with monkeypatch.context() as patch:
            def fail(*args):
                raise RuntimeError('injected capture failure')
            patch.setattr(runner, '_call', fail)
            with pytest.raises(RuntimeError, match='injected capture failure'):
                model.predict(obs, 'pick up the red block')
        assert runner.graph is None and runner.key is None
        compare(obs, rebuild=True)
    finally:
        if frontend.graph_runner is not None:
            frontend.graph_runner.clear()
        monkeypatch.undo()
        del model, frontend
        gc.collect()
        torch.cuda.empty_cache()
