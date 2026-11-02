"""Real-observation GR00T N1.7 BF16 regression tests for RDNA.

Set FLASH_RT_GROOT_N17_RDNA_CKPT, FLASH_RT_GROOT_N17_RDNA_REFERENCE and
FLASH_RT_GROOT_N17_PROCESSOR to a checkpoint, independent official-policy
fixture, and Cosmos processor directory. Reference generation stays outside
FlashRT; expected outputs must not be produced by the path under test.
FLASH_RT_GROOT_N17_RDNA_EXPANDED_REFERENCE additionally enables a matrix of
official outputs across observations, diffusion seeds and denoising schedules.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _required_path(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} for the RDNA GR00T reference tests")
    if not os.path.exists(value):
        pytest.fail(f"{name} does not exist: {value}")
    return value


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires ROCm PyTorch")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    hardware = {"gfx1100": "amd_rdna3", "gfx1151": "amd_rdna35", "gfx1201": "amd_rdna4"}
    if arch not in hardware:
        pytest.skip("requires a supported RDNA target")
    checkpoint = _required_path("FLASH_RT_GROOT_N17_RDNA_CKPT")
    reference = _required_path("FLASH_RT_GROOT_N17_RDNA_REFERENCE")
    processor_path = _required_path("FLASH_RT_GROOT_N17_PROCESSOR")
    import flash_rt
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    model = flash_rt.load_model(checkpoint, config="groot_n17", framework="torch",
                               hardware=hardware[arch], num_views=2, use_fp8=False,
                               fuse_image_embeds=True)
    processor = Gr00tN1d7Processor.from_pretrained(checkpoint, model_name=processor_path)
    model.pipeline.set_hf_processor(processor)
    entries = torch.load(reference, map_location="cpu", weights_only=False)
    assert len(entries) >= 3, "reference must contain multiple independent real observations"
    assert model.pipeline.hardware == hardware[arch]
    return model, entries


def test_official_preprocessing_and_physical_actions(runtime):
    import torch

    model, entries = runtime
    frontend = model.pipeline
    for entry in entries:
        state, aux = frontend.prepare_observation(entry["inputs"])
        for key in ("input_ids", "grid_thw", "visual_pos_masks", "rope_cos", "rope_sin"):
            assert torch.equal(aux[key].cpu(), entry[key].cpu()), key
        assert torch.equal(aux["pixel_values"].bfloat16(), entry["pixel_values"].bfloat16())
        assert torch.equal(state.bfloat16(), entry["state_normalized"].bfloat16())
        frontend.set_prompt(aux=aux)
        eager = frontend.infer(state, aux=aux, initial_noise=entry["initial_noise"], use_graph=False)
        graph = frontend.infer(state, aux=aux, initial_noise=entry["initial_noise"])
        assert torch.equal(graph, eager)
        # Only the embodiment's 17 active channels count toward precision;
        # padded outputs must not inflate the cosine score.
        actual = graph[..., :17].double().cpu().flatten()
        expected = entry["action_pred"][..., :17].double().flatten()
        assert torch.isfinite(actual).all()
        assert torch.nn.functional.cosine_similarity(actual, expected, dim=0) >= 0.999
        assert (actual - expected).square().mean().sqrt() < 0.015
        assert (actual - expected).abs().max() < 0.125
        decoded = frontend.denormalize_action(graph, state_dict=entry["inputs"]["state"])
        for name, limit in (("eef_9d", 0.05), ("joint_position", 0.05), ("gripper_position", 0.05)):
            difference = np.abs(decoded[name].numpy() - np.asarray(entry["actions"][name]))
            # Joint positions are radians; eef_9d is translation + rotation6D.
            assert np.isfinite(difference).all()
            assert difference.max() < limit, (name, difference.max())
            if name == "eef_9d":
                assert difference[..., :3].max() < 0.02  # metres


def test_fresh_observation_noise_and_prompt(runtime):
    import torch

    _, entries = runtime
    frontend = runtime[0].pipeline
    first, second = entries[:2]
    frontend.set_prompt(aux=first)
    state = first["state_normalized"]
    noise = first["initial_noise"]
    baseline = frontend.infer(state, aux=first, initial_noise=noise).clone()
    graph = frontend._graph
    # Change only the frame, keeping prompt, state and diffusion noise fixed.
    changed = {**first, "pixel_values": second["pixel_values"]}
    fresh = frontend.infer(state, aux=changed, initial_noise=noise).clone()
    assert frontend._graph is graph
    assert not torch.equal(fresh, baseline)
    restored = frontend.infer(state, aux=first, initial_noise=noise)
    assert torch.equal(restored, baseline)
    changed_state = state.clone()
    changed_state[..., :17] += 0.25
    different_state = frontend.infer(changed_state, aux=first, initial_noise=noise)
    assert frontend._graph is graph
    assert not torch.equal(different_state, baseline)
    assert torch.equal(frontend.infer(state, aux=first, initial_noise=noise), baseline)
    different_noise = frontend.infer(state, aux=first, initial_noise=-noise)
    assert frontend._graph is graph
    assert not torch.equal(different_noise, baseline)
    changed_prompt = {**first, "input_ids": first["input_ids"].clone()}
    changed_prompt["input_ids"][..., 0] += 1
    with pytest.raises(ValueError, match="set_prompt"):
        frontend.infer(state, aux=changed_prompt, initial_noise=noise)


def test_delegated_infer_and_frontend_predict_contract(runtime):
    import torch

    model, entries = runtime
    frontend = model.pipeline
    entry = entries[0]
    state, aux = frontend.prepare_observation(entry["inputs"])
    model.set_prompt(aux=aux)
    expected = model.infer(state, aux=aux, initial_noise=entry["initial_noise"])
    actual = frontend.predict(state, aux=aux, initial_noise=entry["initial_noise"])
    assert actual.shape == (1, frontend.action_horizon, 132)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    graph = frontend._graph
    # State normalization follows checkpoint modality order, not dict order.
    reversed_state = dict(reversed(list(entry["inputs"]["state"].items())))
    assert torch.equal(frontend.normalize_state(reversed_state).bfloat16(), state.bfloat16())
    for method in (model.infer, frontend.predict):
        with pytest.raises(TypeError, match="normalized state tensor"):
            method(entry["inputs"])
    with pytest.raises(TypeError):
        model.set_prompt("instruction without aux")
    assert frontend._graph is graph  # malformed calls do not invalidate setup


def test_explicit_prompt_update_and_graph_reuse(runtime):
    import torch

    model, entries = runtime
    frontend = model.pipeline
    first = entries[0]
    state, aux = frontend.prepare_observation(first["inputs"])
    model.set_prompt(aux=aux)
    noise = first["initial_noise"]
    original = model.infer(state, aux=aux, initial_noise=noise).clone()
    graph = frontend._graph
    language_key = next(iter(first["inputs"]["language"]))
    observation = {**first["inputs"], "language": {language_key: [["Place the cup on the table."]]}}
    updated_state, updated_aux = frontend.prepare_observation(observation)
    assert not torch.equal(updated_aux["input_ids"], aux["input_ids"])
    assert frontend._graph is graph  # preparation alone never binds a prompt
    with pytest.raises(ValueError, match="set_prompt"):
        model.infer(updated_state, aux=updated_aux, initial_noise=noise)
    model.set_prompt(aux=updated_aux, prompt="Place the cup on the table.")
    assert frontend._graph is None
    changed = model.infer(updated_state, aux=updated_aux, initial_noise=noise).clone()
    assert not torch.equal(changed, original)
    assert torch.equal(changed, model.infer(updated_state, aux=updated_aux,
                                          initial_noise=noise, use_graph=False))
    graph = frontend._graph
    fresh_state, fresh_aux = frontend.prepare_observation(observation)
    assert torch.equal(changed, model.infer(fresh_state, aux=fresh_aux, initial_noise=noise))
    assert frontend._graph is graph
    model.set_prompt(aux=aux)
    assert torch.equal(original, model.infer(state, aux=aux, initial_noise=noise))


def test_invalid_observation_and_noise_fail_clearly(runtime):
    _, entries = runtime
    frontend = runtime[0].pipeline
    entry = entries[0]
    frontend.set_prompt(aux=entry)
    with pytest.raises(ValueError, match="pixel_values"):
        frontend.infer(entry["state_normalized"], aux={})
    with pytest.raises(ValueError, match="initial_noise"):
        frontend.infer(entry["state_normalized"], aux=entry,
                       initial_noise=entry["initial_noise"][..., :1])
    with pytest.raises(ValueError, match="camera"):
        frontend.prepare_observation({**entry["inputs"], "video": {}})
    malformed = {**entry["inputs"], "language": {"annotation.human.action.task_description": [["a"], ["b"]]}}
    with pytest.raises(ValueError, match="language"):
        frontend.prepare_observation(malformed)


@pytest.fixture(scope="module")
def projected_runtime(runtime):
    import flash_rt
    model, entries = runtime
    frontend = model.pipeline
    projected = flash_rt.load_model(str(frontend.checkpoint_path), config="groot_n17",
                                   hardware=frontend.hardware, use_fp8=False)
    projected.pipeline.set_hf_processor(frontend._hf_processor())
    return projected, entries


def test_cuda_cdna_aux_and_call_contract(projected_runtime):
    """Use the existing CUDA/CDNA workflow with independent official tensors."""
    import torch

    model, entries = projected_runtime
    frontend = model.pipeline
    keys = ("pixel_features", "llm_input_embeds", "grid_thw", "visual_pos_masks",
            "rope_cos", "rope_sin")
    for entry in entries:
        aux = {k: entry[k] for k in keys}
        state_dict = {"state." + k: v for k, v in entry["inputs"]["state"].items()}
        state = frontend.normalize_state(state_dict)
        model.set_prompt(aux=aux)
        kwargs = dict(initial_noise=entry["initial_noise"], num_inference_timesteps=4,
                      action_horizon=40, num_timestep_buckets=1000)
        actual = model.infer(state, aux=aux, **kwargs)
        # Rich official bundles include raw pixels too. The explicit mode,
        # never the presence of an extra field, chooses what is computed.
        assert torch.equal(actual, model.infer(state, aux=entry, **kwargs))
        assert torch.equal(actual, model.infer(state, **kwargs))
        assert torch.equal(actual, model.infer(state, aux=aux, use_dit_graph=False, **kwargs))
        assert torch.equal(actual, model.infer(state, aux=aux, use_graph=False, **kwargs))
        error = actual[..., :17].double().cpu() - entry["action_pred"][..., :17].double()
        assert error.square().mean().sqrt() < 0.015
        assert error.abs().max() < 0.125
        assert torch.nn.functional.cosine_similarity(
            actual[..., :17].double().cpu().flatten(),
            entry["action_pred"][..., :17].double().flatten(), dim=0) >= 0.999
        decoded = frontend.denormalize_action(actual, state_dict=state_dict)
        for key, value in decoded.items():
            error = np.abs(value.numpy() - entry["actions"][key])
            assert error.max() < 0.05
            if key == "eef_9d":
                assert error[..., :3].max() < 0.02
        # Keep caller-visible output stable across subsequent graph replay.
        saved = actual.clone()
        model.infer(state, initial_noise=-entry["initial_noise"])
        assert torch.equal(actual, saved)


def test_projected_mode_refuses_raw_only_and_frames(projected_runtime):
    model, entries = projected_runtime
    f, entry = model.pipeline, entries[0]
    raw = {k: entry[k] for k in (*f._PROMPT_KEYS, "pixel_values")}
    with pytest.raises(ValueError, match="pixel_features"):
        f.set_prompt(aux=raw)
    f.set_prompt(aux=entry)
    with pytest.raises(ValueError, match="fuse_image_embeds=True"):
        f.infer(entry["state_normalized"], frames=np.zeros((4, 180, 320, 3), dtype=np.uint8))


def test_frames_official_actions_and_eager_graph(runtime):
    import torch
    from flash_rt.amd.frontends.torch.groot_n17_rdna import frames_from_observation, process_frames

    model, entries = runtime
    f = model.pipeline
    processor = f._hf_processor()
    cameras = processor.modality_configs[f.embodiment_tag]["video"].modality_keys
    for entry in entries:
        state, aux = f.prepare_observation(entry["inputs"])
        frames = frames_from_observation(entry["inputs"], cameras)
        pixels = process_frames(processor, frames, aux["grid_thw"])
        assert torch.equal(pixels.bfloat16(), entry["pixel_values"].bfloat16())
        f.set_prompt(aux=aux)
        noise = entry["initial_noise"]
        expected = f.infer(state, aux=aux, initial_noise=noise)
        actual = model.infer(state, frames=frames, initial_noise=noise)
        assert torch.equal(actual, expected)
        assert torch.equal(actual, model.infer(state, frames=frames, initial_noise=noise, use_graph=False))
        error = actual[..., :17].double().cpu() - entry["action_pred"][..., :17].double()
        assert error.square().mean().sqrt() < 0.015 and error.abs().max() < 0.125
        decoded = f.denormalize_action(actual, state_dict=entry["inputs"]["state"])
        for key, value in decoded.items():
            error = np.abs(value.numpy() - entry["actions"][key])
            assert error.max() < 0.05
            if key == "eef_9d":
                assert error[..., :3].max() < 0.02


def test_frames_update_forms_metadata_and_output_ownership(runtime):
    import torch
    from flash_rt.amd.frontends.torch.groot_n17_rdna import frames_from_observation

    f, entries = runtime[0].pipeline, runtime[1]
    cameras = f._hf_processor().modality_configs[f.embodiment_tag]["video"].modality_keys
    entry = entries[0]
    frames = frames_from_observation(entry["inputs"], cameras)
    other = frames_from_observation(entries[1]["inputs"], cameras)
    state, noise = entry["state_normalized"], entry["initial_noise"]
    f.set_prompt(aux=entry)
    baseline = f.infer(state, frames=frames, initial_noise=noise)
    snapshot, graph = baseline.clone(), f._graph
    for form in (list(frames), torch.from_numpy(frames), torch.from_numpy(frames).to(f.device)):
        assert torch.equal(f.infer(state, frames=form, initial_noise=noise), baseline)
    fresh = f.infer(state, frames=other, initial_noise=noise)
    assert f._graph is graph and not torch.equal(fresh, baseline)
    assert torch.equal(f.infer(state, initial_noise=noise), fresh)
    assert torch.equal(snapshot, baseline)
    with pytest.raises(ValueError, match="cannot both"):
        f.infer(state, frames=frames, aux=entry)
    with pytest.raises(ValueError, match="camera/history"):
        f.infer(state, frames=frames[:2])
    changed = entry["input_ids"].clone()
    changed[..., 0] += 1
    with pytest.raises(ValueError, match="set_prompt"):
        f.infer(state, frames=frames, aux={"input_ids": changed})
    # Refused calls must not have replaced the last valid observation.
    assert torch.equal(f.infer(state, initial_noise=noise), fresh)
    f.set_prompt(aux=entry)
    assert f._graph is None
    assert torch.equal(f.infer(state, frames=frames, initial_noise=noise), baseline)


@pytest.mark.parametrize("albumentations", [True, False])
def test_square_checkpoint_transform(runtime, tmp_path, albumentations):
    import json
    import shutil
    import torch
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from flash_rt.amd.frontends.torch.groot_n17_rdna import (
        prepare_observation, frames_from_observation, process_frames,
    )

    f, entry = runtime[0].pipeline, runtime[1][0]
    _, aux = f.prepare_observation(entry["inputs"])
    # Configured square processing and the original rectangular recipe must
    # both match the full official processor. Do not force a square grid.
    # This pinned processor only forwards a small allowlist of from_pretrained
    # overrides. Configure the recipe in an isolated file, never the checkpoint.
    config = json.loads((f.checkpoint_path / "processor_config.json").read_text())
    config["processor_kwargs"]["letter_box_transform"] = True
    config["processor_kwargs"]["use_albumentations"] = albumentations
    (tmp_path / "processor_config.json").write_text(json.dumps(config))
    for name in ("statistics.json", "embodiment_id.json"):
        source = f.checkpoint_path / name
        if source.exists():
            shutil.copy2(source, tmp_path / name)
    square = Gr00tN1d7Processor.from_pretrained(
        tmp_path, model_name=os.environ["FLASH_RT_GROOT_N17_PROCESSOR"])
    _, square_aux = prepare_observation(square, entry["inputs"], f.embodiment_tag, num_views=2)
    cameras = square.modality_configs[f.embodiment_tag]["video"].modality_keys
    frames = frames_from_observation(entry["inputs"], cameras)
    assert square_aux["pixel_values"].shape[0] == 1024
    pixels = process_frames(square, frames, square_aux["grid_thw"])
    assert torch.equal(pixels.bfloat16(), square_aux["pixel_values"].bfloat16())
    with pytest.raises(ValueError, match="image grid"):
        process_frames(square, frames, aux["grid_thw"])


def test_prepared_metadata_does_not_alias_the_private_cache(runtime):
    import torch

    frontend, entry = runtime[0].pipeline, runtime[1][0]
    _, aux = frontend.prepare_observation(entry["inputs"])
    expected = {key: aux[key].clone() for key in ("rope_cos", "visual_pos_masks")}
    # Mutate fields outside the cache key. A caller modifying its own result
    # must not poison preparation for the next unchanged instruction/grid.
    aux["rope_cos"].zero_()
    aux["visual_pos_masks"].logical_not_()
    _, following = frontend.prepare_observation(entry["inputs"])
    for key, value in expected.items():
        assert torch.equal(following[key], value), key


def test_graph_controls_and_cached_observation(runtime):
    import torch

    model, entries = runtime
    f, entry = model.pipeline, entries[0]
    f.set_prompt(aux=entry)
    state, noise = entry["state_normalized"], entry["initial_noise"]
    # First infer without aux uses the setup observation, as on CUDA/CDNA.
    cached = model.infer(state, initial_noise=noise)
    assert f._graph is None and f._dit_graph is not None
    raw = model.infer(state, aux=entry, initial_noise=noise)
    assert torch.equal(raw, cached)
    original_setting = f.use_backbone_graph
    try:
        for backbone_graph, dit_graph in ((True, False), (False, True), (False, False)):
            f.set_prompt(aux=entry)
            f.use_backbone_graph = backbone_graph
            actual = model.infer(state, aux=entry, initial_noise=noise, use_dit_graph=dit_graph)
            assert torch.equal(actual, raw)
            assert (f._backbone_graph is not None) == backbone_graph
            assert (f._dit_graph is not None) == dit_graph
    finally:
        f.use_backbone_graph = original_setting
    changed = {**entry, "pixel_values": entries[1]["pixel_values"]}
    fresh = model.infer(state, aux=changed, initial_noise=noise)
    assert not torch.equal(fresh, raw)
    assert torch.equal(fresh, model.infer(state, initial_noise=noise))


def test_denoising_options_reconfigure_and_restore(runtime):
    import torch

    model, entries = runtime
    f, entry = model.pipeline, entries[0]
    f.set_prompt(aux=entry)
    state, noise = entry["state_normalized"], entry["initial_noise"]
    baseline = model.infer(state, aux=entry, initial_noise=noise)
    for horizon, steps, buckets in ((40, 2, 1000), (12, 4, 1000), (40, 4, 500)):
        options = dict(action_horizon=horizon, num_inference_timesteps=steps,
                       num_timestep_buckets=buckets, initial_noise=noise[:, :horizon])
        graph = model.infer(state, aux=entry, **options)
        eager = model.infer(state, aux=entry, use_graph=False, **options)
        assert graph.shape == (1, horizon, 132) and torch.isfinite(graph).all()
        assert torch.equal(graph, eager)
        assert not torch.equal(graph, baseline[:, :horizon])
        # Rebinding a prompt must preserve the active schedule's buffer shapes.
        f.set_prompt(aux=entry)
        assert torch.equal(graph, model.infer(state, aux=entry, **options))
    assert torch.equal(baseline, model.infer(state, aux=entry, initial_noise=noise))
    for options in (dict(action_horizon=41), dict(num_inference_timesteps=0),
                    dict(num_timestep_buckets=True), dict(action_horizon=1.5)):
        with pytest.raises(ValueError, match="positive integer"):
            model.infer(state, aux=entry, **options)


def test_setup_owns_device_metadata_and_rejects_padding(runtime):
    import torch

    model, entries = runtime
    f, entry = model.pipeline, entries[0]
    aux = {k: entry[k].to(f.device).clone() for k in (*f._PROMPT_KEYS, "pixel_values")}
    f.set_prompt(aux=aux)
    state, noise = entry["state_normalized"], entry["initial_noise"]
    original = model.infer(state, initial_noise=noise)
    aux["input_ids"].zero_()
    aux["rope_cos"].zero_()
    aux["pixel_values"].zero_()
    # An eager call catches aliasing too; graph could conceal baked-in values.
    assert torch.equal(original, model.infer(state, aux=entry, initial_noise=noise, use_graph=False))
    mask = torch.ones_like(entry["input_ids"])
    mask[..., -1] = 0
    with pytest.raises(ValueError, match="attention_mask"):
        model.infer(state, aux={**entry, "attention_mask": mask})
    assert torch.equal(original, model.infer(state, initial_noise=noise))


def test_official_observation_seed_and_schedule_matrix(runtime):
    import torch
    from flash_rt.amd.frontends.torch.groot_n17_rdna import frames_from_observation

    reference = torch.load(_required_path("FLASH_RT_GROOT_N17_RDNA_EXPANDED_REFERENCE"),
                           map_location="cpu", weights_only=False)
    model, _ = runtime
    f = model.pipeline
    cameras = f._hf_processor().modality_configs[f.embodiment_tag]["video"].modality_keys
    entries = reference["entries"]
    assert len({entry["index"] for entry in entries}) >= 3
    assert len({entry["seed"] for entry in entries}) >= 3
    assert len({entry["steps"] for entry in entries}) >= 3
    for entry in entries:
        case = (entry["index"], entry["seed"], entry["steps"])
        observation = reference["observations"][entry["index"]]
        state, aux = f.prepare_observation(observation)
        model.set_prompt(aux=aux)
        options = dict(initial_noise=entry["initial_noise"],
                       num_inference_timesteps=entry["steps"])
        eager = model.infer(state, aux=aux, use_graph=False, **options)
        graph = model.infer(state, aux=aux, **options)
        frames = model.infer(state, frames=frames_from_observation(observation, cameras), **options)
        assert torch.equal(eager, graph), case
        assert torch.equal(frames, graph), case
        actual = graph[..., :17].double().cpu().flatten()
        expected = entry["action_pred"][..., :17].double().flatten()
        difference = (actual - expected).abs()
        assert torch.isfinite(actual).all(), case
        assert torch.nn.functional.cosine_similarity(actual, expected, dim=0) >= 0.999, case
        assert difference.square().mean().sqrt() < 0.015, case
        assert difference.max() < 0.125, case
        decoded = f.denormalize_action(graph, state_dict=observation["state"])
        for name, value in decoded.items():
            error = np.abs(value.numpy() - entry["actions"][name])
            assert np.isfinite(error).all() and error.max() < 0.05, (case, name)
            if name == "eef_9d":
                assert error[..., :3].max() < 0.02, case
