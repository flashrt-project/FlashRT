"""Checkpoint-gated Pi0.5 BF16 smoke tests for AMD RDNA wave32 targets."""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

_ARCH_TO_HARDWARE = {
    "gfx1151": "amd_rdna35",
    "gfx1201": "amd_rdna4",
}
_LIBERO_NPZ = (
    pathlib.Path(__file__).parents[1] / "examples" / "thor" / "assets" / "pi05_libero.npz"
)
_LIBERO_PROMPT = "pick up the black bowl and place it on the plate"


def _checkpoint():
    path = os.environ.get("FLASH_RT_PI05_RDNA35_CKPT")
    if not path:
        pytest.skip("set FLASH_RT_PI05_RDNA35_CKPT for the RDNA E2E test")
    if not os.path.isfile(os.path.join(path, "model.safetensors")):
        pytest.skip("RDNA checkpoint directory lacks model.safetensors")
    return path


def _visible_rdna_arch():
    torch = pytest.importorskip("torch")
    if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
        pytest.skip("requires ROCm PyTorch and a visible device")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    base = arch.split(":", 1)[0]
    if base not in _ARCH_TO_HARDWARE:
        pytest.skip("requires gfx1151 or gfx1201")
    return base


@pytest.fixture(scope="module")
def model():
    import flash_rt
    arch = _visible_rdna_arch()
    loaded = flash_rt.load_model(
        _checkpoint(), framework="torch", config="pi05",
        hardware=_ARCH_TO_HARDWARE[arch], num_views=2, use_fp8=False,
        action_dim=int(os.environ["FLASH_RT_PI05_ACTION_DIM"]) if "FLASH_RT_PI05_ACTION_DIM" in os.environ else None)
    loaded.set_prompt("pick up the object", state=np.zeros(8, dtype=np.float32))
    return loaded


def _images():
    rng = np.random.default_rng(17)
    return [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for _ in range(2)]


def test_frontend_hardware_matches_visible_isa(model):
    assert model.pipeline.hardware == _ARCH_TO_HARDWARE[_visible_rdna_arch()]


def test_rdna4_routes_decoder_projections_to_smallm(model):
    if _visible_rdna_arch() != "gfx1201":
        pytest.skip("decoder small-M routing is gfx1201-only")
    frontend = model.pipeline
    gemm = frontend.pipeline.gemm
    for name in (
        "decoder_attn_qkv_w",
        "decoder_attn_o_w",
        "decoder_ffn_gate_up_w",
        "decoder_ffn_down_w",
    ):
        stacked = frontend.weights[name]
        for index in range(stacked.shape[0]):
            assert gemm._prepared_weight_nt(stacked[index]) is not None, (
                name, index)


def test_actions_are_finite_and_match_checkpoint_profile(model):
    frontend = model.pipeline
    noise = np.random.default_rng(19).standard_normal(
        (frontend.chunk_size, 32)).astype(np.float32)
    result = frontend.infer({"images": _images()}, debug=True, noise=noise)
    assert result["raw_actions"].shape == (frontend.chunk_size, 32)
    assert result["actions"].shape == (frontend.chunk_size, frontend.action_dim)
    assert np.isfinite(result["raw_actions"]).all()
    assert np.isfinite(result["actions"]).all()


def test_fixed_noise_is_deterministic(model):
    frontend = model.pipeline
    images = _images()
    noise = np.random.default_rng(23).standard_normal(
        (frontend.chunk_size, 32)).astype(np.float32)
    first = frontend.infer({"images": images}, debug=True, noise=noise)
    second = frontend.infer({"images": images}, debug=True, noise=noise)
    assert np.array_equal(first["raw_actions"], second["raw_actions"])


def test_full_graph_matches_eager(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    frontend._fill_images({"images": _images()})
    noise = torch.from_numpy(
        np.random.default_rng(27).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)

    eager = frontend.forward_with_fixed_noise(
        frontend._image_buf, noise, use_graph=False).clone()
    captured = frontend.forward_with_fixed_noise(
        frontend._image_buf, noise, use_graph=True).clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, eager, atol=0, rtol=0)


def test_decoder_only_reuses_last_full_encoder_cache(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    frontend._fill_images({"images": _images()})
    noise = torch.from_numpy(
        np.random.default_rng(31).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)

    full = frontend.forward_with_fixed_noise(
        frontend._image_buf, noise, use_graph=False).clone()
    cached = frontend.pipeline.forward_decode_only(
        noise, use_graph=False).clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(cached, full, atol=0, rtol=0)


def test_decoder_only_graph_matches_full_graph_with_same_context(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    frontend._fill_images({"images": _images()})
    noise = torch.from_numpy(
        np.random.default_rng(37).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)

    full = frontend.forward_with_fixed_noise(
        frontend._image_buf, noise, use_graph=True).clone()
    cached = frontend.pipeline.forward_decode_only(
        noise, use_graph=True).clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(cached, full, atol=0, rtol=0)


@pytest.mark.parametrize(
    "context_change",
    ["same_length_state", "different_length_prompt"],
)
def test_decoder_only_graph_refreshes_changed_context(model, context_change):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    pipeline = frontend.pipeline
    images = _images()
    noise = torch.from_numpy(
        np.random.default_rng(41).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)
    initial_prompt = "pick up the object"
    initial_state = np.zeros(8, dtype=np.float32)

    try:
        frontend.set_prompt(initial_prompt, state=initial_state)
        initial_len = frontend._prompt_len
        initial_embeds = frontend._prompt_buf[:initial_len].clone()
        frontend._fill_images({"images": images})
        frontend.forward_with_fixed_noise(
            frontend._image_buf, noise, use_graph=True)
        pipeline.forward_decode_only(noise, use_graph=True)
        torch.cuda.synchronize()
        decoder_graph_before = pipeline._decoder_only_graph
        prompt_start = frontend.num_views * 256
        initial_prompt_k = pipeline.buf["encoder_k"][
            :, prompt_start:prompt_start + initial_len
        ].clone()

        if context_change == "same_length_state":
            changed_prompt = initial_prompt
            changed_state = np.full(8, 0.01, dtype=np.float32)
        else:
            changed_prompt = (
                "carefully pick up the object and place it on the table"
            )
            changed_state = initial_state
        frontend.set_prompt(changed_prompt, state=changed_state)
        changed_len = frontend._prompt_len

        if context_change == "same_length_state":
            assert changed_len == initial_len
            assert not torch.equal(
                frontend._prompt_buf[:changed_len], initial_embeds)
        else:
            assert changed_len != initial_len
        assert not pipeline.has_encoder_cache

        frontend._fill_images({"images": images})
        full = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise, use_graph=True).clone()
        refreshed_prompt_k = pipeline.buf["encoder_k"][
            :, prompt_start:prompt_start + min(initial_len, changed_len)
        ]
        assert not torch.equal(
            refreshed_prompt_k,
            initial_prompt_k[:, :min(initial_len, changed_len)],
        )
        cached = pipeline.forward_decode_only(
            noise, use_graph=True).clone()
        torch.cuda.synchronize()

        torch.testing.assert_close(cached, full, atol=0, rtol=0)
        assert pipeline._decoder_only_graph_prompt_len == changed_len
        if context_change == "same_length_state":
            assert pipeline._decoder_only_graph is decoder_graph_before
        else:
            assert pipeline._decoder_only_graph is not decoder_graph_before
    finally:
        frontend.set_prompt(initial_prompt, state=initial_state)


def test_frontend_cache_frames_alternates_full_and_decoder_only(model):
    frontend = model.pipeline
    images = _images()
    changed_images = [np.roll(image, 1, axis=1) for image in images]
    noise = np.random.default_rng(43).standard_normal(
        (frontend.chunk_size, 32)).astype(np.float32)
    original_cache_frames = frontend._cache_frames
    try:
        frontend._cache_frames = 2
        frontend.set_prompt(
            "pick up the object", state=np.zeros(8, dtype=np.float32))
        full = frontend.infer({"images": images}, debug=True, noise=noise)
        cached = frontend.infer(
            {"images": changed_images}, debug=True, noise=noise)
        refreshed = frontend.infer(
            {"images": changed_images}, debug=True, noise=noise)

        np.testing.assert_array_equal(cached["raw_actions"], full["raw_actions"])
        assert not np.array_equal(
            refreshed["raw_actions"], full["raw_actions"])
    finally:
        frontend._cache_frames = original_cache_frames
        frontend.set_prompt(
            "pick up the object", state=np.zeros(8, dtype=np.float32))


def _compare_default_with_aten_fallback(frontend, noise):
    torch = pytest.importorskip("torch")
    pipeline = frontend.pipeline
    pipeline_names = (
        "fused_rope",
        "fused_decoder_ops",
        "merged_decoder_ffn",
        "merged_encoder_ffn",
        "fused_large_ops",
    )
    attention_names = ("hip_gqa", "hip_encoder_gqa")
    original_pipeline = {
        name: getattr(pipeline.ops, name) for name in pipeline_names}
    original_attention = {
        name: getattr(pipeline.attn, name) for name in attention_names}
    prepared_smallm = pipeline.gemm._smallm_weights
    try:
        for name in pipeline_names:
            setattr(pipeline.ops, name, False)
        for name in attention_names:
            setattr(pipeline.attn, name, False)
        pipeline.gemm._smallm_weights = {}
        reference = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise).clone()

        for name, enabled in original_pipeline.items():
            setattr(pipeline.ops, name, enabled)
        for name, enabled in original_attention.items():
            setattr(pipeline.attn, name, enabled)
        pipeline.gemm._smallm_weights = prepared_smallm
        optimized = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise).clone()
    finally:
        for name, enabled in original_pipeline.items():
            setattr(pipeline.ops, name, enabled)
        for name, enabled in original_attention.items():
            setattr(pipeline.attn, name, enabled)
        pipeline.gemm._smallm_weights = prepared_smallm

    cosine = torch.nn.functional.cosine_similarity(
        reference.float().flatten(), optimized.float().flatten(), dim=0)
    assert cosine.item() >= 0.999
    # Merged BF16 GEMMs and the runtime-selected hipBLASLt algorithms change
    # accumulation order relative to the split ATen reference. Keep the
    # end-to-end bound wide enough for that composition while the operator
    # tests retain substantially tighter per-kernel tolerances.
    torch.testing.assert_close(optimized, reference, atol=1e-1, rtol=3e-2)


def test_optimized_path_matches_aten_fallback(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    frontend._fill_images({"images": _images()})
    noise = torch.from_numpy(
        np.random.default_rng(29).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)
    _compare_default_with_aten_fallback(frontend, noise)


def test_native_attention_and_smallm_match_library_fallback(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    attention = frontend.pipeline.attn
    gemm = frontend.pipeline.gemm
    missing = []
    if not attention.hip_gqa:
        missing.append("HIP GQA")
    if not gemm._smallm_weights:
        missing.append("small-M WMMA")
    if missing:
        pytest.skip(
            "native-path parity needs " + " and ".join(missing))

    frontend._fill_images({"images": _images()})
    noise = torch.from_numpy(
        np.random.default_rng(41).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)
    packed_weights = gemm._smallm_weights
    try:
        attention.hip_gqa = False
        gemm._smallm_weights = {}
        reference = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise).clone()

        attention.hip_gqa = True
        gemm._smallm_weights = packed_weights
        optimized = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise).clone()
    finally:
        attention.hip_gqa = True
        gemm._smallm_weights = packed_weights

    cosine = torch.nn.functional.cosine_similarity(
        reference.float().flatten(), optimized.float().flatten(), dim=0)
    assert cosine.item() >= 0.999
    torch.testing.assert_close(optimized, reference, atol=1e-2, rtol=1e-2)


def _libero_observations():
    if not _LIBERO_NPZ.is_file():
        pytest.skip(f"LIBERO observations are not present: {_LIBERO_NPZ}")
    with np.load(_LIBERO_NPZ, allow_pickle=False) as data:
        count = int(data["n"])
        return [
            {
                "images": [
                    np.array(data[f"img_{index}"]),
                    np.array(data[f"wrist_{index}"]),
                ],
                "state": np.array(data[f"state_{index}"], dtype=np.float32),
            }
            for index in range(count)
        ]


def test_real_libero_frames_are_deterministic_and_match_aten_fallback(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    observations = _libero_observations()
    assert observations, "LIBERO fixture contains no observations"
    original_cache_frames = frontend._cache_frames
    try:
        frontend._cache_frames = 1
        for index, observation in enumerate(observations):
            frontend.set_prompt(_LIBERO_PROMPT, state=observation["state"])
            noise = np.random.default_rng(1000 + index).standard_normal(
                (frontend.chunk_size, 32)).astype(np.float32)
            first = frontend.infer(
                {"images": observation["images"]}, debug=True, noise=noise)
            second = frontend.infer(
                {"images": observation["images"]}, debug=True, noise=noise)
            assert first["actions"].shape == (
                frontend.chunk_size, frontend.action_dim)
            assert np.isfinite(first["raw_actions"]).all()
            assert np.isfinite(first["actions"]).all()
            np.testing.assert_array_equal(
                first["raw_actions"], second["raw_actions"])

        observation = observations[0]
        frontend.set_prompt(_LIBERO_PROMPT, state=observation["state"])
        frontend._fill_images({"images": observation["images"]})
        noise = torch.from_numpy(
            np.random.default_rng(29).standard_normal(
                (frontend.chunk_size, 32)).astype(np.float32)
        ).to(device="cuda", dtype=torch.bfloat16)
        _compare_default_with_aten_fallback(frontend, noise)
    finally:
        frontend._cache_frames = original_cache_frames
        frontend.set_prompt(
            "pick up the object", state=np.zeros(8, dtype=np.float32))


def test_independent_reference_fixture(model):
    """Optional independent oracle; same-pipeline fallback tests are not an oracle."""
    import hashlib
    import json
    from pathlib import Path
    fixture_path = os.environ.get('FLASH_RT_PI05_REFERENCE')
    if not fixture_path:
        pytest.skip('set FLASH_RT_PI05_REFERENCE to an independently generated NPZ fixture')
    frontend = model.pipeline
    with np.load(fixture_path, allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture['metadata'].item()))
        assert metadata['producer'] == 'openpi', 'fixture must come from independent OpenPI'
        revision = metadata['producer_revision']
        assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
        digest = hashlib.sha256()
        with Path(frontend._checkpoint_path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        assert metadata['checkpoint_sha256'] == digest.hexdigest()
        assert metadata['num_steps'] == frontend.num_steps
        assert metadata['action_dim'] == frontend.action_dim
        assert metadata['action_horizon'] == frontend.chunk_size
        prompt = str(fixture['prompt'].item())
        frontend.set_prompt(prompt, state=fixture['state'])
        result = frontend.infer({'images': list(fixture['images'])}, debug=True, noise=fixture['noise'])
        for key in ('raw_actions', 'actions'):
            actual, expected = result[key], fixture[key]
            assert actual.shape == expected.shape
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            # BF16 end-to-end acceptance bounds; do not loosen them based on a failing run.
            np.testing.assert_allclose(actual, expected, atol=.1, rtol=.03)
            x, y = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            assert denom > 0 and np.dot(x, y) / denom >= .999
