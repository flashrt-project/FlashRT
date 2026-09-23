"""Checkpoint-gated Pi0.5 BF16 smoke tests for AMD RDNA 3.5."""

from __future__ import annotations

import os

import numpy as np
import pytest


def _checkpoint():
    path = os.environ.get("FLASH_RT_PI05_RDNA35_CKPT")
    if not path:
        pytest.skip("set FLASH_RT_PI05_RDNA35_CKPT for the RDNA 3.5 E2E test")
    if not os.path.isfile(os.path.join(path, "model.safetensors")):
        pytest.skip("RDNA 3.5 checkpoint directory lacks model.safetensors")
    return path


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
        pytest.skip("requires ROCm PyTorch and a visible device")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if arch.split(":", 1)[0] != "gfx1151":
        pytest.skip("requires gfx1151")

    import flash_rt
    loaded = flash_rt.load_model(
        _checkpoint(), framework="torch", config="pi05",
        hardware="amd_rdna35", num_views=2, use_fp8=False)
    loaded.set_prompt("pick up the object", state=np.zeros(8, dtype=np.float32))
    return loaded


def _images():
    rng = np.random.default_rng(17)
    return [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for _ in range(2)]


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


def test_optimized_path_matches_aten_fallback(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    frontend._fill_images({"images": _images()})
    noise = torch.from_numpy(
        np.random.default_rng(29).standard_normal(
            (frontend.chunk_size, 32)).astype(np.float32)
    ).to(device="cuda", dtype=torch.bfloat16)
    pipeline = frontend.pipeline
    pipeline_names = (
        "hip_rope",
        "hip_decoder",
        "hip_ffn_gate_up",
        "hip_encoder_ffn",
        "hip_large_ops",
    )
    attention_names = ("hip_gqa", "hip_encoder_gqa")
    original_pipeline = {
        name: getattr(pipeline, name) for name in pipeline_names}
    original_attention = {
        name: getattr(pipeline.attn, name) for name in attention_names}
    prepared_smallm = pipeline.gemm._smallm_weights
    try:
        for name in pipeline_names:
            setattr(pipeline, name, False)
        for name in attention_names:
            setattr(pipeline.attn, name, False)
        pipeline.gemm._smallm_weights = {}
        reference = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise).clone()

        for name, enabled in original_pipeline.items():
            setattr(pipeline, name, enabled)
        for name, enabled in original_attention.items():
            setattr(pipeline.attn, name, enabled)
        pipeline.gemm._smallm_weights = prepared_smallm
        optimized = frontend.forward_with_fixed_noise(
            frontend._image_buf, noise).clone()
    finally:
        for name, enabled in original_pipeline.items():
            setattr(pipeline, name, enabled)
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


def test_native_attention_and_smallm_match_library_fallback(model):
    torch = pytest.importorskip("torch")
    frontend = model.pipeline
    attention = frontend.pipeline.attn
    gemm = frontend.pipeline.gemm
    if not attention.hip_gqa or not gemm._smallm_weights:
        pytest.skip("enable HIP GQA and small-M to run native-path parity")

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
