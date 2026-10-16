"""Orbax initialization and graph replay for the Pi0.5 RTX precision tiers.

Requires PI05_LIBERO_ORBAX_CHECKPOINT and PI05_REAL_OBSERVATIONS_NPZ (the
n/img_i/wrist_i/state_i NPZ format used by tools/bench_pi05_e2e.py), CUDA SM120
and a build with the Pi0.5 NVFP4 kernels. See docs/pi05_prefix_nvfp4.md for the
verified cuBLASLt runtime. No synthetic observations or checkpoint stubs.
"""
import gc
import os

import numpy as np
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not os.environ.get("PI05_LIBERO_ORBAX_CHECKPOINT")
    or not os.environ.get("PI05_REAL_OBSERVATIONS_NPZ"),
    reason="requires a CUDA device, Orbax checkpoint and real observation fixture",
)
@pytest.mark.parametrize("tier", ["fp8", "nvfp4"])
def test_orbax_precision_tier_initializes_and_replays_fresh_observations(tier):
    from flash_rt.api import VLAModel
    from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx

    with np.load(os.environ["PI05_REAL_OBSERVATIONS_NPZ"], allow_pickle=False) as data:
        assert int(data["n"]) >= 2
        observations = [
            {"image": data[f"img_{i}"].copy(),
             "wrist_image": data[f"wrist_{i}"].copy(),
             "state": data[f"state_{i}"].copy()}
            for i in range(int(data["n"]))
        ]
    frontend = Pi05JaxFrontendRtx(
        os.environ["PI05_LIBERO_ORBAX_CHECKPOINT"],
        num_views=2, hardware="rtx_sm120",
        prefix_precision=tier, decoder_precision=tier,
    )
    prompt = "pick up the black bowl and place it on the plate"
    frontend.set_prompt(prompt)
    frontend.calibrate(observations)
    noise = torch.randn(
        frontend.chunk_size, 32, device="cuda", dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(2026),
    )
    first = frontend.infer(observations[0], noise=noise)["actions"].copy()
    second = frontend.infer(observations[1], noise=noise)["actions"].copy()
    repeated = frontend.infer(observations[0], noise=noise)["actions"]
    for actions in (first, second, repeated):
        assert actions.shape == (frontend.chunk_size, 7)
        assert np.isfinite(actions).all()
    np.testing.assert_array_equal(first, repeated)

    model = VLAModel(frontend, framework="jax")
    for observation in observations[:2]:
        actions = model.predict(observation, prompt=prompt)
        assert actions.shape == (frontend.chunk_size, 7)
        assert np.isfinite(actions).all()
    del model, frontend
    gc.collect()
    torch.cuda.empty_cache()
