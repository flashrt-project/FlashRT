"""Graph-safety gate for HyVLATorchFrontendRtx (requires SM120 + checkpoint).

Verifies the invariants the CUDA-graph capture relies on for the SM120
block-128 tier (``use_fp8=True``):

  1. graph == eager — replaying the captured graph reproduces the un-captured
     eager action chunk for identical inputs (cosine >= 0.9999).
  2. replay stability — two replays on identical static inputs agree
     (cosine >= 0.9999). The SM120 expert o/dn GEMMs use split-K with atomic
     accumulation, so replay is *not* bit-identical; the gate is cosine-based,
     mirroring the fused-vs-unfused tolerance used for the Orin path.

Inputs are fully synthetic and deterministic (seed-0 torch.rand images/state,
RandomState(0) noise), so the gate is reproducible with only the checkpoint.
Set FLASHRT_HYVLA_CHECKPOINT to the Hy-Embodied-0.5-VLA directory to run.
"""

import os

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

CKPT = os.environ.get("FLASHRT_HYVLA_CHECKPOINT", "")
if not CKPT or not os.path.isdir(CKPT):
    pytest.skip(
        "set FLASHRT_HYVLA_CHECKPOINT to the Hy-Embodied-0.5-VLA checkpoint "
        "directory to run this gate", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

if tuple(torch.cuda.get_device_capability()) != (12, 0):
    pytest.skip("requires an SM120 (RTX Blackwell) device",
                allow_module_level=True)

from flash_rt.frontends.torch.hyvla_rtx import HyVLATorchFrontendRtx  # noqa: E402


def _inputs(fe):
    """Deterministic synthetic inputs (identical across runs and machines)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    img = torch.rand(1, 6, 3, 224, 224, generator=g)
    state = torch.rand(1, fe.max_state_dim, generator=g) * 0.1
    images = torch.stack([img[0], img[0].clone(), img[0].clone()], 0)
    noise = np.random.RandomState(0).randn(
        1, fe.chunk, fe.max_action_dim).astype(np.float32)
    return images.numpy(), state.numpy(), noise


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.fixture(scope="module")
def frontend():
    fe = HyVLATorchFrontendRtx(CKPT, use_fp8=True)  # validated block-128 tier
    fe.set_prompt("pick up the bottle")
    return fe


def test_graph_matches_eager(frontend):
    images, state, noise = _inputs(frontend)
    a_eager = frontend.predict_actions(images, state=state, noise=noise,
                                       use_graph=False)
    a_graph = frontend.predict_actions(images, state=state, noise=noise,
                                       use_graph=True)
    cos = _cos(a_eager, a_graph)
    assert cos >= 0.9999, f"graph vs eager cosine {cos} < 0.9999"


def test_replay_is_stable(frontend):
    images, state, noise = _inputs(frontend)
    a1 = frontend.predict_actions(images, state=state, noise=noise, use_graph=True)
    a2 = frontend.predict_actions(images, state=state, noise=noise, use_graph=True)
    cos = _cos(a1, a2)
    assert cos >= 0.9999, f"replayed graph cosine {cos} < 0.9999"


def test_output_is_finite_and_shaped(frontend):
    images, state, noise = _inputs(frontend)
    a = frontend.predict_actions(images, state=state, noise=noise, use_graph=True)
    assert np.isfinite(a).all()
    assert a.shape == (1, frontend.chunk, frontend.max_action_dim)
