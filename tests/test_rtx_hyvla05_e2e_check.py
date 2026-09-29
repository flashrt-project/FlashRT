"""Precision / contract gate for HyVLATorchFrontendRtx (requires SM120 + ckpt).

Loads the BF16 baseline, captures the fixed-noise reference action, frees it
(gc.collect + empty_cache: the reference frontend holds a reference cycle), then
loads the default block-128 tier (``use_fp8=True``) and checks the public input
boundaries and determinism.

Synthetic-input cosine is a *smoke/determinism* signal only: the quantized
SM120 tier is input-distribution sensitive, and uniform-random inputs do not
exercise the action manifold (see VERIFIER.md: synthetic tensors are
inadmissible for distribution-level accept/reject). The distribution-level gate
is validated on real RoboTwin frames and documented in
``docs/hyvla05_rtx_sm120.md`` (0.99978 >= 0.999). When a recorded real-frame
fixture is provided via ``HYVLA_RTX_PARITY_FIXTURE`` this module additionally
asserts the strict cosine >= 0.999 gate on it.

Set FLASHRT_HYVLA_CHECKPOINT to the Hy-Embodied-0.5-VLA directory to run.
"""

import gc
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

PROMPT = "pick up the bottle"
_PARITY_FIXTURE = os.environ.get("HYVLA_RTX_PARITY_FIXTURE", "")


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
def ref_and_tier():
    # Sequential loads: the RTX card cannot hold two ~9 GB weight copies.
    fe_bf16 = HyVLATorchFrontendRtx(CKPT, use_fp8=False, use_int8=False,
                                    use_fused=True)
    fe_bf16.set_prompt(PROMPT)
    images, state, noise = _inputs(fe_bf16)
    a_bf16 = fe_bf16.predict_actions(images, state=state, noise=noise,
                                     use_graph=False)
    del fe_bf16
    gc.collect()
    torch.cuda.empty_cache()

    fe = HyVLATorchFrontendRtx(CKPT, use_fp8=True)  # validated block-128 tier
    fe.set_prompt(PROMPT)
    return a_bf16, fe


def test_default_tier_enables_validated_fusion(ref_and_tier):
    _, fe = ref_and_tier
    pipe = fe.pipe
    assert getattr(pipe, "_fp8b128", False), \
        "default tier must select the SM120 block-128 FP8 GEMMs"
    assert getattr(fe, "_exp_fp8_ready", False), \
        "default tier must quantize the expert denoise tower"
    assert getattr(fe, "_vlm_fp8_ready", False), \
        "default tier must quantize the VLM prefill tower"


def test_bf16_vs_tier_synthetic_smoke(ref_and_tier):
    a_bf16, fe = ref_and_tier
    images, state, noise = _inputs(fe)
    a = fe.predict_actions(images, state=state, noise=noise, use_graph=False)
    assert np.isfinite(a).all()
    assert a.shape == (1, fe.chunk, fe.max_action_dim)
    cos = _cos(a, a_bf16)
    assert cos >= 0.95, f"synthetic smoke cosine {cos} < 0.95"


def test_eager_is_reproducible_with_fixed_noise(ref_and_tier):
    _, fe = ref_and_tier
    images, state, noise = _inputs(fe)
    a1 = fe.predict_actions(images, state=state, noise=noise, use_graph=False)
    a2 = fe.predict_actions(images, state=state, noise=noise, use_graph=False)
    # Split-K atomic accumulation is not bit-exact; assert a tight cosine.
    cos = _cos(a1, a2)
    assert cos >= 0.9999, f"eager repeat cosine {cos} < 0.9999"


def test_real_frame_distribution_gate(ref_and_tier):
    if not _PARITY_FIXTURE or not os.path.isfile(_PARITY_FIXTURE):
        pytest.skip("set HYVLA_RTX_PARITY_FIXTURE to a recorded real-frame "
                    ".npz (images/state/noise/raw) to run the 0.999 gate")
    _, fe = ref_and_tier
    d = np.load(_PARITY_FIXTURE)
    a = fe.predict_actions(d["images"], state=d["state"], noise=d["noise"],
                           use_graph=False)
    cos = _cos(a, d["raw"])
    assert cos >= 0.999, f"real-frame action cosine {cos:.6f} < 0.999 gate"


def test_oversized_state_rejected(ref_and_tier):
    _, fe = ref_and_tier
    images, _, noise = _inputs(fe)
    big_state = np.zeros((1, fe.max_state_dim + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="max_state_dim"):
        fe.predict_actions(images, state=big_state, noise=noise, use_graph=False)


def test_wrong_noise_size_rejected(ref_and_tier):
    _, fe = ref_and_tier
    images, state, _ = _inputs(fe)
    bad_noise = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="noise must have"):
        fe.predict_actions(images, state=state, noise=bad_noise, use_graph=False)
