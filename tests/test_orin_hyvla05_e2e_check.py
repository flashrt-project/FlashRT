"""Fixed-noise precision gate for HyVLATorchFrontendOrin (requires SM87 + checkpoint).

Loads the BF16 baseline first, captures the fixed-noise reference action,
frees it, then loads the default INT8 W8A8 tier and asserts action cosine
>= 0.999 against the reference — the documented Orin precision gate. Also
covers the public input boundaries inherited by the Orin override.

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

try:
    from flash_rt.frontends.torch.hyvla_orin import HyVLATorchFrontendOrin
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"hyvla_orin frontend not importable: {exc}", allow_module_level=True)

PROMPT = "pick up the bottle"


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
def ref_and_int8():
    # Sequential loads: Orin unified memory should not hold two 9 GB weight
    # copies at once. Capture the BF16 reference, free it, then load INT8.
    fe_bf16 = HyVLATorchFrontendOrin(CKPT, use_fp8=False, use_int8=False)
    fe_bf16.set_prompt(PROMPT)
    images, state, noise = _inputs(fe_bf16)
    a_bf16 = fe_bf16.predict_actions(images, state=state, noise=noise,
                                     use_graph=False)
    del fe_bf16
    torch.cuda.empty_cache()

    fe_int8 = HyVLATorchFrontendOrin(CKPT)  # default INT8 W8A8 tier
    fe_int8.set_prompt(PROMPT)
    return a_bf16, fe_int8


def test_default_int8_tier_enables_validated_fusion(ref_and_int8):
    _, fe_int8 = ref_and_int8
    pipe = fe_int8.pipe
    assert pipe._fused_attn, \
        "default tier must use the fused RoPE/QKNorm/KV-write kernel"
    assert pipe._vit_fuse_ln or pipe._vit_eff_sdpa, \
        "default tier must enable at least one ViT fusion lever"
    assert pipe._vlm_ffn_int8, "default tier must INT8-quantize the prefill FFN"


def test_int8_vs_bf16_fixed_noise_cosine(ref_and_int8):
    a_bf16, fe_int8 = ref_and_int8
    images, state, noise = _inputs(fe_int8)
    a_int8 = fe_int8.predict_actions(images, state=state, noise=noise,
                                     use_graph=False)
    assert np.isfinite(a_int8).all()
    assert a_int8.shape == (1, fe_int8.chunk, fe_int8.max_action_dim)
    cos = _cos(a_int8, a_bf16)
    assert cos >= 0.999, f"INT8 vs BF16 cosine {cos:.6f} < 0.999 gate"


def test_eager_is_deterministic_with_fixed_noise(ref_and_int8):
    _, fe_int8 = ref_and_int8
    images, state, noise = _inputs(fe_int8)
    a1 = fe_int8.predict_actions(images, state=state, noise=noise, use_graph=False)
    a2 = fe_int8.predict_actions(images, state=state, noise=noise, use_graph=False)
    assert np.array_equal(a1, a2), "eager path must be deterministic for fixed noise"


def test_oversized_state_rejected(ref_and_int8):
    _, fe_int8 = ref_and_int8
    images, _, noise = _inputs(fe_int8)
    big_state = np.zeros((1, fe_int8.max_state_dim + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="max_state_dim"):
        fe_int8.predict_actions(images, state=big_state, noise=noise, use_graph=False)


def test_wrong_noise_size_rejected(ref_and_int8):
    _, fe_int8 = ref_and_int8
    images, state, _ = _inputs(fe_int8)
    bad_noise = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="noise must have"):
        fe_int8.predict_actions(images, state=state, noise=bad_noise, use_graph=False)
