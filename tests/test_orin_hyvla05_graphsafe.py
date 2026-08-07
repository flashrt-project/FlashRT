"""Graph-safety gate for HyVLATorchFrontendOrin (requires SM87 + checkpoint).

Verifies the invariants the CUDA-graph capture relies on:

  1. graph == eager — replaying the captured graph produces the same action
     chunk as the un-captured eager path for identical inputs.
  2. replay-stable — replaying the graph twice on the same static inputs
     yields identical output (no transient-buffer aliasing).
  3. fused == unfused — the fused RoPE/QKNorm/KV-write attention-prep and
     the ViT fused add+LayerNorm produce the same actions (within FP noise)
     as the torch fallback paths.

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

try:
    from flash_rt.frontends.torch.hyvla_orin import HyVLATorchFrontendOrin
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"hyvla_orin frontend not importable: {exc}", allow_module_level=True)


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
    fe = HyVLATorchFrontendOrin(CKPT)  # default INT8 W8A8 tier
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
    assert np.array_equal(a1, a2), \
        "replayed graph output differs run-to-run on identical static inputs"


def test_fused_matches_unfused_attention_prep(frontend):
    images, state, noise = _inputs(frontend)
    if not frontend.pipe._fused_attn:
        pytest.skip("fused RoPE/QKNorm/KV-write kernel not in this build")
    a_fused = frontend.predict_actions(images, state=state, noise=noise,
                                       use_graph=False)
    frontend.pipe._fused_attn = False
    try:
        a_unfused = frontend.predict_actions(images, state=state, noise=noise,
                                             use_graph=False)
    finally:
        frontend.pipe._fused_attn = True
    cos = _cos(a_fused, a_unfused)
    assert cos >= 0.999, f"fused vs unfused attention-prep cosine {cos} < 0.999"


def test_fused_matches_unfused_vit_layer_norm(frontend):
    images, state, noise = _inputs(frontend)
    if not frontend.pipe._vit_fuse_ln:
        pytest.skip("hyvla_vit_add_layer_norm_bf16 not in this build")
    a_fused = frontend.predict_actions(images, state=state, noise=noise,
                                       use_graph=False)
    frontend.pipe._vit_fuse_ln = False
    try:
        a_unfused = frontend.predict_actions(images, state=state, noise=noise,
                                             use_graph=False)
    finally:
        frontend.pipe._vit_fuse_ln = True
    cos = _cos(a_fused, a_unfused)
    assert cos >= 0.999, f"fused vs unfused ViT add+LN cosine {cos} < 0.999"
