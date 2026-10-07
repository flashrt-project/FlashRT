"""Precision + graph-safety gates for GR00T N1.7 on Jetson Orin SM87.

Every number here is measured against the **HF eager reference on real robot
frames**, not synthetic tensors: ``tests/_helpers/groot_orin/gen_reference.py``
captures per-block activations and ``capture_aux.py`` captures the ``aux``
bundle from the same forward. Random inputs mismeasure the activation outliers
that decide the precision tier (AGENTS.md 3.7), and the state channel in
particular has to be in the checkpoint's own units — see ``STATE_UNIT`` in
``gen_reference.py``.

Gates:

  G1  per-stage backbone cosine (ViT tap / DeepStack mergers / LLM / VLSA)
  G2  per-denoise-step DiT input cosine
  G3  final normalized action cosine
  G4  decoded action cosine — the real end-to-end number
  G5  CUDA-Graph safety: graph == eager bit-for-bit, replay deterministic
  G6  the raw-frames GPU image path: bit-exact shrink, ≤1 LSB full chain, and
      three tiers unmoved against a cv2 arm that is bit-identical to HF

Skipped unless ``FLASHRT_GROOT_N17_CHECKPOINT`` points at a GR00T-N1.7
checkpoint and the paired fixtures exist. Run:

    FLASHRT_GROOT_N17_CHECKPOINT=/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=/mnt/Isaac-GR00T:. \
      python -m pytest tests/test_orin_groot_n17_precision.py -v

``/mnt/Isaac-GR00T`` is required, not optional: G4 calls ``denormalize_action``,
which lazily imports ``gr00t.model.gr00t_n1d7`` to build the HF processor.
Without it those two tests fail with ``ModuleNotFoundError: No module named
'gr00t'``. The offline vars skip the HF Hub HEAD requests, which otherwise
burn ~30 s per retry cycle on a machine with no egress.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CKPT = os.environ.get("FLASHRT_GROOT_N17_CHECKPOINT")
FIXTURE_DIR = Path(os.environ.get(
    "FLASHRT_GROOT_N17_FIXTURE_DIR",
    str(Path(__file__).resolve().parent / "fixtures")))
#: Fixture tag and frame list are overridable so the *same* gate suite can be
#: run against a second, independent real dataset without duplicating a single
#: test. The sim-collected SO101 set (50 episodes, task "Pick up green block and
#: put it on the blue block") is captured with ``--tag _sim`` and has
#: **Se=148** rather than 141, so it also exercises a different sequence shape:
#:   FLASHRT_GROOT_N17_FIXTURE_TAG=_sim \
#:   FLASHRT_GROOT_N17_FRAMES=100,107,15000,31000 pytest ...
FIXTURE_TAG = os.environ.get("FLASHRT_GROOT_N17_FIXTURE_TAG", "")
FRAMES = tuple(int(x) for x in
               os.environ.get("FLASHRT_GROOT_N17_FRAMES", "0,300").split(","))

# The ViT chain accumulates ~1 bf16 ULP per layer against HF (measured: layer 0
# rel_l2 0.0034 == bf16 eps, growing smoothly to 0.061 at layer 17 with no jump
# point), and HF's own vision tower falls back off FA2 because a submodule stays
# fp32. The tap that is actually consumed is gated at 0.999 via the merger
# outputs below; the raw tap tensor is a depth diagnostic at a looser bound.
THR_VIT_TAP = 0.998
THR_CONSUMED = 0.999

# Bound for the ViT-derived intermediates when fuse_image_embeds=True.
#
# This is a real, measured trade, not a relaxed gate. With the fusion OFF the
# LLM's image tokens come from HF, which ran its own vision tower -- and HF's
# tower is partly fp32 (it falls back off FA2: "current dtype in
# Qwen3VLVisionModel is torch.float32"), so it is MORE accurate than a bf16
# port can be. With the fusion ON those tokens come from FlashRT's own 24-layer
# bf16 tower, whose measured floor is cos 0.9943 at layer 23 (accumulating
# ~1 ULP/layer from 0.0034 at layer 0 -- see the doc's per-layer table).
# (Fused off: backbone_features 0.999729.)
#
# The fused ViT *input* is bit-identical to HF's (conv3d with the checkpoint's
# own weight, and the pos-embed interpolation matching HF's bf16 weight
# rounding), so layers 0..17 agree with the fused-off mode digit for digit
# (0.999994 / 0.999892 / 0.999618 / 0.998156). The only new term is that the
# tower now runs to layer 23 -- measured 0.993874, the same value the legacy
# 24-layer run produced -- and that reaches the LLM through the merger.
# Measured backbone_features: 0.996951 (frame 0) / 0.998082 (frame 300).
#
# What makes this acceptable is that it does not reach the output. Measured,
# decoded action (G4) per frame, fused off -> fused on:
#   frame 0   cos 1.000000 -> 1.000000,  max|d| 0.004307 -> 0.008613 rad
#   frame 300 cos 1.000000 -> 1.000000,  max|d| 0.003260 -> 0.003260 rad
# So frame 300 is unaffected and frame 0's worst-case error doubles -- but it
# doubles from 0.25 deg to 0.49 deg on a joint range of ~[-133, 152] deg, and
# the cosine stays 1.000000 in all four cases. The DiT reads
# backbone_features only through cross-attention over 141 tokens, which
# averages the per-token error out (velocity cos moves by ~1e-5).
# The fusion itself is gated separately and strictly in
# test_fusion_reproduces_hf_embeds -- text embeds must be bit-identical and the
# merger must match a torch reference to 0.9999.
THR_FUSED_CONSUMED = 0.995

# T1 (backbone_features) bound for the ``infer(frames=...)`` GPU image arm.
#
# The chain's ≤1 LSB enlarge residual rides through a 24-layer bf16 residual
# tower before anything consumes it, which is the same mechanism
# THR_FUSED_CONSUMED documents for the fusion: an intermediate moves and the
# consumed output does not. Measured over **12 real frames** (2 real-robot
# Se=141, 10 sim Se=148), three arms each — HF fixture, a cv2 chain that G6.3
# proves bit-identical to it, and the GPU chain:
#
#   tier              HF fixture / cv2 arm      GPU image arm
#   T1 backbone       0.996034 .. 0.998082      0.990967 .. 0.996485
#   T3 velocity       0.999417 .. 0.999576      0.999285 .. 0.999538
#   T4 decoded cos    >= 0.999999073            >= 0.999999095
#   T4 max error      0.1936 .. 0.4935 deg      0.1868 .. 0.4935 deg
#
# The cv2 arm equals the HF arm to all nine printed decimals on every frame and
# every tier, so the T1 movement is entirely the GPU chain's and not the test's.
# T3's drop spans -0.000087 .. +0.000153 — it is sometimes *better* — and T4's
# max error is better on 4 frames, worse on 4 and equal on 4: symmetric, i.e. no
# systematic degradation (docs §6.20's lesson, now at n=12 rather than n=4).
#
# So T3 and T4 keep THR_CONSUMED and only T1 is relaxed, to one full observed
# spread (0.0055) below the worst frame measured. G6.5 pins that a genuinely
# wrong image chain falls well through it.
THR_IMAGE_GPU_BACKBONE = 0.985

# backbone.forward + action_head.get_action_with_features as a share of HF
# eager's full get_action. Two independent instrumented runs agreed:
# 359.21/387.09 = 0.9280 and 370.76/399.01 = 0.9292. The instrumented
# absolute is inflated ~3% by the per-stage syncs, so only the ratio is used.
_HF_BOUNDARY_SHARE = 0.929

_DEVFREQ = "/sys/class/devfreq/17000000.gpu"


def _pairs(frame):
    fx = FIXTURE_DIR / (
        f"gr00t_n17_ref_new_embodiment_2v{FIXTURE_TAG}_frame{frame}_seed0.pt")
    return fx, fx.with_name(fx.stem + "_aux.pt")


def _missing():
    if not CKPT:
        return "FLASHRT_GROOT_N17_CHECKPOINT is not set"
    if not Path(CKPT).is_dir():
        return f"checkpoint {CKPT} does not exist"
    for frame in FRAMES:
        fx, aux = _pairs(frame)
        for p in (fx, aux):
            if not p.exists():
                return (f"missing {p.name}; run tests/_helpers/groot_orin/"
                        f"gen_reference.py then capture_aux.py --ver n17")
    return None


_skip = _missing()
if _skip:
    pytest.skip(_skip, allow_module_level=True)

try:
    import flash_rt.flash_rt_kernels  # noqa: F401
except ModuleNotFoundError:
    pytest.skip("flash_rt_kernels was not built", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)


def _cos(a, b):
    # float64: the LLM residual stream carries a 1.5e4-magnitude channel and
    # fp32 accumulation of the two norms over 2.9e5 elements can return cos>1.
    a = torch.as_tensor(a).detach().double().flatten()
    b = torch.as_tensor(b).detach().double().flatten()
    assert a.numel() == b.numel(), f"numel {a.numel()} vs {b.numel()}"
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _maxabs(a, b):
    return float((torch.as_tensor(a).detach().double()
                  - torch.as_tensor(b).detach().double()).abs().max())


#: DiT precision tiers under test. ``False`` is the bf16 tier; ``True`` is the
#: SM87 rowwise-INT8 tier (docs §6.8 for the fake-quant gate that admitted it,
#: §6.9 for the integration and its A/B, §6.3 for why it needs the CUDA graph).
TIERS = [pytest.param(False, id="bf16"), pytest.param(True, id="int8")]


def _frontend_cls():
    from flash_rt.frontends.torch.groot_n17_orin import (
        GrootN17TorchFrontendOrin,
    )
    return GrootN17TorchFrontendOrin


def _shipped_default_int8() -> bool:
    """The default tier, read from the constructor rather than restated here.

    The headline gates and the boundary latency test should exercise whatever
    actually ships, so flipping ``use_int8_dit``'s default in the frontend must
    not require a matching edit here -- a test that hardcodes the default would
    quietly keep validating a tier nobody runs.
    """
    import inspect

    sig = inspect.signature(_frontend_cls().__init__)
    return bool(sig.parameters["use_int8_dit"].default)


#: The tier this deployment ships; ``prompted`` and the boundary latency test
#: follow it. Currently True (docs §6.9.6).
DEFAULT_INT8 = _shipped_default_int8()


def _build(int8: bool):
    return _frontend_cls()(
        CKPT, embodiment_tag="new_embodiment", device="cuda:0",
        use_int8_dit=int8)


@pytest.fixture(scope="module")
def frontend():
    return _build(DEFAULT_INT8)


def _load(frame):
    """Reference activations + aux bundle for one real frame."""
    fx, aux = _pairs(frame)
    # Both files are written locally by tests/_helpers/groot_orin/*.py from real
    # SO101 frames and are never downloaded; weights_only=False is required for
    # the numpy arrays and nested dicts they store. Do not point this loader at
    # a fixture of unknown provenance.
    ref = torch.load(fx, map_location="cpu", weights_only=False)
    bundle = torch.load(aux, map_location="cpu", weights_only=False)
    if bundle["meta"].get("paired_fixture") != fx.name:
        raise AssertionError(
            f"{aux.name} was captured against "
            f"{bundle['meta'].get('paired_fixture')!r}, not {fx.name!r}; "
            f"regenerate both from the same frame")
    return ref, bundle


@pytest.fixture(scope="module")
def prompted_by_tier():
    """{tier: {frame: (frontend, ref, bundle)}}, built once per tier.

    set_prompt is one-shot by contract, so a frontend cannot be reused across
    frames -- hence one per (tier, frame). Four live frontends is ~14 GB of the
    61 GB unified pool, which is why this can be module-scoped at all.
    """
    cache = {}

    def get(int8: bool):
        if int8 not in cache:
            out = {}
            for frame in FRAMES:
                ref, bundle = _load(frame)
                fe = _build(int8)
                fe.set_prompt(aux=bundle, prompt=ref["meta"]["task"])
                out[frame] = (fe, ref, bundle)
            cache[int8] = out
        return cache[int8]

    return get


@pytest.fixture(scope="module")
def prompted(prompted_by_tier):
    """The shipped default tier, under the name the backbone gates already use.

    The DiT tier does not touch the backbone, so the G1 gates read the same
    either way; but ``test_latency_at_the_hf_boundary`` uses this fixture, and
    that number must describe the default configuration rather than a tier
    nobody deploys.
    """
    return prompted_by_tier(DEFAULT_INT8)


# ── G1: backbone stages ────────────────────────────────────────────────────

@pytest.mark.parametrize("frame", FRAMES)
def test_backbone_stages_match_hf_reference(prompted, frame):
    fe, ref, bundle = prompted[frame]
    cap = {"vit_layers": True}
    fe._run_kernel_backbone(bundle, capture=cap)
    acts = ref["activations"]

    last_tap = max(fe._DEEPSTACK_TAPS)
    # Compare the per-layer snapshot, NOT cap["vit_h"]: with the fusion on the
    # tower runs all 24 layers, so vit_h holds the layer-23 output and pairing
    # it against vit_block_17 measures nothing (it scored cos 0.27).
    rows = [
        (f"vit_block_{last_tap}", cap[f"vit_block_{last_tap}"],
         acts[f"vit_block_{last_tap}"], THR_VIT_TAP),
    ]
    # deepstack_out_2 is the merger of the layer-17 tap, so it inherits the
    # tower's accumulation; with the fusion on the tower also runs 6 layers
    # deeper before the LLM, and the image tokens it produces reach the LLM
    # through the merger. See THR_FUSED_* below for why the bound is looser.
    ds_thr = THR_FUSED_CONSUMED if fe._fuse_image_embeds else THR_CONSUMED
    for j in range(3):
        rows.append((f"deepstack_out_{j}", cap[f"deepstack_out_{j}"],
                     acts[f"deepstack_merger_{j}"], ds_thr))
    rows.append(("llm_layer_15", cap["llm_h"], acts["llm_layer_15"],
                 THR_CONSUMED))
    # vlsa_block_3 IS backbone_features: gr00t_n1d7 runs vlln then the 4
    # VL-self-attn blocks on the backbone output and nothing else.
    rows.append(("backbone_features", cap["vlsa_h"], acts["vlsa_block_3"],
                 ds_thr))

    for name, mine, theirs, thr in rows:
        got = _cos(mine, theirs)
        assert got >= thr, (
            f"frame {frame} {name}: cos {got:.6f} < {thr} "
            f"(max|d| {_maxabs(mine, theirs):.4g})")


@pytest.mark.parametrize("frame", FRAMES)
def test_backbone_features_are_what_the_dit_consumes(prompted, frame):
    """The whole-backbone gate, on the tensor ``set_prompt`` actually caches."""
    fe, ref, _bundle = prompted[frame]
    got = _cos(fe._backbone_features.squeeze(0).float().cpu(),
               ref["activations"]["vlsa_block_3"].squeeze(0))
    # mode-aware: see THR_FUSED_CONSUMED for the measured trade and why the
    # fusion's own arithmetic is gated strictly, and separately, in
    # test_fusion_reproduces_hf_embeds
    thr = THR_FUSED_CONSUMED if fe._fuse_image_embeds else THR_CONSUMED
    assert got >= thr, f"frame {frame}: backbone cos {got:.6f} < {thr}"


# ── G1c: the image→embeds fusion, gated separately and strictly ────────────

@pytest.mark.parametrize("frame", FRAMES)
def test_fusion_reproduces_hf_embeds(prompted, frame):
    """The fusion is gated here, not by the looser THR_FUSED_CONSUMED bound.

    ``test_backbone_stages_match_hf_reference`` allows the ViT-derived
    intermediates to sit at 0.995+ when the fusion is on, on the argument that
    the deviation is the bf16 tower's own accumulation floor rather than a
    mistake in the fusion. That argument is only credible if the fusion's own
    arithmetic is checked *independently* and strictly -- which is what this
    does, in three parts:

    1. the text-token embeds must be **bit-identical** to HF's (they come from
       a pure gather, so anything else is an indexing bug, not rounding);
    2. the merger's bf16 kernel chain must match a torch fp64 reference run on
       the *same* input to 0.9999 -- this isolates the merger from the tower,
       and is the check that caught the tanh-vs-exact-erf GELU bug;
    3. the full fused embeds must still track HF's, at a bound that reflects
       the tower floor rather than the merger.
    """
    fe, ref, bundle = prompted[frame]
    if not fe._fuse_image_embeds:
        pytest.skip("fusion disabled on this frontend")

    cap = {"vit_layers": True}
    fe._run_kernel_backbone(bundle, capture=cap)
    torch.cuda.synchronize()

    fused = cap["llm_input_embeds"].double()
    want = bundle["llm_input_embeds"].squeeze(0).double()
    assert fused.shape == want.shape, (fused.shape, want.shape)
    vis = bundle["visual_pos_masks"][0].bool()

    # 1. text tokens: pure embedding gather, must be exact
    d_text = (fused[~vis] - want[~vis]).abs().max()
    assert float(d_text) == 0.0, (
        f"frame {frame}: text-token embeds differ by {float(d_text):.4g} -- "
        "embedding_lookup_bf16 or the token-id handling is wrong; a gather has "
        "no rounding to blame")

    # 2. merger, isolated: torch fp64 reference on FlashRT's OWN tower output
    sv = cap["vit_block_23"] if "vit_block_23" in cap else cap["vit_h"]
    x = torch.as_tensor(sv).double().cpu()
    nw = fe._merger_norm_w.double().cpu()
    nb = fe._merger_norm_b.double().cpu()
    xn = (x - x.mean(-1, keepdim=True)) / torch.sqrt(
        x.var(-1, unbiased=False, keepdim=True) + 1e-6) * nw + nb
    # use_postshuffle_norm=False: LayerNorm on (Sv,1024), THEN the 2x2 shuffle
    y = torch.nn.functional.gelu(
        xn.reshape(-1, 4096) @ fe._merger_fc1_w.double().cpu()
        + fe._merger_fc1_b.double().cpu(), approximate="none")
    ref_mg = y @ fe._merger_fc2_w.double().cpu() + fe._merger_fc2_b.double().cpu()
    got_mg = cap["merger_out"].double().cpu()
    c_mg = _cos(got_mg, ref_mg)
    assert c_mg >= 0.9999, (
        f"frame {frame}: kernel merger vs torch fp64 merger on the SAME input "
        f"cos {c_mg:.6f} -- the merger itself is wrong (check the norm-before-"
        "shuffle order and that the GELU is exact erf, not tanh)")

    # 3. the fused result as a whole
    c_all = _cos(fused, want)
    assert c_all >= THR_FUSED_CONSUMED, (
        f"frame {frame}: fused llm_input_embeds cos {c_all:.6f} vs HF")
    c_img = _cos(fused[vis], want[vis])
    assert c_img >= THR_FUSED_CONSUMED, (
        f"frame {frame}: image-token embeds cos {c_img:.6f} vs HF")


# ── G2/G3/G4: denoise loop and decoded action ─────────────────────────────

@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("int8", TIERS)
def test_denoise_loop_and_decoded_action(prompted_by_tier, int8, frame):
    # Both tiers are held to the same THR_CONSUMED. No looser bound is needed
    # for INT8: measured velocity cos floor is 0.999512 (bf16 0.999822) and the
    # decoded action stays cos >= 0.999999, so the tier clears 0.999 with room.
    fe, ref, bundle = prompted_by_tier(int8)[frame]
    tier = "int8" if int8 else "bf16"
    state = ref["raw"]["state_fed"].reshape(1, 1, -1)
    state_norm = fe.normalize_state({"state.state": state})

    # The checkpoint's statistics are the unit contract: a normalized state far
    # outside [-1, 1] means the frame was fed in the wrong unit (see STATE_UNIT
    # in gen_reference.py), which would make every downstream number meaningless.
    finite = state_norm.flatten()[:state.shape[-1]]
    assert float(finite.abs().max()) < 4.0, (
        f"frame {frame}: normalized state {finite.tolist()} is far outside "
        f"[-1, 1]; state_unit={ref['meta'].get('state_unit')!r} looks wrong")

    cap = {}
    out = fe.infer(state_norm, initial_noise=bundle["initial_noise"],
                   use_dit_graph=False, capture=cap).cpu()

    for i, mine in enumerate(cap["dit_step_input"]):
        want = bundle["dit_step_input"][i]
        got = _cos(mine, want)
        assert got >= THR_CONSUMED, (
            f"frame {frame} [{tier}] dit_step_input[{i}]: cos {got:.6f}")

    # HF's action_decoder runs on all Sa tokens (state token first); ours runs
    # on the action tokens only, hence the trailing slice.
    for i, mine in enumerate(cap["velocity_per_step"]):
        want = bundle["velocity_per_step"][i][:, -mine.shape[1]:]
        got = _cos(mine, want)
        assert got >= THR_CONSUMED, (
            f"frame {frame} [{tier}] velocity[{i}]: cos {got:.6f}")

    want = bundle["final_actions_norm"]
    got = _cos(cap["final_actions_norm"], want)
    assert got >= THR_CONSUMED, (
        f"frame {frame} [{tier}] final_actions_norm: cos {got:.6f}")
    assert torch.equal(cap["final_actions_norm"], out.float().cpu())

    decoded = fe.denormalize_action(out, {"state.state": state})
    for key, value in decoded.items():
        got = _cos(value, ref["actions"][key])
        assert got >= THR_CONSUMED, (
            f"frame {frame} [{tier}] decoded {key}: cos {got:.6f} "
            f"(max|d| {_maxabs(value, ref['actions'][key]):.4g})")


# ── G5: CUDA-Graph safety ─────────────────────────────────────────────────

@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("int8", TIERS)
def test_dit_graph_is_bit_identical_to_eager_and_deterministic(
        prompted_by_tier, int8, frame):
    # The INT8 tier is the one that actually needs this gate: its activation
    # scales are computed on the device per launch, so a capture that baked in a
    # stale scale pointer would still produce plausible-looking numbers.
    fe, ref, bundle = prompted_by_tier(int8)[frame]
    tier = "int8" if int8 else "bf16"
    state_norm = fe.normalize_state(
        {"state.state": ref["raw"]["state_fed"].reshape(1, 1, -1)})
    noise = bundle["initial_noise"]

    eager = fe.infer(state_norm, initial_noise=noise, use_dit_graph=False).cpu()
    first = fe.infer(state_norm, initial_noise=noise, use_dit_graph=True).cpu()
    second = fe.infer(state_norm, initial_noise=noise, use_dit_graph=True).cpu()

    # Identical output alone proves nothing in a fallback-capable system, so
    # pin bit-equality: a captured graph that silently replayed stale pointers
    # or fell back to eager would show up here.
    assert torch.equal(first, eager), (
        f"frame {frame} [{tier}]: graph != eager, "
        f"max|d| {_maxabs(first, eager):.4g}")
    assert torch.equal(second, first), (
        f"frame {frame} [{tier}]: replay not deterministic, "
        f"max|d| {_maxabs(second, first):.4g}")


def test_graph_replay_survives_a_changed_input(prompted):
    """Stale-value test: replay must read the new dit_h, not the captured one."""
    fe, ref, bundle = prompted[FRAMES[0]]
    state_norm = fe.normalize_state(
        {"state.state": ref["raw"]["state_fed"].reshape(1, 1, -1)})
    noise = bundle["initial_noise"]

    base = fe.infer(state_norm, initial_noise=noise, use_dit_graph=True).cpu()
    shifted = fe.infer(state_norm + 0.25, initial_noise=noise,
                       use_dit_graph=True).cpu()
    assert not torch.allclose(base, shifted, atol=1e-4), (
        "a different state produced the same actions: the captured graph is "
        "replaying stale input buffers")


@pytest.mark.parametrize("override", ["action_horizon", "num_timestep_buckets"])
def test_a_changed_denoising_parameter_bypasses_the_graphs(
        prompted, override, monkeypatch, caplog):
    """The graphs bake ``Sa = action_horizon + 1`` and the AdaLN modulators in.

    Replaying them at a different horizon does not raise and does not change the
    output *shape* — it denoises the wrong number of action tokens and returns a
    well-formed action. Measured before the guard existed (docs §6.24), against
    this frontend's own eager arm on a real frame:

        captured at ah=40, requested ah=20   0.0586 rad =  3.357 deg
        captured at ah=20, requested ah=40   0.3750 rad = 21.486 deg
        captured at buckets=1000, asked 200  0.0625 rad =  3.581 deg

    against a shipped worst case of 0.4935 deg, and ``THR_CONSUMED`` is a cosine
    gate that §6.15.3 already proved cannot see a stale context. So the guard is
    the only thing standing between a caller and a silently wrong action.

    Two arms, because "the number came out right" is not evidence in a
    fallback-capable path (red line #5): the replay counter proves the graphs
    were *not* used, and the warning proves the bypass was announced rather than
    quiet. The eager arm is correct here — §6.24's V1c measured it
    order-independent (built at Sa=41 vs rebuilt at Sa=21 agree bit for bit).
    """
    import logging

    from flash_rt.frontends.torch import groot_n17_orin as orin_mod

    fe, ref, bundle = prompted[FRAMES[0]]
    state_norm = fe.normalize_state(
        {"state.state": ref["raw"]["state_fed"].reshape(1, 1, -1)})
    noise = bundle["initial_noise"]

    # Capture at the shipped parameters, then count every replay from here on.
    # A proxy list rather than ``monkeypatch.setattr(g, "replay", ...)``:
    # ``torch.cuda.CUDAGraph`` is a pybind object with no ``__dict__``, so
    # setting an attribute on it does not take.
    fe.infer(state_norm, initial_noise=noise, use_dit_graph=True)
    replays = {"n": 0}

    class _CountingGraph:
        def __init__(self, inner):
            self._inner = inner

        def replay(self):
            replays["n"] += 1
            self._inner.replay()

    monkeypatch.setattr(fe, "_dit_graphs",
                        [_CountingGraph(g) for g in fe._dit_graphs])
    captured = fe._dit_graph_params
    assert captured == (fe._num_inference_timesteps, fe._action_horizon,
                        fe._num_timestep_buckets), (
        f"_dit_graph_params {captured} does not describe the shipped config")

    if override == "action_horizon":
        kwargs, want_horizon = {"action_horizon": 20}, 20
        want_noise = noise[:, :20, :].contiguous()
    else:
        kwargs, want_horizon = {"num_timestep_buckets": 200}, None
        want_noise = noise

    # The one-shot flag is process-global, and an earlier test in this module may
    # already have spent it; reset it so the warning is observable here.
    monkeypatch.setattr(orin_mod, "_warned_dit_graph_params", False)
    with caplog.at_level(logging.WARNING, logger=orin_mod.logger.name):
        got = fe.infer(state_norm, initial_noise=want_noise,
                       use_dit_graph=True, **kwargs).cpu()
        eager = fe.infer(state_norm, initial_noise=want_noise,
                         use_dit_graph=False, **kwargs).cpu()

    assert replays["n"] == 0, (
        f"{replays['n']} graph replays on a call whose {override} differs from "
        "the captured triple: the stale graphs were served")
    assert torch.equal(got, eager), (
        f"the bypassed arm diverged from eager, max|d| "
        f"{_maxabs(got, eager):.4g}")
    assert got.shape[1] == (want_horizon if want_horizon is not None
                            else fe._action_horizon)
    assert fe._dit_graph_params == captured, (
        "the graphs were re-captured for a one-off parameter; a caller "
        "alternating horizons would pay the capture cost every observation")
    assert any("DiT CUDA graphs were captured for" in r.message
               for r in caplog.records), (
        "the bypass was silent: a caller would see a 1.66x latency change and "
        "a wrong-shaped denoise with no diagnostic")

    # Positive control: at the captured triple the graphs DO run and DO match the
    # eager arm, so the zero above is the guard firing and not a counter that
    # never increments. ``caplog.clear()`` first -- records accumulate across
    # ``at_level`` blocks within one test, so without it the bypass warning from
    # above is still in the list and this arm looks like it warned too.
    replays["n"] = 0
    caplog.clear()
    monkeypatch.setattr(orin_mod, "_warned_dit_graph_params", False)
    with caplog.at_level(logging.WARNING, logger=orin_mod.logger.name):
        matched = fe.infer(state_norm, initial_noise=noise,
                           use_dit_graph=True).cpu()
        shipped_eager = fe.infer(state_norm, initial_noise=noise,
                                 use_dit_graph=False).cpu()
    assert replays["n"] == len(fe._dit_graphs), (
        f"{replays['n']} replays at the captured triple, expected "
        f"{len(fe._dit_graphs)}: the counter is not wired to the graphs")
    assert not any("DiT CUDA graphs were captured for" in r.message
                   for r in caplog.records), (
        "the matched call warned, so the guard fires when it should not")
    assert torch.equal(matched, shipped_eager), (
        "graph != eager at the shipped triple, so this frontend's graphs were "
        f"already broken before the override, max|d| "
        f"{_maxabs(matched, shipped_eager):.4g}")


# ── G6: the raw-frames GPU image path ─────────────────────────────────────
#
# ``infer(frames=...)`` replaces the vendor's per-observation image path
# (12.916-14.027 ms of host work on Orin's ARM CPU, measured end to end) with
# pure torch on the GPU (1.448-1.485 ms of device work). It contains two
# resizes with *different* guarantees: the shrink
# step is bit-exact against cv2.INTER_AREA and provably so (an odd numerator in
# the scale means no rounding tie can exist for any input), while the enlarge
# step carries a bounded <=1 LSB residual whose cause is declared unknown rather
# than explained. So the gate checks the two separately, then checks that the
# residual costs nothing downstream on the same tiers G1..G4 use.
#
# The camera bytes come from ``ref["raw"]["images"]`` — the exact frames the HF
# reference was captured from, carrying the fixture's own provenance statement —
# so this gate needs no dataset access and cannot drift from what HF saw
# (AGENTS.md 3.7).

#: The normalized-range value of one uint8 LSB: the chain maps [0,255] onto
#: [-1,1] as ``(x/255 - 0.5)/0.5``, so one level is ``2/255``.
_ONE_LSB = 2.0 / 255.0


def _raw_views(ref):
    """The camera bytes HF was captured from, in the prompt's view order.

    Insertion order, not sorted: ``pixel_values`` rows are ordered by view, and
    the dict ``gen_reference.py`` stored is already in the processor's order.
    G6.3's bit-exactness of the cv2 arm against the fixture is what proves the
    order is right — swapped views would give a well-shaped matrix and a large
    difference.
    """
    import numpy as np

    images = ref["raw"]["images"]
    views = [np.asarray(v) for v in images.values()]
    shapes = {v.shape for v in views}
    assert len(shapes) == 1, f"views have mixed geometries: {shapes}"
    return views


def _gpu_plan(fe, views):
    h, w = views[0].shape[:2]
    return fe._image_plan(h, w), (h, w)


def _cv2_pixel_values(fe, plan, views):
    """The host reference arm: cv2 chain, then the module's own patchify."""
    from flash_rt.frontends.torch._groot_n17_preprocess import (
        host_reference_chain, patchify)

    u8 = host_reference_chain(views, plan)
    return patchify((torch.from_numpy(u8).float() / 255.0 - 0.5) / 0.5)


def _shrink_only_plan(plan):
    """A plan whose crop is the whole frame and whose enlarge is the identity, so
    ``frames_to_resized`` through it yields the *first resize alone*.

    Used both to isolate the shrink step (G6.1, where bit-exactness is claimed)
    and as the deliberately-wrong chain in G6.5's negative control. No arithmetic
    is re-derived in the test: the identity matmul is exact over integral fp32
    values and ``.round()`` on an exact integer is a no-op, so the isolation
    adds no error of its own.
    """
    from flash_rt.frontends.torch._groot_n17_preprocess import ImagePlan

    identity = torch.eye(plan.shortest, dtype=plan.shrink.dtype,
                         device=plan.shrink.device)
    return ImagePlan(pad=plan.pad, shrink=plan.shrink,
                     crop=(0, plan.shortest), enlarge=identity,
                     shortest=plan.shortest, grid=plan.grid, rows=plan.rows,
                     shrink_p=plan.shrink_p, shrink_exact=plan.shrink_exact)


@pytest.mark.parametrize("frame", FRAMES)
def test_the_gpu_shrink_step_is_bit_exact_against_cv2(prompted, frame):
    """G6.1: the provable half of the chain, measured on real frames.

    ``max == 0`` over every pixel is the empirical half of the exactness claim;
    the CPU suite's exhaustive pin (all 16.7M inputs per tap pattern) is the
    other half. Together they are why the ≤1 LSB budget can be attributed
    entirely to the enlarge step.
    """
    import cv2
    import numpy as np

    from flash_rt.frontends.torch._groot_n17_preprocess import frames_to_resized

    fe, ref, _ = prompted[frame]
    views = _raw_views(ref)
    plan, (h, w) = _gpu_plan(fe, views)
    assert plan.shrink_exact, (
        f"frame {frame}: this geometry's shrink scale admits rounding ties, so "
        "bit-exactness is not claimable and frames_to_resized should have "
        "refused it")

    frames = torch.from_numpy(np.stack(views)).to(plan.device)
    got = frames_to_resized(frames, _shrink_only_plan(plan)).cpu().numpy()

    side = plan.shortest
    left, right, top, bottom = plan.pad
    worst = total = 0
    for view, g in zip(views, got):
        padded = (cv2.copyMakeBorder(view, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=0)
                  if (top or bottom or left or right) else view)
        want = cv2.resize(padded, (side, side),
                          interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        # int32 before subtracting: np.abs on a uint8 difference wraps mod 256,
        # so a -1 LSB residual would read as 255.
        d = np.abs(g.astype(np.int32) - want.astype(np.int32))
        worst = max(worst, int(d.max()))
        total += int((d > 0).sum())
    print(f"\n[frame {frame}] shrink {max(h, w)}->{side}, {len(views)} real "
          f"views, scale p={plan.shrink_p}: max={worst} "
          f"differing={total}/{got.size}")
    assert worst == 0, (
        f"frame {frame}: the shrink step differs from cv2.INTER_AREA by {worst} "
        f"LSB on {total} of {got.size} pixels")


@pytest.mark.parametrize("frame", FRAMES)
def test_the_gpu_chain_costs_at_most_one_lsb(prompted, frame):
    """G6.2: the enlarge step's declared residual, bounded on real frames.

    ``host_reference_chain`` delegates to OpenCV itself and is bit-exact against
    the vendor's albumentations chain, so this is the module's GPU arithmetic
    against the definition of correct. The bound — not an explanation — is what
    G6.4 then spends downstream.
    """
    import numpy as np

    from flash_rt.frontends.torch._groot_n17_preprocess import (
        frames_to_resized, host_reference_chain)

    fe, ref, _ = prompted[frame]
    views = _raw_views(ref)
    plan, _hw = _gpu_plan(fe, views)
    frames = torch.from_numpy(np.stack(views)).to(plan.device)
    got = frames_to_resized(frames, plan).cpu().numpy().astype(np.int32)
    want = host_reference_chain(views, plan).astype(np.int32)
    d = np.abs(got - want)
    frac = float((d > 0).mean())
    print(f"\n[frame {frame}] full chain vs cv2: max={int(d.max())} "
          f"mean={d.mean():.5f} differing={frac * 100:.3f}%")
    assert d.max() <= 1, (
        f"frame {frame}: the chain differs from cv2 by {int(d.max())} LSB; the "
        "declared envelope is <=1 and only on the enlarge step")
    # Not vacuously zero, and not unbounded: the residual is a real, small
    # fraction of the pixels. A jump to 0 would mean the enlarge step became
    # exact (update the docs); a jump to >20% would mean a different failure.
    assert 0.0 < frac < 0.20, f"frame {frame}: differing fraction {frac:.4f}"


@pytest.mark.parametrize("frame", FRAMES)
def test_pixel_values_against_the_hf_fixture(prompted, frame):
    """G6.3: the cv2 arm must be bit-identical, the GPU arm must stay in budget.

    The asymmetry is the point. The cv2 arm runs the same OpenCV call the vendor
    does, so anything short of ``torch.equal`` in bf16 is a patchify or
    normalization bug — and that equality is also what proves ``_raw_views``
    returns the views in the prompt's order. The GPU arm's budget is the enlarge
    step's ≤1 LSB, i.e. ``2/255`` of the normalized range.
    """
    from flash_rt.frontends.torch._groot_n17_preprocess import (
        frames_to_pixel_values)

    import numpy as np

    fe, ref, bundle = prompted[frame]
    views = _raw_views(ref)
    plan, _hw = _gpu_plan(fe, views)
    want = bundle["pixel_values"]

    # The fixture stores bf16-quantized values in an fp32 tensor, so an fp32
    # comparison against it carries up to half a bf16 ULP of *fixture*
    # quantization that has nothing to do with this chain. Pinned here because
    # the budget decomposition below depends on it.
    assert torch.equal(want, want.to(torch.bfloat16).to(torch.float32)), (
        f"frame {frame}: the fixture's pixel_values are not bf16-quantized, so "
        "the budget split below is wrong — recheck _ONE_LSB against the fp32 "
        "comparison instead")

    pv_cv2 = _cv2_pixel_values(fe, plan, views)
    assert pv_cv2.shape == want.shape, (pv_cv2.shape, want.shape)
    assert torch.equal(pv_cv2.to(torch.bfloat16), want.to(torch.bfloat16)), (
        f"frame {frame}: the cv2 reference arm is not bit-identical to the HF "
        f"fixture in bf16 (max|d| {_maxabs(pv_cv2, want):.3e}); either the view "
        "order in _raw_views is wrong or the patchify/normalization is")

    frames = torch.from_numpy(np.stack(views)).to(plan.device)
    pv_gpu = frames_to_pixel_values(frames, plan).cpu()

    # Against the cv2 arm — both exact fp32 — the budget is exactly one uint8 LSB,
    # which is the enlarge step's whole declared residual.
    d_cv2 = float((pv_gpu - pv_cv2).abs().max())
    assert d_cv2 <= _ONE_LSB + 1e-6, (
        f"frame {frame}: the GPU chain differs from the bit-exact cv2 chain by "
        f"{d_cv2:.3e}, over the one-LSB budget {_ONE_LSB:.3e}")

    # Against the fixture, in the dtype the model actually eats.
    frac = float((pv_gpu.to(torch.bfloat16)
                  != want.to(torch.bfloat16)).float().mean())
    print(f"\n[frame {frame}] pixel_values: cv2 arm bf16 torch.equal=True | "
          f"GPU vs cv2 (fp32) maxdiff={d_cv2:.3e} <= {_ONE_LSB:.3e} | "
          f"GPU vs HF fixture bf16 differing={frac * 100:.2f}%")
    assert frac <= 0.10, f"frame {frame}: {frac * 100:.2f}% of bf16 elements differ"


def _tier_numbers(fe, ref, bundle, *, frames=None, pixel_values=None):
    """Run one observation and return the three tiers plus the raw output."""
    import numpy as np

    state = ref["raw"]["state_fed"].reshape(1, 1, -1)
    state_norm = fe.normalize_state({"state.state": state})
    aux = None
    kwargs = {}
    if frames is not None:
        kwargs["frames"] = frames
    if pixel_values is not None:
        aux = {k: v for k, v in bundle.items() if k != "pixel_values"}
        aux["pixel_values"] = pixel_values
    cap = {}
    out = fe.infer(state_norm, aux=aux, initial_noise=bundle["initial_noise"],
                   use_dit_graph=False, capture=cap, **kwargs).cpu()
    decoded = fe.denormalize_action(out, {"state.state": state})
    return {
        "out": out,
        "T1_backbone": fe._backbone_features.squeeze(0).float().cpu(),
        "T3_velocity": [v.float().cpu() for v in cap["velocity_per_step"]],
        "T4_decoded": decoded["action"],
        "state": state,
    }


@pytest.mark.parametrize("frame", FRAMES)
def test_the_gpu_image_path_does_not_move_any_tier_out_of_band(prompted, frame):
    """G6.4: three arms, three tiers, against the shipped bounds.

    The arms are the HF fixture's ``pixel_values``, the cv2 reference chain and
    the GPU chain. Each is compared against the HF reference *and* against the
    cv2 arm: the second comparison is the one that isolates this change, since
    the cv2 arm is bit-identical to HF by G6.3 and therefore carries none of the
    GPU chain's residual.

    The claim is "the GPU arm must not move any shipped number outside its
    current band", not "the GPU arm is more accurate" — with two frames per
    dataset a directional claim is not supportable (docs §6.20).
    """
    import numpy as np

    from flash_rt.frontends.torch._groot_n17_preprocess import (
        frames_to_pixel_values)

    fe, ref, bundle = prompted[frame]
    views = _raw_views(ref)
    plan, _hw = _gpu_plan(fe, views)
    frames = torch.from_numpy(np.stack(views)).to(plan.device)

    hf = _tier_numbers(fe, ref, bundle)
    cv2arm = _tier_numbers(fe, ref, bundle,
                           pixel_values=_cv2_pixel_values(fe, plan, views))
    gpu = _tier_numbers(fe, ref, bundle, frames=views)

    base_thr = THR_FUSED_CONSUMED if fe._fuse_image_embeds else THR_CONSUMED
    # Only the GPU arm's T1 gets its own bound; T3/T4 keep THR_CONSUMED for both
    # arms. See THR_IMAGE_GPU_BACKBONE for the 12-frame measurement behind it.
    bounds = {"cv2": base_thr,
              "gpu": min(base_thr, THR_IMAGE_GPU_BACKBONE)
              if fe._fuse_image_embeds else THR_IMAGE_GPU_BACKBONE}
    want_bb = ref["activations"]["vlsa_block_3"].squeeze(0)
    rows = []
    for name, arm in (("cv2", cv2arm), ("gpu", gpu)):
        c_bb = _cos(arm["T1_backbone"], want_bb)
        assert c_bb >= bounds[name], (
            f"frame {frame} [{name}] T1 backbone cos {c_bb:.6f} < "
            f"{bounds[name]}")
        vel = []
        for i, mine in enumerate(arm["T3_velocity"]):
            want = bundle["velocity_per_step"][i][:, -mine.shape[1]:]
            c = _cos(mine, want)
            assert c >= THR_CONSUMED, (
                f"frame {frame} [{name}] T3 velocity[{i}] cos {c:.6f}")
            vel.append(c)
        c_dec = _cos(arm["T4_decoded"], ref["actions"]["action"])
        err = _maxabs(arm["T4_decoded"], ref["actions"]["action"]) * 180.0 / np.pi
        assert c_dec >= THR_CONSUMED, (
            f"frame {frame} [{name}] T4 decoded cos {c_dec:.9f}")
        rows.append((name, c_bb, min(vel), c_dec, err))

    c_hf = _cos(hf["T4_decoded"], ref["actions"]["action"])
    err_hf = _maxabs(hf["T4_decoded"], ref["actions"]["action"]) * 180.0 / np.pi
    print(f"\n[frame {frame}] tier table (HF-fixture arm: T4 cos {c_hf:.9f}, "
          f"max {err_hf:.4f} deg)")
    print("  arm   T1 backbone   T3 velocity   T4 decoded    T4 max(deg)")
    for name, c_bb, v, c_dec, err in rows:
        print(f"  {name:<5} {c_bb:.6f}      {v:.6f}      {c_dec:.9f}   {err:.4f}")
    # The GPU arm against the cv2 arm: this is the residual's whole downstream
    # cost, and it must stay an order of magnitude inside the decoded gate.
    c_cross = _cos(gpu["T4_decoded"], cv2arm["T4_decoded"])
    print(f"  gpu vs cv2 decoded cos {c_cross:.9f}; "
          f"T1 max|d| {_maxabs(gpu['T1_backbone'], cv2arm['T1_backbone']):.3e}")
    assert c_cross >= THR_CONSUMED, (
        f"frame {frame}: the GPU image path moved the decoded action to cos "
        f"{c_cross:.9f} against the bit-exact cv2 arm")


@pytest.mark.parametrize("frame", FRAMES[:1])
def test_a_broken_image_chain_is_detected_by_the_tier_gate(prompted, frame):
    """G6.5 negative control: the tier gate must be able to go red.

    A fallback-capable path cannot be validated by output alone (AGENTS.md 3.6),
    and the decoded action on this checkpoint is dominated by the state input —
    the continuous suite's C6 records a *stranded cross-K/V* still scoring
    0.999991673. So the control is gated on the inner tiers, which discriminate.

    The broken chain is the shrink-only plan: right shape, right range, plausible
    image, and the centre crop plus enlarge simply never happened.
    """
    import numpy as np

    from flash_rt.frontends.torch._groot_n17_preprocess import patchify

    fe, ref, bundle = prompted[frame]
    views = _raw_views(ref)
    plan, _hw = _gpu_plan(fe, views)
    frames = torch.from_numpy(np.stack(views)).to(plan.device)

    good = _tier_numbers(fe, ref, bundle, frames=views)
    from flash_rt.frontends.torch._groot_n17_preprocess import frames_to_resized
    skipped = frames_to_resized(frames, _shrink_only_plan(plan)).float()
    bad = _tier_numbers(fe, ref, bundle,
                        pixel_values=patchify((skipped / 255.0 - 0.5) / 0.5))

    c_good = _cos(good["T1_backbone"],
                  ref["activations"]["vlsa_block_3"].squeeze(0))
    c_bad = _cos(bad["T1_backbone"],
                 ref["activations"]["vlsa_block_3"].squeeze(0))
    v_good = _cos(good["T3_velocity"][0],
                  bundle["velocity_per_step"][0][:, -good["T3_velocity"][0].shape[1]:])
    v_bad = _cos(bad["T3_velocity"][0],
                 bundle["velocity_per_step"][0][:, -bad["T3_velocity"][0].shape[1]:])
    print(f"\n[frame {frame}] negative control (centre crop + enlarge skipped): "
          f"T1 {c_good:.6f} -> {c_bad:.6f}, T3 {v_good:.6f} -> {v_bad:.6f}")
    assert c_bad < c_good, "T1 did not degrade, so the tier gate is blind"
    assert v_bad < v_good, "T3 did not degrade, so the tier gate is blind"
    thr = THR_FUSED_CONSUMED if fe._fuse_image_embeds else THR_CONSUMED
    assert c_bad < thr or v_bad < THR_CONSUMED, (
        f"the broken chain still clears every shipped bound (T1 {c_bad:.6f}, "
        f"T3 {v_bad:.6f}); G6.4 cannot detect a wrong image chain and the "
        "identical-output claim it makes is not evidence of anything")


@pytest.mark.parametrize("frame", FRAMES[:1])
def test_the_frames_arm_really_runs_the_gpu_chain(prompted, frame, monkeypatch):
    """G6.6: identical output is not evidence on its own (red line #5).

    ``frames=`` is numerically indistinguishable from an ``aux=`` call carrying
    the same ``pixel_values``, so a frontend that silently fell back to
    ``host_reference_chain`` would pass every gate above while costing 6 ms of
    host time per observation — the exact thing this path exists to remove. Pin
    the call counts instead.
    """
    import numpy as np

    import flash_rt.frontends.torch._groot_n17_preprocess as PP

    fe, ref, bundle = prompted[frame]
    views = _raw_views(ref)
    plan, _hw = _gpu_plan(fe, views)
    counts = {"gpu": 0, "host": 0, "plan_builds": 0}
    real_gpu = PP.frames_to_pixel_values
    real_host = PP.host_reference_chain

    def counting_gpu(*a, **k):
        counts["gpu"] += 1
        return real_gpu(*a, **k)

    def counting_host(*a, **k):
        counts["host"] += 1
        return real_host(*a, **k)

    monkeypatch.setattr(PP, "frames_to_pixel_values", counting_gpu)
    monkeypatch.setattr(PP, "host_reference_chain", counting_host)

    real_plan = type(fe)._image_plan

    def counting_plan(self, h, w):
        before = len(getattr(self, "_image_plans", {}) or {})
        p = real_plan(self, h, w)
        counts["plan_builds"] += len(self._image_plans) > before
        return p

    monkeypatch.setattr(type(fe), "_image_plan", counting_plan)
    # The earlier G6 gates already warmed the cache on this module-scoped
    # frontend; empty it so the one build this asserts on lands inside the window
    # being measured rather than before it.
    monkeypatch.setattr(fe, "_image_plans", {}, raising=False)

    for _ in range(3):
        _tier_numbers(fe, ref, bundle, frames=views)

    assert counts["gpu"] == 3, (
        f"the GPU chain ran {counts['gpu']} times over 3 frames= observations")
    assert counts["host"] == 0, (
        f"host_reference_chain ran {counts['host']} times on the frames= path; "
        "that is the 6 ms-of-host-time fallback this path exists to remove")
    assert counts["plan_builds"] == 1, (
        f"{counts['plan_builds']} operator-matrix builds over 3 observations; "
        "the plan is a property of the camera geometry and must be cached")
    # And the numbers above really came from the GPU chain, not from a bundle
    # that happened to be equal: the fixture's pixel_values differ from it.
    pv = real_gpu(torch.from_numpy(np.stack(views)).to(plan.device), plan).cpu()
    assert not torch.equal(pv.to(torch.bfloat16),
                           bundle["pixel_values"].to(torch.bfloat16))


# ── latency (skipped unless the GPU clocks are locked) ────────────────────

def _clocks_locked():
    try:
        with open(f"{_DEVFREQ}/cur_freq") as f, open(f"{_DEVFREQ}/max_freq") as g:
            return f.read().strip() == g.read().strip()
    except OSError:
        return False


@pytest.mark.skipif(not _clocks_locked(),
                    reason="GPU clocks are not locked; a latency number from a "
                           "DVFS-throttled Orin is not reportable")
def test_latency_at_the_hf_boundary(prompted):
    """Backbone + DiT, the exact boundary HF's backbone+action_head covers."""
    import numpy as np

    fe, ref, bundle = prompted[FRAMES[0]]
    state_norm = fe.normalize_state(
        {"state.state": ref["raw"]["state_fed"].reshape(1, 1, -1)})
    noise = bundle["initial_noise"]

    for _ in range(3):
        fe._run_kernel_backbone(bundle)
        fe.infer(state_norm, initial_noise=noise, use_dit_graph=True)
    torch.cuda.synchronize()

    bb, dit = [], []
    for _ in range(10):
        torch.cuda.synchronize()
        ev0, ev1, ev2 = (torch.cuda.Event(True) for _ in range(3))
        ev0.record()
        fe._run_kernel_backbone(bundle)
        ev1.record()
        fe.infer(state_norm, initial_noise=noise, use_dit_graph=True)
        ev2.record()
        torch.cuda.synchronize()
        bb.append(ev0.elapsed_time(ev1))
        dit.append(ev1.elapsed_time(ev2))

    # HF eager's own E2E comes from the fixture (recorded by gen_reference.py,
    # uninstrumented). The backbone+action_head boundary is a share of it; the
    # share was measured once by wrapping those two modules with a sync each
    # (138.79 + 231.97 of a 399.01 ms instrumented total; the per-stage syncs
    # cost ~3%, so only the *ratio* is used, never the instrumented absolute).
    hf_e2e = float(ref["meta"]["e2e_median_ms"])
    hf_boundary = hf_e2e * _HF_BOUNDARY_SHARE
    ours = np.median(bb) + np.median(dit)
    tier = "int8" if DEFAULT_INT8 else "bf16"
    print(f"\n[clock-locked, default tier={tier}] backbone median "
          f"{np.median(bb):.2f} ms, DiT(graph) median {np.median(dit):.2f} ms, "
          f"total {ours:.2f} ms -> {hf_boundary / ours:.2f}x vs HF eager at the "
          f"same boundary ({hf_boundary:.2f} ms = {hf_e2e:.2f} ms full "
          f"get_action x {_HF_BOUNDARY_SHARE:.3f})")


@pytest.mark.skipif(not _clocks_locked(),
                    reason="GPU clocks are not locked; a latency number from a "
                           "DVFS-throttled Orin is not reportable")
def test_int8_dit_paired_alternating_speedup(prompted_by_tier):
    """DiT 4-step loop, bf16 vs INT8, measured as a paired alternating A/B.

    Single-arm wall clock drifts several percent on this board, so the ratio --
    the only number claimed here -- comes from interleaving the two tiers and
    taking medians. The isolated per-tier absolutes belong to
    ``test_latency_at_the_hf_boundary``; the two calipers differ by ~1.5%
    because interleaving two resident frontends costs L2 locality, and the
    ratio is the caliper that survives that.

    The INT8 tier is only admissible under CUDA graph: §6.3 measured it at
    33.39 ms/step eager vs 9.37 ms/step captured, i.e. 2.1x *slower* than bf16
    without the graph. So this test times the graph path and asserts the tier
    really is INT8 rather than a silent bf16 fallback.
    """
    import numpy as np

    frame = FRAMES[0]
    runs = {}
    for int8 in (False, True):
        fe, ref, bundle = prompted_by_tier(int8)[frame]
        # A silent fallback to bf16 is numerically plausible and would make the
        # speedup claim meaningless, so pin the tier before timing it.
        assert bool(fe._use_int8_dit) == int8
        # Empty shift/scale lists suffice: _dit_weights only maps .data_ptr()
        # over them, and this avoids depending on the lazy
        # _precompute_diffusion_modulators init order.
        assert ("q_w8" in fe._dit_weights([], [])) == int8, (
            "dit_forward would not select the INT8 tier")
        state_norm = fe.normalize_state(
            {"state.state": ref["raw"]["state_fed"].reshape(1, 1, -1)})
        runs[int8] = (fe, state_norm, bundle["initial_noise"])

    def time_one(int8):
        fe, state_norm, noise = runs[int8]
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        fe.infer(state_norm, initial_noise=noise, use_dit_graph=True)
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1)

    for _ in range(3):
        time_one(False)
        time_one(True)
    bf16, int8_t = [], []
    for _ in range(11):
        bf16.append(time_one(False))
        int8_t.append(time_one(True))

    mb, mi = float(np.median(bf16)), float(np.median(int8_t))
    print(f"\n[clock-locked, paired alternating, median of 11] DiT x4 steps: "
          f"bf16 {mb:.2f} ms (min/med/max {min(bf16):.2f}/{mb:.2f}/"
          f"{max(bf16):.2f})  INT8 {mi:.2f} ms ({min(int8_t):.2f}/{mi:.2f}/"
          f"{max(int8_t):.2f})  -> {mb / mi:.3f}x")

    # Asserting a floor rather than a target: the point of the gate is that
    # INT8 is a real win and not noise. §6.3 predicted 1.56x from a synthetic
    # microbenchmark; the in-pipeline measurement came in lower because that
    # benchmark omitted the attention, norm, bias and residual kernels the real
    # loop also runs. Anything below 1.15x means the tier is not paying for
    # its extra quantize passes.
    assert mb / mi >= 1.15, (
        f"INT8 DiT only reached {mb / mi:.3f}x over bf16 ({mb:.2f} -> "
        f"{mi:.2f} ms); the separate quantize passes are eating the "
        f"weight-bandwidth saving")


@pytest.mark.skipif(not _clocks_locked(),
                    reason="GPU clocks are not locked; a latency number from a "
                           "DVFS-throttled Orin is not reportable")
@pytest.mark.parametrize("frame", FRAMES)
def test_the_frames_arm_beats_the_host_image_chain(prompted, frame):
    """G6.7: paired alternating, because single-arm wall clock drifts several
    percent on this board (AGENTS.md 3.8).

    Two arms producing the same observation:

    * **host** — ``host_reference_chain`` (cv2, the vendor's own arithmetic) plus
      ``patchify``, then ``infer(aux=...)``;
    * **gpu** — ``infer(frames=...)``, which does the same work on the device.

    The host arm is already ~4 ms faster than the vendor's real chain, since it
    skips the 3.87 ms of PIL/torch glue (``Image.fromarray`` → ``np.array`` →
    ``torch.stack`` → ``.numpy()``, verified lossless) and the HF image
    processor's own overhead. So the saving asserted here is a **floor** on what
    ``frames=`` buys against a real deployment; the vendor-arm number is measured
    separately in docs §6.23.

    Three calipers:

    1. preprocessing alone — the change itself, isolated from the ~95 ms model
       that would otherwise bury it in noise. This is the number the claim rests
       on;
    2. per-observation wall clock — what a deployment sees;
    3. per-observation **CPU submission** (no sync after) — reported for
       continuity with §6.11.5's caliper, but *not* an independent measurement
       here: ``run_backbone_graph`` and ``_run_kernel_backbone`` both synchronize
       internally, so submission and wall clock coincide to within ~0.2 ms. This
       work does move host time to the device, and §6.11.5 identified Orin's ARM
       CPU as the submission bottleneck, but showing that needs a caliper taken
       outside ``infer`` and is left to docs §6.23's vendor-arm measurement.
    """
    import time

    import numpy as np

    from flash_rt.frontends.torch._groot_n17_preprocess import (
        frames_to_pixel_values, host_reference_chain, patchify)

    fe, ref, bundle = prompted[frame]
    views = _raw_views(ref)
    plan, _hw = _gpu_plan(fe, views)
    frames_dev = torch.from_numpy(np.stack(views)).to(plan.device)
    state = ref["raw"]["state_fed"].reshape(1, 1, -1)
    state_norm = fe.normalize_state({"state.state": state})
    noise = bundle["initial_noise"]
    rest = {k: v for k, v in bundle.items() if k != "pixel_values"}

    def host_pre():
        u8 = host_reference_chain(views, plan)
        return patchify((torch.from_numpy(u8).float() / 255.0 - 0.5) / 0.5)

    def gpu_pre():
        return frames_to_pixel_values(frames_dev, plan)

    def host_arm():
        aux = dict(rest)
        aux["pixel_values"] = host_pre()
        return fe.infer(state_norm, aux=aux, initial_noise=noise)

    def gpu_arm():
        return fe.infer(state_norm, frames=views, initial_noise=noise)

    def wall(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    def submit(fn):
        # Sync *before*, not after: this measures how long the host spends
        # handing one observation to an empty queue, which is the §6.11.5
        # caliper. Syncing after would fold the device time back in.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1e3

    def paired(a, b, timer, reps=11, warm=3):
        for _ in range(warm):
            timer(a)
            timer(b)
        xs, ys = [], []
        for _ in range(reps):
            xs.append(timer(a))
            ys.append(timer(b))
        return float(np.median(xs)), float(np.median(ys)), xs, ys

    hp, gp, _, _ = paired(host_pre, gpu_pre, wall)
    hw, gw, hws, gws = paired(host_arm, gpu_arm, wall)
    hs, gs, _, _ = paired(host_arm, gpu_arm, submit)

    print(f"\n[frame {frame}, clock-locked, paired alternating, median of 11]")
    print(f"  preprocessing only : host cv2 {hp:7.3f} ms   gpu {gp:6.3f} ms"
          f"   -> {hp / gp:.2f}x, saves {hp - gp:.3f} ms")
    print(f"  per observation    : host     {hw:7.3f} ms   gpu {gw:6.3f} ms"
          f"   -> {hw / gw:.4f}x, saves {hw - gw:.3f} ms")
    print(f"  CPU submission     : host     {hs:7.3f} ms   gpu {gs:6.3f} ms"
          f"   -> saves {hs - gs:.3f} ms")
    print(f"  wall spread        : host min/med/max "
          f"{min(hws):.2f}/{hw:.2f}/{max(hws):.2f}   gpu "
          f"{min(gws):.2f}/{gw:.2f}/{max(gws):.2f}")

    # The isolated preprocessing is where the change lives, and the host chain is
    # ~4x the device one; anything under 2x means the pinned upload or the plan
    # caching regressed.
    assert hp / gp >= 2.0, (
        f"frame {frame}: the GPU image path only reached {hp / gp:.2f}x over the "
        f"host cv2 chain ({hp:.3f} -> {gp:.3f} ms)")
    # The per-observation numbers are dominated by the ~95 ms model, so the gate
    # is a direction, not a magnitude — the magnitude is the preprocessing line.
    assert gw < hw, (
        f"frame {frame}: per-observation wall clock did not improve "
        f"({hw:.3f} -> {gw:.3f} ms)")
    assert gs < hs, (
        f"frame {frame}: CPU submission did not improve ({hs:.3f} -> {gs:.3f} "
        "ms); this path exists to move work off Orin's ARM CPU, so a host arm "
        "that submits faster is a regression in the thing being optimized")
