"""Continuous-inference gates for GR00T N1.7 on Jetson Orin SM87.

``infer(state, aux=...)`` lets ONE frontend serve a stream of observations
instead of one prompt-and-one-frame. That changes what has to be true: the
persistent buffers, the captured DiT graphs and the attention backend's
cross-K/V slots all outlive the observation they were built for, so every one
of them has to be *written in place* rather than replaced. Replacing a slot
the graphs captured the ``data_ptr()`` of does not raise and does not change
the output shape -- the graphs replay successfully and simply keep reading the
previous observation's K/V.

Gates:

  C1  per-observation fidelity against that observation's own HF reference
  C3  repeating an observation is bit-identical
  C4  the continuous path equals the one-shot path bit-for-bit
  C5  the DiT graphs are reused, not re-captured (count + object identity)
  C6  negative control: strand the cross-K/V and prove a gate detects it
  C7  ``infer(aux=...)`` really enforces the observation contract

C2 (a different observation gives a different action) and the contract's own
accept/reject matrix are pinned on CPU in ``test_orin_groot_n17_dispatch.py``,
which needs neither a GPU nor a checkpoint.

⚠️ Read §6.15.3 of ``docs/groot_n17_orin_sm87.md`` before "simplifying" C6.
The obvious version -- assert the decoded action moves -- does NOT work: with
the refresh disabled, a distant frame's decoded action still scored cos
0.999991673 against its own reference, far above the 0.999 gate, because on
this checkpoint the decoded action is dominated by the state input and the
visual context is only worth ~1 deg. C6 therefore gates on the cross-K slots
themselves, which discriminate perfectly (1.000000000 good vs 0.809373
stranded).

Skipped unless ``FLASHRT_GROOT_N17_CHECKPOINT`` is set and the fixtures exist.
These gates need *consecutive* frames plus one from a different episode, which
the sim-collected set provides, so it is the default tag here:

    FLASHRT_GROOT_N17_CHECKPOINT=/mnt/GR00T/so101_sim_rynnbot/checkpoint-89-1.000 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/mnt/Isaac-GR00T:. \
      python -m pytest tests/test_orin_groot_n17_continuous.py -v

One frontend is resident for the whole module (~3.4 GB of Orin's 61 GB unified
pool); running this file in the same process as the two-tier precision suite
has OOMed this board when another large model was still resident.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CKPT = os.environ.get("FLASHRT_GROOT_N17_CHECKPOINT")
FIXTURE_DIR = Path(os.environ.get(
    "FLASHRT_GROOT_N17_FIXTURE_DIR",
    str(Path(__file__).resolve().parent / "fixtures")))
#: The default tag differs from the precision suite's on purpose: continuity is
#: only meaningful on frames that are actually adjacent, and only the
#: sim-collected set was captured that way (100-107 consecutive, 31000 from a
#: different episode as the negative control's distant arm).
FIXTURE_TAG = os.environ.get("FLASHRT_GROOT_N17_FIXTURE_TAG", "_sim")
#: Two adjacent frames for the fidelity and repeat gates, one distant frame for
#: the negative control. ``FAR`` must be far enough that its own reference
#: action is clearly distinguishable -- measured cos 0.900 against frame 100's.
RUNUP, NEIGHBOUR = 100, 101
FAR = int(os.environ.get("FLASHRT_GROOT_N17_FAR_FRAME", "31000"))

#: Same bound the precision suite holds every consumed tensor to.
THR_CONSUMED = 0.999
RAD2DEG = 180.0 / 3.141592653589793


def _fx(frame, aux=False):
    return FIXTURE_DIR / (
        f"gr00t_n17_ref_new_embodiment_2v{FIXTURE_TAG}_frame{frame}_seed0"
        f"{'_aux' if aux else ''}.pt")


def _missing():
    if not CKPT:
        return "set FLASHRT_GROOT_N17_CHECKPOINT to a GR00T-N1.7 checkpoint"
    for frame in (RUNUP, NEIGHBOUR, FAR):
        for p in (_fx(frame), _fx(frame, aux=True)):
            if not p.exists():
                return (f"missing {p.name}; run tests/_helpers/groot_orin/"
                        f"gen_reference.py --tag {FIXTURE_TAG or '_sim'} then "
                        f"capture_aux.py --ver n17 --tag {FIXTURE_TAG or '_sim'}")
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
    # float64, and b moved onto a's device: the reference fixture tensors are
    # on CPU while everything captured during a run is not.
    a = torch.as_tensor(a).detach().double().flatten()
    b = torch.as_tensor(b).detach().to(a.device).double().flatten()
    assert a.numel() == b.numel(), f"numel {a.numel()} vs {b.numel()}"
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _load(frame):
    # weights_only=False: both files are written locally by
    # tests/_helpers/groot_orin/*.py from real frames and are never downloaded;
    # the flag is required for the numpy arrays and nested dicts they store.
    ref = torch.load(_fx(frame), map_location="cpu", weights_only=False)
    bundle = torch.load(_fx(frame, aux=True), map_location="cpu",
                        weights_only=False)
    if bundle["meta"].get("paired_fixture") != _fx(frame).name:
        raise AssertionError(
            f"frame {frame}: aux bundle was captured against "
            f"{bundle['meta'].get('paired_fixture')!r}; regenerate both")
    return ref, bundle


def _frontend_cls():
    from flash_rt.frontends.torch.groot_n17_orin import (
        GrootN17TorchFrontendOrin,
    )
    return GrootN17TorchFrontendOrin


@pytest.fixture(scope="module")
def fe():
    """One frontend for the whole module, with a capture counter installed.

    The counter goes in before any test runs so C5 can assert that serving more
    observations adds no captures, rather than merely that some particular call
    did not capture.
    """
    refs = {f: _load(f) for f in (RUNUP, NEIGHBOUR, FAR)}
    frontend = _frontend_cls()(
        CKPT, embodiment_tag="new_embodiment", device="cuda:0")
    first_ref, first_aux = refs[RUNUP]
    frontend.set_prompt(aux=first_aux, prompt=first_ref["meta"]["task"])
    #: ``set_prompt``'s own backbone output, stashed before any test moves the
    #: frontend onto another frame. C4 compares the reload against this rather
    #: than against whatever the previous test left loaded.
    frontend._prompt_backbone = frontend._backbone_features.clone()

    state = {"captures": 0}
    original = frontend._capture_dit_graphs

    def counted(*a, **k):
        state["captures"] += 1
        return original(*a, **k)

    frontend._capture_dit_graphs = counted
    frontend._capture_count = state

    #: Same instrumentation for the backbone graph (lever #11). C4 passing on
    #: its own would not prove the graph ran: ``run_backbone_graph`` is
    #: bit-identical to the eager arm by design, so a silent fall-through to
    #: ``_run_kernel_backbone`` would pass every numerical gate here while
    #: costing 4.54 ms per observation (red line #5).
    bb_state = {"captures": 0}
    bb_original = frontend._capture_backbone_graph

    def bb_counted(*a, **k):
        bb_state["captures"] += 1
        return bb_original(*a, **k)

    frontend._capture_backbone_graph = bb_counted
    frontend._bb_capture_count = bb_state
    frontend._refs = refs
    return frontend


def _observe(fe, frame, capture=None):
    """Serve one observation exactly as a deployment would, and decode it."""
    ref, bundle = fe._refs[frame]
    state = ref["raw"]["state_fed"].reshape(1, 1, -1)
    out = fe.infer(fe.normalize_state({"state.state": state}), aux=bundle,
                   initial_noise=bundle["initial_noise"],
                   use_dit_graph=True, capture=capture).cpu()
    return out, fe.denormalize_action(out, {"state.state": state})


def _err_deg(decoded, ref):
    got, want = decoded["action"], ref["actions"]["action"]
    return (float((got.double() - want.double()).abs().max()) * RAD2DEG,
            _cos(got, want))


def _slot_K(fe, want):
    """The cross-K the captured graphs will actually read, per family length.

    Each slot is padded to ``max(Skv_text, Skv_image)`` and the 16 entries
    alternate between the two families, so every slot has to be truncated by
    its own entry's row count -- one shared length silently compares the wrong
    rows (measured: a 3.7x numel mismatch before this was fixed).
    """
    return torch.cat([
        t.view(t.shape[0], -1)[: want[j].shape[0]].flatten()
        for j, t in enumerate(fe._dit_attn.dit_cross_K)])


def _flat(K):
    # cat, not stack: the 16 entries alternate between the text and image
    # families and so have different row counts
    return torch.cat([t.flatten() for t in K])


def _want_K(fe):
    return _flat(fe._project_dit_cross_kv()[0])


# ── C1 / C3 / C4 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("frame", [RUNUP, NEIGHBOUR])
def test_a_fresh_observation_reproduces_its_own_reference(fe, frame):
    """C1: continuity must not cost fidelity.

    Each frame is compared against *its own* HF reference, which is the only
    comparison that would catch a frontend quietly serving the observation it
    was prompted with.
    """
    ref, _ = fe._refs[frame]
    _, decoded = _observe(fe, frame)
    err, c = _err_deg(decoded, ref)
    assert c >= THR_CONSUMED, (
        f"frame {frame}: decoded action cos {c:.9f} through the continuous path")
    # The one-shot suite's worst case on this dataset is 0.3872 deg; leaving
    # headroom for board noise but none for a wrong context.
    assert err < 1.0, f"frame {frame}: decoded error {err:.4f} deg"


def test_repeating_an_observation_is_bit_identical(fe):
    """C3: nothing may accumulate across observations.

    The cross-K/V slots, the persistent backbone buffers and the DiT graphs are
    all written in place, so a second identical observation has to land on
    exactly the same bytes -- AGENTS.md §1.1(6)'s repeat-identical gate.
    """
    first, _ = _observe(fe, NEIGHBOUR)
    second, _ = _observe(fe, NEIGHBOUR)
    assert torch.equal(first, second), (
        "re-serving an observation changed the result; something is "
        "accumulating in the persistent buffers")


def test_the_continuous_path_equals_the_one_shot_path(fe):
    """C4: ``aux=`` must add a data path, not a second numerics path.

    Two arms, both independent of test order — the fixture is module-scoped and
    C3 leaves the frontend holding NEIGHBOUR, so the one-shot arm has to be
    re-armed with RUNUP first. Without that this gate compared RUNUP's state
    against NEIGHBOUR's context and reported max|d| 3.1e-02, which is C2's
    ordinary inter-frame delta rather than a defect.

    (a) reloading the prompted observation reproduces ``set_prompt``'s own
        backbone features bit-for-bit, so the reload is not a second numerics
        path through the persistent runtime;
    (b) with that observation loaded, ``infer()`` and ``infer(aux=)`` agree
        bit-for-bit, so the cross-K/V refresh and its in-place slot copy add
        nothing either.
    """
    ref, bundle = fe._refs[RUNUP]
    state = ref["raw"]["state_fed"].reshape(1, 1, -1)
    state_norm = fe.normalize_state({"state.state": state})

    _observe(fe, RUNUP)
    reloaded = fe._backbone_features
    assert torch.equal(reloaded, fe._prompt_backbone), (
        "reloading the prompted observation changed the backbone features: "
        f"max|d| {float((reloaded - fe._prompt_backbone).abs().max()):.3e}")

    one_shot = fe.infer(state_norm, initial_noise=bundle["initial_noise"],
                        use_dit_graph=True).cpu()
    continuous, _ = _observe(fe, RUNUP)
    assert torch.equal(continuous, one_shot), (
        f"infer(aux=) diverged from set_prompt+infer(): "
        f"max|d| {float((continuous - one_shot).abs().max()):.3e}")


# ── C5: the graphs are reused ─────────────────────────────────────────────

def test_the_dit_graphs_are_reused_not_recaptured(fe):
    """C5: re-capturing per observation would be correct and ruinous.

    Identity of the output is not evidence on its own (red line #5) -- a
    frontend that rebuilt the attention backend and re-captured all four graphs
    every frame would pass every numerical gate here while costing ~260 ms per
    observation. So this asserts the capture *count* and the identity of the
    backend, the graph list, and the slots' pointers.
    """
    before = fe._capture_count["captures"]
    attn_ids, graph_ids, slot_ptrs = set(), set(), set()
    for frame in (RUNUP, NEIGHBOUR, FAR):
        _observe(fe, frame)
        attn_ids.add(id(fe._dit_attn))
        graph_ids.add(id(fe._dit_graphs))
        slot_ptrs.add(tuple(t.data_ptr() for t in fe._dit_attn.dit_cross_K))

    added = fe._capture_count["captures"] - before
    assert added == 0, (
        f"{added} extra DiT graph captures over 3 observations; the continuous "
        "path is rebuilding graphs it is supposed to reuse")
    assert len(attn_ids) == 1, "the attention backend was rebuilt mid-stream"
    assert len(graph_ids) == 1, "the graph list was replaced mid-stream"
    assert len(slot_ptrs) == 1, (
        "the cross-K slots moved, so every captured graph is now reading "
        "memory that is no longer the slot it was captured against")
    assert len(fe._dit_graphs) == 4


def test_the_dit_graphs_are_replayed_on_every_observation(fe, monkeypatch):
    """C9: reuse (C5) is necessary but not sufficient — they must actually run.

    C5 pins that the graphs are not *re-captured* per observation. That leaves a
    hole the size of the whole optimization: a guard that started bypassing them
    would keep the capture count at zero and the graph-list identity stable, so
    C5 stays green while every observation silently takes the eager arm at
    **1.66–1.69×** the cost (docs §6.24 measured +64.120 / +67.222 ms per
    observation, and 2305.50 MiB of fp32 AdaLN-weight casts per inference). The
    bypass is announced by a warning, but a warning is not a gate.

    So this counts replays directly, over a stream rather than a single call —
    the single-call case is the precision suite's positive control, and what is
    specific to *this* file is that the graphs keep running on observation 2 and
    3, not just the first one after capture.

    A proxy list rather than patching ``replay`` onto each graph:
    ``torch.cuda.CUDAGraph`` is a pybind object with no ``__dict__``, so
    ``setattr`` on it does not take. The proxy delegates, so the graphs still run
    and every numerical gate here is unaffected; ``monkeypatch`` restores the
    real list afterwards, which is what keeps C5's identity assertions pointed at
    the frontend's own object.
    """
    replays = {"n": 0}

    class _CountingGraph:
        def __init__(self, inner):
            self._inner = inner

        def replay(self):
            replays["n"] += 1
            self._inner.replay()

    per_obs = len(fe._dit_graphs)
    monkeypatch.setattr(fe, "_dit_graphs",
                        [_CountingGraph(g) for g in fe._dit_graphs])
    captures_before = fe._capture_count["captures"]
    frames = (RUNUP, NEIGHBOUR, FAR)
    for frame in frames:
        _observe(fe, frame)

    assert replays["n"] == per_obs * len(frames), (
        f"{replays['n']} replays over {len(frames)} observations, expected "
        f"{per_obs * len(frames)}: the continuous path is not running the graphs "
        "it keeps alive, and C5's capture count cannot see that")
    assert fe._capture_count["captures"] == captures_before, (
        "the graphs were re-captured mid-stream; replaying and re-capturing are "
        "opposite failures and both cost the optimization")
    #: The triple the guard compares against. Asserting it here is what makes the
    #: replay count above a statement about the shipped configuration rather than
    #: about one that happens to match. The literal 4 is deliberate: the whole
    #: graph design is "one graph per denoising step", so a checkpoint that
    #: changed the step count should fail here loudly rather than quietly
    #: re-partition the replay budget.
    assert fe._dit_graph_params == (4, fe._action_horizon,
                                    fe._num_timestep_buckets)
    assert fe._num_inference_timesteps == 4 == per_obs


# ── C8: the backbone graph arm actually ran ───────────────────────────────

def test_the_backbone_graph_is_captured_once_and_reused(fe):
    """C8: prove the graph arm ran, don't infer it from identical output.

    ``run_backbone_graph`` is bit-identical to ``_run_kernel_backbone`` by
    design — that is what C4 asserts — so every numerical gate in this file
    would also pass with the graph arm silently bypassed. The capture count and
    the graph object's identity are what distinguish the two (red line #5).

    One capture for the whole module: this fixture has already served many
    observations across C1/C3/C4/C5 by the time this runs, so a per-observation
    capture would show up as a count far above 1. Re-capturing per observation
    would cost 262.1 ms each against the 4.54 ms the graph saves.
    """
    assert hasattr(fe, "_kbb_graph"), (
        "no backbone graph was captured, so infer(aux=...) ran the eager arm "
        "and the 4.54 ms/observation it exists to save was not measured")
    assert fe._bb_capture_count["captures"] == 1, (
        f"{fe._bb_capture_count['captures']} backbone captures; the graph must "
        "be captured once and replayed for every later observation")
    assert fe._use_backbone_graph is True

    # Replay is stable: the same graph object, and the output buffer it writes
    # is the runtime's persistent vlsa_h, whose address the DiT path reads.
    g_before, ptr_before = fe._kbb_graph, fe._kbb_rt["vlsa_h"].data_ptr()
    _observe(fe, RUNUP)
    assert fe._kbb_graph is g_before, "the backbone graph was replaced"
    assert fe._kbb_rt["vlsa_h"].data_ptr() == ptr_before, (
        "the backbone output buffer moved, so anything captured against it is "
        "now reading stale memory")
    assert fe._bb_capture_count["captures"] == 1


def test_the_eager_backbone_arm_stays_reachable(fe):
    """The opt-out has to be real, not decorative.

    A caller serving only a handful of observations through ``aux`` pays the
    262.1 ms capture to save 4.54 ms each, so ``use_backbone_graph=False`` must
    route ``infer`` back to ``_run_kernel_backbone``. Checked on the source
    rather than by building a second frontend: this fixture's graph is already
    captured and a second 2B-parameter load is not worth it for a branch pin.
    """
    import inspect

    src = inspect.signature(type(fe).__init__).parameters["use_backbone_graph"]
    assert src.default is True
    body = inspect.getsource(type(fe).infer)
    assert "if self._use_backbone_graph:" in body
    assert "_run_kernel_backbone(aux)" in body


# ── C6: the negative control ──────────────────────────────────────────────

def test_a_stranded_cross_kv_is_detected_by_the_slot_metric(fe, monkeypatch):
    """C6: break the refresh and prove a gate actually catches it.

    A fallback-capable or cache-capable path cannot be validated by output
    alone (AGENTS.md §3.6), so the refresh is disabled and the metric must
    visibly degrade. It is gated on the cross-K slots because the decoded action
    does not move enough to gate on -- see the module docstring and §6.15.3.

    Measured on this checkpoint, for the record: stranding frame 100's K/V and
    asking for frame 31000 moved the decoded action from 0.2129 deg to 1.2337
    deg, cos 0.999999582 -> 0.999991673. That still clears the 0.999 gate, so
    an action-level assertion here would pass with the bug present.
    """
    ref_far, _ = fe._refs[FAR]

    # 1. A working refresh, verified against the frame's own projection.
    _observe(fe, FAR)
    K_far = fe._project_dit_cross_kv()[0]
    want_far = _flat(K_far)
    good = _cos(_slot_K(fe, K_far), want_far)
    assert good > 0.999999, (
        f"a working refresh left the slots at cos {good:.9f} against the "
        "requested frame's own K")

    # 2. Establish the state that is about to be stranded. This has to run
    # with the refresh still working, or the slots would hold FAR's K and the
    # control below would measure nothing.
    _observe(fe, RUNUP)
    want_runup = _want_K(fe)
    assert _cos(_slot_K(fe, fe._project_dit_cross_kv()[0]),
                want_runup) > 0.999999

    # 3. Break it, then ask for a different observation.
    monkeypatch.setattr(_frontend_cls(), "_refresh_dit_cross_kv",
                        lambda self: None)
    cap_broken = {}
    broken, broken_dec = _observe(fe, FAR, capture=cap_broken)
    K_now = fe._project_dit_cross_kv()[0]      # _backbone_features is FAR's
    stranded = _cos(_slot_K(fe, K_now), want_far)
    tracks_runup = _cos(_slot_K(fe, K_now), want_runup)

    assert stranded < THR_CONSUMED, (
        f"the control did not degrade: with the refresh disabled the slots "
        f"still score cos {stranded:.9f} against the requested frame's K")
    assert tracks_runup > stranded, (
        f"the stranded slots do not track the observation they were left on "
        f"({tracks_runup:.9f} vs {stranded:.9f}); something other than a "
        "stale cross-K/V is being measured")

    # The per-step velocity sits before the action decoder and does move; it is
    # asserted as a second, independent discriminator at the model's own
    # boundary rather than as the primary one.
    monkeypatch.undo()
    cap_good = {}
    _observe(fe, FAR, capture=cap_good)
    v_ref = fe._refs[FAR][1]["velocity_per_step"][0]
    n = cap_broken["velocity_per_step"][0].shape[1]
    c_broken = _cos(cap_broken["velocity_per_step"][0], v_ref[:, -n:])
    c_good = _cos(cap_good["velocity_per_step"][0], v_ref[:, -n:])
    assert c_broken < c_good, (
        f"step-0 velocity did not degrade ({c_broken:.9f} vs {c_good:.9f})")
    assert c_good >= 0.999, f"the working path's velocity cos is {c_good:.9f}"

    # The decoded action is *not* a discriminator on this checkpoint, which is
    # the whole reason C6 gates on the slots: stranding frame 100's K/V and
    # asking for frame 31000 moves it only 0.2129 -> 1.2337 deg (docstring
    # above; §6.15.3), and both ends clear the 0.999 cos gate. It is still
    # asserted, at a catastrophic bound rather than a tight one, so that a
    # total break of the decode path -- NaN, inf, a wrong decoder, a collapsed
    # shape -- fails here instead of passing every gate above it. The bound is
    # ~8x the measured stranded figure and loose enough to survive a different
    # FLASHRT_GROOT_N17_FAR_FRAME; NaN and inf both compare False, so the one
    # assert covers them.
    broken_deg = _err_deg(broken_dec, ref_far)[0]
    assert broken_deg < 10.0, (
        f"the stranded decode is {broken_deg:.4f} deg from the reference; this "
        "control measured 1.2337 deg, so something other than a stale "
        "cross-K/V is broken")


# ── C7: the contract is enforced on the GPU path too ──────────────────────

def test_infer_aux_rejects_a_changed_prompt_structure(fe):
    """C7: the CPU matrix pins the logic; this pins that ``infer`` calls it.

    ``visual_pos_masks`` is the one that matters most: it decides the
    text/image split of the cross-K/V *and* is baked into the backbone
    runtime's ``vis_idx``, so a changed mask would be served silently.
    """
    _, bundle = fe._refs[NEIGHBOUR]
    mutated = dict(bundle)
    mutated["visual_pos_masks"] = bundle["visual_pos_masks"].clone()
    flat = mutated["visual_pos_masks"].flatten()
    flat[0] = not bool(flat[0])
    ref, _ = fe._refs[NEIGHBOUR]
    state_norm = fe.normalize_state(
        {"state.state": ref["raw"]["state_fed"].reshape(1, 1, -1)})
    with pytest.raises(ValueError, match="visual_pos_masks"):
        fe.infer(state_norm, aux=mutated,
                 initial_noise=bundle["initial_noise"])
