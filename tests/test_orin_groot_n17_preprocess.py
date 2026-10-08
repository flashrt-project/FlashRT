"""CPU contract pins for the GR00T N1.7 raw-frames image path.

No GPU, no checkpoint and no dataset are needed here. These pin the parts of
``flash_rt.frontends.torch._groot_n17_preprocess`` and the ``infer(frames=...)``
wiring that are *structural* rather than numerical-on-real-data:

  P1  the area operator: shape, row sums exactly 1.0, tap structure, dtype
  P2  the exactness argument, asserted exhaustively rather than asserted in prose
  P3  agreement with ``cv2.INTER_AREA`` on images built to sit on rounding edges
  P4  the plan: shipped geometry, envelope refusals, the inexact-scale record
  P5  patchify: shape, temporal duplication, row order vs an independent
      index-arithmetic derivation
  P6  ``frames_to_resized`` input contract
  P7  the frontend wiring: refusals, identity re-presentation, plan caching

The **real-data** measurement — the shrink step at ``max=0`` and the full chain
at ``max=1`` LSB against the vendor chain on real frames from both datasets, and
what that costs the decoded action — lives in
``tests/test_orin_groot_n17_precision.py``, which needs a GPU, a checkpoint and
captured fixtures. P3 here deliberately uses structured images instead of real
ones: an operator-exactness pin wants inputs that sit *on* the rounding
boundaries, which a camera frame mostly does not, and the point of P2 is that
the argument is input-independent.

Run:

    python -m pytest tests/test_orin_groot_n17_preprocess.py -q
"""

import json

import pytest

torch = pytest.importorskip("torch")

import flash_rt.frontends.torch.groot_n17_orin as orin_mod
from flash_rt.frontends.torch._groot_n17_preprocess import (
    MERGE,
    PATCH,
    ROW_DIM,
    TEMPORAL,
    ImagePlan,
    _MAX_EXACT_NUMERATOR,
    area_operator,
    build_image_plan,
    frames_to_pixel_values,
    frames_to_resized,
    host_reference_chain,
    patchify,
    shrink_scale_is_exact,
)

CLS = orin_mod.GrootN17TorchFrontendOrin

#: The shipped geometry: both datasets letterbox to 640, so the first resize is
#: 640 -> 256 with scale 5/2.
SIDE, SHORTEST, CROP_FRACTION = 640, 256, 0.95
CROPPED = int(SHORTEST * CROP_FRACTION)          # 243


# ── P1: the area operator ─────────────────────────────────────────────────

@pytest.mark.parametrize("src,dst", [(640, 256), (243, 256), (40, 32),
                                     (1280, 256), (300, 256)])
def test_area_operator_row_sums_are_exactly_one(src, dst):
    """Coverage weights partition each output pixel's source span, so a row must
    sum to 1.0 — and to 1.0 *exactly*, not to within a tolerance: a row that
    summed to 1±1e-7 would bias every pixel of a constant image, which is the
    one input where a resize is trivially checkable by eye."""
    A = area_operator(src, dst, device="cpu")
    assert tuple(A.shape) == (dst, src)
    assert A.dtype == torch.float32
    sums = A.sum(1).to(torch.float64)
    assert bool((sums == 1.0).all()), (
        f"row sums span {sums.min():.17g}..{sums.max():.17g}")
    assert float(A.min()) >= 0.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_area_operator_refuses_narrow_weights(dtype):
    """fp16/bf16 weights measured max=1 LSB against cv2 where fp32 measures
    max=0, so the refusal is the only thing keeping the exactness claim true."""
    with pytest.raises(ValueError, match="fp32 or wider"):
        area_operator(640, 256, device="cpu", dtype=dtype)


def test_the_shipped_shrink_scale_is_three_taps_of_a_fifth_and_two_fifths():
    """640 -> 256 is scale 5/2: exactly 3 taps per output, weights {1/5, 2/5}.

    This is the structure the parity argument in P2 needs, so it is pinned as a
    fact about the operator rather than restated in a comment.
    """
    A = area_operator(SIDE, SHORTEST, device="cpu")
    taps = (A > 0).sum(1)
    assert set(taps.tolist()) == {3}, sorted(set(taps.tolist()))
    weights = sorted({float(w) for w in torch.unique(A[A > 0])})
    assert len(weights) == 2
    # float32 nearest to 1/5 and 2/5; compare against the cast, not the decimal.
    assert weights == [float(torch.tensor(0.2, dtype=torch.float32)),
                       float(torch.tensor(0.4, dtype=torch.float32))]
    # Every row's integer numerators over p=5 must partition 5.
    p, exact = shrink_scale_is_exact(SIDE, SHORTEST)
    assert (p, exact) == (5, True)
    m = (A.to(torch.float64) * p).round().to(torch.int64)
    assert set(m.sum(1).tolist()) == {p}
    assert m.min() >= 0


def test_the_enlarge_operator_is_the_two_tap_case():
    """243 -> 256 enlarges, so OpenCV takes its two-tap emulation branch. The
    *operator* here is still continuous coverage; P3 pins what that costs."""
    A = area_operator(CROPPED, SHORTEST, device="cpu")
    assert set((A > 0).sum(1).tolist()) <= {1, 2}


# ── P2: the exactness argument, exhaustively ──────────────────────────────

def _tie_free(p: int) -> bool:
    """No ``N/p`` with integer ``N`` in range is a half-integer."""
    return not any((2 * N) % (2 * p) == p for N in range(0, p * 255 + 1))


@pytest.mark.parametrize("src,dst", [(640, 256), (1280, 256), (480, 256),
                                     (300, 256), (40, 32)])
def test_no_exact_output_value_can_be_a_rounding_tie(src, dst):
    """The half of the argument that makes fp32 safe.

    With scale ``p/q`` in lowest terms the weights are ``m_i/p`` for integers
    ``m_i`` summing to ``p``, so the exact output is ``N/p`` for an integer
    ``N <= 255p``. A tie needs ``N/p = k + 1/2``, i.e. ``2N = p(2k+1)``; for odd
    ``p`` that forces ``p | N`` and makes the exact value an integer. So there is
    no input at which fp32 error has to break a tie — it only has to stay under
    ``1/(2p)``, which P2's second half measures.
    """
    p, exact = shrink_scale_is_exact(src, dst)
    assert p % 2 == 1 and exact
    assert _tie_free(p)


def test_an_even_numerator_scale_really_does_admit_ties():
    """The converse, so the odd-``p`` condition is not vacuous.

    512 -> 256 is scale 2: ``out = (a+b)/2``, which lands exactly on ``.5`` for
    every odd ``a+b`` — half of all inputs. OpenCV's tie-break there is not what
    continuous coverage does, which is why ``frames_to_resized`` refuses it.
    """
    p, exact = shrink_scale_is_exact(512, 256)
    assert (p, exact) == (2, False)
    assert not _tie_free(p)


@pytest.mark.parametrize("p", [3, 5, 7, 15, 45, 75])
def test_the_tie_margin_exceeds_fp32_error_for_every_admitted_scale(p):
    """``1/(2p)`` against the ~1e-4 an fp32 sum of a few taps over values ≤255
    carries. This is what ``_MAX_EXACT_NUMERATOR`` is for."""
    assert 0.5 / p > 1e-3, f"p={p} leaves a tie margin below the fp32 error"
    assert p <= _MAX_EXACT_NUMERATOR


def _exhaustive_rounding_agreement(weights, exact_weights=None,
                                   chunk: int = 16) -> int:
    """Count inputs where ``round(fp32 Σ w_i v_i)`` != the exact rounding, over
    **every** ``(v0, v1, ...)`` in ``[0, 255]^n``.

    ``exact_weights`` defaults to ``weights``; passing a different reference is
    how the negative control asks "does a perturbed operator still produce the
    right answer".

    The exact rounding is ``floor(N/p + 1/2)`` — round-half-*up* — while torch's
    ``.round()`` is round-half-*even*. They can only differ at a tie, and P2's
    first half establishes there are none, so comparing against half-up is the
    same as comparing against the true answer and does not need a tie rule.
    """
    from fractions import Fraction

    if exact_weights is None:
        exact_weights = weights
    exact = [Fraction(w).limit_denominator(4096) for w in exact_weights]
    p = exact[0].denominator
    assert all(f.denominator == p for f in exact), (exact_weights, exact)
    m = [f.numerator for f in exact]
    assert len(m) == len(weights)
    n = len(weights)
    w = torch.tensor(weights, dtype=torch.float32)
    vals = torch.arange(256, dtype=torch.float32)
    ivals = torch.arange(256, dtype=torch.int64)
    bad = 0
    for lo in range(0, 256, chunk):
        hi = min(chunk, 256 - lo)
        f32 = torch.zeros(hi, *[256] * (n - 1))
        exact_n = torch.zeros(hi, *[256] * (n - 1), dtype=torch.int64)
        for axis in range(n):
            sh = [1] * n
            sh[axis] = hi if axis == 0 else 256
            src = (vals[lo:lo + hi] if axis == 0 else vals).reshape(sh)
            isrc = (ivals[lo:lo + hi] if axis == 0 else ivals).reshape(sh)
            f32 = f32 + src * w[axis]
            exact_n = exact_n + isrc * m[axis]
        want = (2 * exact_n + p) // (2 * p)
        bad += int((f32.round().to(torch.int64) != want).sum())
    return bad


def test_fp32_accumulation_rounds_exactly_for_every_uint8_input():
    """The other half: with no ties to break, fp32 must land on the exact answer
    for *all* inputs, not just for the frames that happened to be measured.

    Exhaustive over every distinct tap pattern of the shipped 640->256 operator
    (256^3 inputs each). This is the pin that turns "max=0 on 4 real frames"
    into "max=0 for this operator".
    """
    A = area_operator(SIDE, SHORTEST, device="cpu")
    patterns = {}
    for r in range(A.shape[0]):
        nz = A[r][A[r] > 0]
        patterns[tuple(float(x) for x in nz)] = int(nz.numel())
    assert patterns, "no taps found"
    for weights, n_taps in patterns.items():
        assert len(weights) == n_taps
        assert _exhaustive_rounding_agreement(list(weights)) == 0, (
            f"fp32 rounding diverged from exact for the tap pattern {weights}")


def test_a_perturbed_operator_loses_the_exhaustive_pin():
    """Negative control: the exhaustive pin must be able to go red.

    Without this, "0 disagreements" is equally consistent with a comparison that
    never compares anything.
    """
    A = area_operator(SIDE, SHORTEST, device="cpu")
    weights = [float(x) for x in A[0][A[0] > 0]]
    assert _exhaustive_rounding_agreement(weights) == 0
    perturbed = list(weights)
    perturbed[0] += 1e-3
    bad = _exhaustive_rounding_agreement(perturbed, exact_weights=weights)
    assert bad > 0, (
        "perturbing a weight by 1e-3 still rounds to the exact answer over all "
        "16.7M inputs, so the pin above is not sensitive to the weights and "
        "proves nothing")


def test_shrink_scale_is_exact_classifies_by_the_odd_numerator():
    cases = {640: (5, True), 320: (5, True), 480: (15, True), 300: (75, True),
             768: (3, True), 1280: (5, True), 512: (2, False),
             1024: (4, False), 2560: (10, False)}
    for src, want in cases.items():
        assert shrink_scale_is_exact(src, 256) == want, src


# ── P3: agreement with cv2 on boundary-seeking images ─────────────────────

def _boundary_images(h: int, w: int):
    """Images chosen to land on rounding edges, not to look like a camera frame.

    A ramp steps through every residue class of the tap sum, so if a tie or an
    fp32 flip is reachable at this scale, one of these rows hits it. A
    checkerboard at the tap period alternates the extreme tap weights.

    The ramps are built in ``int32`` and cast at the end. ``np.arange(w,
    dtype=np.uint8) * 255`` overflows and silently produces wrapped noise
    instead of a ramp — which this file's own negative control caught by
    reporting that a perturbation moved the output by 0.12 rather than 2.5.
    """
    import numpy as np

    xs = np.arange(w, dtype=np.int32)
    ys = np.arange(h, dtype=np.int32)
    h_ramp = (xs * 255 // max(1, w - 1)).astype(np.uint8)
    v_ramp = (ys * 255 // max(1, h - 1)).astype(np.uint8)
    out = {
        "h_ramp": np.tile(h_ramp, (h, 1)),
        "v_ramp": np.tile(v_ramp[:, None], (1, w)),
        "checker": (((xs[None, :] // 2 + ys[:, None] // 2) % 2) * 255).astype(
            np.uint8),
        "const_127": np.full((h, w), 127, dtype=np.uint8),
        "const_128": np.full((h, w), 128, dtype=np.uint8),
        # odd/even adjacency: (a+b)/2 ties for every odd sum at scale 2
        "alt_0_1": np.tile(np.array([0, 1], dtype=np.uint8),
                           (h, w // 2 + 1))[:, :w],
        # every residue class of the tap sum mod p, as flat fields
        **{f"const_{v}": np.full((h, w), v, dtype=np.uint8)
           for v in (1, 2, 3, 4, 127, 128, 129, 253, 254, 255)},
        # A seeded pseudo-random field. Structured images alone are not enough:
        # they probe the residues a ramp happens to hit, and a disagreement that
        # lives in some other combination of tap values would pass unnoticed.
        # Seeded so the pin is reproducible. (This is an operator-exactness pin
        # over the input space, not a model-precision measurement — those use the
        # real captured frames, in G6 of the precision suite.)
        "random": np.random.RandomState(20260).randint(
            0, 256, (h, w, 3)).astype(np.uint8),
    }
    return {k: (v if v.ndim == 3 else
                np.stack([v, np.roll(v, 1, 1), np.roll(v, 1, 0)], axis=-1))
            for k, v in out.items()}


@pytest.fixture(scope="module")
def small_plan():
    """A 40-wide plan: scale 5/4, so 2 taps with weights {1/5, 4/5}.

    A *different* tap structure from the shipped 640->256, under the same odd-
    ``p`` guarantee — which is what makes this a pin on the argument rather than
    on one geometry.
    """
    return build_image_plan(24, 40, device="cpu", shortest=32,
                            crop_fraction=CROP_FRACTION)


def _letterbox_cv2(im, plan):
    """The vendor's LetterBoxPad, in cv2, using the plan's own pad tuple."""
    import cv2

    left, right, top, bottom = plan.pad
    if not (left or right or top or bottom):
        return im
    return cv2.copyMakeBorder(im, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=0)


def test_the_shrink_step_matches_cv2_bit_for_bit(small_plan):
    import cv2
    import numpy as np

    p, exact = shrink_scale_is_exact(40, 32)
    assert (p, exact) == (5, True)
    A = small_plan.shrink
    side = small_plan.shortest
    for name, im in _boundary_images(24, 40).items():
        x = torch.nn.functional.pad(
            torch.from_numpy(im)[None].permute(0, 3, 1, 2).float(),
            small_plan.pad)
        got = (A @ x @ A.t()).round().clamp(0, 255).to(torch.uint8).numpy()[0]
        got = got.transpose(1, 2, 0)          # CHW -> HWC to match cv2
        want = cv2.resize(_letterbox_cv2(im, small_plan), (side, side),
                          interpolation=cv2.INTER_AREA)
        # int32 before subtracting: np.abs on a uint8 difference wraps mod 256,
        # so a -1 LSB residual would read as 255.
        d = np.abs(got.astype(np.int32) - want.astype(np.int32))
        assert d.max() == 0, f"{name}: shrink differs from cv2 by {d.max()}"


def test_a_perturbed_operator_is_detected_by_the_cv2_gate(small_plan):
    """Negative control for P3: the cv2 comparison must be able to go red.

    The perturbation is placed at a tap the test image actually excites and its
    effect on the fp32 output is asserted first — the obvious version (row 0,
    first tap) lands on a source column the ramp holds at 0, perturbs nothing,
    and the control passes for the wrong reason.
    """
    import cv2
    import numpy as np

    im = _boundary_images(24, 40)["h_ramp"]
    x = torch.nn.functional.pad(
        torch.from_numpy(im)[None].permute(0, 3, 1, 2).float(), small_plan.pad)
    A = small_plan.shrink.clone()
    row = A.shape[0] // 2
    col = int((A[row] > 0).nonzero()[-1, 0])
    A[row, col] += 1e-2
    base = small_plan.shrink @ x @ small_plan.shrink.t()
    moved = float((base - A @ x @ A.t()).abs().max())
    assert moved > 0.5, (
        f"the perturbation moved the fp32 output by only {moved:.3e}, so this "
        "control proves nothing about the gate")

    got = (A @ x @ A.t()).round().clamp(0, 255).to(torch.uint8).numpy()[0]
    got = got.transpose(1, 2, 0)              # CHW -> HWC to match cv2
    want = cv2.resize(_letterbox_cv2(im, small_plan), (32, 32),
                      interpolation=cv2.INTER_AREA)
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    assert d.max() > 0, "the cv2 gate did not notice a perturbed operator"


def test_the_full_chain_costs_at_most_one_lsb(small_plan):
    """The enlarge step's declared residual, bounded rather than explained.

    ``host_reference_chain`` delegates to OpenCV itself, so this is the module's
    GPU-shaped arithmetic against the definition of correct. The bound is what
    the precision suite then spends downstream.
    """
    import numpy as np

    views = list(_boundary_images(24, 40).values())
    want = host_reference_chain(views, small_plan)
    got = frames_to_resized(torch.from_numpy(np.stack(views)), small_plan)
    d = np.abs(got.numpy().astype(np.int32) - want.astype(np.int32))
    assert d.max() <= 1, f"full chain differs by {d.max()} LSB, expected <=1"
    # Not vacuous: the chain really resized (it is not the padded input passed
    # through), and skipping the centre crop — one scalar of the plan — is
    # detected by exactly this comparison.
    assert got.shape == want.shape == (len(views), 3, 32, 32)
    uncropped = ImagePlan(
        pad=small_plan.pad, shrink=small_plan.shrink, crop=(0, 32),
        enlarge=area_operator(32, 32, device="cpu"),
        shortest=small_plan.shortest, grid=small_plan.grid,
        rows=small_plan.rows, shrink_p=small_plan.shrink_p,
        shrink_exact=small_plan.shrink_exact)
    other = frames_to_resized(torch.from_numpy(np.stack(views)), uncropped)
    assert not np.array_equal(other.numpy(), got.numpy()), (
        "the crop had no effect on the output, so this comparison is not "
        "sensitive to the plan")


# ── P4: the plan ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("h,w,pad,crop", [
    (480, 640, (0, 0, 80, 80), (6, 249)),      # real-robot SO101 geometry
    (360, 640, (0, 0, 140, 140), (6, 249)),    # sim-collected geometry
])
def test_build_image_plan_reproduces_the_shipped_geometries(h, w, pad, crop):
    plan = build_image_plan(h, w, device="cpu")
    assert plan.pad == pad
    assert plan.crop == crop
    assert plan.shortest == SHORTEST
    assert plan.grid == SHORTEST // PATCH == 16
    assert plan.rows == 256
    assert tuple(plan.shrink.shape) == (SHORTEST, max(h, w))
    assert tuple(plan.enlarge.shape) == (SHORTEST, CROPPED)
    assert plan.shrink_exact is True and plan.shrink_p == 5
    assert plan.device == torch.device("cpu")


@pytest.mark.parametrize("args,kwargs,exc", [
    ((480, 640), {"shortest": 250}, ValueError),          # does not tile
    ((480, 640), {"shortest": 200}, ValueError),          # does not tile
    ((256, 256), {}, NotImplementedError),                # would not shrink
    ((200, 200), {}, NotImplementedError),                # would not shrink
    ((480, 640), {"crop_fraction": 1.0}, NotImplementedError),   # no enlarge
    ((480, 640), {"crop_fraction": 1.5}, NotImplementedError),   # crop > frame
])
def test_build_image_plan_refuses_outside_the_verified_envelope(args, kwargs, exc):
    with pytest.raises(exc):
        build_image_plan(*args, device="cpu", **kwargs)


def test_a_degenerate_crop_fraction_clamps_like_the_vendor():
    """``crop_fraction=0`` is not refused, and that is deliberate: the vendor's
    own transform does ``max(1, int(size * fraction))``, so a zero fraction
    yields a 1-pixel crop there too. Reproducing the clamp rather than inventing
    a refusal keeps this chain a reproduction. Pinned so the behaviour is a
    decision on the record and not an accident.
    """
    plan = build_image_plan(480, 640, device="cpu", crop_fraction=0.0)
    lo, hi = plan.crop
    assert hi - lo == 1


def test_an_inexact_scale_is_recorded_on_the_plan_and_refused_at_use():
    """A 512-wide camera letterboxes to 512, so the shrink is scale 2 and ties
    are reachable. The plan still builds — ``host_reference_chain`` reads only
    its scalar geometry and stays correct at any scale — but the GPU chain must
    refuse rather than serve a ~1 LSB drift on an unbounded pixel set."""
    plan = build_image_plan(288, 512, device="cpu")
    assert plan.shrink_exact is False and plan.shrink_p == 2
    frames = torch.zeros(1, 288, 512, 3, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="ties are then reachable"):
        frames_to_resized(frames, plan)
    # The documented workaround really works at that geometry.
    out = host_reference_chain([frames[0].numpy()], plan)
    assert out.shape == (1, 3, SHORTEST, SHORTEST)
    assert out.dtype.name == "uint8"


def test_tf32_matmul_is_refused(monkeypatch):
    """TF32's 10-bit mantissa puts ~0.25 of error on a value of 255 against a
    tie margin of 1/(2p) = 0.1 at the shipped scale, so the shrink step would
    stop being bit-exact and nothing downstream would report it."""
    plan = build_image_plan(24, 40, device="cpu", shortest=32,
                            crop_fraction=CROP_FRACTION)
    frames = torch.zeros(1, 24, 40, 3, dtype=torch.uint8)
    assert frames_to_resized(frames, plan) is not None      # fp32 is fine
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="TF32"):
        frames_to_resized(frames, plan)


# ── P5: patchify ──────────────────────────────────────────────────────────

def _patchify_reference(x):
    """The row order spelled out as index arithmetic, with no reshape or permute.

    One row per patch, ordered by view, then merged block, then position within
    the block; within a row by channel, then temporal copy, then pixel. Written
    independently of the implementation so that agreeing with it means
    something.
    """
    V, C, H, W = x.shape
    gh, gw = H // PATCH, W // PATCH
    rows = []
    for v in range(V):
        for hm in range(gh // MERGE):
            for wm in range(gw // MERGE):
                for dh in range(MERGE):
                    for dw in range(MERGE):
                        ph, pw = hm * MERGE + dh, wm * MERGE + dw
                        row = []
                        for c in range(C):
                            for _t in range(TEMPORAL):     # identical copies
                                for a in range(PATCH):
                                    for b in range(PATCH):
                                        row.append(
                                            x[v, c, ph * PATCH + a,
                                              pw * PATCH + b])
                        rows.append(torch.stack(row))
    return torch.stack(rows)


def test_patchify_matches_an_independent_row_order_derivation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 2 * PATCH, 2 * PATCH, generator=g)
    got, want = patchify(x), _patchify_reference(x)
    assert got.shape == want.shape == (2 * 4, ROW_DIM)
    assert torch.equal(got, want)


def test_patchify_shape_and_temporal_duplication():
    V, H = 2, SHORTEST
    x = torch.arange(V * 3 * H * H, dtype=torch.float32).reshape(V, 3, H, H)
    out = patchify(x)
    assert out.shape == (V * (H // PATCH) ** 2, ROW_DIM) == (512, 1536)
    assert out.dtype == x.dtype and out.is_contiguous()
    # temporal_patch_size=2 duplicates each frame, so within a row the second
    # half of each channel block is the first half again.
    r = out[0].reshape(3, TEMPORAL, PATCH * PATCH)
    assert torch.equal(r[:, 0], r[:, 1])


@pytest.mark.parametrize("shape", [(3, 32, 32), (2, 4, 32, 32), (2, 3, 32, 20)])
def test_patchify_refusals(shape):
    with pytest.raises(ValueError):
        patchify(torch.zeros(*shape))


# ── P6: frames_to_resized input contract ──────────────────────────────────

def test_frames_to_resized_input_refusals(small_plan):
    ok = torch.zeros(1, 24, 40, 3, dtype=torch.uint8)
    assert frames_to_resized(ok, small_plan).shape == (1, 3, 32, 32)
    with pytest.raises(ValueError, match="uint8"):
        frames_to_resized(ok.float(), small_plan)
    with pytest.raises(ValueError, match=r"\(views, H, W, 3\)"):
        frames_to_resized(torch.zeros(24, 40, 3, dtype=torch.uint8), small_plan)
    with pytest.raises(ValueError, match=r"\(views, H, W, 3\)"):
        frames_to_resized(torch.zeros(1, 24, 40, 4, dtype=torch.uint8),
                          small_plan)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_frames_to_resized_refuses_a_device_mismatch():
    """A host tensor handed to a device plan faults on the device instead of
    producing a wrong answer, so the check is worth its cost."""
    host = build_image_plan(24, 40, device="cpu", shortest=32,
                            crop_fraction=CROP_FRACTION)
    dev = build_image_plan(24, 40, device="cuda", shortest=32,
                           crop_fraction=CROP_FRACTION)
    frames = torch.zeros(1, 24, 40, 3, dtype=torch.uint8)
    assert frames_to_resized(frames, host).device.type == "cpu"
    with pytest.raises(ValueError, match="upload the frames first"):
        frames_to_resized(frames, dev)


def test_frames_to_pixel_values_shape_and_range(small_plan):
    frames = torch.randint(0, 256, (2, 24, 40, 3), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(1))
    pv = frames_to_pixel_values(frames, small_plan)
    assert pv.shape == (2 * small_plan.rows, ROW_DIM)
    assert pv.dtype == torch.float32
    # (x/255 - 0.5)/0.5 maps [0,255] onto [-1, 1]
    assert float(pv.min()) >= -1.0 - 1e-6
    assert float(pv.max()) <= 1.0 + 1e-6


# ── P7: the frontend wiring ───────────────────────────────────────────────

#: A stub checkpoint whose processor config asks for a 32px shortest edge, so
#: the wiring tests cost milliseconds instead of a 640->256 resize.
_STUB_CFG = {
    "processor_class": "Gr00tN1d7Processor",
    "processor_kwargs": {"shortest_image_edge": 32, "crop_fraction": 0.95,
                         "use_albumentations": True},
}
_STUB_VIEWS, _STUB_H, _STUB_W = 2, 24, 40
_STUB_ROWS = (32 // PATCH) ** 2                            # 4


def _stub_fe(tmp_path, *, fuse=True, cfg=None, views=_STUB_VIEWS,
             shortest=32):
    """A frontend shell with a real contract and no weights, kernels or CUDA."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "processor_config.json").write_text(
        json.dumps(cfg if cfg is not None else _STUB_CFG))
    fe = object.__new__(CLS)
    fe.device = "cpu"
    fe.checkpoint_path = str(ckpt)
    fe._fuse_image_embeds = fuse
    g = torch.Generator().manual_seed(0)
    Se = 9
    aux = {
        "grid_thw": torch.tensor([[1, 2, 2]] * views),
        "visual_pos_masks": torch.zeros(1, Se, dtype=torch.bool),
        "rope_cos": torch.randn(1, Se, 128, generator=g),
        "rope_sin": torch.randn(1, Se, 128, generator=g),
        "input_ids": torch.arange(Se).reshape(1, Se),
    }
    if fuse:
        aux["pixel_values"] = torch.randn(views * _STUB_ROWS, ROW_DIM,
                                          generator=g)
    else:
        aux["pixel_features"] = torch.randn(views * _STUB_ROWS, 1024,
                                            generator=g)
        aux["llm_input_embeds"] = torch.randn(1, Se, 2048, generator=g)
    fe._observation_contract = fe._snapshot_observation_contract(aux)
    fe._stub_aux = aux
    return fe


def _stub_frames(views=_STUB_VIEWS, h=_STUB_H, w=_STUB_W):
    g = torch.Generator().manual_seed(2)
    return torch.randint(0, 256, (views, h, w, 3), dtype=torch.uint8,
                         generator=g)


def test_infer_exposes_frames_as_an_optional_keyword():
    import inspect

    sig = inspect.signature(CLS.infer)
    assert "frames" in sig.parameters
    p = sig.parameters["frames"]
    assert p.default is None
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
        "frames= must be keyword-only: infer's first positional is the state, "
        "and a positional frames argument would silently swap them")


def test_frames_and_aux_pixel_values_together_are_refused(tmp_path):
    fe = _stub_fe(tmp_path)
    aux = dict(fe._stub_aux)
    with pytest.raises(ValueError, match="both given"):
        fe._observation_aux(_stub_frames(), aux)
    # aux *without* pixel_values is a legal override of the prompt-scoped keys.
    rest = {k: v for k, v in aux.items() if k != "pixel_values"}
    built = fe._observation_aux(_stub_frames(), rest)
    assert built["pixel_values"].shape == (_STUB_VIEWS * _STUB_ROWS, ROW_DIM)


def test_prompt_scoped_tensors_are_re_presented_by_identity(tmp_path):
    """The whole reason this arm is cheap. A copy would be numerically identical
    and would put the host compare straight back."""
    fe = _stub_fe(tmp_path)
    built = fe._observation_aux(_stub_frames(), None)
    for name, entry in fe._observation_contract.items():
        if name.endswith("_shape"):
            continue
        assert built[name] is entry["source"], name
    assert "pixel_values" in built


def test_the_identity_re_presentation_takes_the_validators_fast_path(tmp_path):
    """Pin it by what the validator does, not by what the dict holds.

    ``torch.equal`` is called only on the slow path, and ``validated_source`` is
    written only there too — so both staying untouched across repeated
    observations is the receipt that no host compare happened (red line #5).
    """
    import numpy as np

    fe = _stub_fe(tmp_path)
    real_equal = torch.equal
    calls = {"n": 0}

    def counted(a, b):
        calls["n"] += 1
        return real_equal(a, b)

    frames = _stub_frames().numpy()
    try:
        torch.equal = counted
        for _ in range(3):
            built = fe._observation_aux(frames, None)
            fe._validate_observation_contract(built)
    finally:
        torch.equal = real_equal
    assert calls["n"] == 0, (
        f"{calls['n']} host compares over 3 observations; the prompt-scoped "
        "tensors are not reaching the validator by identity")
    metadata = [k for k in fe._observation_contract if not k.endswith("_shape")]
    assert metadata, "nothing to pin"
    untouched = [k for k in metadata
                 if fe._observation_contract[k]["validated_source"] is None]
    assert untouched == metadata, (
        f"{sorted(set(metadata) - set(untouched))} went down the validator's "
        "slow path, which writes validated_source")


def test_a_rebuilt_bundle_misses_the_fast_path(tmp_path):
    """The control for the test above: fresh objects with equal values do fall
    through to the host compare. Without this, "0 calls" could mean the
    validator never ran."""
    fe = _stub_fe(tmp_path)
    real_equal = torch.equal
    calls = {"n": 0}

    def counted(a, b):
        calls["n"] += 1
        return real_equal(a, b)

    rebuilt = {k: v["source"].clone()
               for k, v in fe._observation_contract.items()
               if not k.endswith("_shape")}
    rebuilt["pixel_values"] = torch.zeros(_STUB_VIEWS * _STUB_ROWS, ROW_DIM)
    try:
        torch.equal = counted
        fe._validate_observation_contract(rebuilt)
    finally:
        torch.equal = real_equal
    assert calls["n"] > 0, (
        "equal-valued fresh tensors took the fast path too, so the identity "
        "check above is not measuring identity")


def test_the_frames_arm_refuses_unfused_mode(tmp_path):
    """Without the fusion the backbone also consumes ``llm_input_embeds``, which
    is an LLM forward over the prompt's text — not something an image path can
    produce. Serving a half-observation would be silent."""
    fe = _stub_fe(tmp_path, fuse=False)
    with pytest.raises(ValueError, match="fuse_image_embeds=True"):
        fe._pixel_values_from_frames(_stub_frames())


def test_a_dict_of_views_is_refused(tmp_path):
    fe = _stub_fe(tmp_path)
    f = _stub_frames()
    with pytest.raises(ValueError, match="not a dict"):
        fe._frames_to_device({"front": f[0], "wrist": f[1]})


def test_the_accepted_input_forms_agree(tmp_path):
    import numpy as np

    fe = _stub_fe(tmp_path)
    f = _stub_frames()
    ref = fe._pixel_values_from_frames(f)
    for form in (list(f), tuple(f), f.numpy(),
                 [f[0].numpy(), f[1].numpy()]):
        assert torch.equal(fe._pixel_values_from_frames(form), ref), type(form)
    with pytest.raises(ValueError, match="must be"):
        fe._frames_to_device(42)


def test_uint8_is_required_and_a_float_frame_is_not_silently_rescaled(tmp_path):
    """A caller who pre-normalized would otherwise be transformed twice, which
    yields a plausible image and a quietly wrong action."""
    fe = _stub_fe(tmp_path)
    with pytest.raises(ValueError, match="transformed twice"):
        fe._frames_to_device(_stub_frames().float())
    with pytest.raises(ValueError, match=r"\(views, H, W, 3\)"):
        fe._frames_to_device(_stub_frames()[0])


def test_the_plan_is_built_once_per_geometry(tmp_path):
    fe = _stub_fe(tmp_path)
    builds = {"n": 0}
    real = type(fe)._image_plan

    def counted(self, h, w):
        before = len(getattr(self, "_image_plans", {}) or {})
        p = real(self, h, w)
        builds["n"] += len(self._image_plans) > before
        return p

    fe._image_plan = counted.__get__(fe)
    for _ in range(4):
        fe._pixel_values_from_frames(_stub_frames())
    assert builds["n"] == 1, "the operator matrices were rebuilt per frame"
    # A different geometry is a new plan, not an error.
    fe._pixel_values_from_frames(_stub_frames(h=20, w=48))
    assert builds["n"] == 2
    assert len(fe._image_plans) == 2


def test_the_row_count_is_checked_against_the_prompt(tmp_path):
    """The authoritative geometry gate: rows are ``views * (shortest/16)**2``, so
    a changed camera count is caught even though every view still resizes fine."""
    fe = _stub_fe(tmp_path)
    with pytest.raises(ValueError, match="different camera"):
        fe._pixel_values_from_frames(_stub_frames(views=1))
    with pytest.raises(ValueError, match="different camera"):
        fe._pixel_values_from_frames(_stub_frames(views=3))


def test_a_lower_resolution_is_accepted_because_the_resize_is_smallest_edge(
        tmp_path):
    """Not a bug and worth pinning: the vendor transform is a smallest-edge
    resize, so 18x36 letterboxes to 36 and still yields the same grid. Refusing
    it would refuse a legitimate camera — as long as the scale stays exact, which
    36/32 = 9/8 does."""
    fe = _stub_fe(tmp_path)
    pv = fe._pixel_values_from_frames(_stub_frames(h=18, w=36))
    assert pv.shape == (_STUB_VIEWS * _STUB_ROWS, ROW_DIM)
    assert fe._image_plan(18, 36).shrink_p == 9


def test_the_processor_geometry_comes_from_the_checkpoint(tmp_path):
    fe = _stub_fe(tmp_path)
    assert fe._read_processor_geometry() == {"shortest": 32,
                                             "crop_fraction": 0.95}
    # ...and the plan really uses it rather than the module default of 256.
    plan = fe._image_plan(_STUB_H, _STUB_W)
    assert plan.shortest == 32 and plan.grid == 2
    assert fe._read_processor_geometry() is fe._proc_geom      # cached


@pytest.mark.parametrize("cfg,exc,match", [
    ({"processor_kwargs": {"shortest_image_edge": 32,
                           "use_albumentations": True}},
     RuntimeError, "crop_fraction"),
    ({"processor_kwargs": {"shortest_image_edge": 32, "crop_fraction": 0.95,
                           "use_albumentations": False}},
     NotImplementedError, "use_albumentations=False"),
    ({}, RuntimeError, "shortest_image_edge"),
])
def test_the_processor_config_is_not_guessed_at(tmp_path, cfg, exc, match):
    """The vendor's *code* default for ``crop_fraction`` is 0.9 while this
    checkpoint ships 0.95, so a missing key must not fall back to anything."""
    fe = _stub_fe(tmp_path, cfg=cfg)
    with pytest.raises(exc, match=match):
        fe._read_processor_geometry()


def test_a_missing_processor_config_names_the_workaround(tmp_path):
    ckpt = tmp_path / "bare"
    ckpt.mkdir()
    fe = object.__new__(CLS)
    fe.device = "cpu"
    fe.checkpoint_path = str(ckpt)
    fe._fuse_image_embeds = True
    with pytest.raises(RuntimeError, match="already-processed aux bundle"):
        fe._read_processor_geometry()


def test_the_image_path_stays_outside_every_capture():
    """§6.17's correction: a per-observation source must not be baked into a
    graph. The plan and its operators are cached on the instance, so a future
    capture is possible without re-deriving them — but nothing captures today.
    """
    import inspect

    for name in ("_capture_backbone_graph", "_capture_dit_graphs",
                 "_kbb_forward"):
        body = inspect.getsource(getattr(CLS, name))
        for needle in ("_pixel_values_from_frames", "_frames_to_device",
                       "_observation_aux", "_groot_n17_preprocess"):
            assert needle not in body, f"{name} reaches the image path"

    # infer builds the aux before it dispatches to either backbone arm, so both
    # arms see an ordinary bundle and neither has to know about frames.
    infer_src = inspect.getsource(CLS.infer)
    assert infer_src.index("_observation_aux") < \
        infer_src.index("if self._use_backbone_graph:")


def test_the_pinned_staging_buffer_is_used_for_a_host_upload():
    """Source pin, since allocating pinned memory needs a CUDA context.

    A pageable ``.cuda()`` costs ~0.5 ms more per observation here — a third of
    this path's whole budget — and ``hyvla_rtx.py`` is the precedent for the
    persistent pinned buffer.
    """
    import inspect

    src = inspect.getsource(CLS._frames_to_device)
    assert "pin_memory=True" in src
    assert "non_blocking=True" in src
    assert '"_frames_pin"' in src and '"_frames_dev"' in src, (
        "both buffers must persist; reallocating per frame defeats the point")
    # A CPU-device frontend must not try to allocate pinned memory at all, which
    # is also what lets this whole file run without CUDA.
    assert 'torch.device(self.device).type == "cpu"' in src
