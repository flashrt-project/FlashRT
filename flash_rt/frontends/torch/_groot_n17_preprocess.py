"""The GR00T N1.7 per-observation image path, in pure torch on the GPU.

Why this module exists
----------------------
Every N1.7 CUDA frontend in this repo (Thor, Thor FP8, RTX, RTX FP16/FP8, SM89,
AMD, Orin) requires the caller to hand it an already-processed ``aux`` bundle,
so the vendor's ``Gr00tN1d7Processor`` runs per observation. On Orin that costs
**12.916-14.027 ms of host time on the weak ARM CPU** — the whole chain measured
end to end on 4 real frames across both datasets, clocks locked, median of 11.
Stage by stage (median of 9, 2 views), the same work divides as:

    albumentations eval chain (letterbox -> INTER_AREA -> crop -> INTER_AREA)  5.27 ms
    HF Qwen2VLImageProcessorFast (rescale/normalize/patchify)              2.02-5.30 ms
    PIL/torch glue around them (``Image.fromarray`` -> ``np.array`` -> ...)      3.87 ms

Those three sum to 11.16-14.44 ms, which brackets the whole-chain number — but
the sum of separately measured stage medians is *not* a measurement, and quoting
it as one is how an earlier draft of this work booked "~10.3 ms" (see
docs §6.23.6). The end-to-end figure is the one to cite.

The 3.87 ms of glue is pure waste (``apply_with_replay`` converts numpy → PIL →
numpy, verified lossless), and the rest is resize + rescale + normalize +
shuffle: all device work that happens to have been written for a CPU. This
module replaces the whole thing. Measured on 2 views: **1.448-1.485 ms** of
device work in isolation, **1.803-1.854 ms** paired-alternating against the cv2
reference chain (AGENTS.md §3.8's caliper, and the one the gate asserts);
including a pinned host-to-device upload the per-observation total lands at
**99.579-100.428 ms** against the vendor arm's **112.374-115.499 ms**, i.e.
**11.945-15.410 ms saved (1.1189-1.1540x)**.

That host time is serialized *before* the model runs, so removing it removes
wall clock directly. It is **not** a submission-overlap win: docs §6.13.3
withdrew the "backbone is CPU submission bound" reading (the launch-bound share
measured 7.3%, not 91.2%), so the mechanism here is simply that 13 ms of ARM
scalar code no longer executes.

The prompt-scoped half of the processor (``input_ids``, ``attention_mask``,
``image_grid_thw``, ``embodiment_id``, the rope tables) is **not** here: those
do not change between observations, and ``set_prompt`` already consumes them
once. Verified directly — across observations only ``pixel_values`` and
``state`` differ.

The chain being reproduced
---------------------------
``Gr00tN1d7Processor`` at eval time, with this checkpoint's own
``processor_config.json`` (``crop_fraction=0.95``, *not* the 0.9 code default;
``image_crop_size=[230,230]`` is unused because ``fraction_to_use`` prefers
``crop_fraction`` when it is not None)::

    LetterBoxPad        pad to square with black bars, top/bottom = (max-h)//2
    SmallestMaxSize     cv2.INTER_AREA to shortest_image_edge=256
    FractionalCenterCrop  int(256*0.95) = 243, offset (256-243)//2 = 6
    SmallestMaxSize     cv2.INTER_AREA back to 256

then the HF image processor's rescale (1/255), normalize (mean=std=0.5) and the
Qwen2-VL merge-block patchify to ``(views*256, 1536)``.

It is fully deterministic: ``FractionalCenterCrop`` is a centre crop
(``image_augmentations.py:257``, wired at ``:484``). The one that draws with
``np.random.randint`` is ``FractionalRandomCrop`` (``:188``), which only the
*train* ``ReplayCompose`` uses. Both names are in the same file, which is worth
knowing before trusting either.

What is exact, and what is not
------------------------------
Two different answers, and the difference is the whole design:

* **The shrink step (640→256) is bit-exact**, and provably so — and the proof
  generalizes to any geometry, which is why it is *checked* rather than assumed.
  Write the scale ``s = src/dst`` in lowest terms as ``p/q``. Continuous coverage
  then gives weights ``m_i/p`` with non-negative integers ``m_i`` summing to
  ``p``, so the exact output of a pixel is ``N/p`` for an integer ``N``. A
  rounding tie needs ``N/p = k + 1/2``, i.e. ``2N = p(2k+1)``; for **odd ``p``**
  that forces ``p | N``, making ``N/p`` an integer and never a half-integer. No
  ties for *any* input, so fp32 error (~1e-4 against a tie margin of ``1/(2p)``)
  cannot flip a rounding. At the shipped 640→256 the scale is ``5/2``: weights
  exactly {0.2, 0.4}, 3 taps, ``out = (a + 2b + 2c)/5``. Measured ``max=0``
  against ``cv2.INTER_AREA`` on 4 real frames × 2 datasets (393216 pixels each),
  in numpy fp64 and torch fp32 on CUDA. ``build_image_plan`` records whether the
  condition holds and ``frames_to_resized`` refuses when it does not — e.g. a
  512-wide camera gives ``p=2``, where ``out = (a+b)/2`` ties on every odd sum
  and OpenCV's tie-break rule is not what this reproduces.

  Note that torch's own ``interpolate(mode="area")`` is off by 18 LSB even where
  this is exact: it uses integer source bounds and a plain mean instead of
  fractional coverage weights. **fp16/bf16 weights are not exact** (``max=1``)
  and neither is TF32, whose 10-bit mantissa puts ~0.25 of error against a
  0.1 tie margin — both are refused.

  Exactness was measured at three shrinking scales — 640→256 (5/2), 320→256 and
  40→32 (both 5/4) — in fp32 *and* fp64, on random images as well as ramps,
  checkerboards and flat fields: ``max=0`` on every one. It is a property of
  shrinking, not of the shipped geometry.
* **The enlarge step (243→256) is not exact**, and the cause is *not* the
  weights. Probing OpenCV's effective operator directly — one-hot columns into
  ``cv2.resize`` on float32 input, which reads the matrix out exactly — shows its
  weights **are** continuous coverage, agreeing to 7.2e-8 at 243→256, 320→256,
  640→256 and 40→32 alike. So the enlarging branch does not use a different
  kernel shape.

  The residual is in OpenCV's **uint8 accumulation/rounding**, and it is not a
  precision problem: running this module's own arithmetic in fp64 instead of
  fp32 gives the *same* ``max=1``. Four formulations were tried on the same
  images — float coverage with one rounding (this module), 1/2048 fixed-point
  with round-half-up and an intermediate uint8 rounding, 1/2048 fixed-point with
  truncation, and a single ``>>22`` over both passes — and **all four converge on
  max=1**, with float coverage carrying the *lowest* differing fraction (12.2%
  random / 11.5% ramp / 13.8% checkerboard at 243→256, against 21.0% and 77.7%
  for the fixed-point forms). What ships is therefore the best of the available
  reproductions, and the remaining LSB is declared unknown rather than guessed
  at. On real frames it measures **max=1, mean 0.051-0.069, on 5.1-6.9% of
  pixels** — lower than on random noise, because a camera frame has fewer pixels
  sitting near a rounding boundary.

That ≤1 LSB was measured end to end before this module was written (4 frames ×
2 datasets, both tiers): decoded action against the HF fixture is **never worse
and is better on 2 of 4 frames** (0.2626→0.1868°, 0.3872→0.2499°), with
GPU-vs-reference decoded cos 0.9999994–0.9999998. It sits *inside* the existing
bf16 noise floor rather than on top of it. Per §6.20's lesson n=4 supports "no
systematic degradation" and nothing stronger, which is why the precision suite
gates this path rather than assuming it.

:func:`host_reference_chain` keeps the cv2 version for exactly that reason —
fallback and numerical oracle, the role ``_rope_rotate_half`` plays in
``pipeline_orin.py``. It is bit-exact against the vendor's albumentations chain
(``np.array_equal``, 6/6 real frames across both datasets) and needs neither
albumentations nor PIL nor the vendor package.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PATCH", "MERGE", "TEMPORAL", "ROW_DIM",
    "shrink_scale_is_exact",
    "area_operator",
    "ImagePlan",
    "build_image_plan",
    "patchify",
    "frames_to_resized",
    "frames_to_pixel_values",
    "host_reference_chain",
]

#: Qwen3-VL vision geometry. A checkpoint whose processor differs is a different
#: model, so :func:`build_image_plan` checks rather than adapts.
PATCH = 16
MERGE = 2
TEMPORAL = 2
#: Row width of ``pixel_values``: channels × temporal copies × patch area.
ROW_DIM = 3 * TEMPORAL * PATCH * PATCH          # 1536

#: Largest shrink-scale numerator ``p`` for which the no-ties argument still
#: survives fp32 accumulation. The exact output is ``N/p``, so its distance from
#: a rounding tie is at least ``1/(2p)``; an fp32 sum of ~3 taps over values
#: ≤255 carries ~1e-4 of error (one ULP at 255 is 3.0e-5), so ``1/(2p) > 1e-4``
#: needs ``p < 5000``. The bound is set an order of magnitude inside that: any
#: real camera geometry lands at ``p`` ≤ 15 (640→256 is 5/2, 1280→256 is 5), and
#: a ``p`` above 1000 means a source thousands of pixels wide being resized to
#: 256, which is not this checkpoint's envelope.
_MAX_EXACT_NUMERATOR = 1000


def shrink_scale_is_exact(src: int, dst: int) -> tuple:
    """Whether a ``src``→``dst`` area resize can be reproduced bit-exactly.

    Returns ``(p, exact)`` where ``p`` is the numerator of ``src/dst`` in lowest
    terms. The argument is in the module docstring: the continuous-coverage
    weights are ``m_i/p`` for integers ``m_i`` summing to ``p``, so the exact
    output is ``N/p``; a rounding tie needs ``2N = p(2k+1)``, which for odd
    ``p`` forces ``p | N`` and makes the exact value an integer. No ties ⇒ an
    fp32 sum cannot round differently from an exact one, provided the tie margin
    ``1/(2p)`` still exceeds the accumulation error (see
    ``_MAX_EXACT_NUMERATOR``).

    This is a *precondition*, not a measurement. It says the arithmetic is capable
    of being exact; ``max=0`` against ``cv2.INTER_AREA`` was then measured at
    three shrinking scales — 5/2 (640→256, the shipped one) and 5/4 (320→256 and
    40→32) — in fp32 and fp64, on random and structured images alike. An odd
    ``p`` that is neither of those satisfies the same argument but has not itself
    been measured, which is why the flag is recorded on the plan rather than
    hidden inside it.

    Args:
        src: source edge length.
        dst: target edge length.

    Returns:
        ``(p, exact)``.
    """
    from fractions import Fraction

    scale = Fraction(int(src), int(dst))
    p = scale.numerator
    return p, bool(p % 2 == 1 and p <= _MAX_EXACT_NUMERATOR)


def area_operator(src: int, dst: int, *, device=None, dtype=None):
    """OpenCV INTER_AREA's coverage weights as a ``(dst, src)`` matrix.

    Output pixel ``x`` covers source span ``[x*s, (x+1)*s)`` with ``s = src/dst``,
    and source pixel ``i`` gets the fraction of that span it overlaps::

        A[x, i] = max(0, min(i+1, hi) - max(i, lo)) / s

    Rows therefore sum to exactly 1.0 and the operator is separable, so a resize
    is ``A @ x @ A.T``. Built in fp64 and cast down: the weights are the one part
    of this that must not carry rounding error, since §6.22's lesson about the
    shrink step being exact rests on them being exact.

    Args:
        src: source edge length.
        dst: target edge length.
        device: target device for the matrix.
        dtype: target dtype. **Must be fp32 or wider** — fp16/bf16 weights were
            measured at max=1 LSB against cv2 where fp32 is max=0.

    Returns:
        ``(dst, src)`` on ``device``.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"the area operator needs fp32 or wider weights, got {dtype}: at "
            "fp16/bf16 the shrink step measures max=1 LSB against "
            "cv2.INTER_AREA where fp32 measures max=0, and the exactness "
            "argument for the shrink step assumes the weights are exact")
    scale = src / dst
    x = torch.arange(dst, dtype=torch.float64)
    i = torch.arange(src, dtype=torch.float64)
    lo = (x * scale)[:, None]
    hi = ((x + 1.0) * scale)[:, None]
    overlap = torch.minimum(i + 1.0, hi) - torch.maximum(i, lo)
    return (overlap.clamp_min(0.0) / scale).to(dtype=dtype, device=device)


@dataclass(frozen=True)
class ImagePlan:
    """One camera geometry's resize chain, with its operators already built.

    Built once per geometry and reused for every observation — the same lifecycle
    as the frontend's rope tables. Both resizes act on a **square** image (the
    letterbox guarantees it), so one operator per step serves the H and W axes.

    Attributes:
        pad: ``(left, right, top, bottom)`` for ``torch.nn.functional.pad``.
        shrink: ``(shortest, side)`` area operator for the first resize.
        crop: ``(lo, hi)`` half-open slice applied to both spatial axes.
        enlarge: ``(shortest, cropped)`` area operator for the second resize.
        shortest: the edge length both resizes target.
        grid: patches per axis after the final resize, i.e. ``shortest // PATCH``.
        rows: ``pixel_values`` rows per view (``grid ** 2``).
        shrink_p: numerator of ``side/shortest`` in lowest terms.
        shrink_exact: whether the no-ties argument covers this geometry, i.e.
            whether :func:`frames_to_resized` may claim ``max=0`` against
            ``cv2.INTER_AREA`` for the first resize. ``False`` makes it refuse.
    """

    pad: tuple
    shrink: object
    crop: tuple
    enlarge: object
    shortest: int
    grid: int
    rows: int
    shrink_p: int = 1
    shrink_exact: bool = True

    @property
    def device(self):
        return self.shrink.device


def build_image_plan(height: int, width: int, *, device,
                     shortest: int = 256, crop_fraction: float = 0.95):
    """Build the :class:`ImagePlan` for one camera geometry.

    Refuses loudly outside the envelope this chain was verified in, rather than
    silently running a *different* OpenCV branch: INTER_AREA has separate
    shrinking and enlarging code paths, and applying the wrong one produces
    plausible images and a quietly degraded cosine.

    A geometry whose shrink scale has an **even** numerator is *not* refused
    here — it is recorded as ``shrink_exact=False`` and
    :func:`frames_to_resized` refuses it. That split keeps this function usable
    for :func:`host_reference_chain`, which reads only the scalar geometry and
    stays correct at any scale because it delegates to OpenCV itself.

    Args:
        height: raw frame height.
        width: raw frame width.
        device: device for the operator matrices.
        shortest: ``shortest_image_edge`` from the checkpoint's processor config.
        crop_fraction: ``crop_fraction`` from the same config. This checkpoint
            ships **0.95**; the code default of 0.9 is not what runs.

    Returns:
        An :class:`ImagePlan`.
    """
    import torch

    if shortest % (PATCH * MERGE):
        raise ValueError(
            f"shortest_image_edge={shortest} is not a multiple of "
            f"patch*merge={PATCH * MERGE}, so the patches would not tile the "
            "resized frame. Refusing to guess a padding rule: a wrong grid here "
            "shows up as a shape mismatch three stages downstream.")
    side = max(int(height), int(width))
    if not shortest < side:
        raise NotImplementedError(
            f"{height}x{width} letterboxes to {side}x{side}, and the first "
            f"resize to {shortest} would not shrink. OpenCV's INTER_AREA takes "
            "a different branch when enlarging, which this plan does not "
            "reproduce exactly (max=1 LSB); refusing rather than silently "
            "changing the arithmetic.")
    pad_h, pad_w = side - int(height), side - int(width)
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    cropped = max(1, int(shortest * crop_fraction))
    if not cropped < shortest:
        raise NotImplementedError(
            f"crop_fraction={crop_fraction} of {shortest} leaves {cropped}, so "
            "the second resize would not enlarge — a different OpenCV branch. "
            "Refusing rather than silently changing the arithmetic.")
    lo = (shortest - cropped) // 2

    p, exact = shrink_scale_is_exact(side, shortest)
    return ImagePlan(
        pad=pad,
        shrink=area_operator(side, shortest, device=device),
        crop=(lo, lo + cropped),
        enlarge=area_operator(cropped, shortest, device=device),
        shortest=shortest,
        grid=shortest // PATCH,
        rows=(shortest // PATCH) ** 2,
        shrink_p=p,
        shrink_exact=exact,
    )


def patchify(x):
    """``(V, 3, H, W)`` normalized floats → ``(V*patches, 1536)`` rows.

    The Qwen2-VL merge-block shuffle: one row per patch, ordered by view, then
    merged block, then position within the block; within a row by channel, then
    temporal copy, then pixel. The two temporal slots are identical copies.

    Verified ``torch.equal`` against ``Qwen2VLImageProcessorFast`` **in bf16**
    (0 of 786432 elements differing, 4/4 real frames). In fp32 it differs by
    5.9e-08 because HF rescales by multiplying with ``1/255`` as a float32
    constant while this divides by ``255.0``; the difference vanishes under the
    bf16 cast the model actually eats, so the gate is on bf16.

    Args:
        x: ``(V, 3, H, W)`` float, already rescaled and normalized. ``H`` and
            ``W`` must be multiples of ``PATCH``.

    Returns:
        ``(V * (H//PATCH) * (W//PATCH), ROW_DIM)`` contiguous, dtype ``x.dtype``.
    """
    import torch

    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(
            f"patchify takes (views, 3, H, W), got {tuple(x.shape)}")
    views, channels, height, width = x.shape
    if height % PATCH or width % PATCH:
        raise ValueError(
            f"{height}x{width} does not tile into {PATCH}px patches")
    grid_h, grid_w = height // PATCH, width // PATCH

    x = x.reshape(views, channels, grid_h, PATCH, grid_w, PATCH)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(views, grid_h // MERGE, MERGE, grid_w // MERGE, MERGE,
                  channels, PATCH * PATCH)
    x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
    rows = views * grid_h * grid_w
    x = x.reshape(rows, channels, 1, PATCH * PATCH)
    x = x.expand(rows, channels, TEMPORAL, PATCH * PATCH)
    return x.reshape(rows, channels * TEMPORAL * PATCH * PATCH).contiguous()


def frames_to_resized(frames, plan: ImagePlan):
    """Raw ``uint8`` frames → the ``(V, 3, shortest, shortest)`` ``uint8`` the
    patchify reads. Directly comparable with :func:`host_reference_chain`.

    Exposed separately from :func:`frames_to_pixel_values` because the two
    resizes have different exactness guarantees (bit-exact vs ≤1 LSB) and a gate
    that can only see the final ``pixel_values`` cannot tell them apart.

    The matmul association is pinned to ``(A @ x) @ A.T`` (vertical pass then
    horizontal) because that is the order the ``max=0`` exactness measurement was
    taken in. The right-associated ``A @ (x @ A.T)`` is the same mathematics and
    rounds differently in fp32: measured on a real frame it moves **2 of 393216**
    uint8 pixels by one LSB (4 of 786432 bf16 elements downstream). Small, but it
    is the difference between a gate that reproduces its recorded number and one
    that does not, so do not "optimize" the order without re-running it.

    Args:
        frames: ``(V, H, W, 3)`` ``uint8`` on ``plan.device``.
        plan: the :class:`ImagePlan` for this geometry.

    Returns:
        ``(V, 3, shortest, shortest)`` ``uint8`` on ``plan.device``.
    """
    import torch

    if frames.dtype != torch.uint8:
        raise ValueError(
            f"frames must be uint8 (the vendor chain resizes integers and only "
            f"then rescales), got {frames.dtype}")
    if frames.dim() != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must be (views, H, W, 3), got {tuple(frames.shape)}")
    if frames.device != plan.device:
        raise ValueError(
            f"frames are on {frames.device} but this plan's operators are on "
            f"{plan.device}; upload the frames first (a pinned staging buffer "
            "makes that ~0.2 ms cheaper than a pageable copy)")
    if not plan.shrink_exact:
        side = int(plan.shrink.shape[1])
        reason = ("even" if plan.shrink_p % 2 == 0
                  else f"above {_MAX_EXACT_NUMERATOR}")
        raise NotImplementedError(
            f"this geometry letterboxes to {side}x{side} and resizes to "
            f"{plan.shortest}, a scale whose numerator in lowest terms is "
            f"{plan.shrink_p} — {reason}. Rounding ties are then reachable (at "
            "p=2 the resize is (a+b)/2 and ties on every odd sum) and OpenCV's "
            "tie-break rule is not what the continuous-coverage operator "
            "reproduces, so the result would drift from cv2.INTER_AREA by ~1 LSB "
            "on an unbounded set of pixels instead of the bounded 5.5% the "
            "enlarge step already costs. Use host_reference_chain (it delegates "
            "to OpenCV itself and is correct at any scale) or feed an "
            "already-processed aux bundle.")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(
            "TF32 matmul is enabled (torch.backends.cuda.matmul.allow_tf32). "
            "Its 10-bit mantissa puts ~0.25 of error on a value of 255 against "
            "a rounding-tie margin of 1/(2p) = "
            f"{0.5 / plan.shrink_p:.4f}, so the shrink step would stop being "
            "bit-exact against cv2.INTER_AREA and nothing would report it. The "
            "measured max=0 for this path is an fp32 result. Set it to False, "
            "or use host_reference_chain.")

    x = torch.nn.functional.pad(frames.permute(0, 3, 1, 2).to(torch.float32),
                                plan.pad)
    # Round-to-nearest-even after each resize: that is where uint8 quantization
    # happens in the vendor chain, and rounding once at the end instead would
    # feed the crop a different image.
    x = (plan.shrink @ x @ plan.shrink.t()).round().clamp(0.0, 255.0)
    lo, hi = plan.crop
    x = x[:, :, lo:hi, lo:hi]
    x = (plan.enlarge @ x @ plan.enlarge.t()).round().clamp(0.0, 255.0)
    return x.to(torch.uint8)


def frames_to_pixel_values(frames, plan: ImagePlan, *, mean: float = 0.5,
                           std: float = 0.5):
    """Raw ``uint8`` camera frames → the ``pixel_values`` matrix, on the GPU.

    Args:
        frames: ``(V, H, W, 3)`` ``uint8``, **already on ``plan.device``**. The
            upload is deliberately not done here: it is part of the frame's cost
            and belongs where the caller can see it and reuse a pinned buffer.
        plan: the :class:`ImagePlan` for this geometry.
        mean: the processor's ``image_mean``.
        std: the processor's ``image_std``.

    Returns:
        ``(V * plan.rows, ROW_DIM)`` contiguous fp32 on ``plan.device``. The
        caller's ``.to(bf16)`` is what makes it match HF bit for bit; see
        :func:`patchify`.
    """
    import torch

    x = frames_to_resized(frames, plan).to(torch.float32)
    return patchify((x / 255.0 - mean) / std)


def host_reference_chain(views, plan: ImagePlan):
    """The cv2 chain this module replaces. **Fallback and numerical oracle.**

    Bit-exact against the vendor's albumentations ``eval_image_transform``
    (``np.array_equal``, zero differing pixels, 6/6 real frames across both the
    480x640 real-robot and 360x640 sim datasets) and needs no albumentations,
    no PIL and no import of the vendor package.

    Kept for two reasons: it is the reference the GPU path is gated against, and
    it is the correct answer on a build without CUDA. It is ~4x slower than
    :func:`frames_to_pixel_values` and runs on the host, so nothing in the
    serving path should reach for it.

    Args:
        views: sequence of ``(H, W, 3)`` ``uint8`` numpy arrays.
        plan: the :class:`ImagePlan` describing the same geometry. Only its
            scalar geometry is used; the operator matrices are not.

    Returns:
        ``(V, 3, shortest, shortest)`` ``uint8`` numpy array.
    """
    import numpy as np

    try:
        import cv2
    except ImportError as exc:                    # pragma: no cover - env specific
        raise ImportError(
            "the host reference chain needs opencv (cv2): it reproduces "
            "OpenCV's INTER_AREA, which is the definition of correct here. "
            "Install opencv-python, or use frames_to_pixel_values on a CUDA "
            "build and gate it against a captured reference instead.") from exc

    side = plan.shortest
    lo, hi = plan.crop
    out = []
    for v in views:
        a = np.asarray(v)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(
                f"each view must be (H, W, 3) uint8, got {a.shape} {a.dtype}")
        h, w = a.shape[:2]
        m = max(h, w)
        ph, pw = m - h, m - w
        if ph or pw:
            a = cv2.copyMakeBorder(a, ph // 2, ph - ph // 2, pw // 2,
                                   pw - pw // 2, cv2.BORDER_CONSTANT, value=0)
        a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
        a = a[lo:hi, lo:hi]
        a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA)
        out.append(a.transpose(2, 0, 1))
    return np.stack(out)
