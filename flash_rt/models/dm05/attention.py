"""FlashRT FA2 attention backends for the DM05 text prefix and action expert.

DM05's three attention sites default to framework operators: the Gemma3 text
prefix runs the HF ``eager`` path (two BF16 BMMs plus an online softmax over a
materialised ``[1, H, S, S]`` score matrix), while SigLIP and the action expert
run SDPA. This module provides FlashRT FA2 implementations for the prefix and
the action expert, selected through ``llm_attn_implementation`` and
``action_attn_implementation``.

Both entry points take raw device pointers, explicit strides and an explicit
stream, and allocate nothing internally, so both are CUDA-graph capture safe.
Scratch buffers are cached for exactly that reason: a buffer allocated during
capture must keep its address across replays.

Two properties of the FA2 bindings drive the design and neither is obvious:

  * FA2's causal diagonal is BOTTOM-RIGHT aligned. With ``Sq != Sk`` query *i*
    attends keys ``<= i + (Sk - Sq)``, which is exactly "a causal run of rows on
    top of a fully-visible prefix". The prefix therefore decomposes into plain
    causal/non-causal calls with no mask tensor at all.
  * At the action-expert shape (``Sq = 10``) a non-split kernel leaves most SMs
    idle: ``fwd_bf16`` with null accumulators costs 34.96 us against 19.33 us
    for the same entry point with ``lse_accum``/``o_accum`` supplied, and
    ``fwd_bf16_seqused`` forces ``num_splits = 1`` internally (35.06 us). The
    masked action-expert path must therefore use ``fwd_bf16_seqused_splitkv``.

Measured on an RTX 5060 Ti (sm120, 36 SMs) against the SDPA/eager defaults, the
prefix site is a clear win and the action expert a smaller one; the SigLIP tower
is left on SDPA because FA2 there was measured slower at the served
two-image, 1024-patch shape. ``num_splits`` is not numerically neutral -- each
split reduces its own FP32 partial, so the split count fixes the summation order
and therefore the rounding. It defaults to one split per SM; see
:func:`set_num_splits`.
"""
from __future__ import annotations

import logging
import os

import torch

_LOG = logging.getLogger(__name__)

try:
    from flash_rt import flash_rt_fa2 as _fa2
except ImportError:  # pragma: no cover - optional dependency
    _fa2 = None


def is_available() -> bool:
    """True when the FlashRT FA2 extension is importable on this install."""
    return _fa2 is not None


def _require() -> None:
    if _fa2 is None:
        raise RuntimeError(
            "attn_implementation='flash_rt_fa2' needs the FlashRT FA2 "
            "extension, but `from flash_rt import flash_rt_fa2` failed. "
            "Install FlashRT for this GPU architecture or select 'sdpa'.")


_NSMS = (torch.cuda.get_device_properties(0).multi_processor_count
         if torch.cuda.is_available() else 1)
_MAX_SPLITS = 128          # worst-case bound for the num_splits heuristic
# Split-KV for the causal prefix text segments; off restores one CTA row per head.
_PREFIX_SPLIT = os.environ.get("DM05_FA2_PREFIX_SPLIT", "1") == "1"
_NEG = torch.finfo(torch.float32).min

# How many key-splits the split paths request. The bindings take this as an
# explicit argument where `pytorch_flash` applies its own internal heuristic, and
# that difference is NOT numerically neutral: each split accumulates its own fp32
# partial which the combine kernel then reduces, so the split count fixes the
# summation order and therefore the rounding.
#
# This is the named suspect for a measured accuracy movement. On one quantized
# boundary, enabling FA2 at the action expert and SigLIP together moved a
# 444-observation 6-DoF cosine minimum from 0.993429 to 0.984050, with 3
# observations dropping below a 0.99 acceptance threshold.
# An earlier version of this module asserted that at the suffix and vision sites
# `pytorch_flash` and `flash_rt_fa2` are the same kernel instantiation and that
# the change therefore "buys the contract, not the clock" -- but a bit-identical
# kernel over identical inputs cannot move a cosine at all, so that assertion is
# incomplete: the instantiation matches while the split count need not.
#
# set_num_splits(0) asks the kernel for its own heuristic instead of one split
# per SM, which is the cheapest test of whether the split count carries the loss.
# It is expected to cost time at Sq=10 (the docstring measures 34.96 us at
# num_splits=1 against 19.33 us split), so this knob trades the clock for the
# gate and both numbers have to be reported together.
_SPLITS = _NSMS


def set_num_splits(num_splits: int) -> None:
    """Override the key-split count used by the split-KV paths.

    ``0`` asks the kernel for its own heuristic instead of one split per SM.
    Because each split reduces a private FP32 partial, this changes the
    summation order and therefore the numerics; at ``Sq = 10`` it is also
    measurably slower (34.96 us at one split against 19.33 us split). Change it
    only with both the accuracy and the latency number in hand.
    """
    global _SPLITS
    _SPLITS = int(num_splits)


def _bshd_strides(t: torch.Tensor) -> tuple[int, int, int]:
    """Strides of the (B, S, H, D) view of a [B, H, S, D] model tensor.

    The model hands attention [B, H, S, D]; FA2 indexes (batch, row, head).
    Passing the permuted strides means the transpose costs nothing -- no
    repacking copy, which is the whole reason the binding takes strides at all.
    """
    return (t.stride(0), t.stride(2), t.stride(1))


# --------------------------------------------------------------------- scratch
# lse is never read back at any of these sites, so one buffer per (B, Hq, Sq)
# is shared by every segment of every layer: the segments run in issue order on
# one stream, so overlapping writes cannot race. `o_accum`/`lse_accum` are only
# allocated for the split paths that ask for them. Everything is cached, because
# a buffer allocated inside a CUDA graph capture must keep its address across
# replays and a fresh `torch.empty` per call would not.
_SCRATCH: dict[tuple, tuple[int, int, int]] = {}


_SCRATCH_KEEP: list = []   # owns the storage the cached pointers point into


def _scratch(dev, b: int, hq: int, sq: int, d: int, split: bool):
    """Return (lse_ptr, lse_accum_ptr, o_accum_ptr, lse_accum_tensor).

    The tensor comes back as well as the pointer because the seqused split-KV
    entry REQUIRES `lse_accum` to be re-initialised to -inf on every call; see
    the comment on that call in `_fwd`. Pointers alone cannot express that.
    """
    key = (b, hq, sq, d, split)
    got = _SCRATCH.get(key)
    if got is None:
        lse = torch.empty(b, hq, sq, device=dev, dtype=torch.float32)
        la = oa = None
        if split:
            la = torch.empty(_MAX_SPLITS, b, hq, sq, device=dev, dtype=torch.float32)
            oa = torch.empty(_MAX_SPLITS, b, hq, sq, d, device=dev, dtype=torch.float32)
        _SCRATCH_KEEP.append((lse, la, oa))
        got = (lse.data_ptr(),
               la.data_ptr() if la is not None else 0,
               oa.data_ptr() if oa is not None else 0,
               la)
        _SCRATCH[key] = got
    return got


# ------------------------------------------------------------------- raw calls
def _fwd(qp, kp, vp, op, qs, ks, vs, os_, b, sq, sk, hq, hkv, d, dev,
         scale, causal, split, seqused_ptr=0):
    lp, lap, oap, la = _scratch(dev, b, hq, sq, d, split)
    strm = torch.cuda.current_stream().cuda_stream
    if seqused_ptr:
        # seqused forces num_splits=1 in the non-splitkv entry, so the split
        # variant is the only usable one at Sq=10 (35.06 us vs 19.44 us).
        #
        # THIS PREFILL IS MANDATORY. Omitting it is a real bug, not a micro-opt:
        # The repository's own RTX caller states the contract, at
        # flash_rt/hardware/rtx/attn_backend.py:_call_fvk_fa2_seqused_splitkv:
        #   "At replay with a smaller `seqused`, splits whose K-range lies
        #    entirely past `seqused` are empty and the kernel does NOT write
        #    `-inf` into their `lse_accum` slot -- so we MUST pre-fill
        #    `lse_accum` with `-inf` each call (captured in the graph). Empty
        #    splits then keep `-inf` -> combine weight `exp(-inf)=0` ->
        #    correct. `o_accum` needs no init (it is multiplied by the zero
        #    weight)."
        # `_scratch` handed out a `torch.empty` buffer and nothing ever
        # initialised it, so every empty split contributed `exp(garbage - max)`
        # times a garbage `o_accum` row to the combine. This is the suffix
        # site's exact shape: `sk` is the fixed max K length while `seqused` is
        # the real one, so empty splits are the normal case, not an edge case.
        #
        # This is the leading candidate cause of the accuracy regression
        # described above (0.993429 -> 0.984050). A wrong-integration result is
        # not evidence against the kernel: the `_SPLITS` mechanism stays a real
        # second-order effect on summation order, but it is no longer the first
        # suspect, because an uninitialised reduction accumulator outranks a
        # rounding-order change by orders of magnitude.
        #
        # The whole buffer is filled rather than `la[:_SPLITS]` so the fill is
        # also correct when `_SPLITS == 0` hands the split count to the
        # kernel's own heuristic. 128 x b x hq x sq fp32 is ~41 KB at the
        # suffix shape; the fill is captured in the graph like any other
        # kernel and costs no host work at replay.
        la.fill_(float("-inf"))
        _fa2.fwd_bf16_seqused_splitkv(
            qp, kp, vp, op, lp, seqused_ptr, lap, oap,
            b, sq, sk, hq, hkv, d, qs, ks, vs, os_, scale, _SPLITS, strm)
    elif causal:
        # With accumulators the causal wrapper picks num_splits itself (its last
        # argument is the SM count, not a split count) and stays unsplit when a
        # split would not pay, e.g. the 41-row opening text segment. The late text
        # segments are 7-8 query rows over 305-568 keys, i.e. ceil(Sq/64) x Hq = 8
        # CTAs on 36 SMs, so splitting K is what fills the GPU there: 32.05 -> 14.45
        # us and 18.68 -> 13.16 us per call at the production shape.
        _fa2.fwd_bf16_causal(qp, kp, vp, op, lp, lap, oap,
                             b, sq, sk, hq, hkv, d, qs, ks, vs, os_, scale,
                             _NSMS if split else 0, strm)
    elif split:
        _fa2.fwd_bf16(qp, kp, vp, op, lp, lap, oap,
                      b, sq, sk, hq, hkv, d, qs, ks, vs, os_, scale, _SPLITS, strm)
    else:
        _fa2.fwd_bf16(qp, kp, vp, op, lp, 0, 0,
                      b, sq, sk, hq, hkv, d, qs, ks, vs, os_, scale, 0, strm)


def _dense(q, k, v, scale, causal=False, split=False, seqused=None):
    """q/k/v are [B, H, S, D]; returns [B, Sq, Hq, D] (the layout all three
    call sites already want, so no transpose is added anywhere)."""
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1], k.shape[2]
    o = torch.empty(b, sq, hq, d, device=q.device, dtype=q.dtype)
    _fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
         _bshd_strides(q), _bshd_strides(k), _bshd_strides(v),
         (o.stride(0), o.stride(1), o.stride(2)),
         b, sq, sk, hq, hkv, d, q.device, scale, causal, split,
         seqused.data_ptr() if seqused is not None else 0)
    return o


# ------------------------------------------------------------- prefix segments
# The served prefix mask, characterised over strided real observations:
# it is identical for sliding_attention and full_attention layers (the 4096
# window is inert at S~570) and has 0 dead keys, because single-request
# inference never pads. Every row's allowed key set is a contiguous run from 0
# and the bound is monotone, so the mask carries nothing a native entry point
# cannot express -- five calls, no mask tensor:
#
#   q[  0: 41) -> keys[0: 41)  causal    41 text tokens
#   q[ 41:297) -> keys[0:297)  full     256 image-1 tokens (bidirectional block)
#   q[297:305) -> keys[0:305)  causal     8 text tokens
#   q[305:561) -> keys[0:561)  full     256 image-2 tokens
#   q[561:568) -> keys[0:568)  causal     7 text tokens
#
# That shape is a property of the prompt template, not of the kernel, so it is
# re-derived per predict rather than hard-coded, and the structural assumption
# is re-checked every time it is derived.
_SEGS: dict[tuple, tuple] = {}
_BY_LEN: dict[int, tuple] = {}
_GEN = [0]


def new_predict() -> None:
    """Invalidate the per-predict segment cache.

    Keying the cache on (data_ptr, shape) alone would be the `_fast_ms` bug
    again: the caching allocator hands back the same address with the same
    shape for the next predict's mask, and a stale segment table would then be
    applied to different content. The generation counter is what makes the key
    sound, so it has to be bumped by something outside the cache.
    """
    _GEN[0] += 1


def _segments_from_mask(mask: torch.Tensor):
    """Derive the segment table, and verify the structure rather than assume it.

    Deriving it requires one `.tolist()` of ~10 boundary values, which is a
    device->host sync and therefore illegal inside a CUDA graph capture. The
    sync is avoided under capture by a second cache keyed on prefix length
    alone, and that key is sound because it was measured to be: over all 444
    real observations the table takes exactly 5 distinct values, one per prefix
    length, each a pure shift of the first text block --

        S=567  q[0:40)C  q[40:296)F  q[296:304)C  q[304:560)F  q[560:567)C
        S=568  q[0:41)C  q[41:297)F  q[297:305)C  q[305:561)F  q[561:568)C
        S=569 / 571 / 572 likewise

    -- because the prompt template is fixed (two 256-token image blocks and an
    8/7-token text tail) and only the leading task text varies in length. That
    is a property of the corpus, not of the kernel, so it is re-validated on
    every non-capturing call rather than trusted: a mismatch raises instead of
    silently baking a stale launch geometry into a graph. The same reasoning
    licenses the other shape-dependent caches on this path: capture is always
    preceded by eager warmup on the same prompt.

    Outside capture this runs at most twice per predict (Gemma3 builds one mask
    per attention type, and collapses to one while the 4096 window is inert at
    S~570) and costs ~30 us against the 2.62 ms the decomposition saves.
    """
    key = (_GEN[0], mask.data_ptr(), tuple(mask.shape))
    got = _SEGS.get(key)
    if got is not None:
        return got
    s = int(mask.shape[-2])
    if torch.cuda.is_current_stream_capturing():
        got = _BY_LEN.get(s)
        if got is None:
            raise RuntimeError(
                f"flash_rt_fa2 prefix: CUDA graph capture at prefix_len={s} with no "
                f"validated segment table. Deriving one needs a host sync, which "
                f"capture forbids; the eager warmup that precedes capture should "
                f"have populated it.")
        return got
    allow = mask[0, 0] if mask.dtype == torch.bool else (mask[0, 0] == 0)
    idx = torch.arange(s, device=allow.device)
    # r[i] = 1 + (last allowed key of row i); n[i] = number of allowed keys.
    r = torch.where(allow, idx, torch.full_like(idx, -1)).amax(-1) + 1
    n = allow.sum(-1)
    ok = torch.stack([(n == r).all(), (r[1:] >= r[:-1]).all()])
    ok_l, rl = ok.tolist(), r.tolist()
    if not ok_l[0]:
        raise RuntimeError(
            "flash_rt_fa2 prefix: a row's visible keys are not a contiguous run from "
            "0, so this mask is not expressible as causal/full segments. Refusing "
            "to approximate it -- select attn_implementation='eager' or 'sdpa' "
            "for this model instead.")
    if not ok_l[1]:
        raise RuntimeError(
            "flash_rt_fa2 prefix: the per-row key bound is not monotone.")
    segs, i = [], 0
    while i < s:
        if rl[i] == i + 1:                                  # causal run
            j = i
            while j + 1 < s and rl[j + 1] == j + 2:
                j += 1
            segs.append((i, j + 1, rl[j], True))
        else:                                               # constant-bound run
            bound, j = rl[i], i
            while j + 1 < s and rl[j + 1] == bound:
                j += 1
            segs.append((i, j + 1, bound, False))
        i = j + 1
    segs = tuple(segs)
    prev = _BY_LEN.get(s)
    if prev is not None and prev != segs:
        raise RuntimeError(
            f"flash_rt_fa2 prefix: prefix_len={s} produced two different segment "
            f"tables,\n  before {prev}\n  now    {segs}\nThe length-keyed table "
            f"used during CUDA graph capture is therefore unsound for this "
            f"corpus. Disable CUDA graph capture for the text prefix.")
    _BY_LEN[s] = segs
    _SEGS.clear()          # at most 2 live entries; never let this grow
    _SEGS[key] = segs
    _LOG.debug("prefix S=%d segments=%s", s, " ".join(
        f"q[{a}:{b})->k[0:{c}){'C' if d else 'F'}" for a, b, c, d in segs))
    return segs


# --------------------------------------------------------------- custom ops
# Dynamo must not trace into these: they call an opaque binding and (for the
# prefix) do data-dependent work. Registering them as custom ops makes each one
# a single opaque node, so the LM/AE compiled graphs keep their shape instead of
# breaking 34 times -- a graph break per layer is exactly the Python dispatch
# the contract forbids.
@torch.library.custom_op("dm05::fa2_prefix", mutates_args=())
def fa2_prefix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: torch.Tensor | None, scale: float) -> torch.Tensor:
    b, hq, sq, d = q.shape
    if mask is None:
        # Never silently fall back to a dense call here. transformers' mask
        # pipeline returns None for any `_attn_implementation` it does not
        # recognise ("for TGI/vLLM backends... we don't need a mask!"), so a
        # missing registration in ALL_MASK_ATTENTION_FUNCTIONS looks exactly
        # like "bidirectional attention" -- and the first run of this module
        # did precisely that: 34 dense calls instead of 170 segment calls,
        # every text token attending its own future, and a latency number
        # that looked plausible. The DM05 prefix always contains causal text
        # spans, so no-mask is never correct at this site.
        raise RuntimeError(
            "flash_rt_fa2 prefix got attention_mask=None. transformers built no "
            "mask for this attention implementation -- register it in "
            "ALL_MASK_ATTENTION_FUNCTIONS (install() does this) before use.")
    out = torch.empty(b, sq, hq, d, device=q.device, dtype=q.dtype)
    hkv = k.shape[1]
    # Segment slicing is pointer arithmetic, not tensor slicing: 10 Python
    # slice objects per layer x 34 layers is real CPU at a 110 us kernel. The
    # strides are hoisted out of the loop for the same reason -- the binding
    # costs 6.7 us/call fully hoisted against 11.1 us recomputing them.
    qs, ks, vs = _bshd_strides(q), _bshd_strides(k), _bshd_strides(v)
    os_ = (out.stride(0), out.stride(1), out.stride(2))
    q_row, o_row, esz = q.stride(2), out.stride(1), q.element_size()
    qp, kp, vp, op = q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr()
    dev = q.device
    for s0, s1, kk, causal in _segments_from_mask(mask):
        # Causal (text) segments take the split-KV path; full (image) segments
        # already fill the GPU and gain nothing from splitting.
        _fwd(qp + s0 * q_row * esz, kp, vp, op + s0 * o_row * esz,
             qs, ks, vs, os_, b, s1 - s0, kk, hq, hkv, d, dev,
             scale, causal, causal and _PREFIX_SPLIT)
    return out


@fa2_prefix.register_fake
def _(q, k, v, mask, scale):
    b, hq, sq, d = q.shape
    return q.new_empty(b, sq, hq, d)


@torch.library.custom_op("dm05::fa2_suffix", mutates_args=())
def fa2_suffix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: torch.Tensor | None, scale: float) -> torch.Tensor:
    if mask is None:
        return _dense(q, k, v, scale, split=True)
    # Under the suffix CUDA graph the prefix is bucket-padded with `padding_idx`
    # and `_build_suffix_metadata` masks exactly that contiguous tail, so the
    # mask is a pure length and `seqused_k` is exact. The length is computed on
    # device (two tiny reductions, no sync) because a replayed graph has to
    # recompute it each time; reading it to the host would break capture.
    seqused = (mask[:, 0, 0] == 0).sum(-1).to(torch.int32)
    return _dense(q, k, v, scale, split=True, seqused=seqused)


@fa2_suffix.register_fake
def _(q, k, v, mask, scale):
    b, hq, sq, d = q.shape
    return q.new_empty(b, sq, hq, d)


@torch.library.custom_op("dm05::fa2_vision", mutates_args=())
def fa2_vision(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scale: float) -> torch.Tensor:
    return _dense(q, k, v, scale)


@fa2_vision.register_fake
def _(q, k, v, scale):
    b, hq, sq, d = q.shape
    return q.new_empty(b, sq, hq, d)


# -------------------------------------------------------- HF attention interface
def _prefix_forward(module, query, key, value, attention_mask, scaling=None,
                    dropout=0.0, **kw):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    if attention_mask is not None:
        attention_mask = attention_mask[..., : key.shape[-2]]
    return fa2_prefix(query, key, value, attention_mask, float(scaling)), None


def _vision_forward(module, query, key, value, attention_mask, scaling=None,
                    dropout=0.0, **kw):
    if attention_mask is not None:
        raise RuntimeError(
            "flash_rt_fa2 vision: SigLIP was given an attention mask. The served "
            "tower runs two full 1024-patch images with no padding and no mask; "
            "a mask here means the input shape changed and the dense call would "
            "silently attend to padding.")
    if scaling is None:
        scaling = module.head_dim ** -0.5
    return fa2_vision(query, key, value, float(scaling)), None


PREFIX_ATTN_NAME = "flash_rt_fa2_prefix"
VISION_ATTN_NAME = "flash_rt_fa2_vision"


def register_hf_attention(*, prefix: bool = True, vision: bool = False) -> None:
    """Register the FA2 forwards in the ``transformers`` attention registries.

    Must run before ``set_attn_implementation`` -- that call validates the
    requested name against ``ALL_ATTENTION_FUNCTIONS`` and would reject a name
    that is not registered yet.
    """
    _require()
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if prefix:
        # `.register()` (classmethod -> _global_mapping), NOT `[...] = `:
        # __setitem__ only writes the instance's _local_mapping, and
        # `_preprocess_mask_arguments` reads `_global_mapping` directly.
        ALL_ATTENTION_FUNCTIONS.register(PREFIX_ATTN_NAME, _prefix_forward)
        # Mandatory, not optional. `_preprocess_mask_arguments` returns a None
        # mask for any implementation missing from this registry, which would
        # turn the prefix into one dense bidirectional call. Binding to
        # `eager_mask` reproduces byte-for-byte the 4D additive mask the eager
        # path builds, which is the mask this module was characterised against.
        ALL_MASK_ATTENTION_FUNCTIONS.register(PREFIX_ATTN_NAME, eager_mask)
    if vision:
        ALL_ATTENTION_FUNCTIONS.register(VISION_ATTN_NAME, _vision_forward)


def suffix_attention(query_states, key_states, value_states, attention_mask, layer):
    """Drop-in for `DM05ActionExpert._attn_fn`, the sanctioned backend hook."""
    _require()
    return fa2_suffix(query_states, key_states, value_states, attention_mask,
                      float(layer.self_attn.scaling))


# ----------------------------------------------------------------------- install
def install(lm, vt=None, ae=None, sites: str = "prefix,suffix,vision"):
    """Point the requested sites at FA2. Returns the list actually installed."""
    _require()
    want = {s.strip() for s in sites.split(",") if s.strip()}
    unknown = want - {"prefix", "suffix", "vision"}
    if unknown:
        raise ValueError(f"flash_rt_fa2: unknown site(s) {sorted(unknown)}")
    register_hf_attention(prefix="prefix" in want, vision="vision" in want)
    done = []
    if "prefix" in want:
        lm.config._attn_implementation = PREFIX_ATTN_NAME
        for layer in lm.layers:
            layer.self_attn.config._attn_implementation = PREFIX_ATTN_NAME
        done.append("prefix")
    if "vision" in want:
        vt.vision_model.config._attn_implementation = VISION_ATTN_NAME
        for layer in vt.vision_model.encoder.layers:
            layer.self_attn.config._attn_implementation = VISION_ATTN_NAME
        done.append("vision")
    if "suffix" in want:
        ae._attn_fn = suffix_attention
        ae._suffix_attn_backend = "flash_rt_fa2"
        done.append("suffix")
    return done
