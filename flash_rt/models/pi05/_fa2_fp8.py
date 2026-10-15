"""FA2 causal attention with an FP8 attention-output epilogue.

The prefix o_proj reads a static FP8 copy of the attention output. Stock FA2 writes
only bf16, so the pipeline would otherwise pay a separate quantize pass over every
layer's output. ``flash_rt_fa2``'s ``fwd_bf16_causal_fp8out`` / ``fwd_bf16_fp8out``
write that copy from the same registers that feed the bf16 store, and can skip the
bf16 store where the FP8 copy is the only consumer (FlashRT local patch 4; see
``csrc/attention/flash_attn_2_src/flash_attn/flash_fwd_kernel.h``).

This module is the adapter between those two entries and the call shape the prefix
uses -- one ``fwd_bf16(..., causal)`` instead of a causal and a noncausal name. The
arithmetic is the shipped kernel's; there is no second FA2 build here.

An earlier revision of this file vendored a patched copy of the FA2 headers under
``csrc/fa2_fp8`` and JIT-compiled it. The two implementations were measured
bit-identical -- bf16 output, ``lse`` and the FP8 copy all exact -- across every
prefix configuration (causal and noncausal, split and unsplit, at both the 630-row
production segment and the 41-row opening segment), and both reproduce a standalone
``quantize_fp8_static`` bit for bit. The duplicate is gone; the shipped entries are
the only path, and ``tests/test_fa2_fp8_output_epilogue.py`` covers them.

``load()`` returns None when those entries are absent, and the caller keeps the
stock FA2 + quantize path.
"""
from __future__ import annotations

#: False = not attempted, None = unavailable, otherwise the adapter below.
_EXT = False


class _BuiltinEpilogue:
    """Presents the two shipped FA2 entries as the one ``fwd_bf16`` the prefix calls."""

    def __init__(self, fa2):
        self._fa2 = fa2

    def fwd_bf16(self, q, k, v, o, lse, lse_a, o_a, b, sq, sk, nhq, nhkv, hd,
                 qs, ks, vs, os_, scale, num_sms, stream,
                 o_fp8, o_fp8_scale, o8s, skip_bf16, causal):
        fn = (self._fa2.fwd_bf16_causal_fp8out if causal
              else self._fa2.fwd_bf16_fp8out)
        fn(q, k, v, o, lse, lse_a, o_a, b, sq, sk, nhq, nhkv, hd,
           qs, ks, vs, os_, o_fp8, o_fp8_scale, o8s, skip_bf16,
           scale, num_sms, stream)


def load():
    """The FP8-epilogue extension, or None when this build does not carry it."""
    global _EXT
    if _EXT is not False:
        return _EXT
    _EXT = None
    try:
        from flash_rt import flash_rt_fa2 as fa2
    except ImportError:
        return _EXT
    if hasattr(fa2, "fwd_bf16_causal_fp8out") and hasattr(fa2, "fwd_bf16_fp8out"):
        _EXT = _BuiltinEpilogue(fa2)
    return _EXT
