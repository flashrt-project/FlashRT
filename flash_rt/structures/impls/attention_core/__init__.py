import torch

from .fa2_seqused import (DenseAttention, PackedKVAttention,
                          bind_attention_core,
                          bind_dense_attention, plan_packed_kv)
from .two_way_fa2 import FactoredTwoWayAttention, bind_two_way_attention


def bind_dense_attention_best(captures):
    """Dense attention across the variant family, precision-descending.

    One structure, parallel executable forms, and no second hardware
    table: each variant's kernel package declares its own archs (or
    ships build variants only for the hosts it serves), so the device
    split *is* the refusal machinery. The order is precision first:
    FA2 (BF16, numerics preserving) binds wherever its runtime
    executes — every current receipt keeps its exact form. Then the
    FA4 CuTe DSL form (BF16, the SM100-family hot path), the
    allocation-free masked MHA (BF16/FP16, batch-of-one sites), and
    last the FP8 FA4 form. All are judged by the same downstream
    gates; a device none serves keeps the host's own attention.

    The winner carries the trail: ``_frt_variant`` names the bound
    form and ``_frt_variant_trail`` holds what each preferred variant
    said when it stepped aside. Without that, a host whose preferred
    package is merely absent looks identical to one where it was
    weighed and rejected — the two need different fixes, and only the
    trail tells them apart.
    """
    # a package can refuse a device three ways: its arch declaration
    # (ValueError from the loader's metadata check), the kernels
    # library finding no build variant for the host (OSError), or a
    # bind smoke the runtime cannot execute (RuntimeError) — each
    # means "not this variant here", never an error for the family
    def _fa2(caps):
        return bind_dense_attention(caps)

    def _fa4_cute(caps):
        from . import fa4_cute
        return fa4_cute.bind_dense_attention(caps)

    def _masked_mha(caps):
        from . import masked_mha
        return masked_mha.bind_dense_attention(caps)

    def _fa4_fp8(caps):
        from . import fa4_fp8
        return fa4_fp8.bind_dense_attention(caps)

    refusals, declined = [], 0
    for name, binder in (("fa2", _fa2), ("fa4_cute", _fa4_cute),
                         ("masked_mha", _masked_mha),
                         ("fa4_fp8", _fa4_fp8)):
        try:
            core = binder(captures)
        except (ValueError, RuntimeError, OSError) as refusal:
            refusals.append(f"{name}: {str(refusal)[:120]}")
            continue
        if core is not None and not _beats_host(core, captures[0]):
            refusals.append(
                f"{name}: bound but measured slower than the host "
                "attention at the captured shape — stepped aside")
            declined += 1
            core = None
        if core is not None:
            # the seam is served, but which variant served it and what
            # the preferred ones said are both load-bearing facts: a
            # host silently falling back to a lower-precision or slower
            # form is exactly the failure the ordering exists to make
            # visible, and it is invisible unless the superseded
            # refusals travel with the bound module
            core._frt_variant = name
            core._frt_variant_trail = tuple(refusals)
            return core
        declined += 1
        refusals.append(f"{name}: declined the captured shape form")
    if declined:
        # at least one variant executed its qualification and declined
        # the shape form — a site-level refusal the adapter records,
        # same contract as a single binder returning None
        return None
    raise ValueError(
        "attention_core: no variant serves this device — "
        + "; ".join(refusals))


def _beats_host(core, capture, margin: float = 0.02, iters: int = 20):
    """The family's speed gate: a variant seats only if it measures at
    least as fast as the host's own attention on the captured shape.
    Availability and precision order decide who gets weighed first;
    this decides whether the winner actually serves — bands are
    measured, not conceded, in this family too."""
    import torch.nn.functional as F

    query = capture.get("q")
    key = capture.get("key", capture.get("k"))
    value = capture.get("value", capture.get("v"))
    mask = capture.get("mask")
    if query is None or key is None or value is None:
        return True
    if not query.is_cuda:
        return True

    def _time(fn):
        with torch.no_grad():
            for _ in range(4):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    try:
        ours = _time(lambda: core(query, key, value))
        host = _time(lambda: F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask))
    except (RuntimeError, ValueError):
        return False
    return ours <= host * (1.0 + margin)


__all__ = ["DenseAttention", "PackedKVAttention",
           "bind_attention_core", "bind_dense_attention",
           "bind_dense_attention_best", "plan_packed_kv",
           "FactoredTwoWayAttention", "bind_two_way_attention"]
# variant modules (fa4_fp8, fa4_cute, masked_mha) import lazily inside
# the family binder: loading one must not require the others' runtimes
