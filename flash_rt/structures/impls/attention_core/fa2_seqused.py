"""attention_core — run attention on the FlashRT FA2 kernel.

A fused attention kernel wants contiguous keys and values. This module
provides two executable forms: a cadence-aware packed-KV form for decoder
loops, and a stateless dense form for hosts that supply complete Q/K/V on
every call. Both reduce supported masks to contiguous allowed key ranges
before dispatching one dense attention call.

The packed prefix (everything before the blocked run) belongs to a
slower cadence — it is the encoder's output for the current
observation, unchanged across the denoise loop — so it is filled once
at bind time and refreshed through an update function, exactly as
:mod:`..cadence_static` does for whole modules. The suffix is written
per call.

Three qualifications, all decided from real captures:

- ``head_dim`` must be one the kernel supports; otherwise the caller
  keeps its own path (``bind_attention_core`` returns ``None`` so the
  host can fall back to a community kernel rather than fail),
- every query row must see the same mask, and the blocked positions
  must form a single contiguous run — anything else is not expressible
  as a packed dense attention,
- the prefix keys must not move across the loop; a moving prefix means
  the split is wrong and the parity gate would catch it downstream, so
  it is rejected here where the reason is still legible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam

@lru_cache(maxsize=1)
def supported_head_dims() -> tuple[int, ...]:
    """Read the executable envelope from the installed Hub artifact."""
    package = hub_kernel("flashrt/fa2-seqused-runtime", ">=1")
    advertised = getattr(package, "SUPPORTED_HEAD_DIMS", None)
    if advertised is None:
        raise ValueError(
            "attention_core: FA2 Hub artifact does not advertise "
            "SUPPORTED_HEAD_DIMS; refusing to duplicate backend capability "
            "inside the structure layer")
    dims = tuple(sorted({int(dim) for dim in advertised}))
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(
            "attention_core: FA2 Hub artifact advertised invalid head dims")
    return dims


@dataclass
class _Scratch:
    """Output/LSE/workspace shared by same-shaped attention sites."""

    out: torch.Tensor
    lse: torch.Tensor
    workspace: object


@dataclass
class PackedKVPlan:
    """How the host's masked attention maps onto a dense one."""

    prefix: int
    suffix_start: int
    suffix_len: int
    seq_kv: int

    @property
    def packed(self) -> int:
        return self.prefix + self.suffix_len


def plan_packed_kv(mask: torch.Tensor | None, seq_kv: int) -> PackedKVPlan:
    """Derive the packing plan from one captured attention mask."""
    if mask is None:
        return PackedKVPlan(seq_kv, seq_kv, 0, seq_kv)
    if mask.dim() < 3:
        raise ValueError("attention_core: unexpected mask rank")
    rows = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0]
    if not bool((rows == rows[0]).all()):
        raise ValueError("attention_core: mask differs per query row")
    row = rows[0].float()
    blocked = ((row < -1e5) | row.isneginf()).nonzero().flatten()
    if blocked.numel() == 0:
        return PackedKVPlan(seq_kv, seq_kv, 0, seq_kv)
    runs, start, prev = [], None, None
    for i in blocked.tolist():
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    if len(runs) != 1:
        raise ValueError(
            f"attention_core: mask blocks {len(runs)} separate runs — "
            "not expressible as one packed dense attention")
    lo, hi = runs[0]
    return PackedKVPlan(lo, hi + 1, seq_kv - (hi + 1), seq_kv)


class PackedKVAttention(GuardedSeam, torch.nn.Module):
    """Attention over packed keys/values, on the FlashRT FA2 kernel.

    Holds one host module's packed buffers. Call it in place of the
    host's attention body; the prefix half is refreshed by the update
    function returned from :func:`bind_attention_core`.

    The packed region, the output and the split-KV workspace are all
    allocated for one query length, and this module is reached through a
    routed call rather than a module path, so there is no host module to
    revert to per call: a query outside the calibrated form raises. That
    is the whole reason to check — handing the kernel a scratch buffer
    sized for a different sequence is the failure that does not announce
    itself.
    """

    def __init__(self, plan: PackedKVPlan, q_shape, kv_heads: int,
                 dtype: torch.dtype, device, prefix_kv=None,
                 scratch: "_Scratch | None" = None):
        super().__init__()
        self.plan = plan
        b, heads, seq_q, head_dim = q_shape
        self.seq_q = seq_q
        # set by alias_suffix when a producer writes that side in place
        self._alias_k = False
        self._alias_v = False
        self._kfa = hub_kernel("flashrt/fa2-seqused-runtime", ">=1")
        self.register_buffer("packed_k", torch.zeros(
            b, plan.packed, kv_heads, head_dim, device=device,
            dtype=dtype))
        self.register_buffer("packed_v", torch.zeros_like(self.packed_k))
        if prefix_kv is not None:
            k0, v0 = prefix_kv
            self.packed_k[:, :plan.prefix] = k0
            self.packed_v[:, :plan.prefix] = v0
        # Output, LSE and split-KV workspace are scratch: each site
        # consumes its result before the next one runs, so sites with
        # the same shapes share one set. Per-site copies cost real
        # latency (pi05 r16: 18 private scratches ran 1.5% slower than
        # one shared set) and buy nothing.
        if scratch is None:
            q_sample = torch.empty(b, seq_q, heads, head_dim,
                                   device=device, dtype=dtype)
            out, lse = self._kfa.allocate_outputs(q_sample)
            scratch = _Scratch(out, lse, self._kfa.allocate_workspace(
                q_sample, self.packed_k))
        self._scratch = scratch
        self._frt_arm(dtypes=(dtype,), device=self.packed_k.device,
                      k=int(head_dim), rows=int(b * heads * seq_q))

    def forward(self, query, key, value, *, scale=None):
        """``query``/``key``/``value`` in the host's (B, H, S, D)."""
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:            # unreachable: no host path
            return admitted                    # to revert to, so it raises
        plan = self.plan
        q = query.transpose(1, 2).contiguous()
        if plan.suffix_len:
            self.packed_k[:, plan.prefix:].copy_(
                key.transpose(1, 2)[:, plan.suffix_start:])
            self.packed_v[:, plan.prefix:].copy_(
                value.transpose(1, 2)[:, plan.suffix_start:])
        sc = self._scratch
        return self._kfa.forward_static(
            q, self.packed_k, self.packed_v, out=sc.out,
            softmax_lse=sc.lse, workspace=sc.workspace,
            softmax_scale=scale)

    def alias_suffix(self, *, key: bool = False, value: bool = False):
        """Hand out the packed suffix rows for a producer to write into.

        The declarative form of what a hand-written runtime does when it
        gives the next stage a pointer: instead of the producer filling
        its own buffer and this module copying it in, the producer's
        output *is* the region.

        Only legal where nothing transforms the tensor between the two —
        a rotary embedding applied after the projection would leave the
        untransformed values here. Callers therefore alias key and value
        independently, and take ``None`` for whatever does not qualify.
        """
        plan = self.plan
        if not plan.suffix_len or self.packed_k.shape[0] != 1:
            return None, None          # only a single batch row slices
        regions = []                   # into a contiguous suffix
        for want, packed, flag in ((key, self.packed_k, "_alias_k"),
                                   (value, self.packed_v, "_alias_v")):
            if not want:
                regions.append(None)
                continue
            region = packed[0, plan.prefix:]
            if not region.is_contiguous():
                regions.append(None)
                continue
            setattr(self, flag, True)
            regions.append(region)
        return tuple(regions)

    def forward_suffix(self, query, key, value, *, scale=None):
        """Kernel layout (B, S, H, D), with **suffix-only** keys/values.

        Two things separate this from :meth:`forward`. The layout: the
        host builds (B, H, S, D) because that is what eager SDPA wants
        and :meth:`forward` transposes it back — two cancelling
        transposes plus the copies that make each of them contiguous. A
        caller that owns the whole attention sublayer never builds the
        host layout at all; the projections' own output view *is* this
        one. And the extent: the host hands attention the full
        cache-concatenated KV and the suffix gets sliced back out here,
        while a sublayer's projections produce exactly the new tokens —
        which is the suffix already. Binding checks that equality rather
        than assuming it.
        """
        # the contract is the same one ``forward`` checks: it counts rows
        # against the head dim, which is what both layouts agree on
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        plan = self.plan
        q = query if query.is_contiguous() else query.contiguous()
        if plan.suffix_len:
            # an aliased side already wrote itself here
            if not self._alias_k:
                self.packed_k[:, plan.prefix:].copy_(key)
            if not self._alias_v:
                self.packed_v[:, plan.prefix:].copy_(value)
        sc = self._scratch
        return self._kfa.forward_static(
            q, self.packed_k, self.packed_v, out=sc.out,
            softmax_lse=sc.lse, workspace=sc.workspace,
            softmax_scale=scale)


class DenseAttention(GuardedSeam, torch.nn.Module):
    """FA2 replacement for an ordinary dense SDPA call.

    Unlike :class:`PackedKVAttention`, this form owns no observation-cadence
    state.  Every call receives the complete Q/K/V tensors, so it is suitable
    for Diffusers self- and cross-attention and remains correct when the
    conditioning changes between graph replays.

    Inputs and outputs use the host SDPA layout ``[B, H, S, D]``.  The package
    consumes ``[B, S, H, D]``; reversing the host's projection view is normally
    already contiguous and therefore does not materialise a transpose.
    """

    def __init__(
        self, q_shape, kv_shape, dtype: torch.dtype, device,
        allowed_ranges=None, scratch: "_Scratch | None" = None,
    ):
        super().__init__()
        b, heads, seq_q, head_dim = q_shape
        kb, kv_heads, seq_kv, kv_dim = kv_shape
        if kb != b or kv_dim != head_dim:
            raise ValueError(
                "attention_core dense: Q and KV batch/head dimensions differ")
        if heads % kv_heads:
            raise ValueError(
                "attention_core dense: query heads must be divisible by "
                "KV heads")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        self.allowed_ranges = tuple(allowed_ranges or ())
        self._kfa = hub_kernel("flashrt/fa2-seqused-runtime", ">=1")
        q_sample = torch.empty(
            b, seq_q, heads, head_dim, device=device, dtype=dtype)
        packed_seq = (
            sum(hi - lo for lo, hi in self.allowed_ranges)
            if self.allowed_ranges else seq_kv)
        kv_sample = torch.empty(
            b, packed_seq, kv_heads, head_dim, device=device, dtype=dtype)
        if self.allowed_ranges:
            self.register_buffer("packed_k", torch.empty_like(kv_sample))
            self.register_buffer("packed_v", torch.empty_like(kv_sample))
        if scratch is None:
            out, lse = self._kfa.allocate_outputs(q_sample)
            scratch = _Scratch(
                out, lse, self._kfa.allocate_workspace(q_sample, kv_sample))
        elif (
            scratch.out.shape != q_sample.shape
            or scratch.out.dtype != dtype
            or scratch.out.device != q_sample.device
            or scratch.lse.shape != (b, heads, seq_q)
            or scratch.lse.device != q_sample.device
        ):
            raise ValueError(
                "attention_core dense: shared scratch does not match "
                "the bound attention form"
            )
        self._scratch = scratch
        self._frt_arm(
            dtypes=(dtype,), device=q_sample.device, k=int(head_dim),
            rows=int(b * heads * seq_q))

    def forward(self, query, key, value, *, scale=None):
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        if tuple(query.shape) != self.q_shape:
            raise ValueError(
                "attention_core dense: query shape moved from "
                f"{self.q_shape} to {tuple(query.shape)}")
        if tuple(key.shape) != self.kv_shape or value.shape != key.shape:
            raise ValueError(
                "attention_core dense: key/value shape moved from "
                f"{self.kv_shape} to {tuple(key.shape)}/{tuple(value.shape)}")
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        if not q.is_contiguous():
            q = q.contiguous()
        if self.allowed_ranges:
            offset = 0
            for lo, hi in self.allowed_ranges:
                length = hi - lo
                self.packed_k[:, offset:offset + length].copy_(k[:, lo:hi])
                self.packed_v[:, offset:offset + length].copy_(v[:, lo:hi])
                offset += length
            k, v = self.packed_k, self.packed_v
        else:
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()
        sc = self._scratch
        out = self._kfa.forward_static(
            q, k, v, out=sc.out, softmax_lse=sc.lse,
            workspace=sc.workspace, softmax_scale=scale)
        return out.transpose(1, 2)


def _allowed_ranges(mask):
    if mask is None:
        return ()
    rows = mask.reshape(-1, mask.shape[-1])
    first = rows[0]
    if mask.dtype == torch.bool:
        allowed = first
        if not bool((rows == first).all()):
            raise ValueError(
                "attention_core dense: mask differs per query row")
    else:
        allowed = ~((first.float() < -1e5) | first.float().isneginf())
        other = ~((rows.float() < -1e5) | rows.float().isneginf())
        if not bool((other == allowed).all()):
            raise ValueError(
                "attention_core dense: mask differs per query row")
    indices = allowed.nonzero().flatten().tolist()
    if not indices:
        raise ValueError("attention_core dense: mask permits no keys")
    ranges = []
    for index in indices:
        if not ranges or index != ranges[-1][1]:
            ranges.append([index, index + 1])
        else:
            ranges[-1][1] += 1
    if len(ranges) > 8:
        # the packed-copy loop is linear in segments; past a handful
        # the copies outweigh the masked-out keys and the masked
        # executable form serves better
        return None
    if len(ranges) == 1 and ranges[0] == [0, mask.shape[-1]]:
        return ()
    return tuple((lo, hi) for lo, hi in ranges)


def bind_dense_attention(captures):
    """Bind one stateless dense FA2 core from repeated host captures."""
    if not captures:
        raise ValueError("attention_core dense: no captures")
    first = captures[0]
    query, key, value = first["q"], first["key"], first["value"]
    head_dim = query.shape[-1]
    if head_dim not in supported_head_dims():
        return None
    allowed_ranges = _allowed_ranges(first.get("mask"))
    if allowed_ranges is None:
        return None
    expected = (tuple(query.shape), tuple(key.shape), tuple(value.shape),
                query.dtype, key.dtype, value.dtype)
    for capture in captures[1:]:
        got = (
            tuple(capture["q"].shape),
            tuple(capture["key"].shape),
            tuple(capture["value"].shape),
            capture["q"].dtype,
            capture["key"].dtype,
            capture["value"].dtype,
        )
        if got != expected:
            raise ValueError(
                "attention_core dense: shape, dtype, or mask moved within "
                f"one calibration call: {expected} -> {got}")
        if _allowed_ranges(capture.get("mask")) != allowed_ranges:
            raise ValueError(
                "attention_core dense: mask pattern moved within one "
                "calibration call")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("attention_core dense: Q/K/V dtypes differ")
    return DenseAttention(
        query.shape, key.shape, query.dtype, query.device,
        allowed_ranges=allowed_ranges)


def bind_attention_core(captures, *, prefix_static_rtol: float = 1e-3):
    """Bind one packed-KV attention per site from real captures.

    ``captures`` is a sequence of per-site dicts holding ``q`` (one
    captured query, host layout), ``keys``/``values`` (the tensors that
    site produced across the hot loop, oldest first) and ``mask``.
    Returns ``(modules, update)``, or ``None`` when the head dim is
    unsupported so the caller can keep a fallback path.
    """
    if not captures:
        raise ValueError("attention_core: no captures")
    head_dim = captures[0]["q"].shape[-1]
    if head_dim not in supported_head_dims():
        return None

    modules, scratch = [], None
    for site, cap in enumerate(captures):
        keys = cap["keys"]
        plan = plan_packed_kv(cap.get("mask"), keys[0].shape[2])
        first = keys[0][:, :, :plan.prefix]
        for other in keys[1:]:
            if not torch.allclose(first, other[:, :, :plan.prefix],
                                  rtol=prefix_static_rtol,
                                  atol=prefix_static_rtol):
                raise ValueError(
                    f"attention_core: site {site} prefix keys move "
                    "across the loop — the cadence split is wrong")
        q = cap["q"]
        core = PackedKVAttention(
            plan, q.shape, keys[0].shape[1], q.dtype, q.device,
            prefix_kv=(keys[0].transpose(1, 2)[:, :plan.prefix],
                       cap["values"][0].transpose(1, 2)[:, :plan.prefix]),
            scratch=scratch)
        scratch = scratch or core._scratch
        modules.append(core)

    def update(fresh_kv) -> None:
        """Refresh every site's prefix from freshly computed K/V."""
        with torch.no_grad():
            for mod, (k, v) in zip(modules, fresh_kv):
                p = mod.plan.prefix
                mod.packed_k[:, :p].copy_(k.transpose(1, 2)[:, :p])
                mod.packed_v[:, :p].copy_(v.transpose(1, 2)[:, :p])

    return modules, update
