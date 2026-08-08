"""qkv_pack — pack sibling linears that share one input into one GEMM.

Sibling projections consumed in a fixed call order (q/k/v of an
attention block, gate/up of an MLP) each pay a small-M GEMM whose cost
is launch/latency floor, not bandwidth. Packing their weights into one
``[sum(N_i), K]`` matrix turns the group into a single GEMM; the later
siblings become buffer reads. Two bind forms cover the hosts seen so
far:

- **leaf**: the host calls the sibling modules separately and there is
  no enclosing attention-module boundary. The first sibling's slot gets
  a :class:`PackedLinear` (runs the packed GEMM, writes the other
  outputs into preallocated buffers); the later slots get
  :class:`StashReader` (return the buffer). The host's own call order
  is the data dependency — functionalization keeps copy/read ordered
  inside compiled and captured graphs.
- **module**: the host has an attention module with
  ``q_proj/k_proj/v_proj/out_proj`` attributes and a standard
  projections → attention → out_proj forward. :class:`AttnBlockPacked`
  replaces the whole module: packed GEMM, SDPA at a declared compute
  dtype, original ``out_proj``.

Both forms quantize the packed weight to FP8 with one joint per-tensor
scale (the joint-scale rounding difference is covered by the parity
gate). Inputs enter either as FP8 (a producer seam supplies the shared
``act_scale``) or as BF16 through the fused-quantize entry with a
calibrated ``act_scale``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .. import hub_kernel
from ...workspace import lease
from ...guard import CAST_OK, FP8_ONLY, PROCEED, GuardRefused, GuardedSeam

_FP8 = torch.float8_e4m3fn


def _all_zero(t: torch.Tensor) -> bool:
    return not bool(t.any())


def _pack_weights(mods: Sequence[torch.nn.Module]):
    ws = [m.weight.detach() for m in mods]
    w = torch.cat(ws, 0)
    scale = (w.float().abs().max() / 448.0).clamp(min=1e-8).view(1)
    w8 = (w.float() / scale).clamp(-448, 448).to(_FP8)
    splits = [wi.shape[0] for wi in ws]
    biases = []
    for m in mods:
        b = getattr(m, "bias", None)
        biases.append(b.detach().to(torch.bfloat16) if b is not None
                      else torch.zeros(m.weight.shape[0],
                                       device=w.device,
                                       dtype=torch.bfloat16))
    return w8, scale, torch.cat(biases), splits


class PackedLinear(GuardedSeam, torch.nn.Module):
    """Leaf-form head: one packed GEMM, later siblings stashed.

    The stash and quantize buffers are allocated once at the largest row
    count observed during calibration. The Hub entry accepts a logical M
    no larger than those buffers and returns only the logical rows, so the
    contract is a row *capacity*, not one exact row count. Calls above the
    capacity fall back before they can hand the kernel a short buffer.
    """

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, mods: Sequence[torch.nn.Module],
                 act_scale: torch.Tensor, rows: int,
                 in_dtype: str = "fp8_static", joint_slots: int = 0):
        super().__init__()
        self.host_linear = mods[0]
        kf = hub_kernel("flashrt/flashrt-fp8-ffn", ">=1")
        self.in_dtype = in_dtype
        w8, w_scale, bias, splits = _pack_weights(mods)
        self.splits = splits
        self.rows = rows
        self.register_buffer("w8", w8)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("bias_cat", bias)
        self.register_buffer("act_scale", act_scale)
        dev = w8.device
        for i, n in enumerate(splits[1:], 1):
            setattr(self, f"stash{i}", lease(
                (rows, n), torch.bfloat16, dev,
                tag=f"qkv_stash{i}"))
        # a bias-add is its own kernel. Hosts whose projections carry no
        # bias (the whole Gemma family) would otherwise pay a launch per
        # call to add zeros — measured 3 kernels/call with the bias
        # entry against 1 without it at the same shapes.
        self.no_bias = _all_zero(bias)
        # A caller that consumes the first `joint_slots` siblings
        # together takes the packed output whole: they are one
        # contiguous run in it, and splitting them apart only to apply
        # the same elementwise transform to each half costs a kernel and
        # two copies. Zero keeps the sibling-by-sibling contract.
        self.joint_slots = joint_slots
        if joint_slots:
            self.packed = lease((rows, sum(splits)), torch.bfloat16,
                                w8.device, tag="qkv_joint")
        if in_dtype == "fp8_static":
            self._fn = (kf.fp8_gemm_bf16 if self.no_bias
                        else kf.fp8_linear_bias_bf16)
        else:
            self._fn = kf.bf16_fp8_linear_bias_bf16
            k = mods[0].weight.shape[1]
            # Call-lifetime scratch, so the pool owns it: the quantize
            # scratch is written and read inside the kernel call, and
            # the packed output is fully consumed before forward returns
            # (q sliced out by copy, later siblings copied to stashes).
            # Layers run sequentially, so every same-shape pack shares
            # one allocation instead of paying ~900 MiB per layer — the
            # difference between binding a 52-layer host and refusing
            # most of it on budget.
            self.x8_buf = lease((rows, k), _FP8, dev, tag="qkv_x8")
            self.y_buf = lease((rows, sum(splits)), torch.bfloat16, dev,
                               tag="qkv_y")
        self._frt_arm(
            dtypes=FP8_ONLY if in_dtype == "fp8_static" else CAST_OK,
            device=dev, k=int(mods[0].weight.shape[1]),
            row_capacity=rows)

    def alias_stash(self, index: int, region: torch.Tensor) -> None:
        """Write sibling ``index`` straight into a buffer someone else owns.

        The stash exists because the later siblings' outputs have to live
        somewhere until the host asks for them. When the consumer of that
        output already owns a region of the right shape, that region can
        *be* the stash and the consumer's own copy disappears — one
        buffer instead of two, which is the join the two structures could
        never see from inside either of them.
        """
        if not 1 <= index < len(self.splits):
            raise ValueError(f"qkv_pack: no sibling {index} to alias")
        want = (self.rows, self.splits[index])
        if region.dtype is not torch.bfloat16:
            raise ValueError(
                f"qkv_pack: aliased region is {region.dtype}, the packed "
                "output is bfloat16")
        # An alias has to be checked for actually aliasing, not for a
        # property that usually comes with it. Two ways to lose it, both
        # silent: reshape *copies* when it cannot view, leaving a
        # detached buffer that looks right and is connected to nothing;
        # and a view that does succeed can still be strided, so the
        # writes would land on every other row of the consumer's region.
        try:
            buf = region.view(want)
        except RuntimeError as exc:
            raise ValueError(
                f"qkv_pack: aliased region is not viewable at {want} "
                "without a copy") from exc
        if buf.data_ptr() != region.data_ptr():
            raise ValueError(
                "qkv_pack: aliased view does not start at the region")
        if not buf.is_contiguous():
            raise ValueError(
                f"qkv_pack: aliased region is strided at {want} — the "
                "stash write would skip rows of the consumer's buffer")
        setattr(self, f"stash{index}", buf)

    def enable_joint(self, slots: int) -> None:
        """Let a caller take the first ``slots`` siblings together.

        Enabled after binding, because whether anyone consumes them
        jointly is a property of the composition around this pack, not
        of the pack. Refused when the siblings do not divide evenly by
        the head dim they would be viewed at — then they are not one
        run of equal-width heads and the caller cannot treat them alike.
        """
        if not 2 <= slots <= len(self.splits):
            raise ValueError(f"qkv_pack: cannot join {slots} sibling(s)")
        self.joint_slots = slots
        if not hasattr(self, "packed"):
            self.packed = lease((self.rows, sum(self.splits)),
                                torch.bfloat16, self.w8.device,
                                tag="qkv_joint")

    def disable_joint(self) -> None:
        """Restore the sibling-by-sibling stash contract."""
        self.joint_slots = 0

    def joint(self, x):
        """Run the pack and return the first ``joint_slots`` siblings whole.

        They are one contiguous run of the packed output, so a caller
        that applies the same transform to all of them (a rotary
        embedding over q and k, whose head dims match by construction)
        can do it in one pass instead of splitting them apart first.
        """
        if not self.joint_slots:
            raise ValueError("qkv_pack: this pack has no joint slots")
        flat = x.reshape(-1, x.shape[-1])
        if not torch.compiler.is_compiling():
            reason = self._frt_guard.admit(flat)
            if reason is not None:
                # Joint consumption has no q-only host fallback: all sibling
                # projections have already been claimed by the composition.
                # Still write the refusal into the ordinary seam ledger.
                self._frt_guard.refuse(reason)
                raise GuardRefused(f"qkv_pack: joint refused — {reason}")
        self._run(flat, stash_all=False)
        width = sum(self.splits[:self.joint_slots])
        return self.packed[:flat.shape[0], :width]

    def _run(self, flat, stash_all: bool = True):
        logical_rows = flat.shape[0]
        out = (self.packed[:logical_rows]
               if self.joint_slots and self.in_dtype == "fp8_static"
               else self.packed if self.joint_slots else None)
        if self.in_dtype == "fp8_static":
            y = (self._fn(flat, self.w8, self.act_scale, self.w_scale,
                          out=out)
                 if self.no_bias else
                 self._fn(flat, self.w8, self.bias_cat, self.act_scale,
                          self.w_scale, out=out))
        else:
            y = self._fn(flat.to(torch.bfloat16).contiguous(),
                         self.w8, self.bias_cat, self.act_scale,
                         self.w_scale, input_fp8=self.x8_buf,
                         out=out if out is not None else self.y_buf)
        # siblings the caller takes jointly are read straight out of the
        # packed buffer; only the rest need stashing. A plain forward
        # always stashes — a host-form call on a module whose joint
        # consumer is enabled but not routed must leave fresh stashes,
        # not silently stale ones (measured cos ~1e-5 on the sibling
        # read when this was skipped).
        if not torch.compiler.is_compiling():
            self._stash_epoch = getattr(self, "_stash_epoch", 0) + 1
        skip = 0 if stash_all else self.joint_slots
        off = sum(self.splits[:max(1, skip)]) if skip else self.splits[0]
        for i, n in enumerate(self.splits[1:], 1):
            if i < skip:
                continue
            getattr(self, f"stash{i}")[:logical_rows].copy_(
                y[:, off:off + n])
            if not torch.compiler.is_compiling():
                epochs = getattr(self, "_stash_epochs", None)
                if epochs is None:
                    epochs = {}
                    self._stash_epochs = epochs
                epochs[i] = self._stash_epoch
            off += n
        return y

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        flat = x.reshape(-1, x.shape[-1])
        y = self._run(flat)
        out = y[:, :self.splits[0]].contiguous()
        out = out.reshape(*x.shape[:-1], self.splits[0])
        # the kernel's output dtype is BF16 by contract; only cast back
        # when the host boundary itself is a compute dtype. On the
        # fp8_static entry the input is FP8 (a producer seam supplies
        # it) and casting to it would hand FP8 activations to the
        # host's next op.
        return out if x.dtype is _FP8 else out.to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


class StashReader(GuardedSeam, torch.nn.Module):
    """Leaf-form tail: return the packed head's stashed output.

    Shares the head's contract, because it shares the head's input: the
    host hands the same activation to every sibling, so head and tails
    admit or refuse a call together and the group never half-runs.
    """

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True
    # Its value is valid only after the packed head ran on the same input.
    # A slower-cadence updater calls one projection independently, so using
    # this replacement there would refresh from a previous sibling call.
    _frt_requires_sibling_order = True

    def __init__(self, orig: torch.nn.Module, packed: PackedLinear,
                 index: int):
        super().__init__()
        self.host_linear = orig
        self._packed = (packed,)
        self.index = index
        head = packed._frt_guard
        self._frt_arm(dtypes=head.dtypes, device=head.device, k=head.k,
                      row_capacity=head.row_capacity)

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        head = self._packed[0]
        if not torch.compiler.is_compiling():
            epoch = getattr(head, "_stash_epoch", 0)
            written = getattr(head, "_stash_epochs", {}).get(self.index)
            if epoch and written != epoch:
                raise GuardRefused(
                    "qkv_pack: sibling stash is stale — the head's last "
                    "run did not write this slot (a joint consumer "
                    "skipped it); reading it would be silently wrong")
        logical_rows = x.numel() // x.shape[-1]
        buf = getattr(self._packed[0], f"stash{self.index}")[:logical_rows]
        out = buf.reshape(*x.shape[:-1], buf.shape[-1])
        return out if x.dtype is _FP8 else out.to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


def bind_qkv_pack(mods: Sequence[torch.nn.Module],
                  act_scale: torch.Tensor, rows: int,
                  in_dtype: str = "fp8_static"):
    """Bind a sibling group; returns replacements in sibling order.

    Each host weight is checkpoint-native ``[out_features, in_features]``;
    the binder concatenates along the output axis and packs once. ``rows``
    is the preallocated row capacity, not an exact runtime M.
    """
    if len(mods) < 2:
        raise ValueError("qkv_pack: need at least two siblings")
    kdims = {m.weight.shape[1] for m in mods}
    if len(kdims) != 1:
        raise ValueError(f"qkv_pack: sibling K dims differ {kdims}")
    packed = PackedLinear(mods, act_scale, rows, in_dtype=in_dtype)
    out = [packed]
    for i, m in enumerate(mods[1:], 1):
        out.append(StashReader(m, packed, i))
    return out


class AttnBlockPacked(GuardedSeam, torch.nn.Module):
    """Module-form: packed QKV + SDPA at a declared dtype + out_proj.

    Fits attention modules exposing ``q_proj/k_proj/v_proj/out_proj``,
    ``head_dim`` and ``scale`` with the standard block forward
    (SigLIP/CLIP-family vision towers and friends).
    """

    _frt_host_attr = "host_attn"
    _frt_can_fallback = True

    def __init__(self, orig: torch.nn.Module, act_scale: torch.Tensor,
                 rows: int, sdpa_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.host_attn = orig
        kf = hub_kernel("flashrt/flashrt-fp8-ffn", ">=1")
        self._fn = kf.bf16_fp8_linear_bias_bf16
        w8, w_scale, bias, splits = _pack_weights(
            [orig.q_proj, orig.k_proj, orig.v_proj])
        if len(set(splits)) != 1:
            raise ValueError("attn_block: q/k/v widths differ")
        self.e = splits[0]
        self.register_buffer("w8", w8)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("bias_cat", bias)
        self.register_buffer("in_scale", act_scale)
        k = orig.q_proj.weight.shape[1]
        dev = w8.device
        self.register_buffer("x8_buf", torch.empty(
            rows, k, device=dev, dtype=_FP8))
        self.register_buffer("y_buf", torch.empty(
            rows, 3 * self.e, device=dev, dtype=torch.bfloat16))
        self.sdpa_dtype = sdpa_dtype
        self._frt_arm(dtypes=CAST_OK, device=dev, k=int(k),
                      row_capacity=rows)

    def forward(self, hidden_states, attention_mask=None, **kw):
        admitted = self._frt_admit(hidden_states, attention_mask, **kw)
        if admitted is not PROCEED:
            return admitted
        a = self.host_attn
        bsz, seq, dim = hidden_states.shape
        flat = hidden_states.reshape(-1, dim).to(
            torch.bfloat16).contiguous()
        y = self._fn(flat, self.w8, self.bias_cat, self.in_scale,
                     self.w_scale, input_fp8=self.x8_buf,
                     out=self.y_buf)
        hd = a.head_dim

        def split(t):
            return t.contiguous().view(bsz, seq, -1, hd).transpose(
                1, 2).to(self.sdpa_dtype)

        e = self.e
        mask = (attention_mask.to(self.sdpa_dtype)
                if attention_mask is not None else None)
        o = torch.nn.functional.scaled_dot_product_attention(
            split(y[:, :e]), split(y[:, e:2 * e]), split(y[:, 2 * e:]),
            attn_mask=mask, scale=a.scale)
        o = o.to(hidden_states.dtype).transpose(1, 2).reshape(
            bsz, seq, dim).contiguous()
        return a.out_proj(o), None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_attn"), name)


def bind_attn_block(orig: torch.nn.Module, act_scale: torch.Tensor,
                    rows: int,
                    sdpa_dtype: torch.dtype = torch.bfloat16
                    ) -> AttnBlockPacked:
    """Bind the module form with checkpoint-native Q/K/V weights.

    ``orig.{q,k,v}_proj.weight`` are ``[out_features, in_features]`` and
    ``rows`` is the maximum logical row count covered by the preallocated
    quantize/output buffers.
    """
    for attr in ("q_proj", "k_proj", "v_proj", "out_proj", "head_dim",
                 "scale"):
        if not hasattr(orig, attr):
            raise ValueError(f"attn_block: host lacks {attr!r}")
    return AttnBlockPacked(orig, act_scale, rows, sdpa_dtype=sdpa_dtype)
