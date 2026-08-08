"""decoder_block — the pre-norm transformer block as one boundary.

Every region structure in this library binds inside a block: the norm,
the packed projections, the attention, the MLP. What none of them can
see is the *dataflow between them* — and that is where the residual
adds, the gate broadcasts and the dtype round trips live. A host block
is a fixed shape:

    r = h;  h, g = norm_in(h, cond);   h = attn(h);  h = r + h * g
    r = h;  h, g = norm_out(h, cond);  h = mlp(h);   h = r + h * g

so the second norm's real input is not a hidden state, it is a pending
``residual + attn_out * gate``. The adaptive-norm kernel already takes
exactly that (it computes the gated residual, norms it, modulates and
quantizes in one pass) — bound at the norm boundary there is nothing to
hand it, so the residual argument gets zeros, the host keeps its own
elementwise add, and the fused producer looks like dead weight. It was
measured as such and refused. The boundary was wrong, not the kernel.

Binding the block puts the residual back in the producer's hands. Three
things follow, none of which is available one seam at a time:

- the pending residual add disappears into the producer's kernel;
- the producer emits FP8, so the MLP takes the FP8 entry and its own
  input quantization goes away;
- the step lookup is resolved once and shared by both producers instead
  of being recomputed per norm.

The block owns composition, not kernels: every slot is a module bound by
its own structure, and a slot that did not bind keeps the host's child.
It can therefore only add to what the region structures already do.
"""

from __future__ import annotations

import torch
from torch import nn

from ...guard import CAST_OK, PROCEED, GuardedSeam

_BLOCK_ATTRS = ("self_attn", "mlp", "input_layernorm",
                "post_attention_layernorm")


def _parameterised_children(module: nn.Module) -> set[str]:
    return {name for name, child in module.named_children()
            if any(True for _ in child.parameters())}


class FusedDecoderBlock(GuardedSeam, nn.Module):
    """Pre-norm block whose sublayer dataflow runs through the producers.

    The host block is retained whole, which makes this the widest way back
    in the library: a call outside the calibrated form runs the entire
    original block. That is exact rather than approximate, because the
    block never swapped the host's own children — it holds separately
    bound copies and leaves the host's sublayers where they were.
    """

    _frt_host_attr = "host"
    _frt_can_fallback = True

    def __init__(self, host: nn.Module, producer_in, producer_out,
                 ffn: nn.Module, *, cond_kw: str = "adarms_cond",
                 returns_tuple: bool = False, attn=None):
        super().__init__()
        self.host = host
        self.producer_in = producer_in
        self.producer_out = producer_out
        self.ffn = ffn
        self.cond_kw = cond_kw
        self.returns_tuple = returns_tuple
        # an attention sublayer that owns its own layout, or None to keep
        # the host's attention module (which owns the layout itself)
        self.attn = attn
        self.own_attn = attn is not None
        rows, dim = producer_in.resid.shape
        self._frt_arm(dtypes=CAST_OK, device=producer_in.resid.device,
                      k=int(dim), rows=int(rows))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # before the conditioning is taken out of kwargs: the host block
        # expects its own signature back if this call has to go to it
        admitted = self._frt_admit(hidden_states, *args, **kwargs)
        if admitted is not PROCEED:
            return admitted
        cond = kwargs.pop(self.cond_kw, None)
        idx = self.producer_in.resolve(cond)

        y, gate = self.producer_in.produce(hidden_states, idx)
        y = y.reshape(hidden_states.shape)
        if self.own_attn:
            attn_out = self.attn(
                y, position_embeddings=kwargs.get("position_embeddings"))
        else:
            attn_out = self.host.self_attn(y, *args, **kwargs)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]

        # the pending "hidden_states + attn_out * gate" is the second
        # norm's input; the kernel takes it whole
        resid, y, gate = self.producer_out.absorb(
            hidden_states, attn_out, gate, idx)

        out = resid + self.ffn(y) * gate
        out = out.reshape(hidden_states.shape)
        return (out,) if self.returns_tuple else out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host"), name)


def qualify(host: nn.Module) -> None:
    """Refuse a host block whose dataflow this structure does not model.

    The check is structural, not behavioural: the four sublayer slots
    must be there and nothing else that carries weights may be, because
    a fifth parameterised child (a second pair of norms, a cross
    attention) is a sublayer this block would silently drop. What the
    forward then does with those slots is adjudicated by the parity gate
    at the tick boundary, which is where this structure's reference is
    declared.
    """
    missing = [a for a in _BLOCK_ATTRS
               if not isinstance(getattr(host, a, None), nn.Module)]
    if missing:
        raise ValueError(f"decoder_block: host lacks {missing}")
    extra = _parameterised_children(host) - set(_BLOCK_ATTRS)
    if extra:
        raise ValueError(
            "decoder_block: host carries sublayers this structure does "
            f"not model ({sorted(extra)}) — keeping the host block")


def bind_decoder_block(host: nn.Module, producer_in, producer_out,
                       ffn: nn.Module, *, cond_kw: str = "adarms_cond",
                       returns_tuple: bool = False,
                       attn=None) -> FusedDecoderBlock:
    """Compose bound sublayer structures into one block.

    ``producer_in`` / ``producer_out`` are bound ``adaln_producer``
    modules on the same conditioning stream (they must share a locator
    for the step lookup to be shared); ``ffn`` is a bound
    ``decoder_ffn`` on the FP8 entry, which is what makes the producer's
    quantize load-bearing.
    """
    qualify(host)
    for name, prod in (("producer_in", producer_in),
                       ("producer_out", producer_out)):
        if not hasattr(prod, "resolve"):
            raise ValueError(
                f"decoder_block: {name} is not an adaln_producer")
    if not producer_out.can_absorb:
        raise ValueError(
            "decoder_block: the second producer cannot absorb a residual "
            "(it is not the rms form with fp8 output) — without that "
            "fold the block boundary buys nothing over the region seams")
    if producer_in.locator is not producer_out.locator:
        raise ValueError(
            "decoder_block: the two producers do not share a step "
            "locator, so the lookup would still run twice")
    return FusedDecoderBlock(host, producer_in, producer_out, ffn,
                             cond_kw=cond_kw,
                             returns_tuple=returns_tuple, attn=attn)
