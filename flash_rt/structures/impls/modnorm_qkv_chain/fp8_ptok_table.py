"""Per-token-table implementation of the ``modnorm_qkv_chain`` structure.

The video-DiT block form: modulation parameters live in the block's own
``[1, chunks, D]`` table combined per token with a ``[B, M, chunks, D]``
timestep embedding, inline in the block's forward. Only a block owner
can reroute that math, so this impl binds the whole block:

- both producer sites run the ``adaptive-layernorm-producers`` per-token
  table entry (table add + chunk selection + no-affine LayerNorm +
  modulation + static FP8 quantize, one pass — the per-block six-chunk
  materialization never exists);
- the self-attention Q/K/V consume the shared FP8 wire through wire
  projections: the block hands them the quantized activation explicitly
  before calling the host attention, so rotary/SDPA internals stay the
  host's. The handoff is an explicit attribute set per call — never an
  identity-keyed cache (a pointer-keyed bank measurably cross-fed CFG
  branches on this host);
- the FFN runs the fused FP8 GELU MLP straight from the second producer
  site's wire;
- the output projection and the whole cross-attention are *not* owned:
  the forward calls whatever is attached there, so their individual
  seams keep composing.

Bind needs the four static activation scales the composition consumes
(``attn_in``/``o_in``/``ffn_in``/``ffn_hid``), measured at the block's
own sublayer inputs by the ordinary calibration pass.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

PRODUCER_DEP = {
    "provider": "hf",
    "repo": "flashrt/adaptive-layernorm-producers",
    "version": ">=1",
}
FFN_DEP = {
    "provider": "hf",
    "repo": "flashrt/flashrt-fp8-ffn",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0

#: chunk indices in the block table: (shift, scale, gate) for the
#: attention site and the FFN site, in table order
_ATTN_CHUNKS = (0, 1, 2)
_FFN_CHUNKS = (3, 4, 5)

SUPPORT = {
    "chunks": 6,
    "D": {"min": 512, "max": 16384, "multiple_of": 2},
}


@lru_cache(maxsize=1)
def _producer():
    from flash_rt.structures.impls import hub_kernel

    pkg = hub_kernel(PRODUCER_DEP["repo"], PRODUCER_DEP["version"])
    if not hasattr(pkg, "ada_layer_norm_quant_fp8_ptok_table_bf16"):
        raise ValueError(
            "refused: the installed adaptive-layernorm-producers build "
            "predates the per-token table entry; a package release with "
            "ada_layer_norm_quant_fp8_ptok_table_bf16 is required")
    return pkg


@lru_cache(maxsize=1)
def _ffn_kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(FFN_DEP["repo"], FFN_DEP["version"])


def _q8(w: torch.Tensor):
    s = (w.float().abs().amax() / _FP8_MAX).clamp_min(1e-8)
    packed = (w.float() / s).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8) \
        .contiguous()
    return packed, s.reshape(1)


class WireProj(torch.nn.Module):
    """Projection that consumes the chain's FP8 wire.

    The owning block sets ``take(x8, scale)`` immediately before the
    host attention runs and the projection consumes it exactly once.
    Called without a wire armed (someone invoking the projection outside
    the chain), it falls back to the retained host projection — counted,
    like every dispatch.
    """

    def __init__(self, lin, gemm):
        super().__init__()
        w8, ws = _q8(lin.weight.detach())
        self.register_buffer("_w8", w8)
        self.register_buffer("_ws", ws)
        self._bias = (None if lin.bias is None
                      else lin.bias.detach().to(torch.bfloat16))
        self._gemm = gemm
        self.host_linear = lin
        self._wire = None
        self.off_wire_calls = 0

    def take(self, x8, scale):
        self._wire = (x8, scale)

    def forward(self, x):
        wire = self._wire
        if wire is None:
            if not torch.compiler.is_compiling():
                self.off_wire_calls += 1
            return self.host_linear(x)
        x8, scale = wire
        y = self._gemm(x8, self._w8, scale, self._ws)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*x.shape[:-1], self._w8.shape[0]).type_as(x)


class PerTokenModChainBlock(GuardedSeam, torch.nn.Module):
    """Drop-in replacement for one per-token-table DiT block."""

    _frt_host_attr = "host_block"
    _frt_can_fallback = True

    def __init__(self, block, wires, scales, ffn_state, producer_fn,
                 ffn_fn, table, eps):
        super().__init__()
        self.host_block = block
        self._wires = wires                  # (q, k, v) WireProj modules
        self._scales = scales                # dict of [1] fp32 tensors
        self._ffn = ffn_state
        self._producer = producer_fn
        self._ffn_fn = ffn_fn
        self.register_buffer("_table", table)
        self._eps = eps
        guard = self._frt_arm(dtypes=CAST_OK, device=table.device,
                              k=int(table.shape[1]))
        guard.notes["host_form_calls"] = 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_block":
                raise
            return getattr(super().__getattr__("host_block"), name)

    def _host_form(self, *args, **kwargs):
        guard = self._frt_guard
        if guard is not None and not torch.compiler.is_compiling():
            guard.notes["host_form_calls"] += 1
        return self.host_block(*args, **kwargs)

    def forward(self, hidden_states, encoder_hidden_states, temb,
                rotary_emb, *args, **kwargs):
        admitted = self._frt_admit(hidden_states)
        if admitted is not PROCEED:
            return admitted
        if temb.dim() != 4 or temb.shape[2] != self._table.shape[0]:
            # the broadcast (per-sample) form is the host's own path
            return self._host_form(hidden_states, encoder_hidden_states,
                                   temb, rotary_emb, *args, **kwargs)
        block = self.host_block
        x = hidden_states.contiguous()
        bsz, seq, dim = x.shape
        tb = getattr(temb, "_frt_bf16", None)
        if tb is None:
            # one cast per transformer call, shared by every block: the
            # attribute dies with the tensor, so there is no cross-call
            # identity to poison
            tb = temb.reshape(-1, temb.shape[2], dim) \
                .to(torch.bfloat16).contiguous()
            temb._frt_bf16 = tb
        s_idx, c_idx, g_idx = _ATTN_CHUNKS
        gate_msa = (self._table[g_idx]
                    + temb[0, :, g_idx, :].float()).unsqueeze(0)
        x2d = x.view(-1, dim)
        x8 = self._producer(x2d, tb, self._table,
                            self._scales["attn_in"], s_idx, c_idx,
                            self._eps)
        for wire in self._wires:
            wire.take(x8, self._scales["attn_in"])
        try:
            # x is passed for its shape only: the wire projections
            # consume the quantized activation, not this tensor's values
            attn = block.attn1(x, None, None, rotary_emb)
        finally:
            for wire in self._wires:
                wire._wire = None
        x = (x.float() + attn * gate_msa).type_as(x)
        n2 = block.norm2(x.float()).type_as(x)
        x = x + block.attn2(n2, encoder_hidden_states, None, None)
        fs_idx, fc_idx, fg_idx = _FFN_CHUNKS
        c_gate = (self._table[fg_idx]
                  + temb[0, :, fg_idx, :].float()).unsqueeze(0)
        x8f = self._producer(x.contiguous().view(-1, dim), tb,
                             self._table, self._scales["ffn_in"],
                             fs_idx, fc_idx, self._eps)
        st = self._ffn
        ff = self._ffn_fn(x8f, st["up_w8"], st["up_b"], st["dn_w8"],
                          st["dn_b"], self._scales["ffn_in"],
                          st["up_ws"], self._scales["ffn_hid"],
                          st["dn_ws"])
        ff = ff.reshape(bsz, seq, dim)
        return (x.float() + ff.float() * c_gate).type_as(x)


@torch.no_grad()
def bind_block_seam(model, seam, *, points):
    """Bind one per-token-table block; returns the swap dict.

    The dict carries the block wrapper plus the three wire projections
    under the host attention, so attach/detach treats the whole
    composition as one transaction.
    """
    from flash_rt.structures.discover import _resolve

    block = _resolve(model, seam.path)
    table_param = block.scale_shift_table.detach()
    chunks = int(table_param.shape[1])
    dim = int(table_param.shape[2])
    if chunks != SUPPORT["chunks"]:
        raise ValueError(
            f"refused: {chunks}-chunk table; this impl serves the "
            f"6-chunk (dual-site) layout")
    bounds = SUPPORT["D"]
    if not bounds["min"] <= dim <= bounds["max"] or dim % 2:
        raise ValueError(f"D={dim} outside support envelope")

    # the collector keys each point by its own placement path (the
    # block's sublayer input), exactly where points.resolve put it
    sites = {"attn_in": ".attn1.to_q", "o_in": ".attn1.to_out.0",
             "ffn_in": ".ffn", "ffn_hid": ".ffn.net.2"}
    scales = {}
    for name, child in sites.items():
        value = (points.scale(seam.path + child, name)
                 if points is not None else None)
        if value is None:
            raise ValueError(
                f"refused: calibration point {name!r} was not measured "
                "for this block")
        scales[name] = torch.tensor([float(value)], device="cuda",
                                    dtype=torch.float32)

    producer_pkg = _producer()
    ffn_pkg = _ffn_kernel()
    gemm = ffn_pkg.fp8_gemm_bf16

    wires = tuple(WireProj(getattr(block.attn1, a), gemm)
                  for a in ("to_q", "to_k", "to_v"))
    up, dn = block.ffn.net[0].proj, block.ffn.net[2]
    ffn_state = {}
    ffn_state["up_w8"], ffn_state["up_ws"] = _q8(up.weight.detach())
    ffn_state["dn_w8"], ffn_state["dn_ws"] = _q8(dn.weight.detach())
    ffn_state["up_b"] = up.bias.detach().to(torch.bfloat16).contiguous()
    ffn_state["dn_b"] = dn.bias.detach().to(torch.bfloat16).contiguous()
    for key in ("up_w8", "up_ws", "dn_w8", "dn_ws"):
        ffn_state[key] = ffn_state[key].to("cuda")

    eps = float(getattr(block.norm1, "eps", 1e-6))
    table = table_param.reshape(chunks, dim).float().contiguous() \
        .to("cuda")
    wrapper = PerTokenModChainBlock(
        block, wires, scales, ffn_state,
        producer_pkg.ada_layer_norm_quant_fp8_ptok_table_bf16,
        (getattr(ffn_pkg, "fp8_gelu_mlp_v2_bf16", None)
         or ffn_pkg.fp8_gelu_mlp_bf16), table, eps)

    # bind-time smoke: both producer sites launch once on zeros before
    # the seam is handed out
    z = torch.zeros(4, dim, device="cuda", dtype=torch.bfloat16)
    zt = torch.zeros(4, chunks, dim, device="cuda", dtype=torch.bfloat16)
    for s_idx, c_idx in (_ATTN_CHUNKS[:2], _FFN_CHUNKS[:2]):
        probe = producer_pkg.ada_layer_norm_quant_fp8_ptok_table_bf16(
            z, zt, table, scales["attn_in"], s_idx, c_idx, eps)
        if probe.shape != (4, dim):
            raise ValueError("refused: producer bind smoke shape "
                             f"{tuple(probe.shape)}")

    swaps = {seam.path: wrapper}
    for attr, wire in zip(("to_q", "to_k", "to_v"), wires):
        swaps[f"{seam.path}.attn1.{attr}"] = wire
    return swaps
