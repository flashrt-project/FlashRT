"""The fused static-FP8 launch chain over an AdaRMS decoder stack.

Per layer: one fused gated-residual/AdaRMS/quantize producer feeds a
merged-QKV FP8 GEMM, a split+RoPE kernel writes the suffix K/V into a
chain-owned cache behind the copied prefix, FA2 attends over exactly
the used keys, and the same producer carries the post-attention
residual into the FFN's merged gate/up GEMM. The activation quantizer
sites are calibrated on the probe run against the pristine host.

Two equivalences are established at bind, not assumed:

- The host rotates pairs ``(i, i + half)`` (rotate-half); the split
  kernel rotates adjacent pairs. The q/k projection rows are permuted
  at quantize time so the kernel's layout carries the host's rotation
  — attention dot products are invariant under a shared permutation
  of the head dimension, and V stays in host order so the output
  projection sees host layout.
- The host masks with a dense additive mask over
  ``[prefix | pad | suffix]``. The chain checks on the probe call
  that every query row shares that exact pattern, then expresses it
  as a used-key count: prefix rows copied, suffix rows appended, FA2
  told the total. A mask outside this shape refuses the bind.

This is a region candidate: adapter contract in, receipts decide
activation, out-of-contract calls fall back to the retained host
forward. Contract checks run eager-only and step aside during
capture — the captured window is certified by its own gate.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .. import KernelUnavailable, hub_kernel
from ..chain_elements import (
    ATTN_RUNGS, FP8_MAX, attention_rungs as _attention_rungs,
    cache_kv as _cache_kv, fp8_weight as _fp8_weight,
    gelu_tanh_like as _gelu_tanh_like,
    interleave_rows as _interleave_rows)
from ...guard import GuardedSeam

GEMM_PACKAGE = "flashrt/fp8-gemm"
NORM_PACKAGE = "flashrt/flashrt-adaptive-norms"
ROPE_PACKAGE = "flashrt/flashrt-qkv-cache-rope"
GEMM_SYMBOLS = ("fp8_linear_bf16",)
NORM_SYMBOLS = ("gate_residual_ada_norm_fp8_static_bf16",)
ROPE_SYMBOLS = ("qkv_split_rope_kvcache_bf16",)
FUSE_PACKAGE = "flashrt/transformer-fused-ops"
FUSE_SYMBOLS = ("gate_geglu_merged_quant_fp8_static_bf16",
                "quantize_fp8_static_bf16")
#: the mixed-precision band, the native decoder's own form: the three
#: norm-fed GEMMs ride W4A4 NVFP4 with dynamic block scales, the down
#: projection stays static FP8 (its input is the fused GEGLU's FP8)
FP4_GEMM_PACKAGE = "flashrt/fp4-gemm"
FP4_NORM_PACKAGE = "flashrt/fp4-fused-ops"
FP4_GEMM_SYMBOLS = ("nvfp4_gemm_bias_bf16", "quantize_fp4_sfa_bf16")
FP4_NORM_SYMBOLS = ("gate_res_ada_rms_norm_quant_nvfp4_swizzled_bf16",)

#: the band table IS the recipe: one row per precision band, naming
#: the extra packages the band assembles and its candidate rank. The
#: chain body stays one form; adding a band means adding a row (and
#: its two element closures in :func:`_band_elements`), never another
#: branch in the loop.
BANDS: dict[str, dict] = {
    "fp8": {"packages": (), "precision_rank": 0},
    "fp4": {"packages": ((FP4_GEMM_PACKAGE, FP4_GEMM_SYMBOLS),
                         (FP4_NORM_PACKAGE, FP4_NORM_SYMBOLS)),
            "precision_rank": 1},
}

#: the attention element resolves per host: the house CuTe FA4
#: runtime first — it carries the D256 2CTA forward this stack's
#: 8-query/1-KV heads need, which the community FA4 package does not
#: expose — then the FA2 used-keys entry. The chain cache holds
#: exactly the used keys, so both rungs run the same dense
#: non-causal call into a caller-owned output; a rung that loads but
#: cannot execute is eliminated by the bind-time functional probe,
#: not by device lists.
#: whole-stack smoke on every probe call; the arm's end-to-end parity
#: gate (0.99 vs the host's own eager run) stays the judge
SMOKE_FLOOR = 0.985


def missing_symbols_fp4() -> list[str]:
    return missing_symbols(band="fp4")


def missing_symbols(band: str = "fp8") -> list[str]:
    """The factual prerequisites this box does not meet (may be empty)."""
    gaps: list[str] = []
    packages = ((GEMM_PACKAGE, GEMM_SYMBOLS),
                (NORM_PACKAGE, NORM_SYMBOLS),
                (ROPE_PACKAGE, ROPE_SYMBOLS),
                (FUSE_PACKAGE, FUSE_SYMBOLS)) + BANDS[band]["packages"]
    for repo, symbols in packages:
        try:
            kern = hub_kernel(repo, ">=1")
        except KernelUnavailable:
            gaps.append(repo)
            continue
        gaps.extend(f"{repo}:{s}" for s in symbols
                    if not hasattr(kern, s))
    if not _attention_rungs():
        gaps.append("attention: " + " or ".join(
            r[1] for r in ATTN_RUNGS))
    return gaps


class BoundAdaRmsFp8Chain(GuardedSeam, torch.nn.Module):
    """Bind-time state: FP8 weights, style stack, chain-owned caches.

    Plain tensor attributes, not buffers — a ledger citizen, not a
    state_dict citizen; the truth of every weight stays with the host
    modules the chain absorbs, which keeps revert bit-exact for free.
    """

    _frt_can_fallback = False   # fallback is the routed closure's job

    def __init__(self) -> None:
        super().__init__()
        self.table: list[dict] = []
        self.dims: dict = {}
        self.buf: dict = {}
        self.style_w_t = None
        self.style_b = None
        self.rope = None
        self.scaling = 1.0
        self.eps = 1e-6
        self.p_used = 0
        self.total_keys = 0
        self.out_ctor = None
        self.kperm = None
        self.kernels: dict = {}
        #: the step-table form (the stack's own house pattern): style
        #: modulations are baked per probed step at bind, and the run
        #: resolves the live conditioning to a table by on-device
        #: nearest-neighbour match — no Python state, so the same
        #: data-driven selection is what a compile traces and a
        #: capture bakes
        self.step_conds = None
        self.style_table = None
        self.fin_table = None
        self.band = "fp8"
        self.zero_bias: dict = {}
        #: armed by the region wire (autobuild) once a bound producer
        #: guarantees the chain caches' prefix rows are written
        #: in-graph before the first decoder step: the per-step prefix
        #: gather leaves the graph. Bind-time state — constant by
        #: capture time, so both a capture and a compile bake one arm.
        self.prefix_wired = False


def _stack_parts(stack):
    layers = list(stack.layers)
    attn = layers[0].self_attn
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int):
        raise ValueError("attention exposes no integer head_dim")
    nh = attn.q_proj.out_features // head_dim
    kv = attn.k_proj.out_features // head_dim
    dim = attn.q_proj.in_features
    hidden = layers[0].mlp.gate_proj.out_features
    return layers, nh, kv, head_dim, dim, hidden


@torch.no_grad()
def _quantize(bound: BoundAdaRmsFp8Chain, layers, amax: dict) -> None:
    """Pack every stack GEMM from the pristine host: static FP8 with
    the calibrated activation scale folded into alpha, or — on the
    mixed band — NVFP4 with dynamic block scales for the norm-fed
    GEMMs (the down projection stays FP8; its input already is)."""
    nh, kv, hd = (bound.dims[k] for k in ("nh", "kv", "hd"))
    fp4 = bound.band == "fp4"
    quant4 = (bound.kernels["kg4"].quantize_fp4_sfa_bf16
              if fp4 else None)
    for i, ly in enumerate(layers):
        attn, mlp = ly.self_attn, ly.mlp
        a_qkv, a_o, a_gu, a_dn = (amax[(i, s)] / FP8_MAX for s in
                                  ("qkv", "o", "gu", "dn"))
        qkv_w = torch.cat([
            _interleave_rows(attn.q_proj.weight, nh, hd),
            _interleave_rows(attn.k_proj.weight, kv, hd),
            attn.v_proj.weight], dim=0)
        gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        entry: dict[str, Any] = {}
        for name, w, act in (("qkv", qkv_w, a_qkv),
                             ("o", attn.o_proj.weight, a_o),
                             ("gu", gu_w, a_gu),
                             ("dn", mlp.down_proj.weight, a_dn)):
            if fp4 and name != "dn":
                wp, wsf = quant4(
                    w.detach().to("cuda", torch.bfloat16)
                    .contiguous(), is_sfb=True)
                entry[name] = (wp, wsf)
                if w.shape[0] not in bound.zero_bias:
                    bound.zero_bias[w.shape[0]] = torch.zeros(
                        w.shape[0], device="cuda",
                        dtype=torch.bfloat16)
                continue
            packed, w_scale = _fp8_weight(w)
            entry[name] = packed
            entry[f"a_{name}"] = act * w_scale
        entry["sc_qkv"] = torch.tensor([a_qkv], device="cuda",
                                       dtype=torch.float32)
        entry["sc_gu"] = torch.tensor([a_gu], device="cuda",
                                      dtype=torch.float32)
        entry["inv_o"] = 1.0 / a_o if a_o > 0 else 1.0
        entry["inv_dn"] = 1.0 / a_dn if a_dn > 0 else 1.0
        entry["t_sc_o"] = torch.tensor([a_o], device="cuda",
                                       dtype=torch.float32)
        entry["t_sc_dn"] = torch.tensor([a_dn], device="cuda",
                                        dtype=torch.float32)
        bound.table.append(entry)


def _style_stack(bound: BoundAdaRmsFp8Chain, stack, layers) -> None:
    """One stacked projection serves every norm's (scale, shift, gate)."""
    norms = []
    for ly in layers:
        norms.extend((ly.input_layernorm, ly.post_attention_layernorm))
    norms.append(stack.norm)
    w = torch.cat([n.dense.weight.detach().float() for n in norms], dim=0)
    b = torch.cat([n.dense.bias.detach().float() for n in norms], dim=0)
    bound.style_w_t = w.t().contiguous().to("cuda")
    bound.style_b = b.to("cuda").unsqueeze(0)
    bound.eps = float(getattr(norms[0], "eps", 1e-6))


def _mask_facts(mask: torch.Tensor, seq: int) -> tuple[int, int] | None:
    """Read ``[prefix | pad | suffix]`` out of the additive mask, or
    refuse: every row identical, valid keys one prefix run plus the
    whole suffix."""
    if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[-2] != seq:
        return None
    rows = mask[0, 0] if mask.shape[1] == 1 else mask[0, :1, :, :][0]
    valid = rows == 0
    if not bool((valid == valid[:1]).all()):
        return None
    row = valid[0]
    total = row.shape[0]
    p_raw = total - seq
    if p_raw < 1 or not bool(row[p_raw:].all()):
        return None
    prefix = row[:p_raw]
    p_used = int(prefix.sum())
    if p_used < 1 or not bool(prefix[:p_used].all()):
        return None
    return p_used, p_raw


def _build_rope(bound: BoundAdaRmsFp8Chain, stack,
                position_ids: torch.Tensor) -> bool:
    hd = bound.dims["hd"]
    half = hd // 2
    dummy = torch.zeros(1, position_ids.shape[1], hd, device="cuda",
                        dtype=torch.float32)
    cos, sin = stack.rotary_emb(dummy, position_ids.to("cuda"))
    cos, sin = cos[0].float(), sin[0].float()
    if not torch.allclose(cos[:, :half], cos[:, half:], atol=1e-5):
        return False
    rope = torch.empty(position_ids.shape[1], hd, device="cuda",
                       dtype=torch.bfloat16)
    rope[:, 0::2] = cos[:, :half].to(torch.bfloat16)
    rope[:, 1::2] = sin[:, :half].to(torch.bfloat16)
    bound.rope = rope
    return True


def _alloc(bound: BoundAdaRmsFp8Chain) -> None:
    S, D, nh, kv, hd, H, L = (bound.dims[k] for k in
                              ("seq", "dim", "nh", "kv", "hd",
                               "hidden", "layers"))
    T = bound.total_keys
    dev, bf = "cuda", torch.bfloat16
    b = bound.buf
    b["zero"] = torch.zeros(S, D, device=dev, dtype=bf)
    b["ones_w"] = torch.ones(D, device=dev, dtype=bf)
    b["res"] = torch.empty(S, D, device=dev, dtype=bf)
    b["xn8"] = torch.empty(S, D, device=dev, dtype=torch.float8_e4m3fn)
    b["g1"] = torch.empty(S, D, device=dev, dtype=bf)
    b["g2"] = torch.empty(S, D, device=dev, dtype=bf)
    b["dn"] = torch.empty(S, D, device=dev, dtype=bf)
    b["qkv"] = torch.empty(S, (nh + 2 * kv) * hd, device=dev, dtype=bf)
    b["q"] = torch.empty(1, S, nh, hd, device=dev, dtype=bf)
    b["gu"] = torch.empty(S, 2 * H, device=dev, dtype=bf)
    b["hid8"] = torch.empty(S, H, device=dev,
                            dtype=torch.float8_e4m3fn)
    b["o8"] = torch.empty(S, nh * hd, device=dev,
                          dtype=torch.float8_e4m3fn)
    b["kc"] = [torch.zeros(1, T, kv, hd, device=dev, dtype=bf)
               for _ in range(L)]
    b["vc"] = [torch.zeros(1, T, kv, hd, device=dev, dtype=bf)
               for _ in range(L)]
    b["seqused"] = torch.full((1,), T, device=dev, dtype=torch.int32)
    b["att"] = torch.empty(1, S, nh, hd, device=dev, dtype=bf)
    if bound.band == "fp4":
        # static FP4 out-buffers, sized by the quantizer's own
        # allocator: the SF tensor's tile-padding entries are zeroed
        # once here and never written by the producers, so one
        # zero-padded pair per site stays valid across every replay —
        # no per-call allocation, no per-call SF memset in the graph
        quant4 = bound.kernels["kg4"].quantize_fp4_sfa_bf16
        b["xp4"], b["xsf4"] = quant4(b["zero"])
        b["op4"], b["osf4"] = quant4(
            torch.zeros(S, nh * hd, device=dev, dtype=bf))
        # the norm kernel's style contract is contiguous (rows, 3*dim):
        # one static stage, filled by a single broadcast copy per step
        b["st4"] = torch.empty(2 * L, S, 3 * D, device=dev, dtype=bf)


def _make_attend(bound: BoundAdaRmsFp8Chain, mode: str, kern):
    """One rung of the attention ladder, closed over the buffers."""
    b = bound.buf
    S, nh, hd = (bound.dims[k] for k in ("seq", "nh", "hd"))
    scaling = bound.scaling
    att2 = b["att"].view(S, nh * hd)
    if mode == "fa4_cute":
        def attend(layer_index):
            kern.forward_static(
                b["q"], b["kc"][layer_index], b["vc"][layer_index],
                b["att"], softmax_scale=scaling, causal=False,
                pack_gqa=True, seqused_k=b["seqused"])
            return att2
        return attend
    lse = kern.allocate_outputs(b["q"])[1]

    def attend(layer_index):
        kern.forward_seqused_static(
            b["q"], b["kc"][layer_index], b["vc"][layer_index],
            b["seqused"], out=b["att"], softmax_lse=lse,
            softmax_scale=scaling)
        return att2
    return attend


def _band_elements(bound: BoundAdaRmsFp8Chain):
    """The band row's two element closures, plus the style stager.

    Every band expresses the same three moves — stage the step's
    styles, feed a style-conditioned producer into a projection site,
    project the attention output — so the chain loop stays one form
    and a band is a table row plus this closure pair.

    - ``stage_styles(styles)``: per-step staging; returns the handle
      the producer consumes.
    - ``norm_project(styles, slot, x, prev_gate, e, site, out,
      gate_out)``: gated-residual AdaRMS producer into the ``site``
      GEMM, result in ``out``, the norm's gate in ``gate_out``.
    - ``out_project(att2, e, out)``: quantize the attention output and
      run the output projection into ``out``.
    """
    b = bound.buf
    kg = bound.kernels["kg"]
    kn = bound.kernels["kn"]
    kf = bound.kernels["kf"]
    eps = bound.eps
    S, D, L = (bound.dims[k] for k in ("seq", "dim", "layers"))
    if bound.band == "fp4":
        norm4 = bound.kernels["kn4"]             .gate_res_ada_rms_norm_quant_nvfp4_swizzled_bf16
        gemm4 = bound.kernels["kg4"].nvfp4_gemm_bias_bf16
        quant4 = bound.kernels["kg4"].quantize_fp4_sfa_bf16
        zb = bound.zero_bias
        xp4, xsf4 = b["xp4"], b["xsf4"]
        op4, osf4 = b["op4"], b["osf4"]
        st4 = b["st4"]

        def stage_styles(styles):
            # one broadcast copy stages every layer's rows for the
            # step — the per-norm expand+contiguous kernels (and
            # their per-call allocations) never enter the graph
            st4.copy_(styles[:2 * L].expand(-1, S, -1))
            return st4

        def norm_project(styles, slot, x, prev_gate, e, site, out,
                         gate_out):
            norm4(x, prev_gate, b["res"], styles[slot], packed=xp4,
                  sf_swizzled=xsf4, gate=gate_out)
            wp, wsf = e[site]
            gemm4(xp4, wp, xsf4, wsf, zb[out.shape[1]], out=out)

        def out_project(att2, e, out):
            quant4(att2, op4, osf4)
            gemm4(op4, e["o"][0], osf4, e["o"][1], zb[D], out=out)

        return stage_styles, norm_project, out_project

    def stage_styles(styles):
        return styles

    def norm_project(styles, slot, x, prev_gate, e, site, out,
                     gate_out):
        kn.gate_residual_ada_norm_fp8_static_bf16(
            b["res"], x, prev_gate, b["ones_w"], styles[slot],
            e["sc_" + site], eps, out=b["xn8"], gate_out=gate_out)
        kg.fp8_linear_bf16(b["xn8"], e[site], alpha=e["a_" + site],
                           out=out)

    def out_project(att2, e, out):
        kf.quantize_fp8_static_bf16(att2, e["t_sc_o"], out=b["o8"])
        kg.fp8_linear_bf16(b["o8"], e["o"], alpha=e["a_o"], out=out)

    return stage_styles, norm_project, out_project


def _make_run(bound: BoundAdaRmsFp8Chain):
    kg = bound.kernels["kg"]
    kn = bound.kernels["kn"]
    kr = bound.kernels["kr"]
    kf = bound.kernels["kf"]
    attend = bound.kernels["attend"]
    S, D, nh, kv, hd, H, L = (bound.dims[k] for k in
                              ("seq", "dim", "nh", "kv", "hd",
                               "hidden", "layers"))
    P, T = bound.p_used, bound.total_keys
    b = bound.buf
    table = bound.table
    eps = bound.eps
    qkv3 = b["qkv"].view(1, S, (nh + 2 * kv) * hd)
    fp8 = torch.float8_e4m3fn
    stage_styles, norm_project, out_project = _band_elements(bound)

    kperm = bound.kperm

    rf = torch.profiler.record_function

    def run(x2d, cond, prefix_kv):
        if not bound.prefix_wired:
            with rf("ad:prefix"):
                for l, (pk, pv) in enumerate(prefix_kv):
                    # rotated pairs are adjacent; the host cached
                    # prefix keys in rotate-half layout — gather them
                    # into the shared permutation so q·k stays
                    # layout-consistent. A wired prefix skips this:
                    # the producer already wrote these rows in chain
                    # layout, and kperm∘kperm_inv is the identity.
                    torch.index_select(pk, -1, kperm,
                                       out=b["kc"][l][0, :P, 0])
                    b["vc"][l][0, :P, 0].copy_(pv)
        # nearest-neighbour step resolution, fully on-device: the
        # schedule is a bind-time fact, so the live conditioning names
        # its baked table without a host round-trip or Python state
        with rf("ad:style"):
            step = (bound.step_conds
                    - cond.float()).abs().sum(-1).argmin()
            styles = bound.style_table.index_select(
                0, step.view(1))[0]
            fin = bound.fin_table.index_select(0, step.view(1))[0]
            styles = stage_styles(styles)
        res = b["res"]
        res.copy_(x2d)
        delta, gate = b["zero"], b["zero"]
        rf_layers = rf("ad:layers")
        rf_layers.__enter__()
        for l in range(L):
            e = table[l]
            norm_project(styles, 2 * l, delta, gate, e, "qkv",
                         b["qkv"], b["g1"])
            kr.qkv_split_rope_kvcache_bf16(
                qkv3, bound.rope, nh, kv, hd, P, q_out=b["q"],
                k_cache=b["kc"][l], v_cache=b["vc"][l], max_seq_len=T)
            att2 = attend(l)
            out_project(att2, e, b["dn"])
            norm_project(styles, 2 * l + 1, b["dn"], b["g1"], e, "gu",
                         b["gu"], b["g2"])
            kf.gate_geglu_merged_quant_fp8_static_bf16(
                b["gu"], e["t_sc_dn"], out=b["hid8"])
            kg.fp8_linear_bf16(b["hid8"], e["dn"], alpha=e["a_dn"],
                               out=b["dn"])
            delta, gate = b["dn"], b["g2"]
        rf_layers.__exit__(None, None, None)
        with rf("ad:tail"):
            res = res.float() + gate.float() * delta.float()
            normed = res * torch.rsqrt(
                res.square().mean(-1, keepdim=True) + eps)
            out = (normed * (1 + fin[0]) + fin[1]).to(torch.bfloat16)
        return bound.out_ctor(last_hidden_state=out.view(1, S, D),
                              past_key_values=None)

    return run


def bind_adarms_fp4_chain(model, root: str,
                          probe: Callable[[], Any]) -> dict:
    return bind_adarms_fp8_chain(model, root, probe, band="fp4")


def bind_adarms_fp8_chain(model, root: str,
                          probe: Callable[[], Any],
                          band: str = "fp8") -> dict:
    """Bind the chain onto the stack at ``root``; adapter contract out.

    One probe run does all the observation: the suffix calls (smoke
    references), the mask facts, the prefix K/V, and the activation
    amax at every quantizer site. The routed form must track the host
    on **every** probe call above ``SMOKE_FLOOR`` or the whole bind
    refuses — no partial routing. Returns ``{"refused": reason}`` on
    any refusal, with the host untouched.
    """
    try:
        kg = hub_kernel(GEMM_PACKAGE, ">=1")
        kn = hub_kernel(NORM_PACKAGE, ">=1")
        kr = hub_kernel(ROPE_PACKAGE, ">=1")
        kf = hub_kernel(FUSE_PACKAGE, ">=1")
        kg4 = kn4 = None
        if band == "fp4":
            kg4 = hub_kernel(FP4_GEMM_PACKAGE, ">=1")
            kn4 = hub_kernel(FP4_NORM_PACKAGE, ">=1")
    except KernelUnavailable as exc:
        return {"refused": f"adarms_{band}_chain: {exc}"}
    rungs = _attention_rungs()
    gaps = (missing_symbols_fp4() if band == "fp4"
            else missing_symbols())
    if gaps:
        return {"refused": f"adarms_fp8_chain missing: {', '.join(gaps)}"}

    stack = model.get_submodule(root) if root else model
    layers, nh, kv, hd, dim, hidden = _stack_parts(stack)
    if kv != 1:
        return {"refused": f"adarms_fp8_chain: kv_heads {kv} outside "
                           "the single-KV band"}
    if not _gelu_tanh_like(layers[0].mlp.act_fn):
        return {"refused": "adarms_fp8_chain: FFN activation is not "
                           "tanh-GELU"}
    scalings = {float(ly.self_attn.scaling) for ly in layers}
    if len(scalings) != 1:
        return {"refused": "adarms_fp8_chain: per-layer attention "
                           "scaling differs"}

    bound = BoundAdaRmsFp8Chain()
    bound.kernels = {"kg": kg, "kn": kn, "kr": kr, "kf": kf,
                     "kg4": kg4, "kn4": kn4}
    bound.band = band
    bound.scaling = scalings.pop()
    bound.dims = {"nh": nh, "kv": kv, "hd": hd, "dim": dim,
                  "hidden": hidden, "layers": len(layers)}

    # ---- one probe: calls, mask facts, prefix K/V, amax sites ----
    calls: list[dict] = []
    amax: dict[tuple[int, str], float] = {}

    def note(site):
        def hook(_m, args):
            x = args[0]
            peak = float(x.detach().abs().amax())
            amax[site] = max(amax.get(site, 0.0), peak)
        return hook

    hooks = []
    for i, ly in enumerate(layers):
        hooks.append(ly.self_attn.q_proj.register_forward_pre_hook(
            note((i, "qkv"))))
        hooks.append(ly.self_attn.o_proj.register_forward_pre_hook(
            note((i, "o"))))
        hooks.append(ly.mlp.gate_proj.register_forward_pre_hook(
            note((i, "gu"))))
        hooks.append(ly.mlp.down_proj.register_forward_pre_hook(
            note((i, "dn"))))

    # the host calls the stack's ``forward`` directly, so capture is an
    # instance-attribute wrap, not a forward hook
    saved_probe = stack.__dict__.get("forward")
    host_forward = stack.forward

    def capturing(_self, *args, **kwargs):
        out = host_forward(*args, **kwargs)
        embs = kwargs.get("inputs_embeds")
        cond = kwargs.get("adarms_cond")
        pkv = kwargs.get("past_key_values")
        hidden = getattr(out, "last_hidden_state", None)
        if (embs is not None and cond is not None and pkv is not None
                and hidden is not None
                and embs.dim() == 3 and embs.shape[0] == 1):
            entry = {
                "x": embs.detach().clone(),
                "cond": cond.detach().clone(),
                "mask": kwargs.get("attention_mask"),
                "pos": kwargs.get("position_ids"),
                "out": hidden.detach().clone(),
                "out_type": type(out),
            }
            entry["mask"] = (entry["mask"].detach().clone()
                             if entry["mask"] is not None else None)
            entry["pos"] = (entry["pos"].detach().clone()
                            if entry["pos"] is not None else None)
            if not any("kv" in c for c in calls):
                entry["kv"] = [
                    (_cache_kv(pkv, i)[0].detach().clone(),
                     _cache_kv(pkv, i)[1].detach().clone())
                    for i in range(len(layers))]
            calls.append(entry)
        return out

    stack.forward = types.MethodType(capturing, stack)
    try:
        with torch.inference_mode():
            probe()
    finally:
        for hook in hooks:
            hook.remove()
        if saved_probe is not None:
            stack.forward = saved_probe
        else:
            stack.__dict__.pop("forward", None)

    if not calls:
        return {"refused": "adarms_fp8_chain: probe never made a "
                           "suffix call"}
    first = calls[0]
    if first["mask"] is None or first["pos"] is None:
        return {"refused": "adarms_fp8_chain: probe call carried no "
                           "mask or positions"}
    S = first["x"].shape[1]
    facts = _mask_facts(first["mask"], S)
    if facts is None:
        return {"refused": "adarms_fp8_chain: mask outside the "
                           "[prefix|pad|suffix] band"}
    p_used, p_raw = facts
    pos = first["pos"][0]
    want = torch.arange(p_used, p_used + S, device=pos.device)
    if not torch.equal(pos.to(want.dtype), want):
        return {"refused": "adarms_fp8_chain: positions are not the "
                           "contiguous suffix run"}
    for c in calls[1:]:
        if (c["x"].shape != first["x"].shape
                or (c["mask"] is not None
                    and c["mask"].shape != first["mask"].shape)):
            return {"refused": "adarms_fp8_chain: probe calls disagree "
                               "on shape"}
    if any((i, s) not in amax or amax[(i, s)] <= 0.0
           for i in range(len(layers))
           for s in ("qkv", "o", "gu", "dn")):
        return {"refused": "adarms_fp8_chain: calibration saw a dead "
                           "quantizer site"}

    bound.dims["seq"] = S
    bound.p_used = p_used
    bound.total_keys = p_used + S
    bound.out_ctor = first["out_type"]

    _style_stack(bound, stack, layers)
    if not _build_rope(bound, stack, first["pos"]):
        return {"refused": "adarms_fp8_chain: rotary table is not "
                           "half-duplicated"}
    _quantize(bound, layers, amax)
    _alloc(bound)

    # ---- the attention ladder: first rung that actually executes ----
    attend, attn_mode = None, None
    rung_trail = []
    for mode, kern in rungs:
        try:
            candidate = _make_attend(bound, mode, kern)
            candidate(0)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — a dead rung, next one
            rung_trail.append(f"{mode}: {type(exc).__name__}")
            continue
        attend, attn_mode = candidate, mode
        break
    if attend is None:
        return {"refused": "adarms_fp8_chain: no attention rung "
                           f"executes here ({'; '.join(rung_trail)})"}
    bound.kernels["attend"] = attend

    half = hd // 2
    kperm = torch.empty(hd, dtype=torch.long, device="cuda")
    kperm[0::2] = torch.arange(half, device="cuda")
    kperm[1::2] = torch.arange(half, hd, device="cuda")
    bound.kperm = kperm

    # bake one style table per distinct probed step (the step-table
    # form): the run resolves the live conditioning by nearest match
    n_norms = 2 * len(layers) + 1
    step_conds, style_tables, fin_tables = [], [], []
    with torch.no_grad():
        for c in calls:
            cnd = c["cond"].float().cuda()
            if any(torch.allclose(cnd, prev) for prev in step_conds):
                continue
            st = torch.addmm(bound.style_b, cnd, bound.style_w_t)
            step_conds.append(cnd)
            # rows=1 broadcast: the norm kernels accept a single
            # style row, so the table stays one row per norm
            style_tables.append(
                st.view(n_norms, 1, 3 * dim)
                  .to(torch.bfloat16).contiguous())
            fin_tables.append(
                st[0, (n_norms - 1) * 3 * dim:].view(3, dim).clone())
    bound.step_conds = torch.cat(step_conds, dim=0)
    bound.style_table = torch.stack(style_tables)
    bound.fin_table = torch.stack(fin_tables)
    prefix_kv = [(k[0, 0, :p_used].contiguous().clone(),
                  v[0, 0, :p_used].contiguous().clone())
                 for k, v in first["kv"]]
    run = _make_run(bound)
    guard = bound._frt_arm(dtypes=(torch.bfloat16,),
                           device=torch.device("cuda"))
    guard.notes["n_layers"] = len(layers)
    guard.notes["suffix_calls"] = len(calls)
    guard.notes["p_used"] = p_used
    guard.notes["attention"] = attn_mode
    if rung_trail:
        guard.notes["attention_fell_through"] = rung_trail

    # ---- smoke: the routed stack against every captured call ----
    worst = None
    with torch.inference_mode():
        for c in calls:
            got = run(c["x"][0].to(torch.bfloat16), c["cond"], prefix_kv)
            cos = torch.nn.functional.cosine_similarity(
                got.last_hidden_state.float().flatten(),
                c["out"].float().flatten(), dim=0)
            worst = float(cos) if worst is None else min(worst,
                                                         float(cos))
    if worst is None or worst < SMOKE_FLOOR:
        return {"refused": f"adarms_fp8_chain smoke cos {worst} < "
                           f"{SMOKE_FLOOR} across {len(calls)} probe "
                           "call(s)"}
    guard.notes["smoke_cos"] = round(worst, 6)

    # ---- route ----
    saved = stack.__dict__.get("forward")
    n_layers = len(layers)
    x_shape = tuple(first["x"].shape)
    mask_shape = tuple(first["mask"].shape)

    def routed(_self, *args, **kwargs):
        compiling = torch.compiler.is_compiling()
        capturing_now = (False if compiling
                         else torch.cuda.is_current_stream_capturing())
        eager = not compiling and not capturing_now
        if eager:
            guard.calls += 1
        embs = kwargs.get("inputs_embeds")
        cond = kwargs.get("adarms_cond")
        pkv = kwargs.get("past_key_values")
        mask = kwargs.get("attention_mask")
        ok = (not args and embs is not None and cond is not None
              and pkv is not None
              and tuple(embs.shape) == x_shape
              and (mask is None or tuple(mask.shape) == mask_shape))
        if not ok:
            if not eager:
                raise RuntimeError(
                    "adarms_fp8_chain: out-of-contract call during "
                    "capture/compile — fix the eager path first")
            guard.fallbacks += 1
            guard.last_reason = "call outside the routed contract"
            return host_forward(*args, **kwargs)
        prefix = []
        for i in range(n_layers):
            k, v = _cache_kv(pkv, i)
            prefix.append((k[0, 0, :bound.p_used],
                           v[0, 0, :bound.p_used]))
        return run(embs[0].to(torch.bfloat16), cond, prefix)

    def enable() -> None:
        stack.forward = types.MethodType(routed, stack)

    def disable() -> None:
        if saved is not None:
            stack.forward = saved
        elif "forward" in stack.__dict__:
            del stack.forward

    def revert() -> None:
        disable()
        bound.table.clear()
        bound.buf.clear()

    enable()
    return {
        "observed": {f"{root}::adarms_fp8_chain": bound},
        "revert": [revert],
        "toggle": (enable, disable),
        "smoke_cos": worst,
    }
