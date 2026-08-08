"""The fused static-FP8 launch chain over a plain-norm prefill tower.

Per layer: RMSNorm→FP8 (the host's ``(1+w)`` folded into the kernel's
weight), one merged-QKV FP8 GEMM, split+RoPE into a chain-owned
cache, dense non-causal attention over exactly the used keys (pads
trail, so a used-key count carries the host's mask), FP8 output
projection, and a fused residual+RMSNorm→FP8 into the merged gate/up
GEMM. The keys are un-permuted back to host rotate-half layout and
appended to the host's own cache object — the tower's whole product
is that cache, and every downstream reader keeps its contract.

Same equivalences as the sibling conditioned-stack chain (shared
helpers): the rotate-half↔adjacent-pair permutation on q/k rows, the
probe-calibrated activation scales, the attention ladder. The mask
fact here is simpler: one valid-query row names the used-key run;
pad-query rows are garbage-in-garbage-out by construction (their
hidden states feed only pad positions, their keys are masked by
every downstream mask).
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .. import KernelUnavailable, hub_kernel
from ...guard import GuardedSeam
from ..adarms_stack.fp8_chain import _make_attend
from ..chain_elements import (
    ATTN_RUNGS, FP8_MAX, attention_rungs as _attention_rungs,
    cache_kv as _cache_kv, fp8_weight as _fp8_weight,
    gelu_tanh_like as _gelu_tanh_like,
    interleave_rows as _interleave_rows)

GEMM_PACKAGE = "flashrt/fp8-gemm"
NORM_PACKAGE = "flashrt/flashrt-residual-norm-quant"
ROPE_PACKAGE = "flashrt/flashrt-qkv-cache-rope"
FFN_PACKAGE = "flashrt/transformer-fused-ops"
GEMM_SYMBOLS = ("fp8_linear_bf16",)
NORM_SYMBOLS = ("rms_norm_quant_fp8_static_bf16",
                "residual_add_rms_norm_quant_fp8_static_bf16")
ROPE_SYMBOLS = ("qkv_split_rope_kvcache_bf16",)
FFN_SYMBOLS = ("gate_geglu_merged_quant_fp8_static_bf16",
               "quantize_fp8_static_bf16")

#: whole-tower smoke over hidden states and every layer's cached K/V
#: — a min over ~37 tensors after 18 residual layers of static FP8,
#: so the compounding tail sits lower than a single-output floor
#: (measured: worst deep-layer V ≈0.97, hidden ≈0.96, while the
#: end-to-end action parity the arm gates on holds ≥0.9999). The
#: arm's 0.99 parity gate stays the judge.
SMOKE_FLOOR = 0.95

#: transition rung (author-gated): a native kernel module whose
#: cutlass_fp8_*_bf16out entries serve the rows the hub entry refuses.
#: Test-only — the hub rung wins the moment its package covers the
#: band, with no code change here.
FVK_SO_ENV = "FRT_FVK_SO"
_FVK_SITE = {"qkv": "cutlass_fp8_sq_bf16out",
             "o": "cutlass_fp8_sq_bf16out",
             "gu": "cutlass_fp8_t1_bf16out",
             "dn": "cutlass_fp8_wide_bf16out"}


def _load_fvk():
    import importlib.util
    import os
    so = os.environ.get(FVK_SO_ENV)
    if not so or not os.path.exists(so):
        return None
    spec = importlib.util.spec_from_file_location(
        "flash_rt_kernels", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def missing_symbols() -> list[str]:
    gaps: list[str] = []
    for repo, symbols in ((GEMM_PACKAGE, GEMM_SYMBOLS),
                          (NORM_PACKAGE, NORM_SYMBOLS),
                          (ROPE_PACKAGE, ROPE_SYMBOLS),
                          (FFN_PACKAGE, FFN_SYMBOLS)):
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


class BoundPrefillFp8Chain(GuardedSeam, torch.nn.Module):
    """Bind-time state: FP8 weights, folded norm weights, caches."""

    _frt_can_fallback = False

    def __init__(self) -> None:
        super().__init__()
        self.table: list[dict] = []
        self.dims: dict = {}
        self.buf: dict = {}
        self.rope = None
        self.scaling = 1.0
        self.eps = 1e-6
        self.s_used = 0
        self.out_ctor = None
        self.kperm = None
        self.kperm_inv = None
        self.final_w = None
        self.cache_type = None
        self.kernels: dict = {}
        #: region wire (armed by autobuild): per-layer (k, v) sink
        #: views of a bound consumer's chain caches. The tower's K is
        #: already in chain layout (same split kernel, same adjacent-
        #: pair convention), so writing the sink is the identity of
        #: the consumer's own gather — the consumer drops it in-graph.
        self.prefix_sinks = None


def _plain_norm_weight(norm) -> torch.Tensor | None:
    """The affine vector of an unconditioned RMS norm, or None."""
    if getattr(norm, "dense", None) is not None:
        return None
    w = getattr(norm, "weight", None)
    if w is None or getattr(w, "ndim", 0) != 1:
        return None
    return w


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


def _prefill_mask_facts(mask: torch.Tensor, seq: int) -> int | None:
    """One valid-query row names the used-key run ``[0, s_used)``."""
    if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[-1] != seq:
        return None
    rows = mask[0, 0] if mask.shape[1] == 1 else mask[0, :1][0]
    if rows.shape[-2] != seq:
        return None
    row = rows[0] == 0
    s_used = int(row.sum())
    if s_used < 1 or not bool(row[:s_used].all()):
        return None
    return s_used


def _build_rope(bound, stack, position_ids) -> bool:
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


@torch.no_grad()
def _quantize(bound, layers, amax) -> None:
    nh, kv, hd = (bound.dims[k] for k in ("nh", "kv", "hd"))
    for i, ly in enumerate(layers):
        attn, mlp = ly.self_attn, ly.mlp
        a_qkv, a_o, a_gu, a_dn = (amax[(i, s)] / FP8_MAX for s in
                                  ("qkv", "o", "gu", "dn"))
        fold_in = (1.0 + ly.input_layernorm.weight.detach()
                   .float()).to(attn.q_proj.weight.device)
        fold_post = (1.0 + ly.post_attention_layernorm.weight.detach()
                     .float()).to(attn.q_proj.weight.device)
        qkv_w = torch.cat([
            _interleave_rows(attn.q_proj.weight, nh, hd),
            _interleave_rows(attn.k_proj.weight, kv, hd),
            attn.v_proj.weight], dim=0).float() * fold_in[None, :]
        gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight],
                         dim=0).float() * fold_post[None, :]
        entry: dict[str, Any] = {}
        for name, w, act in (("qkv", qkv_w, a_qkv),
                             ("o", attn.o_proj.weight, a_o),
                             ("gu", gu_w, a_gu),
                             ("dn", mlp.down_proj.weight, a_dn)):
            packed, w_scale = _fp8_weight(w)
            entry[name] = packed
            entry[f"a_{name}"] = act * w_scale
        entry["sc_qkv"] = torch.tensor([a_qkv], device="cuda",
                                       dtype=torch.float32)
        entry["sc_gu"] = torch.tensor([a_gu], device="cuda",
                                      dtype=torch.float32)
        entry["inv_o"] = 1.0 / a_o if a_o > 0 else 1.0
        entry["inv_dn"] = 1.0 / a_dn if a_dn > 0 else 1.0
        def _t(v):
            return torch.tensor([v], device="cuda",
                                dtype=torch.float32)
        entry["t_wsc_gu"] = _t(entry["a_gu"] / a_gu)
        entry["t_sc_dn"] = _t(a_dn)
        entry["t_wsc_dn"] = _t(entry["a_dn"] / a_dn)
        entry["t_sc_o"] = _t(a_o)
        bound.table.append(entry)


def _alloc(bound) -> None:
    S, D, nh, kv, hd, H = (bound.dims[k] for k in
                           ("seq", "dim", "nh", "kv", "hd", "hidden"))
    dev, bf = "cuda", torch.bfloat16
    b = bound.buf
    b["res"] = torch.empty(S, D, device=dev, dtype=bf)
    b["xn8"] = torch.empty(S, D, device=dev, dtype=torch.float8_e4m3fn)
    b["qkv"] = torch.empty(S, (nh + 2 * kv) * hd, device=dev, dtype=bf)
    b["q"] = torch.empty(1, S, nh, hd, device=dev, dtype=bf)
    kc = torch.zeros(1, S, kv, hd, device=dev, dtype=bf)
    vc = torch.zeros(1, S, kv, hd, device=dev, dtype=bf)
    L = bound.dims["layers"]
    b["kc"] = [kc] * L
    b["vc"] = [vc] * L
    b["fg"] = torch.empty(S, D, device=dev, dtype=bf)
    b["gu"] = torch.empty(S, 2 * H, device=dev, dtype=bf)
    b["hid8"] = torch.empty(S, H, device=dev,
                            dtype=torch.float8_e4m3fn)
    b["o8"] = torch.empty(S, nh * hd, device=dev,
                          dtype=torch.float8_e4m3fn)
    b["seqused"] = torch.full((1,), bound.s_used, device=dev,
                              dtype=torch.int32)
    b["ones_w"] = torch.ones(D, device=dev, dtype=bf)
    b["out_full"] = torch.zeros(bound.dims["seq_full"], D, device=dev,
                                dtype=bf)
    b["att"] = torch.empty(1, S, nh, hd, device=dev, dtype=bf)


def _make_run(bound: BoundPrefillFp8Chain):
    kg = bound.kernels["kg"]
    kn = bound.kernels["kn"]
    kr = bound.kernels["kr"]
    kf = bound.kernels["kf"]
    attend = bound.kernels["attend"]
    S, D, nh, kv, hd, H, L = (bound.dims[k] for k in
                              ("seq", "dim", "nh", "kv", "hd",
                               "hidden", "layers"))
    b = bound.buf
    table = bound.table
    eps = bound.eps
    qkv3 = b["qkv"].view(1, S, (nh + 2 * kv) * hd)
    fp8 = torch.float8_e4m3fn
    kperm_inv = bound.kperm_inv
    kh = b["kc"][0].view(S, hd)

    gemm = bound.kernels["gemm"]

    rf = torch.profiler.record_function

    S_full = bound.dims["seq_full"]

    def run(x2d, pkv):
        res = b["res"]
        res.copy_(x2d[:S])
        for l in range(L):
            e = table[l]
            if l == 0:
                kn.rms_norm_quant_fp8_static_bf16(
                    res, b["ones_w"], e["sc_qkv"], eps, out=b["xn8"])
            gemm(e, "qkv", b["xn8"], b["qkv"])
            with rf("pf:rope"):
                kr.qkv_split_rope_kvcache_bf16(
                    qkv3, bound.rope, nh, kv, hd, 0, q_out=b["q"],
                    k_cache=b["kc"][0], v_cache=b["vc"][0],
                    max_seq_len=S)
            if bound.prefix_sinks is not None:
                with rf("pf:wire"):
                    dk, dv = bound.prefix_sinks[l]
                    dk.copy_(kh)
                    dv.copy_(b["vc"][0].view(S, hd))
            if pkv is not None:
                # fresh tensors per layer: the cache keeps references,
                # and the shared buffers are overwritten next layer
                with rf("pf:cache"):
                    k_host = torch.index_select(kh, -1, kperm_inv)
                    pkv.update(k_host.view(1, 1, S, hd),
                               b["vc"][0].reshape(1, 1, S, hd)
                               .clone(), l)
            with rf("pf:attn"):
                att2 = attend(l)
            with rf("pf:o"):
                kf.quantize_fp8_static_bf16(att2, e["t_sc_o"],
                                            out=b["o8"])
                gemm(e, "o", b["o8"], b["fg"])
                res.add_(b["fg"])
            kn.rms_norm_quant_fp8_static_bf16(
                res, b["ones_w"], e["sc_gu"], eps, out=b["xn8"])
            gemm(e, "gu", b["xn8"], b["gu"])
            kf.gate_geglu_merged_quant_fp8_static_bf16(
                b["gu"], e["t_sc_dn"], out=b["hid8"])
            gemm(e, "dn", b["hid8"], b["fg"])
            if l < L - 1:
                nxt = table[l + 1]
                kn.residual_add_rms_norm_quant_fp8_static_bf16(
                    res, b["fg"], b["ones_w"], nxt["sc_qkv"], eps,
                    out=b["xn8"])
            else:
                res.add_(b["fg"])
        fin = res.float()
        fin = fin * torch.rsqrt(fin.square().mean(-1, keepdim=True)
                                + eps)
        b["out_full"][:S] = (fin * bound.final_w).to(torch.bfloat16)
        return bound.out_ctor(
            last_hidden_state=b["out_full"].view(1, S_full, D),
            past_key_values=pkv)

    return run


def bind_prefill_fp8_chain(model, root: str,
                           probe: Callable[[], Any]) -> dict:
    """Bind the chain onto the tower at ``root``; adapter contract out."""
    try:
        kg = hub_kernel(GEMM_PACKAGE, ">=1")
        kn = hub_kernel(NORM_PACKAGE, ">=1")
        kr = hub_kernel(ROPE_PACKAGE, ">=1")
        kf = hub_kernel(FFN_PACKAGE, ">=1")
    except KernelUnavailable as exc:
        return {"refused": f"prefill_fp8_chain: {exc}"}
    rungs = _attention_rungs()
    gaps = missing_symbols()
    if gaps:
        return {"refused": "prefill_fp8_chain missing: "
                           f"{', '.join(gaps)}"}

    stack = model.get_submodule(root) if root else model
    layers, nh, kv, hd, dim, hidden = _stack_parts(stack)
    if kv != 1:
        return {"refused": f"prefill_fp8_chain: kv_heads {kv} outside "
                           "the single-KV band"}
    if not _gelu_tanh_like(layers[0].mlp.act_fn):
        return {"refused": "prefill_fp8_chain: FFN activation is not "
                           "tanh-GELU"}
    scalings = {float(ly.self_attn.scaling) for ly in layers}
    if len(scalings) != 1:
        return {"refused": "prefill_fp8_chain: per-layer attention "
                           "scaling differs"}
    final_w = _plain_norm_weight(stack.norm)
    if final_w is None or any(
            _plain_norm_weight(ly.input_layernorm) is None
            or _plain_norm_weight(ly.post_attention_layernorm) is None
            for ly in layers):
        return {"refused": "prefill_fp8_chain: a norm is not a plain "
                           "affine RMS norm"}

    bound = BoundPrefillFp8Chain()
    bound.kernels = {"kg": kg, "kn": kn, "kr": kr, "kf": kf}
    bound.scaling = scalings.pop()
    bound.dims = {"nh": nh, "kv": kv, "hd": hd, "dim": dim,
                  "hidden": hidden, "layers": len(layers)}
    bound.eps = float(getattr(stack.norm, "eps", 1e-6))
    bound.final_w = (1.0 + final_w.detach().float()).to("cuda")

    # ---- one probe: the prefill call, mask fact, amax sites ----
    calls: list[dict] = []
    amax: dict[tuple[int, str], float] = {}
    # the quantizer statistic is a high element quantile, not the raw
    # peak: one outlier element must cost itself saturation, not the
    # whole tensor's resolution (the house two-level rule, single-call
    # analog). FRT_CALIB_Q=1.0 restores the raw peak for A/B runs.
    import os as _os
    calib_q = float(_os.environ.get("FRT_CALIB_Q", "1.0"))

    def note(site, unfold=None):
        safe = (unfold.abs().clamp(min=1e-6)
                if unfold is not None else None)

        def hook(_m, args):
            x = args[0].detach().float().abs()
            if safe is not None:
                # the chain quantizes the pure-RMS output and folds the
                # norm's (1+w) into the GEMM weight columns (the tight
                # distribution is what FP8 can afford); measure what
                # the chain will actually see. A (1+w)=0 channel's
                # host output is exactly zero — its weight column is
                # zero too, so its recovered magnitude may be anything
                # and the clamp keeps it out of the statistic.
                x = x / safe
            x = x.flatten()
            if calib_q >= 1.0 or x.numel() < 1000:
                peak = float(x.amax())
            else:
                k = max(1, int(x.numel() * (1.0 - calib_q)))
                peak = float(x.kthvalue(x.numel() - k + 1).values)
            amax[site] = max(amax.get(site, 0.0), peak)
        return hook

    hooks = []
    for i, ly in enumerate(layers):
        fold_in = (1.0 + ly.input_layernorm.weight.detach().float())
        fold_post = (1.0 + ly.post_attention_layernorm.weight
                     .detach().float())
        hooks.append(ly.self_attn.q_proj.register_forward_pre_hook(
            note((i, "qkv"), fold_in)))
        hooks.append(ly.self_attn.o_proj.register_forward_pre_hook(
            note((i, "o"))))
        hooks.append(ly.mlp.gate_proj.register_forward_pre_hook(
            note((i, "gu"), fold_post)))
        hooks.append(ly.mlp.down_proj.register_forward_pre_hook(
            note((i, "dn"))))

    saved_probe = stack.__dict__.get("forward")
    host_forward = stack.forward

    def capturing(_self, *args, **kwargs):
        out = host_forward(*args, **kwargs)
        embs = kwargs.get("inputs_embeds")
        pkv = getattr(out, "past_key_values", None)
        hidden_out = getattr(out, "last_hidden_state", None)
        if (embs is not None and kwargs.get("use_cache")
                and hidden_out is not None
                and embs.dim() == 3 and embs.shape[0] == 1
                and kwargs.get("adarms_cond") is None):
            entry = {
                "x": embs.detach().clone(),
                "mask": kwargs.get("attention_mask"),
                "pos": kwargs.get("position_ids"),
                "out": hidden_out.detach().clone(),
                "out_type": type(out),
                "cache_type": type(pkv) if pkv is not None else None,
                "kv": [tuple(t.detach().clone()
                             for t in _cache_kv(pkv, i))
                       for i in range(len(layers))] if pkv is not None
                      else None,
            }
            entry["mask"] = (entry["mask"].detach().clone()
                             if entry["mask"] is not None else None)
            entry["pos"] = (entry["pos"].detach().clone()
                            if entry["pos"] is not None else None)
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
        return {"refused": "prefill_fp8_chain: probe never made a "
                           "prefill call"}
    first = calls[0]
    if first["mask"] is None or first["pos"] is None:
        return {"refused": "prefill_fp8_chain: probe call carried no "
                           "mask or positions"}
    S = first["x"].shape[1]
    s_used = _prefill_mask_facts(first["mask"], S)
    if s_used is None:
        return {"refused": "prefill_fp8_chain: mask outside the "
                           "[valid|pad] band"}
    dead = [(i, s) for i in range(len(layers))
            for s in ("qkv", "o", "gu", "dn")
            if (i, s) not in amax or not amax[(i, s)] > 0.0]
    if dead:
        return {"refused": "prefill_fp8_chain: dead quantizer "
                           f"site(s) {dead[:6]} of {len(dead)}; "
                           f"sample={ {k: amax[k] for k in list(amax)[:4]} }"}

    bound.dims["seq"] = s_used
    bound.dims["seq_full"] = S
    bound.s_used = s_used
    bound.out_ctor = first["out_type"]
    bound.cache_type = first["cache_type"]
    # the GEMM entry's row band is a runtime fact, not a symbol fact:
    # probe it at the bound shape, then fall to the transition rung
    gemm_mode = "hub"
    try:
        kg.fp8_linear_bf16(
            torch.zeros(S, dim, device="cuda",
                        dtype=torch.float8_e4m3fn),
            torch.zeros(dim, dim, device="cuda",
                        dtype=torch.float8_e4m3fn), alpha=1.0)
    except Exception as hub_exc:  # noqa: BLE001 — a band fact
        fvk = _load_fvk()
        if fvk is None or any(not hasattr(fvk, n)
                              for n in set(_FVK_SITE.values())):
            return {"refused": "prefill_fp8_chain: the FP8 GEMM entry "
                               f"refuses {S} rows ({hub_exc})"}
        gemm_mode = "fvk"
        bound.kernels["fvk"] = fvk
    if gemm_mode == "hub":
        def gemm(e, site, x8, out):
            kg.fp8_linear_bf16(x8, e[site], alpha=e["a_" + site],
                               out=out)
    else:
        fvk = bound.kernels["fvk"]
        fns = {site: getattr(fvk, name)
               for site, name in _FVK_SITE.items()}

        def gemm(e, site, x8, out):
            w = e[site]
            stream = torch.cuda.current_stream().cuda_stream
            fns[site](x8.data_ptr(), w.data_ptr(), out.data_ptr(),
                      x8.shape[0], w.shape[0], w.shape[1],
                      e["a_" + site], 0.0, stream)
    bound.kernels["gemm"] = gemm
    if not _build_rope(bound, stack, first["pos"]):
        return {"refused": "prefill_fp8_chain: rotary table is not "
                           "half-duplicated"}
    _quantize(bound, layers, amax)
    _alloc(bound)
    half = hd // 2
    kperm = torch.empty(hd, dtype=torch.long, device="cuda")
    kperm[0::2] = torch.arange(half, device="cuda")
    kperm[1::2] = torch.arange(half, hd, device="cuda")
    bound.kperm = kperm
    bound.kperm_inv = torch.argsort(kperm)

    attend, attn_mode = None, None
    rung_trail = []
    for mode, kern in rungs:
        try:
            candidate = _make_attend(bound, mode, kern)
            candidate(0)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — a dead rung, next
            rung_trail.append(f"{mode}: {type(exc).__name__}")
            continue
        attend, attn_mode = candidate, mode
        break
    if attend is None:
        return {"refused": "prefill_fp8_chain: no attention rung "
                           f"executes here ({'; '.join(rung_trail)})"}
    bound.kernels["attend"] = attend

    run = _make_run(bound)
    guard = bound._frt_arm(dtypes=(torch.bfloat16,),
                           device=torch.device("cuda"))
    guard.notes["n_layers"] = len(layers)
    guard.notes["s_used"] = s_used
    guard.notes["attention"] = attn_mode
    guard.notes["gemm"] = gemm_mode

    # ---- smoke: hidden states and the cache the tower left behind ----
    import os as _os2
    floor = float(_os2.environ.get("FRT_PREFILL_SMOKE_FLOOR",
                                   str(SMOKE_FLOOR)))
    scores: list[tuple[float, str]] = []
    with torch.inference_mode():
        for c in calls:
            fresh = c["cache_type"]() if c["cache_type"] else None
            got = run(c["x"][0].to(torch.bfloat16), fresh)
            valid = slice(0, s_used)
            cos = torch.nn.functional.cosine_similarity(
                got.last_hidden_state[0, valid].float().flatten(),
                c["out"][0, valid].float().flatten(), dim=0)
            scores.append((float(cos), "hidden"))
            if fresh is not None and c["kv"] is not None:
                for l in range(len(layers)):
                    gk, gv = _cache_kv(fresh, l)
                    hk, hv = c["kv"][l]
                    for tag, a, bb in (("k", gk, hk), ("v", gv, hv)):
                        cc = torch.nn.functional.cosine_similarity(
                            a[..., :s_used, :].float().flatten(),
                            bb[..., :s_used, :].float().flatten(),
                            dim=0)
                        scores.append((float(cc), f"{tag}{l}"))
    worst = min(s0 for s0, _ in scores) if scores else None
    trail = sorted(scores)[:4]
    if worst is None or worst < floor:
        return {"refused": f"prefill_fp8_chain smoke cos {worst} < "
                           f"{floor}; worst={trail}"}
    guard.notes["smoke_cos"] = round(worst, 6)
    guard.notes["smoke_worst"] = [(round(a, 4), t) for a, t in trail]

    # ---- route ----
    saved = stack.__dict__.get("forward")
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
        mask = kwargs.get("attention_mask")
        pkv = kwargs.get("past_key_values")
        ok = (not args and embs is not None
              and kwargs.get("use_cache")
              and kwargs.get("adarms_cond") is None
              and tuple(embs.shape) == x_shape
              and (mask is None or tuple(mask.shape) == mask_shape))
        if not ok:
            if not eager:
                raise RuntimeError(
                    "prefill_fp8_chain: out-of-contract call during "
                    "capture/compile — fix the eager path first")
            guard.fallbacks += 1
            guard.last_reason = "call outside the routed contract"
            return host_forward(*args, **kwargs)
        if pkv is None and bound.cache_type is not None:
            pkv = bound.cache_type()
        return run(embs[0].to(torch.bfloat16), pkv)

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
        "observed": {f"{root}::prefill_fp8_chain": bound},
        "revert": [revert],
        "toggle": (enable, disable),
        "smoke_cos": worst,
    }
