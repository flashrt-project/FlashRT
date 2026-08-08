"""The fused NVFP4 launch chain over an alternating DiT block stack.

Eight kernels per layer, no elementwise traffic between them: the
AdaLN and the pre-FFN norm emit FP4 directly, the attention output
projection and the FFN down-projection carry the residual add in
their epilogues, and the FFN up-projection emits bias+GELU straight
back to FP4 for the down GEMM. The per-layer AdaLN modulators come
from one stacked projection per distinct timestep, resolved at run
time by nearest-neighbour match — the step-table form, computed at
bind. Cross-attention keys and values go through the block's own
``to_k``/``to_v`` **resolved at call time**, so whatever seat holds
that path when the call happens (a cadence bank, a quantized linear,
the host) serves it — the chain never captures a module reference at
bind (the o_proj dead-seat lesson).

This is a region candidate, not a specialist: it binds through the
adapter contract (observed guard, enable/disable toggle, bit-exact
revert), its activation is decided by :mod:`..regions` receipts, and
every out-of-contract call falls back to the retained host forward
with the reason on the ledger. Contract checks run eager-only and
step aside during CUDA graph capture — the captured window is
certified by its own gate.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .. import KernelUnavailable, hub_kernel
from ...guard import GuardedSeam

GEMM_PACKAGE = "flashrt/fp4-gemm"
NORM_PACKAGE = "flashrt/adaptive-layernorm-producers"
GEMM_SYMBOLS = ("nvfp4_gemm_bias_bf16", "nvfp4_gemm_bias_residual_bf16",
                "nvfp4_gemm_bias_gelu_nvfp4", "quantize_fp4_sfa_bf16")
NORM_SYMBOLS = ("ada_layer_norm_quant_nvfp4_swizzled_bf16",
                "layer_norm_no_affine_quant_nvfp4_swizzled_bf16")

#: bind-time smoke over the whole routed stack, on every probe step.
#: The late-binding lesson calibrates this: a per-layer error class of
#: ~2e-3 compounds across ~32 layers, so the stack floor sits below
#: the per-seam 0.995 line on purpose; the arm's end-to-end parity
#: gate (0.99 against the host's own eager run) stays the judge.
SMOKE_FLOOR = 0.985


def missing_symbols() -> list[str]:
    """The factual prerequisites this box does not meet (may be empty)."""
    gaps: list[str] = []
    for repo, symbols in ((GEMM_PACKAGE, GEMM_SYMBOLS),
                          (NORM_PACKAGE, NORM_SYMBOLS)):
        try:
            kern = hub_kernel(repo, ">=1")
        except KernelUnavailable:
            gaps.append(repo)
            continue
        gaps.extend(f"{repo}:{s}" for s in symbols
                    if not hasattr(kern, s))
    return gaps


class BoundDitFp4Chain(GuardedSeam, torch.nn.Module):
    """The chain's bind-time state: quantized weights and step tables.

    Weight tables and step tables are plain tensor attributes, not
    buffers — this module is a ledger citizen (``plan.observed``), not
    a state_dict citizen; the truth of every weight stays with the
    host modules it absorbs, which is what makes revert and fallback
    bit-exact for free.
    """

    _frt_can_fallback = False   # fallback is the routed closure's job

    def __init__(self) -> None:
        super().__init__()
        self.table: list[dict] = []
        self.blocks: list = []          # plain list: no child registration
        self.dims: dict = {}
        self.t_keys = None
        self.mods_table = None
        self.tails_table = None
        self.text_idx = None
        self.image_idx = None
        self.mask_shape = None
        self.kernels: dict = {}


def _stack_parts(dit) -> tuple[list, int, int, int, int]:
    blocks = list(dit.transformer_blocks)
    head = blocks[0]
    return (blocks, head.num_attention_heads, head.attention_head_dim,
            head.dim, len(blocks))


@torch.no_grad()
def _quantize(bound: BoundDitFp4Chain, dit, kg) -> None:
    """NVFP4-pack every stack GEMM weight from the pristine host."""
    blocks, _nh, _hd, _dim, _n = _stack_parts(dit)

    def quant(w: torch.Tensor):
        return kg.quantize_fp4_sfa_bf16(
            w.detach().to("cuda", torch.bfloat16).contiguous(),
            is_sfb=True)

    def bias(module: torch.nn.Module):
        return module.bias.detach().to(
            "cuda", torch.bfloat16).contiguous()

    for block in blocks:
        attn = block.attn1
        is_self = attn.to_k.in_features == attn.to_q.in_features
        entry: dict[str, Any] = {"is_self": is_self}
        if is_self:
            w = torch.cat([attn.to_q.weight, attn.to_k.weight,
                           attn.to_v.weight], dim=0)
            entry["qkv"] = quant(w)
            entry["qkv_b"] = torch.cat(
                [attn.to_q.bias, attn.to_k.bias, attn.to_v.bias]
            ).detach().to("cuda", torch.bfloat16).contiguous()
        else:
            entry["q"] = quant(attn.to_q.weight)
            entry["q_b"] = bias(attn.to_q)
        entry["o"] = quant(attn.to_out[0].weight)
        entry["o_b"] = bias(attn.to_out[0])
        entry["up"] = quant(block.ff.net[0].proj.weight)
        entry["up_b"] = bias(block.ff.net[0].proj)
        entry["down"] = quant(block.ff.net[2].weight)
        entry["down_b"] = bias(block.ff.net[2])
        bound.table.append(entry)


@torch.no_grad()
def _step_tables(bound: BoundDitFp4Chain, dit, seen, masks) -> None:
    """Per-step modulators from one stacked projection per timestep.

    Computing them in-graph reads the stacked modulator weights once
    per step — pure bandwidth, and the profile named that one skinny
    GEMM as a whole regression. Here they are computed once per
    distinct timestep the probe saw; the run resolves the step by
    nearest-neighbour match, pure tensor ops, capture-safe.
    """
    blocks, _nh, _hd, dim, n_layers = _stack_parts(dit)
    ada_w = torch.cat([b.norm1.linear.weight for b in blocks], dim=0)
    ada_b = torch.cat([b.norm1.linear.bias for b in blocks], dim=0)
    silu = torch.nn.functional.silu
    keys, mods, tails = [], [], []
    for t, temb in seen:
        keys.append(t)
        mods.append(torch.nn.functional.linear(silu(temb), ada_w, ada_b)
                    .view(n_layers, 2, dim).to(torch.bfloat16))
        shift, scale = dit.proj_out_1(silu(temb)).chunk(2, dim=1)
        tails.append(torch.stack(
            [shift.reshape(dim), scale.reshape(dim)]).to(torch.bfloat16))
    image_rows = (masks["image"] & masks["backbone"]).reshape(-1)
    text_rows = (~masks["image"] & masks["backbone"]).reshape(-1)
    bound.t_keys = torch.cat(keys).contiguous()
    bound.mods_table = torch.stack(mods).contiguous()
    bound.tails_table = torch.stack(tails).contiguous()
    bound.text_idx = torch.where(text_rows)[0].contiguous()
    bound.image_idx = torch.where(image_rows)[0].contiguous()
    bound.mask_shape = tuple(masks["image"].shape)


def _make_run(bound: BoundDitFp4Chain, dit) -> Callable:
    """The chain body: eight launches per layer over the stack."""
    kg, kq = bound.kernels["kg"], bound.kernels["kq"]
    ada_fp4 = kq.ada_layer_norm_quant_nvfp4_swizzled_bf16
    ln_fp4 = kq.layer_norm_no_affine_quant_nvfp4_swizzled_bf16
    gemm_bias = kg.nvfp4_gemm_bias_bf16
    gemm_bias_res = kg.nvfp4_gemm_bias_residual_bf16
    gemm_gelu_fp4 = kg.nvfp4_gemm_bias_gelu_nvfp4
    quant_act = kg.quantize_fp4_sfa_bf16
    sdpa = torch.nn.functional.scaled_dot_product_attention
    nh, hd = bound.dims["nh"], bound.dims["hd"]
    dim, n_layers = bound.dims["dim"], bound.dims["n_layers"]
    every_n = bound.dims["every_n"]
    table = bound.table
    blocks = bound.blocks

    def layer(li, h, sa, scale, shift, enc, rows):
        entry = table[li]
        xp, xs = ada_fp4(h, scale, shift)
        if entry["is_self"]:
            qkv = gemm_bias(xp, entry["qkv"][0], xs, entry["qkv"][1],
                            entry["qkv_b"])
            packs = qkv.view(sa, 3, nh, hd).permute(1, 2, 0, 3)
            o = sdpa(packs[0].unsqueeze(0), packs[1].unsqueeze(0),
                     packs[2].unsqueeze(0))
        else:
            q = gemm_bias(xp, entry["q"][0], xs, entry["q"][1],
                          entry["q_b"])
            # call-time resolution: whoever seats to_k/to_v now (a
            # cadence bank, a quantized linear, the host) serves this
            attn = blocks[li].attn1
            kb = attn.to_k(enc).index_select(-2, rows)
            vb = attn.to_v(enc).index_select(-2, rows)
            skv = rows.shape[0]
            o = sdpa(q.view(1, sa, nh, hd).transpose(1, 2),
                     kb.reshape(1, skv, nh, hd).transpose(1, 2),
                     vb.reshape(1, skv, nh, hd).transpose(1, 2))
        o = o.transpose(1, 2).reshape(sa, dim).contiguous()
        op, osf = quant_act(o)
        h = gemm_bias_res(op, entry["o"][0], osf, entry["o"][1],
                          entry["o_b"], h)
        np_, ns = ln_fp4(h)
        hp, hs = gemm_gelu_fp4(np_, entry["up"][0], ns, entry["up"][1],
                               entry["up_b"])
        return gemm_bias_res(hp, entry["down"][0], hs, entry["down"][1],
                             entry["down_b"], h)

    def run(hidden_states, encoder_hidden_states, timestep,
            return_all_hidden_states):
        t = timestep.reshape(-1)[:1].float()
        idx = (bound.t_keys - t).abs().argmin().reshape(1)
        mods = bound.mods_table.index_select(0, idx)[0]
        tail = bound.tails_table.index_select(0, idx)[0]

        b, sa, d = hidden_states.shape
        h = hidden_states.reshape(sa, d).to(torch.bfloat16).contiguous()
        all_h = [hidden_states] if return_all_hidden_states else None
        for li in range(n_layers):
            if li % 2 == 1:
                h = layer(li, h, sa, mods[li, 0], mods[li, 1], None, None)
            else:
                rows = (bound.text_idx if li % (2 * every_n) == 0
                        else bound.image_idx)
                h = layer(li, h, sa, mods[li, 0], mods[li, 1],
                          encoder_hidden_states, rows)
            if all_h is not None:
                all_h.append(h.reshape(b, sa, d))

        h = h.reshape(b, sa, d).type_as(hidden_states)
        h = (dit.norm_out(h) * (1 + tail[1].reshape(1, 1, d))
             + tail[0].reshape(1, 1, d))
        out = dit.proj_out_2(h)
        if return_all_hidden_states:
            return out, all_h
        return out

    return run


def bind_dit_fp4_chain(model, root: str,
                       probe: Callable[[], Any]) -> dict:
    """Bind the chain onto the stack at ``root``; adapter contract out.

    One probe run does all the observation: the distinct timesteps
    (step tables), the attention masks (compact-row banks), and every
    stack call's arguments and output (the smoke reference). The
    routed form must track the host on **every** probe step above
    ``SMOKE_FLOOR`` or the whole bind refuses — no partial routing of
    a stack. Returns ``{"refused": reason}`` on any refusal, with the
    host untouched.
    """
    try:
        kg = hub_kernel(GEMM_PACKAGE, ">=1")
        kq = hub_kernel(NORM_PACKAGE, ">=1")
    except KernelUnavailable as exc:
        return {"refused": f"dit_fp4_chain: {exc}"}
    gaps = missing_symbols()
    if gaps:
        return {"refused": f"dit_fp4_chain missing: {', '.join(gaps)}"}

    dit = model.get_submodule(root) if root else model
    blocks, nh, hd, dim, n_layers = _stack_parts(dit)

    bound = BoundDitFp4Chain()
    bound.kernels = {"kg": kg, "kq": kq}
    bound.blocks = blocks
    bound.dims = {"nh": nh, "hd": hd, "dim": dim, "n_layers": n_layers,
                  "every_n": getattr(dit, "attend_text_every_n_blocks", 2)}
    _quantize(bound, dit, kg)

    # ---- one probe: timesteps, masks, and the smoke reference ----
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    masks: dict = {}
    calls: list[tuple[tuple, dict, torch.Tensor]] = []

    def note(_module, args, output):
        t = args[0].reshape(-1)[:1].float()
        if not any(torch.allclose(t, prev) for prev, _ in seen):
            seen.append((t.detach().clone(), output.detach().clone()))

    def grab(_module, args, kwargs, output):
        img = kwargs.get("image_mask")
        bb = kwargs.get("backbone_attention_mask")
        if img is not None and bb is not None and not masks:
            masks["image"] = img.detach().clone()
            masks["backbone"] = bb.detach().clone()
        out = output[0] if isinstance(output, tuple) else output
        calls.append((args, dict(kwargs), out.detach().clone()))

    hooks = [dit.timestep_encoder.register_forward_hook(note),
             dit.register_forward_hook(grab, with_kwargs=True)]
    try:
        with torch.inference_mode():
            probe()
    finally:
        for hook in hooks:
            hook.remove()
    if not seen:
        return {"refused": "dit_fp4_chain: probe saw no timesteps"}
    if not masks:
        return {"refused": "dit_fp4_chain: probe saw no attention masks"}
    if not calls:
        return {"refused": "dit_fp4_chain: probe never called the stack"}

    _step_tables(bound, dit, seen, masks)
    run = _make_run(bound, dit)
    guard = bound._frt_arm(dtypes=(torch.bfloat16,),
                           device=bound.t_keys.device)
    guard.notes["n_layers"] = n_layers
    guard.notes["steps"] = len(seen)

    # ---- smoke: the routed stack against every captured host call ----
    worst = None
    with torch.inference_mode():
        for args, kwargs, ref in calls:
            parsed = _parse_call(args, kwargs)
            if parsed is None:
                return {"refused": "dit_fp4_chain: probe call shape "
                                   "outside the routed contract"}
            got = run(*parsed)
            got = got[0] if isinstance(got, tuple) else got
            cos = torch.nn.functional.cosine_similarity(
                got.float().flatten(), ref.float().flatten(), dim=0)
            worst = float(cos) if worst is None else min(worst,
                                                         float(cos))
    if worst is None or worst < SMOKE_FLOOR:
        return {"refused": f"dit_fp4_chain smoke cos {worst} < "
                           f"{SMOKE_FLOOR} across {len(calls)} probe "
                           "step(s)"}
    guard.notes["smoke_cos"] = round(worst, 6)

    # ---- route ----
    saved = dit.__dict__.get("forward")
    host_forward = dit.forward

    def routed(_dit_self, *args, **kwargs):
        # the ledger is eager-only, like every guard: a compiler
        # tracing this sees constant-False branches and no side
        # effects (no graph breaks in the hot loop), and a capturing
        # stream skips the Python state the replay would never run
        compiling = torch.compiler.is_compiling()
        capturing = (False if compiling
                     else torch.cuda.is_current_stream_capturing())
        eager = not compiling and not capturing
        if eager:
            guard.calls += 1
        parsed = _parse_call(args, kwargs)
        if parsed is None:
            if not eager:
                raise RuntimeError(
                    "dit_fp4_chain: out-of-contract call during "
                    "capture/compile — fix the eager path first")
            guard.fallbacks += 1
            guard.last_reason = "call outside the routed contract"
            return host_forward(*args, **kwargs)
        if eager:
            img = kwargs.get("image_mask")
            if (img is not None
                    and tuple(img.shape) != bound.mask_shape):
                guard.fallbacks += 1
                guard.last_reason = "attention mask shape changed"
                return host_forward(*args, **kwargs)
        return run(*parsed)

    def enable() -> None:
        dit.forward = types.MethodType(routed, dit)

    def disable() -> None:
        if saved is not None:
            dit.forward = saved
        elif "forward" in dit.__dict__:
            del dit.forward

    def revert() -> None:
        disable()
        bound.table.clear()
        bound.blocks = []

    enable()
    return {
        "observed": {f"{root}::dit_fp4_chain": bound},
        "revert": [revert],
        "toggle": (enable, disable),
        "smoke_cos": worst,
    }


def _parse_call(args: tuple, kwargs: dict):
    """The routed contract: the native stack signature, batch of one.

    Anything else — an unexpected keyword, a missing timestep, a
    batched call — is the host's, not the chain's.
    """
    known = ("hidden_states", "encoder_hidden_states", "timestep",
             "encoder_attention_mask", "return_all_hidden_states",
             "image_mask", "backbone_attention_mask")
    if any(k not in known for k in kwargs):
        return None
    merged = dict(zip(known, args))
    if set(merged).intersection(kwargs):
        return None
    merged.update(kwargs)
    hidden = merged.get("hidden_states")
    enc = merged.get("encoder_hidden_states")
    t = merged.get("timestep")
    if hidden is None or enc is None or t is None:
        return None
    if hidden.dim() != 3 or hidden.shape[0] != 1:
        return None
    return (hidden, enc, t,
            bool(merged.get("return_all_hidden_states", False)))
