"""The fused static-FP8 launch chain over a biased vision tower.

Per layer: LayerNorm→FP8 with the affine pair in the kernel, one
merged-QKV FP8 GEMM with the bias in the epilogue, dense per-call
attention (no cache, no mask — the patch sequence is full and
unpadded), and the output/down projections carry their bias *and*
the residual add in the GEMM epilogue — the native vision form,
kernel for kernel. The activation quantizer sites are calibrated on
the probe run against the pristine host.

The attention element is a ladder: the house FA4 entry probed at the
bound head shape first, plain SDPA as the floor — SDPA is a single
capture-safe primitive and this tower's attention is a small slice
of its time; the GEMM epilogues are where the native form wins.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch

from .. import KernelUnavailable, hub_kernel
from ...guard import GuardedSeam
from ..chain_elements import (
    fp8_weight as _fp8_weight, gelu_tanh_like as _gelu_tanh_like)

GEMM_PACKAGE = "flashrt/fp8-gemm"
FUSE_PACKAGE = "flashrt/transformer-fused-ops"
GEMM_SYMBOLS = ("fp8_linear_bias_bf16", "fp8_linear_bias_residual_bf16",
                "fp8_linear_bias_gelu_bf16")
FUSE_SYMBOLS = ("layer_norm_quant_fp8_static_bf16",
                "quantize_fp8_static_bf16")
FA4_REPO = "flashrt/fa4-cute-runtime"

SMOKE_FLOOR = 0.97
FP8_MAX = 448.0


def missing_symbols() -> list[str]:
    gaps: list[str] = []
    for repo, symbols in ((GEMM_PACKAGE, GEMM_SYMBOLS),
                          (FUSE_PACKAGE, FUSE_SYMBOLS)):
        try:
            kern = hub_kernel(repo, ">=1")
        except KernelUnavailable:
            gaps.append(repo)
            continue
        gaps.extend(f"{repo}:{s}" for s in symbols
                    if not hasattr(kern, s))
    return gaps


class BoundVisionFp8Chain(GuardedSeam, torch.nn.Module):
    """Bind-time state: FP8 weights with epilogue biases, buffers."""

    _frt_can_fallback = False

    def __init__(self) -> None:
        super().__init__()
        self.table: list[dict] = []
        self.dims: dict = {}
        self.buf: dict = {}
        self.scaling = 1.0
        self.out_ctor = None
        self.out_dtype = None
        self.kernels: dict = {}


def _stack_parts(stack):
    layers = list(stack.layers)
    attn = layers[0].self_attn
    dim = attn.q_proj.in_features
    heads = getattr(attn, "num_heads", None)
    if not isinstance(heads, int):
        head_dim = getattr(attn, "head_dim", None)
        if not isinstance(head_dim, int):
            raise ValueError("attention exposes neither num_heads "
                             "nor head_dim")
        heads = dim // head_dim
    hidden = layers[0].mlp.fc1.out_features
    return layers, heads, dim // heads, dim, hidden


@torch.no_grad()
def _quantize(bound, layers, amax) -> None:
    for i, ly in enumerate(layers):
        attn, mlp = ly.self_attn, ly.mlp
        a_qkv, a_o, a_fc1, a_fc2 = (amax[(i, s)] / FP8_MAX for s in
                                    ("qkv", "o", "fc1", "fc2"))
        qkv_w = torch.cat([attn.q_proj.weight, attn.k_proj.weight,
                           attn.v_proj.weight], dim=0)
        qkv_b = torch.cat([attn.q_proj.bias, attn.k_proj.bias,
                           attn.v_proj.bias], dim=0)
        entry: dict[str, Any] = {}
        for name, w, act in (("qkv", qkv_w, a_qkv),
                             ("o", attn.out_proj.weight, a_o),
                             ("fc1", mlp.fc1.weight, a_fc1),
                             ("fc2", mlp.fc2.weight, a_fc2)):
            packed, w_scale = _fp8_weight(w)
            entry[name] = packed
            entry[f"a_{name}"] = act * w_scale
        for name, b in (("qkv_b", qkv_b), ("o_b", attn.out_proj.bias),
                        ("fc1_b", mlp.fc1.bias), ("fc2_b", mlp.fc2.bias)):
            entry[name] = b.detach().to("cuda", torch.bfloat16)
        for name, norm in (("ln1", ly.layer_norm1),
                           ("ln2", ly.layer_norm2)):
            entry[f"{name}_w"] = norm.weight.detach().to(
                "cuda", torch.bfloat16)
            entry[f"{name}_b"] = norm.bias.detach().to(
                "cuda", torch.bfloat16)
            entry[f"{name}_eps"] = float(getattr(norm, "eps", 1e-6))
        entry["sc_qkv"] = torch.tensor([a_qkv], device="cuda",
                                       dtype=torch.float32)
        entry["sc_o"] = torch.tensor([a_o], device="cuda",
                                     dtype=torch.float32)
        entry["sc_fc1"] = torch.tensor([a_fc1], device="cuda",
                                       dtype=torch.float32)
        entry["sc_fc2"] = torch.tensor([a_fc2], device="cuda",
                                       dtype=torch.float32)
        bound.table.append(entry)


def _make_attend(bound, mode: str, kern):
    nh, hd = bound.dims["nh"], bound.dims["hd"]
    scaling = bound.scaling
    if mode == "fa4_cute":
        def attend(q, k, v):
            out = torch.empty_like(q)
            kern.forward_static(q, k, v, out, softmax_scale=scaling,
                                causal=False)
            return out
        return attend

    def attend(q, k, v):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=scaling)
        return o.transpose(1, 2)
    return attend


def _make_run(bound: BoundVisionFp8Chain):
    kg = bound.kernels["kg"]
    kf = bound.kernels["kf"]
    attend = bound.kernels["attend"]
    B, S, nh, hd, D, H = (bound.dims[k] for k in
                          ("batch", "seq", "nh", "hd", "dim",
                           "hidden"))
    b = bound.buf
    table = bound.table
    rf = torch.profiler.record_function

    def run(x3d):
        res = b["res"]
        res.copy_(x3d.reshape(B * S, D))
        for e in table:
            with rf("vi:qkv"):
                kf.layer_norm_quant_fp8_static_bf16(
                    res, e["ln1_w"], e["ln1_b"], e["sc_qkv"],
                    eps=e["ln1_eps"], out=b["xn8"])
                kg.fp8_linear_bias_bf16(b["xn8"], e["qkv"], e["qkv_b"],
                                        alpha=e["a_qkv"], out=b["qkv"])
            with rf("vi:attn"):
                q = b["qkv"][:, :D].view(B, S, nh, hd)
                k = b["qkv"][:, D:2 * D].view(B, S, nh, hd)
                v = b["qkv"][:, 2 * D:].view(B, S, nh, hd)
                att = attend(q, k, v)
            with rf("vi:o"):
                kf.quantize_fp8_static_bf16(
                    att.reshape(B * S, D), e["sc_o"], out=b["o8"])
                kg.fp8_linear_bias_residual_bf16(
                    b["o8"], e["o"], e["o_b"], res, alpha=e["a_o"])
            with rf("vi:ffn"):
                kf.layer_norm_quant_fp8_static_bf16(
                    res, e["ln2_w"], e["ln2_b"], e["sc_fc1"],
                    eps=e["ln2_eps"], out=b["xn8"])
                kg.fp8_linear_bias_gelu_bf16(
                    b["xn8"], e["fc1"], e["fc1_b"], alpha=e["a_fc1"],
                    out=b["hid"])
                kf.quantize_fp8_static_bf16(b["hid"], e["sc_fc2"],
                                            out=b["h8"])
                kg.fp8_linear_bias_residual_bf16(
                    b["h8"], e["fc2"], e["fc2_b"], res,
                    alpha=e["a_fc2"])
        return bound.out_ctor(
            last_hidden_state=res.view(B, S, D)
            .to(bound.out_dtype).clone())

    return run


def bind_vision_fp8_chain(model, root: str,
                          probe: Callable[[], Any]) -> dict:
    """Bind the chain onto the tower at ``root``; adapter contract out."""
    try:
        kg = hub_kernel(GEMM_PACKAGE, ">=1")
        kf = hub_kernel(FUSE_PACKAGE, ">=1")
    except KernelUnavailable as exc:
        return {"refused": f"vision_fp8_chain: {exc}"}
    gaps = missing_symbols()
    if gaps:
        return {"refused": f"vision_fp8_chain missing: "
                           f"{', '.join(gaps)}"}

    stack = model.get_submodule(root) if root else model
    layers, nh, hd, dim, hidden = _stack_parts(stack)
    if not _gelu_tanh_like(layers[0].mlp.activation_fn
                           if hasattr(layers[0].mlp, "activation_fn")
                           else layers[0].mlp.act_fn):
        return {"refused": "vision_fp8_chain: MLP activation is not "
                           "tanh-GELU"}
    scale_attr = getattr(layers[0].self_attn, "scale", None)
    scaling = float(scale_attr) if scale_attr else hd ** -0.5

    bound = BoundVisionFp8Chain()
    bound.kernels = {"kg": kg, "kf": kf}
    bound.scaling = scaling
    bound.dims = {"nh": nh, "hd": hd, "dim": dim, "hidden": hidden,
                  "layers": len(layers)}

    calls: list[dict] = []
    amax: dict = {}

    def note(site):
        def hook(_m, args):
            peak = float(args[0].detach().abs().amax())
            amax[site] = max(amax.get(site, 0.0), peak)
        return hook

    hooks = []
    for i, ly in enumerate(layers):
        hooks.append(ly.self_attn.q_proj.register_forward_pre_hook(
            note((i, "qkv"))))
        hooks.append(ly.self_attn.out_proj.register_forward_pre_hook(
            note((i, "o"))))
        hooks.append(ly.mlp.fc1.register_forward_pre_hook(
            note((i, "fc1"))))
        hooks.append(ly.mlp.fc2.register_forward_pre_hook(
            note((i, "fc2"))))

    saved_probe = stack.__dict__.get("forward")
    host_forward = stack.forward

    def capturing(_self, *args, **kwargs):
        out = host_forward(*args, **kwargs)
        embs = kwargs.get("inputs_embeds",
                          args[0] if args else None)
        hidden_out = getattr(out, "last_hidden_state", None)
        if (embs is not None and hidden_out is not None
                and embs.dim() == 3
                and kwargs.get("attention_mask") is None):
            calls.append({"x": embs.detach().clone(),
                          "out": hidden_out.detach().clone(),
                          "out_type": type(out)})
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
        return {"refused": "vision_fp8_chain: probe never made an "
                           "unmasked encoder call"}
    first = calls[0]
    if any(tuple(c["x"].shape) != tuple(first["x"].shape)
           for c in calls[1:]):
        return {"refused": "vision_fp8_chain: probe calls disagree "
                           "on shape"}
    if any((i, s) not in amax or amax[(i, s)] <= 0.0
           for i in range(len(layers))
           for s in ("qkv", "o", "fc1", "fc2")):
        return {"refused": "vision_fp8_chain: calibration saw a dead "
                           "quantizer site"}

    B, S, _ = first["x"].shape
    bound.dims["batch"], bound.dims["seq"] = B, S
    bound.out_ctor = first["out_type"]
    bound.out_dtype = first["out"].dtype
    _quantize(bound, layers, amax)
    dev, bf = "cuda", torch.bfloat16
    b = bound.buf
    b["res"] = torch.empty(B * S, dim, device=dev, dtype=bf)
    b["xn8"] = torch.empty(B * S, dim, device=dev,
                           dtype=torch.float8_e4m3fn)
    b["qkv"] = torch.empty(B * S, 3 * dim, device=dev, dtype=bf)
    b["o8"] = torch.empty(B * S, dim, device=dev,
                          dtype=torch.float8_e4m3fn)
    b["hid"] = torch.empty(B * S, hidden, device=dev, dtype=bf)
    b["h8"] = torch.empty(B * S, hidden, device=dev,
                          dtype=torch.float8_e4m3fn)

    attend, attn_mode = None, None
    try:
        ka = hub_kernel(FA4_REPO, ">=1")
        cand = _make_attend(bound, "fa4_cute", ka)
        cand(torch.zeros(B, S, nh, hd, device=dev, dtype=bf),
             torch.zeros(B, S, nh, hd, device=dev, dtype=bf),
             torch.zeros(B, S, nh, hd, device=dev, dtype=bf))
        torch.cuda.synchronize()
        attend, attn_mode = cand, "fa4_cute"
    except Exception:       # noqa: BLE001 — the floor rung serves
        attend, attn_mode = _make_attend(bound, "sdpa", None), "sdpa"
    bound.kernels["attend"] = attend

    run = _make_run(bound)
    guard = bound._frt_arm(dtypes=(torch.bfloat16,),
                           device=torch.device("cuda"))
    guard.notes["n_layers"] = len(layers)
    guard.notes["attention"] = attn_mode

    worst = None
    with torch.inference_mode():
        for c in calls:
            got = run(c["x"].to(torch.bfloat16))
            cos = torch.nn.functional.cosine_similarity(
                got.last_hidden_state.float().flatten(),
                c["out"].float().flatten(), dim=0)
            worst = float(cos) if worst is None else min(worst,
                                                         float(cos))
    if worst is None or worst < SMOKE_FLOOR:
        return {"refused": f"vision_fp8_chain smoke cos {worst} < "
                           f"{SMOKE_FLOOR} across {len(calls)} "
                           "probe call(s)"}
    guard.notes["smoke_cos"] = round(worst, 6)

    saved = stack.__dict__.get("forward")
    x_shape = tuple(first["x"].shape)

    def routed(_self, *args, **kwargs):
        compiling = torch.compiler.is_compiling()
        capturing_now = (False if compiling
                         else torch.cuda.is_current_stream_capturing())
        eager = not compiling and not capturing_now
        if eager:
            guard.calls += 1
        embs = kwargs.get("inputs_embeds",
                          args[0] if args else None)
        ok = (embs is not None
              and kwargs.get("attention_mask") is None
              and tuple(embs.shape) == x_shape)
        if not ok:
            if not eager:
                raise RuntimeError(
                    "vision_fp8_chain: out-of-contract call during "
                    "capture/compile — fix the eager path first")
            guard.fallbacks += 1
            guard.last_reason = "call outside the routed contract"
            return host_forward(*args, **kwargs)
        return run(embs.to(torch.bfloat16))

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
        "observed": {f"{root}::vision_fp8_chain": bound},
        "revert": [revert],
        "toggle": (enable, disable),
        "smoke_cos": worst,
    }
