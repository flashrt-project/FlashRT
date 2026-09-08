"""Pi0.5 capture-time graph construction with CANN fused operators.

Weights, merged projections, rotary tables and timestep styles are prepared
once. Runtime replay lives in the independent AscendCL pointer/stream layer.
The separate CPU pipeline is a diagnostic; correctness promotion requires
an independent full-model reference on real observations.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from flash_rt.npu.core.linear import linear

from flash_rt.npu.models.pi05.pipeline import (
    DEC_L, DEC_D, DEC_HD, DEC_NH, DEC_NKV, _DP, _TOP_EXP_NORM,
    ENC_L, ENC_D, ENC_HD, ENC_NH, ENC_NKV, _EP,
    VIS_L, VIS_D, VIS_NH, VIS_HD, VIS_TOKENS_PER_VIEW, _VP, _MP,
    GELU_TANH_APPROX,
    EPS, rope_half_split, attention, vision_attention, rms_norm, _rope_inv_freqs,
)

# Optional fused K/V suffix-store kernel (option-c library, 16-bit copy
# only). Falls back to the two torch slice copies when the extension is not
# installed or the mode is disabled. `_kv_store_mode` is a runtime knob so
# paired A/B can interleave arms inside one process (never compare across
# processes).
#
# A/B (paired, 910B4, 2026-09-08): fused kernel p50 88.55 ms vs torch copies
# 86.31 ms on the same process (raw cos 1.0) — the serialized per-row kernel
# is ~2.2 ms slower than the captured aclnn copies. DEFAULT IS TORCH COPIES
# (mode 0); the kernel stays as a validated library capability and can be
# re-enabled per experiment via `set_kv_store_mode(1)` / env.
try:
    import kv_cache_store_ext as _kv_cache_store_ext
    _KV_EXT_OK = True
except Exception:  # pragma: no cover - env without the extension
    _kv_cache_store_ext = None
    _KV_EXT_OK = False

_kv_store_mode = int(os.environ.get("FLASH_RT_NPU_PI05_KV_KERNEL", "0"))


def set_kv_store_mode(mode: int) -> None:
    """0 = torch copies, 1 = fused kernel (if importable). For A/B only."""
    global _kv_store_mode
    _kv_store_mode = int(mode)


def kv_store_fused_enabled() -> bool:
    return _KV_EXT_OK and _kv_store_mode == 1


# ── build-time weight/style preparation ───────────────────────────────


def make_fast_weights(wb: dict) -> dict:
    """Shallow overlay of ``wb`` with per-decoder-layer merged GEMM weights.

    Adds ``qkv`` (q|k|v concatenated over rows) and ``gu`` (gate|up
    concatenated) for every decoder layer; all other entries stay the same
    objects so no extra device copies are created.
    """
    wf = dict(wb)
    for i in range(DEC_L):
        p = f"{_DP}.{i}"
        # Gemma decoder attention/MLP linears are bias-free (mirrors the
        # reference recipe, which passes weight only).
        wf[f"{p}.qkv.weight"] = torch.cat(
            [wb[f"{p}.self_attn.q_proj.weight"],
             wb[f"{p}.self_attn.k_proj.weight"],
             wb[f"{p}.self_attn.v_proj.weight"]], dim=0)
        wf[f"{p}.gu.weight"] = torch.cat(
            [wb[f"{p}.mlp.gate_proj.weight"],
             wb[f"{p}.mlp.up_proj.weight"]], dim=0)
    return wf


def make_styles(wb: dict, conds: list) -> tuple:
    """Precompute every AdaRMS ``dense`` linear output (3·DEC_D fp32).

    Returns ``(attn, mlp, top)`` indexed ``[step][layer]`` / ``[step]`` with
    the raw (3072,) tensor whose chunks are (scale, shift, gate) — identical
    values to the reference's per-call ``linear(cond, dense)``, computed
    once outside the captured graph.
    """
    attn, mlp, top = [], [], []
    for c in conds:
        a = [linear(c, wb[f"{_DP}.{i}.input_layernorm.dense.weight"],
                      wb[f"{_DP}.{i}.input_layernorm.dense.bias"])
             for i in range(DEC_L)]
        m = [linear(c, wb[f"{_DP}.{i}.post_attention_layernorm.dense.weight"],
                      wb[f"{_DP}.{i}.post_attention_layernorm.dense.bias"])
             for i in range(DEC_L)]
        t = linear(c, wb[f"{_TOP_EXP_NORM}.weight"],
                     wb[f"{_TOP_EXP_NORM}.bias"])
        attn.append(a)
        mlp.append(m)
        top.append(t)
    return attn, mlp, top


def make_styles_opt(wb: dict, conds: list) -> tuple:
    """AdaRMS styles precomputed for the fused path. Each entry is
    ``(gamma_bf16, shift_f32, gate_bf16)`` where ``gamma = (1 + scale)`` so
    ``npu_rms_norm(x, gamma)`` reproduces ``x*rms*(1+scale)`` and only the
    ``+ shift`` stays elementwise. Indexed ``[step][layer]`` / ``[step]``."""
    attn, mlp, top = [], [], []
    for c in conds:
        a = []
        for i in range(DEC_L):
            s = linear(c, wb[f"{_DP}.{i}.input_layernorm.dense.weight"],
                         wb[f"{_DP}.{i}.input_layernorm.dense.bias"])
            sc, sh, g = s.chunk(3, dim=-1)
            a.append(((1.0 + sc).to(torch.bfloat16), sh.contiguous(),
                      g.to(torch.bfloat16)))
        attn.append(a)
        m = []
        for i in range(DEC_L):
            s = linear(c, wb[f"{_DP}.{i}.post_attention_layernorm.dense.weight"],
                         wb[f"{_DP}.{i}.post_attention_layernorm.dense.bias"])
            sc, sh, g = s.chunk(3, dim=-1)
            m.append(((1.0 + sc).to(torch.bfloat16), sh.contiguous(),
                      g.to(torch.bfloat16)))
        mlp.append(m)
        s = linear(c, wb[f"{_TOP_EXP_NORM}.weight"],
                     wb[f"{_TOP_EXP_NORM}.bias"])
        sc, sh, g = s.chunk(3, dim=-1)
        top.append(((1.0 + sc).to(torch.bfloat16), sh.contiguous(),
                    g.to(torch.bfloat16)))
    return attn, mlp, top


def make_encoder_fast_weights(wb: dict) -> dict:
    """Overlay of ``wb`` with merged per-encoder-layer GEMM weights
    (``qkv`` and ``gu``), mirroring the decoder overlay. Bias-free."""
    wfe = dict(wb)
    for i in range(ENC_L):
        p = f"{_EP}.{i}"
        wfe[f"{p}.qkv.weight"] = torch.cat(
            [wb[f"{p}.self_attn.q_proj.weight"],
             wb[f"{p}.self_attn.k_proj.weight"],
             wb[f"{p}.self_attn.v_proj.weight"]], dim=0)
        wfe[f"{p}.gu.weight"] = torch.cat(
            [wb[f"{p}.mlp.gate_proj.weight"],
             wb[f"{p}.mlp.up_proj.weight"]], dim=0)
    return wfe


def encoder_pass_fast(prefix_emb: torch.Tensor, wf: dict):
    """Gemma encoder with merged QKV/GateUp GEMMs — same cache contract as
    ``pipeline.encoder_pass`` (list of 18 (K_rot, V), each (S,256)); the
    decoder's prefix fill reads it unchanged."""
    S = prefix_emb.shape[0]
    dev = prefix_emb.device
    inv = _rope_inv_freqs(ENC_HD, device=dev)
    pos = torch.arange(S, device=dev)
    x = prefix_emb
    cache = []
    for i in range(ENC_L):
        p = f"{_EP}.{i}"
        xn = rms_norm(x, wf[f"{p}.input_layernorm.weight"])
        qkv = linear(xn, wf[f"{p}.qkv.weight"])
        q, k, v = qkv.split([ENC_NH * ENC_HD, ENC_HD, ENC_HD], dim=-1)
        q_rot = rope_half_split(q, pos, inv, ENC_HD)
        k_rot = rope_half_split(k, pos, inv, ENC_HD)
        qh = q_rot.reshape(S, ENC_NH, ENC_HD).transpose(0, 1)
        kh = k_rot.reshape(S, ENC_NKV, ENC_HD).transpose(0, 1).expand(
            ENC_NH, S, ENC_HD)
        vh = v.reshape(S, ENC_NKV, ENC_HD).transpose(0, 1).expand(
            ENC_NH, S, ENC_HD)
        o = attention(qh, kh, vh)
        o = linear(o, wf[f"{p}.self_attn.o_proj.weight"])
        x = x + o
        xn = rms_norm(x, wf[f"{p}.post_attention_layernorm.weight"])
        gu = linear(xn, wf[f"{p}.gu.weight"])
        import torch_npu
        hidden = torch_npu.npu_geglu(gu, dim=-1, approximate=1, activate_left=True)[0]
        d = linear(hidden, wf[f"{p}.mlp.down_proj.weight"])
        x = x + d
        cache.append((k_rot.contiguous(), v.contiguous()))
    return cache


# ── encoder with fused residual+rms (P0 reuse: npu_rms_norm / add_rms) ─

def make_encoder_opt_weights(wb: dict) -> dict:
    """Overlay with per-encoder-layer bf16 gamma=(1+w) for the fused norm
    ops (reference rms multiplies by (1+w); npu_rms_norm expects gamma)."""
    wfe = dict(wb)
    for i in range(ENC_L):
        p = f"{_EP}.{i}"
        a = wb[f"{p}.input_layernorm.weight"]
        f = wb[f"{p}.post_attention_layernorm.weight"]
        wfe[f"{p}.gamma_a"] = (1.0 + a.float()).to(torch.bfloat16)
        wfe[f"{p}.gamma_f"] = (1.0 + f.float()).to(torch.bfloat16)
    return wfe


def encoder_pass_opt(prefix_emb: torch.Tensor, wf: dict, cos_t, sin_t):
    """Encoder with fused norm ops and table-based rope: ``npu_rms_norm``
    for the pre-attention norm and ``npu_add_rms_norm`` for the
    post-attention residual+norm. Static INT8 groups share input quantization
    while keeping separate GEMMs. Same cache contract as pipeline.encoder_pass."""
    import torch_npu
    S = prefix_emb.shape[0]
    x = prefix_emb
    cache = []
    for i in range(ENC_L):
        p = f"{_EP}.{i}"
        x = x.to(torch.bfloat16)  # keep the bf16 contract (residual cast etc.)
        xn = torch_npu.npu_rms_norm(x, wf[f"{p}.gamma_a"], EPS)[0]
        if f"{p}.qkv.group" in wf:
            q, k, v = wf[f"{p}.qkv.group"](xn)
        else:
            q = linear(xn, wf[f"{p}.self_attn.q_proj.weight"])
            k = linear(xn, wf[f"{p}.self_attn.k_proj.weight"])
            v = linear(xn, wf[f"{p}.self_attn.v_proj.weight"])
        q_rot = rope_fast(q, cos_t, sin_t, 0, ENC_HD)
        k_rot = rope_fast(k, cos_t, sin_t, 0, ENC_HD)
        o = attention_flash(q_rot, k_rot, v, ENC_NH, ENC_NKV)
        o = linear(o, wf[f"{p}.self_attn.o_proj.weight"])
        xn_ff, _, x = torch_npu.npu_add_rms_norm(x, o, wf[f"{p}.gamma_f"], EPS)
        x = x.to(torch.bfloat16)  # npu residual comes back fp32; keep bf16 chain
        if f"{p}.gu.group" in wf:
            g, u = wf[f"{p}.gu.group"](xn_ff)
        else:
            g = linear(xn_ff, wf[f"{p}.mlp.gate_proj.weight"])
            u = linear(xn_ff, wf[f"{p}.mlp.up_proj.weight"])
        if f"{p}.down.fused" in wf:
            d = wf[f"{p}.down.fused"](g, u)
        else:
            d = linear(F.gelu(g, approximate="tanh") * u,
                         wf[f"{p}.mlp.down_proj.weight"])
        x = x + d
        cache.append((k_rot.contiguous(), v.contiguous()))
    return cache


# ── in-graph decoder math (capture-safe subset of the reference) ──────

def _ada(x: torch.Tensor, style: torch.Tensor) -> tuple:
    """AdaRMSNorm with a precomputed style tensor; same math as pipeline.ada_rms_norm."""
    x32 = x.to(torch.float32)
    scale, shift, gate = style.chunk(3, dim=-1)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    xn = x32 * torch.rsqrt(var + EPS)
    y = (xn * (1.0 + scale) + shift).to(x.dtype)
    return y, gate.to(x.dtype)


def _ada_opt(x: torch.Tensor, style: tuple) -> tuple:
    """AdaRMS with ``npu_rms_norm`` + elementwise shift: style is
    ``(gamma_bf16=(1+scale), shift_f32, gate_bf16)``. Replaces the manual
    var/rsqrt/mul fp32 chain with one fused rms node per call."""
    import torch_npu
    gamma_b, shift_f, gate_b = style
    xb = x.to(torch.bfloat16)
    xn = torch_npu.npu_rms_norm(xb, gamma_b, EPS)[0]
    y = (xn.to(torch.float32) + shift_f).to(xb.dtype)
    return y, gate_b


def decoder_step_fast(x_t, enc_cache, wf, style_attn, style_mlp, style_top,
                      prefix_len: int, chunk: int):
    """One denoise step over the whole chunk, merged-GEMM / precomputed-style
    decoder. Mirrors ``pipeline._decoder_step``; returns (chunk,1024)."""
    dev = x_t.device
    inv = _rope_inv_freqs(DEC_HD, device=dev)
    pos = torch.arange(prefix_len, prefix_len + chunk, device=dev)
    x = x_t
    for i in range(DEC_L):
        p = f"{_DP}.{i}"
        x_mod, gate_a = _ada(x, style_attn[i])
        qkv = linear(x_mod, wf[f"{p}.qkv.weight"])
        q, k, v = qkv.split([DEC_NH * DEC_HD, DEC_HD, DEC_HD], dim=-1)
        q_rot = rope_half_split(q, pos, inv, DEC_HD)
        k_rot = rope_half_split(k, pos, inv, DEC_HD)
        k_full = torch.cat([enc_cache[i][0], k_rot], dim=0)
        v_full = torch.cat([enc_cache[i][1], v], dim=0)
        qh = q_rot.reshape(chunk, DEC_NH, DEC_HD).transpose(0, 1)
        kh = k_full.reshape(-1, DEC_NKV, DEC_HD).transpose(0, 1).expand(
            DEC_NH, -1, DEC_HD)
        vh = v_full.reshape(-1, DEC_NKV, DEC_HD).transpose(0, 1).expand(
            DEC_NH, -1, DEC_HD)
        o = attention(qh, kh, vh)
        o = linear(o, wf[f"{p}.self_attn.o_proj.weight"])
        x = x + o * gate_a
        x_mod, gate_f = _ada(x, style_mlp[i])
        gu = linear(x_mod, wf[f"{p}.gu.weight"])
        import torch_npu
        hidden = torch_npu.npu_geglu(gu, dim=-1, approximate=1, activate_left=True)[0]
        d = linear(hidden, wf[f"{p}.mlp.down_proj.weight"])
        x = x + d * gate_f
    x_final, _ = _ada(x, style_top)
    return x_final


# ── no-cat cross-attention K/V (L1b) ──────────────────────────────────

def make_kv_buffers(prefix_len: int, chunk_cap: int) -> tuple:
    """One contiguous (prefix_len + chunk_cap, 256) K/V buffer per decoder
    layer. Encoder prefix rows are copied in once per frame
    (``fill_kv_prefix``); each denoise step overwrites only its own suffix
    rows in place. Removes the two per-layer ``torch.cat`` (+ their repeated
    copy of the whole prefix) from every decoder step."""
    rows = prefix_len + chunk_cap
    kbufs = [torch.empty(rows, DEC_HD, dtype=torch.bfloat16, device="npu")
             for _ in range(DEC_L)]
    vbufs = [torch.empty(rows, DEC_HD, dtype=torch.bfloat16, device="npu")
             for _ in range(DEC_L)]
    return kbufs, vbufs


def fill_kv_prefix(enc_cache, kbufs, vbufs, prefix_len: int) -> None:
    """Copy the encoder prefix K/V into the buffer head — once per frame."""
    for i in range(DEC_L):
        kbufs[i][:prefix_len].copy_(enc_cache[i][0])
        vbufs[i][:prefix_len].copy_(enc_cache[i][1])


def decoder_step_fast_nocat(x_t, kbufs, vbufs, wf, style_attn, style_mlp,
                            style_top, prefix_len: int, chunk: int,
                            cos_t=None, sin_t=None, rope_kernel=None, ada_kernel=None):
    """decoder_step_fast without per-step K/V ``cat``: suffix rows are
    written in place into the preallocated buffers and one cross-attention
    runs over rows ``[0, prefix_len + chunk)``. Values are identical to the
    cat form (same rows, same order), so the numerics are unchanged."""
    dev = x_t.device
    use_tbl = cos_t is not None and sin_t is not None
    x = x_t
    pending = pending_gate = None
    for i in range(DEC_L):
        p = f"{_DP}.{i}"
        if ada_kernel is not None:
            gamma, shift, gate_a = style_attn[i]
            x_mod, x = ada_kernel(x, pending, pending_gate, gamma, shift)
        else:
            x_mod, gate_a = _ada_opt(x, style_attn[i])
        qkv = linear(x_mod, wf[f"{p}.qkv.weight"])
        total = prefix_len + chunk
        if rope_kernel is not None:
            q_rot = rope_kernel(qkv, cos_t, sin_t, kbufs[i], vbufs[i], prefix_len)
        else:
            q, k, v = qkv.split([DEC_NH * DEC_HD, DEC_HD, DEC_HD], dim=-1)
            if use_tbl:
                q_rot = rope_fast(q, cos_t, sin_t, prefix_len, DEC_HD)
                k_rot = rope_fast(k, cos_t, sin_t, prefix_len, DEC_HD)
            else:
                inv = _rope_inv_freqs(DEC_HD, device=dev)
                pos = torch.arange(prefix_len, prefix_len + chunk, device=dev)
                q_rot = rope_half_split(q, pos, inv, DEC_HD)
                k_rot = rope_half_split(k, pos, inv, DEC_HD)
            if kv_store_fused_enabled() and qkv.is_contiguous():
                _kv_cache_store_ext.kv_cache_store(
                    k_rot, qkv, kbufs[i], vbufs[i], prefix_len)
            else:
                kbufs[i][prefix_len:total].copy_(k_rot)
                vbufs[i][prefix_len:total].copy_(v)
        o = attention_flash(q_rot, kbufs[i][:total], vbufs[i][:total],
                            DEC_NH, DEC_NKV)
        o = linear(o, wf[f"{p}.self_attn.o_proj.weight"])
        if ada_kernel is not None:
            gamma, shift, gate_f = style_mlp[i]
            x_mod, x = ada_kernel(x, o, gate_a, gamma, shift)
        else:
            x = x + o * gate_a
            x_mod, gate_f = _ada_opt(x, style_mlp[i])
        gu = linear(x_mod, wf[f"{p}.gu.weight"])
        import torch_npu
        hidden = torch_npu.npu_geglu(gu, dim=-1, approximate=1, activate_left=True)[0]
        d = linear(hidden, wf[f"{p}.mlp.down_proj.weight"])
        if ada_kernel is not None:
            pending, pending_gate = d, gate_f
        else:
            x = x + d * gate_f
    if ada_kernel is not None:
        x_final, _ = ada_kernel(x, pending, pending_gate, style_top[0], style_top[1])
    else:
        x_final, _ = _ada_opt(x, style_top)
    return x_final


def make_rope_tables(capacity: int, hd: int, device) -> tuple:
    """Precomputed per-position cos/sin rows (capacity, hd) for the
    half-split rope. Built once outside the graph with the same arithmetic
    as ``pipeline.rope_half_split`` (identical values, so results are
    bit-identical to the eager path)."""
    half = hd // 2
    inv = _rope_inv_freqs(hd, device=device)
    pos = torch.arange(capacity, device=device).float().unsqueeze(1)
    emb = pos * inv.unsqueeze(0)                       # (capacity, hd/2)
    emb2 = torch.cat([emb, emb], dim=-1)               # (capacity, hd)
    return torch.cos(emb2), torch.sin(emb2)


def rope_fast(x: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor,
              pos0: int, hd: int) -> torch.Tensor:
    """Half-split RoPE from a cos/sin table (no per-call trig). Same
    rotation math as pipeline.rope_half_split; x (rows, D) with D % hd == 0."""
    R = x.shape[0]
    c = cos_t[pos0:pos0 + R][:, None, :]
    s = sin_t[pos0:pos0 + R][:, None, :]
    import torch_npu
    xr = x.to(torch.float32).reshape(1, R, -1, hd)
    return torch_npu.npu_rotary_mul(xr, c.unsqueeze(0), s.unsqueeze(0)).reshape(x.shape).to(x.dtype)


def attention_flash(q, k, v, nh: int, nkv: int) -> torch.Tensor:
    """Full (non-causal) attention via ``npu_prompt_flash_attention`` — one
    capturable aclnn node replacing the manual bmm/softmax chain. ``q`` is
    (Sq, nh*hd), ``k``/``v`` are (Skv, nkv*hd). Scale is 1/sqrt(hd)."""
    import torch_npu
    hd = q.shape[-1] // nh
    out = torch_npu.npu_prompt_flash_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        num_heads=nh, num_key_value_heads=nkv,
        scale_value=1.0 / math.sqrt(hd),
        pre_tokens=2147483647, next_tokens=2147483647,
        input_layout="BSH", sparse_mode=0)
    return out.reshape(q.shape)


def vision_tower_opt(images: torch.Tensor, w: dict, zeros: torch.Tensor):
    """SigLIP vision tower with fused LayerNorms.

    ``npu_layer_norm_eval`` (the natural fit) is an aclop operator and cannot
    be captured, so each LayerNorm runs as ``npu_add_layer_norm(x, zeros,
    gamma, beta, eps)[0]`` — same plain-LN result (fp32 gamma/beta, measured
    cos 1.0) but one capturable aclnn node instead of the ~7-op manual fp32
    chain. ``zeros`` is a persistent (S, VIS_D) bf16 zero tensor."""
    import torch_npu
    nv = images.shape[0]
    vp = _VP
    w_dtype = images.dtype
    pe_w = w[f"{vp}.embeddings.patch_embedding.weight"].to(torch.float32)
    pe_b = w[f"{vp}.embeddings.patch_embedding.bias"].to(torch.float32)
    imgs32 = images.to(torch.float32)
    blocks = imgs32.view(nv, 3, 16, 14, 16, 14).permute(0, 2, 4, 1, 3, 5)
    tokens = blocks.reshape(nv, VIS_TOKENS_PER_VIEW, 3 * 14 * 14).contiguous()
    x = torch.matmul(tokens.reshape(-1, 588), pe_w.reshape(VIS_D, -1).t())
    x = (x + pe_b).reshape(nv, VIS_TOKENS_PER_VIEW, VIS_D).to(w_dtype)
    x = x + w[f"{vp}.embeddings.position_embedding.weight"].unsqueeze(0)

    def ln(xn, lw, lb):
        # npu_add_layer_norm: bf16 x1/x2 with matching-dtype gamma/beta
        # (probe: fp32 gamma is also accepted with bf16 x; matching bf16 is
        # the safe combo across the model's stored dtypes).
        x2 = xn.to(torch.bfloat16).reshape(-1, xn.shape[-1])
        z2 = zeros.expand(x2.shape[0], xn.shape[-1])
        y = torch_npu.npu_add_layer_norm(
            x2, z2, lw.to(torch.bfloat16), lb.to(torch.bfloat16), EPS)[0]
        return y.to(xn.dtype).reshape(xn.shape)

    for i in range(VIS_L):
        xn = ln(x, w[f"{vp}.encoder.layers.{i}.layer_norm1.weight"],
                w[f"{vp}.encoder.layers.{i}.layer_norm1.bias"])
        q = linear(xn, w[f"{vp}.encoder.layers.{i}.self_attn.q_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.q_proj.bias"])
        k = linear(xn, w[f"{vp}.encoder.layers.{i}.self_attn.k_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.k_proj.bias"])
        v = linear(xn, w[f"{vp}.encoder.layers.{i}.self_attn.v_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.v_proj.bias"])
        o = torch_npu.npu_prompt_flash_attention(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            num_heads=16, num_key_value_heads=16, scale_value=72 ** -0.5,
            input_layout="BSH", pre_tokens=2147483647, next_tokens=2147483647,
            sparse_mode=0).to(q.dtype)
        o = linear(o, w[f"{vp}.encoder.layers.{i}.self_attn.out_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.out_proj.bias"])
        x = x + o
        res2 = x
        xn = ln(x, w[f"{vp}.encoder.layers.{i}.layer_norm2.weight"],
                w[f"{vp}.encoder.layers.{i}.layer_norm2.bias"])
        h = linear(xn, w[f"{vp}.encoder.layers.{i}.mlp.fc1.weight"],
                     w[f"{vp}.encoder.layers.{i}.mlp.fc1.bias"])
        h = F.gelu(h, approximate=GELU_TANH_APPROX)
        h = linear(h, w[f"{vp}.encoder.layers.{i}.mlp.fc2.weight"],
                     w[f"{vp}.encoder.layers.{i}.mlp.fc2.bias"])
        x = res2 + h
    x = ln(x, w[f"{vp}.post_layernorm.weight"],
           w[f"{vp}.post_layernorm.bias"])
    x = linear(x, w[f"{_MP}.weight"], w[f"{_MP}.bias"])
    return x.reshape(nv * VIS_TOKENS_PER_VIEW, ENC_D)
