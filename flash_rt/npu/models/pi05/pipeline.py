"""Pi0.5 Ascend NPU pipeline (plain torch ops, fp32 eager reference first).

Reimplements the openpi Pi0.5 model math faithfully in torch ops (no
custom kernels, no raw-pointer ABI) so the same weight tensors and code
run on a CPU fp32 reference and, after moving to an Ascend device, on
torch_npu. Numerics follow ``training/_vendor/openpi_pi0_pytorch/*``:

- vision tower  = SigLIP-L (27 layers), LayerNorm eps 1e-6, tanh-GELU
                  MLP, non-causal per-view attention, 14x14 conv patch
- encoder tower = Gemma-2B (18 layers), RMSNorm ``x*rms*(1+w)``,
                  half-split RoPE, GQA 8Q/1KV, bidirectional prefix
- action expert = Gemma-300M (18 layers), AdaRMSNorm style modulation
                  ([scale|shift|gate]), Euler flow matching
                  ``x += (-1/num_steps)*v`` over num_steps (default 10)
- the encoder runs once and yields 18 rotated K/V caches; every denoise
  step cross-attends to those caches.

B = 1 only (the LIBERO deployment shape). This fp32 eager path is the
correctness floor for the captured BF16 graph path in this same module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ── model geometry (fixed by the openpi pi0.5 config) ──────────────────
VIS_L, VIS_D, VIS_H, VIS_NH, VIS_HD = 27, 1152, 4304, 16, 72
VIS_PATCH, VIS_TOKENS_PER_VIEW = 14, 256
ENC_L, ENC_D, ENC_H, ENC_NH, ENC_NKV, ENC_HD = 18, 2048, 16384, 8, 1, 256
DEC_L, DEC_D, DEC_H, DEC_NH, DEC_NKV, DEC_HD = 18, 1024, 4096, 8, 1, 256
ACTION_DIM = 32
NUM_STEPS_DEFAULT = 10
CHUNK_DEFAULT = 10
IMG_HW = 224
EPS = 1e-6
ROPE_THETA = 10000.0
GELU_TANH_APPROX = "tanh"

_PALIGEMMA = "paligemma_with_expert.paligemma"
_VP = _PALIGEMMA + ".model.vision_tower.vision_model"
_MP = _PALIGEMMA + ".model.multi_modal_projector.linear"
_LANG = _PALIGEMMA + ".model.language_model.layers"
_EP = _LANG
_DP = "paligemma_with_expert.gemma_expert.model.layers"
_LM = _PALIGEMMA + ".lm_head.weight"
_TOP_EXP_NORM = "paligemma_with_expert.gemma_expert.model.norm.dense"


def load_weights_fp32(path: str) -> dict:
    """Load openpi Pi0.5 safetensors into a flat fp32 CPU dict.

    Handles both bare openpi keys and the lerobot ``model.``-wrapped
    layout (auto-detected). No fusion or folding: RMSNorm uses the live
    ``(1+w)`` semantics and RoPE the native half-split form, exactly as
    the vendored openpi reference computes them.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
    strip = all(k.startswith("model.") for k in keys)
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in keys:
            raw = f.get_tensor(k)
            out[k[len("model."):] if strip else k] = raw.to(torch.float32)
    return out


# ── building blocks (fp32 math wherever the reference is fp32) ────────
def layer_norm(x: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor) -> torch.Tensor:
    """Standard LayerNorm computed in fp32 (gain/bias kept fp32 like the
    reference recipe). Avoids aclnn's input/weight same-dtype constraint."""
    x32 = x.to(torch.float32)
    var = x32.var(dim=-1, keepdim=True, unbiased=False)
    y = (x32 - x32.mean(dim=-1, keepdim=True)) * torch.rsqrt(var + 1e-6)
    return (y * weight.to(torch.float32) + bias.to(torch.float32)).to(x.dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x * rsqrt(mean(x^2, -1)+eps) * (1 + w).  w is the stored gain-minus-1."""
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return ((x32 * torch.rsqrt(var + EPS)) * (1.0 + w.to(torch.float32))).to(x.dtype)


def ada_rms_norm(x, dense_w, dense_b, cond):
    """AdaRMSNorm. Returns (y, gate); gate is returned raw (no activation)."""
    x32 = x.to(torch.float32)
    cond32 = cond.to(torch.float32)
    mod = F.linear(cond32, dense_w.to(torch.float32), dense_b.to(torch.float32))
    scale, shift, gate = mod.chunk(3, dim=-1)   # order [scale | shift | gate]
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    xn = x32 * torch.rsqrt(var + EPS)
    y = (xn * (1.0 + scale) + shift).to(x.dtype)
    return y, gate.to(x.dtype)


def _rope_inv_freqs(hd: int, device=None) -> torch.Tensor:
    half = hd // 2
    return 1.0 / (ROPE_THETA ** (torch.arange(half, device=device).float() / half))


def rope_half_split(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor, hd: int):
    """Half-split RoPE. x is (S, D) with D % hd == 0; rotation within each
    hd-sized head; pos (S,)."""
    x32 = x.to(torch.float32)
    inv = inv.to(x.device)
    emb = (pos.to(x.device).float().unsqueeze(1) * inv.unsqueeze(0))  # (S, hd/2)
    emb = torch.cat([emb, emb], dim=-1)                  # (S, hd)
    cos = torch.cos(emb)[:, None, :]                     # (S, 1, hd)
    sin = torch.sin(emb)[:, None, :]
    shape = x32.shape
    xr = x32.reshape(shape[0], -1, hd)                   # (S, nhead, hd)
    half = hd // 2
    x_rot = torch.cat([-xr[..., half:], xr[..., :half]], dim=-1)
    y = (xr * cos + x_rot * sin).reshape(shape)
    return y.to(x.dtype)


def attention(q, k, v, mask=None, scale=None):
    """q/k/v as (H, S, hd). Returns (S, H*hd). mask (Sq, Sk) additive or None.

    Manual bmm/softmax/bmm **in the input dtype** (bf16 on NPU, fp32 on the
    CPU reference where the inputs already are fp32) — the only attention
    path proven safe to capture with ``torch.npu.graph``. Dropping the old
    per-call fp32 upcasts removes three dtype-conversion passes and runs the
    logits/AV GEMMs on the bf16 Cube path (~204 TF here) instead of the
    ~44 TF fp32 path. NPU GEMM kernels accumulate in fp32 internally; the
    fp32 reference gate holds accuracy.
    """
    hd = q.shape[-1]
    scale = 1.0 / math.sqrt(hd) if scale is None else scale
    logits = torch.bmm(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        logits = logits + mask.unsqueeze(0).to(logits.dtype)
    att = torch.softmax(logits, dim=-1)
    o = torch.bmm(att, v)
    return o.transpose(0, 1).reshape(o.shape[1], -1).to(q.dtype)


def vision_attention(q, k, v, num_heads: int = VIS_NH):
    """Independent image attention with explicit batch, token and head axes.

    Input and output are (views, tokens, width). Folding contiguous tokens
    directly into heads changes which tokens attend to one another.
    """
    batch, tokens, width = q.shape
    head_dim = width // num_heads
    def heads(x):
        return x.reshape(batch, tokens, num_heads, head_dim).transpose(1, 2)
    qh, kh, vh = heads(q), heads(k), heads(v)
    logits = (qh.float() @ kh.float().transpose(-1, -2)) * (head_dim ** -0.5)
    out = logits.softmax(dim=-1).to(vh.dtype) @ vh
    return out.transpose(1, 2).reshape(batch, tokens, width)


def time_embedding(t: float, dim: int) -> torch.Tensor:
    """openpi sinusoidal time embedding (fp64), fp32 (dim,)."""
    half = dim // 2
    frac = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
    period = 4e-3 * ((4.0 / 4e-3) ** frac)
    freq = 2.0 * math.pi / period
    arg = torch.tensor(t, dtype=torch.float64) * freq
    return torch.cat([torch.sin(arg), torch.cos(arg)]).to(torch.float32)


# ── SigLIP vision tower ────────────────────────────────────────────────
def vision_tower(images: torch.Tensor, w: dict) -> torch.Tensor:
    """images (nv,3,224,224) in [-1,1] → (nv*256, 2048)."""
    nv = images.shape[0]
    vp = _VP
    w_dtype = images.dtype
    # Patch embed via reshape/im2col + matmul instead of conv2d. Ascend's
    # conv2d is an aclop operator and cannot be captured by torch.npu.graph
    # (full-frame capture is the point of this backend). The reshape path is
    # reduction-order-identical to the conv and stays in fp32 like the conv
    # did (see notes in docs/pi05_npu_numerics.md).
    pe_w = w[f"{vp}.embeddings.patch_embedding.weight"].to(torch.float32)  # (1152,3,14,14)
    pe_b = w[f"{vp}.embeddings.patch_embedding.bias"].to(torch.float32)
    imgs32 = images.to(torch.float32)                                     # (nv,3,224,224)
    blocks = imgs32.view(nv, 3, 16, 14, 16, 14).permute(0, 2, 4, 1, 3, 5)
    tokens = blocks.reshape(nv, VIS_TOKENS_PER_VIEW, 3 * 14 * 14).contiguous()
    x = torch.matmul(tokens.reshape(-1, 588), pe_w.reshape(VIS_D, -1).t())
    x = (x + pe_b).reshape(nv, VIS_TOKENS_PER_VIEW, VIS_D).to(w_dtype)
    x = x + w[f"{vp}.embeddings.position_embedding.weight"].unsqueeze(0)

    for i in range(VIS_L):
        ln1w = w[f"{vp}.encoder.layers.{i}.layer_norm1.weight"]
        ln1b = w[f"{vp}.encoder.layers.{i}.layer_norm1.bias"]
        xn = layer_norm(x, ln1w, ln1b)
        q = F.linear(xn, w[f"{vp}.encoder.layers.{i}.self_attn.q_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.q_proj.bias"])
        k = F.linear(xn, w[f"{vp}.encoder.layers.{i}.self_attn.k_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.k_proj.bias"])
        v = F.linear(xn, w[f"{vp}.encoder.layers.{i}.self_attn.v_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.v_proj.bias"])
        o = vision_attention(q, k, v)
        o = F.linear(o, w[f"{vp}.encoder.layers.{i}.self_attn.out_proj.weight"],
                     w[f"{vp}.encoder.layers.{i}.self_attn.out_proj.bias"])
        x = x + o
        res2 = x
        ln2w = w[f"{vp}.encoder.layers.{i}.layer_norm2.weight"]
        ln2b = w[f"{vp}.encoder.layers.{i}.layer_norm2.bias"]
        xn = layer_norm(x, ln2w, ln2b)
        h = F.linear(xn, w[f"{vp}.encoder.layers.{i}.mlp.fc1.weight"],
                     w[f"{vp}.encoder.layers.{i}.mlp.fc1.bias"])
        h = F.gelu(h, approximate=GELU_TANH_APPROX)
        h = F.linear(h, w[f"{vp}.encoder.layers.{i}.mlp.fc2.weight"],
                     w[f"{vp}.encoder.layers.{i}.mlp.fc2.bias"])
        x = res2 + h
    x = layer_norm(x, w[f"{vp}.post_layernorm.weight"],
                   w[f"{vp}.post_layernorm.bias"])
    x = F.linear(x, w[f"{_MP}.weight"], w[f"{_MP}.bias"])
    return x.reshape(nv * VIS_TOKENS_PER_VIEW, ENC_D)


def embed_language(tokens: torch.Tensor, w: dict) -> torch.Tensor:
    """tokens (L,) → (L, 2048) embeddings scaled by sqrt(hidden)."""
    emb = F.embedding(tokens, w[_LM])
    return emb * math.sqrt(ENC_D)


# ── Gemma-2B encoder (prefix pass, builds K/V caches) ──────────────────
def encoder_pass(prefix_emb: torch.Tensor, w: dict):
    """prefix_emb (S,2048), all tokens valid (eager, no padding).
    Returns list of 18 (K_rot (S,256), V (S,256))."""
    S = prefix_emb.shape[0]
    dev = prefix_emb.device
    inv = _rope_inv_freqs(ENC_HD, device=dev)
    pos = torch.arange(S, device=dev)
    x = prefix_emb
    cache = []
    for i in range(ENC_L):
        xn = rms_norm(x, w[f"{_EP}.{i}.input_layernorm.weight"])
        q = F.linear(xn, w[f"{_EP}.{i}.self_attn.q_proj.weight"])   # (S,2048)
        k = F.linear(xn, w[f"{_EP}.{i}.self_attn.k_proj.weight"])   # (S,256)
        v = F.linear(xn, w[f"{_EP}.{i}.self_attn.v_proj.weight"])
        q_rot = rope_half_split(q, pos, inv, ENC_HD)
        k_rot = rope_half_split(k, pos, inv, ENC_HD)
        # q has 8 heads, k/v 1 head (repeat to 8 for the manual bmm path)
        qh = q_rot.reshape(S, ENC_NH, ENC_HD).transpose(0, 1)
        kh = k_rot.reshape(S, ENC_NKV, ENC_HD).transpose(0, 1).expand(ENC_NH, S, ENC_HD)
        vh = v.reshape(S, ENC_NKV, ENC_HD).transpose(0, 1).expand(ENC_NH, S, ENC_HD)
        o = attention(qh, kh, vh)                                    # (S,2048)
        o = F.linear(o, w[f"{_EP}.{i}.self_attn.o_proj.weight"])
        x = x + o
        xn = rms_norm(x, w[f"{_EP}.{i}.post_attention_layernorm.weight"])
        g = F.linear(xn, w[f"{_EP}.{i}.mlp.gate_proj.weight"])
        u = F.linear(xn, w[f"{_EP}.{i}.mlp.up_proj.weight"])
        d = F.linear(F.gelu(g, approximate=GELU_TANH_APPROX) * u,
                     w[f"{_EP}.{i}.mlp.down_proj.weight"])
        x = x + d
        cache.append((k_rot.contiguous(), v.contiguous()))
    return cache


# ── Gemma-300M action expert (one denoise step) ───────────────────────
def _decoder_step(x_t, enc_cache, cond, w, prefix_len: int, chunk: int):
    """One decoder forward over the whole chunk. x_t (chunk,1024) is the
    action projection. enc_cache: 18 (K_rot,V) each (prefix_len,256).
    cond (1024,). Returns (chunk,1024)."""
    dev = x_t.device
    inv = _rope_inv_freqs(DEC_HD, device=dev)
    pos = torch.arange(prefix_len, prefix_len + chunk, device=dev)  # suffix positions
    x = x_t
    for i in range(DEC_L):
        dense_w = w[f"{_DP}.{i}.input_layernorm.dense.weight"]
        dense_b = w[f"{_DP}.{i}.input_layernorm.dense.bias"]
        x_mod, gate_a = ada_rms_norm(x, dense_w, dense_b, cond)
        q = F.linear(x_mod, w[f"{_DP}.{i}.self_attn.q_proj.weight"])  # (C,2048)
        k = F.linear(x_mod, w[f"{_DP}.{i}.self_attn.k_proj.weight"])  # (C,256)
        v = F.linear(x_mod, w[f"{_DP}.{i}.self_attn.v_proj.weight"])
        q_rot = rope_half_split(q, pos, inv, DEC_HD)
        k_rot = rope_half_split(k, pos, inv, DEC_HD)
        k_full = torch.cat([enc_cache[i][0], k_rot], dim=0)          # (P+C,256)
        v_full = torch.cat([enc_cache[i][1], v], dim=0)
        qh = q_rot.reshape(chunk, DEC_NH, DEC_HD).transpose(0, 1)
        kh = k_full.reshape(-1, DEC_NKV, DEC_HD).transpose(0, 1).expand(
            DEC_NH, -1, DEC_HD)
        vh = v_full.reshape(-1, DEC_NKV, DEC_HD).transpose(0, 1).expand(
            DEC_NH, -1, DEC_HD)
        o = attention(qh, kh, vh)                                   # (C,2048)
        o = F.linear(o, w[f"{_DP}.{i}.self_attn.o_proj.weight"])    # →(C,1024)
        x = x + o * gate_a
        dense_w = w[f"{_DP}.{i}.post_attention_layernorm.dense.weight"]
        dense_b = w[f"{_DP}.{i}.post_attention_layernorm.dense.bias"]
        x_mod, gate_f = ada_rms_norm(x, dense_w, dense_b, cond)
        g = F.linear(x_mod, w[f"{_DP}.{i}.mlp.gate_proj.weight"])
        u = F.linear(x_mod, w[f"{_DP}.{i}.mlp.up_proj.weight"])
        d = F.linear(F.gelu(g, approximate=GELU_TANH_APPROX) * u,
                     w[f"{_DP}.{i}.mlp.down_proj.weight"])
        x = x + d * gate_f
    dense_w = w[f"{_TOP_EXP_NORM}.weight"]
    dense_b = w[f"{_TOP_EXP_NORM}.bias"]
    x_final, _ = ada_rms_norm(x, dense_w, dense_b, cond)
    return x_final


# ── top-level flow-matching sample ────────────────────────────────────
def sample(images: torch.Tensor, prompt_tokens: torch.Tensor, noise: torch.Tensor,
           w: dict, num_steps: int = NUM_STEPS_DEFAULT) -> torch.Tensor:
    """images (nv,3,224,224) in [-1,1]; prompt_tokens (L,); noise (chunk,32) fp32
    (or None → fresh randn on images device). Returns normalized actions
    (chunk, ACTION_DIM) fp32."""
    dev = images.device
    nv = images.shape[0]
    chunk = noise.shape[0] if noise is not None else CHUNK_DEFAULT
    if noise is None:
        noise = torch.randn(chunk, ACTION_DIM, device=dev)
    x_t = noise.to(torch.float32)

    vis = vision_tower(images, w)                              # (nv*256, 2048)
    lang = embed_language(prompt_tokens.to(dev), w)            # (L, 2048)
    prefix_emb = torch.cat([vis, lang], dim=0)                 # (S, 2048)
    prefix_len = prefix_emb.shape[0]
    enc_cache = encoder_pass(prefix_emb, w)

    dt = -1.0 / num_steps
    time = 1.0
    for _ in range(num_steps):
        te = time_embedding(time, DEC_D).to(dev)               # (1024,)
        cond = F.linear(te, w["time_mlp_in.weight"], w["time_mlp_in.bias"])
        cond = F.silu(cond)
        cond = F.linear(cond, w["time_mlp_out.weight"], w["time_mlp_out.bias"])
        cond = F.silu(cond)
        act = F.linear(x_t, w["action_in_proj.weight"], w["action_in_proj.bias"])
        out = _decoder_step(act, enc_cache, cond, w, prefix_len, chunk)
        v_t = F.linear(out, w["action_out_proj.weight"], w["action_out_proj.bias"])
        x_t = x_t + dt * v_t
        time += dt
    return x_t
