"""FlashRT -- Nex-N2-mini (qwen3_5_moe) kernelized NVFP4 forward.

Production forward that drives the fvk kernels off the pre-quantized
:class:`WeightHandles` produced by ``extract_weights_nexn2_nvfp4`` -- the
load-once seam that replaces the Phase-1 HF shim. It reproduces the
component-validated assembly (GDN recurrent, full GQA attn, fine-grained
MoE) used to lock the golden cosine fixture, sourcing every weight from
the loader instead of re-reading safetensors.

Compute split (this milestone -- prefill, S>1):
  * NVFP4 W4A16 GEMMs (full-attn q/k/v/o, GDN out_proj, MoE routed +
    shared experts): ``quantize_bf16_to_nvfp4_swizzled`` +
    ``fp4_w4a16_gemm_sm120_bf16out`` off the loader's packed / SF / alpha.
  * GDN gating / recurrent / gated-norm, partial RoPE: shared fvk kernels
    (parameterised, validated at the Nex-N2 head counts).
  * BF16-kept projections (GDN in_proj -- red line, router, shared gate,
    embed, lm_head): batched cuBLAS matmul (fp32 accumulate).
  * Glue still on torch (kernelised in the decode/graph milestone, where
    the nexn2-specific dims need new .cu): conv1d, 16->32 q/k broadcast,
    q/gate split, causal SDPA, MoE routing, residual adds.

All fvk pointer args bind to named tensors first -- an inline
``x.to(bf16).contiguous().data_ptr()`` temporary is GC'd before the
kernel launches and reads freed memory (validated regression: 0.479 vs
1.0). See feedback_ctypes_temp_tensor_gc.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import _sf_swz_bytes

# Static Nex-N2-mini dims (config.json:text_config). Kept module-local so
# the forward reads like the validation script it was lifted from.
HID = 2048
NK, NV, HK, HV, KS = 16, 32, 128, 128, 4      # GDN: 16 K-heads / 32 V-heads
KD, VD = NK * HK, NV * HV                       # 2048 / 4096
CONV = KD + KD + VD                             # 8192 in_proj_qkv conv channels
NQ, NKV, HD, ROPE = 16, 2, 256, 64              # full-attn GQA + partial rope
INTER, TOPK = 512, 8


def build_rope_tables(seq_len, theta, rope_dim, device):
    """(cos, sin) each (S, rope_dim) bf16 -- HF cat([freqs, freqs]) layout."""
    inv = 1.0 / (theta ** (
        torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    ang = torch.arange(seq_len, device=device).float()[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _rms(x, w, eps):
    """Plain (already (1+w)-folded) RMSNorm in fp32, bf16 out."""
    v = x.float().pow(2).mean(-1, keepdim=True)
    return ((x.float() * torch.rsqrt(v + eps)) * w.float()).to(torch.bfloat16)


def _proj(x2d, ld, base, n, fvk, device):
    """y = x @ w.T for one projection, dispatching on the loader's scope.

    NVFP4 site (``<base>_packed`` present) -> fp4 W4A16 GEMM. Otherwise the
    weight was kept BF16 (``<base>_w_t``, quant_scope='experts') -> cuBLAS
    matmul with fp32 accumulate. ``n`` is ignored on the BF16 path (taken
    from the weight).
    """
    if ld.get(base + '_packed') is not None:
        return _nvfp4_gemm(x2d, ld[base + '_packed'], ld[base + '_sf'],
                           ld[base + '_alpha'], n, fvk, device)
    w = ld[base + '_w_t']
    if (_DENSE_W4A16 and x2d.shape[0] >= 64 and (w.shape[0] % 64) == 0
            and (x2d.shape[1] % 64) == 0):
        return _gemm_w4a16(x2d, w, ld, base + '_w_t', fvk, device)
    if (_DENSE_W16A16 and x2d.shape[0] >= _DENSE_BF16_MIN_M
            and (x2d.shape[1] % 64) == 0):
        return _gemm_w16a16(x2d, w, fvk, device)
    if _DENSE_FP4:
        return _gemm_fp4(x2d, w, ld, base + '_w_t', fvk, device)
    if _DENSE_BF16 and x2d.shape[0] >= _DENSE_BF16_MIN_M:
        # BF16 cuBLAS matmul. Faster than the hand-tuned kernel but its split-K
        # accumulation is non-deterministic (flips near-tie argmaxes run-to-run,
        # breaking the token-exact red line) -- kept only for A/B, not default.
        return (x2d.to(torch.bfloat16) @ w.t()).to(torch.bfloat16)
    return (x2d.float() @ w.float().T).to(torch.bfloat16)


def _gemm_w16a16(x2d, w, fvk, device):
    """y = x @ w.T via the deterministic bf16-act x bf16-weight tensor-core
    GEMM (fp32 register accumulate). Matches the fp32 path's argmax (cos 1.0)
    and is bit-identical run-to-run, at ~1.75x the fp32/TF32 op."""
    m, k = x2d.shape
    n = w.shape[0]
    xc = x2d.contiguous()
    wc = w.contiguous()
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.w16a16_gemm_sm120_bf16(xc.data_ptr(), wc.data_ptr(), y.data_ptr(),
                               m, n, k, 1.0, 0)
    return y


# True W4A16 (bf16 activation x fp4 weight) GEMM for the large prefill dense
# projections: 2.18x the fp32 path. BUT the cos cost of the dense proj is the
# fp4 *weight* (not the activation), so W4A16 lands at the same ~0.987 as W4A4
# while being slower than the CUTLASS W4A4 -- dominated. Default OFF; the 0.994
# path needs a bf16-*weight* GEMM (repurpose this kernel's 2.18x structure).
_DENSE_W4A16 = False

# BF16 tensor-core dense projections (vs the default fp32/TF32 matmul). The
# experts-scope q/k/v/o/out/shared/router projections dominate the prefill
# profile as fp32 GEMMs; bf16 inputs with fp32 accumulate roughly halve that
# bucket at near-identical cos (no fp4 weight rounding -> stays ~0.994).
# Gated to M >= _DENSE_BF16_MIN_M (decode M=1 is weight-bound -> fp4 wins).
_DENSE_BF16 = False
_DENSE_BF16_MIN_M = 64

# Deterministic hand-tuned bf16-act x bf16-weight tensor-core GEMM (fp32 reg
# accumulate, single pass over K). Matches the fp32 path's argmax (cos 1.0),
# bit-identical run-to-run, ~1.75x the fp32/TF32 op -- so it replaces the fp32
# dense projections at prefill M with no precision or determinism cost. Default
# ON; decode (M=1) stays on its own GEMV (weight-bound -> fp4). Set False to
# fall back to the fp32 path. (K must be a multiple of 64.)
_DENSE_W16A16 = True


def _gemm_w4a16(x2d, w, ld, key, fvk, device):
    """y = x @ w.T via the bf16-act x fp4-weight tensor-core GEMM. Weight
    quantised to NVFP4 once (cached); activation stays BF16 (precise)."""
    m, k = x2d.shape
    p, s, a = _wquant(w, ld, key, fvk, device)
    n = p.shape[0]
    xc = x2d.contiguous()
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.w4a16_gemm_sm120_bf16(xc.data_ptr(), p.data_ptr(), s.data_ptr(),
                              y.data_ptr(), m, n, k, a, 0)
    return y


# The experts-scope dense projections run as `.float() @ .float().T` -- a full
# fp32 (TF32) tensor-op which on sm120 is ~10x slower than the fp4 block-scaled
# GEMM. Routing the non-red-line projections (full-attn q/k/v/o, GDN out_proj,
# shared expert) through the fp4 W4A4 GEMM (weight quantised once) crosses the
# llama.cpp 9.5k prefill target (9686 tok/s) but the fp4 *activation* drops cos
# to 0.987 (within the 0.984 red line, tight). Off by default; the true W4A16
# kernel (w4a16_gemm_sm120, bf16 activation, correct but needs occupancy tuning
# to beat the fp32 path) is the precision-preserving replacement in progress.
_DENSE_FP4 = False


def _wquant(w, ld, key, fvk, device):
    """Quantise a bf16 weight to swizzled NVFP4 once, cached on ld[key+'_w4*']."""
    pk = key + '_w4p'
    if pk not in ld:
        nn, kk = w.shape
        p = torch.empty(nn, kk // 2, dtype=torch.uint8, device=device)
        s = torch.zeros(_sf_swz_bytes(nn, kk), dtype=torch.uint8, device=device)
        scr = torch.zeros(1, dtype=torch.float32, device=device)
        og = torch.zeros(1, dtype=torch.float32, device=device)
        fvk.bf16_weight_to_nvfp4_swizzled(
            w.contiguous().data_ptr(), p.data_ptr(), s.data_ptr(),
            scr.data_ptr(), og.data_ptr(), nn, kk, 0)
        torch.cuda.synchronize()
        ld[pk] = p
        ld[key + '_w4s'] = s
        ld[key + '_w4a'] = float(og.item())
    return ld[pk], ld[key + '_w4s'], ld[key + '_w4a']


def _gemm_fp4(x2d, w, ld, key, fvk, device, xp=None, xsf=None):
    """y = x @ w.T via the fp4 block-scaled GEMM (W4A4, bf16 out). 6-10x the
    fp32 path at large M. Weight quantised once (cached); activation quantised
    per call unless (xp, xsf) are supplied (shared across same-input projs)."""
    m, k = x2d.shape
    p, s, a = _wquant(w, ld, key, fvk, device)
    n = ld[key + '_w4p'].shape[0]
    if xp is None:
        xp, xsf = _quant_act(x2d, fvk, device)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.fp4_w4a16_gemm_sm120_bf16out(
        xp.data_ptr(), p.data_ptr(), y.data_ptr(), m, n, k,
        xsf.data_ptr(), s.data_ptr(), a, 0)
    return y


def _quant_act(x2d, fvk, device, stream=0):
    """Quantise one (M,K) bf16 activation to NVFP4 swizzled; return (xp, xsf).

    Split out so a shared activation (e.g. the M=1 decode token routed to
    several experts) is quantised once and reused across GEMMs.
    """
    m, kk = x2d.shape
    xc = x2d.contiguous()
    xp = torch.empty(m, kk // 2, dtype=torch.uint8, device=device)
    xsf = torch.zeros(_sf_swz_bytes(m, kk), dtype=torch.uint8, device=device)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        xc.data_ptr(), xp.data_ptr(), xsf.data_ptr(), m, kk, stream)
    return xp, xsf


def _nvfp4_gemm_preq(xp, xsf, wp_ptr, wsf_ptr, alpha, m, n, k, fvk, device,
                     stream=0):
    """y = x @ w.T from a pre-quantised activation (xp, xsf)."""
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.fp4_w4a16_gemm_sm120_bf16out(
        xp.data_ptr(), wp_ptr, y.data_ptr(), m, n, k,
        xsf.data_ptr(), wsf_ptr, alpha, stream)
    return y


def _nvfp4_gemm(x2d, wp_ptr, wsf_ptr, alpha, n, fvk, device, stream=0):
    """y = x @ w.T via NVFP4. x2d is (M,K) bf16; weight given by ptrs+alpha.

    Activation quantised per call (swizzled). All ptr args are bound to
    named tensors that outlive the launch.
    """
    m, kk = x2d.shape
    xp, xsf = _quant_act(x2d, fvk, device, stream)
    return _nvfp4_gemm_preq(xp, xsf, wp_ptr, wsf_ptr, alpha, m, n, kk,
                            fvk, device, stream)


def _silu_mul(g, u, fvk, device):
    """out = silu(g) * u via one fused kernel (was 4 torch ops). g, u bf16."""
    n = g.numel()
    gc = g.reshape(-1).contiguous()
    uc = u.reshape(-1).contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    fvk.silu_mul_sm120_bf16(gc.data_ptr(), uc.data_ptr(), out.data_ptr(), n, 0)
    return out.reshape(g.shape)


# WY chunked gated-delta-rule scan for the GDN prefill. The seq-scan kernel is
# O(S) sequential per head with only NV=32 blocks (19% occupancy on the 5090),
# so it is the prefill wall (~9.4 ms/layer at S=2048). The WY/UT chunked form
# (FLA delta rule) runs the intra-chunk work as tensor-core matmuls and only
# the inter-chunk state recurrence is sequential -> 11x faster at S=2048,
# bit-exact (out cos 0.99998, state cos 0.99997 vs the seq-scan). Default on.
import os as _os
_USE_WY_GDN = _os.environ.get('NEXN2_WY_GDN', '1') != '0'
_WY_MIN_S = 64        # below this the seq-scan's lower fixed overhead wins


def _wy_pack_t(x, ch=64):
    """(S, H, D) -> (chunks, H, ch, D): x_pack[ci, h, i, d] = x[ci*ch+i, h, d]
    (zero-padded last chunk). The packed chunk-major layout the mma kernels read."""
    s, hh, d = x.shape
    pad = (-s) % ch
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
    return x.reshape(-1, ch, hh, d).permute(0, 2, 1, 3).contiguous()


def _wy_l2(x):
    """l2norm over the last dim, eps inside rsqrt (matches the seq-scan kEps)."""
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16)


def _wy_gcumsum(g, ch=64):
    """(S, NV) -> (S, NV) per-chunk cumulative sum of the (log-space) gate."""
    s = g.shape[0]
    pad = (-s) % ch
    gp = F.pad(g, (0, 0, 0, pad)) if pad else g
    return torch.cumsum(gp.float().reshape(-1, ch, g.shape[1]), 1).reshape(
        -1, g.shape[1])[:s].to(torch.bfloat16)


def _gdn_wy_chunk(q16, k16, v, g, beta, fvk, device):
    """WY chunked scan. q16/k16 (S,16,128) raw post-conv, v (S,32,128),
    g/beta (S,32). Returns core (S,32,128) + final state (32,128,128).

    Pipeline (FLA chunked delta rule, all add-only existing kernels):
    l2norm + per-chunk g-cumsum (torch glue) -> kkt -> solve_tril(+pack) ->
    recompute_wu -> chunk_h (inter-chunk state) -> output_o."""
    S = q16.shape[0]
    chunks = (S + 63) // 64
    CH, QKG = 64, NV // NK
    q_l2 = _wy_l2(q16)
    k_l2 = _wy_l2(k16).contiguous()
    gc = _wy_gcumsum(g).contiguous()
    betac = beta.contiguous()
    vc = v.contiguous()

    k_pack = torch.empty(chunks, NK, CH, HK, dtype=torch.bfloat16, device=device)
    kkt_base = torch.empty(chunks, NK, CH, CH, dtype=torch.float32, device=device)
    A = torch.empty(chunks, NV, CH, CH, dtype=torch.float32, device=device)
    fvk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
        k_l2.data_ptr(), betac.data_ptr(), gc.data_ptr(), k_pack.data_ptr(),
        kkt_base.data_ptr(), A.data_ptr(), S, NK, NV, HK, QKG, 0)

    Ai = torch.empty(chunks, NV, CH, CH, dtype=torch.float32, device=device)
    Ai_pack = torch.empty(chunks, NV, CH, CH, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_solve_tril_b64_f32_parallel_pack(
        A.data_ptr(), Ai.data_ptr(), Ai_pack.data_ptr(), S, NV, 0)

    w_pack = torch.empty(chunks, NV, CH, HV, dtype=torch.bfloat16, device=device)
    u_pack = torch.empty(chunks, NV, CH, HV, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_recompute_wu_b64_bf16_mma_fla(
        k_l2.data_ptr(), vc.data_ptr(), betac.data_ptr(), gc.data_ptr(),
        Ai_pack.data_ptr(), w_pack.data_ptr(), u_pack.data_ptr(),
        S, NK, NV, HK, QKG, 0)

    state = torch.zeros(NV, HK, HV, dtype=torch.bfloat16, device=device)
    h0 = torch.empty(chunks, NV, HK, HV, dtype=torch.bfloat16, device=device)
    v_new = torch.empty(S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla(
        k_l2.data_ptr(), w_pack.data_ptr(), u_pack.data_ptr(), gc.data_ptr(),
        state.data_ptr(), h0.data_ptr(), v_new.data_ptr(), 0, 0,
        S, NK, NV, HK, QKG, 0)

    q_pack = _wy_pack_t(q_l2.repeat_interleave(QKG, 1))
    k_pack_hv = _wy_pack_t(k_l2.repeat_interleave(QKG, 1))
    v_pack = _wy_pack_t(v_new)
    core = torch.empty(S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.linear_attn_gdn_wy_output_o_b64_bf16_mma_fla(
        q_pack.data_ptr(), k_pack_hv.data_ptr(), v_pack.data_ptr(),
        h0.data_ptr(), gc.data_ptr(), core.data_ptr(),
        S, NV, HV, float(HV ** -0.5), 0)
    return core, state


def _gdn_layer(h, ld, fvk, device, eps, cap=None, rank=None):
    """Gated DeltaNet (linear_attention) layer. h (1,S,HID) -> (1,S,HID).

    When ``cap`` is given (a Nexn2DecodeState), the final recurrent state and
    the last KS-1 conv inputs are written into its decode buffers so a batched
    prefill leaves exactly the state the per-token decode path would.
    """
    B, S, _ = h.shape
    Wqkv = ld['in_proj_qkv_w_t']
    Wz = ld['in_proj_z_w_t']
    Wb, Wa = ld['in_proj_b_w_t'], ld['in_proj_a_w_t']
    convw = ld['conv1d_w_t']
    A_log, dtb = ld['A_log_t'].float(), ld['dt_bias_t'].float()
    nw = ld['gdn_norm_w_t']

    # in_proj must NOT be quantized (red line: fp4 weight/act collapses GDN).
    # The big mixed (N=CONV) and z (N=NV*HV) projections route through the
    # deterministic bf16-weight w16a16 GEMM -- bf16 weight + bf16 act + fp32
    # accumulate is the same precision as the fp32 path (no quantization), just
    # on bf16 tensor cores. The tiny b/a (N=NV=32) stay on the fp32 matmul.
    h2 = h.reshape(B * S, HID)
    if _DENSE_W16A16 and (B * S) >= _DENSE_BF16_MIN_M:
        mixed = _gemm_w16a16(h2, Wqkv, fvk, device).reshape(B, S, -1)
        z = _gemm_w16a16(h2, Wz, fvk, device).reshape(B, S, NV, HV)
    else:
        mixed = (h.float() @ Wqkv.float().T).to(torch.bfloat16)
        z = (h.float() @ Wz.float().T).reshape(B, S, NV, HV)
    b = h.float() @ Wb.float().T
    a = h.float() @ Wa.float().T

    # causal depthwise conv1d + silu (torch glue; nexn2 dim=8192).
    xc = F.silu(F.conv1d(mixed.transpose(1, 2).float(), convw.float(),
                         groups=CONV, padding=KS - 1)[:, :, :S]).transpose(1, 2)
    # split conv output + broadcast q/k 16 -> 32 heads in one fvk kernel.
    xc_bf = xc.to(torch.bfloat16).reshape(B * S, CONV).contiguous()
    qb = torch.empty(B, S, NV, HK, dtype=torch.bfloat16, device=device)
    kb = torch.empty(B, S, NV, HK, dtype=torch.bfloat16, device=device)
    vb = torch.empty(B, S, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_lin_split_qkv_broadcast_bf16(
        xc_bf.data_ptr(), qb.data_ptr(), kb.data_ptr(), vb.data_ptr(),
        B * S, 0)

    neg = (-A_log.exp()).float().contiguous()
    dtb_c = dtb.contiguous()
    a_bf = a.to(torch.bfloat16).contiguous()
    b_bf = b.to(torch.bfloat16).contiguous()
    g_out = torch.empty(B, S, NV, dtype=torch.bfloat16, device=device)
    bo = torch.empty(B, S, NV, dtype=torch.bfloat16, device=device)
    fvk.qwen36_gdn_gating_bf16(
        a_bf.data_ptr(), b_bf.data_ptr(), neg.data_ptr(), dtb_c.data_ptr(),
        g_out.data_ptr(), bo.data_ptr(), B * S, NV, 0)

    if _USE_WY_GDN and S >= _WY_MIN_S:
        # WY chunked delta-rule scan: 11x faster than the seq-scan at S=2048,
        # bit-exact. qb/kb are the 16->32 broadcast heads (src_h = h//2), so the
        # 16 unique K-heads are the even slots; the WY kernels re-expand by GQA.
        q16 = qb.reshape(S, NV, HK)[:, 0::2, :]
        k16 = kb.reshape(S, NV, HK)[:, 0::2, :]
        core, state = _gdn_wy_chunk(
            q16, k16, vb.reshape(S, NV, HV), g_out.reshape(S, NV),
            bo.reshape(S, NV), fvk, device)
        core = core.reshape(B, S, NV, HV)
    else:
        # Sequential scan over the whole prompt in ONE launch (state stays in
        # registers across all S timesteps -> no per-token state HBM round-trip
        # / S kernel launches). Bit-equivalent to the per-token recurrent loop
        # (out cos 0.99999); the short-prompt fallback below _WY_MIN_S.
        state = torch.zeros(NV, HK, HV, dtype=torch.bfloat16, device=device)
        core = torch.empty(S, NV, HV, dtype=torch.bfloat16, device=device)
        fvk.gdn_recurrent_seq_sm120_bf16(
            qb.reshape(S, NV, HK).contiguous().data_ptr(),
            kb.reshape(S, NV, HK).contiguous().data_ptr(),
            vb.reshape(S, NV, HV).contiguous().data_ptr(),
            g_out.reshape(S, NV).contiguous().data_ptr(),
            bo.reshape(S, NV).contiguous().data_ptr(),
            state.data_ptr(), core.data_ptr(), S, NV, HK, True, 0)
        core = core.reshape(B, S, NV, HV)

    if cap is not None:
        # GDN recurrent final state = `state` after the S-step scan; conv state
        # = the last KS-1 `mixed` inputs (channel-major, newest at index -1),
        # matching the causal_conv1d_update rolling buffer (1, CONV, KS-1).
        cap.lin_state[rank].copy_(state)
        cs = mixed[0, S - (KS - 1):S, :].transpose(0, 1).contiguous()
        cap.lin_conv_state[rank].copy_(cs.unsqueeze(0))

    cf = core.reshape(-1, HV).contiguous()
    zf = z.reshape(-1, HV).to(torch.bfloat16).contiguous()
    nf = torch.empty_like(cf)
    fvk.rms_norm_gated_silu_qwen36_bf16(
        cf.data_ptr(), zf.data_ptr(), nw.data_ptr(), nf.data_ptr(),
        cf.shape[0], HV, eps, 0)
    out = _proj(nf.reshape(B * S, VD), ld, 'out_proj', HID, fvk, device)
    return out.reshape(B, S, HID)


# bf16 SDPA for the prefill full-attn (vs the fp32/TF32 cutlass fmha). The
# softmax accumulates in fp32 inside the kernel, so cos is preserved while the
# QK^T / PV matmuls run on bf16 tensor cores (~2x). Default ON.
_ATTN_BF16 = True


def _full_attn_layer(h, ld, ct, st, fvk, device, eps, cap=None, rank=None):
    """Full GQA attention layer with output gate + partial RoPE.

    When ``cap`` is given, the RoPE'd K and V for all S positions are written
    into the decode KV cache so a batched prefill seeds the same cache the
    per-token decode would.
    """
    B, S, _ = h.shape
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']      # already (1+w)-folded
    x2 = h.reshape(B * S, HID)

    qg = _proj(x2, ld, 'q_proj', NQ * 2 * HD, fvk, device).contiguous()
    # split interleaved [q_pre(256), gate(256)] per head via fvk kernel.
    q_pre = torch.empty(B * S, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(B * S, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_split_q_gate_bf16(
        qg.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), B * S, 0)
    q = q_pre.view(B, S, NQ, HD)
    gate = gate.view(B, S, NQ * HD)
    q = _rms(q.to(torch.bfloat16), qnw, eps)
    k = _proj(x2, ld, 'k_proj', NKV * HD, fvk, device).view(B, S, NKV, HD)
    k = _rms(k, knw, eps)
    v = _proj(x2, ld, 'v_proj', NKV * HD, fvk, device).view(B, S, NKV, HD)

    qo = torch.empty(S, NQ, HD, dtype=torch.bfloat16, device=device)
    ko = torch.empty(S, NKV, HD, dtype=torch.bfloat16, device=device)
    qin = q.reshape(S, NQ, HD).contiguous()
    kin = k.reshape(S, NKV, HD).contiguous()
    ctc, stc = ct.contiguous(), st.contiguous()
    fvk.qwen36_partial_rope_qk_bf16(
        qin.data_ptr(), kin.data_ptr(), ctc.data_ptr(), stc.data_ptr(),
        qo.data_ptr(), ko.data_ptr(), S, NQ, NKV, HD, ROPE, 0)

    if cap is not None:
        # Seed the decode KV cache: RoPE'd K + raw V for all S positions,
        # exactly what the per-token decode writes at each pos.
        cap.attn.K_cache[rank, :S].copy_(ko.reshape(S, NKV, HD))
        cap.attn.V_cache[rank, :S].copy_(v.reshape(S, NKV, HD))

    qa = qo.reshape(B, S, NQ, HD).transpose(1, 2)
    ka = ko.reshape(B, S, NKV, HD).transpose(1, 2).repeat_interleave(NQ // NKV, 1)
    va = v.transpose(1, 2).repeat_interleave(NQ // NKV, 1)
    if _ATTN_BF16:
        # bf16 SDPA (mem-efficient/flash) instead of the fp32/TF32 fmha: ~2x
        # that bucket, cos preserved (softmax accumulates in fp32 internally).
        at = F.scaled_dot_product_attention(qa, ka, va, is_causal=True)
    else:
        at = F.scaled_dot_product_attention(
            qa.float(), ka.float(), va.float(), is_causal=True)
    at = at.transpose(1, 2).reshape(B, S, NQ * HD)
    at = (at.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return _proj(at.reshape(B * S, NQ * HD), ld, 'o_proj', HID,
                 fvk, device).reshape(B, S, HID)


# Grouped MoE for prefill (on by default); set False to use the per-expert loop.
_USE_GROUPED_MOE = True
# M=16 tensor-core mma MoE: tokens are sorted into 16-row expert tiles and the
# SM120 block-scaled mma runs each expert once at full M-utilisation -- ~5.6x
# the SIMT grouped W4A16 at large S (the compute wall). W4A4 (FP4 activation),
# so cos is a touch lower; used for S >= _M16_MIN_S (small S keeps W4A16).
_USE_M16_MOE = True
_M16_MIN_S = 64
_N_EXPERTS = 256
# Multi-warp block-tile (BM=BN=64, 4 warps) W4A4 GEMM: 4.0x the M16 tile (the
# activation + weight loaded once into smem and shared across warps). Default on
# for S >= _M16_MIN_S; set False to fall back to the M16 tile.
_USE_BT_MOE = True


def _moe_experts_m16(x, ti, tw, ld, fvk, device):
    """Routed experts via the M=16 tensor-core block-scaled mma. Sort the
    S*TOPK assignments by expert, pack into zero-padded 16-row tiles, quant once
    and run the mma (16 real tokens/tile -> full tensor-core M, each expert
    weight once). gate_up + silu(bf16) + down + scatter."""
    S = x.shape[0]
    E = _N_EXPERTS
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']

    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    order = exp_flat.argsort()
    se = exp_flat[order].long()
    stok = tok_flat[order]
    sw = tw.reshape(-1)[order]
    counts = torch.bincount(se, minlength=E)
    tile_counts = (counts + 15) // 16
    tile_off = torch.cumsum(tile_counts, 0) - tile_counts
    total_tiles = int(tile_counts.sum().item())
    tile_expert = torch.repeat_interleave(
        torch.arange(E, device=device), tile_counts).to(torch.int32)
    cumcount = torch.cumsum(counts, 0) - counts
    pos = torch.arange(S * TOPK, device=device) - cumcount[se]
    tiled_row = (tile_off[se] + pos // 16) * 16 + (pos % 16)

    A_t = torch.zeros(total_tiles * 16, HID, dtype=torch.bfloat16, device=device)
    A_t[tiled_row] = x[stok]
    ap, asf = _quant_act(A_t, fvk, device)
    d_gu = torch.empty(total_tiles * 16, n_gu, dtype=torch.bfloat16, device=device)
    fvk.moe_m16_mma_sm120_bf16(
        ap.data_ptr(), gu_p.data_ptr(), asf.data_ptr(), gu_s.data_ptr(),
        d_gu.data_ptr(), gu_a.data_ptr(), tile_expert.data_ptr(),
        total_tiles, n_gu, HID, 0, gu_p[0].numel(), gu_s[0].numel(), 0)
    inter = _silu_mul(d_gu[:, :INTER], d_gu[:, INTER:], fvk, device).contiguous()
    ip, isf = _quant_act(inter, fvk, device)
    d_dn = torch.empty(total_tiles * 16, n_dn, dtype=torch.bfloat16, device=device)
    fvk.moe_m16_mma_sm120_bf16(
        ip.data_ptr(), dn_p.data_ptr(), isf.data_ptr(), dn_s.data_ptr(),
        d_dn.data_ptr(), dn_a.data_ptr(), tile_expert.data_ptr(),
        total_tiles, n_dn, INTER, 0, dn_p[0].numel(), dn_s[0].numel(), 0)
    out = torch.zeros(S, HID, device=device)
    out.index_add_(0, stok, d_dn[tiled_row].float() * sw.unsqueeze(-1))
    return out


def _moe_experts_bt(x, ti, tw, ld, fvk, device):
    """Routed experts via the multi-warp block-tile (BM=BN=64, 4 warps) W4A4
    block-scaled GEMM. 64-row expert tiles; the activation rows and weight cols
    are loaded once into smem and shared across the 4 warps, so traffic ~ (1/64
    + 1/64) -- 4.0x the M16 tile / 1.96x the M64 tile at 1024 rows, cos identical
    (same FP4 mma). The sm120 hand-tuned equivalent of a DeepGEMM/SGLang tile."""
    S = x.shape[0]
    E = _N_EXPERTS
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']

    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    order = exp_flat.argsort()
    se = exp_flat[order].long()
    stok = tok_flat[order]
    sw = tw.reshape(-1)[order]
    counts = torch.bincount(se, minlength=E)
    tile_counts = (counts + 63) // 64
    tile_off = torch.cumsum(tile_counts, 0) - tile_counts
    total_tiles = int(tile_counts.sum().item())
    tile_expert = torch.repeat_interleave(
        torch.arange(E, device=device), tile_counts).to(torch.int32)
    cumcount = torch.cumsum(counts, 0) - counts
    pos = torch.arange(S * TOPK, device=device) - cumcount[se]
    tiled_row = (tile_off[se] + pos // 64) * 64 + (pos % 64)

    A_t = torch.zeros(total_tiles * 64, HID, dtype=torch.bfloat16, device=device)
    A_t[tiled_row] = x[stok]
    ap, asf = _quant_act(A_t, fvk, device)
    d_gu = torch.empty(total_tiles * 64, n_gu, dtype=torch.bfloat16, device=device)
    fvk.moe_blocktile_mma_sm120_bf16(
        ap.data_ptr(), gu_p.data_ptr(), asf.data_ptr(), gu_s.data_ptr(),
        d_gu.data_ptr(), gu_a.data_ptr(), tile_expert.data_ptr(),
        total_tiles, n_gu, HID, 0, gu_p[0].numel(), gu_s[0].numel(), 0)
    inter = _silu_mul(d_gu[:, :INTER], d_gu[:, INTER:], fvk, device).contiguous()
    ip, isf = _quant_act(inter, fvk, device)
    d_dn = torch.empty(total_tiles * 64, n_dn, dtype=torch.bfloat16, device=device)
    fvk.moe_blocktile_mma_sm120_bf16(
        ip.data_ptr(), dn_p.data_ptr(), isf.data_ptr(), dn_s.data_ptr(),
        d_dn.data_ptr(), dn_a.data_ptr(), tile_expert.data_ptr(),
        total_tiles, n_dn, INTER, 0, dn_p[0].numel(), dn_s[0].numel(), 0)
    out = torch.zeros(S, HID, device=device)
    out.index_add_(0, stok, d_dn[tiled_row].float() * sw.unsqueeze(-1))
    return out


def _moe_experts_grouped(x, ti, tw, ld, fvk, device):
    """Routed experts via the grouped W4A16 GEMV. Flatten the S*TOPK
    (token, expert) assignments, sort by expert so consecutive slots share a
    weight (L2-amortised -> each expert weight read ~once), and run one grouped
    GEMV per gate_up / down (BF16 activation, no per-expert launch/quant). The
    Python expert loop's ~5000 tiny-M GEMMs collapse to 2 kernel launches."""
    S = x.shape[0]
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]
    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']

    slots = S * TOPK
    exp_flat = ti.reshape(-1).to(torch.int32)
    tok_flat = torch.arange(S, device=device).repeat_interleave(TOPK)
    order = exp_flat.argsort()
    se = exp_flat[order].contiguous()
    stok = tok_flat[order]
    sw = tw.reshape(-1)[order]

    A = x[stok].contiguous()                          # (slots, HID) bf16
    d_gu = torch.empty(slots, n_gu, dtype=torch.bfloat16, device=device)
    fvk.moe_grouped_w4a16_sm120_bf16(
        A.data_ptr(), gu_p.data_ptr(), gu_s.data_ptr(), gu_a.data_ptr(),
        se.data_ptr(), d_gu.data_ptr(), slots, n_gu, HID,
        HID, gu_p[0].numel(), gu_s[0].numel(), 0)
    g, u = d_gu[:, :INTER], d_gu[:, INTER:]
    inter = _silu_mul(g, u, fvk, device).contiguous()
    d_dn = torch.empty(slots, n_dn, dtype=torch.bfloat16, device=device)
    fvk.moe_grouped_w4a16_sm120_bf16(
        inter.data_ptr(), dn_p.data_ptr(), dn_s.data_ptr(), dn_a.data_ptr(),
        se.data_ptr(), d_dn.data_ptr(), slots, n_dn, INTER,
        INTER, dn_p[0].numel(), dn_s[0].numel(), 0)
    out = torch.zeros(S, HID, device=device)
    out.index_add_(0, stok, d_dn.float() * sw.unsqueeze(-1))
    return out


def _moe_layer(h, ld, fvk, device):
    """Fine-grained MoE FFN: 256 experts top-8 routed + 1 shared expert."""
    B, S, _ = h.shape
    x = h.reshape(-1, HID)
    rw = ld['router_w_t']
    gu_p, gu_s, gu_a = (ld['experts_gate_up_packed_t'],
                        ld['experts_gate_up_sf_t'], ld['experts_gate_up_alpha_t'])
    dn_p, dn_s, dn_a = (ld['experts_down_packed_t'],
                        ld['experts_down_sf_t'], ld['experts_down_alpha_t'])
    n_gu = gu_p.shape[1]          # 2 * inter
    n_dn = dn_p.shape[1]          # hidden

    logit = F.softmax(x.float() @ rw.float().T, -1)
    tw, ti = torch.topk(logit, TOPK, -1)
    tw = tw / tw.sum(-1, keepdim=True)

    if _USE_BT_MOE and x.shape[0] >= _M16_MIN_S:
        out = _moe_experts_bt(x, ti, tw, ld, fvk, device)
    elif _USE_M16_MOE and x.shape[0] >= _M16_MIN_S:
        out = _moe_experts_m16(x, ti, tw, ld, fvk, device)
    elif _USE_GROUPED_MOE:
        out = _moe_experts_grouped(x, ti, tw, ld, fvk, device)
    else:
        out = torch.zeros(x.shape[0], HID, device=device)
        gu_a_l = gu_a.tolist()
        dn_a_l = dn_a.tolist()
        for e in torch.unique(ti).tolist():
            m = (ti == e)
            tok = m.any(-1).nonzero(as_tuple=True)[0]
            w = (tw * m)[tok].sum(-1)
            gu_e_p, gu_e_s = gu_p[e], gu_s[e]
            gu = _nvfp4_gemm(x[tok].contiguous(), gu_e_p.data_ptr(),
                             gu_e_s.data_ptr(), gu_a_l[e], n_gu,
                             fvk, device)
            g, u = gu.chunk(2, -1)
            inter = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
            dn_e_p, dn_e_s = dn_p[e], dn_s[e]
            dpj = _nvfp4_gemm(inter, dn_e_p.data_ptr(), dn_e_s.data_ptr(),
                              dn_a_l[e], n_dn, fvk, device)
            out[tok] += dpj.float() * w.unsqueeze(-1)

    sg = _proj(x, ld, 'shared_gate_proj', INTER, fvk, device)
    su = _proj(x, ld, 'shared_up_proj', INTER, fvk, device)
    si = _silu_mul(sg, su, fvk, device)
    shared = _proj(si, ld, 'shared_down_proj', HID, fvk, device)
    sgate = torch.sigmoid(x.float() @ ld['shared_gate_w_t'].float().T)
    return (out + shared.float() * sgate).reshape(B, S, HID).to(torch.bfloat16)


def nexn2_forward_nvfp4(handles, input_ids, fvk, device, cap=None,
                        return_hidden=False):
    """Full kernelized NVFP4 prefill forward: token ids -> logits.

    Args:
        handles: WeightHandles from extract_weights_nexn2_nvfp4.
        input_ids: (1, S) long on device.
        fvk: flash_rt_kernels module.
        device: cuda device string.
        cap: optional Nexn2DecodeState; when given, the GDN recurrent/conv
            state and the full-attn KV cache are seeded so a subsequent decode
            continues from position S (batched prefill).

    Returns:
        logits: (S, vocab) fp32 on device.
    """
    p = handles.ptrs
    eps = float(p['rms_norm_eps'])
    theta = float(p['rope_theta'])
    rope_dim = int(p['head_dim'] * p['partial_rotary_factor'])
    types = p['layer_types']
    layers = p['layers']

    h = F.embedding(input_ids, p['embed_w_t'])
    S = h.shape[1]
    ct, st = build_rope_tables(S, theta, rope_dim, device)
    lin_rank = full_rank = 0
    for L in range(p['num_layers']):
        ld = layers[L]
        res = h
        n = _rms(h, ld['input_norm_w_t'], eps)
        if types[L] == 'linear_attention':
            attn = _gdn_layer(n, ld, fvk, device, eps, cap, lin_rank)
            lin_rank += 1
        else:
            attn = _full_attn_layer(n, ld, ct, st, fvk, device, eps,
                                    cap, full_rank)
            full_rank += 1
        h = res + attn
        res = h
        n = _rms(h, ld['post_norm_w_t'], eps)
        h = res + _moe_layer(n, ld, fvk, device)

    hidden = h[0]                       # (S, HID) residual stream, pre-final-norm
    h = _rms(h, p['final_norm_w_t'], eps)
    logits = h[0].float() @ p['lm_head_w_t'].float().T
    if return_hidden:
        return logits, hidden
    return logits
