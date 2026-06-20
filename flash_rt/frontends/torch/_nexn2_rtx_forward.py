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
    return (x2d.float() @ w.float().T).to(torch.bfloat16)


def _nvfp4_gemm(x2d, wp_ptr, wsf_ptr, alpha, n, fvk, device, stream=0):
    """y = x @ w.T via NVFP4. x2d is (M,K) bf16; weight given by ptrs+alpha.

    Activation quantised per call (swizzled). All ptr args are bound to
    named tensors that outlive the launch.
    """
    m, kk = x2d.shape
    xc = x2d.contiguous()
    xp = torch.empty(m, kk // 2, dtype=torch.uint8, device=device)
    xsf = torch.zeros(_sf_swz_bytes(m, kk), dtype=torch.uint8, device=device)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        xc.data_ptr(), xp.data_ptr(), xsf.data_ptr(), m, kk, stream)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    fvk.fp4_w4a16_gemm_sm120_bf16out(
        xp.data_ptr(), wp_ptr, y.data_ptr(), m, n, kk,
        xsf.data_ptr(), wsf_ptr, alpha, stream)
    return y


def _gdn_layer(h, ld, fvk, device, eps):
    """Gated DeltaNet (linear_attention) layer. h (1,S,HID) -> (1,S,HID)."""
    B, S, _ = h.shape
    Wqkv = ld['in_proj_qkv_w_t']
    Wz = ld['in_proj_z_w_t']
    Wb, Wa = ld['in_proj_b_w_t'], ld['in_proj_a_w_t']
    convw = ld['conv1d_w_t']
    A_log, dtb = ld['A_log_t'].float(), ld['dt_bias_t'].float()
    nw = ld['gdn_norm_w_t']

    # in_proj stays BF16 (red line: quantizing it collapses GDN).
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
    fvk.nexn2_lin_split_qkv_broadcast_bf16(
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

    state = torch.zeros(NV, HK, HV, dtype=torch.bfloat16, device=device)
    core = torch.empty(B, S, NV, HV, dtype=torch.bfloat16, device=device)
    for t in range(S):
        qt = qb[:, t].reshape(NV, HK).to(torch.bfloat16).contiguous()
        kt = kb[:, t].reshape(NV, HK).to(torch.bfloat16).contiguous()
        vt = vb[:, t].reshape(NV, HV).to(torch.bfloat16).contiguous()
        gtt = g_out[:, t].reshape(NV).contiguous()
        btt = bo[:, t].reshape(NV).contiguous()
        ot = core[:, t].reshape(NV, HV)
        fvk.gated_deltanet_recurrent_qwen36_bf16(
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), gtt.data_ptr(),
            btt.data_ptr(), state.data_ptr(), ot.data_ptr(),
            1, NV, HK, HV, True, 0)

    cf = core.reshape(-1, HV).contiguous()
    zf = z.reshape(-1, HV).to(torch.bfloat16).contiguous()
    nf = torch.empty_like(cf)
    fvk.rms_norm_gated_silu_qwen36_bf16(
        cf.data_ptr(), zf.data_ptr(), nw.data_ptr(), nf.data_ptr(),
        cf.shape[0], HV, eps, 0)
    out = _proj(nf.reshape(B * S, VD), ld, 'out_proj', HID, fvk, device)
    return out.reshape(B, S, HID)


def _full_attn_layer(h, ld, ct, st, fvk, device, eps):
    """Full GQA attention layer with output gate + partial RoPE."""
    B, S, _ = h.shape
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']      # already (1+w)-folded
    x2 = h.reshape(B * S, HID)

    qg = _proj(x2, ld, 'q_proj', NQ * 2 * HD, fvk, device).contiguous()
    # split interleaved [q_pre(256), gate(256)] per head via fvk kernel.
    q_pre = torch.empty(B * S, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(B * S, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.nexn2_split_q_gate_bf16(
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

    qa = qo.reshape(B, S, NQ, HD).transpose(1, 2)
    ka = ko.reshape(B, S, NKV, HD).transpose(1, 2).repeat_interleave(NQ // NKV, 1)
    va = v.transpose(1, 2).repeat_interleave(NQ // NKV, 1)
    at = F.scaled_dot_product_attention(
        qa.float(), ka.float(), va.float(), is_causal=True)
    at = at.transpose(1, 2).reshape(B, S, NQ * HD)
    at = (at * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return _proj(at.reshape(B * S, NQ * HD), ld, 'o_proj', HID,
                 fvk, device).reshape(B, S, HID)


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
    out = torch.zeros(x.shape[0], HID, device=device)
    for e in torch.unique(ti).tolist():
        m = (ti == e)
        tok = m.any(-1).nonzero(as_tuple=True)[0]
        w = (tw * m)[tok].sum(-1)
        gu_e_p, gu_e_s = gu_p[e], gu_s[e]
        gu = _nvfp4_gemm(x[tok].contiguous(), gu_e_p.data_ptr(),
                         gu_e_s.data_ptr(), float(gu_a[e].item()), n_gu,
                         fvk, device)
        g, u = gu.chunk(2, -1)
        inter = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
        dn_e_p, dn_e_s = dn_p[e], dn_s[e]
        dpj = _nvfp4_gemm(inter, dn_e_p.data_ptr(), dn_e_s.data_ptr(),
                          float(dn_a[e].item()), n_dn, fvk, device)
        out[tok] += dpj.float() * w.unsqueeze(-1)

    sg = _proj(x, ld, 'shared_gate_proj', INTER, fvk, device)
    su = _proj(x, ld, 'shared_up_proj', INTER, fvk, device)
    si = (F.silu(sg.float()) * su.float()).to(torch.bfloat16)
    shared = _proj(si, ld, 'shared_down_proj', HID, fvk, device)
    sgate = torch.sigmoid(x.float() @ ld['shared_gate_w_t'].float().T)
    return (out + shared.float() * sgate).reshape(B, S, HID).to(torch.bfloat16)


def nexn2_forward_nvfp4(handles, input_ids, fvk, device):
    """Full kernelized NVFP4 prefill forward: token ids -> logits.

    Args:
        handles: WeightHandles from extract_weights_nexn2_nvfp4.
        input_ids: (1, S) long on device.
        fvk: flash_rt_kernels module.
        device: cuda device string.

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
    for L in range(p['num_layers']):
        ld = layers[L]
        res = h
        n = _rms(h, ld['input_norm_w_t'], eps)
        if types[L] == 'linear_attention':
            attn = _gdn_layer(n, ld, fvk, device, eps)
        else:
            attn = _full_attn_layer(n, ld, ct, st, fvk, device, eps)
        h = res + attn
        res = h
        n = _rms(h, ld['post_norm_w_t'], eps)
        h = res + _moe_layer(n, ld, fvk, device)

    h = _rms(h, p['final_norm_w_t'], eps)
    logits = h[0].float() @ p['lm_head_w_t'].float().T
    return logits
