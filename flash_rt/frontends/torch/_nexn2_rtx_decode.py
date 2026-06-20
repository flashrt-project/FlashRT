"""FlashRT -- Nex-N2-mini (qwen3_5_moe) M=1 decode forward.

Single-token autoregressive decode driving the fvk kernels off the loader
handles, with persistent per-layer state:
  * Gated DeltaNet: recurrent state (NV, HK, HV) + causal-conv rolling state
    (1, conv_dim, k-1), both carried across decode steps.
  * Full attention: KV cache owned by RtxFlashAttnBackendNexn2; the new
    token's rope'd K and V are written at ``pos`` and attention runs 1 query
    vs the [0..pos] history.

Prefill is seeded by running this same step over the prompt tokens 0..S-1,
so position p's output integrates exactly tokens 0..p -- identical math to
the batched prefill forward (the self-consistency check in phase4d).

This is the correctness substrate: scratch is allocated per call. The
graph milestone (2d) pre-allocates everything and captures the step. Routed
MoE stays on the eager prefill _moe_layer (dynamic top-8 routing is the
known graph blocker, handled separately).

All fvk pointer args bind to named tensors (ctypes GC rule).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_rt.frontends.torch._nexn2_rtx_forward import (
    CONV, HD, HID, HK, HV, KD, KS, NK, NKV, NQ, NV, ROPE, VD,
    _moe_layer, _proj, _rms, build_rope_tables,
)
from flash_rt.hardware.rtx.attn_backend_nexn2 import RtxFlashAttnBackendNexn2


class Nexn2DecodeState:
    """Persistent decode state: GDN recurrent/conv caches, KV cache, RoPE."""

    def __init__(self, handles, max_seq, device):
        self.handles = handles
        self.device = device
        self.max_seq = int(max_seq)
        p = handles.ptrs
        self.eps = float(p['rms_norm_eps'])
        self.types = p['layer_types']
        self.num_layers = int(p['num_layers'])

        # Map each layer to its rank within its regime.
        self._lin_rank = {}
        self._full_rank = {}
        nlin = nfull = 0
        for L, t in enumerate(self.types):
            if t == 'linear_attention':
                self._lin_rank[L] = nlin
                nlin += 1
            else:
                self._full_rank[L] = nfull
                nfull += 1
        self.n_lin, self.n_full = nlin, nfull

        bf16 = torch.bfloat16
        # GDN recurrent state (NV, HK, HV) + conv rolling state (1, CONV, KS-1).
        self.lin_state = [
            torch.zeros(NV, HK, HV, dtype=bf16, device=device)
            for _ in range(nlin)]
        self.lin_conv_state = [
            torch.zeros(1, CONV, KS - 1, dtype=bf16, device=device)
            for _ in range(nlin)]

        # Full-attn KV cache.
        self.attn = RtxFlashAttnBackendNexn2(max_seq=self.max_seq, max_q_seq=1)

        # RoPE tables for the whole window.
        theta = float(p['rope_theta'])
        rope_dim = int(p['head_dim'] * p['partial_rotary_factor'])
        self.rope_cos, self.rope_sin = build_rope_tables(
            self.max_seq, theta, rope_dim, device)

    def reset(self):
        for s in self.lin_state:
            s.zero_()
        for c in self.lin_conv_state:
            c.zero_()
        self.attn.reset_cache()


def _decode_gdn(h, ld, state, lin_rank, fvk, device):
    """GDN layer at one token, updating recurrent + conv state in place."""
    eps = state.eps
    Wqkv = ld['in_proj_qkv_w_t']
    Wz = ld['in_proj_z_w_t']
    Wb, Wa = ld['in_proj_b_w_t'], ld['in_proj_a_w_t']
    convw = ld['conv1d_w_t'].reshape(CONV, KS).contiguous()
    A_log, dtb = ld['A_log_t'].float(), ld['dt_bias_t'].float()
    nw = ld['gdn_norm_w_t']

    h2 = h.reshape(1, HID)
    mixed = (h2.float() @ Wqkv.float().T).to(torch.bfloat16).contiguous()
    z = (h2.float() @ Wz.float().T).to(torch.bfloat16).reshape(NV, HV).contiguous()
    a = (h2.float() @ Wa.float().T).to(torch.bfloat16).contiguous()
    b = (h2.float() @ Wb.float().T).to(torch.bfloat16).contiguous()

    # causal conv1d state-update (no bias) + silu.
    conv_out = torch.empty(1, CONV, dtype=torch.bfloat16, device=device)
    conv_state = state.lin_conv_state[lin_rank]
    fvk.causal_conv1d_qwen36_update_bf16(
        mixed.data_ptr(), convw.data_ptr(), 0,
        conv_out.data_ptr(), conv_state.data_ptr(),
        1, CONV, KS, True, 0)

    # split + broadcast 16 -> 32 heads.
    qb = torch.empty(1, NV, HK, dtype=torch.bfloat16, device=device)
    kb = torch.empty(1, NV, HK, dtype=torch.bfloat16, device=device)
    vb = torch.empty(1, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.nexn2_lin_split_qkv_broadcast_bf16(
        conv_out.data_ptr(), qb.data_ptr(), kb.data_ptr(), vb.data_ptr(),
        1, 0)

    neg = (-A_log.exp()).float().contiguous()
    dtb_c = dtb.contiguous()
    g_out = torch.empty(1, NV, dtype=torch.bfloat16, device=device)
    bo = torch.empty(1, NV, dtype=torch.bfloat16, device=device)
    fvk.qwen36_gdn_gating_bf16(
        a.data_ptr(), b.data_ptr(), neg.data_ptr(), dtb_c.data_ptr(),
        g_out.data_ptr(), bo.data_ptr(), 1, NV, 0)

    qt = qb.reshape(NV, HK).contiguous()
    kt = kb.reshape(NV, HK).contiguous()
    vt = vb.reshape(NV, HV).contiguous()
    gt = g_out.reshape(NV).contiguous()
    bt = bo.reshape(NV).contiguous()
    core = torch.empty(NV, HV, dtype=torch.bfloat16, device=device)
    fvk.gated_deltanet_recurrent_qwen36_bf16(
        qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), gt.data_ptr(),
        bt.data_ptr(), state.lin_state[lin_rank].data_ptr(),
        core.data_ptr(), 1, NV, HK, HV, True, 0)

    nf = torch.empty(NV, HV, dtype=torch.bfloat16, device=device)
    fvk.rms_norm_gated_silu_qwen36_bf16(
        core.data_ptr(), z.data_ptr(), nw.data_ptr(), nf.data_ptr(),
        NV, HV, eps, 0)
    out = _proj(nf.reshape(1, VD), ld, 'out_proj', HID, fvk, device)
    return out.reshape(1, 1, HID)


def _decode_full(h, ld, state, full_rank, pos, fvk, device):
    """Full-attn layer at one token; writes KV at pos, attends [0..pos]."""
    eps = state.eps
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']
    x2 = h.reshape(1, HID)

    qg = _proj(x2, ld, 'q_proj', NQ * 2 * HD, fvk, device).contiguous()
    q_pre = torch.empty(1, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(1, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.nexn2_split_q_gate_bf16(
        qg.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), 1, 0)
    q = _rms(q_pre.reshape(NQ, HD), qnw, eps).reshape(1, NQ, HD)
    k = _proj(x2, ld, 'k_proj', NKV * HD, fvk, device).reshape(NKV, HD)
    k = _rms(k, knw, eps).reshape(1, NKV, HD)
    v = _proj(x2, ld, 'v_proj', NKV * HD, fvk, device).reshape(1, NKV, HD)

    ct = state.rope_cos[pos:pos + 1].contiguous()
    st = state.rope_sin[pos:pos + 1].contiguous()
    qin = q.reshape(1, NQ, HD).contiguous()
    kin = k.reshape(1, NKV, HD).contiguous()
    qo = torch.empty(1, NQ, HD, dtype=torch.bfloat16, device=device)
    ko = torch.empty(1, NKV, HD, dtype=torch.bfloat16, device=device)
    fvk.qwen36_partial_rope_qk_bf16(
        qin.data_ptr(), kin.data_ptr(), ct.data_ptr(), st.data_ptr(),
        qo.data_ptr(), ko.data_ptr(), 1, NQ, NKV, HD, ROPE, 0)

    attn = state.attn
    attn.Q_buf[:, :1].copy_(qo.reshape(1, 1, NQ, HD))
    attn.K_cache[full_rank, pos:pos + 1].copy_(ko.reshape(1, NKV, HD))
    attn.V_cache[full_rank, pos:pos + 1].copy_(v.reshape(1, NKV, HD))
    attn.run('full', layer_idx=full_rank, q_seq=1, kv_seq=pos + 1,
             softmax_scale=float(HD) ** -0.5)
    at = attn.O_buf[:, :1].reshape(1, NQ * HD)
    at = (at.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return _proj(at, ld, 'o_proj', HID, fvk, device).reshape(1, 1, HID)


def decode_step(state, token_id, pos, fvk, device):
    """One decode step: token id at position pos -> (1, vocab) logits."""
    handles = state.handles
    p = handles.ptrs
    layers = p['layers']
    if not isinstance(token_id, torch.Tensor):
        token_id = torch.tensor([token_id], device=device, dtype=torch.long)
    h = F.embedding(token_id.view(1, 1), p['embed_w_t'])

    for L in range(state.num_layers):
        ld = layers[L]
        res = h
        n = _rms(h, ld['input_norm_w_t'], state.eps)
        if state.types[L] == 'linear_attention':
            attn = _decode_gdn(n, ld, state, state._lin_rank[L], fvk, device)
        else:
            attn = _decode_full(n, ld, state, state._full_rank[L], pos,
                                fvk, device)
        h = res + attn
        res = h
        n = _rms(h, ld['post_norm_w_t'], state.eps)
        h = res + _moe_layer(n, ld, fvk, device)

    h = _rms(h, p['final_norm_w_t'], state.eps)
    logits = h[0].float() @ p['lm_head_w_t'].float().T
    return logits


def seed_prefill(state, input_ids, fvk, device):
    """Run the decode step over prompt tokens 0..S-1, building all state.

    Returns the last-token logits (1, vocab).
    """
    state.reset()
    ids = input_ids.view(-1)
    last = None
    for pos in range(ids.shape[0]):
        last = decode_step(state, ids[pos:pos + 1], pos, fvk, device)
    return last


def generate_greedy(state, input_ids, max_new_tokens, fvk, device):
    """Greedy decode: seed the prompt then emit max_new_tokens tokens."""
    ids = input_ids.view(-1).tolist()
    pos = len(ids)
    logits = seed_prefill(state, input_ids, fvk, device)
    out = []
    for _ in range(max_new_tokens):
        nxt = int(logits[0].argmax().item())
        out.append(nxt)
        logits = decode_step(state, nxt, pos, fvk, device)
        pos += 1
    return out
