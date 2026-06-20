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
    CONV, HD, HID, HK, HV, INTER, KS, NKV, NQ, NV, ROPE, TOPK, VD,
    _quant_act, _rms, build_rope_tables,
)
from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import _sf_swz_bytes
from flash_rt.hardware.rtx.attn_backend_nexn2 import RtxFlashAttnBackendNexn2


def _cs():
    """Current CUDA stream handle. Inside torch.cuda.graph capture this is
    the capture stream; eager, the default stream. fvk calls MUST use it --
    a hard-coded stream=0 silently no-ops on graph replay (-> NaN)."""
    return torch.cuda.current_stream().cuda_stream


def _mma_preq(xp, xsf, wp_ptr, wsf_ptr, alpha, n, k, fvk, device):
    """M=1 NVFP4 GEMV via the hand-tuned SM120 mma kernel (full-N).

    cos=1.0 vs the CUTLASS fp4_w4a16 GEMM at every Nex-N2 decode shape, and
    far higher HBM-BW utilisation at M=1 (CUTLASS tiles for M>=16). Same
    swizzled SF layout, so it consumes the loader weights + _quant_act
    activation directly.
    """
    y = torch.empty(1, n, dtype=torch.bfloat16, device=device)
    fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
        xp.data_ptr(), wp_ptr, y.data_ptr(), n, k,
        xsf.data_ptr(), wsf_ptr, alpha, _cs())
    return y


def _mma(x2d, wp_ptr, wsf_ptr, alpha, n, fvk, device):
    """Quantise the M=1 activation then GEMV via the mma kernel."""
    _, k = x2d.shape
    xp, xsf = _quant_act(x2d, fvk, device, _cs())
    return _mma_preq(xp, xsf, wp_ptr, wsf_ptr, alpha, n, k, fvk, device)


def _bf16_mv(x1k, w, fvk, device):
    """M=1 BF16 GEMV x(1,K) @ w(N,K).T -> (1,N) via the hand-tuned kernel.

    cos 0.999999 vs torch fp32 matmul; reads the bf16 weight directly (no
    fp32 up-cast / temporary), so it is both faster and lighter on HBM.
    """
    n, k = w.shape
    xc = x1k.contiguous()
    y = torch.empty(1, n, dtype=torch.bfloat16, device=device)
    fvk.bf16_matvec_qwen36_bf16(xc.data_ptr(), w.data_ptr(), y.data_ptr(),
                                n, k, _cs())
    return y


def _proj_mma(x2d, ld, base, n, fvk, device):
    """Decode projection dispatch: NVFP4 -> mma GEMV, else BF16 mv kernel."""
    if ld.get(base + '_packed') is not None:
        return _mma(x2d, ld[base + '_packed'], ld[base + '_sf'],
                    ld[base + '_alpha'], n, fvk, device)
    return _bf16_mv(x2d, ld[base + '_w_t'], fvk, device)


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

        # ── CUDA graph decode ──
        # One graph per position (KV slot / attn length / RoPE slice baked);
        # the only varying input is the device token id, re-read each replay.
        self._static_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._graph_stream = torch.cuda.Stream()
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._graphs = {}                       # pos -> (CUDAGraph, out tensor)
        self._snap_lin = [torch.empty_like(t) for t in self.lin_state]
        self._snap_conv = [torch.empty_like(t) for t in self.lin_conv_state]

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

    s = _cs()
    h2 = h.reshape(1, HID)
    mixed = _bf16_mv(h2, Wqkv, fvk, device).contiguous()
    z = _bf16_mv(h2, Wz, fvk, device).reshape(NV, HV).contiguous()
    a = _bf16_mv(h2, Wa, fvk, device).contiguous()
    b = _bf16_mv(h2, Wb, fvk, device).contiguous()

    # causal conv1d state-update (no bias) + silu.
    conv_out = torch.empty(1, CONV, dtype=torch.bfloat16, device=device)
    conv_state = state.lin_conv_state[lin_rank]
    fvk.causal_conv1d_qwen36_update_bf16(
        mixed.data_ptr(), convw.data_ptr(), 0,
        conv_out.data_ptr(), conv_state.data_ptr(),
        1, CONV, KS, True, s)

    # split + broadcast 16 -> 32 heads.
    qb = torch.empty(1, NV, HK, dtype=torch.bfloat16, device=device)
    kb = torch.empty(1, NV, HK, dtype=torch.bfloat16, device=device)
    vb = torch.empty(1, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.nexn2_lin_split_qkv_broadcast_bf16(
        conv_out.data_ptr(), qb.data_ptr(), kb.data_ptr(), vb.data_ptr(),
        1, s)

    neg = (-A_log.exp()).float().contiguous()
    dtb_c = dtb.contiguous()
    g_out = torch.empty(1, NV, dtype=torch.bfloat16, device=device)
    bo = torch.empty(1, NV, dtype=torch.bfloat16, device=device)
    fvk.qwen36_gdn_gating_bf16(
        a.data_ptr(), b.data_ptr(), neg.data_ptr(), dtb_c.data_ptr(),
        g_out.data_ptr(), bo.data_ptr(), 1, NV, s)

    qt = qb.reshape(NV, HK).contiguous()
    kt = kb.reshape(NV, HK).contiguous()
    vt = vb.reshape(NV, HV).contiguous()
    gt = g_out.reshape(NV).contiguous()
    bt = bo.reshape(NV).contiguous()
    core = torch.empty(NV, HV, dtype=torch.bfloat16, device=device)
    fvk.gated_deltanet_recurrent_qwen36_bf16(
        qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), gt.data_ptr(),
        bt.data_ptr(), state.lin_state[lin_rank].data_ptr(),
        core.data_ptr(), 1, NV, HK, HV, True, s)

    nf = torch.empty(NV, HV, dtype=torch.bfloat16, device=device)
    fvk.rms_norm_gated_silu_qwen36_bf16(
        core.data_ptr(), z.data_ptr(), nw.data_ptr(), nf.data_ptr(),
        NV, HV, eps, s)
    out = _proj_mma(nf.reshape(1, VD), ld, 'out_proj', HID, fvk, device)
    return out.reshape(1, 1, HID)


def _decode_full(h, ld, state, full_rank, pos, fvk, device):
    """Full-attn layer at one token; writes KV at pos, attends [0..pos]."""
    eps = state.eps
    s = _cs()
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']
    x2 = h.reshape(1, HID)

    qg = _proj_mma(x2, ld, 'q_proj', NQ * 2 * HD, fvk, device).contiguous()
    q_pre = torch.empty(1, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(1, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.nexn2_split_q_gate_bf16(
        qg.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), 1, s)
    q = _rms(q_pre.reshape(NQ, HD), qnw, eps).reshape(1, NQ, HD)
    k = _proj_mma(x2, ld, 'k_proj', NKV * HD, fvk, device).reshape(NKV, HD)
    k = _rms(k, knw, eps).reshape(1, NKV, HD)
    v = _proj_mma(x2, ld, 'v_proj', NKV * HD, fvk, device).reshape(1, NKV, HD)

    ct = state.rope_cos[pos:pos + 1].contiguous()
    st = state.rope_sin[pos:pos + 1].contiguous()
    qin = q.reshape(1, NQ, HD).contiguous()
    kin = k.reshape(1, NKV, HD).contiguous()
    qo = torch.empty(1, NQ, HD, dtype=torch.bfloat16, device=device)
    ko = torch.empty(1, NKV, HD, dtype=torch.bfloat16, device=device)
    fvk.qwen36_partial_rope_qk_bf16(
        qin.data_ptr(), kin.data_ptr(), ct.data_ptr(), st.data_ptr(),
        qo.data_ptr(), ko.data_ptr(), 1, NQ, NKV, HD, ROPE, s)

    attn = state.attn
    attn.Q_buf[:, :1].copy_(qo.reshape(1, 1, NQ, HD))
    attn.K_cache[full_rank, pos:pos + 1].copy_(ko.reshape(1, NKV, HD))
    attn.V_cache[full_rank, pos:pos + 1].copy_(v.reshape(1, NKV, HD))
    attn.run('full', layer_idx=full_rank, q_seq=1, kv_seq=pos + 1,
             stream=s, softmax_scale=float(HD) ** -0.5)
    at = attn.O_buf[:, :1].reshape(1, NQ * HD)
    at = (at.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return _proj_mma(at, ld, 'o_proj', HID, fvk, device).reshape(1, 1, HID)


def _moe_layer_decode(h, ld, fvk, device):
    """M=1 fine-grained MoE via the grouped GEMV kernel: the 8 routed experts
    run in one launch each for gate_up (shared act) and down (per-slot act),
    indexed by a device top-k id buffer (the same buffer drives a graph)."""
    s = _cs()
    x = h.reshape(1, HID)
    logit = F.softmax(_bf16_mv(x, ld['router_w_t'], fvk, device).float(), -1)
    tw, ti = torch.topk(logit, TOPK, -1)
    tw_row = (tw / tw.sum(-1, keepdim=True))[0]              # (TOPK,) device
    idx = ti[0].to(torch.int32).contiguous()                # (TOPK,) device

    if 'experts_gate_up_alpha_dev' not in ld:               # cache once/layer
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]               # 1024 / HID

    # gate_up: one shared activation, grouped over the 8 experts.
    xp, xsf = _quant_act(x, fvk, device, s)
    d_gu = torch.empty(TOPK, n_gu, dtype=torch.bfloat16, device=device)
    fvk.nexn2_moe_grouped_gemv_bf16(
        xp.data_ptr(), gu_p.data_ptr(), d_gu.data_ptr(),
        xsf.data_ptr(), gu_s.data_ptr(), gu_a.data_ptr(), idx.data_ptr(),
        TOPK, n_gu, HID, 0, 0, n_gu * (HID // 2),
        _sf_swz_bytes(n_gu, HID), s)

    g, u = d_gu[:, :INTER], d_gu[:, INTER:]
    inter = (F.silu(g.float()) * u.float()).to(torch.bfloat16)   # (TOPK, INTER)

    # down: per-slot activation (quantise each M=1 into the stack).
    a_stack = torch.empty(TOPK, INTER // 2, dtype=torch.uint8, device=device)
    sfa_stack = torch.zeros(TOPK, _sf_swz_bytes(1, INTER),
                            dtype=torch.uint8, device=device)
    for j in range(TOPK):
        xc = inter[j:j + 1].contiguous()
        fvk.quantize_bf16_to_nvfp4_swizzled(
            xc.data_ptr(), a_stack[j].data_ptr(), sfa_stack[j].data_ptr(),
            1, INTER, s)
    d_dn = torch.empty(TOPK, n_dn, dtype=torch.bfloat16, device=device)
    fvk.nexn2_moe_grouped_gemv_bf16(
        a_stack.data_ptr(), dn_p.data_ptr(), d_dn.data_ptr(),
        sfa_stack.data_ptr(), dn_s.data_ptr(), dn_a.data_ptr(), idx.data_ptr(),
        TOPK, n_dn, INTER, INTER // 2, _sf_swz_bytes(1, INTER),
        n_dn * (INTER // 2), _sf_swz_bytes(n_dn, INTER), s)
    out = (d_dn.float() * tw_row.unsqueeze(-1)).sum(0, keepdim=True)

    sg = _proj_mma(x, ld, 'shared_gate_proj', INTER, fvk, device)
    su = _proj_mma(x, ld, 'shared_up_proj', INTER, fvk, device)
    si = (F.silu(sg.float()) * su.float()).to(torch.bfloat16)
    shared = _proj_mma(si, ld, 'shared_down_proj', HID, fvk, device)
    sgate = torch.sigmoid(x.float() @ ld['shared_gate_w_t'].float().T)
    return (out + shared.float() * sgate).reshape(1, 1, HID).to(torch.bfloat16)


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
        h = res + _moe_layer_decode(n, ld, fvk, device)

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


def _ensure_decode_graph(state, pos, fvk, device):
    """Lazily capture a CUDA graph of one decode step at ``pos``.

    The KV-write slot, attention length and RoPE slice are baked per pos, so
    each pos owns a graph; the only varying input is ``state._static_token``,
    re-read each replay. The lin/conv/KV state mutated by the 2 warmup runs +
    the capture run is snapshotted and restored, so a later replay advances
    from the true pre-step state. Returns (graph, logits_buffer).
    """
    cached = state._graphs.get(pos)
    if cached is not None:
        return cached

    gs = state._graph_stream
    for i, t in enumerate(state.lin_state):
        state._snap_lin[i].copy_(t)
    for i, t in enumerate(state.lin_conv_state):
        state._snap_conv[i].copy_(t)
    snap_k = state.attn.K_cache[:, pos:pos + 1].clone()
    snap_v = state.attn.V_cache[:, pos:pos + 1].clone()

    def _restore():
        for i, t in enumerate(state.lin_state):
            t.copy_(state._snap_lin[i])
        for i, t in enumerate(state.lin_conv_state):
            t.copy_(state._snap_conv[i])
        state.attn.K_cache[:, pos:pos + 1].copy_(snap_k)
        state.attn.V_cache[:, pos:pos + 1].copy_(snap_v)

    with torch.no_grad():               # settle allocator + kernel order
        for _ in range(2):
            decode_step(state, state._static_token, pos, fvk, device)
        _restore()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=gs, pool=state._graph_pool), \
            torch.no_grad():
        out = decode_step(state, state._static_token, pos, fvk, device)
    with torch.no_grad():
        _restore()

    state._graphs[pos] = (g, out)
    return state._graphs[pos]


def generate_greedy_graph(state, input_ids, max_new_tokens, fvk, device):
    """Greedy decode replaying a per-position CUDA graph.

    First visit to each pos captures (and runs) its graph; subsequent visits
    (warm cache / later generations) are pure replays. The graph reads the
    next token from ``state._static_token``.
    """
    pos = input_ids.view(-1).shape[0]
    logits = seed_prefill(state, input_ids, fvk, device)
    out = []
    for _ in range(max_new_tokens):
        nxt = int(logits[0].argmax().item())
        out.append(nxt)
        state._static_token.fill_(nxt)
        g, out_buf = _ensure_decode_graph(state, pos, fvk, device)
        g.replay()
        logits = out_buf
        pos += 1
    return out
