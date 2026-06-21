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

import collections
import os

import torch
import torch.nn.functional as F

from flash_rt.frontends.torch._nexn2_rtx_forward import (
    CONV, HD, HID, HK, HV, INTER, KD, KS, NKV, NQ, NV, ROPE, TOPK, VD,
    _quant_act, build_rope_tables, nexn2_forward_nvfp4,
)
from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import _sf_swz_bytes
from flash_rt.hardware.rtx.attn_backend_nexn2 import RtxFlashAttnBackendNexn2

# Prompt length at/above which the batched prefill wins over the per-token loop
# (batched has a fixed forward overhead; below this the loop's lower latency
# wins). See Nexn2DecodeState.batched_prefill.
_BATCHED_PREFILL_MIN_S = 8


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
    # MLP variant: 8 int4 loads in flight per warp -> bandwidth-bound at M=1
    # (1.5-3.4x the qwen36 matvec on the Nex-N2 shapes, cos=1.0).
    fvk.bf16_matvec_sm120_bf16(xc.data_ptr(), w.data_ptr(), y.data_ptr(),
                               n, k, _cs())
    return y


def _w4a16_mv(x1k, w_bf16, ld, key, fvk, device):
    """M=1 W4A16 GEMV: NVFP4 weight x BF16 activation. The bf16 weight is
    quantised to swizzled NVFP4 once (cached in `ld`, done on the first eager
    call before graph capture -> the .item()/sync is graph-safe), then the
    matvec reads 4-bit weight. ~2.2x the bf16 GEMV on the big projections,
    cos 0.994 (BF16 activation -> no activation-quant error)."""
    n, k = w_bf16.shape
    pk = key + '_w4a16_p'
    if pk not in ld:
        packed = torch.empty(n, k // 2, dtype=torch.uint8, device=device)
        sf = torch.zeros(_sf_swz_bytes(n, k), dtype=torch.uint8, device=device)
        scr = torch.zeros(1, dtype=torch.float32, device=device)
        og = torch.zeros(1, dtype=torch.float32, device=device)
        fvk.bf16_weight_to_nvfp4_swizzled(
            w_bf16.contiguous().data_ptr(), packed.data_ptr(), sf.data_ptr(),
            scr.data_ptr(), og.data_ptr(), n, k, _cs())
        torch.cuda.synchronize()
        ld[pk] = packed
        ld[key + '_w4a16_sf'] = sf
        ld[key + '_w4a16_a'] = float(og.item())
    xc = x1k.contiguous()
    y = torch.empty(1, n, dtype=torch.bfloat16, device=device)
    fvk.w4a16_matvec_sm120_bf16(
        xc.data_ptr(), ld[pk].data_ptr(), ld[key + '_w4a16_sf'].data_ptr(),
        y.data_ptr(), n, k, ld[key + '_w4a16_a'], _cs())
    return y


def _dense_mv(x1k, w_bf16, ld, key, state, fvk, device):
    """Dense projection GEMV: W4A16 when enabled, else the BF16 kernel."""
    if state.dense_w4a16:
        return _w4a16_mv(x1k, w_bf16, ld, key, fvk, device)
    return _bf16_mv(x1k, w_bf16, fvk, device)


def _silu_mul(g, u, fvk, device):
    """out = silu(g) * u via one fused kernel (was 4 torch ops). g, u bf16."""
    n = g.numel()
    gc = g.reshape(-1).contiguous()
    uc = u.reshape(-1).contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    fvk.silu_mul_sm120_bf16(gc.data_ptr(), uc.data_ptr(), out.data_ptr(),
                            n, _cs())
    return out.reshape(g.shape)


def _sigmoid_mul(x, gate, fvk, device):
    """out = x * sigmoid(gate) via one fused kernel. x, gate bf16."""
    n = x.numel()
    xc = x.reshape(-1).contiguous()
    gc = gate.reshape(-1).contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    fvk.sigmoid_mul_sm120_bf16(xc.data_ptr(), gc.data_ptr(), out.data_ptr(),
                               n, _cs())
    return out.reshape(x.shape)


def _rms_fvk(x, w, fvk, device, eps):
    """RMSNorm via the fused fvk kernel (fp32 internal) -- one launch vs the
    ~6 torch elementwise ops of the reference _rms. w is the (1+w)-folded
    weight; the kernel multiplies by it directly."""
    shp = x.shape
    k = shp[-1]
    x2 = x.reshape(-1, k).contiguous()
    out = torch.empty(x2.shape[0], k, dtype=torch.bfloat16, device=device)
    fvk.rms_norm(x2.data_ptr(), w.data_ptr(), out.data_ptr(),
                 x2.shape[0], k, eps, _cs())
    return out.reshape(shp)


def _proj_mma(x2d, ld, base, n, fvk, device, state=None):
    """Decode projection dispatch: NVFP4(W4A4) -> mma GEMV; else BF16 weight
    via W4A16 (when state.dense_w4a16) or the plain BF16 GEMV."""
    if ld.get(base + '_packed') is not None:
        return _mma(x2d, ld[base + '_packed'], ld[base + '_sf'],
                    ld[base + '_alpha'], n, fvk, device)
    w = ld[base + '_w_t']
    if state is not None and state.dense_w4a16:
        return _w4a16_mv(x2d, w, ld, base + '_w_t', fvk, device)
    return _bf16_mv(x2d, w, fvk, device)


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
        # pos -> (CUDAGraph, out tensor). LRU-bounded: each captured graph owns
        # its memory pool, so an unbounded cache leaks VRAM across a long
        # generation (one graph per absolute position). Evict the least-recently
        # used pos once over the cap; 0/negative disables the bound (legacy).
        self._graphs = collections.OrderedDict()
        self.graph_cache_max = int(
            os.environ.get('FLASHRT_NEXN2_GRAPH_CACHE_MAX', '256'))
        self._snap_lin = [torch.empty_like(t) for t in self.lin_state]
        self._snap_conv = [torch.empty_like(t) for t in self.lin_conv_state]
        # Pre-allocated KV snapshot rows (one [.,1,.] slice each) reused every
        # capture instead of a fresh clone() per pos -- zero alloc in the hot
        # capture path.
        self._snap_k = torch.empty_like(self.attn.K_cache[:, 0:1])
        self._snap_v = torch.empty_like(self.attn.V_cache[:, 0:1])
        # On-device greedy sampling. The captured decode graph runs argmax on
        # its own logits and writes the next token id straight back into
        # _static_token (the buffer the next replay re-reads), so a warm replay
        # is a single launch -- no per-step logits.argmax().item() D2H sync, no
        # separate argmax launch. Emitted ids accumulate in _out_tokens and are
        # read to the host once at the end. _snap_token preserves the input
        # token across the warmup/capture runs (which overwrite it via argmax).
        # NB decode is GPU (HBM)-bound, so this is ~parity for plain greedy
        # decode; its purpose is to keep the loop fully on-device as the
        # foundation for the spec-decode verify chain.
        self._out_tokens = torch.zeros(1, dtype=torch.long, device=device)
        self._snap_token = torch.zeros_like(self._static_token)
        # NVFP4 lm_head: 1GB -> 0.25GB read, +7% tok/s, decode cos 0.973 ->
        # 0.965. On by default (SOTA speed); set False for the bf16 lm_head.
        self.lm_head_nvfp4 = True
        # W4A16 dense projections (NVFP4 weight x BF16 activation): the dense
        # BF16 projections are the largest decode HBM bucket; reading 4-bit
        # weight instead of 16-bit is ~2.2x on the big shapes. BF16 activation
        # keeps cos high (no activation quant). `gdn_in_proj` is the GDN
        # in_proj_qkv (the W4A4 red line); W4A16 keeps BF16 activation so it is
        # gated separately and validated by E2E cos before enabling.
        self.dense_w4a16 = True
        self.gdn_in_proj_w4a16 = True
        # Batched prefill (one M=S forward seeding the decode state) instead of
        # the O(S) per-token loop: ~15x at S=128 and higher cos vs golden. Used
        # for prompts >= _BATCHED_PREFILL_MIN_S (below that the per-token path's
        # lower fixed overhead wins). Set False to force the per-token path.
        self.batched_prefill = True

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
    # Fuse the 4 K=2048 in_proj GEMVs into one matvec: under CUDA graph the
    # kernels run serially and each pays a fixed K-loop latency regardless of
    # N, so one (12352, 2048) matvec replaces four (saves 3 K-loops/layer).
    if 'in_proj_fused_w' not in ld:
        ld['in_proj_fused_w'] = torch.cat(
            [Wqkv, Wz, Wa, Wb], 0).contiguous()
    if state.dense_w4a16 and state.gdn_in_proj_w4a16:
        fused = _w4a16_mv(h2, ld['in_proj_fused_w'], ld, 'in_proj_fused',
                          fvk, device)
    else:
        fused = _bf16_mv(h2, ld['in_proj_fused_w'], fvk, device)
    mixed = fused[:, :KD * 2 + VD].contiguous()
    z = fused[:, KD * 2 + VD:KD * 2 + VD + NV * HV].reshape(NV, HV).contiguous()
    a = fused[:, -2 * NV:-NV].contiguous()
    b = fused[:, -NV:].contiguous()

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
    fvk.qwen35moe_lin_split_qkv_broadcast_bf16(
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
    out = _proj_mma(nf.reshape(1, VD), ld, 'out_proj', HID, fvk, device, state)
    return out.reshape(1, 1, HID)


def _decode_full(h, ld, state, full_rank, pos, fvk, device):
    """Full-attn layer at one token; writes KV at pos, attends [0..pos]."""
    eps = state.eps
    s = _cs()
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']
    x2 = h.reshape(1, HID)

    nqg = NQ * 2 * HD
    if ld.get('q_proj_packed') is None:     # experts-scope: fuse BF16 q/k/v
        if 'qkv_fused_w' not in ld:
            ld['qkv_fused_w'] = torch.cat(
                [ld['q_proj_w_t'], ld['k_proj_w_t'], ld['v_proj_w_t']],
                0).contiguous()
        fused = _dense_mv(x2, ld['qkv_fused_w'], ld, 'qkv_fused', state,
                          fvk, device)
        qg = fused[:, :nqg].contiguous()
        k = fused[:, nqg:nqg + NKV * HD].reshape(NKV, HD)
        v = fused[:, nqg + NKV * HD:].reshape(1, NKV, HD)
    else:
        qg = _proj_mma(x2, ld, 'q_proj', nqg, fvk, device, state).contiguous()
        k = _proj_mma(x2, ld, 'k_proj', NKV * HD, fvk, device, state).reshape(
            NKV, HD)
        v = _proj_mma(x2, ld, 'v_proj', NKV * HD, fvk, device, state).reshape(
            1, NKV, HD)
    q_pre = torch.empty(1, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(1, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_split_q_gate_bf16(
        qg.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), 1, s)
    q = _rms_fvk(q_pre.reshape(NQ, HD), qnw, fvk, device, eps).reshape(
        1, NQ, HD)
    k = _rms_fvk(k, knw, fvk, device, eps).reshape(1, NKV, HD)

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
    at = _sigmoid_mul(at, gate, fvk, device)
    return _proj_mma(at, ld, 'o_proj', HID, fvk, device, state).reshape(
        1, 1, HID)


def _moe_layer_decode(h, ld, state, fvk, device):
    """M=1 fine-grained MoE via the grouped GEMV kernel: the 8 routed experts
    run in one launch each for gate_up (shared act) and down (per-slot act),
    indexed by a device top-k id buffer (the same buffer drives a graph)."""
    s = _cs()
    x = h.reshape(1, HID)
    # Router + shared gate/up all read the same post-norm activation at K=HID,
    # so when they take the W4A16 path they fuse into one GEMV (concat weights,
    # split outputs). Under graph replay each tiny GEMV pays a ~2 us latency
    # floor, so collapsing three latency-bound launches into one big-N read is a
    # real saving (the elementwise glue is not -- no launch cost to remove).
    ne = ld['router_w_t'].shape[0]                          # n_experts (256)
    fused_rs = (state.dense_w4a16 and ld.get('router_packed') is None
                and ld.get('shared_gate_proj_packed') is None)
    if fused_rs:
        if 'router_shared_fused_w' not in ld:
            ld['router_shared_fused_w'] = torch.cat(
                [ld['router_w_t'], ld['shared_gate_proj_w_t'],
                 ld['shared_up_proj_w_t']], 0).contiguous()
        rs = _w4a16_mv(x, ld['router_shared_fused_w'], ld,
                       'router_shared_fused', fvk, device)
        logit_raw = rs[:, :ne]
        sg_f = rs[:, ne:ne + INTER]
        su_f = rs[:, ne + INTER:]
    else:
        logit_raw = _dense_mv(x, ld['router_w_t'], ld, 'router', state,
                              fvk, device)
    # Router top-8 of the raw logits via one kernel (was softmax(256) +
    # torch.topk bitonic sort). Re-normalising top-8 of softmax(256) equals
    # softmax(top-8 logits), so softmax the 8 returned logits.
    lr = logit_raw.reshape(-1).contiguous()
    idx = torch.empty(TOPK, dtype=torch.int32, device=device)
    topv = torch.empty(TOPK, dtype=torch.float32, device=device)
    fvk.moe_router_topk_sm120_bf16(lr.data_ptr(), idx.data_ptr(), topv.data_ptr(),
                               lr.numel(), TOPK, s)
    tw_row = F.softmax(topv, -1)                             # (TOPK,) device

    if 'experts_gate_up_alpha_dev' not in ld:               # cache once/layer
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]               # 1024 / HID

    # gate_up: shared BF16 activation, grouped W4A16 over the 8 experts. BF16
    # activation -> no activation quant, higher cos than the W4A4 mma, and
    # faster at this scale (6.2 vs 8.2 us standalone).
    xc = x.contiguous()
    d_gu = torch.empty(TOPK, n_gu, dtype=torch.bfloat16, device=device)
    fvk.moe_grouped_w4a16_sm120_bf16(
        xc.data_ptr(), gu_p.data_ptr(), gu_s.data_ptr(), gu_a.data_ptr(),
        idx.data_ptr(), d_gu.data_ptr(), TOPK, n_gu, HID,
        0, gu_p[0].numel(), gu_s[0].numel(), s)

    # down: silu(gate)*up (BF16, fused) then grouped W4A16 (per-slot activation).
    g_, u_ = d_gu[:, :INTER], d_gu[:, INTER:]
    inter = _silu_mul(g_, u_, fvk, device).contiguous()
    d_dn = torch.empty(TOPK, n_dn, dtype=torch.bfloat16, device=device)
    fvk.moe_grouped_w4a16_sm120_bf16(
        inter.data_ptr(), dn_p.data_ptr(), dn_s.data_ptr(), dn_a.data_ptr(),
        idx.data_ptr(), d_dn.data_ptr(), TOPK, n_dn, INTER,
        INTER, dn_p[0].numel(), dn_s[0].numel(), s)
    out = (tw_row.float() @ d_dn.float()).unsqueeze(0)   # (1, n_dn) weighted sum

    if fused_rs:                                      # already projected above
        sg, su = sg_f, su_f
    elif ld.get('shared_gate_proj_packed') is None:  # experts-scope: fuse g/u
        if 'shared_gu_fused_w' not in ld:
            ld['shared_gu_fused_w'] = torch.cat(
                [ld['shared_gate_proj_w_t'], ld['shared_up_proj_w_t']],
                0).contiguous()
        gu = _dense_mv(x, ld['shared_gu_fused_w'], ld, 'shared_gu_fused',
                       state, fvk, device)
        sg, su = gu[:, :INTER], gu[:, INTER:]
    else:
        sg = _proj_mma(x, ld, 'shared_gate_proj', INTER, fvk, device, state)
        su = _proj_mma(x, ld, 'shared_up_proj', INTER, fvk, device, state)
    si = _silu_mul(sg, su, fvk, device)
    shared = _proj_mma(si, ld, 'shared_down_proj', HID, fvk, device, state)
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
        n = _rms_fvk(h, ld['input_norm_w_t'], fvk, device, state.eps)
        if state.types[L] == 'linear_attention':
            attn = _decode_gdn(n, ld, state, state._lin_rank[L], fvk, device)
        else:
            attn = _decode_full(n, ld, state, state._full_rank[L], pos,
                                fvk, device)
        h = res + attn
        res = h
        n = _rms_fvk(h, ld['post_norm_w_t'], fvk, device, state.eps)
        h = res + _moe_layer_decode(n, ld, state, fvk, device)

    h = _rms_fvk(h, p['final_norm_w_t'], fvk, device, state.eps)
    # lm_head as NVFP4 W4A16: 4x less weight read (1GB -> 0.25GB) via the
    # hand-tuned mma (3.1x the bf16 GEMV; the CUTLASS widen is M=1-broken).
    # The weight is quantised once during the eager seed (cached on p), so
    # the captured graph only runs the activation quant + fp4 GEMM.
    vocab = p['vocab_size']
    logits = torch.empty(1, vocab, dtype=torch.bfloat16, device=device)
    if not state.lm_head_nvfp4:
        fvk.bf16_matvec_sm120_bf16(
            h.reshape(1, HID).contiguous().data_ptr(),
            p['lm_head_w_t'].data_ptr(), logits.data_ptr(), vocab, HID, _cs())
        return logits
    if 'lm_head_packed_t' not in p:
        w = p['lm_head_w_t'].contiguous()
        nn, kk = w.shape
        packed = torch.empty(nn, kk // 2, dtype=torch.uint8, device=device)
        sf = torch.zeros(_sf_swz_bytes(nn, kk), dtype=torch.uint8, device=device)
        scr = torch.zeros(1, dtype=torch.float32, device=device)
        og = torch.zeros(1, dtype=torch.float32, device=device)
        fvk.bf16_weight_to_nvfp4_swizzled(
            w.data_ptr(), packed.data_ptr(), sf.data_ptr(),
            scr.data_ptr(), og.data_ptr(), nn, kk, 0)
        torch.cuda.synchronize()
        p['lm_head_packed_t'] = packed
        p['lm_head_sf_t'] = sf
        p['lm_head_alpha'] = float(og.item())
    xp, xsf = _quant_act(h.reshape(1, HID), fvk, device, _cs())
    fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
        xp.data_ptr(), p['lm_head_packed_t'].data_ptr(), logits.data_ptr(),
        vocab, HID, xsf.data_ptr(), p['lm_head_sf_t'].data_ptr(),
        p['lm_head_alpha'], _cs())
    return logits


def seed_prefill(state, input_ids, fvk, device):
    """Run the decode step over prompt tokens 0..S-1, building all state.

    Returns the last-token logits (1, vocab).
    """
    ids = input_ids.view(-1)
    if state.batched_prefill and ids.shape[0] >= _BATCHED_PREFILL_MIN_S:
        return seed_prefill_batched(state, input_ids, fvk, device)
    state.reset()
    last = None
    for pos in range(ids.shape[0]):
        last = decode_step(state, ids[pos:pos + 1], pos, fvk, device)
    return last


def seed_prefill_batched(state, input_ids, fvk, device):
    """Batched prefill: one forward pass over the whole prompt seeds the decode
    state (GDN recurrent/conv + KV cache), instead of looping the per-token
    decode S times. Returns the last-token logits (1, vocab).

    Produces the same decode state as ``seed_prefill`` (the per-token path) but
    runs the prompt through batched (M=S) projections / attention, which is far
    faster for long prompts. The state capture is done inside the forward layers
    when ``cap`` is passed (see _gdn_layer / _full_attn_layer)."""
    state.reset()
    logits = nexn2_forward_nvfp4(
        state.handles, input_ids.view(1, -1), fvk, device, cap=state)
    return logits[-1:].contiguous()


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
    re-read each replay. When ``qwen36_argmax_bf16`` is available the greedy
    argmax of the step's logits is captured *inside* the graph and written back
    into ``_static_token``, so a warm replay both decodes and produces the next
    input token in one launch (the token buffer is also snapshotted/restored,
    since the warmup+capture argmax overwrites it). The lin/conv/KV state
    mutated by the 2 warmup runs + the capture run is snapshotted and restored,
    so a later replay advances from the true pre-step state.
    Returns (graph, logits_buffer).
    """
    cached = state._graphs.get(pos)
    if cached is not None:
        state._graphs.move_to_end(pos)          # mark MRU
        return cached

    bake_argmax = hasattr(fvk, 'qwen36_argmax_bf16')
    vocab = state.handles.ptrs['vocab_size']
    gs = state._graph_stream
    for i, t in enumerate(state.lin_state):
        state._snap_lin[i].copy_(t)
    for i, t in enumerate(state.lin_conv_state):
        state._snap_conv[i].copy_(t)
    # Reuse the pre-allocated [.,1,.] KV rows instead of cloning a fresh tensor
    # every capture (zero alloc in the hot capture path).
    state._snap_k.copy_(state.attn.K_cache[:, pos:pos + 1])
    state._snap_v.copy_(state.attn.V_cache[:, pos:pos + 1])
    state._snap_token.copy_(state._static_token)

    def _restore():
        for i, t in enumerate(state.lin_state):
            t.copy_(state._snap_lin[i])
        for i, t in enumerate(state.lin_conv_state):
            t.copy_(state._snap_conv[i])
        state.attn.K_cache[:, pos:pos + 1].copy_(state._snap_k)
        state.attn.V_cache[:, pos:pos + 1].copy_(state._snap_v)
        state._static_token.copy_(state._snap_token)

    with torch.no_grad():               # settle allocator + kernel order
        for _ in range(2):
            decode_step(state, state._static_token, pos, fvk, device)
        _restore()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=gs, pool=state._graph_pool), \
            torch.no_grad():
        out = decode_step(state, state._static_token, pos, fvk, device)
        if bake_argmax:
            fvk.qwen36_argmax_bf16(
                out.data_ptr(), state._static_token.data_ptr(),
                1, vocab, _cs())
    with torch.no_grad():
        _restore()

    state._graphs[pos] = (g, out)
    cap = state.graph_cache_max
    if cap > 0 and len(state._graphs) > cap:
        state._graphs.popitem(last=False)       # evict LRU
    return state._graphs[pos]


def generate_greedy_graph(state, input_ids, max_new_tokens, fvk, device):
    """Greedy decode replaying a per-position CUDA graph.

    First visit to each pos captures (and runs) its graph; subsequent visits
    (warm cache / later generations) are pure replays. The graph reads the
    next token from ``state._static_token``.

    Greedy sampling runs on-device: ``qwen36_argmax_bf16`` writes the argmax
    token id straight into ``_static_token`` (which the next replay re-reads)
    and the emitted ids accumulate in ``_out_tokens``, read back to the host
    once at the end. This removes the per-step ``logits.argmax().item()`` D2H
    sync that otherwise serialises the decode loop (CPU blocks on the GPU every
    token). Falls back to the host argmax when the kernel is unavailable.
    """
    pos = input_ids.view(-1).shape[0]
    logits = seed_prefill(state, input_ids, fvk, device)
    vocab = state.handles.ptrs['vocab_size']

    if not hasattr(fvk, 'qwen36_argmax_bf16'):
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

    if state._out_tokens.numel() < max_new_tokens:
        state._out_tokens = torch.zeros(
            max_new_tokens, dtype=torch.long, device=device)
    out_tokens = state._out_tokens
    # Seed token: argmax of the last prompt-step logits -> _static_token.
    # Every subsequent next-token argmax is baked into the replayed graph,
    # so the loop body is a single launch + a device-side token copy.
    fvk.qwen36_argmax_bf16(
        logits.data_ptr(), state._static_token.data_ptr(), 1, vocab, _cs())
    for i in range(max_new_tokens):
        out_tokens[i].copy_(state._static_token.view(-1)[0])
        g, _ = _ensure_decode_graph(state, pos, fvk, device)
        g.replay()                # decode + argmax -> _static_token (next tok)
        pos += 1
    return out_tokens[:max_new_tokens].tolist()       # single D2H sync
