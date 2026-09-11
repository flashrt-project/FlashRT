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
    _quant_act, _w4a16_mrows, build_rope_tables, kernel_policy,
    moe_grouped_w4a16, nexn2_forward_nvfp4, set_spec_verify, w4a16_matvec,
)
from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import _sf_swz_bytes
from flash_rt.hardware.rtx.attn_backend_nexn2 import RtxFlashAttnBackendNexn2


def _qwen35moe_env(name: str, default: str) -> str:
    generic = f"FLASHRT_QWEN35MOE_{name}"
    legacy = f"FLASHRT_NEXN2_{name}"
    return os.environ.get(generic, os.environ.get(legacy, default))


# Prompt length at/above which the batched prefill wins over the per-token loop
# (batched has a fixed forward overhead; below this the loop's lower latency
# wins). See Nexn2DecodeState.batched_prefill.
_BATCHED_PREFILL_MIN_S = 8


def _cache_put(cache, key, value, cap):
    """Insert ``value`` and evict least-recently-used down to ``cap``.

    Both graph caches go through this. A captured graph owns its memory pool,
    so an unbounded cache leaks device memory across a long generation -- one
    graph per absolute position. ``cap <= 0`` disables the bound.

    Returns the value, so a caller can insert and use in one expression.
    """
    cache[key] = value
    if cap > 0:
        while len(cache) > cap:
            cache.popitem(last=False)           # evict LRU
    return value


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
    w4a16_matvec(fvk)(
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

    def __init__(self, handles, max_seq, device, *,
                 spec_graph_cache_max=None):
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

        # Full-attn KV cache. A loaded draft head is one more full-attention
        # layer and takes the slot after the model's own.
        self.mtp = p.get('mtp')
        self.mtp_rank = nfull if self.mtp is not None else None
        self.attn = RtxFlashAttnBackendNexn2(
            max_seq=self.max_seq, max_q_seq=1,
            num_full_layers=nfull + (1 if self.mtp is not None else 0))
        # The pre-final-norm hidden state of the last step, which is what the
        # draft head reads. Written every step whether or not one is loaded:
        # a 4 KB device copy, and making it conditional would put a Python
        # branch inside the captured region.
        self.last_hidden = torch.zeros(HID, dtype=bf16, device=device)

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
            _qwen35moe_env("GRAPH_CACHE_MAX", "256"))
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
        # Above this prompt length the batched prefill is run in token-blocks
        # of this size (chunked prefill), carrying the GDN recurrent/conv state
        # and KV cache across blocks so the per-layer activation memory stays
        # bounded by the block (not S) -- this is what lets context scale past
        # the ~16k single-pass ceiling on a 32 GB card. A multiple of the WY
        # chunk (64). 0 disables (always single-pass).
        self.prefill_chunk = int(
            _qwen35moe_env("PREFILL_CHUNK", "8192"))
        # Optional eager-only traces used to size edge expert caches and to
        # score expert quantization against real activations. Keep these
        # disabled during CUDA Graph capture.
        self.router_trace = None
        self.moe_input_trace = None
        self._active_layer = -1
        # Per-token recurrent/conv snapshots for a speculative window, sized
        # on first use. Only allocated when speculation runs: 30 layers of
        # (NV, HK, HV) bf16 per window slot.
        self.spec_states = None
        self.spec_conv = None
        # Set only around a verify block. Prefill runs the same layer code and
        # would otherwise pay for -- and overrun -- snapshots it never uses.
        self.spec_capture = False
        # One captured graph per (pos, window): the KV slots, attention length
        # and RoPE slice are baked per position exactly as the decode graph's
        # are, and each owns a memory pool, so this is LRU-bounded the same way.
        #
        # It is NOT bounded at the same number. A speculative graph covers k+1
        # positions through the whole stack, so its pool is several times a
        # decode step's, and the decode cap of 256 is sized for a step. Holding
        # 256 of these alongside the model is more than a 32 GB board has at a
        # 2048-token context -- measured there, it is what runs it out of
        # memory. Sixteen keeps the windows a generation actually revisits
        # (recapture costs two warmup runs) and bounds the pools at something
        # the smallest supported board carries.
        self._spec_graphs = collections.OrderedDict()
        self.spec_graph_cache_max = int(
            spec_graph_cache_max if spec_graph_cache_max is not None
            else _qwen35moe_env("SPEC_GRAPH_CACHE_MAX", "16"))
        # Its own memory pool, not the decode graphs'. The two are replayed
        # interleaved -- a window, then whatever the caller does next -- and
        # sharing a pool between graphs used that way is the case the runtime
        # does not promise to handle. Measured: with the pool shared, a 64-token
        # speculative run ran at half the rate of a 32-token one on identical
        # code, the cost growing with the number of live graphs.
        self._spec_pool = torch.cuda.graph_pool_handle()
        self._spec_tokens = None
        self._spec_argmax = None
        # Which half of the draft head's fc input carries the hidden state.
        # The checkpoint does not say and fc is square in the concatenated
        # width, so it was settled by measuring acceptance both ways -- and the
        # answer is the embedding first. Over 48 decoded tokens:
        #
        #   cat[embed, hidden]   first draft 0.896, chained 0.646, 0.417
        #   cat[hidden, embed]   0.000, 0.000, 0.000
        #
        # The wrong half drafts noise, so nothing is ever accepted and every
        # window pays for a verify that keeps one token. It agrees with the
        # reference implementation of this head, which concatenates the
        # embedding first as well.
        self.mtp_hidden_first = (
            _qwen35moe_env("MTP_HIDDEN_FIRST", "0") != "0")
        # Set to an ExpertCache to read the routed experts from storage. Only
        # meaningful when the loader skipped them; see _moe_experts_streamed.
        self.expert_cache = None
        self._scratch = None
        self._hadamard = None

    def _streamed_scratch(self, device):
        """The two decode buffers a streamed expert is unpacked into.

        Allocated once and reused: 4 MiB for gate_up and 2 MiB for down, which
        would otherwise be allocated 8 times per layer per token.
        """
        if self._scratch is None:
            self._scratch = {
                'gate_up': torch.empty(
                    2 * INTER, HID, dtype=torch.bfloat16, device=device),
                'down': torch.empty(
                    HID, INTER, dtype=torch.bfloat16, device=device),
            }
        return self._scratch

    def reset(self):
        for s in self.lin_state:
            s.zero_()
        for c in self.lin_conv_state:
            c.zero_()
        self.attn.reset_cache()


def router_topk(fvk):
    """The router top-k entry this build should call.

    The warp variant returns identical indices and values -- argmax under a
    total order picks one element regardless of the reduction tree, checked
    over 800 inputs of which 397 had a tie inside the top-8 -- without the
    block kernel's 24 barriers.
    """
    if kernel_policy().warp_router_topk:
        fn = getattr(fvk, 'moe_router_topk_warp_sm120_bf16', None)
        if fn is not None:
            return fn
    return fvk.moe_router_topk_sm120_bf16


def gdn_recurrent(fvk):
    """The single-token GDN recurrence entry this build should call.

    The edge variant is the same arithmetic in the same order -- checked
    exactly, over chained steps so the state drift is exercised too -- and it is
    1.70x the shipped one. The shipped one holds the thread's whole state column
    in a 128-float array that cannot live in registers, so it is in local
    memory and walked five times; ncu measures 39 registers per thread for a
    128-float array. 51% of bandwidth against 87%.
    """
    if kernel_policy().gdn_recurrent_edge:
        fn = getattr(fvk, 'gated_deltanet_recurrent_edge_qwen36_bf16', None)
        if fn is not None:
            return fn
    return fvk.gated_deltanet_recurrent_qwen36_bf16


def _gdn_gate_consts(ld, device):
    """The gating kernel's two constant inputs, derived once per layer.

    ``A_log`` and ``dt_bias`` are weights, so -exp(A_log) and the fp32 bias are
    the same on every step -- but deriving them per call put four elementwise
    launches per GDN layer inside the captured region, thirty layers of them,
    recomputing values identical to the previous replay's. Each is a couple of
    microseconds of dispatch quantum for no arithmetic anyone reads.

    Same expressions in the same order, so the bytes handed to the kernel are
    the bytes it was getting before.
    """
    if 'gdn_neg_exp_a' not in ld:
        ld['gdn_neg_exp_a'] = (
            -ld['A_log_t'].float().exp()).float().contiguous()
        ld['gdn_dt_bias_f'] = ld['dt_bias_t'].float().contiguous()
    return ld['gdn_neg_exp_a'], ld['gdn_dt_bias_f']


def _decode_gdn(h, ld, state, lin_rank, fvk, device):
    """GDN layer at one token, updating recurrent + conv state in place."""
    eps = state.eps
    Wqkv = ld['in_proj_qkv_w_t']
    Wz = ld['in_proj_z_w_t']
    Wb, Wa = ld['in_proj_b_w_t'], ld['in_proj_a_w_t']
    convw = ld['conv1d_w_t'].reshape(CONV, KS).contiguous()
    neg, dtb_c = _gdn_gate_consts(ld, device)
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
    gdn_recurrent(fvk)(
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


def _hadamard16(device):
    """The block-16 transform, built once. Symmetric and its own inverse."""
    m = torch.ones(1, 1, dtype=torch.float32, device=device)
    for _ in range(4):
        m = torch.cat((torch.cat((m, m), 1), torch.cat((m, -m), 1)), 0)
    return m / 4.0


def _rotate16(x, h):
    """Apply the transform along the last dimension, in blocks of 16."""
    shape = x.shape
    return (x.reshape(-1, 16).float() @ h).reshape(shape).to(x.dtype)


def _moe_experts_streamed(x, idx, state, fvk, device, s):
    """The routed experts' outputs, read from storage instead of memory.

    Returns only ``d_dn`` -- the per-slot expert outputs. The weighted sum, the
    shared expert and its gate are identical to the resident path and stay
    there; replacing the whole layer here is how an earlier version silently
    dropped the shared expert from every layer.

    Reachable only when the loader was told to stream, in which case the
    per-layer stacked tensors were never allocated. Each block is decoded to
    bf16 and multiplied with the shared bf16 GEMV, because the block-scaled
    4-bit GEMMs read neither this codebook nor this scale layout.

    When the bundle was written with the transform applied, the stored weight is
    H*W, so the activation entering each GEMM has to be rotated the same way or
    the products are wrong -- while staying finite and plausible, which is
    exactly how it goes unnoticed.
    """
    cache = state.expert_cache
    layer = state._active_layer
    experts = [int(value) for value in idx.cpu().tolist()]
    cache.get_many(layer, experts)

    rotated = bool(cache.manifest.get('rht'))
    if rotated and state._hadamard is None:
        state._hadamard = _hadamard16(device)
    h16 = state._hadamard

    scratch = state._streamed_scratch(device)
    d_gu = torch.empty(TOPK, 2 * INTER, dtype=torch.bfloat16, device=device)
    d_dn = torch.empty(TOPK, HID, dtype=torch.bfloat16, device=device)
    xc = (_rotate16(x, h16) if rotated else x).contiguous()

    for slot, expert in enumerate(experts):
        parts = cache.components(layer, expert)
        gu_alpha, dn_alpha = parts['global_scales'].tolist()
        rc = fvk.qwen35moe_e0m3_dequant_bf16(
            parts['gate_up_weight'].data_ptr(),
            parts['gate_up_scale'].data_ptr(),
            scratch['gate_up'].data_ptr(),
            2 * INTER, HID, cache.group_size, gu_alpha, s)
        if rc:
            raise RuntimeError(f'gate_up decode failed with {rc}')
        fvk.bf16_matvec_sm120_bf16(
            xc.data_ptr(), scratch['gate_up'].data_ptr(),
            d_gu[slot].data_ptr(), 2 * INTER, HID, s)

        gated = _silu_mul(
            d_gu[slot:slot + 1, :INTER], d_gu[slot:slot + 1, INTER:],
            fvk, device)
        if rotated:
            gated = _rotate16(gated, h16)
        gated = gated.contiguous()
        rc = fvk.qwen35moe_e0m3_dequant_bf16(
            parts['down_weight'].data_ptr(),
            parts['down_scale'].data_ptr(),
            scratch['down'].data_ptr(),
            HID, INTER, cache.group_size, dn_alpha, s)
        if rc:
            raise RuntimeError(f'down decode failed with {rc}')
        fvk.bf16_matvec_sm120_bf16(
            gated.data_ptr(), scratch['down'].data_ptr(),
            d_dn[slot].data_ptr(), HID, INTER, s)
    return d_dn


def _shared_combine(routed, shared, glog, rows, fvk, device):
    """out = routed(fp32) + shared(bf16) * sigmoid(gate), in one kernel.

    The tensor-op form is a cast, a sigmoid, a broadcast multiply, an add and a
    cast: five launches a layer, forty layers, in a step that is 99% kernel
    time and where a launch costs its dispatch quantum whether or not it
    computes much. The kernel does the same arithmetic in the same order and
    rounds once at the store, so it stands in for the chain rather than
    approximating it -- checked bit for bit at the shapes and scales decode and
    the window issue, because the fixture and the speculative verify both rest
    on it.
    """
    if (kernel_policy().fused_shared_combine
            and hasattr(fvk, 'moe_shared_gate_combine_edge_bf16')):
        out = torch.empty(rows, HID, dtype=torch.bfloat16, device=device)
        fvk.moe_shared_gate_combine_edge_bf16(
            routed.data_ptr(), shared.data_ptr(), glog.data_ptr(),
            out.data_ptr(), rows, HID, _cs())
        return out
    sgate = torch.sigmoid(glog.float()).reshape(rows, 1)
    return (routed + shared.float() * sgate).to(torch.bfloat16)


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
    # idx and topv come from torch.empty, so an unchecked failure here leaves
    # uninitialised memory to be used as expert indices -- which reaches a file
    # offset before anything notices.
    rc = router_topk(fvk)(
        lr.data_ptr(), idx.data_ptr(), topv.data_ptr(), lr.numel(), TOPK, s)
    if rc:
        raise RuntimeError(
            f'router top-k failed with {rc} for {lr.numel()} experts, k={TOPK}')
    tw_row = F.softmax(topv, -1)                             # (TOPK,) device
    if state.router_trace is not None:
        state.router_trace[state._active_layer].append(
            tuple(int(v) for v in idx.cpu().tolist()))
    if state.moe_input_trace is not None:
        state.moe_input_trace[state._active_layer].append(
            x.detach().to("cpu", copy=True))

    # Streaming replaces only the routed experts' own GEMVs. Everything after
    # this -- the weighted sum, the shared expert, its gate -- is identical, and
    # returning early from here is how an earlier version silently dropped the
    # shared expert from every layer.
    if ld.get('experts_streamed'):
        d_dn = _moe_experts_streamed(x, idx, state, fvk, device, s)
        n_dn = HID
    else:
        if 'experts_gate_up_alpha_dev' not in ld:           # cache once/layer
            ld['experts_gate_up_alpha_dev'] = \
                ld['experts_gate_up_alpha_t'].to(device).contiguous()
            ld['experts_down_alpha_dev'] = \
                ld['experts_down_alpha_t'].to(device).contiguous()
        gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
        dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
        gu_a = ld['experts_gate_up_alpha_dev']
        dn_a = ld['experts_down_alpha_dev']
        n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]           # 1024 / HID

        # gate_up: shared BF16 activation, grouped W4A16 over the 8 experts.
        # BF16 activation -> no activation quant, higher cos than the W4A4 mma,
        # and faster at this scale (6.2 vs 8.2 us standalone).
        xc = x.contiguous()
        d_gu = torch.empty(TOPK, n_gu, dtype=torch.bfloat16, device=device)
        moe_grouped_w4a16(fvk)(
            xc.data_ptr(), gu_p.data_ptr(), gu_s.data_ptr(), gu_a.data_ptr(),
            idx.data_ptr(), d_gu.data_ptr(), TOPK, n_gu, HID,
            0, gu_p[0].numel(), gu_s[0].numel(), s)

        # down: silu(gate)*up (BF16, fused) then grouped W4A16 (per-slot act).
        g_, u_ = d_gu[:, :INTER], d_gu[:, INTER:]
        inter = _silu_mul(g_, u_, fvk, device).contiguous()
        d_dn = torch.empty(TOPK, n_dn, dtype=torch.bfloat16, device=device)
        moe_grouped_w4a16(fvk)(
            inter.data_ptr(), dn_p.data_ptr(), dn_s.data_ptr(), dn_a.data_ptr(),
            idx.data_ptr(), d_dn.data_ptr(), TOPK, n_dn, INTER,
            INTER, dn_p[0].numel(), dn_s[0].numel(), s)
    # Fixed-order weighted sum. The generic torch matmul may choose a
    # reduction whose accumulation order changes between launches, which can
    # flip a later greedy decision when two logits are nearly tied.
    if 'decode_topk_rows' not in ld:
        ld['decode_topk_rows'] = torch.arange(
            TOPK, dtype=torch.int32, device=device)
    out = torch.empty(n_dn, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), ld['decode_topk_rows'].data_ptr(),
        tw_row.data_ptr(), out.data_ptr(),
        1, TOPK, n_dn, n_dn, s)
    out = out.unsqueeze(0)

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
    # shared-expert scalar gate: N=1 GEMV via the bf16 matvec kernel (was a
    # torch matmul -- the last fp32 matmul in the captured decode step). The
    # sigmoid, the broadcast multiply, the add and the cast are one kernel.
    glog = _bf16_mv(x, ld['shared_gate_w_t'], fvk, device)
    return _shared_combine(out, shared, glog, 1, fvk, device).reshape(
        1, 1, HID)


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
        state._active_layer = L
        h = res + _moe_layer_decode(n, ld, state, fvk, device)

    # The pre-final-norm hidden state is what a DeepSeek-V3-style draft head
    # consumes. Keeping it in a fixed buffer costs one 4 KB device copy and
    # survives graph capture, unlike reading it out per step.
    state.last_hidden.copy_(h.reshape(HID))
    h = _rms_fvk(h, p['final_norm_w_t'], fvk, device, state.eps)
    # lm_head as NVFP4 W4A16: 4x less weight read (1GB -> 0.25GB) via the
    # hand-tuned mma (3.1x the bf16 GEMV; the CUTLASS widen is M=1-broken).
    # The weight is quantised once during the eager seed (cached on p), so
    # the captured graph only runs the activation quant + fp4 GEMM.
    return _lm_head(state, h, fvk, device)


def _ensure_lm_head_nvfp4(state, fvk, device):
    """Quantise the lm_head to swizzled NVFP4 once, on the handles.

    Both the single-row decode head and the M-row verify read this one copy,
    so the verify cannot drift from decode by having been handed a second
    quantisation of the same weight. The .item() lands here, on the first
    eager call, and never inside a captured region.
    """
    p = state.handles.ptrs
    if 'lm_head_packed_t' in p:
        return
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


def _lm_head(state, h, fvk, device):
    """Project a hidden state to logits over the full vocabulary.

    Taken out of decode_step so the speculative draft head, which ends the
    same way, does not carry a second copy of the quantise-once bookkeeping.
    """
    p = state.handles.ptrs
    vocab = p['vocab_size']
    logits = torch.empty(1, vocab, dtype=torch.bfloat16, device=device)
    if not state.lm_head_nvfp4:
        fvk.bf16_matvec_sm120_bf16(
            h.reshape(1, HID).contiguous().data_ptr(),
            p['lm_head_w_t'].data_ptr(), logits.data_ptr(), vocab, HID, _cs())
        return logits
    _ensure_lm_head_nvfp4(state, fvk, device)
    if hasattr(fvk, 'fp4_w4a4_mma_sm120_full_n_bf16out'):
        xp, xsf = _quant_act(h.reshape(1, HID), fvk, device, _cs())
        fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
            xp.data_ptr(), p['lm_head_packed_t'].data_ptr(),
            logits.data_ptr(), vocab, HID, xsf.data_ptr(),
            p['lm_head_sf_t'].data_ptr(), p['lm_head_alpha'], _cs())
        return logits
    # That kernel is built only for GPU_ARCH 120/121, so on every other target
    # this path had no implementation at all. The W4A16 matvec reads the same
    # swizzled weight and the same scale factors, leaves the activation in
    # bf16 -- so it also skips the activation quantisation and its error -- and
    # lives in a tier that builds wherever the core does.
    w4a16_matvec(fvk)(
        h.reshape(1, HID).contiguous().data_ptr(),
        p['lm_head_packed_t'].data_ptr(), p['lm_head_sf_t'].data_ptr(),
        logits.data_ptr(), vocab, HID, p['lm_head_alpha'], _cs())
    return logits


def mtp_draft(state, token_id, pos, fvk, device, *, hidden=None):
    """Draft the token after next with the MTP head.

    A DeepSeek-V3 single-module head: it sees the main model's last hidden
    state for position p-1 and the token emitted at p, and predicts p+1. The
    layer under it is an ordinary full-attention layer with its own MoE, so it
    runs through the same per-layer code as the model -- which is the point of
    loading it through the same loader.

    ``hidden`` defaults to the buffer the last decode step wrote. The head
    carries its own KV at the same absolute positions as the model, so calling
    this advances that cache and nothing else.
    """
    p = state.handles.ptrs
    mtp = state.mtp
    if mtp is None:
        raise RuntimeError(
            'no MTP head is loaded; build the frontend with speculation '
            'enabled so the loader reads it')
    ld = mtp['layer']
    h_prev = state.last_hidden if hidden is None else hidden

    # The draft head runs on its BF16 weights, not on the runtime W4A16 the
    # model's own projections take. A draft is one layer -- its weights are a
    # rounding error against the window's traffic -- while its accuracy is the
    # whole point, since a rejected draft costs a verified position. The
    # sibling frontends keep the head BF16 for the same reason; this path had
    # been quantising it along with everything else.
    was_w4a16, state.dense_w4a16 = state.dense_w4a16, False
    try:
        return _mtp_draft_bf16(state, mtp, ld, p, token_id, h_prev, pos,
                               fvk, device)
    finally:
        state.dense_w4a16 = was_w4a16


def _mtp_draft_bf16(state, mtp, ld, p, token_id, h_prev, pos, fvk, device):
    e = F.embedding(token_id.view(1, 1), p['embed_w_t']).reshape(1, HID)
    hn = _rms_fvk(h_prev.reshape(1, HID), mtp['pre_h_w_t'], fvk, device,
                  state.eps)
    en = _rms_fvk(e, mtp['pre_e_w_t'], fvk, device, state.eps)
    # Which half goes first is a checkpoint convention, not something the
    # shapes pin down -- fc is square in the concatenated width. It is
    # measured, not assumed: the wrong order drafts noise.
    cat = (torch.cat([hn, en], -1) if state.mtp_hidden_first
           else torch.cat([en, hn], -1))
    h = _dense_mv(cat, mtp['fc_w_t'], mtp, 'fc_w_t', state, fvk, device)

    res = h
    n = _rms_fvk(h, ld['input_norm_w_t'], fvk, device, state.eps)
    h = res + _decode_full(n, ld, state, state.mtp_rank, pos, fvk, device)
    res = h
    n = _rms_fvk(h, ld['post_norm_w_t'], fvk, device, state.eps)
    prev_layer, state._active_layer = state._active_layer, None
    try:
        h = res + _moe_layer_decode(n, ld, state, fvk, device)
    finally:
        state._active_layer = prev_layer
    # Return the state before the head's own final norm as well: chaining a
    # second draft means feeding the head what the model would have fed it,
    # and that is a pre-final-norm hidden state. Handing it the model's stale
    # one instead costs real acceptance -- measured 0.208 against 0.539 on the
    # second draft.
    return _lm_head(state, _rms_fvk(h, mtp['norm_w_t'], fvk, device,
                                    state.eps), fvk, device), h.reshape(HID)


def _ensure_spec_buffers(state, window, device):
    """Allocate what a window of `window` tokens needs."""
    if state._spec_tokens is None or state._spec_tokens.numel() < window:
        state._spec_tokens = torch.zeros(window, dtype=torch.long,
                                         device=device)
        state._spec_argmax = torch.zeros(window, dtype=torch.long,
                                         device=device)
    have = (state.spec_states is not None
            and len(state.spec_states[0]) >= window)
    if have:
        return
    state.spec_states = [
        [torch.empty(NV, HK, HV, dtype=torch.bfloat16, device=device)
         for _ in range(window)]
        for _ in range(state.n_lin)]
    state.spec_conv = [
        [torch.empty(1, CONV, KS - 1, dtype=torch.bfloat16, device=device)
         for _ in range(window)]
        for _ in range(state.n_lin)]


def _rewind_to(state, kept):
    """Put the recurrent and conv states where `kept` tokens of the window end.

    The KV cache needs nothing: it is written by absolute position, so the
    rejected tail is simply overwritten by whatever comes next. The recurrent
    state is the opposite -- it has already absorbed the whole window -- which
    is what the per-token snapshots are for.
    """
    for rank in range(state.n_lin):
        state.lin_state[rank].copy_(state.spec_states[rank][kept - 1])
        state.lin_conv_state[rank].copy_(state.spec_conv[rank][kept - 1])


def _verify_dense(x2d, w_bf16, ld, key, fvk, device):
    """A window's rows against the 4-bit weight the decode GEMV reads.

    Same tensor under the same cache key, and the M-row form of that GEMV,
    whose per-row accumulation order is the GEMV's -- so row t of the result
    equals what the decode step at that position would have computed, bit for
    bit, while the weight crosses the bus once for the whole window.
    """
    return _w4a16_mrows(x2d, w_bf16, ld, key, fvk, device)


def _verify_gdn(h, ld, state, lin_rank, w, fvk, device):
    """The GDN layer over a window of w tokens, snapshotting per token.

    The projections and the elementwise stages run at w rows; the two stages
    that carry state -- the causal conv and the recurrence -- run a token at a
    time through the very kernels the decode step calls, because a window is
    accepted up to a prefix and the state at that prefix has to be the state
    decode would have been in. The sequential-scan variant would do both in one
    launch, but it is cos 0.99999 against the per-token kernel rather than
    equal to it, and this layer is ~6% of a step: not worth paying for in
    tokens that diverge.
    """
    eps = state.eps
    convw = ld['conv1d_w_t'].reshape(CONV, KS).contiguous()
    neg, dtb_c = _gdn_gate_consts(ld, device)
    nw = ld['gdn_norm_w_t']
    s = _cs()
    x = h.reshape(w, HID)

    if 'in_proj_fused_w' not in ld:
        ld['in_proj_fused_w'] = torch.cat(
            [ld['in_proj_qkv_w_t'], ld['in_proj_z_w_t'],
             ld['in_proj_a_w_t'], ld['in_proj_b_w_t']], 0).contiguous()
    fused = _verify_dense(x, ld['in_proj_fused_w'], ld, 'in_proj_fused',
                          fvk, device)
    mixed = fused[:, :KD * 2 + VD].contiguous()
    z = fused[:, KD * 2 + VD:KD * 2 + VD + NV * HV].reshape(
        w * NV, HV).contiguous()
    a = fused[:, -2 * NV:-NV].contiguous()
    b = fused[:, -NV:].contiguous()

    conv_out = torch.empty(w, CONV, dtype=torch.bfloat16, device=device)
    st_in = state.lin_conv_state[lin_rank]
    for t in range(w):
        st_out = state.spec_conv[lin_rank][t]
        fvk.causal_conv1d_qwen36_update_inout_bf16(
            mixed[t].data_ptr(), convw.data_ptr(), 0,
            conv_out[t].data_ptr(), st_in.data_ptr(), st_out.data_ptr(),
            1, CONV, KS, True, s)
        st_in = st_out
    state.lin_conv_state[lin_rank].copy_(st_in)

    qb = torch.empty(w, NV, HK, dtype=torch.bfloat16, device=device)
    kb = torch.empty(w, NV, HK, dtype=torch.bfloat16, device=device)
    vb = torch.empty(w, NV, HV, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_lin_split_qkv_broadcast_bf16(
        conv_out.data_ptr(), qb.data_ptr(), kb.data_ptr(), vb.data_ptr(),
        w, s)

    g_out = torch.empty(w, NV, dtype=torch.bfloat16, device=device)
    bo = torch.empty(w, NV, dtype=torch.bfloat16, device=device)
    fvk.qwen36_gdn_gating_bf16(
        a.data_ptr(), b.data_ptr(), neg.data_ptr(), dtb_c.data_ptr(),
        g_out.data_ptr(), bo.data_ptr(), w, NV, s)

    core = torch.empty(w, NV, HV, dtype=torch.bfloat16, device=device)
    lin_state = state.lin_state[lin_rank]
    for t in range(w):
        qt, kt, vt = qb[t], kb[t], vb[t]
        gt, bt = g_out[t], bo[t]
        gdn_recurrent(fvk)(
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), gt.data_ptr(),
            bt.data_ptr(), lin_state.data_ptr(), core[t].data_ptr(),
            1, NV, HK, HV, True, s)
        state.spec_states[lin_rank][t].copy_(lin_state)

    nf = torch.empty(w * NV, HV, dtype=torch.bfloat16, device=device)
    fvk.rms_norm_gated_silu_qwen36_bf16(
        core.reshape(w * NV, HV).data_ptr(), z.data_ptr(), nw.data_ptr(),
        nf.data_ptr(), w * NV, HV, eps, s)
    out = _verify_dense(nf.reshape(w, VD), ld['out_proj_w_t'], ld,
                        'out_proj_w_t', fvk, device)
    return out.reshape(1, w, HID)


def _verify_full(h, ld, state, full_rank, pos, w, fvk, device):
    """The full-attention layer over a window of w tokens.

    Projections, norms and rope run at w rows. The attention itself runs a
    token at a time at q_seq=1 against [0..pos+t], which is the call the decode
    step makes: a batched q_seq=w call would have to carry a bottom-right
    causal mask and would reduce over a different tiling, and this is the one
    place where the two would stop being the same function. The KV it reads is
    small next to the weights the window is here to amortise.
    """
    eps = state.eps
    s = _cs()
    qnw, knw = ld['q_norm_w_t'], ld['k_norm_w_t']
    x2 = h.reshape(w, HID)

    nqg = NQ * 2 * HD
    if 'qkv_fused_w' not in ld:
        ld['qkv_fused_w'] = torch.cat(
            [ld['q_proj_w_t'], ld['k_proj_w_t'], ld['v_proj_w_t']],
            0).contiguous()
    fused = _verify_dense(x2, ld['qkv_fused_w'], ld, 'qkv_fused', fvk, device)
    qg = fused[:, :nqg].contiguous()
    kk = fused[:, nqg:nqg + NKV * HD].reshape(w * NKV, HD).contiguous()
    v = fused[:, nqg + NKV * HD:].reshape(w, NKV, HD).contiguous()

    q_pre = torch.empty(w, NQ, HD, dtype=torch.bfloat16, device=device)
    gate = torch.empty(w, NQ * HD, dtype=torch.bfloat16, device=device)
    fvk.qwen35moe_split_q_gate_bf16(
        qg.data_ptr(), q_pre.data_ptr(), gate.data_ptr(), w, s)
    q = _rms_fvk(q_pre.reshape(w * NQ, HD), qnw, fvk, device, eps)
    kn = _rms_fvk(kk, knw, fvk, device, eps)

    ct = state.rope_cos[pos:pos + w].contiguous()
    st = state.rope_sin[pos:pos + w].contiguous()
    qin = q.reshape(w, NQ, HD).contiguous()
    kin = kn.reshape(w, NKV, HD).contiguous()
    qo = torch.empty(w, NQ, HD, dtype=torch.bfloat16, device=device)
    ko = torch.empty(w, NKV, HD, dtype=torch.bfloat16, device=device)
    fvk.qwen36_partial_rope_qk_bf16(
        qin.data_ptr(), kin.data_ptr(), ct.data_ptr(), st.data_ptr(),
        qo.data_ptr(), ko.data_ptr(), w, NQ, NKV, HD, ROPE, s)

    attn = state.attn
    at = torch.empty(w, NQ * HD, dtype=torch.bfloat16, device=device)
    for t in range(w):
        attn.Q_buf[:, :1].copy_(qo[t].reshape(1, 1, NQ, HD))
        attn.K_cache[full_rank, pos + t:pos + t + 1].copy_(
            ko[t].reshape(1, NKV, HD))
        attn.V_cache[full_rank, pos + t:pos + t + 1].copy_(
            v[t].reshape(1, NKV, HD))
        attn.run('full', layer_idx=full_rank, q_seq=1, kv_seq=pos + t + 1,
                 stream=s, softmax_scale=float(HD) ** -0.5)
        at[t].copy_(attn.O_buf[:, :1].reshape(NQ * HD))
    at = _sigmoid_mul(at, gate, fvk, device)
    out = _verify_dense(at, ld['o_proj_w_t'], ld, 'o_proj_w_t', fvk, device)
    return out.reshape(1, w, HID)


def _verify_moe(h, ld, state, w, fvk, device):
    """The MoE layer over a window of w tokens.

    The routed experts are the one part of a window that does not amortise:
    w tokens pick up to w*TOPK distinct experts out of 256, so the weight
    traffic here scales with the window where everything else is read once.
    They still go through one grouped launch rather than w of them -- the
    kernel already takes the slot count, and a slot is an independent GEMV, so
    w*TOPK slots compute exactly what w separate TOPK-slot launches would.
    """
    s = _cs()
    x = h.reshape(w, HID)
    ne = ld['router_w_t'].shape[0]

    if 'router_shared_fused_w' not in ld:
        ld['router_shared_fused_w'] = torch.cat(
            [ld['router_w_t'], ld['shared_gate_proj_w_t'],
             ld['shared_up_proj_w_t']], 0).contiguous()
    rs = _verify_dense(x, ld['router_shared_fused_w'], ld,
                       'router_shared_fused', fvk, device)
    logit_raw = rs[:, :ne].contiguous()
    sg, su = rs[:, ne:ne + INTER], rs[:, ne + INTER:]

    # Top-8 a row at a time through the decode router. It is a single-block
    # kernel, so w launches is w small launches -- and the selected set has to
    # be the set decode selects, ties included, or the window keeps a token
    # from a different mixture.
    idx = torch.empty(w, TOPK, dtype=torch.int32, device=device)
    topv = torch.empty(w, TOPK, dtype=torch.float32, device=device)
    for t in range(w):
        rc = router_topk(fvk)(
            logit_raw[t].data_ptr(), idx[t].data_ptr(), topv[t].data_ptr(),
            ne, TOPK, s)
        if rc:
            raise RuntimeError(
                f'router top-k failed with {rc} for {ne} experts, k={TOPK}')
    tw = F.softmax(topv, -1)

    if 'experts_gate_up_alpha_dev' not in ld:
        ld['experts_gate_up_alpha_dev'] = \
            ld['experts_gate_up_alpha_t'].to(device).contiguous()
        ld['experts_down_alpha_dev'] = \
            ld['experts_down_alpha_t'].to(device).contiguous()
    gu_p, gu_s = ld['experts_gate_up_packed_t'], ld['experts_gate_up_sf_t']
    dn_p, dn_s = ld['experts_down_packed_t'], ld['experts_down_sf_t']
    gu_a, dn_a = ld['experts_gate_up_alpha_dev'], ld['experts_down_alpha_dev']
    n_gu, n_dn = gu_p.shape[1], dn_p.shape[1]

    slots = w * TOPK
    eidx = idx.reshape(-1).contiguous()
    # One activation row per slot: the grouped kernel indexes A by slot, and
    # the decode call gets the same effect from a zero stride over its single
    # row. The copy is w*TOPK*HID bf16 -- tens of KB.
    xrep = x.repeat_interleave(TOPK, 0).contiguous()
    d_gu = torch.empty(slots, n_gu, dtype=torch.bfloat16, device=device)
    moe_grouped_w4a16(fvk)(
        xrep.data_ptr(), gu_p.data_ptr(), gu_s.data_ptr(), gu_a.data_ptr(),
        eidx.data_ptr(), d_gu.data_ptr(), slots, n_gu, HID,
        HID, gu_p[0].numel(), gu_s[0].numel(), s)

    g_, u_ = d_gu[:, :INTER], d_gu[:, INTER:]
    inter = _silu_mul(g_, u_, fvk, device).contiguous()
    d_dn = torch.empty(slots, n_dn, dtype=torch.bfloat16, device=device)
    moe_grouped_w4a16(fvk)(
        inter.data_ptr(), dn_p.data_ptr(), dn_s.data_ptr(), dn_a.data_ptr(),
        eidx.data_ptr(), d_dn.data_ptr(), slots, n_dn, INTER,
        INTER, dn_p[0].numel(), dn_s[0].numel(), s)

    rk = f'verify_topk_rows_{w}'
    if rk not in ld:
        ld[rk] = torch.arange(slots, dtype=torch.int32, device=device)
    twf = tw.reshape(-1).contiguous()
    out = torch.empty(w, n_dn, dtype=torch.float32, device=device)
    fvk.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(), ld[rk].data_ptr(), twf.data_ptr(),
        out.data_ptr(), w, TOPK, n_dn, n_dn, s)

    si = _silu_mul(sg.contiguous(), su.contiguous(), fvk, device)
    shared = _verify_dense(si, ld['shared_down_proj_w_t'], ld,
                           'shared_down_proj_w_t', fvk, device)
    # The scalar gate is an N=1 GEMV over a 4 KB weight, so a row at a time
    # costs nothing and is the decode kernel's own arithmetic.
    xc = x.contiguous()
    gsc = torch.empty(w, 1, dtype=torch.bfloat16, device=device)
    for t in range(w):
        xr = xc[t]
        fvk.bf16_matvec_sm120_bf16(
            xr.data_ptr(), ld['shared_gate_w_t'].data_ptr(),
            gsc[t].data_ptr(), 1, HID, s)
    return _shared_combine(out, shared, gsc, w, fvk, device).reshape(
        1, w, HID)


def _verify_block_K(state, toks, pos, w, fvk, device):
    """Run a window of w tokens through the decode kernels, at w rows.

    This is what makes the verify the same function as the steps it verifies.
    Every stage is the kernel decode calls, at w rows instead of one, over the
    weights decode caches; the two state-carrying stages and the attention run
    per token for the reasons given above. So the window's row t is the decode
    step at pos+t, and the largest weights cross the bus once instead of w
    times.

    Returns (logits (w, vocab), hidden (w, HID) pre-final-norm).
    """
    p = state.handles.ptrs
    layers = p['layers']
    h = F.embedding(toks.view(1, w), p['embed_w_t'])

    for L in range(state.num_layers):
        ld = layers[L]
        res = h
        n = _rms_fvk(h, ld['input_norm_w_t'], fvk, device, state.eps)
        if state.types[L] == 'linear_attention':
            attn = _verify_gdn(n, ld, state, state._lin_rank[L], w,
                               fvk, device)
        else:
            attn = _verify_full(n, ld, state, state._full_rank[L], pos, w,
                                fvk, device)
        h = res + attn
        res = h
        n = _rms_fvk(h, ld['post_norm_w_t'], fvk, device, state.eps)
        state._active_layer = L
        h = res + _verify_moe(n, ld, state, w, fvk, device)

    hidden = h.reshape(w, HID)
    hn = _rms_fvk(h, p['final_norm_w_t'], fvk, device, state.eps)
    vocab = p['vocab_size']
    _ensure_lm_head_nvfp4(state, fvk, device)
    logits = torch.empty(w, vocab, dtype=torch.bfloat16, device=device)
    hc = hn.reshape(w, HID).contiguous()
    rc = fvk.w4a16_mrows_edge_sm120_bf16(
        hc.data_ptr(), p['lm_head_packed_t'].data_ptr(),
        p['lm_head_sf_t'].data_ptr(), logits.data_ptr(),
        w, vocab, HID, p['lm_head_alpha'], _cs())
    if rc:
        raise RuntimeError(f'M-row lm_head failed with {rc} at M={w}')
    return logits, hidden


def _verify_block_usable(state) -> bool:
    """Can the window run on the decode kernels?

    The M-row GEMV stages a window's activations in shared memory, so it has a
    width limit; and the whole point is that the window reads what decode
    reads, which is only true where decode takes the W4A16 dense path over
    BF16-scope weights. Anywhere else the prefill forward is still the answer.
    """
    # gdn_in_proj_w4a16 is gated separately from the rest of the dense path, so
    # with it off decode reads the GDN in_proj at BF16 while the window reads
    # it at four bits -- a different function in thirty of the forty layers,
    # which is exactly the thing this block exists to rule out.
    if not kernel_policy().verify_k_rows or not state.dense_w4a16:
        return False
    if not state.gdn_in_proj_w4a16:
        return False
    # The window fuses the router with the shared gate/up and reads every
    # projection at four bits, which is what decode does only when the loader
    # kept these BF16. One NVFP4 site among them and decode takes the W4A4 mma
    # instead, so ask about each of the three the window assumes.
    ld = state.handles.ptrs['layers'][0]
    return (ld.get('router_packed') is None
            and ld.get('shared_gate_proj_packed') is None
            and ld.get('out_proj_packed') is None
            and not ld.get('experts_streamed'))


def _spec_block(state, pos, k, fvk, device):
    """The whole window as one dependency chain: k drafts, then the verify.

    Written to be capturable end to end. Each draft's token is chosen on the
    device -- ``qwen36_argmax_bf16`` writes it straight into the token buffer
    the next draft reads -- so the chain never leaves the GPU, and the only
    host decision left is how much of the window to keep.
    """
    vocab = state.handles.ptrs['vocab_size']
    toks = state._spec_tokens
    window = k + 1

    hidden = state.last_hidden
    for j in range(k):
        d_logits, hidden = mtp_draft(state, toks[j:j + 1], pos + j, fvk,
                                     device, hidden=hidden)
        fvk.qwen36_argmax_bf16(d_logits.data_ptr(),
                               toks[j + 1:j + 2].data_ptr(), 1, vocab, _cs())

    if _verify_block_usable(state):
        logits, hid = _verify_block_K(state, toks[:window], pos, window,
                                      fvk, device)
    else:
        state.spec_capture = True
        set_spec_verify(True)
        try:
            logits, hid = nexn2_forward_nvfp4(
                state.handles, toks[:window].view(1, window), fvk, device,
                cap=state, pos_offset=pos, last_logits_only=False,
                return_hidden=True)
        finally:
            state.spec_capture = False
            set_spec_verify(False)
    logits = logits.reshape(window, -1)
    fvk.qwen36_argmax_bf16(logits.data_ptr(), state._spec_argmax.data_ptr(),
                           window, vocab, _cs())
    return hid


def _ensure_spec_graph(state, pos, k, fvk, device):
    """Capture the draft-and-verify window at ``pos``, or return the cached one.

    Everything the block mutates is snapshotted and restored around the warmup
    and capture runs -- the recurrent and conv states, the KV rows the window
    writes across every rank including the draft head's, and the drafted token
    slots -- so a later replay advances from the true pre-window state rather
    than from whatever the capture left behind.
    """
    key = (pos, k)
    cached = state._spec_graphs.get(key)
    if cached is not None:
        state._spec_graphs.move_to_end(key)
        return cached

    window = k + 1
    snap_lin = [t.clone() for t in state.lin_state]
    snap_conv = [t.clone() for t in state.lin_conv_state]
    snap_k = state.attn.K_cache[:, pos:pos + window].clone()
    snap_v = state.attn.V_cache[:, pos:pos + window].clone()
    snap_tok = state._spec_tokens.clone()

    def _restore():
        for i, t in enumerate(state.lin_state):
            t.copy_(snap_lin[i])
        for i, t in enumerate(state.lin_conv_state):
            t.copy_(snap_conv[i])
        state.attn.K_cache[:, pos:pos + window].copy_(snap_k)
        state.attn.V_cache[:, pos:pos + window].copy_(snap_v)
        state._spec_tokens.copy_(snap_tok)

    with torch.no_grad():           # settle allocator, kernel order, and the
        for _ in range(2):          # weight quantisation the draft does lazily
            _spec_block(state, pos, k, fvk, device)
        _restore()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=state._graph_stream,
                          pool=state._spec_pool), torch.no_grad():
        hid = _spec_block(state, pos, k, fvk, device)
    with torch.no_grad():
        _restore()

    return _cache_put(state._spec_graphs, key, (g, hid),
                      state.spec_graph_cache_max)


def spec_decode_step(state, token_id, pos, k, fvk, device):
    """One speculative step: draft k tokens, verify k+1 positions, keep a prefix.

    Returns (tokens, next_pos) where `tokens` are the ids actually emitted --
    between 1 and k+1 of them.

    Verifying the window is one batched forward over k+1 positions, which reads
    the dense weights once instead of k+1 times; that, and nothing about the
    drafts being good, is where the time comes from. A draft is kept only where
    the model's own argmax agrees with it, so the emitted sequence is what that
    verifier's greedy decode would have produced.
    """
    window = k + 1
    _ensure_spec_buffers(state, window, device)
    state._spec_tokens[0].copy_(token_id.view(1)[0])

    g, hid = _ensure_spec_graph(state, pos, k, fvk, device)
    g.replay()

    # One D2H for the whole decision: the drafted ids and what the model said
    # at each position. Everything before this stayed on the device.
    drafted = state._spec_tokens[:window].tolist()
    argmax = state._spec_argmax[:window].tolist()
    kept = 1
    for j in range(k):
        if argmax[j] != drafted[j + 1]:
            break
        kept += 1

    if kept < window:
        _rewind_to(state, kept)
    # The draft head reads the pre-final-norm hidden state of the last emitted
    # position. Without this the next window would draft off a stale one.
    state.last_hidden.copy_(hid[kept - 1])
    tokens = drafted[1:kept] + [argmax[kept - 1]]
    return tokens, pos + kept


def generate_greedy_spec(state, input_ids, max_new_tokens, k, fvk, device):
    """Greedy decode through the draft-and-verify step.

    Emits exactly what generate_greedy would; the tokens are a check on the
    machinery, not an approximation of it.
    """
    logits = seed_prefill(state, input_ids, fvk, device)
    pos = input_ids.view(-1).shape[0]
    nxt = logits[0].argmax().view(1)
    out = []
    state.spec_windows = 0
    state.spec_kept = 0
    while len(out) < max_new_tokens:
        tokens, pos = spec_decode_step(state, nxt, pos, k, fvk, device)
        emitted = [int(nxt)] + tokens[:-1]
        state.spec_windows += 1
        state.spec_kept += len(emitted)
        out.extend(emitted)
        nxt = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
    return out[:max_new_tokens]


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
    when ``cap`` is passed (see _gdn_layer / _full_attn_layer). Above
    ``prefill_chunk`` tokens the pass is chunked to bound activation memory."""
    S = input_ids.view(-1).shape[0]
    if state.prefill_chunk and S > state.prefill_chunk:
        return seed_prefill_chunked(state, input_ids, fvk, device,
                                    state.prefill_chunk)
    state.reset()
    logits, hidden = nexn2_forward_nvfp4(
        state.handles, input_ids.view(1, -1), fvk, device, cap=state,
        last_logits_only=True, return_hidden=True)
    # The last prompt position's pre-final-norm hidden state, which is what a
    # draft head reads. The per-token path writes it every step; this one has
    # to do it explicitly, and without it the first window drafts off whatever
    # the previous generation left behind -- so how much of that window is kept
    # depends on what ran before it, and the run stops being reproducible.
    state.last_hidden.copy_(hidden[-1])
    return logits           # already (1, vocab): only the seeding logit


def seed_prefill_chunked(state, input_ids, fvk, device, block):
    """Chunked batched prefill: process the prompt in token-blocks through all
    layers, carrying the GDN recurrent/conv state + the full-attn KV cache
    across blocks. Each block's per-layer activations are bounded by ``block``
    instead of the full prompt, so context scales past the single-pass memory
    ceiling. Produces the same decode state as ``seed_prefill_batched`` (each
    GDN layer continues from cap's carried state; each full-attn block attends
    to cap's accumulated KV). Returns the last-token logits (1, vocab)."""
    state.reset()
    ids = input_ids.view(1, -1)
    S = ids.shape[1]
    logits = None
    for b0 in range(0, S, block):
        b1 = min(b0 + block, S)
        logits, hidden = nexn2_forward_nvfp4(
            state.handles, ids[:, b0:b1], fvk, device, cap=state,
            pos_offset=b0, last_logits_only=True, compute_logits=(b1 == S),
            return_hidden=True)
    state.last_hidden.copy_(hidden[-1])          # see seed_prefill_batched
    return logits


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

    return _cache_put(state._graphs, pos, (g, out), state.graph_cache_max)


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
