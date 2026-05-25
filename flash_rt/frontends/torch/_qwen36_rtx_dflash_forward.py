"""FlashRT — Qwen3.6 DFlash drafter S=16 forward (NVFP4).

Block-diffusion drafter forward path. One call per spec-decode cycle.
Inputs:

  prev_token_id  : scalar int (the last accepted target token)
  hidden_taps    : (5, hidden=5120) bf16 — captured target hidden state
                    at layers [1, 16, 31, 46, 61], the most recent
                    target step's outputs at those layer indices
  drafter_cur_pos: int — reserved for future persistent-KV variant.
                    The reference impl uses RELATIVE positions per cycle
                    (positions_q = [ctx_len..ctx_len+q_len-1]) so this
                    parameter is currently unused; positions are baked
                    in as ctx_len=1, positions_q=[1..16], positions_k=[0..16].

Output: ``logits`` (16, vocab_size=248320) bf16. Caller does argmax
per row to get 16 candidate tokens. Position-to-token mapping (per
the lucebox / megaqwen3_27b_dflash reference, ctx_len=1):

  output[0]   predicts position cur_pos-1 (i.e., a re-prediction of
              prev_token; not used for spec)
  output[i]   for i in 1..15 predicts position cur_pos+i-1

Algorithm (matches lucebox-hub/dflash/src/qwen3_dflash_graph.cpp):

  1.  input_ids = [prev_token_id, MASK_ID × 15]   shape (16,) int64
  2.  embed[16, 5120]      = main.embed_tokens[input_ids]
  3.  tap_concat[1, 25600] = hidden_taps.flatten().view(1, 5*hidden)
  4.  target_feat[1, 5120] = NVFP4 GEMM(fc, tap_concat)           [M=1]
  5.  target_feat          = rms_norm(target_feat) * hidden_norm.weight
  6.  h[16, 5120]          = embed     ← NO residual add of target_feat
  7.  for L in 0..4:
        hn = rms_norm(h) * input_norm.weight                      [M=16]
        Q  = wq @ hn               -> per-head q_norm             [M=16]
        K_ctx = wk @ target_feat   (1 row);  V_ctx = wv @ target_feat
        K_q   = wk @ hn (16 rows); V_q   = wv @ hn
        K = concat[K_ctx, K_q]     -> per-head k_norm (17 rows)
        V = concat[V_ctx, V_q]
        RoPE(Q, positions_q=[1..16]); RoPE(K, positions_k=[0..16])
                                        (NEOX style, theta=1e7, full head_dim)
        attn = SDPA(Q, K, V, is_causal=False, scale=1/sqrt(head_dim))
                                        (q_seq=16, kv_seq=17, GQA 4:1)
        h += wo @ attn
        hf = rms_norm(h) * post_attn_norm.weight                  [M=16]
        h += w_down @ (silu(w_gate @ hf) * (w_up @ hf))
  8.  h_final = rms_norm(h) * final_norm.weight                   [M=16]
  9.  logits  = NVFP4 GEMM(main.lm_head, h_final)                 [M=16]
  10. return logits

Reuses kernels:
  fvk.rms_norm
  fvk.quantize_bf16_to_nvfp4_swizzled
  fvk.fp4_w4a16_gemm_sm120_bf16out (+_widen for large N)
  fvk.silu_mul_qwen36_bf16

Add-only: no existing kernel touched. New scratch + new methods on
the frontend instance, namespaced under ``_dflash_*``.
"""
from __future__ import annotations

import torch


def _swz_bytes(rows: int, cols: int) -> int:
    """Bytes for swizzled SF tensor at (rows × cols) NVFP4 quant."""
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 512


def alloc_drafter_scratch(frontend, device: str = 'cuda:0') -> dict:
    """Allocate all drafter scratch buffers + RoPE table on the frontend.

    Stores everything under the single attribute ``frontend._dflash_buf``
    (a dict). Idempotent: re-call returns the existing dict.

    Sized for the stateless first-cut forward (S=16 per cycle, ctx_len=1
    target_feat slot prepended to each layer's K/V).
    """
    if getattr(frontend, '_dflash_buf', None) is not None:
        return frontend._dflash_buf

    bf16 = torch.bfloat16
    d = frontend._weights.ptrs['dflash']
    hidden = d['hidden']
    q_dim = d['q_dim']
    kv_dim = d['kv_dim']
    inter = d['intermediate']
    head_dim = d['head_dim']
    num_q = d['num_q_heads']
    num_kv = d['num_kv_heads']
    fc_in = d['fc_in']
    vocab = d['vocab_size']
    block = d['block_size']      # 16
    n_layers = d['num_layers']   # 5
    # P5: variable ctx_len up to MAX_CTX. Each cycle the orchestration
    # appends one new target_feat row (computed from the cycle's hidden
    # taps) and the drafter forward attends to ALL valid past slots
    # (sliding window). lucebox uses 4096-slot ring; we cap at 256 by
    # default. Empirically AL is best at modest ctx (8-32) and falls
    # off as ctx grows, so the actual hot-path window is capped via
    # frontend._dflash_eff_ctx (defaults to max_ctx, env-overridable).
    max_ctx = int(getattr(frontend, '_dflash_max_ctx', 256))
    total_k_max = block + max_ctx     # max K/V positions per layer

    buf: dict = {
        'block': block,
        'max_ctx': max_ctx,
        'total_k_max': total_k_max,
        'hidden': hidden,
        'head_dim': head_dim,
        'num_q': num_q,
        'num_kv': num_kv,
        'q_dim': q_dim,
        'kv_dim': kv_dim,
        'inter': inter,
        'fc_in': fc_in,
        'vocab': vocab,
        'n_layers': n_layers,
    }

    # ---- per-shape NVFP4 activation scratch (act packed + sf) -------
    # M=16 K=5120 : pre-q/k/v + pre-mlp norm output -> quantize for proj
    # M=16 K=4096 : o_proj input
    # M=16 K=17408: mlp_down input (after silu_mul)
    # M=1  K=25600: fc input (one-shot per cycle, hidden-tap concat)
    # M=1  K=5120 : target_feat input to wk/wv (one row per layer)
    def _alloc_act(M: int, K: int):
        ap = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
        sf = torch.zeros(_swz_bytes(M, K), dtype=torch.uint8, device=device)
        return ap, sf

    buf['act_M16_K5120'] = _alloc_act(block, hidden)
    buf['act_M16_K4096'] = _alloc_act(block, q_dim)
    buf['act_M16_K17408'] = _alloc_act(block, inter)
    buf['act_M1_K25600'] = _alloc_act(1, fc_in)
    # P5: ctx-row activation buf for wk/wv on target_feat. Sized for
    # max_ctx so we can quantize all past target_feat in one launch.
    buf['act_Mctx_K5120'] = _alloc_act(max_ctx, hidden)

    # ---- output buffers (bf16) --------------------------------------
    # Per-layer GEMM outputs:
    buf['q_out']  = torch.empty(block, q_dim,  dtype=bf16, device=device)
    buf['k_q_out'] = torch.empty(block, kv_dim, dtype=bf16, device=device)
    buf['v_q_out'] = torch.empty(block, kv_dim, dtype=bf16, device=device)
    # K/V_ctx sized for the max ctx slots; per-cycle writes the first
    # ctx_len_now rows.
    buf['k_ctx_out'] = torch.empty(max_ctx, kv_dim, dtype=bf16, device=device)
    buf['v_ctx_out'] = torch.empty(max_ctx, kv_dim, dtype=bf16, device=device)
    # Combined K/V (concatenated along sequence) staging for k_norm.
    # Sized for max_ctx + block; per-cycle uses first total_k_now rows.
    buf['k_cat_out'] = torch.empty(total_k_max, kv_dim, dtype=bf16, device=device)
    buf['v_cat_out'] = torch.empty(total_k_max, kv_dim, dtype=bf16, device=device)
    buf['o_out']  = torch.empty(block, hidden, dtype=bf16, device=device)
    buf['gate_out'] = torch.empty(block, inter, dtype=bf16, device=device)
    buf['up_out']   = torch.empty(block, inter, dtype=bf16, device=device)
    buf['silu_mul_out'] = torch.empty(block, inter, dtype=bf16, device=device)
    buf['down_out'] = torch.empty(block, hidden, dtype=bf16, device=device)
    # tap projection (one row per cycle = one new ctx slot)
    buf['tap_proj_out'] = torch.empty(1, hidden, dtype=bf16, device=device)

    # P5: target_feat ring buffer. Each cycle appends one row (the
    # current cycle's rms_norm(fc(hidden_taps)) * hidden_norm). Drafter
    # attends to the most recent ctx_len_now rows (sliding).
    # Layout: rows [0..max_ctx-1] cycle through circularly; we always
    # pass a contiguous slice to layers via an explicit slice/copy when
    # ring wraps (rare for short bench). max_ctx=256 covers up to 256
    # cycles without wrapping.
    buf['target_feat_ring'] = torch.zeros(
        max_ctx, hidden, dtype=bf16, device=device)
    buf['ctx_len_now'] = 0   # how many slots are valid (starts at 0)
    buf['ring_pos'] = 0      # next write position (0..max_ctx-1)

    # final norm output + logits
    buf['h_final_norm'] = torch.empty(block, hidden, dtype=bf16, device=device)
    buf['logits'] = torch.empty(block, vocab, dtype=bf16, device=device)

    # Hidden state ping-pong (between layers + residuals)
    buf['h_a']    = torch.empty(block, hidden, dtype=bf16, device=device)
    buf['h_b']    = torch.empty(block, hidden, dtype=bf16, device=device)
    buf['x_norm'] = torch.empty(block, hidden, dtype=bf16, device=device)
    buf['h_mid']  = torch.empty(block, hidden, dtype=bf16, device=device)
    buf['x_mlp']  = torch.empty(block, hidden, dtype=bf16, device=device)
    # Per-head Q/K rmsnorm output buffers — sized for max_ctx+block
    buf['q_norm_out'] = torch.empty(block * num_q, head_dim,
                                     dtype=bf16, device=device)
    buf['k_norm_out'] = torch.empty(total_k_max * num_kv, head_dim,
                                     dtype=bf16, device=device)

    # Q/K rotated buffers (for SDPA: shape (1, n_heads, seq, head_dim)).
    # Q has block (16) positions; K/V have total_k_now (variable) positions.
    buf['q_rope'] = torch.empty(1, num_q,  block,       head_dim,
                                dtype=bf16, device=device)
    buf['k_rope'] = torch.empty(1, num_kv, total_k_max, head_dim,
                                dtype=bf16, device=device)
    buf['v_for_sdpa'] = torch.empty(1, num_kv, total_k_max, head_dim,
                                     dtype=bf16, device=device)
    buf['attn_out'] = torch.empty(1, num_q, block, head_dim,
                                   dtype=bf16, device=device)

    # ---- P7: static buffers for graph capture path -----------------
    # Pre-allocated ids buffer ([prev_tok, MASK*15]). External caller
    # writes new prev_tok to ids_static[0:1] before each replay.
    mask_id = int(d['mask_token_id'])
    buf['ids_static'] = torch.full(
        (block,), mask_id, dtype=torch.long, device=device)
    # Embed buffer for index_select(embed_t, 0, ids_static, out=embed_buf)
    buf['embed_buf'] = torch.empty(
        block, hidden, dtype=bf16, device=device)
    # Static input for hidden_taps. External caller copies real taps in
    # before replay. hidden_taps_static shape (5, hidden) bf16.
    buf['hidden_taps_static'] = torch.zeros(
        5, hidden, dtype=bf16, device=device)
    # Shift-window target_feat buffer. Always size = eff_ctx_capture.
    # eff_ctx_capture is set when graph is captured; cached graphs are
    # keyed by this value. Each cycle: shift left, write new tap at end.
    buf['target_feat_window'] = None    # alloc'd at first capture
    buf['eff_ctx_capture'] = None       # eff_ctx the graph was captured for

    # ---- RoPE table for drafter (head_dim=128 full, theta=1e7) ------
    # NEOX-style: pair (x[..., i], x[..., i+half]). Per the reference,
    # positions are RELATIVE within a cycle: positions_q = [ctx_len..
    # ctx_len + block - 1], positions_k = [0..total_k-1]. With variable
    # ctx_len (up to max_ctx), we need max_ctx + block positions.
    theta = float(d['rope_theta'])
    rotary_dim = head_dim
    half = rotary_dim // 2
    max_drafter_pos = total_k_max + 16   # generous headroom
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    positions = torch.arange(max_drafter_pos, device=device).float()
    freqs = positions[:, None] * inv_freq[None, :]   # (max, half)
    buf['cos_cache'] = freqs.cos().to(bf16).contiguous()
    buf['sin_cache'] = freqs.sin().to(bf16).contiguous()

    # MASK token id stays as a python int (used to build input_ids).
    buf['mask_token_id'] = d['mask_token_id']

    frontend._dflash_buf = buf
    return buf


def _rope_apply(x_in, x_out, cos, sin, half: int):
    """Apply NEOX-style full RoPE.

    x_in / x_out: (1, n_heads, seq, head_dim) bf16
    cos / sin   : (seq, half) bf16
    Pairs (x[..., i], x[..., i+half]):
      out_lo = x_lo * cos - x_hi * sin
      out_hi = x_hi * cos + x_lo * sin
    """
    cs = cos.view(1, 1, cos.shape[0], half)
    ss = sin.view(1, 1, sin.shape[0], half)
    x_lo = x_in[..., :half]
    x_hi = x_in[..., half:]
    out_lo = x_out[..., :half]
    out_hi = x_out[..., half:]
    out_lo.copy_(x_lo).mul_(cs).addcmul_(x_hi, ss, value=-1.0)
    out_hi.copy_(x_hi).mul_(cs).addcmul_(x_lo, ss, value=1.0)


def _gemm_nvfp4(fvk, ap_ptr, sf_ptr, w_packed, w_sf, alpha,
                out_ptr, M: int, N: int, K: int, stream,
                widen: bool = False) -> None:
    """One-shot wrapper to keep call sites compact."""
    fn = (fvk.fp4_w4a16_gemm_sm120_bf16out_widen if widen
          else fvk.fp4_w4a16_gemm_sm120_bf16out)
    fn(int(ap_ptr), int(w_packed), int(out_ptr),
       M, N, K,
       int(sf_ptr), int(w_sf),
       float(alpha),
       int(stream))


def _quant_act(fvk, x, ap, sf, M: int, K: int, stream) -> None:
    """Quantize bf16 (M, K) -> NVFP4 packed + swizzled SF."""
    fvk.quantize_bf16_to_nvfp4_swizzled(
        int(x.data_ptr()), int(ap.data_ptr()), int(sf.data_ptr()),
        M, K, int(stream))


def _drafter_layer_forward(frontend, fvk, L: int, h_in,
                            target_feat, ctx_len_now: int, stream):
    """One drafter layer forward at q_seq=16 / kv_seq=ctx_len_now+16.

    h_in:         (16, hidden) bf16  — current block hidden states
    target_feat:  (ctx_len_now, hidden) bf16  — fused target hidden context
    ctx_len_now:  number of valid past target_feat rows

    Returns: (16, hidden) bf16
    """
    buf = frontend._dflash_buf
    d = frontend._weights.ptrs['dflash']
    lw = d['layers'][L]
    eps = float(d['rms_norm_eps'])
    M = buf['block']                    # 16
    CTX = ctx_len_now                   # variable, 1..max_ctx
    T = CTX + M                         # variable, 17..max_ctx+16
    H = buf['hidden']
    HD = buf['head_dim']
    NQ = buf['num_q']
    NKV = buf['num_kv']
    QD = buf['q_dim']
    KVD = buf['kv_dim']
    INTER = buf['inter']
    half = HD // 2
    s = stream

    # ---- 1) input rms_norm on h ----
    fvk.rms_norm(
        int(h_in.data_ptr()), int(lw['input_norm_w']),
        int(buf['x_norm'].data_ptr()),
        M, H, eps, int(s),
    )

    # ---- 2) NVFP4-quantize x_norm (M=16, K=5120) for q/k/v over noise ----
    ap_h, sf_h = buf['act_M16_K5120']
    _quant_act(fvk, buf['x_norm'], ap_h, sf_h, M, H, s)

    # ---- 3) Q from noise only ----
    _gemm_nvfp4(fvk, ap_h.data_ptr(), sf_h.data_ptr(),
                lw['q_proj_packed'], lw['q_proj_sf'], lw['q_proj_alpha'],
                buf['q_out'].data_ptr(), M, QD, H, s)

    # ---- 4) K/V_q from noise (16 rows) ----
    _gemm_nvfp4(fvk, ap_h.data_ptr(), sf_h.data_ptr(),
                lw['k_proj_packed'], lw['k_proj_sf'], lw['k_proj_alpha'],
                buf['k_q_out'].data_ptr(), M, KVD, H, s)
    _gemm_nvfp4(fvk, ap_h.data_ptr(), sf_h.data_ptr(),
                lw['v_proj_packed'], lw['v_proj_sf'], lw['v_proj_alpha'],
                buf['v_q_out'].data_ptr(), M, KVD, H, s)

    # ---- 5) K/V_ctx from target_feat (CTX rows): one-shot NVFP4 quant + GEMM ----
    ap_t, sf_t = buf['act_Mctx_K5120']
    _quant_act(fvk, target_feat, ap_t, sf_t, CTX, H, s)
    # Write into the slice [:CTX] of the full max_ctx-sized buffers
    _gemm_nvfp4(fvk, ap_t.data_ptr(), sf_t.data_ptr(),
                lw['k_proj_packed'], lw['k_proj_sf'], lw['k_proj_alpha'],
                buf['k_ctx_out'].data_ptr(), CTX, KVD, H, s)
    _gemm_nvfp4(fvk, ap_t.data_ptr(), sf_t.data_ptr(),
                lw['v_proj_packed'], lw['v_proj_sf'], lw['v_proj_alpha'],
                buf['v_ctx_out'].data_ptr(), CTX, KVD, H, s)

    # ---- 6) Concat along sequence: K = [K_ctx (CTX), K_q (M)]; V same ----
    buf['k_cat_out'][:CTX].copy_(buf['k_ctx_out'][:CTX])
    buf['k_cat_out'][CTX:T].copy_(buf['k_q_out'])
    buf['v_cat_out'][:CTX].copy_(buf['v_ctx_out'][:CTX])
    buf['v_cat_out'][CTX:T].copy_(buf['v_q_out'])

    # ---- 7) Per-head q_norm (M*NQ rows) and k_norm (T*NKV rows) ----
    q_view = buf['q_out'].view(M * NQ, HD)
    fvk.rms_norm(
        int(q_view.data_ptr()), int(lw['q_norm_w']),
        int(buf['q_norm_out'].data_ptr()),
        M * NQ, HD, eps, int(s),
    )
    k_view = buf['k_cat_out'][:T].contiguous().view(T * NKV, HD)
    fvk.rms_norm(
        int(k_view.data_ptr()), int(lw['k_norm_w']),
        int(buf['k_norm_out'].data_ptr()),
        T * NKV, HD, eps, int(s),
    )

    # ---- 8) RoPE (NEOX, full head_dim=128) ----
    # positions_q = [CTX..CTX+M-1]
    # positions_k = [0..T-1]
    q_pre = buf['q_norm_out'].view(M, NQ, HD).permute(1, 0, 2).unsqueeze(0)
    k_pre_full = buf['k_norm_out'][:T * NKV].view(T, NKV, HD).permute(
        1, 0, 2).unsqueeze(0)
    cos_q = buf['cos_cache'][CTX:CTX + M]
    sin_q = buf['sin_cache'][CTX:CTX + M]
    cos_k = buf['cos_cache'][0:T]
    sin_k = buf['sin_cache'][0:T]
    q_pre_c = q_pre.contiguous()
    k_pre_c = k_pre_full.contiguous()
    # k_rope is sized for max; use only [..., :T, :] slice
    _rope_apply(q_pre_c, buf['q_rope'], cos_q, sin_q, half)
    k_rope_view = buf['k_rope'][:, :, :T, :]
    _rope_apply(k_pre_c, k_rope_view, cos_k, sin_k, half)

    # V doesn't get RoPE'd; reshape to SDPA layout (slice of v_for_sdpa).
    v_for_sdpa_view = buf['v_for_sdpa'][:, :, :T, :]
    v_for_sdpa_view.copy_(
        buf['v_cat_out'][:T].view(T, NKV, HD).permute(1, 0, 2).unsqueeze(0))

    # ---- 9) Non-causal SDPA, GQA 4:1 (q_seq=M, kv_seq=T) ----
    scale = float(HD) ** -0.5
    try:
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            buf['q_rope'], k_rope_view, v_for_sdpa_view,
            attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=scale, enable_gqa=True,
        )
    except TypeError:
        rep = NQ // NKV
        k_rep = k_rope_view.repeat_interleave(rep, dim=1)
        v_rep = v_for_sdpa_view.repeat_interleave(rep, dim=1)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            buf['q_rope'], k_rep, v_rep,
            attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=scale,
        )
    # attn_out: (1, NQ, M, HD) -> (M, q_dim)
    attn_2d = attn_out.squeeze(0).permute(1, 0, 2).contiguous().view(M, QD)

    # ---- 10) o_proj (NVFP4, M=16, N=5120, K=4096) + residual ----
    ap_q, sf_q = buf['act_M16_K4096']
    _quant_act(fvk, attn_2d, ap_q, sf_q, M, QD, s)
    _gemm_nvfp4(fvk, ap_q.data_ptr(), sf_q.data_ptr(),
                lw['o_proj_packed'], lw['o_proj_sf'], lw['o_proj_alpha'],
                buf['o_out'].data_ptr(), M, H, QD, s)
    fvk.add_bf16_out(
        h_in.data_ptr(), buf['o_out'].data_ptr(),
        buf['h_mid'].data_ptr(), M * H, s,
    )

    # ---- 11) post-attn rms_norm + MLP gate/up/down ----
    fvk.rms_norm(
        int(buf['h_mid'].data_ptr()), int(lw['post_attn_norm_w']),
        int(buf['x_mlp'].data_ptr()),
        M, H, eps, int(s),
    )
    _quant_act(fvk, buf['x_mlp'], ap_h, sf_h, M, H, s)
    _gemm_nvfp4(fvk, ap_h.data_ptr(), sf_h.data_ptr(),
                lw['mlp_gate_packed'], lw['mlp_gate_sf'],
                lw['mlp_gate_alpha'],
                buf['gate_out'].data_ptr(), M, INTER, H, s, widen=True)
    _gemm_nvfp4(fvk, ap_h.data_ptr(), sf_h.data_ptr(),
                lw['mlp_up_packed'], lw['mlp_up_sf'], lw['mlp_up_alpha'],
                buf['up_out'].data_ptr(), M, INTER, H, s, widen=True)
    fvk.silu_mul_qwen36_bf16(
        int(buf['gate_out'].data_ptr()),
        int(buf['up_out'].data_ptr()),
        int(buf['silu_mul_out'].data_ptr()),
        M * INTER, int(s),
    )
    ap_i, sf_i = buf['act_M16_K17408']
    _quant_act(fvk, buf['silu_mul_out'], ap_i, sf_i, M, INTER, s)
    _gemm_nvfp4(fvk, ap_i.data_ptr(), sf_i.data_ptr(),
                lw['mlp_down_packed'], lw['mlp_down_sf'],
                lw['mlp_down_alpha'],
                buf['down_out'].data_ptr(), M, H, INTER, s)
    h_out = buf['h_a'] if (L % 2 == 0) else buf['h_b']
    fvk.add_bf16_out(
        buf['h_mid'].data_ptr(), buf['down_out'].data_ptr(),
        h_out.data_ptr(), M * H, s,
    )
    return h_out


def reset_drafter_ring(frontend) -> None:
    """Clear the target_feat ring buffer. Call at start of each
    generate() so taps from a previous prompt don't leak in."""
    buf = frontend._dflash_buf
    if buf is None:
        return
    buf['target_feat_ring'].zero_()
    buf['ctx_len_now'] = 0
    buf['ring_pos'] = 0


def dflash_drafter_forward(frontend, prev_token_id: int,
                            hidden_taps,
                            drafter_cur_pos: int = 0):
    """Run one DFlash drafter cycle. Returns logits (16, vocab).

    P5 sliding-ring variant: each cycle appends the current cycle's
    fc(hidden_taps) to a ring buffer of past target_feat, and the
    drafter attention attends to ALL valid past slots (sliding window
    matches lucebox 'sliding target_feat ring' design).

    Args:
      frontend: Qwen36TorchFrontendRtx (NVFP4 main + DFlash drafter loaded)
      prev_token_id: int, the last accepted target token
      hidden_taps: bf16 tensor (5, hidden=5120) — captured target hidden
        states at layers [1, 16, 31, 46, 61], current step
      drafter_cur_pos: reserved (relative positions are baked in)

    Returns:
      bf16 tensor (16, vocab_size). Caller does argmax per row.
    """
    from flash_rt import flash_rt_kernels as fvk

    bf16 = torch.bfloat16
    s = torch.cuda.current_stream().cuda_stream
    buf = frontend._dflash_buf
    if buf is None:
        raise RuntimeError(
            'drafter scratch not allocated; call alloc_drafter_scratch '
            'before dflash_drafter_forward'
        )
    d = frontend._weights.ptrs['dflash']
    M = buf['block']
    H = buf['hidden']
    FC_IN = buf['fc_in']
    VOCAB = buf['vocab']
    MAX_CTX = buf['max_ctx']
    eps = float(d['rms_norm_eps'])

    # ---- 1) Build input_ids = [prev, MASK, MASK, ...] ---------------
    mask_id = int(buf['mask_token_id'])
    ids = torch.full((M,), mask_id, dtype=torch.long,
                     device=hidden_taps.device)
    ids[0] = int(prev_token_id)

    # ---- 2) Embed via main embed_tokens ------------------------------
    embed_t = frontend._weights.anchors[0]    # (vocab, hidden) bf16
    embed = embed_t[ids].to(bf16)             # (16, hidden)

    # ---- 3) New cycle: fc + rms_norm on this cycle's hidden_taps -----
    if hidden_taps.shape != (5, H):
        raise RuntimeError(
            f'hidden_taps shape {tuple(hidden_taps.shape)} != (5, {H})')
    tap_flat = hidden_taps.contiguous().view(1, FC_IN)
    ap_fc, sf_fc = buf['act_M1_K25600']
    _quant_act(fvk, tap_flat, ap_fc, sf_fc, 1, FC_IN, s)
    _gemm_nvfp4(fvk, ap_fc.data_ptr(), sf_fc.data_ptr(),
                d['fc_packed'], d['fc_sf'], d['fc_alpha'],
                buf['tap_proj_out'].data_ptr(), 1, H, FC_IN, s)

    # ---- 4) Append rms_norm(tap_proj, hidden_norm) into ring -----
    ring = buf['target_feat_ring']
    ring_pos = buf['ring_pos']
    fvk.rms_norm(
        int(buf['tap_proj_out'].data_ptr()), int(d['hidden_norm_w']),
        int(ring[ring_pos:ring_pos + 1].data_ptr()),
        1, H, eps, int(s),
    )
    # Advance ring; cap ctx_len_now at MAX_CTX (slide window).
    buf['ring_pos'] = (ring_pos + 1) % MAX_CTX
    raw_ctx = min(buf['ctx_len_now'] + 1, MAX_CTX)
    buf['ctx_len_now'] = raw_ctx

    # Effective ctx may be capped tighter than physical buffer to use
    # only the most recent N slots. Empirically AL peaks below MAX_CTX.
    eff_ctx_cap = int(getattr(frontend, '_dflash_eff_ctx', MAX_CTX))
    new_ctx_len = min(raw_ctx, eff_ctx_cap)

    # For the most-recent-N window: take the last new_ctx_len writes
    # ending at ring_pos-1 (the slot we JUST wrote). Two cases:
    #   1) No wrap and we want the last new_ctx_len: rows
    #      [ring_pos - new_ctx_len .. ring_pos - 1]
    #   2) Wrap: split via cat
    rp = buf['ring_pos']  # next write slot
    end_slot = (rp - 1) % MAX_CTX     # most recent slot we just wrote
    start_slot = (end_slot - new_ctx_len + 1) % MAX_CTX
    if start_slot <= end_slot:
        target_feat = ring[start_slot:end_slot + 1]
    else:
        # Wrap: oldest..end_of_ring + start_of_ring..most-recent
        target_feat = torch.cat(
            [ring[start_slot:], ring[:end_slot + 1]], dim=0)

    # ---- 5) h = noise embeddings only (NO residual add of target_feat)
    h = buf['h_b']
    h.copy_(embed)

    # ---- 6) Drafter layers 0..4 (target_feat shared across layers) ---
    for L in range(buf['n_layers']):
        h = _drafter_layer_forward(
            frontend, fvk, L, h, target_feat, new_ctx_len, s)

    # ---- 6) Final rmsnorm * norm.weight ------------------------------
    fvk.rms_norm(
        int(h.data_ptr()), int(d['final_norm_w']),
        int(buf['h_final_norm'].data_ptr()),
        M, H, eps, int(s),
    )

    # ---- 7) lm_head (shared with main, NVFP4 widen GEMM) -------------
    ap_lm, sf_lm = buf['act_M16_K5120']
    _quant_act(fvk, buf['h_final_norm'], ap_lm, sf_lm, M, H, s)
    _gemm_nvfp4(fvk, ap_lm.data_ptr(), sf_lm.data_ptr(),
                frontend._weights.ptrs['lm_head_packed'],
                frontend._weights.ptrs['lm_head_sf'],
                frontend._weights.ptrs['lm_head_alpha'],
                buf['logits'].data_ptr(),
                M, VOCAB, H, s, widen=True)
    return buf['logits']


# ====================================================================
# P7: Capture-friendly drafter forward (static buffers, fixed eff_ctx)
# ====================================================================
#
# Goal: capture the entire drafter forward into one CUDA Graph so the
# 220 per-call kernel launches collapse to ONE replay launch (~10us).
#
# Design choices for capture-friendliness:
#   1. Pre-allocated ids buffer (`ids_static`) initialized to
#      [PLACEHOLDER, MASK, MASK, ...]. External caller writes prev_token
#      to ids_static[0:1] before each replay.
#   2. Pre-allocated embed buffer; use index_select(... out=embed_buf)
#      instead of `embed_t[ids]` (which allocates).
#   3. Shift-window target_feat instead of circular ring: fixed-size
#      buffer (eff_ctx, hidden); each cycle shift left by 1, write the
#      new tap_proj_norm at the last slot. Read slice is fixed
#      [:eff_ctx] with stable pointers.
#   4. Static hidden_taps input buffer; caller copies real taps in.
#   5. ctx_len passed as a Python constant (eff_ctx) baked into the
#      captured graph; one graph per eff_ctx value.


def alloc_drafter_capture_window(frontend, eff_ctx: int) -> None:
    """Allocate the (eff_ctx, hidden) shift-window for the captured graph."""
    buf = frontend._dflash_buf
    if (buf['target_feat_window'] is not None
            and buf['eff_ctx_capture'] == eff_ctx):
        return
    H = buf['hidden']
    buf['target_feat_window'] = torch.zeros(
        eff_ctx, H, dtype=torch.bfloat16,
        device=frontend.device)
    buf['eff_ctx_capture'] = eff_ctx


def reset_drafter_capture_state(frontend) -> None:
    """Clear shift-window state. Call at the start of each generation."""
    buf = frontend._dflash_buf
    if buf is None:
        return
    if buf['target_feat_window'] is not None:
        buf['target_feat_window'].zero_()


def dflash_drafter_forward_capture_eager(frontend,
                                          valid_ctx: int) -> torch.Tensor:
    """Eager variant of capture forward used during ramp-up.

    Same as ``dflash_drafter_forward_capture`` but reads only the most
    recent ``valid_ctx`` slots of the shift-window — avoiding the
    zero-dilution that hurts AL when the window is partially filled.

    During ramp-up (first eff_ctx cycles), the orchestration calls this
    with valid_ctx growing 1..eff_ctx. Once valid_ctx == eff_ctx, the
    captured graph (which uses the full window) takes over. Both paths
    share the same target_feat_window state.
    """
    from flash_rt import flash_rt_kernels as fvk

    s = torch.cuda.current_stream().cuda_stream
    buf = frontend._dflash_buf
    d = frontend._weights.ptrs['dflash']
    M = buf['block']
    H = buf['hidden']
    FC_IN = buf['fc_in']
    VOCAB = buf['vocab']
    eps = float(d['rms_norm_eps'])
    eff_ctx = buf['eff_ctx_capture']
    if eff_ctx is None:
        raise RuntimeError(
            'eff_ctx_capture not set; call alloc_drafter_capture_window first')
    if not (1 <= valid_ctx <= eff_ctx):
        raise ValueError(
            f'valid_ctx={valid_ctx} must be in [1, eff_ctx={eff_ctx}]')

    # 1) Embed via static ids buffer.
    fvk.qwen36_embedding_lookup_bf16(
        buf['ids_static'].data_ptr(),
        int(frontend._weights.ptrs['embed_w']),
        buf['embed_buf'].data_ptr(), M, H, s,
    )

    # 2) fc(hidden_taps_static) -> rms_norm -> shift-write into window
    tap_flat = buf['hidden_taps_static'].view(1, FC_IN)
    ap_fc, sf_fc = buf['act_M1_K25600']
    _quant_act(fvk, tap_flat, ap_fc, sf_fc, 1, FC_IN, s)
    _gemm_nvfp4(fvk, ap_fc.data_ptr(), sf_fc.data_ptr(),
                d['fc_packed'], d['fc_sf'], d['fc_alpha'],
                buf['tap_proj_out'].data_ptr(), 1, H, FC_IN, s)
    win = buf['target_feat_window']
    if eff_ctx > 1:
        win[:-1].copy_(win[1:].clone())
    fvk.rms_norm(
        int(buf['tap_proj_out'].data_ptr()), int(d['hidden_norm_w']),
        int(win[eff_ctx - 1:eff_ctx].data_ptr()),
        1, H, eps, int(s),
    )

    # 3) h = embed
    fvk.gpu_copy(
        buf['h_b'].data_ptr(), buf['embed_buf'].data_ptr(),
        M * H * 2, s,
    )
    h = buf['h_b']

    # 4) Layer forward with ONLY the most-recent valid_ctx slots
    # (slice [eff_ctx-valid_ctx : eff_ctx] of the window).
    target_feat_valid = win[eff_ctx - valid_ctx:eff_ctx]
    for L in range(buf['n_layers']):
        h = _drafter_layer_forward(
            frontend, fvk, L, h, target_feat_valid, valid_ctx, s)

    # 5) Final rms_norm
    fvk.rms_norm(
        int(h.data_ptr()), int(d['final_norm_w']),
        int(buf['h_final_norm'].data_ptr()),
        M, H, eps, int(s),
    )

    # 6) lm_head NVFP4 widen
    ap_lm, sf_lm = buf['act_M16_K5120']
    _quant_act(fvk, buf['h_final_norm'], ap_lm, sf_lm, M, H, s)
    _gemm_nvfp4(fvk, ap_lm.data_ptr(), sf_lm.data_ptr(),
                frontend._weights.ptrs['lm_head_packed'],
                frontend._weights.ptrs['lm_head_sf'],
                frontend._weights.ptrs['lm_head_alpha'],
                buf['logits'].data_ptr(),
                M, VOCAB, H, s, widen=True)
    return buf['logits']


def dflash_drafter_forward_capture(frontend) -> torch.Tensor:
    """Capture-friendly drafter forward — reads from static buffers.

    Caller is responsible for setting up:
      - frontend._dflash_buf['ids_static'][0:1] := prev_token tensor
      - frontend._dflash_buf['hidden_taps_static'] := current taps

    Then either:
      - Call this function eagerly for warmup / first cycles
      - Or replay the captured graph (which is just this function baked in)

    Reads ctx_len from the pre-set frontend._dflash_buf['eff_ctx_capture'].

    Returns: logits tensor (16, vocab) bf16. Same buffer as
    dflash_drafter_forward, so callers can use either path.
    """
    from flash_rt import flash_rt_kernels as fvk

    s = torch.cuda.current_stream().cuda_stream
    buf = frontend._dflash_buf
    d = frontend._weights.ptrs['dflash']
    M = buf['block']
    H = buf['hidden']
    FC_IN = buf['fc_in']
    VOCAB = buf['vocab']
    eps = float(d['rms_norm_eps'])
    eff_ctx = buf['eff_ctx_capture']
    if eff_ctx is None:
        raise RuntimeError(
            'eff_ctx_capture not set; call alloc_drafter_capture_window first')

    # ---- 1) Embed via static ids buffer ----
    fvk.qwen36_embedding_lookup_bf16(
        buf['ids_static'].data_ptr(),
        int(frontend._weights.ptrs['embed_w']),
        buf['embed_buf'].data_ptr(), M, H, s,
    )

    # ---- 2) fc(hidden_taps_static) -> rms_norm -> shift-write window ----
    tap_flat = buf['hidden_taps_static'].view(1, FC_IN)
    ap_fc, sf_fc = buf['act_M1_K25600']
    _quant_act(fvk, tap_flat, ap_fc, sf_fc, 1, FC_IN, s)
    _gemm_nvfp4(fvk, ap_fc.data_ptr(), sf_fc.data_ptr(),
                d['fc_packed'], d['fc_sf'], d['fc_alpha'],
                buf['tap_proj_out'].data_ptr(), 1, H, FC_IN, s)

    # Shift-window: target_feat_window[0:-1] := target_feat_window[1:]
    # then write rms_norm(tap_proj, hidden_norm) into [-1].
    win = buf['target_feat_window']
    # Use roll-and-write style. For graph capture, the slice copies must
    # use pre-determined pointers — this works because win is fixed.
    if eff_ctx > 1:
        win[:-1].copy_(win[1:].clone())
    fvk.rms_norm(
        int(buf['tap_proj_out'].data_ptr()), int(d['hidden_norm_w']),
        int(win[eff_ctx - 1:eff_ctx].data_ptr()),
        1, H, eps, int(s),
    )

    # ---- 3) h = embed (no residual add of target_feat) ----
    fvk.gpu_copy(
        buf['h_b'].data_ptr(), buf['embed_buf'].data_ptr(),
        M * H * 2, s,
    )
    h = buf['h_b']

    # ---- 4) Drafter layers 0..4 with eff_ctx context ----
    for L in range(buf['n_layers']):
        h = _drafter_layer_forward(
            frontend, fvk, L, h, win, eff_ctx, s)

    # ---- 5) Final rmsnorm ----
    fvk.rms_norm(
        int(h.data_ptr()), int(d['final_norm_w']),
        int(buf['h_final_norm'].data_ptr()),
        M, H, eps, int(s),
    )

    # ---- 6) lm_head (NVFP4 widen) ----
    ap_lm, sf_lm = buf['act_M16_K5120']
    _quant_act(fvk, buf['h_final_norm'], ap_lm, sf_lm, M, H, s)
    _gemm_nvfp4(fvk, ap_lm.data_ptr(), sf_lm.data_ptr(),
                frontend._weights.ptrs['lm_head_packed'],
                frontend._weights.ptrs['lm_head_sf'],
                frontend._weights.ptrs['lm_head_alpha'],
                buf['logits'].data_ptr(),
                M, VOCAB, H, s, widen=True)
    return buf['logits']
