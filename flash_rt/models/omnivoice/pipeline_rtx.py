"""
OmniVoice FlashRT Engine — flashcli bundle module.

Mixed BF16 CFG + FP4 noCFG acceleration: BF16 establishes token structure
(step 1 with CFG), FP4 fills remaining steps. Preserves audio quality at 5x
speedup with RTF 0.032.

Architecture:
  FlashRTLlm       — FP4 W4A4 forward (CUDA Graph + FA2 + fused kernels)
  FlashRTLlmBF16   — BF16 forward (FlashRT kernels, no quantization)
  inject()         — Patch OmniVoice model for FlashRT acceleration
  free_encoder()   — Release encoder weights to save VRAM
"""
from __future__ import annotations
import logging, math, torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
BF16 = torch.bfloat16

# ── Qwen3-1.5B model constants ──
D = 1024; L = 28; NH = 16; NKV = 8; HD = 128; FFN = 3072; EPS = 1e-6
THETA = 1_000_000.0; NQK = NH * HD; KVD = NKV * HD
QKVD = NQK + 2 * KVD; GUD = 2 * FFN; MAX_S = 2048

from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample, _filter_top_k

# ── FlashRT kernel detection (supports flashcli deployment + local dev) ──
try:
    from flash_rt import flash_rt_kernels as _fvk
except ImportError:
    try:
        import flash_rt_kernels as _fvk
    except ImportError:
        _fvk = None

try:
    from flash_rt import flash_rt_omnivoice as _fvo
except ImportError:
    try:
        import flash_rt_omnivoice as _fvo
    except ImportError:
        _fvo = None

_has_cfg_kernel = _fvo is not None and hasattr(_fvo, 'cfg_combine_log_softmax_bf16')


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def _swizzled_sf_bytes(rows, cols):
    assert cols % 16 == 0
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 128 * 64

def _pick_gemm_variant(N, K):
    return 'pingpong' if N >= 4096 else 'default'

def _call_fp4_gemm(fvk, A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, variant, stream):
    if variant == 'pingpong':
        fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream)
    else:
        fvk.fp4_w4a16_gemm_sm120_bf16out(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream)


# ═══════════════════════════════════════════════════════════════
# FlashRTLlm — FP4 W4A4 forward engine
# ═══════════════════════════════════════════════════════════════

class FlashRTLlm:
    """FlashRT FP4 LLM — NVFP4 W4A4 GEMM + fused QK norm+rope + FA2 + CUDA Graph."""

    def __init__(self, qm, dev="cuda:0"):
        self.device = dev
        layers = qm.layers
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_fa2 as fa2
        self.fvk = fvk; self.fa2 = fa2

        self.WL_bf16 = []
        for i in range(L):
            ly = layers[i]
            self.WL_bf16.append({
                'qkv': torch.cat([ly.self_attn.q_proj.weight.data,
                                  ly.self_attn.k_proj.weight.data,
                                  ly.self_attn.v_proj.weight.data], 0).contiguous(),
                'gu': torch.cat([ly.mlp.gate_proj.weight.data,
                                  ly.mlp.up_proj.weight.data], 0).contiguous(),
                'o': ly.self_attn.o_proj.weight.data.contiguous(),
                'down': ly.mlp.down_proj.weight.data.contiguous(),
                'in_norm': ly.input_layernorm.weight.data.contiguous(),
                'post_norm': ly.post_attention_layernorm.weight.data.contiguous(),
                'qn': ly.self_attn.q_norm.weight.data.contiguous(),
                'kn': ly.self_attn.k_norm.weight.data.contiguous(),
            })
        self.final_norm = qm.norm.weight.data.contiguous()

        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
        pos = torch.arange(MAX_S, device=dev, dtype=torch.float32)
        f = torch.outer(pos, inv)
        self.rc = f.cos().to(BF16); self.rs = f.sin().to(BF16)
        self._rope_tables = {}; self._buf = None
        self._attn_o_buf = None; self._attn_lse = None
        self._fp4_ready = False; self.WL_fp4 = None; self._fp4_act = None; self._alphas = None
        self._graph = None; self._graph_pool = None; self._graph_stream = None
        self._graph_bs = 0; self._graph_input = None; self._graph_output = None
        self._num_sms = torch.cuda.get_device_properties(dev).multi_processor_count

    def _get_rope_bs(self, B, S):
        key = (B, S)
        if key not in self._rope_tables:
            cos_s = torch.cat([self.rc[:S], self.rc[:S]], -1)
            sin_s = torch.cat([self.rs[:S], self.rs[:S]], -1)
            self._rope_tables[key] = (cos_s.repeat(B, 1).contiguous(), sin_s.repeat(B, 1).contiguous())
        return self._rope_tables[key]

    def _alloc(self, B, S):
        BS = B * S; d = self.device
        self._buf = {
            'h': torch.empty(BS, D, device=d, dtype=BF16),
            'h2': torch.empty(BS, D, device=d, dtype=BF16),
            'xn': torch.empty(BS, D, device=d, dtype=BF16),
            'tmp': torch.empty(BS, D, device=d, dtype=BF16),
            'Dq': torch.empty(BS, QKVD, device=d, dtype=BF16),
            'Dg': torch.empty(BS, GUD, device=d, dtype=BF16),
            'q_flat': torch.empty(BS * NH, HD, device=d, dtype=BF16),
            'k_flat': torch.empty(BS * NKV, HD, device=d, dtype=BF16),
            'q_temp': torch.empty(BS * NH, HD, device=d, dtype=BF16),
            'k_temp': torch.empty(BS * NKV, HD, device=d, dtype=BF16),
        }
        self._attn_o_buf = torch.empty(B, S, NH, HD, device=d, dtype=BF16)
        sq = ((S + 127) // 128) * 128
        self._attn_lse = torch.empty(B, NH, sq, device=d, dtype=torch.float32)

    def _alloc_fp4_act(self, BS):
        d = self.device
        self._fp4_act = {
            'inp_packed': torch.empty(BS, D // 2, dtype=torch.uint8, device=d),
            'inp_sf': torch.zeros(_swizzled_sf_bytes(BS, D), dtype=torch.uint8, device=d),
            'ao_packed': torch.empty(BS, NQK // 2, dtype=torch.uint8, device=d),
            'ao_sf': torch.zeros(_swizzled_sf_bytes(BS, NQK), dtype=torch.uint8, device=d),
            'act_packed': torch.empty(BS, FFN // 2, dtype=torch.uint8, device=d),
            'act_sf': torch.zeros(_swizzled_sf_bytes(BS, FFN), dtype=torch.uint8, device=d),
        }

    def _quantize_weight(self, w_bf16):
        N, K = w_bf16.shape; d = w_bf16.device
        packed = torch.empty(N, K // 2, dtype=torch.uint8, device=d)
        sf = torch.zeros(_swizzled_sf_bytes(N, K), dtype=torch.uint8, device=d)
        self.fvk.quantize_bf16_to_nvfp4_swizzled_mse(
            w_bf16.data_ptr(), packed.data_ptr(), sf.data_ptr(), N, K,
            torch.cuda.current_stream().cuda_stream)
        return packed, sf

    @torch.inference_mode()
    def calibrate(self, calibration_input):
        B, S = calibration_input.shape[:2]; BS = B * S
        self._alloc(B, S)
        self.WL_fp4 = []; self._alphas = []
        for li in range(L):
            wb = self.WL_bf16[li]; wf = {}; al = {}
            for key in ['in_norm', 'post_norm', 'qn', 'kn']:
                wf[key] = wb[key]
            for key, N, K in [('qkv', QKVD, D), ('gu', GUD, D), ('o', D, NQK), ('down', D, FFN)]:
                packed, sf = self._quantize_weight(wb[key])
                wf[key + '_packed'] = packed; wf[key + '_sf'] = sf; al[key] = 1.0
            self.WL_fp4.append(wf); self._alphas.append(al)
        self._fp4_ready = True
        log.info("FP4: weight quantization complete (%d layers)", L)

    @torch.inference_mode()
    def forward(self, inputs_embeds, attention_mask=None, **kw):
        if not self._fp4_ready: self.calibrate(inputs_embeds)
        fvk = self.fvk; st = torch.cuda.current_stream().cuda_stream
        B, S0, _ = inputs_embeds.shape; BS = B * S0
        if self._buf is None or self._buf['h'].shape[0] != BS: self._alloc(B, S0)
        if self._fp4_act is None or self._fp4_act['inp_packed'].shape[0] != BS: self._alloc_fp4_act(BS)
        b = self._buf; fa = self._fp4_act; cos_bs, sin_bs = self._get_rope_bs(B, S0)
        b['h'].copy_(inputs_embeds.to(BF16).contiguous().reshape(BS, D))
        fvk.rms_norm_to_nvfp4_swizzled_bf16(b['h'].data_ptr(), self.WL_fp4[0]['in_norm'].data_ptr(),
            fa['inp_packed'].data_ptr(), fa['inp_sf'].data_ptr(), BS, D, EPS, st)
        h_ptr = b['h'].data_ptr(); h2_ptr = b['h2'].data_ptr()
        for li in range(L):
            w = self.WL_fp4[li]; al = self._alphas[li]; is_last = (li == L - 1)
            next_in_norm_ptr = (self.WL_fp4[li + 1]['in_norm'].data_ptr() if not is_last else 0)
            _call_fp4_gemm(fvk, fa['inp_packed'].data_ptr(), w['qkv_packed'].data_ptr(),
                b['Dq'].data_ptr(), BS, QKVD, D, fa['inp_sf'].data_ptr(), w['qkv_sf'].data_ptr(),
                al['qkv'], _pick_gemm_variant(QKVD, D), st)
            fvk.fused_qk_norm_rope_v4_bf16(b['Dq'].data_ptr(), w['qn'].data_ptr(), w['kn'].data_ptr(),
                cos_bs.data_ptr(), sin_bs.data_ptr(),
                b['q_temp'].data_ptr(), b['k_temp'].data_ptr(), BS, NH, NKV, HD, QKVD, EPS, st)
            b['q_flat'], b['q_temp'] = b['q_temp'], b['q_flat']
            b['k_flat'], b['k_temp'] = b['k_temp'], b['k_flat']
            qr = b['q_flat'].view(B, S0, NH, HD).contiguous()
            kr = b['k_flat'].view(B, S0, NKV, HD).contiguous()
            vv = b['Dq'][:, NQK + KVD:].contiguous().reshape(B, S0, NKV, HD).contiguous()
            self._fa2_fwd(qr, kr, vv, B, S0, NH, NKV, HD, st)
            ao_flat = self._attn_o_buf.reshape(BS, NQK).contiguous()
            fvk.quantize_bf16_to_nvfp4_swizzled(ao_flat.data_ptr(), fa['ao_packed'].data_ptr(), fa['ao_sf'].data_ptr(), BS, NQK, st)
            _call_fp4_gemm(fvk, fa['ao_packed'].data_ptr(), w['o_packed'].data_ptr(), b['tmp'].data_ptr(),
                BS, D, NQK, fa['ao_sf'].data_ptr(), w['o_sf'].data_ptr(), al['o'], _pick_gemm_variant(D, NQK), st)
            fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(h_ptr, b['tmp'].data_ptr(), h2_ptr,
                w['post_norm'].data_ptr(), fa['inp_packed'].data_ptr(), fa['inp_sf'].data_ptr(), BS, D, EPS, st)
            h_ptr, h2_ptr = h2_ptr, h_ptr
            _call_fp4_gemm(fvk, fa['inp_packed'].data_ptr(), w['gu_packed'].data_ptr(), b['Dg'].data_ptr(),
                BS, GUD, D, fa['inp_sf'].data_ptr(), w['gu_sf'].data_ptr(), al['gu'], _pick_gemm_variant(GUD, D), st)
            fvk.silu_mul_merged_to_nvfp4_swizzled_bf16(b['Dg'].data_ptr(), fa['act_packed'].data_ptr(), fa['act_sf'].data_ptr(), BS, FFN, st)
            _call_fp4_gemm(fvk, fa['act_packed'].data_ptr(), w['down_packed'].data_ptr(), b['tmp'].data_ptr(),
                BS, D, FFN, fa['act_sf'].data_ptr(), w['down_sf'].data_ptr(), al['down'], _pick_gemm_variant(D, FFN), st)
            if not is_last:
                fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(h_ptr, b['tmp'].data_ptr(), h2_ptr,
                    next_in_norm_ptr, fa['inp_packed'].data_ptr(), fa['inp_sf'].data_ptr(), BS, D, EPS, st)
                h_ptr, h2_ptr = h2_ptr, h_ptr
            else:
                fvk.residual_add_rms_norm(h_ptr, b['tmp'].data_ptr(), self.final_norm.data_ptr(),
                    b['xn'].data_ptr(), BS, D, EPS, st)
        return (b['xn'].reshape(B, S0, D).to(inputs_embeds.dtype),)

    def _fa2_fwd(self, q, k, v, B, S, NH, NKV, HD, stream):
        o = self._attn_o_buf
        self.fa2.fwd_bf16(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
            self._attn_lse.data_ptr(), 0, 0, B, S, S, NH, NKV, HD,
            (q.stride(0), q.stride(1), q.stride(2)), (k.stride(0), k.stride(1), k.stride(2)),
            (v.stride(0), v.stride(1), v.stride(2)), (o.stride(0), o.stride(1), o.stride(2)),
            HD ** -0.5, self._num_sms, stream)

    @torch.inference_mode()
    def _capture_graph(self, inputs_embeds):
        if self._graph is not None: return
        B, S, _ = inputs_embeds.shape; BS = B * S
        self._graph_stream = torch.cuda.Stream(device=self.device)
        if self._graph_pool is None: self._graph_pool = torch.cuda.graph_pool_handle()
        self._graph_input = torch.empty(B, S, D, device=self.device, dtype=BF16)
        self._graph_output = torch.empty(B, S, D, device=self.device, dtype=BF16)
        gs = self._graph_stream; gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            for _ in range(3):
                self._graph_input.copy_(inputs_embeds.to(BF16).contiguous()); self.forward(self._graph_input)
            self._graph_input.copy_(inputs_embeds.to(BF16).contiguous())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._graph_pool, stream=gs):
                out = self.forward(self._graph_input)
                self._graph_output.copy_(out[0].contiguous())
        torch.cuda.current_stream().wait_stream(gs)
        self._graph = g; self._graph_bs = BS
        log.info("FP4: CUDA Graph captured (B=%d,S=%d)", B, S)

    @torch.inference_mode()
    def forward_graph(self, inputs_embeds, attention_mask=None, **kw):
        if self._graph is None or inputs_embeds.shape[0]*inputs_embeds.shape[1] != self._graph_bs:
            self._graph = None; self._capture_graph(inputs_embeds)
            return self.forward(inputs_embeds, attention_mask, **kw)
        self._graph_input.copy_(inputs_embeds.to(BF16).contiguous())
        self._graph.replay()
        return (self._graph_output.to(inputs_embeds.dtype),)


# ═══════════════════════════════════════════════════════════════
# FlashRTLlmBF16 — BF16 forward engine (no quantization)
# ═══════════════════════════════════════════════════════════════

class FlashRTLlmBF16(FlashRTLlm):
    """BF16 FlashRT — fused kernels + FA2, zero quantization overhead."""

    @torch.inference_mode()
    def calibrate(self, calibration_input):
        B, S = calibration_input.shape[:2]
        self._alloc(B, S)
        self.WL_fp4 = self.WL_bf16
        self._alphas = [{'qkv':1.0,'o':1.0,'gu':1.0,'down':1.0} for _ in range(L)]
        self._fp4_ready = True; self._fp4_act = None
        log.info("BF16 engine: ready (%d layers, no quantization)", L)

    @torch.inference_mode()
    def forward(self, inputs_embeds, attention_mask=None, **kw):
        if not self._fp4_ready: self.calibrate(inputs_embeds)
        fvk = self.fvk; st = torch.cuda.current_stream().cuda_stream
        B, S0, _ = inputs_embeds.shape; BS = B * S0
        if self._buf is None or self._buf['h'].shape[0] != BS: self._alloc(B, S0)
        b = self._buf; cos_bs, sin_bs = self._get_rope_bs(B, S0)
        b['h'].copy_(inputs_embeds.to(BF16).contiguous().reshape(BS, D))
        for li in range(L):
            wb = self.WL_bf16[li]
            fvk.rms_norm(b['h'].data_ptr(), wb['in_norm'].data_ptr(), b['xn'].data_ptr(), BS, D, EPS, st)
            torch.matmul(b['xn'], wb['qkv'].t(), out=b['Dq'])
            fvk.fused_qk_norm_rope_v4_bf16(b['Dq'].data_ptr(), wb['qn'].data_ptr(), wb['kn'].data_ptr(),
                cos_bs.data_ptr(), sin_bs.data_ptr(),
                b['q_temp'].data_ptr(), b['k_temp'].data_ptr(), BS, NH, NKV, HD, QKVD, EPS, st)
            b['q_flat'], b['q_temp'] = b['q_temp'], b['q_flat']
            b['k_flat'], b['k_temp'] = b['k_temp'], b['k_flat']
            qr = b['q_flat'].view(B, S0, NH, HD).contiguous()
            kr = b['k_flat'].view(B, S0, NKV, HD).contiguous()
            vv = b['Dq'][:, NQK + KVD:].contiguous().reshape(B, S0, NKV, HD).contiguous()
            self._fa2_fwd(qr, kr, vv, B, S0, NH, NKV, HD, st)
            ao_flat = self._attn_o_buf.reshape(BS, NQK).contiguous()
            torch.matmul(ao_flat, wb['o'].t(), out=b['tmp'])
            b['h'] += b['tmp']
            fvk.rms_norm(b['h'].data_ptr(), wb['post_norm'].data_ptr(), b['xn'].data_ptr(), BS, D, EPS, st)
            torch.matmul(b['xn'], wb['gu'].t(), out=b['Dg'])
            act = F.silu(b['Dg'][:, :FFN]) * b['Dg'][:, FFN:]
            torch.matmul(act, wb['down'].t(), out=b['tmp'])
            b['h'] += b['tmp']
        fvk.rms_norm(b['h'].data_ptr(), self.final_norm.data_ptr(), b['xn'].data_ptr(), BS, D, EPS, st)
        return (b['xn'].reshape(B, S0, D).to(inputs_embeds.dtype),)


# ═══════════════════════════════════════════════════════════════
# MaskGIT loop — BF16 CFG + FP4 noCFG hybrid
# ═══════════════════════════════════════════════════════════════

def _make_cfg(gs, mask_id):
    if _has_cfg_kernel and gs > 0:
        _sfn = lambda: torch.cuda.current_stream().cuda_stream
        def _cfg(c, u):
            B,C,S,V = c.shape; r=B*C*S
            cf=c.reshape(r,V).contiguous(); uf=u.reshape(r,V).contiguous()
            out=torch.empty_like(cf)
            _fvo.cfg_combine_log_softmax_bf16(cf.data_ptr(),uf.data_ptr(),out.data_ptr(),r,V,mask_id,gs,_sfn())
            return out.view(B,C,S,V)
        return _cfg
    def _cfg(c, u):
        cl=torch.nn.functional.log_softmax(c.float(),dim=-1); cl[...,mask_id]=-float('inf')
        if gs <= 0: return cl
        ul=torch.nn.functional.log_softmax(u.float(),dim=-1)
        return torch.log_softmax(cl+gs*(cl-ul),dim=-1)
    return _cfg


def _optimize_maskgit(model, frt_fp4, cfg_ratio=0.05, bookend=False):
    num_cb = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    _dev = model.device
    _arange_cb = torch.arange(num_cb, device=_dev).view(1, -1, 1)

    def _sample_step(logits, step, tokens, schedules, c_lens, t_lens, lp_fn, ct, pt, lpf, B_n, u_logits=None):
        for i in range(B_n):
            k = schedules[i][step]
            if k <= 0: continue
            cl, tl = c_lens[i], t_lens[i]
            lp = lp_fn(logits[i:i+1,:,cl-tl:cl,:], u_logits[i:i+1,:,:tl,:]) if u_logits is not None else lp_fn(logits[i:i+1,:,cl-tl:cl,:])
            pred = _gumbel_sample(_filter_top_k(lp,0.1),ct).argmax(-1) if ct>0 else lp.argmax(-1)
            scores = lp.max(-1)[0] - _arange_cb * lpf
            if pt > 0: scores = _gumbel_sample(scores, pt)
            st_ = tokens[i:i+1,:,:tl]; scores.masked_fill_(st_!=mask_id, -float("inf"))
            _, idx = torch.topk(scores.flatten(), k)
            ft = st_.flatten(); ft[idx] = pred.flatten()[idx]; st_.copy_(ft.view_as(st_))
            tokens[i:i+1,:,:tl] = st_

    def _gen(task, gen_config):
        B = task.batch_size
        inputs = [model._prepare_inference_inputs(
            task.texts[i], task.target_lens[i], task.ref_texts[i],
            task.ref_audio_tokens[i], task.langs[i], task.instructs[i], gen_config.denoise) for i in range(B)]
        c_lens = [inp["input_ids"].size(2) for inp in inputs]
        max_c = max(c_lens); max_t = max(task.target_lens)
        gs = gen_config.guidance_scale; NS = gen_config.num_step
        cfg_step = min(int(NS * cfg_ratio), NS - 1 if bookend else NS)
        ct, pt, lpf = gen_config.class_temperature, gen_config.position_temperature, gen_config.layer_penalty_factor

        ts = _get_time_steps(0.0, 1.0, NS, gen_config.t_shift).tolist()
        schedules = []
        for tl in task.target_lens:
            tm=tl*num_cb; rem=tm; sc=[]
            for step in range(NS):
                n = rem if step==NS-1 else min(math.ceil(tm*(ts[step+1]-ts[step])), rem)
                sc.append(int(n)); rem -= int(n)
            schedules.append(sc)

        tokens = torch.full((B, num_cb, max_t), mask_id, dtype=torch.long, device=_dev)
        _cfg = _make_cfg(gs, mask_id)
        _model = model.__call__

        def _nocfg_lp(c_lg):
            lp = torch.nn.functional.log_softmax(c_lg, dim=-1); lp[...,mask_id] = -float('inf'); return lp

        # Phase 1: BF16 + CFG (B=2)
        bm = 2
        bid2 = torch.full((bm*B, num_cb, max_c), mask_id, dtype=torch.long, device=_dev)
        bam2 = torch.zeros((bm*B, max_c), dtype=torch.bool, device=_dev)
        battn2 = torch.zeros((bm*B, 1, max_c, max_c), dtype=torch.bool, device=_dev)
        for i, inp in enumerate(inputs):
            cl = c_lens[i]; bid2[i,:,:cl]=inp["input_ids"]; bam2[i,:cl]=inp["audio_mask"]; battn2[i,:,:cl,:cl]=True
            ul = task.target_lens[i]
            bid2[B+i,:,:ul]=inp["input_ids"][...,-ul:]; bam2[B+i,:ul]=inp["audio_mask"][...,-ul:]
            battn2[B+i,:,:ul,:ul]=True
            if max_c>ul: pd=torch.arange(ul,max_c,device=_dev); battn2[B+i,:,pd,pd]=True

        for step in range(cfg_step):
            logits = _model(input_ids=bid2, audio_mask=bam2, attention_mask=battn2).logits
            _sample_step(logits, step, tokens, schedules, c_lens, task.target_lens, _cfg, ct, pt, lpf, B, u_logits=logits[B:])
            for i in range(B):
                cl,tl = c_lens[i], task.target_lens[i]
                bid2[i:i+1,:,cl-tl:cl]=tokens[i:i+1,:,:tl]; bid2[B+i:B+i+1,:,:tl]=tokens[i:i+1,:,:tl]

        # Phase 2: FP4 + noCFG (B=1)
        orig_fwd = model.llm.forward
        def _fp4_fwd(*a, **kw):
            e = kw.get('inputs_embeds')
            if e is not None: del kw['inputs_embeds']
            elif len(a) >= 2 and isinstance(a[1], torch.Tensor): e = a[1]
            else: return orig_fwd(*a, **kw)
            if frt_fp4._graph is not None: return frt_fp4.forward_graph(e, **kw)
            return frt_fp4.forward(e, **kw)
        model.llm.forward = _fp4_fwd
        try:
            bid1 = bid2[:B].clone(); bam1 = bam2[:B].clone(); battn1 = battn2[:B].clone()
            fp4_end = NS - 1 if bookend else NS
            for step in range(cfg_step, fp4_end):
                logits = _model(input_ids=bid1, audio_mask=bam1, attention_mask=battn1).logits
                _sample_step(logits, step, tokens, schedules, c_lens, task.target_lens, _nocfg_lp, ct, pt, lpf, B)
                for i in range(B):
                    cl,tl = c_lens[i], task.target_lens[i]
                    bid1[i:i+1,:,cl-tl:cl]=tokens[i:i+1,:,:tl]
        finally:
            model.llm.forward = orig_fwd

        # Phase 3: BF16 + CFG bookend
        if bookend and fp4_end < NS:
            for i in range(B):
                cl,tl = c_lens[i], task.target_lens[i]
                bid2[i:i+1,:,cl-tl:cl]=tokens[i:i+1,:,:tl]; bid2[B+i:B+i+1,:,:tl]=tokens[i:i+1,:,:tl]
            step = NS - 1
            logits = _model(input_ids=bid2, audio_mask=bam2, attention_mask=battn2).logits
            _sample_step(logits, step, tokens, schedules, c_lens, task.target_lens, _cfg, ct, pt, lpf, B, u_logits=logits[B:])

        return [tokens[i,:,:task.target_lens[i]] for i in range(B)]

    model._generate_iterative = _gen


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

_injected = False; _frt_bf16 = None; _frt_fp4 = None; _orig = None; _orig_gen = None

def _check_kernels():
    """Verify all required FlashRT kernel modules are available.

    Raises RuntimeError with rebuild instructions if any kernel is missing.
    Call before model loading to fail fast.
    """
    missing = []
    if _fvk is None:
        missing.append("flash_rt_kernels — FlashRT base not installed")
    else:
        for sym in (
            "fp4_w4a16_gemm_sm120_bf16out",
            "rms_norm",
            "residual_add_rms_norm_to_nvfp4_swizzled_bf16",
            "silu_mul_merged_to_nvfp4_swizzled_bf16",
            "fused_qk_norm_rope_v4_bf16",
            "quantize_bf16_to_nvfp4_swizzled_mse",
        ):
            if not hasattr(_fvk, sym):
                missing.append("flash_rt_kernels." + sym)

    if _fvo is None or not hasattr(_fvo, "cfg_combine_log_softmax_bf16"):
        missing.append(
            "flash_rt_omnivoice — cfg_combine_log_softmax_bf16 not found; "
            "rebuild FlashRT with: cmake -DFLASHRT_ENABLE_OMNIVOICE=ON -DGPU_ARCH=120"
        )

    if missing:
        raise RuntimeError(
            "OmniVoice FlashRT engine: required kernel symbols missing.\n"
            + "\n".join("  missing: " + s for s in missing)
        )

def inject(m, cfg_ratio=0.05, bookend=False):
    """Patch OmniVoice model for FlashRT acceleration.

    Args:
        m: OmniVoice model instance
        cfg_ratio: Fraction of steps using BF16 CFG forward (0.05 = 5% BF16, 95% FP4)
        bookend: If True, the final step also uses BF16 CFG
    """
    global _injected, _frt_bf16, _frt_fp4, _orig, _orig_gen
    if _injected: return
    _check_kernels()

    _frt_bf16 = FlashRTLlmBF16(m.llm, str(m.device))
    _orig = m.llm.forward
    _frt_fp4 = FlashRTLlm(m.llm, str(m.device))

    c2 = torch.randn(2, 178, 1024, device=m.device, dtype=BF16) * 0.02
    _frt_bf16.calibrate(c2)
    _frt_fp4.calibrate(c2)

    _frt_fp4.WL_bf16 = None
    _frt_bf16.WL_fp4 = None; _frt_bf16._fp4_act = None; _frt_bf16._alphas = None
    for layer in m.llm.layers:
        for param in layer.parameters(recurse=True):
            param.data = torch.empty(0, device=param.device, dtype=param.dtype)
    torch.cuda.empty_cache()

    c1 = torch.randn(1, 178, 1024, device=m.device, dtype=BF16) * 0.02
    for _ in range(3): _frt_fp4.forward(c1)
    torch.cuda.synchronize(); _frt_fp4._capture_graph(c1)
    for _ in range(3): _frt_fp4.forward_graph(c1); torch.cuda.synchronize()
    _frt_bf16._graph = None

    def _fwd_bf16(e, **kw): return _frt_bf16.forward(e, **kw)
    def p(*a, **kw):
        e = kw.get('inputs_embeds')
        if e is not None: del kw['inputs_embeds']; return _fwd_bf16(e, **kw)
        if len(a) >= 2 and isinstance(a[1], torch.Tensor): return _fwd_bf16(a[1], **kw)
        return _orig(*a, **kw)
    m.llm.forward = p; m._frt = _frt_bf16

    _orig_gen = m._generate_iterative
    _optimize_maskgit(m, _frt_fp4, cfg_ratio, bookend)
    _injected = True
    log.info("Engine injected: BF16 CFG(%.0f%%) + FP4 noCFG + bookend=%s", cfg_ratio*100, bookend)

def free_encoder(m):
    """Release AudioTokenizer encoder weights (~600 MB).

    After freeing, new voice clone prompts cannot be created.
    """
    if getattr(m, '_encoder_freed', False): return 0
    freed = 0
    for name in ['semantic_model', 'acoustic_encoder', 'encoder_semantic']:
        mod = getattr(m.audio_tokenizer, name, None)
        if mod is not None:
            s = sum(p.numel()*p.element_size() for p in mod.parameters())
            freed += s
            for p in mod.parameters(): p.data = torch.empty(0, device=p.device, dtype=p.dtype)
            log.info("Freed %s (%d MB)", name, s//1024//1024)
    torch.cuda.empty_cache()
    m._encoder_freed = True
    return freed // 1024 // 1024

def eject(m):
    """Restore original forward and generate methods."""
    global _injected, _orig, _orig_gen
    if not _injected: return
    if _orig is not None: m.llm.forward = _orig
    if _orig_gen is not None: m._generate_iterative = _orig_gen
    _injected = False
