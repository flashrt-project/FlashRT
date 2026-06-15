#!/usr/bin/env python3
"""Production all-kernel fused Cosmos3 denoise step — ZERO torch compute in the
hot path. Joint contiguous buffer (und=[0:103], gen=[103:736]); per-tower slices.
Kernels: NVFP4 GEMM (both towers), rms_norm (qk-norm batched), qwen36 batched
rope (q+k one launch), residual_add_rms_norm (fused residual+next-norm),
silu_mul_qwen36, FA2. Timestep embeds precomputed outside the graph (static table).
No cat / scatter / torch-rope / torch-residual.
"""
import os, sys, math, time, torch, torch.nn.functional as F
from safetensors import safe_open
import flash_rt.flash_rt_kernels as fvk
from flash_rt import flash_rt_fa2 as fa2
from .fm_solvers_unipc import FlowUniPCMultistepScheduler

W = os.environ.get("COSMOS_W", os.environ.get("W", "/work/cosmos3_weights_dom27.safetensors"))
R = os.environ.get("COSMOS_R", os.environ.get("R", "/work/ref_dump/tensors.safetensors"))
DEV = os.environ.get("COSMOS_DEV", "cuda"); EPS = 1e-6; H, KV, D, FF, HID, NL = 32, 8, 128, 12288, 4096, 64*64*0+36

def _shape_from_ref():
    try:
        with safe_open(R, "pt", device="cpu") as rf:
            nu = int(rf.get_tensor("s00/lm_in__causal_seq").shape[0])
            ng = int(rf.get_tensor("s00/lm_in__full_only_seq").shape[0])
            if os.environ.get("COSMOS_AV") or os.environ.get("AV"):
                av = int(os.environ.get("COSMOS_AV", os.environ.get("AV")))
            elif "once/vae2llm_out" in rf.keys():
                av = int(rf.get_tensor("once/vae2llm_out").shape[0])
            else:
                av = 600
            return nu, ng, nu + ng, av
    except Exception:
        return 103, 633, 736, 600

NU, NG, NJ, AV = _shape_from_ref()    # und, gen, joint, n_vision

def _action_shape_from_ref():
    try:
        with safe_open(R, "pt", device="cpu") as rf:
            al = int(rf.get_tensor("s00/action2llm_in").shape[0])
            vl = int(rf.get_tensor("s00/velocity").shape[0])
            return al, vl, al - vl
    except Exception:
        return 33, 32, 1

AL, VL, AC = _action_shape_from_ref()  # all action tokens, velocity/noisy tokens, clean prefix tokens
BF = torch.bfloat16; F8 = torch.float8_e4m3fn
NSM = torch.cuda.get_device_properties(0).multi_processor_count
def sf_bytes(rows, cols): return ((rows + 127) // 128) * ((cols // 16 + 3) // 4) * 128 * 64
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")


class CosmosV2:
    def __init__(self, bf16_projs=()):
        self.bf16_projs = set(bf16_projs)   # projs kept in bf16 (rest NVFP4)
        wf = safe_open(W, "pt", device=DEV); rf = safe_open(R, "pt", device=DEV)
        T = lambda k: wf.get_tensor(k).to(BF).t().contiguous()
        N = lambda k: wf.get_tensor(k).to(BF)
        # NVFP4-quantize all proj weights ([N,K] original orientation)
        self.Wp = {}; self.Wsf = {}; self.Wn = {}; self.Wt_bf16 = {}
        wq = os.environ.get("COSMOS_WQUANT", "default")  # default | mse (mse: MSE-optimal block scale, zero runtime cost)
        def qz(w_nk):
            n_, k_ = w_nk.shape
            p = torch.empty(n_, k_ // 2, dtype=torch.uint8, device=DEV)
            sf = torch.zeros(sf_bytes(n_, k_), dtype=torch.uint8, device=DEV)
            if wq == "mse":
                fvk.quantize_bf16_to_nvfp4_swizzled_mse(w_nk.contiguous().data_ptr(), p.data_ptr(), sf.data_ptr(), n_, k_, 0)
            elif wq == "secondmax":
                fvk.quantize_bf16_to_nvfp4_swizzled_secondmax(w_nk.contiguous().data_ptr(), p.data_ptr(), sf.data_ptr(), n_, k_, float(os.environ.get("COSMOS_SMAX_MULT", "1.0")), 0)
            else:
                fvk.quantize_bf16_to_nvfp4_swizzled(w_nk.contiguous().data_ptr(), p.data_ptr(), sf.data_ptr(), n_, k_, 0)
            return p, sf
        for li in range(NL):
            P = f"language_model.model.layers.{li}."
            for suf in ("", "_moe_gen"):
                for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    wk = P + f"self_attn.{nm}{suf}.weight"
                    if nm in self.bf16_projs: self.Wt_bf16[(li, suf, nm)] = T(wk)
                    else: self.Wp[(li, suf, nm)], self.Wsf[(li, suf, nm)] = qz(N(wk))
                for nm, mk in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                    wk = P + f"mlp{suf}.{mk}.weight"
                    if nm in self.bf16_projs: self.Wt_bf16[(li, suf, nm)] = T(wk)
                    else: self.Wp[(li, suf, nm)], self.Wsf[(li, suf, nm)] = qz(N(wk))
                for nm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
                    self.Wn[(li, suf, nm)] = N(P + f"{nm}{suf}.weight")
        self.norm_g = N("language_model.model.norm_moe_gen.weight")
        self.norm_u = N("language_model.model.norm.weight")
        self.Wll = N("llm2action.weight"); self.bll = N("llm2action.bias").float()
        self.Wa = N("action2llm.weight"); self.ba_mod = (N("action2llm.bias").float() + N("action_modality_embed").float()).to(BF)
        self.te0w = wf.get_tensor("time_embedder.mlp.0.weight").float(); self.te0b = wf.get_tensor("time_embedder.mlp.0.bias").float()
        self.te2w = wf.get_tensor("time_embedder.mlp.2.weight").float(); self.te2b = wf.get_tensor("time_embedder.mlp.2.bias").float()
        self.freqs = torch.exp(-math.log(10000) * torch.arange(0, 64, dtype=torch.float32, device=DEV) / 64)
        r = lambda k: rf.get_tensor("s00/" + k)
        # joint cos/sin [736,128] = [und ; gen]
        self.cos = torch.cat([r("rope_cos__causal_seq"), r("rope_cos__full_only_seq")], 0).contiguous()
        self.sin = torch.cat([r("rope_sin__causal_seq"), r("rope_sin__full_only_seq")], 0).contiguous()
        self.gemm = fvk.GemmRunner()
        # static conditioning
        self.vision = r("lm_in__full_only_seq")[:AV].clone()
        self.text = r("lm_in__causal_seq").clone()
        self.clean = rf.get_tensor("s00/action2llm_in")[:AC].clone() if AC else None
        # ---- preallocated buffers ----
        z = lambda *s: torch.zeros(*s, device=DEV, dtype=BF)
        self.Hb = z(NJ, HID); self.nrm = z(NJ, HID); self.nrm2 = z(NJ, HID)
        self.Qb = z(NJ, H, D); self.Kb = z(NJ, KV, D); self.Vb = z(NJ, KV, D)
        self.attn = z(NJ, H * D); self.ob = z(NJ, HID)
        self.g = z(NJ, FF); self.u = z(NJ, FF); self.act = z(NJ, FF); self.dn = z(NJ, HID)
        self.xfull = z(AL, 64); self.aenc = z(AL, HID)
        self.x = self.xfull[AC:AL]                 # graph input slot (noisy latents)
        self.te = z(VL, HID)                       # graph input slot (timestep emb, precomputed)
        self.vel = z(VL, 64)
        self.ba_modA = self.ba_mod.unsqueeze(0).expand(AL, HID).contiguous()  # static broadcast bias
        self.bllV = N("llm2action.bias").unsqueeze(0).expand(VL, 64).contiguous()  # static
        self.vtmp = z(VL, 64)
        self.lse = torch.empty(1, H, max(NU, NG), dtype=torch.float32, device=DEV)
        # nvfp4 activation scratch (per distinct (M,K))
        self.aq = {}
        for (M, K) in [(NU, HID), (NG, HID), (NU, FF), (NG, FF)]:
            self.aq[(M, K)] = (torch.empty(M, K // 2, dtype=torch.uint8, device=DEV), torch.zeros(sf_bytes(M, K), dtype=torch.uint8, device=DEV))
        if AC:
            self.xfull[:AC].copy_(self.clean)
        self.gr = None
        self.calib = False; self.cal_q = {}; self.cal_g = {}   # AWQ per-channel amax capture
        torch.cuda.synchronize()

    def _rec(self, store, key, x):
        a = x.detach().abs().amax(0).float()
        store[key] = a if key not in store else torch.maximum(store[key], a)

    def _s(self): return torch.cuda.current_stream().cuda_stream
    def _nvfp4(self, A, key, out, Nn):  # A[M,K] bf16 -> out[M,Nn] bf16 via nvfp4
        s = self._s(); M, K = A.shape; ap, asf = self.aq[(M, K)]
        fvk.quantize_bf16_to_nvfp4_swizzled(A.data_ptr(), ap.data_ptr(), asf.data_ptr(), M, K, s)
        fvk.fp4_w4a16_gemm_sm120_bf16out(ap.data_ptr(), self.Wp[key].data_ptr(), out.data_ptr(),
                                         M, Nn, K, asf.data_ptr(), self.Wsf[key].data_ptr(), 1.0, s)
    def _proj(self, A, key, out, Nn):   # route: bf16 if in bf16_projs else nvfp4
        if key[2] in self.bf16_projs:
            M, K = A.shape
            self.gemm.bf16_nn(A.data_ptr(), self.Wt_bf16[key].data_ptr(), out.data_ptr(), M, Nn, K, self._s())
        else:
            self._nvfp4(A, key, out, Nn)
    def _rms(self, x, w, out, rows, dim):
        fvk.rms_norm(x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, dim, EPS, self._s())
    def _radd_rms(self, h, x, w, out, rows):
        fvk.residual_add_rms_norm(h.data_ptr(), x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, HID, EPS, self._s())
    def _silu(self, g, u, out, n):
        fvk.silu_mul_qwen36_bf16(g.data_ptr(), u.data_ptr(), out.data_ptr(), n, self._s())
    def _fa(self, q, k, v, o, nq, nk, causal):
        s = self._s()
        fwd = fa2.fwd_bf16_causal if causal else fa2.fwd_bf16
        qs = (nq * H * D, H * D, D); ks = (nk * KV * D, KV * D, D)   # (batch, seq, head) strides
        fwd(Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=o.data_ptr(), softmax_lse=self.lse.data_ptr(),
            softmax_lse_accum=0, o_accum=0, batch=1, seqlen_q=nq, seqlen_k=nk, num_heads_q=H, num_heads_kv=KV, head_dim=D,
            q_strides=qs, k_strides=ks, v_strides=ks, o_strides=qs,
            softmax_scale=D ** -0.5, num_sms=NSM, stream=s)

    def forward(self):
        s = self._s()
        # ---- encode (all-kernel) ----
        # action2llm: aenc = xfull @ Wa ; + (ba+mod) broadcast ; + timestep on noisy
        self.gemm.bf16_nn(self.xfull.data_ptr(), self.Wa.data_ptr(), self.aenc.data_ptr(), AL, HID, 64, s)
        fvk.add_bf16_out(self.aenc.data_ptr(), self.ba_modA.data_ptr(), self.aenc.data_ptr(), AL * HID, s)
        fvk.add_bf16_out(self.aenc[AC:AL].data_ptr(), self.te.data_ptr(), self.aenc[AC:AL].data_ptr(), VL * HID, s)  # +timestep
        # assemble joint H: und=text, gen=[vision ; aenc]
        self.Hb[0:NU].copy_(self.text)
        self.Hb[NU:NU + AV].copy_(self.vision)
        self.Hb[NU + AV:NJ].copy_(self.aenc)
        # initial input norm per tower
        self._rms(self.Hb[0:NU], self.Wn[(0, "", "input_layernorm")], self.nrm[0:NU], NU, HID)
        self._rms(self.Hb[NU:NJ], self.Wn[(0, "_moe_gen", "input_layernorm")], self.nrm[NU:NJ], NG, HID)
        for li in range(NL):
            last = (li == NL - 1)
            # ---- QKV both towers into joint Qb/Kb/Vb ----
            for (suf, lo, hi, n) in (("", 0, NU, NU), ("_moe_gen", NU, NJ, NG)):
                if self.calib: self._rec(self.cal_q, (li, suf), self.nrm[lo:hi])
                self._proj(self.nrm[lo:hi], (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), HID)
                self._proj(self.nrm[lo:hi], (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), KV * D)
                self._proj(self.nrm[lo:hi], (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), KV * D)
                # qk-norm (rms over [n*heads,128])
                self._rms(self.Qb[lo:hi], self.Wn[(li, suf, "self_attn.q_norm")], self.Qb[lo:hi], n * H, D)
                self._rms(self.Kb[lo:hi], self.Wn[(li, suf, "self_attn.k_norm")], self.Kb[lo:hi], n * KV, D)
                # batched rope q+k (one launch)
                fvk.qwen36_partial_rope_qk_bf16(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
                    self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(),
                    self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(), n, H, KV, D, D, s)
            # ---- attention ----
            self._fa(self.Qb[0:NU], self.Kb[0:NU], self.Vb[0:NU], self.attn[0:NU].view(NU, H, D), NU, NU, True)
            self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)
            # ---- o_proj + residual+norm + ffn + residual+norm, per tower ----
            for (suf, lo, hi, n) in (("", 0, NU, NU), ("_moe_gen", NU, NJ, NG)):
                self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)
                self._radd_rms(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], self.nrm2[lo:hi], n)
                if self.calib: self._rec(self.cal_g, (li, suf), self.nrm2[lo:hi])
                self._proj(self.nrm2[lo:hi], (li, suf, "gate"), self.g[lo:hi], FF)
                self._proj(self.nrm2[lo:hi], (li, suf, "up"), self.u[lo:hi], FF)
                self._silu(self.g[lo:hi], self.u[lo:hi], self.act[lo:hi], n * FF)
                self._proj(self.act[lo:hi], (li, suf, "down"), self.dn[lo:hi], HID)
                if not last:
                    self._radd_rms(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], self.nrm[lo:hi], n)
                else:
                    fvk.residual_add(self.Hb[lo:hi].data_ptr(), self.dn[lo:hi].data_ptr(), n * HID, s)
        # final gen norm + llm2action
        self._rms(self.Hb[NU + AV + AC:NJ], self.norm_g, self.nrm[NU + AV + AC:NJ], VL, HID)
        self.gemm.bf16_nn(self.nrm[NU + AV + AC:NJ].data_ptr(), self.Wll.data_ptr(), self.vtmp.data_ptr(), VL, 64, HID, s)
        fvk.add_bf16_out(self.vtmp.data_ptr(), self.bllV.data_ptr(), self.vel.data_ptr(), VL * 64, s)
        return self.vel

    def set_input(self, x, te): self.x.copy_(x); self.te.copy_(te)
    def capture(self):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): self.forward()
        torch.cuda.current_stream().wait_stream(st)
        self.gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.gr): self.forward()
    def replay(self): self.gr.replay(); return self.vel


def te_table(timesteps_in, m):  # precompute timestep embeds OUTSIDE graph
    out = []
    tf = torch.exp(-math.log(10000) * torch.arange(0, 128, dtype=torch.float32, device=DEV) / 128)  # dim=256 -> 128 freqs
    for t in timesteps_in:
        args = t[:, None] * tf[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
        te = F.linear(F.silu(F.linear(emb, m.te0w, m.te0b)), m.te2w, m.te2b)
        out.append(te.to(BF))
    return out


def main():
    rf = safe_open(R, "pt", device=DEV); r = lambda k: rf.get_tensor(k)
    cosv = lambda a, b: F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()
    m = CosmosV2(); print("[init] all-kernel V2 ready")
    tes = te_table([r(f"s{st:02d}/timesteps_in") for st in range(10)], m)
    print("=== EAGER all-kernel velocity cos ===")
    for st in range(10):
        m.set_input(r(f"s{st:02d}/action2llm_in")[1:33], tes[st]); v = m.forward().clone(); torch.cuda.synchronize()
        print(f"  step {st}: cos = {cosv(v, r(f's{st:02d}/velocity')):.5f}")
    m.set_input(r("s00/action2llm_in")[1:33], tes[0]); m.capture(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): m.replay()
    torch.cuda.synchronize(); print(f"[timing] V2 all-kernel graphed: {(time.perf_counter()-t0)/20*1000:.2f} ms (nvfp4-torch was 32, bf16 48.6)")
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
    sched.set_timesteps(10, device=DEV, shift=5.0)
    lat = r("s00/action2llm_in")[1:33].float()
    for i, t in enumerate(sched.timesteps):
        m.set_input(lat.to(BF), tes[i]); v = m.replay().clone().float(); v[:, 29:] = 0
        lat = sched.step(v, t, lat, return_dict=True).prev_sample
    fin = r("once/final_action_norm")[1:33].float()
    print(f"[end-to-end] final active[:29] cos = {cosv(lat[:,:29], fin[:,:29]):.5f}")


if __name__ == "__main__":
    main()
