#!/usr/bin/env python3
"""V3 — extreme fusion: fuse activation-quant INTO the norm, and quantize the
shared activation ONCE (qkv reads one nvfp4 input, not 3). Per-tower quant count
drops 7 -> ~2/layer. Uses residual_add_rms_norm_to_nvfp4_swizzled_bf16 (residual
+ norm + nvfp4-quant in ONE launch) + rms_norm_to_nvfp4 for layer-0 input.
o_proj/down read non-norm inputs (attn/silu) so keep quant-then-gemm (or bf16).

  sudo docker exec cosmos-dev python3 /work/cosmos_v3.py
"""
import sys, time, torch, torch.nn.functional as F
from safetensors import safe_open
import flash_rt.flash_rt_kernels as fvk
from .cosmos_v2 import CosmosV2, te_table, sf_bytes, NL, H, KV, D, FF, HID, NU, NG, NJ, AV, AL, VL, AC, EPS, BF, DEV, R
from .fm_solvers_unipc import FlowUniPCMultistepScheduler


class CosmosV3(CosmosV2):
    def __init__(self, bf16_projs=()):
        super().__init__(bf16_projs=bf16_projs)
        # per-tower shared nvfp4 norm-output buffers (K=HID for both qkv & gate/up inputs)
        z8 = lambda r, c: (torch.empty(r, c // 2, dtype=torch.uint8, device=DEV), torch.zeros(sf_bytes(r, c), dtype=torch.uint8, device=DEV))
        self.pn = {"": z8(NU, HID), "_moe_gen": z8(NG, HID)}
        torch.cuda.synchronize()

    def _rms_q(self, x, w, pk, sf, rows):  # rms_norm -> nvfp4 (fused)
        fvk.rms_norm_to_nvfp4_swizzled_bf16(x.data_ptr(), w.data_ptr(), pk.data_ptr(), sf.data_ptr(), rows, HID, EPS, self._s())
    def _radd_rms_q(self, h, x, w, pk, sf, rows):  # residual+rms_norm -> nvfp4 (fused, h updated in place)
        fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(h.data_ptr(), x.data_ptr(), h.data_ptr(), w.data_ptr(), pk.data_ptr(), sf.data_ptr(), rows, HID, EPS, self._s())
    def _gemm_pre(self, pk, sf, key, out, M, N, K):  # fp4 GEMM from pre-quantized activation
        fvk.fp4_w4a16_gemm_sm120_bf16out(pk.data_ptr(), self.Wp[key].data_ptr(), out.data_ptr(), M, N, K, sf.data_ptr(), self.Wsf[key].data_ptr(), 1.0, self._s())

    def forward(self):
        s = self._s()
        self.gemm.bf16_nn(self.xfull.data_ptr(), self.Wa.data_ptr(), self.aenc.data_ptr(), AL, HID, 64, s)
        fvk.add_bf16_out(self.aenc.data_ptr(), self.ba_modA.data_ptr(), self.aenc.data_ptr(), AL * HID, s)
        fvk.add_bf16_out(self.aenc[AC:AL].data_ptr(), self.te.data_ptr(), self.aenc[AC:AL].data_ptr(), VL * HID, s)
        self.Hb[0:NU].copy_(self.text); self.Hb[NU:NU + AV].copy_(self.vision); self.Hb[NU + AV:NJ].copy_(self.aenc)
        # layer-0 input norm -> nvfp4 (fused) per tower
        self._rms_q(self.Hb[0:NU], self.Wn[(0, "", "input_layernorm")], *self.pn[""], NU)
        self._rms_q(self.Hb[NU:NJ], self.Wn[(0, "_moe_gen", "input_layernorm")], *self.pn["_moe_gen"], NG)
        for li in range(NL):
            last = (li == NL - 1)
            for (suf, lo, hi, n) in (("", 0, NU, NU), ("_moe_gen", NU, NJ, NG)):
                pk, sf = self.pn[suf]
                self._gemm_pre(pk, sf, (li, suf, "q_proj"), self.Qb[lo:hi].view(n, H * D), n, HID, HID)
                self._gemm_pre(pk, sf, (li, suf, "k_proj"), self.Kb[lo:hi].view(n, KV * D), n, KV * D, HID)
                self._gemm_pre(pk, sf, (li, suf, "v_proj"), self.Vb[lo:hi].view(n, KV * D), n, KV * D, HID)
                self._rms(self.Qb[lo:hi], self.Wn[(li, suf, "self_attn.q_norm")], self.Qb[lo:hi], n * H, D)
                self._rms(self.Kb[lo:hi], self.Wn[(li, suf, "self_attn.k_norm")], self.Kb[lo:hi], n * KV, D)
                fvk.qwen36_partial_rope_qk_bf16(self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(),
                    self.cos[lo:hi].data_ptr(), self.sin[lo:hi].data_ptr(),
                    self.Qb[lo:hi].data_ptr(), self.Kb[lo:hi].data_ptr(), n, H, KV, D, D, s)
            self._fa(self.Qb[0:NU], self.Kb[0:NU], self.Vb[0:NU], self.attn[0:NU].view(NU, H, D), NU, NU, True)
            self._fa(self.Qb[NU:NJ], self.Kb[0:NJ], self.Vb[0:NJ], self.attn[NU:NJ].view(NG, H, D), NG, NJ, False)
            for (suf, lo, hi, n) in (("", 0, NU, NU), ("_moe_gen", NU, NJ, NG)):
                pk, sf = self.pn[suf]
                self._proj(self.attn[lo:hi], (li, suf, "o_proj"), self.ob[lo:hi], HID)         # o: quant attn or bf16
                self._radd_rms_q(self.Hb[lo:hi], self.ob[lo:hi], self.Wn[(li, suf, "post_attention_layernorm")], pk, sf, n)  # H+=o ; nrm2->nvfp4
                self._gemm_pre(pk, sf, (li, suf, "gate"), self.g[lo:hi], n, FF, HID)
                self._gemm_pre(pk, sf, (li, suf, "up"), self.u[lo:hi], n, FF, HID)
                self._silu(self.g[lo:hi], self.u[lo:hi], self.act[lo:hi], n * FF)
                self._proj(self.act[lo:hi], (li, suf, "down"), self.dn[lo:hi], HID)             # down: quant act or bf16
                if not last:
                    self._radd_rms_q(self.Hb[lo:hi], self.dn[lo:hi], self.Wn[(li + 1, suf, "input_layernorm")], pk, sf, n)  # H+=down ; next nrm->nvfp4
                else:
                    fvk.residual_add(self.Hb[lo:hi].data_ptr(), self.dn[lo:hi].data_ptr(), n * HID, s)
        self._rms(self.Hb[NU + AV + AC:NJ], self.norm_g, self.nrm[NU + AV + AC:NJ], VL, HID)
        self.gemm.bf16_nn(self.nrm[NU + AV + AC:NJ].data_ptr(), self.Wll.data_ptr(), self.vtmp.data_ptr(), VL, 64, HID, s)
        fvk.add_bf16_out(self.vtmp.data_ptr(), self.bllV.data_ptr(), self.vel.data_ptr(), VL * 64, s)
        return self.vel


def main():
    rf = safe_open(R, "pt", device=DEV); r = lambda k: rf.get_tensor(k)
    cosv = lambda a, b: F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()
    keys = set(rf.keys())
    n_steps = sum(1 for k in keys if k.endswith("/timesteps_in"))
    final_key = "once/final_action_norm" if "once/final_action_norm" in keys else "once/final_action"
    active_dims = int(__import__("os").environ.get("COSMOS_ACTIVE_DIMS", "29" if final_key.endswith("_norm") else "9"))
    for projset in [(), ("down",)]:
        m = CosmosV3(bf16_projs=projset); tes = te_table([r(f"s{st:02d}/timesteps_in") for st in range(n_steps)], m)
        # quick eager cos
        m.set_input(r("s00/action2llm_in")[AC:AL], tes[0]); v = m.forward().clone(); torch.cuda.synchronize()
        c0 = cosv(v, r("s00/velocity"))
        m.set_input(r("s00/action2llm_in")[AC:AL], tes[0]); m.capture(); torch.cuda.synchronize()
        sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False); sched.set_timesteps(n_steps, device=DEV, shift=5.0)
        lat = r("s00/action2llm_in")[AC:AL].float()
        for i, t in enumerate(sched.timesteps):
            m.set_input(lat.to(BF), tes[i]); v = m.replay().clone().float(); v[:, active_dims:] = 0
            lat = sched.step(v, t, lat, return_dict=True).prev_sample
        fin = r(final_key)[AC:AL].float()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): m.replay()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"V3 fused bf16_projs={projset}: step0 cos={c0:.5f}  end-to-end cos={cosv(lat[:,:active_dims],fin[:,:active_dims]):.5f}  {ms:.2f} ms/step")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
