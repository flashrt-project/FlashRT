"""HyVLA forward path for RTX consumer Blackwell (SM120).

SM120 has the tcgen05 FP8 block-128 and NVFP4 tensor-core routes but not
the SM110 (Thor) FP8 megakernel path, and it does not build the SM80-family
INT8 W8A8 kernels the Orin SM87 path uses. This module owns the SM120 lowered
execution plan:

  * block-128-scaled FP8 GEMMs for the expert denoise tower and the VLM prefill
    tower (all K/N 128-aligned);
  * NVFP4 (W4A4) expert and ViT MLP weights with fused norm / residual-add /
    SiLU-Mul producers;
  * the repository-native sm120 FA2 attention backend for the denoise and ViT
    spatial attention, plus the fused FA2 gather->NVFP4-quant proj producer;
  * the ViT head-dim-72 native path (no 72->96 padding) and the im2col patch
    embed.

The shared ``pipeline_thor`` / ``pipeline_orin`` semantic pipeline keeps the
BF16 correctness path; this target subclasses ``HyVLAOrinBF16Pipeline`` and
supplies both the SM120 kernel bindings **and** the SM120 scheduling overrides
(``_block`` / ``prefill`` / ``merger_forward`` with the fused residual
producers, the ``_norm`` / ``_branch_gemm2`` hooks). No hardware fork lives in
the shared files. Reachable via
``flash_rt.load_model(ckpt, config="hyvla", framework="torch")`` on SM120.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:  # pragma: no cover - torch < 2.2
    sdpa_kernel = None
    SDPBackend = None

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.hyvla.pipeline_thor import HyVLAThorBF16Pipeline, _rot_half
from flash_rt.models.hyvla.pipeline_orin import (
    HyVLAOrinBF16Pipeline, _W8Int8, _vit_int8_linear)


def _rms_norm_torch(x, weight, eps):
    y = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def _rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    # Prefer the repository-native rms_norm kernel on CUDA bf16 (removes a
    # framework dispatch from the hot path); otherwise use F.rms_norm.
    if (x.is_cuda and x.dtype == torch.bfloat16 and weight is not None
            and weight.dtype == torch.bfloat16
            and len(normalized_shape) == 1
            and normalized_shape[0] == x.shape[-1]
            and hasattr(fvk, "rms_norm")):
        xc = x if x.is_contiguous() else x.contiguous()
        wc = weight if weight.is_contiguous() else weight.contiguous()
        rows = xc.numel() // xc.shape[-1]
        out = torch.empty_like(xc)
        fvk.rms_norm(xc.data_ptr(), wc.data_ptr(), out.data_ptr(),
                     rows, xc.shape[-1], eps,
                     torch.cuda.current_stream().cuda_stream)
        return out.reshape(x.shape)

    native = getattr(F, "rms_norm", None)
    if native is not None:
        return native(x, normalized_shape, weight, eps)
    return _rms_norm_torch(x, weight, eps)



class HyVLARTXBF16Pipeline(HyVLAOrinBF16Pipeline):
    """SM120 lowered execution plan for Hy-Embodied-0.5-VLA."""

    def _block_ffn8(self, hs, n_vis, w_text, w_vis, qk_w, mask, cos, sin,
                    kbuf, vbuf, off, ffnv, ffnt):
        S = hs.shape[1]
        D = hs.shape[2]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv

        hs_v = _rms_norm(hs[0, :n_vis], (D,), w_vis[4], self.rms_eps)
        hs_t = _rms_norm(hs[0, n_vis:], (D,), w_text[4], self.rms_eps)
        qkv = torch.cat([hs_v @ w_vis[0].t(), hs_t @ w_text[0].t()], 0)

        if getattr(self, "_fused_attn", False):
            q = torch.empty(1, nh, S, hd, dtype=torch.bfloat16, device=hs.device)
            self._rope_qknorm_kvwrite(qkv, cos, sin, qk_w, q, kbuf, vbuf, S, off)
        else:
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], -1)
            q = q.view(S, nh, hd).transpose(0, 1)[None]
            k = k.view(S, nkv, hd).transpose(0, 1)[None]
            v = v.view(S, nkv, hd).transpose(0, 1)[None]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            q = _rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = _rms_norm(k, (hd,), qk_w[1], self.rms_eps)
            if kbuf.shape[1] != nkv:
                r = kbuf.shape[1] // nkv
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            kbuf[:, :, off:off + S].copy_(k)
            vbuf[:, :, off:off + S].copy_(v)
        att = self._attn(q, kbuf[:, :, : off + S], vbuf[:, :, : off + S], mask)
        att = att.transpose(1, 2).reshape(1, S, self.q_dim)

        o_v = att[0, :n_vis] @ w_vis[1].t()
        o_t = att[0, n_vis:] @ w_text[1].t()
        hs = hs + torch.cat([o_v, o_t], 0)[None]

        hs_v = _rms_norm(hs[0, :n_vis], (D,), w_vis[5], self.rms_eps)
        hs_t = _rms_norm(hs[0, n_vis:], (D,), w_text[5], self.rms_eps)
        gu = torch.cat([self._int8_rowwise_gemm(hs_v, ffnv[0], ffnv[1]),
                        self._int8_rowwise_gemm(hs_t, ffnt[0], ffnt[1])], 0)
        act = _silu_mul_gu(gu)
        dn = torch.cat([self._int8_rowwise_gemm(act[:n_vis], ffnv[2], ffnv[3]),
                        self._int8_rowwise_gemm(act[n_vis:], ffnt[2], ffnt[3])], 0)
        return hs + dn[None]


    def _exp_block_r(self, hs, w, qk_w, mask, cos, sin, kbuf, vbuf, off,
                     fp8w, pending):
        S = hs.shape[1]
        D = hs.shape[2]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv

        if pending is None:
            hs_n = _rms_norm(hs, (D,), w[4], self.rms_eps)
        else:
            hs_n = self._res_add_rms_norm(hs, pending, w[4])

        qkv = self._int8_rowwise_gemm(hs_n[0], fp8w[0], fp8w[1])
        if getattr(self, "_fused_attn", False):
            q = torch.empty(1, nh, S, hd, dtype=torch.bfloat16, device=hs.device)
            self._rope_qknorm_kvwrite(qkv, cos, sin, qk_w, q, kbuf, vbuf, S, off)
        else:
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], -1)
            q = q.view(S, nh, hd).transpose(0, 1)[None]
            k = k.view(S, nkv, hd).transpose(0, 1)[None]
            v = v.view(S, nkv, hd).transpose(0, 1)[None]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            q = _rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = _rms_norm(k, (hd,), qk_w[1], self.rms_eps)
            if kbuf.shape[1] != nkv:
                r = kbuf.shape[1] // nkv
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            kbuf[:, :, off:off + S].copy_(k)
            vbuf[:, :, off:off + S].copy_(v)
        att = self._attn(q, kbuf[:, :, : off + S], vbuf[:, :, : off + S], mask)
        att = att.transpose(1, 2).reshape(1, S, self.q_dim)

        o = self._int8_rowwise_gemm(att[0], fp8w[2], fp8w[3])
        hs_n2 = self._res_add_rms_norm(hs, o[None], w[5])
        gu = self._int8_rowwise_gemm(hs_n2[0], fp8w[4], fp8w[5])
        act = _silu_mul_gu(gu)
        dn = self._int8_rowwise_gemm(act, fp8w[6], fp8w[7])
        return hs, dn[None]


    def _fp8_gemm(self, x, w8, ws):
        wmap = getattr(self, "_fp4_weight_map", None)
        if wmap is not None:
            entry = wmap.get(w8.data_ptr())
            if entry is not None:
                p, s, a = entry
                return self._nvfp4_gemm(x, p, s, a, None)
        if getattr(self, "_fp8b128", False):
            return self._fp8_block128_gemm(x, w8, ws)
        if not getattr(self, "_orin_int8", False):
            raise RuntimeError(
                "HyVLA Orin lower-precision GEMM requires enable_int8() or "
                "enable_fp8_block128().")
        return self._int8_rowwise_gemm(x, w8, ws)


    def _vit_mlp(self, x):
        if getattr(self, "_vit_fp4", False):
            bk, n, d = x.shape
            if getattr(self, "_vit_fp4_fc1_only", False):
                o1 = self._nvfp4_gemm(
                    x, self._vit_fc1_p4_cur, self._vit_fc1_s4_cur,
                    self._vit_fc1_a4_cur, None)
                return self._vit_mlp_fc2(
                    o1.reshape(-1, o1.shape[-1]), bk, n, d)
            o1 = self._nvfp4_gemm(x, self._vit_fc1_p4_cur, self._vit_fc1_s4_cur,
                                  self._vit_fc1_a4_cur, self._vit_fc1_b_pad4_cur)
            o1 = _gelu_erf(o1)
            return self._nvfp4_gemm(o1, self._vit_fc2_p4_cur,
                                    self._vit_fc2_s4_cur, self._vit_fc2_a4_cur,
                                    self._vit_fc2_b_cur)
        if isinstance(self._vit_fc1_w_cur, _W8Int8):
            x = _vit_int8_linear(x, self._vit_fc1_w_cur, self._vit_fc1_b_cur)
            x = _gelu_erf(x)
            return _vit_int8_linear(x, self._vit_fc2_w_cur, self._vit_fc2_b_cur)
        if (getattr(self, "_vit_f8", False)
                and getattr(self.W, "_vit_fc_fp8b128", False)
                and hasattr(fvk, "gelu_erf_bias_to_fp8_block128_bf16")):
            # fc1 GEMM (no bias) -> fused bias+erf-GELU+quant -> fc2 GEMM.
            # Removes the standalone out+bias, gelu and quant kernels.
            bk, n, d = x.shape
            out = self._fp8_block128_gemm(x, self._vit_fc1_w8c,
                                          self._vit_fc1_wsc)
            return self._vit_mlp_fc2(out, bk, n, d)
        return super()._vit_mlp(x)


    def _vit_qkv(self, h):
        if getattr(self, "_vit_fp4_qkv", False):
            bias = self._vit_qkv_b_pad_cur if self._vit_qkv_b_pad_cur is not None \
                else self._vit_qkv_b_cur
            qkv = self._nvfp4_gemm(h, self._vit_qkv_p4_cur, self._vit_qkv_s4_cur,
                                   self._vit_qkv_a4_cur, bias)
        elif isinstance(self._vit_qkv_w_cur, _W8Int8):
            qkv = _vit_int8_linear(h, self._vit_qkv_w_cur, self._vit_qkv_b_cur)
        elif getattr(self, "_vit_f8b128", False):
            bias = self._vit_qkv_b_pad_cur if self._vit_qkv_b_pad_cur is not None \
                else self._vit_qkv_b_cur
            qkv = self._fp8_block128_gemm_bias(h, self._vit_qkv_w8_cur,
                                               self._vit_qkv_ws_cur, bias)
        else:
            return super()._vit_qkv(h)
        bk, N, _ = h.shape
        hd_pad = getattr(self, "_vit_hd_pad", self.vit_hd)
        qkv = qkv.reshape(bk, N, 3, self.vit_heads, hd_pad).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]


    def _vit_spatial_attn(self, q, k, v):
        bk, _, N, _ = q.shape
        if (getattr(self, "_vit_hd_pad", None) is not None
                and getattr(self, "_fa2", None) is not None
                and getattr(self, "_vit_fp4_proj", False)
                and hasattr(fvk, "hyvla_vit_proj_gather_nvfp4_swizzled_bf16")):
            # Fused: FA2 -> slice/transpose + NVFP4 quant -> proj GEMM.
            o = self._vit_spatial_attn_fa2(q, k, v)
            ap, asf = self._vit_proj_gather_nvfp4(o)
            return self._nvfp4_gemm_prequant(
                ap, asf, self._vit_proj_p4_cur, self._vit_proj_s4_cur,
                self._vit_proj_a4_cur, self._vit_proj_b_cur)
        if (getattr(self, "_vit_hd_pad", None) is not None
                and getattr(self, "_fa2", None) is not None):
            # Native head dim (72 with HYVLA_VIT_HD72) or the 72->96 padding
            # both run the repository-native sm120 FA2 backend
            # (flash_rt_fa2.fwd_bf16_tile) instead of the framework SDPA.
            out = self._vit_spatial_attn_fa2(q, k, v)
            out = out[..., :self.vit_hd]
        elif getattr(self, "_vit_eff_sdpa", False):
            # The q/k/v slices out of the packed QKV GEMM are strided; the
            # flash backend force-copies them contiguous (4 copy_ per block,
            # ~25% of ViT time). The memory-efficient backend accepts the
            # strides directly and reads the strided layout natively.
            if sdpa_kernel is not None:
                ctx = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
            else:
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=False, enable_mem_efficient=True)
            with ctx:
                out = F.scaled_dot_product_attention(q, k, v, scale=self.vit_scale)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.vit_scale)
        out = out.transpose(1, 2).reshape(bk, N, -1)
        if getattr(self, "_vit_fp4_proj", False):
            return self._nvfp4_gemm(out, self._vit_proj_p4_cur,
                                    self._vit_proj_s4_cur,
                                    self._vit_proj_a4_cur,
                                    self._vit_proj_b_cur)
        if isinstance(self._vit_proj_w_cur, _W8Int8):
            return _vit_int8_linear(out, self._vit_proj_w_cur, self._vit_proj_b_cur)
        if getattr(self, "_vit_f8b128", False):
            return self._fp8_block128_gemm_bias(out, self._vit_proj_w8_cur,
                                                self._vit_proj_ws_cur,
                                                self._vit_proj_b_cur)
        return F.linear(out, self._vit_proj_w_cur, self._vit_proj_b_cur)


    @torch.no_grad()
    def denoise(self, state, x_t, time_embs, smask, scos, ssin,
                kbuf, vbuf, S_p, num_steps=10):
        W = self.W
        if getattr(self, "_fp8b128", False) and getattr(W, "_exp_fp8_ready", False):
            return self._denoise_fp8b128(state, x_t, time_embs, smask, scos,
                                         ssin, kbuf, vbuf, S_p, num_steps)
        if not (getattr(self, "_orin_int8", False)
                and getattr(W, "_exp_fp8_ready", False)):
            return super().denoise(state, x_t, time_embs, smask, scos, ssin,
                                   kbuf, vbuf, S_p, num_steps=num_steps)
        S_s = 1 + x_t.shape[1]
        dt = -1.0 / num_steps
        state_emb = (F.linear(state.to(torch.bfloat16), W._state_w, W._state_b))[:, None]
        for s in range(num_steps):
            action_emb = F.linear(x_t.to(torch.bfloat16), W._ain_w, W._ain_b)
            t_emb = time_embs[s].expand_as(action_emb)
            ate = torch.cat([action_emb, t_emb], 2)
            ate = F.linear(ate, W._atmlp_in_w, W._atmlp_in_b)
            ate = _silu(ate)
            ate = F.linear(ate, W._atmlp_out_w, W._atmlp_out_b)
            hs = torch.cat([state_emb, ate], 1)
            pending = None
            for li in range(32):
                exp = self._exp_w(li)
                qk = (W._qk_norm_q[li], W._qk_norm_k[li])
                hs, pending = self._exp_block_r(
                    hs, exp, qk, smask, scos, ssin,
                    kbuf[li], vbuf[li], S_p, self._exp_w_fp8(li), pending)
            hs_n = self._res_add_rms_norm(hs, pending, W._exp_final_norm_w)
            v_t = F.linear(hs_n[:, -x_t.shape[1]:], W._aout_w, W._aout_b)
            x_t.add_(dt * v_t.to(x_t.dtype))
        return x_t


    def enable_fp4(self):
        # SM120 (RTX Blackwell) native FP4 via flash_rt_kernels' NVFP4 GEMM
        # (see _fp4_gemm_f4 below); no flash_rt_fp4 add-on module needed.
        self._fp4 = True


    @torch.no_grad()
    def vit_forward(self, imgs):
        if not getattr(self, "_vit_fuse_ln", False):
            return super().vit_forward(imgs)
        W = self.W
        self._vit_ln_fp8 = (
            getattr(self, "_fp8b128", False)
            and getattr(W, "_vit_qkv_proj_fp8b128_ready", False)
            and getattr(W, "_vit_fc_fp8b128", False)
            and hasattr(fvk, "hyvla_vit_add_layer_norm_to_fp8_block128_bf16"))
        if getattr(self, "_vit_fp4", False):
            # NVFP4 MLP test: the fused LN->FP8 producer feeds an FP8 fc1
            # GEMM, so bypass it and use the bf16 LN + FP4 MLP path.
            self._vit_ln_fp8 = False
        num_cam, K = imgs.shape[0], imgs.shape[1]
        bk = num_cam * K
        x = imgs.reshape(bk, 3, 224, 224)
        if getattr(self, "_vit_patch_gemm", False):
            # Non-overlapping strided conv == im2col matmul. cudnn's
            # precomputed_convolve for this in_channels=3 / stride=16 shape runs
            # at ~12 TFLOP/s; the equivalent bf16 GEMM (M=bk*14*14, K=768,
            # N=1152) is ~3x faster and bit-compatible to ~1 bf16 ULP.
            C = W._vit_patch_w.shape[1]
            P = W._vit_patch_w.shape[-1]
            d = W._vit_patch_w.shape[0]
            hh, ww = x.shape[-2] // P, x.shape[-1] // P
            patch = x.reshape(bk, C, hh, P, ww, P).permute(
                0, 2, 4, 1, 3, 5).reshape(bk * hh * ww, C * P * P)
            x = (patch @ W._vit_patch_w.reshape(d, C * P * P).t())
            x = x.reshape(bk, hh, ww, d).permute(0, 3, 1, 2).contiguous()
        else:
            x = F.conv2d(x, W._vit_patch_w, None, stride=16)
        hh = ww = x.shape[-1]
        n = hh * ww
        d = x.shape[1]
        # Native patch-embed bias add (bias-free conv above).
        fvk.hyvla_vit_patch_bias_bf16(
            x.data_ptr(), W._vit_patch_b.to(x.dtype).data_ptr(), bk, d, hh, ww,
            torch.cuda.current_stream().cuda_stream)
        xb = x.flatten(2)                       # (bk, d, n) contiguous
        pe = self._vit_pos_embed_rescale(hh, ww, xb.dtype)  # (1, n, d)
        out = torch.empty(bk, n, d, dtype=xb.dtype, device=xb.device)
        fvk.hyvla_vit_pos_add_bf16(
            xb.data_ptr(), pe.reshape(n, d).data_ptr(), out.data_ptr(),
            bk, n, d, torch.cuda.current_stream().cuda_stream)
        x = out
        last_st = max(self.vit_spacetime_ids) if K > 1 else -1
        sliced = False
        pending = None
        for li in range(27):
            Wcur = self.W
            self._vit_qkv_w_cur = Wcur._vit_qkv_w[li]; self._vit_qkv_b_cur = Wcur._vit_qkv_b[li]
            self._vit_proj_w_cur = Wcur._vit_proj_w[li]; self._vit_proj_b_cur = Wcur._vit_proj_b[li]
            self._vit_fc1_w_cur = Wcur._vit_fc1_w[li]; self._vit_fc1_b_cur = Wcur._vit_fc1_b[li]
            self._vit_fc2_w_cur = Wcur._vit_fc2_w[li]; self._vit_fc2_b_cur = Wcur._vit_fc2_b[li]
            if getattr(self, "_vit_fp4", False):
                self._vit_fc1_p4_cur = Wcur._vit_fc1_p4[li]
                self._vit_fc1_s4_cur = Wcur._vit_fc1_s4[li]
                self._vit_fc1_a4_cur = Wcur._vit_fc1_a4[li]
                self._vit_fc1_b_pad4_cur = Wcur._vit_fc1_b_pad4[li]
                self._vit_fc2_p4_cur = Wcur._vit_fc2_p4[li]
                self._vit_fc2_s4_cur = Wcur._vit_fc2_s4[li]
                self._vit_fc2_a4_cur = Wcur._vit_fc2_a4[li]
                self._vit_proj_p4_cur = Wcur._vit_proj_p4[li]
                self._vit_proj_s4_cur = Wcur._vit_proj_s4[li]
                self._vit_proj_a4_cur = Wcur._vit_proj_a4[li]
                self._vit_qkv_p4_cur = Wcur._vit_qkv_p4[li]
                self._vit_qkv_s4_cur = Wcur._vit_qkv_s4[li]
                self._vit_qkv_a4_cur = Wcur._vit_qkv_a4[li]
                if getattr(Wcur, "_vit_fc2_awq", False):
                    self._vit_fc2_inv_s_cur = Wcur._vit_fc2_inv_s[li]
            self._vit_f8 = self._fp8 and getattr(Wcur, "_vit_fp8_ready", False)
            if self._vit_f8:
                self._vit_qkv_w8c = Wcur._vit_qkv_w8[li]; self._vit_qkv_wsc = Wcur._vit_qkv_ws[li]
                self._vit_proj_w8c = Wcur._vit_proj_w8[li]; self._vit_proj_wsc = Wcur._vit_proj_ws[li]
                self._vit_fc1_w8c = Wcur._vit_fc1_w8[li]; self._vit_fc1_wsc = Wcur._vit_fc1_ws[li]
                self._vit_fc2_w8c = Wcur._vit_fc2_w8[li]; self._vit_fc2_wsc = Wcur._vit_fc2_ws[li]
            self._vit_f8b128 = (getattr(self, "_fp8b128", False)
                                and getattr(Wcur, "_vit_qkv_proj_fp8b128_ready", False))
            if self._vit_f8b128:
                self._vit_qkv_w8_cur = Wcur._vit_qkv_w8[li]; self._vit_qkv_ws_cur = Wcur._vit_qkv_ws[li]
                self._vit_proj_w8_cur = Wcur._vit_proj_w8[li]; self._vit_proj_ws_cur = Wcur._vit_proj_ws[li]
                if hasattr(Wcur, "_vit_qkv_b_pad"):
                    self._vit_qkv_b_pad_cur = Wcur._vit_qkv_b_pad[li]
                    self._vit_hd_pad = getattr(Wcur, "_vit_hd_pad", self.vit_hd)
                else:
                    self._vit_qkv_b_pad_cur = None
            else:
                self._vit_qkv_b_pad_cur = None
            if getattr(Wcur, "_vit_fc_fp8b128", False):
                self._vit_fc1_b_cur = Wcur._vit_fc1_b_pad[li]
            ln1w, ln1b = Wcur._vit_ln1_w[li], Wcur._vit_ln1_b[li]
            ln2w, ln2b = Wcur._vit_ln2_w[li], Wcur._vit_ln2_b[li]

            if li in self.vit_spacetime_ids and K > 1:
                b, kf = bk // K, K
                pe = self._vit_time_pe(kf, x.device, x.dtype)
                if getattr(self, "_native_spacetime_ln", False):
                    h = torch.empty_like(x)
                    xp = pending.data_ptr() if pending is not None else 0
                    fvk.hyvla_vit_res_add_ln_time_bf16(
                        x.data_ptr(), xp, pe.data_ptr(), ln1w.data_ptr(),
                        ln1b.data_ptr(), h.data_ptr(), bk * n, d, n, kf,
                        self.vit_eps, torch.cuda.current_stream().cuda_stream)
                    pending = None
                else:
                    if pending is not None:
                        x = x + pending
                        pending = None
                    h = F.layer_norm(x.view(b, kf, n, d) + pe.view(1, kf, 1, d),
                                     (d,), ln1w, ln1b, self.vit_eps).view(bk, n, d)
                q, k, v = self._vit_qkv(h)
                v = self._vit_time_mix(q, k, v, b, kf)
                attn_out = self._vit_spatial_attn(q, k, v)
            else:
                if pending is not None:
                    if self._vit_ln_fp8:
                        a8, ascale = self._vit_add_ln_to_fp8(x, pending, ln1w, ln1b)
                        q, k, v = self._vit_qkv_prequant(a8, ascale, x)
                    elif getattr(self, "_vit_fp4", False):
                        ap, asf = self._vit_add_ln_to_nvfp4(x, pending, ln1w, ln1b)
                        q, k, v = self._vit_qkv_prequant_fp4(ap, asf, x)
                    else:
                        h = self._vit_add_ln(x, pending, ln1w, ln1b)
                        q, k, v = self._vit_qkv(h)
                else:
                    h = self._vit_ln(x, ln1w, ln1b)
                    q, k, v = self._vit_qkv(h)
                attn_out = self._vit_spatial_attn(q, k, v)

            if self._vit_ln_fp8:
                a8_2, ascale_2 = self._vit_add_ln_to_fp8(x, attn_out, ln2w, ln2b)
                pending = self._vit_mlp_prequant(a8_2, ascale_2, x)
            elif getattr(self, "_vit_fp4", False):
                ap2, asf2 = self._vit_add_ln_to_nvfp4(x, attn_out, ln2w, ln2b)
                pending = self._vit_mlp_prequant_fp4(
                    ap2, asf2, x.shape[0], x.shape[1], x.shape[2])
            else:
                h2 = self._vit_add_ln(x, attn_out, ln2w, ln2b)
                pending = self._vit_mlp(h2)
            if li == last_st and li < 26:
                # Fused residual add + last-frame select + contiguous copy.
                out = torch.empty(num_cam, n, d, dtype=x.dtype,
                                  device=x.device)
                fvk.hyvla_vit_tail_slice_bf16(
                    x.data_ptr(), pending.data_ptr(), out.data_ptr(),
                    num_cam, K, n, d, torch.cuda.current_stream().cuda_stream)
                x = out
                pending = None
                sliced = True
        if pending is not None:
            fvk.residual_add(x.data_ptr(), pending.data_ptr(), x.numel(),
                             torch.cuda.current_stream().cuda_stream)
        if not sliced:
            x = x.view(num_cam, K, n, d)[:, -1]
        return x


    def _attn(self, q, k, v, mask):
        if (getattr(self, "_native_prefill_attn", False)
                and mask is not None and mask.shape[-1] == q.shape[2]
                and k.shape[1] == q.shape[1]):
            return self._prefill_attn_native(q, k, v, mask)
        return super()._attn(q, k, v, mask)


    def _bf16_gemm_bias(self, x, w, bias):
        """Native FlashRT BF16 GEMM (...,K)->(...,N) + bias for the action MLP
        (replaces F.linear so the hot path has no framework GEMM)."""
        orig = x.shape
        xc = x.reshape(-1, orig[-1]).contiguous()
        M, K = xc.shape
        N = w.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        fvk.tq_cutlass_bf16_gemm(xc.data_ptr(), w.data_ptr(), out.data_ptr(),
                                 M, N, K, torch.cuda.current_stream().cuda_stream)
        if bias is not None:
            fvk.add_bias_bf16(out.data_ptr(), bias.data_ptr(), M, N,
                              torch.cuda.current_stream().cuda_stream)
        return out.reshape(*orig[:-1], N)


    def _bf16_gemm_bias_out(self, x, w, bias, out):
        """Native BF16 GEMM + bias writing into a caller-provided contiguous
        (M, N) bf16 buffer (removes the concat that assembled the expert input)."""
        xc = x.reshape(-1, x.shape[-1]).contiguous()
        M, K = xc.shape
        N = w.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        fvk.tq_cutlass_bf16_gemm(xc.data_ptr(), w.data_ptr(), out.data_ptr(),
                                 M, N, K, st)
        if bias is not None:
            fvk.add_bias_bf16(out.data_ptr(), bias.data_ptr(), M, N, st)
        return out


    def _branch_gemm2(self, x_v, w_v, x_t, w_t, kind, N=None, K=None):
        """Two-branch MoT GEMM writing both branches into one (S,N) buffer
        (no torch.cat) for the NVFP4 prefill path. Accepts either bf16
        activations or prequantized (packed, sfa) tuples."""
        if kind == "fp8":
            wmap = getattr(self, "_fp4_weight_map", None)
            ev = wmap.get(w_v[0].data_ptr()) if wmap is not None else None
            et = wmap.get(w_t[0].data_ptr()) if wmap is not None else None
            if ev is None or et is None:
                return self._branch_gemm2_plain(x_v, w_v, x_t, w_t, kind, N, K)
            pv, sv, av = ev
            pt, st_, at = et
            Nout = pv.shape[0]
        elif kind == "fp4":
            pv, sv = w_v[0], w_v[1]
            pt, st_ = w_t[0], w_t[1]
            amap = getattr(self, "_fp4_alpha", {})
            av = amap.get(pv.data_ptr(), 1.0)
            at = amap.get(pt.data_ptr(), 1.0)
            Nout = N
        else:
            return self._branch_gemm2_plain(x_v, w_v, x_t, w_t, kind, N, K)
        pre = isinstance(x_v, tuple)
        if pre:
            av8, asf_v = x_v
            at8, asf_t = x_t
            M_v, M_t = av8.shape[0], at8.shape[0]
        else:
            M_v, M_t = x_v.shape[0], x_t.shape[0]
        out = torch.empty(M_v + M_t, Nout, dtype=torch.bfloat16,
                          device=x_v[0].device if pre else x_v.device)
        if pre:
            self._nvfp4_gemm_out_prequant(av8, asf_v, pv, sv, av, out[:M_v])
            self._nvfp4_gemm_out_prequant(at8, asf_t, pt, st_, at, out[M_v:])
        else:
            self._nvfp4_gemm_out(x_v, pv, sv, av, out[:M_v])
            self._nvfp4_gemm_out(x_t, pt, st_, at, out[M_v:])
        return out

    def _branch_gemm2_plain(self, x_v, w_v, x_t, w_t, kind, N=None, K=None):
        """Self-contained two-branch MoT GEMM fallback: dispatch each branch
        through the native ``_fp8_gemm`` / ``_fp4_gemm_f4`` slot and ``cat``.
        Used when the NVFP4 weight map is absent (prefill NVFP4 disabled) or an
        unknown ``kind`` is requested — the SM120 pipeline must not depend on a
        base hook that lives in another hardware's file."""
        if kind == "fp8":
            v = self._fp8_gemm(x_v, w_v[0], w_v[1])
            t = self._fp8_gemm(x_t, w_t[0], w_t[1])
        else:
            v = self._fp4_gemm_f4(x_v, w_v[0], w_v[1], N, K)
            t = self._fp4_gemm_f4(x_t, w_t[0], w_t[1], N, K)
        return torch.cat([v, t], 0)


    @torch.no_grad()
    def _denoise_fp8b128(self, state, x_t, time_embs, smask, scos, ssin,
                         kbuf, vbuf, S_p, num_steps=10):
        W = self.W
        S_s = 1 + x_t.shape[1]
        dt = -1.0 / num_steps
        # Preallocated expert-tower input buffer (1, S_s, D_exp): row 0 is the
        # state embedding, rows 1: are the action tokens. Writing both directly
        # via output-buffer GEMMs removes the per-step framework concat.
        state_bf16 = state.to(torch.bfloat16)
        D_exp = W._state_w.shape[0]
        hs_buf = getattr(self, "_denoise_hs_buf", None)
        if hs_buf is None or hs_buf.shape[1] != S_s or hs_buf.shape[2] != D_exp:
            hs_buf = torch.empty(1, S_s, D_exp, dtype=torch.bfloat16,
                                 device=x_t.device)
            self._denoise_hs_buf = hs_buf
        # bf16 mirror of the flow state, maintained by the fused Euler kernel.
        x_bf16 = getattr(self, "_denoise_x_bf16", None)
        if x_bf16 is None or x_bf16.numel() != x_t.numel():
            x_bf16 = torch.empty_like(x_t, dtype=torch.bfloat16)
            self._denoise_x_bf16 = x_bf16
        st = torch.cuda.current_stream().cuda_stream
        fvk.hyvla_euler_update_bf16_fp32(
            x_t.data_ptr(), 0, x_bf16.data_ptr(), 0.0, x_t.numel(), st)
        # Fold the per-step time term of the action-MLP input GEMM into the bias:
        #   cat([action_emb, t_emb]) @ W_in.T + b
        #     == action_emb @ W_in[:, :D].T + (time_embs @ W_in[:, D:].T) + b
        # removing one framework cat per denoise step with no extra per-step op.
        d_act = W._ain_w.shape[0]
        Wl = getattr(self, "_atmlp_in_l", None)
        if Wl is None:
            Wl = W._atmlp_in_w[:, :d_act].contiguous()
            self._atmlp_in_l = Wl
            self._atmlp_in_r = W._atmlp_in_w[:, d_act:].contiguous()
        tproj = time_embs @ self._atmlp_in_r.t()          # (num_steps, 1, H)
        # b_all is decision-invariant (time_embs and the weights are static):
        # precompute once so the per-decision broadcast add is not in graph.
        b_all = getattr(self, "_atmlp_b_all_cache", None)
        if b_all is None:
            b_all = W._atmlp_in_b[None] + tproj[:, 0]     # (num_steps, H)
            self._atmlp_b_all_cache = b_all
        for s in range(num_steps):
            action_emb = self._bf16_gemm_bias(x_bf16, W._ain_w, W._ain_b)
            ate = self._bf16_gemm_bias(action_emb, Wl, b_all[s])
            ate = _silu(ate)
            self._bf16_gemm_bias_out(ate, W._atmlp_out_w, W._atmlp_out_b,
                                     hs_buf[0, 1:])
            self._bf16_gemm_bias_out(state_bf16, W._state_w, W._state_b,
                                     hs_buf[0, :1])
            hs = hs_buf
            pending = None
            _fp4 = getattr(self, "_exp_fp4", False)
            for li in range(32):
                exp = self._exp_w(li)
                qk = (W._qk_norm_q[li], W._qk_norm_k[li])
                if _fp4:
                    hs, pending = self._exp_block_fp4(
                        hs, exp, qk, smask, scos, ssin,
                        kbuf[li], vbuf[li], S_p, self._exp_w_fp4(li), pending,
                        self._exp_w_fp8(li))
                else:
                    hs, pending = self._exp_block_fp8b128(
                        hs, exp, qk, smask, scos, ssin,
                        kbuf[li], vbuf[li], S_p, self._exp_w_fp8(li), pending)
            hs_n = self._res_add_rms_norm(hs, pending, W._exp_final_norm_w)
            v_t = self._bf16_gemm_bias(hs_n[:, -x_t.shape[1]:], W._aout_w, W._aout_b)
            # Fused Euler update x_t += dt * v_t (one native kernel; replaces
            # v_t.to(fp32), dt*v and x_t.add_ framework launches).
            fvk.hyvla_euler_update_bf16_fp32(
                x_t.data_ptr(), v_t.data_ptr(), x_bf16.data_ptr(), dt,
                x_t.numel(), st)
        return x_t


    def _exp_block_fp4(self, hs, w, qk_w, mask, cos, sin, kbuf, vbuf, off,
                       fp4w, pending, fp8w=None):
        """NVFP4 (W4A4) expert tower block: mirrors ``_exp_block_fp8b128`` with
        the NVFP4 fused producers + M=41 prequant NVFP4 GEMMs and the fused
        FA2-output gather + NVFP4 quant for the o-projection input.

        When ``fp8w`` is supplied and ``_exp_od_fp8`` is set, the narrow-N o/dn
        GEMMs (N=1024, K=2048) stay on the FP8 block-128 split-K path, which
        beats the 8-CTA NVFP4 persistent GEMM at these shapes (mixed route)."""
        S = hs.shape[1]
        D = hs.shape[2]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv
        _fq = getattr(self, "_fused_quant", False)
        _mixed = fp8w is not None and getattr(self, "_exp_od_fp8", False)

        if _fq:
            if pending is None:
                ap, asf = self._norm_to_nvfp4(hs, w[4])
            else:
                ap, asf = self._res_add_norm_to_nvfp4(hs, pending, w[4], hs)
            qkv = self._nvfp4_gemm_prequant(ap, asf, fp4w[0], fp4w[1],
                                            fp4w[2], None)
        else:
            if pending is None:
                hs_n = _rms_norm(hs, (D,), w[4], self.rms_eps)
            else:
                hs_n = self._res_add_rms_norm(hs, pending, w[4])
            qkv = self._nvfp4_gemm(hs_n[0], fp4w[0], fp4w[1], fp4w[2], None)
        if getattr(self, "_fused_attn", False):
            q = torch.empty(1, nh, S, hd, dtype=torch.bfloat16, device=hs.device)
            if getattr(self, "_fused_prepare", False):
                _qb = self._fa2_buffers(q, off, S)[1]
                self._rope_qknorm_kvwrite_qb(qkv, cos, sin, qk_w, _qb,
                                             kbuf, vbuf, S, off)
            else:
                _qb = None
                self._rope_qknorm_kvwrite(qkv, cos, sin, qk_w, q, kbuf, vbuf,
                                          S, off)
        else:
            _qb = None
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], -1)
            q = q.view(S, nh, hd).transpose(0, 1)[None]
            k = k.view(S, nkv, hd).transpose(0, 1)[None]
            v = v.view(S, nkv, hd).transpose(0, 1)[None]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            q = _rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = _rms_norm(k, (hd,), qk_w[1], self.rms_eps)
            if kbuf.shape[1] != nkv:
                r = kbuf.shape[1] // nkv
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            kbuf[:, :, off:off + S].copy_(k)
            vbuf[:, :, off:off + S].copy_(v)
        if getattr(self, "_fa2", None) is not None:
            att = self._fa2_denoise_attn(q, kbuf[:, :, : off + S],
                                         vbuf[:, :, : off + S], off,
                                         force_fp8=_mixed, qb_pre=_qb)
        else:
            att = self._attn(q, kbuf[:, :, : off + S], vbuf[:, :, : off + S], mask)
        if _mixed:
            if isinstance(att, tuple):
                o = self._od_b128_splitk_prequant(att[0], att[1],
                                                  fp8w[2], fp8w[3])
            else:
                att = att.transpose(1, 2).reshape(1, S, self.q_dim)
                o = self._od_b128_splitk_gemm(att[0], fp8w[2], fp8w[3])
        elif isinstance(att, tuple):
            o = self._nvfp4_gemm_prequant_od(att[0], att[1], fp4w[3], fp4w[4],
                                             fp4w[5])
        else:
            att = att.transpose(1, 2).reshape(1, S, self.q_dim)
            o = self._nvfp4_gemm(att[0], fp4w[3], fp4w[4], fp4w[5], None)
        if _fq:
            a8_2, asc_2 = self._res_add_norm_to_nvfp4(hs, o[None], w[5], hs)
            gu = self._nvfp4_gemm_prequant(a8_2, asc_2, fp4w[6], fp4w[7],
                                           fp4w[8], None)
            if _mixed:
                act8, act_s = self._silu_mul_to_fp8(gu)
                dn = self._od_b128_splitk_prequant(act8, act_s,
                                                   fp8w[6], fp8w[7])
            else:
                act_p, act_s = self._silu_mul_to_nvfp4(gu)
                dn = self._nvfp4_gemm_prequant_od(act_p, act_s, fp4w[9], fp4w[10],
                                                  fp4w[11])
        else:
            hs_n2 = self._res_add_rms_norm(hs, o[None], w[5])
            gu = self._nvfp4_gemm(hs_n2[0], fp4w[6], fp4w[7], fp4w[8], None)
            if _mixed:
                act8, act_s = self._silu_mul_to_fp8(gu)
                dn = self._od_b128_splitk_prequant(act8, act_s,
                                                   fp8w[6], fp8w[7])
            else:
                act = _silu_mul_gu(gu)
                dn = self._nvfp4_gemm(act, fp4w[9], fp4w[10], fp4w[11], None)
        return hs, dn[None]


    def _exp_block_fp8b128(self, hs, w, qk_w, mask, cos, sin, kbuf, vbuf, off,
                           fp8w, pending):
        S = hs.shape[1]
        D = hs.shape[2]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv

        _fq = getattr(self, "_fused_quant", False)

        if _fq:
            if pending is None:
                a8, asc = self._rms_norm_to_fp8(hs, w[4])
            else:
                a8, asc = self._res_add_rms_norm_to_fp8(hs, pending, w[4])
            qkv = self._fp8_block128_gemm_prequant(a8, asc, fp8w[0], fp8w[1])
        else:
            if pending is None:
                hs_n = _rms_norm(hs, (D,), w[4], self.rms_eps)
            else:
                hs_n = self._res_add_rms_norm(hs, pending, w[4])
            qkv = self._fp8_block128_gemm(hs_n[0], fp8w[0], fp8w[1])
        if getattr(self, "_fused_attn", False):
            q = torch.empty(1, nh, S, hd, dtype=torch.bfloat16, device=hs.device)
            self._rope_qknorm_kvwrite(qkv, cos, sin, qk_w, q, kbuf, vbuf, S, off)
        else:
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], -1)
            q = q.view(S, nh, hd).transpose(0, 1)[None]
            k = k.view(S, nkv, hd).transpose(0, 1)[None]
            v = v.view(S, nkv, hd).transpose(0, 1)[None]
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            q = _rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = _rms_norm(k, (hd,), qk_w[1], self.rms_eps)
            if kbuf.shape[1] != nkv:
                r = kbuf.shape[1] // nkv
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            kbuf[:, :, off:off + S].copy_(k)
            vbuf[:, :, off:off + S].copy_(v)
        if getattr(self, "_fa2", None) is not None:
            att = self._fa2_denoise_attn(q, kbuf[:, :, : off + S],
                                         vbuf[:, :, : off + S], off)
        else:
            att = self._attn(q, kbuf[:, :, : off + S], vbuf[:, :, : off + S], mask)
        _sk = getattr(self, "_exp_od_b128", False)
        if isinstance(att, tuple):
            if _sk:
                o = self._od_b128_splitk_prequant(att[0], att[1], fp8w[2], fp8w[3])
            else:
                o = self._fp8_block128_gemm_prequant(att[0], att[1], fp8w[2], fp8w[3])
        else:
            att = att.transpose(1, 2).reshape(1, S, self.q_dim)
            if _sk:
                o = self._od_b128_splitk_gemm(att[0], fp8w[2], fp8w[3])
            else:
                o = self._fp8_block128_gemm(att[0], fp8w[2], fp8w[3])
        if _fq:
            a8_2, asc_2 = self._res_add_rms_norm_to_fp8(hs, o[None], w[5])
            gu = self._fp8_block128_gemm_prequant(a8_2, asc_2, fp8w[4], fp8w[5])
            if _sk:
                act = _silu_mul_gu(gu)
                dn = self._od_b128_splitk_gemm(act, fp8w[6], fp8w[7])
            else:
                act8, act_s = self._silu_mul_to_fp8(gu)
                dn = self._fp8_block128_gemm_prequant(act8, act_s, fp8w[6], fp8w[7])
        else:
            hs_n2 = self._res_add_rms_norm(hs, o[None], w[5])
            gu = self._fp8_block128_gemm(hs_n2[0], fp8w[4], fp8w[5])
            act = _silu_mul_gu(gu)
            if _sk:
                dn = self._od_b128_splitk_gemm(act, fp8w[6], fp8w[7])
            else:
                dn = self._fp8_block128_gemm(act, fp8w[6], fp8w[7])
        return hs, dn[None]


    def _exp_w_fp4(self, li):
        """NVFP4 expert weights for layer ``li``.

        Returns (qkv_p, qkv_s, qkv_a, o_p, o_s, o_a, gu_p, gu_s, gu_a,
        dn_p, dn_s, dn_a) or None when the NVFP4 expert tower is not prepared.
        """
        W = self.W
        if not getattr(W, "_exp_fp4_ready", False):
            return None
        amap = W._exp_fp4_alpha

        def a(p):
            return amap[p.data_ptr()]

        return (W._exp_qkv_p4[li], W._exp_qkv_s4[li], a(W._exp_qkv_p4[li]),
                W._exp_o_p4[li], W._exp_o_s4[li], a(W._exp_o_p4[li]),
                W._exp_gu_p4[li], W._exp_gu_s4[li], a(W._exp_gu_p4[li]),
                W._exp_d_p4[li], W._exp_d_s4[li], a(W._exp_d_p4[li]))


    def _fa2_buffers(self, q, S_p, S):
        """Return the per-(S_p,S) FA2 denoise buffers (seqused, qb, ob, lse),
        creating the zero-filled ``qb`` once so the zero-fill is not repeated."""
        key = (S_p, S)
        cached = self._fa2_seqused_cache.get(key)
        if cached is None:
            B, H, D = q.shape[0], q.shape[1], q.shape[3]
            Skv = S_p + S
            seqused = torch.tensor([S_p + 1, Skv], dtype=torch.int32,
                                   device=q.device)
            # qb is zeroed once; the prepare kernel (or the fused RoPE variant)
            # writes only the non-dummy rows, so the zero-fill and the q
            # transpose/contiguous copy are not repeated per denoise step.
            qb = torch.zeros(B * 2, S, H, D, dtype=q.dtype, device=q.device)
            ob = torch.empty(B * 2, S, H, D, dtype=q.dtype, device=q.device)
            lse = torch.empty(B * 2, H, S, dtype=torch.float32,
                              device=q.device)
            cached = (seqused, qb, ob, lse)
            self._fa2_seqused_cache[key] = cached
        return cached


    def _fa2_denoise_attn(self, q, k, v, S_p, force_fp8=False, qb_pre=None):
        """Native SM120 FA2 for the denoise suffix (single call, batch=2).

        The denoise mask is prefix-full + suffix near-global with the state
        token (row 0) self-only. Expressed as one FA2 seqused call: batch 0 =
        [state + 39 zero dummies] with seqused_k=S_p+1, batch 1 = [actions] with
        seqused_k=S_p+S. q/k/v are (B,H,S,D); FA2 reinterprets (B,S,H,D) via
        strides (no copy). ``qb_pre`` (when given) is the already-filled
        (2,S,H,D) query packing produced by the fused RoPE kernel."""
        fa2 = self._fa2
        B, H, S, D = q.shape
        Skv = S_p + S
        scale = 1.0 / (D ** 0.5)
        num_sms = self._fa2_num_sms
        st = torch.cuda.current_stream().cuda_stream

        seqused, qb, ob, lse = self._fa2_buffers(q, S_p, S)
        if qb_pre is not None:
            qb = qb_pre
        else:
            prepare = getattr(self, "_fa2_prepare_q", None)
            if prepare is not None:
                prepare(q.data_ptr(), qb.data_ptr(), S, H, D, st)
            else:
                qf = q.transpose(1, 2).contiguous()  # (B,S,H,D)
                qb[0, 0] = qf[0, 0]
                qb[1, :S - 1] = qf[0, 1:]
        k2 = k.expand(2, -1, -1, -1)  # broadcast KV to the 2 batches (stride 0)
        v2 = v.expand(2, -1, -1, -1)
        attn = getattr(fa2, "fwd_bf16_tile", fa2.fwd_bf16_seqused)
        attn(
            qb.data_ptr(), k2.data_ptr(), v2.data_ptr(), ob.data_ptr(),
            lse.data_ptr(), seqused.data_ptr(), B * 2, S, Skv, H, H, D,
            (qb.stride(0), qb.stride(1), qb.stride(2)),
            (k2.stride(0), k2.stride(2), k2.stride(1)),
            (v2.stride(0), v2.stride(2), v2.stride(1)),
            (ob.stride(0), ob.stride(1), ob.stride(2)),
            scale, num_sms, st)
        o4 = getattr(self, "_fa2_gather_quant_o4", None)
        if (getattr(self, "_exp_fp4", False) and not force_fp8
                and o4 is not None and (H * D) % 16 == 0):
            ap = torch.empty(S, (H * D) // 2, dtype=torch.uint8, device=q.device)
            asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(S, H * D),
                              dtype=torch.uint8, device=q.device)
            o4(ob.data_ptr(), ap.data_ptr(), asf.data_ptr(), S, H, D, st)
            return ap, asf
        gq = getattr(self, "_fa2_gather_quant_o", None)
        if gq is not None and (H * D) % 128 == 0:
            a8 = torch.empty(S, H * D, dtype=torch.uint8, device=q.device)
            asc = torch.empty(S, (H * D) // 128, dtype=torch.float32,
                              device=q.device)
            gq(ob.data_ptr(), a8.data_ptr(), asc.data_ptr(), S, H, D, st)
            return a8, asc
        return torch.cat([ob[0:1, 0:1], ob[1:2, :S - 1]], dim=1).transpose(1, 2)


    def _fp4_gemm_f4(self, x, w_packed, w_sf, N, K):
        """Prefill FFN NVFP4 GEMM using the flash_rt_kernels NVFP4 family
        (weights from bf16_weight_to_nvfp4_swizzled)."""
        alpha = getattr(self, "_fp4_alpha", {}).get(w_packed.data_ptr(), 1.0)
        return self._nvfp4_gemm(x, w_packed, w_sf, alpha, None)


    def _fp4_gemm_prepacked(self, a_packed, w_packed, out, a_sfa, w_sfb,
                            w_alpha, bias):
        """Dispatch a prequantized-A NVFP4 GEMM: pingpong (no-bias) /
        pingpong+bias / bias epilogue / base (+ separate bias add)."""
        M = a_packed.shape[0]
        N = w_packed.shape[0]
        K = a_packed.shape[1] * 2
        st = torch.cuda.current_stream().cuda_stream
        pp = getattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_pingpong", None)
        ppb = getattr(fvk, "fp4_w4a16_gemm_sm120_bf16out_pingpong_bias", None)
        if bias is None and pp is not None and self._pp_ok(M, N, K):
            pp(a_packed.data_ptr(), w_packed.data_ptr(), out.data_ptr(),
               M, N, K, a_sfa.data_ptr(), w_sfb.data_ptr(), w_alpha, st)
        elif bias is not None and ppb is not None and self._pp_bias_ok(M, N, K):
            ppb(a_packed.data_ptr(), w_packed.data_ptr(), out.data_ptr(),
                bias.data_ptr(), M, N, K, a_sfa.data_ptr(), w_sfb.data_ptr(),
                w_alpha, st)
        elif bias is not None and hasattr(
                fvk, "fp4_w4a16_gemm_sm120_bf16out_bias"):
            fvk.fp4_w4a16_gemm_sm120_bf16out_bias(
                a_packed.data_ptr(), w_packed.data_ptr(), out.data_ptr(),
                bias.data_ptr(), M, N, K, a_sfa.data_ptr(), w_sfb.data_ptr(),
                w_alpha, st)
        else:
            fvk.fp4_w4a16_gemm_sm120_bf16out(
                a_packed.data_ptr(), w_packed.data_ptr(), out.data_ptr(),
                M, N, K, a_sfa.data_ptr(), w_sfb.data_ptr(), w_alpha, st)
            if bias is not None:
                fvk.add_bias_bf16(out.data_ptr(), bias.data_ptr(), M, N, st)
        return out


    def _fp8_block128_gemm(self, x, w8, wscale):
        """x (...,K) bf16 -> (...,N) bf16 via native SM120a FP8 block-128 GEMM.

        ``w8`` is the e4m3 weight in (N, K) layout (uint8 bits); ``wscale`` is
        the per-128x128 block scale (N/128, K/128) fp32. Activations are
        quantized per-token per-128-K block by the native kernel on each call
        (graph-safe: static pointers, no CPU sync)."""
        N, K = w8.shape
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        a8 = torch.empty(M, K, dtype=torch.uint8, device=x.device)
        ascale = torch.empty(M, K // 128, dtype=torch.float32, device=x.device)
        fvk.fp8_per_token_block128_quant_bf16(
            xc.data_ptr(), a8.data_ptr(), ascale.data_ptr(), M, K, st)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            a8.data_ptr(), w8.data_ptr(), out.data_ptr(), M, N, K,
            ascale.data_ptr(), wscale.data_ptr(), st)
        return out


    def _fp8_block128_gemm_bias(self, x, w8, ws, bias):
        """FP8 block-128 GEMM (...,K)->(...,N) + bias via the fused epilogue."""
        orig = x.shape
        N, K = w8.shape
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        a8 = torch.empty(M, K, dtype=torch.uint8, device=x.device)
        ascale = torch.empty(M, K // 128, dtype=torch.float32, device=x.device)
        fvk.fp8_per_token_block128_quant_bf16(
            xc.data_ptr(), a8.data_ptr(), ascale.data_ptr(), M, K, st)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out_bias(
            a8.data_ptr(), w8.data_ptr(), out.data_ptr(), bias.data_ptr(),
            M, N, K, ascale.data_ptr(), ws.data_ptr(), st)
        return out.reshape(*orig[:-1], N)


    def _fp8_block128_gemm_prequant(self, a8, ascale, w8, wscale):
        """SM120a FP8 block-128 GEMM on a pre-quantized activation (a8, ascale)
        — the backend half of _fp8_block128_gemm, for fused quant producers."""
        N, K = w8.shape
        M = a8.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a8.device)
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            a8.data_ptr(), w8.data_ptr(), out.data_ptr(), M, N, K,
            ascale.data_ptr(), wscale.data_ptr(), st)
        return out


    def _fp8_block128_gemm_prequant_bias(self, a8, ascale, w8, wscale, bias):
        """Pre-quantized FP8 GEMM with a fused per-column bias epilogue."""
        N, K = w8.shape
        M = a8.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a8.device)
        fvk.fp8_block128_gemm_cutlass_sm120_bf16out_bias(
            a8.data_ptr(), w8.data_ptr(), out.data_ptr(), bias.data_ptr(),
            M, N, K, ascale.data_ptr(), wscale.data_ptr(), st)
        return out


    def _norm(self, x, normalized_shape, weight=None, eps=1e-5):
        """Prefer the repository-native rms_norm kernel (hot-path RMSNorm sites)."""
        return _rms_norm(x, normalized_shape, weight, eps)


    def _norm_to_nvfp4(self, x, weight):
        """Fused RMSNorm + NVFP4 swizzled quant -> (packed, sfa)."""
        if not hasattr(fvk, "rms_norm_to_nvfp4_swizzled_bf16"):
            return None
        d = x.shape[-1]
        if d % 16 != 0 or weight is None:
            return None
        xc = x.reshape(-1, d)
        M = xc.shape[0]
        ap = torch.empty(M, d // 2, dtype=torch.uint8, device=x.device)
        asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, d),
                          dtype=torch.uint8, device=x.device)
        fvk.rms_norm_to_nvfp4_swizzled_bf16(
            xc.data_ptr(), weight.data_ptr(), ap.data_ptr(), asf.data_ptr(),
            M, d, self.rms_eps, torch.cuda.current_stream().cuda_stream)
        return ap, asf


    def _nvfp4_gemm(self, x, w_packed, w_sfb, w_alpha, bias):
        """NVFP4 W4A4 GEMM (...,K)->(...,N) bf16: dynamic activation quant."""
        orig = x.shape
        K = orig[-1]
        x2d = x.reshape(-1, K).contiguous()
        M = x2d.shape[0]
        N = w_packed.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        a_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)
        a_sfa = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K), dtype=torch.uint8,
                            device=x.device)
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x2d.data_ptr(), a_packed.data_ptr(), a_sfa.data_ptr(), M, K, st)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        self._fp4_gemm_prepacked(a_packed, w_packed, out, a_sfa, w_sfb,
                                 w_alpha, bias)
        return out.reshape(*orig[:-1], N)


    def _nvfp4_gemm_out(self, x, w_packed, w_sfb, w_alpha, out):
        """NVFP4 GEMM writing into a caller-owned (M,N) bf16 buffer."""
        K = w_packed.shape[1] * 2
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        a_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)
        a_sfa = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K),
                            dtype=torch.uint8, device=x.device)
        fvk.quantize_bf16_to_nvfp4_swizzled(
            xc.data_ptr(), a_packed.data_ptr(), a_sfa.data_ptr(), M, K, st)
        self._fp4_gemm_prepacked(a_packed, w_packed, out, a_sfa, w_sfb,
                                 w_alpha, None)
        return out


    def _nvfp4_gemm_out_prequant(self, a8, sfa, w_packed, w_sfb, w_alpha, out):
        """NVFP4 GEMM on a prequantized activation, writing into ``out``."""
        self._fp4_gemm_prepacked(a8, w_packed, out, sfa, w_sfb, w_alpha, None)
        return out


    def _nvfp4_gemm_prequant(self, a_packed, a_sfa, w_packed, w_sfb, w_alpha,
                             bias):
        """NVFP4 GEMM on a pre-quantized (packed, swizzled SFA) activation."""
        N = w_packed.shape[0]
        M = a_packed.shape[0]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a_packed.device)
        return self._fp4_gemm_prepacked(a_packed, w_packed, out, a_sfa, w_sfb,
                                        w_alpha, bias)


    def _nvfp4_gemm_prequant_od(self, a_packed, a_sfa, w_packed, w_sfb, w_alpha,
                                cols=32, kg=4, stages=2):
        """NVFP4 prequant GEMM for the narrow-N expert o/dn shapes via the
        col-tile x K-group split kernel (A read once per block, K-group
        partials reduced in smem, graph-replay safe). Falls back to the
        CUTLASS persistent NVFP4 GEMM when the shape/config is unsupported."""
        f = getattr(fvk, "fp4_w4a4_mma_sm120_cksplit_bf16out", None)
        N = w_packed.shape[0]
        M = a_packed.shape[0]
        K = a_packed.shape[1] * 2
        if (f is None or M > 48 or N % cols or (K // 64) % kg):
            return self._nvfp4_gemm_prequant(a_packed, a_sfa, w_packed, w_sfb,
                                             w_alpha, None)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a_packed.device)
        rc = f(a_packed.data_ptr(), w_packed.data_ptr(), out.data_ptr(), M, N, K,
               a_sfa.data_ptr(), w_sfb.data_ptr(), w_alpha, cols, kg, stages,
               torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            return self._nvfp4_gemm_prequant(a_packed, a_sfa, w_packed, w_sfb,
                                             w_alpha, None)
        return out


    def _od_b128_splitk_gemm(self, x, w8, wscale, k_split=4):
        """Block-128-scaled split-K FP8 GEMM (M=41 o/dn): quantize the
        activation block-128, then sum K-split partials with per-128-block
        descaling. Preserves block-128 accuracy with split-K parallelism."""
        N, K = w8.shape
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        a8 = torch.empty(M, K, dtype=torch.uint8, device=x.device)
        asc = torch.empty(M, K // 128, dtype=torch.float32, device=x.device)
        fvk.fp8_per_token_block128_quant_bf16(
            xc.data_ptr(), a8.data_ptr(), asc.data_ptr(), M, K, st)
        return self._od_b128_splitk_prequant(a8, asc, w8, wscale, k_split)


    def _od_b128_splitk_prequant(self, a8, asc, w8, wscale, k_split=4):
        N, K = w8.shape
        M = a8.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a8.device)
        cache = getattr(self, "_splitk_scratch", None)
        if cache is None:
            cache = {}
            self._splitk_scratch = cache
        key = (M, N, k_split)
        scr = cache.get(key)
        if scr is None:
            scr = torch.empty(M * N * k_split, dtype=torch.float32,
                              device=a8.device)
            cache[key] = scr
        fvk.splitk_b128_fp8_gemm_32x64x128_w4(
            a8.data_ptr(), w8.data_ptr(), asc.data_ptr(), wscale.data_ptr(),
            out.data_ptr(), M, N, K, K // 128, k_split, scr.data_ptr(), st)
        return out


    def _pp_bias_ok(self, M, N, K):
        """Route large-M fused-bias NVFP4 GEMMs to the pingpong+bias kernel.
        It is bit-identical to the base bias kernel and measured 6-21% faster on
        the ViT qkv/proj/fc2 shapes (M=3528/588). All bias NVFP4 GEMMs in this
        pipeline are ViT-scale (M>=512); small-M bias shapes do not occur."""
        if not getattr(self, "_use_pingpong", True):
            return False
        return M >= 512


    def _pp_ok(self, M, N, K):
        """Route large no-bias NVFP4 GEMMs to the <128,256,128> pingpong tile.
        It is bit-identical to the base kernel and measured faster on the large-M
        ViT fc1 (3528/588, 4352, 1152) and prefix MLP (147/34, 12288, 2048)
        shapes; pingpong is equal/worse on the small-M expert and prefix
        qkv/proj/down shapes, which stay on the base kernel."""
        if not getattr(self, "_use_pingpong", True):
            return False
        return (M >= 512 and N >= 4096 and K <= 2048) or (N >= 8192 and K >= 2048)


    def _prefill_attn_native(self, q, k, v, mask):
        """Repository-native block-sparse BF16 attention for the prefill
        segment mask (block-diagonal 3x49 vision + causal text). Replaces the
        mem-efficient SDPA (fmha_cutlassF sm80) on the prefill path; q/k/v are
        (1, nh, S, hd) with the KV cache already pre-expanded, mask is
        (1, 1, S, S) bf16 additive (0/-inf)."""
        key = mask.data_ptr()
        cached = self._prefill_native_cache.get(key)
        if cached is None:
            S = mask.shape[-1]
            nh = q.shape[1]
            hd = q.shape[3]
            mask_b = mask[0, 0]  # (S, S) bf16 0/-inf
            mask_f32 = mask_b.float().contiguous()
            QT, KT = 64, 16
            num_qb = (S + QT - 1) // QT
            num_kb = (S + KT - 1) // KT
            active = torch.zeros(num_qb, nh, dtype=torch.uint32, device=q.device)
            is_zero = (mask_b != float("-inf"))
            for qb in range(num_qb):
                qs, qe = qb * QT, min((qb + 1) * QT, S)
                m = 0
                for kb in range(num_kb):
                    ks, ke = kb * KT, min((kb + 1) * KT, S)
                    if is_zero[qs:qe, ks:ke].any():
                        m |= (1 << kb)
                active[qb, :] = m
            cached = (mask_f32, active)
            self._prefill_native_cache[key] = cached
        mask_f32, active = cached
        S, nh, hd = q.shape[2], q.shape[1], q.shape[3]
        # Output in the consumer (S, H, D) layout so the caller's
        # transpose(1,2).reshape(1,S,q_dim) is a view (no copy).
        o = torch.empty(S, nh, hd, dtype=torch.bfloat16, device=q.device)
        scale = 1.0 / (hd ** 0.5)
        fvk.hyvla_prefill_attn_bf16(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
            mask_f32.data_ptr(), active.data_ptr(), S, nh, scale,
            q.stride(1), k.stride(1), v.stride(1),
            torch.cuda.current_stream().cuda_stream)
        return o.permute(1, 0, 2)[None]


    def _res_add_norm_to_nvfp4(self, residual, x_add, weight, out):
        """Fused (residual + x_add) + RMSNorm + NVFP4 swizzled quant: writes the
        new residual into ``out``; returns (packed, sfa)."""
        if (not hasattr(fvk, "residual_add_rms_norm_to_nvfp4_swizzled_bf16")
                or weight is None):
            return None
        d = residual.shape[-1]
        if d % 16 != 0:
            return None
        r = residual.reshape(-1, d)
        a = x_add.reshape(-1, d)
        o_ = out.reshape(-1, d)
        M = r.shape[0]
        ap = torch.empty(M, d // 2, dtype=torch.uint8, device=residual.device)
        asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, d),
                          dtype=torch.uint8, device=residual.device)
        st = torch.cuda.current_stream().cuda_stream
        v3 = getattr(fvk, "residual_add_rms_norm_to_nvfp4_swizzled_bf16_v3", None)
        if v3 is not None and d <= 1024:
            v3(r.data_ptr(), a.data_ptr(), o_.data_ptr(), weight.data_ptr(),
               ap.data_ptr(), asf.data_ptr(), M, d, self.rms_eps,
               max(128, min(512, d // 2)), st)
        elif v3 is not None and d <= 2048 and M <= 64:
            v3(r.data_ptr(), a.data_ptr(), o_.data_ptr(), weight.data_ptr(),
               ap.data_ptr(), asf.data_ptr(), M, d, self.rms_eps, 1024, st)
        else:
            fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16(
                r.data_ptr(), a.data_ptr(), o_.data_ptr(), weight.data_ptr(),
                ap.data_ptr(), asf.data_ptr(), M, d, self.rms_eps, st)
        return ap, asf


    def _res_add_rms_norm_to_fp8(self, residual, x, weight):
        K = residual.shape[-1]
        r = residual.reshape(-1, K).contiguous()
        xc = x.reshape(-1, K).contiguous()
        M = r.shape[0]
        a8 = torch.empty(M, K, dtype=torch.uint8, device=r.device)
        asc = torch.empty(M, K // 128, dtype=torch.float32, device=r.device)
        # residual_out == residual: the fused kernel updates the residual
        # stream in place, matching the standalone residual_add_rms_norm.
        fvk.residual_add_rms_norm_to_fp8_block128_bf16(
            r.data_ptr(), xc.data_ptr(), r.data_ptr(), weight.data_ptr(),
            a8.data_ptr(), asc.data_ptr(), M, K, self.rms_eps,
            torch.cuda.current_stream().cuda_stream)
        return a8, asc


    def _rms_norm_to_fp8(self, x, weight):
        K = x.shape[-1]
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        a8 = torch.empty(M, K, dtype=torch.uint8, device=x.device)
        asc = torch.empty(M, K // 128, dtype=torch.float32, device=x.device)
        fvk.rms_norm_to_fp8_block128_bf16(
            xc.data_ptr(), weight.data_ptr(), a8.data_ptr(), asc.data_ptr(),
            M, K, self.rms_eps, torch.cuda.current_stream().cuda_stream)
        return a8, asc


    def _rope_qknorm_kvwrite_qb(self, qkv, cos, sin, qk_w, qb, kbuf, vbuf,
                                S, off):
        """Fused RoPE+QK-Norm+KV-write writing Q directly in the FA2 denoise
        (2,S,nq,hd) packing (replaces the separate prepare_q transpose)."""
        hd = self.head_dim
        S_tot = kbuf.shape[2]
        kv_rep = kbuf.shape[1] // self.n_kv
        fvk.hyvla_rope_qknorm_kvwrite_qb_bf16(
            qkv.data_ptr(),
            cos.reshape(S, hd).contiguous().data_ptr(),
            sin.reshape(S, hd).contiguous().data_ptr(),
            qk_w[0].data_ptr(), qk_w[1].data_ptr(),
            qb.data_ptr(), kbuf.data_ptr(), vbuf.data_ptr(),
            S, self.n_heads, self.n_kv, hd, S_tot, off, self.rms_eps,
            kv_rep, torch.cuda.current_stream().cuda_stream)


    def _silu_mul_to_fp8(self, gu):
        K = gu.shape[-1] // 2
        g = gu.reshape(-1, 2 * K).contiguous()
        M = g.shape[0]
        a8 = torch.empty(M, K, dtype=torch.uint8, device=gu.device)
        asc = torch.empty(M, K // 128, dtype=torch.float32, device=gu.device)
        fvk.silu_mul_merged_to_fp8_block128_bf16(
            g.data_ptr(), a8.data_ptr(), asc.data_ptr(), M, K,
            torch.cuda.current_stream().cuda_stream)
        return a8, asc


    def _silu_mul_to_nvfp4(self, gu):
        """SwiGLU producer -> NVFP4 swizzled activation (packed, sfa)."""
        K = gu.shape[-1] // 2
        g = gu.reshape(-1, 2 * K).contiguous()
        M = g.shape[0]
        ap = torch.empty(M, K // 2, dtype=torch.uint8, device=gu.device)
        asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K), dtype=torch.uint8,
                          device=gu.device)
        # grouped32 (32 scale groups/CTA, atomic-free) is the fast path for the
        # wide SwiGLU rows (cols=2048); falls back to the plain producer if the
        # grouped binding is absent.
        fn = getattr(fvk, "silu_mul_merged_to_nvfp4_swizzled_grouped32_bf16",
                     None) or fvk.silu_mul_merged_to_nvfp4_swizzled_bf16
        fn(g.data_ptr(), ap.data_ptr(), asf.data_ptr(), M, K,
           torch.cuda.current_stream().cuda_stream)
        return ap, asf


    def _vit_add_ln_to_fp8(self, x, add, lnw, lnb):
        """Fused (x += add) + LayerNorm + block-128 FP8 quant -> (a8, ascale).

        Same in-place residual add as _vit_add_ln, but the normed activation is
        emitted straight to FP8 (no bf16 HBM round-trip before the GEMM)."""
        bk, n, d = x.shape
        M = bk * n
        a8 = torch.empty(M, d, dtype=torch.uint8, device=x.device)
        ascale = torch.empty(M, d // 128, dtype=torch.float32, device=x.device)
        fvk.hyvla_vit_add_layer_norm_to_fp8_block128_bf16(
            x.data_ptr(), add.data_ptr(), lnw.data_ptr(), lnb.data_ptr(),
            a8.data_ptr(), ascale.data_ptr(), M, d, self.vit_eps,
            torch.cuda.current_stream().cuda_stream)
        return a8, ascale


    def _vit_add_ln_to_nvfp4(self, x, add, lnw, lnb):
        """Fused (x += add) + LayerNorm + NVFP4 swizzled quant -> (packed, sfa)."""
        bk, n, d = x.shape
        M = bk * n
        ap = torch.empty(M, d // 2, dtype=torch.uint8, device=x.device)
        asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, d), dtype=torch.uint8,
                          device=x.device)
        fvk.hyvla_vit_add_layer_norm_to_nvfp4_swizzled_bf16(
            x.data_ptr(), add.data_ptr(), lnw.data_ptr(), lnb.data_ptr(),
            ap.data_ptr(), asf.data_ptr(), M, d, self.vit_eps,
            torch.cuda.current_stream().cuda_stream)
        return ap, asf


    def _vit_mlp_fc2(self, fc1_out, bk, n, d):
        M, K1 = fc1_out.shape[0], fc1_out.shape[1]
        if getattr(self.W, "_vit_fc2_awq", False):
            # SmoothQuant NVFP4 fc2: fused (fc1_out + bias) - erf-GELU - *inv_s
            # - NVFP4 block quant, then the pre-scaled-weight NVFP4 GEMM.
            ap = torch.empty(M, K1 // 2, dtype=torch.uint8,
                             device=fc1_out.device)
            asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K1),
                              dtype=torch.uint8, device=fc1_out.device)
            fvk.awq_bias_gelu_quant_bf16_to_nvfp4_swizzled(
                fc1_out.data_ptr(), self._vit_fc1_b_cur.data_ptr(),
                self._vit_fc2_inv_s_cur.data_ptr(), ap.data_ptr(),
                asf.data_ptr(), M, K1,
                torch.cuda.current_stream().cuda_stream)
            out2 = self._nvfp4_gemm_prequant(
                ap, asf, self._vit_fc2_p4_cur, self._vit_fc2_s4_cur,
                self._vit_fc2_a4_cur, self._vit_fc2_b_cur)
            return out2.reshape(bk, n, d)
        if (getattr(self, "_vit_fc2_fused_nvfp4", False)
                and hasattr(fvk, "bias_gelu_quant_bf16_to_nvfp4_swizzled")):
            # Non-AWQ fused NVFP4 fc2: (fc1_out + bias) -> GELU -> NVFP4
            # block quant in one producer, then the prequant GEMM. Static
            # weights, dynamic activation quant, no data-dependent calibration.
            ap = torch.empty(M, K1 // 2, dtype=torch.uint8,
                             device=fc1_out.device)
            asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K1),
                              dtype=torch.uint8, device=fc1_out.device)
            # Prefer the register-resident producer (single GELU pass, no
            # shared-memory atomics); fall back to the atomic kernel.
            fn = getattr(fvk, "bias_gelu_quant_bf16_to_nvfp4_swizzled_v2", None) \
                or fvk.bias_gelu_quant_bf16_to_nvfp4_swizzled
            fn(fc1_out.data_ptr(), self._vit_fc1_b_cur.data_ptr(),
               ap.data_ptr(), asf.data_ptr(), M, K1,
               torch.cuda.current_stream().cuda_stream)
            out2 = self._nvfp4_gemm_prequant(
                ap, asf, self._vit_fc2_p4_cur, self._vit_fc2_s4_cur,
                self._vit_fc2_a4_cur, self._vit_fc2_b_cur)
            return out2.reshape(bk, n, d)
        a8 = torch.empty(M, K1, dtype=torch.uint8, device=fc1_out.device)
        ascale = torch.empty(M, K1 // 128, dtype=torch.float32,
                             device=fc1_out.device)
        fvk.gelu_erf_bias_to_fp8_block128_bf16(
            fc1_out.data_ptr(), self._vit_fc1_b_cur.data_ptr(), a8.data_ptr(),
            ascale.data_ptr(), M, K1,
            torch.cuda.current_stream().cuda_stream)
        out2 = self._fp8_block128_gemm_prequant_bias(
            a8, ascale, self._vit_fc2_w8c, self._vit_fc2_wsc,
            self._vit_fc2_b_cur)
        return out2.reshape(bk, n, d)


    def _vit_mlp_prequant(self, a8, ascale, x):
        """ViT MLP on a pre-quantized fc1 activation (from the fused LN->FP8)."""
        bk, n, d = x.shape
        out = self._fp8_block128_gemm_prequant(
            a8, ascale, self._vit_fc1_w8c, self._vit_fc1_wsc)
        return self._vit_mlp_fc2(out, bk, n, d)


    def _vit_mlp_prequant_fp4(self, a_packed, a_sfa, bk, n, d):
        M = a_packed.shape[0]
        if (getattr(self, "_vit_fc1_gelu_fuse", False) and M >= 1024
                and hasattr(fvk, "fp4_w4a16_gemm_bias_gelu_fp4out_sm120")):
            # Fused fc1 NVFP4 GEMM + per-col bias + tanh-GELU + NVFP4 block
            # quant epilogue (cutlass-swizzled SF), replacing the bf16-output
            # GEMM plus the standalone bias_gelu_quant producer.
            N1 = self._vit_fc1_p4_cur.shape[0]
            ap = torch.empty(M, N1 // 2, dtype=torch.uint8,
                             device=a_packed.device)
            asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, N1),
                              dtype=torch.uint8, device=a_packed.device)
            fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
                a_packed.data_ptr(), self._vit_fc1_p4_cur.data_ptr(),
                a_sfa.data_ptr(), self._vit_fc1_s4_cur.data_ptr(),
                self._vit_fc1_b_cur.data_ptr(), ap.data_ptr(), asf.data_ptr(),
                M, N1, d, self._vit_fc1_a4_cur,
                torch.cuda.current_stream().cuda_stream)
            out2 = self._nvfp4_gemm_prequant(
                ap, asf, self._vit_fc2_p4_cur, self._vit_fc2_s4_cur,
                self._vit_fc2_a4_cur, self._vit_fc2_b_cur)
            return out2.reshape(bk, n, d)
        o1 = self._nvfp4_gemm_prequant(
            a_packed, a_sfa, self._vit_fc1_p4_cur, self._vit_fc1_s4_cur,
            self._vit_fc1_a4_cur, None)
        return self._vit_mlp_fc2(o1.reshape(-1, o1.shape[-1]), bk, n, d)


    def _vit_proj_gather_nvfp4(self, o):
        """Gather the FA2 spatial output (bk,H,N,Ds) into the proj NVFP4
        activation (rows=bk*N, K=H*Dh) in one pass (slice+transpose+quant)."""
        bk, H, N, Ds = o.shape
        Dh = self.vit_hd
        K = H * Dh
        M = bk * N
        ap = torch.empty(M, K // 2, dtype=torch.uint8, device=o.device)
        asf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K),
                          dtype=torch.uint8, device=o.device)
        fvk.hyvla_vit_proj_gather_nvfp4_swizzled_bf16(
            o.data_ptr(), ap.data_ptr(), asf.data_ptr(), bk, H, N, Ds, Dh,
            torch.cuda.current_stream().cuda_stream)
        return ap, asf


    def _vit_qkv_prequant(self, a8, ascale, x):
        bk, N, _ = x.shape
        bias = self._vit_qkv_b_pad_cur if self._vit_qkv_b_pad_cur is not None \
            else self._vit_qkv_b_cur
        qkv = self._fp8_block128_gemm_prequant_bias(
            a8, ascale, self._vit_qkv_w8_cur, self._vit_qkv_ws_cur, bias)
        hd_pad = getattr(self, "_vit_hd_pad", self.vit_hd)
        qkv = qkv.reshape(bk, N, 3, self.vit_heads, hd_pad).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]


    def _vit_qkv_prequant_fp4(self, a_packed, a_sfa, x):
        bk, N, _ = x.shape
        bias = self._vit_qkv_b_pad_cur if self._vit_qkv_b_pad_cur is not None \
            else self._vit_qkv_b_cur
        qkv = self._nvfp4_gemm_prequant(
            a_packed, a_sfa, self._vit_qkv_p4_cur, self._vit_qkv_s4_cur,
            self._vit_qkv_a4_cur, bias)
        hd_pad = getattr(self, "_vit_hd_pad", self.vit_hd)
        qkv = qkv.reshape(bk, N, 3, self.vit_heads, hd_pad).permute(
            2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]


    def _vit_spatial_attn_fa2(self, q, k, v):
        """Repository-native SM120 FA2 for the ViT spatial attention (full,
        non-causal attention over N tokens). q/k/v are (bk, heads, N, 96) with
        the head-dim padded 72->96 (zero columns). FA2 reinterprets the (bk,
        heads, N, 96) layout as (batch, seqlen, heads, hd) via strides, so no
        copy is needed. GQA is MHA here (num_heads_kv == num_heads_q)."""
        fa2 = self._fa2
        bk, H, N, D = q.shape
        st = torch.cuda.current_stream().cuda_stream
        o = torch.empty(bk, H, N, D, dtype=q.dtype, device=q.device)
        lse = torch.empty(bk, H, N, dtype=torch.float32, device=q.device)
        # Small-N query tile (hd96 <64,32,4>) when the specialisation is built.
        attn = getattr(fa2, "fwd_bf16_tile", None)
        if attn is not None:
            attn(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                lse.data_ptr(), 0,
                bk, N, N, H, H, D,
                (q.stride(0), q.stride(2), q.stride(1)),
                (k.stride(0), k.stride(2), k.stride(1)),
                (v.stride(0), v.stride(2), v.stride(1)),
                (o.stride(0), o.stride(2), o.stride(1)),
                self.vit_scale, self._fa2_num_sms, st)
        else:
            fa2.fwd_bf16(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                lse.data_ptr(), 0, 0,
                bk, N, N, H, H, D,
                (q.stride(0), q.stride(2), q.stride(1)),
                (k.stride(0), k.stride(2), k.stride(1)),
                (v.stride(0), v.stride(2), v.stride(1)),
                (o.stride(0), o.stride(2), o.stride(1)),
                self.vit_scale, self._fa2_num_sms, st)
        return o


    def enable_fp8_block128(self):
        """Route the ``_fp8`` GEMM slots to the SM120a FP8 block-128 cutlass
        GEMM (fp8_block128_gemm_cutlass_sm120_bf16out). The frontend stores the
        block-128 e4m3 weights in the same ``_exp_*8``/``_vlm_*8`` slots the
        Thor FP8 path uses, so the parent ``_block`` FP8 branch routes through
        ``_fp8_gemm`` unchanged — only the GEMM backend swaps."""
        self._fp8 = True
        self._fp8b128 = True
        self.gemm = None
        # Fused producer->FP8 quant for the denoise tower: the residual-add /
        # RMSNorm / SwiGLU producers emit the block-128 FP8 activation directly,
        # removing the standalone quant launch and the bf16 round-trip.
        # HYVLA_FUSED_QUANT=0 disables the route (A/B isolation).
        self._fused_quant = (
            os.environ.get("HYVLA_FUSED_QUANT", "1") == "1"
            and hasattr(fvk, "residual_add_rms_norm_to_fp8_block128_bf16")
            and hasattr(fvk, "silu_mul_merged_to_fp8_block128_bf16"))
        self._fa2 = None
        self._fa2_seqused_cache = {}
        self._native_prefill_attn = hasattr(fvk, "hyvla_prefill_attn_bf16")
        self._prefill_native_cache = {}
        try:
            import flash_rt.flash_rt_fa2 as fa2
            if hasattr(fa2, "fwd_bf16_seqused"):
                self._fa2 = fa2
                self._fa2_num_sms = torch.cuda.get_device_properties(
                    0).multi_processor_count
        except ImportError:
            pass

    # ---- moved from the shared HyVLA Thor pipeline (SM120 overrides) ----

    def __init__(self, W):
        self.W = W
        self.n_heads = 16
        self.n_kv = 4
        self.head_dim = 128
        self.q_dim = self.n_heads * self.head_dim      # 2048
        self.kv_dim = self.n_kv * self.head_dim        # 512
        self.rms_eps = 1e-5
        self._fp8 = False
        self.gemm = None
        self._fused_attn = False
        self._fp4 = False
        self._F4 = None
        # When M <= this, use the single-CTA fused dynamic FP8 quant
        # (hyvla_quant_fp8_dyn_bf16, 1 launch) instead of quantize_fp8_device
        # (4 nodes). 0 disables. Set by the frontend for the denoise tower.
        self._small_quant_m = 0
        # When a set(), _fp8_gemm records the (M,N,K) it sees so the frontend
        # can autotune the cuBLASLt FP8 algo per shape before graph capture.
        self._gemm_shapes = None
        # Fuse the expert denoise FFN (gu+silu_mul, dn+residual) into two
        # occupancy-preserving persistent megakernels (hyvla_ffn_*).
        self._ffn_mega = False
        # ViT (HYViT2-400M)
        self.vit_heads = 16
        self.vit_hd = 72
        self.vit_scale = self.vit_hd ** -0.5
        self.vit_eps = 1e-6
        self.vit_time_base = 100.0
        self.vit_spacetime_ids = set(range(3, 27, 4))   # {3,7,11,15,19,23}
        # Static ViT positional embeddings (spatial rescale + time PE) depend
        # only on the fixed spatial/temporal sizes, so they are computed once
        # and reused across every forward instead of being recomputed inside
        # each captured graph replay.
        self._vit_pos_embed_cache = {}
        self._vit_time_pe_cache = {}

    def _vit_pos_embed_rescale(self, h, w, dtype):
        """Bilinear-rescale learned pos_embed (128x128) to (h,w). (1, h*w, 1152).

        Static per (h, w, dtype) -> cached, so the rescale is not recomputed
        inside every captured graph replay."""
        key = (h, w, dtype)
        cached = self._vit_pos_embed_cache.get(key)
        if cached is not None:
            return cached
        pos = self.W._vit_pos_embed  # (1, 16384, 1152)
        g = int(pos.shape[1] ** 0.5)  # 128
        if (h, w) == (g, g):
            self._vit_pos_embed_cache[key] = pos
            return pos
        pe2d = pos[0].T.contiguous().view(1, -1, g, g).float()
        pe2d = F.interpolate(pe2d, (h, w), mode="bilinear", align_corners=False)
        out = pe2d.view(-1, h * w).T.contiguous()[None].to(dtype)
        self._vit_pos_embed_cache[key] = out
        return out

    def _vit_time_pe(self, kf, device, dtype):
        """Fixed sinusoidal e(t), base 100, e(0)=0. (kf, 1152). Static per
        (kf, dtype) -> cached so graph replay carries no time-PE math."""
        key = (kf, str(device), dtype)
        cached = self._vit_time_pe_cache.get(key)
        if cached is not None:
            return cached
        dim = self.vit_heads * self.vit_hd
        t = torch.arange(kf, dtype=torch.float32, device=device).unsqueeze(1)
        inv_freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                             * (-torch.log(torch.tensor(self.vit_time_base)) / dim))
        pe = torch.empty(kf, dim, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(t * inv_freq)
        pe[:, 1::2] = torch.cos(t * inv_freq) - 1.0
        pe = pe.to(dtype)
        self._vit_time_pe_cache[key] = pe
        return pe

    def _vit_time_mix(self, q, k, v, b, kf):
        """Causal-in-time softmax over K frames folded onto V. (bk,H,N,d).

        Uses the repository-native fused kernel when available (one launch
        instead of the framework bmm/mask/softmax composite); falls back to the
        torch composite for unsupported shapes or when disabled."""
        native = getattr(self, "_native_temporal_mix", None)
        if native is not None:
            bk, H, N, D = v.shape
            if 1 <= kf <= 8 and D <= 128:
                out = torch.empty(bk, H, N, D, dtype=v.dtype, device=v.device)
                native(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                       out.data_ptr(), b, kf, H, N, D,
                       q.stride(0), q.stride(1), q.stride(2),
                       self.vit_scale, torch.cuda.current_stream().cuda_stream)
                return out
        bk, heads, n, d = v.shape
        rs = lambda t: t.view(b, kf, heads, n, d).permute(0, 3, 2, 1, 4).reshape(b * n, heads, kf, d)
        q_t, k_t, v_t = rs(q), rs(k), rs(v)
        scores = (q_t @ k_t.transpose(-2, -1)) * self.vit_scale
        mask = torch.triu(torch.ones(kf, kf, device=scores.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        vm = scores.softmax(dim=-1).to(v_t.dtype) @ v_t
        return vm.view(b, n, heads, kf, d).permute(0, 3, 2, 1, 4).reshape(bk, heads, n, d)

    @torch.no_grad()
    def merger_forward(self, x, grid=14):
        """x (num_cam, 196, 1152) -> (num_cam, 49, 2048). NormalizedDwPooler 2x2."""
        W = self.W
        B = x.shape[0]
        h = w = grid
        x = x.reshape(B, h, w, -1)
        x = F.linear(x, W._mg_proj1_w, W._mg_proj1_b)               # (B,14,14,2048)
        C = x.shape[-1]
        if (getattr(self, "_merger_native", False)
                and hasattr(fvk, "hyvla_merger_pool_bf16")):
            new_x = torch.empty(B, h // 2, w // 2, 4, C, dtype=x.dtype,
                                device=x.device)
            fused = torch.empty(B, h // 2, w // 2, 4, 2 * C, dtype=x.dtype,
                                device=x.device)
            st = torch.cuda.current_stream().cuda_stream
            fvk.hyvla_merger_pool_bf16(
                x.data_ptr(), new_x.data_ptr(), fused.data_ptr(),
                B, h, w, C, st)
            score = F.linear(fused, W._mg_pred0_w, W._mg_pred0_b)
            score = _gelu_erf(score)
            score = F.linear(score, W._mg_pred2_w, W._mg_pred2_b)
            gated = torch.empty(B, h // 2, w // 2, C, dtype=x.dtype,
                                device=x.device)
            fvk.hyvla_merger_gate_bf16(
                score.data_ptr(), new_x.data_ptr(), gated.data_ptr(),
                B, h // 2, w // 2, C, st)
            x = _gelu_erf(gated)
            x = F.linear(x, W._mg_proj2_w, W._mg_proj2_b)
            return x.reshape(B, -1, C)
        new_x = (x.reshape(B, h // 2, 2, w // 2, 2, C)
                 .permute(0, 1, 3, 2, 4, 5).reshape(B, h // 2, w // 2, 4, C))
        pooled = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
        fused = torch.cat([new_x, pooled], dim=-1)                  # (B,7,7,4,4096)
        score = F.linear(fused, W._mg_pred0_w, W._mg_pred0_b)
        score = _gelu_erf(score)
        score = F.linear(score, W._mg_pred2_w, W._mg_pred2_b)       # (B,7,7,4,2048)
        x = (new_x * score.softmax(dim=-2)).sum(dim=-2)             # (B,7,7,2048)
        x = _gelu_erf(x)
        x = F.linear(x, W._mg_proj2_w, W._mg_proj2_b)
        return x.reshape(B, -1, C)

    def _block(self, hs, n_vis, w_text, w_vis, qk_w, mask, cos, sin,
               kbuf, vbuf, off, fp8w=None, fp8v=None, fp8t=None,
               fp4v=None, fp4t=None, ffn_mk=None, pending=None,
               return_dn=False):
        """One MoT transformer block over sorted ``[vision|text]`` tokens.

        ``w_text``/``w_vis`` are per-branch weight tuples
        ``(qkv, o, gu, d, ln_in, ln_post)``. When ``w_text is None`` every
        token uses ``w_vis`` (the all-vision expert suffix). ``fp8w`` (expert)
        or ``fp8v``/``fp8t`` (prefill vision/text branches) =
        ``(qkv8,qkv_ws,o8,o_ws,gu8,gu_ws,d8,d_ws)`` enable graph-safe dynamic
        FP8 for the GEMMs. Writes rope+norm'd K/V into ``kbuf``/``vbuf`` at row
        ``off`` and attends over ``[:off+S]``.
        """
        B, S, D = hs.shape
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv
        _fp8 = self._fp8 and fp8w is not None and w_text is None
        _fp8p = self._fp8 and fp8v is not None and w_text is not None
        _fp4p = self._fp4 and fp4v is not None and w_text is not None
        # The fused RMSNorm/LayerNorm->NVFP4 producers may only run when the
        # consumer GEMM actually has NVFP4 weights; otherwise they emit a
        # (packed, sfa) tuple into an FP8/bf16 GEMM. This keeps the prefill
        # tower on pure FP8 when NVFP4 is disabled via the frontend.
        _wmap_on = getattr(self, "_fp4_weight_map", None) is not None

        if w_text is None:
            hs_n = self._norm(hs, (D,), w_vis[4], self.rms_eps)
            qkv = self._fp8_gemm(hs_n[0], fp8w[0], fp8w[1]) if _fp8 else hs_n[0] @ w_vis[0].t()
        else:
            nq_v = nq_t = None
            if _fp8p and _wmap_on:
                if pending is not None:
                    hs_out = torch.empty_like(hs)
                    nq_v = self._res_add_norm_to_nvfp4(
                        hs[0, :n_vis], pending[:n_vis], w_vis[4], hs_out[0, :n_vis])
                    nq_t = self._res_add_norm_to_nvfp4(
                        hs[0, n_vis:], pending[n_vis:], w_text[4], hs_out[0, n_vis:])
                    if nq_v is not None and nq_t is not None:
                        hs = hs_out
                    else:
                        hs = hs + pending[None]
                        nq_v = nq_t = None
                if nq_v is None:
                    nq_v = self._norm_to_nvfp4(hs[0, :n_vis], w_vis[4])
                    nq_t = self._norm_to_nvfp4(hs[0, n_vis:], w_text[4])
            if nq_v is not None and nq_t is not None:
                qkv = self._branch_gemm2(nq_v, (fp8v[0], fp8v[1]), nq_t,
                                         (fp8t[0], fp8t[1]), "fp8")
            else:
                if pending is not None:
                    # Fused-residual scheme: the previous layer's down-proj was
                    # returned as `pending` instead of being added in-place, so
                    # the non-quantized (BF16) path must fold it in here before
                    # the input norm, matching the FP8/FP4 `_res_add_norm_to_nvfp4`
                    # producer path.
                    hs = hs + pending[None]
                hs_v = self._norm(hs[0, :n_vis], (D,), w_vis[4], self.rms_eps)
                hs_t = self._norm(hs[0, n_vis:], (D,), w_text[4], self.rms_eps)
                if _fp8p:
                    qkv = self._branch_gemm2(hs_v, (fp8v[0], fp8v[1]), hs_t,
                                             (fp8t[0], fp8t[1]), "fp8")
                else:
                    qkv = torch.cat([hs_v @ w_vis[0].t(), hs_t @ w_text[0].t()], 0)

        if getattr(self, "_fused_attn", False):
            st = torch.cuda.current_stream().cuda_stream
            S_tot = kbuf.shape[2]
            kv_rep = kbuf.shape[1] // nkv       # 4 when the cache is pre-expanded
            q = torch.empty(1, nh, S, hd, dtype=torch.bfloat16, device=hs.device)
            fvk.hyvla_rope_qknorm_kvwrite_parallel_bf16(
                qkv.contiguous().data_ptr(),
                cos.reshape(S, hd).contiguous().data_ptr(),
                sin.reshape(S, hd).contiguous().data_ptr(),
                qk_w[0].data_ptr(), qk_w[1].data_ptr(),
                q.data_ptr(), kbuf.data_ptr(), vbuf.data_ptr(),
                S, nh, nkv, hd, S_tot, off, self.rms_eps, kv_rep, st)
        else:
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], -1)
            q = q.view(S, nh, hd).transpose(0, 1)[None]
            k = k.view(S, nkv, hd).transpose(0, 1)[None]
            v = v.view(S, nkv, hd).transpose(0, 1)[None]

            # RoPE (rotate_half) THEN QK-Norm (RMSNorm over head_dim).
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            q = F.rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = F.rms_norm(k, (hd,), qk_w[1], self.rms_eps)

            if kbuf.shape[1] != nkv:            # pre-expanded cache
                r = kbuf.shape[1] // nkv
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            kbuf[:, :, off:off + S].copy_(k)
            vbuf[:, :, off:off + S].copy_(v)
        k_use = kbuf[:, :, : off + S]
        v_use = vbuf[:, :, : off + S]

        att = self._attn(q, k_use, v_use, mask)
        att = att.transpose(1, 2).reshape(1, S, self.q_dim)

        if w_text is None:
            o = self._fp8_gemm(att[0], fp8w[2], fp8w[3]) if _fp8 else att[0] @ w_vis[1].t()
            hs = hs + o[None]
            if self._ffn_mega and ffn_mk is not None and _fp8:
                hs = self._ffn_mega_bf16(hs, D, w_vis[5], ffn_mk)
            else:
                hs_n = self._norm(hs, (D,), w_vis[5], self.rms_eps)
                gu = self._fp8_gemm(hs_n[0], fp8w[4], fp8w[5]) if _fp8 else hs_n[0] @ w_vis[2].t()
                act = _silu_mul_gu(gu)
                dn = self._fp8_gemm(act, fp8w[6], fp8w[7]) if _fp8 else act @ w_vis[3].t()
                hs = hs + dn[None]
        else:
            if _fp8p:
                o = self._branch_gemm2(att[0, :n_vis], (fp8v[2], fp8v[3]),
                                       att[0, n_vis:], (fp8t[2], fp8t[3]), "fp8")
            else:
                o = torch.cat([att[0, :n_vis] @ w_vis[1].t(),
                               att[0, n_vis:] @ w_text[1].t()], 0)
            nqg_v = nqg_t = None
            _resfn = getattr(self, "_res_add_norm_to_nvfp4", None)
            if (_fp4p or (_fp8p and _wmap_on)) and _resfn is not None:
                # Fused in-place attention residual + per-branch RMSNorm->NVFP4
                # producer: `hs += o` is written by the norm producer, removing
                # the standalone framework add.
                d_ = hs.shape[-1]
                hs2 = hs.reshape(-1, d_)
                o2 = o.reshape(-1, d_)
                nqg_v = _resfn(hs2[:n_vis], o2[:n_vis], w_vis[5], hs2[:n_vis])
                nqg_t = _resfn(hs2[n_vis:], o2[n_vis:], w_text[5], hs2[n_vis:])
                hs = hs2.reshape(hs.shape)
            if nqg_v is None or nqg_t is None:
                hs = hs + o[None]
                nqg_v = self._norm_to_nvfp4(
                    hs[0, :n_vis], w_vis[5]) if (_fp4p or (_fp8p and _wmap_on)) else None
                nqg_t = self._norm_to_nvfp4(
                    hs[0, n_vis:], w_text[5]) if (_fp4p or (_fp8p and _wmap_on)) else None
            if nqg_v is not None and nqg_t is not None:
                if _fp4p:
                    gu = self._branch_gemm2(nqg_v, (fp4v[0], fp4v[1]), nqg_t,
                                            (fp4t[0], fp4t[1]), "fp4", fp4v[4], D)
                else:
                    gu = self._branch_gemm2(nqg_v, (fp8v[4], fp8v[5]), nqg_t,
                                            (fp8t[4], fp8t[5]), "fp8")
            elif _fp4p:
                hs_v = self._norm(hs[0, :n_vis], (D,), w_vis[5], self.rms_eps)
                hs_t = self._norm(hs[0, n_vis:], (D,), w_text[5], self.rms_eps)
                N_gu = fp4v[4]
                gu = self._branch_gemm2(hs_v, (fp4v[0], fp4v[1]), hs_t,
                                        (fp4t[0], fp4t[1]), "fp4", N_gu, D)
            elif _fp8p:
                hs_v = self._norm(hs[0, :n_vis], (D,), w_vis[5], self.rms_eps)
                hs_t = self._norm(hs[0, n_vis:], (D,), w_text[5], self.rms_eps)
                gu = self._branch_gemm2(hs_v, (fp8v[4], fp8v[5]), hs_t,
                                        (fp8t[4], fp8t[5]), "fp8")
            else:
                # Non-quantized path: recompute the attention-residual state's
                # post-attention norm (ln_post = w_vis[5]) for the FFN. The
                # earlier hs_v/hs_t were ln_in (w_vis[4]) on the PRE-attention
                # state and must not be reused here.
                hs_v = self._norm(hs[0, :n_vis], (D,), w_vis[5], self.rms_eps)
                hs_t = self._norm(hs[0, n_vis:], (D,), w_text[5], self.rms_eps)
                gu = torch.cat([hs_v @ w_vis[2].t(), hs_t @ w_text[2].t()], 0)
            if _fp4p:
                inter, Dh = fp4v[5], fp4v[6]
                sm = getattr(self, "_silu_mul_to_nvfp4", None)
                if sm is not None:
                    # Fused SwiGLU -> NVFP4 producer per branch: removes the
                    # bf16 act write+read that the standalone quant would do.
                    nqd_v, nqd_t = sm(gu[:n_vis]), sm(gu[n_vis:])
                    dn = self._branch_gemm2(nqd_v, (fp4v[2], fp4v[3]),
                                            nqd_t, (fp4t[2], fp4t[3]),
                                            "fp4", Dh, inter)
                else:
                    act = _silu_mul_gu(gu)
                    dn = self._branch_gemm2(act[:n_vis], (fp4v[2], fp4v[3]),
                                            act[n_vis:], (fp4t[2], fp4t[3]),
                                            "fp4", Dh, inter)
            else:
                act = _silu_mul_gu(gu)
                if _fp8p:
                    dn = self._branch_gemm2(act[:n_vis], (fp8v[6], fp8v[7]),
                                            act[n_vis:], (fp8t[6], fp8t[7]), "fp8")
                else:
                    dn = torch.cat([act[:n_vis] @ w_vis[3].t(),
                                    act[n_vis:] @ w_text[3].t()], 0)
            if return_dn:
                return hs, dn
            hs = hs + dn[None]
        return hs

    @torch.no_grad()
    def prefill(self, prefix_embs, n_vis, pmask, pcos, psin, kbuf, vbuf):
        """Run the 32-layer MoT VLM tower; fills kbuf/vbuf rows [0:S_p]."""
        hs = prefix_embs
        pending = None
        for li in range(32):
            text, vis = self._vlm_w(li)
            qk = (self.W._qk_norm_q[li], self.W._qk_norm_k[li])
            fp8v, fp8t = self._vlm_w_fp8(li)
            fp4v, fp4t = self._vlm_w_fp4(li)
            hs, pending = self._block(hs, n_vis, text, vis, qk, pmask, pcos,
                                      psin, kbuf[li], vbuf[li], 0, fp8v=fp8v,
                                      fp8t=fp8t, fp4v=fp4v, fp4t=fp4t,
                                      pending=pending, return_dn=True)
        if pending is not None:
            fvk.residual_add(hs.data_ptr(), pending.data_ptr(), hs.numel(),
                             torch.cuda.current_stream().cuda_stream)
        return hs

def _gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """In-place erf-GELU via the native kernel (framework-free hot path);
    falls back to F.gelu when the kernel is absent or x is non-contiguous."""
    if hasattr(fvk, "gelu_erf_bf16") and x.is_contiguous():
        fvk.gelu_erf_bf16(x.data_ptr(), x.numel(),
                          torch.cuda.current_stream().cuda_stream)
        return x
    return F.gelu(x)

def _silu(x: torch.Tensor) -> torch.Tensor:
    """In-place SiLU via the native kernel (framework-free hot path)."""
    if hasattr(fvk, "silu_bf16") and x.is_contiguous():
        fvk.silu_bf16(x.data_ptr(), x.numel(),
                      torch.cuda.current_stream().cuda_stream)
        return x
    return F.silu(x)

def _silu_mul_gu(gu: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up over a (S, 2D) [gate|up] tensor -> (S, D) bf16.

    Uses the native fused ``silu_mul_merged_bf16`` kernel when present (one
    launch, bit-exact with the PyTorch ``F.silu(g) * u`` composite); otherwise
    falls back to the two-op torch path."""
    if hasattr(fvk, "silu_mul_merged_bf16"):
        S, full = gu.shape
        D = full // 2
        gu = gu.contiguous()
        out = torch.empty(S, D, dtype=torch.bfloat16, device=gu.device)
        fvk.silu_mul_merged_bf16(gu.data_ptr(), out.data_ptr(), S, D,
                                 torch.cuda.current_stream().cuda_stream)
        return out
    g, u = gu.chunk(2, -1)
    return F.silu(g) * u

__all__ = ["HyVLARTXBF16Pipeline"]
