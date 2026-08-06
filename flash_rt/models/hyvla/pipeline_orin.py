"""HyVLA forward path for Jetson Orin SM87.

The BF16 math is inherited from the Thor correctness path. When enabled, the
GEMM slots that Thor names ``fp8`` are backed by SM87 INT8 W8A8 rowwise kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:  # pragma: no cover - torch < 2.2
    sdpa_kernel = None
    SDPBackend = None

import flash_rt.flash_rt_kernels as fvk
import flash_rt.models.hyvla.pipeline_thor as _thor_pipeline
from flash_rt.models.hyvla.pipeline_thor import HyVLAThorBF16Pipeline, _rot_half


if not hasattr(_thor_pipeline.F, "rms_norm"):
    def _rms_norm_torch(x, weight, eps):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        if weight is not None:
            y = y * weight.float()
        return y.to(x.dtype)

    def _rms_norm(x, normalized_shape, weight=None, eps=1e-5):
        # torch 2.3 lacks F.rms_norm. The single-launch fvk.rms_norm kernel
        # implements the exact reference math (fp32 sum-of-squares -> rsqrt ->
        # weight multiply -> bf16 rounding) and is bit-equal to the torch
        # expansion on contiguous rows; fall back for anything exotic.
        if (x.is_cuda and x.dtype == torch.bfloat16 and weight is not None
                and weight.dtype == torch.bfloat16
                and len(normalized_shape) == 1
                and normalized_shape[0] == x.shape[-1]):
            xc = x if x.is_contiguous() else x.contiguous()
            wc = weight if weight.is_contiguous() else weight.contiguous()
            rows = xc.numel() // xc.shape[-1]
            out = torch.empty_like(xc)
            fvk.rms_norm(xc.data_ptr(), wc.data_ptr(), out.data_ptr(),
                         rows, xc.shape[-1], eps,
                         torch.cuda.current_stream().cuda_stream)
            return out.reshape(x.shape)
        return _rms_norm_torch(x, weight, eps)

    _thor_pipeline.F.rms_norm = _rms_norm


class _W8Int8:
    """Per-output-row INT8 weight + FP32 scale pair for the ViT INT8 path."""

    __slots__ = ("w", "s")

    def __init__(self, w, s):
        self.w = w
        self.s = s


def _vit_int8_linear(x, w8, bias):
    """x (..., K) bf16 @ W8_int8.T -> (..., N) bf16 + bias via SM87 rowwise INT8."""
    wq, ws = w8.w, w8.s
    N, K = wq.shape
    orig = x.shape
    xc = x.reshape(-1, K).contiguous()
    M = xc.shape[0]
    st = torch.cuda.current_stream().cuda_stream
    a8 = torch.empty(M, K, dtype=torch.int8, device=x.device)
    act_scale = torch.empty(M, dtype=torch.float32, device=x.device)
    fvk.quantize_int8_rowwise(
        xc.data_ptr(), a8.data_ptr(), act_scale.data_ptr(), M, K, st)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    status = fvk.cutlass_int8_rowwise_bf16out(
        a8.data_ptr(), wq.data_ptr(), act_scale.data_ptr(), ws.data_ptr(),
        out.data_ptr(), M, N, K, st)
    if status != 0:
        raise RuntimeError(
            f"cutlass_int8_rowwise_bf16out failed: status={status} "
            f"shape=({M},{N},{K})")
    out = out + bias
    return out.reshape(*orig[:-1], N)


class HyVLAOrinBF16Pipeline(HyVLAThorBF16Pipeline):
    def enable_fp8(self):
        raise RuntimeError(
            "HyVLA Orin does not support Thor FP8 GEMMs; use enable_int8().")

    def enable_fp4(self):
        raise RuntimeError(
            "HyVLA Orin does not support FP4 because SM87 has no native FP4 tensor cores.")

    def enable_int8(self):
        self._fp8 = True
        self._orin_int8 = True
        self.gemm = None

    def autotune_gemms(self, shapes, num_algos=16):
        return 0

    def _attn(self, q, k, v, mask):
        out = super()._attn(q, k, v, mask)
        return torch.nan_to_num(out)

    # ------------------------------------------------------------------
    #  ViT INT8 GEMM sites (SM87). Weight lists are replaced in-place by
    #  the frontend with _W8Int8 wrappers when use_int8_vit is enabled;
    #  otherwise the parent BF16 F.linear path is kept.
    # ------------------------------------------------------------------
    def _vit_qkv(self, h):
        if isinstance(self._vit_qkv_w_cur, _W8Int8):
            qkv = _vit_int8_linear(h, self._vit_qkv_w_cur, self._vit_qkv_b_cur)
            bk, N, _ = h.shape
            qkv = qkv.reshape(bk, N, 3, self.vit_heads, self.vit_hd).permute(2, 0, 3, 1, 4)
            return qkv[0], qkv[1], qkv[2]
        return super()._vit_qkv(h)

    def _vit_spatial_attn(self, q, k, v):
        bk, _, N, _ = q.shape
        if getattr(self, "_vit_eff_sdpa", False):
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
        if isinstance(self._vit_proj_w_cur, _W8Int8):
            return _vit_int8_linear(out, self._vit_proj_w_cur, self._vit_proj_b_cur)
        return F.linear(out, self._vit_proj_w_cur, self._vit_proj_b_cur)

    def _vit_mlp(self, x):
        if isinstance(self._vit_fc1_w_cur, _W8Int8):
            x = _vit_int8_linear(x, self._vit_fc1_w_cur, self._vit_fc1_b_cur)
            x = F.gelu(x)
            return _vit_int8_linear(x, self._vit_fc2_w_cur, self._vit_fc2_b_cur)
        return super()._vit_mlp(x)

    # ------------------------------------------------------------------
    #  Fused ViT forward: residual-add + LayerNorm pairs collapse into one
    #  hyvla_vit_add_layer_norm_bf16 launch. The previous block's MLP
    #  output is carried as ``pending`` and fused into the next block's
    #  entry LN; the post-attention add is fused into the pre-MLP LN.
    #  Spacetime blocks keep the torch path (their entry LN also adds the
    #  time positional embedding).
    # ------------------------------------------------------------------
    def _vit_add_ln(self, x, add, lnw, lnb):
        bk, n, d = x.shape
        out = torch.empty_like(x)
        fvk.hyvla_vit_add_layer_norm_bf16(
            x.data_ptr(), add.data_ptr(), lnw.data_ptr(), lnb.data_ptr(),
            out.data_ptr(), bk * n, d, self.vit_eps,
            torch.cuda.current_stream().cuda_stream)
        return out

    def _vit_ln(self, x, lnw, lnb):
        bk, n, d = x.shape
        out = torch.empty_like(x)
        fvk.layer_norm(x.data_ptr(), lnw.data_ptr(), lnb.data_ptr(),
                       out.data_ptr(), bk * n, d, self.vit_eps,
                       torch.cuda.current_stream().cuda_stream)
        return out

    @torch.no_grad()
    def vit_forward(self, imgs):
        if not getattr(self, "_vit_fuse_ln", False):
            return super().vit_forward(imgs)
        W = self.W
        num_cam, K = imgs.shape[0], imgs.shape[1]
        bk = num_cam * K
        x = imgs.reshape(bk, 3, 224, 224)
        x = F.conv2d(x, W._vit_patch_w, W._vit_patch_b, stride=16)
        hh = ww = x.shape[-1]
        n = hh * ww
        d = x.shape[1]
        x = x.flatten(2).transpose(1, 2)
        x = (x + self._vit_pos_embed_rescale(hh, ww, x.dtype)).contiguous()
        last_st = max(self.vit_spacetime_ids) if K > 1 else -1
        sliced = False
        pending = None
        for li in range(27):
            Wcur = self.W
            self._vit_qkv_w_cur = Wcur._vit_qkv_w[li]; self._vit_qkv_b_cur = Wcur._vit_qkv_b[li]
            self._vit_proj_w_cur = Wcur._vit_proj_w[li]; self._vit_proj_b_cur = Wcur._vit_proj_b[li]
            self._vit_fc1_w_cur = Wcur._vit_fc1_w[li]; self._vit_fc1_b_cur = Wcur._vit_fc1_b[li]
            self._vit_fc2_w_cur = Wcur._vit_fc2_w[li]; self._vit_fc2_b_cur = Wcur._vit_fc2_b[li]
            self._vit_f8 = self._fp8 and getattr(Wcur, "_vit_fp8_ready", False)
            if self._vit_f8:
                self._vit_qkv_w8c = Wcur._vit_qkv_w8[li]; self._vit_qkv_wsc = Wcur._vit_qkv_ws[li]
                self._vit_proj_w8c = Wcur._vit_proj_w8[li]; self._vit_proj_wsc = Wcur._vit_proj_ws[li]
                self._vit_fc1_w8c = Wcur._vit_fc1_w8[li]; self._vit_fc1_wsc = Wcur._vit_fc1_ws[li]
                self._vit_fc2_w8c = Wcur._vit_fc2_w8[li]; self._vit_fc2_wsc = Wcur._vit_fc2_ws[li]
            ln1w, ln1b = Wcur._vit_ln1_w[li], Wcur._vit_ln1_b[li]
            ln2w, ln2b = Wcur._vit_ln2_w[li], Wcur._vit_ln2_b[li]

            if li in self.vit_spacetime_ids and K > 1:
                if pending is not None:
                    x = x + pending
                    pending = None
                b, kf = bk // K, K
                pe = self._vit_time_pe(kf, x.device, x.dtype)
                h = F.layer_norm(x.view(b, kf, n, d) + pe.view(1, kf, 1, d),
                                 (d,), ln1w, ln1b, self.vit_eps).view(bk, n, d)
                q, k, v = self._vit_qkv(h)
                v = self._vit_time_mix(q, k, v, b, kf)
                attn_out = self._vit_spatial_attn(q, k, v)
            else:
                if pending is not None:
                    h = self._vit_add_ln(x, pending, ln1w, ln1b)
                else:
                    h = self._vit_ln(x, ln1w, ln1b)
                q, k, v = self._vit_qkv(h)
                attn_out = self._vit_spatial_attn(q, k, v)

            h2 = self._vit_add_ln(x, attn_out, ln2w, ln2b)
            pending = self._vit_mlp(h2)
            if li == last_st and li < 26:
                x = (x + pending).contiguous()
                pending = None
                x = x.view(num_cam, K, n, d)[:, -1].contiguous()
                sliced = True
        if pending is not None:
            x = x + pending
        if not sliced:
            x = x.view(num_cam, K, n, d)[:, -1]
        return x

    def _int8_rowwise_gemm(self, x, wq, ws):
        """x (..., K) bf16 -> (M, N) bf16 via dynamic per-row INT8 W8A8."""
        N, K = wq.shape
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        a8 = torch.empty(M, K, dtype=torch.int8, device=x.device)
        act_scale = torch.empty(M, dtype=torch.float32, device=x.device)
        fvk.quantize_int8_rowwise(
            xc.data_ptr(), a8.data_ptr(), act_scale.data_ptr(), M, K, st)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        status = fvk.cutlass_int8_rowwise_bf16out(
            a8.data_ptr(), wq.data_ptr(), act_scale.data_ptr(), ws.data_ptr(),
            out.data_ptr(), M, N, K, st)
        if status != 0:
            raise RuntimeError(
                f"cutlass_int8_rowwise_bf16out failed: status={status} "
                f"shape=({M},{N},{K})")
        return out

    def _fp8_gemm(self, x, w8, ws):
        if not getattr(self, "_orin_int8", False):
            raise RuntimeError("HyVLA Orin lower-precision GEMM requires enable_int8().")
        return self._int8_rowwise_gemm(x, w8, ws)

    # ------------------------------------------------------------------
    #  Prefill FFN-only INT8: QKV / O stay BF16 (their outputs feed the KV
    #  cache read by 10 denoise steps), gate/up/down run INT8 W8A8.
    # ------------------------------------------------------------------
    def _block_ffn8(self, hs, n_vis, w_text, w_vis, qk_w, mask, cos, sin,
                    kbuf, vbuf, off, ffnv, ffnt):
        S = hs.shape[1]
        D = hs.shape[2]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv

        hs_v = F.rms_norm(hs[0, :n_vis], (D,), w_vis[4], self.rms_eps)
        hs_t = F.rms_norm(hs[0, n_vis:], (D,), w_text[4], self.rms_eps)
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
            q = F.rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = F.rms_norm(k, (hd,), qk_w[1], self.rms_eps)
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

        hs_v = F.rms_norm(hs[0, :n_vis], (D,), w_vis[5], self.rms_eps)
        hs_t = F.rms_norm(hs[0, n_vis:], (D,), w_text[5], self.rms_eps)
        gu = torch.cat([self._int8_rowwise_gemm(hs_v, ffnv[0], ffnv[1]),
                        self._int8_rowwise_gemm(hs_t, ffnt[0], ffnt[1])], 0)
        g, u = gu.chunk(2, -1)
        act = F.silu(g) * u
        dn = torch.cat([self._int8_rowwise_gemm(act[:n_vis], ffnv[2], ffnv[3]),
                        self._int8_rowwise_gemm(act[n_vis:], ffnt[2], ffnt[3])], 0)
        return hs + dn[None]

    @torch.no_grad()
    def prefill(self, prefix_embs, n_vis, pmask, pcos, psin, kbuf, vbuf):
        if getattr(self, "_vlm_ffn_int8", False):
            W = self.W
            hs = prefix_embs
            for li in range(32):
                text, vis = self._vlm_w(li)
                qk = (W._qk_norm_q[li], W._qk_norm_k[li])
                ffnv = (W._vlm_gu_v8[li], W._vlm_gu_v_ws[li],
                        W._vlm_d_v8[li], W._vlm_d_v_ws[li])
                ffnt = (W._vlm_gu_t8[li], W._vlm_gu_t_ws[li],
                        W._vlm_d_t8[li], W._vlm_d_t_ws[li])
                hs = self._block_ffn8(hs, n_vis, text, vis, qk, pmask,
                                      pcos, psin, kbuf[li], vbuf[li], 0,
                                      ffnv, ffnt)
            return hs
        return super().prefill(prefix_embs, n_vis, pmask, pcos, psin, kbuf, vbuf)

    # ------------------------------------------------------------------
    #  Expert denoise with fused residual-add + RMSNorm. The down-proj
    #  output of layer N is carried as ``pending`` and fused into layer
    #  N+1's input norm (and into the final norm after layer 31), turning
    #  (add + norm) pairs into single launches across all 32x10 blocks.
    # ------------------------------------------------------------------
    def _res_add_rms_norm(self, residual, x, weight):
        rows = residual.shape[-2]
        dim = residual.shape[-1]
        out = torch.empty_like(residual)
        fvk.residual_add_rms_norm(
            residual.data_ptr(), x.data_ptr(), weight.data_ptr(),
            out.data_ptr(), rows, dim, self.rms_eps,
            torch.cuda.current_stream().cuda_stream)
        return out

    def _rope_qknorm_kvwrite(self, qkv, cos, sin, qk_w, q, kbuf, vbuf, S, off):
        """Fused RoPE(q,k)+QK-Norm+KV-write (1 launch). ``qkv`` must be a
        contiguous (S, (nq+2*nkv)*hd) bf16 tensor; writes ``q`` and the
        GQA-pre-expanded KV cache rows at ``off``."""
        hd = self.head_dim
        S_tot = kbuf.shape[2]
        kv_rep = kbuf.shape[1] // self.n_kv
        fvk.hyvla_rope_qknorm_kvwrite_bf16(
            qkv.data_ptr(),
            cos.reshape(S, hd).contiguous().data_ptr(),
            sin.reshape(S, hd).contiguous().data_ptr(),
            qk_w[0].data_ptr(), qk_w[1].data_ptr(),
            q.data_ptr(), kbuf.data_ptr(), vbuf.data_ptr(),
            S, self.n_heads, self.n_kv, hd, S_tot, off, self.rms_eps,
            kv_rep, torch.cuda.current_stream().cuda_stream)

    def _exp_block_r(self, hs, w, qk_w, mask, cos, sin, kbuf, vbuf, off,
                     fp8w, pending):
        S = hs.shape[1]
        D = hs.shape[2]
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv

        if pending is None:
            hs_n = F.rms_norm(hs, (D,), w[4], self.rms_eps)
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
            q = F.rms_norm(q, (hd,), qk_w[0], self.rms_eps)
            k = F.rms_norm(k, (hd,), qk_w[1], self.rms_eps)
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
        g, u = gu.chunk(2, -1)
        act = F.silu(g) * u
        dn = self._int8_rowwise_gemm(act, fp8w[6], fp8w[7])
        return hs, dn[None]

    @torch.no_grad()
    def denoise(self, state, x_t, time_embs, smask, scos, ssin,
                kbuf, vbuf, S_p, num_steps=10):
        W = self.W
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
            ate = F.silu(ate)
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


__all__ = ["HyVLAOrinBF16Pipeline"]
