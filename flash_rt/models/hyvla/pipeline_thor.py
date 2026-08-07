"""Hy-Embodied-0.5-VLA forward path on Thor SM110 (BF16 baseline).

This module owns the model-specific forwards (repo contract §0 rule 1/3 —
they may NOT live in ``hardware/thor/shared_primitives.py``):

  * ``vit_forward``      — 27-block HYViT2 incl. the 6 spacetime blocks,
                           per camera, over 6 history frames.
  * ``merger_forward``   — proj1 → 2x2 NormalizedDwPooler → GELU → proj2.
  * ``prefill_forward``  — 32-layer MoT VLM tower over the sorted
                           ``[vision|text]`` prefix; fills the per-layer
                           KV cache the denoise loop reads.
  * ``denoise_forward``  — 32-layer expert tower (``_v`` only) + action
                           head, 10 flow-matching Euler steps.

The BF16 baseline is **correctness-first**: heavy GEMMs are plain
``torch`` bf16 matmuls and attention is ``F.scaled_dot_product_attention``
(the numerically-exact reference path, GQA 16/4 + materialized mask).
The math mirrors the validated reference implementation
(action-chunk cosine ≥ 0.999 vs the HF reference eager path). The
optimized path swaps the GEMMs/norms for ``fvk`` pointer kernels, the
two masked-attention bodies for fused kernels, and captures one CUDA
graph — the forward signatures and the ``[vision|text]`` static-routing
layout are chosen so that swap is local (no structural change). No
external training code is imported.

Key model constants (verified from the checkpoint + reference source):
  D_vlm=2048, D_exp=1024, n_heads=16, n_kv=4, head_dim=128,
  q_dim=2048, kv_dim=512, inter_vlm=6144, inter_exp=2048,
  rms_eps=1e-5 (all towers + QK-norm + final norm),
  chunk=40, num_steps=10, dt=-1/steps, t: 1.0→0.1.
  QK-Norm (RMSNorm over head_dim) is applied AFTER RoPE (rotate_half).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class HyVLAThorBF16Pipeline:
    """BF16 forward for the Hy-VLA dual-tower + action head.

    Weights are read from ``W`` (the frontend), which carries the
    declarative-spec attributes (``_vlm_qkv_t`` lists, ``_ain_w`` etc.).
    Buffers (KV cache, x_t, time embeddings) are pre-allocated by the
    frontend and passed in, so the forward performs no Python-level
    allocation — the graph-capture safety precondition.
    """

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

    # ══════════════════════════════════════════════════════════════════
    #  FP8 (dynamic per-tensor, graph-safe) — expert denoise GEMMs
    # ══════════════════════════════════════════════════════════════════
    def enable_fp8(self):
        from flash_rt.core.context import FvkContext
        self.gemm = FvkContext().gemm
        self._fp8 = True

    def enable_fp4(self):
        import flash_rt.flash_rt_fp4 as _F4
        self._F4 = _F4
        self._fp4 = True

    def autotune_gemms(self, shapes, num_algos=16):
        """Per-shape cuBLASLt FP8 algo autotune (motus pattern). Runs BEFORE
        graph capture on ``self.gemm`` — the SAME GemmRunner _fp8_gemm calls, so
        the tuned algo is cached (keyed on (M,N,K)) and every captured
        ``fp8_nn_dev`` for that shape picks it up automatically. Dummy buffers:
        only the algo is timed; real scale pointers are set per call."""
        if self.gemm is None or not hasattr(self.gemm, "autotune_fp8_nn_dev"):
            return 0
        dev = "cuda"
        n = 0
        for (M, N, K) in sorted(shapes):
            A = torch.empty(M, K, dtype=torch.uint8, device=dev)
            B = torch.empty(K, N, dtype=torch.uint8, device=dev)
            D = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
            sa = torch.ones(1, dtype=torch.float32, device=dev)
            sb = torch.ones(1, dtype=torch.float32, device=dev)
            self.gemm.autotune_fp8_nn_dev(A.data_ptr(), B.data_ptr(), D.data_ptr(),
                                          M, N, K, sa.data_ptr(), sb.data_ptr(), num_algos)
            n += 1
        torch.cuda.synchronize()
        return n

    def _fp4_gemm_f4(self, x, w4, wsf, N, K):
        """F4-family W4A4 GEMM (Thor). x (M,K) bf16 -> (M,N) bf16.

        Activation is dynamically quantized to NVFP4 with the SAME swizzled SF
        family as the weight (both from flash_rt_fp4), then cutlass_fp4_sq_fp16
        (A@Bᵀ, weight stored (N,K/2)+SF). Verified cos 0.990/GEMM. Graph-safe."""
        F4 = self._F4
        st = torch.cuda.current_stream().cuda_stream
        xc = x.reshape(-1, K).contiguous().to(torch.float16)
        M = xc.shape[0]
        xp = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)
        xsf = torch.empty(F4.sfa_size_bytes(M, K, False), dtype=torch.uint8, device=x.device)
        F4.quantize_fp4_dynamic_sfa_fp16(xc.data_ptr(), xp.data_ptr(), xsf.data_ptr(), M, K, False, st)
        out = torch.empty(M, N, dtype=torch.float16, device=x.device)
        F4.cutlass_fp4_sq_fp16(xp.data_ptr(), xsf.data_ptr(), w4.data_ptr(), wsf.data_ptr(),
                               out.data_ptr(), M, N, K, 1.0, 0.0, st)
        return out.to(torch.bfloat16)

    def _fp8_gemm(self, x, w8, ws):
        """x (M,K) bf16 -> (M,N) bf16 via dynamic-scale FP8 (graph-safe).

        w8 is the fp8 weight stored (K,N); activation amax is computed on-GPU
        each call (device scale). cuBLASLt device-scale FP8 GEMM is
        CUDA-graph-capturable on Thor (verified)."""
        K, N = w8.shape
        xc = x.reshape(-1, K).contiguous()
        M = xc.shape[0]
        if self._gemm_shapes is not None:
            self._gemm_shapes.add((M, N, K))
        st = torch.cuda.current_stream().cuda_stream
        a8 = torch.empty(M, K, dtype=torch.uint8, device=x.device)
        dsa = torch.empty(1, dtype=torch.float32, device=x.device)
        if 0 < self._small_quant_m and M <= self._small_quant_m:
            fvk.hyvla_quant_fp8_dyn_bf16(xc.data_ptr(), a8.data_ptr(),
                                         dsa.data_ptr(), M * K, st)
        else:
            fvk.quantize_fp8_device(xc.data_ptr(), a8.data_ptr(), dsa.data_ptr(), M * K, st)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        self.gemm.fp8_nn_dev(a8.data_ptr(), w8.data_ptr(), out.data_ptr(),
                             M, N, K, dsa.data_ptr(), ws.data_ptr(), st)
        return out

    def _fp8_gemm_bias(self, x, w8, ws, bias):
        """FP8 GEMM (…,K)->(…,N) + bias, for the biased ViT projections."""
        orig = x.shape
        out = self._fp8_gemm(x.reshape(-1, orig[-1]), w8, ws)   # (M,N) bf16
        out = out + bias
        return out.reshape(*orig[:-1], out.shape[-1])

    def _ffn_mega_bf16(self, hs_post, D, norm_w, mk):
        """Expert denoise FFN via the two persistent megakernels.

        hs_post (1,S_s,D) bf16 is the post-attention residual stream (input to
        the FFN AND the residual). mk = (gu8, sgu, dn8, sdn). Returns the new
        hidden (1,S_s,D). Dynamic FP8: activation amax on-GPU each call
        (graph-safe); weight scale is the baked host float. Scratch tensors go
        to the graph private pool (per-call torch.empty, alias-safe)."""
        gu8, sgu, dn8, sdn = mk
        dev = hs_post.device
        st = torch.cuda.current_stream().cuda_stream
        S_s = hs_post.shape[1]
        Nout = gu8.shape[0] // 2         # inter (gate+up merged -> inter)
        INTER = dn8.shape[1]             # dn K
        hs_n = F.rms_norm(hs_post, (D,), norm_w, self.rms_eps)[0].contiguous()
        x8 = torch.empty(S_s, D, dtype=torch.uint8, device=dev)
        sx = torch.empty(1, dtype=torch.float32, device=dev)
        fvk.quantize_fp8_device(hs_n.data_ptr(), x8.data_ptr(), sx.data_ptr(), S_s * D, st)
        act = torch.empty(S_s, Nout, dtype=torch.bfloat16, device=dev)
        fvk.hyvla_ffn_gu_silu_bf16(x8.data_ptr(), gu8.data_ptr(), act.data_ptr(),
                                   S_s, D, Nout, sx.data_ptr(), sgu, st)
        a8 = torch.empty(S_s, INTER, dtype=torch.uint8, device=dev)
        sa = torch.empty(1, dtype=torch.float32, device=dev)
        fvk.quantize_fp8_device(act.data_ptr(), a8.data_ptr(), sa.data_ptr(), S_s * INTER, st)
        y = torch.empty(S_s, D, dtype=torch.bfloat16, device=dev)
        fvk.hyvla_ffn_dn_res_bf16(a8.data_ptr(), dn8.data_ptr(), hs_post[0].contiguous().data_ptr(),
                                  y.data_ptr(), S_s, INTER, D, sa.data_ptr(), sdn, st)
        return y[None]

    # ══════════════════════════════════════════════════════════════════
    #  ViT (vision tower) + merger — BF16
    # ══════════════════════════════════════════════════════════════════
    def _vit_qkv(self, h):
        """h (bk, N, 1152) -> q,k,v each (bk, heads, N, 72)."""
        bk, N, _ = h.shape
        if getattr(self, "_vit_f8", False):
            qkv = self._fp8_gemm_bias(h, self._vit_qkv_w8c, self._vit_qkv_wsc, self._vit_qkv_b_cur)
        else:
            qkv = F.linear(h, self._vit_qkv_w_cur, self._vit_qkv_b_cur)
        qkv = qkv.reshape(bk, N, 3, self.vit_heads, self.vit_hd).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _vit_spatial_attn(self, q, k, v):
        """(bk, heads, N, 72) full non-causal attention -> proj. (bk, N, 1152)."""
        bk, _, N, _ = q.shape
        out = F.scaled_dot_product_attention(q, k, v, scale=self.vit_scale)
        out = out.transpose(1, 2).reshape(bk, N, -1)
        if getattr(self, "_vit_f8", False):
            return self._fp8_gemm_bias(out, self._vit_proj_w8c, self._vit_proj_wsc, self._vit_proj_b_cur)
        return F.linear(out, self._vit_proj_w_cur, self._vit_proj_b_cur)

    def _vit_time_pe(self, kf, device, dtype):
        """Fixed sinusoidal e(t), base 100, e(0)=0. (kf, 1152)."""
        dim = self.vit_heads * self.vit_hd
        t = torch.arange(kf, dtype=torch.float32, device=device).unsqueeze(1)
        inv_freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                             * (-torch.log(torch.tensor(self.vit_time_base)) / dim))
        pe = torch.empty(kf, dim, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(t * inv_freq)
        pe[:, 1::2] = torch.cos(t * inv_freq) - 1.0
        return pe.to(dtype)

    def _vit_time_mix(self, q, k, v, b, kf):
        """Causal-in-time softmax over K frames folded onto V. (bk,H,N,d)."""
        bk, heads, n, d = v.shape
        rs = lambda t: t.view(b, kf, heads, n, d).permute(0, 3, 2, 1, 4).reshape(b * n, heads, kf, d)
        q_t, k_t, v_t = rs(q), rs(k), rs(v)
        scores = (q_t @ k_t.transpose(-2, -1)) * self.vit_scale
        mask = torch.triu(torch.ones(kf, kf, device=scores.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        vm = scores.softmax(dim=-1).to(v_t.dtype) @ v_t
        return vm.view(b, n, heads, kf, d).permute(0, 3, 2, 1, 4).reshape(bk, heads, n, d)

    def _vit_mlp(self, x):
        if getattr(self, "_vit_f8", False):
            x = self._fp8_gemm_bias(x, self._vit_fc1_w8c, self._vit_fc1_wsc, self._vit_fc1_b_cur)
            x = F.gelu(x)
            return self._fp8_gemm_bias(x, self._vit_fc2_w8c, self._vit_fc2_wsc, self._vit_fc2_b_cur)
        x = F.linear(x, self._vit_fc1_w_cur, self._vit_fc1_b_cur)
        x = F.gelu(x)
        return F.linear(x, self._vit_fc2_w_cur, self._vit_fc2_b_cur)

    def _vit_block(self, x, li, num_frames):
        """One ViT block; spacetime when li in vit_spacetime_ids."""
        W = self.W
        self._vit_qkv_w_cur = W._vit_qkv_w[li]; self._vit_qkv_b_cur = W._vit_qkv_b[li]
        self._vit_proj_w_cur = W._vit_proj_w[li]; self._vit_proj_b_cur = W._vit_proj_b[li]
        self._vit_fc1_w_cur = W._vit_fc1_w[li]; self._vit_fc1_b_cur = W._vit_fc1_b[li]
        self._vit_fc2_w_cur = W._vit_fc2_w[li]; self._vit_fc2_b_cur = W._vit_fc2_b[li]
        self._vit_f8 = self._fp8 and getattr(W, "_vit_fp8_ready", False)
        if self._vit_f8:
            self._vit_qkv_w8c = W._vit_qkv_w8[li]; self._vit_qkv_wsc = W._vit_qkv_ws[li]
            self._vit_proj_w8c = W._vit_proj_w8[li]; self._vit_proj_wsc = W._vit_proj_ws[li]
            self._vit_fc1_w8c = W._vit_fc1_w8[li]; self._vit_fc1_wsc = W._vit_fc1_ws[li]
            self._vit_fc2_w8c = W._vit_fc2_w8[li]; self._vit_fc2_wsc = W._vit_fc2_ws[li]
        ln1w, ln1b = W._vit_ln1_w[li], W._vit_ln1_b[li]
        ln2w, ln2b = W._vit_ln2_w[li], W._vit_ln2_b[li]
        bk, n, d = x.shape

        if li in self.vit_spacetime_ids and num_frames > 1:
            b, kf = bk // num_frames, num_frames
            pe = self._vit_time_pe(kf, x.device, x.dtype)
            h = F.layer_norm(x.view(b, kf, n, d) + pe.view(1, kf, 1, d),
                             (d,), ln1w, ln1b, self.vit_eps).view(bk, n, d)
            q, k, v = self._vit_qkv(h)
            v = self._vit_time_mix(q, k, v, b, kf)
            attn_out = self._vit_spatial_attn(q, k, v)
        else:
            h = F.layer_norm(x, (d,), ln1w, ln1b, self.vit_eps)
            q, k, v = self._vit_qkv(h)
            attn_out = self._vit_spatial_attn(q, k, v)

        x = x + attn_out
        x = x + self._vit_mlp(F.layer_norm(x, (d,), ln2w, ln2b, self.vit_eps))
        return x

    def _vit_pos_embed_rescale(self, h, w, dtype):
        """Bilinear-rescale learned pos_embed (128x128) to (h,w). (1, h*w, 1152)."""
        pos = self.W._vit_pos_embed  # (1, 16384, 1152)
        g = int(pos.shape[1] ** 0.5)  # 128
        if (h, w) == (g, g):
            return pos
        pe2d = pos[0].T.contiguous().view(1, -1, g, g).float()
        pe2d = F.interpolate(pe2d, (h, w), mode="bilinear", align_corners=False)
        return pe2d.view(-1, h * w).T.contiguous()[None].to(dtype)

    @torch.no_grad()
    def vit_forward(self, imgs):
        """imgs (num_cam, K, 3, 224, 224) bf16 in [-1,1] -> (num_cam, 196, 1152)."""
        W = self.W
        num_cam, K = imgs.shape[0], imgs.shape[1]
        bk = num_cam * K
        x = imgs.reshape(bk, 3, 224, 224)
        x = F.conv2d(x, W._vit_patch_w, W._vit_patch_b, stride=16)  # (bk,1152,14,14)
        hh = ww = x.shape[-1]
        x = x.flatten(2).transpose(1, 2)                            # (bk,196,1152)
        x = x + self._vit_pos_embed_rescale(hh, ww, x.dtype)
        # History frames only feed the spacetime time-mix (causal over K, with
        # the current frame last). After the FINAL spacetime block the remaining
        # blocks are per-frame independent and only the current frame reaches the
        # merger — so drop the history there. Numerically identical, and those
        # blocks then do 1/K of the work.
        last_st = max(self.vit_spacetime_ids) if K > 1 else -1
        sliced = False
        for li in range(27):
            x = self._vit_block(x, li, K)
            if li == last_st and li < 26:
                x = x.view(num_cam, K, hh * ww, -1)[:, -1]          # (num_cam,N,D)
                sliced = True
        if not sliced:
            x = x.view(num_cam, K, hh * ww, -1)[:, -1]              # current frame
        return x

    @torch.no_grad()
    def merger_forward(self, x, grid=14):
        """x (num_cam, 196, 1152) -> (num_cam, 49, 2048). NormalizedDwPooler 2x2."""
        W = self.W
        B = x.shape[0]
        h = w = grid
        x = x.reshape(B, h, w, -1)
        x = F.linear(x, W._mg_proj1_w, W._mg_proj1_b)               # (B,14,14,2048)
        C = x.shape[-1]
        new_x = (x.reshape(B, h // 2, 2, w // 2, 2, C)
                 .permute(0, 1, 3, 2, 4, 5).reshape(B, h // 2, w // 2, 4, C))
        pooled = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
        fused = torch.cat([new_x, pooled], dim=-1)                  # (B,7,7,4,4096)
        score = F.linear(fused, W._mg_pred0_w, W._mg_pred0_b)
        score = F.gelu(score)
        score = F.linear(score, W._mg_pred2_w, W._mg_pred2_b)       # (B,7,7,4,2048)
        x = (new_x * score.softmax(dim=-2)).sum(dim=-2)             # (B,7,7,2048)
        x = F.gelu(x)
        x = F.linear(x, W._mg_proj2_w, W._mg_proj2_b)
        return x.reshape(B, -1, C)

    # ------------------------------------------------------------------
    def _attn(self, q, k, v, mask):
        """GQA attention with a materialized bool mask.

        q (1, n_heads, S, hd) ; k/v (1, n_kv, S_kv, hd) ; mask (1,1,S,S_kv).
        Expand KV heads to n_heads and drop enable_gqa so SDPA can pick the
        memory-efficient backend (the bool-mask + enable_gqa combo forces the
        slow math backend — measured ~41ms of E2E)."""
        if k.shape[1] != q.shape[1]:
            r = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                 torch.nn.attention.SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # ------------------------------------------------------------------
    def _block(self, hs, n_vis, w_text, w_vis, qk_w, mask, cos, sin,
               kbuf, vbuf, off, fp8w=None, fp8v=None, fp8t=None,
               fp4v=None, fp4t=None, ffn_mk=None):
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

        if w_text is None:
            hs_n = F.rms_norm(hs, (D,), w_vis[4], self.rms_eps)
            qkv = self._fp8_gemm(hs_n[0], fp8w[0], fp8w[1]) if _fp8 else hs_n[0] @ w_vis[0].t()
        else:
            hs_v = F.rms_norm(hs[0, :n_vis], (D,), w_vis[4], self.rms_eps)
            hs_t = F.rms_norm(hs[0, n_vis:], (D,), w_text[4], self.rms_eps)
            if _fp8p:
                qkv = torch.cat([self._fp8_gemm(hs_v, fp8v[0], fp8v[1]),
                                 self._fp8_gemm(hs_t, fp8t[0], fp8t[1])], 0)
            else:
                qkv = torch.cat([hs_v @ w_vis[0].t(), hs_t @ w_text[0].t()], 0)

        if getattr(self, "_fused_attn", False):
            st = torch.cuda.current_stream().cuda_stream
            S_tot = kbuf.shape[2]
            kv_rep = kbuf.shape[1] // nkv       # 4 when the cache is pre-expanded
            q = torch.empty(1, nh, S, hd, dtype=torch.bfloat16, device=hs.device)
            fvk.hyvla_rope_qknorm_kvwrite_bf16(
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
                hs_n = F.rms_norm(hs, (D,), w_vis[5], self.rms_eps)
                gu = self._fp8_gemm(hs_n[0], fp8w[4], fp8w[5]) if _fp8 else hs_n[0] @ w_vis[2].t()
                g, u = gu.chunk(2, -1)
                act = F.silu(g) * u
                dn = self._fp8_gemm(act, fp8w[6], fp8w[7]) if _fp8 else act @ w_vis[3].t()
                hs = hs + dn[None]
        else:
            if _fp8p:
                o_v = self._fp8_gemm(att[0, :n_vis], fp8v[2], fp8v[3])
                o_t = self._fp8_gemm(att[0, n_vis:], fp8t[2], fp8t[3])
            else:
                o_v = att[0, :n_vis] @ w_vis[1].t()
                o_t = att[0, n_vis:] @ w_text[1].t()
            hs = hs + torch.cat([o_v, o_t], 0)[None]
            hs_v = F.rms_norm(hs[0, :n_vis], (D,), w_vis[5], self.rms_eps)
            hs_t = F.rms_norm(hs[0, n_vis:], (D,), w_text[5], self.rms_eps)
            if _fp4p:
                N_gu = fp4v[4]
                gu = torch.cat([self._fp4_gemm_f4(hs_v, fp4v[0], fp4v[1], N_gu, D),
                                self._fp4_gemm_f4(hs_t, fp4t[0], fp4t[1], N_gu, D)], 0)
            elif _fp8p:
                gu = torch.cat([self._fp8_gemm(hs_v, fp8v[4], fp8v[5]),
                                self._fp8_gemm(hs_t, fp8t[4], fp8t[5])], 0)
            else:
                gu = torch.cat([hs_v @ w_vis[2].t(), hs_t @ w_text[2].t()], 0)
            g, u = gu.chunk(2, -1)
            act = F.silu(g) * u
            if _fp4p:
                inter, Dh = fp4v[5], fp4v[6]
                dn = torch.cat([self._fp4_gemm_f4(act[:n_vis], fp4v[2], fp4v[3], Dh, inter),
                                self._fp4_gemm_f4(act[n_vis:], fp4t[2], fp4t[3], Dh, inter)], 0)
            elif _fp8p:
                dn = torch.cat([self._fp8_gemm(act[:n_vis], fp8v[6], fp8v[7]),
                                self._fp8_gemm(act[n_vis:], fp8t[6], fp8t[7])], 0)
            else:
                dn = torch.cat([act[:n_vis] @ w_vis[3].t(),
                                act[n_vis:] @ w_text[3].t()], 0)
            hs = hs + dn[None]
        return hs

    # ------------------------------------------------------------------
    def _vlm_w(self, li):
        W = self.W
        text = (W._vlm_qkv_t[li], W._vlm_o_t[li], W._vlm_gu_t[li],
                W._vlm_d_t[li], W._vlm_ln_in_t[li], W._vlm_ln_post_t[li])
        vis = (W._vlm_qkv_v[li], W._vlm_o_v[li], W._vlm_gu_v[li],
               W._vlm_d_v[li], W._vlm_ln_in_v[li], W._vlm_ln_post_v[li])
        return text, vis

    def _vlm_w_fp8(self, li):
        W = self.W
        if not getattr(W, "_vlm_fp8_ready", False):
            return None, None
        vis = (W._vlm_qkv_v8[li], W._vlm_qkv_v_ws[li], W._vlm_o_v8[li], W._vlm_o_v_ws[li],
               W._vlm_gu_v8[li], W._vlm_gu_v_ws[li], W._vlm_d_v8[li], W._vlm_d_v_ws[li])
        text = (W._vlm_qkv_t8[li], W._vlm_qkv_t_ws[li], W._vlm_o_t8[li], W._vlm_o_t_ws[li],
                W._vlm_gu_t8[li], W._vlm_gu_t_ws[li], W._vlm_d_t8[li], W._vlm_d_t_ws[li])
        return vis, text

    def _vlm_w_fp4(self, li):
        W = self.W
        if not getattr(W, "_vlm_fp4_ready", False):
            return None, None
        vis = (W._vlm_gu_v4[li], W._vlm_gu_v4sf[li], W._vlm_d_v4[li], W._vlm_d_v4sf[li],
               W._vlm_gu_N, W._vlm_inter, W._vlm_D)
        text = (W._vlm_gu_t4[li], W._vlm_gu_t4sf[li], W._vlm_d_t4[li], W._vlm_d_t4sf[li],
                W._vlm_gu_N, W._vlm_inter, W._vlm_D)
        return vis, text

    def _exp_w(self, li):
        W = self.W
        return (W._exp_qkv_v[li], W._exp_o_v[li], W._exp_gu_v[li],
                W._exp_d_v[li], W._exp_ln_in_v[li], W._exp_ln_post_v[li])

    def _exp_w_fp8(self, li):
        W = self.W
        if not getattr(W, "_exp_fp8_ready", False):
            return None
        return (W._exp_qkv8[li], W._exp_qkv_ws[li], W._exp_o8[li], W._exp_o_ws[li],
                W._exp_gu8[li], W._exp_gu_ws[li], W._exp_d8[li], W._exp_d_ws[li])

    def _exp_ffn_mk(self, li):
        W = self.W
        if not getattr(W, "_exp_ffn_mega_ready", False):
            return None
        return (W._exp_gu_mk[li], W._exp_gu_mk_s[li], W._exp_d_mk[li], W._exp_d_mk_s[li])

    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prefix_embs, n_vis, pmask, pcos, psin, kbuf, vbuf):
        """Run the 32-layer MoT VLM tower; fills kbuf/vbuf rows [0:S_p]."""
        hs = prefix_embs
        for li in range(32):
            text, vis = self._vlm_w(li)
            qk = (self.W._qk_norm_q[li], self.W._qk_norm_k[li])
            fp8v, fp8t = self._vlm_w_fp8(li)
            fp4v, fp4t = self._vlm_w_fp4(li)
            hs = self._block(hs, n_vis, text, vis, qk, pmask, pcos, psin,
                             kbuf[li], vbuf[li], 0, fp8v=fp8v, fp8t=fp8t,
                             fp4v=fp4v, fp4t=fp4t)
        return hs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def denoise(self, state, x_t, time_embs, smask, scos, ssin,
                kbuf, vbuf, S_p, num_steps=10):
        """32-layer expert tower + action head, ``num_steps`` Euler steps.

        ``x_t`` (1, chunk, 32) fp32 is updated in place and returned.
        ``time_embs`` (num_steps, 1, D_exp) bf16 precomputed by frontend.
        """
        W = self.W
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
            for li in range(32):
                exp = self._exp_w(li)
                qk = (W._qk_norm_q[li], W._qk_norm_k[li])
                hs = self._block(hs, S_s, None, exp, qk, smask, scos, ssin,
                                 kbuf[li], vbuf[li], S_p, fp8w=self._exp_w_fp8(li),
                                 ffn_mk=self._exp_ffn_mk(li))
            hs = F.rms_norm(hs, (hs.shape[-1],), W._exp_final_norm_w, self.rms_eps)
            v_t = F.linear(hs[:, -x_t.shape[1]:], W._aout_w, W._aout_b)
            x_t.add_(dt * v_t.to(x_t.dtype))   # in-place: static buffer for CUDA-graph replay
        return x_t


__all__ = ["HyVLAThorBF16Pipeline"]
