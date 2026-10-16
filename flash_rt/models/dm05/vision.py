"""Native SigLIP vision tower + Gemma3 multimodal projector for DM05.

Mirrors ``SiglipVisionModel`` (vision_use_head=False) followed by
``Gemma3MultiModalProjector`` with native kernels on preallocated buffers:

    pixels (nv, 3, 448, 448) bf16 NCHW
      -> im2col (nv*1024, 588) -> patch GEMM + bias -> + position embedding
      -> 27 x [LN1 -> QKV -> FA2 -> O -> +res -> LN2 -> fc1 -> gelu_tanh -> fc2 -> +res]
      -> post_layernorm
      -> 2x2 avg-pool (32x32 -> 16x16 per view)
      -> Gemma3 RMSNorm ((1 + w), fp32) -> matmul (1152 -> 2560)
      -> image tokens (nv*256, 2560) bf16

Each biased linear is one GEMM with the bias in the cuBLASLt epilogue
(``dm05_native.gemm_bf16``), rounded to bf16 once, as HF's addmm does. FlashRT's
``bf16_nn_bias`` is rejected by this cuBLASLt (row-order layouts + BIAS
epilogue -> CUBLAS_STATUS_NOT_SUPPORTED), and splitting the bias into a second
kernel was measured as the largest per-op error on SigLIP layer 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_rt import flash_rt_kernels as fvk

from . import _ext
from .attention import _bshd_strides, _fwd

bf16 = torch.bfloat16


def _kn(w: torch.Tensor) -> torch.Tensor:
    """nn.Linear weight (out, in) -> (in, out) contiguous, the bf16_nn B layout."""
    return w.detach().to(bf16).contiguous()


class NativeVision:
    """SigLIP + projector with fixed shapes and stable buffer addresses."""

    def __init__(self, vision_tower, projector, num_views: int, device="cuda"):
        vm = vision_tower.vision_model
        cfg = vm.config
        self.nv = int(num_views)
        self.P = int(cfg.patch_size)
        self.img = int(cfg.image_size)
        self.grid = self.img // self.P                    # 32
        self.seq_per_view = self.grid * self.grid         # 1024
        self.S = self.nv * self.seq_per_view              # 2048
        self.D = int(cfg.hidden_size)                     # 1152
        self.FF = int(cfg.intermediate_size)              # 4304
        self.NH = int(cfg.num_attention_heads)            # 16
        self.HD = self.D // self.NH                       # 72
        self.eps = float(cfg.layer_norm_eps)              # 1e-6
        self.L = len(vm.encoder.layers)
        self.patch_feat = 3 * self.P * self.P             # 588
        self.pool = self.grid // int(round(projector.tokens_per_side))  # 2
        self.tok_side = self.grid // self.pool            # 16
        self.tok_per_view = self.tok_side * self.tok_side  # 256
        self.out_dim = int(projector.mm_input_projection_weight.shape[1])  # 2560
        self.proj_eps = float(projector.mm_soft_emb_norm.eps)
        self.scale = self.HD ** -0.5
        d = torch.device(device)

        emb = vm.embeddings
        self.w_patch = _kn(emb.patch_embedding.weight.reshape(self.D, -1))
        self.b_patch = emb.patch_embedding.bias.detach().to(bf16).contiguous()
        self.pos = emb.position_embedding.weight.detach().to(bf16).repeat(self.nv, 1).contiguous()

        self.layers = []
        for ly in vm.encoder.layers:
            a = ly.self_attn
            self.layers.append(dict(
                ln1_w=ly.layer_norm1.weight.detach().to(bf16).contiguous(),
                ln1_b=ly.layer_norm1.bias.detach().to(bf16).contiguous(),
                qkv_w=_kn(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0)),
                qkv_b=torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).detach().to(bf16).contiguous(),
                o_w=_kn(a.out_proj.weight), o_b=a.out_proj.bias.detach().to(bf16).contiguous(),
                ln2_w=ly.layer_norm2.weight.detach().to(bf16).contiguous(),
                ln2_b=ly.layer_norm2.bias.detach().to(bf16).contiguous(),
                fc1_w=_kn(ly.mlp.fc1.weight), fc1_b=ly.mlp.fc1.bias.detach().to(bf16).contiguous(),
                fc2_w=_kn(ly.mlp.fc2.weight), fc2_b=ly.mlp.fc2.bias.detach().to(bf16).contiguous(),
            ))
        self.post_w = vm.post_layernorm.weight.detach().to(bf16).contiguous()
        self.post_b = vm.post_layernorm.bias.detach().to(bf16).contiguous()
        # Gemma3RMSNorm multiplies by (1 + w) in fp32; dm05_native.gemma_rms_norm forms
        # (1 + w) per element in fp32. Folding it into a bf16 weight instead would
        # quantise it to 1/128 steps near 1.0.
        self.mm_norm_w = projector.mm_soft_emb_norm.weight.detach().to(bf16).contiguous()
        # (2560, 1152) = (N, K): the projector weight is the one tensor the checkpoint
        # already holds (K, N), so it needs the transpose the _kn helper no longer does
        self.mm_proj_w = projector.mm_input_projection_weight.detach().to(bf16).t().contiguous()

        z = lambda *s: torch.empty(*s, dtype=bf16, device=d)
        self.pixels = z(self.nv, 3, self.img, self.img)
        self.patches = z(self.S, self.patch_feat)
        self.x = z(self.S, self.D)
        self.xn = z(self.S, self.D)
        self.qkv = z(self.S, 3 * self.D)
        self.attn = z(self.S, self.D)
        self.tmp = z(self.S, self.D)
        self.h = z(self.S, self.FF)
        self.pooled = z(self.nv * self.tok_per_view, self.D)
        self.pooled_n = z(self.nv * self.tok_per_view, self.D)
        self.out = z(self.nv * self.tok_per_view, self.out_dim)

        self.fp8: dict[tuple[int, str], "Fp8Linear"] = {}
        self.attn_hd72 = False
        self.attn_quant_epilogue = False
        self.gemm = fvk.GemmRunner()
        self.ext = _ext.load()
        qkv5 = self.qkv.view(self.nv, self.seq_per_view, 3, self.NH, self.HD)
        # [B, H, S, D] views into the fused QKV buffer; FA2 reads them through
        # strides, so no split/copy kernel is needed.
        self._q = qkv5[:, :, 0].permute(0, 2, 1, 3)
        self._k = qkv5[:, :, 1].permute(0, 2, 1, 3)
        self._v = qkv5[:, :, 2].permute(0, 2, 1, 3)
        self._qs, self._ks, self._vs = (_bshd_strides(self._q), _bshd_strides(self._k),
                                        _bshd_strides(self._v))
        o4 = self.attn.view(self.nv, self.seq_per_view, self.NH, self.HD)
        self._os = (o4.stride(0), o4.stride(1), o4.stride(2))

    _DIMS = {"qkv": ("qkv_w", "qkv_b"), "o": ("o_w", "o_b"), "fc1": ("fc1_w", "fc1_b"),
             "fc2": ("fc2_w", "fc2_b")}

    def fold_kv_bias(self, drop_k: bool = True) -> None:
        """Exact algebraic removal of the key and value biases. softmax(q.(k + b_k)) =
        softmax(q.k) because q.b_k shifts every score of a query equally, and the attention
        weights sum to one, so b_v passes straight through to the output, where
        (a + b_v) W_o + b_o = a W_o + (b_v W_o + b_o). Only the query bias remains; the
        fused forward adds it to the q third of the qkv buffer alone. Must run before the
        FP8 sites release the bf16 o_proj weight."""
        if self.fp8:
            raise ValueError("fold_kv_bias must run before enable_fp8")
        D = self.D
        for w in self.layers:
            b = w["qkv_b"].float()
            w["o_b"] = (w["o_b"].float() + b[2 * D:] @ w["o_w"].float().t()).to(w["o_b"].dtype).contiguous()
            keep = D if drop_k else 2 * D
            w["qkv_b"] = torch.cat([b[:keep], torch.zeros(3 * D - keep, device=b.device)]).to(w["qkv_b"].dtype).contiguous()
            w["q_b"] = w["qkv_b"][:keep]
        self.kv_bias_folded = D if drop_k else 2 * D

    def enable_fp8(self, kinds, layers=None) -> None:
        """Route SigLIP projections through static per-tensor FP8 (bias added after)."""
        from .fp8 import Fp8Linear
        for i in (range(self.L) if layers is None else layers):
            for kind in kinds:
                wkey, _ = self._DIMS[kind]
                w_nk = self.layers[i][wkey]                       # already (N, K)
                self.fp8[(i, kind)] = Fp8Linear(w_nk, self.S, slots=1, name=f"vt.{i}.{kind}")
                self.layers[i][wkey] = None                          # bf16 copy no longer used

    def _lin(self, i, kind, x, out, n, k, stream, defer_bias=False):
        """Returns True when the bias is still pending (deferred to the caller's next op)."""
        w = self.layers[i]
        wkey, bkey = self._DIMS[kind]
        site = self.fp8.get((i, kind))
        if site is None:
            self.ext.gemm_bf16(x.data_ptr(), w[wkey].data_ptr(), out.data_ptr(), w[bkey].data_ptr(),
                               self.S, n, k, stream, True)
            return False
        if getattr(site, "_prequantized", False):
            site._prequantized = False       # activation already written by the fused kernel
        else:
            site.quantize(x, self.S, stream)
        # FlashRT fp8_nt_dev (sm120 nvjet). The cuBLASLt FP8 + BIAS epilogue was measured
        # slower even autotuned (it only offers an sm89 xmma kernel: 21.7 vs 14.3 ms per
        # predict), so the bias is added separately or deferred into the next op.
        site.gemm(out, self.S, stream)
        if defer_bias:
            return True
        self.ext.add_bias_v8(out.data_ptr(), w[bkey].data_ptr(), self.S, n, stream)
        return False

    @staticmethod
    def _p(t: torch.Tensor) -> int:
        return t.data_ptr()

    def forward(self, stream: int) -> None:
        """Run on the preallocated buffers: ``self.pixels`` -> ``self.out``."""
        p, g, S, D = self._p, self.gemm, self.S, self.D
        mm = self.ext.gemm_bf16
        if self._fused_ready():
            return self._forward_fused(stream)
        self.ext.patch_im2col_nchw_bf16(p(self.pixels), p(self.patches), self.nv, 3,
                                        self.img, self.img, self.P, stream)
        mm(p(self.patches), p(self.w_patch), p(self.x), p(self.b_patch), S, D, self.patch_feat, stream, True)
        fvk.residual_add(p(self.x), p(self.pos), S * D, stream)
        for i, w in enumerate(self.layers):
            fvk.layer_norm(p(self.x), p(w["ln1_w"]), p(w["ln1_b"]), p(self.xn), S, D, self.eps, stream)
            self._lin(i, "qkv", self.xn, self.qkv, 3 * D, D, stream)
            self._attention(stream)
            # FP8 sites defer their bias into the next elementwise op (x += y + b;
            # gelu(h + b)), so the fast nvjet FP8 GEMM needs no separate bias pass
            if self._lin(i, "o", self.attn, self.tmp, D, D, stream, defer_bias=True):
                fvk.bias_residual(p(self.x), p(self.tmp), p(w["o_b"]), S, D, stream)
            else:
                fvk.residual_add(p(self.x), p(self.tmp), S * D, stream)
            fvk.layer_norm(p(self.x), p(w["ln2_w"]), p(w["ln2_b"]), 0, S, D, self.eps, stream)
            fc2 = self.fp8.get((i, "fc2"))
            if self._lin(i, "fc1", self.xn, self.h, self.FF, D, stream, defer_bias=True):
                if fc2 is not None and fc2.calibrated and not fc2.calibrating:
                    # bias + gelu + fc2's static FP8 quantize in one pass (no bf16 write-back)
                    self.ext.bias_gelu_quant_fp8_v8(p(self.h), p(w["fc1_b"]), p(fc2.a_scale[0:1]),
                                                    p(fc2.a8), S, self.FF, stream)
                    fc2._prequantized = True
                else:
                    fvk.bias_gelu_bf16(p(self.h), p(w["fc1_b"]), S, self.FF, stream)
            else:
                fvk.gelu_inplace(p(self.h), S * self.FF, stream)
            if self._lin(i, "fc2", self.h, self.tmp, D, self.FF, stream, defer_bias=True):
                fvk.bias_residual(p(self.x), p(self.tmp), p(w["fc2_b"]), S, D, stream)
            else:
                fvk.residual_add(p(self.x), p(self.tmp), S * D, stream)
        fvk.layer_norm(p(self.x), p(self.post_w), p(self.post_b), p(self.xn), S, D, self.eps, stream)
        fvk.avg_pool_vision_tokens(p(self.xn), p(self.pooled), self.nv, self.grid, self.grid, D,
                                   self.pool, stream)
        T = self.nv * self.tok_per_view
        self.ext.gemma_rms_norm(p(self.pooled), D, p(self.mm_norm_w), p(self.pooled_n), D, 0, D,
                                T, T, 0, D, self.proj_eps, stream)
        mm(p(self.pooled_n), p(self.mm_proj_w), p(self.out), 0, T, self.out_dim, D, stream, True)

    def _attention(self, stream: int, o_site=None) -> bool:
        """Returns True when the attention wrote o_site's FP8 input itself."""
        if self.attn_hd72:
            # native head_dim-72 kernel: no padding of the 72-wide heads to FA2's hdim-96 tile;
            # with o_site its epilogue also quantizes straight into o_proj's static FP8 input
            q8 = o_site is not None
            self.ext.vit_attn_hd72(self.qkv.data_ptr(), 3 * self.D, self.D, self.attn.data_ptr(), self.D,
                                   self.seq_per_view, self.NH, self.nv, self.scale, 2, stream,
                                   o_site.a8.data_ptr() if q8 else 0, o_site.a_scale.data_ptr() if q8 else 0)
            return q8
        p = self._p
        _fwd(p(self._q), p(self._k), p(self._v), p(self.attn), self._qs, self._ks, self._vs, self._os,
             self.nv, self.seq_per_view, self.seq_per_view, self.NH, self.NH, self.HD,
             self.qkv.device, self.scale, False, False)
        return False

    def _fused_ready(self) -> bool:
        """All four projections of every layer are calibrated static FP8 sites."""
        for i in range(self.L):
            for kind in ("qkv", "o", "fc1", "fc2"):
                site = self.fp8.get((i, kind))
                if site is None or not site.calibrated or site.calibrating:
                    return False
        return True

    def _forward_fused(self, stream: int) -> None:
        """All-FP8 SigLIP with fused layer boundaries (bit-identical FP8 activations):
        res+LN2+fc1-quant after o, bias+gelu+fc2-quant after fc1, res+next-LN1+qkv-quant
        after fc2. Only the post-layernorm, whose bf16 output is pooled, keeps FlashRT's
        layer_norm."""
        p, S, D, e = self._p, self.S, self.D, self.ext
        e.patch_im2col_nchw_bf16(p(self.pixels), p(self.patches), self.nv, 3, self.img, self.img, self.P, stream)
        e.gemm_bf16(p(self.patches), p(self.w_patch), p(self.x), p(self.b_patch), S, D, self.patch_feat, stream, True)
        fvk.residual_add(p(self.x), p(self.pos), S * D, stream)
        L0 = self.layers[0]
        q0 = self.fp8[(0, "qkv")]
        e.res_ln_quant(p(self.x), 0, 0, p(L0["ln1_w"]), p(L0["ln1_b"]), 0, p(q0.a8), p(q0.a_scale),
                       S, D, self.eps, stream)
        for i, w in enumerate(self.layers):
            sq, so, s1, s2 = (self.fp8[(i, k)] for k in ("qkv", "o", "fc1", "fc2"))
            sq.gemm(self.qkv, S, stream)
            if getattr(self, "kv_bias_folded", False):
                e.add_bias_rows_v8(p(self.qkv), 3 * D, p(w["q_b"]), S, self.kv_bias_folded, stream)
            else:
                e.add_bias_v8(p(self.qkv), p(w["qkv_b"]), S, 3 * D, stream)
            if not self._attention(stream, so if self.attn_quant_epilogue else None):
                so.quantize(self.attn, S, stream)
            so.gemm(self.tmp, S, stream)
            e.res_ln_quant(p(self.x), p(self.tmp), p(w["o_b"]), p(w["ln2_w"]), p(w["ln2_b"]), 0,
                           p(s1.a8), p(s1.a_scale), S, D, self.eps, stream)
            s1.gemm(self.h, S, stream)
            e.bias_gelu_quant_fp8_v8(p(self.h), p(w["fc1_b"]), p(s2.a_scale), p(s2.a8), S, self.FF, stream)
            s2.gemm(self.tmp, S, stream)
            if i + 1 < self.L:
                nx = self.layers[i + 1]
                nq = self.fp8[(i + 1, "qkv")]
                e.res_ln_quant(p(self.x), p(self.tmp), p(w["fc2_b"]), p(nx["ln1_w"]), p(nx["ln1_b"]), 0,
                               p(nq.a8), p(nq.a_scale), S, D, self.eps, stream)
            else:
                fvk.bias_residual(p(self.x), p(self.tmp), p(w["fc2_b"]), S, D, stream)
        fvk.layer_norm(p(self.x), p(self.post_w), p(self.post_b), p(self.xn), S, D, self.eps, stream)
        fvk.avg_pool_vision_tokens(p(self.xn), p(self.pooled), self.nv, self.grid, self.grid, D, self.pool, stream)
        T = self.nv * self.tok_per_view
        e.gemma_rms_norm(p(self.pooled), D, p(self.mm_norm_w), p(self.pooled_n), D, 0, D, T, T, 0, D,
                         self.proj_eps, stream)
        e.gemm_bf16(p(self.pooled_n), p(self.mm_proj_w), p(self.out), 0, T, self.out_dim, D, stream, True)
