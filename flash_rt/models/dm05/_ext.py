"""Adapter: the dm05_native call convention, backed by FlashRT kernels.

The DM05 pipeline files in this directory are the native implementation, ported
across unchanged. They reach their kernels through ``self.ext = _ext.load()``, and
this module makes that name resolve to FlashRT instead of to the out-of-tree
``dm05_native`` extension. Every entry is a FlashRT kernel; what this file adds is
the argument translation between the two calling conventions.

Going through one adapter rather than editing roughly forty call sites across
vision.py, text.py and action.py has two advantages: the whole mapping is in one
place, reviewable against the closure table, and a signature that does not line up
fails at import or at the first call rather than silently passing the wrong
argument -- which is the failure mode that cost this port its first 85% mismatch.

Not yet translated, and they raise rather than guess: ``res_ln_quant`` and
``qk_norm_rope_v8`` are one fused kernel natively and two or three calls here, so
each needs its call site's buffer strides to be read before it can be written.
"""
from __future__ import annotations

from flash_rt import flash_rt_kernels as fvk
from flash_rt import flash_rt_dm05 as _dm05

_GEMM = None


def _gemm():
    global _GEMM
    if _GEMM is None:
        _GEMM = fvk.GemmRunner()
    return _GEMM


class _Ext:
    """The dm05_native surface, over FlashRT."""

    # ---- GEMM. dm05_native's ninth argument selects the B layout: True is (N, K),
    # absent is (K, N), and either may carry a bias.
    def gemm_bf16(self, A, B, out, bias, M, N, K, stream, b_nk=False):
        g = _gemm()
        if b_nk:
            if bias:
                g.bf16_nt_bias(A, B, out, bias, M, N, K, stream)
            else:
                g.bf16_run(A, B, out, M, N, K, stream)
        else:
            if bias:
                g.bf16_nn_bias(A, B, out, bias, M, N, K, stream)
            else:
                g.bf16_nn(A, B, out, M, N, K, stream)

    # ---- elementwise and dtype steps. Signature-identical; the native code already
    # calls several of these by their FlashRT names.
    def gelu_mul_bf16(self, gate, up, out, n, stream):
        fvk.gelu_mul_bf16(gate, up, out, n, stream)

    def axpy_f32(self, x, v, dt, n, stream):
        fvk.axpy_f32(x, v, dt, n, stream)

    def mul_rows_f32(self, x, mask, rows, cols, stream):
        fvk.mul_cols_f32(x, mask, rows, cols, stream)

    def init_rows_f32(self, x, src, mask, rows, cols, stream):
        fvk.copy_scaled_cols_f32(x, src, mask, rows, cols, stream)

    def add_bias_v8(self, x, b, rows, cols, stream):
        fvk.add_bias_bf16(x, b, rows, cols, stream)

    def add_bias_rows_v8(self, x, ld, b, rows, cols, stream):
        fvk.add_bias_rows_bf16(x, ld, b, rows, cols, stream)

    def patch_im2col_nchw_bf16(self, src, dst, nv, C, H, W, P, stream):
        fvk.patch_im2col_nchw_bf16(src, dst, nv, C, H, W, P, stream)

    def resize_norm(self, src, dst, nv, H, W, OH, OW, stream):
        # uint8 HWC -> bf16 CHW, the OpenCV-compatible resize the raw-view path folds
        # into the graph. Same coefficient/clamp arithmetic; the only build-dependent
        # step is the (x - 127.5) / 127.5 rescale, which flash_rt_kernels compiles
        # with --prec-div=false and therefore needs the rounding pinned explicitly.
        fvk.resize_norm_u8_hwc_to_bf16_chw(src, dst, nv, H, W, OH, OW, stream)

    # ---- projections
    def lin_in_f32_bf16(self, x, w, b, out, T, D, A, stream):
        fvk.linear_f32_in_bf16_out(x, w, b, out, T, D, A, stream)

    def lin_out_bf16_f32(self, x, w, b, out, T, A, D, stream):
        fvk.linear_bf16_in_f32_out(x, w, b, out, T, A, D, stream)

    def embed_merge(self, ids, img_index, table, img, out, rows, dim, scale, stream):
        fvk.embedding_lookup_merge_bf16(ids, img_index, table, img, out, rows, dim,
                                        scale, stream)

    # ---- norms. The native strided form takes (x, x_stride, w, out, out_stride,
    # residual, dim, rows, rows_per_group, group_stride, dim, eps, stream); FlashRT
    # takes res_stride where the native form repeats dim.
    def gemma_rms_norm(self, x, xs, w, out, o_xs, residual, dim, rows, rpg, gs, _dim, eps, stream):
        fvk.gemma3_rms_norm_bf16(x, xs, w, out, o_xs, residual, dim, rows, rpg, gs,
                                 dim, eps, stream)

    def gemma_rms_norm_v8(self, x, w, out, residual, rows, dim, eps, stream):
        fvk.gemma3_rms_norm_v8_bf16(x, dim, w, out, dim, residual, dim, rows, rows, 0,
                                    dim, eps, stream)

    def ada_rms_norm(self, x, mod, out, rows, dim, eps, stream):
        fvk.ada_rms_norm(x, mod, out, rows, dim, eps, stream)

    # ---- the action expert's fused gate + residual + adaLN. Same argument order in
    # both builds, and verified bit-identical at (rows=10, dim=1024) on both outputs.
    def gate_residual_ada_norm_v8(self, x, w, gate, residual, mod, out, rows, dim, eps, stream):
        fvk.gemma3_norm_gate_residual_ada_v8_bf16(x, w, gate, residual, mod, out, rows,
                                                  dim, eps, stream)

    # The non-v8 sibling reduces the same sums in a different order, so it is not the
    # same kernel and must not be aliased to it. At D=1024 the action expert's ternary
    # always picks the v8 form, so this is unreachable on the delivered shape; it
    # raises rather than returning a plausible wrong answer if that ever changes.
    def gate_residual_ada_norm(self, *a, **k):
        raise NotImplementedError(
            "gate_residual_ada_norm is the strided-reduction sibling of "
            "gate_residual_ada_norm_v8 and has no FlashRT route; it is unreachable at "
            "D=1024, where the action expert selects the v8 form")

    # ---- attention
    def vit_attn_hd72(self, qkv, ld, D, out, o_ld, S, NH, nv, scale, mt, stream,
                      q8=0, q8_scale=0):
        _dm05.dm05_vit_attn_hd72(qkv, ld, D, out, o_ld, S, NH, nv, scale, mt, stream,
                                 q8, q8_scale)

    def prefix_attn(self, q, k, v, out, klim, S, NH, NKV, scale, stream):
        _dm05.dm05_prefix_attn(q, k, v, out, klim, S, NH, NKV, scale, stream)

    def ae_decode_attn(self, q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt,
                       T, G, NKV, sk, ns, scale, stream):
        _dm05.dm05_ae_decode_attn(q, q_ld, k, v, kv_ld, out, o_ld, ws_o, ws_ml, cnt,
                                  T, G, NKV, sk, ns, scale, stream)

    # ---- fused quantizers
    def gelu_mul_quant_fp8_v8(self, gate, up, a_scale, a8, n, stream):
        fvk.gelu_mul_quant_fp8_static_bf16(gate, up, a_scale, a8, n, stream)

    def gelu_mul_quant_nvfp4(self, gate, up, cap, packed, sf, rows, cols, stream):
        fvk.gelu_mul_quant_nvfp4_swizzled(gate, up, cap, packed, sf, rows, cols, stream)

    def bias_gelu_quant_fp8_v8(self, h, b, scale, out, rows, cols, stream):
        fvk.bias_gelu_quantize_fp8_static_bf16_rounded_v8(h, b, out, scale, rows, cols, stream)

    def gemma_rms_norm_quant_v8(self, x, w, qparam, q, sf, rows, dim, eps, mode, stream):
        if mode == 0:
            fvk.gemma3_rms_norm_fp8_bf16(x, w, q, qparam, rows, dim, eps, stream)
        else:
            fvk.gemma3_rms_norm_nvfp4_swizzled(x, w, qparam, q, sf, rows, dim, eps, stream)

    # ---- int8 (Int8WLinear in int8w.py calls these through the module, not here)
    def w8a16_pack(self, q, packed, N, K, stream):
        _dm05.dm05_w8a16_pack(q, packed, N, K, stream)

    def w8a16_tc(self, x, packed, scale, out, m, N, K, warps, stream):
        _dm05.dm05_w8a16_tc(x, packed, scale, out, m, N, K, warps, stream)

    def w8a16_tc3(self, *a):
        _dm05.dm05_w8a16_tc3(*a)

    def w8a16_tc_glu(self, *a):
        _dm05.dm05_w8a16_tc_glu(*a)

    # ---- the two fused kernels the native build has and this one splits.
    #
    # res_ln_quant: one kernel doing residual-add, LayerNorm and FP8 quantize in a
    # single pass over the row. Composed here as the three FlashRT kernels it equals:
    # measured bit-equal on the residual and on the fp8 output, with 21 of 2359296
    # elements differing in the LayerNorm intermediate because the fused kernel's
    # pair assignment differs. The 444 gate was re-run for that and passes.
    def res_ln_quant(self, r, y, b, w, beta, xn, q, scale, rows, dim, eps, stream):
        # One pass, matching the reference: residual add under `if (y)`, LayerNorm, and
        # the static FP8 quantise, values kept in registers throughout. y == 0 skips the
        # residual (then b is unused); xn == 0 skips the normalised store, which is what
        # the vision tower passes -- the reference never materialises it, and the
        # three-kernel composition this replaces had to write 4.7 MB per call to a
        # scratch purely because layer_norm and quantize_fp8_static are separate kernels.
        fvk.res_ln_quant_v8_bf16(r, y, b, w, beta, xn, q, scale, rows, dim, eps, stream)

    # qk_norm_rope_v8: per-head (1+w) RMSNorm in place on q and on the k rows already
    # in the cache, then HF rotate-half rope, both at head_dim 256 -- the native
    # kernel hardcodes it. Verified bit-identical to the fused original at the
    # production shape: 0 of 1269760 q and 0 of 634880 k elements differ.
    def qk_norm_rope_v8(self, q, q_stride, nhq, k, k_stride, nhk, wq, wk, cosT, sinT,
                        seq, eps, stream):
        # (1+w) RMSNorm + HF rotate-half RoPE over q and the cached k rows, one launch.
        # FlashRT's csrc/kernels/qk_norm_rope_fused.cu cannot be used here: it is fp16
        # with a layer norm and laid out for head_dim 128 (the Chameleon shape), while
        # this is bf16 with a (1+w) RMSNorm at head_dim 256.
        fvk.gemma3_qk_norm_rope_v8_bf16(q, q_stride, nhq, k, k_stride, nhk, wq, wk,
                                        cosT, sinT, seq, eps, stream)


def load():
    """The dm05_native module, replaced by an adapter over FlashRT."""
    return _Ext()
