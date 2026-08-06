"""GR00T N1.7 Thor frontend with an NVFP4 DiT action head.

``GrootN17TorchFrontendThorFP4`` keeps the production FP8 backbone
(``GrootN17TorchFrontendThorFP8``) and runs every DiT GEMM — fused
self-attention QKV, cross-attention Q, attention output projection and
both FFN projections — as a block-scaled NVFP4 GEMM with fused bf16
bias epilogues:

* QKV / cross-Q:   ``cutlass_fp4_gemm_bias_bf16``
* O / FFN down:    ``cutlass_fp4_gemm_bias_res_bf16``  (residual fused)
* FFN up:          ``cutlass_fp4_gemm_bias_gelu_fp4out_bf16`` (bias +
                   tanh-GELU + FP4/SFA output feeding the down GEMM)

Activation quantization is dynamic per-16-element (no calibration pass):
the AdaLN and pre-FFN norms emit FP4 directly via the fused
``ada_layer_norm_fp4_sfa_bf16`` / ``layer_norm_no_affine_fp4_sfa_bf16``
kernels, and the attention output goes through
``quantize_fp4_dynamic_sfa_bf16_vec``.

At M = 41 (1 state + 40 action tokens) the DiT is weight-bandwidth
bound; NVFP4 halves the weight bytes of the FP8 tier on its dominant
cost and removes most per-layer elementwise launches.
"""

from __future__ import annotations

import torch

from flash_rt.frontends.torch.groot_n17_thor_fp8 import (
    GrootN17TorchFrontendThorFP8,
)


class GrootN17TorchFrontendThorFP4(GrootN17TorchFrontendThorFP8):
    """N1.7 Thor inference frontend: FP8 backbone + NVFP4 DiT."""

    _DIT_QUANT = "fp4"
    # Backbone small-kernel tier: vectorized (16-byte-load) norm / rope /
    # quantize / GQA-expand rewrites. Same element math as the scalar
    # kernels; the scalar originals are latency-bound at these shapes.
    _KBB_VEC = True
    # FA4 (CuTe-DSL) attention for the backbone MHA sites; falls back to
    # the fmha/cublas chain when the FA4 runtime deps are missing.
    _KBB_FA4 = True

    # Bindings this tier needs beyond the always-compiled set. They ship
    # only on SM100-class builds (CMake: FLASHRT_HAVE_THOR_VLA_KERNELS), so
    # probe once and fail with the build flag named rather than dying on an
    # AttributeError deep inside graph capture.
    _REQUIRED_BINDINGS = (
        "rms_norm_fp16_vec", "layer_norm_fp16_vec",
        "layer_norm_fp8_static_fp16_vec", "rope_rotate_half_fp16_vec",
        "quantize_fp8_static_fp16_vec", "gpu_repeat_interleave_heads_vec",
        "residual_add_fp16_vec", "attention_mha_fp16_masked",
        "attention_mha_bf16_masked",
    )

    def __init__(self, *args, **kwargs):
        import flash_rt.flash_rt_kernels as fvk
        missing = [n for n in self._REQUIRED_BINDINGS if not hasattr(fvk, n)]
        if missing:
            raise RuntimeError(
                "GROOT N1.7 Thor NVFP4 tier needs the Thor VLA helper "
                "kernels, which this build of flash_rt_kernels does not "
                f"carry (missing: {', '.join(missing)}). Rebuild for an "
                "SM100-class GPU (cmake -B build -S . -DGPU_ARCH=110) or "
                "use the FP8 tier (GrootN17TorchFrontendThorFP8).")
        super().__init__(*args, **kwargs)

    def _quantize_dit_fp4(self, step_weights, bp, action_horizon) -> None:
        """Quantize every DiT GEMM weight to NVFP4 (per-16 MSE block scales)
        and splice the pointers + activation scratch buffers into
        ``step_weights`` / ``bp`` so ``dit_forward`` takes the FP4 path."""
        import flash_rt.flash_rt_fp4 as fvk_fp4
        self._fvk_fp4 = fvk_fp4

        dev = self.device
        Sa = action_horizon + 1
        D, FF = 1536, 6144
        keep: list = []
        self._dit_fp4_keep = keep

        def quant_w(w_kn: torch.Tensor):
            # Weights are stored [K, N] (input-major, bf16); the NVFP4 GEMM
            # consumes B as packed [N, K] with SFB tile-interleaved scales.
            w_nk = w_kn.t().contiguous().to(torch.float16)
            N, K = w_nk.shape
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
            # SF buffers must be zero-initialized: the tile-interleaved
            # layout rounds rows/K up to atom sizes and the quantize kernel
            # only writes real coordinates — garbage in the padding decodes
            # as UE4M3 NaN and poisons the accumulator.
            sfb = torch.zeros(fvk_fp4.sfa_size_bytes(N, K, True),
                              dtype=torch.uint8, device=dev)
            rc = fvk_fp4.quantize_fp4_dynamic_sfa_mse_fp16(
                w_nk.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
                N, K, True, 0)
            if rc != 0:
                raise RuntimeError(f"DiT FP4 weight quantize failed rc={rc}")
            keep.extend([packed, sfb])
            return packed.data_ptr(), sfb.data_ptr()

        qkv_w, qkv_sfb, qkv_b = [], [], []
        q_w, q_sfb = [], []
        o_w, o_sfb = [], []
        up_w, up_sfb = [], []
        dn_w, dn_sfb = [], []
        for li in range(32):
            if li % 2 == 1:
                pw, ps = quant_w(torch.cat(
                    [self._dit_q_w[li], self._dit_k_w[li], self._dit_v_w[li]],
                    dim=1))
                qkv_w.append(pw)
                qkv_sfb.append(ps)
                qb = torch.cat([self._dit_q_b[li], self._dit_k_b[li],
                                self._dit_v_b[li]]).bfloat16().contiguous()
                keep.append(qb)
                qkv_b.append(qb.data_ptr())
            else:
                pw, ps = quant_w(self._dit_q_w[li])
                q_w.append(pw)
                q_sfb.append(ps)
            pw, ps = quant_w(self._dit_o_w[li])
            o_w.append(pw)
            o_sfb.append(ps)
            pw, ps = quant_w(self._dit_ff_proj_w[li])
            up_w.append(pw)
            up_sfb.append(ps)
            pw, ps = quant_w(self._dit_ff_down_w[li])
            dn_w.append(pw)
            dn_sfb.append(ps)
        torch.cuda.synchronize()

        # Activation scratch: packed e2m1 + zero-initialized SFA buffers.
        def act_buf(cols):
            packed = torch.empty(Sa, cols // 2, dtype=torch.uint8, device=dev)
            sfa = torch.zeros(fvk_fp4.sfa_size_bytes(Sa, cols, False),
                              dtype=torch.uint8, device=dev)
            keep.extend([packed, sfa])
            return packed, sfa

        xn_fp4, xn_sfa = act_buf(D)
        octx_fp4, octx_sfa = act_buf(D)
        hid_fp4, hid_sfa = act_buf(FF)
        self._k_qkv_buf = torch.empty(Sa, 3 * D, dtype=torch.bfloat16,
                                      device=dev)

        # Read the fused QKV GEMM output in place from the self-attn site
        # (token stride 3D) — the three per-layer split copies disappear.
        base = self._k_qkv_buf.data_ptr()
        self._dit_attn._slots["dit_self"].update(
            Q=base, K=base + 2 * D, V=base + 4 * D, qkv_stride=3 * D)

        for sw in step_weights:
            sw.update(
                qkv_w_fp4=qkv_w, qkv_sfb=qkv_sfb, qkv_b_fp4=qkv_b,
                q_w_fp4=q_w, q_sfb=q_sfb,
                o_w_fp4=o_w, o_sfb=o_sfb,
                ff_proj_w_fp4=up_w, ff_proj_sfb=up_sfb,
                ff_down_w_fp4=dn_w, ff_down_sfb=dn_sfb)
        bp.update(
            xn_fp4=xn_fp4.data_ptr(), xn_sfa=xn_sfa.data_ptr(),
            octx_fp4=octx_fp4.data_ptr(), octx_sfa=octx_sfa.data_ptr(),
            hid_fp4=hid_fp4.data_ptr(), hid_sfa=hid_sfa.data_ptr(),
            qkv_buf=self._k_qkv_buf.data_ptr(), qkv_strided=True)
