"""HyVLA frontend for RTX consumer Blackwell (SM120).

This backend targets SM120 only (validated on an RTX 5060 Ti) and is built as
``sm_120a``. SM121 / GB10 (DGX Spark) is out of scope for this build: it needs
a separate ``-DGPU_ARCH=121`` build and has not been validated, so the
capability gate rejects it up front.

SM120 has no SM110 (Thor) FP8 megakernel path, but it can reuse the
Orin SM87 implementation: BF16 math, the plain-CUDA fused attention-prep
kernels (``hyvla_rope_qknorm_kvwrite_parallel_bf16`` and
``hyvla_rope_qknorm_kvwrite_qb_bf16``), the fused ViT add+LayerNorm
kernel, and memory-efficient SDPA.

Precision tiers on SM120a (the native tcgen05 routes):

  * ``use_fp8_block128=True`` — the SM120a FP8 block-128 cutlass GEMM
    (``fp8_block128_gemm_cutlass_sm120_bf16out``) with per-token x per-128-K
    activation scales and per-128x128 weight scales. Backs the expert denoise
    tower and the VLM prefill tower GEMMs (all K/N are 128-aligned). The ViT
    fc1/fc2 (K=4304) are NOT 128-aligned and stay BF16.

The Orin SM87 INT8 W8A8 rowwise tier requires ``ENABLE_SM80_INT8_CUTLASS``,
which is DISABLED in the sm120 build.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.frontends.torch.hyvla_orin import HyVLATorchFrontendOrin
from flash_rt.frontends.torch.hyvla_thor import (
    _BF16, _camera_tensor, _TOK_BOS, _TOK_HY_USER, _TOK_VISION_START,
    _TOK_VISION_END, _TOK_VISION_SPLIT, _TOK_HY_ASSISTANT)
from flash_rt.models.hyvla.pipeline_rtx import HyVLARTXBF16Pipeline

_FP8_MAX = 448.0
_FP8_BLOCK = 128


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a documented ``HYVLA_*`` opt-out lever (default-on)."""
    return os.environ.get(name, "1" if default else "0") == "1"


def _flags() -> dict:
    """Centralized SM120 optimization-lever flags.

    These are internal, default-on performance levers validated as part of the
    block-128 tier — **not** user-facing precision tiers (those are the
    ``use_fp8`` / ``use_fp4`` constructor kwargs). Each is an opt-out
    environment variable, documented in ``docs/hyvla05_rtx_sm120.md``.
    """
    return {
        "native_temporal": _env_flag("HYVLA_NATIVE_TEMPORAL"),
        "vit_spacetime_fused": _env_flag("HYVLA_VIT_SPACETIME_FUSED"),
        "merger_native": _env_flag("HYVLA_MERGER_NATIVE"),
        "prefix_native": _env_flag("HYVLA_PREFIX_NATIVE"),
        "fa2_prepare_q": _env_flag("HYVLA_FA2_PREPARE_Q"),
        "fa2_gather_quant_o": _env_flag("HYVLA_FA2_GATHER_QUANT_O"),
        "expert_od_splitk_b128": _env_flag("HYVLA_EXPERT_OD_SPLITK_B128"),
        "expert_od_fp8": _env_flag("HYVLA_EXPERT_OD_FP8", default=False),
        "vit_hd72": _env_flag("HYVLA_VIT_HD72"),
        "vit_proj_nvfp4": _env_flag("HYVLA_VIT_PROJ_NVFP4"),
        "vit_qkv_nvfp4": _env_flag("HYVLA_VIT_QKV_NVFP4"),
        "vit_patch_gemm": _env_flag("HYVLA_VIT_PATCH_GEMM"),
        "vit_fc2_nvfp4": _env_flag("HYVLA_VIT_FC2_NVFP4"),
        "vit_fc1_gelu_fuse": _env_flag("HYVLA_VIT_FC1_GELU_FUSE"),
    }


def _block128_quant_weight(w_bf16):
    """(N,K) bf16 -> (N,K) e4m3 uint8 + (N/128,K/128) fp32 scale.

    Matches the fp8_block128_gemm_cutlass_sm120_bf16out dequant convention:
    reconstruct = e4m3(w / scale) * scale, scale = amax/448 per 128x128 block.
    """
    N, K = w_bf16.shape
    if N % _FP8_BLOCK or K % _FP8_BLOCK:
        raise ValueError(
            f"FP8 block-128 weight requires N,K multiples of 128, got {N},{K}")
    w = w_bf16.float()
    wr = w.reshape(N // _FP8_BLOCK, _FP8_BLOCK, K // _FP8_BLOCK, _FP8_BLOCK)
    amax = wr.abs().amax(dim=(1, 3))
    scale = (amax / _FP8_MAX).clamp(min=1e-12)
    q = (wr / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    q8 = q.to(torch.float8_e4m3fn).view(torch.uint8).reshape(N, K)
    return q8, scale.contiguous()


class HyVLATorchFrontendRtx(HyVLATorchFrontendOrin):
    _REQUIRED_CAPABILITY = (12, 0)
    _SUPPORTED_CAPABILITIES = ((12, 0),)
    _ARCH_NAME = "RTX Blackwell SM120"
    _PIPE_CLS = HyVLARTXBF16Pipeline

    def _require_arch(self):
        if os.environ.get(self._FORCE_ARCH_ENV) == "1":
            return  # explicit documented dev override: skip the probe
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"HyVLA frontend requires a CUDA device ({self._ARCH_NAME}); "
                "CUDA is not available.")
        cap = tuple(torch.cuda.get_device_capability())
        if cap not in self._SUPPORTED_CAPABILITIES:
            raise RuntimeError(
                f"HyVLA frontend requires {self._ARCH_NAME} "
                f"(capability {self._REQUIRED_CAPABILITY}), found capability "
                f"{cap}. This backend is built for sm_120a; SM121 / GB10 needs "
                "a separate -DGPU_ARCH=121 build and is not validated. Set "
                f"{self._FORCE_ARCH_ENV}=1 to bypass this check for "
                "development only.")

    def __init__(self, checkpoint_dir: str, *, hardware: str = "rtx_sm120",
                 use_fp8: bool = False, use_fp4: bool = False,
                 use_fp4_expert: bool = False,
                 use_fused: bool = True, use_fp8_block128: bool | None = None,
                 **kwargs):
        """SM120 frontend.

        Precision tiers follow the same named-kwarg convention as
        ``HyVLATorchFrontendThor`` so that ``load_model`` (which forwards
        ``use_fp8`` / ``use_fused``) reaches them:

          * ``use_fp8=True`` — validated FP8 tier: SM120 block-128 FP8
            GEMMs throughout (expert + ViT + VLM prefill); **no NVFP4**.
            Action cosine 0.99994 (>= 0.999 gate).
          * ``use_fp4=True`` — master switch for the NVFP4 **ViT + VLM
            prefill** tower (qkv/proj/fc1/fc2, prefill QKV/O + FFN); the
            expert denoise tower stays on FP8 unless ``use_fp4_expert=True``.
            Action cosine 0.99977 (>= 0.999 gate), 55.7 ms. ``load_model``'s
            default on sm120.
          * ``use_fp4_expert`` — independently promotes the expert denoise
            tower to NVFP4 (requires ``use_fp4``); default ``False``.
            ``use_fp4=True, use_fp4_expert=True`` is the fastest tier
            (52 ms, cosine ~0.9986, below the 0.999 gate) — opt-in.
          * ``use_fp8=False, use_fp4=False`` — pure BF16 reference path (Orin
            SM87 INT8 is never selected on sm120).

        ``use_fp8_block128`` is a legacy alias for ``use_fp8`` kept for the
        internal A/B harness. Fine-grained optimization levers are documented
        opt-out ``HYVLA_*`` environment variables (see
        ``docs/hyvla05_rtx_sm120.md``).
        """
        # Orin SM87 INT8 is not built on sm120; never route to it.
        kwargs["use_fp8"] = False
        kwargs["use_int8"] = False
        kwargs.setdefault("use_fused", use_fused)
        block128 = bool(use_fp8 or use_fp4 or use_fp8_block128)
        # ``use_fp4`` gates all NVFP4 (ViT + prefill); ``use_fp4_expert``
        # independently promotes the expert denoise tower (default False).
        expert_fp4 = bool(use_fp4_expert)
        super().__init__(checkpoint_dir, hardware=hardware, **kwargs)
        import flash_rt.flash_rt_kernels as fvk
        # Graph caches for the SM120 one-graph steady-state path, and the fused
        # FA2 prepare_q (writes Q straight into the denoise packing).
        self._full_graph_cache = {}
        self._prefix_static_cache = None
        if (getattr(self.pipe, "_fused_attn", False)
                and hasattr(fvk, "hyvla_rope_qknorm_kvwrite_qb_bf16")):
            self.pipe._fused_prepare = True
        f = _flags()
        if f["native_temporal"] and hasattr(fvk, "hyvla_vit_temporal_mix_bf16"):
            self.pipe._native_temporal_mix = fvk.hyvla_vit_temporal_mix_bf16
        self.pipe._native_spacetime_ln = (
            f["vit_spacetime_fused"]
            and hasattr(fvk, "hyvla_vit_res_add_ln_time_bf16"))
        self.pipe._merger_native = f["merger_native"]
        self._prefix_native = f["prefix_native"]
        if (f["fa2_prepare_q"]
                and hasattr(fvk, "hyvla_fa2_denoise_prepare_q_bf16")):
            self.pipe._fa2_prepare_q = fvk.hyvla_fa2_denoise_prepare_q_bf16
        if (f["fa2_gather_quant_o"]
                and hasattr(fvk, "hyvla_fa2_denoise_gather_o_fp8_block128_bf16")):
            self.pipe._fa2_gather_quant_o = \
                fvk.hyvla_fa2_denoise_gather_o_fp8_block128_bf16
        self.pipe._exp_od_b128 = f["expert_od_splitk_b128"]
        # ViT head-dim unpadding: run the native 72 head dim end-to-end (no
        # 72->96 zero-pad); the vendored FA2 <96,64,32,4> tile handles a runtime
        # head_dim=72 through the is_even_K=false path, shrinking the qkv GEMM,
        # the proj-gather read and the temporal-mix traffic by 25%.
        self._vit_hd72 = f["vit_hd72"]
        if block128:
            self._quantize_fp8_block128()
            self._quantize_vit_fc_fp8()
            self.pipe.enable_fp8_block128()
        # ── NVFP4 (W4A4) master switch ────────────────────────────────────
        # ``use_fp4`` enables NVFP4 in the ViT (qkv/proj/fc1/fc2) and the VLM
        # prefill tower (QKV/O + FFN). With ``use_fp4=False`` nothing below
        # runs and the whole model stays on block-128 FP8 (cosine 0.99994).
        if block128 and use_fp4:
            self._quantize_vit_fc_fp4()
            self.pipe._vit_fp4 = True
            # fc1 NVFP4 + fused NVFP4 fc2 producer (the validated ViT route).
            self.pipe._vit_fp4_fc1_only = True
            self.pipe._vit_fp4_proj = f["vit_proj_nvfp4"]
            self.pipe._vit_fp4_qkv = f["vit_qkv_nvfp4"]
            self._quantize_prefill_ffn_fp4()
            self.pipe._fp4_alpha = self._fp4_alpha
            self.pipe._fp4_weight_map = getattr(self, "_fp4_weight_map", None)
            self.pipe.enable_fp4()
        # Patch embed: replace the cudnn stride-16 conv with the equivalent
        # im2col bf16 matmul (measured ~3x faster at in_channels=3).
        self.pipe._vit_patch_gemm = f["vit_patch_gemm"]
        # SmoothQuant/AWQ NVFP4 for ViT fc2: pre-scale the fc2 weight by
        # per-input-channel scales s_k (migrating activation dynamic range into
        # the weight) and divide the activation by s_k in the fused producer.
        # Enabled only when an explicit calibration file is supplied.
        # NOTE: the flag is read by the pipeline off ``self.W`` (this frontend),
        # so it must be set here, not on ``self.pipe``.
        self._vit_fc2_awq = False
        _awq_path = os.environ.get("HYVLA_VIT_FC2_AWQ")
        if block128 and use_fp4 and _awq_path:
            self._quantize_vit_fc2_fp4_awq(_awq_path)
            self._vit_fc2_awq = True
        # Non-AWQ fused NVFP4 fc2 producer (bias+GELU+NVFP4 quant) + prequant
        # GEMM. Measured winner over the FP8 block-128 producer under the 0.99
        # gate. AWQ (above) takes precedence when a calibration is supplied.
        self.pipe._vit_fc2_fused_nvfp4 = (
            block128 and use_fp4 and f["vit_fc2_nvfp4"]
            and not self._vit_fc2_awq)
        # Fuse the large-M ViT fc1 NVFP4 GEMM + per-col bias + tanh-GELU +
        # NVFP4 block-quant into one cutlass bias_gelu_fp4out epilogue launch,
        # replacing the bf16-output GEMM plus the standalone producer.
        self.pipe._vit_fc1_gelu_fuse = (
            self.pipe._vit_fc2_fused_nvfp4 and f["vit_fc1_gelu_fuse"])
        # ── Expert denoise NVFP4 (independent; opt-in via use_fp4_expert) ──
        # qkv/o/gu/dn -> NVFP4 with fused producers + the M=41 prequant GEMM;
        # halves the expert weight HBM traffic. Fastest tier but action cosine
        # ~0.9986 (below the 0.999 gate) — opt-in.
        _o4 = getattr(fvk, "hyvla_fa2_denoise_gather_o_nvfp4_bf16", None)
        if (block128 and expert_fp4
                and hasattr(fvk, "fp4_w4a16_gemm_sm120_bf16out")
                and _o4 is not None):
            self._quantize_expert_fp4()
            self.pipe._exp_fp4 = True
            self.pipe._fa2_gather_quant_o4 = _o4
            # Expert o/dn route: full NVFP4 W4A4 unless HYVLA_EXPERT_OD_FP8=1.
            self.pipe._exp_od_fp8 = f["expert_od_fp8"]

    def _quantize_prefill_ffn_fp4(self):
        """NVFP4 (flash_rt_kernels family) for the prefill FFN gu/down weights."""
        import flash_rt.flash_rt_kernels as fvk
        alpha = {}

        def qw(w):
            N, K = w.shape
            w = w.to(torch.bfloat16).contiguous()
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
            sf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(N, K), dtype=torch.uint8,
                             device=w.device)
            scratch = torch.empty(1, dtype=torch.float32, device=w.device)
            og = torch.empty(1, dtype=torch.float32, device=w.device)
            fvk.bf16_weight_to_nvfp4_swizzled(
                w.data_ptr(), packed.data_ptr(), sf.data_ptr(),
                scratch.data_ptr(), og.data_ptr(), N, K, 0)
            torch.cuda.synchronize()
            alpha[packed.data_ptr()] = float(og.item())
            return packed, sf

        def ql(src):
            ps, ss = [], []
            for w in src:
                p, s = qw(w)
                ps.append(p); ss.append(s)
            return ps, ss

        self._vlm_gu_v4, self._vlm_gu_v4sf = ql(self._vlm_gu_v)
        self._vlm_d_v4, self._vlm_d_v4sf = ql(self._vlm_d_v)
        self._vlm_gu_t4, self._vlm_gu_t4sf = ql(self._vlm_gu_t)
        self._vlm_d_t4, self._vlm_d_t4sf = ql(self._vlm_d_t)
        self._vlm_gu_N = self._vlm_gu_v[0].shape[0]
        self._vlm_D = self._vlm_gu_v[0].shape[1]
        self._vlm_inter = self._vlm_d_v[0].shape[1]
        self._vlm_fp4_ready = True
        self._fp4_alpha = alpha
        # qkv/o prefill GEMMs -> NVFP4, dispatched by the FP8 weight pointer
        # that _block passes to _fp8_gemm.
        if _env_flag("HYVLA_PREFILL_QKVO_NVFP4"):
            wmap = {}
            for bf16_list, fp8_list in (
                    (self._vlm_qkv_v, self._vlm_qkv_v8),
                    (self._vlm_o_v, self._vlm_o_v8),
                    (self._vlm_qkv_t, self._vlm_qkv_t8),
                    (self._vlm_o_t, self._vlm_o_t8)):
                for bf, f8 in zip(bf16_list, fp8_list):
                    p, s = qw(bf)
                    wmap[f8.data_ptr()] = (p, s, alpha[p.data_ptr()])
            self._fp4_weight_map = wmap
        torch.cuda.synchronize()

    def _quantize_vit_fc_fp4(self):
        """Experiment: NVFP4 W4A4 weights for the ViT fc1/fc2 (padded 4352)."""
        import flash_rt.flash_rt_kernels as fvk

        def qw(w):
            N, K = w.shape
            w = w.contiguous()
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
            sfb = torch.empty(fvk.nvfp4_sf_swizzled_bytes(N, K),
                              dtype=torch.uint8, device=w.device)
            scratch = torch.empty(1, dtype=torch.float32, device=w.device)
            og = torch.empty(1, dtype=torch.float32, device=w.device)
            fvk.bf16_weight_to_nvfp4_swizzled(
                w.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
                scratch.data_ptr(), og.data_ptr(), N, K, 0)
            torch.cuda.synchronize()
            return packed, sfb, float(og.item())

        p1, s1, a1, p2, s2, a2, b1 = [], [], [], [], [], [], []
        for i in range(len(self._vit_fc1_w)):
            w1 = torch.nn.functional.pad(
                self._vit_fc1_w[i].to(torch.bfloat16), (0, 0, 0, 48))
            w2 = torch.nn.functional.pad(
                self._vit_fc2_w[i].to(torch.bfloat16), (0, 48, 0, 0))
            q1, r1, g1 = qw(w1)
            q2, r2, g2 = qw(w2)
            p1.append(q1); s1.append(r1); a1.append(g1)
            p2.append(q2); s2.append(r2); a2.append(g2)
            b1.append(torch.nn.functional.pad(
                self._vit_fc1_b[i].to(torch.bfloat16), (0, 48)))
        self._vit_fc1_p4, self._vit_fc1_s4, self._vit_fc1_a4 = p1, s1, a1
        self._vit_fc2_p4, self._vit_fc2_s4, self._vit_fc2_a4 = p2, s2, a2
        self._vit_fc1_b_pad4 = b1
        # proj (attention output -> 1152); 1152-aligned, no padding.
        pp, ps_, pa = [], [], []
        for i in range(len(self._vit_proj_w)):
            q, s, g = qw(self._vit_proj_w[i].to(torch.bfloat16))
            pp.append(q); ps_.append(s); pa.append(g)
        self._vit_proj_p4, self._vit_proj_s4, self._vit_proj_a4 = pp, ps_, pa
        # qkv (head-padded 72->96 to N=4608 unless HYVLA_VIT_HD72).
        def pad_qkv(w):
            hd = 72 if self._vit_hd72 else 96
            w = w.reshape(3, 16, 72, w.shape[1])
            if hd == 96:
                w = torch.nn.functional.pad(w, (0, 0, 0, 24))
            return w.reshape(3 * 16 * hd, w.shape[-1]).contiguous()
        qp, qs, qa = [], [], []
        for i in range(len(self._vit_qkv_w)):
            q, s, g = qw(pad_qkv(self._vit_qkv_w[i].to(torch.bfloat16)))
            qp.append(q); qs.append(s); qa.append(g)
        self._vit_qkv_p4, self._vit_qkv_s4, self._vit_qkv_a4 = qp, qs, qa
        torch.cuda.synchronize()

    def _quantize_vit_fc2_fp4_awq(self, calib_path):
        """SmoothQuant pre-scaled NVFP4 weights for the ViT fc2 (padded 4352).

        Reads per-input-channel ``s`` / ``inv_s`` scales produced from real
        RoboTwin frames, pre-scales the (N, K) fc2 weight by ``s_k`` and
        quantizes it to NVFP4. The fused fc2 producer multiplies the activation
        by ``inv_s_k`` so the product is unchanged while the per-16 block
        dynamic range is equalized.
        """
        import json
        import flash_rt.flash_rt_kernels as fvk

        with open(calib_path) as fh:
            cal = json.load(fh)
        s_list, inv_list = cal["s"], cal["inv_s"]
        dev = self._vit_fc2_w[0].device

        p2, s2, a2, invs = [], [], [], []
        for i in range(len(self._vit_fc2_w)):
            w2 = torch.nn.functional.pad(
                self._vit_fc2_w[i].to(torch.bfloat16), (0, 48, 0, 0))
            s = torch.tensor(s_list[i], dtype=torch.bfloat16, device=dev)
            wp = (w2.float() * s.unsqueeze(0).float()).to(torch.bfloat16)
            wp = wp.contiguous()
            N, K = wp.shape
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
            sfb = torch.empty(fvk.nvfp4_sf_swizzled_bytes(N, K),
                              dtype=torch.uint8, device=dev)
            scratch = torch.empty(1, dtype=torch.float32, device=dev)
            og = torch.empty(1, dtype=torch.float32, device=dev)
            fvk.bf16_weight_to_nvfp4_swizzled(
                wp.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
                scratch.data_ptr(), og.data_ptr(), N, K, 0)
            torch.cuda.synchronize()
            p2.append(packed)
            s2.append(sfb)
            a2.append(float(og.item()))
            invs.append(torch.tensor(inv_list[i], dtype=torch.bfloat16,
                                     device=dev).contiguous())
        self._vit_fc2_p4, self._vit_fc2_s4, self._vit_fc2_a4 = p2, s2, a2
        self._vit_fc2_inv_s = invs
        torch.cuda.synchronize()

    def _quantize_fp8_block128(self):
        """Quantize the expert + VLM tower GEMM weights to FP8 block-128.

        Weights are stored in the (N, K) e4m3 layout + (N/128, K/128) fp32
        block scale consumed by ``fp8_block128_gemm_cutlass_sm120_bf16out``.
        The quantized tensors reuse the Thor FP8 slot names so the parent
        ``_block`` FP8 branch routes through ``HyVLAOrinBF16Pipeline._fp8_gemm``
        (which dispatches to the block-128 backend). ViT fc1/fc2 (K=4304) are
        not 128-aligned and remain BF16."""
        def q_list(src):
            q8s, sss = [], []
            for w in src:
                q8, ss = _block128_quant_weight(w)
                q8s.append(q8)
                sss.append(ss)
            return q8s, sss

        self._exp_qkv8, self._exp_qkv_ws = q_list(self._exp_qkv_v)
        self._exp_o8, self._exp_o_ws = q_list(self._exp_o_v)
        self._exp_gu8, self._exp_gu_ws = q_list(self._exp_gu_v)
        self._exp_d8, self._exp_d_ws = q_list(self._exp_d_v)
        self._exp_fp8_ready = True

        self._vlm_qkv_v8, self._vlm_qkv_v_ws = q_list(self._vlm_qkv_v)
        self._vlm_o_v8, self._vlm_o_v_ws = q_list(self._vlm_o_v)
        self._vlm_gu_v8, self._vlm_gu_v_ws = q_list(self._vlm_gu_v)
        self._vlm_d_v8, self._vlm_d_v_ws = q_list(self._vlm_d_v)
        self._vlm_qkv_t8, self._vlm_qkv_t_ws = q_list(self._vlm_qkv_t)
        self._vlm_o_t8, self._vlm_o_t_ws = q_list(self._vlm_o_t)
        self._vlm_gu_t8, self._vlm_gu_t_ws = q_list(self._vlm_gu_t)
        self._vlm_d_t8, self._vlm_d_t_ws = q_list(self._vlm_d_t)
        self._vlm_fp8_ready = True

        # ViT qkv/proj are 128-aligned (3456/1152); fc1/fc2 (K=N=4304) are
        # NOT 128-aligned and stay BF16. By default the qkv head-dim is padded
        # 72 -> 96 (per-head, interleaved) so the spatial attention uses the
        # native sm120 flash backend; with HYVLA_VIT_HD72 the padding is dropped
        # (the <96,64,32,4> tile runs head_dim=72 through is_even_K=false).
        def pad_qkv(w):
            hd = 72 if self._vit_hd72 else 96
            w = w.reshape(3, 16, 72, w.shape[1])
            if hd == 96:
                w = torch.nn.functional.pad(w, (0, 0, 0, 24))
            return w.reshape(3 * 16 * hd, w.shape[-1]).contiguous()

        def pad_qkv_bias(b):
            hd = 72 if self._vit_hd72 else 96
            b = b.reshape(3, 16, 72)
            if hd == 96:
                b = torch.nn.functional.pad(b, (0, 24))
            return b.reshape(3 * 16 * hd).contiguous()

        self._vit_qkv_w8, self._vit_qkv_ws = q_list(
            [pad_qkv(w) for w in self._vit_qkv_w])
        self._vit_proj_w8, self._vit_proj_ws = q_list(self._vit_proj_w)
        self._vit_qkv_b_pad = [pad_qkv_bias(b.to(torch.bfloat16))
                               for b in self._vit_qkv_b]
        self._vit_qkv_proj_fp8b128_ready = True
        self._vit_hd_pad = 72 if self._vit_hd72 else 96
        torch.cuda.synchronize()

    def _quantize_expert_fp4(self):
        """NVFP4 (W4A4) expert-tower weights: qkv/o/gu/dn of all 32 layers.

        Same wire format as the ViT/prefill NVFP4 path (per-16 UE4M3 SF +
        per-tensor global scale, swizzled). Static weights only; activation
        quant stays dynamic per-16 at run time. Dispatched by
        ``HYVLA_EXPERT_NVFP4`` (default on for the FP8 block-128 tier)."""
        import flash_rt.flash_rt_kernels as fvk
        alpha = {}

        def qw(w):
            N, K = w.shape
            w = w.to(torch.bfloat16).contiguous()
            packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
            sf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(N, K),
                             dtype=torch.uint8, device=w.device)
            scratch = torch.empty(1, dtype=torch.float32, device=w.device)
            og = torch.empty(1, dtype=torch.float32, device=w.device)
            fvk.bf16_weight_to_nvfp4_swizzled(
                w.data_ptr(), packed.data_ptr(), sf.data_ptr(),
                scratch.data_ptr(), og.data_ptr(), N, K, 0)
            torch.cuda.synchronize()
            alpha[packed.data_ptr()] = float(og.item())
            return packed, sf

        def ql(src):
            ps, ss = [], []
            for w in src:
                p, s = qw(w)
                ps.append(p); ss.append(s)
            return ps, ss

        self._exp_qkv_p4, self._exp_qkv_s4 = ql(self._exp_qkv_v)
        self._exp_o_p4, self._exp_o_s4 = ql(self._exp_o_v)
        self._exp_gu_p4, self._exp_gu_s4 = ql(self._exp_gu_v)
        self._exp_d_p4, self._exp_d_s4 = ql(self._exp_d_v)
        self._exp_fp4_alpha = alpha
        self._exp_fp4_ready = True
        torch.cuda.synchronize()

    def _quantize_vit_fc_fp8(self):
        """Zero-pad fc1/fc2 (4304 -> 4352) and quantize to FP8 block-128."""
        fc1_w8, fc1_ws, fc2_w8, fc2_ws, fc1_bp = [], [], [], [], []
        for i in range(len(self._vit_fc1_w)):
            w1 = torch.nn.functional.pad(self._vit_fc1_w[i].to(torch.bfloat16),
                                         (0, 0, 0, 48))
            w2 = torch.nn.functional.pad(self._vit_fc2_w[i].to(torch.bfloat16),
                                         (0, 48, 0, 0))
            q1, s1 = _block128_quant_weight(w1)
            q2, s2 = _block128_quant_weight(w2)
            fc1_w8.append(q1); fc1_ws.append(s1)
            fc2_w8.append(q2); fc2_ws.append(s2)
            fc1_bp.append(torch.nn.functional.pad(
                self._vit_fc1_b[i].to(torch.bfloat16), (0, 48)))
        self._vit_fc1_w8 = fc1_w8
        self._vit_fc1_ws = fc1_ws
        self._vit_fc2_w8 = fc2_w8
        self._vit_fc2_ws = fc2_ws
        self._vit_fc1_b_pad = fc1_bp
        self._vit_fp8_ready = True
        self._vit_fc_fp8b128 = True
        torch.cuda.synchronize()

    # ---- moved from the shared HyVLA Thor/Orin frontend (SM120 overrides) ----

    @torch.no_grad()
    def _assemble_prefix(self, merged):
        """merged: (num_cam, 49, 2048) merged vision tokens. Returns prefix tensors."""
        dev = self.device
        stat = self._prefix_static_embs(dev, merged.shape[0])

        prefix_embs = self._prefix_embs_from_merged(merged)
        pad_masks = torch.cat(stat["pad"], dim=1).bool()
        att_masks = torch.tensor(stat["att"], dtype=torch.bool, device=dev)[None]
        mm_prefix = torch.tensor(stat["mm"], dtype=torch.bool, device=dev)[None]
        return (prefix_embs, pad_masks, att_masks, mm_prefix,
                list(stat["idx_ranges"]), list(stat["full_ranges"]))

    def _build_graph(self, key, pmask=None, pcos=None, psin=None,
                     smask=None, scos=None, ssin=None):
        S_p, n_vis = key
        dev = self.device
        L, nkv, hd = 32, self.n_kv, self.head_dim
        # KV cache is stored PRE-EXPANDED to all query heads: the megakernel
        # replicates each KV head kv_rep times, so attention reads it directly
        # and skips the per-call repeat_interleave over the whole cache
        # (measured ~36us/call x 640 calls on the 281-row cache).
        n_kvc = self.n_heads
        D, S_s = self.d_vlm, 1 + self.chunk
        z = lambda *s, dt=_BF16: torch.zeros(*s, dtype=dt, device=dev)
        b = {
            "pe": z(1, S_p, D),
            "pmask": z(1, 1, S_p, S_p, dt=_BF16),
            "pcos": z(1, 1, S_p, hd), "psin": z(1, 1, S_p, hd),
            "smask": z(1, 1, S_s, S_p + S_s, dt=_BF16),
            "scos": z(1, 1, S_s, hd), "ssin": z(1, 1, S_s, hd),
            "state": z(1, self.max_state_dim),
            "x": z(1, self.chunk, self.max_action_dim, dt=torch.float32),
            "kbuf": z(L, 1, n_kvc, S_p + S_s, hd),
            "vbuf": z(L, 1, n_kvc, S_p + S_s, hd),
        }
        # Static masks/rope are baked once (they are prompt-fixed); only
        # prefix_embs/state/noise are re-copied per replay.
        if pmask is not None:
            b["pmask"].copy_(pmask); b["pcos"].copy_(pcos); b["psin"].copy_(psin)
            b["smask"].copy_(smask); b["scos"].copy_(scos); b["ssin"].copy_(ssin)
        # Per-shape FP8 GEMM autotune BEFORE capture (graph-safe: mutates only
        # the GemmRunner algo cache). A dry eager body records the exact (M,N,K)
        # set the captured path will hit, then we tune each on self.pipe.gemm.
        if self.use_autotune and self.use_fp8:
            self.pipe._gemm_shapes = set()
            self._captured_body(b, n_vis, S_p)
            shapes = self.pipe._gemm_shapes
            self.pipe._gemm_shapes = None
            self.pipe.autotune_gemms(shapes)
            torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._captured_body(b, n_vis, S_p)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._captured_body(b, n_vis, S_p)
        return {"graph": graph, "buf": b}

    def _build_vit_graph(self, shape):
        dev = self.device
        img = torch.zeros(shape, dtype=_BF16, device=dev)
        perm = self._prefix_perm(shape[0])

        def body():
            resized = self._resize_cam_imgs(img)
            m = self.pipe.merger_forward(self.pipe.vit_forward(resized))
            pe.copy_(self._prefix_embs_from_merged(m)[:, perm])

        merged = self.pipe.merger_forward(
            self.pipe.vit_forward(self._resize_cam_imgs(img))).clone()
        pe = self._prefix_embs_from_merged(merged)[:, perm].clone()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return {"graph": graph, "img": img, "pe": pe, "perm": perm}

    def _graph_forward(self, S_p, n_vis, prefix_embs, pmask, pcos, psin,
                       smask, scos, ssin, state_t, noise_t, use_graph=True):
        dev = self.device
        if not use_graph:
            L, nkv, hd = 32, self.n_kv, self.head_dim
            kbuf = torch.zeros(L, 1, self.n_heads, S_p + 1 + self.chunk, hd, dtype=_BF16, device=dev)
            vbuf = torch.zeros_like(kbuf)
            x = noise_t.clone().float()
            self.pipe.prefill(prefix_embs.clone(), n_vis, pmask, pcos, psin, kbuf, vbuf)
            return self.pipe.denoise(state_t, x, self._time_embs, smask, scos, ssin,
                                     kbuf, vbuf, S_p, num_steps=self.num_steps)
        g = self._graph_cache.get((S_p, n_vis))
        if g is None:
            g = self._build_graph((S_p, n_vis), pmask, pcos, psin, smask, scos, ssin)
            self._graph_cache[(S_p, n_vis)] = g
        b = g["buf"]
        b["pe"].copy_(prefix_embs)
        b["state"].copy_(state_t); b["x"].copy_(noise_t)
        g["graph"].replay()
        return b["x"].clone()

    def _preprocess_images(self, images):
        """Normalize public inputs to (num_cam,K,3,H,W) in [0,1]."""
        if isinstance(images, dict):
            keys = [k for k in self.image_keys if k in images]
            if not keys:
                keys = [k for k in ("image", "wrist_image", "wrist_image_right") if k in images]
            if not keys:
                raise ValueError("images dict does not contain configured camera keys")
            images = [images[k] for k in keys]

        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("images list must have at least one camera")
            images = torch.stack([_camera_tensor(im) for im in images], 0)
        else:
            images = torch.as_tensor(np.asarray(images))
            scale_uint8 = images.dtype == torch.uint8
            if images.ndim == 5:
                if images.shape[-1] == 3:
                    images = images.permute(0, 1, 4, 2, 3)
                elif images.shape[2] != 3:
                    raise ValueError(
                        f"images must have channel dimension of size 3, got {tuple(images.shape)}")
            elif images.ndim == 4:
                images = torch.stack([_camera_tensor(im) for im in images], 0)
            else:
                raise ValueError(
                    f"images must be list/dict or rank 4/5 tensor, got {tuple(images.shape)}")
            images = images.contiguous().float()
            if scale_uint8:
                images = images / 255.0

        if images.device.type != "cpu":
            # Already on-device (or meta): keep the direct path.
            out = []
            for cam in range(images.shape[0]):
                out.append(images[cam].to(self.device, _BF16)[None])
            return out

        # Stage the pageable CPU input through a persistent pinned host buffer so
        # each camera H2D is a pinned DMA instead of a pageable copy. The CPU
        # dtype cast is identical to the previous path; only the transfer changes.
        # The cast of camera ``c+1`` is interleaved with the pinned H2D of camera
        # ``c`` so the DMA runs under the cast instead of serializing after the
        # whole batch has been staged.
        pin = getattr(self, "_pin_stage", None)
        if pin is None or tuple(pin.shape) != tuple(images.shape):
            pin = torch.empty(tuple(images.shape), dtype=_BF16, pin_memory=True)
            self._pin_stage = pin
        dev = self.device
        out = []
        for cam in range(images.shape[0]):
            pin[cam].copy_(images[cam])
            out.append(pin[cam].to(dev, non_blocking=True)[None])
        return out

    def _vit_merge(self, imgs5, use_graph=True):
        """imgs5 (num_cam,K,3,H,W) bf16 [0,1] -> prefix_embs (1,S_p,2048).

        The resize-with-pad + normalize + ViT + merger + prefix assembly + the
        [vision|text] permutation are captured in one graph (the static token
        embeddings and perm are cached, so the body has no torch.tensor())."""
        if not use_graph:
            resized = self._resize_cam_imgs(imgs5)
            merged = self.pipe.merger_forward(self.pipe.vit_forward(resized))
            return self._prefix_embs_from_merged(merged)[
                :, self._prefix_perm(imgs5.shape[0])]
        key = tuple(imgs5.shape[:2])
        g = self._vit_graph_cache.get(key)
        if g is None:
            g = self._build_vit_graph(imgs5.shape)
            self._vit_graph_cache[key] = g
        g["img"].copy_(imgs5)
        g["graph"].replay()
        return g["pe"]

    def _build_full_graph(self, img_shape, stat):
        """One CUDA graph for the whole steady-state pipeline: resize-with-pad +
        normalize + ViT + merger + [vision|text] perm + prefill + 10-step
        denoise. Only the raw images / state / noise are graph inputs (copied
        into the static buffers before each replay); the masks/rope/perm/static
        embeddings are baked at capture time."""
        dev = self.device
        S_p, n_vis, perm = stat["S_p"], stat["n_vis"], stat["perm"]
        L, nkv, hd = 32, self.n_kv, self.head_dim
        n_kvc = self.n_heads
        D, S_s = self.d_vlm, 1 + self.chunk
        z = lambda *s, dt=_BF16: torch.zeros(*s, dtype=dt, device=dev)
        img = torch.zeros(img_shape, dtype=_BF16, device=dev)
        b = {
            "pe": z(1, S_p, D),
            "pmask": z(1, 1, S_p, S_p), "pcos": z(1, 1, S_p, hd),
            "psin": z(1, 1, S_p, hd),
            "smask": z(1, 1, S_s, S_p + S_s), "scos": z(1, 1, S_s, hd),
            "ssin": z(1, 1, S_s, hd),
            "state": z(1, self.max_state_dim),
            "x": z(1, self.chunk, self.max_action_dim, dt=torch.float32),
            "kbuf": z(L, 1, n_kvc, S_p + S_s, hd),
            "vbuf": z(L, 1, n_kvc, S_p + S_s, hd),
        }
        b["pmask"].copy_(stat["pmask"]); b["pcos"].copy_(stat["pcos"])
        b["psin"].copy_(stat["psin"])
        b["smask"].copy_(stat["smask"]); b["scos"].copy_(stat["scos"])
        b["ssin"].copy_(stat["ssin"])

        def body():
            resized = self._resize_cam_imgs(img)
            m = self.pipe.merger_forward(self.pipe.vit_forward(resized))
            b["pe"].copy_(self._prefix_embs_from_merged(m, perm))
            self.pipe.prefill(b["pe"], n_vis, b["pmask"], b["pcos"], b["psin"],
                              b["kbuf"], b["vbuf"])
            self.pipe.denoise(b["state"], b["x"], self._time_embs, b["smask"],
                              b["scos"], b["ssin"], b["kbuf"], b["vbuf"], S_p,
                              num_steps=self.num_steps)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return {"graph": graph, "img": img, "buf": b, "perm": perm}

    def _full_graph_forward(self, imgs5, stat, state_t, noise_t, use_graph=True):
        dev = self.device
        if not use_graph:
            resized = self._resize_cam_imgs(imgs5)
            merged = self.pipe.merger_forward(self.pipe.vit_forward(resized))
            pe = self._prefix_embs_from_merged(merged)[:, stat["perm"]]
            S_p, n_vis = stat["S_p"], stat["n_vis"]
            L, nkv, hd = 32, self.n_kv, self.head_dim
            kbuf = torch.zeros(L, 1, self.n_heads, S_p + 1 + self.chunk, hd,
                               dtype=_BF16, device=dev)
            vbuf = torch.zeros_like(kbuf)
            x = noise_t.clone().float()
            self.pipe.prefill(pe, n_vis, stat["pmask"], stat["pcos"],
                              stat["psin"], kbuf, vbuf)
            return self.pipe.denoise(state_t, x, self._time_embs, stat["smask"],
                                     stat["scos"], stat["ssin"], kbuf, vbuf, S_p,
                                     num_steps=self.num_steps)
        key = (tuple(imgs5.shape[:2]), stat["S_p"], stat["n_vis"])
        g = self._full_graph_cache.get(key)
        if g is None:
            g = self._build_full_graph(imgs5.shape, stat)
            self._full_graph_cache[key] = g
        g["img"].copy_(imgs5)
        g["buf"]["state"].copy_(state_t)
        g["buf"]["x"].copy_(noise_t)
        g["graph"].replay()
        return g["buf"]["x"].clone()

    @torch.no_grad()
    def _prefix_embs_from_merged(self, merged, perm=None):
        """Image-dependent prefix embeddings: merged -> [BOS|HY_USER|per-cam
        vision(+split)|lang] concatenated. Static token embeddings come from the
        cached _prefix_static_embs so this is graph-capturable (no torch.tensor).
        If ``perm`` is given the native path returns the prefix already in
        permuted order (caller must not permute again)."""
        stat = self._prefix_static_embs(self.device, merged.shape[0])
        if getattr(self, "_prefix_native", False):
            import flash_rt.flash_rt_kernels as fvk
            if hasattr(fvk, "hyvla_prefix_scatter_bf16"):
                return self._prefix_native_assemble(merged, stat, fvk, perm)
        embs = [stat["bos"], stat["hy_user"]]
        for ci in range(merged.shape[0]):
            img_emb = merged[ci][None]
            g = int(img_emb.shape[1] ** 0.5)
            embs.append(stat["vstart"])
            grid = img_emb.view(1, g, g, -1)
            split_exp = stat["vsplit"].unsqueeze(1).expand(1, g, 1, grid.shape[-1])
            embs.append(torch.cat([grid, split_exp], dim=2).reshape(1, -1, grid.shape[-1]))
            embs.append(stat["vend"])
        embs.append(stat["lang"])
        return torch.cat(embs, dim=1)

    @torch.no_grad()
    def _prefix_native_assemble(self, merged, stat, fvk, perm=None):
        """Native prefix assembly: static tokens are pre-placed once into a
        persistent buffer; the image-dependent merged vision grid is scattered
        into it per decision (removes the interleave cats/concat). ``perm``
        optionally writes every token at its permuted position (the static
        vision-first permutation derived on CPU from the mm mask)."""
        ncam = merged.shape[0]
        g = 7
        C = merged.shape[-1]
        if perm is not None:
            mm = stat["mm"]
            # pl[q] = natural position emitted at permuted slot q, i.e. the
            # caller wants natural[:, pl]. Emitting that directly means the
            # token at natural position p must be written to slot inv[p] (the
            # inverse permutation), not to pl[p].
            pl = ([i for i, v in enumerate(mm) if v]
                  + [i for i, v in enumerate(mm) if not v])
            inv = [0] * len(pl)
            for q, p in enumerate(pl):
                inv[p] = q
        else:
            pl = inv = None
        key = (ncam, C, self._prompt, None if pl is None else tuple(pl))
        cache = getattr(self, "_prefix_native_cache", None)
        if cache is None or cache.get("key") != key:
            fin = (lambda p: inv[p]) if inv is not None else (lambda p: p)
            row = g + 1
            block = g * row
            stride = 1 + block + 1
            n_lang = stat["lang"].shape[1]
            S = 2 + ncam * stride + n_lang
            buf = torch.empty(1, S, C, dtype=merged.dtype, device=merged.device)
            dest = torch.empty(ncam * g * g, dtype=torch.int32,
                               device=merged.device)
            buf[0, fin(0)] = stat["bos"][0, 0]
            buf[0, fin(1)] = stat["hy_user"][0, 0]
            for ci in range(ncam):
                base = 2 + ci * stride
                buf[0, fin(base)] = stat["vstart"][0, 0]
                buf[0, fin(base + 1 + block)] = stat["vend"][0, 0]
                for r in range(g):
                    buf[0, fin(base + 1 + r * row + g)] = stat["vsplit"][0, 0]
                    for cc in range(g):
                        dest[ci * g * g + r * g + cc] = fin(base + 1 + r * row + cc)
            for k in range(n_lang):
                buf[0, fin(2 + ncam * stride + k)] = stat["lang"][0, k]
            cache = {"key": key, "buf": buf, "dest": dest, "C": C}
            self._prefix_native_cache = cache
        buf = cache["buf"]
        src = merged.reshape(-1, C)
        fvk.hyvla_prefix_scatter_bf16(
            src.data_ptr(), buf.data_ptr(), cache["dest"].data_ptr(),
            cache["dest"].numel(), C,
            torch.cuda.current_stream().cuda_stream)
        return buf

    def _prefix_perm(self, num_cam):
        """Static [vision | non-vision] token permutation for ``num_cam``.

        Derived from the prompt-fixed ``mm`` (multimodal) mask in the static
        prefix cache: all vision tokens first, then the text/control tokens.
        Baked into the ViT graph so the prefix reorder has no eager aten math."""
        stat = self._prefix_static_embs(self.device, num_cam)
        mm = torch.tensor(stat["mm"], dtype=torch.bool, device=self.device)
        return torch.cat([torch.nonzero(mm).squeeze(-1),
                          torch.nonzero(~mm).squeeze(-1)])

    @torch.no_grad()
    def _prefix_static_embs(self, dev, num_cam):
        """Static (prompt + num_cam) prefix parts: token embeddings + the
        segment/attention mask structure, computed once and cached so the graph
        capture and per-call assembly reuse them without torch.tensor() in the
        hot path."""
        key = (self._prompt, num_cam)
        cache = getattr(self, "_prefix_static_cache", None)
        if cache is not None and cache.get("key") == key:
            return cache

        bos = self._embed_ids(torch.tensor([[_TOK_BOS]], device=dev))
        hy_user = self._embed_ids(torch.tensor([[_TOK_HY_USER]], device=dev))
        att = [1, 1]
        mm = [False, False]
        pad = [torch.ones((1, 2), dtype=torch.bool, device=dev)]
        idx_ranges, full_ranges = [], []

        vstart = self._embed_ids(torch.tensor([[_TOK_VISION_START]], device=dev))
        vend = self._embed_ids(torch.tensor([[_TOK_VISION_END]], device=dev))
        vsplit = self._embed_ids(torch.tensor([[_TOK_VISION_SPLIT]], device=dev))

        for _ in range(num_cam):
            g = 7
            att.append(1); mm.append(False)
            pad.append(torch.ones((1, 1), dtype=torch.bool, device=dev))
            row_len = g + 1
            total = g * row_len
            start = len(att)
            idx_ranges.extend([(start + r * row_len, start + r * row_len + g) for r in range(g)])
            full_ranges.append((start, start + total))
            att.extend([1] * total)
            mm.extend(([True] * g + [False]) * g)
            pad.append(torch.ones((1, total), dtype=torch.bool, device=dev))
            att.append(1); mm.append(False)
            pad.append(torch.ones((1, 1), dtype=torch.bool, device=dev))

        lang = self._embed_ids(self._lang_tokens)             # (1,64,2048)
        lang_valid = self._lang_masks[0]                      # (64,) bool
        lang = lang[:, lang_valid]                            # (1,n_valid,2048)
        n_lang = lang.shape[1]
        pad.append(torch.ones((1, n_lang), dtype=torch.bool, device=dev))
        att.extend([1] * n_lang)
        mm.extend([False] * n_lang)

        stat = {"key": key, "bos": bos, "hy_user": hy_user,
                "vstart": vstart, "vend": vend, "vsplit": vsplit, "lang": lang,
                "att": att, "mm": mm, "pad": pad,
                "idx_ranges": idx_ranges, "full_ranges": full_ranges}
        self._prefix_static_cache = stat
        return stat

    def _resize_cam_imgs(self, imgs5):
        """imgs5 (num_cam,K,3,H,W) bf16 [0,1] -> (num_cam,K,3,224,224) [-1,1].

        Graph-capturable (fixed input shape): the resize-with-pad + *2-1
        normalize now run inside the ViT graph so the preprocess has no eager
        aten math (only the pageable HtoD remains host-side)."""
        nc = imgs5.shape[0]
        out = torch.empty(nc, *imgs5.shape[1:3], 224, 224, dtype=imgs5.dtype,
                          device=imgs5.device)
        for cam in range(nc):
            _resize_with_pad(imgs5[cam], 224, 224, pad_value=0.0,
                             out=out[cam], scale=2.0, offset=-1.0)
        return out

    @torch.no_grad()
    def predict_actions(self, images, prompt=None, state=None, noise=None,
                        use_graph=True):
        """Orin variant of the Thor ``predict_actions`` with a static-prefix
        cache: everything that depends only on (prompt, num_cam) — the segment
        mask, the [vision|text] permutation, the bf16-rounded RoPE tables
        (fp64 math), and the suffix mask — is computed once per prompt and
        reused across frames. Only the image-dependent ``prefix_embs`` are
        rebuilt each call."""
        if state is not None:
            state_size = (
                state.numel() if torch.is_tensor(state)
                else np.asarray(state).size
            )
            if state_size > self.max_state_dim:
                raise ValueError(
                    f"state has {state_size} dims, max_state_dim is "
                    f"{self.max_state_dim}")
        if noise is not None:
            noise_size = (
                noise.numel() if torch.is_tensor(noise)
                else np.asarray(noise).size
            )
            want = self.chunk * self.max_action_dim
            if noise_size != want:
                raise ValueError(
                    f"noise must have {want} elements "
                    f"(chunk={self.chunk} x max_action_dim={self.max_action_dim}), "
                    f"got {noise_size}")

        if prompt is not None and prompt != self._prompt:
            self.set_prompt(prompt)
        if self._lang_tokens is None:
            raise RuntimeError("call set_prompt() before predict_actions()")
        dev = self.device

        if not torch.is_tensor(images):
            images = torch.as_tensor(np.asarray(images))
        cam_imgs = self._preprocess_images(images)
        imgs5 = torch.cat(cam_imgs, 0)
        num_cam = imgs5.shape[0]

        if state is None:
            state_t = torch.zeros(1, self.max_state_dim, device=dev, dtype=_BF16)
        else:
            source = state if torch.is_tensor(state) else np.asarray(state)
            st = torch.as_tensor(source, device=dev, dtype=_BF16).reshape(1, -1)
            if st.shape[1] < self.max_state_dim:
                st = F.pad(st, (0, self.max_state_dim - st.shape[1]))
            state_t = st

        key = (self._prompt, num_cam)
        stat = getattr(self, "_static_prefix_cache", None)
        if stat is None or stat["key"] != key:
            pstat = self._prefix_static_embs(dev, num_cam)
            pad_masks = torch.cat(pstat["pad"], dim=1).bool()
            att_masks = torch.tensor(pstat["att"], dtype=torch.bool, device=dev)[None]
            mm_prefix = torch.tensor(pstat["mm"], dtype=torch.bool, device=dev)[None]
            idx_ranges = pstat["idx_ranges"]
            full_ranges = pstat["full_ranges"]
            att2d = self._make_att_2d(pad_masks, att_masks)
            att2d = self._apply_segment_mask(att2d, idx_ranges, full_ranges)
            prefix_pos = torch.cumsum(pad_masks.long(), dim=1) - 1
            mm = mm_prefix[0]
            perm = torch.cat([torch.nonzero(mm).squeeze(-1),
                              torch.nonzero(~mm).squeeze(-1)])
            n_vis = int(mm.sum().item())
            att2d = att2d[:, perm][:, :, perm]
            prefix_pos = prefix_pos[:, perm]
            pcos, psin = self._rope_cos_sin(prefix_pos)
            S_p = pad_masks.shape[1]
            S_s = 1 + self.chunk
            suffix_pad = torch.ones(1, S_s, dtype=torch.bool, device=dev)
            suffix_att = torch.tensor([1, 1] + [0] * (self.chunk - 1),
                                      dtype=torch.bool, device=dev)[None]
            suffix_att2d = self._make_att_2d(suffix_pad, suffix_att)
            prefix_pad_2d = pad_masks[:, perm][:, None, :].expand(1, S_s, S_p)
            smask = torch.cat([prefix_pad_2d, suffix_att2d], 2)[:, None]
            suffix_pos = (pad_masks.long().sum(-1)[:, None]
                          + torch.cumsum(suffix_pad.long(), 1) - 1)
            scos, ssin = self._rope_cos_sin(suffix_pos)

            def _float_mask(m):
                # bool -> additive float mask (0 / -inf). Bit-exact with the
                # bool mask in SDPA but ~20% faster (avoids the bool->bias
                # conversion in the mem-efficient backend).
                return torch.zeros_like(m, dtype=_BF16).masked_fill_(
                    ~m, -float("inf"))

            stat = {"key": key, "perm": perm, "n_vis": n_vis, "S_p": S_p,
                    "pmask": _float_mask(att2d)[:, None], "pcos": pcos, "psin": psin,
                    "smask": _float_mask(smask), "scos": scos, "ssin": ssin}
            self._static_prefix_cache = stat

        if noise is None:
            noise_t = torch.randn(1, self.chunk, self.max_action_dim,
                                  dtype=torch.float32, device=dev)
        else:
            source = noise if torch.is_tensor(noise) else np.asarray(noise)
            noise_t = torch.as_tensor(source, device=dev, dtype=torch.float32)
            noise_t = noise_t.reshape(1, self.chunk, self.max_action_dim)

        x_t = self._full_graph_forward(imgs5, stat, state_t, noise_t,
                                       use_graph=use_graph)
        return x_t.float().cpu().numpy()


def _resize_with_pad(img, height=224, width=224, pad_value=-1.0, mode="bilinear",
                     out=None, scale=1.0, offset=0.0):
    """Pi0-style resize with aspect-preserving center pad. (B,C,H,W).

    ``scale``/``offset`` apply a fused affine to the result; ``out`` may be a
    preallocated buffer (avoids the per-camera concat)."""
    ch, cw = img.shape[2:]
    import flash_rt.flash_rt_kernels as fvk
    if (ch, cw) == (height, width):
        if out is None:
            out = torch.empty_like(img)
        else:
            out.copy_(img)
        if scale != 1.0 or offset != 0.0:
            out = out * scale + offset
        return out
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    ph, pw = max(0, height - rh), max(0, width - rw)
    t, l = ph // 2, pw // 2
    if (mode == "bilinear" and img.is_cuda and img.dtype == torch.bfloat16
            and img.is_contiguous()
            and hasattr(fvk, "hyvla_resize_pad_bilinear_scale_bf16")):
        B, C = img.shape[0], img.shape[1]
        if out is None:
            out = torch.empty(B, C, height, width, dtype=img.dtype,
                              device=img.device)
        fvk.hyvla_resize_pad_bilinear_scale_bf16(
            img.data_ptr(), out.data_ptr(), B, C, ch, cw, rh, rw, height, width,
            t, l, float(pad_value), float(scale), float(offset),
            torch.cuda.current_stream().cuda_stream)
        return out
    resized = F.interpolate(img, size=(rh, rw), mode=mode, align_corners=False)
    resized = F.pad(resized, (l, pw - l, t, ph - t), value=pad_value)
    if scale != 1.0 or offset != 0.0:
        resized = resized * scale + offset
    return resized
