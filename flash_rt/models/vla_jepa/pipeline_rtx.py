"""FlashRT — RTX/SM120 pipeline for LeRobot VLA-JEPA (LIBERO).

Dataflow (one action chunk per call = one robot decision):

    pixels (2 views)  ──► Qwen3-VL ViT tower (256 patches/view, DeepStack taps)
                            │  merged features (64/view) + 3 deepstack features
                            ▼
    input_ids ──► embed gather ──► scatter image features into the image spans
                            │      (+ deepstack add after LLM layers 0..2)
                            ▼
                      28 × Qwen3-VL decoder layer (RMSNorm → QKV GEMM →
                          fused q/k-norm + interleaved-MRoPE → FA2 causal →
                          o_proj → residual → RMSNorm → gate/up → SiLU-mul →
                          down_proj → residual)
                            │
                            ▼  gather the 32 embodied-action tokens
                      DiT-B flow head  (4 Euler steps × 16 blocks,
                          cross-attn to the embodied tokens on even layers,
                          self-attn on odd layers, AdaLN styles precomputed)
                            │
                            ▼
                      action chunk (7 × 7) → static unnormalize + gripper snap

Every shape is static and every buffer is allocated at construction, so the
whole ``forward`` can be captured as one CUDA Graph and replayed per decision.

Precision: BF16 residual streams with per-op fused kernels
(``w16a16_gemm_sm120_bf16``, ``rms_norm``, ``silu_mul_merged_bf16``) for the
bring-up tier; the FP8 / NVFP4 tiers swap the GEMM entry points without
touching the dataflow.
"""
from __future__ import annotations

import json
import math
import os

import torch

from flash_rt.frontends.torch._vla_jepa_rtx_spec import (
    CHUNK,
    DEFAULT_LLM_ATTENTION,
    DEFAULT_LLM_ATTENTION_TILE,
    DIT_ATTN_BF16_BLOCKS,
    DIT_FP4_CFG,
    DIT_HEAD_DIM,
    DIT_NUM_HEADS,
    LLM_ATTENTION_ROUTES,
    NVFP4_DOWN_ACT_SHIFT,
    NVFP4_DOWN_FP8_LAYERS,
    VIT_BF16_BLOCKS,
)


def _read_cfg(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class VlaJepaRtxPipeline:
    """Static VLA-JEPA pipeline (Qwen3-VL-2B + DiT-B) on RTX/SM120."""

    def __init__(self, qwen_dir: str, *, action_ckpt: str, weights: dict,
                 device: str = "cuda:0",
                 num_views: int = 2, chunk_size: int = CHUNK, action_dim: int = 7,
                 state_dim: int = 8, seq_len: int = 256, num_image_tokens: int = 128,
                 num_inference_steps: int = 4, timestep_buckets: int = 1000,
                 num_cond_tokens: int = 32, attention_backend: str = "fa2",
                 llm_attention: str = DEFAULT_LLM_ATTENTION,
                 llm_attention_tile: str | None = DEFAULT_LLM_ATTENTION_TILE,
                 gemm: str = "cublaslt", llm_precision: str = "bf16",
                 nvfp4_skip_edges: int = 0, nvfp4_sites: str = "mlp",
                 nvfp4_act_mode: str = "dynamic",
                 dit_precision: str = "bf16", dit_fp4_cfg: dict | None = None,
                 verbose: bool = False) -> None:
        if not isinstance(weights, dict) or "llm" not in weights or "dit" not in weights:
            raise ValueError(
                "VlaJepaRtxPipeline consumes preloaded weights, not checkpoint "
                "paths: build them with "
                "flash_rt.frontends.torch._vla_jepa_rtx_spec.load_weights() "
                "(the VlaJepaTorchFrontendRtx constructor does this).")
        from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx

        self.device = device
        self.qwen_dir = qwen_dir
        self.action_ckpt = action_ckpt
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.seq_len = int(seq_len)
        self.num_image_tokens = int(num_image_tokens)
        self.n_steps = int(num_inference_steps)
        self.timestep_buckets = int(timestep_buckets)
        self.num_cond = int(num_cond_tokens)
        # last decoder layer: post-attention work on the conditioning rows' tile
        self.tail_rows_enabled = True
        # VLM-independent head inputs on a forked graph branch (see forward)
        self.dit_prep_side = True
        self._side = None
        self.dit_attn_tile = True
        self.gemm_bias_epilogue = True
        self._tail_row0 = 0
        self.gemm_mode = gemm
        if llm_precision not in ("bf16", "fp8", "fp8_residual_safe", "nvfp4"):
            raise ValueError(
                "llm_precision must be 'bf16', 'fp8', 'fp8_residual_safe' or 'nvfp4'")
        self.llm_precision = llm_precision
        if llm_attention not in LLM_ATTENTION_ROUTES:
            raise ValueError(
                f"llm_attention must be one of {LLM_ATTENTION_ROUTES}, "
                f"got {llm_attention!r}")
        self.llm_attention = llm_attention
        if llm_attention_tile not in (None, "auto", "64", "128"):
            raise ValueError(
                "llm_attention_tile must be None, 'auto', '64' or '128', "
                f"got {llm_attention_tile!r}")
        self.llm_attention_tile = llm_attention_tile
        # The edge-layer FP8 exception came from the FP8-era accuracy budget;
        # measured on the current pipeline the all-layer NVFP4 MLP both wins
        # (15.77 vs 16.35 ms) and keeps the gate (min cosine 0.9958), so the
        # delivered default is no edge exception.
        self.nvfp4_skip_edges = int(nvfp4_skip_edges)
        self.nvfp4_sites = nvfp4_sites
        self.nvfp4_act_mode = nvfp4_act_mode
        if dit_precision not in ("bf16", "nvfp4"):
            raise ValueError("dit_precision must be 'bf16' or 'nvfp4'")
        self.dit_precision = dit_precision
        # Conditioning K/V on the SM120 block-scaled NVFP4 GEMM (activation
        # padded to the 128-row block-scale tile). It was opt-in while the
        # action head skipped three attention blocks (0.99013 vs 0.99104 min
        # cosine then); on the corrected head it passes: -0.015 ms vs the FP8
        # block-128 projection, min cosine 0.993750 on the 16-frame delivery
        # set and 0.998806 on the disjoint 8-frame set.
        self.dit_cond_fp4 = dit_precision == "nvfp4"
        # Conditioning K/V on the FP8 block-128 GEMM (per-token per-128-K
        # activation scales): the M=32 projection dispatches the kernel's
        # 64-row pingpong tile, 2.6x the bf16 cuBLASLt call at this shape
        # while keeping e4m3 activation precision.
        self.dit_cond_fp8 = dit_precision == "nvfp4" and not self.dit_cond_fp4
        self.dit_fp4_cfg_override = dict(dit_fp4_cfg or {})
        # Which DiT sites run on the skinny NVFP4 family: the FFN pair carries
        # two thirds of the head's weights and the attention projections run
        # W4A4 on every block (DIT_ATTN_BF16_BLOCKS is empty on the delivered
        # calibration). Per-site overrides remain available for revalidation.
        self.dit_fp4_sites = "all"
        self.dit_attn_bf16_blocks = DIT_ATTN_BF16_BLOCKS
        self.dit_attn_q_bf16_blocks = self.dit_attn_bf16_blocks
        self.dit_attn_o_bf16_blocks = frozenset()
        # BF16 attention blocks: fold the o-projection bias into the residual
        # add epilogue.
        self.dit_o_fuse = True

        cfg = _read_cfg(os.path.join(qwen_dir, "config.json"))
        tc = cfg["text_config"]
        self.hidden = int(tc["hidden_size"])
        self.num_layers = int(tc["num_hidden_layers"])
        self.n_q = int(tc["num_attention_heads"])
        self.n_kv = int(tc["num_key_value_heads"])
        self.head_dim = int(tc.get("head_dim") or self.hidden // self.n_q)
        self.inter = int(tc["intermediate_size"])
        self.rms_eps = float(tc.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(tc.get("rope_theta", 5_000_000.0))
        self.mrope_section = tuple(tc["rope_scaling"]["mrope_section"])
        vc = cfg["vision_config"]
        self.vis_hidden = int(vc["hidden_size"])
        self.vis_out = int(vc["out_hidden_size"])
        self.vis_head_dim = self.vis_hidden // int(vc["num_heads"])
        self.num_grid_per_side = int(vc["num_position_embeddings"] ** 0.5)
        self.deepstack_layers = len(vc["deepstack_visual_indexes"])
        self.image_token_id = int(cfg["image_token_id"])
        self.video_token_id = int(cfg.get("video_token_id", 151656))
        self.vision_start_token_id = int(cfg["vision_start_token_id"])
        self.patch_dim = 3 * int(vc["temporal_patch_size"]) * int(vc["patch_size"]) ** 2
        self.merge = int(vc["spatial_merge_size"])
        self.patches_per_view = self.num_image_tokens // self.num_views * self.merge ** 2
        self.vis_tokens_per_view = self.num_image_tokens // self.num_views

        # weights
        self._fvk = None
        self._fa2 = None
        # The repository's Qwen3-VL frontend keeps the first three ViT blocks in
        # BF16 to protect the image-embedding cosine of its own LLM path; on this
        # model the all-FP8 tower is both faster (16.80 vs 17.15 ms) and closer to
        # the reference chunk (min cosine 0.9979 vs 0.9921) - the ViT feeds the
        # same quantized decoder, so its rounding does not add an independent
        # error term. See _vla_jepa_rtx_spec.VIT_BF16_BLOCKS.
        self.vision = Qwen3VlVisionRtx(
            qwen_dir, device=device, attention_backend=attention_backend,
            bf16_backend=("cublaslt" if gemm == "cublaslt" else "w16a16"),
            bf16_first_blocks=VIT_BF16_BLOCKS,
            vla_jepa=True)
        self._apply_llm_weights(weights["llm"])
        if self.llm_precision != "bf16":
            self._pack_llm_fp8()
        if self.llm_precision == "nvfp4":
            self._pack_llm_nvfp4()
        self._setup_attention()
        self._apply_dit_weights(weights["dit"])
        self._alloc_buffers()
        if self.dit_precision == "nvfp4":
            self._pack_dit_nvfp4()
            self._alloc_dit_fp4()
        # model-specific preprocessing / postprocessing constants (staged by
        # the frontend from the checkpoint's own processor configs)
        self.resize_images_to = (224, 224)
        self.vis_grid = (int(self.patches_per_view ** 0.5), int(self.patches_per_view ** 0.5))
        self.img_mean = 0.5
        self.img_std = 0.5
        # fp32 scratches for the native preprocessing chain (area resize ->
        # antialias bicubic resize); see csrc/kernels/vla_preprocess.cu.
        h_img, w_img = self.resize_images_to
        gh_img, gw_img = self.vis_grid
        self.pre_area = torch.empty(self.num_views, 3, h_img, w_img,
                                    dtype=torch.float32, device=self.device)
        self.pre_bic = torch.empty(self.num_views, 3, gh_img * 16, gw_img * 16,
                                   dtype=torch.float32, device=self.device)
        # Exact adaptive-average pooling windows (int2 {start, end} per output
        # index), built on the host: the kernel must not recompute
        # `o * in / out` because --use_fast_math may reassociate the division
        # and move an exact window boundary by an ulp.
        self.pre_ybounds = self._aa_bounds(256, h_img)
        self.pre_xbounds = self._aa_bounds(256, w_img)
        self.llm_anchors += [self.pre_area, self.pre_bic, self.pre_ybounds,
                             self.pre_xbounds]
        self.processor = None
        self.state_mean = self.state_std = self.state_eps = None
        self.action_min = self.action_max = None
        self.clip_normalized_actions = True
        self.pre_snap_gripper_action = True
        self.binarize_gripper_action = True
        self._precompute_dit_constants()
        self._build_geometry()
        self._fvk = None
        self._fa2 = None
        self._vlk = None
        if verbose:
            print(f"VLA-JEPA pipeline: S={self.seq_len} layers={self.num_layers} "
                  f"hidden={self.hidden} views={self.num_views} dit_steps={self.n_steps}")

    # ───────────────────────── weight application ─────────────────────────

    def _apply_llm_weights(self, weights: dict) -> None:
        """Attach the frontend-loaded Qwen3-VL decoder tensors."""
        self.embed_w = weights["embed_w"]
        self.final_norm_w = weights["final_norm_w"]
        self.llm_layers = weights["layers"]
        self.llm_anchors = [self.embed_w, self.final_norm_w] + [
            t for ld in self.llm_layers for t in ld.values()]

    def _pack_llm_fp8(self) -> None:
        from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
            _quant_fp8_block128,
        )
        # 'fp8_residual_safe' keeps the residual-writing projections
        # (o_proj, down_proj) in bf16: their output feeds the late-layer
        # massive-activation channel, where FP8 noise is amplified.
        fp8_keys = (("qkv_w", "gate_up_w")
                    if self.llm_precision == "fp8_residual_safe"
                    else ("qkv_w", "o_w", "gate_up_w", "down_w"))
        for ld in self.llm_layers:
            for k in fp8_keys:
                w8, ws = _quant_fp8_block128(ld[k])
                ld[k + "_fp8"] = w8
                ld[k + "_ws"] = ws
                self.llm_anchors += [w8, ws]

    def _pack_dit_nvfp4(self) -> None:
        """Pack the DiT's projections for the skinny NVFP4 family (sm_120a).

        Every GEMM of the flow head (fused self-attn QKV, cross-attn Q, output
        projection, FFN up/down) becomes a block-scaled W4A4 ``nvfp4_skinny_gemm``
        whose fused consumers apply the bias / GELU / residual / AdaLN and emit
        the next GEMM's NVFP4 operand directly -- the same composition the
        repository's GROOT N1.7 and Pi0.5 action decoders use on this hardware.
        Activations use per-16 dynamic scales with a global scale of 1 (no
        calibration); weights carry a per-tensor global scale.
        """
        fvk = self._kernels()
        keep = []
        for blk in self.dit_blocks:
            sites = ("qkv",) if not blk["cross"] else ("q",)
            blk["fp4"] = {}
            for name in sites + ("o", "ff1", "ff2"):
                w = blk[name]["weight"]
                N, K = w.shape
                gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
                q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
                sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
                packed = torch.empty_like(q)
                sf_packed = torch.empty_like(sf)
                rc = fvk.nvfp4_skinny_quantize_weight(w.data_ptr(), gs.data_ptr(), q.data_ptr(),
                                                      sf.data_ptr(), N, K, 0, 4)
                if rc != 0:
                    raise RuntimeError(f"DiT NVFP4 weight quantize failed for [{N}, {K}]: {rc}")
                rc = fvk.nvfp4_skinny_pack_weight(q.data_ptr(), sf.data_ptr(), packed.data_ptr(),
                                                  sf_packed.data_ptr(), N, K, 0)
                if rc != 0:
                    raise RuntimeError(f"DiT NVFP4 weight pack failed for [{N}, {K}]: {rc}")
                blk["fp4"][name] = (packed.data_ptr(), sf_packed.data_ptr(), gs.data_ptr(),
                                    blk[name]["bias"].data_ptr())
                keep += [q, sf, packed, sf_packed, gs]
        self.dit_fp4_keep = keep
        if not self.dit_cond_fp4:
            return
        # Conditioning K/V projection: one wide-N NVFP4 GEMM instead of the
        # bf16 cuBLASLt call (128.6 -> 31.6 us at the delivered shape). The
        # activation is padded to the 128-row block-scale tile; rows past
        # num_cond stay zero and no consumer reads them.
        w = self.d_cond_kv["weight"]
        N, K = w.shape
        gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
        qp = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf_lin = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_quantize_weight(
            w.data_ptr(), gs.data_ptr(), qp.data_ptr(), sf_lin.data_ptr(), N, K, 0, 4)
        if rc != 0:
            raise RuntimeError(f"NVFP4 weight quantization failed for cond K/V [{N},{K}]")
        swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(N, K), dtype=torch.uint8,
                          device=self.device)
        fvk.nvfp4_sf_linear_to_swizzled(sf_lin.data_ptr(), swz.data_ptr(), N, K, True, 0)
        self.d_cond_kv_fp4 = (qp, swz, float(gs.item()))
        self.dit_fp4_keep += [qp, sf_lin, swz, gs]
        self.cond4 = torch.empty(128, K // 2, dtype=torch.uint8, device=self.device)
        self.cond4sf = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(128, K),
                                   dtype=torch.uint8, device=self.device)
        self.dit_fp4_keep += [self.cond4, self.cond4sf]

    def _alloc_dit_fp4(self) -> None:
        """NVFP4 operand / partials buffers for the DiT action head."""
        fvk = self._kernels()
        d = torch.device(self.device)
        N, D, FF = self.dit_seq, self.dim_dit, self.dit_ff_dim
        cfg = dict(DIT_FP4_CFG)
        cfg.update(self.dit_fp4_cfg_override)
        for (n, k), c in cfg.items():
            if not fvk.nvfp4_skinny_config_supports(c, n, k):
                raise ValueError(f"NVFP4 DiT config {c} does not tile N={n}, K={k}")
        self.dit_fp4_cfg = cfg
        floats = max((k // fvk.nvfp4_skinny_config_k_chunk(c)) * N * n
                     for (n, k), c in cfg.items())
        # attention output as the o projection's NVFP4 operand (emitted by the
        # partials attention itself)
        self.dit_ao_fp4 = True
        # self-attention blocks: Q/K/V projection fused into the attention
        # launch (needs the single-split K = 768 projection config)
        self.dit_qkv_fused = (D == 768 and N <= 48 and self.dit_head_dim == 64
                              and fvk.nvfp4_skinny_config_k_chunk(cfg[(3 * D, D)]) == D)
        self.dit_q_cross_fused = (D == 768 and N <= 48 and self.dit_head_dim == 64
                                  and D // fvk.nvfp4_skinny_config_k_chunk(cfg[(D, D)]) == 3)
        self.dit_ao4 = torch.zeros(N, D // 2, dtype=torch.uint8, device=d)
        self.dit_aosf4 = torch.zeros(N, D // 16, dtype=torch.uint8, device=d)
        self.dit_xn4 = torch.zeros(N, D // 2, dtype=torch.uint8, device=d)
        self.dit_xnsf4 = torch.zeros(N, D // 16, dtype=torch.uint8, device=d)
        self.dit_hid4 = torch.zeros(N, FF // 2, dtype=torch.uint8, device=d)
        self.dit_hidsf4 = torch.zeros(N, FF // 16, dtype=torch.uint8, device=d)
        self.dit_partials = torch.zeros(floats, dtype=torch.float32, device=d)
        self.dit_one = torch.ones(1, dtype=torch.float32, device=d)

    def _pack_llm_nvfp4(self) -> None:
        """NVFP4 (e2m1 + UE4M3 block-16 scales, per-tensor global scale) for the
        decoder MLP GEMMs: gate/up and down carry the bulk of the prefill FLOPs
        and their inputs are post-RMSNorm activations with bounded tails."""
        fvk = self._kernels()
        for ld in self.llm_layers:
            for k in ("gate_up_w", "down_w", "qkv_w", "o_w"):
                w = ld[k]
                N, K = w.shape
                gs = (w.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
                qp = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
                sf_lin = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
                rc = fvk.nvfp4_skinny_quantize_weight(
                    w.data_ptr(), gs.data_ptr(), qp.data_ptr(), sf_lin.data_ptr(), N, K, 0, 4)
                if rc != 0:
                    raise RuntimeError(f"NVFP4 weight quantization failed for [{N},{K}]")
                swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(N, K), dtype=torch.uint8,
                                  device=self.device)
                fvk.nvfp4_sf_linear_to_swizzled(sf_lin.data_ptr(), swz.data_ptr(), N, K, True, 0)
                ld[k + "_fp4"] = qp
                ld[k + "_fp4sf"] = swz
                ld[k + "_fp4alpha"] = float(gs.item())
                self.llm_anchors += [qp, sf_lin, swz, gs]
        fvk = self._kernels()
        S, I = self.seq_len, self.inter
        # SwiGLU epilogue fold (the repository Qwen3 prefill / GR00T N1.7
        # backbone route): gate and up quantized separately, rows interleaved
        # (gate i -> 2i, up i -> 2i + 1); the GEMM epilogue applies
        # silu(alpha_g * gate) * alpha_u * up and emits the down projection's
        # NVFP4 operand scales, an even-column compaction packs its data.
        # Replaces [gate_up GEMM bf16 -> SiLU*up->NVFP4 producer] on every
        # NVFP4-down layer: -0.10 ms paired against the merger-fc2 NVFP4 tier
        # at the same margin (16-frame delivery set min cosine 0.992235,
        # disjoint 8-frame set 0.998907).
        self.swiglu_fold = True
        self.down_act_shift = dict(NVFP4_DOWN_ACT_SHIFT)
        self.swiglu_fold_skip = frozenset()
        if self.swiglu_fold:
            for ld in self.llm_layers:
                w = ld["gate_up_w"]
                parts = []
                for half in (w[:I], w[I:]):
                    half = half.contiguous()
                    gs = (half.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
                    q = torch.empty(I, self.hidden // 2, dtype=torch.uint8, device=self.device)
                    sf = torch.empty(I, self.hidden // 16, dtype=torch.uint8, device=self.device)
                    rc = fvk.nvfp4_skinny_quantize_weight(half.data_ptr(), gs.data_ptr(), q.data_ptr(),
                                                          sf.data_ptr(), I, self.hidden, 0, 4)
                    if rc != 0:
                        raise RuntimeError(f"NVFP4 SwiGLU weight quantization failed: {rc}")
                    parts.append((q, sf, float(gs.item())))
                (gq, gsf, ga), (uq, usf, ua) = parts
                il = torch.empty(2 * I, self.hidden // 2, dtype=torch.uint8, device=self.device)
                il[0::2], il[1::2] = gq, uq
                il_sf = torch.empty(2 * I, self.hidden // 16, dtype=torch.uint8, device=self.device)
                il_sf[0::2], il_sf[1::2] = gsf, usf
                swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(2 * I, self.hidden), dtype=torch.uint8,
                                  device=self.device)
                fvk.nvfp4_sf_linear_to_swizzled(il_sf.data_ptr(), swz.data_ptr(), 2 * I, self.hidden,
                                                False, 0)
                ld["gate_up_il"], ld["gate_up_il_sf"] = il, swz
                ld["gate_up_il_alpha"] = (ga, ua)
                self.llm_anchors += [il, swz]
            self.swiglu_fold_out = torch.zeros(S, I, dtype=torch.uint8, device=self.device)
        self.xf4 = torch.empty(S, self.hidden // 2, dtype=torch.uint8, device=self.device)
        self.xsf4 = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(S, self.hidden),
                                dtype=torch.uint8, device=self.device)
        self.af4 = torch.empty(S, I // 2, dtype=torch.uint8, device=self.device)
        self.asf4 = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(S, I), dtype=torch.uint8,
                                device=self.device)
        # real-data static activation calibration: per-16-channel group amax of
        # the post-RMSNorm MLP input (gate_up) and of the SiLU output (down),
        # accumulated over representative observations by calibrate_nvfp4().
        self.nvfp4_calibrated = False
        # 'dynamic' per-block amax, 'secondmax' (outlier-robust scale search),
        # or 'static' calibrated group amax
        # 'mse' searches per-block scales (8 percent lower reconstruction error
        # on real activations, measured +0.83 ms E2E for +0.002 min cosine);
        # 'dynamic' is the delivered default because the frozen gate is what
        # decides and the faster passing config wins. 'mse:<site>[,<site>]'
        # restricts the search to those sites (qkv / o / gup / down).
        self.nvfp4_clip_scale = 1.0
        # 'secondmax' mode: outlier-robust per-16 scale = second-largest
        # magnitude x this factor (1.0 = hard outlier clip, >= ~16 = dynamic).
        self.nvfp4_secondmax_mult = 1.0
        # Per-layer gate for the 'down' site: the calibrated exception list
        # above, or a runtime selection via select_nvfp4_down_layers().
        self.nvfp4_down_layers = frozenset(
            set(range(self.num_layers)) - NVFP4_DOWN_FP8_LAYERS)
        qd = self.n_q * self.head_dim
        self.xq4 = torch.empty(S, self.hidden // 2, dtype=torch.uint8, device=self.device)
        self.xqsf4 = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(S, self.hidden),
                                 dtype=torch.uint8, device=self.device)
        self.xo4 = torch.empty(S, qd // 2, dtype=torch.uint8, device=self.device)
        self.xosf4 = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(S, qd),
                                 dtype=torch.uint8, device=self.device)
        self.o_group_amax = torch.zeros(self.num_layers, self.hidden // 16,
                                        dtype=torch.float32, device=self.device)
        self.gup_group_amax = torch.zeros(self.num_layers, self.hidden // 16,
                                          dtype=torch.float32, device=self.device)
        self.down_group_amax = torch.zeros(self.num_layers, self.inter // 16,
                                           dtype=torch.float32, device=self.device)
        self.llm_anchors += [self.af4, self.asf4, self.xf4, self.xsf4,
                             self.xq4, self.xqsf4, self.xo4, self.xosf4]
        self.down4 = None

    def _setup_attention(self) -> None:
        """Build the RTX attention backend that owns the decoder's Q/K/V/O
        scratch and its route dispatch. The route is a model-level choice
        (constructor ``llm_attention``); the layer code never calls an
        attention kernel directly.

        The delivered default is the repository all-FP8 FMHA prefill kernel,
        whose NVFP4 epilogue *is* the o-projection operand (byte-identical to
        the standalone quantizer), so the bf16 O round-trip and that
        quantizer launch both disappear. Measured on 16 real frames at the
        delivered boundary: 15.02 ms / min cosine 0.99517 vs 15.47 ms /
        0.99697 for the bf16 FA2 route.
        """
        from flash_rt.hardware.rtx.attn_backend_qwen3 import (
            RtxFlashAttnBackendQwen3,
        )
        S = self.seq_len
        qd = self.n_q * self.head_dim
        kv = self.n_kv * self.head_dim
        if self.llm_attention in ("fp8", "fp8_fp4out", "sage2") and S % 128:
            raise ValueError(
                f"llm_attention={self.llm_attention!r} requires the fused "
                f"norm+RoPE prefill producers, which need seq_len % 128 == 0; "
                f"got {S}")
        if self.llm_attention in ("fp8", "fp8_fp4out") and (
                (self.n_q, self.n_kv) not in ((32, 8), (16, 8))):
            raise ValueError(
                "llm_attention='fp8'/'fp8_fp4out' use the all-fp8 FMHA prefill "
                "kernel, which is compiled for the 32/8 and 16/8 GQA shapes; "
                "use 'sage2' for any other head count")
        self._attn = RtxFlashAttnBackendQwen3(
            max_seq=S, max_q_seq=S, num_layers=self.num_layers,
            num_q_heads=self.n_q, num_kv_heads=self.n_kv,
            head_dim=self.head_dim, device=self.device,
            prefill_attention=self.llm_attention,
            prefill_tile=self.llm_attention_tile)
        # The fused norm+RoPE producers write straight into the backend's
        # staging buffers; aliasing them here keeps the pointer math in one
        # place and keeps the tensors alive for the model's lifetime.
        self.q_buf = self._attn.Q_buf.view(S, qd)
        self.attn_out = self._attn.O_buf.view(S, qd)
        ops = self._attn.prefill_operands()
        if ops is None:
            self.q8_attn = self.k8_attn = self.v8_attn = None
            self.v16_attn = self.qs_attn = self.ks_attn = None
        else:
            self.q8_attn = ops["q_fp8"].view(S, qd)
            self.k8_attn = ops["k_fp8"].view(S, kv)
            if "v_fp8" in ops:
                self.v8_attn = ops["v_fp8"].view(S, kv)
                self.v16_attn = self.qs_attn = self.ks_attn = None
            else:
                self.v16_attn = ops["v_fp16"].view(S, kv)
                self.qs_attn = ops["q_scale"].view(self.n_q, S)
                self.ks_attn = ops["k_scale"].view(self.n_kv, S)
                self.v8_attn = None

    def _llm_attention(self, L: int, stream: int) -> None:
        """Layer L's causal prefill attention, dispatched by the RTX backend.

        The fused norm+RoPE producer writes whichever Q/K/V staging buffers
        the route needs (bf16, e4m3, or e4m3 + per-token scales); the backend
        then selects FA2 / all-FP8 FMHA / sage2. On ``fp8_fp4out`` the
        attention kernel emits the o-projection's NVFP4 operand itself.
        """
        fvk = self._fvk
        lw = self.llm_layers[L]
        S, nq, nkv = self.seq_len, self.n_q, self.n_kv
        q_dim, kv_dim = nq * self.head_dim, nkv * self.head_dim
        qkv_n = q_dim + 2 * kv_dim
        route = self._attn.prefill_route
        if route in ("fp8", "fp8_fp4out"):
            ops = self._attn.prefill_operands()
            r = fvk.qwen3_qk_norm_rope_prefill_v3_fp8_direct(
                self.qkv_out.data_ptr(), self.qkv_out.data_ptr() + q_dim * 2,
                self.qkv_out.data_ptr() + (q_dim + kv_dim) * 2,
                lw["q_norm_w"].data_ptr(), lw["k_norm_w"].data_ptr(),
                self.mcos.data_ptr(), self.msin.data_ptr(),
                ops["q_fp8"].data_ptr(), ops["k_fp8"].data_ptr(),
                ops["v_fp8"].data_ptr(), nq, nkv, S, qkv_n, self.rms_eps,
                stream)
            if r != 0:
                raise RuntimeError(f"fused qkv norm+rope failed: {r}")
        elif route == "sage2":
            ops = self._attn.prefill_operands()
            slot = self._attn.get_slot_ptrs("full", L)
            r = fvk.qwen3_q_norm_rope_qstage_prefill_v3_fp8(
                self.qkv_out.data_ptr(), lw["q_norm_w"].data_ptr(),
                self.mcos.data_ptr(), self.msin.data_ptr(), 0,
                ops["q_fp8"].data_ptr(), ops["q_scale"].data_ptr(),
                nq, S, qkv_n, q_dim, self.rms_eps, stream)
            if r != 0:
                raise RuntimeError(f"q_norm_rope v3_fp8 failed: {r}")
            r = fvk.qwen3_k_norm_rope_kvwrite_prefill_v3_fp8(
                self.qkv_out.data_ptr() + q_dim * 2,
                self.qkv_out.data_ptr() + (q_dim + kv_dim) * 2,
                lw["k_norm_w"].data_ptr(), self.mcos.data_ptr(),
                self.msin.data_ptr(), slot["K"], slot["V"],
                ops["k_fp8"].data_ptr(), ops["k_scale"].data_ptr(),
                ops["v_fp16"].data_ptr(), nkv, S, qkv_n, kv_dim,
                self.rms_eps, stream)
            if r != 0:
                raise RuntimeError(f"k_norm_rope v3_fp8 failed: {r}")
        else:
            slot = self._attn.get_slot_ptrs("full", L)
            fvk.qwen3_q_norm_rope_qstage_prefill_bf16(
                self.qkv_out.data_ptr(), lw["q_norm_w"].data_ptr(),
                self.mcos.data_ptr(), self.msin.data_ptr(),
                self.q_buf.data_ptr(), nq, S, qkv_n, q_dim, self.rms_eps,
                stream)
            fvk.qwen3_k_norm_rope_kvwrite_prefill_bf16(
                self.qkv_out.data_ptr() + q_dim * 2,
                self.qkv_out.data_ptr() + (q_dim + kv_dim) * 2,
                lw["k_norm_w"].data_ptr(), self.mcos.data_ptr(),
                self.msin.data_ptr(), slot["K"], slot["V"], nkv, S,
                qkv_n, kv_dim, self.rms_eps, stream)
        if route == "fp8_fp4out" and self._use_fp4_site("o"):
            self._attn.set_o_fp4(self.xo4.data_ptr(), self.xosf4.data_ptr())
        else:
            self._attn.clear_o_fp4()
        out = self._attn.run("full", L, q_seq=S, kv_seq=S, stream=stream,
                             causal=True)
        if out != self.attn_out.data_ptr():
            raise RuntimeError(
                "attention backend returned an unexpected output buffer")

    def _apply_dit_weights(self, weights: dict) -> None:
        """Attach the frontend-loaded DiT-B flow-head tensors and derive the
        head geometry from them."""
        self.d_act_enc = weights["d_act_enc"]
        self.d_state_enc = weights["d_state_enc"]
        self.d_future = weights["d_future"]
        self.d_pos_emb = weights["d_pos_emb"]
        self.d_proj_out1 = weights["d_proj_out1"]
        self.d_proj_out2 = weights["d_proj_out2"]
        self.d_time_emb = weights["d_time_emb"]
        self.d_act_dec = weights["d_act_dec"]
        self.dit_blocks = weights["blocks"]
        self.d_cond_kv = weights["d_cond_kv"]
        if self.dit_cond_fp8:
            from flash_rt.frontends.torch._qwen3_vl_vision_rtx import (
                _quant_fp8_block128,
            )
            w8, wsk = _quant_fp8_block128(self.d_cond_kv["weight"])
            self.d_cond_kv["w8"] = w8
            self.d_cond_kv["ws"] = wsk
        self.dit_cross_index = [i for i, blk in enumerate(self.dit_blocks)
                                if blk["cross"]]
        self.dit_anchor = [t for d in (self.d_act_enc, self.d_state_enc, self.d_proj_out1,
                                       self.d_proj_out2, self.d_time_emb, self.d_act_dec)
                           for v in d.values()
                           for t in (v.values() if isinstance(v, dict) else (v,))] + [
            self.d_future, self.d_pos_emb] + [
            t for blk in self.dit_blocks for v in blk.values()
            if isinstance(v, dict) for t in v.values()]

        self.dim_dit = self.dit_blocks[0]["q"]["weight"].shape[0]          # 768
        self.dit_head_dim = DIT_HEAD_DIM
        self.dit_heads = DIT_NUM_HEADS
        if self.dim_dit != self.dit_heads * self.dit_head_dim:
            raise ValueError(
                f"DiT width {self.dim_dit} != {self.dit_heads} heads x "
                f"{self.dit_head_dim}; check action_num_heads / "
                "action_attention_head_dim in the checkpoint config")
        self.dit_ff_dim = self.dit_blocks[0]["ff1"]["weight"].shape[0]     # 3072
        self.dit_dec_h = self.d_act_dec["layer1"]["weight"].shape[0]       # 1024
        self.cond_dim = self.dit_blocks[0]["k"]["weight"].shape[1]         # 2048
        self.dit_seq = 1 + self.num_cond + self.chunk_size                 # 40

    # ───────────────────────── buffers ─────────────────────────

    def _alloc_buffers(self) -> None:
        d = torch.device(self.device)
        bf16 = torch.bfloat16
        S = self.seq_len
        H, I = self.hidden, self.inter
        n_patch = self.num_views * self.patches_per_view

        self.h = torch.empty(S, H, dtype=bf16, device=d)
        # staged prompt embeddings (text rows per task, image rows per
        # decision) read by decoder layer 0; NVFP4 decoder tier only
        self.embed_hoist = self.llm_precision == "nvfp4"
        self.h0 = torch.zeros(S, H, dtype=bf16, device=d)
        self.xn = torch.empty(S, H, dtype=bf16, device=d)
        self.qkv_out = torch.empty(S, (self.n_q + 2 * self.n_kv) * self.head_dim, dtype=bf16, device=d)
        # Q/O and the KV cache are owned by the attention backend (self._attn,
        # built in _setup_attention); q_buf/attn_out alias its buffers.
        self.proj_out = torch.empty(S, H, dtype=bf16, device=d)
        self.gate_up = torch.empty(S, 2 * I, dtype=bf16, device=d)
        self.act = torch.empty(S, I, dtype=bf16, device=d)
        self.down_out = torch.empty(S, H, dtype=bf16, device=d)
        # FP8 tier scratch (per-token, per-128-K-block activation scales)
        self.xf8 = torch.empty(S, H, dtype=torch.float8_e4m3fn, device=d)
        self.xs8 = torch.empty(S, H // 128, dtype=torch.float32, device=d)
        self.af8 = torch.empty(S, I, dtype=torch.float8_e4m3fn, device=d)
        self.as8 = torch.empty(S, I // 128, dtype=torch.float32, device=d)
        self.attnf8 = torch.empty(S, self.n_q * self.head_dim,
                                  dtype=torch.float8_e4m3fn, device=d)
        self.attns8 = torch.empty(S, self.n_q * self.head_dim // 128,
                                  dtype=torch.float32, device=d)

        D, FF = self.dim_dit, self.dit_ff_dim
        self.dit_x = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_xn = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_q = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_k = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_v = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_o = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_tmp = torch.empty(self.dit_seq, D, dtype=bf16, device=d)
        self.dit_qkv = torch.empty(self.dit_seq, 3 * D, dtype=bf16, device=d)
        self.dit_ff = torch.empty(self.dit_seq, FF, dtype=bf16, device=d)
        self.dit_cat = torch.empty(self.chunk_size, 2 * D, dtype=bf16, device=d)
        # DiT tail folds: ReLU in the output GEMM's epilogue, and the action
        # encoder's first linear written straight into the left half of a
        # per-step [. | time] buffer (the concat launch goes away)
        self.dit_tail_fold = (self.gemm_mode == 'cublaslt'
                              and self.gemm_bias_epilogue)
        self.dit_cat_steps = torch.zeros(self.n_steps, self.chunk_size, 2 * D, dtype=bf16, device=d)
        self.dit_act = torch.empty(self.chunk_size, D, dtype=bf16, device=d)
        # Cross-attention K/V live in one wide buffer (row stride 2*ncross*D)
        # that a single GEMM fills; each cross block reads its slice with that
        # stride (partials_attention's kv_ld / FA2's k_strides).
        nc = len(self.dit_cross_index)
        cond_rows = 128 if self.dit_cond_fp4 else self.num_cond
        self.dit_cond_kv = torch.empty(cond_rows, 2 * nc * D, dtype=bf16, device=d)
        self.dit_cond_ld = 2 * nc * D
        self.dit_cond_k = [None] * 16
        self.dit_cond_v = [None] * 16
        for ord_, i in enumerate(self.dit_cross_index):
            self.dit_cond_k[i] = self.dit_cond_kv[:self.num_cond,
                                                  ord_ * D:(ord_ + 1) * D]
            self.dit_cond_v[i] = self.dit_cond_kv[:self.num_cond,
                                                  (nc + ord_) * D:(nc + ord_ + 1) * D]
        self.dit_lse = torch.empty(1, self.dit_heads, self.dit_seq, dtype=torch.float32, device=d)
        self.dit_pred = torch.empty(self.chunk_size, self.dit_dec_h, dtype=bf16, device=d)
        # proj_out_2 composed with the action decoder's first linear (no
        # nonlinearity between them): one 768->1024 GEMM per step instead of
        # 768->1024->1024. 16-frame delivery set: -0.02 ms, min cosine
        # 0.992235 -> 0.992801 (the bf16 rounding of the intermediate goes).
        self.dit_out_merge = True
        self.dit_enc_fold = True
        if self.dit_out_merge:
            # W1 (W2 x + b2) + b1 = (W1 W2) x + (W1 b2 + b1), composed in fp32
            w2, b2 = self.d_proj_out2["weight"].float(), self.d_proj_out2["bias"].float()
            w1, b1 = self.d_act_dec["layer1"]["weight"].float(), self.d_act_dec["layer1"]["bias"].float()
            self.d_out_merged = ((w1 @ w2).to(bf16).contiguous(), (w1 @ b2 + b1).to(bf16).contiguous())
        self.dit_dec1 = torch.empty(self.chunk_size, self.dit_dec_h, dtype=bf16, device=d)
        self.dit_vel = torch.empty(self.chunk_size, self.action_dim, dtype=bf16, device=d)
        self.actions = torch.zeros(self.chunk_size, self.action_dim, dtype=bf16, device=d)

        self.input_ids = torch.zeros(S, dtype=torch.int64, device=d)
        # raw E2E boundary: camera frames (uint8, CHW so the host staging copy
        # and the graph's H2D memcpy are flat), raw state, delivered chunk
        self.raw_frames = torch.zeros(self.num_views, 3, 256, 256, dtype=torch.uint8,
                                      device=d)
        self.raw_state = torch.zeros(self.state_dim, dtype=torch.float32, device=d)
        self.out_actions = torch.zeros(self.chunk_size, self.action_dim,
                                       dtype=torch.float32, device=d)
        self.gripper_dim = 6
        self.gripper_threshold = 0.5
        self.pix = torch.empty(n_patch, self.patch_dim, dtype=bf16, device=d)
        self.pos_embeds = torch.empty(n_patch, self.vis_hidden, dtype=bf16, device=d)
        self.vcos = torch.empty(n_patch, self.vis_head_dim // 2, dtype=bf16, device=d)
        self.vsin = torch.empty(n_patch, self.vis_head_dim // 2, dtype=bf16, device=d)
        self.mcos = torch.zeros(S, self.head_dim // 2, dtype=bf16, device=d)
        self.msin = torch.zeros(S, self.head_dim // 2, dtype=bf16, device=d)
        self.state_norm = torch.zeros(1, 1, self.state_dim, dtype=bf16, device=d)
        self.emb_idx = torch.zeros(self.num_cond, dtype=torch.int64, device=d)
        # 128 rows: the NVFP4 conditioning projection pads the 32 conditioning
        # tokens to the block-scale tile (the tail rows stay zero and no
        # consumer reads them).
        self.cond_pad = torch.zeros(128, self.cond_dim, dtype=bf16, device=d)
        self.cond = self.cond_pad[:self.num_cond]
        # FP8 block-128 operand for the conditioning K/V projection: the
        # kernel's M<=64 pingpong tile runs the real 32 rows (no padding).
        self.cond_f8 = torch.empty(self.num_cond, self.cond_dim,
                                   dtype=torch.float8_e4m3fn, device=d)
        self.cond_fs = torch.empty(self.num_cond, self.cond_dim // 128,
                                   dtype=torch.float32, device=d)
        # DiT step-input head rows: [state token | future tokens], refreshed
        # into dit_x with one copy per Euler step (state token written per
        # decision, future tokens constant).
        self.dit_head = torch.empty(1 + self.num_cond, D, dtype=bf16, device=d)
        self.dit_head[1:].copy_(self.d_future[:self.num_cond])
        self.state_tok = self.dit_head[:1]

        # per-step DiT constants (filled by _precompute_dit_constants)
        self.st_norm1_s = torch.empty(self.n_steps, 16, D, dtype=bf16, device=d)
        self.st_norm1_b = torch.empty(self.n_steps, 16, D, dtype=bf16, device=d)
        self.st_q_scale = torch.empty(self.n_steps, D, dtype=bf16, device=d)
        self.st_q_share = torch.empty(self.n_steps, D, dtype=bf16, device=d)
        self.st_time = torch.empty(self.n_steps, self.chunk_size, D, dtype=bf16, device=d)
        self.st_pos = self.d_pos_emb[:self.chunk_size]

    def _precompute_dit_constants(self) -> None:
        """The 4 Euler timesteps are fixed, so every time-conditioned DiT term
        (per-block AdaLN styles, output style, ActionEncoder time embedding) is
        computed once here and uploaded as a static graph input."""
        dev = torch.device(self.device)

        def lin(x, w, b):
            y = x @ w.t().to(x.dtype)
            return y if b is None else y + b.to(x.dtype)

        def lin32(x, w, b):
            y = x.float() @ w.t().float()
            return y if b is None else y + b.float()

        def silu(x):
            return torch.nn.functional.silu(x.float()).to(x.dtype)

        emb_dim, half = 256, 128
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64,
                                                            device=dev) / (half - 1.0))
        styles_s, styles_b, qs, qsh, times = [], [], [], [], []
        te = self.d_time_emb
        step_t = [int(s / float(max(self.n_steps, 1)) * self.timestep_buckets)
                  for s in range(self.n_steps)]
        for t in step_t:
            # diffusers get_timestep_embedding: emb = t * exp(exponent), then
            # cat([sin, cos]) flipped to [cos, sin] for flip_sin_to_cos=True.
            args = float(t) * freqs
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=0).to(torch.float32)
            temb = lin32(silu(lin32(emb[None], te["linear_1"]["weight"], te["linear_1"]["bias"])),
                         te["linear_2"]["weight"], te["linear_2"]["bias"])
            times.append(temb)
            for i in range(16):
                blk = self.dit_blocks[i]
                sb = lin32(silu(temb), blk["norm1"]["weight"],
                           blk["norm1"]["bias"]).view(2, self.dim_dit)
                styles_s.append(sb[0])
                styles_b.append(sb[1])
            shsc = lin32(silu(temb), self.d_proj_out1["weight"],
                         self.d_proj_out1["bias"]).view(2, self.dim_dit)
            qsh.append(shsc[0])
            qs.append(shsc[1])
        self.st_norm1_s.copy_(torch.stack(styles_s).view(self.n_steps, 16, self.dim_dit))
        self.st_norm1_b.copy_(torch.stack(styles_b).view(self.n_steps, 16, self.dim_dit))
        self.st_q_scale.copy_(torch.stack(qs))
        self.st_q_share.copy_(torch.stack(qsh))
        dhalf = self.dim_dit // 2
        freq2 = torch.exp(-torch.arange(dhalf, dtype=torch.float32, device=dev)
                          * (math.log(10000.0) / max(dhalf, 1)))
        rows = []
        for t in step_t:
            fr = torch.full((self.chunk_size,), float(t), device=dev)[:, None] * freq2[None, :]
            rows.append(torch.cat([torch.sin(fr), torch.cos(fr)], dim=-1))
        self.st_time.copy_(torch.stack(rows).to(torch.bfloat16))
        # the action encoder's concat input per step: [layer1 out | time]
        self.dit_cat_steps[:, :, self.dim_dit:].copy_(self.st_time)
        # layer1 and layer2 of the action encoder have no nonlinearity between
        # them, and layer2's time half is constant per step: compose them into
        # one K = action_dim linear with a per-step bias
        #   W2[:, :D] (W1 a + b1) + W2[:, D:] t_s + b2 = W12 a + c_s
        ae, D = self.d_act_enc, self.dim_dit
        w2 = ae["layer2"]["weight"].float()
        self.dit_enc_w12 = (w2[:, :D] @ ae["layer1"]["weight"].float()).to(torch.bfloat16).contiguous()
        c = (w2[:, :D] @ ae["layer1"]["bias"].float() + ae["layer2"]["bias"].float())[None, :] \
            + self.st_time[:, 0].float() @ w2[:, D:].t()
        self.dit_enc_c = c.to(torch.bfloat16).contiguous()


    # ───────────────────────── geometry ─────────────────────────

    def _build_geometry(self) -> None:
        from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

        self._geo = geo
        self._mrope_cos_cache, self._mrope_sin_cache = geo.build_mrope_cache(
            max_pos=self.seq_len + self.num_grid_per_side,
            head_dim=self.head_dim, rope_theta=self.rope_theta, device=self.device)
        self._vision_rope_cos_cache, self._vision_rope_sin_cache = geo.build_vision_rope_cache(
            max_hw=self.num_views * self.patches_per_view * self.merge,
            head_dim=self.vis_head_dim, device=self.device)

    def stage_prompt(self, input_ids, image_grid) -> None:
        """Stage a tokenized prompt into the static buffers (called once per
        task/prompt, outside the captured region)."""
        ii = torch.as_tensor(input_ids, dtype=torch.long).view(-1)
        S = self.seq_len
        text_len = int(ii.numel())
        if text_len < S:
            ii = torch.cat([ii, torch.full((S - text_len,), 0, dtype=torch.long)])
        self.text_len = text_len
        self.input_ids.copy_(ii[:S].to(self.input_ids.device))

        grid = torch.as_tensor(image_grid, dtype=torch.long)
        pos_ids = self._geo.mrope_position_ids(
            ii.cpu()[:S], grid.cpu(), None, image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            spatial_merge_size=self.merge)
        mcos, msin = self._geo.mrope_cos_sin_cached(
            pos_ids, self._mrope_cos_cache, self._mrope_sin_cache,
            mrope_section=self.mrope_section)
        pad = S - mcos.shape[0]
        if pad > 0:
            mcos = torch.cat([mcos, mcos[-1:].expand(pad, -1)], 0)
            msin = torch.cat([msin, msin[-1:].expand(pad, -1)], 0)
        self.mcos.copy_(mcos[:S].to(self.mcos.device))
        self.msin.copy_(msin[:S].to(self.msin.device))

        segs = self._geo.vision_segments(
            ii.cpu()[:S], grid.cpu(), None, image_token_id=self.image_token_id,
            video_token_id=self.video_token_id, spatial_merge_size=self.merge)
        self.vision_spans = [(int(s["span"][0]), int(s["span"][1])) for s in segs]
        self.pos_embeds.copy_(self._geo.vision_pos_embeds(
            grid, self.vision.pos_embed, num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.merge).to(self.pos_embeds.device))
        vcos, vsin = self._geo.vision_rope_cos_sin_cached(
            grid, self._vision_rope_cos_cache, self._vision_rope_sin_cache,
            spatial_merge_size=self.merge)
        self.vcos.copy_(vcos.to(self.vcos.device))
        self.vsin.copy_(vsin.to(self.vsin.device))
        if self.embed_hoist:
            # the prompt's token embeddings are constant per task: gathered
            # once here; each decision only writes its image rows into h0
            # and decoder layer 0 reads h0 (its o-projection epilogue writes h)
            fvk = self._kernels()
            fvk.embedding_lookup_bf16(self.input_ids.data_ptr(), self.embed_w.data_ptr(),
                                      self.h0.data_ptr(), S, self.hidden,
                                      torch.cuda.current_stream().cuda_stream)

    def set_embodied_positions(self, token_id: int) -> None:
        idx = (self.input_ids == int(token_id)).nonzero(as_tuple=True)[0]
        n = min(idx.numel(), self.num_cond)
        self.emb_idx[:n].copy_(idx[:n].to(self.emb_idx.device))
        # first 128-row tile boundary at or below the first conditioning row:
        # the last decoder layer only needs rows from there on (0 = all rows)
        first = int(idx[:n].min()) if n == self.num_cond else 0
        self._tail_row0 = (first // 128) * 128 if self.tail_rows_enabled else 0

    # ───────────────────────── kernel helpers ─────────────────────────

    def _kernels(self):
        if self._fvk is None:
            from flash_rt import flash_rt_fa2 as fa2
            from flash_rt import flash_rt_kernels as fvk
            from flash_rt import flash_rt_qwen3_vl_kernels as vlk
            self._fvk = fvk
            self._fa2 = fa2
            self._vlk = vlk
            self._num_sms = torch.cuda.get_device_properties(
                torch.device(self.device)).multi_processor_count
        return self._fvk

    def _fa2_short(self, **kw) -> None:
        """Non-causal BF16 FA2 for the action head's 40-query attention: the
        repository short-query tile entry (64-row query / 64-wide key tile at
        hd64, bit-identical to the vendored 128x128 default at these lengths,
        about half its time), else the default entry."""
        fa2 = self._fa2
        fn = getattr(fa2, "fwd_bf16_tile_vla", None)
        if fn is None:
            fn = getattr(fa2, "fwd_bf16_tile", None)
        if self.dit_attn_tile and fn is not None:
            kw.pop("softmax_lse_accum", None)
            kw.pop("o_accum", None)
            fn(seqused_k=0, **kw)
        else:
            fa2.fwd_bf16(**kw)

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def _gemm(self, x, w, out, M, N, K, stream) -> None:
        """y = x @ W^T in bf16.

        cuBLASLt is the default backend: at the DiT flow head's short-M shapes
        (M=40) it measures 2-3x the hand-written SM120 W16A16 tile.
        ``gemm='w16a16'`` selects the deterministic in-repo kernel instead.
        """
        if self.gemm_mode == 'cublaslt':
            self._fvk.bf16_matmul_cublaslt_bf16(
                x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, stream)
        else:
            self._fvk.w16a16_gemm_sm120_bf16(
                x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, 1.0, stream)

    def _gemm_bias(self, x, w, b, out, M, N, K, stream) -> None:
        if (b is not None and self.gemm_mode == 'cublaslt' and self.gemm_bias_epilogue):
            # bias in the cuBLASLt epilogue: one launch, one bf16 rounding
            self._fvk.bf16_matmul_cublaslt_bias_bf16(
                x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(), M, N, K, stream)
            return
        self._gemm(x, w, out, M, N, K, stream)
        if b is not None:
            self._fvk.add_bias_bf16(out.data_ptr(), b.data_ptr(), M, N, stream)

    def _gemm_bias_generic(self, x, w, b, out, M, N, K, stream) -> None:
        """bf16 GEMM for shapes the tensor-core tiles cannot take: the action /
        state projections have K = 7 / 8 (the SM120 W16A16 tile needs
        K % 64 == 0). cuBLASLt handles them at full speed."""
        if self.gemm_mode == 'cublaslt':
            if b is not None and self.gemm_bias_epilogue:
                self._fvk.bf16_matmul_cublaslt_bias_bf16(
                    x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(), M, N, K, stream)
                return
            self._fvk.bf16_matmul_cublaslt_bf16(
                x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, stream)
        else:
            self._fvk.bf16_matmul_bf16(
                x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, stream)
        if b is not None:
            self._fvk.add_bias_bf16(out.data_ptr(), b.data_ptr(), M, N, stream)

    # ───────────────────────── stages ─────────────────────────

    def vision_stage(self, stream: int) -> None:
        fvk = self._fvk
        S = self.seq_len
        H = self.hidden
        # token embedding gather for the whole prompt (hoisted to
        # stage_prompt when the decoder's first layer reads h0)
        if not self.embed_hoist:
            fvk.embedding_lookup_bf16(self.input_ids.data_ptr(), self.embed_w.data_ptr(),
                                      self.h.data_ptr(), S, H, stream)
        # ViT tower over both views in one batched forward: the tower is
        # per-patch except for attention, which runs batched so views never
        # attend to each other (~20% faster than two per-view forwards).
        emb, ds = self.vision.forward(
            self.pix, self.pos_embeds, self.vcos, self.vsin,
            nviews=self.num_views)
        per_view = self.vis_tokens_per_view
        # The DeepStack adds read the tower's merger outputs in place (their
        # addresses are fixed inside the captured graph), so only the image
        # embeddings are copied into the prompt's image spans.
        self._ds_keep = ds
        self._ds_ptr = [ds[k].data_ptr() + v * per_view * self.vis_out * 2
                        for v in range(self.num_views)
                        for k in range(self.deepstack_layers)]
        hin = self.h0 if self.embed_hoist else self.h
        for v in range(self.num_views):
            a, b = self.vision_spans[v]
            fvk.gpu_copy(hin.data_ptr() + a * H * 2,
                         emb.data_ptr() + v * per_view * self.vis_out * 2,
                         (b - a) * H * 2, stream)

    def llm_stage(self, stream: int) -> None:
        layer = {"bf16": self._llm_layer, "nvfp4": self._llm_layer_nvfp4}.get(
            self.llm_precision, self._llm_layer_fp8)
        for L in range(self.num_layers):
            layer(L, stream)
        self._gather_conditioning(stream)

    def _quant_act_fp8(self, x, xf8, xs8, M, K, stream) -> None:
        self._fvk.fp8_per_token_block128_quant_bf16(
            x.data_ptr(), xf8.data_ptr(), xs8.data_ptr(), M, K, stream)

    def _fp4_gemm_out(self, a, w, d, M, N, K, sfa, sfb, alpha, stream,
                      site: str) -> None:
        """Dispatch the decoder's non-residual NVFP4 GEMM by site/shape.

        The repository exposes the shared 128x128x256 base tile, a
        128x256x128 ``widen`` variant (faster at very large N) and a
        ``pingpong`` variant. This model's ``base`` entry is the additive
        128x128x128 twin (fp4_w4a16_gemm_sm120_bf16out_k128); the delivered
        decoder shapes select ``pingpong``.
        """
        variant = "pingpong"
        fn = {
            "base": self._fvk.fp4_w4a16_gemm_sm120_bf16out_k128,
            "widen": self._fvk.fp4_w4a16_gemm_sm120_bf16out_widen,
            "pingpong": self._fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong,
        }[variant]
        fn(a, w, d, M, N, K, sfa, sfb, alpha, stream)

    def _llm_layer_fp8(self, L: int, stream: int) -> None:
        """Qwen3-VL decoder layer with FP8 block-128 W8A8 GEMMs (the FlashRT
        SM120 prefill route: dynamic per-token/per-128-K activation scales,
        fp32 accumulate, bf16 output)."""
        fvk = self._fvk
        S, H, I = self.seq_len, self.hidden, self.inter
        nq, nkv, hd = self.n_q, self.n_kv, self.head_dim
        q_dim, kv_dim = nq * hd, nkv * hd
        qkv_n = q_dim + 2 * kv_dim
        lw = self.llm_layers[L]
        if self._use_fp4_site("qkv"):
            # RMSNorm fused into the NVFP4 activation quantizer (the repository's
            # Qwen3.5/OmniVoice prefill fold, bit-identical to the chain): the
            # normed activation never round-trips through bf16.
            fvk.rms_norm_to_nvfp4_swizzled_bf16_v3(
                self.h.data_ptr(), lw["input_norm_w"].data_ptr(),
                self.xq4.data_ptr(), self.xqsf4.data_ptr(), S, H, self.rms_eps,
                stream)
            self._fp4_gemm_out(
                self.xq4.data_ptr(), lw["qkv_w_fp4"].data_ptr(), self.qkv_out.data_ptr(),
                S, qkv_n, H, self.xqsf4.data_ptr(), lw["qkv_w_fp4sf"].data_ptr(),
                lw["qkv_w_fp4alpha"], stream, "qkv")
        else:
            fvk.rms_norm_to_fp8_block128_bf16(
                self.h.data_ptr(), lw["input_norm_w"].data_ptr(),
                self.xf8.data_ptr(), self.xs8.data_ptr(), S, H, self.rms_eps, stream)
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.xf8.data_ptr(), lw["qkv_w_fp8"].data_ptr(), self.qkv_out.data_ptr(),
                S, qkv_n, H, self.xs8.data_ptr(), lw["qkv_w_ws"].data_ptr(), stream)
        self._llm_attention(L, stream)
        self._quant_act_fp8(self.attn_out, self.attnf8, self.attns8, S, q_dim, stream)
        if "o_w_fp8" in lw:
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.attnf8.data_ptr(), lw["o_w_fp8"].data_ptr(),
                self.proj_out.data_ptr(), S, H, q_dim, self.attns8.data_ptr(),
                lw["o_w_ws"].data_ptr(), stream)
        else:
            self._gemm(self.attn_out, lw["o_w"], self.proj_out, S, H, q_dim, stream)
        fvk.residual_add(self.h.data_ptr(), self.proj_out.data_ptr(), S * H, stream)
        fvk.rms_norm(self.h.data_ptr(), lw["post_norm_w"].data_ptr(),
                     self.xn.data_ptr(), S, H, self.rms_eps, stream)
        self._quant_act_fp8(self.xn, self.xf8, self.xs8, S, H, stream)
        self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
            self.xf8.data_ptr(), lw["gate_up_w_fp8"].data_ptr(),
            self.gate_up.data_ptr(), S, 2 * I, H, self.xs8.data_ptr(),
            lw["gate_up_w_ws"].data_ptr(), stream)
        fvk.silu_mul_merged_bf16(self.gate_up.data_ptr(), self.act.data_ptr(),
                                 S, I, stream)
        if "down_w_fp8" in lw:
            self._quant_act_fp8(self.act, self.af8, self.as8, S, I, stream)
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.af8.data_ptr(), lw["down_w_fp8"].data_ptr(),
                self.down_out.data_ptr(), S, H, I, self.as8.data_ptr(),
                lw["down_w_ws"].data_ptr(), stream)
        else:
            self._gemm(self.act, lw["down_w"], self.down_out, S, H, I, stream)
        fvk.residual_add(self.h.data_ptr(), self.down_out.data_ptr(), S * H, stream)
        if L < self.deepstack_layers:
            for v in range(self.num_views):
                a, b = self.vision_spans[v]
                fvk.residual_add(self.h.data_ptr() + a * H * 2,
                                 self._ds_ptr[v * self.deepstack_layers + L],
                                 (b - a) * H, stream)

    @torch.no_grad()
    def calibrate_nvfp4(self, observations, warm: int = 2, verbose: bool = False) -> dict:
        """Real-data activation calibration for the NVFP4 MLP tier.

        Runs the FP8 pipeline over real observations and accumulates, per layer,
        the per-16-channel-group amax of the two quantized activation tensors
        (post-RMSNorm gate_up input, SiLU output). The maxima are then uploaded
        as static scales for ``quantize_bf16_to_nvfp4_swizzled_static_groups``.
        """
        fvk = self._kernels()
        S, H, I = self.seq_len, self.hidden, self.inter
        gup = torch.zeros_like(self.gup_group_amax)
        down = torch.zeros_like(self.down_group_amax)
        self.llm_precision_saved = self.llm_precision
        self.llm_precision = "fp8"          # capture activations on the FP8 path
        try:
            for i, obs in enumerate(observations):
                self.raw_frames.copy_(
                    torch.as_tensor(obs["frames"]).permute(0, 3, 1, 2)
                    .to(self.raw_frames.device))
                self.raw_state.copy_(torch.as_tensor(obs["state"], dtype=torch.float32).to(self.raw_state.device))
                self.preprocess_stage(0)
                self.state_stage(0)
                self.vision_stage(0)
                for L in range(self.num_layers):
                    self._llm_layer_fp8(L, 0)
                    gup[L] = torch.maximum(
                        gup[L], self.xn.abs().view(S, H // 16, 16).amax(dim=(0, 2)))
                    down[L] = torch.maximum(
                        down[L], self.act.abs().view(S, I // 16, 16).amax(dim=(0, 2)))
                if verbose:
                    print(f"  calibrated frame {i}: gate_up amax {float(gup.max()):.2f} "
                          f"down amax {float(down.max()):.2f}")
        finally:
            self.llm_precision = self.llm_precision_saved
        self.gup_group_amax.copy_(gup.clamp_min(1e-6))
        self.down_group_amax.copy_(down.clamp_min(1e-6))
        self.nvfp4_calibrated = True
        return {"frames": len(observations),
                "gate_up_amax_max": float(gup.max()), "down_amax_max": float(down.max())}

    def _attn_fp4o(self) -> bool:
        """True when the attention kernel itself emits the o-projection's NVFP4
        operand (the repo FMHA fp4out epilogue, byte-identical to the standalone
        quantizer), so the o-site must skip its own activation quantizer."""
        return self._attn.prefill_route == "fp8_fp4out" and self._use_fp4_site("o")

    def _act_mode_for(self, site: str) -> str:
        """Resolve the activation-quantizer mode for one site."""
        m = self.nvfp4_act_mode
        if m.startswith("mse:"):
            return "mse" if site in m.split(":", 1)[1].split(",") else "dynamic"
        return m

    def _quant_act_fp4(self, x, xq, xsf, M, K, L: int, site: str, stream: int) -> None:
        fvk = self._fvk
        mode = self._act_mode_for(site)
        if mode == "static" and self.nvfp4_calibrated:
            amax = {"gup": self.gup_group_amax, "down": self.down_group_amax}.get(
                site, self.gup_group_amax)[L]
            fvk.quantize_bf16_to_nvfp4_swizzled_static_groups(
                x.data_ptr(), amax.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, K, stream)
        elif mode == "clipped" and self.nvfp4_calibrated:
            # dynamic per-block scale, but the virtual amax is capped by the
            # real-data calibrated group amax: outliers saturate in FP4 instead
            # of destroying the other 15 values' resolution (AWQ-style).
            amax = (self.gup_group_amax[L] if site == "gup" else self.down_group_amax[L])
            if self.nvfp4_clip_scale != 1.0:
                amax = amax * self.nvfp4_clip_scale
                self._clip_buf = amax.contiguous()
                amax = self._clip_buf
            fvk.quantize_bf16_to_nvfp4_swizzled_clipped(
                x.data_ptr(), amax.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, K, stream)
        elif mode == "mse":
            # MSE scale search for the activation blocks (the same search the
            # weight quantizer uses): 8 percent lower reconstruction error than
            # the ceil-of-amax scale on real post-RMSNorm activations.
            fvk.quantize_bf16_to_nvfp4_swizzled_mse(
                x.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, K, stream)
        elif mode == "secondmax":
            fvk.quantize_bf16_to_nvfp4_swizzled_secondmax(
                x.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, K,
                self.nvfp4_secondmax_mult, stream)
        else:
            # v2 is bit-identical to the v1 kernel and reads the input once.
            fvk.quantize_bf16_to_nvfp4_swizzled_v2(
                x.data_ptr(), xq.data_ptr(), xsf.data_ptr(), M, K, stream)

    def _use_fp4_mlp(self, L: int) -> bool:
        """NVFP4 for the decoder MLP, except the edge layers whose activations
        carry the largest tails (FlashRT's precision-tier pattern)."""
        return self.nvfp4_skip_edges <= L < self.num_layers - self.nvfp4_skip_edges

    def _use_fp4_site(self, site: str) -> bool:
        """Which GEMM sites run on NVFP4 (''/site name / 'attn' group)."""
        if self.llm_precision != "nvfp4":
            # The NVFP4 operands/scales only exist on the NVFP4 route; the
            # FP8 and BF16 tiers (and the FP8 pass of calibrate_nvfp4) never
            # take an NVFP4 site.
            return False
        if self.nvfp4_sites == "all":
            return True
        if self.nvfp4_sites == "attn" and site in ("qkv", "o"):
            return True
        if self.nvfp4_sites == "mlp" and site in ("gate_up", "down"):
            return True
        return site in self.nvfp4_sites.split(",")

    def _tail_ok(self, L: int) -> bool:
        """The trailing-tile path needs the all-NVFP4 post-attention chain
        (fused o operand, NVFP4 MLP and down) and no deepstack add."""
        return (self._attn_fp4o() and self._use_fp4_site("o")
                and self._use_fp4_mlp(L) and self._use_fp4_site("gate_up")
                and self._use_fp4_site("down") and self._use_fp4_down(L)
                and L >= self.deepstack_layers)

    def _use_fp4_down(self, L: int) -> bool:
        """Per-layer gate for the ``down`` site (see select_nvfp4_down_layers).
        A layer that needs the operand pre-scale (NVFP4_DOWN_ACT_SHIFT) runs
        NVFP4 only through the SwiGLU fold, which applies it."""
        if L in self.down_act_shift and not (self.swiglu_fold and L not in self.swiglu_fold_skip):
            return False
        return self.nvfp4_down_layers is None or L in self.nvfp4_down_layers

    def select_nvfp4_down_layers(self, keep_frac: float) -> list:
        """Real-data layer selection for the NVFP4 ``down`` site.

        The down projection's input is the SiLU output, whose per-layer amax
        spans four orders of magnitude (calibrated: up to 4.4e4). Layers with
        the largest tails lose the most from W4A4 there, so run NVFP4 on the
        ``keep_frac`` fraction of layers with the *smallest* calibrated amax
        and leave the rest on FP8. Requires calibrate_nvfp4() first.
        """
        if not self.nvfp4_calibrated:
            raise RuntimeError("select_nvfp4_down_layers needs calibrate_nvfp4() first")
        amax = self.down_group_amax.amax(dim=1)          # per layer
        n_keep = max(0, min(self.num_layers, int(round(self.num_layers * keep_frac))))
        order = torch.argsort(amax)                       # ascending
        keep = sorted(int(i) for i in order[:n_keep].tolist())
        self.nvfp4_down_layers = set(keep)
        self.down_layer_amax = [float(v) for v in amax.tolist()]
        return keep

    def _use_fp4_site_legacy(self, site: str) -> bool:
        """Which MLP GEMM runs on NVFP4.

        ``mlp``   - gate_up and down (default).
        ``gate_up`` - only gate_up. Its input is the post-RMSNorm activation
                      (calibrated amax ~2.6e2); the ``down`` input is the SiLU
                      output, whose calibrated amax reaches 4.4e4 and whose
                      per-16-block FP4 scales lose too much resolution.
        """
        return self.nvfp4_sites == "mlp" or site == self.nvfp4_sites

    def _llm_layer_nvfp4(self, L: int, stream: int) -> None:
        """Decoder layer with the MLP linears on NVFP4 W4A16 (real-weights
        calibrated global scales) and the attention linears on FP8 block-128."""
        fvk = self._fvk
        S, H, I = self.seq_len, self.hidden, self.inter
        nq, nkv, hd = self.n_q, self.n_kv, self.head_dim
        q_dim, kv_dim = nq * hd, nkv * hd
        qkv_n = q_dim + 2 * kv_dim
        lw = self.llm_layers[L]
        # layer 0 reads the staged embeddings (h0); its o-projection
        # epilogue (D = A B + C, C = h0) writes the residual stream h
        hin = self.h0 if (L == 0 and self.embed_hoist) else self.h
        if self._use_fp4_site("qkv"):
            # RMSNorm fused into the NVFP4 activation quantizer (the repository's
            # OmniVoice/Qwen3.5 prefill fold, validated bit-identical to the
            # [rms_norm -> bf16 -> quantize] chain): one launch fewer and no bf16
            # round-trip for the attention input.
            fvk.rms_norm_to_nvfp4_swizzled_bf16_v3(
                hin.data_ptr(), lw["input_norm_w"].data_ptr(),
                self.xq4.data_ptr(), self.xqsf4.data_ptr(), S, H, self.rms_eps,
                stream)
            self._fp4_gemm_out(
                self.xq4.data_ptr(), lw["qkv_w_fp4"].data_ptr(), self.qkv_out.data_ptr(),
                S, qkv_n, H, self.xqsf4.data_ptr(), lw["qkv_w_fp4sf"].data_ptr(),
                lw["qkv_w_fp4alpha"], stream, "qkv")
        else:
            fvk.rms_norm(self.h.data_ptr(), lw["input_norm_w"].data_ptr(),
                         self.xn.data_ptr(), S, H, self.rms_eps, stream)
            self._quant_act_fp8(self.xn, self.xf8, self.xs8, S, H, stream)
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.xf8.data_ptr(), lw["qkv_w_fp8"].data_ptr(), self.qkv_out.data_ptr(),
                S, qkv_n, H, self.xs8.data_ptr(), lw["qkv_w_ws"].data_ptr(), stream)
        self._llm_attention(L, stream)
        # Only the embodied-action rows of the last layer feed the action head,
        # and everything after attention is row-local: the last layer runs its
        # o-projection and MLP on the trailing 128-row tile(s) that hold them
        # (exact for those rows; the NVFP4 operands' swizzled scales are laid
        # out per 128-row block, so the tile offset is a plain pointer offset).
        r0 = self._tail_row0 if L == self.num_layers - 1 and self._tail_ok(L) else 0
        if r0:
            Mt = S - r0
            hp = self.h.data_ptr() + r0 * H * 2
            self._fvk.fp4_w4a16_gemm_residual_sm120_bf16out_k128(
                self.xo4.data_ptr() + r0 * q_dim // 2, lw["o_w_fp4"].data_ptr(),
                hp, hp, Mt, H, q_dim,
                self.xosf4.data_ptr() + (r0 // 128) * (q_dim // 64) * 512,
                lw["o_w_fp4sf"].data_ptr(), lw["o_w_fp4alpha"], stream)
            fvk.rms_norm_to_nvfp4_swizzled_bf16_v3(
                hp, lw["post_norm_w"].data_ptr(),
                self.xf4.data_ptr(), self.xsf4.data_ptr(), Mt, H, self.rms_eps,
                stream)
            self._fp4_gemm_out(
                self.xf4.data_ptr(), lw["gate_up_w_fp4"].data_ptr(), self.gate_up.data_ptr(),
                Mt, 2 * I, H, self.xsf4.data_ptr(), lw["gate_up_w_fp4sf"].data_ptr(),
                lw["gate_up_w_fp4alpha"], stream, "gate_up")
            fvk.silu_mul_merged_to_nvfp4_swizzled_vec_bf16(
                self.gate_up.data_ptr(), self.af4.data_ptr(), self.asf4.data_ptr(),
                Mt, I, stream)
            self._fvk.fp4_w4a16_gemm_residual_sm120_bf16out_k128(
                self.af4.data_ptr(), lw["down_w_fp4"].data_ptr(), hp, hp, Mt, H, I,
                self.asf4.data_ptr(), lw["down_w_fp4sf"].data_ptr(),
                lw["down_w_fp4alpha"], stream)
            return
        if self._use_fp4_site("o"):
            if not self._attn_fp4o():
                self._quant_act_fp4(self.attn_out, self.xo4, self.xosf4, S, q_dim, L, "o", stream)
            # residual folded into the GEMM epilogue (D = alpha*A@B + C), the
            # repository Qwen3 prefill default: one launch fewer and one fewer
            # bf16 rounding than gemm + residual_add
            self._fvk.fp4_w4a16_gemm_residual_sm120_bf16out_k128(
                self.xo4.data_ptr(), lw["o_w_fp4"].data_ptr(),
                hin.data_ptr(), self.h.data_ptr(), S, H, q_dim,
                self.xosf4.data_ptr(), lw["o_w_fp4sf"].data_ptr(),
                lw["o_w_fp4alpha"], stream)
        else:
            self._quant_act_fp8(self.attn_out, self.attnf8, self.attns8, S, q_dim, stream)
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.attnf8.data_ptr(), lw["o_w_fp8"].data_ptr(), self.proj_out.data_ptr(),
                S, H, q_dim, self.attns8.data_ptr(), lw["o_w_ws"].data_ptr(), stream)
            fvk.residual_add(self.h.data_ptr(), self.proj_out.data_ptr(), S * H, stream)
        # MLP on NVFP4 (edge layers stay FP8: their inputs carry the largest
        # activation tails); the post-attention RMSNorm is fused into whichever
        # quantizer feeds the gate_up GEMM.
        fold = (self.swiglu_fold and L not in self.swiglu_fold_skip and self._use_fp4_mlp(L) and self._use_fp4_site("gate_up")
                and self._use_fp4_site("down") and self._use_fp4_down(L))
        if fold:
            fvk.rms_norm_to_nvfp4_swizzled_bf16_v3(
                self.h.data_ptr(), lw["post_norm_w"].data_ptr(),
                self.xf4.data_ptr(), self.xsf4.data_ptr(), S, H, self.rms_eps,
                stream)
            ga, ua = lw["gate_up_il_alpha"]
            # power-of-two pre-scale of the SiLU*up operand (exact; undone in
            # the down GEMM's alpha) for layers whose activation tail exceeds
            # the NVFP4 block-scale range at a unit global scale
            sh = self.down_act_shift.get(L, 0)
            fvk.fp4_w4a16_dual_gemm_silu_fp4out_sm120(
                self.xf4.data_ptr(), lw["gate_up_il"].data_ptr(), lw["gate_up_il"].data_ptr(),
                self.xsf4.data_ptr(), lw["gate_up_il_sf"].data_ptr(), lw["gate_up_il_sf"].data_ptr(),
                self.swiglu_fold_out.data_ptr(), self.asf4.data_ptr(), S, 2 * I, H, ga,
                ua * 2.0 ** (-sh), stream)
            fvk.fp4_swiglu_even_col_compact_v2(self.swiglu_fold_out.data_ptr(), self.af4.data_ptr(),
                                               S, I, stream)
            self._fvk.fp4_w4a16_gemm_residual_sm120_bf16out_k128(
                self.af4.data_ptr(), lw["down_w_fp4"].data_ptr(),
                self.h.data_ptr(), self.h.data_ptr(), S, H, I,
                self.asf4.data_ptr(), lw["down_w_fp4sf"].data_ptr(),
                lw["down_w_fp4alpha"] * 2.0 ** sh, stream)
        elif self._use_fp4_mlp(L) and self._use_fp4_site("gate_up"):
            fvk.rms_norm_to_nvfp4_swizzled_bf16_v3(
                self.h.data_ptr(), lw["post_norm_w"].data_ptr(),
                self.xf4.data_ptr(), self.xsf4.data_ptr(), S, H, self.rms_eps,
                stream)
            self._fp4_gemm_out(
                self.xf4.data_ptr(), lw["gate_up_w_fp4"].data_ptr(), self.gate_up.data_ptr(),
                S, 2 * I, H, self.xsf4.data_ptr(), lw["gate_up_w_fp4sf"].data_ptr(),
                lw["gate_up_w_fp4alpha"], stream, "gate_up")
        else:
            fvk.rms_norm_to_fp8_block128_bf16(
                self.h.data_ptr(), lw["post_norm_w"].data_ptr(),
                self.xf8.data_ptr(), self.xs8.data_ptr(), S, H, self.rms_eps, stream)
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.xf8.data_ptr(), lw["gate_up_w_fp8"].data_ptr(),
                self.gate_up.data_ptr(), S, 2 * I, H, self.xs8.data_ptr(),
                lw["gate_up_w_ws"].data_ptr(), stream)
        if fold:
            pass
        elif self._use_fp4_mlp(L) and self._use_fp4_site("down") and self._use_fp4_down(L):
            # SiLU(gate)*up straight into the down GEMM's NVFP4 operand (the same
            # fold the repository Qwen3 prefill path uses), then the residual in
            # that GEMM's epilogue.
            # The base entry point shares the grouped variants' math and now
            # dispatches to the warp-per-block v2 kernel (606 vs 317 GB/s at
            # this shape); the grouped32 form is bit-identical.
            fvk.silu_mul_merged_to_nvfp4_swizzled_vec_bf16(
                self.gate_up.data_ptr(), self.af4.data_ptr(), self.asf4.data_ptr(),
                S, I, stream)
            self._fvk.fp4_w4a16_gemm_residual_sm120_bf16out_k128(
                self.af4.data_ptr(), lw["down_w_fp4"].data_ptr(),
                self.h.data_ptr(), self.h.data_ptr(), S, H, I,
                self.asf4.data_ptr(), lw["down_w_fp4sf"].data_ptr(),
                lw["down_w_fp4alpha"], stream)
        else:
            # SiLU(gate)*up and the block-128 FP8 quantization in one launch (the
            # repository's Qwen3 prefill fold): the bf16 activation never round-trips.
            fvk.silu_mul_merged_to_fp8_block128_bf16(
                self.gate_up.data_ptr(), self.af8.data_ptr(), self.as8.data_ptr(),
                S, I, stream)
            self._fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.af8.data_ptr(), lw["down_w_fp8"].data_ptr(),
                self.down_out.data_ptr(), S, H, I, self.as8.data_ptr(),
                lw["down_w_ws"].data_ptr(), stream)
            fvk.residual_add(self.h.data_ptr(), self.down_out.data_ptr(), S * H, stream)
        if L < self.deepstack_layers:
            for v in range(self.num_views):
                a, b = self.vision_spans[v]
                fvk.residual_add(self.h.data_ptr() + a * H * 2,
                                 self._ds_ptr[v * self.deepstack_layers + L],
                                 (b - a) * H, stream)

    def _llm_layer(self, L: int, stream: int) -> None:
        fvk = self._fvk
        S, H, I = self.seq_len, self.hidden, self.inter
        nq, nkv, hd = self.n_q, self.n_kv, self.head_dim
        q_dim, kv_dim = nq * hd, nkv * hd
        qkv_n = q_dim + 2 * kv_dim
        if True:
            lw = self.llm_layers[L]
            fvk.rms_norm(self.h.data_ptr(), lw["input_norm_w"].data_ptr(),
                         self.xn.data_ptr(), S, H, self.rms_eps, stream)
            self._gemm(self.xn, lw["qkv_w"], self.qkv_out, S, qkv_n, H, stream)
            self._llm_attention(L, stream)
            self._gemm(self.attn_out, lw["o_w"], self.proj_out, S, H, q_dim, stream)
            fvk.residual_add(self.h.data_ptr(), self.proj_out.data_ptr(), S * H, stream)
            fvk.rms_norm(self.h.data_ptr(), lw["post_norm_w"].data_ptr(),
                         self.xn.data_ptr(), S, H, self.rms_eps, stream)
            self._gemm(self.xn, lw["gate_up_w"], self.gate_up, S, 2 * I, H, stream)
            fvk.silu_mul_merged_bf16(self.gate_up.data_ptr(), self.act.data_ptr(),
                                     S, I, stream)
            self._gemm(self.act, lw["down_w"], self.down_out, S, H, I, stream)
            fvk.residual_add(self.h.data_ptr(), self.down_out.data_ptr(), S * H, stream)
            if L < self.deepstack_layers:
                for v in range(self.num_views):
                    a, b = self.vision_spans[v]
                    fvk.residual_add(self.h.data_ptr() + a * H * 2,
                                     self._ds_ptr[v * self.deepstack_layers + L],
                                     (b - a) * H, stream)

    def _gather_conditioning(self, stream: int) -> None:
        fvk = self._fvk
        H = self.hidden
        # embodied-action tokens -> DiT conditioning
        fvk.embedding_lookup_bf16(self.emb_idx.data_ptr(), self.h.data_ptr(),
                                  self.cond.data_ptr(), self.num_cond, H, stream)
        # cross-attention K/V for the even DiT blocks (constant per decision):
        # one wide-N projection writes every cross block's K and V.
        n_kv = 2 * len(self.dit_cross_index) * self.dim_dit
        if self.dit_cond_fp4:
            fvk.quantize_bf16_to_nvfp4_swizzled_v2(
                self.cond_pad.data_ptr(), self.cond4.data_ptr(),
                self.cond4sf.data_ptr(), 128, self.cond_dim, stream)
            qp, swz, alpha = self.d_cond_kv_fp4
            # bias in the GEMM epilogue (one rounding, no 128-row bias pass)
            fvk.fp4_w4a16_gemm_sm120_bf16out_bias_k128(
                self.cond4.data_ptr(), qp.data_ptr(), self.dit_cond_kv.data_ptr(),
                self.d_cond_kv["bias"].data_ptr(), 128, n_kv, self.cond_dim,
                self.cond4sf.data_ptr(), swz.data_ptr(), alpha, stream)
        elif self.dit_cond_fp8:
            fvk.fp8_per_token_block128_quant_bf16(
                self.cond.data_ptr(), self.cond_f8.data_ptr(),
                self.cond_fs.data_ptr(), self.num_cond, self.cond_dim, stream)
            fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
                self.cond_f8.data_ptr(), self.d_cond_kv["w8"].data_ptr(),
                self.dit_cond_kv.data_ptr(), self.num_cond, n_kv, self.cond_dim,
                self.cond_fs.data_ptr(), self.d_cond_kv["ws"].data_ptr(), stream)
            fvk.add_bias_bf16(self.dit_cond_kv.data_ptr(),
                              self.d_cond_kv["bias"].data_ptr(),
                              self.num_cond, n_kv, stream)
        else:
            self._gemm_bias(self.cond, self.d_cond_kv["weight"],
                            self.d_cond_kv["bias"], self.dit_cond_kv,
                            self.num_cond, n_kv, self.cond_dim, stream)

    def dit_state_token(self, stream: int) -> None:
        """state_encoder(state): [1,8] -> [1,1024] -> relu -> [1,768]."""
        fvk = self._fvk
        D = self.dim_dit
        se = self.d_state_enc
        self._gemm_bias_generic(self.state_norm, se["layer1"]["weight"],
                                se["layer1"]["bias"], self.dit_ff[:1, :self.dit_dec_h],
                                1, self.dit_dec_h, self.state_dim, stream)
        fvk.relu_inplace_bf16(self.dit_ff.data_ptr(), self.dit_dec_h, stream)
        self._gemm_bias(self.dit_ff[:1, :self.dit_dec_h], se["layer2"]["weight"],
                        se["layer2"]["bias"], self.state_tok, 1, D, self.dit_dec_h, stream)

    def dit_stage(self, stream: int, first_inputs_ready: bool = False) -> None:
        """4 Euler steps over the 16-block DiT-B flow head."""
        for step in range(self.n_steps):
            self._dit_step(step, stream, first_inputs_ready and step == 0)

    def _dit_step(self, step: int, stream: int, inputs_ready: bool = False) -> None:
        if not inputs_ready:
            self._assemble_step_inputs(step, stream)
        if self.dit_precision == "nvfp4":
            for i in range(16):
                self._dit_block_fp4(i, step, stream)
            self._dit_output(step, stream, norm_done=True)
        else:
            self._dit_blocks(step, stream)
            self._dit_output(step, stream)
        self._euler(step, stream)

    def _assemble_step_inputs(self, step: int, stream: int) -> None:
        """x = [state_token | future_tokens | action_features]."""
        fvk = self._fvk
        D, CH = self.dim_dit, self.chunk_size
        ae = self.d_act_enc
        a1 = self.dit_act
        if self.dit_enc_fold:
            self._gemm_bias_generic(self.actions, self.dit_enc_w12, self.dit_enc_c[step],
                                    a1, CH, D, self.action_dim, stream)
        elif self.dit_tail_fold:
            cat = self.dit_cat_steps[step]
            fvk.bf16_matmul_cublaslt_bias_ex_bf16(
                self.actions.data_ptr(), ae["layer1"]["weight"].data_ptr(),
                ae["layer1"]["bias"].data_ptr(), cat.data_ptr(), CH, D, self.action_dim,
                2 * D, False, stream)
        else:
            cat = self.dit_cat
            self._gemm_bias_generic(self.actions, ae["layer1"]["weight"],
                                    ae["layer1"]["bias"], a1, CH, D,
                                    self.action_dim, stream)
            fvk.concat2_bf16(a1.data_ptr(), self.st_time[step].data_ptr(),
                             self.dit_cat.data_ptr(), CH, D, D, stream)
        if not self.dit_enc_fold:
            self._gemm_bias(cat, ae["layer2"]["weight"], ae["layer2"]["bias"],
                            a1, CH, D, 2 * D, stream)
        fvk.silu_bf16(a1.data_ptr(), CH * D, stream)
        self._gemm_bias(a1, ae["layer3"]["weight"], ae["layer3"]["bias"],
                        self.dit_x[self.num_cond + 1:], CH, D, D, stream)
        # head rows (state + future tokens) and the action rows' positional
        # embedding in one kernel (a graph memcpy node would also break the
        # programmatic-dependent-launch chain)
        r = fvk.vla_dit_step_rows(self.dit_x.data_ptr(), self.dit_head.data_ptr(),
                                  self.st_pos.data_ptr(), 1 + self.num_cond, CH, D, stream)
        if r != 0:
            raise RuntimeError(f"vla_dit_step_rows failed: {r}")

    def _dit_blocks(self, step: int, stream: int) -> None:
        fvk, fa2 = self._fvk, self._fa2
        D = self.dim_dit
        N = self.dit_seq
        hd, nh = self.dit_head_dim, self.dit_heads
        scale = 1.0 / math.sqrt(hd)
        for i in range(16):
            self._dit_block(i, step, stream)

    def _dit_block(self, i: int, step: int, stream: int) -> None:
        fvk, fa2 = self._fvk, self._fa2
        D = self.dim_dit
        N = self.dit_seq
        hd, nh = self.dit_head_dim, self.dit_heads
        scale = 1.0 / math.sqrt(hd)
        if True:
            blk = self.dit_blocks[i]
            fvk.ada_layer_norm_bf16(self.dit_x.data_ptr(),
                                    self.st_norm1_s[step, i].data_ptr(),
                                    self.st_norm1_b[step, i].data_ptr(),
                                    self.dit_xn.data_ptr(), N, D, 1e-5, stream)
            if blk["cross"]:
                self._gemm_bias(self.dit_xn, blk["q"]["weight"], blk["q"]["bias"],
                                self.dit_q, N, D, D, stream)
                q_ptr, q_row = self.dit_q.data_ptr(), D
                k_src, v_src, kv_seq = self.dit_cond_k[i], self.dit_cond_v[i], self.num_cond
                k_row = v_row = self.dit_cond_ld
            else:
                self._gemm_bias(self.dit_xn, blk["qkv"]["weight"], blk["qkv"]["bias"],
                                self.dit_qkv, N, 3 * D, D, stream)
                q_ptr, q_row = self.dit_qkv.data_ptr(), 3 * D
                k_src, v_src, kv_seq = self.dit_qkv, self.dit_qkv, N
                k_row = v_row = 3 * D
            self._fa2_short(
                Q=q_ptr,
                K=k_src.data_ptr() + (D * 2 if not blk["cross"] else 0),
                V=v_src.data_ptr() + (D * 4 if not blk["cross"] else 0),
                O=self.dit_o.data_ptr(), softmax_lse=self.dit_lse.data_ptr(),
                softmax_lse_accum=0, o_accum=0, batch=1, seqlen_q=N, seqlen_k=kv_seq,
                num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                q_strides=(N * q_row, q_row, hd), k_strides=(kv_seq * k_row, k_row, hd),
                v_strides=(kv_seq * v_row, v_row, hd), o_strides=(N * D, D, hd),
                softmax_scale=scale, num_sms=0, stream=stream)
            self._gemm_bias(self.dit_o, blk["o"]["weight"], blk["o"]["bias"],
                            self.dit_tmp, N, D, D, stream)
            fvk.residual_add(self.dit_x.data_ptr(), self.dit_tmp.data_ptr(), N * D, stream)
            fvk.layer_norm_no_affine_bf16(self.dit_x.data_ptr(), self.dit_xn.data_ptr(),
                                          N, D, 1e-5, stream)
            self._gemm_bias(self.dit_xn, blk["ff1"]["weight"], blk["ff1"]["bias"],
                            self.dit_ff, N, self.dit_ff_dim, D, stream)
            fvk.gelu_inplace(self.dit_ff.data_ptr(), N * self.dit_ff_dim, stream)
            self._gemm_bias(self.dit_ff, blk["ff2"]["weight"], blk["ff2"]["bias"],
                            self.dit_tmp, N, D, self.dit_ff_dim, stream)
            fvk.residual_add(self.dit_x.data_ptr(), self.dit_tmp.data_ptr(), N * D, stream)

    def _dit_block_fp4(self, i: int, step: int, stream: int) -> None:
        """One DiT block on the skinny NVFP4 family, the same composition the
        repository's GROOT N1.7 / Pi0.5 action heads use on sm_120a: every
        projection is a block-scaled W4A4 K-split GEMM whose fused consumer
        applies alpha, bias and the activation, and emits the next GEMM's NVFP4
        operand (with its block scales) in the same launch.

        Block 0's AdaLN is an explicit launch; afterwards each block's AdaLN
        (before attention) and no-affine LayerNorm (before the FFN) are the
        epilogues of the previous block's consumers.
        """
        fvk = self._fvk
        D, N, FF = self.dim_dit, self.dit_seq, self.dit_ff_dim
        hd, nh = self.dit_head_dim, self.dit_heads
        qk_scale = 1.0 / math.sqrt(hd)
        blk = self.dit_blocks[i]
        W = blk["fp4"]
        one = self.dit_one.data_ptr()
        part = self.dit_partials.data_ptr()
        xn4, xnsf4 = self.dit_xn4.data_ptr(), self.dit_xnsf4.data_ptr()
        hid4, hidsf4 = self.dit_hid4.data_ptr(), self.dit_hidsf4.data_ptr()

        def _splits(n: int, k: int) -> int:
            c = self.dit_fp4_cfg[(n, k)]
            return k // fvk.nvfp4_skinny_config_k_chunk(c)

        # The Q/K/V projection and the output projection have independent
        # precision policies: W4A4 on both is the delivered ten-block set, and
        # the remaining blocks can run either site on BF16 (their whole-block
        # W4A4 form misses the gate). The operand format each consumer emits
        # follows the *next* block's q policy, not this block's.
        q_fp4 = (self.dit_fp4_sites == "all"
                 and i not in self.dit_attn_q_bf16_blocks)
        o_fp4 = (self.dit_fp4_sites == "all"
                 and i not in self.dit_attn_o_bf16_blocks)
        # W4A4 attention + W4A4 o projection: the attention kernel emits the
        # o GEMM's NVFP4 operand (same bytes the GEMM's on-load quantizer
        # would produce from the BF16 output).
        ao_fp4 = self.dit_ao_fp4 and q_fp4 and o_fp4
        ao4 = ((self.dit_ao4.data_ptr(), self.dit_aosf4.data_ptr()) if ao_fp4 else (0, 0))
        if q_fp4:
            if i == 0:
                # AdaLN of the step input -> NVFP4 operand for the first GEMM.
                fvk.nvfp4_skinny_layer_norm(self.dit_x.data_ptr(),
                                            self.st_norm1_s[step, 0].data_ptr(),
                                            self.st_norm1_b[step, 0].data_ptr(),
                                            xn4, xnsf4, one, 0, N, D, 1e-5, True, stream)
            # Attention straight off the projection partials: Q (+ K/V for the
            # self blocks) in the same launch as the K-split sum; the cross
            # blocks read this decision's conditioning K/V.
            if blk["cross"]:
                w, sf, gs, b = W["q"]
                if self.dit_q_cross_fused:
                    # Q projection (same K splits) + attention in one launch
                    r = fvk.nvfp4_skinny_q_cross_attention(
                        xn4, xnsf4, w, sf, one, gs, b,
                        self.dit_cond_k[i].data_ptr(), self.dit_cond_v[i].data_ptr(),
                        self.dit_cond_ld, self.num_cond, self.dit_o.data_ptr(), N, nh, hd, D,
                        _splits(D, D), qk_scale, True, stream, *ao4)
                else:
                    fvk.nvfp4_skinny_gemm(xn4, xnsf4, w, sf, part, N, D, D,
                                          self.dit_fp4_cfg[(D, D)], True, stream)
                    r = fvk.nvfp4_skinny_partials_attention(
                        part, _splits(D, D), one, gs, b, D,
                        self.dit_cond_k[i].data_ptr(), self.dit_cond_v[i].data_ptr(),
                        self.dit_cond_ld, self.num_cond, True, self.dit_o.data_ptr(), N,
                        nh, hd, qk_scale, True, stream, *ao4)
            else:
                w, sf, gs, b = W["qkv"]
                if self.dit_qkv_fused:
                    # projection + attention in one launch per (head, q tile)
                    r = fvk.nvfp4_skinny_qkv_self_attention(
                        xn4, xnsf4, w, sf, one, gs, b, self.dit_o.data_ptr(), N, nh, hd, D,
                        qk_scale, True, stream, *ao4)
                else:
                    fvk.nvfp4_skinny_gemm(xn4, xnsf4, w, sf, part, N, 3 * D, D,
                                          self.dit_fp4_cfg[(3 * D, D)], True, stream)
                    # self attention: K/V come from the same partials, kv_seq == rows
                    r = fvk.nvfp4_skinny_partials_attention(
                        part, _splits(3 * D, D), one, gs, b, 3 * D, 0, 0, 0, N, False,
                        self.dit_o.data_ptr(), N, nh, hd, qk_scale, True, stream, *ao4)
            if r != 0:
                raise RuntimeError(f"DiT block {i} partials attention failed: {r}")
        else:
            # BF16 attention projections; the no-affine LayerNorm that feeds
            # the FFN still emits the NVFP4 operand.
            if i == 0:
                fvk.ada_layer_norm_bf16(self.dit_x.data_ptr(),
                                        self.st_norm1_s[step, 0].data_ptr(),
                                        self.st_norm1_b[step, 0].data_ptr(),
                                        self.dit_xn.data_ptr(), N, D, 1e-5, stream)
            if blk["cross"]:
                self._gemm_bias(self.dit_xn, blk["q"]["weight"], blk["q"]["bias"],
                                self.dit_q, N, D, D, stream)
                q_ptr, q_row = self.dit_q.data_ptr(), D
                k_src, v_src, kv_seq = self.dit_cond_k[i], self.dit_cond_v[i], self.num_cond
                k_row = v_row = self.dit_cond_ld
            else:
                self._gemm_bias(self.dit_xn, blk["qkv"]["weight"], blk["qkv"]["bias"],
                                self.dit_qkv, N, 3 * D, D, stream)
                q_ptr, q_row = self.dit_qkv.data_ptr(), 3 * D
                k_src, v_src, kv_seq = self.dit_qkv, self.dit_qkv, N
                k_row = v_row = 3 * D
            self._fa2_short(
                Q=q_ptr,
                K=k_src.data_ptr() + (D * 2 if not blk["cross"] else 0),
                V=v_src.data_ptr() + (D * 4 if not blk["cross"] else 0),
                O=self.dit_o.data_ptr(), softmax_lse=self.dit_lse.data_ptr(),
                softmax_lse_accum=0, o_accum=0, batch=1, seqlen_q=N, seqlen_k=kv_seq,
                num_heads_q=nh, num_heads_kv=nh, head_dim=hd,
                q_strides=(N * q_row, q_row, hd), k_strides=(kv_seq * k_row, k_row, hd),
                v_strides=(kv_seq * v_row, v_row, hd), o_strides=(N * D, D, hd),
                softmax_scale=qk_scale, num_sms=0, stream=stream)
        if o_fp4:
            # Output projection (BF16 attention output quantized on load) ->
            # bias + residual into the stream -> no-affine LayerNorm -> operand.
            w, sf, gs, b = W["o"]
            if ao_fp4:
                fvk.nvfp4_skinny_gemm(self.dit_ao4.data_ptr(), self.dit_aosf4.data_ptr(), w, sf,
                                      part, N, D, D, self.dit_fp4_cfg[(D, D)], True, stream)
            else:
                fvk.nvfp4_skinny_gemm_bf16_act(self.dit_o.data_ptr(), one, w, sf, part,
                                               N, D, D, self.dit_fp4_cfg[(D, D)], True, stream)
            fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, _splits(D, D), one, gs, b, self.dit_x.data_ptr(), 0, 0,
                xn4, xnsf4, one, 0, N, D, 1e-5, True, stream)
        else:
            if self.dit_o_fuse:
                # Bias rides the residual epilogue (one launch and one pass
                # fewer; the fused kernel the ViT proj/fc2 residuals use).
                self._gemm(self.dit_o, blk["o"]["weight"], self.dit_tmp,
                           N, D, D, stream)
                self._vlk.residual_add_bias_bf16(
                    self.dit_x.data_ptr(), self.dit_tmp.data_ptr(),
                    blk["o"]["bias"].data_ptr(), N, D, stream)
            else:
                self._gemm_bias(self.dit_o, blk["o"]["weight"],
                                blk["o"]["bias"], self.dit_tmp, N, D, D, stream)
                fvk.residual_add(self.dit_x.data_ptr(), self.dit_tmp.data_ptr(),
                                 N * D, stream)
            fvk.nvfp4_skinny_layer_norm(self.dit_x.data_ptr(), 0, 0, xn4, xnsf4,
                                        one, 0, N, D, 1e-5, True, stream)

        # FFN up -> bias + tanh-GELU -> operand for the down projection.
        w, sf, gs, b = W["ff1"]
        fvk.nvfp4_skinny_gemm(xn4, xnsf4, w, sf, part, N, FF, D,
                              self.dit_fp4_cfg[(FF, D)], True, stream)
        fvk.nvfp4_skinny_sum_bias_gelu(part, _splits(FF, D), one, gs, b,
                                       hid4, hidsf4, one, N, FF, True, stream)

        # FFN down -> bias + residual -> the next block's AdaLN (final block:
        # the step's output AdaLN, written BF16 for the output projection).
        w, sf, gs, b = W["ff2"]
        fvk.nvfp4_skinny_gemm(hid4, hidsf4, w, sf, part, N, D, FF,
                              self.dit_fp4_cfg[(D, FF)], True, stream)
        if i < 15:
            # The epilogue feeds the *next* block's attention, whose operand
            # format follows the next block's precision, not this block's.
            nxt_attn_fp4 = (self.dit_fp4_sites == "all"
                            and (i + 1) not in self.dit_attn_q_bf16_blocks)
            nxt_s = self.st_norm1_s[step, i + 1].data_ptr()
            nxt_b = self.st_norm1_b[step, i + 1].data_ptr()
            fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, _splits(D, FF), one, gs, b, self.dit_x.data_ptr(),
                nxt_s, nxt_b,
                xn4 if nxt_attn_fp4 else 0, xnsf4 if nxt_attn_fp4 else 0,
                one if nxt_attn_fp4 else 0, 0 if nxt_attn_fp4 else self.dit_xn.data_ptr(),
                N, D, 1e-5, True, stream)
        else:
            fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, _splits(D, FF), one, gs, b, self.dit_x.data_ptr(),
                self.st_q_scale[step].data_ptr(), self.st_q_share[step].data_ptr(),
                0, 0, 0, self.dit_xn.data_ptr(), N, D, 1e-6, True, stream)

    def _dit_output(self, step: int, stream: int, norm_done: bool = False) -> None:
        fvk = self._fvk
        D, CH = self.dim_dit, self.chunk_size
        N = self.dit_seq
        if not norm_done:
            fvk.ada_layer_norm_bf16(self.dit_x.data_ptr(), self.st_q_scale[step].data_ptr(),
                                    self.st_q_share[step].data_ptr(), self.dit_xn.data_ptr(),
                                    N, D, 1e-6, stream)
        relu_done = False
        if self.dit_out_merge and self.dit_tail_fold:
            w, b = self.d_out_merged
            fvk.bf16_matmul_cublaslt_bias_ex_bf16(
                self.dit_xn[self.num_cond + 1:].data_ptr(), w.data_ptr(), b.data_ptr(),
                self.dit_dec1.data_ptr(), CH, self.dit_dec_h, D, 0, True, stream)
            relu_done = True
        elif self.dit_out_merge:
            # proj_out_2 and the action decoder's first linear have no
            # nonlinearity between them: one GEMM with the composed weight
            w, b = self.d_out_merged
            self._gemm_bias(self.dit_xn[self.num_cond + 1:], w, b, self.dit_dec1, CH,
                            self.dit_dec_h, D, stream)
        else:
            self._gemm_bias(self.dit_xn[self.num_cond + 1:], self.d_proj_out2["weight"],
                            self.d_proj_out2["bias"], self.dit_pred, CH, self.dit_dec_h,
                            D, stream)
            self._gemm_bias(self.dit_pred, self.d_act_dec["layer1"]["weight"],
                            self.d_act_dec["layer1"]["bias"], self.dit_dec1, CH,
                            self.dit_dec_h, self.dit_dec_h, stream)
        if not relu_done:
            fvk.relu_inplace_bf16(self.dit_dec1.data_ptr(), CH * self.dit_dec_h, stream)
        self._gemm_bias_generic(self.dit_dec1, self.d_act_dec["layer2"]["weight"],
                                self.d_act_dec["layer2"]["bias"], self.dit_vel, CH,
                                self.action_dim, self.dit_dec_h, stream)

    def _euler(self, step: int, stream: int) -> None:
        """actions += dt * velocity; the action decoder is pre-scaled by dt."""
        self._fvk.residual_add(self.actions.data_ptr(), self.dit_vel.data_ptr(),
                               self.chunk_size * self.action_dim, stream,
                               include_tail=True)

    # ───────────────────────── entry point ─────────────────────────

    @staticmethod
    def _aa_bounds(in_size: int, out_size: int) -> "torch.Tensor":
        """start = floor(o*in/out), end = ceil((o+1)*in/out) per output index."""
        o = torch.arange(out_size, dtype=torch.int32)
        start = (o * in_size) // out_size
        end = -(-((o + 1) * in_size) // out_size)
        return torch.stack([start, end], dim=1).contiguous().cuda()

    def preprocess_stage(self, stream: int) -> None:
        """Raw uint8 camera frames -> patchified pixel_values, on device.

        The reference chain is: float/[0,1] -> area resize to
        ``resize_images_to`` -> Qwen3-VL smart resize (grid factor
        patch*merge, bicubic + antialias) -> mean/std 0.5 normalize -> 16x16
        patchify with the temporal tile, in the processor's reshape/permute
        order. All of it runs as repository-native kernels
        (``vla_frames_area_resize`` / ``vla_frames_bicubic_resize`` /
        ``vla_frames_patchify``), so the captured graph holds no framework
        kernels; the kernels reproduce the reference arithmetic (same pooling
        windows and accumulation order, the antialias kernel's a=-0.5 cubic
        with renormalised taps, the same normalize/patchify order).
        """
        v = self.num_views
        gh, gw = self.vis_grid
        tps, ps, m = 2, 16, self.merge
        fvk = self._kernels()
        h, w = self.resize_images_to
        fvk.vla_frames_area_resize(
            self.raw_frames.data_ptr(), self.pre_area.data_ptr(),
            self.pre_ybounds.data_ptr(), self.pre_xbounds.data_ptr(), v, 256, 256,
            3, h, w, stream)
        if fvk.vla_frames_bicubic_patchify(
                self.pre_area.data_ptr(), self.pix.data_ptr(), v, 3, h, w, gh * ps,
                gw * ps, tps, ps, m, float(self.img_mean), float(self.img_std), stream):
            return
        fvk.vla_frames_bicubic_resize(
            self.pre_area.data_ptr(), self.pre_bic.data_ptr(), v * 3, h, w,
            gh * ps, gw * ps, stream)
        fvk.vla_frames_patchify(
            self.pre_bic.data_ptr(), self.pix.data_ptr(), v, 3, gh * ps, gw * ps,
            tps, ps, m, float(self.img_mean), float(self.img_std), stream)

    def state_stage(self, stream: int) -> None:
        """Raw state -> MEAN_STD normalized (checkpoint statistics), native."""
        self._kernels().vla_state_normalize(
            self.raw_state.data_ptr(), self.state_mean.data_ptr(),
            self.state_std.data_ptr(), float(self.state_eps), self.state_dim,
            self.state_norm.data_ptr(), stream)

    def postprocess_stage(self, stream: int) -> None:
        """Delivered chunk: clip -> gripper pre-snap -> MIN_MAX inverse ->
        gripper binarize (lerobot's postprocessor, in the same order), as one
        fused launch with the exact per-column fp32 arithmetic of the chain."""
        if not (self.clip_normalized_actions and self.pre_snap_gripper_action
                and self.binarize_gripper_action):
            raise RuntimeError("postprocess_stage is fused for the delivered "
                               "clip + pre-snap + binarize policy")
        self._fvk.vla_action_postprocess(
            self.actions.data_ptr(), self.out_actions.data_ptr(),
            self.action_min.data_ptr(), self.action_max.data_ptr(),
            self.chunk_size, self.action_dim, self.gripper_dim,
            self.gripper_threshold, stream)

    def forward(self, stream: int | None = None) -> int:
        """Run the whole pipeline (raw frames -> delivered chunk); returns the
        delivered action buffer pointer."""
        self._kernels()
        s = self._stream() if stream is None else int(stream)
        self.preprocess_stage(s)
        if not self.dit_prep_side:
            self.state_stage(s)
            self.vision_stage(s)
            self.llm_stage(s)
            self.dit_state_token(s)
            self.dit_stage(s)
            self.postprocess_stage(s)
            return int(self.out_actions.data_ptr())
        self.vision_stage(s)
        # The state token and the first Euler step's input rows depend only
        # on the raw state and the noise: they run on a forked branch beside
        # the decoder (forked after the tower, the last cuBLASLt user before
        # the head, since the BF16 cuBLASLt entries share one workspace).
        # The branch joins before the head: a join inside the head's PDL
        # chain (e.g. per-cross-block conditioning K/V) slows every step.
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.device)
        main = torch.cuda.ExternalStream(s)
        ev = torch.cuda.Event()
        ev.record(main)
        self._side.wait_event(ev)
        ss = self._side.cuda_stream
        self.state_stage(ss)
        self.dit_state_token(ss)
        self._assemble_step_inputs(0, ss)
        ev = torch.cuda.Event()
        ev.record(self._side)
        self.llm_stage(s)
        main.wait_event(ev)
        self.dit_stage(s, first_inputs_ready=True)
        self.postprocess_stage(s)
        return int(self.out_actions.data_ptr())
