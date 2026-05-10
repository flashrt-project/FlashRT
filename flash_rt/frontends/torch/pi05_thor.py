"""FlashRT -- Pi05TorchFrontendThor: complete inference using ONLY flash_rt_kernels.so.

All computation goes through flash_rt_kernels (pybind11 + CUTLASS + cuBLASLt).

Usage:
    pipe = Pi05TorchFrontendThor("/path/to/checkpoint", num_views=2)
    pipe.set_prompt("pick up the red cup")
    result = pipe.infer({"image": img1, "wrist_image": img2})
    actions = result["actions"]  # (10, 7) numpy
"""

import ctypes
import json
import math
import logging
import pathlib
import time
from typing import Optional, Union

from flash_rt.hardware.thor.shared_primitives import (
    siglip_forward,
    postln_project,
    encoder_forward,
    encoder_forward_calibrate,
)
from flash_rt.models.pi05.pipeline_thor import (
    decoder_forward,
    decoder_forward_calibrate,
)
from flash_rt.hardware.thor.attn_backend import (
    ThorFlashAttnBackend,
    make_pi05_attention_spec,
)

import numpy as np
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
from flash_rt.core.quant.calibrator import load_calibration, save_calibration

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

_cudart = ctypes.CDLL("libcudart.so")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from flash_rt.core.thor_frontend_utils import embed_prompt_torch as embed_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Calibration functions (replaces old C encoder_full_scaled / ae_forward_scaled)
# ---------------------------------------------------------------------------

# ===========================================================================
# ThorPipeline
# ===========================================================================

class Pi05TorchFrontendThor:
    """Complete Pi0.5 inference pipeline using only flash_rt_kernels.so.

    Interface compatible with FlashRTModel.predict():
        set_prompt(prompt_text)
        infer(observation) -> {"actions": np.ndarray}
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, checkpoint_dir: str, num_views: int = 2,
                 use_cuda_graph: bool = True, autotune: int = 3,
                 use_fp8: bool = True):
        """
        Args:
            autotune: CUDA Graph autotune trials per set_prompt().
                0 = off, 3 = default (fast), 5+ = thorough.
                Torch usually finds fast graph on trial 0-1.
        """
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = num_views
        self.use_cuda_graph = use_cuda_graph
        self.use_fp8 = bool(use_fp8)
        self.autotune = int(autotune) if autotune is not True else 3
        if autotune is False:
            self.autotune = 0
        self.latency_records = []
        self.calibrated = False
        self.graph_captured = False
        self._real_data_calibrated = False

        # ---- RL CFG state (set via set_rl_mode) ----
        # When ``_rl_config`` is non-None, ``set_prompt`` builds an
        # advantage-conditioned (cond) + raw (uncond) prompt pair, and
        # ``infer`` runs the encoder + decoder twice (once per branch)
        # then combines the two action chunks via
        # ``cfg_combine_into_residual_fp16``. Mirrors the RTX
        # ``Pi05CFGPipeline`` contract per-chunk (vs RTX per-step) — the
        # math collapses to cond-only at ``cfg_beta=1.0`` either way.
        self._rl_config: Optional[dict] = None
        self._lang_emb_cond: Optional[torch.Tensor] = None
        self._lang_emb_uncond: Optional[torch.Tensor] = None
        # Lazy-allocated per-set_prompt (sized by Sa once known).
        self._noise_R_snapshot: Optional[torch.Tensor] = None
        self._v_cond_buf: Optional[torch.Tensor] = None
        self._v_uncond_buf: Optional[torch.Tensor] = None
        self._rl_current_prompt_text: Optional[str] = None

        # ---- FvkContext + GemmRunner ----
        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()

        # ---- Load CUTLASS FMHA (compiled from csrc/attention/) ----
        # Search order: next to the checkpoint, then the installed
        # ``flash_rt/`` package dir (pip + editable installs land here via
        # the ``package-data = ["*.so"]`` glob in pyproject.toml), then the
        # uncopied ``build/`` output of a fresh cmake run, then the docker
        # container convention ``/workspace/``.
        fmha_paths = [
            str(checkpoint_dir.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent.parent / "build" / "libfmha_fp16_strided.so"),
            "/workspace/libfmha_fp16_strided.so",
        ]
        fmha_loaded = False
        for p in fmha_paths:
            if pathlib.Path(p).exists():
                ret = fvk.load_fmha_strided_library(p)
                if ret == 0:
                    fmha_loaded = True
                    logger.info("CUTLASS FMHA loaded from %s", p)
                    break
        if not fmha_loaded:
            logger.warning("CUTLASS strided FMHA not found — SigLIP will use cuBLAS attention fallback")

        # ---- Norm stats ----
        self._load_norm_stats(checkpoint_dir)

        # ---- Weights (safetensors only — JAX/Orbax uses ThorPipelineJax) ----
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path}. "
                "For Orbax/JAX checkpoints, use ThorPipelineJax.")
        self._checkpoint_path = str(safetensors_path)
        self._load_weights(safetensors_path)
        logger.info("Pi05TorchFrontendThor initialised (num_views=%d)", num_views)

    # -----------------------------------------------------------------------
    # norm_stats
    # -----------------------------------------------------------------------

    def _load_norm_stats(self, checkpoint_dir):
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir.parent / "pi05_libero" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            pathlib.Path("/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/"
                         "assets/physical-intelligence/libero/norm_stats.json"),
            *lerobot_candidates(checkpoint_dir),
        ]
        self.norm_stats = load_norm_stats(
            candidates, checkpoint_dir=checkpoint_dir)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def _load_weights(self, safetensors_path):
        from safetensors import safe_open

        from flash_rt.executors.torch_weights import (
            SafetensorsSource, WeightLoader,
        )
        from flash_rt.frontends.torch._pi05_thor_spec import build_spec

        from flash_rt.executors.torch_weights import _autodetect_strip_prefix
        sf = safe_open(str(safetensors_path), framework='pt', device='cuda')
        # The lerobot HF policy releases (e.g. lerobot/pi05_libero
        # _finetuned_v044) wrap every weight key under an extra
        # ``model.`` namespace. Auto-detect and prepend it on the raw
        # safetensors lookup so the rest of this loader can stay
        # written against the openpi-converted bare keys.
        _strip = _autodetect_strip_prefix(set(sf.keys()))
        def g_raw(k): return sf.get_tensor((_strip + k) if _strip else k)
        def g(k): return g_raw(k).to(fp16)

        # Declarative weight-loader pass (stage 7.3). Populates:
        #   self._sig_{ln_attn,ln_ffn,qkv,o,up,down}_{w,b}  (27-layer lists)
        #   self._sig_alpha                                  (108 fp32 scales)
        #   self._enc_{qkv,o,gu,d}_w                          (18-layer lists)
        #   self._enc_w_scales                                (72 fp32 scales)
        #   self._dec_{qkv,o,gu,d}_flat                       (flat cat tensors)
        #   self._ae_w_scales                                 (72 fp32 scales)
        #   self._{attn,ffn}_mod_{w,b}                        (18-layer lists)
        _src = SafetensorsSource(str(safetensors_path), device='cuda')
        WeightLoader(source=_src, target=self, spec=build_spec()).run()

        self.embedding_weight = g('paligemma_with_expert.paligemma.lm_head.weight')

        # ===============================================================
        # SigLIP  (27 layers)
        # ===============================================================
        vp = 'paligemma_with_expert.paligemma.model.vision_tower.vision_model'
        nv = self.num_views
        S_sig = nv * 256
        D_sig, H_sig, NH_sig, HD_sig, L_sig = 1152, 4304, 16, 72, 27
        self.sig_S = S_sig
        self.sig_D = D_sig
        self.sig_H = H_sig
        self.sig_NH = NH_sig
        self.sig_HD = HD_sig
        self.sig_L = L_sig

        # Per-layer weights, scales, and biases loaded declaratively above.
        # Compose the pointer dict consumed by shared_primitives.siglip_forward.
        self._sig_weights = {
            'ln_attn_w': [w.data_ptr() for w in self._sig_ln_attn_w],
            'ln_attn_b': [w.data_ptr() for w in self._sig_ln_attn_b],
            'qkv_w':     [w.data_ptr() for w in self._sig_qkv_w],
            'qkv_b':     [w.data_ptr() for w in self._sig_qkv_b],
            'o_w':       [w.data_ptr() for w in self._sig_o_w],
            'o_b':       [w.data_ptr() for w in self._sig_o_b],
            'ln_ffn_w':  [w.data_ptr() for w in self._sig_ln_ffn_w],
            'ln_ffn_b':  [w.data_ptr() for w in self._sig_ln_ffn_b],
            'up_w':      [w.data_ptr() for w in self._sig_up_w],
            'up_b':      [w.data_ptr() for w in self._sig_up_b],
            'down_w':    [w.data_ptr() for w in self._sig_down_w],
            'down_b':    [w.data_ptr() for w in self._sig_down_b],
            'alpha':     self._sig_alpha,
        }
        # Unit scale (1.0) for FP8 casts without calibration (LN output already normalized)
        self._unit_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        self._sig_weights['unit_scale'] = self._unit_scale.data_ptr()
        # SiLU scale: 1/SILU_SCALE = 5.0 (kernel does val * 1/d_scale = val * 0.2)
        self._silu_scale = torch.tensor([5.0], dtype=torch.float32, device='cuda')

        # SigLIP buffers
        self._sig_x     = torch.zeros(S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_x_fp8 = torch.zeros(S_sig * D_sig, dtype=torch.uint8, device='cuda')
        self._sig_qkv   = torch.empty(S_sig, 3 * D_sig, dtype=fp16, device='cuda')
        self._sig_attn   = torch.empty(S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_hidden = torch.empty(S_sig, H_sig, dtype=fp16, device='cuda')
        self._sig_hid_fp8 = torch.zeros(S_sig * H_sig, dtype=torch.uint8, device='cuda')

        self._sig_bufs = {
            'x':       self._sig_x.data_ptr(),
            'x_fp8':   self._sig_x_fp8.data_ptr(),
            'qkv':     self._sig_qkv.data_ptr(),
            'attn_out': self._sig_attn.data_ptr(),
            'hidden':  self._sig_hidden.data_ptr(),
            'hid_fp8': self._sig_hid_fp8.data_ptr(),
        }
        self._sig_dims = {
            'S': S_sig, 'D': D_sig, 'H': H_sig,
            'NH': NH_sig, 'HD': HD_sig, 'L': L_sig,
            'num_views': nv, 'seq_per_view': 256,
        }

        # Patch embedding weights (HWC im2col order)
        pe_w_2d = (g(f'{vp}.embeddings.patch_embedding.weight')
                   .reshape(D_sig, 3, 14, 14)
                   .permute(0, 2, 3, 1)
                   .reshape(D_sig, -1)
                   .T.contiguous())
        self._pe_w = CudaBuffer.from_numpy(pe_w_2d.cpu().numpy().copy())
        self._pe_b = CudaBuffer.from_numpy(
            g(f'{vp}.embeddings.patch_embedding.bias').cpu().numpy().copy())
        self._pos_emb = CudaBuffer.from_numpy(
            g(f'{vp}.embeddings.position_embedding.weight')[:256].cpu().numpy().copy())
        self._img_buf = CudaBuffer.device_empty(nv * 224 * 224 * 3, np.float16)
        self._patches_buf = CudaBuffer.device_empty(S_sig * 588, np.float16)

        # PostLN weights
        self._postln_w = g(f'{vp}.post_layernorm.weight')
        self._postln_b = g(f'{vp}.post_layernorm.bias')
        mp = 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear'
        self._proj_w = g(f'{mp}.weight').T.contiguous()  # [D_sig, D_enc] for bf16_nn
        self._proj_b = g(f'{mp}.bias')
        self._postln_scratch = torch.empty(S_sig, max(D_sig, H_sig), dtype=fp16, device='cuda')

        # ===============================================================
        # Encoder  (18 layers, GQA)
        # ===============================================================
        De, He, Le = 2048, 16384, 18
        NHe, HDe = 8, 256
        Se_max = nv * 256 + 256  # max sequence length (views*256 + max prompt)
        self.De = De; self.He = He; self.Le = Le
        self.NHe = NHe; self.HDe = HDe; self.Se_max = Se_max

        # Encoder per-layer weights loaded declaratively above:
        #   self._enc_qkv_w   list of [2560, D]   fp8 (fused-norm + GQA interleave)
        #   self._enc_o_w     list of [2048, 2048] fp8
        #   self._enc_gu_w    list of [2*H, D]    fp8 (fused post-attn norm)
        #   self._enc_d_w     list of [H, D]      fp8
        #   self._enc_w_scales  host list of Le*4 floats (order: q, o, gu, d per layer)
        self._enc_w_dev = torch.tensor(self._enc_w_scales, dtype=torch.float32, device='cuda')

        # RoPE table
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256))
        kp = inv_freq[None, :] * torch.arange(1200, device='cuda')[:, None].float()
        self._kc_t = torch.cos(kp).to(fp16)
        self._ks_t = torch.sin(kp).to(fp16)
        self._enc_rope = torch.empty(Se_max, 256, dtype=fp16, device='cuda')

        # KV cache
        Sa, Da, Ha, La = 10, 1024, 4096, 18
        self.Sa = Sa; self.Da = Da; self.Ha = Ha; self.La = La
        total_keys_max = Se_max + Sa
        self._Kc = torch.zeros(Le, total_keys_max, HDe, dtype=fp16, device='cuda')
        self._Vc = torch.zeros(Le, total_keys_max, HDe, dtype=fp16, device='cuda')

        # Encoder buffers
        self._enc_x      = torch.empty(Se_max, De, dtype=fp16, device='cuda')
        self._enc_x_fp8  = torch.zeros(Se_max * De, dtype=torch.uint8, device='cuda')
        self._enc_qkv_buf = torch.empty(Se_max, 2560, dtype=fp16, device='cuda')
        self._enc_logits = torch.empty(Se_max * NHe, total_keys_max, dtype=fp16, device='cuda')
        self._enc_attn   = torch.empty(Se_max, NHe * HDe, dtype=fp16, device='cuda')
        self._enc_o_fp8  = torch.zeros(Se_max * NHe * HDe, dtype=torch.uint8, device='cuda')
        self._enc_gate   = torch.empty(Se_max, 2 * He, dtype=fp16, device='cuda')
        self._enc_hidden = torch.empty(Se_max, He, dtype=fp16, device='cuda')
        self._enc_hid_fp8 = torch.zeros(Se_max * He, dtype=torch.uint8, device='cuda')
        self._enc_fg     = torch.empty(Se_max, De, dtype=fp16, device='cuda')

        # ===============================================================
        # Decoder / AE  (18 layers, 10 steps)
        # ===============================================================
        dp = 'paligemma_with_expert.gemma_expert'
        steps = 10
        D3a = 3 * Da

        # Decoder/AE per-layer weights loaded declaratively above as flat cats:
        #   self._dec_qkv_flat  per-layer [q|k|v] fp8  (.t().contiguous(), interleaved Q/K)
        #   self._dec_o_flat    per-layer O   fp8
        #   self._dec_gu_flat   per-layer [gate|up] fp8
        #   self._dec_d_flat    per-layer down  fp8
        #   self._ae_w_scales   host list of La*4 floats (order: q, o, gu, d per layer)

        # Action in/out projections (singletons)
        self._ain_w = g('action_in_proj.weight').t().contiguous()
        self._ain_b = g('action_in_proj.bias')
        # Note: action_out_proj has -1/steps scaling baked in
        self._aow = g('action_out_proj.weight').t().contiguous() * (-1.0 / steps)
        self._aob = g('action_out_proj.bias') * (-1.0 / steps)

        self._ae_w_dev = torch.tensor(self._ae_w_scales, dtype=torch.float32, device='cuda')

        # Time conditioning weights
        self._time_mlp_in_w  = g('time_mlp_in.weight')
        self._time_mlp_in_b  = g('time_mlp_in.bias')
        self._time_mlp_out_w = g('time_mlp_out.weight')
        self._time_mlp_out_b = g('time_mlp_out.bias')
        # Per-layer AdaRMS modulation Dense layers (self._{attn,ffn}_mod_{w,b})
        # loaded declaratively above. Final norm modulation stays inline.
        dp_full = 'paligemma_with_expert.gemma_expert'
        self._final_mod_w = g(f'{dp_full}.model.norm.dense.weight')
        self._final_mod_b = g(f'{dp_full}.model.norm.dense.bias')

        # Decoder buffers (pre-allocate at max sizes)
        self._dec_rope = torch.empty(Sa, 256, dtype=fp16, device='cuda')
        self._ae_x   = torch.empty(Sa, Da, dtype=fp16, device='cuda')
        self._ae_xn  = torch.empty(Sa, Da, dtype=fp16, device='cuda')
        self._ae_gate = torch.empty(Sa, Da, dtype=fp16, device='cuda')
        self._ae_qkv = torch.empty(Sa, 2560, dtype=fp16, device='cuda')
        self._ae_logits = torch.empty(Sa * 8, total_keys_max, dtype=fp16, device='cuda')
        self._ae_attn = torch.empty(Sa * 8, 256, dtype=fp16, device='cuda')
        self._ae_hid  = torch.empty(Sa, 2 * Ha, dtype=fp16, device='cuda')
        self._ae_fg   = torch.empty(Sa, 2 * Ha, dtype=fp16, device='cuda')  # must fit Gate+Up GEMM output [Sa, 2H]
        self._ae_xn_fp8  = torch.zeros(Sa * Da, dtype=torch.uint8, device='cuda')
        self._ae_hid_fp8 = torch.zeros(Sa * Ha, dtype=torch.uint8, device='cuda')
        self._ae_ctx_fp8 = torch.zeros(Sa * 8 * 256, dtype=torch.uint8, device='cuda')
        self._g_noise = torch.zeros(Sa, 32, dtype=fp16, device='cuda')

        # Calibration scale buffers
        self._enc_calib_scales = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        self._ae_calib_scales  = torch.zeros(La * 4, dtype=torch.float32, device='cuda')

        # ── B=N batched mode state (Stage 2 of Thor batched-CFG port) ──
        # Inactive by default — buffers + B=N graph capture are
        # lazily created when ``set_batched_mode(enable=True)`` fires.
        self._batched = False
        self.B = 1
        # Lazy-allocated b2-suffixed buffers (None → not yet created).
        # See ``_alloc_b2_buffers`` for the full inventory.
        self._Kc_b2 = None
        self._Vc_b2 = None
        self._enc_x_b2 = None
        self._enc_x_fp8_b2 = None
        self._enc_qkv_buf_b2 = None
        self._enc_logits_b2 = None
        self._enc_attn_b2 = None
        self._enc_o_fp8_b2 = None
        self._enc_gate_b2 = None
        self._enc_hid_fp8_b2 = None
        self._enc_fg_b2 = None
        self._ae_x_b2 = None
        self._ae_xn_b2 = None
        self._ae_gate_b2 = None
        self._ae_qkv_b2 = None
        self._ae_logits_b2 = None
        self._ae_attn_b2 = None
        self._ae_fg_b2 = None
        self._ae_xn_fp8_b2 = None
        self._ae_hid_fp8_b2 = None
        self._ae_ctx_fp8_b2 = None
        self._g_noise_b2 = None
        self._v_b2 = None
        # B-tiled style buffers — built in set_prompt when ``_batched``.
        self._sa_all_b2 = None
        self._sf_all_b2 = None
        self._fs_all_b2 = None
        # B=N captured graph (separate from the B=1 ``_enc_ae_graph``).
        self._enc_ae_graph_b2 = None
        # Outer fused-CFG graph: lang swap (×2) + SigLIP (×2) +
        # encoder_b2 + decoder_b2 (with per-step CFG inside). One
        # ``replay()`` per CFG inference. Captured by
        # ``_capture_cfg_b2_outer_graph`` and consumed via
        # ``Pi05ThorCFGBatchedPipeline.forward``. Mirrors RTX
        # ``Pi05CFGBatchedPipeline.forward`` /
        # ``self._graph.replay()`` at
        # ``pipeline_rtx_cfg_batched.py:419``.
        self._cfg_b2_outer_graph = None
        # When non-None, ``_capture_enc_ae_graph_b2`` bakes the per-step
        # CFG combine + noise mirror into the graph (paper-correct
        # per-step CFG; mirrors RTX
        # ``Pi05CFGBatchedPipeline.transformer_decoder_batched``).
        # Cleared by ``set_rl_mode(cfg_enable=False)``; set by
        # ``_build_cfg_batched_pipeline``.
        self._enc_ae_graph_b2_cfg_beta = None

        logger.info("Weights loaded for Pi05TorchFrontendThor")

    def _alloc_b2_buffers(self, B: int = 2) -> None:
        """Allocate all B=N inference buffers.

        Stage 2 of the Thor batched-CFG port. Called by
        :meth:`set_batched_mode` to lazily set up B-folded buffers
        without disturbing the B=1 hot path. The B=1 buffers stay
        allocated; the b2 set is parallel.

        Buffer layout convention: leading row dim is ``B * <per-sample>``,
        flat in memory. ``_Kc_b2`` and ``_Vc_b2`` are
        ``(B, La, total_keys_max, HD)`` so per-sample slabs can be
        addressed via ``_Kc_b2[b].view(-1).data_ptr()``.
        """
        if B < 2:
            raise ValueError(f"_alloc_b2_buffers requires B >= 2; got {B}")
        Le = self.Le; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha; La = self.La
        Se_max = self._enc_x.shape[0]
        total_keys_max = self._Kc.shape[1]

        # ── KV cache (B, La, total_keys_max, HD) ──
        self._Kc_b2 = torch.zeros(B, Le, total_keys_max, HDe,
                                   dtype=fp16, device='cuda')
        self._Vc_b2 = torch.zeros(B, Le, total_keys_max, HDe,
                                   dtype=fp16, device='cuda')

        # ── Encoder buffers (B*Se, ...) ──
        self._enc_x_b2     = torch.empty(B * Se_max, De, dtype=fp16, device='cuda')
        self._enc_x_fp8_b2 = torch.zeros(B * Se_max * De, dtype=torch.uint8, device='cuda')
        self._enc_qkv_buf_b2 = torch.empty(B * Se_max, 2560, dtype=fp16, device='cuda')
        self._enc_logits_b2  = torch.empty(Se_max * NHe, total_keys_max,
                                            dtype=fp16, device='cuda')  # scratch reused per sample
        self._enc_attn_b2    = torch.empty(B * Se_max, NHe * HDe, dtype=fp16, device='cuda')
        self._enc_o_fp8_b2   = torch.zeros(B * Se_max * NHe * HDe, dtype=torch.uint8, device='cuda')
        self._enc_gate_b2    = torch.empty(B * Se_max, 2 * He, dtype=fp16, device='cuda')
        self._enc_hid_fp8_b2 = torch.zeros(B * Se_max * He, dtype=torch.uint8, device='cuda')
        self._enc_fg_b2      = torch.empty(B * Se_max, De, dtype=fp16, device='cuda')

        # ── Decoder buffers (B*Sa, ...) ──
        self._ae_x_b2   = torch.empty(B * Sa, Da, dtype=fp16, device='cuda')
        self._ae_xn_b2  = torch.empty(B * Sa, Da, dtype=fp16, device='cuda')
        self._ae_gate_b2 = torch.empty(B * Sa, Da, dtype=fp16, device='cuda')
        self._ae_qkv_b2 = torch.empty(B * Sa, 2560, dtype=fp16, device='cuda')
        self._ae_logits_b2 = torch.empty(Sa * 8, total_keys_max,
                                          dtype=fp16, device='cuda')  # scratch reused per sample
        self._ae_attn_b2 = torch.empty(B * Sa * 8, 256, dtype=fp16, device='cuda')
        self._ae_fg_b2   = torch.empty(B * Sa, 2 * Ha, dtype=fp16, device='cuda')
        self._ae_xn_fp8_b2  = torch.zeros(B * Sa * Da, dtype=torch.uint8, device='cuda')
        self._ae_hid_fp8_b2 = torch.zeros(B * Sa * Ha, dtype=torch.uint8, device='cuda')
        self._ae_ctx_fp8_b2 = torch.zeros(B * Sa * 8 * 256, dtype=torch.uint8, device='cuda')
        self._g_noise_b2 = torch.zeros(B * Sa, 32, dtype=fp16, device='cuda')
        # Per-step velocity scratch for the CFG-batched graph: each step
        # writes (v_cond, v_uncond) here before the in-graph cfg_combine
        # mixes them into ``_g_noise_b2`` slot 0. Always allocated; the
        # non-CFG b2 path simply doesn't reference it.
        self._v_b2 = torch.zeros(B * Sa, 32, dtype=fp16, device='cuda')

        self.B = int(B)
        logger.info(
            "Allocated B=%d buffers (Se_max=%d, Sa=%d, total_keys_max=%d)",
            B, Se_max, Sa, total_keys_max)

    # NOTE: _load_weights_orbax removed — Orbax loading belongs to ThorPipelineJax
    # -----------------------------------------------------------------------
    # RL CFG inference (opt-in via set_rl_mode)
    # -----------------------------------------------------------------------

    def set_rl_mode(
        self,
        *,
        cfg_enable: bool = True,
        cfg_beta: float = 1.5,
        advantage_positive: bool = True,
    ) -> None:
        """Enable / configure advantage-conditioned RL inference (opt-in).

        When enabled, subsequent :meth:`set_prompt` calls build a
        conditioned prompt with an "Advantage: positive" / "negative"
        ACP tag appended; the unconditioned prompt is the raw task text.
        :meth:`infer` then runs the encoder + decoder twice (cond /
        uncond) and combines the two action chunks via
        ``fvk.cfg_combine_into_residual_fp16`` with strength ``cfg_beta``.

        Mirrors :meth:`Pi05TorchFrontendRtx.set_rl_mode` (the public API
        contract is byte-equal). The Thor fused B=2 path now does
        per-step CFG inside the captured ``_enc_ae_graph_b2`` (matches
        RTX :meth:`Pi05CFGBatchedPipeline.transformer_decoder_batched`
        and arXiv:2511.14759 Appendix E); the serial path is still
        per-chunk (cheaper, no graph recapture needed).

        Args:
            cfg_enable: ``True`` activates CFG inference; ``False`` clears
                any previous RL config and reverts the next
                :meth:`set_prompt` to the standard single-forward path.
            cfg_beta: Guidance strength. Must be ``>= 1.0`` (1.0 collapses
                to cond-only). Common deployment range ``[1.5, 2.5]``.
            advantage_positive: Whether the conditioned prompt uses the
                positive advantage tag (the standard "select for high
                advantage" use case). Set ``False`` only for debugging.
        """
        if not cfg_enable:
            self._rl_config = None
            self._lang_emb_cond = None
            self._lang_emb_uncond = None
            # Force the next set_prompt to rebuild the standard graph.
            self.graph_captured = False
            # If a CFG-batched B=2 graph was previously captured (with
            # in-graph cfg_combine), invalidate it so the next batched
            # use recaptures without the CFG schedule baked in.
            if self._enc_ae_graph_b2_cfg_beta is not None:
                self._enc_ae_graph_b2_cfg_beta = None
                self._enc_ae_graph_b2 = None
            return
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        new_config = {
            "cfg_beta": float(cfg_beta),
            "advantage_positive": bool(advantage_positive),
        }
        if self._rl_config != new_config:
            self._rl_config = new_config
            # Force graph rebuild on next set_prompt so the new beta /
            # mode takes effect.
            self.graph_captured = False
        logger.info(
            "RL mode enabled: cfg_beta=%.2f, advantage_positive=%s",
            new_config["cfg_beta"], new_config["advantage_positive"])

    def _ensure_cfg_buffers(self):
        """Lazy-allocate the CFG-only intermediate buffers."""
        n = self.Sa * 32
        if self._noise_R_snapshot is None:
            self._noise_R_snapshot = torch.empty(
                n, dtype=fp16, device='cuda')
        if self._v_cond_buf is None:
            self._v_cond_buf = torch.empty(n, dtype=fp16, device='cuda')
        if self._v_uncond_buf is None:
            self._v_uncond_buf = torch.empty(n, dtype=fp16, device='cuda')

    def _set_prompt_rl(self, prompt_text):
        """Build cond + uncond embeds, drive the standard set_prompt path
        with cond_text (cond is always >= uncond in length because the
        ACP tag appends to the task), then pad uncond to cond_len and
        stash it for the runtime swap.

        The captured ``_siglip_graph`` reads from the cond tensor at
        ``self._lang_emb.data_ptr()``; :meth:`_infer_cfg` ``copy_``s
        either ``_lang_emb_cond`` or ``_lang_emb_uncond`` into that
        same buffer between the two siglip replays without touching
        graph capture.
        """
        if isinstance(prompt_text, (np.ndarray, list)):
            raise ValueError(
                "set_rl_mode requires a text prompt (the ACP tag is "
                "appended at the string level); pass a str, not token IDs")
        from flash_rt.core.rl import build_acp_tagged_task

        cfg = self._rl_config
        cond_text = build_acp_tagged_task(
            prompt_text, is_positive=cfg["advantage_positive"])
        uncond_text = prompt_text

        # Drive the standard graph-capture path with cond_text; this
        # sets self._lang_emb / self._S_lang / self.Se at cond_len.
        self.graph_captured = False
        self.set_prompt(cond_text)
        target_len = self._S_lang  # actual padded language slot length

        # Compute uncond embeds and pad to target_len (cond is >= uncond
        # because the ACP tag adds tokens).
        uncond_emb, uncond_len = embed_prompt(
            uncond_text, self.embedding_weight, max_len=48)
        if uncond_emb.shape[0] > target_len:
            raise RuntimeError(
                f"uncond_len={uncond_emb.shape[0]} > cond target_len="
                f"{target_len}; the ACP tag is supposed to make cond at "
                f"least as long as uncond")
        if uncond_emb.shape[0] < target_len:
            pad = uncond_emb[-1:].expand(target_len - uncond_emb.shape[0], -1)
            uncond_emb = torch.cat([uncond_emb, pad], dim=0)
        self._lang_emb_uncond = uncond_emb.contiguous()
        # Snapshot the cond embeds the standard set_prompt produced so
        # we can restore them after the uncond replay.
        self._lang_emb_cond = self._lang_emb.detach().clone()
        self._rl_current_prompt_text = prompt_text

        self._ensure_cfg_buffers()

        # Build the appropriate CFG pipeline based on whether batched
        # mode is active. Stage 0 (serial) when ``_batched`` is False;
        # Stage 3 (B=2 fused) when ``_batched`` is True.
        if self._batched:
            self._build_cfg_batched_pipeline(cfg["cfg_beta"])
        else:
            self._build_cfg_serial_pipeline(cfg["cfg_beta"])

        logger.info(
            "Set RL prompt: '%s' (cond_len=%d, uncond_len=%d, padded=%d, "
            "cfg_beta=%.2f, batched=%s)",
            prompt_text, target_len, uncond_len, target_len,
            cfg["cfg_beta"], self._batched)

    def _build_cfg_serial_pipeline(self, cfg_beta: float) -> None:
        """Build the Stage 0 serial CFG pipeline (per-chunk, B=1 graphs)."""
        from flash_rt.models.pi05.pipeline_thor_cfg import (
            Pi05ThorCFGPipeline)
        self._cfg_pipeline = Pi05ThorCFGPipeline(
            fvk,
            cfg_beta=cfg_beta,
            Sa=int(self.Sa),
            replay_siglip=lambda: self._siglip_graph.replay(),
            replay_enc_ae=lambda: self._enc_ae_graph.replay(),
            upload_cond_lang_emb=lambda: self._lang_emb.copy_(
                self._lang_emb_cond),
            upload_uncond_lang_emb=lambda: self._lang_emb.copy_(
                self._lang_emb_uncond),
            snapshot_noise=lambda: self._noise_R_snapshot.copy_(
                self._g_noise.view(-1)),
            restore_noise=lambda: self._g_noise.view(-1).copy_(
                self._noise_R_snapshot),
            snapshot_g_noise_to_v_cond=lambda: self._v_cond_buf.copy_(
                self._g_noise.view(-1)),
            snapshot_g_noise_to_v_uncond=lambda: self._v_uncond_buf.copy_(
                self._g_noise.view(-1)),
            zero_g_noise=lambda: self._g_noise.zero_(),
            g_noise_ptr=lambda: self._g_noise.data_ptr(),
            v_cond_ptr=lambda: self._v_cond_buf.data_ptr(),
            v_uncond_ptr=lambda: self._v_uncond_buf.data_ptr(),
            sync=torch.cuda.synchronize,
            stream_int=0,
        )

    def _build_cfg_batched_pipeline(self, cfg_beta: float) -> None:
        """Build the Stage 3 fused CFG pipeline (paper-correct per-step CFG).

        The fused-CFG pipeline runs as a SINGLE outer CUDA Graph: lang
        swap (×2) + SigLIP (×2) + encoder_b2 + decoder_b2 (with the
        per-step ``cfg_combine_into_residual_fp16`` + noise mirror
        baked into each denoise step). One ``forward()`` =
        ``outer_graph.replay()`` + final sync, matching RTX
        :meth:`Pi05CFGBatchedPipeline.forward` /
        ``self._graph.replay(...)``.

        Changing ``cfg_beta`` requires rebuilding the pipeline (the
        beta is baked into the captured cfg_combine kernel calls).
        """
        from flash_rt.models.pi05.pipeline_thor_cfg_batched import (
            Pi05ThorCFGBatchedPipeline)
        # Mark this beta on the inner enc_ae_b2 graph for backward
        # compatibility with any callers that still trigger the eager
        # ``run_pipeline`` fallback. The hot path uses the outer graph.
        self._enc_ae_graph_b2_cfg_beta = float(cfg_beta)
        self._enc_ae_graph_b2 = None
        self._capture_enc_ae_graph_b2()
        # Capture the full fused-CFG pipeline as one graph. When the
        # frontend is constructed with ``autotune > 0``, recapture N
        # times and keep the fastest — same parameterisation as the
        # B=1 path (``self.autotune``).
        self._cfg_b2_outer_graph = None
        if self.autotune > 0:
            self._autotune_cfg_b2_outer_graph(
                cfg_beta, n_trials=self.autotune, n_bench=10)
        else:
            self._capture_cfg_b2_outer_graph(cfg_beta)

        Se = self.Se
        Sa = self.Sa

        # B=1 SigLIP graph writes into ``_enc_x[:Se]`` (vision +
        # whatever is currently in ``_lang_emb``). For batched-CFG
        # we run it twice (once with cond_lang in _lang_emb, once
        # with uncond_lang) and copy each result into the respective
        # slot of ``_enc_x_b2``. This is identical to the serial
        # CFG pattern, just routed into B=2 buffers.
        def _siglip_for_cond():
            self._lang_emb.copy_(self._lang_emb_cond)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            self._enc_x_b2[0:Se].copy_(self._enc_x[:Se])

        def _siglip_for_uncond():
            self._lang_emb.copy_(self._lang_emb_uncond)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            self._enc_x_b2[Se:2 * Se].copy_(self._enc_x[:Se])

        def _seed_b2_noise_from_R():
            # Frontend-side R: a fresh normal draw of (Sa, 32) at
            # FP16. Both slots see the SAME R so the per-step CFG
            # mirror at the end of step 0 starts from identical x_t.
            #
            # numpy CPU RNG (not torch CUDA randn) so the bit pattern
            # matches the JAX frontend's noise draw at the same
            # ``np.random.seed`` — required for cross-backend cos
            # ≥ 0.999 at all β.
            R_np = np.random.randn(Sa, 32).astype(np.float16)
            R = torch.from_numpy(R_np).to('cuda', non_blocking=True)
            self._g_noise_b2.view(-1, 32)[0:Sa].copy_(R)
            self._g_noise_b2.view(-1, 32)[Sa:2 * Sa].copy_(R)

        self._cfg_pipeline = Pi05ThorCFGBatchedPipeline(
            fvk,
            cfg_beta=cfg_beta,
            Sa=int(self.Sa),
            replay_siglip_for_cond=_siglip_for_cond,
            replay_siglip_for_uncond=_siglip_for_uncond,
            replay_enc_ae_b2=lambda: self._enc_ae_graph_b2.replay(),
            seed_b2_noise_from_R=_seed_b2_noise_from_R,
            sync=torch.cuda.synchronize,
            stream_int=0,
            outer_graph_replay=lambda: self._cfg_b2_outer_graph.replay(),
        )

    def _infer_cfg(self, observation, debug=False):
        """CFG inference dispatcher.

        Routes to either the serial (Stage 0) per-chunk path —
        2× (siglip + enc_ae) + cfg_combine into ``g_noise`` — or
        the Stage 3 fused B=2 path: 2× SigLIP-into-different-slots,
        single B=2 enc_ae graph replay, then cfg_combine into
        ``g_noise_b2[0:Sa]``. The Pi05ThorCFGPipeline /
        Pi05ThorCFGBatchedPipeline subclass instance picks the
        right code path under the hood; the frontend just hands
        in observations + reads back the result.
        """
        t0 = time.perf_counter()
        nv = self.num_views

        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(
                    observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(
                    observation.get('wrist_image_right', img_list[-1]))

        def _to_np16(im):
            if isinstance(im, torch.Tensor):
                return im.to(dtype=torch.float16).cpu().numpy()
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)
        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
        self._img_buf.upload(images_np)

        # SigLIP needs to run once for the lazy-recal hook below; the
        # CFG pipeline will run it again per branch. The hook only
        # fires on the first infer call after construction, so this
        # extra siglip replay is amortized across the lifetime of the
        # frontend.
        if not self._real_data_calibrated:
            self._lang_emb.copy_(self._lang_emb_cond)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            self._recalibrate_with_real_data()
            self._real_data_calibrated = True
            # When batched, recalibrate reallocates ``_enc_calib_scales``
            # and ``_ae_calib_scales`` (new tensors) — the B=2 graph
            # captured by pointer therefore reads stale memory. The
            # torch caching allocator usually keeps the old contents
            # around long enough for the math to roughly track, but
            # this is fragile; recapture B=2 against the fresh scales
            # so the result matches B=1 cond-only deterministically.
            if self._batched and self._enc_ae_graph_b2 is not None:
                self._enc_ae_graph_b2 = None
                self._capture_enc_ae_graph_b2()
                # The outer fused-CFG graph also references the freed
                # scale buffers; recapture against the new scales,
                # honouring autotune.
                if self._cfg_b2_outer_graph is not None:
                    self._cfg_b2_outer_graph = None
                    if self.autotune > 0:
                        self._autotune_cfg_b2_outer_graph(
                            self._enc_ae_graph_b2_cfg_beta,
                            n_trials=self.autotune, n_bench=10)
                    else:
                        self._capture_cfg_b2_outer_graph(
                            self._enc_ae_graph_b2_cfg_beta)

        if self._batched:
            # Stage 3 fused B=2 CFG path. The pipeline's ``forward()``
            # is one outer-graph replay (lang swap + SigLIP×2 +
            # encoder_b2 + decoder_b2 with per-step CFG) + sync; result
            # lives in ``_g_noise_b2[0:Sa]``.
            self._cfg_pipeline.forward()
            raw_actions = self._g_noise_b2[0:self.Sa].float().cpu().numpy()
        else:
            # Stage 0 serial CFG path — pre-seed noise R and let the
            # pipeline run the full dual-branch + combine into
            # ``_g_noise``.
            #
            # numpy CPU RNG so the bit pattern matches the JAX frontend
            # under the same ``np.random.seed`` (cross-backend cos
            # contract — see ``_seed_b2_noise_from_R`` for the rationale).
            R_np = np.random.randn(self.Sa, 32).astype(np.float16)
            R = torch.from_numpy(R_np).to('cuda', non_blocking=True)
            self._g_noise.view(-1, 32).copy_(R)
            self._cfg_pipeline.run_pipeline()
            raw_actions = self._g_noise.float().cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]
        if debug:
            logger.info(
                "CFG raw[0,:5]: %s, latency: %.1f ms (beta=%.2f)",
                raw_actions[0, :5], latency_ms,
                self._cfg_pipeline.cfg_beta)
        return {"actions": robot_actions}

    def set_prompt(self, prompt_text):
        """Tokenize prompt, compute time conditioning, calibrate scales, capture graphs.

        When :meth:`set_rl_mode` has activated CFG inference and the
        caller supplies a text prompt, this routes to
        :meth:`_set_prompt_rl` which builds the cond + uncond pair and
        captures the graph at the padded length. The re-entry guard
        ``_in_rl_set_prompt`` prevents infinite recursion when
        ``_set_prompt_rl`` calls back into ``set_prompt`` with token IDs
        to drive the standard capture path.
        """
        if (self._rl_config is not None
                and not getattr(self, "_in_rl_set_prompt", False)
                and isinstance(prompt_text, str)):
            self._in_rl_set_prompt = True
            try:
                self._set_prompt_rl(prompt_text)
            finally:
                self._in_rl_set_prompt = False
            return

        S_sig = self.sig_S
        nv = self.num_views

        # ---- Tokenize ----
        if isinstance(prompt_text, (np.ndarray, list)):
            token_ids = np.asarray(prompt_text, dtype=np.int64)
            prompt_len = len(token_ids)
            embeds = F.embedding(
                torch.from_numpy(token_ids).long().cuda(), self.embedding_weight)
            embeds = embeds * float(embeds.shape[-1] ** 0.5)
        else:
            embeds, prompt_len = embed_prompt(prompt_text, self.embedding_weight, max_len=48)

        # Se must be EVEN for cuBLASLt FP8
        Se = S_sig + prompt_len
        if Se % 2 != 0:
            Se += 1
        self.Se = Se
        self.total_keys = Se + self.Sa

        # Stage 1.4 — build AttentionBackend. total_keys must be set first
        # because the encoder/decoder KV cache layer_stride is computed from
        # the runtime total_keys (the kernel treats Kc/Vc as a contiguous
        # [L, total_keys, HD] buffer, even though it was allocated with
        # total_keys_max along dim 1). Rebuilt on every set_prompt since
        # total_keys depends on prompt length.
        attn_scale = 1.0 / math.sqrt(float(self.HDe))
        layer_stride = int(self.total_keys) * int(self.HDe) * 2  # fp16 bytes
        kc_ptr = self._Kc.reshape(-1).data_ptr()
        vc_ptr = self._Vc.reshape(-1).data_ptr()
        self._attn = ThorFlashAttnBackend(
            make_pi05_attention_spec(
                num_views=self.num_views,
                enc_seq_max=self.Se,
                chunk_size=self.Sa,
            ),
            self._ctx,
            siglip_slots={
                "qkv": self._sig_qkv.data_ptr(),
                "O":   self._sig_attn.data_ptr(),
                "D":   self.sig_D,
            },
            encoder_slots={
                "Q_O":          self._enc_attn.data_ptr(),
                "Kc":           kc_ptr,
                "Vc":           vc_ptr,
                "logits":       self._enc_logits.data_ptr(),
                "layer_stride": layer_stride,
                "scale":        attn_scale,
            },
            decoder_slots={
                "Q_O":          self._ae_attn.data_ptr(),
                "Kc":           kc_ptr,
                "Vc":           vc_ptr,
                "logits":       self._ae_logits.data_ptr(),
                "layer_stride": layer_stride,
                "scale":        attn_scale,
            },
        )

        actual_lang = Se - S_sig
        if actual_lang > prompt_len:
            embeds = torch.cat([embeds, embeds[-1:]], dim=0)
        self._lang_emb = embeds
        self._S_lang = actual_lang

        # ---- RoPE tables ----
        self._enc_rope[:Se].copy_(
            torch.cat([self._kc_t[:Se, :, None],
                       self._ks_t[:Se, :, None]], dim=2).reshape(Se, 256))
        dec_start = Se
        self._dec_rope.copy_(
            torch.cat([self._kc_t[dec_start:dec_start + self.Sa, :, None],
                       self._ks_t[dec_start:dec_start + self.Sa, :, None]], dim=2)
            .reshape(self.Sa, 256))

        # ---- Time conditioning (precompute per-step AdaRMSNorm styles) ----
        Sa, Da, La = self.Sa, self.Da, self.La
        steps = 10; D3a = 3 * Da
        sa_all = torch.zeros(steps * La * Sa, D3a, dtype=fp16, device='cuda')
        sf_all = torch.zeros(steps * La * Sa, D3a, dtype=fp16, device='cuda')
        fs_all = torch.zeros(steps * Sa, D3a, dtype=fp16, device='cuda')

        time_embeds = []
        for step in range(steps):
            t_val = 1.0 - step / steps
            t_tensor = torch.tensor([t_val], device='cuda')
            fraction = torch.linspace(0, 1, Da // 2, device='cuda', dtype=torch.float64)
            period = 4e-3 * (4.0 / 4e-3) ** fraction
            scaling = 1.0 / period * 2 * math.pi
            sin_input = scaling * t_tensor.double()
            emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1).to(fp16)
            time_embeds.append(emb)

        for step in range(steps):
            te = time_embeds[step].unsqueeze(0)
            tmp = (te @ self._time_mlp_in_w.t() + self._time_mlp_in_b.unsqueeze(0)).float()
            tmp = (tmp * torch.sigmoid(tmp)).to(fp16)
            tmp2 = (tmp @ self._time_mlp_out_w.t() + self._time_mlp_out_b.unsqueeze(0)).float()
            tmp2 = (tmp2 * torch.sigmoid(tmp2)).to(fp16)
            time_emb = tmp2.expand(Sa, -1).contiguous()
            for layer in range(La):
                idx = (step * La + layer) * Sa
                sa_all[idx:idx + Sa] = (time_emb @ self._attn_mod_w[layer].t()
                                        + self._attn_mod_b[layer].unsqueeze(0))
                sf_all[idx:idx + Sa] = (time_emb @ self._ffn_mod_w[layer].t()
                                        + self._ffn_mod_b[layer].unsqueeze(0))
            fidx = step * Sa
            fs_all[fidx:fidx + Sa] = (time_emb @ self._final_mod_w.t()
                                      + self._final_mod_b.unsqueeze(0))

        self._sa_all = sa_all
        self._sf_all = sf_all
        self._fs_all = fs_all

        # ── B-tile style buffers when batched mode is active ──
        # Stage 2 of the Thor batched-CFG port: ``decoder_forward_b2``
        # passes ``M = B*Sa`` to the AdaRMSNorm fused kernels and walks
        # ``sa[step, layer, row]`` for ``row ∈ [0, B*Sa)``. Each
        # (step, layer) slice must be tiled B times along the row dim
        # so both samples see the same per-step style. Mirrors RTX
        # Bug 6 fix at pipeline_rtx_batched.py:195–220.
        if self._batched and self.B >= 2:
            B = self.B
            self._sa_all_b2 = sa_all.view(steps, La, Sa, D3a).repeat_interleave(
                B, dim=2).reshape(steps * La * B * Sa, D3a).contiguous()
            self._sf_all_b2 = sf_all.view(steps, La, Sa, D3a).repeat_interleave(
                B, dim=2).reshape(steps * La * B * Sa, D3a).contiguous()
            self._fs_all_b2 = fs_all.view(steps, Sa, D3a).repeat_interleave(
                B, dim=1).reshape(steps * B * Sa, D3a).contiguous()

        # ---- Capture SigLIP graph first (warmup writes enc_x with real PostLN output) ----
        self._capture_siglip_graph()

        # ---- Calibrate FP8 scales (using SigLIP warmup output in enc_x) ----
        self._calibrate(Se)

        # ---- Capture encoder+decoder graph ----
        if self.autotune > 0:
            self._autotune_enc_ae(n_trials=self.autotune, n_bench=10)
        else:
            self._capture_enc_ae_graph()

        self.graph_captured = True
        self.calibrated = True
        logger.info("set_prompt done: '%s' (%d tokens, Se=%d)", prompt_text, prompt_len, Se)

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------

    def _calibrate(self, Se):
        """Calibrate encoder + decoder FP8 activation scales."""
        Le = self.Le; La = self.La
        total_keys = self.total_keys

        # Try cache first
        cached = load_calibration(self._checkpoint_path, Se)
        if cached is not None:
            self._enc_calib_scales = torch.tensor(
                cached["enc_scales"], dtype=torch.float32, device='cuda')
            enc_ws = self._enc_w_dev.cpu().tolist()
            # f32 multiply to match production C float arithmetic (not f64!)
            self._enc_alpha_host = [
                float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
                for i in range(Le * 4)]
            self._ae_calib_scales = torch.tensor(
                cached["ae_scales"], dtype=torch.float32, device='cuda')
            logger.info("Calibration loaded from cache (enc=%d, ae=%d scales)",
                        Le * 4, La * 4)
            return

        HDe = self.HDe; NHe = self.NHe; De = self.De; He = self.He

        # Build encoder weight/buffer dicts for calibration
        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights = {
            'qkv_w':   [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':     [w.data_ptr() for w in self._enc_o_w],
            'gate_w':  [w.data_ptr() for w in self._enc_gu_w],
            'down_w':  [w.data_ptr() for w in self._enc_d_w],
            'rope':    self._enc_rope.data_ptr(),
            'Kc':      self._Kc.reshape(-1).data_ptr(),
            'Vc':      self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Encoder calibration — scratch buffers allocated by caller
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch = torch.zeros(Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')
        enc_bufs['norm_scratch'] = _norm_scratch.data_ptr()
        enc_bufs['x_scratch'] = _x_scratch.data_ptr()
        enc_bufs['calib_buf'] = _calib_buf.data_ptr()
        enc_bufs['d_scale'] = _d_scale.data_ptr()
        enc_bufs['fp8_scratch'] = _fp8_scratch.data_ptr()
        enc_bufs['ones'] = _ones.data_ptr()

        self._Kc.zero_(); self._Vc.zero_()
        enc_max = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        encoder_forward_calibrate(
            self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
            enc_max.data_ptr(), stream=0)

        self._enc_calib_scales = enc_max
        enc_ws = self._enc_w_dev.cpu().tolist()
        # f32 multiply to match production C float arithmetic
        self._enc_alpha_host = [
            float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]
        logger.info("Encoder calibrated: %d scales", Le * 4)

        # Decoder calibration
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            'ain_w':     self._ain_w.data_ptr(),
            'ain_b':     self._ain_b.data_ptr(),
            'sa':        self._sa_all.data_ptr(),
            'qw':        self._dec_qkv_flat.data_ptr(),
            'Kc':        self._Kc.reshape(-1).data_ptr(),
            'Vc':        self._Vc.reshape(-1).data_ptr(),
            'ow':        self._dec_o_flat.data_ptr(),
            'sf':        self._sf_all.data_ptr(),
            'gw':        self._dec_gu_flat.data_ptr(),
            'dw':        self._dec_d_flat.data_ptr(),
            'aow':       self._aow.data_ptr(),
            'aob':       self._aob.data_ptr(),
            'fs':        self._fs_all.data_ptr(),
            'rope':      self._dec_rope.data_ptr(),
            'w_scales':  self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': self.La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        # Decoder scratch buffers
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        _ae_calib_buf = torch.zeros(self.La * 4, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(Sa * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(Sa * max(Da, Ha), dtype=torch.uint8, device='cuda')
        ae_bufs['calib_buf'] = _ae_calib_buf.data_ptr()
        ae_bufs['d_scale'] = _ae_d_scale.data_ptr()
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.data_ptr()
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.data_ptr()

        self._g_noise.normal_()
        ae_max = torch.zeros(self.La * 4, dtype=torch.float32, device='cuda')
        decoder_forward_calibrate(
            self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
            ae_max.data_ptr(), stream=0)

        self._ae_calib_scales = ae_max
        logger.info("Decoder calibrated: %d scales", La * 4)

        # Save to cache
        try:
            save_calibration(
                checkpoint_path=self._checkpoint_path,
                Se=Se,
                enc_scales=self._enc_calib_scales.cpu().tolist(),
                enc_alpha=self._enc_alpha_host,
                ae_scales=self._ae_calib_scales.cpu().tolist(),
                enc_w_scales=enc_ws,
            )
        except Exception as e:
            logger.warning("Failed to save calibration cache: %s", e)

    # -----------------------------------------------------------------------
    # Patch embedding (SigLIP input)
    # -----------------------------------------------------------------------

    def _patch_embed_ops(self, stream_int):
        """im2col -> GEMM -> bias+pos.  Output in self._sig_x."""
        S_sig, D_sig = self.sig_S, self.sig_D
        fvk.patch_im2col(self._img_buf.ptr.value, self._patches_buf.ptr.value,
                         self.num_views, stream_int)
        self._gemm.fp16_nn(self._patches_buf.ptr.value, self._pe_w.ptr.value,
                           self._sig_x.data_ptr(), S_sig, D_sig, 588, stream_int)
        fvk.patch_embed_bias_pos(self._sig_x.data_ptr(), self._pe_b.ptr.value,
                                 self._pos_emb.ptr.value, S_sig, D_sig, 256, stream_int)

    # -----------------------------------------------------------------------
    # PostLN + projection
    # -----------------------------------------------------------------------

    def _postln_project_ops(self, stream_int):
        """LayerNorm + projection + lang concat.  Writes into self._enc_x."""
        S_sig = self.sig_S; D_sig = self.sig_D; De = self.De
        postln_bufs = {
            'x_sig':   self._sig_x.data_ptr(),
            'enc_x':   self._enc_x.data_ptr(),
            'scratch': self._postln_scratch.data_ptr(),
        }
        postln_weights = {
            'ln_w':    self._postln_w.data_ptr(),
            'ln_b':    self._postln_b.data_ptr(),
            'proj_w':  self._proj_w.data_ptr(),
            'proj_b':  self._proj_b.data_ptr(),
            'lang_emb': self._lang_emb.data_ptr(),
        }
        postln_dims = {
            'S_sig': S_sig, 'D_sig': D_sig,
            'D_enc': De, 'S_lang': self._S_lang,
        }
        postln_project(self._gemm, fvk, postln_bufs, postln_weights,
                       postln_dims, stream=stream_int)

    # -----------------------------------------------------------------------
    # CUDA graph capture
    # -----------------------------------------------------------------------

    def _capture_siglip_graph(self):
        """Capture patch_embed + SigLIP + PostLN as CUDA graph."""
        # Warmup: zero SigLIP input to match production (g_xs = zeros).
        # patch_embed runs but its output is zeroed before SigLIP to avoid
        # inf from bias accumulation through 27 layers.
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)
        for _ in range(3):
            self._patch_embed_ops(0)
            self._sig_x.zero_()  # match production: SigLIP sees zeros
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                           self._sig_dims, stream=0, attn=self._attn)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._siglip_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._siglip_graph.capture_begin()
            self._patch_embed_ops(s_int)
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                           self._sig_dims, stream=s_int, attn=self._attn)
            self._postln_project_ops(s_int)
            self._siglip_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("SigLIP CUDA graph captured (S=%d)", self.sig_S)

    def _capture_enc_ae_graph(self):
        """Capture encoder + decoder (static FP8) as CUDA graph."""
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha

        # Build dicts for encoder_forward
        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc':        self._Kc.reshape(-1).data_ptr(),
            'Vc':        self._Vc.reshape(-1).data_ptr(),
            'act_scales':  self._enc_calib_scales.data_ptr(),  # device float ptr
            'alpha_host':  self._enc_alpha_host,              # host float list [L*4]
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Build dicts for decoder_forward
        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        # Warmup
        for _ in range(3):
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=0, attn=self._attn)
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                            ae_dims, stream=0, attn=self._attn)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._enc_ae_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._enc_ae_graph.capture_begin()
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=s_int, attn=self._attn)
            decoder_forward(self._ctx, fvk, ae_bufs, ae_weights,
                            ae_dims, stream=s_int, attn=self._attn)
            self._enc_ae_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("Enc+AE CUDA graph captured (Se=%d)", Se)

    def _capture_enc_ae_graph_b2(self):
        """Capture B=N encoder + decoder as a CUDA graph (Stage 2).

        Mirror of :meth:`_capture_enc_ae_graph` for the batched path:
        wires up the ``_b2``-suffixed buffers, the per-sample KV-cache
        slabs, and the B-tiled style buffers, then drives a warmup +
        capture using
        :func:`flash_rt.hardware.thor.shared_primitives_batched.encoder_forward_b2`
        and :func:`flash_rt.models.pi05.pipeline_thor_batched.decoder_forward_b2`.

        Preconditions:
          * ``self._batched`` is True and ``self._alloc_b2_buffers``
            has run.
          * ``self._sa_all_b2`` / ``_sf_all_b2`` / ``_fs_all_b2`` have
            been built by ``set_prompt`` (B-tiled).
          * Encoder + decoder FP8 scales (``_enc_calib_scales``,
            ``_ae_calib_scales``) have been calibrated. Stage 2 reuses
            the B=1 scales — Pi05BatchedPipeline's parent contract is
            "B=1 calibration transfers to B=N". Stage 3 CFG-batched
            will override with a joint cond+uncond pass.
        """
        from flash_rt.hardware.thor.shared_primitives_batched import (
            encoder_forward_b2)
        from flash_rt.models.pi05.pipeline_thor_batched import (
            decoder_forward_b2)

        B = self.B
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha

        # Per-sample KV slab device pointers (one per b).
        kc_b2 = [self._Kc_b2[b].view(-1).data_ptr() for b in range(B)]
        vc_b2 = [self._Vc_b2[b].view(-1).data_ptr() for b in range(B)]

        enc_bufs_b2 = {
            'x':       self._enc_x_b2.data_ptr(),
            'x_fp8':   self._enc_x_fp8_b2.data_ptr(),
            'qkv':     self._enc_qkv_buf_b2.data_ptr(),
            'logits':  self._enc_logits_b2.data_ptr(),
            'attn_out': self._enc_attn_b2.data_ptr(),
            'o_fp8':   self._enc_o_fp8_b2.data_ptr(),
            'gate':    self._enc_gate_b2.data_ptr(),
            'hid_fp8': self._enc_hid_fp8_b2.data_ptr(),
            'fg':      self._enc_fg_b2.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights_b2 = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc_b2':     kc_b2,
            'Vc_b2':     vc_b2,
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims_b2 = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs_b2 = {
            'noise':   self._g_noise_b2.data_ptr(),
            'x':       self._ae_x_b2.data_ptr(),
            'xn':      self._ae_xn_b2.data_ptr(),
            'gate':    self._ae_gate_b2.data_ptr(),
            'qkv':     self._ae_qkv_b2.data_ptr(),
            'logits':  self._ae_logits_b2.data_ptr(),
            'attn_out': self._ae_attn_b2.data_ptr(),
            'fg':      self._ae_fg_b2.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8_b2.data_ptr(),
            'hid_fp8': self._ae_hid_fp8_b2.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8_b2.data_ptr(),
            # Per-step velocity scratch, only consumed when cfg_beta is
            # set (CFG-batched capture path); harmless otherwise.
            'v_b2':    self._v_b2.data_ptr(),
        }
        ae_weights_b2 = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all_b2.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc_b2':      kc_b2,
            'Vc_b2':      vc_b2,
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all_b2.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'fs':         self._fs_all_b2.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims_b2 = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        # When the CFG-batched pipeline is active, bake the per-step
        # CFG combine + noise mirror into the captured graph (matches
        # RTX). Non-CFG batched paths leave this None and use the
        # standard independent-slot integration.
        cfg_beta = self._enc_ae_graph_b2_cfg_beta

        def _b2_run(stream):
            for b in range(B):
                self._Kc_b2[b].zero_(); self._Vc_b2[b].zero_()
            encoder_forward_b2(self._gemm, fvk, enc_bufs_b2, enc_weights_b2,
                                enc_dims_b2, stream=stream, B=B)
            decoder_forward_b2(self._ctx, fvk, ae_bufs_b2, ae_weights_b2,
                                ae_dims_b2, stream=stream, B=B,
                                cfg_beta=cfg_beta)

        # Warmup
        for _ in range(3):
            _b2_run(0)
        torch.cuda.synchronize()

        # Capture
        cstream = torch.cuda.Stream()
        self._enc_ae_graph_b2 = torch.cuda.CUDAGraph()
        s_int = cstream.cuda_stream
        with torch.cuda.stream(cstream):
            self._enc_ae_graph_b2.capture_begin()
            _b2_run(s_int)
            self._enc_ae_graph_b2.capture_end()
        torch.cuda.synchronize()
        logger.info(
            "Enc+AE CUDA graph captured at B=%d (Se=%d, total_keys=%d)",
            B, Se, total_keys)

    def _capture_cfg_b2_outer_graph(self, cfg_beta: float) -> None:
        """Capture the entire fused-CFG B=2 pipeline as one outer graph.

        Replaces the multi-call Python orchestration in
        :class:`Pi05ThorCFGBatchedPipeline.run_pipeline` (lang swap +
        SigLIP×2 + enc_ae_b2) with a single ``CUDAGraph.replay()``,
        matching RTX
        :meth:`Pi05CFGBatchedPipeline.forward` /
        ``self._graph.replay(...)``.

        Captures (in order, on a fresh stream):
          1. ``lang_emb := lang_emb_cond``        (D2D)
          2. ``patch_embed + siglip + postln``    (writes _enc_x with cond)
          3. ``_enc_x → _enc_x_b2[0:Se]``         (D2D snapshot)
          4. ``lang_emb := lang_emb_uncond``      (D2D)
          5. ``patch_embed + siglip + postln``    (writes _enc_x with uncond)
          6. ``_enc_x → _enc_x_b2[Se:2*Se]``      (D2D snapshot)
          7. ``encoder_forward_b2 + decoder_forward_b2(cfg_beta=...)``
             (the latter carries the 10-step decoder loop AND the
             per-step cfg_combine + noise mirror).

        Pre-capture inputs (frontend stages these per inference, before
        ``forward()``):
          * ``_img_buf``     — observation images (H2D)
          * ``_g_noise_b2``  — fresh noise R replicated into both slots
        After replay: ``_g_noise_b2[0:Sa]`` holds the guided action chunk.

        Requires ``self._lang_emb_cond`` / ``_lang_emb_uncond`` to be
        device tensors (set up by ``_set_prompt_rl``).
        """
        from flash_rt.hardware.thor.shared_primitives_batched import (
            encoder_forward_b2)
        from flash_rt.models.pi05.pipeline_thor_batched import (
            decoder_forward_b2)

        B = self.B
        Se = self.Se
        De = self.De
        total_keys = self.total_keys
        Le = self.Le; La = self.La; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha

        kc_b2 = [self._Kc_b2[b].view(-1).data_ptr() for b in range(B)]
        vc_b2 = [self._Vc_b2[b].view(-1).data_ptr() for b in range(B)]

        enc_bufs_b2 = {
            'x':       self._enc_x_b2.data_ptr(),
            'x_fp8':   self._enc_x_fp8_b2.data_ptr(),
            'qkv':     self._enc_qkv_buf_b2.data_ptr(),
            'logits':  self._enc_logits_b2.data_ptr(),
            'attn_out': self._enc_attn_b2.data_ptr(),
            'o_fp8':   self._enc_o_fp8_b2.data_ptr(),
            'gate':    self._enc_gate_b2.data_ptr(),
            'hid_fp8': self._enc_hid_fp8_b2.data_ptr(),
            'fg':      self._enc_fg_b2.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights_b2 = {
            'qkv_w':     [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':       [w.data_ptr() for w in self._enc_o_w],
            'gate_w':    [w.data_ptr() for w in self._enc_gu_w],
            'down_w':    [w.data_ptr() for w in self._enc_d_w],
            'rope':      self._enc_rope.data_ptr(),
            'Kc_b2':     kc_b2,
            'Vc_b2':     vc_b2,
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims_b2 = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        ae_bufs_b2 = {
            'noise':   self._g_noise_b2.data_ptr(),
            'x':       self._ae_x_b2.data_ptr(),
            'xn':      self._ae_xn_b2.data_ptr(),
            'gate':    self._ae_gate_b2.data_ptr(),
            'qkv':     self._ae_qkv_b2.data_ptr(),
            'logits':  self._ae_logits_b2.data_ptr(),
            'attn_out': self._ae_attn_b2.data_ptr(),
            'fg':      self._ae_fg_b2.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8_b2.data_ptr(),
            'hid_fp8': self._ae_hid_fp8_b2.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8_b2.data_ptr(),
            'v_b2':    self._v_b2.data_ptr(),
        }
        ae_weights_b2 = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all_b2.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc_b2':      kc_b2,
            'Vc_b2':      vc_b2,
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all_b2.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'fs':         self._fs_all_b2.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
            'act_scales': self._ae_calib_scales.data_ptr(),
        }
        ae_dims_b2 = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        lang_nbytes = self._S_lang * De * 2  # fp16
        enc_x_slot_bytes = Se * De * 2

        def _outer_run(stream_int):
            # Cond branch
            fvk.gpu_copy(
                self._lang_emb.data_ptr(),
                self._lang_emb_cond.data_ptr(),
                lang_nbytes, stream_int)
            self._patch_embed_ops(stream_int)
            siglip_forward(self._gemm, fvk, self._sig_bufs,
                           self._sig_weights, self._sig_dims,
                           stream=stream_int, attn=self._attn)
            self._postln_project_ops(stream_int)
            fvk.gpu_copy(
                self._enc_x_b2.data_ptr(),
                self._enc_x.data_ptr(),
                enc_x_slot_bytes, stream_int)
            # Uncond branch (overwrites _sig_x / _enc_x; we already
            # snapshot'd cond into _enc_x_b2[0]).
            fvk.gpu_copy(
                self._lang_emb.data_ptr(),
                self._lang_emb_uncond.data_ptr(),
                lang_nbytes, stream_int)
            self._patch_embed_ops(stream_int)
            siglip_forward(self._gemm, fvk, self._sig_bufs,
                           self._sig_weights, self._sig_dims,
                           stream=stream_int, attn=self._attn)
            self._postln_project_ops(stream_int)
            fvk.gpu_copy(
                self._enc_x_b2.data_ptr() + enc_x_slot_bytes,
                self._enc_x.data_ptr(),
                enc_x_slot_bytes, stream_int)
            # Encoder + Decoder at B=2 with per-step CFG.
            for b in range(B):
                self._Kc_b2[b].zero_(); self._Vc_b2[b].zero_()
            encoder_forward_b2(
                self._gemm, fvk, enc_bufs_b2, enc_weights_b2,
                enc_dims_b2, stream=stream_int, B=B)
            decoder_forward_b2(
                self._ctx, fvk, ae_bufs_b2, ae_weights_b2,
                ae_dims_b2, stream=stream_int, B=B,
                cfg_beta=float(cfg_beta))

        # Warmup so cuBLAS / cuDNN selects tactics + workspace before
        # the capture freezes their kernel choice.
        for _ in range(3):
            _outer_run(0)
        torch.cuda.synchronize()

        cstream = torch.cuda.Stream()
        self._cfg_b2_outer_graph = torch.cuda.CUDAGraph()
        s_int = cstream.cuda_stream
        with torch.cuda.stream(cstream):
            self._cfg_b2_outer_graph.capture_begin()
            _outer_run(s_int)
            self._cfg_b2_outer_graph.capture_end()
        torch.cuda.synchronize()
        logger.info(
            "CFG-B=2 outer CUDA graph captured (Se=%d, S_lang=%d, beta=%.2f)",
            Se, self._S_lang, cfg_beta)

    def _autotune_cfg_b2_outer_graph(self, cfg_beta: float,
                                       n_trials: int = 3,
                                       n_bench: int = 10) -> None:
        """Capture the fused-CFG outer graph N times, keep the fastest.

        Same rationale as :meth:`_autotune_enc_ae` (B=1 path): cuBLASLt
        heuristic state and CUDA graph instantiation are not
        deterministic across captures on Thor, especially under
        process-state variation between backends. Recapturing N times
        and benching each lets each backend converge on its
        locally-optimal schedule instead of being stuck with whatever
        the first heuristic call returned.

        Driven by the frontend's ``self.autotune`` knob, parameterised
        identically to the B=1 path: ``autotune=N > 0`` runs N trials
        for the B=2 outer graph too; ``autotune=0`` does a single
        capture (current behaviour).
        """
        candidates = []
        for trial in range(n_trials):
            self._cfg_b2_outer_graph = None
            self._capture_cfg_b2_outer_graph(cfg_beta)
            graph = self._cfg_b2_outer_graph

            # Benchmark this capture: outer replay + sync. The outer
            # graph already contains lang swap + SigLIP×2 + enc_ae_b2,
            # so we don't need any pre-bench priming.
            latencies = []
            for _ in range(n_bench):
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            candidates.append((p50, graph))
            logger.info("  [B2 autotune] trial %d/%d: p50=%.2f ms",
                        trial + 1, n_trials, p50)

        best_p50, best_graph = min(candidates, key=lambda x: x[0])
        self._cfg_b2_outer_graph = best_graph
        # Drop refs to losers so torch frees their CUDAGraph state.
        for p50, g in candidates:
            if g is not best_graph:
                del g
        logger.info("  [B2 autotune] kept best: p50=%.2f ms (of %d trials)",
                    best_p50, n_trials)

    def set_batched_mode(self, *, enable: bool = True,
                          batch_size: int = 2) -> None:
        """Switch the frontend to / from B=N batched inference (Stage 2).

        When ``enable=True``, allocates the ``_b2``-suffixed buffer
        set, B-tiles the AdaRMSNorm style buffers, and (after the
        next :meth:`set_prompt`) captures a separate
        ``_enc_ae_graph_b2`` at B*Seq shapes. The B=1 hot path is
        preserved and remains available — this method only flips a
        flag; the :meth:`infer_batch` entry point uses the b2
        buffers, while :meth:`infer` continues to use the B=1 ones.

        When ``enable=False``, drops the b2 graph reference so a
        future re-enable will re-capture, but keeps the b2 buffers
        allocated (cheap to keep around; expensive to re-alloc).

        Args:
            enable: ``True`` activates batched mode; ``False`` clears
                the active b2 graph reference.
            batch_size: Number of fused samples. Stage 2 supports
                ``batch_size=2``; future stages may extend.

        Mirrors :meth:`flash_rt.frontends.torch.pi05_rtx.Pi05TorchFrontendRtx.set_batched_mode`
        (line 1140) at the API level.
        """
        if not enable:
            self._batched = False
            self._enc_ae_graph_b2 = None
            return

        if batch_size < 2:
            raise ValueError(
                f"set_batched_mode requires batch_size >= 2; "
                f"got {batch_size}. Use the standard infer() for B=1.")

        # Lazy alloc on first enable (or on B change).
        if self._Kc_b2 is None or self.B != batch_size:
            self._alloc_b2_buffers(B=batch_size)

        self._batched = True
        self.B = int(batch_size)
        # Force re-capture: next set_prompt or explicit
        # _capture_enc_ae_graph_b2() will rebuild the b2 graph against
        # the new prompt length.
        self._enc_ae_graph_b2 = None
        logger.info(
            "Batched mode ENABLED (B=%d). Call set_prompt() then "
            "_capture_enc_ae_graph_b2() to capture the b2 graph.",
            batch_size)

    def infer_batch(self, observations):
        """Run B=N batched inference on a list of observations.

        Stage 2 of the Thor batched-CFG port. Each observation must
        be a dict with the same keys as :meth:`infer`'s ``observation``
        argument (``image``, optional ``wrist_image``, optional
        ``wrist_image_right``). The returned list has one ``actions``
        ndarray per slot.

        For Stage 2 generic batched, all samples share the same
        SigLIP B=1 graph (we replay it once and copy the vision tokens
        into both ``_enc_x_b2`` slots). The two language slots receive
        each sample's prompt embeddings — but Stage 2 uses the
        currently-set prompt for both, so callers wanting two
        different prompts should use Stage 3's
        Pi05ThorCFGBatchedPipeline (which manages cond/uncond
        explicitly).
        """
        if not self._batched:
            raise RuntimeError(
                "set_batched_mode(enable=True) must be called first")
        if self._enc_ae_graph_b2 is None:
            self._capture_enc_ae_graph_b2()

        if isinstance(observations, dict):
            observations = [observations] * self.B
        if len(observations) != self.B:
            raise ValueError(
                f"infer_batch expected {self.B} observations; "
                f"got {len(observations)}")

        nv = self.num_views
        Se = self.Se
        S_lang = self._S_lang
        S_sig = self.sig_S
        De = self.De

        t0 = time.perf_counter()

        # ── Run SigLIP per slot (B=1 graph, B times) and stage into _enc_x_b2 ──
        # The B=1 _siglip_graph writes into _enc_x[:Se]. We then
        # memcpy into _enc_x_b2[b*Se : b*Se+Se]. For homogeneous
        # observations this could be optimized to one SigLIP run +
        # broadcast; Stage 2 keeps the simple correct path.
        for b, obs in enumerate(observations):
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(
                        obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(
                        obs.get('wrist_image_right', img_list[-1]))

            def _to_np16(im):
                if isinstance(im, torch.Tensor):
                    return im.to(dtype=torch.float16).cpu().numpy()
                if im.dtype == np.float16:
                    return im
                return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)
            images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()
            torch.cuda.synchronize()
            # _enc_x[:Se] now has vision + lang for this sample.
            self._enc_x_b2[b * Se : (b + 1) * Se].copy_(self._enc_x[:Se])

        # ── Seed noise per slot ──
        self._g_noise_b2.normal_()

        # ── Run B=N enc+ae graph ──
        self._enc_ae_graph_b2.replay()
        torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        # ── Unpack per-slot actions ──
        results = []
        for b in range(self.B):
            raw = self._g_noise_b2[b * self.Sa : (b + 1) * self.Sa
                                    ].float().cpu().numpy()
            unnorm = unnormalize_actions(raw, self.norm_stats)
            results.append(
                {"actions": unnorm[:, :LIBERO_ACTION_DIM]})
        return results

    # -----------------------------------------------------------------------
    # Autotune: try multiple graph captures, keep the fastest
    # -----------------------------------------------------------------------

    def _autotune_enc_ae(self, n_trials=5, n_bench=10):
        """Autotune Enc+AE graph: recapture until fast schedule is found.

        CUDA Graph instantiation is non-deterministic on Thor — the same kernels
        can produce different schedules with ~2ms variance. This recaptures the
        graph until a fast schedule is obtained or max trials are exhausted.
        The LAST captured graph is always used (no stale references).

        Benchmark replicates real infer() flow: SigLIP replay → noise → Enc+AE,
        because SigLIP changes L2 cache state which affects Enc+AE performance.

        Called once per set_prompt(). Adds ~1-5s to startup.
        """
        import ctypes
        _crt = ctypes.CDLL("libcudart.so")

        def _make_ev():
            e = ctypes.c_void_p()
            _crt.cudaEventCreate(ctypes.byref(e))
            return e

        def _elapsed(a, b):
            ms = ctypes.c_float()
            _crt.cudaEventElapsedTime(ctypes.byref(ms), a, b)
            return ms.value

        # Prepare image for SigLIP (replicate real infer flow)
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)

        logger.info("Autotune: up to %d trials for best Enc+AE graph...", n_trials)

        for trial in range(n_trials):
            self._capture_enc_ae_graph()

            # Benchmark with SigLIP in front (replicates real infer flow)
            latencies = []
            for _ in range(n_bench):
                self._siglip_graph.replay()
                e0, e1 = _make_ev(), _make_ev()
                _crt.cudaEventRecord(e0, ctypes.c_void_p(0))
                self._g_noise.normal_()
                self._enc_ae_graph.replay()
                _crt.cudaEventRecord(e1, ctypes.c_void_p(0))
                torch.cuda.synchronize()
                latencies.append(_elapsed(e0, e1))

            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            logger.info("  Trial %d: %.2f ms", trial, p50)

            # Accept if fast enough (< 38.5ms = within fast regime)
            if p50 < 38.5:
                logger.info("Autotune done: Enc+AE = %.2f ms (trial %d)", p50, trial)
                return

        logger.info("Autotune done: Enc+AE = %.2f ms (best of %d)", p50, n_trials)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def infer(self, observation, debug=False):
        """Run inference: images -> CUDA graph replay -> actions.

        Args:
            observation: dict with 'image' and 'wrist_image' (or 'images' list).
                         Each image is (224,224,3) uint8 or float16 numpy.
        Returns:
            {"actions": np.ndarray}  shape (Sa, LIBERO_ACTION_DIM)
        """
        if self._rl_config is not None:
            return self._infer_cfg(observation, debug)
        t0 = time.perf_counter()
        nv = self.num_views

        # ---- Collect and upload images ----
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image']]
            if nv >= 2:
                img_list.append(
                    observation.get('wrist_image', observation['image']))
            if nv >= 3:
                img_list.append(
                    observation.get('wrist_image_right', img_list[-1]))

        def _to_np16(im):
            if isinstance(im, torch.Tensor):
                return im.to(dtype=torch.float16).cpu().numpy()
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
        self._img_buf.upload(images_np)

        # ---- Graph 1: SigLIP + PostLN ----
        self._siglip_graph.replay()

        # ---- Lazy real-data recalibration on first call ----
        if not self._real_data_calibrated:
            torch.cuda.synchronize()
            self._recalibrate_with_real_data()
            self._real_data_calibrated = True

        # ---- Graph 2: Encoder + Decoder ----
        # numpy CPU RNG so the bit pattern matches the JAX frontend's
        # standard infer path (cross-backend determinism + lets the RL
        # CFG path's β=1.0 output collapse cleanly to this cond-only
        # baseline — both use the same numpy seed → same R).
        R_np = np.random.randn(self.Sa, 32).astype(np.float16)
        R = torch.from_numpy(R_np).to('cuda', non_blocking=True)
        self._g_noise.view(-1, 32).copy_(R)
        self._enc_ae_graph.replay()
        torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        # ---- Post-process ----
        raw_actions = self._g_noise.float().cpu().numpy()
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :LIBERO_ACTION_DIM]

        if debug:
            logger.info("Raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("Robot actions[0]: %s", robot_actions[0])
            logger.info("Latency: %.1f ms", latency_ms)

        return {"actions": robot_actions}

    # -----------------------------------------------------------------------
    # Real-data recalibration (called once after first real image)
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration API (Thor).

        N=1: runs the existing implicit recalibration path via ``infer``
        (bit-equal to the legacy "first call auto-calibrates" behaviour).

        N>=2: per-sample amax collection on both encoder and decoder
        calibrate passes, reduced via ``np.percentile(axis=0)``. The CUDA
        graph is recaptured once at the end so the caller can run
        ``infer`` immediately afterwards without the lazy-calibrate
        branch firing again.
        """
        if isinstance(observations, dict):
            obs_list = [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = list(observations)
        if max_samples is not None:
            obs_list = obs_list[:max_samples]
        n = len(obs_list)
        if n == 0:
            raise ValueError("observations must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        if n == 1:
            # Legacy implicit-calibrate path: one infer() flips
            # _real_data_calibrated via the lazy branch in infer().
            from flash_rt.core.calibration_api import implicit_calibrate
            implicit_calibrate(
                self, obs_list,
                percentile=percentile, max_samples=None, verbose=verbose,
            )
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """Thor frontends do not yet surface a structured PrecisionSpec.

        The scale buffers are populated in-device by the calibration
        kernels. A future release will add a snapshot path parallel to
        the RTX frontends.
        """
        return None

    def _recalibrate_with_real_data(self):
        """Recalibrate using real SigLIP output, then recapture enc+ae graph."""
        Se = self.Se; Le = self.Le; La = self.La
        total_keys = self.total_keys
        De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe

        enc_bufs = {
            'x':       self._enc_x.data_ptr(),
            'x_fp8':   self._enc_x_fp8.data_ptr(),
            'qkv':     self._enc_qkv_buf.data_ptr(),
            'logits':  self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8':   self._enc_o_fp8.data_ptr(),
            'gate':    self._enc_gate.data_ptr(),
            'hidden':  self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg':      self._enc_fg.data_ptr(),
            'ctx':     self._ctx,
        }
        enc_weights = {
            'qkv_w':   [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':     [w.data_ptr() for w in self._enc_o_w],
            'gate_w':  [w.data_ptr() for w in self._enc_gu_w],
            'down_w':  [w.data_ptr() for w in self._enc_d_w],
            'rope':    self._enc_rope.data_ptr(),
            'Kc':      self._Kc.reshape(-1).data_ptr(),
            'Vc':      self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Scratch buffers for recalibration
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch = torch.zeros(Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')
        enc_bufs['norm_scratch'] = _norm_scratch.data_ptr()
        enc_bufs['x_scratch'] = _x_scratch.data_ptr()
        enc_bufs['calib_buf'] = _calib_buf.data_ptr()
        enc_bufs['d_scale'] = _d_scale.data_ptr()
        enc_bufs['fp8_scratch'] = _fp8_scratch.data_ptr()
        enc_bufs['ones'] = _ones.data_ptr()

        self._enc_calib_scales.zero_()
        self._Kc.zero_(); self._Vc.zero_()
        encoder_forward_calibrate(
            self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
            self._enc_calib_scales.data_ptr(), stream=0)
        torch.cuda.synchronize()

        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]

        # Recalibrate decoder
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        ae_bufs = {
            'noise':   self._g_noise.data_ptr(),
            'x':       self._ae_x.data_ptr(),
            'xn':      self._ae_xn.data_ptr(),
            'gate':    self._ae_gate.data_ptr(),
            'qkv':     self._ae_qkv.data_ptr(),
            'logits':  self._ae_logits.data_ptr(),
            'attn_out': self._ae_attn.data_ptr(),
            'hid':     self._ae_hid.data_ptr(),
            'fg':      self._ae_fg.data_ptr(),
            'xn_fp8':  self._ae_xn_fp8.data_ptr(),
            'hid_fp8': self._ae_hid_fp8.data_ptr(),
            'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': self.La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        _ae_calib_buf = torch.zeros(self.La * 4, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(Sa * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(Sa * max(Da, Ha), dtype=torch.uint8, device='cuda')
        ae_bufs['calib_buf'] = _ae_calib_buf.data_ptr()
        ae_bufs['d_scale'] = _ae_d_scale.data_ptr()
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.data_ptr()
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.data_ptr()

        self._ae_calib_scales.zero_()
        self._g_noise.normal_()
        decoder_forward_calibrate(
            self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
            self._ae_calib_scales.data_ptr(), stream=0)
        torch.cuda.synchronize()

        # Recapture graph with updated scales
        self._capture_enc_ae_graph()
        logger.info("Recalibrated with real data + graph recaptured")

    # -----------------------------------------------------------------------
    # Multi-sample dataset calibration (N>=2)
    # -----------------------------------------------------------------------

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        """Per-sample amax across N observations, reduced via percentile.

        Reuses the existing ``encoder_forward_calibrate`` /
        ``decoder_forward_calibrate`` kernels (no new CUDA code). Each
        sample writes amax into ``_enc_calib_scales[Le*4]`` and
        ``_ae_calib_scales[La*4]``; we snapshot both to CPU per sample,
        then ``accumulate_amax(percentile)`` column-wise, upload the
        reduced scales once, recompute ``_enc_alpha_host`` (the cuBLASLt
        alpha-fold list), and recapture the encoder+decoder CUDA graph.

        Mirrors :meth:`Pi05TorchFrontendRtx._calibrate_multi_frame` in
        shape so user code is portable, but the underlying scale
        buffers differ (Thor exposes single device tensors per stage
        rather than a per-tensor dict).
        """
        from flash_rt.core.calibration import (
            accumulate_amax,
            check_scale_ceiling,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Pi0.5 Thor: calibrating FP8 across %d real samples "
            "(percentile=%.2f)...", n, percentile)

        Se = self.Se; Le = self.Le; La = self.La
        total_keys = self.total_keys
        De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        nv = self.num_views

        # Buffer setup identical to _recalibrate_with_real_data — allocate
        # scratch once and reuse across the N forward calibrate passes.
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch_enc = torch.zeros(
            Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')

        _ae_calib_buf = torch.zeros(La * 4, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(Sa * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(
            Sa * max(Da, Ha), dtype=torch.uint8, device='cuda')

        enc_weights = {
            'qkv_w':   [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':     [w.data_ptr() for w in self._enc_o_w],
            'gate_w':  [w.data_ptr() for w in self._enc_gu_w],
            'down_w':  [w.data_ptr() for w in self._enc_d_w],
            'rope':    self._enc_rope.data_ptr(),
            'Kc':      self._Kc.reshape(-1).data_ptr(),
            'Vc':      self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }
        ae_weights = {
            'ain_w':      self._ain_w.data_ptr(),
            'ain_b':      self._ain_b.data_ptr(),
            'sa':         self._sa_all.data_ptr(),
            'qw':         self._dec_qkv_flat.data_ptr(),
            'Kc':         self._Kc.reshape(-1).data_ptr(),
            'Vc':         self._Vc.reshape(-1).data_ptr(),
            'ow':         self._dec_o_flat.data_ptr(),
            'sf':         self._sf_all.data_ptr(),
            'gw':         self._dec_gu_flat.data_ptr(),
            'dw':         self._dec_d_flat.data_ptr(),
            'aow':        self._aow.data_ptr(),
            'aob':        self._aob.data_ptr(),
            'fs':         self._fs_all.data_ptr(),
            'rope':       self._dec_rope.data_ptr(),
            'w_scales':   self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'S': Sa, 'D': Da, 'H': Ha, 'NH': 8, 'HD': 256,
            'steps': 10, 'layers': La, 'enc_seq': Se,
            'total_keys': total_keys,
        }

        per_sample_enc: list[np.ndarray] = []
        per_sample_ae:  list[np.ndarray] = []

        # Deterministic decoder noise per-sample so amax variation reflects
        # encoder input only (not the random noise tail).
        noise_gen = torch.Generator(device='cuda').manual_seed(0)

        for i, obs in enumerate(obs_list):
            # 1. Stage images through SigLIP to populate _enc_x.
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(obs.get('wrist_image', obs['image']))
                if nv >= 3:
                    img_list.append(obs.get('wrist_image_right',
                                            img_list[-1]))

            def _to_np16(im):
                if isinstance(im, torch.Tensor):
                    if im.dtype == torch.uint8 or (
                            im.is_floating_point() and
                            torch.max(im).item() > 1.5):
                        im = (im.float() / 127.5 - 1.0)
                    return im.to(dtype=fp16).cpu().numpy()
                if im.dtype == np.float16:
                    return im
                return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

            images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
            self._img_buf.upload(images_np)
            self._siglip_graph.replay()
            torch.cuda.synchronize()

            # 2. Encoder calibrate pass.
            enc_bufs = {
                'x':       self._enc_x.data_ptr(),
                'x_fp8':   self._enc_x_fp8.data_ptr(),
                'qkv':     self._enc_qkv_buf.data_ptr(),
                'logits':  self._enc_logits.data_ptr(),
                'attn_out': self._enc_attn.data_ptr(),
                'o_fp8':   self._enc_o_fp8.data_ptr(),
                'gate':    self._enc_gate.data_ptr(),
                'hidden':  self._enc_hidden.data_ptr(),
                'hid_fp8': self._enc_hid_fp8.data_ptr(),
                'fg':      self._enc_fg.data_ptr(),
                'ctx':     self._ctx,
                'norm_scratch': _norm_scratch.data_ptr(),
                'x_scratch':    _x_scratch.data_ptr(),
                'calib_buf':    _calib_buf.data_ptr(),
                'd_scale':      _d_scale.data_ptr(),
                'fp8_scratch':  _fp8_scratch_enc.data_ptr(),
                'ones':         _ones.data_ptr(),
            }
            self._enc_calib_scales.zero_()
            self._Kc.zero_(); self._Vc.zero_()
            encoder_forward_calibrate(
                self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                self._enc_calib_scales.data_ptr(), stream=0)
            torch.cuda.synchronize()
            per_sample_enc.append(self._enc_calib_scales.cpu().numpy().copy())

            # 3. Decoder calibrate pass.
            ae_bufs = {
                'noise':   self._g_noise.data_ptr(),
                'x':       self._ae_x.data_ptr(),
                'xn':      self._ae_xn.data_ptr(),
                'gate':    self._ae_gate.data_ptr(),
                'qkv':     self._ae_qkv.data_ptr(),
                'logits':  self._ae_logits.data_ptr(),
                'attn_out': self._ae_attn.data_ptr(),
                'hid':     self._ae_hid.data_ptr(),
                'fg':      self._ae_fg.data_ptr(),
                'xn_fp8':  self._ae_xn_fp8.data_ptr(),
                'hid_fp8': self._ae_hid_fp8.data_ptr(),
                'ctx_fp8': self._ae_ctx_fp8.data_ptr(),
                'calib_buf':      _ae_calib_buf.data_ptr(),
                'd_scale':        _ae_d_scale.data_ptr(),
                'hidden_scratch': _ae_hidden_scratch.data_ptr(),
                'fp8_scratch':    _ae_fp8_scratch.data_ptr(),
            }
            self._ae_calib_scales.zero_()
            # Reset noise-gen per sample for deterministic per-sample input.
            noise = torch.empty_like(self._g_noise).normal_(generator=noise_gen)
            self._g_noise.copy_(noise)
            decoder_forward_calibrate(
                self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
                self._ae_calib_scales.data_ptr(), stream=0)
            torch.cuda.synchronize()
            per_sample_ae.append(self._ae_calib_scales.cpu().numpy().copy())

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  calibration sample %d/%d", i + 1, n)

        # 4. Per-point percentile reduce.
        final_enc = accumulate_amax(per_sample_enc, percentile=percentile)
        final_ae  = accumulate_amax(per_sample_ae,  percentile=percentile)

        if verbose:
            logger.info("encoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample_enc, final_enc)))
            logger.info("decoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample_ae, final_ae)))

        # 5. Write reduced scales back onto the device buffers.
        self._enc_calib_scales.copy_(
            torch.from_numpy(final_enc.astype(np.float32)))
        self._ae_calib_scales.copy_(
            torch.from_numpy(final_ae.astype(np.float32)))

        # 6. Recompute encoder alpha-fold list (cuBLASLt alpha path).
        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(final_enc[i]) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]

        # 7. Outlier warning (same ratio-based check as RTX path).
        check_scale_ceiling(final_enc, label=f"pi05_thor_enc_N{n}")
        check_scale_ceiling(final_ae,  label=f"pi05_thor_ae_N{n}")

        # 8. Recapture CUDA graph with the new scales.
        self._capture_enc_ae_graph()
        self._real_data_calibrated = True
        logger.info(
            "Pi0.5 Thor multi-frame calibration + graph recapture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_latency_stats(self):
        """Return latency statistics."""
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat),
            "mean_ms": float(np.mean(lat)),
            "std_ms": float(np.std(lat)),
            "min_ms": float(np.min(lat)),
            "max_ms": float(np.max(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "hz": float(1000 / np.mean(lat)),
        }
