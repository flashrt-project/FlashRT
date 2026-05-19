"""FlashRT — Motus torch frontend for RTX (sm120) — G2.

Class: ``MotusTorchFrontendRtx``. Owner of:
    * the loaded upstream Motus model (8.02B params, ~16GB BF16)
    * the AttentionBackend (declared but not yet wired into kernels at G2)
    * the MotusPipelineRtx (orchestrates the denoise loop)
    * lifecycle: __init__ -> set_prompt(...) -> infer(obs) [...]

VRAM contract on RTX 5090 (32 GB):
    * load Motus first (~16 GB BF16 weights + ~9 GB activations
      ≈ 25 GB peak) — DO NOT also hold T5 (3.7B = 7.4 GB) at the
      same time, or we OOM (verified during baseline)
    * caller pre-encodes T5 ctx + builds VLM inputs once outside this
      frontend (matches the Motus input-bundle contract),
      passes them in via ``set_prompt(prompt, t5_embeds=, vlm_inputs=)``

Reference baseline: PyTorch BF16 P50 = 1226.9 ms / 24.8 GB / cos = 1.0
(see the upstream Motus baseline artifacts).

────────────────────────────────────────────────────────────────────
G2 status
────────────────────────────────────────────────────────────────────
At G2 the frontend is functional end-to-end. Internally:
    * weights load via the upstream ``model.load_checkpoint(...)``
    * pipeline delegates to upstream Motus methods (cos = 1.0 baseline)
    * attention does NOT yet route through ``self.attn.run(...)`` — that
      hookpoint is in G3c (the upstream wan_layer.self_attn calls
      ``flash_attention(...)`` from bak/wan/modules/attention.py; we
      can replace either the global symbol or pull QKV out and call
      attn.run directly)
    * NO CUDA Graph capture (G5)
    * NO FP8 quantization (G4)

Acceptance: ``tests/test_motus_g2_cosine.py`` — cos vs baseline ≥ 0.999,
3× A/B std < 0.0005, latency P50 measurable (expected 1200-1500 ms).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from flash_rt.models.motus.pipeline_rtx import (
    MotusPipelineDims, MotusPipelineRtx,
)

logger = logging.getLogger(__name__)


def _apply_fp4_profile_from_env() -> None:
    """Normalize Motus FP4 profile flags before any swap modules import."""
    profile = os.environ.get("FLASH_RT_MOTUS_FP4_PROFILE", "").strip().lower()
    if not profile:
        legacy = os.environ.get("FLASH_RT_MOTUS_USE_FP4", "").strip().lower()
        if legacy in ("1", "true", "on", "yes", "fp4", "nvfp4"):
            profile = "on"
        elif legacy in ("0", "false", "off", "no", "fp8", "none"):
            profile = "off"

    if not profile:
        return

    if profile in ("1", "true", "on", "yes", "fp4", "nvfp4",
                   "balanced", "vae-fp4"):
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_VIDEO_QKV"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_VIDEO_O"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO"] = "1"
        os.environ.setdefault("FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_MODE", "down")
        os.environ.setdefault("FLASH_RT_MOTUS_FFN_DOWN_FUSED_BGQ", "0")
        os.environ.setdefault("FLASH_RT_MOTUS_USE_FP4_VAE", "1")
        os.environ.setdefault("FLASH_RT_MOTUS_VAE_FP4_AGGRESSIVE_CACHE", "0")
        os.environ.setdefault("FLASH_RT_MOTUS_VAE_FP4_MAX_CI", "2048")
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_CROSS_Q"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_CROSS_O"] = "1"
        os.environ.pop("FLASH_RT_MOTUS_NO_NVFP4_VIDEO_O", None)
        os.environ.pop("FLASH_RT_MOTUS_NO_G7_39_NVFP4_ADALN", None)
        os.environ["FLASH_RT_MOTUS_NVFP4_VIDEO_QKV_AWQ"] = "0"
        os.environ["FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ"] = "0"
        os.environ.setdefault("FLASH_RT_MOTUS_AWQ_FFN", "1")
        os.environ.setdefault("FLASH_RT_MOTUS_AWQ_FFN_SCOPE", "action")
        os.environ.setdefault("FLASH_RT_MOTUS_NO_CUDNN_FP8_CONV2D", "1")
        return

    if profile in ("fast-cache", "action-fast",
                   "fast", "fast-tiny", "tiny-cache"):
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_VIDEO_QKV"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_VIDEO_O"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO"] = "1"
        os.environ.setdefault("FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_MODE", "down")
        os.environ.setdefault("FLASH_RT_MOTUS_FFN_DOWN_FUSED_BGQ", "0")
        os.environ.setdefault("FLASH_RT_MOTUS_USE_FP4_VAE", "1")
        os.environ.setdefault("FLASH_RT_MOTUS_VAE_FP4_AGGRESSIVE_CACHE", "1")
        os.environ.setdefault("FLASH_RT_MOTUS_VAE_FP4_MAX_CI", "2048")
        os.environ.setdefault(
            "FLASH_RT_MOTUS_VAE_FP8_T1_SITES",
            "30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47")
        os.environ.setdefault("FLASH_RT_MOTUS_NO_CUDNN_FP8_CONV2D", "1")
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_CROSS_Q"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_CROSS_O"] = "1"
        os.environ.pop("FLASH_RT_MOTUS_NO_NVFP4_VIDEO_O", None)
        os.environ.pop("FLASH_RT_MOTUS_NO_G7_39_NVFP4_ADALN", None)
        os.environ["FLASH_RT_MOTUS_NVFP4_VIDEO_QKV_AWQ"] = "0"
        os.environ["FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ"] = "0"
        os.environ.setdefault("FLASH_RT_MOTUS_AWQ_FFN", "1")
        os.environ.setdefault("FLASH_RT_MOTUS_AWQ_FFN_SCOPE", "action")
        if profile in ("fast", "fast-tiny", "tiny-cache"):
            os.environ["FLASH_RT_MOTUS_AWQ_FFN_SCOPE"] = "all"
            os.environ.setdefault("FLASH_RT_MOTUS_USE_UND_FFN_V5SPLIT_STAGE3",
                                  "1")
            os.environ["FLASH_RT_MOTUS_USE_TINYFP8_DISPATCH"] = "1"
            os.environ.setdefault("FLASH_RT_MOTUS_USE_WFFN_DN_STREAMK", "1")
            os.environ.setdefault("FLASH_RT_MOTUS_USE_FP8_CONV2D_V2", "1")
            os.environ.setdefault("FLASH_RT_MOTUS_USE_FP8_CONV2D_V2_NCDHWOUT",
                                  "1")
            os.environ.setdefault("FLASH_RT_MOTUS_USE_VAE_TIME_CONV_FP8", "1")
            os.environ.setdefault("FLASH_RT_MOTUS_VAE_T1_CONV2D", "1")
            os.environ.setdefault("FLASH_RT_MOTUS_FFN_MULTI_STREAM", "1")
            # Stage4 (2026-05-17): handtuned FP8 small-M dispatcher is
            # ship-ready (-0.88 ms / inf verified, cos preserved).
            # Production-default via this profile, opt-out by setting
            # MOTUS_HANDTUNED_FP8=0 explicitly.
            os.environ.setdefault("MOTUS_HANDTUNED_FP8", "1")
        return

    if profile in ("0", "false", "off", "no", "fp8", "none"):
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_VIDEO_QKV"] = "0"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_VIDEO_O"] = "0"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO"] = "0"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_CROSS_Q"] = "0"
        os.environ["FLASH_RT_MOTUS_USE_NVFP4_CROSS_O"] = "0"
        os.environ["FLASH_RT_MOTUS_USE_FP4_VAE"] = "0"
        os.environ["FLASH_RT_MOTUS_NO_NVFP4_VIDEO_O"] = "1"
        os.environ["FLASH_RT_MOTUS_NO_G7_39_NVFP4_ADALN"] = "1"
        os.environ["FLASH_RT_MOTUS_USE_G7_49_CROSS_NORM3_NVFP4_Q"] = "0"
        os.environ.setdefault("FLASH_RT_MOTUS_AWQ_FFN", "1")
        os.environ.setdefault("FLASH_RT_MOTUS_AWQ_FFN_SCOPE", "action")
        os.environ.setdefault("FLASH_RT_MOTUS_USE_UND_FFN_V5T", "0")
        return

    raise ValueError(
        "FLASH_RT_MOTUS_FP4_PROFILE must be one of "
        "fast/fast-cache/on/off/fp8/nvfp4")


# Make the upstream Motus repo importable. We do not modify any source
# under the Motus checkout; we just import its modules. ``bak/`` ships
# vendored Wan2.2 + qwen2_5_vl deps that Motus's own code expects on
# sys.path.
_MOTUS_ROOT_ENV = os.environ.get("FLASH_RT_MOTUS_ROOT") or os.environ.get("MOTUS_ROOT")
_MOTUS_ROOT = pathlib.Path(_MOTUS_ROOT_ENV).expanduser() if _MOTUS_ROOT_ENV else None
if _MOTUS_ROOT is not None:
    _BAK_ROOT = _MOTUS_ROOT / "bak"
    for _p in (str(_MOTUS_ROOT), str(_BAK_ROOT)):
        if _p not in sys.path:
            sys.path.insert(0, _p)


class MotusTorchFrontendRtx:
    """Motus torch frontend for RTX 5090 (sm120). G2 implementation."""

    # ────────────────────────────────────────────────────────────
    # Static shape constants (Stage-2 ckpt geometry — see
    # baseline_artifacts/README.md §2 for the C2 derivation).
    # ────────────────────────────────────────────────────────────
    NUM_LAYERS = 30
    NUM_INFERENCE_STEPS = 10
    NUM_VIDEO_FRAMES = 8
    VIDEO_HEIGHT = 384
    VIDEO_WIDTH = 320
    ACTION_DIM = 14
    STATE_DIM = 14
    ACTION_CHUNK_SIZE = 8        # = num_video_frames * video_action_freq_ratio(=1)
    NUM_REGISTERS = 0            # pretrain mode, no register tokens

    def __init__(
        self,
        checkpoint_dir,
        wan_path: Optional[str] = None,
        vlm_path: Optional[str] = None,
        num_views: int = 3,
        autotune: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ):
        _apply_fp4_profile_from_env()

        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.num_views = num_views
        self.autotune = autotune
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else self.NUM_INFERENCE_STEPS)
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        # ─── G6.4: enable cudnn benchmark so cudnn explores algos for
        # each unique VAE conv shape (~6 unique shapes, each fired at
        # warmup) and locks the fastest into the captured graph.
        # The default heuristic on sm_120 picks an implicit_gemm tile
        # that is not always optimal for our 384×320 latent geometry.
        # ────────────────────────────────────────────────────────────
        if os.environ.get("FLASH_RT_MOTUS_NO_G6_4", "0") != "1":
            self._g6_4_cudnn = {
                "prev_benchmark": torch.backends.cudnn.benchmark,
                "prev_deterministic": torch.backends.cudnn.deterministic,
            }
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info(
                f"[motus] G6.4 cudnn.benchmark=True (was "
                f"{self._g6_4_cudnn['prev_benchmark']}); .deterministic="
                f"False (was {self._g6_4_cudnn['prev_deterministic']})")
        else:
            self._g6_4_cudnn = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G6_4=1, default cudnn config")

        # Default paths if not provided. They require FLASH_RT_MOTUS_ROOT or
        # MOTUS_ROOT to point at an upstream Motus checkout.
        if _MOTUS_ROOT is None and (wan_path is None or vlm_path is None):
            raise ValueError(
                "wan_path/vlm_path must be provided unless FLASH_RT_MOTUS_ROOT "
                "or MOTUS_ROOT points at an upstream Motus checkout.")
        self.wan_path = pathlib.Path(
            wan_path or _MOTUS_ROOT / "pretrained_models" / "Wan2.2-TI2V-5B")
        self.vlm_path = pathlib.Path(
            vlm_path or _MOTUS_ROOT / "pretrained_models" / "Qwen3-VL-2B-Instruct")

        ckpt_file = self.checkpoint_dir / "mp_rank_00_model_states.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"Motus checkpoint not found: {ckpt_file}. Expected "
                "DeepSpeed ZeRO-1 layout (mp_rank_00_model_states.pt).")

        self._load_checkpoint_geometry()

        # ─── Construct + load Motus (dominates wall time at construction) ───
        t0 = time.perf_counter()
        self.model = self._build_motus_model()
        t_build = time.perf_counter() - t0
        logger.info(f"[motus] model built in {t_build:.2f}s")

        t0 = time.perf_counter()
        self.model.load_checkpoint(str(self.checkpoint_dir), strict=False)
        t_ckpt = time.perf_counter() - t0
        logger.info(f"[motus] checkpoint loaded in {t_ckpt:.2f}s")
        self.model.eval()

        # G5 prep: move Wan's CPU-resident RoPE `freqs` buffer to cuda.
        # Upstream wan/modules/model.py:521 builds `self.freqs` on CPU
        # at __init__; line 562 has a lazy `.to(device)` inside the
        # WanModel.forward path we never traverse. As a result, our
        # process_joint_attention (motus.py:256) hits an implicit
        # CPU→GPU copy on every call — which is forbidden inside
        # torch.cuda.graph(). Move once, here.
        wm = self.model.video_model.wan_model
        if wm.freqs.device != self.device:
            wm.freqs = wm.freqs.to(self.device)
            logger.info(
                f"[motus] G5 prep: moved wan_model.freqs to {self.device}")

        # Replace upstream rope_apply with a graph-safe variant. The
        # original uses torch.unique + .tolist() (CPU↔GPU sync ops
        # forbidden inside torch.cuda.graph capture). The replacement
        # reads a cached freq_grid built once per (T, H, W) — for B=1
        # Motus this is set during graph_capture warmup.
        from flash_rt.models.motus._rope_swap import (
            install_graph_safe_rope)
        install_graph_safe_rope(model=self.model)

        # ─── G6.2: cast Wan2_2_VAE to bf16. Upstream defaults to fp32
        # which dispatches cudnn TF32 conv (~245 ms/call at G5 per
        # g6.1 profile, ~42% of wall). bf16 conv tensorop is 1.7-2×
        # faster on sm_120.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._vae_swap import install_vae_bf16
        if os.environ.get("FLASH_RT_MOTUS_NO_G6_2", "0") != "1":
            self._vae_swap_stats = install_vae_bf16(
                self.model, dtype=self.dtype)
            logger.info(
                f"[motus] G6.2 VAE bf16: {self._vae_swap_stats}")
        else:
            self._vae_swap_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G6_2=1, keeping fp32 VAE")

        from flash_rt.models.motus._vae_fp8_resample_swap import (
            install_vae_fp8_resample)
        self._vae_fp8_resample_stats = install_vae_fp8_resample(self.model)
        self._vae_fp8_resample_committed = (
            self._vae_fp8_resample_stats.get('installed', 0) == 0)
        self._vae_fp8_g723_installed = False
        self._vae_fp8_g723_committed = False
        self._vae_fp8_g723_stats = None
        self._vae_fp8_g723_commit_stats = None
        if self._vae_fp8_resample_stats.get('installed', 0):
            logger.info(
                f"[motus] G7.30 VAE FP8 resample hooks: "
                f"{self._vae_fp8_resample_stats}")

        # ─── G3a: replace WanRMSNorm / WanLayerNorm forwards with fvk
        # BF16 kernels. Pure substitution; cos must stay >= 0.999.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._norm_swap import install_fvk_norms
        if os.environ.get("FLASH_RT_MOTUS_NO_G3A", "0") != "1":
            self._norm_swap_stats = install_fvk_norms(self.model)
            logger.info(
                f"[motus] G3a norm swap: "
                f"{self._norm_swap_stats['rms']} RMSNorm + "
                f"{self._norm_swap_stats['layer_no_affine']} LayerNorm(no-affine) "
                f"+ {self._norm_swap_stats['layer_affine']} LayerNorm(affine) "
                f"replaced; {self._norm_swap_stats['skipped']} skipped")
        else:
            self._norm_swap_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G3A=1, keeping torch norms")

        # ─── G3b: replace nn.Linear under hot-path scopes with fvk
        # GemmRunner.bf16_nn[_bias]. Skips VLM + VAE.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._linear_swap import install_fvk_linears
        if os.environ.get("FLASH_RT_MOTUS_NO_G3B", "0") != "1":
            self._linear_swap_stats = install_fvk_linears(self.model)
            logger.info(
                f"[motus] G3b linear swap: "
                f"{self._linear_swap_stats['replaced']} Linears -> fvk.bf16_nn; "
                f"scope-skipped {self._linear_swap_stats['skipped_scope']}, "
                f"dtype-skipped {self._linear_swap_stats['skipped_dtype']}")
        else:
            self._linear_swap_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G3B=1, keeping torch linears")

        # ─── G3c: replace wan.modules.attention.flash_attention with
        # vendored FA2 (flash_rt_fa2.fwd_bf16). Covers Tri-modal joint
        # self-attention + Wan cross-attn into T5 ctx.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._attn_swap import install_fa2_attention
        if os.environ.get("FLASH_RT_MOTUS_NO_G3C", "0") != "1":
            self._attn_swap_stats = install_fa2_attention(self.model)
            logger.info(
                f"[motus] G3c attention swap (vendored FA2): "
                f"{self._attn_swap_stats}")
        else:
            self._attn_swap_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G3C=1, keeping shim SDPA")

        # ─── G3d: replace Wan/Action/Und FFN nn.Sequential with a
        # direct fvk chain (bf16_nn + add_bias + gelu_inplace + bf16_nn
        # + add_bias). Bypasses Sequential dispatch + per-Linear closure.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._ffn_swap import install_fvk_ffns
        if os.environ.get("FLASH_RT_MOTUS_NO_G3D", "0") != "1":
            self._ffn_swap_stats = install_fvk_ffns(self.model)
            logger.info(
                f"[motus] G3d FFN swap: "
                f"{self._ffn_swap_stats['ffn_replaced']} FFNs -> direct fvk; "
                f"struct-skipped {self._ffn_swap_stats['ffn_skipped_struct']}")
        else:
            self._ffn_swap_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G3D=1, keeping G3b FFN")

        # ─── G4: FP8 W8A8 static quantization. Pre-quantizes weights
        # at install (frees BF16 storage), allocates per-site act_scale.
        # First inference call runs in calibration mode (writes
        # act_scale via quantize_fp8_device), subsequent calls use
        # static scales.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._fp8_swap import install_fp8_swap
        if os.environ.get("FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO", "0") == "1":
            # _Fp8Site sees this during construction and keeps an NVFP4
            # copy of the video FFN weights before the BF16 storage is
            # replaced by FP8. Runtime replacement still happens after
            # calibration, immediately before graph capture.
            os.environ.setdefault("FLASH_RT_MOTUS_PREP_NVFP4_VIDEO_FFN", "1")
        self._nvfp4_ffn_video_installed = False
        self._nvfp4_ffn_video_stats = None
        self._nvfp4_video_qkv_installed = False
        self._nvfp4_video_qkv_stats = None
        if os.environ.get("FLASH_RT_MOTUS_NO_G4", "0") != "1":
            self._fp8_swap_stats = install_fp8_swap(self.model)
            self._fp8_calibrated = False
            logger.info(
                f"[motus] G4 FP8 swap: "
                f"linear={self._fp8_swap_stats['linear_replaced']}, "
                f"ffn={self._fp8_swap_stats['ffn_replaced']}, "
                f"scope-skip={self._fp8_swap_stats['skipped_scope']}, "
                f"dtype-skip={self._fp8_swap_stats['skipped_dtype']}, "
                f"bypassed={self._fp8_swap_stats['skipped_bypassed_by_ffn']}; "
                f"first infer() will calibrate")
        else:
            self._fp8_swap_stats = None
            self._fp8_calibrated = True   # nothing to calibrate
            logger.info("[motus] FLASH_RT_MOTUS_NO_G4=1, keeping BF16")

        # ─── G7.13: AWQ calibration hooks for action_expert + und_expert.
        # Naive PTQ (per-tensor FP8 / block-128 / NVFP4) all collapse cos
        # on these sites because their per-channel activation magnitude
        # variance is large at small M (chunk=8). SmoothQuant rebalances
        # by absorbing per-K activation magnitude into the weight; folded
        # scale lets per-tensor FP8 capture both with one scalar.
        # Sequence:
        #   __init__       : install pre-hooks; sites stay BF16
        #   first infer()  : 1st pass = AWQ calib (BF16); apply AWQ-FP8;
        #                    2nd pass = G4 calibration covering new sites
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._awq_fp8_swap import (
            install_awq_calibration_hooks)
        if os.environ.get("FLASH_RT_MOTUS_NO_G7_13", "0") != "1":
            self._awq_calib_states = install_awq_calibration_hooks(self.model)
            self._awq_applied = False
            logger.info(
                f"[motus] G7.13.awq: hooks installed on "
                f"{len(self._awq_calib_states)} sites")
        else:
            self._awq_calib_states = None
            self._awq_applied = True

        # ─── G7.24: action_qkv / und_qkv (BAGEL packed) FP8 W8A8 swap.
        # Calibration markers attached here so the AWQ BF16 calib pass
        # (which is the same first forward) records norm_*.abs().amax
        # per-K. After AWQ FP8 swap completes, install_g724_fp8 folds
        # SmoothQuant scale into wan_*_qkv and quantizes the (K, 9216)
        # flat to FP8. Runtime path lives inside _modulate_fuse_swap.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._action_und_qkv_fp8_swap import (
            install_g724_calibration)
        if os.environ.get("FLASH_RT_MOTUS_NO_G7_24", "0") != "1":
            self._g724_states = install_g724_calibration(self.model)
            self._g724_applied = False
        else:
            self._g724_states = None
            self._g724_applied = True

        # ─── G6.5: AdaLN modulate + gated-residual fusion (FFN paths).
        # Replaces upstream's `(1+e[k+1])*norm + e[k]` chain with
        # ada_layer_norm_bf16, and `x + y * e[k]` with gate_mul_residual.
        # Installed BEFORE the first infer() so FP8 calibration measures
        # the fused-modulate input distribution and writes self-consistent
        # act_scales.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._modulate_fuse_swap import (
            install_modulate_fuse)
        if os.environ.get("FLASH_RT_MOTUS_NO_G6_5", "0") != "1":
            self._modulate_fuse_stats = install_modulate_fuse(self.model)
            logger.info(
                f"[motus] G6.5 modulate fuse: {self._modulate_fuse_stats}")
        else:
            self._modulate_fuse_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G6_5=1, keeping upstream modulate")

        # ─── G7.8: fuse Wan video Q/K/V into one FP8 GEMM per block.
        # Must run AFTER FP8 swap (G4) and modulate fuse (G6.5/6/7).
        # Patches WanSelfAttention.forward at the class level.
        # ────────────────────────────────────────────────────────────
        from flash_rt.models.motus._wan_qkv_fuse_swap import (
            install_wan_qkv_fuse)
        if os.environ.get("FLASH_RT_MOTUS_NO_G7_8", "0") != "1":
            self._wan_qkv_fuse_stats = install_wan_qkv_fuse(self.model)
            logger.info(
                f"[motus] G7.8 wan QKV fuse: {self._wan_qkv_fuse_stats}")
        else:
            self._wan_qkv_fuse_stats = None
            logger.info("[motus] FLASH_RT_MOTUS_NO_G7_8=1, "
                        "keeping 3-GEMM Q/K/V path")

        # ─── G6.3: Wan cross-attn T5 K/V cache. We do NOT install yet
        # (the cross-attn self.k / self.v FP8 sites still need
        # calibration via the original forward path). Install + populate
        # happens between FP8 calibration and graph capture in infer().
        # ────────────────────────────────────────────────────────────
        self._kv_cache_handles = None
        self._kv_cache_disabled = (
            os.environ.get("FLASH_RT_MOTUS_NO_G6_3", "0") == "1")
        if self._kv_cache_disabled:
            logger.info(
                "[motus] FLASH_RT_MOTUS_NO_G6_3=1, skipping T5 KV cache")

        # ─── G5: CUDA Graph capture state. Populated lazily on second
        # infer() call (first one runs FP8 calibration in eager mode,
        # then we capture).
        # ────────────────────────────────────────────────────────────
        self._graph_state = None
        self._graph_disabled = (
            os.environ.get("FLASH_RT_MOTUS_NO_G5", "0") == "1")
        if self._graph_disabled:
            logger.info("[motus] FLASH_RT_MOTUS_NO_G5=1, skipping graph capture")

        # ─── AttentionBackend: built but not wired at G2 (kept here so
        #     G3c can swap in fvk attention without touching __init__) ───
        from flash_rt.hardware.rtx.attn_backend_motus import (
            make_motus_attention_spec)
        self._attn_spec = make_motus_attention_spec()
        self._attn_backend = _G2AttentionBackendStub(self._attn_spec)

        # ─── Pipeline ─────────────────────────────────────────────────
        self._dims = MotusPipelineDims(
            num_layers=self.NUM_LAYERS,
            num_inference_steps=self.num_inference_steps,
            num_video_frames=self.num_video_frames,
            video_height=self.video_height,
            video_width=self.video_width,
            action_chunk_size=self.action_chunk_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            is_pretrain=self.training_mode == "pretrain",
            num_registers=self.num_registers,
        )
        self.pipeline = MotusPipelineRtx(
            model=self.model,
            attn_backend=self._attn_backend,
            dims=self._dims,
            dtype=self.dtype,
            device=self.device,
        )

        # Per-prompt state (filled by set_prompt)
        self._t5_embeds: Optional[List[torch.Tensor]] = None
        self._vlm_inputs: Optional[List[Dict[str, Any]]] = None
        self._instruction: Optional[str] = None

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[motus] G2 ready: params={n_params/1e9:.2f}B, "
            f"build={t_build:.1f}s, ckpt_load={t_ckpt:.1f}s, "
            f"vram={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ──────────────────────────────────────────────────────────────
    # Model construction (mirrors the baseline driver)
    # ──────────────────────────────────────────────────────────────

    def _load_checkpoint_geometry(self) -> None:
        """Load Motus shape metadata from checkpoint config when present.

        Stage-2 pretrain checkpoints do not require a config override and keep
        the class defaults. Stage-3 Robotwin finetune checkpoints carry
        ``config.json`` with video/action ratio and register-token geometry.
        Environment variables can override the detected values for debugging.
        """
        cfg: Dict[str, Any] = {}
        cfg_path = self.checkpoint_dir / "config.json"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                cfg = raw

        common = cfg.get("common", {}) if isinstance(cfg.get("common", {}), dict) else {}
        action_cfg = (
            cfg.get("action_expert", {})
            if isinstance(cfg.get("action_expert", {}), dict)
            else {}
        )
        und_cfg = (
            cfg.get("und_expert", {})
            if isinstance(cfg.get("und_expert", {}), dict)
            else {}
        )
        vlm_cfg = (
            und_cfg.get("vlm", {})
            if isinstance(und_cfg.get("vlm", {}), dict)
            else {}
        )

        self.action_dim = int(os.environ.get(
            "FLASH_RT_MOTUS_ACTION_DIM",
            common.get("action_dim", self.ACTION_DIM)))
        self.state_dim = int(os.environ.get(
            "FLASH_RT_MOTUS_STATE_DIM",
            common.get("state_dim", self.STATE_DIM)))
        self.num_video_frames = int(os.environ.get(
            "FLASH_RT_MOTUS_NUM_VIDEO_FRAMES",
            common.get("num_video_frames", self.NUM_VIDEO_FRAMES)))
        self.video_height = int(os.environ.get(
            "FLASH_RT_MOTUS_VIDEO_HEIGHT",
            common.get("video_height", self.VIDEO_HEIGHT)))
        self.video_width = int(os.environ.get(
            "FLASH_RT_MOTUS_VIDEO_WIDTH",
            common.get("video_width", self.VIDEO_WIDTH)))
        self.global_downsample_rate = int(os.environ.get(
            "FLASH_RT_MOTUS_GLOBAL_DOWNSAMPLE_RATE",
            common.get("global_downsample_rate", 3)))
        self.video_action_freq_ratio = int(os.environ.get(
            "FLASH_RT_MOTUS_VIDEO_ACTION_FREQ_RATIO",
            common.get("video_action_freq_ratio", 1)))

        default_mode = "finetune" if self.video_action_freq_ratio > 1 else "pretrain"
        self.training_mode = os.environ.get(
            "FLASH_RT_MOTUS_TRAINING_MODE", default_mode).strip().lower()
        if self.training_mode not in ("pretrain", "finetune"):
            raise ValueError(
                "FLASH_RT_MOTUS_TRAINING_MODE must be pretrain or finetune")
        if self.training_mode == "finetune":
            os.environ.setdefault("FLASH_RT_MOTUS_USE_TINYFP8_DISPATCH", "0")

        default_registers = self.NUM_REGISTERS if self.training_mode == "pretrain" else 4
        self.num_registers = int(os.environ.get(
            "FLASH_RT_MOTUS_NUM_REGISTERS", default_registers))
        self.action_chunk_size = int(os.environ.get(
            "FLASH_RT_MOTUS_ACTION_CHUNK_SIZE",
            self.num_video_frames * self.video_action_freq_ratio))

        self.und_expert_hidden_size = int(und_cfg.get(
            "hidden_size", 512))
        self.und_expert_ffn_dim_multiplier = int(und_cfg.get(
            "ffn_dim_multiplier", 4))
        self.und_expert_norm_eps = float(und_cfg.get(
            "norm_eps", 1e-5))
        self.vlm_adapter_input_dim = int(vlm_cfg.get(
            "input_dim", 2048))
        self.vlm_adapter_projector_type = str(vlm_cfg.get(
            "projector_type", "mlp3x_silu"))
        self.action_expert_dim = int(action_cfg.get(
            "hidden_size", 1024))
        self.action_expert_ffn_dim_multiplier = int(action_cfg.get(
            "ffn_dim_multiplier", 4))
        self.action_expert_norm_eps = float(action_cfg.get(
            "norm_eps", 1e-5))

        logger.info(
            "[motus] checkpoint geometry: mode=%s, frames=%d, action_chunk=%d, "
            "registers=%d, action_dim=%d, state_dim=%d, video=%dx%d, ratio=%d",
            self.training_mode, self.num_video_frames, self.action_chunk_size,
            self.num_registers, self.action_dim, self.state_dim,
            self.video_height, self.video_width, self.video_action_freq_ratio)

    def _build_motus_model(self):
        """Construct Motus with checkpoint geometry. Imports upstream code."""
        from models.motus import Motus, MotusConfig  # upstream

        mc = MotusConfig(
            wan_checkpoint_path=str(self.wan_path),
            vae_path=str(self.wan_path / "Wan2.2_VAE.pth"),
            wan_config_path=str(self.wan_path),
            video_precision="bfloat16",
            vlm_checkpoint_path=str(self.vlm_path),
            und_expert_hidden_size=self.und_expert_hidden_size,
            und_expert_ffn_dim_multiplier=self.und_expert_ffn_dim_multiplier,
            und_expert_norm_eps=self.und_expert_norm_eps,
            vlm_adapter_input_dim=self.vlm_adapter_input_dim,
            vlm_adapter_projector_type=self.vlm_adapter_projector_type,
            num_layers=self.NUM_LAYERS,
            action_state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_expert_dim=self.action_expert_dim,
            action_expert_ffn_dim_multiplier=self.action_expert_ffn_dim_multiplier,
            action_expert_norm_eps=self.action_expert_norm_eps,
            global_downsample_rate=self.global_downsample_rate,
            video_action_freq_ratio=self.video_action_freq_ratio,
            num_video_frames=self.num_video_frames,
            video_height=self.video_height,
            video_width=self.video_width,
            batch_size=1,
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            training_mode=self.training_mode,
            load_pretrained_backbones=False,
        )
        return Motus(mc).to(self.device).eval()

    # ──────────────────────────────────────────────────────────────
    # Public API (matches docs/stable_api.md predict shape)
    # ──────────────────────────────────────────────────────────────

    def set_prompt(
        self,
        prompt: str,
        *,
        t5_embeds: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Pre-encode + cache instruction conditioning.

        At G2 the caller MUST pre-encode T5 + build VLM inputs (32 GB
        VRAM cannot hold both Motus + T5 simultaneously). At G7 we will
        add an optional inline T5/VLM path that loads/frees opportunistically.

        Args:
            prompt: text instruction (stored for logging only at G2).
            t5_embeds: list of per-batch [seq_len, 4096] BF16 tensors
                       (output of T5EncoderModel from the baseline driver).
            vlm_inputs: list of per-batch dicts with keys input_ids,
                        attention_mask, pixel_values, image_grid_thw.
        """
        if t5_embeds is None or vlm_inputs is None:
            raise ValueError(
                "G2 requires pre-encoded t5_embeds and vlm_inputs. "
                "Use the Motus input-bundle contract for the expected shapes.")
        # Move to device + dtype, list-of-tensors form (Motus convention)
        self._t5_embeds = [e.to(self.device).to(self.dtype) for e in t5_embeds]
        self._vlm_inputs = [
            {k: (v.to(self.device) if torch.is_tensor(v) else v)
             for k, v in vi.items()}
            for vi in vlm_inputs
        ]
        self._instruction = prompt
        logger.info(f"[motus] set_prompt: {prompt!r}")

        # G5: precompute VLM und_tokens + T5 ctx OUTSIDE the captured
        # graph. The VLM uses torch.linspace (CPU-side) which is
        # forbidden inside torch.cuda.graph() — moving it to set_prompt
        # is also the §7-O1 optimization (saves 9× VLM forwards inside
        # the per-step loop).
        # If the prompt changes, we also need to drop the captured
        # graph and re-capture — invalidate by clearing _graph_state.
        self.pipeline.clear_caches()
        self.pipeline.precompute_und_tokens(self._vlm_inputs)
        self.pipeline.precompute_t5_context(self._t5_embeds)
        # Sprint 1.4: free Qwen3-VL-2B vlm_model weights — only used in
        # precompute_und_tokens above. After cache is populated, the VLM
        # is dead for all subsequent infer() calls. Saves ~4.2 GB VRAM.
        # Note: a second set_prompt() with new vlm_inputs would need the
        # VLM reloaded; this trades that re-init cost for VRAM headroom.
        # Disable with FLASH_RT_MOTUS_FREE_VLM_AFTER_PROMPT=0.
        if os.environ.get(
                'FLASH_RT_MOTUS_FREE_VLM_AFTER_PROMPT', '1') == '1':
            vlm_mod = getattr(self.model, 'vlm_model', None)
            if vlm_mod is not None:
                freed = 0
                for p in vlm_mod.parameters():
                    if p.is_cuda and p.numel() > 0:
                        freed += p.numel() * p.element_size()
                        p.data = torch.empty(
                            0, dtype=p.dtype, device=p.device)
                if freed > 0:
                    logger.info(
                        f'[motus.set_prompt] freed vlm_model weights: '
                        f'{freed/1e9:.2f} GB')
        # G6.3: repopulate K/V cache for the new T5 ctx (only if it's
        # already installed — first set_prompt before any infer() leaves
        # this as None and infer() does the install + populate itself).
        if self._kv_cache_handles is not None:
            from flash_rt.models.motus._kv_cache_swap import (
                populate_wan_t5_kv_cache)
            populate_wan_t5_kv_cache(
                self.model, self.pipeline._cached_t5_ctx)
        if self._graph_state is not None:
            logger.info("[motus] set_prompt changed; dropping captured graph")
            self._graph_state = None

    def infer(
        self,
        first_frame: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ):
        """Run one denoise call using the configured denoising step count.

        On first call (when ``self._fp8_calibrated`` is False AND the
        G4 FP8 swap is installed), runs the forward in CALIBRATION
        mode: every fvk-FP8 site uses ``quantize_fp8_device`` to
        write its activation amax/448 to the persistent device
        scale buffer. After that one pass, calibration is locked
        and subsequent calls use ``quantize_fp8_static``.

        Args:
            first_frame: [B, 3, H, W] in [0, 1].
            state: [B, state_dim] (ignored in pretrain mode).

        Returns:
            Tuple of (predicted_frames, predicted_actions) — see
            MotusPipelineRtx.run() for shapes.
        """
        if self._t5_embeds is None or self._vlm_inputs is None:
            raise RuntimeError(
                "Call set_prompt(prompt, t5_embeds=, vlm_inputs=) before infer()")

        first_frame = first_frame.to(self.device).to(self.dtype)
        if state is not None:
            state = state.to(self.device).to(self.dtype)

        # G4 calibration on first inference. G7.13.awq runs an extra
        # BF16 pass first to capture per-K activation amax for action /
        # und expert sites, then folds the SmoothQuant scale into weight
        # before the main G4 calibration runs (which now covers the new
        # AWQ-FP8 sites too).
        if (self._fp8_swap_stats is not None) and (not self._fp8_calibrated):
            from flash_rt.models.motus._fp8_swap import set_calibrating

            # 0) AWQ activation-amax capture in BF16 (action/und still BF16).
            if (self._awq_calib_states is not None) and (not self._awq_applied):
                logger.info("[motus] G7.13.awq: BF16 calibration pass…")
                from flash_rt.models.motus._vae_fp8_resample_swap import (
                    set_calibrating as set_resample_calibrating)
                set_resample_calibrating(True)
                with torch.no_grad():
                    try:
                        _ = self.pipeline.run(
                            first_frame=first_frame, state=state,
                            t5_embeds=self._t5_embeds, vlm_inputs=self._vlm_inputs,
                        )
                    finally:
                        set_resample_calibrating(False)
                torch.cuda.synchronize()
                from flash_rt.models.motus._awq_fp8_swap import (
                    install_awq_fp8_swap, remove_awq_calibration_hooks)
                remove_awq_calibration_hooks(self.model)
                self._awq_swap_stats = install_awq_fp8_swap(
                    self.model, self._awq_calib_states)
                self._awq_applied = True
                logger.info(
                    f"[motus] G7.13.awq: applied — {self._awq_swap_stats}")

                # G7.24: AWQ calib pass also accumulated norm_*.abs().amax
                # into _g724_state.act_amax_K (via modulate_fuse hook).
                # Now fold SmoothQuant + FP8 quantize flat QKV weights;
                # subsequent G4 calibration runs the FP8 path with
                # dynamic act_scale capture.
                if (self._g724_states is not None) and (not self._g724_applied):
                    from flash_rt.models.motus._action_und_qkv_fp8_swap import (
                        install_g724_fp8)
                    self._g724_swap_stats = install_g724_fp8(self.model)
                    self._g724_applied = True
                    logger.info(
                        f"[motus] G7.24: applied — {self._g724_swap_stats}")

            # 1) G4 calibration for ALL FP8 sites (Wan + new AWQ sites).
            logger.info("[motus] G4: running calibration forward...")
            from flash_rt.models.motus._vae_fp8_resample_swap import (
                set_calibrating as set_resample_calibrating)
            set_calibrating(True)
            set_resample_calibrating(True)
            try:
                _ = self.pipeline.run(
                    first_frame=first_frame, state=state,
                    t5_embeds=self._t5_embeds, vlm_inputs=self._vlm_inputs,
                )
                torch.cuda.synchronize()
            finally:
                set_calibrating(False)
                set_resample_calibrating(False)
            self._fp8_calibrated = True
            logger.info("[motus] G4: calibration complete; switching to static")

            from flash_rt.models.motus._fp8_swap import (
                autotune_motus_hot_fp8_gemms)
            self._fp8_gemm_autotune_stats = autotune_motus_hot_fp8_gemms(
                self.model)
            logger.info(
                f"[motus] G7.29 FP8 GEMM autotune: "
                f"{self._fp8_gemm_autotune_stats}")

            # NOTE: handtuned FP8 dispatch install moved BELOW the V6 stack
            # block (after tinyfp8 install) so that `gemm.fp8_nn_dev` lookup
            # at install-time captures the tinyfp8 router as our fallback
            # `original_fn` — shapes not in SHAPE_TO_KERNEL fall to tinyfp8,
            # not raw cuBLASLt. See bench data 2026-05-16.

        if (os.environ.get("FLASH_RT_MOTUS_AUTO_G7_23_VAE_FP8", "0") == "1"
                and not self._vae_fp8_g723_committed):
            from flash_rt.models.motus._vae_fp8_swap import (
                install_vae_fp8,
                set_calibrating as set_vae_g723_calibrating,
                commit_calibration as commit_vae_g723_calibration,
            )
            if not self._vae_fp8_g723_installed:
                self._vae_fp8_g723_stats = install_vae_fp8(self.model)
                self._vae_fp8_g723_installed = True
                self._graph_state = None
                logger.info(
                    f"[motus] G7.23 VAE FP8 installed: "
                    f"{self._vae_fp8_g723_stats}")

            # Stage4 (2026-05-17): also install (3,1,1) time_conv FP8
            # swap before calibration, so both calibrate in same pass.
            try:
                from flash_rt.models.motus._vae_time_conv_fp8_swap import (
                    install_time_conv_fp8_swap,
                    finalize_time_conv_fp8,
                )
                _time_conv_install_stats = install_time_conv_fp8_swap(
                    self.model)
                print(
                    f"[motus] VAE time_conv FP8 install: "
                    f"{_time_conv_install_stats}", flush=True)
            except Exception as _exc:
                print(
                    f"[motus] VAE time_conv FP8 install FAILED: {_exc}",
                    flush=True)
                _time_conv_install_stats = {'enabled': False}

            if not (self._vae_fp8_g723_stats or {}).get("skipped", False):
                logger.info("[motus] G7.23 VAE FP8 calibration forward...")
                set_vae_g723_calibrating(True)
                try:
                    with torch.no_grad():
                        _ = self.pipeline.run(
                            first_frame=first_frame, state=state,
                            t5_embeds=self._t5_embeds,
                            vlm_inputs=self._vlm_inputs,
                        )
                    torch.cuda.synchronize()
                finally:
                    set_vae_g723_calibrating(False)
                self._vae_fp8_g723_commit_stats = (
                    commit_vae_g723_calibration(self.model))
                logger.info(
                    f"[motus] G7.23 VAE FP8 committed: "
                    f"{self._vae_fp8_g723_commit_stats}")
                if _time_conv_install_stats.get('enabled', False):
                    _tc_final = finalize_time_conv_fp8()
                    print(
                        f"[motus] VAE time_conv FP8 finalized: {_tc_final}",
                        flush=True)
            self._vae_fp8_g723_committed = True

            # Phase 8 — chain FP4 swap on top of FP8 swap (additive).
            # Status (2026-05-12): integration + router work; per-conv cos
            # 0.989 compounds across 30+ convs to <0.997 floor; wall savings
            # limited to -2.2ms by dequant/requant overhead (no fused
            # bf16_rms_silu_quant_nvfp4 kernel yet). See
            # MOTUS_HANDOFF_FP4_NEXT.md Phase 8.
            if os.environ.get("FLASH_RT_MOTUS_USE_FP4_VAE", "0") == "1":
                from flash_rt.models.motus._vae_fp4_swap import install_vae_fp4
                self._vae_fp4_stats = install_vae_fp4(self.model)
                self._graph_state = None
                print(f"[motus] VAE FP4 installed: {self._vae_fp4_stats}",
                      flush=True)
                # AWQ calibration pass: collect per-Ci max|x| on every
                # FP4-routed Conv3d during a one-shot bf16 forward, then
                # dump stats (and optionally apply AWQ scales).
                # FLASH_RT_MOTUS_VAE_FP4_AWQ_CALIB=1 → dump stats only.
                # =2 → also apply AWQ scaling to weights (re-quant FP4).
                awq_mode = int(os.environ.get(
                    "FLASH_RT_MOTUS_VAE_FP4_AWQ_CALIB", "0"))
                if awq_mode > 0:
                    from flash_rt.models.motus._vae_fp4_swap import (
                        awq_calibrate_and_dump)
                    awq_calibrate_and_dump(
                        self.model, self.pipeline,
                        first_frame=first_frame, state=state,
                        t5_embeds=self._t5_embeds,
                        vlm_inputs=self._vlm_inputs,
                        apply_scales=(awq_mode >= 2))
                    self._graph_state = None

        if (os.environ.get("FLASH_RT_MOTUS_USE_NVFP4_VIDEO_QKV", "0") == "1"
                and not self._nvfp4_video_qkv_installed):
            from flash_rt.models.motus._wan_qkv_fuse_swap import (
                install_wan_qkv_nvfp4)
            self._nvfp4_video_qkv_stats = install_wan_qkv_nvfp4(self.model)
            self._nvfp4_video_qkv_installed = True
            logger.info(
                f"[motus] NVFP4 video QKV: {self._nvfp4_video_qkv_stats}")

        if (os.environ.get("FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO", "0") == "1"
                and not self._nvfp4_ffn_video_installed):
            from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                install_motus_nvfp4_ffn_video)
            self._nvfp4_ffn_video_stats = install_motus_nvfp4_ffn_video(
                self.model)
            self._nvfp4_ffn_video_installed = True
            logger.info(
                f"[motus] NVFP4 video FFN: {self._nvfp4_ffn_video_stats}")

        if (self._nvfp4_video_qkv_installed or self._nvfp4_ffn_video_installed):
            torch.cuda.empty_cache()

        # Recipe C step 1 (cutlass NVFP4 GEMM_up + bias + GELU bf16-out)
        # is now baked into _motus_nvfp4_ffn_video_swap.py — default on
        # when fvk.fp4_w4a16_gemm_bias_gelu_bf16out_sm120 is built into
        # flash_rt_kernels. Env disable:
        # FLASH_RT_MOTUS_USE_WFFN_CUTLASS_FUSED=0.

        if not self._vae_fp8_resample_committed:
            from flash_rt.models.motus._vae_fp8_resample_swap import (
                commit_calibration_resample)
            self._vae_fp8_resample_commit_stats = (
                commit_calibration_resample(self.model))
            self._vae_fp8_resample_committed = True
            logger.info(
                f"[motus] G7.30 VAE FP8 resample committed: "
                f"{self._vae_fp8_resample_commit_stats}")

        # G6.3: install + populate Wan cross-attn T5 KV cache. Must be
        # AFTER calibration (so self.k / self.v have valid act_scale)
        # and BEFORE graph capture. Idempotent: re-populates cache
        # contents on subsequent set_prompt via repopulate_kv_cache().
        if (not self._kv_cache_disabled) and (self._kv_cache_handles is None):
            from flash_rt.models.motus._kv_cache_swap import (
                install_wan_t5_kv_cache, populate_wan_t5_kv_cache)
            t5_ctx = self.pipeline._cached_t5_ctx
            assert t5_ctx is not None, (
                "G6.3 expects pipeline._cached_t5_ctx populated by "
                "set_prompt -> precompute_t5_context")
            self._kv_cache_handles = install_wan_t5_kv_cache(
                self.model,
                dtype=self.dtype,
                t5_seq_len=int(t5_ctx.size(1)),
                batch=int(t5_ctx.size(0)),
            )
            populate_wan_t5_kv_cache(self.model, t5_ctx)
            logger.info(
                f"[motus] G6.3: T5 KV cache ready "
                f"({len(self._kv_cache_handles)} layers)")

        # Action expert FFN V6tuned + und FFN V5tuned megakernels + tiny_fp8
        # GEMM dispatcher. All three are additive monkey-patches on
        # model.action_module.process_ffn / model.und_module.process_ffn /
        # fvk.GemmRunner.fp8_nn_dev. Installed AFTER AWQ FP8 swap so the
        # required _awq_*_site attributes are present, AFTER VAE memory
        # has been allocated, and BEFORE graph capture so the megakernel
        # paths are baked into the captured graph.
        if not getattr(self, "_v6_stack_installed", False):
            print(f"[motus] V6 stack install: hasattr action_expert="
                  f"{hasattr(self.model, 'action_expert')} "
                  f"und_expert={hasattr(self.model, 'und_expert')} "
                  f"action_module={hasattr(self.model, 'action_module')} "
                  f"und_module={hasattr(self.model, 'und_module')}",
                  flush=True)
            try:
                from flash_rt.models.motus._action_ffn_v6t_install import (
                    install as _install_action_ffn_v6t)
                n_act = _install_action_ffn_v6t(self.model)
                print(f"[motus] action FFN V6tuned: {n_act} layers",
                      flush=True)
            except Exception as e:
                print(f"[motus] action FFN V6tuned install failed: {e}",
                      flush=True)
            try:
                from flash_rt.models.motus._und_ffn_v5t_install import (
                    install as _install_und_ffn_v5t)
                n_und = _install_und_ffn_v5t(self.model)
                print(f"[motus] und FFN V5tuned: {n_und} layers",
                      flush=True)
            except Exception as e:
                print(f"[motus] und FFN V5tuned install failed: {e}",
                      flush=True)
            try:
                from flash_rt.models.motus._tinyfp8_dispatch_install import (
                    install as _install_tinyfp8_dispatch)
                n_tiny = _install_tinyfp8_dispatch(self.model)
                print(f"[motus] tiny_fp8 dispatch: {n_tiny} weights",
                      flush=True)
            except Exception as e:
                print(f"[motus] tiny_fp8 dispatch install failed: {e}",
                      flush=True)
            # Handtuned FP8 dispatch — env-gated, additive, MUST install
            # AFTER tinyfp8 so the instance-attr wrapper captures the
            # tinyfp8-routed bound method as its `original_fn` fallback.
            # Routes 2 verified-winner shapes (action_o, und_o) to ht_*
            # inline-PTX kernels. See _handtuned_fp8_dispatch.py header.
            try:
                from flash_rt.models.motus._handtuned_fp8_dispatch import (
                    install_handtuned_fp8_dispatch)
                gemm = getattr(self.model, '_g3b_gemm', None)
                if gemm is not None:
                    self._handtuned_fp8_stats = install_handtuned_fp8_dispatch(
                        self.model, gemm)
                    print(f"[motus] handtuned FP8 dispatch: "
                          f"{self._handtuned_fp8_stats}", flush=True)
            except Exception as e:
                print(f"[motus] handtuned FP8 dispatch install failed: {e}",
                      flush=True)
            self._v6_stack_installed = True

        # TeaCache install (env-gated, additive) — MUST install BEFORE
        # graph capture so the skip/compute schedule is baked into the
        # captured graph (CUDA Graph replay cannot evaluate Python
        # branches per-replay).
        if not getattr(self, "_teacache_install_attempted", False):
            try:
                from flash_rt.models.motus._teacache_swap import (
                    install_motus_teacache)
                self._teacache_stats = install_motus_teacache(self.pipeline)
                if self._teacache_stats.get("enabled", False):
                    print(f"[motus] TeaCache install: "
                          f"{self._teacache_stats}", flush=True)
            except Exception as _exc:
                print(f"[motus] TeaCache install failed: {_exc}",
                      flush=True)
            try:
                from flash_rt.models.motus._easycache_swap import (
                    install_motus_easycache)
                self._easycache_stats = install_motus_easycache(
                    self.pipeline)
                if self._easycache_stats.get("enabled", False):
                    print(f"[motus] EasyCache install: "
                          f"{self._easycache_stats}", flush=True)
            except Exception as _exc:
                print(f"[motus] EasyCache install failed: {_exc}",
                      flush=True)
            try:
                from flash_rt.models.motus._taylorseer_swap import (
                    install_motus_taylorseer)
                self._taylorseer_stats = install_motus_taylorseer(
                    self.pipeline)
                if self._taylorseer_stats.get("enabled", False):
                    print(f"[motus] TaylorSeer install: "
                          f"{self._taylorseer_stats}", flush=True)
            except Exception as _exc:
                print(f"[motus] TaylorSeer install failed: {_exc}",
                      flush=True)
            try:
                from flash_rt.models.motus._mixcache_swap import (
                    install_motus_mixcache)
                self._mixcache_stats = install_motus_mixcache(
                    self.pipeline)
                if self._mixcache_stats.get("enabled", False):
                    print(f"[motus] MixCache install: "
                          f"{self._mixcache_stats}", flush=True)
            except Exception as _exc:
                print(f"[motus] MixCache install failed: {_exc}",
                      flush=True)
            self._teacache_install_attempted = True

        # G5 graph capture: lazy, on the FIRST infer() AFTER calibration.
        # Calibration writes per-site act_scale ON DEVICE — those live in
        # the same buffers the captured graph reads from (no re-capture
        # needed when scales change because buffers are pointer-stable).
        if (not self._graph_disabled) and (self._graph_state is None):
            from flash_rt.models.motus._graph_capture import (
                capture_motus_graph)
            logger.info("[motus] G5: capturing CUDA graph (first replayable infer)")
            self._graph_state = capture_motus_graph(
                self.pipeline,
                first_frame=first_frame,
                state=state,
                t5_embeds=self._t5_embeds,
                vlm_inputs=self._vlm_inputs,
            )
            logger.info("[motus] G5: capture complete; subsequent infer() = replay")

        # Steady-state path.
        if self._graph_state is not None:
            from flash_rt.models.motus._graph_capture import (
                replay_motus_graph)
            return replay_motus_graph(
                self._graph_state, first_frame=first_frame, state=state)

        # Fallback: eager (used when G5 disabled by env).
        return self.pipeline.run(
            first_frame=first_frame,
            state=state,
            t5_embeds=self._t5_embeds,
            vlm_inputs=self._vlm_inputs,
        )

    def predict(self, images, prompt=None, state=None):
        """``api.VLAModel.predict`` ABI — required by load_model wrapper.

        At G2 we expose the lower-level (set_prompt + infer) API since
        Motus's prompt conditioning includes pre-encoded T5/VLM that
        the predict() shape doesn't naturally fit. G7 will add an
        inline T5/VLM path so predict() works directly.
        """
        raise NotImplementedError(
            "Motus G2 exposes set_prompt() / infer(); predict() requires "
            "the inline T5+VLM path scheduled for G7. Use the lower-level "
            "API for now — see tests/test_motus_g2_cosine.py.")


class _G2AttentionBackendStub:
    """G2 placeholder for the AttentionBackend. Holds the spec for shape
    declarations; no kernel routing yet. G3c replaces this with a live
    backend that dispatches ``run("mot_joint", ...)`` to fvk kernels.
    """

    def __init__(self, spec):
        self.spec = spec

    def __repr__(self):
        sites = sorted(self.spec.sites.keys())
        return f"_G2AttentionBackendStub(sites={sites})"
