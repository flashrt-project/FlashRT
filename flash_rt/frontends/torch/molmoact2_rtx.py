"""FlashRT — MolmoAct2 torch frontend for RTX consumer GPUs (SM120/SM89).

Loads ``allenai/MolmoAct2`` safetensors directly, builds the static prompt for
a fixed task, precomputes the flow-matching modulation tables, and drives
:class:`flash_rt.models.molmoact2.pipeline_rtx.MolmoAct2Pipeline`.

Public surface (mirrors the other RTX VLA frontends)::

    fe = MolmoAct2TorchFrontendRtx(ckpt_dir, precision="fp8")
    fe.set_prompt(task, norm_tag="yam_dual_molmoact2")
    fe.calibrate(samples)            # real observations
    fe.record_infer_graph()
    out = fe.infer(images_uint8, state)   # {"actions": (horizon, action_dim)}

The steady-state graph includes the pinned-host→device input copies, the whole
model, and the device→pinned-host action copy, so ``infer`` is one graph launch
plus a stream synchronisation.

This module is the only place that touches PyTorch tensors; the pipeline below
it receives raw device pointers.
"""

from __future__ import annotations

import ctypes
import json
import os
import logging
import pathlib
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_fa2 as fa2
import flash_rt.flash_rt_kernels as fvk
fvk_mod = fvk

# MolmoAct2 ships its model-specific kernels behind FLASHRT_ENABLE_MOLMOACT2
# (SM120 build option). Check the exported surface up front so a build without
# them raises one actionable error instead of an AttributeError halfway
# through weight upload.
_MOLMO_KERNELS = (
    "molmo_action_denorm_bf16", "molmo_action_mk_layers",
    "molmo_action_mk_pack_fp8", "molmo_action_mk_struct_sizes",
    "molmo_add_bias_bf16", "molmo_bias_gelu_quant_fp8_vec",
    "molmo_bias_rms_norm_rows96_exact", "molmo_bias_round_add_bf16",
    "molmo_bias_round_bf16", "molmo_concat2_quant_fp8",
    "molmo_gate_residual_bf16", "molmo_gate_residual_bias",
    "molmo_layer_norm_quant_fp8", "molmo_mask_dims_bf16",
    "molmo_mod_rms_norm_bf16", "molmo_mod_rms_norm_rows768_exact",
    "molmo_mod_rms_norm_rows768_quant_fp8_exact", "molmo_mul_bf16",
    "molmo_pool_attn_bf16", "molmo_pool_mean_bf16",
    "molmo_qk_norm_rope_bf16", "molmo_qkv_norm_rope_fp8_hd128",
    "molmo_qkv_split_bias", "molmo_qkv_split_bias_norm_rope",
    "molmo_qkv_split_vec", "molmo_reciprocal_f32",
    "molmo_resize_patchify_bf16", "molmo_rms_norm768_ctx_warp",
    "molmo_rms_norm768_heads96", "molmo_rms_norm_bf16",
    "molmo_rms_norm_rows96_exact", "molmo_silu_mul_bias",
    "molmo_silu_mul_quant_fp8", "molmo_skinny_bias_rms_norm",
    "molmo_skinny_gate_residual",
    "molmo_skinny_gate_residual_mod_rms768_quant_fp8",
    "molmo_skinny_qkv_split_bias", "molmo_skinny_silu_mul",
    "molmo_state_discretize_f32", "molmo_smallm_gemm_bias_res_bf16out",
    # Shared SM120 FP8 prefill attention entries the text stage calls.
    "fmha_fp8_causal_gqa_nhd_d128_fp4out",
    "fmha_fp8_causal_gqa_q64_nhd_d128_fp4out",
    "fmha_fp8_causal_gqa_q64_nhd_d128_fp4out_wsp",
    "molmo_vit_attn_bf16", "molmo_vit_attn_bf16_fp8out", "molmo_vit_attn_bf16_persist",
    "molmo_vit_attn_f16pv",
)


def _require_kernels(module=fvk) -> None:
    missing = [name for name in _MOLMO_KERNELS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "flash_rt_kernels was built without the MolmoAct2 kernels "
            f"({len(missing)} missing, e.g. {missing[0]!r}). Rebuild on "
            "GPU_ARCH=120/121 with -DFLASHRT_ENABLE_MOLMOACT2=ON; see "
            "docs/molmoact2_usage.md.")
from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.frontends._fp8_layout import select_fp8_layout
from flash_rt.models.molmoact2.pipeline_rtx import (
    ACT_D, ACT_H, ACT_HD, ACT_LAYERS, ACT_MAX_DIM, ACT_NH, ACT_STEPS,
    CONTEXT_DIM, EPS, GRID, IMG_H, IMG_W, MolmoAct2Pipeline, NUM_VIEWS,
    PATCH, PATCH_DIM, PATCH_DIM_PAD, POOL, POOL_GRID, POOL_TOKENS, POOL_WINDOW, POS_TOKENS,
    TILE, TOTAL_PATCHES, TOTAL_POOLED, TXT_D, TXT_FFN, TXT_HD, TXT_KV_DIM,
    TXT_LAYERS, TXT_NH, TXT_NKV, TXT_Q_DIM, TXT_QKV_DIM, VIS_D, VIS_FFN,
    VIS_FFN_PAD, VIS_HD, VIS_LAYERS, VIS_NH,
)

logger = logging.getLogger(__name__)

ACT_HORIZON_DEFAULT = 30

# ── processor / prompt constants (mirrors the HF custom code) ─────────────
IM_PATCH_TOKEN = "<im_patch>"
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
IM_END_TOKEN = "<im_end>"
IMAGE_PROMPT = "<|image|>"
ACTION_OUTPUT_TOKEN = "<action_output>"
STATE_START_TOKEN = "<state_start>"
STATE_END_TOKEN = "<state_end>"
STATE_TOKEN_PREFIX = "<state_"
SETUP_START_TOKEN = "<setup_start>"
SETUP_END_TOKEN = "<setup_end>"
CONTROL_START_TOKEN = "<control_start>"
CONTROL_END_TOKEN = "<control_end>"
QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,…"
QUESTION_TRAILING_CLOSERS = "\"'”’)]}"
QUESTION_SURROUNDING_DELIMITERS = "\"'`“”‘’[](){}"
QUESTION_PREFIX_PATTERNS = (
    re.compile(r"^\s*(?:please\s+)?(?:what|which|how|where|when|why)\b", re.I),
    re.compile(r"^\s*(?:please\s+)?(?:can|could|would|will)\s+you\b", re.I),
    re.compile(r"^\s*(?:please\s+)?(?:tell|show|describe|explain)\s+me\b", re.I),
)

# Default FP8 eligibility: every large GEMM, but keep the tiny per-step action
# input/output projections in BF16 (K=32 / N=32) where FP8 has no bandwidth
# benefit and the output is most sensitive.
_FP8_EXCLUDE_PREFIXES = ("act_embed", "act_final", "vis_patch_embed")

# Measured best skinny-NVFP4 tile configurations per action-expert shape
# (sm120 block-scaled W4A4, K-split partials; see the campaign tile sweep).
_SKINNY_CFG = {(2304, 768): 14, (3072, 768): 6, (768, 768): 6, (768, 3072): 6}
_SKINNY_KINDS = ("act_qkv", "act_so", "act_cq", "act_co", "act_up",
                 "act_gate", "act_down")


def _normalize_question_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in QUESTION_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(
            QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(
            QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    chunks = [c.strip() for c in re.split(r"[.!?]+", normalized) if c.strip()]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)
    return normalized.lower()


def _build_robot_text(*, task: str, discrete_state_string: str,
                      setup_type: str, control_mode: str,
                      num_images: int) -> str:
    setup_text = f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"
    control_text = f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"
    state_clause = (
        f" The current state of the robot is {discrete_state_string}."
        if discrete_state_string else "")
    prompt = (
        f"The task is to {task}. The setup is {setup_text}.{state_clause} "
        f"The expected control mode is {control_text}. Given these, what "
        "action should the robot take to complete the task?")
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = IMAGE_PROMPT
    else:
        image_prefix = "".join(
            f"Image {idx + 1}{IMAGE_PROMPT}" for idx in range(num_images))
    return (f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}")


def _discrete_state_string(state: np.ndarray, num_tokens: int) -> str:
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_tokens - 1)
    bins = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_tokens) - 1)
    return (f"{STATE_START_TOKEN}"
            + "".join(f"{STATE_TOKEN_PREFIX}{int(b)}>" for b in bins.reshape(-1))
            + f"{STATE_END_TOKEN}")


class _SafetensorReader:
    """Random-access checkpoint reader (one file handle at a time)."""

    def __init__(self, root: pathlib.Path):
        from safetensors import safe_open
        self._safe_open = safe_open
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self._root = root
        self._map = index["weight_map"]
        self._handles: dict[str, Any] = {}

    def keys(self):
        return self._map.keys()

    def get(self, key: str) -> torch.Tensor:
        fname = self._map[key]
        handle = self._handles.get(fname)
        if handle is None:
            handle = self._safe_open(str(self._root / fname), framework="pt")
            self._handles[fname] = handle
        return handle.get_tensor(key)


class MolmoAct2TorchFrontendRtx:
    """MolmoAct2 RTX torch frontend."""

    def __init__(self, checkpoint_dir, *, device: str = "cuda",
                 precision: str = "bf16", num_steps: int = ACT_STEPS,
                 norm_tag: str = "yam_dual_molmoact2",
                 use_cuda_graph: bool = True,
                 fp8_layout: Optional[str] = None,
                 hardware: Optional[str] = None):
        _require_kernels()
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.device = torch.device(device)
        self.precision = str(precision).lower()
        if self.precision not in ("bf16", "fp8", "nvfp4"):
            raise ValueError(
                f"precision must be 'bf16', 'fp8' or 'nvfp4', got {precision!r}")
        self.num_steps = int(num_steps)
        self.norm_tag = str(norm_tag)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.fp8_layout = select_fp8_layout(hardware, fp8_layout)
        if self.fp8_layout != "kn":
            raise ValueError(
                "MolmoAct2 RTX pipeline currently ships the 'kn' weight layout "
                f"(bf16_nn/fp8_nn_dev), got {self.fp8_layout!r}")
        self._cudart = ctypes.CDLL("libcudart.so")
        self._cudart.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_void_p]
        self._cudart.cudaMemcpyAsync.restype = ctypes.c_int
        self._cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._cudart.cudaStreamSynchronize.restype = ctypes.c_int

        self._keep: list[torch.Tensor] = []
        self._bf16: dict[str, int] = {}
        self._fp8: dict[str, tuple[int, int]] = {}
        # [N, K] e4m3 copies (+ host scale) for the fused action SwiGLU
        # producer; only the action FFN up/gate sites are materialized.
        self._fp8_nk: dict[str, tuple[int, int, float]] = {}
        # Fused up+gate+SwiGLU producer: one launch instead of up GEMM +
        # gate GEMM + silu_mul_bias (-0.36 ms E2E at the production shapes).
        self._act_fused = os.environ.get(
            "FLASHRT_MOLMO_ACT_FUSED", "1") == "1"
        self._bias: dict[str, int] = {}
        self._nvfp4_scope = os.environ.get(
            "FLASHRT_MOLMO_NVFP4_SCOPE", "tvp").strip().lower()
        self._vis_fused_layers = self._parse_layer_set(
            os.environ.get("FLASHRT_MOLMO_VIS_FUSED",
                           "all" if self._nvfp4_scope == "tvp" else "0"))
        self._nvfp4: dict[str, tuple] = {}
        # Zero-pad the FP8 vision w2 reduction to K=4352 (cuBLASLt runs a
        # 128-multiple K faster; the padded products are exact zeros).
        self._vis_w2_pad = (self.precision in ("fp8", "nvfp4")
                            and not self._vis_fused_layers
                            and os.environ.get("FLASHRT_MOLMO_VIS_W2PAD",
                                               "1") == "1")
        # Text o-projection weights in the skinny packed layout, used only by
        # the small-M experimental route (FLASHRT_MOLMO_SMALLM_GEMM).
        self._smallm: dict[str, tuple] = {}
        # SwiGLU epilogue fold for the text FFN (same route as the Qwen3 RTX
        # prefill): txt_ff1 is quantized over the row-interleaved gate|up
        # weight and the GEMM epilogue writes silu(gate)*up as the ff2 NVFP4
        # operand. Maps site -> host alpha (the fold takes scalar alphas).
        self._swiglu_fold_enabled = os.environ.get(
            "FLASHRT_MOLMO_SWIGLU_FOLD", "1") == "1"
        self._swiglu_fold: dict[str, float] = {}
        self._nvfp4_keep: list[torch.Tensor] = []
        self._skinny: dict[str, tuple] = {}
        self._skinny_keep: list[torch.Tensor] = []

        self._stats = self._load_stats()
        self._reader = _SafetensorReader(self.checkpoint_dir)
        self._tokenizer = self._load_tokenizer()

        self._load_weights()
        self._embedding_ptr = self._build_embedding_table()

        self.pipeline: Optional[MolmoAct2Pipeline] = None
        self.prompt_len = 0
        self.task = ""
        self.actions: Optional[np.ndarray] = None
        self.graph_recorded = False
        self._calibrated = False
        self._stream = torch.cuda.Stream()

        # Pinned host staging (captured into the graph).
        self._host_images = torch.empty(
            NUM_VIEWS * IMG_H * IMG_W * 3, dtype=torch.uint8, pin_memory=True)
        self._state_dim = len(self._stats["state_q01"])
        self._host_state = torch.empty(
            self._state_dim, dtype=torch.float32, pin_memory=True)
        horizon = int(self._stats["action_horizon"])
        self._host_noise = torch.empty(
            horizon, ACT_MAX_DIM, dtype=torch.int16, pin_memory=True)
        self._host_actions = torch.empty(
            horizon, int(self._stats["action_q01"].shape[0]),
            dtype=torch.float32, pin_memory=True)
        self._host_images_np = self._host_images.numpy()
        self._host_state_np = self._host_state.numpy()
        self._host_noise_np = self._host_noise.numpy().view(np.uint16)
        self._host_actions_np = self._host_actions.numpy()
        self._images_buf = CudaBuffer.empty(
            NUM_VIEWS * IMG_H * IMG_W * 3, np.uint8)

    # ══════════════════════════════════════════════════════════════════
    #  Checkpoint / stats
    # ══════════════════════════════════════════════════════════════════
    def _load_stats(self) -> dict:
        payload = json.loads(
            (self.checkpoint_dir / "norm_stats.json").read_text())
        tag = payload["metadata_by_tag"][self.norm_tag]
        mode = str(payload.get("norm_mode", "min_max"))
        if mode not in ("q01_q99", "q10_q90"):
            raise ValueError(f"unsupported norm_mode {mode!r}")
        low_key, high_key = ("q01", "q99") if mode == "q01_q99" else ("q10", "q90")

        def _arr(stats, key):
            v = stats.get(key)
            return None if v is None else np.asarray(v, dtype=np.float32)

        out = {"mode": mode,
               "setup_type": str(tag.get("setup_type", "") or ""),
               "control_mode": str(tag.get("control_mode", "") or ""),
               "action_horizon": int(tag.get("action_horizon") or 30),
               "n_action_steps": int(tag.get("n_action_steps") or 30)}
        for kind in ("state", "action"):
            st = tag[f"{kind}_stats"]
            q01 = _arr(st, low_key)
            q99 = _arr(st, high_key)
            if q01 is None or q99 is None:
                raise ValueError(f"missing {kind} q01/q99 stats")
            mask = st.get("mask")
            mask = (np.ones_like(q01, dtype=bool) if mask is None
                    else np.asarray(mask, dtype=bool))
            mn, mx = _arr(st, "min"), _arr(st, "max")
            zero = (np.zeros_like(mask) if mn is None or mx is None
                    else (mn == mx))
            out[f"{kind}_q01"] = q01.astype(np.float32)
            out[f"{kind}_q99"] = q99.astype(np.float32)
            out[f"{kind}_mask"] = mask.astype(np.int8)
            out[f"{kind}_zero_mask"] = zero.astype(np.int8)
        return out

    def _load_tokenizer(self):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MolmoAct2 prompt construction needs the `tokenizers` package "
                "(shipped with transformers). Install `tokenizers`.") from exc
        return Tokenizer.from_file(str(self.checkpoint_dir / "tokenizer.json"))

    # ══════════════════════════════════════════════════════════════════
    #  Weight loading / quantization
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _parse_layer_set(spec: str) -> frozenset[int]:
        """Parse ``FLASHRT_MOLMO_VIS_FUSED``: "all", "0-9", "0,2,4" or "0"."""
        spec = (spec or "0").strip().lower()
        if spec in ("", "0", "off", "none"):
            return frozenset()
        if spec == "all":
            return frozenset(range(VIS_LAYERS))
        layers: set[int] = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                layers.update(range(int(lo), int(hi) + 1))
            else:
                layers.add(int(part))
        return frozenset(i for i in layers if 0 <= i < VIS_LAYERS)

    def _nvfp4_eligible(self, name: str, n: int, k: int) -> bool:
        """NVFP4 prefill kernels need K % 128 == 0 and N % 8 == 0.

        ``FLASHRT_MOLMO_NVFP4_SCOPE`` selects the stage set. "tvp" (the
        default with the checkpoint 10-step flow schedule) is the text stage,
        every vision projection (w2 through the fused GELU->NVFP4 MLP route)
        and the pooling adapter; the action context and expert keep their
        FP8 / skinny routes. It keeps >= 0.99 delivery similarity on the eight
        calibration observations and twelve held-out real frames. "tvmqp"
        (vision o/w2 in FP8), "tvm", "tv", "tvmq", "text", "textx", "vision",
        "action", and "all" are sensitivity variants;
        ``FLASHRT_MOLMO_NVFP4_EXTRA`` adds further site prefixes for
        calibration sweeps.
        """
        if self.precision != "nvfp4":
            return False
        if name.startswith(("vis_patch_embed", "act_embed", "act_final")):
            return False
        if name.startswith("vis_w2_"):
            # The fused bias+GELU->NVFP4 epilogue route requires the padded
            # 4352-wide MLP; it is enabled per layer by the calibration sweep
            # (`FLASHRT_MOLMO_VIS_FUSED`).
            if int(name.rsplit("_", 1)[1]) not in self._vis_fused_layers:
                return False
        scope = self._nvfp4_scope
        extra = os.environ.get("FLASHRT_MOLMO_NVFP4_EXTRA", "")
        if extra and any(p and name.startswith(p)
                         for p in extra.split(",")):
            return k % 128 == 0 and n % 8 == 0
        if scope == "text" and not name.startswith("txt_"):
            return False
        if scope == "vision" and not name.startswith(("vis_", "pool_",
                                                       "proj_")):
            return False
        if scope == "action" and not name.startswith("act_"):
            return False
        if scope == "tv" and not name.startswith(
                ("txt_", "vis_", "pool_", "proj_")) and name not in (
                    "act_ck", "act_cv"):
            return False
        if scope == "tvmq":
            if not (name.startswith("txt_") or name.startswith(
                    ("vis_w1_", "vis_qkv_"))):
                return False
        if scope == "tvm":
            # Text-stage NVFP4 plus the vision MLP input projection only.
            if not (name.startswith("txt_") or name.startswith("vis_w1_")):
                return False
        if scope == "tvp":
            # Delivery scope: text stage, all vision projections and the
            # pooling adapter (the action context stays on the FP8 route).
            if not name.startswith(("txt_", "vis_", "pool_", "proj_")):
                return False
        if scope == "tvmqp":
            # Text stage, vision w1 + qkv, and the pooling adapter
            # projections.
            # vis_o/vis_w2 stay FP8 (the 2-step schedule needed it; at the
            # checkpoint 10 steps "tvp" passes the gate).
            if not (name.startswith(("txt_", "pool_", "proj_"))
                    or name.startswith(("vis_w1_", "vis_qkv_"))):
                return False
        if scope == "textx":
            # Text-stage NVFP4 with the QKV projection left in FP8: its K/V
            # feed both the text attention and the action context.
            if not name.startswith("txt_") or name.startswith("txt_qkv"):
                return False
        if scope == "va" and not name.startswith(
                ("vis_", "pool_", "proj_", "act_")):
            return False
        return k % 128 == 0 and n % 8 == 0

    def _quantize_nvfp4_weight(self, w_nk: torch.Tensor, bias):
        """BF16 [N, K] -> packed NVFP4 + swizzled SF + per-tensor alpha."""
        fvk = fvk_mod
        N, K = w_nk.shape
        w_nk = w_nk.to(device=self.device, dtype=torch.bfloat16).contiguous()
        gs = (w_nk.float().abs().amax() / (6.0 * 448.0)).clamp_min(
            1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_quantize_weight(
            w_nk.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
            N, K, 0, 4)
        if rc != 0:
            raise RuntimeError(
                f"NVFP4 weight quantization failed for [{N}, {K}]: {rc}")
        swz = torch.zeros(fvk.nvfp4_sf_swizzled_bytes(N, K),
                          dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_sf_linear_to_swizzled(
            sf.data_ptr(), swz.data_ptr(), N, K, True, 0)
        if rc != 0:
            raise RuntimeError(
                f"NVFP4 SF swizzle failed for [{N}, {K}]: {rc}")
        if bias is None:
            b = torch.zeros(N, dtype=torch.bfloat16, device=self.device)
        else:
            b = bias.to(device=self.device, dtype=torch.bfloat16).contiguous()
        self._nvfp4_keep.extend([q, swz, gs, b])
        return (q.data_ptr(), swz.data_ptr(), gs.data_ptr(), b.data_ptr())

    def _prepare_smallm_weight(self, name: str, w_nk: torch.Tensor) -> None:
        """BF16 [N, K] -> skinny-packed NVFP4 weight for the small-M kernel."""
        fvk = fvk_mod
        N, K = int(w_nk.shape[0]), int(w_nk.shape[1])
        gs = (w_nk.float().abs().amax() / (6.0 * 448.0)).clamp_min(
            1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_quantize_weight(
            w_nk.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
            N, K, 0, 4)
        if rc != 0:
            raise RuntimeError(f"small-M NVFP4 quant failed {name}")
        qpk = torch.empty(N * K // 2, dtype=torch.uint8, device=self.device)
        sfpk = torch.empty(N * K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_pack_weight(
            q.data_ptr(), sf.data_ptr(), qpk.data_ptr(), sfpk.data_ptr(),
            N, K, 0)
        if rc != 0:
            raise RuntimeError(f"small-M NVFP4 pack failed {name}")
        self._keep.extend([qpk, sfpk])
        self._smallm[name] = (qpk.data_ptr(), sfpk.data_ptr())

    def _skinny_eligible(self, name: str) -> bool:
        """Action-expert GEMMs served by the skinny NVFP4 (W4A4) family."""
        if self.precision != "nvfp4":
            return False
        scope = os.environ.get(
            "FLASHRT_MOLMO_SKINNY_SCOPE", "outd").strip().lower()
        if scope == "off":
            return False
        if scope in ("outd", "out") and not name.startswith(
                ("act_so_", "act_co_", "act_down_")):
            return False
        if scope == "outfg" and not name.startswith(
                ("act_so_", "act_co_", "act_up_", "act_gate_")):
            return False
        return any(name.startswith(k + "_") for k in _SKINNY_KINDS)

    def _prepare_skinny_weight(self, name: str, w_nk: torch.Tensor) -> None:
        """BF16 [N, K] -> fragment-major NVFP4 weight for the skinny GEMM."""
        N, K = int(w_nk.shape[0]), int(w_nk.shape[1])
        cfg = _SKINNY_CFG.get((N, K))
        if cfg is None or not fvk.nvfp4_skinny_config_supports(cfg, N, K):
            cfg = next(c for c in range(fvk.nvfp4_skinny_config_count())
                       if fvk.nvfp4_skinny_config_supports(c, N, K))
        gs = (w_nk.float().abs().amax() / (6.0 * 448.0)).clamp_min(
            1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_quantize_weight(
            w_nk.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
            N, K, 0, 4)
        if rc != 0:
            raise RuntimeError(f"skinny NVFP4 weight quant failed {name}")
        wq = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sfq = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        rc = fvk.nvfp4_skinny_pack_weight(
            q.data_ptr(), sf.data_ptr(), wq.data_ptr(), sfq.data_ptr(),
            N, K, 0)
        if rc != 0:
            raise RuntimeError(f"skinny NVFP4 weight pack failed {name}")
        torch.cuda.synchronize()
        splits = K // fvk.nvfp4_skinny_config_k_chunk(cfg)
        self._skinny_keep.extend([wq, sfq, gs])
        self._skinny[name] = (wq.data_ptr(), sfq.data_ptr(), gs.data_ptr(),
                              cfg, splits, N, K)

    def _fp8_eligible(self, name: str) -> bool:
        force_bf16 = os.environ.get("FLASHRT_MOLMO_BF16_SITES", "")
        if any(p and name.startswith(p) for p in force_bf16.split(",")):
            return False
        return self.precision in ("fp8", "nvfp4") and not name.startswith(
            _FP8_EXCLUDE_PREFIXES)

    def _upload_bf16(self, t: torch.Tensor) -> torch.Tensor:
        out = t.to(device=self.device, dtype=torch.bfloat16).contiguous()
        self._keep.append(out)
        return out

    def _add_gemm(self, name: str, w: torch.Tensor,
                  bias: Optional[torch.Tensor] = None) -> None:
        """Upload one ``nn.Linear`` weight as a FlashRT GEMM operand.

        ``w`` is the checkpoint ``[N, K]`` weight. The "kn" layout stores
        ``[K, N]`` for ``bf16_nn`` / ``fp8_nn_dev``.
        """
        w_bf = w.to(device=self.device, dtype=torch.bfloat16)
        n_out, k_in = int(w.shape[0]), int(w.shape[1])
        if self._skinny_eligible(name):
            self._prepare_skinny_weight(name, w_bf.contiguous())
        if self._nvfp4_eligible(name, n_out, k_in):
            if (self._swiglu_fold_enabled and name.startswith("txt_ff1_")
                    and bias is None):
                # [gate; up] -> gate row i at 2i, up row i at 2i + 1. The
                # amax (global scale) and the per-row 16-element blocks are
                # unchanged, so the packed rows are a permutation of the
                # merged weight's.
                half = n_out // 2
                w_bf = torch.stack([w_bf[:half], w_bf[half:]], 1).reshape(
                    n_out, k_in)
                self._nvfp4[name] = self._quantize_nvfp4_weight(w_bf, bias)
                # weight global scale (activation global scale is 1)
                self._swiglu_fold[name] = float(self._nvfp4_keep[-2].item())
                return
            self._nvfp4[name] = self._quantize_nvfp4_weight(w_bf, bias)
            if (os.environ.get("FLASHRT_MOLMO_SMALLM_GEMM", "0") == "1"
                    and name.startswith("txt_o_")):
                self._prepare_smallm_weight(name, w_bf.contiguous())
            return
        if self._act_fused and (name.startswith("act_up_")
                                or name.startswith("act_gate_")):
            nk_scale = (w_bf.float().abs().amax().double() / 448.0).clamp_min(
                1e-12).float().reshape(1)
            wf_nk = (w_bf.float() * nk_scale.reciprocal()).clamp(
                -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            self._keep.extend([wf_nk, nk_scale])
            self._fp8_nk[name] = (wf_nk.data_ptr(), nk_scale.data_ptr(),
                                  float(nk_scale.item()))
        wt = w_bf.t().contiguous()
        if self._fp8_eligible(name):
            scale = (wt.float().abs().amax().double() / 448.0).clamp_min(
                1e-12).float().reshape(1)
            wf = (wt.float() * scale.reciprocal()).clamp(
                -448.0, 448.0).to(torch.float8_e4m3fn)
            self._keep.extend([wf, scale])
            self._fp8[name] = (wf.data_ptr(), scale.data_ptr())
        else:
            self._keep.append(wt)
            self._bf16[name] = wt.data_ptr()
        if bias is not None:
            b = bias.to(device=self.device, dtype=torch.bfloat16).contiguous()
            self._keep.append(b)
            self._bias[name] = b.data_ptr()

    def _add_norm(self, name: str, w: torch.Tensor) -> None:
        self._bf16[name] = self._upload_bf16(w.reshape(-1)).data_ptr()

    def _add_bias_tensor(self, name: str, b: torch.Tensor) -> None:
        self._bias[name] = self._upload_bf16(b.reshape(-1)).data_ptr()

    def _load_weights(self) -> None:
        R = self._reader

        # ── vision tower ──
        # K = 588 padded with zero columns to 592 (16-byte rows) so cuBLASLt
        # can use an aligned kernel; the patch rows carry matching zeros.
        w_pe = R.get("model.vision_backbone.image_vit.patch_embedding.weight")
        w_pe = torch.nn.functional.pad(w_pe, (0, PATCH_DIM_PAD - w_pe.shape[1]))
        self._add_gemm("vis_patch_embed", w_pe,
                       R.get("model.vision_backbone.image_vit.patch_embedding.bias"))
        pos = R.get("model.vision_backbone.image_vit.positional_embedding")
        self._pos_emb = pos.float().numpy()
        for i in range(VIS_LAYERS):
            p = f"model.vision_backbone.image_vit.transformer.resblocks.{i}"
            wq, wk, wv = (R.get(f"{p}.attention.wq.weight"),
                          R.get(f"{p}.attention.wk.weight"),
                          R.get(f"{p}.attention.wv.weight"))
            bq, bk, bv = (R.get(f"{p}.attention.wq.bias"),
                          R.get(f"{p}.attention.wk.bias"),
                          R.get(f"{p}.attention.wv.bias"))
            self._add_gemm(f"vis_qkv_{i}", torch.cat([wq, wk, wv], 0),
                           torch.cat([bq, bk, bv], 0))
            self._add_gemm(f"vis_o_{i}", R.get(f"{p}.attention.wo.weight"),
                           R.get(f"{p}.attention.wo.bias"))
            self._add_norm(f"vis_ln1_w_{i}", R.get(f"{p}.attention_norm.weight"))
            self._add_bias_tensor(f"vis_ln1_b_{i}", R.get(f"{p}.attention_norm.bias"))
            self._add_norm(f"vis_ln2_w_{i}", R.get(f"{p}.ffn_norm.weight"))
            self._add_bias_tensor(f"vis_ln2_b_{i}", R.get(f"{p}.ffn_norm.bias"))
            w1 = R.get(f"{p}.feed_forward.w1.weight")
            b1 = R.get(f"{p}.feed_forward.w1.bias")
            w2 = R.get(f"{p}.feed_forward.w2.weight")
            # The fused NVFP4 bias+GELU-epilogue GEMM tiles M in 128 rows and
            # N in 32 columns: pad the 4304-wide MLP to 4352 (zero rows of
            # w1/bias, zero columns of w2) when both projections take that
            # route.
            if self.precision == "nvfp4" and self._nvfp4_eligible(
                    f"vis_w2_{i}", int(w2.shape[0]), VIS_FFN_PAD):
                w1 = F.pad(w1, (0, 0, 0, VIS_FFN_PAD - int(w1.shape[0])))
                b1 = F.pad(b1, (0, VIS_FFN_PAD - int(b1.shape[0])))
                w2 = F.pad(w2, (0, VIS_FFN_PAD - int(w2.shape[1])))
            elif self._vis_w2_pad:
                w1 = F.pad(w1, (0, 0, 0, VIS_FFN_PAD - int(w1.shape[0])))
                b1 = F.pad(b1, (0, VIS_FFN_PAD - int(b1.shape[0])))
                w2 = F.pad(w2, (0, VIS_FFN_PAD - int(w2.shape[1])))
            self._add_gemm(f"vis_w1_{i}", w1, b1)
            self._add_gemm(f"vis_w2_{i}", w2,
                           R.get(f"{p}.feed_forward.w2.bias"))

        # ── pooled adapter ──
        p = "model.vision_backbone.image_pooling_2d"
        for q in ("wq", "wk", "wv"):
            self._add_gemm(f"pool_{q[-1]}", R.get(f"{p}.{q}.weight"),
                           R.get(f"{p}.{q}.bias"))
        self._add_gemm("pool_o", R.get(f"{p}.wo.weight"),
                       R.get(f"{p}.wo.bias"))
        p = "model.vision_backbone.image_projector"
        self._add_gemm("proj_w1", R.get(f"{p}.w1.weight"))
        self._add_gemm("proj_w3", R.get(f"{p}.w3.weight"))
        self._add_gemm("proj_w2", R.get(f"{p}.w2.weight"))

        # ── text tower ──
        for i in range(TXT_LAYERS):
            p = f"model.transformer.blocks.{i}"
            self._add_gemm(f"txt_qkv_{i}", R.get(f"{p}.self_attn.att_proj.weight"))
            self._add_norm(f"txt_qnorm_{i}", R.get(f"{p}.self_attn.q_norm.weight"))
            self._add_norm(f"txt_knorm_{i}", R.get(f"{p}.self_attn.k_norm.weight"))
            self._add_gemm(f"txt_o_{i}", R.get(f"{p}.self_attn.attn_out.weight"))
            self._add_norm(f"txt_ln1_{i}", R.get(f"{p}.attn_norm.weight"))
            self._add_norm(f"txt_ln2_{i}", R.get(f"{p}.ff_norm.weight"))
            # HF computes silu(second_half) * first_half.  Store the gate
            # half first so the stride-aware merged SwiGLU kernel can be used.
            wf = R.get(f"{p}.mlp.ff_proj.weight")
            wf = torch.cat([wf[TXT_FFN:], wf[:TXT_FFN]], dim=0)
            self._add_gemm(f"txt_ff1_{i}", wf)
            self._add_gemm(f"txt_ff2_{i}", R.get(f"{p}.mlp.ff_out.weight"))

        # ── action expert ──
        self._add_gemm("act_ck",
                       R.get("model.action_expert.context_k_proj.weight"))
        self._add_gemm("act_cv",
                       R.get("model.action_expert.context_v_proj.weight"))
        for i in range(ACT_LAYERS):
            p = f"model.action_expert.blocks.{i}"
            self._add_gemm(f"act_qkv_{i}", R.get(f"{p}.self_attn.qkv.weight"),
                           R.get(f"{p}.self_attn.qkv.bias"))
            self._add_gemm(f"act_so_{i}", R.get(f"{p}.self_attn.out_proj.weight"),
                           R.get(f"{p}.self_attn.out_proj.bias"))
            self._add_gemm(f"act_cq_{i}", R.get(f"{p}.cross_attn.q_proj.weight"),
                           R.get(f"{p}.cross_attn.q_proj.bias"))
            self._add_gemm(f"act_co_{i}",
                           R.get(f"{p}.cross_attn.out_proj.weight"),
                           R.get(f"{p}.cross_attn.out_proj.bias"))
            self._add_gemm(f"act_up_{i}", R.get(f"{p}.mlp.up_proj.weight"),
                           R.get(f"{p}.mlp.up_proj.bias"))
            self._add_gemm(f"act_gate_{i}", R.get(f"{p}.mlp.gate_proj.weight"),
                           R.get(f"{p}.mlp.gate_proj.bias"))
            self._add_gemm(f"act_down_{i}", R.get(f"{p}.mlp.down_proj.weight"),
                           R.get(f"{p}.mlp.down_proj.bias"))
        self._add_gemm("act_embed",
                       R.get("model.action_expert.action_embed.weight"),
                       R.get("model.action_expert.action_embed.bias"))
        self._add_gemm("act_final",
                       R.get("model.action_expert.final_layer.linear.weight"),
                       R.get("model.action_expert.final_layer.linear.bias"))

        # Modulation / time-embedding weights are consumed host-side below.
        self._time_embed = [
            R.get("model.action_expert.time_embed.1.weight").float(),
            R.get("model.action_expert.time_embed.1.bias").float(),
            R.get("model.action_expert.time_embed.3.weight").float(),
            R.get("model.action_expert.time_embed.3.bias").float(),
        ]
        self._mod_weights = [
            (R.get(f"model.action_expert.blocks.{i}.modulation.linear.weight").float(),
             R.get(f"model.action_expert.blocks.{i}.modulation.linear.bias").float())
            for i in range(ACT_LAYERS)
        ]
        self._final_mod = (
            R.get("model.action_expert.final_layer.modulation.linear.weight").float(),
            R.get("model.action_expert.final_layer.modulation.linear.bias").float())

    def _build_embedding_table(self) -> int:
        emb = self._reader.get("model.transformer.wte.embedding")
        new = self._reader.get("model.transformer.wte.new_embedding")
        table = torch.cat([emb, new], 0).to(
            device=self.device, dtype=torch.bfloat16).contiguous()
        self._keep.append(table)
        return table.data_ptr()

    # ══════════════════════════════════════════════════════════════════
    #  Static prompt / precomputed tables
    # ══════════════════════════════════════════════════════════════════
    def _prompt_ids(self, task: str, state: np.ndarray):
        """Tokenise the production prompt and locate dynamic spans."""
        norm_state = self._normalize_state(state)
        state_str = _discrete_state_string(
            norm_state, int(self._num_state_tokens()))
        # The processor's single-crop ("resize") branch uses <im_start>.
        img_tokens = IM_START_TOKEN + IM_PATCH_TOKEN * POOL_TOKENS + IM_END_TOKEN
        text = _build_robot_text(
            task=_normalize_question_text(task),
            discrete_state_string=state_str,
            setup_type=self._stats["setup_type"],
            control_mode=self._stats["control_mode"],
            num_images=NUM_VIEWS,
        )
        text = text.replace(IMAGE_PROMPT, img_tokens)
        ids = self._tokenizer.encode(text).ids
        bos = 151645
        if not ids or ids[0] != bos:
            ids = [bos] + ids
        ids_arr = np.asarray(ids, dtype=np.int64)

        state_start = int(self._state_token_start)
        state_pos = np.nonzero(
            (ids_arr >= state_start) & (ids_arr < state_start + 256))[0]
        if state_pos.size != self._state_dim or not np.all(
                np.diff(state_pos) == 1):
            raise RuntimeError(
                "MolmoAct2 prompt state span is not a contiguous run of "
                f"{self._state_dim} tokens (found {state_pos.tolist()})")
        patch_id = int(self._image_patch_id)
        patch_pos = np.nonzero(ids_arr == patch_id)[0]
        if patch_pos.size != TOTAL_POOLED:
            raise RuntimeError(
                f"expected {TOTAL_POOLED} image patch tokens, got "
                f"{patch_pos.size}")
        starts = []
        for v in range(NUM_VIEWS):
            block = patch_pos[v * POOL_TOKENS:(v + 1) * POOL_TOKENS]
            if not np.all(np.diff(block) == 1):
                raise RuntimeError("image patch tokens are not contiguous")
            starts.append(int(block[0]))
        return ids_arr, int(state_pos[0]), starts

    def _num_state_tokens(self) -> int:
        return int(self._config["num_state_tokens"])

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        q01 = self._stats["state_q01"]
        q99 = self._stats["state_q99"]
        mode = self._stats["mode"]
        normed = 2.0 * (arr - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
        normed = np.clip(normed, -1.0, 1.0)
        mask = self._stats["state_mask"].astype(bool)
        normed = np.where(mask, normed, arr)
        normed = np.where(
            self._stats["state_zero_mask"].astype(bool), 0.0, normed)
        return normed.astype(np.float32)

    def _precompute(self, ids: np.ndarray, state_id_offset: int,
                    img_starts: Sequence[int]) -> dict:
        import ml_dtypes

        def bf16(a):
            return np.ascontiguousarray(
                np.asarray(a, dtype=np.float32).astype(ml_dtypes.bfloat16))

        pre: dict[str, Any] = {}
        # Expanded positional embedding: 3 identical 27x27 grids.
        pos = self._pos_emb.astype(np.float32)
        pos = np.tile(pos[None], (NUM_VIEWS, 1, 1)).reshape(
            TOTAL_PATCHES, VIS_D)
        pre["pos_emb_expanded"] = bf16(pos)

        # 2x2 pooling index map: pad 27x27 to 28x28 with -1 exactly like the
        # processor's arange_for_pooling (bottom/right padded).
        idx = np.arange(POS_TOKENS, dtype=np.int32).reshape(GRID, GRID)
        padded = np.full((POOL_GRID * POOL, POOL_GRID * POOL), -1, np.int32)
        padded[:GRID, :GRID] = idx
        base = padded.reshape(POOL_GRID, POOL, POOL_GRID, POOL).transpose(
            0, 2, 1, 3).reshape(POOL_TOKENS, POOL_WINDOW)
        # Flattened patch rows are image-major, so image v's window indices are
        # offset by v * POS_TOKENS (matches build_batched_images).
        pool_idx = np.concatenate(
            [np.where(base >= 0, base + v * POS_TOKENS, -1)
             for v in range(NUM_VIEWS)], axis=0)
        pre["pool_idx"] = np.ascontiguousarray(pool_idx.astype(np.int32))

        # Flow-matching modulation tables (static: fixed timesteps + weights).
        # Compute them with the same bf16 ops the HF module uses so the tables
        # are bit-identical, not merely close.
        steps = self.num_steps
        t = torch.arange(steps, dtype=torch.float32) / float(steps)
        half = 128
        freq = torch.exp(torch.arange(half, dtype=torch.float32)
                         * (-np.log(10000.0) / (half - 1)))
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        w1, b1, w2, b2 = (x.to(self.device, torch.bfloat16)
                          for x in self._time_embed)
        emb = emb.to(self.device, torch.bfloat16)
        cond = F.silu(F.linear(emb, w1, b1))
        cond = F.linear(cond, w2, b2)                        # [steps, D]

        def u16(tensor):
            return np.ascontiguousarray(
                tensor.view(torch.int16).cpu().numpy().view(np.uint16))

        mod_self = torch.zeros((steps, ACT_LAYERS, 3, ACT_D),
                               dtype=torch.bfloat16, device=self.device)
        mod_cross = torch.zeros_like(mod_self)
        mod_ffn = torch.zeros_like(mod_self)
        for li, (w, b) in enumerate(self._mod_weights):
            wb = w.to(self.device, torch.bfloat16)
            bb = b.to(self.device, torch.bfloat16)
            m = F.linear(F.silu(cond), wb, bb)
            chunks = torch.chunk(m, 9, dim=1)
            sec = {
                "self": (chunks[1], chunks[0], chunks[2]),
                "cross": (chunks[4], chunks[3], chunks[5]),
                "ffn": (chunks[7], chunks[6], chunks[8]),
            }
            for name, (scale, shift, gate) in sec.items():
                dst = {"self": mod_self, "cross": mod_cross,
                       "ffn": mod_ffn}[name]
                dst[:, li, 0] = scale
                dst[:, li, 1] = shift
                dst[:, li, 2] = gate
        pre["mod_self"] = u16(mod_self)
        pre["mod_cross"] = u16(mod_cross)
        pre["mod_ffn"] = u16(mod_ffn)

        fw, fb = self._final_mod
        fm = F.linear(F.silu(cond), fw.to(self.device, torch.bfloat16),
                      fb.to(self.device, torch.bfloat16))
        fchunks = torch.chunk(fm, 2, dim=1)       # (shift, scale)
        mod_final = torch.zeros((steps, 2, ACT_D), dtype=torch.bfloat16,
                                device=self.device)
        mod_final[:, 0] = fchunks[1]              # scale
        mod_final[:, 1] = fchunks[0]              # shift
        pre["mod_final"] = u16(mod_final)

        pre["noise"] = bf16(np.zeros(
            (self._stats["action_horizon"], ACT_MAX_DIM), np.float32))
        pre["state_q01"] = self._stats["state_q01"].astype(np.float32)
        pre["state_q99"] = self._stats["state_q99"].astype(np.float32)
        pre["state_mask"] = self._stats["state_mask"].astype(np.int8)
        pre["state_zero_mask"] = self._stats["state_zero_mask"].astype(np.int8)
        pre["act_q01"] = self._stats["action_q01"].astype(np.float32)
        pre["act_q99"] = self._stats["action_q99"].astype(np.float32)
        pre["act_mask"] = self._stats["action_mask"].astype(np.int8)
        pre["state_dim"] = self._state_dim
        pre["state_id_offset"] = int(state_id_offset)
        pre["num_state_tokens"] = self._num_state_tokens()
        pre["state_token_start_id"] = int(self._state_token_start)
        pre["img_row_start"] = np.asarray(img_starts, dtype=np.int64)
        return pre

    # ══════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════
    def set_prompt(self, task: str, *, state: Optional[np.ndarray] = None,
                   norm_tag: Optional[str] = None) -> None:
        """Build the static prompt for ``task`` and instantiate the pipeline."""
        if norm_tag is not None and norm_tag != self.norm_tag:
            self.norm_tag = str(norm_tag)
            self._stats = self._load_stats()
        cfg = json.loads((self.checkpoint_dir / "config.json").read_text())
        self._config = cfg
        self._state_token_start = int(cfg["state_token_start_id"])
        self._image_patch_id = int(cfg["image_patch_id"])
        self.task = str(task)
        if state is None:
            state = np.zeros(self._state_dim, np.float32)
        ids, state_off, img_starts = self._prompt_ids(self.task, state)
        self.prompt_len = int(ids.shape[0])
        pre = self._precompute(ids, state_off, img_starts)

        weights = {
            "bf16": self._bf16,
            "fp8": self._fp8,
            "bias": self._bias,
            "nvfp4": self._nvfp4,
            "skinny": self._skinny,
            "smallm": self._smallm,
            "swiglu_fold": self._swiglu_fold,
            "fp8_nk": self._fp8_nk,
            "act_fused": self._act_fused,
            "vis_w2_pad": self._vis_w2_pad,
            "embedding": self._embedding_ptr,
            "precomputed": pre,
        }
        self.pipeline = MolmoAct2Pipeline(
            fvk.GemmRunner(), fvk, fa2, weights,
            prompt_len=self.prompt_len, num_steps=self.num_steps,
            use_fp8=(self.precision in ("fp8", "nvfp4")),
            act_dim=int(self._stats["action_q01"].shape[0]),
            action_horizon=int(self._stats["action_horizon"]),
            vis_fused=bool(self._vis_fused_layers))
        ids_buf = self.pipeline.bufs["ids"]
        ids_np = np.ascontiguousarray(ids.astype(np.int64))
        ids_buf.upload(ids_np)
        self.pipeline.input_images_ptr = self._images_buf.ptr.value
        self.pipeline.host_images_ptr = self._host_images.data_ptr()
        self.pipeline.host_state_ptr = self._host_state.data_ptr()
        self.pipeline.host_noise_ptr = self._host_noise.data_ptr()
        self.pipeline.host_actions_ptr = self._host_actions.data_ptr()
        self._host_actions = torch.empty(
            self.pipeline.horizon, self.pipeline.act_dim,
            dtype=torch.float32, pin_memory=True)
        self._host_actions_np = self._host_actions.numpy()
        self.pipeline.host_actions_ptr = self._host_actions.data_ptr()
        logger.info("MolmoAct2 prompt len=%d images start=%s state@%d",
                    self.prompt_len, img_starts, state_off)

    def upload_observations(self, images_u8: np.ndarray,
                            state: np.ndarray,
                            noise: Optional[np.ndarray] = None) -> None:
        """Fill the pinned staging buffers (cheap CPU copies)."""
        img = np.ascontiguousarray(images_u8, dtype=np.uint8).reshape(-1)
        if img.size != self._host_images.numel():
            raise ValueError(
                f"expected {self._host_images.numel()} image bytes, got "
                f"{img.size}")
        self._host_images_np[:] = img
        st = np.ascontiguousarray(state, dtype=np.float32).reshape(-1)
        if st.size != self._state_dim:
            raise ValueError(
                f"expected state dim {self._state_dim}, got {st.size}")
        self._host_state_np[:] = st
        if noise is not None:
            import ml_dtypes
            nz = np.asarray(noise)
            if nz.shape != (self.pipeline.horizon, ACT_MAX_DIM):
                raise ValueError(
                    f"noise shape {nz.shape} != "
                    f"({self.pipeline.horizon}, {ACT_MAX_DIM})")
            nz = np.array(nz, dtype=np.float32, copy=True)
            # The reference masks padded action dims before the flow loop;
            # the same masking is applied here so the trajectory matches.
            nz[:, self.pipeline.act_dim:] = 0.0
            nz = np.ascontiguousarray(nz.astype(ml_dtypes.bfloat16))
            self._host_noise_np[:] = nz.view(np.uint16)

    def calibrate(self, samples: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        """Collect static FP8 activation scales from real observations."""
        if self.precision not in ("fp8", "nvfp4") or self._calibrated:
            self._calibrated = True
            return
        pipe = self.pipeline
        pipe.fp8_calibrated = False
        for images, state in samples:
            self.upload_observations(images, state)
            pipe.run_pipeline(stream=0)
        self._cudart.cudaStreamSynchronize(None)
        pipe.fp8_calibrated = True
        pipe.prepare_scale_host()
        pipe.prepare_skinny_scales()
        pipe.prepare_vis_gelu_inv_scales()
        pipe.prepare_action_kernel()
        self._calibrated = True
        logger.info("MolmoAct2 FP8 calibrated on %d real frames (%d scales)",
                    len(samples), len(pipe.fp8_scales))

    def record_infer_graph(self) -> None:
        if self.precision in ("fp8", "nvfp4") and not self._calibrated:
            raise RuntimeError("calibrate() must run before graph capture")
        self._with_stream = torch.cuda.stream(self._stream)
        self._with_stream.__enter__()
        try:
            self.pipeline.record_infer_graph(
                external_stream_int=self._stream.cuda_stream)
        finally:
            self._with_stream.__exit__(None, None, None)
        self.graph_recorded = True

    def run_eager(self, images_u8: np.ndarray, state: np.ndarray,
                  noise: Optional[np.ndarray] = None) -> dict:
        """Execute the pipeline without a captured graph (debug/validation).

        The full path still runs on one CUDA stream, including the pinned
        input copies and the action download.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt() first")
        self.upload_observations(images_u8, state, noise)
        stream_int = self._stream.cuda_stream
        with torch.cuda.stream(self._stream):
            self.pipeline.run_pipeline(stream=stream_int)
        self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream_int))
        return {"actions": self._host_actions_np.copy()}

    def infer(self, images_u8: np.ndarray, state: np.ndarray,
              noise: Optional[np.ndarray] = None) -> dict:
        if self.pipeline is None or not self.graph_recorded:
            raise RuntimeError("set_prompt() and record_infer_graph() first")
        self.upload_observations(images_u8, state, noise)
        stream_int = self._stream.cuda_stream
        self.pipeline.replay(stream_int)
        self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream_int))
        actions = self._host_actions_np.copy()
        return {"actions": actions}

    # ── introspection ──
    @property
    def precision_spec(self) -> dict:
        return {"precision": self.precision, "fp8_weights": len(self._fp8),
                "nvfp4_weights": len(self._nvfp4),
                "bf16_weights": len(self._bf16)}

    def state_dict_ptrs(self) -> dict:
        return {"bf16": dict(self._bf16), "fp8": dict(self._fp8),
                "bias": dict(self._bias)}
