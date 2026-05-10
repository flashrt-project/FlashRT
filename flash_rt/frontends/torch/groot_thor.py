"""FlashRT — GrootTorchFrontendThor: GROOT N1.6 inference via flash_rt_kernels.so.

Architecture: Eagle3-VL (SigLIP2 + Qwen3-1.7B) + AlternateVLDiT (32L, 4 flow-matching steps)

Usage:
    pipe = GrootTorchFrontendThor("/path/to/groot/checkpoint", num_views=2)
    pipe.set_prompt("pick up the red block")
    result = pipe.infer({"image": img1, "wrist_image": img2})
    actions = result["actions"]  # (16, action_dim) numpy
"""

import ctypes
import json
import logging
import math
import pathlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.hardware.thor.shared_primitives import siglip_forward  # SigLIP2 == Pi0.5 SigLIP
from flash_rt.hardware.thor.attn_backend_groot import (
    ThorGrootAttnBackend,
    make_groot_attention_spec,
)
from flash_rt.models.groot.pipeline_thor import (
    siglip2_forward,
    eagle_project,
    qwen3_forward,
    dit_forward,
    embodiment_encode_state,
    embodiment_encode_action,
    embodiment_decode_action,
)

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.quant.calibrator import load_calibration, save_calibration

logger = logging.getLogger(__name__)

fp16 = torch.float16
bf16 = torch.bfloat16
fp8 = torch.float8_e4m3fn

# ★ GROOT uses fp16 throughout (not bf16) — Thor fp16 has better perf ★
# SigLIP2/Qwen3: FP8 GEMM → fp16 intermediates (same as Pi0.5)
# DiT/mlp1/Embodiment: fp16 GEMM (fp16_nn) + fp16 kernels
# Kernel dtype rules:
#   layer_norm_fp16 = fp16,  layer_norm = bf16 (do not mix)
#   gelu_inplace_fp16 = fp16,  gelu_inplace = bf16
#   add_bias_fp16 = fp16 only
#   softmax_fp16 = fp16 (cols must be aligned, otherwise use PyTorch SDPA)
COMPUTE_DTYPE = fp16

# Embodiment tag → projector index. Shared with rtx — see
# flash_rt/hardware/groot_embodiments.py. Same 32-slot per-embodiment
# MLP layout; only a subset of slots have trained weights in the
# GR00T-N1.6-3B base checkpoint (see TRAINED_EMBODIMENT_IDS).
from flash_rt.models.groot.embodiments import (
    EMBODIMENT_TAG_TO_INDEX,
    PUBLIC_TRAINED_TAGS,
    is_embodiment_trained,
)

# ── Checkpoint key prefixes (C2: double vision_model!) ──
VIS_PREFIX = "backbone.model.vision_model.vision_model"  # ★ double vision_model ★
LLM_PREFIX = "backbone.model.language_model.model"
MLP1_PREFIX = "backbone.model.mlp1"
DIT_PREFIX = "action_head.model"
AH_PREFIX  = "action_head"


from flash_rt.core.thor_frontend_utils import quant_fp8  # noqa: E402


class GrootTorchFrontendThor:
    """GROOT N1.6 inference pipeline on Thor SM110."""

    def __init__(self, checkpoint, num_views=2, autotune=3,
                 embodiment_tag="new_embodiment", use_fp8=True):
        """Initialize GROOT pipeline.

        Args:
            checkpoint: path to GROOT safetensors checkpoint directory
            num_views: camera views (default 2)
            autotune: CUDA Graph autotune intensity (0=off, 3=default)
            embodiment_tag: target embodiment for per-embodiment MLPs
        """
        if embodiment_tag not in EMBODIMENT_TAG_TO_INDEX:
            raise ValueError(
                f"Unknown embodiment_tag {embodiment_tag!r}. "
                f"Known tags: {sorted(EMBODIMENT_TAG_TO_INDEX.keys())}. "
                f"Trained in GR00T-N1.6-3B: {PUBLIC_TRAINED_TAGS}.")
        self._checkpoint_path = pathlib.Path(checkpoint)
        self._num_views = num_views
        self._autotune = autotune
        self.use_fp8 = bool(use_fp8)
        self._embodiment_tag = embodiment_tag
        self._embodiment_id = EMBODIMENT_TAG_TO_INDEX[embodiment_tag]
        self._real_data_calibrated = False
        self.calibrated = False

        logger.info("GROOT pipeline init: checkpoint=%s, views=%d, embodiment=%s (id=%d)",
                     checkpoint, num_views, embodiment_tag, self._embodiment_id)
        if not is_embodiment_trained(embodiment_tag):
            logger.warning(
                "embodiment_tag=%r (id=%d) is NOT trained in the GR00T-N1.6-3B "
                "base checkpoint — per-embodiment MLP weights are at "
                "initialization and the model will emit noise-like actions. "
                "Pick one of %s for a demo, or fine-tune this slot before "
                "deployment.",
                embodiment_tag, self._embodiment_id, PUBLIC_TRAINED_TAGS,
            )

        # ── Init kernels ──
        self._gemm = fvk.GemmRunner()
        self._ctx = fvk.FvkContext()

        # ── Load FMHA ──
        fmha_path = pathlib.Path(fvk.__file__).parent / "libfmha_fp16_strided.so"
        if fmha_path.exists():
            fvk.load_fmha_library(str(fmha_path))
            fvk.load_fmha_strided_library(str(fmha_path))
            logger.info("FMHA loaded: standard=%s, strided=OK", fvk.has_cutlass_fmha())

        # ── Model dimensions (verified from checkpoint 2026-04-04) ──
        # SigLIP2 (27 layers, full attention, no RoPE)
        self.D_sig = 1152
        self.NH_sig = 16
        self.HD_sig = 72
        self.H_sig = 4304
        self.L_sig = 27
        self.spv_raw = 256   # raw patches per view (before pixel unshuffle)
        self.spv = 64        # tokens per view after 2x2 pixel_unshuffle (C4)
        self.mlp1_in = 4608  # 1152 * 4 after pixel unshuffle (C5)

        # Qwen3 (★16 layers★, select_layer=16, checkpoint truncated) (C1)
        self.D_llm = 2048
        self.NHQ = 16
        self.NHKV = 8
        self.HD_llm = 128
        self.H_llm = 6144
        self.L_llm = 16      # ★ NOT 28 — checkpoint only has 16 layers ★
        self.QKV_DIM = self.NHQ * self.HD_llm + 2 * self.NHKV * self.HD_llm  # 4096

        # DiT (32 layers, ★GELU not GEGLU★) (C3)
        self.D_dit = 1536
        self.NH_dit = 32
        self.HD_dit = 48
        self.H_dit = 6144    # ★ GELU FFN dim, NOT 2x for GEGLU ★
        self.L_dit = 32
        self.output_dim = 1024

        # Action (★padded max dims★) (C6, C12)
        self.action_dim = 128    # ★ NOT 29 — padded max, per-embodiment actual varies ★
        self.state_dim = 128     # ★ NOT 29 ★
        self.action_horizon = 50 # ★ NOT 16 — padded max ★
        self.num_steps = 4       # flow-matching Euler steps
        self.Sa = self.action_horizon + 1  # 51 = 1 state + 50 actions

        # ── Load checkpoint (keep full sd; SigLIP init deferred to _capture_all_graphs) ──
        self._load_checkpoint()

        logger.info("GROOT pipeline ready. GPU mem: %.1fGB free",
                     torch.cuda.mem_get_info()[0] / 1e9)

    # ─────────────────────────────────────────────────────────────
    # Weight loading
    # ─────────────────────────────────────────────────────────────

    def _load_checkpoint(self):
        """Load checkpoint to GPU. SigLIP init deferred to _capture_all_graphs."""
        from safetensors import safe_open

        ckpt_path = self._checkpoint_path
        st_files = sorted(ckpt_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors found in {ckpt_path}")

        logger.info("Loading %d safetensors files...", len(st_files))
        state_dict = {}
        for f in st_files:
            with safe_open(str(f), framework="pt", device="cuda") as sf:
                for key in sf.keys():
                    state_dict[key] = sf.get_tensor(key)
        logger.info("Loaded %d tensors", len(state_dict))

        # Extract embeddings (needed for set_prompt)
        self._qwen3_embed = state_dict[f"{LLM_PREFIX}.embed_tokens.weight"]
        self._vlln_w = state_dict["action_head.vlln.weight"]
        self._vlln_b = state_dict["action_head.vlln.bias"]
        # Keep full sd (SigLIP + Qwen3 + DiT all initialized in _capture_all_graphs)
        self._full_sd = state_dict

    def _load_siglip2_weights(self, sd):
        """Load SigLIP2 vision encoder weights.

        Key prefix: backbone.model.vision_model.vision_model.encoder.layers.{i}
        (C2: double vision_model in key path!)

        Structure per layer: LayerNorm → QKV merged → FMHA → O → LN → FFN up → FFN down
        All weights FP8 quantized (Q/K/V/O, fc1, fc2).
        """
        logger.info("Loading SigLIP2 weights (27 layers)...")

        D = self.D_sig     # 1152
        H = self.H_sig     # 4304
        L = self.L_sig     # 27

        # Declarative weight-loader pass (stage 7.8, partial). Populates:
        #   self._sig_{ln_attn,ln_ffn,qkv,o,up,down}_{w,b}  (27-layer lists)
        #   self._sig_alpha                                  (108 fp32 scales)
        # Qwen3 / DiT / embodiment remain inline — see _groot_thor_spec.py docstring.
        from flash_rt.executors.torch_weights import DictSource, WeightLoader
        from flash_rt.frontends.torch._groot_thor_spec import build_siglip2_spec
        WeightLoader(source=DictSource(sd), target=self,
                     spec=build_siglip2_spec()).run()

        # Post-layernorm (applied after SigLIP before pixel unshuffle)
        self._sig_post_ln_w = sd[f"{VIS_PREFIX}.post_layernorm.weight"].to(fp16)
        self._sig_post_ln_b = sd[f"{VIS_PREFIX}.post_layernorm.bias"].to(fp16)

        # Patch embedding: Linear [1152, 588] (not Conv2d!) — 588 = 14*14*3
        self._sig_patch_w = sd[f"{VIS_PREFIX}.embeddings.patch_embedding.weight"].to(fp16)  # [1152, 588]
        self._sig_patch_b = sd[f"{VIS_PREFIX}.embeddings.patch_embedding.bias"].to(fp16)    # [1152]

        # Position embedding: [256, 1152] — 256 patches, NO CLS (C15)
        self._sig_pos_embed = sd[f"{VIS_PREFIX}.embeddings.position_embedding.weight"].to(fp16)  # [256, 1152]

        # mlp1: LN(4608) → Linear(4608,2048) → GELU → Linear(2048,2048) (C5)
        self._mlp1_ln_w = sd[f"{MLP1_PREFIX}.0.weight"].to(fp16)    # [4608]
        self._mlp1_ln_b = sd[f"{MLP1_PREFIX}.0.bias"].to(fp16)      # [4608]
        self._mlp1_fc1_w = sd[f"{MLP1_PREFIX}.1.weight"].T.contiguous().to(fp16)  # [2048,4608]→[4608,2048] for NN
        self._mlp1_fc1_b = sd[f"{MLP1_PREFIX}.1.bias"].to(fp16)     # [2048]
        self._mlp1_fc2_w = sd[f"{MLP1_PREFIX}.3.weight"].T.contiguous().to(fp16)  # [2048,2048]→[2048,2048]
        self._mlp1_fc2_b = sd[f"{MLP1_PREFIX}.3.bias"].to(fp16)     # [2048]

        # Unit scale for FP8 casts
        self._unit_scale = torch.ones(1, dtype=torch.float32, device='cuda')

        # Build weight dicts (data pointers) — per-layer tensors are kept
        # alive as self._sig_*_{w,b} lists by the loader above.
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
            'unit_scale': self._unit_scale.data_ptr(),
        }

        logger.info("SigLIP2 weights loaded: %d layers, patch_embed [%s], pos_embed [%s], mlp1 ready",
                     L, list(self._sig_patch_w.shape), list(self._sig_pos_embed.shape))

    def _load_qwen3_weights(self, sd):
        """Load Qwen3-1.7B weights with FP8 quantization.

        Key differences from Pi0.5 Gemma:
        - ★16 layers★ (select_layer=16, checkpoint already truncated) (C1)
        - GQA 16Q/8KV: QKV merged = [D, 4096]
        - q_norm/k_norm weights: [128] per layer
        - Gate+Up merge: [D, 12288]
        - QKV interleave for RoPE compatibility
        """
        logger.info("Loading Qwen3 weights (%d layers)...", self.L_llm)
        prefix = f"{LLM_PREFIX}.layers"

        self._qwen3_w = {
            'ln_attn_w': [], 'qkv_w': [], 'qkv_s': [],
            'qkv_w_fp16': [],  # fp16 for fp16_nn path
            'q_norm_w': [], 'k_norm_w': [],
            'o_w': [], 'o_s': [],
            'o_w_fp16': [],
            'ln_ffn_w': [], 'gate_up_w': [], 'gate_up_s': [],
            'gate_up_w_fp16': [],
            'down_w': [], 'down_s': [],
            'down_w_fp16': [],
        }

        for i in range(self.L_llm):
            lp = f"{prefix}.{i}"

            # Input LayerNorm (RMSNorm, no bias)
            self._qwen3_w['ln_attn_w'].append(sd[f"{lp}.input_layernorm.weight"].to(fp16))

            # QKV: separate → merge [D, QKV_DIM]
            q_w = sd[f"{lp}.self_attn.q_proj.weight"]   # [2048, 2048]
            k_w = sd[f"{lp}.self_attn.k_proj.weight"]   # [1024, 2048]
            v_w = sd[f"{lp}.self_attn.v_proj.weight"]   # [1024, 2048]

            # Non-interleaved QKV (for PyTorch rotate_half RoPE)
            qkv_raw = torch.cat([q_w, k_w, v_w], dim=0)  # [4096, 2048]
            qkv_raw_T = qkv_raw.T.contiguous()  # [2048, 4096]
            self._qwen3_w['qkv_w_fp16'].append(qkv_raw_T.to(fp16))
            qkv_fp8_raw, qkv_s_raw = quant_fp8(qkv_raw_T)
            self._qwen3_w['qkv_w'].append(qkv_fp8_raw)  # FP8 non-interleaved
            self._qwen3_w['qkv_s'].append(qkv_s_raw)

            # Interleaved QKV (for future csrc rope_apply, not used now)
            # q_w_il = _interleave_qk(q_w, self.NHQ)
            # k_w_il = _interleave_qk(k_w, self.NHKV)

            # q_norm / k_norm weights [128]
            self._qwen3_w['q_norm_w'].append(sd[f"{lp}.self_attn.q_norm.weight"].to(fp16))
            self._qwen3_w['k_norm_w'].append(sd[f"{lp}.self_attn.k_norm.weight"].to(fp16))

            # O projection [2048, 2048]
            o_w = sd[f"{lp}.self_attn.o_proj.weight"]
            o_wT = o_w.T.contiguous()
            o_fp8, o_s = quant_fp8(o_wT)
            self._qwen3_w['o_w'].append(o_fp8)
            self._qwen3_w['o_s'].append(o_s)
            self._qwen3_w['o_w_fp16'].append(o_wT.to(fp16))

            # Post-attention RMSNorm
            self._qwen3_w['ln_ffn_w'].append(sd[f"{lp}.post_attention_layernorm.weight"].to(fp16))

            # FFN: gate_proj + up_proj → merged [D, 2H]
            gate_w = sd[f"{lp}.mlp.gate_proj.weight"]  # [6144, 2048]
            up_w = sd[f"{lp}.mlp.up_proj.weight"]      # [6144, 2048]
            gate_up = torch.cat([gate_w, up_w], dim=0)  # [12288, 2048]
            gu_T = gate_up.T.contiguous()  # [2048, 12288]
            gu_fp8, gu_s = quant_fp8(gu_T)
            self._qwen3_w['gate_up_w'].append(gu_fp8)
            self._qwen3_w['gate_up_s'].append(gu_s)
            self._qwen3_w['gate_up_w_fp16'].append(gu_T.to(fp16))

            # down_proj [2048, 6144]
            down_w = sd[f"{lp}.mlp.down_proj.weight"]  # [2048, 6144]
            dn_T = down_w.T.contiguous()  # [6144, 2048]
            dn_fp8, dn_s = quant_fp8(dn_T)
            self._qwen3_w['down_w'].append(dn_fp8)
            self._qwen3_w['down_s'].append(dn_s)
            self._qwen3_w['down_w_fp16'].append(dn_T.to(fp16))

        # Pre-allocate FP8 scale device tensors
        self._qwen3_w['qkv_s_dev'] = [torch.tensor([s], dtype=torch.float32, device='cuda')
                                       for s in self._qwen3_w['qkv_s']]
        self._qwen3_w['gate_up_s_dev'] = [torch.tensor([s], dtype=torch.float32, device='cuda')
                                           for s in self._qwen3_w['gate_up_s']]

        # Final RMSNorm
        self._qwen3_final_norm_w = sd[f"{LLM_PREFIX}.norm.weight"]

        # Token embeddings (for prompt encoding)
        self._qwen3_embed = sd[f"{LLM_PREFIX}.embed_tokens.weight"]  # [vocab, 2048]

        logger.info("Qwen3 weights loaded: %d layers, embed [%s]",
                     self.L_llm, list(self._qwen3_embed.shape))

    def _load_dit_weights(self, sd):
        """Load AlternateVLDiT weights.

        32 blocks with alternating self-attn (odd) / cross-attn (even).
        Self-attn blocks: to_k/to_v input=1536
        Cross-attn blocks: to_k/to_v input=2048 (from backbone)
        All DiT weights stay BF16 (FP8 optimization deferred).

        ★ CORRECTIONS (C3, C7, C8, C9, C11): ★
        - FFN is GELU, NOT GEGLU — ff.net.0.proj shape [6144, 1536] not [12288, 1536]
        - norm1.norm has NO parameters (elementwise_affine=False)
        - norm3 has NO parameters (elementwise_affine=False)
        - norm_out has NO parameters (elementwise_affine=False)
        - Output conditioning: shift, scale = chunk(2) — shift FIRST
        """
        logger.info("Loading DiT weights (32 layers)...")
        prefix = f"{DIT_PREFIX}.transformer_blocks"

        self._dit_w = {
            # AdaLayerNorm conditioning (no norm1.norm weights — C7)
            'norm1_linear_w': [], 'norm1_linear_b': [],
            # Attention (vary by block type) — fp16 for bias, FP8 for weights
            'q_w': [], 'q_b': [], 'q_w_fp8': [], 'q_s': [],
            'k_w': [], 'k_b': [], 'k_w_fp8': [], 'k_s': [],
            'v_w': [], 'v_b': [], 'v_w_fp8': [], 'v_s': [],
            'o_w': [], 'o_b': [], 'o_w_fp8': [], 'o_s': [],
            # For self-attn blocks: merged QKV
            'qkv_w': [], 'qkv_b': [], 'qkv_w_fp8': [], 'qkv_s': [],
            # FFN ★GELU★ (no norm3 — C8)
            'ff_up_w': [], 'ff_up_b': [], 'ff_up_w_fp8': [], 'ff_up_s': [],
            'ff_down_w': [], 'ff_down_b': [], 'ff_down_w_fp8': [], 'ff_down_s': [],
        }

        for l in range(self.L_dit):
            lp = f"{prefix}.{l}"
            is_self_attn = (l % 2 == 1)

            # AdaLayerNorm: only norm1.linear (conditioning projection)
            # norm1.norm has NO learnable parameters (elementwise_affine=False) (C7)
            # ★ All DiT weights converted to fp16 for Thor perf ★
            self._dit_w['norm1_linear_w'].append(
                sd[f"{lp}.norm1.linear.weight"].T.contiguous().to(fp16))  # [1536,3072]
            self._dit_w['norm1_linear_b'].append(
                sd[f"{lp}.norm1.linear.bias"].to(fp16))    # [3072]

            # Attention projections
            q_w = sd[f"{lp}.attn1.to_q.weight"]        # [1536, 1536]
            k_w = sd[f"{lp}.attn1.to_k.weight"]        # [1536, 1536] or [1536, 2048]
            v_w = sd[f"{lp}.attn1.to_v.weight"]        # same
            o_w = sd[f"{lp}.attn1.to_out.0.weight"]    # [1536, 1536]

            # fp16 weights (kept for non-FP8 path / bias)
            q_wT = q_w.T.contiguous()
            k_wT = k_w.T.contiguous()
            v_wT = v_w.T.contiguous()
            o_wT = o_w.T.contiguous()

            self._dit_w['q_w'].append(q_wT.to(fp16))
            self._dit_w['q_b'].append(sd[f"{lp}.attn1.to_q.bias"].to(fp16))
            self._dit_w['k_w'].append(k_wT.to(fp16))
            self._dit_w['k_b'].append(sd[f"{lp}.attn1.to_k.bias"].to(fp16))
            self._dit_w['v_w'].append(v_wT.to(fp16))
            self._dit_w['v_b'].append(sd[f"{lp}.attn1.to_v.bias"].to(fp16))
            self._dit_w['o_w'].append(o_wT.to(fp16))
            self._dit_w['o_b'].append(sd[f"{lp}.attn1.to_out.0.bias"].to(fp16))

            # FP8 quantized weights for GEMM
            q_fp8, q_s = quant_fp8(q_wT); self._dit_w['q_w_fp8'].append(q_fp8); self._dit_w['q_s'].append(q_s)
            k_fp8, k_s = quant_fp8(k_wT); self._dit_w['k_w_fp8'].append(k_fp8); self._dit_w['k_s'].append(k_s)
            v_fp8, v_s = quant_fp8(v_wT); self._dit_w['v_w_fp8'].append(v_fp8); self._dit_w['v_s'].append(v_s)
            o_fp8, o_s = quant_fp8(o_wT); self._dit_w['o_w_fp8'].append(o_fp8); self._dit_w['o_s'].append(o_s)

            # Self-attn blocks: merge QKV for single GEMM
            if is_self_attn:
                qkv_merged = torch.cat([q_w, k_w, v_w], dim=0)  # [4608, 1536]
                qkv_mT = qkv_merged.T.contiguous()
                self._dit_w['qkv_w'].append(qkv_mT.to(fp16))
                qkv_bias = torch.cat([sd[f"{lp}.attn1.to_q.bias"],
                                      sd[f"{lp}.attn1.to_k.bias"],
                                      sd[f"{lp}.attn1.to_v.bias"]], dim=0).to(fp16)
                self._dit_w['qkv_b'].append(qkv_bias)
                qkv_fp8, qkv_s = quant_fp8(qkv_mT)
                self._dit_w['qkv_w_fp8'].append(qkv_fp8)
                self._dit_w['qkv_s'].append(qkv_s)
            else:
                self._dit_w['qkv_w'].append(None)
                self._dit_w['qkv_b'].append(None)
                self._dit_w['qkv_w_fp8'].append(None)
                self._dit_w['qkv_s'].append(None)

            # FFN ★GELU★ (NOT GEGLU) (C3) — no norm3 params (C8)
            ff_up = sd[f"{lp}.ff.net.0.proj.weight"]    # [6144, 1536]
            ff_down = sd[f"{lp}.ff.net.2.weight"]        # [1536, 6144]
            ff_up_T = ff_up.T.contiguous()
            ff_down_T = ff_down.T.contiguous()
            self._dit_w['ff_up_w'].append(ff_up_T.to(fp16))
            self._dit_w['ff_up_b'].append(sd[f"{lp}.ff.net.0.proj.bias"].to(fp16))
            self._dit_w['ff_down_w'].append(ff_down_T.to(fp16))
            self._dit_w['ff_down_b'].append(sd[f"{lp}.ff.net.2.bias"].to(fp16))
            fu_fp8, fu_s = quant_fp8(ff_up_T); self._dit_w['ff_up_w_fp8'].append(fu_fp8); self._dit_w['ff_up_s'].append(fu_s)
            fd_fp8, fd_s = quant_fp8(ff_down_T); self._dit_w['ff_down_w_fp8'].append(fd_fp8); self._dit_w['ff_down_s'].append(fd_s)

        # Output layers — norm_out has NO parameters (C9)
        self._dit_proj_out_1_w = sd[f"{DIT_PREFIX}.proj_out_1.weight"].T.contiguous().to(fp16)
        self._dit_proj_out_1_b = sd[f"{DIT_PREFIX}.proj_out_1.bias"].to(fp16)
        self._dit_proj_out_2_w = sd[f"{DIT_PREFIX}.proj_out_2.weight"].T.contiguous().to(fp16)
        self._dit_proj_out_2_b = sd[f"{DIT_PREFIX}.proj_out_2.bias"].to(fp16)

        # Timestep encoder — fp16
        ts_prefix = f"{DIT_PREFIX}.timestep_encoder.timestep_embedder"
        self._ts_linear1_w = sd[f"{ts_prefix}.linear_1.weight"].T.contiguous().to(fp16)
        self._ts_linear1_b = sd[f"{ts_prefix}.linear_1.bias"].to(fp16)
        self._ts_linear2_w = sd[f"{ts_prefix}.linear_2.weight"].T.contiguous().to(fp16)
        self._ts_linear2_b = sd[f"{ts_prefix}.linear_2.bias"].to(fp16)

        # vlln — fp16
        self._vlln_w = sd[f"{AH_PREFIX}.vlln.weight"].to(fp16)
        self._vlln_b = sd[f"{AH_PREFIX}.vlln.bias"].to(fp16)

        # Position embedding — fp16
        self._position_embedding = sd[f"{AH_PREFIX}.position_embedding.weight"].to(fp16)

        logger.info("DiT weights loaded: %d layers, pos_embed [%s]",
                     self.L_dit, list(self._position_embedding.shape))

    def _load_embodiment_weights(self, sd):
        """Extract per-embodiment weights for target embodiment.

        ★ CORRECTIONS (C6, C10, C13, C14): ★
        - action_dim=128, state_dim=128 (padded max, not 29)
        - action_encoder hidden=1536 (not 1024)
        - W layout is [in, out] — NO transpose needed for bmm(x, W) (C13)
        - mask_token doesn't exist (state_dropout_prob=0.0) (C14)

        CategorySpecificLinear: W=[num_categories, input_dim, hidden_dim]
        Forward: output = bmm(x, W[cat_id]) + b[cat_id]
        For bf16_nn: A=[M,K], B=[K,N] → B=W[eid] which is already [in, out] = [K, N] ✓
        """
        eid = self._embodiment_id
        logger.info("Extracting embodiment weights for '%s' (id=%d)", self._embodiment_tag, eid)

        # State encoder: layer1 [128→1024] + ReLU + layer2 [1024→1536]
        # W[eid] is already [in, out] = [K, N] for fp16_nn — NO transpose! (C13)
        # ★ All embodiment weights → fp16 ★
        self._state_enc_w1 = sd[f"{AH_PREFIX}.state_encoder.layer1.W"][eid].contiguous().to(fp16)  # [128, 1024]
        self._state_enc_b1 = sd[f"{AH_PREFIX}.state_encoder.layer1.b"][eid].to(fp16)                # [1024]
        self._state_enc_w2 = sd[f"{AH_PREFIX}.state_encoder.layer2.W"][eid].contiguous().to(fp16)  # [1024, 1536]
        self._state_enc_b2 = sd[f"{AH_PREFIX}.state_encoder.layer2.b"][eid].to(fp16)                # [1536]

        # Action encoder: W1 [128→1536], W2 [3072→1536] (concat action+time), W3 [1536→1536]
        self._action_enc_w1 = sd[f"{AH_PREFIX}.action_encoder.W1.W"][eid].contiguous().to(fp16)
        self._action_enc_b1 = sd[f"{AH_PREFIX}.action_encoder.W1.b"][eid].to(fp16)
        self._action_enc_w2 = sd[f"{AH_PREFIX}.action_encoder.W2.W"][eid].contiguous().to(fp16)
        self._action_enc_b2 = sd[f"{AH_PREFIX}.action_encoder.W2.b"][eid].to(fp16)
        self._action_enc_w3 = sd[f"{AH_PREFIX}.action_encoder.W3.W"][eid].contiguous().to(fp16)
        self._action_enc_b3 = sd[f"{AH_PREFIX}.action_encoder.W3.b"][eid].to(fp16)

        # Action decoder: layer1 [1024→1024] + ReLU + layer2 [1024→128]
        self._action_dec_w1 = sd[f"{AH_PREFIX}.action_decoder.layer1.W"][eid].contiguous().to(fp16)
        self._action_dec_b1 = sd[f"{AH_PREFIX}.action_decoder.layer1.b"][eid].to(fp16)
        self._action_dec_w2 = sd[f"{AH_PREFIX}.action_decoder.layer2.W"][eid].contiguous().to(fp16)
        self._action_dec_b2 = sd[f"{AH_PREFIX}.action_decoder.layer2.b"][eid].to(fp16)

        # No mask_token — state_dropout_prob=0.0 (C14)

        logger.info("Embodiment weights extracted: state[128→1024→1536], "
                     "action_enc[128→1536, 3072→1536, 1536→1536], "
                     "action_dec[1024→1024→128]")

    # ─────────────────────────────────────────────────────────────
    # Precompute
    # ─────────────────────────────────────────────────────────────

    def _precompute_rope(self):
        """Precompute Qwen3 RoPE weights (theta=1e6, HD=128)."""
        theta = 1000000.0
        max_seq = 2048
        HD = self.HD_llm
        freqs = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float32, device='cuda') / HD))
        positions = torch.arange(max_seq, dtype=torch.float32, device='cuda')
        angles = torch.outer(positions, freqs)  # [max_seq, HD//2]

        # For Qwen3 rotate_half RoPE: cos/sin broadcasted to full HD
        self._rope_cos_cache = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(fp16)  # [max_seq, HD]
        self._rope_sin_cache = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(fp16)

        # Legacy format for csrc rope_apply (pair-interleaved)
        self._rope_weights = torch.cat([
            torch.cos(angles), torch.sin(angles)
        ], dim=-1).to(fp16)
        logger.info("RoPE precomputed: theta=%.0e, max_seq=%d, HD=%d", theta, max_seq, HD)

    def _precompute_timesteps(self):
        """Precompute timestep embeddings for 4 flow-matching steps.

        For each step t ∈ {0, 1, 2, 3}:
          t_cont = t / 4.0  → {0.0, 0.25, 0.5, 0.75}
          t_disc = int(t_cont * 1000)  → {0, 250, 500, 750}

        Pipeline:
          t_disc → Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)
          → sinusoidal [256]
          → linear_1 [256→1536] → SiLU → linear_2 [1536→1536]
          → temb [1536]

        Per-block AdaLN conditioning (precomputed for all 32 layers × 4 steps):
          SiLU(temb) → norm1.linear [1536→3072] → split → (shift, scale) (C11: shift first!)

        Final output conditioning (precomputed for 4 steps):
          SiLU(temb) → proj_out_1 [1536→3072] → split → (shift, scale)
        """
        with torch.no_grad():
            D = self.D_dit  # 1536

            # Step 1: Sinusoidal time encoding (matches diffusers Timesteps)
            # flip_sin_to_cos=True, downscale_freq_shift=1
            half_dim = 128  # 256 / 2
            exponent = -torch.arange(half_dim, dtype=torch.float32, device='cuda') * \
                       (math.log(10000.0) / half_dim)
            emb_freqs = exponent.exp()  # [128]

            t_values = [0, 250, 500, 750]
            self._tembs = []        # [4, D] — one temb per step
            self._ada_scales = []   # [4, L, D] — per-step per-layer scale
            self._ada_shifts = []   # [4, L, D]
            self._out_scales = []   # [4, D]
            self._out_shifts = []   # [4, D]

            for t_disc in t_values:
                # Sinusoidal encoding (flip_sin_to_cos=True, downscale_freq_shift=1)
                t_tensor = torch.tensor([t_disc], dtype=torch.float32, device='cuda')
                args = t_tensor[:, None] * emb_freqs[None, :]  # [1, 128]
                sincos = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [1, 256]

                # TimestepEmbedding: linear_1 → SiLU → linear_2
                temb = sincos.to(fp16) @ self._ts_linear1_w + self._ts_linear1_b  # [1, 1536]
                temb = F.silu(temb)
                temb = temb @ self._ts_linear2_w + self._ts_linear2_b  # [1, 1536]
                self._tembs.append(temb.squeeze(0))  # [1536]

                # Per-layer AdaLN: SiLU(temb) → linear → split(scale, shift)
                # ★ AdaLayerNorm: scale FIRST, shift SECOND (line 80 in dit.py)
                # ★ Output conditioning: shift FIRST, scale SECOND (line 394)
                silu_temb = F.silu(temb)  # [1, 1536]
                layer_scales = []
                layer_shifts = []
                for l in range(self.L_dit):
                    ada_out = silu_temb @ self._dit_w['norm1_linear_w'][l]  # [1, 3072]
                    ada_out = ada_out + self._dit_w['norm1_linear_b'][l]
                    scale, shift = ada_out.squeeze(0).chunk(2, dim=0)  # ★ scale first! ★
                    layer_scales.append(scale)
                    layer_shifts.append(shift)
                self._ada_scales.append(torch.stack(layer_scales))  # [L, D]
                self._ada_shifts.append(torch.stack(layer_shifts))  # [L, D]

                # Final output conditioning: SiLU(temb) → proj_out_1 → split(shift, scale)
                out_cond = silu_temb @ self._dit_proj_out_1_w + self._dit_proj_out_1_b  # [1, 3072]
                out_shift, out_scale = out_cond.squeeze(0).chunk(2, dim=0)  # ★ shift first ★
                self._out_scales.append(out_scale)
                self._out_shifts.append(out_shift)

            # Stack for efficient indexing
            self._tembs = torch.stack(self._tembs)           # [4, D]
            self._ada_scales = torch.stack(self._ada_scales) # [4, L, D]
            self._ada_shifts = torch.stack(self._ada_shifts) # [4, L, D]
            self._out_scales = torch.stack(self._out_scales) # [4, D]
            self._out_shifts = torch.stack(self._out_shifts) # [4, D]

        logger.info("Timestep embeddings precomputed: 4 steps × %d layers, temb[%s]",
                     self.L_dit, list(self._tembs.shape))

    def _allocate_buffers(self):
        """Allocate GPU buffers for pipeline."""
        nv = self._num_views
        S_sig = nv * self.spv_raw  # 2 * 256 = 512 (raw SigLIP patches)
        S_img = nv * self.spv      # 2 * 64 = 128 (after pixel unshuffle)
        D_sig = self.D_sig         # 1152
        D_llm = self.D_llm        # 2048
        D_dit = self.D_dit         # 1536
        H_sig = self.H_sig        # 4304
        Sa = self.Sa               # 51

        # ── SigLIP2 buffers ──
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

        # ── mlp1 buffer (after pixel unshuffle) ──
        self._mlp1_in = torch.empty(S_img, self.mlp1_in, dtype=fp16, device='cuda')   # [128, 4608]
        self._mlp1_mid = torch.empty(S_img, D_llm, dtype=fp16, device='cuda')         # [128, 2048]
        self._vision_features = torch.empty(S_img, D_llm, dtype=fp16, device='cuda')  # [128, 2048]

        # ── Qwen3/backbone placeholder (Se depends on prompt_len) ──
        # Allocated dynamically in set_prompt()

        # DiT/Qwen3 buffers managed by CKernel classes (created in _capture_all_graphs)

        logger.info("SigLIP buffers allocated. S_sig=%d, S_img=%d", S_sig, S_img)

    # ─────────────────────────────────────────────────────────────
    # Kernel helpers (validated E2E cos=0.999975)
    # ─────────────────────────────────────────────────────────────

    def _fp16_gemm(self, A, B, M, N, K):
        """fp16_nn wrapper: C[M,N] = A[M,K] @ B[K,N]. All fp16."""
        C = torch.empty(M, N, dtype=fp16, device='cuda')
        self._gemm.fp16_nn(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, 0)
        return C

    def _fp8_gemm(self, A_fp16, B_fp8, w_scale_ptr, M, N, K):
        """FP8 GEMM with pre-quantized weight + pre-allocated scale ptr.
        Quantizes activation on-the-fly.
        """
        A_fp8 = self._fp8_act_buf  # pre-allocated in _allocate_buffers
        fvk.quantize_fp8_static_fp16(
            A_fp16.data_ptr(), A_fp8.data_ptr(),
            self._unit_scale.data_ptr(), M * K, 0)
        C = torch.empty(M, N, dtype=fp16, device='cuda')
        self._gemm.fp8_descale_fp16(
            A_fp8.data_ptr(), B_fp8.data_ptr(), C.data_ptr(),
            M, N, K,
            self._unit_scale.data_ptr(), w_scale_ptr, 0)
        return C

    def _sinusoidal_time_embed(self, timesteps, dim=1536):
        """Sinusoidal positional encoding for action encoder."""
        half_dim = dim // 2
        exp = -torch.arange(half_dim, dtype=torch.float, device='cuda') * \
              (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1).float() * exp.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

    def _timestep_encode(self, t_disc):
        """Timesteps(256) → TimestepEmbedding → temb [1, 1536]."""
        half_dim = 128
        exp = -torch.arange(half_dim, dtype=torch.float32, device='cuda') * \
              (math.log(10000.0) / half_dim)
        t_tensor = torch.tensor([t_disc], dtype=torch.float32, device='cuda')
        args = t_tensor[:, None] * exp.exp()
        sincos = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(fp16)
        temb = F.silu(sincos @ self._ts_linear1_w + self._ts_linear1_b)
        temb = temb @ self._ts_linear2_w + self._ts_linear2_b
        return temb  # [1, 1536]

    def _state_encode(self, state):
        """[1, state_dim] → [1, 1, D_dit]. Embodiment-specific MLP."""
        state = state.to(fp16).contiguous()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = F.relu(self._fp16_gemm(state, self._state_enc_w1, 1, 1024, self.state_dim)
                   + self._state_enc_b1)
        h = self._fp16_gemm(h, self._state_enc_w2, 1, self.D_dit, 1024) + self._state_enc_b2
        return h.unsqueeze(0)  # [1, 1, D_dit]

    def _action_encode(self, actions, t_disc, action_horizon):
        """[1, T, action_dim] + timestep → [1, T, D_dit]."""
        T = action_horizon
        D = self.D_dit
        actions_2d = actions.squeeze(0).to(fp16).contiguous()

        a_emb = self._fp16_gemm(actions_2d, self._action_enc_w1, T, D, self.action_dim) \
                + self._action_enc_b1
        t_expanded = torch.full((T,), t_disc, device='cuda')
        time_emb = self._sinusoidal_time_embed(t_expanded, D).to(fp16)
        concat = torch.cat([a_emb, time_emb], dim=-1)  # [T, 2*D]
        h = F.silu(self._fp16_gemm(concat, self._action_enc_w2, T, D, 2 * D)
                   + self._action_enc_b2)
        h = self._fp16_gemm(h, self._action_enc_w3, T, D, D) + self._action_enc_b3
        pos_ids = torch.arange(T, device='cuda')
        h = h + self._position_embedding[pos_ids]
        return h.unsqueeze(0)  # [1, T, D]

    def _action_decode(self, model_output, action_horizon):
        """[1, Sa, output_dim] → velocity [1, T, action_dim]."""
        Sa = model_output.shape[1]
        x = model_output.squeeze(0).to(fp16)
        h = F.relu(self._fp16_gemm(x, self._action_dec_w1, Sa, 1024, self.output_dim)
                   + self._action_dec_b1)
        pred = self._fp16_gemm(h, self._action_dec_w2, Sa, self.action_dim, 1024) \
               + self._action_dec_b2
        return pred[-action_horizon:].unsqueeze(0)

    def _dit_layer(self, hidden, l, temb, is_self, backbone=None, attn_mask=None):
        """Single DiT layer: AdaLN → Attention → Residual → LN → GELU FFN → Residual.

        Uses FP8 GEMM for large matrix ops (QKV merged, FFN up/down, O proj).
        Small ops (bias, LN, GELU, SDPA) remain fp16.
        """
        D = self.D_dit
        NH, HD = self.NH_dit, self.HD_dit
        H = self.H_dit
        B, S_q = hidden.shape[0], hidden.shape[1]
        w = self._dit_w

        # AdaLayerNorm: LN(x, no params) * (1+scale) + shift
        ada_out = F.silu(temb) @ w['norm1_linear_w'][l] + w['norm1_linear_b'][l]
        scale, shift = ada_out.chunk(2, dim=-1)  # scale first (AdaLayerNorm)
        h_norm = F.layer_norm(hidden.float(), [D], eps=1e-5).to(fp16)
        h_norm = h_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Attention — FP8 for large GEMMs (QKV merged, FFN), fp16 for small (Q/K/V cross, O)
        h_norm_2d = h_norm.squeeze(0).contiguous()
        if is_self:
            # Self-attn: merged QKV is large (S×4608) → FP8
            qkv = self._fp8_gemm(h_norm_2d, w['qkv_w_fp8'][l], w['qkv_s'][l].data_ptr(), S_q, 3*D, D)
            qkv = qkv + w['qkv_b'][l]
            q = qkv[:, :D].view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
            k = qkv[:, D:2*D].view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
            v = qkv[:, 2*D:].view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v)
        else:
            # Cross-attn: Q/K/V separate, small M → keep fp16
            q = self._fp16_gemm(h_norm_2d, w['q_w'][l], S_q, D, D) + w['q_b'][l]
            kv_src = backbone.to(fp16)
            if attn_mask is not None:
                kv_src = kv_src * attn_mask.unsqueeze(-1).to(fp16)
            S_kv, D_kv = kv_src.shape[1], kv_src.shape[2]
            kv_2d = kv_src.squeeze(0).contiguous()
            k = self._fp16_gemm(kv_2d, w['k_w'][l], S_kv, D, D_kv) + w['k_b'][l]
            v = self._fp16_gemm(kv_2d, w['v_w'][l], S_kv, D, D_kv) + w['v_b'][l]
            q = q.view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
            k = k.view(S_kv, NH, HD).unsqueeze(0).transpose(1, 2)
            v = v.view(S_kv, NH, HD).unsqueeze(0).transpose(1, 2)
            amask = None
            if attn_mask is not None:
                amask = attn_mask.unsqueeze(1).unsqueeze(2).expand(B, NH, S_q, S_kv)
                amask = torch.where(amask, 0.0, float('-inf')).to(fp16)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)

        attn = attn.transpose(1, 2).reshape(B, S_q, D).to(fp16)
        attn_2d = attn.squeeze(0).contiguous()
        o = self._fp16_gemm(attn_2d, w['o_w'][l], S_q, D, D) + w['o_b'][l]
        hidden = hidden.float() + o.unsqueeze(0).float()

        # FFN: LN(no params) → GELU → down — FP8 for large FFN GEMMs
        ff_norm = F.layer_norm(hidden, [D], eps=1e-5).to(fp16)
        ff_norm_2d = ff_norm.squeeze(0).contiguous()
        ff_h = self._fp8_gemm(ff_norm_2d, w['ff_up_w_fp8'][l], w['ff_up_s'][l].data_ptr(), S_q, H, D)
        ff_h = ff_h + w['ff_up_b'][l]
        fvk.gelu_inplace_fp16(ff_h.data_ptr(), S_q * H, 0)
        ff_out = self._fp8_gemm(ff_h, w['ff_down_w_fp8'][l], w['ff_down_s'][l].data_ptr(), S_q, D, H)
        ff_out = ff_out + w['ff_down_b'][l]
        hidden = (hidden + ff_out.unsqueeze(0).float()).to(fp16)
        return hidden

    def _dit_forward(self, sa_embs, backbone_features, image_mask, backbone_mask, temb):
        """Full 32-layer DiT forward for a single flow-matching step."""
        D = self.D_dit
        non_image_mask = (~image_mask) & backbone_mask
        image_attn_mask = image_mask & backbone_mask

        hidden = sa_embs
        for l in range(self.L_dit):
            is_self = (l % 2 == 1)
            if is_self:
                hidden = self._dit_layer(hidden, l, temb, True)
            else:
                curr_mask = non_image_mask if l % 4 == 0 else image_attn_mask
                hidden = self._dit_layer(hidden, l, temb, False,
                                         backbone_features, curr_mask)

        # Output conditioning: shift first, scale second
        out_cond = F.silu(temb) @ self._dit_proj_out_1_w + self._dit_proj_out_1_b
        out_shift, out_scale = out_cond.chunk(2, dim=-1)
        hidden = F.layer_norm(hidden.float(), [D], eps=1e-6).to(fp16)
        hidden = hidden * (1 + out_scale.unsqueeze(1)) + out_shift.unsqueeze(1)

        Sa = hidden.shape[1]
        output = self._fp16_gemm(hidden.squeeze(0), self._dit_proj_out_2_w, Sa, self.output_dim, D)
        return output.unsqueeze(0)  # [1, Sa, output_dim]

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def set_prompt(self, prompt):
        """Tokenize prompt and prepare text embeddings for Qwen3 backbone."""
        from transformers import AutoTokenizer

        if not hasattr(self, '_tokenizer'):
            eagle_dir = pathlib.Path(__file__).parent.parent.parent.parent / "configs"
            # Try multiple tokenizer locations
            for tok_path in [
                str(self._checkpoint_path),  # checkpoint dir may have tokenizer
                str(self._checkpoint_path / "tokenizer"),  # subfolder
                # Local GROOT code Eagle dir
                str(pathlib.Path(__file__).parent.parent.parent.parent.parent /
                    "GR00T" / "Isaac-GR00T" / "gr00t" / "model" / "modules" / "nvidia" / "Eagle-Block2A-2B-v2"),
                "nvidia/Eagle-Block2A-2B-v2",  # HF hub (fallback)
            ]:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
                    break
                except Exception:
                    continue
            if not hasattr(self, '_tokenizer'):
                raise RuntimeError("Cannot load Qwen3 tokenizer")
            self._img_token_id = 151669   # <IMG_CONTEXT>
            self._img_start_id = 151670   # <img>
            self._img_end_id = 151671     # </img>

        S_img = self._num_views * self.spv  # image tokens after pixel unshuffle
        text_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        # Build: text + <img> + <IMG_CONTEXT>*S_img + </img>
        full_ids = text_ids + [self._img_start_id] + [self._img_token_id] * S_img + [self._img_end_id]

        self._input_ids = torch.tensor([full_ids], dtype=torch.long, device='cuda')
        self._text_len = len(text_ids)
        self._Se = len(full_ids)
        self._prompt_text = prompt

        # Pre-compute text embeddings (text portion only; image tokens replaced at infer time)
        self._text_embeds = F.embedding(self._input_ids, self._qwen3_embed)  # [1, Se, 2048]

        # Masks
        self._image_mask = (self._input_ids == self._img_token_id)  # [1, Se]
        self._backbone_mask = torch.ones(1, self._Se, dtype=torch.bool, device='cuda')

        logger.info("Prompt set: '%s' (%d text + %d img = %d total tokens)",
                     prompt[:50], self._text_len, S_img, self._Se)

    def infer_action_head(self, backbone_features, image_mask, backbone_mask,
                          state, action_horizon=None, noise_seed=None):
        """Run GROOT action head inference (DiT + embodiment MLPs).

        Assumes backbone features are already computed (SigLIP2 + Qwen3).
        This is the validated E2E path (cos=0.999975 vs PyTorch reference).

        Args:
            backbone_features: [1, Se, 2048] — output of vlln(Qwen3(SigLIP2+text))
            image_mask: [1, Se] boolean — True for image tokens
            backbone_mask: [1, Se] boolean — True for valid tokens
            state: [state_dim] or [1, state_dim] tensor
            action_horizon: number of action steps (default: self.action_horizon)
            noise_seed: random seed for initial noise (default: None = random)

        Returns:
            actions: [1, action_horizon, action_dim] tensor
        """
        if action_horizon is None:
            action_horizon = self.action_horizon
        B = 1
        D = self.D_dit

        # State encoding
        state_feat = self._state_encode(state)  # [1, 1, D_dit]

        # Init noise
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
        actions = torch.randn(B, action_horizon, self.action_dim,
                              dtype=torch.float32, device='cuda')
        dt = 1.0 / self.num_steps

        # 4-step flow matching
        for step in range(self.num_steps):
            t_cont = step / float(self.num_steps)
            t_disc = int(t_cont * 1000)

            temb = self._timestep_encode(t_disc)
            action_feat = self._action_encode(actions, t_disc, action_horizon)
            sa_embs = torch.cat([state_feat, action_feat], dim=1)  # [1, 1+T, D_dit]

            model_output = self._dit_forward(
                sa_embs, backbone_features, image_mask, backbone_mask, temb)

            velocity = self._action_decode(model_output, action_horizon)
            actions = actions + dt * velocity.float()

        return actions

    # ─────────────────────────────────────────────────────────────
    # CUDA Graph (OPT-2)
    # ─────────────────────────────────────────────────────────────

    def _precompute_action_time_embeds(self, action_horizon):
        """Precompute sinusoidal time embeddings for all 4 flow-matching steps."""
        D = self.D_dit
        half_dim = D // 2
        exp = -torch.arange(half_dim, dtype=torch.float, device='cuda') * \
              (math.log(10000.0) / half_dim)
        exp_table = exp.exp()  # [768]

        self._action_time_embeds = []  # [4, T, D]
        for step in range(self.num_steps):
            t_disc = int(step / float(self.num_steps) * 1000)
            t_expanded = torch.full((action_horizon,), t_disc, device='cuda')
            freqs = t_expanded.unsqueeze(-1).float() * exp_table
            te = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).to(fp16)
            self._action_time_embeds.append(te)
        self._action_time_embeds = torch.stack(self._action_time_embeds)  # [4, T, D]

    def _graph_dit_step(self, step_idx, actions_buf, state_feat_buf,
                        backbone_buf, non_img_mask, img_mask):
        """Single flow-matching step — designed for CUDA Graph capture.
        No tensor allocation, no Python control flow dependent on data.
        """
        D = self.D_dit
        T = actions_buf.shape[1]
        dt = 1.0 / self.num_steps

        temb = self._tembs[step_idx].unsqueeze(0)  # [1, D] precomputed

        # Action encoding (all pre-allocated buffers)
        actions_2d = actions_buf.squeeze(0).to(fp16)
        a_emb = self._fp16_gemm(actions_2d, self._action_enc_w1, T, D, self.action_dim)
        a_emb = a_emb + self._action_enc_b1
        time_emb = self._action_time_embeds[step_idx]  # [T, D] precomputed
        concat = torch.cat([a_emb, time_emb], dim=-1)
        h = F.silu(self._fp16_gemm(concat, self._action_enc_w2, T, D, 2 * D) + self._action_enc_b2)
        h = self._fp16_gemm(h, self._action_enc_w3, T, D, D) + self._action_enc_b3
        pos_ids = torch.arange(T, device='cuda')
        action_feat = (h + self._position_embedding[:T]).unsqueeze(0)

        sa_embs = torch.cat([state_feat_buf, action_feat], dim=1)

        # DiT 32 layers
        hidden = sa_embs
        S_q = hidden.shape[1]
        w = self._dit_w
        NH, HD, H = self.NH_dit, self.HD_dit, self.H_dit

        ada_scales = self._ada_scales[step_idx]  # [L, D]
        ada_shifts = self._ada_shifts[step_idx]  # [L, D]

        for l in range(self.L_dit):
            is_self = (l % 2 == 1)

            # AdaLN with precomputed scale/shift
            scale = ada_scales[l].unsqueeze(0).unsqueeze(0)
            shift = ada_shifts[l].unsqueeze(0).unsqueeze(0)
            h_norm = F.layer_norm(hidden.float(), [D], eps=1e-5).to(fp16)
            h_norm = h_norm * (1 + scale) + shift
            h_norm_2d = h_norm.squeeze(0)

            if is_self:
                qkv = self._fp8_gemm(h_norm_2d, w['qkv_w_fp8'][l], w['qkv_s'][l].data_ptr(), S_q, 3*D, D)
                qkv = qkv + w['qkv_b'][l]
                q = qkv[:, :D].view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
                k = qkv[:, D:2*D].view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
                v = qkv[:, 2*D:].view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
                attn = F.scaled_dot_product_attention(q, k, v)
            else:
                q = self._fp16_gemm(h_norm_2d, w['q_w'][l], S_q, D, D) + w['q_b'][l]
                curr_mask = non_img_mask if l % 4 == 0 else img_mask
                kv_src = backbone_buf.to(fp16) * curr_mask.unsqueeze(-1).to(fp16)
                S_kv, D_kv = kv_src.shape[1], kv_src.shape[2]
                kv_2d = kv_src.squeeze(0)
                k = self._fp16_gemm(kv_2d, w['k_w'][l], S_kv, D, D_kv) + w['k_b'][l]
                v = self._fp16_gemm(kv_2d, w['v_w'][l], S_kv, D, D_kv) + w['v_b'][l]
                q = q.view(S_q, NH, HD).unsqueeze(0).transpose(1, 2)
                k = k.view(S_kv, NH, HD).unsqueeze(0).transpose(1, 2)
                v = v.view(S_kv, NH, HD).unsqueeze(0).transpose(1, 2)
                amask = curr_mask.unsqueeze(1).unsqueeze(2).expand(1, NH, S_q, S_kv)
                amask = torch.where(amask, 0.0, float('-inf')).to(fp16)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)

            attn_2d = attn.transpose(1, 2).reshape(1, S_q, D).squeeze(0).to(fp16)
            o = self._fp16_gemm(attn_2d, w['o_w'][l], S_q, D, D) + w['o_b'][l]
            hidden = hidden.float() + o.unsqueeze(0).float()

            ff_norm = F.layer_norm(hidden, [D], eps=1e-5).to(fp16).squeeze(0)
            ff_h = self._fp8_gemm(ff_norm, w['ff_up_w_fp8'][l], w['ff_up_s'][l].data_ptr(), S_q, H, D)
            ff_h = ff_h + w['ff_up_b'][l]
            fvk.gelu_inplace_fp16(ff_h.data_ptr(), S_q * H, 0)
            ff_out = self._fp8_gemm(ff_h, w['ff_down_w_fp8'][l], w['ff_down_s'][l].data_ptr(), S_q, D, H)
            ff_out = ff_out + w['ff_down_b'][l]
            hidden = (hidden + ff_out.unsqueeze(0).float()).to(fp16)

        # Output conditioning (precomputed)
        out_scale = self._out_scales[step_idx].unsqueeze(0).unsqueeze(0)
        out_shift = self._out_shifts[step_idx].unsqueeze(0).unsqueeze(0)
        hidden = F.layer_norm(hidden.float(), [D], eps=1e-6).to(fp16)
        hidden = hidden * (1 + out_scale) + out_shift

        model_output = self._fp16_gemm(hidden.squeeze(0), self._dit_proj_out_2_w, S_q, self.output_dim, D)

        # Action decode
        Sa = model_output.shape[0]
        dec_h = F.relu(self._fp16_gemm(model_output, self._action_dec_w1, Sa, 1024, self.output_dim)
                       + self._action_dec_b1)
        velocity = self._fp16_gemm(dec_h, self._action_dec_w2, Sa, self.action_dim, 1024) + self._action_dec_b2
        velocity = velocity[-T:].unsqueeze(0)

        # Euler step (in-place update)
        actions_buf.add_(dt * velocity.float())

    def capture_graph(self, backbone_features, image_mask, backbone_mask,
                      state, action_horizon):
        """Capture CUDA Graph for the complete 4-step flow-matching action head."""
        logger.info("Capturing CUDA Graph for action head (T=%d)...", action_horizon)

        self._precompute_action_time_embeds(action_horizon)

        # Allocate graph-persistent buffers
        non_img_mask = (~image_mask) & backbone_mask
        img_mask = image_mask & backbone_mask

        # State encoding (runs once, outside graph)
        state_feat = self._state_encode(state)  # [1, 1, D]

        # Graph input buffers (data written before replay)
        self._graph_backbone = backbone_features.to(fp16).clone()
        self._graph_non_img_mask = non_img_mask.clone()
        self._graph_img_mask = img_mask.clone()
        self._graph_state_feat = state_feat.clone()
        self._graph_actions = torch.randn(1, action_horizon, self.action_dim,
                                          dtype=torch.float32, device='cuda')
        self._graph_action_horizon = action_horizon

        # Warmup (3 runs without graph)
        for _ in range(3):
            self._graph_actions.normal_()
            for step in range(self.num_steps):
                self._graph_dit_step(step, self._graph_actions, self._graph_state_feat,
                                     self._graph_backbone, self._graph_non_img_mask,
                                     self._graph_img_mask)

        # Capture
        self._graph_actions.normal_()
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            for step in range(self.num_steps):
                self._graph_dit_step(step, self._graph_actions, self._graph_state_feat,
                                     self._graph_backbone, self._graph_non_img_mask,
                                     self._graph_img_mask)

        self._graph_captured = True
        logger.info("CUDA Graph captured for action head (T=%d)", action_horizon)

    def infer_action_head_graph(self, backbone_features, image_mask, backbone_mask,
                                state, action_horizon=None, noise_seed=None):
        """Run action head with CUDA Graph replay."""
        if not hasattr(self, '_graph_captured') or not self._graph_captured:
            self.capture_graph(backbone_features, image_mask, backbone_mask,
                              state, action_horizon or self.action_horizon)

        # Update graph input buffers
        self._graph_backbone.copy_(backbone_features.to(fp16))
        non_img = (~image_mask) & backbone_mask
        img = image_mask & backbone_mask
        self._graph_non_img_mask.copy_(non_img)
        self._graph_img_mask.copy_(img)
        self._graph_state_feat.copy_(self._state_encode(state))

        if noise_seed is not None:
            torch.manual_seed(noise_seed)
        self._graph_actions.normal_()

        # Replay
        self._cuda_graph.replay()

        return self._graph_actions.clone()

    def _patch_embed_image(self, img_np):
        """Preprocess image: numpy uint8 → patches → SigLIP2 → pixel_unshuffle → mlp1.

        Uses verified FP8 kernel pipeline (same as Pi0.5 siglip_forward).

        Args:
            img_np: numpy uint8 (224, 224, 3)
        Returns:
            vision_features: [spv, D_llm] fp16 — 64 tokens per view
        """
        D = self.D_sig  # 1152
        S = self.spv_raw  # 256 patches per view
        nv = 1
        spv = S
        patch_size = 14
        nH = nW = int(math.sqrt(S))  # 16 for 224x224

        # ── Patch embed: numpy RGB → [S, D] fp16 ──
        arr = img_np.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        pixel = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(fp16).cuda()

        patches = pixel.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(S, -1)  # [S, 588]

        # Linear patch embed + position embed → write into SigLIP buffer
        self._sig_x[:S].copy_(
            (F.linear(patches.float(), self._sig_patch_w.float(), self._sig_patch_b.float())
             + self._sig_pos_embed[:S].float()).to(fp16))

        # ── SigLIP2 27 layers (FP8 kernel pipeline, same as Pi0.5) ──
        sig_dims = {
            'S': S, 'D': D, 'H': self.H_sig, 'NH': self.NH_sig,
            'HD': self.HD_sig, 'L': self.L_sig,
            'num_views': nv, 'seq_per_view': spv,
        }
        siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                        sig_dims, stream=0)

        # Read SigLIP output from buffer
        siglip_out = self._sig_x[:S].clone()

        # ── Post-LN ──
        post_ln = torch.empty(S, D, dtype=fp16, device='cuda')
        fvk.layer_norm_fp16(siglip_out.data_ptr(), self._sig_post_ln_w.data_ptr(),
                            self._sig_post_ln_b.data_ptr(), post_ln.data_ptr(),
                            S, D, 1e-6, 0)

        # Pixel unshuffle: [256, 1152] → [64, 4608]
        spatial = post_ln.view(1, nH, nW, D).permute(0, 3, 1, 2)
        unshuffle = F.pixel_unshuffle(spatial, 2)
        flat = unshuffle.view(self.mlp1_in, -1).T.contiguous()  # [64, 4608]

        # mlp1: LN → FC → GELU → FC (fp16 kernels)
        S_img = flat.shape[0]
        ln_out = torch.empty(S_img, self.mlp1_in, dtype=fp16, device='cuda')
        fvk.layer_norm_fp16(flat.data_ptr(), self._mlp1_ln_w.data_ptr(),
                            self._mlp1_ln_b.data_ptr(), ln_out.data_ptr(),
                            S_img, self.mlp1_in, 1e-5, 0)

        fc1_out = torch.empty(S_img, self.D_llm, dtype=fp16, device='cuda')
        self._gemm.fp16_nn(ln_out.data_ptr(), self._mlp1_fc1_w.data_ptr(),
                           fc1_out.data_ptr(), S_img, self.D_llm, self.mlp1_in, 0)
        fvk.add_bias_fp16(fc1_out.data_ptr(), self._mlp1_fc1_b.data_ptr(), S_img, self.D_llm, 0)
        fvk.gelu_inplace_fp16(fc1_out.data_ptr(), S_img * self.D_llm, 0)

        fc2_out = torch.empty(S_img, self.D_llm, dtype=fp16, device='cuda')
        self._gemm.fp16_nn(fc1_out.data_ptr(), self._mlp1_fc2_w.data_ptr(),
                           fc2_out.data_ptr(), S_img, self.D_llm, self.D_llm, 0)
        fvk.add_bias_fp16(fc2_out.data_ptr(), self._mlp1_fc2_b.data_ptr(), S_img, self.D_llm, 0)

        return fc2_out  # [64, 2048]

    def _qwen3_backbone(self, input_embeds):
        """Run Qwen3 16-layer LLM backbone using fp16 kernels.

        Verified: cos=0.9999 per-layer (test_qwen3_layerwise.py).
        Uses: rms_norm_fp16 + fp16_nn for GEMM + PyTorch for RoPE/q_norm/k_norm/SDPA.
        FP8 optimization deferred to Phase C.

        Args:
            input_embeds: [1, Se, 2048] fp16
        Returns:
            backbone_features: [1, Se, 2048] fp16
        """
        Se = input_embeds.shape[1]
        D = self.D_llm
        NHQ, NHKV, HD, H = self.NHQ, self.NHKV, self.HD_llm, self.H_llm
        QKV_DIM = self.QKV_DIM
        w = self._qwen3_w

        x = input_embeds.squeeze(0).to(fp16).contiguous()  # [Se, D]
        x_normed = torch.empty(Se, D, dtype=fp16, device='cuda')

        for i in range(self.L_llm):
            # ── RMSNorm (fp16 kernel) ──
            fvk.rms_norm_fp16(x.data_ptr(), w['ln_attn_w'][i].data_ptr(),
                              x_normed.data_ptr(), Se, D, 0)

            # ── QKV GEMM (FP8: quantize activation + fp8_descale) ──
            x_fp8 = torch.empty(Se * D, dtype=torch.uint8, device='cuda')
            fvk.quantize_fp8_static_fp16(x_normed.data_ptr(), x_fp8.data_ptr(),
                                          self._unit_scale.data_ptr(), Se * D, 0)
            qkv = torch.empty(Se, QKV_DIM, dtype=fp16, device='cuda')
            self._gemm.fp8_descale_fp16(x_fp8.data_ptr(), w['qkv_w'][i].data_ptr(),
                                         qkv.data_ptr(), Se, QKV_DIM, D,
                                         self._unit_scale.data_ptr(),
                                         w['qkv_s_dev'][i].data_ptr(), 0)

            # ── Split QKV ──
            Q = qkv[:, :NHQ*HD].contiguous().view(Se, NHQ, HD)
            K = qkv[:, NHQ*HD:NHQ*HD+NHKV*HD].contiguous().view(Se, NHKV, HD)
            V = qkv[:, NHQ*HD+NHKV*HD:].contiguous().view(Se, NHKV, HD)

            # ── q_norm / k_norm (rms_norm_fp16 per head) ──
            for h in range(NHQ):
                q_h = Q[:, h].contiguous()
                fvk.rms_norm_fp16(q_h.data_ptr(), w['q_norm_w'][i].data_ptr(),
                                  q_h.data_ptr(), Se, HD, 0)
                Q[:, h] = q_h
            for h in range(NHKV):
                k_h = K[:, h].contiguous()
                fvk.rms_norm_fp16(k_h.data_ptr(), w['k_norm_w'][i].data_ptr(),
                                  k_h.data_ptr(), Se, HD, 0)
                K[:, h] = k_h

            # ── RoPE (PyTorch, theta=1e6) ──
            cos_cache = self._rope_cos_cache[:Se]  # precomputed in _precompute_rope
            sin_cache = self._rope_sin_cache[:Se]

            def rotate_half(t):
                t1, t2 = t[..., :t.shape[-1]//2], t[..., t.shape[-1]//2:]
                return torch.cat([-t2, t1], dim=-1)

            cos_r = cos_cache.unsqueeze(1)  # [Se, 1, HD] — broadcasts over heads
            sin_r = sin_cache.unsqueeze(1)
            Q = (Q.float() * cos_r + rotate_half(Q.float()) * sin_r).to(fp16)
            K = (K.float() * cos_r + rotate_half(K.float()) * sin_r).to(fp16)

            # ── GQA SDPA ──
            q_s = Q.unsqueeze(0).transpose(1, 2)
            k_s = K.unsqueeze(0).transpose(1, 2).repeat_interleave(NHQ // NHKV, dim=1)
            v_s = V.to(fp16).unsqueeze(0).transpose(1, 2).repeat_interleave(NHQ // NHKV, dim=1)
            attn = F.scaled_dot_product_attention(q_s, k_s, v_s)
            attn_flat = attn.transpose(1, 2).reshape(Se, D).to(fp16).contiguous()

            # ── O projection (fp16 — small M, FP8 not faster) + residual ──
            o_out = torch.empty(Se, D, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(attn_flat.data_ptr(), w['o_w_fp16'][i].data_ptr(),
                               o_out.data_ptr(), Se, D, D, 0)
            fvk.residual_add_fp16(x.data_ptr(), o_out.data_ptr(), Se * D, 0)

            # ── Post-attn RMSNorm ──
            fvk.rms_norm_fp16(x.data_ptr(), w['ln_ffn_w'][i].data_ptr(),
                              x_normed.data_ptr(), Se, D, 0)

            # ── FFN: gate+up merged (FP8) → SiLU(gate)*up → down ──
            fvk.quantize_fp8_static_fp16(x_normed.data_ptr(), x_fp8.data_ptr(),
                                          self._unit_scale.data_ptr(), Se * D, 0)
            ffn_merged = torch.empty(Se, 2 * H, dtype=fp16, device='cuda')
            self._gemm.fp8_descale_fp16(x_fp8.data_ptr(), w['gate_up_w'][i].data_ptr(),
                                         ffn_merged.data_ptr(), Se, 2 * H, D,
                                         self._unit_scale.data_ptr(),
                                         w['gate_up_s_dev'][i].data_ptr(), 0)

            gate = ffn_merged[:, :H].contiguous()
            up = ffn_merged[:, H:].contiguous()
            ffn_act = (F.silu(gate.float()) * up.float()).to(fp16).contiguous()

            down_out = torch.empty(Se, D, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(ffn_act.data_ptr(), w['down_w_fp16'][i].data_ptr(),
                               down_out.data_ptr(), Se, D, H, 0)
            fvk.residual_add_fp16(x.data_ptr(), down_out.data_ptr(), Se * D, 0)

        # ── Final RMSNorm ──
        fvk.rms_norm_fp16(x.data_ptr(), self._qwen3_final_norm_w.to(fp16).data_ptr(),
                          x_normed.data_ptr(), Se, D, 0)

        return x_normed.unsqueeze(0)  # [1, Se, D]

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Unified FP8 calibration entry point.

        Args:
            observations: single observation dict, list of dicts, or any
                iterable of dicts. ``N=1`` falls back to the legacy
                single-sample calibration path (bit-equal). ``N>=2`` runs
                per-sample shadow forwards through both Qwen3 and DiT,
                percentile-reduces per-tensor amax along the sample
                axis, and uploads the final FP8 scales into the captured
                graph buffers (no graph re-capture needed).
            percentile: percentile to apply in multi-sample mode.
                ``100.0`` is equivalent to a traditional max reduction;
                ``99.9`` (default) clips outlier frames. Ignored for
                ``N=1``.
            max_samples: optional cap on samples consumed from the
                iterable.
            verbose: if True, log per-stage amax dispersion summaries
                after reduction.
        """
        if not hasattr(self, "_text_embeds"):
            raise RuntimeError("set_prompt() must be called before calibrate()")
        if getattr(self, "_calibrated", False):
            logger.warning(
                "calibrate() called a second time; returning without re-running.")
            return

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
            self._calibrate_single_frame(obs_list[0])
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """:class:`ModelPrecisionSpec` captured at calibration time, or ``None``."""
        return getattr(self, "_precision_spec", None)

    def infer(self, obs):
        """Run full GROOT E2E inference: images → actions.

        First call captures CUDA Graphs (~5s). Subsequent calls use graph replay (~44ms).

        Args:
            obs: dict with keys:
                'image': numpy uint8 (224, 224, 3)
                'wrist_image': numpy uint8 (224, 224, 3) [optional for 2-view]
                'state': numpy float32 (state_dim,) — robot proprioception
                'action_horizon': int [optional]

        Returns:
            dict with 'actions': numpy (action_horizon, action_dim)
        """
        if not hasattr(self, '_input_ids'):
            raise RuntimeError("Call set_prompt() before infer()")

        # ── Lazy graph capture on first call ──
        if not getattr(self, '_graphs_built', False):
            self._capture_all_graphs(obs)

        # ── 1. SigLIP: patch embed + graph replay + mlp1 ──
        views = [obs['image']]
        if 'wrist_image' in obs and self._num_views >= 2:
            views.append(obs['wrist_image'])
        self._patch_embed_2views(views)
        self._siglip_graph.replay()
        torch.cuda.synchronize()  # pixel_unshuffle needs SigLIP output
        self._run_pixel_unshuffle_mlp1()

        # ── 2. Build input embeddings (pre-allocated buffer) ──
        self._g_ie_buf.copy_(self._text_embeds.squeeze(0).to(fp16))
        self._g_ie_buf[self._image_mask[0]] = self._g_vision_out.to(fp16)
        fvk.gpu_copy(self._g_qwen3.b_x.data_ptr(), self._g_ie_buf.data_ptr(),
                     self._Se * self.D_llm * 2, 0)

        # ── 3. Qwen3 graph replay (+ vlln) ──
        self._qwen3_graph.replay()
        torch.cuda.synchronize()  # KV update needs Qwen3 output

        # ── 4. Update DiT KV + init noise ──
        bb = self._g_vlln_buf.unsqueeze(0)
        self._g_dit.b_kv_text.copy_((bb * self._g_non_img.unsqueeze(-1).to(fp16)).squeeze(0))
        self._g_dit.b_kv_img.copy_((bb * self._g_img_m.unsqueeze(-1).to(fp16)).squeeze(0))

        # Precompute cross-attention K/V projections (runs once, reused across 4 steps)
        self._g_dit.precompute_cross_kv()
        self._g_dit.b_actions.normal_()

        # ── 5. DiT graph replay ──
        self._dit_graph.replay()
        torch.cuda.synchronize()

        return {'actions': self._g_dit.b_actions.squeeze(0).cpu().numpy()}

    # ─────────────────────────────────────────────────────────────
    # FP8 Activation Calibration
    # ─────────────────────────────────────────────────────────────

    def _collect_qwen3_amax(self, ie_fp16):
        """Run an FP16 shadow forward through Qwen3 and return raw amax.

        Stateless: each call processes one input embedding tensor and
        returns ``3*L`` raw activation amax values (one per FP8
        quantization point per layer: QKV input, Gate+Up input, Down
        input). No FP8 scale conversion, no cache, no buffer upload —
        the caller wraps these as needed.

        Used by both the single-sample :meth:`_calibrate_qwen3` wrapper
        and the multi-sample :meth:`_calibrate_multi_frame` reducer.
        """
        Se = self._Se
        sd = self._full_sd
        prefix = "backbone.model.language_model.model.layers"
        fp16_w = {'ln_w': [], 'qkv_w': [], 'q_norm_w': [], 'k_norm_w': [],
                  'o_w': [], 'ln2_w': [], 'gu_w': [], 'down_w': []}
        for i in range(16):
            lp = f"{prefix}.{i}"
            fp16_w['ln_w'].append(sd[f"{lp}.input_layernorm.weight"].to(fp16))
            fp16_w['qkv_w'].append(torch.cat([sd[f"{lp}.self_attn.{p}_proj.weight"]
                                               for p in ('q', 'k', 'v')], dim=0).T.contiguous().to(fp16))
            fp16_w['q_norm_w'].append(sd[f"{lp}.self_attn.q_norm.weight"].to(fp16))
            fp16_w['k_norm_w'].append(sd[f"{lp}.self_attn.k_norm.weight"].to(fp16))
            fp16_w['o_w'].append(sd[f"{lp}.self_attn.o_proj.weight"].T.contiguous().to(fp16))
            fp16_w['ln2_w'].append(sd[f"{lp}.post_attention_layernorm.weight"].to(fp16))
            fp16_w['gu_w'].append(torch.cat([sd[f"{lp}.mlp.gate_proj.weight"],
                                              sd[f"{lp}.mlp.up_proj.weight"]], dim=0).T.contiguous().to(fp16))
            fp16_w['down_w'].append(sd[f"{lp}.mlp.down_proj.weight"].T.contiguous().to(fp16))
        w = fp16_w
        D, H, L = 2048, 6144, 16
        x = ie_fp16.clone()
        xn = torch.empty_like(x)
        amax_list = []

        for i in range(L):
            fvk.rms_norm_fp16(x.data_ptr(), w['ln_w'][i].data_ptr(),
                              xn.data_ptr(), Se, D, 1e-6, 0)
            torch.cuda.synchronize()
            amax_list.append(float(xn[:Se].float().abs().max().item()))

            QKV_DIM = self._g_qwen3.QKV
            qkv = torch.empty(Se, QKV_DIM, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(xn.data_ptr(), w['qkv_w'][i].data_ptr(),
                               qkv.data_ptr(), Se, QKV_DIM, D, 0)

            NHQ, NHKV, HD = 16, 8, 128
            Q = qkv[:, :NHQ*HD].contiguous().view(Se, NHQ, HD)
            K = qkv[:, NHQ*HD:NHQ*HD+NHKV*HD].contiguous().view(Se, NHKV, HD)
            V = qkv[:, NHQ*HD+NHKV*HD:].contiguous().view(Se, NHKV, HD)
            for h in range(NHQ):
                q_h = Q[:, h].contiguous()
                fvk.rms_norm_fp16(q_h.data_ptr(), w['q_norm_w'][i].data_ptr(),
                                  q_h.data_ptr(), Se, HD, 1e-6, 0)
                Q[:, h] = q_h
            for h in range(NHKV):
                k_h = K[:, h].contiguous()
                fvk.rms_norm_fp16(k_h.data_ptr(), w['k_norm_w'][i].data_ptr(),
                                  k_h.data_ptr(), Se, HD, 1e-6, 0)
                K[:, h] = k_h
            cos_r = self._g_qwen3.cos_table[:Se].unsqueeze(1)
            sin_r = self._g_qwen3.sin_table[:Se].unsqueeze(1)

            def rotate_half(t):
                return torch.cat([-t[..., t.shape[-1]//2:], t[..., :t.shape[-1]//2]], dim=-1)

            Q = (Q.float() * cos_r + rotate_half(Q.float()) * sin_r).to(fp16)
            K = (K.float() * cos_r + rotate_half(K.float()) * sin_r).to(fp16)
            q_s = Q.unsqueeze(0).transpose(1, 2)
            k_s = K.unsqueeze(0).transpose(1, 2).repeat_interleave(NHQ // NHKV, dim=1)
            v_s = V.to(fp16).unsqueeze(0).transpose(1, 2).repeat_interleave(NHQ // NHKV, dim=1)
            attn = F.scaled_dot_product_attention(q_s, k_s, v_s)
            attn_flat = attn.transpose(1, 2).reshape(Se, D).to(fp16).contiguous()

            o_out = torch.empty(Se, D, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(attn_flat.data_ptr(), w['o_w'][i].data_ptr(),
                               o_out.data_ptr(), Se, D, D, 0)
            fvk.residual_add_fp16(x.data_ptr(), o_out.data_ptr(), Se * D, 0)

            fvk.rms_norm_fp16(x.data_ptr(), w['ln2_w'][i].data_ptr(),
                              xn.data_ptr(), Se, D, 1e-6, 0)
            torch.cuda.synchronize()
            amax_list.append(float(xn[:Se].float().abs().max().item()))

            ffn_merged = torch.empty(Se, 2 * H, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(xn.data_ptr(), w['gu_w'][i].data_ptr(),
                               ffn_merged.data_ptr(), Se, 2 * H, D, 0)
            gate = ffn_merged[:, :H].contiguous()
            up = ffn_merged[:, H:].contiguous()
            ffn_act = (F.silu(gate.float()) * up.float()).to(fp16).contiguous()

            torch.cuda.synchronize()
            amax_list.append(float(ffn_act.float().abs().max().item()))

            down_out = torch.empty(Se, D, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(ffn_act.data_ptr(), w['down_w'][i].data_ptr(),
                               down_out.data_ptr(), Se, D, H, 0)
            fvk.residual_add_fp16(x.data_ptr(), down_out.data_ptr(), Se * D, 0)

        return amax_list

    def _calibrate_qwen3(self, ie_fp16):
        """Single-sample Qwen3 calibration: cache → collect → set scales.

        Bit-equal to the pre-refactor implementation: cache hit short-circuits;
        otherwise delegates the FP16 shadow forward to
        :meth:`_collect_qwen3_amax`, applies the FP8 scale conversion
        ``max(amax, 1e-12) / 448.0``, uploads scales, and writes the cache.
        """
        Se = self._Se
        cached = load_calibration(str(self._checkpoint_path), Se)
        if cached is not None and len(cached.get("qwen3_act_scales", [])) > 0:
            self._g_qwen3.set_act_scales(cached["qwen3_act_scales"])
            logger.info("Qwen3 calibration loaded from cache (%d scales)",
                        len(cached["qwen3_act_scales"]))
            return

        amax_list = self._collect_qwen3_amax(ie_fp16)
        scales = [max(float(a), 1e-12) / 448.0 for a in amax_list]
        self._g_qwen3.set_act_scales(scales)
        logger.info("Qwen3 calibrated: %d act scales", len(scales))

        try:
            save_calibration(
                checkpoint_path=self._checkpoint_path,
                Se=Se,
                enc_scales=[], enc_alpha=[], ae_scales=[],
                enc_w_scales=[],
            )
            from flash_rt.core.quant.calibrator import _checkpoint_hash, _cache_path
            import json
            ckpt_hash = _checkpoint_hash(self._checkpoint_path)
            cache_file = _cache_path(ckpt_hash, Se)
            with open(cache_file) as f:
                data = json.load(f)
            data["qwen3_act_scales"] = scales
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Qwen3 calibration saved to cache")
        except Exception as e:
            logger.warning("Failed to save Qwen3 calibration cache: %s", e)

    def _collect_dit_amax(self):
        """Run an FP16 shadow forward through DiT (step=0) and return raw amax.

        Returns ``3*L`` raw amax values per layer: pre-attn h_norm,
        pre-FFN h_norm (post-GELU FFN-up output is the third). Cross-
        attention layers (``l % 2 == 0``) have no learnable
        pre-attention scale; their first slot is filled with
        ``float('nan')`` as a sentinel so multi-sample reducers can
        preserve the constant placeholder ``1.0/448.0`` that the FP8
        path expects.
        """
        Se = self._Se
        dit = self._g_dit
        D, H, Sa = dit.D, dit.H, dit.Sa
        NH, HD = dit.NH, dit.HD
        S_kv = dit.S_kv

        dit.b_actions.normal_()
        torch.cuda.synchronize()

        T = dit.T
        actions_fp16 = dit.b_actions.squeeze(0).to(fp16)
        a_emb_out = torch.empty(T, D, dtype=fp16, device='cuda')
        ae_concat = torch.empty(T, 2*D, dtype=fp16, device='cuda')
        self._gemm.fp16_nn(actions_fp16.data_ptr(), dit.ae_w1.data_ptr(), a_emb_out.data_ptr(), T, D, dit.action_dim, 0)
        fvk.add_bias_fp16(a_emb_out.data_ptr(), dit.ae_b1.data_ptr(), T, D, 0)
        fvk.gpu_copy(ae_concat.data_ptr(), a_emb_out.data_ptr(), T*D*2, 0)
        fvk.gpu_copy(ae_concat.data_ptr()+T*D*2, dit.action_time_embeds[0].data_ptr(), T*D*2, 0)
        enc_h = torch.empty(T, D, dtype=fp16, device='cuda')
        self._gemm.fp16_nn(ae_concat.data_ptr(), dit.ae_w2.data_ptr(), enc_h.data_ptr(), T, D, 2*D, 0)
        fvk.add_bias_fp16(enc_h.data_ptr(), dit.ae_b2.data_ptr(), T, D, 0)
        fvk.silu_inplace_fp16(enc_h.data_ptr(), T*D, 0)
        self._gemm.fp16_nn(enc_h.data_ptr(), dit.ae_w3.data_ptr(), a_emb_out.data_ptr(), T, D, D, 0)
        fvk.add_bias_fp16(a_emb_out.data_ptr(), dit.ae_b3.data_ptr(), T, D, 0)
        fvk.residual_add_fp16(a_emb_out.data_ptr(), dit.pos_emb[:T].data_ptr(), T*D, 0)

        hidden = torch.empty(Sa, D, dtype=fp16, device='cuda')
        fvk.gpu_copy(hidden.data_ptr(), dit.b_state_feat.data_ptr(), D*2, 0)
        fvk.gpu_copy(hidden.data_ptr()+D*2, a_emb_out.data_ptr(), T*D*2, 0)
        torch.cuda.synchronize()

        h_norm = torch.empty(Sa, D, dtype=fp16, device='cuda')
        amax_list = []

        for l in range(dit.L):
            is_self = (l % 2 == 1)
            w = dit.dit[l]

            fvk.ada_layer_norm_fp16(hidden.data_ptr(), dit.ada_scales[0, l].data_ptr(),
                                     dit.ada_shifts[0, l].data_ptr(), h_norm.data_ptr(), Sa, D, 1e-5, 0)
            torch.cuda.synchronize()

            if is_self:
                amax_list.append(float(h_norm[:Sa].float().abs().max().item()))
            else:
                amax_list.append(float('nan'))

            if is_self:
                Q = torch.empty(Sa, D, dtype=fp16, device='cuda')
                K = torch.empty(Sa, D, dtype=fp16, device='cuda')
                V = torch.empty(Sa, D, dtype=fp16, device='cuda')
                self._gemm.fp16_nn(h_norm.data_ptr(), w['q_w'].data_ptr(), Q.data_ptr(), Sa, D, D, 0)
                fvk.add_bias_fp16(Q.data_ptr(), w['q_b'].data_ptr(), Sa, D, 0)
                self._gemm.fp16_nn(h_norm.data_ptr(), w['k_w'].data_ptr(), K.data_ptr(), Sa, D, D, 0)
                fvk.add_bias_fp16(K.data_ptr(), w['k_b'].data_ptr(), Sa, D, 0)
                self._gemm.fp16_nn(h_norm.data_ptr(), w['v_w'].data_ptr(), V.data_ptr(), Sa, D, D, 0)
                fvk.add_bias_fp16(V.data_ptr(), w['v_b'].data_ptr(), Sa, D, 0)
                q_s = Q.unsqueeze(0).view(1, Sa, NH, HD).transpose(1, 2)
                k_s = K.unsqueeze(0).view(1, Sa, NH, HD).transpose(1, 2)
                v_s = V.unsqueeze(0).view(1, Sa, NH, HD).transpose(1, 2)
                attn = F.scaled_dot_product_attention(q_s, k_s, v_s).transpose(1, 2).reshape(Sa, D).to(fp16).contiguous()
            else:
                Q = torch.empty(Sa, D, dtype=fp16, device='cuda')
                self._gemm.fp16_nn(h_norm.data_ptr(), w['q_w'].data_ptr(), Q.data_ptr(), Sa, D, D, 0)
                fvk.add_bias_fp16(Q.data_ptr(), w['q_b'].data_ptr(), Sa, D, 0)
                cross_idx = l // 2
                K_pre = dit._precomp_k[cross_idx]
                V_pre = dit._precomp_v[cross_idx]
                q_s = Q.unsqueeze(0).view(1, Sa, NH, HD).transpose(1, 2)
                k_s = K_pre.unsqueeze(0).view(1, S_kv, NH, HD).transpose(1, 2)
                v_s = V_pre.unsqueeze(0).view(1, S_kv, NH, HD).transpose(1, 2)
                attn = F.scaled_dot_product_attention(q_s, k_s, v_s).transpose(1, 2).reshape(Sa, D).to(fp16).contiguous()

            o_out = torch.empty(Sa, D, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(attn.data_ptr(), w['o_w'].data_ptr(), o_out.data_ptr(), Sa, D, D, 0)
            fvk.add_bias_fp16(o_out.data_ptr(), w['o_b'].data_ptr(), Sa, D, 0)
            fvk.residual_add_fp16(hidden.data_ptr(), o_out.data_ptr(), Sa*D, 0)

            fvk.layer_norm_no_affine_fp16(hidden.data_ptr(), h_norm.data_ptr(), Sa, D, 1e-5, 0)
            torch.cuda.synchronize()
            amax_list.append(float(h_norm[:Sa].float().abs().max().item()))

            ff_up_fp16 = (w['ff_up_fp8'].view(D, H).float() * w['ff_up_alpha']).to(fp16).contiguous()
            ff_h = torch.empty(Sa, H, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(h_norm.data_ptr(), ff_up_fp16.data_ptr(), ff_h.data_ptr(), Sa, H, D, 0)
            fvk.add_bias_fp16(ff_h.data_ptr(), w['ff_up_b'].data_ptr(), Sa, H, 0)
            fvk.gelu_inplace_fp16(ff_h.data_ptr(), Sa*H, 0)
            torch.cuda.synchronize()
            amax_list.append(float(ff_h[:Sa].float().abs().max().item()))

            ff_dn_fp16 = (w['ff_dn_fp8'].view(H, D).float() * w['ff_dn_alpha']).to(fp16).contiguous()
            ff_out = torch.empty(Sa, D, dtype=fp16, device='cuda')
            self._gemm.fp16_nn(ff_h.data_ptr(), ff_dn_fp16.data_ptr(), ff_out.data_ptr(), Sa, D, H, 0)
            fvk.add_bias_fp16(ff_out.data_ptr(), w['ff_dn_b'].data_ptr(), Sa, D, 0)
            fvk.residual_add_fp16(hidden.data_ptr(), ff_out.data_ptr(), Sa*D, 0)

        return amax_list

    def _calibrate_dit(self):
        """Single-sample DiT calibration: cache → collect → set scales.

        Bit-equal to the pre-refactor implementation: cache hit
        short-circuits; otherwise delegates the FP16 shadow forward to
        :meth:`_collect_dit_amax`, maps the ``NaN`` cross-layer
        sentinels to the constant placeholder ``1.0/448.0``, applies
        the FP8 scale conversion ``max(amax, 1e-12) / 448.0`` to the
        finite slots, uploads scales, and writes the cache.
        """
        Se = self._Se
        cached = load_calibration(str(self._checkpoint_path), Se)
        if cached is not None and len(cached.get("dit_act_scales", [])) > 0:
            self._g_dit.set_dit_act_scales(cached["dit_act_scales"])
            logger.info("DiT calibration loaded from cache (%d scales)",
                        len(cached["dit_act_scales"]))
            return

        amax_list = self._collect_dit_amax()
        scales = [(1.0 / 448.0) if math.isnan(a) else max(float(a), 1e-12) / 448.0
                  for a in amax_list]
        self._g_dit.set_dit_act_scales(scales)
        logger.info("DiT calibrated: %d act scales", len(scales))

        try:
            from flash_rt.core.quant.calibrator import _checkpoint_hash, _cache_path
            import json
            ckpt_hash = _checkpoint_hash(str(self._checkpoint_path))
            cache_file = _cache_path(ckpt_hash, Se)
            if cache_file.exists():
                with open(cache_file) as f:
                    data = json.load(f)
            else:
                data = {"version": 1, "ckpt_hash": ckpt_hash, "Se": Se,
                        "enc_scales": [], "enc_alpha": [], "ae_scales": [], "enc_w_scales": []}
            data["dit_act_scales"] = scales
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("DiT calibration saved to cache")
        except Exception as e:
            logger.warning("Failed to save DiT calibration cache: %s", e)

    # ─────────────────────────────────────────────────────────────
    # Public calibration entry-point implementations
    # ─────────────────────────────────────────────────────────────

    def _calibrate_single_frame(self, obs):
        """N=1 path: build pipeline (which calibrates) and snapshot the spec."""
        if not getattr(self, "_graphs_built", False):
            self._capture_all_graphs(obs)
        self._calibrated = True
        self._precision_spec = self._snapshot_precision_spec(
            method="single_frame", n=1, percentile=None)
        self._warn_if_scale_ceiling_exceeded()
        logger.info("GROOT single-frame calibration complete")

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        """N>=2 path: refine FP8 scales using N samples with percentile reduce.

        Bootstraps the pipeline on the first sample (single-frame path)
        with ``release_full_sd=False`` so the FP16 weights survive the
        Qwen3 multi-sample pass, then drives Qwen3 and DiT shadow
        forwards over all N samples, reduces per-tensor amax via
        ``numpy.percentile`` along the sample axis, and uploads the
        final scales into the existing graph buffers (no graph
        re-capture needed; the captured kernels read scales from the
        buffers at replay time).
        """
        from flash_rt.core.calibration import (
            accumulate_amax,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Calibrating GROOT FP8 across %d samples (percentile=%.2f)...",
            n, percentile)

        if not getattr(self, "_graphs_built", False):
            self._capture_all_graphs(obs_list[0], release_full_sd=False)

        # ── Pass A: collect per-sample Qwen3 amax (uses _full_sd) ──
        qwen3_rows = [
            self._collect_qwen3_amax_for_obs(obs) for obs in obs_list
        ]
        qwen3_reduced = accumulate_amax(qwen3_rows, percentile=percentile)
        if verbose:
            logger.info("Qwen3 amax dispersion:\n%s", format_summary(
                summarize_amax_dispersion(qwen3_rows, qwen3_reduced)))
        qwen3_scales = [max(float(a), 1e-12) / 448.0 for a in qwen3_reduced.tolist()]
        self._g_qwen3.set_act_scales(qwen3_scales)

        # Free _full_sd now that Qwen3 multi-pass is done; DiT uses
        # dequantized FP8 weights and no longer needs the raw state dict.
        if hasattr(self, "_full_sd"):
            del self._full_sd
            torch.cuda.empty_cache()

        # ── Pass B: collect per-sample DiT amax (uses updated Qwen3 scales) ──
        dit_rows = [
            self._collect_dit_amax_for_obs(obs) for obs in obs_list
        ]
        dit_reduced = self._reduce_amax_with_placeholders(
            dit_rows, percentile=percentile)
        dit_scales = [(1.0 / 448.0) if math.isnan(a) else max(float(a), 1e-12) / 448.0
                      for a in dit_reduced.tolist()]
        self._g_dit.set_dit_act_scales(dit_scales)

        self._calibrated = True
        self._precision_spec = self._snapshot_precision_spec(
            method="percentile", n=n, percentile=percentile)
        self._warn_if_scale_ceiling_exceeded(label=f"groot_thor_N{n}")
        logger.info(
            "GROOT multi-frame calibration complete (N=%d, percentile=%.2f)",
            n, percentile)

    def _collect_qwen3_amax_for_obs(self, obs):
        """Drive vision/embed for ``obs`` then collect Qwen3 raw amax."""
        views = [obs['image']]
        if 'wrist_image' in obs and self._num_views >= 2:
            views.append(obs['wrist_image'])
        self._patch_embed_2views(views)
        self._siglip_graph.replay()
        torch.cuda.synchronize()
        self._run_pixel_unshuffle_mlp1()
        torch.cuda.synchronize()
        input_embeds = self._text_embeds.clone()
        input_embeds[0, self._image_mask[0]] = self._g_vision_out.to(input_embeds.dtype)
        ie_fp16 = input_embeds.squeeze(0).to(fp16).contiguous()
        amax = self._collect_qwen3_amax(ie_fp16)
        return np.asarray(amax, dtype=np.float32)

    def _collect_dit_amax_for_obs(self, obs):
        """Run quantized backbone + DiT setup for ``obs`` then collect DiT raw amax."""
        Se = self._Se
        views = [obs['image']]
        if 'wrist_image' in obs and self._num_views >= 2:
            views.append(obs['wrist_image'])
        self._patch_embed_2views(views)
        self._siglip_graph.replay()
        torch.cuda.synchronize()
        self._run_pixel_unshuffle_mlp1()
        torch.cuda.synchronize()
        input_embeds = self._text_embeds.clone()
        input_embeds[0, self._image_mask[0]] = self._g_vision_out.to(input_embeds.dtype)
        ie_fp16 = input_embeds.squeeze(0).to(fp16).contiguous()

        # Quantized Qwen3 backbone → vlln → kv split
        self._g_qwen3.forward(ie_fp16)
        vlln_w = self._vlln_w.to(fp16)
        vlln_b = self._vlln_b.to(fp16)
        fvk.layer_norm_fp16(self._g_qwen3.b_xn.data_ptr(), vlln_w.data_ptr(),
                            vlln_b.data_ptr(), self._g_vlln_buf.data_ptr(),
                            Se, self.D_llm, 1e-5, 0)
        torch.cuda.synchronize()
        bb = self._g_vlln_buf.unsqueeze(0)
        self._g_dit.b_kv_text.copy_((bb * self._g_non_img.unsqueeze(-1).to(fp16)).squeeze(0))
        self._g_dit.b_kv_img.copy_((bb * self._g_img_m.unsqueeze(-1).to(fp16)).squeeze(0))
        self._g_dit.precompute_cross_kv()
        torch.cuda.synchronize()

        # State encode → b_state_feat
        state = obs.get('state', np.zeros(self.state_dim, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(torch.float32).cuda()
        state_fp16 = state.to(fp16).contiguous()
        if state_fp16.dim() == 1:
            state_fp16 = state_fp16.unsqueeze(0)
        h = torch.empty(1, 1024, dtype=fp16, device='cuda')
        self._g_dit.gemm.fp16_nn(state_fp16.data_ptr(), self._g_dit.se_w1.data_ptr(),
                                  h.data_ptr(), 1, 1024, self.state_dim, 0)
        fvk.add_bias_fp16(h.data_ptr(), self._g_dit.se_b1.data_ptr(), 1, 1024, 0)
        fvk.relu_inplace_fp16(h.data_ptr(), 1024, 0)
        sf = torch.empty(1, self.D_dit, dtype=fp16, device='cuda')
        self._g_dit.gemm.fp16_nn(h.data_ptr(), self._g_dit.se_w2.data_ptr(),
                                  sf.data_ptr(), 1, self.D_dit, 1024, 0)
        fvk.add_bias_fp16(sf.data_ptr(), self._g_dit.se_b2.data_ptr(), 1, self.D_dit, 0)
        torch.cuda.synchronize()
        self._g_dit.b_state_feat.copy_(sf)

        amax = self._collect_dit_amax()
        return np.asarray(amax, dtype=np.float32)

    @staticmethod
    def _reduce_amax_with_placeholders(rows, *, percentile: float) -> np.ndarray:
        """Percentile-reduce per-sample amax rows, preserving NaN placeholders."""
        if not rows:
            raise ValueError("rows must contain at least one entry")
        stacked = np.stack(rows, axis=0).astype(np.float64)
        if stacked.ndim != 2:
            raise ValueError(
                f"each amax row must be 1-D, got shapes {[r.shape for r in rows]}")
        out = np.empty(stacked.shape[1], dtype=np.float32)
        for j in range(stacked.shape[1]):
            col = stacked[:, j]
            mask = np.isnan(col)
            if mask.all():
                out[j] = np.float32(np.nan)
            else:
                out[j] = np.float32(np.percentile(col[~mask], percentile))
        return out

    def _read_qwen3_scales(self) -> np.ndarray:
        """Snapshot the current Qwen3 FP8 scale buffer to host."""
        return self._g_qwen3.act_scales.cpu().numpy().astype(np.float32)

    def _read_dit_scales(self) -> np.ndarray:
        """Snapshot the current DiT FP8 scale buffer to host."""
        return self._g_dit._dit_act_scales_dev.cpu().numpy().astype(np.float32)

    def _warn_if_scale_ceiling_exceeded(self, label: str = "groot_thor") -> None:
        """Diagnostic warning if any FP8 scale is far above the median."""
        from flash_rt.core.calibration import check_scale_ceiling
        scales: dict = {}
        for i, s in enumerate(self._read_qwen3_scales().tolist()):
            scales[f"qwen3_{i}"] = float(s)
        for i, s in enumerate(self._read_dit_scales().tolist()):
            if not math.isclose(float(s), 1.0 / 448.0):
                scales[f"dit_{i}"] = float(s)
        if scales:
            check_scale_ceiling(scales, label=label)

    def _snapshot_precision_spec(self, *, method: str, n: int,
                                 percentile: Optional[float]):
        """Build a :class:`ModelPrecisionSpec` from current FP8 scales."""
        from flash_rt.core.precision_spec import (
            ModelPrecisionSpec,
            PrecisionSpec,
        )

        spec = ModelPrecisionSpec(source="calibration")

        def _entry(scale_val: float):
            entry = PrecisionSpec(
                dtype="fp8_e4m3",
                granularity="per_tensor",
                scheme="symmetric",
                scale_source="calibration",
                scale=np.array([scale_val], dtype=np.float32),
                calibration_method=method,
                calibration_samples=n,
                calibration_percentile=percentile,
            )
            entry.validate()
            return entry

        for i, s in enumerate(self._read_qwen3_scales().tolist()):
            spec.encoder_layer_specs[f"qwen3_{i}"] = _entry(float(s))
        for i, s in enumerate(self._read_dit_scales().tolist()):
            spec.decoder_layer_specs[f"dit_{i}"] = _entry(float(s))
        return spec

    # ─────────────────────────────────────────────────────────────
    # CUDA Graph capture (called lazily on first infer())
    # ─────────────────────────────────────────────────────────────

    def _patch_embed_2views(self, views):
        """Preprocess images → SigLIP buffer (batched for multi-view)."""
        for idx, img_np in enumerate(views):
            arr = img_np.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            pixel = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(fp16).cuda()
            patches = pixel.unfold(2, 14, 14).unfold(3, 14, 14)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(self.spv_raw, -1)
            embedded = (F.linear(patches.float(), self._sig_patch_w.float(),
                                 self._sig_patch_b.float())
                        + self._sig_pos_embed[:self.spv_raw].float()).to(fp16)
            self._sig_x[idx * self.spv_raw:(idx + 1) * self.spv_raw].copy_(embedded)

    def _run_pixel_unshuffle_mlp1(self):
        """Pixel unshuffle + mlp1 (runs outside graph, ~0.3ms)."""
        nH = int(math.sqrt(self.spv_raw))
        D, S_img = self.D_sig, self._num_views * self.spv
        all_flat = []
        for v in range(self._num_views):
            view_out = self._g_sig_postln[v * self.spv_raw:(v + 1) * self.spv_raw]
            spatial = view_out.view(1, nH, nH, D).permute(0, 3, 1, 2)
            flat = F.pixel_unshuffle(spatial, 2).view(self.mlp1_in, -1).T.contiguous()
            all_flat.append(flat)
        combined = torch.cat(all_flat, dim=0)
        fvk.layer_norm_fp16(combined.data_ptr(), self._mlp1_ln_w.data_ptr(),
                            self._mlp1_ln_b.data_ptr(), self._g_mlp1_ln.data_ptr(),
                            S_img, self.mlp1_in, 1e-5, 0)
        self._gemm.fp16_nn(self._g_mlp1_ln.data_ptr(), self._mlp1_fc1_w.data_ptr(),
                           self._g_mlp1_fc1.data_ptr(), S_img, self.D_llm, self.mlp1_in, 0)
        fvk.add_bias_fp16(self._g_mlp1_fc1.data_ptr(), self._mlp1_fc1_b.data_ptr(),
                          S_img, self.D_llm, 0)
        fvk.gelu_inplace_fp16(self._g_mlp1_fc1.data_ptr(), S_img * self.D_llm, 0)
        self._gemm.fp16_nn(self._g_mlp1_fc1.data_ptr(), self._mlp1_fc2_w.data_ptr(),
                           self._g_vision_out.data_ptr(), S_img, self.D_llm, self.D_llm, 0)
        fvk.add_bias_fp16(self._g_vision_out.data_ptr(), self._mlp1_fc2_b.data_ptr(),
                          S_img, self.D_llm, 0)

    def _capture_all_graphs(self, obs, release_full_sd: bool = True):
        """One-time graph capture: SigLIP + Qwen3 + DiT.

        Args:
            obs: a single observation used to drive the initial single-sample
                calibration through SigLIP, Qwen3, and DiT.
            release_full_sd: if True (default), free ``self._full_sd`` after
                Qwen3 calibration to reclaim ~8 GB of GPU memory. Set to
                False when the caller intends to refine Qwen3 calibration
                with additional samples (the FP16 shadow forward requires
                the raw state dict to be alive).
        """
        from flash_rt.models.groot.pipeline_thor import CKernelQwen3, CKernelDiTHead
        logger.info("Capturing CUDA Graphs for E2E pipeline...")

        Se = self._Se
        S_sig = self._num_views * self.spv_raw
        S_img = self._num_views * self.spv

        # ── 1. Build CKernel objects FIRST (from sd dict, before SigLIP FP8 init) ──
        # This ensures cuBLASLt workspace is not polluted by SigLIP FP8 quantization
        self._g_qwen3 = CKernelQwen3(self._full_sd, Se)
        vlln_w = self._vlln_w.to(fp16)
        vlln_b = self._vlln_b.to(fp16)

        self._g_non_img = (~self._image_mask) & self._backbone_mask
        self._g_img_m = self._image_mask & self._backbone_mask

        T = self.action_horizon
        self._g_dit = CKernelDiTHead(self._full_sd, self._embodiment_id, T,
                                      (1, Se, self.D_llm))

        # ── 2. SigLIP init (after CKernel objects, so FP8 quant doesn't pollute cuBLAS state) ──
        self._load_siglip2_weights(self._full_sd)
        self._allocate_buffers()
        # NOTE: _full_sd deleted AFTER calibration (needs FP16 weights for amax measurement)
        torch.cuda.empty_cache()

        # ── 2.5 Build ThorGrootAttnBackend (stage 4.3) ──
        # All underlying buffers exist at this point:
        #   * sig_qkv / sig_attn  — from _allocate_buffers() above
        #   * CKernelQwen3 scratch — from _g_qwen3.__init__ (b_q/b_k_exp/…)
        #   * CKernelDiTHead scratch — from _g_dit.__init__ incl. _precomp_k/_precomp_v
        # _precomp_k/_precomp_v pointers are stable across precompute_cross_kv()
        # (verified: torch.empty once in _alloc_buffers, written in-place via GEMM).
        self._attn = ThorGrootAttnBackend(
            make_groot_attention_spec(
                num_views=self._num_views,
                qwen3_seq_max=Se,
                sa=self._g_dit.Sa,
                s_kv=self._g_dit.S_kv,
            ),
            siglip_slots={
                "qkv": self._sig_qkv.data_ptr(),
                "O":   self._sig_attn.data_ptr(),
                "D":   self.D_sig,
            },
            qwen3_slots={
                "ctx":    self._g_qwen3.ctx,       # Qwen3's cuBLAS handle
                "Q":      self._g_qwen3.b_q.data_ptr(),
                "K":      self._g_qwen3.b_k_exp.data_ptr(),
                "V":      self._g_qwen3.b_v_exp.data_ptr(),
                "O":      self._g_qwen3.b_o.data_ptr(),
                "logits": self._g_qwen3.b_logits.data_ptr(),
                "scale":  1.0 / math.sqrt(self._g_qwen3.HD),
            },
            dit_self_slots={
                "ctx":    self._g_dit.ctx,         # DiT's own cuBLAS handle
                "Q":      self._g_dit.b_q_self.data_ptr(),
                "K":      self._g_dit.b_k_self.data_ptr(),
                "V":      self._g_dit.b_v_self.data_ptr(),
                "O":      self._g_dit.b_attn_out.data_ptr(),
                "logits": self._g_dit.b_attn_logits_self.data_ptr(),
                "scale":  1.0 / math.sqrt(self._g_dit.HD),
            },
            dit_cross_slots={
                "ctx":      self._g_dit.ctx,       # same DiT handle
                "Q":        self._g_dit.b_q_cross.data_ptr(),
                "K_layers": [k.data_ptr() for k in self._g_dit._precomp_k],
                "V_layers": [v.data_ptr() for v in self._g_dit._precomp_v],
                "O":        self._g_dit.b_attn_out.data_ptr(),
                "logits":   self._g_dit.b_attn_logits_cross.data_ptr(),
                "scale":    1.0 / math.sqrt(self._g_dit.HD),
            },
        )
        # Bind into the two CKernel instances — their forward / _run_step now
        # dispatches through self.attn.run(...) instead of the direct fvk call.
        self._g_qwen3.attn = self._attn
        self._g_dit.attn = self._attn

        sig_dims = {'S': S_sig, 'D': self.D_sig, 'H': self.H_sig,
                    'NH': self.NH_sig, 'HD': self.HD_sig, 'L': self.L_sig,
                    'num_views': self._num_views, 'seq_per_view': self.spv_raw}

        # Allocate graph-persistent buffers
        self._g_sig_postln = torch.empty(S_sig, self.D_sig, dtype=fp16, device='cuda')
        self._g_mlp1_ln = torch.empty(S_img, self.mlp1_in, dtype=fp16, device='cuda')
        self._g_mlp1_fc1 = torch.empty(S_img, self.D_llm, dtype=fp16, device='cuda')
        self._g_vision_out = torch.empty(S_img, self.D_llm, dtype=fp16, device='cuda')
        self._g_vlln_buf = torch.empty(Se, self.D_llm, dtype=fp16, device='cuda')

        # ── 3. SigLIP graph ──
        stream = torch.cuda.Stream()
        s = stream.cuda_stream
        def _run_sig(si):
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights, sig_dims,
                           stream=si, attn=self._attn)
            fvk.layer_norm_fp16(self._sig_x.data_ptr(), self._sig_post_ln_w.data_ptr(),
                                self._sig_post_ln_b.data_ptr(), self._g_sig_postln.data_ptr(),
                                S_sig, self.D_sig, 1e-6, si)
        with torch.cuda.stream(stream):
            for _ in range(3):
                _run_sig(s)
        torch.cuda.synchronize()
        self._siglip_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            self._siglip_graph.capture_begin()
            _run_sig(s)
            self._siglip_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("  SigLIP graph captured")

        # Run SigLIP to get vision features for Qwen3
        views = [obs['image']]
        if 'wrist_image' in obs and self._num_views >= 2:
            views.append(obs['wrist_image'])
        self._patch_embed_2views(views)
        self._siglip_graph.replay()
        torch.cuda.synchronize()
        self._run_pixel_unshuffle_mlp1()
        torch.cuda.synchronize()

        input_embeds = self._text_embeds.clone()
        input_embeds[0, self._image_mask[0]] = self._g_vision_out.to(input_embeds.dtype)
        ie_fp16 = input_embeds.squeeze(0).to(fp16).contiguous()

        # ── 4. Calibrate + run Qwen3 ──
        self._calibrate_qwen3(ie_fp16)
        if release_full_sd:
            del self._full_sd  # free ~8GB after calibration extracted FP16 weights
            torch.cuda.empty_cache()
        self._g_qwen3.forward(ie_fp16)
        fvk.layer_norm_fp16(self._g_qwen3.b_xn.data_ptr(), vlln_w.data_ptr(),
                            vlln_b.data_ptr(), self._g_vlln_buf.data_ptr(),
                            Se, self.D_llm, 1e-5, 0)
        torch.cuda.synchronize()

        bb = self._g_vlln_buf.unsqueeze(0)
        self._g_dit.b_kv_text.copy_((bb * self._g_non_img.unsqueeze(-1).to(fp16)).squeeze(0))
        self._g_dit.b_kv_img.copy_((bb * self._g_img_m.unsqueeze(-1).to(fp16)).squeeze(0))

        # Precompute cross-attention K/V projections (constant across steps)
        self._g_dit.precompute_cross_kv()
        torch.cuda.synchronize()

        # Calibrate DiT activation scales (needs precomputed KV)
        self._calibrate_dit()

        # State encode (outside graph)
        state = obs.get('state', np.zeros(self.state_dim, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(torch.float32).cuda()
        state_fp16 = state.to(fp16).contiguous()
        if state_fp16.dim() == 1:
            state_fp16 = state_fp16.unsqueeze(0)
        h = torch.empty(1, 1024, dtype=fp16, device='cuda')
        self._g_dit.gemm.fp16_nn(state_fp16.data_ptr(), self._g_dit.se_w1.data_ptr(),
                                  h.data_ptr(), 1, 1024, self.state_dim, 0)
        fvk.add_bias_fp16(h.data_ptr(), self._g_dit.se_b1.data_ptr(), 1, 1024, 0)
        fvk.relu_inplace_fp16(h.data_ptr(), 1024, 0)
        sf = torch.empty(1, self.D_dit, dtype=fp16, device='cuda')
        self._g_dit.gemm.fp16_nn(h.data_ptr(), self._g_dit.se_w2.data_ptr(),
                                  sf.data_ptr(), 1, self.D_dit, 1024, 0)
        fvk.add_bias_fp16(sf.data_ptr(), self._g_dit.se_b2.data_ptr(), 1, self.D_dit, 0)
        torch.cuda.synchronize()
        self._g_dit.b_state_feat.copy_(sf)

        # ── 5. DiT graph capture (before Qwen3 graph for optimal cuBLAS tactic) ──
        stream_d = torch.cuda.Stream()
        sd_int = stream_d.cuda_stream
        with torch.cuda.stream(stream_d):
            for _ in range(3):
                self._g_dit.b_actions.normal_()
                for step in range(self._g_dit.num_steps):
                    self._g_dit._run_step(step, s=sd_int)
        torch.cuda.synchronize()
        self._dit_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream_d):
            self._g_dit.b_actions.normal_()
            self._dit_graph.capture_begin()
            for step in range(self._g_dit.num_steps):
                self._g_dit._run_step(step, s=sd_int)
            self._dit_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("  DiT graph captured (T=%d)", T)

        # ── 6. Qwen3 graph capture ──
        stream_q = torch.cuda.Stream()
        sq = stream_q.cuda_stream
        with torch.cuda.stream(stream_q):
            for _ in range(3):
                self._g_qwen3.forward(ie_fp16, s=sq)
        torch.cuda.synchronize()
        self._qwen3_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream_q):
            fvk.gpu_copy(self._g_qwen3.b_x.data_ptr(), ie_fp16.data_ptr(), Se * self.D_llm * 2, sq)
            self._qwen3_graph.capture_begin()
            self._g_qwen3.forward(ie_fp16, s=sq)
            fvk.layer_norm_fp16(self._g_qwen3.b_xn.data_ptr(), vlln_w.data_ptr(),
                                vlln_b.data_ptr(), self._g_vlln_buf.data_ptr(),
                                Se, self.D_llm, 1e-5, sq)
            self._qwen3_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("  Qwen3 graph captured (Se=%d)", Se)

        # Pre-allocate input_embeds buffer
        self._g_ie_buf = torch.empty(Se, self.D_llm, dtype=fp16, device='cuda')

        self._graphs_built = True
        logger.info("All CUDA Graphs ready")
