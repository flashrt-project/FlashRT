"""FlashRT -- Pi0FastTorchFrontend: Autoregressive Pi0-FAST inference.

Unlike Pi0/Pi0.5 (diffusion), Pi0-FAST generates actions as discrete tokens
via autoregressive decoding through a single Gemma 2B model.

Architecture:
  - SigLIP So400m/14 (27 layers, shared)
  - Gemma 2B (18 layers, D=2048) — both prefill and decode
  - Embedding table [257152, 2048] — input lookup + output logit projection
  - No separate action expert decoder

Usage:
    pipe = Pi0FastTorchFrontend("/path/to/orbax_checkpoint", num_views=2)
    pipe.set_prompt("pick up the red block", state=np.zeros(32))
    result = pipe.infer({"image": img1, "wrist_image": img2, "state": state})
    actions = result["actions"]
"""

import ctypes
import json
import math
import logging
import os
import pathlib
import time
from typing import Optional, Union

from flash_rt.hardware.thor.shared_primitives import (
    siglip_forward,
    postln_project,
)
from flash_rt.models.pi0fast.pipeline import (
    prefill_forward_pi0fast,
    decode_step_pi0fast,
    decode_step_pi0fast_bf16,
    prefill_calibrate_pi0fast,
    siglip_forward_sm120,
)

import numpy as np
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.quant.calibrator import load_calibration, save_calibration

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

_cudart = ctypes.CDLL("libcudart.so")

PALIGEMMA_EOS_TOKEN = 1
PALIGEMMA_PIPE_TOKEN = 235371  # "|" — Pi0-FAST action sequence end marker
FAST_SKIP_TOKENS = 128

# Pi0-FAST always generates "Action: " as the first 3 decoded tokens.
# These are deterministic (training protocol), independent of input.
_TEXT_PHASE_TOKENS = [4022, 235292, 235248]  # "Action", ":", " "


from flash_rt.core.thor_frontend_utils import quant_fp8  # noqa: E402


from flash_rt.core.thor_frontend_utils import (  # noqa: E402
    interleave_qk as _interleave_qk,
)


class Pi0FastTorchFrontend:

    def __init__(self, checkpoint_dir: str, num_views: int = 2,
                 use_cuda_graph: bool = True, autotune: int = 3,
                 max_decode_steps: int = 256,
                 decode_cuda_graph: bool = False,
                 decode_graph_steps: int = 80):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = num_views
        self.use_cuda_graph = use_cuda_graph
        self.autotune = int(autotune) if autotune is not True else 3
        if autotune is False:
            self.autotune = 0
        self.max_decode_steps = max_decode_steps
        self.decode_cuda_graph = decode_cuda_graph
        self.decode_graph_steps = decode_graph_steps
        self._n_text_steps = 3  # "Action", ":", " " — deterministic for Pi0-FAST
        self.latency_records = []
        self.calibrated = False
        self.graph_captured = False
        self._real_data_calibrated = False

        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()

        # Load FMHA
        fmha_paths = [
            str(checkpoint_dir.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(__file__).parent.parent.parent / "libfmha_fp16_strided.so"),
        ]
        for p in fmha_paths:
            if pathlib.Path(p).exists():
                ret = fvk.load_fmha_strided_library(p)
                if ret == 0:
                    logger.info("CUTLASS FMHA loaded from %s", p)
                    break

        # Load norm stats
        self._load_norm_stats(checkpoint_dir)

        # Load sentencepiece tokenizer
        self._load_tokenizer()

        # Load weights from safetensors
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path}. "
                "Convert Orbax checkpoint first: "
                "python examples/convert_pi0fast_orbax_to_safetensors.py "
                f"--orbax_dir /path/to/pi0_fast_base --output_dir {checkpoint_dir}")
        self._checkpoint_path = str(safetensors_path)
        self._load_weights(safetensors_path)

        # SM dispatch probe. Thor SM100/SM110 has CUTLASS FP8 kernels compiled
        # in (``cutlass_fp8_sq`` et al); RTX 5090 SM120 build does not and must
        # route prefill GEMMs through cuBLASLt (``fp8_gemm_descale_*``) using
        # the flat weight buffers. Decode path already uses cuBLASLt on both
        # platforms except for the optional ``cutlass_fp8_wide`` gate+up
        # optimization, which we skip on SM120 by not passing ``gate_w_list``.
        self._has_sm100 = hasattr(fvk, 'cutlass_fp8_sq')

        logger.info("Pi0FastTorchFrontend initialised (num_views=%d, max_decode=%d, sm100=%s)",
                     num_views, max_decode_steps, self._has_sm100)

    def _load_norm_stats(self, checkpoint_dir):
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "franka" / "norm_stats.json",
            checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            *lerobot_candidates(checkpoint_dir),
        ]
        # Pi0-FAST has historically tolerated missing stats (decoder-only
        # path can run without them), so keep ``strict=False`` here.
        self.norm_stats = load_norm_stats(
            candidates, checkpoint_dir=checkpoint_dir, strict=False)
        if self.norm_stats is None:
            logger.warning("norm_stats not found — using dummy stats")

    def _load_tokenizer(self):
        # Locate via the shared FlashRT helper (env var → XDG cache →
        # legacy paths → openpi auto-download). Raises a clear
        # FileNotFoundError with a copy-pasteable curl command if the
        # tokenizer isn't installed; see USAGE.md → 'PaliGemma
        # tokenizer setup'.
        self._sp_tokenizer = None
        try:
            from flash_rt.utils.paligemma_tokenizer import (
                load_paligemma_sentencepiece,
            )
            self._sp_tokenizer = load_paligemma_sentencepiece()
            self._vocab_size = self._sp_tokenizer.GetPieceSize()
            logger.info(
                "PaliGemma SentencePiece loaded (vocab=%d)",
                self._vocab_size)
        except (ImportError, FileNotFoundError, OSError, RuntimeError) as e:
            logger.warning(
                "PaliGemma SentencePiece not available (%s); "
                "falling back to vocab=257152 default", e)
            self._vocab_size = 257152

        # Try loading FAST tokenizer (local only, skip network)
        self._fast_tokenizer = None
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            from transformers import AutoProcessor
            # Try a local snapshot first, then fall back to HF cache
            for tok_src in ['/workspace/fast_tokenizer',
                            'physical-intelligence/fast']:
                try:
                    self._fast_tokenizer = AutoProcessor.from_pretrained(
                        tok_src, trust_remote_code=True, local_files_only=True)
                    logger.info("FAST tokenizer loaded from %s", tok_src)
                    break
                except Exception:
                    continue
        except Exception:
            logger.info("FAST tokenizer not cached locally, skipping")

    def _load_weights(self, safetensors_path):
        """Load weights from safetensors (converted from Orbax via convert_pi0fast_orbax_to_safetensors.py)."""
        from safetensors import safe_open
        from flash_rt.executors.torch_weights import _autodetect_strip_prefix

        sf = safe_open(str(safetensors_path), framework='pt', device='cuda')
        # Auto-strip the lerobot HF policy ``model.`` namespace wrap if
        # present (no-op on existing openpi-converted ckpts).
        _strip = _autodetect_strip_prefix(set(sf.keys()))
        def g_raw(k): return sf.get_tensor((_strip + k) if _strip else k)
        def g(k): return g_raw(k).to(fp16)

        nv = self.num_views

        # Key prefixes (match convert_pi0fast_orbax_to_safetensors.py output)
        vp = 'paligemma.model.vision_tower.vision_model'
        ep = 'paligemma.model.language_model.layers'

        # ============================================================
        # Embedding table (shared input/output projection)
        # ============================================================
        self.embedding_weight = g('paligemma.lm_head.weight')  # [257152, 2048]
        self._vocab_size = self.embedding_weight.shape[0]

        # ============================================================
        # SigLIP (27 layers) — identical to Pi0 Torch
        # ============================================================
        S_sig = nv * 256
        D_sig, H_sig, NH_sig, HD_sig, L_sig = 1152, 4304, 16, 72, 27
        self.sig_S = S_sig
        self.sig_D = D_sig
        self.sig_H = H_sig
        self.sig_NH = NH_sig
        self.sig_HD = HD_sig
        self.sig_L = L_sig

        sig_ln_attn_w, sig_ln_attn_b = [], []
        sig_ln_ffn_w, sig_ln_ffn_b = [], []
        sig_qkv_w, sig_qkv_b = [], []
        sig_o_w, sig_o_b = [], []
        sig_up_w, sig_up_b = [], []
        sig_down_w, sig_down_b = [], []
        sig_alpha = []

        for i in range(L_sig):
            lp = f'{vp}.encoder.layers.{i}'
            sig_ln_attn_w.append(g(f'{lp}.layer_norm1.weight'))
            sig_ln_attn_b.append(g(f'{lp}.layer_norm1.bias'))
            sig_ln_ffn_w.append(g(f'{lp}.layer_norm2.weight'))
            sig_ln_ffn_b.append(g(f'{lp}.layer_norm2.bias'))

            qkv_cat = torch.cat([g(f'{lp}.self_attn.{p}_proj.weight')
                                 for p in 'qkv'], dim=0)
            qkv_fp8, qs = quant_fp8(qkv_cat.T.contiguous())
            sig_qkv_w.append(qkv_fp8)
            sig_qkv_b.append(torch.cat([g(f'{lp}.self_attn.{p}_proj.bias')
                                        for p in 'qkv'], dim=0))

            o_fp8, os_ = quant_fp8(g(f'{lp}.self_attn.out_proj.weight').T.contiguous())
            sig_o_w.append(o_fp8)
            sig_o_b.append(g(f'{lp}.self_attn.out_proj.bias'))

            up_fp8, us = quant_fp8(g(f'{lp}.mlp.fc1.weight').T.contiguous())
            sig_up_w.append(up_fp8)
            sig_up_b.append(g(f'{lp}.mlp.fc1.bias'))

            down_fp8, ds = quant_fp8(g(f'{lp}.mlp.fc2.weight').T.contiguous())
            sig_down_w.append(down_fp8)
            sig_down_b.append(g(f'{lp}.mlp.fc2.bias'))

            sig_alpha.extend([qs, os_, us, ds])

        self._sig_weights = {
            'ln_attn_w': [w.data_ptr() for w in sig_ln_attn_w],
            'ln_attn_b': [w.data_ptr() for w in sig_ln_attn_b],
            'qkv_w': [w.data_ptr() for w in sig_qkv_w],
            'qkv_b': [w.data_ptr() for w in sig_qkv_b],
            'o_w': [w.data_ptr() for w in sig_o_w],
            'o_b': [w.data_ptr() for w in sig_o_b],
            'ln_ffn_w': [w.data_ptr() for w in sig_ln_ffn_w],
            'ln_ffn_b': [w.data_ptr() for w in sig_ln_ffn_b],
            'up_w': [w.data_ptr() for w in sig_up_w],
            'up_b': [w.data_ptr() for w in sig_up_b],
            'down_w': [w.data_ptr() for w in sig_down_w],
            'down_b': [w.data_ptr() for w in sig_down_b],
            'alpha': sig_alpha,
        }
        self._unit_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        self._sig_weights['unit_scale'] = self._unit_scale.data_ptr()
        # SM120 SigLIP path needs a device tensor of per-layer weight scales
        # (layer-major qkv / o / up / down, len = L_sig*4). SM100 uses
        # host-side ``sig_alpha``; SM120 uses this device pointer directly
        # via ``fp8_gemm_descale_fp16``. The tensor is built unconditionally
        # — it's tiny (432 bytes for L=27) and keeps the buffer layout
        # symmetric between platforms.
        self._sig_w_scales_dev = torch.tensor(
            sig_alpha, dtype=torch.float32, device='cuda')
        self._sig_weights['w_scales_dev'] = self._sig_w_scales_dev.data_ptr()
        self._sig_tensors = (sig_ln_attn_w, sig_ln_attn_b, sig_ln_ffn_w, sig_ln_ffn_b,
                             sig_qkv_w, sig_qkv_b, sig_o_w, sig_o_b,
                             sig_up_w, sig_up_b, sig_down_w, sig_down_b)

        # SigLIP buffers
        self._sig_x = torch.zeros(S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_x_fp8 = torch.zeros(S_sig * D_sig, dtype=torch.uint8, device='cuda')
        self._sig_qkv = torch.empty(S_sig, 3 * D_sig, dtype=fp16, device='cuda')
        self._sig_attn = torch.empty(S_sig, D_sig, dtype=fp16, device='cuda')
        self._sig_hidden = torch.empty(S_sig, H_sig, dtype=fp16, device='cuda')
        self._sig_hid_fp8 = torch.zeros(S_sig * H_sig, dtype=torch.uint8, device='cuda')

        self._sig_bufs = {
            'x': self._sig_x.data_ptr(),
            'x_fp8': self._sig_x_fp8.data_ptr(),
            'qkv': self._sig_qkv.data_ptr(),
            'attn_out': self._sig_attn.data_ptr(),
            'hidden': self._sig_hidden.data_ptr(),
            'hid_fp8': self._sig_hid_fp8.data_ptr(),
            # SM120-only keys (ignored by thor siglip_forward): persistent
            # torch tensor views aliasing the above device buffers, used
            # by siglip_forward_sm120 for SDPA-based attention.
            'qkv_t': self._sig_qkv,
            'attn_t': self._sig_attn,
        }
        self._sig_dims = {
            'S': S_sig, 'D': D_sig, 'H': H_sig,
            'NH': NH_sig, 'HD': HD_sig, 'L': L_sig,
            'num_views': nv, 'seq_per_view': 256,
        }

        # Patch embedding
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

        # PostLN
        self._postln_w = g(f'{vp}.post_layernorm.weight')
        self._postln_b = g(f'{vp}.post_layernorm.bias')
        mp = 'paligemma.model.multi_modal_projector.linear'
        self._proj_w = g(f'{mp}.weight').T.contiguous()
        self._proj_b = g(f'{mp}.bias')
        self._postln_scratch = torch.empty(S_sig, max(D_sig, H_sig), dtype=fp16, device='cuda')

        # ============================================================
        # Encoder / LLM (18 layers, GQA 8:1) — same as Pi0 Torch encoder
        # Pi0-FAST: no separate decoder, encoder IS the full model
        # ============================================================
        De, He, Le = 2048, 16384, 18
        NHe, HDe = 8, 256
        Se_max = nv * 256 + 256
        max_total_keys = Se_max + self.max_decode_steps
        self.De = De; self.He = He; self.Le = Le
        self.NHe = NHe; self.HDe = HDe; self.Se_max = Se_max
        self.max_total_keys = max_total_keys

        enc_qkv_w, enc_o_w, enc_gu_w, enc_d_w = [], [], [], []
        enc_qkv_fp16, enc_o_fp16, enc_gu_fp16, enc_d_fp16 = [], [], [], []
        enc_w_scales = []

        for i in range(Le):
            # Fuse RMSNorm weight into QKV (same as Pi0 Torch encoder)
            asc = g_raw(f'{ep}.{i}.input_layernorm.weight').float()
            fa = 1.0 + asc
            q = _interleave_qk(g_raw(f'{ep}.{i}.self_attn.q_proj.weight').float(), 8) * fa.unsqueeze(0)
            k = _interleave_qk(g_raw(f'{ep}.{i}.self_attn.k_proj.weight').float(), 1) * fa.unsqueeze(0)
            v = g_raw(f'{ep}.{i}.self_attn.v_proj.weight').float() * fa.unsqueeze(0)
            qkv_cat = torch.cat([q, k, v], dim=0).to(fp16)
            qw8, qs = quant_fp8(qkv_cat)
            enc_qkv_w.append(qw8)
            enc_qkv_fp16.append(qkv_cat)  # keep FP16 for calibration

            o_fp16_w = g(f'{ep}.{i}.self_attn.o_proj.weight')
            ow8, os_ = quant_fp8(o_fp16_w)
            enc_o_w.append(ow8)
            enc_o_fp16.append(o_fp16_w)

            # Fuse RMSNorm weight into GateUp
            fsc = g_raw(f'{ep}.{i}.post_attention_layernorm.weight').float()
            ff = 1.0 + fsc
            gw = (g_raw(f'{ep}.{i}.mlp.gate_proj.weight').float() * ff.unsqueeze(0)).to(fp16)
            uw = (g_raw(f'{ep}.{i}.mlp.up_proj.weight').float() * ff.unsqueeze(0)).to(fp16)
            gu_cat = torch.cat([gw, uw], dim=0)
            guw8, gus = quant_fp8(gu_cat)
            enc_gu_w.append(guw8)
            enc_gu_fp16.append(gu_cat)

            d_fp16_w = g(f'{ep}.{i}.mlp.down_proj.weight')
            dw8, ds = quant_fp8(d_fp16_w)
            enc_d_w.append(dw8)
            enc_d_fp16.append(d_fp16_w)

            enc_w_scales.extend([qs, os_, gus, ds])

        self._enc_qkv_w = enc_qkv_w
        self._enc_o_w = enc_o_w
        self._enc_gu_w = enc_gu_w
        self._enc_d_w = enc_d_w
        self._enc_qkv_fp16 = enc_qkv_fp16
        self._enc_o_fp16 = enc_o_fp16
        self._enc_gu_fp16 = enc_gu_fp16
        self._enc_d_fp16 = enc_d_fp16
        self._enc_w_dev = torch.tensor(enc_w_scales, dtype=torch.float32, device='cuda')

        # Flat weight tensors for decode_step (fp8_gemm_descale_fp16 needs [K, N] layout)
        def _transpose_fp8_flat(weights_list, N, K):
            parts = []
            for w in weights_list:
                w_t = w.reshape(N, K).T.contiguous()
                parts.append(w_t.reshape(-1))
            return torch.cat(parts)

        self._enc_qkv_flat = _transpose_fp8_flat(enc_qkv_w, 2560, De)
        self._enc_o_flat = _transpose_fp8_flat(enc_o_w, De, De)
        self._enc_gu_flat = _transpose_fp8_flat(enc_gu_w, He * 2, De)
        self._enc_d_flat = _transpose_fp8_flat(enc_d_w, De, He)

        # Final RMSNorm weight — need BOTH BF16 (for prefill's rms_norm bf16 variant)
        # AND FP16 (for decode_step's rms_norm_fp16). Weights are O(1), safe in either.
        final_norm_raw = g_raw('paligemma.model.language_model.norm.weight').float()
        self._final_norm_w = (1.0 + final_norm_raw).to(torch.bfloat16)   # BF16 for prefill
        self._final_norm_w_fp16 = (1.0 + final_norm_raw).to(fp16)        # FP16 for decode

        # RoPE table
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256))
        kp = inv_freq[None, :] * torch.arange(Se_max + self.max_decode_steps, device='cuda')[:, None].float()
        self._kc_t = torch.cos(kp).to(fp16)
        self._ks_t = torch.sin(kp).to(fp16)
        # Full rope table for all positions
        self._full_rope = torch.cat([self._kc_t[:, :, None], self._ks_t[:, :, None]],
                                     dim=2).reshape(-1, 256).contiguous()
        self._enc_rope = torch.empty(Se_max, 256, dtype=fp16, device='cuda')

        # KV cache (shared prefill + decode)
        self._Kc = torch.zeros(Le, max_total_keys, HDe, dtype=fp16, device='cuda')
        self._Vc = torch.zeros(Le, max_total_keys, HDe, dtype=fp16, device='cuda')

        # Encoder/prefill buffers (sized for Se_max)
        # BF16 for residual stream (x, fg, xn) — Pi0-FAST hidden states reach ~569K
        bf16 = torch.bfloat16
        self._enc_x = torch.empty(Se_max, De, dtype=bf16, device='cuda')       # BF16 residual
        # postln_project (shared with Pi0/Pi0.5) writes FP16 bytes; we need FP16
        # staging buffer to receive its output, then explicitly cast → BF16 _enc_x.
        # Without this, the FP16 byte pattern is misinterpreted as BF16 (5e19 garbage).
        self._enc_x_fp16_postln = torch.empty(Se_max, De, dtype=fp16, device='cuda')
        self._enc_x_fp8 = torch.zeros(Se_max * De, dtype=torch.uint8, device='cuda')
        self._enc_qkv_buf = torch.empty(Se_max, 2560, dtype=fp16, device='cuda')  # FP16 (QKV safe)
        self._enc_logits = torch.empty(Se_max * NHe, max_total_keys, dtype=fp16, device='cuda')
        self._enc_attn = torch.empty(Se_max, NHe * HDe, dtype=fp16, device='cuda')
        self._enc_o_fp8 = torch.zeros(Se_max * NHe * HDe, dtype=torch.uint8, device='cuda')
        self._enc_gate = torch.empty(Se_max, 2 * He, dtype=fp16, device='cuda')  # FP16 (GELU safe)
        self._enc_hidden = torch.empty(Se_max, He, dtype=fp16, device='cuda')
        self._enc_hid_fp8 = torch.zeros(Se_max * He, dtype=torch.uint8, device='cuda')
        self._enc_fg = torch.empty(Se_max, De, dtype=bf16, device='cuda')       # BF16 (GEMM output)
        self._enc_fg_f32 = torch.empty(Se_max, De, dtype=torch.float32, device='cuda')  # FP32 scratch (unused with BF16 path)
        self._enc_xn = torch.empty(Se_max, De, dtype=bf16, device='cuda')       # BF16 (norm output)

        # Decode step buffers (M=1) — legacy FP16 path (decode_step_pi0fast).
        # Kept for backward-compat; the BF16 buffers below are what infer() uses.
        self._dec_x = torch.empty(1, De, dtype=fp16, device='cuda')
        self._dec_x_fp8 = torch.zeros(1 * De, dtype=torch.uint8, device='cuda')
        self._dec_qkv = torch.empty(1, 2560, dtype=fp16, device='cuda')
        self._dec_logits = torch.empty(1 * NHe, max_total_keys, dtype=fp16, device='cuda')
        self._dec_attn = torch.empty(1, NHe * HDe, dtype=fp16, device='cuda')
        self._dec_o_fp8 = torch.zeros(1 * NHe * HDe, dtype=torch.uint8, device='cuda')
        self._dec_gate = torch.empty(1, 2 * He, dtype=fp16, device='cuda')
        self._dec_hid_fp8 = torch.zeros(1 * He, dtype=torch.uint8, device='cuda')
        self._dec_fg = torch.empty(1, De, dtype=fp16, device='cuda')
        self._dec_xn = torch.empty(1, De, dtype=fp16, device='cuda')

        # Decode step BF16 buffers — Pi0-FAST hidden state can reach ~569K, which
        # exceeds FP16 max (65504). Only the residual chain (x, fg, xn) is BF16;
        # qkv/attn/gate stay FP16 because their consumers (qkv_split_rope_kvcache_fp16,
        # attention_qkv_fp16, gate_geglu_merged_fp8_fp16) only have FP16 variants
        # and Gemma 2B values at those points are well within FP16 range.
        self._dec_x_bf16 = torch.empty(1, De, dtype=bf16, device='cuda')
        self._dec_fg_bf16 = torch.empty(1, De, dtype=bf16, device='cuda')
        self._dec_xn_bf16 = torch.empty(1, De, dtype=bf16, device='cuda')
        # Scratch for SM120 split-K Down GEMM workaround (see pipeline
        # decode_step_pi0fast_bf16 _need_down_split comment). Only read on
        # SM120; allocated unconditionally since it's only 4KB.
        self._dec_fg_scratch = torch.empty(1, De, dtype=bf16, device='cuda')

        # Logit output buffer (full vocab)
        self._logit_buf = torch.empty(1, self._vocab_size, dtype=fp16, device='cuda')

        # Vocab pruning: precompute action token sub-matrix
        # FAST tokens map to PaliGemma vocab tail: vocab_size - 1 - 128 - fast_id
        # Conservative range: last 2048 tokens before the 128 special tokens
        self._action_vocab_size = 2048
        self._action_range_start = self._vocab_size - FAST_SKIP_TOKENS - self._action_vocab_size
        self._action_range_end = self._vocab_size - FAST_SKIP_TOKENS
        self._action_embed = self.embedding_weight[
            self._action_range_start:self._action_range_end].contiguous()  # [2048, De] ~8MB
        self._action_logit_buf = torch.empty(1, self._action_vocab_size, dtype=fp16, device='cuda')
        logger.info("Action vocab pruning: range [%d, %d) (%d tokens, %.1f MB)",
                     self._action_range_start, self._action_range_end,
                     self._action_vocab_size,
                     self._action_vocab_size * De * 2 / 1e6)

        # Staging buffer: BF16 → FP16 cast for last_hidden inside prefill graph.
        # .to(torch.float16) allocates; .copy_() into static buffer does not.
        self._last_hidden_fp16 = torch.empty(1, De, dtype=fp16, device='cuda')

        # Calibration scales
        self._enc_calib_scales = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')

        self._prefill_graph = None   # set by _capture_prefill_graph()
        self._decode_graph = None    # set by _capture_decode_action_graph()
        if self.decode_cuda_graph:
            self._decode_tok_buf = torch.zeros(self.decode_graph_steps, dtype=torch.long, device='cuda')

        torch.cuda.empty_cache()
        logger.info("Weights loaded for Pi0-FAST (embedding=%dx%d)",
                     self._vocab_size, De)

    # -------------------------------------------------------------------
    # set_prompt
    # -------------------------------------------------------------------

    def set_prompt(self, prompt_text, state=None):
        S_sig = self.sig_S
        nv = self.num_views

        # Tokenize prefix
        if isinstance(prompt_text, (np.ndarray, list)):
            token_ids = np.asarray(prompt_text, dtype=np.int64)
        else:
            token_ids = self._tokenize_prefix(prompt_text, state)

        prompt_len = len(token_ids)

        # Embed prefix tokens
        token_ids_t = torch.from_numpy(token_ids).long().cuda()
        embeds = F.embedding(token_ids_t, self.embedding_weight)
        embeds = embeds * float(self.De ** 0.5)

        # Se must be EVEN for cuBLASLt FP8
        Se = S_sig + prompt_len
        if Se % 2 != 0:
            Se += 1
        self.Se = Se
        self.prefill_len = Se  # KV cache positions 0..Se-1 filled by prefill
        actual_lang = Se - S_sig
        if actual_lang > prompt_len:
            embeds = torch.cat([embeds, embeds[-1:]], dim=0)
        self._lang_emb = embeds
        self._S_lang = actual_lang

        # RoPE for prefill
        self._enc_rope[:Se].copy_(self._full_rope[:Se])

        # Capture SigLIP graph
        self._capture_siglip_graph()

        # Calibrate FP8 scales
        self._calibrate(Se)

        # Capture prefill graph (must be AFTER calibrate — alpha_host baked in)
        self._capture_prefill_graph()

        # Optionally capture decode action graph (trades startup for per-token speed)
        if self.decode_cuda_graph:
            self._capture_decode_action_graph()

        self.graph_captured = True
        self.calibrated = True
        logger.info("set_prompt done: %d tokens, Se=%d", prompt_len, Se)

    def _tokenize_prefix(self, prompt_text, state=None):
        # Match JAX FASTTokenizer.tokenize exactly: prefix ends in ";\n", NOT
        # ";\nAction: ". The model is trained to generate "Action: " itself as the
        # first 3 decoded tokens. Pre-injecting "Action: " puts those 3 tokens in
        # the bidirectional-attention prefix region, which is NOT how the model
        # was trained — it expects them in the causal autoregressive region. This
        # changes the attention computation and produces different first action
        # token logits than JAX, breaking equivalence.
        cleaned = prompt_text.lower().strip().replace("_", " ")
        if state is not None:
            discretized = np.digitize(
                np.asarray(state, dtype=np.float32),
                bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized.astype(int)))
            full_prompt = f"Task: {cleaned}, State: {state_str};\n"
        else:
            full_prompt = f"Task: {cleaned};\n"

        if self._sp_tokenizer is not None:
            tokens = self._sp_tokenizer.Encode(full_prompt)
            return np.array([self._sp_tokenizer.bos_id()] + tokens, dtype=np.int64)
        else:
            return np.zeros(10, dtype=np.int64)

    # -------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------

    def _calibrate(self, Se, force=False):
        Le = self.Le; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        total_keys_max = self.max_total_keys

        # Try cache first (matches Pi0/Pi0.5 pattern)
        if not force:
            cached = load_calibration(self._checkpoint_path, Se)
            if cached is not None:
                self._enc_calib_scales = torch.tensor(
                    cached["enc_scales"], dtype=torch.float32, device='cuda')
                enc_ws = self._enc_w_dev.cpu().tolist()
                self._enc_alpha_host = [
                    float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
                    for i in range(Le * 4)]
                logger.info("Calibration loaded from cache (%d scales)", Le * 4)
                return

        enc_bufs = {
            'x': self._enc_x.data_ptr(),
            'x_fp8': self._enc_x_fp8.data_ptr(),
            'qkv': self._enc_qkv_buf.data_ptr(),
            'logits': self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8': self._enc_o_fp8.data_ptr(),
            'gate': self._enc_gate.data_ptr(),
            'hidden': self._enc_hidden.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg': self._enc_fg.data_ptr(),
            'fg_f32': self._enc_fg_f32.data_ptr(),
            'ctx': self._ctx,
        }
        enc_weights = {
            'qkv_w': [w.data_ptr() for w in self._enc_qkv_w],
            'o_w': [w.data_ptr() for w in self._enc_o_w],
            'gate_w': [w.data_ptr() for w in self._enc_gu_w],
            'down_w': [w.data_ptr() for w in self._enc_d_w],
            # FP16 weights for calibration (pure FP16 forward, no FP8 error)
            'qkv_w_fp16': [w.data_ptr() for w in self._enc_qkv_fp16],
            'o_w_fp16': [w.data_ptr() for w in self._enc_o_fp16],
            'gate_w_fp16': [w.data_ptr() for w in self._enc_gu_fp16],
            'down_w_fp16': [w.data_ptr() for w in self._enc_d_fp16],
            # Flat FP8 weight buffers for SM120 cuBLASLt path
            'qkv_w_flat':  self._enc_qkv_flat.data_ptr(),
            'o_w_flat':    self._enc_o_flat.data_ptr(),
            'gate_w_flat': self._enc_gu_flat.data_ptr(),
            'down_w_flat': self._enc_d_flat.data_ptr(),
            'rope': self._enc_rope.data_ptr(),
            'Kc': self._Kc.reshape(-1).data_ptr(),
            'Vc': self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys_max': total_keys_max,
        }

        _norm_scratch = torch.empty(Se * De, dtype=torch.bfloat16, device='cuda')  # BF16
        _x_scratch = torch.empty(Se * De, dtype=torch.bfloat16, device='cuda')      # BF16
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch = torch.zeros(Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=torch.bfloat16, device='cuda')  # BF16 for BF16 rms_norm
        enc_bufs['norm_scratch'] = _norm_scratch.data_ptr()
        enc_bufs['x_scratch'] = _x_scratch.data_ptr()
        enc_bufs['calib_buf'] = _calib_buf.data_ptr()
        enc_bufs['d_scale'] = _d_scale.data_ptr()
        enc_bufs['fp8_scratch'] = _fp8_scratch.data_ptr()
        enc_bufs['ones'] = _ones.data_ptr()

        # Multi-sample calibration with light perturbations + max aggregation.
        # Restored after the postln BF16 bug fix: with correct prefix values,
        # multi-sample max gives better coverage of activation extremes than
        # single-sample on the bare real prefix.
        n_calib_samples = 16
        enc_x_backup = self._enc_x[:Se].clone()
        x_std = enc_x_backup.std()
        all_scales = []

        for sample_idx in range(n_calib_samples):
            torch.manual_seed(42 + sample_idx)
            if sample_idx == 0:
                self._enc_x[:Se].copy_(enc_x_backup)
            else:
                noise_scale = 0.05 + 0.03 * sample_idx
                noise = torch.randn_like(enc_x_backup) * noise_scale * x_std
                self._enc_x[:Se].copy_(enc_x_backup + noise)

            self._Kc.zero_(); self._Vc.zero_()
            sample_scales = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
            prefill_calibrate_pi0fast(
                self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                sample_scales.data_ptr(), stream=0)
            torch.cuda.synchronize()
            s = sample_scales.cpu()
            if not (torch.isinf(s).any() or torch.isnan(s).any()):
                all_scales.append(s)

        self._enc_x[:Se].copy_(enc_x_backup)

        if len(all_scales) >= 2:
            stacked = torch.stack(all_scales, dim=0)
            best_scales = stacked.float().max(dim=0).values.to(torch.float32).cuda()
        elif len(all_scales) == 1:
            best_scales = all_scales[0].cuda()
        else:
            logger.warning("All calibration samples failed! Using single-pass.")
            self._enc_x[:Se].copy_(enc_x_backup)
            self._Kc.zero_(); self._Vc.zero_()
            best_scales = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
            prefill_calibrate_pi0fast(
                self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                best_scales.data_ptr(), stream=0)
            torch.cuda.synchronize()

        self._enc_calib_scales = best_scales
        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]
        logger.info("Calibrated: %d scales (%d samples)", Le * 4, n_calib_samples)

        # Save to cache (Pi0-FAST has no decoder scales → ae_scales=[])
        try:
            save_calibration(
                checkpoint_path=self._checkpoint_path,
                Se=Se,
                enc_scales=self._enc_calib_scales.cpu().tolist(),
                enc_alpha=self._enc_alpha_host,
                ae_scales=[],
                enc_w_scales=enc_ws,
            )
        except Exception as e:
            logger.warning("Failed to save calibration cache: %s", e)

    # -------------------------------------------------------------------
    # Patch embed + PostLN
    # -------------------------------------------------------------------

    def _patch_embed_ops(self, stream_int):
        S_sig, D_sig = self.sig_S, self.sig_D
        fvk.patch_im2col(self._img_buf.ptr.value, self._patches_buf.ptr.value,
                         self.num_views, stream_int)
        self._gemm.fp16_nn(self._patches_buf.ptr.value, self._pe_w.ptr.value,
                           self._sig_x.data_ptr(), S_sig, D_sig, 588, stream_int)
        fvk.patch_embed_bias_pos(self._sig_x.data_ptr(), self._pe_b.ptr.value,
                                 self._pos_emb.ptr.value, S_sig, D_sig, 256, stream_int)

    def _postln_project_ops(self, stream_int):
        S_sig = self.sig_S; D_sig = self.sig_D; De = self.De
        # postln_project writes FP16 bytes — direct it at our FP16 staging buffer.
        postln_bufs = {
            'x_sig': self._sig_x.data_ptr(),
            'enc_x': self._enc_x_fp16_postln.data_ptr(),
            'scratch': self._postln_scratch.data_ptr(),
        }
        postln_weights = {
            'ln_w': self._postln_w.data_ptr(),
            'ln_b': self._postln_b.data_ptr(),
            'proj_w': self._proj_w.data_ptr(),
            'proj_b': self._proj_b.data_ptr(),
            'lang_emb': self._lang_emb.data_ptr(),
        }
        postln_dims = {
            'S_sig': S_sig, 'D_sig': D_sig,
            'D_enc': De, 'S_lang': self._S_lang,
        }
        postln_project(self._gemm, fvk, postln_bufs, postln_weights,
                       postln_dims, stream=stream_int)
        # Cast FP16 → BF16 into the residual stream buffer.
        # postln_project writes FP16 bytes into _enc_x_fp16_postln; we then
        # copy_ to BF16 _enc_x which performs implicit dtype conversion in-place
        # (no allocation, graph-safe). This bridges the v1.6 BF16 migration boundary.
        Se = getattr(self, "Se", None)
        if Se is not None:
            self._enc_x[:Se].copy_(self._enc_x_fp16_postln[:Se])

    # -------------------------------------------------------------------
    # CUDA Graph capture
    # -------------------------------------------------------------------

    def _capture_siglip_graph(self):
        # SigLIP dispatch: SM100 uses fused FP8 epilogue kernels
        # (fp8_nn_bias / _bias_res / _gelu_bias) which are rejected by
        # cuBLASLt on SM120 with CUBLAS_STATUS_NOT_SUPPORTED. SM120 takes
        # the decomposed path (fp8_gemm_descale_fp16 + separate bias /
        # residual / gelu kernels) defined in pipeline_pi0fast.
        siglip_fn = siglip_forward if self._has_sm100 else siglip_forward_sm120

        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)
        for _ in range(3):
            self._patch_embed_ops(0)
            self._sig_x.zero_()
            siglip_fn(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                      self._sig_dims, stream=0)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        self._siglip_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._siglip_graph.capture_begin()
            self._patch_embed_ops(s_int)
            siglip_fn(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                      self._sig_dims, stream=s_int)
            self._postln_project_ops(s_int)
            self._siglip_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("SigLIP graph captured (S=%d, sm100=%s)",
                     self.sig_S, self._has_sm100)

    def _capture_prefill_graph(self):
        """Capture prefill forward + first logit projection as CUDA Graph.

        Mirrors Pi0's _capture_enc_ae_graph pattern.  Must be called AFTER
        _calibrate() because alpha_host values are baked into the graph.
        """
        Se = self.Se
        De = self.De; He = self.He; Le = self.Le
        NHe = self.NHe; HDe = self.HDe

        prefill_bufs = {
            'x': self._enc_x.data_ptr(),
            'x_fp8': self._enc_x_fp8.data_ptr(),
            'qkv': self._enc_qkv_buf.data_ptr(),
            'logits': self._enc_logits.data_ptr(),
            'attn_out': self._enc_attn.data_ptr(),
            'o_fp8': self._enc_o_fp8.data_ptr(),
            'gate': self._enc_gate.data_ptr(),
            'hid_fp8': self._enc_hid_fp8.data_ptr(),
            'fg': self._enc_fg.data_ptr(),
            'xn': self._enc_xn.data_ptr(),
            'ctx': self._ctx,
        }
        prefill_weights = {
            # SM100 path (CUTLASS) — per-layer [N,K] weight pointers + host alphas
            'qkv_w': [w.data_ptr() for w in self._enc_qkv_w],
            'o_w':   [w.data_ptr() for w in self._enc_o_w],
            'gate_w': [w.data_ptr() for w in self._enc_gu_w],
            'down_w': [w.data_ptr() for w in self._enc_d_w],
            'alpha_host':   self._enc_alpha_host,
            # SM120 path (cuBLASLt) — flat [K,N] weight buffers + device scales
            'qkv_w_flat':  self._enc_qkv_flat.data_ptr(),
            'o_w_flat':    self._enc_o_flat.data_ptr(),
            'gate_w_flat': self._enc_gu_flat.data_ptr(),
            'down_w_flat': self._enc_d_flat.data_ptr(),
            'w_scales':    self._enc_w_dev.data_ptr(),
            # Shared
            'rope':  self._enc_rope.data_ptr(),
            'Kc':    self._Kc.reshape(-1).data_ptr(),
            'Vc':    self._Vc.reshape(-1).data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'act_scales':   self._enc_calib_scales.data_ptr(),
        }
        prefill_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys_max': self.max_total_keys,
        }

        # Warmup 3 times (PyTorch CUDAGraph requirement)
        # T1 opt: skip full-vocab logit projection — first token is hardcoded
        for _ in range(3):
            self._Kc.zero_(); self._Vc.zero_()
            prefill_forward_pi0fast(self._gemm, fvk, prefill_bufs,
                                    prefill_weights, prefill_dims, stream=0)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        self._prefill_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._prefill_graph.capture_begin()
            self._Kc.zero_(); self._Vc.zero_()
            prefill_forward_pi0fast(self._gemm, fvk, prefill_bufs,
                                    prefill_weights, prefill_dims, stream=s_int)
            self._prefill_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("Prefill graph captured (Se=%d)", Se)

    def _capture_decode_action_graph(self):
        """Capture action-phase decode loop as a single CUDA graph.

        Unrolls decode_graph_steps iterations of:
          decode_step_bf16 → action_logit → GPU argmax → GPU embed → next step
        All GPU-only (no D2H, no CPU branch). Same principle as Pi0's
        10-step diffusion loop graph, but with per-step S_kv/offset/RoPE.
        """
        Se = self.Se; De = self.De; He = self.He
        Le = self.Le; NHe = self.NHe; HDe = self.HDe
        M = self.decode_graph_steps
        n_text = self._n_text_steps
        sqrt_D = float(De ** 0.5)

        decode_bufs = {
            'x': self._dec_x_bf16.data_ptr(),
            'x_fp8': self._dec_x_fp8.data_ptr(),
            'qkv': self._dec_qkv.data_ptr(),
            'logits': self._dec_logits.data_ptr(),
            'attn_out': self._dec_attn.data_ptr(),
            'o_fp8': self._dec_o_fp8.data_ptr(),
            'gate': self._dec_gate.data_ptr(),
            'hid_fp8': self._dec_hid_fp8.data_ptr(),
            'fg': self._dec_fg_bf16.data_ptr(),
            'xn': self._dec_xn_bf16.data_ptr(),
            'fg_scratch': self._dec_fg_scratch.data_ptr(),
        }
        decode_weights = {
            'qkv_w_flat': self._enc_qkv_flat.data_ptr(),
            'o_w_flat': self._enc_o_flat.data_ptr(),
            'gate_w_flat': self._enc_gu_flat.data_ptr(),
            'alpha_host': self._enc_alpha_host,
            'down_w_flat': self._enc_d_flat.data_ptr(),
            'rope_base': self._full_rope.data_ptr(),
            'Kc': self._Kc.reshape(-1).data_ptr(),
            'Vc': self._Vc.reshape(-1).data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'act_scales': self._enc_calib_scales.data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        # SM100-only gate+up optimization: cutlass_fp8_wide is 1.41x faster at
        # M=1 but doesn't exist on SM120. Omitting gate_w_list routes the
        # decode_step_pi0fast_bf16 Gate+Up GEMM through fp8_gemm_descale_fp16
        # (cuBLASLt) which is the only option on Blackwell consumer silicon.
        if self._has_sm100:
            decode_weights['gate_w_list'] = [w.data_ptr() for w in self._enc_gu_w]
        decode_dims = {
            'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'prefill_len': Se,
            'total_keys_max': self.max_total_keys,
        }

        def _run_action_steps(s):
            for i in range(M):
                step = n_text + i
                decode_step_pi0fast_bf16(self._ctx, fvk, decode_bufs,
                                          decode_weights, decode_dims,
                                          step=step, stream=s)
                self._last_hidden_fp16.copy_(self._dec_xn_bf16)
                torch.matmul(self._last_hidden_fp16, self._action_embed.T,
                             out=self._action_logit_buf)
                # GPU-only argmax + record + embed (no D2H, graph-safe)
                # Use index_select instead of fancy indexing (fancy indexing
                # triggers cudaErrorStreamCaptureUnsupported during capture).
                local_id = torch.argmax(self._action_logit_buf, dim=1)  # [1]
                self._decode_tok_buf[i:i+1] = local_id + self._action_range_start
                token_embed = torch.index_select(
                    self._action_embed, 0, local_id) * sqrt_D  # [1, D]
                self._dec_x_bf16.copy_(token_embed)

        for _ in range(3):
            _run_action_steps(0)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            self._decode_graph.capture_begin()
            _run_action_steps(stream.cuda_stream)
            self._decode_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("Decode action graph captured (%d steps, Se=%d)", M, Se)

    # -------------------------------------------------------------------
    # Unified calibration API
    # -------------------------------------------------------------------

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration API (Pi0-FAST Thor).

        N=1: legacy path — one ``infer(obs)`` call triggers the lazy
        recalibrate that runs ``_calibrate(force=True)`` (internally
        does 16-sample synthetic-perturbation max reduction on the
        real SigLIP prefix) + recaptures prefill + decode graphs.

        N>=2: replaces the synthetic-perturbation reduction with a
        real-dataset percentile reduction. For each observation, run
        SigLIP to populate ``_enc_x[:Se]``, then run
        ``prefill_calibrate_pi0fast`` and collect the per-layer scale
        vector. Reduce across N samples via ``np.percentile(axis=0)``.
        Pi0-FAST has no decoder graph calibration — only encoder
        scales are reduced. After the reduction, alpha_host is
        recomputed and prefill + decode graphs are recaptured once.
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
            from flash_rt.core.calibration_api import implicit_calibrate
            implicit_calibrate(
                self, obs_list,
                percentile=percentile, max_samples=None, verbose=verbose,
            )
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        """Pi0-FAST real-dataset multi-sample calibration.

        Replaces the synthetic 16-sample perturbation reduction with N
        real observations. Each sample runs SigLIP + one
        ``prefill_calibrate_pi0fast`` pass; per-layer amax vectors are
        percentile-reduced, uploaded back, alpha_host is recomputed,
        and prefill + decode graphs are recaptured once.
        """
        from flash_rt.core.calibration import (
            accumulate_amax,
            check_scale_ceiling,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Pi0-FAST Thor: calibrating FP8 across %d real samples "
            "(percentile=%.2f)...", n, percentile)

        Se = self.Se; Le = self.Le; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        total_keys_max = self.max_total_keys
        nv = self.num_views

        _norm_scratch = torch.empty(Se * De, dtype=torch.bfloat16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=torch.bfloat16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch = torch.zeros(
            Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=torch.bfloat16, device='cuda')

        enc_weights = {
            'qkv_w': [w.data_ptr() for w in self._enc_qkv_w],
            'o_w': [w.data_ptr() for w in self._enc_o_w],
            'gate_w': [w.data_ptr() for w in self._enc_gu_w],
            'down_w': [w.data_ptr() for w in self._enc_d_w],
            'qkv_w_fp16': [w.data_ptr() for w in self._enc_qkv_fp16],
            'o_w_fp16': [w.data_ptr() for w in self._enc_o_fp16],
            'gate_w_fp16': [w.data_ptr() for w in self._enc_gu_fp16],
            'down_w_fp16': [w.data_ptr() for w in self._enc_d_fp16],
            'qkv_w_flat':  self._enc_qkv_flat.data_ptr(),
            'o_w_flat':    self._enc_o_flat.data_ptr(),
            'gate_w_flat': self._enc_gu_flat.data_ptr(),
            'down_w_flat': self._enc_d_flat.data_ptr(),
            'rope': self._enc_rope.data_ptr(),
            'Kc': self._Kc.reshape(-1).data_ptr(),
            'Vc': self._Vc.reshape(-1).data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys_max': total_keys_max,
        }

        per_sample: list[np.ndarray] = []

        for i, obs in enumerate(obs_list):
            # Upload images + run SigLIP so _enc_x[:Se] reflects this sample.
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image'], obs.get('wrist_image', obs['image'])]
                if nv >= 3 and 'wrist_image_right' in obs:
                    img_list.append(obs['wrist_image_right'])

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

            # Run prefill calibrate on the real SigLIP output.
            enc_bufs = {
                'x': self._enc_x.data_ptr(),
                'x_fp8': self._enc_x_fp8.data_ptr(),
                'qkv': self._enc_qkv_buf.data_ptr(),
                'logits': self._enc_logits.data_ptr(),
                'attn_out': self._enc_attn.data_ptr(),
                'o_fp8': self._enc_o_fp8.data_ptr(),
                'gate': self._enc_gate.data_ptr(),
                'hidden': self._enc_hidden.data_ptr(),
                'hid_fp8': self._enc_hid_fp8.data_ptr(),
                'fg': self._enc_fg.data_ptr(),
                'fg_f32': self._enc_fg_f32.data_ptr(),
                'ctx': self._ctx,
                'norm_scratch': _norm_scratch.data_ptr(),
                'x_scratch': _x_scratch.data_ptr(),
                'calib_buf': _calib_buf.data_ptr(),
                'd_scale': _d_scale.data_ptr(),
                'fp8_scratch': _fp8_scratch.data_ptr(),
                'ones': _ones.data_ptr(),
            }
            sample_scales = torch.zeros(
                Le * 4, dtype=torch.float32, device='cuda')
            self._Kc.zero_(); self._Vc.zero_()
            prefill_calibrate_pi0fast(
                self._gemm, fvk, enc_bufs, enc_weights, enc_dims,
                sample_scales.data_ptr(), stream=0)
            torch.cuda.synchronize()
            s = sample_scales.cpu().numpy().copy()
            if not (np.isinf(s).any() or np.isnan(s).any()):
                per_sample.append(s)
            elif verbose:
                logger.warning(
                    "  sample %d/%d produced inf/nan scales — skipped", i + 1, n)

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  calibration sample %d/%d", i + 1, n)

        if len(per_sample) == 0:
            raise RuntimeError(
                "all Pi0-FAST multi-frame calibration samples produced "
                "inf/nan scales — falling back to single-frame recommended")

        final_enc = accumulate_amax(per_sample, percentile=percentile)
        if verbose:
            logger.info("encoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample, final_enc)))

        self._enc_calib_scales = torch.from_numpy(
            final_enc.astype(np.float32)).cuda()
        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(final_enc[i]) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]
        check_scale_ceiling(final_enc, label=f"pi0fast_thor_enc_N{n}")

        # Recapture graphs so new scales are baked in.
        self._capture_prefill_graph()
        if getattr(self, "decode_cuda_graph", False):
            self._capture_decode_action_graph()

        # Save to cache (no decoder scales in pi0fast).
        try:
            save_calibration(
                checkpoint_path=self._checkpoint_path,
                Se=Se,
                enc_scales=self._enc_calib_scales.cpu().tolist(),
                enc_alpha=self._enc_alpha_host,
                ae_scales=[],
                enc_w_scales=enc_ws,
            )
        except Exception as e:
            logger.warning("Failed to save calibration cache: %s", e)

        self._real_data_calibrated = True
        logger.info(
            "Pi0-FAST Thor multi-frame calibration complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """Pi0-FAST does not yet surface a structured PrecisionSpec."""
        return None

    # -------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------

    def infer(self, observation, max_steps=None, temperature=0.0):
        t0 = time.perf_counter()
        nv = self.num_views
        max_steps = max_steps or self.max_decode_steps

        # Upload images
        if 'images' in observation:
            img_list = observation['images']
        else:
            img_list = [observation['image'], observation['wrist_image']]
            if nv >= 3 and 'wrist_image_right' in observation:
                img_list.append(observation['wrist_image_right'])

        def _to_np16(im):
            if isinstance(im, torch.Tensor):
                return im.to(dtype=torch.float16).cpu().numpy()
            if im.dtype == np.float16:
                return im
            return (im.astype(np.float32) / 127.5 - 1.0).astype(np.float16)

        images_np = np.stack([_to_np16(im) for im in img_list[:nv]])
        self._img_buf.upload(images_np)

        # SigLIP graph replay
        self._siglip_graph.replay()
        torch.cuda.synchronize()

        # Lazy real-data recalibration on first infer.
        # Calibration during set_prompt uses synthetic perturbations.
        # Real SigLIP output may have different activation distribution.
        # Recalibrate with actual image data, then recapture prefill graph
        # (alpha_host values are baked into the captured graph).
        if not self._real_data_calibrated:
            self._calibrate(self.Se, force=True)  # force: use real image data, not cache
            self._capture_prefill_graph()
            if self.decode_cuda_graph:
                self._capture_decode_action_graph()
            self._real_data_calibrated = True
            logger.info("Recalibrated with real image data, recaptured graphs")

        Se = self.Se; De = self.De; He = self.He
        Le = self.Le; NHe = self.NHe; HDe = self.HDe

        # Prefill graph replay: KV cache zero + 18-layer forward
        self._prefill_graph.replay()
        torch.cuda.synchronize()

        t_prefill = time.perf_counter()

        # Autoregressive decode loop — uses BF16 residual chain (decode_step_pi0fast_bf16)
        output_tokens = []
        decode_bufs = {
            'x': self._dec_x_bf16.data_ptr(),
            'x_fp8': self._dec_x_fp8.data_ptr(),
            'qkv': self._dec_qkv.data_ptr(),
            'logits': self._dec_logits.data_ptr(),
            'attn_out': self._dec_attn.data_ptr(),
            'o_fp8': self._dec_o_fp8.data_ptr(),
            'gate': self._dec_gate.data_ptr(),
            'hid_fp8': self._dec_hid_fp8.data_ptr(),
            'fg': self._dec_fg_bf16.data_ptr(),
            'xn': self._dec_xn_bf16.data_ptr(),
            'fg_scratch': self._dec_fg_scratch.data_ptr(),
        }
        decode_weights = {
            'qkv_w_flat': self._enc_qkv_flat.data_ptr(),
            'o_w_flat': self._enc_o_flat.data_ptr(),
            'gate_w_flat': self._enc_gu_flat.data_ptr(),
            'alpha_host': self._enc_alpha_host,
            'down_w_flat': self._enc_d_flat.data_ptr(),
            'rope_base': self._full_rope.data_ptr(),
            'Kc': self._Kc.reshape(-1).data_ptr(),
            'Vc': self._Vc.reshape(-1).data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'act_scales': self._enc_calib_scales.data_ptr(),
            'w_scales': self._enc_w_dev.data_ptr(),
        }
        # Only enable the SM100 cutlass_fp8_wide fast-path if we're on Thor;
        # SM120 must route through fp8_gemm_descale_fp16 (cuBLASLt).
        if self._has_sm100:
            decode_weights['gate_w_list'] = [w.data_ptr() for w in self._enc_gu_w]
        decode_dims = {
            'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'prefill_len': Se,
            'total_keys_max': self.max_total_keys,
        }

        # ── T1: Text phase uses hardcoded tokens, skip full-vocab logit ──
        # Pi0-FAST always generates "Action: " (3 tokens) before action tokens.
        # Skipping the [1,2048]×[2048,257152] logit matmul saves ~5.5ms per step.
        sqrt_D = float(De ** 0.5)
        n_text = len(_TEXT_PHASE_TOKENS)
        for i, tok_id in enumerate(_TEXT_PHASE_TOKENS):
            output_tokens.append(tok_id)
            token_embed = self.embedding_weight[tok_id] * sqrt_D
            self._dec_x_bf16.copy_(token_embed.unsqueeze(0))
            decode_step_pi0fast_bf16(self._ctx, fvk, decode_bufs, decode_weights,
                                      decode_dims, step=i, stream=0)

        # Transition: action-vocab logit for first action token (~0.03ms vs 5.5ms)
        self._last_hidden_fp16.copy_(self._dec_xn_bf16)
        torch.matmul(self._last_hidden_fp16, self._action_embed.T,
                     out=self._action_logit_buf)

        if self._decode_graph is not None:
            # ── Graph-accelerated decode: action phase (graph) ──
            if temperature > 0:
                probs = torch.softmax(self._action_logit_buf[0].float() / temperature, dim=0)
                local_id = torch.multinomial(probs, 1).item()
            else:
                local_id = self._action_logit_buf[0].argmax().item()
            token_id = self._action_range_start + local_id
            output_tokens.append(token_id)
            token_embed = self.embedding_weight[token_id] * sqrt_D
            self._dec_x_bf16.copy_(token_embed.unsqueeze(0))

            # Action phase: single graph replay (all GPU, no D2H per step)
            self._decode_graph.replay()
            torch.cuda.synchronize()

            # Post-process: read tokens from GPU, find EOS/pipe, truncate
            graph_toks = self._decode_tok_buf[:self.decode_graph_steps].cpu().tolist()
            for i, tid in enumerate(graph_toks):
                if tid == PALIGEMMA_EOS_TOKEN or tid == PALIGEMMA_PIPE_TOKEN:
                    graph_toks = graph_toks[:i + 1]
                    break
            output_tokens.extend(graph_toks)
        else:
            # ── Standard decode: action phase with per-step pruned vocab ──
            for step in range(n_text, max_steps):
                if temperature > 0:
                    probs = torch.softmax(self._action_logit_buf[0].float() / temperature, dim=0)
                    local_id = torch.multinomial(probs, 1).item()
                else:
                    local_id = self._action_logit_buf[0].argmax().item()
                token_id = self._action_range_start + local_id

                if token_id == PALIGEMMA_EOS_TOKEN or token_id == PALIGEMMA_PIPE_TOKEN:
                    output_tokens.append(token_id)
                    break
                output_tokens.append(token_id)

                token_embed = self.embedding_weight[token_id] * sqrt_D
                self._dec_x_bf16.copy_(token_embed.unsqueeze(0))
                decode_step_pi0fast_bf16(self._ctx, fvk, decode_bufs, decode_weights,
                                          decode_dims, step=step, stream=0)
                self._last_hidden_fp16.copy_(self._dec_xn_bf16)
                torch.matmul(self._last_hidden_fp16, self._action_embed.T,
                             out=self._action_logit_buf)

        torch.cuda.synchronize()
        t_decode = time.perf_counter()
        latency_ms = (t_decode - t0) * 1000
        prefill_ms = (t_prefill - t0) * 1000
        decode_ms = (t_decode - t_prefill) * 1000
        self.latency_records.append(latency_ms)

        n_tokens = len(output_tokens)
        per_token = decode_ms / max(n_tokens, 1)
        logger.info("Infer: %d tokens, prefill=%.1fms, decode=%.1fms (%.1fms/tok), total=%.1fms",
                     n_tokens, prefill_ms, decode_ms, per_token, latency_ms)

        # Decode tokens to actions
        actions = self._decode_actions(output_tokens)
        return {"actions": actions, "tokens": output_tokens,
                "n_tokens": n_tokens, "latency_ms": latency_ms,
                "prefill_ms": prefill_ms, "decode_ms": decode_ms,
                "per_token_ms": per_token}

    def _decode_actions(self, token_ids, action_horizon=10, action_dim=7):
        if not token_ids:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        if self._fast_tokenizer is not None and self._sp_tokenizer is not None:
            import io, contextlib
            try:
                decoded_text = self._sp_tokenizer.Decode(token_ids)
                if "Action: " not in decoded_text:
                    return np.zeros((action_horizon, action_dim), dtype=np.float32)
                raw_action_str = decoded_text.split("Action: ")[1].split("|")[0].strip()
                raw_tokens = np.array(self._sp_tokenizer.Encode(raw_action_str))
                action_tokens = self._vocab_size - 1 - FAST_SKIP_TOKENS - raw_tokens
                # FAST tokenizer prints to stdout on decode error (out-of-distribution
                # token sequences). Capture and discard.
                with contextlib.redirect_stdout(io.StringIO()):
                    actions = self._fast_tokenizer.decode(
                        [action_tokens.tolist()],
                        time_horizon=action_horizon,
                        action_dim=action_dim)[0]
                return np.array(actions, dtype=np.float32)
            except Exception as e:
                logger.warning("FAST decode failed: %s", e)

        return np.zeros((action_horizon, action_dim), dtype=np.float32)

    # -------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------

    def get_latency_stats(self):
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat), "mean_ms": float(np.mean(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "min_ms": float(np.min(lat)), "max_ms": float(np.max(lat)),
        }
