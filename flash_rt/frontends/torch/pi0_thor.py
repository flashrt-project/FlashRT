"""FlashRT -- Pi0TorchFrontendThor: Pi0 inference using ONLY flash_rt_kernels.so.

Adapted from ThorPipelineTorch (Pi0.5). Key differences:
  - Standard RMSNorm (not AdaRMSNorm) → fuse norm weight into QKV/GateUp
  - action_time_mlp: concat(action, time) → MLP (replaces time_mlp → AdaRMS)
  - state_proj: continuous state → single suffix token
  - S_dec = Sa + 1 (1 state + Sa actions, no padding)

Usage:
    pipe = Pi0TorchFrontendThor("/path/to/checkpoint", num_views=2)
    pipe.set_prompt("pick up the red cup")
    result = pipe.infer({"image": img1, "wrist_image": img2, "state": state_vec})
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
from flash_rt.hardware.thor.attn_backend import (
    ThorFlashAttnBackend,
    make_pi0_attention_spec,
)
from flash_rt.models.pi0.pipeline_thor import (
    decoder_forward_pi0,
    decoder_forward_calibrate_pi0,
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
# Helpers (shared with Pi0.5)
# ---------------------------------------------------------------------------

from flash_rt.core.thor_frontend_utils import embed_prompt_torch as embed_prompt  # noqa: E402


# ===========================================================================
# Pi0TorchFrontendThor
# ===========================================================================

class Pi0TorchFrontendThor:
    """Complete Pi0 inference pipeline using only flash_rt_kernels.so.

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

        # ---- FvkContext + GemmRunner ----
        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()

        # ---- Load CUTLASS FMHA ----
        # Search order: ckpt-adjacent, ``flash_rt/`` package dir (pip
        # install lands here), fresh cmake ``build/`` output, docker
        # ``/workspace/`` convention.
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

        # ---- Weights ----
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path}. "
                "For Orbax/JAX checkpoints, use ThorPipelineJax.")
        self._checkpoint_path = str(safetensors_path)
        self._load_weights(safetensors_path)
        logger.info("Pi0TorchFrontendThor initialised (num_views=%d)", num_views)

    # -----------------------------------------------------------------------
    # norm_stats
    # -----------------------------------------------------------------------

    def _load_norm_stats(self, checkpoint_dir):
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir.parent / "pi0_base" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            pathlib.Path("/root/.cache/openpi/openpi-assets/checkpoints/pi0_base/"
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
            SafetensorsSource, WeightLoader, _autodetect_strip_prefix,
        )
        from flash_rt.frontends.torch._pi0_thor_spec import build_spec

        sf = safe_open(str(safetensors_path), framework='pt', device='cuda')
        # Auto-strip the lerobot HF policy ``model.`` namespace wrap so
        # this loader can stay written against openpi-converted bare
        # keys. Returns ``""`` (no-op) for already-openpi checkpoints.
        _strip = _autodetect_strip_prefix(set(sf.keys()))
        def g_raw(k): return sf.get_tensor((_strip + k) if _strip else k)
        def g(k): return g_raw(k).to(fp16)

        # Declarative weight-loader pass (stage 7.4). Populates:
        #   self._sig_{ln_attn,ln_ffn,qkv,o,up,down}_{w,b}  (27-layer lists)
        #   self._sig_alpha                                  (108 fp32 scales)
        #   self._enc_{qkv,o,gu,d}_w                          (18-layer lists)
        #   self._enc_w_scales                                (72 fp32 scales)
        #   self._dec_{qkv,o,gu,d}_flat                       (flat cat tensors)
        #   self._ae_w_scales                                 (72 fp32 scales)
        _src = SafetensorsSource(str(safetensors_path), device='cuda')
        WeightLoader(source=_src, target=self, spec=build_spec()).run()

        self.embedding_weight = g('paligemma_with_expert.paligemma.lm_head.weight')

        # ===============================================================
        # SigLIP  (27 layers) — identical to Pi0.5
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

        # Per-layer weights loaded declaratively above. Compose the pointer
        # dict consumed by shared_primitives.siglip_forward.
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
        self._unit_scale = torch.ones(1, dtype=torch.float32, device='cuda')
        self._sig_weights['unit_scale'] = self._unit_scale.data_ptr()
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

        # Patch embedding weights
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
        self._proj_w = g(f'{mp}.weight').T.contiguous()
        self._proj_b = g(f'{mp}.bias')
        self._postln_scratch = torch.empty(S_sig, max(D_sig, H_sig), dtype=fp16, device='cuda')

        # ===============================================================
        # Encoder  (18 layers, GQA) — identical to Pi0.5
        # ===============================================================
        De, He, Le = 2048, 16384, 18
        NHe, HDe = 8, 256
        Se_max = nv * 256 + 256
        self.De = De; self.He = He; self.Le = Le
        self.NHe = NHe; self.HDe = HDe; self.Se_max = Se_max

        # Encoder per-layer weights loaded declaratively above:
        #   self._enc_{qkv,o,gu,d}_w lists (fused-norm, GQA interleave)
        #   self._enc_w_scales  host list of Le*4 floats (order: q, o, gu, d per layer)
        self._enc_w_dev = torch.tensor(self._enc_w_scales, dtype=torch.float32, device='cuda')

        # RoPE table
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256))
        kp = inv_freq[None, :] * torch.arange(1200, device='cuda')[:, None].float()
        self._kc_t = torch.cos(kp).to(fp16)
        self._ks_t = torch.sin(kp).to(fp16)
        self._enc_rope = torch.empty(Se_max, 256, dtype=fp16, device='cuda')

        # ===============================================================
        # Decoder / AE  (18 layers, 10 steps)
        # Pi0: fuse RMSNorm weight into QKV/GateUp (like encoder)
        # ===============================================================
        dp = 'paligemma_with_expert.gemma_expert'
        Sa, Da, Ha, La = 10, 1024, 4096, 18
        steps = 10
        S_dec = Sa + 1  # 1 state + Sa actions (no padding)
        self.Sa = Sa; self.Da = Da; self.Ha = Ha; self.La = La
        self.S_dec = S_dec

        # KV cache: total_keys_max = Se_max + S_dec
        total_keys_max = Se_max + S_dec
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

        # Decoder per-layer weights loaded declaratively above as flat cats:
        #   self._dec_{qkv,o,gu,d}_flat  (QKV + GateUp fuse norm_fuse weight)
        #   self._ae_w_scales            host list of La*4 floats (q, o, gu, d)

        # Action in/out projections (singletons)
        self._ain_w = g('action_in_proj.weight').t().contiguous()  # [32, Da]
        self._ain_b = g('action_in_proj.bias')
        self._aow = g('action_out_proj.weight').t().contiguous() * (-1.0 / steps)
        self._aob = g('action_out_proj.bias') * (-1.0 / steps)

        self._ae_w_dev = torch.tensor(self._ae_w_scales, dtype=torch.float32, device='cuda')

        # ---- Pi0-specific: state_proj + action_time_mlp ----
        self._state_proj_w = g('state_proj.weight').t().contiguous()  # [32, Da]
        self._state_proj_b = g('state_proj.bias')

        # action_time_mlp_in: [Da, 2*Da] → split into W_action [Da, Da] and W_time [Da, Da]
        atmlp_in_w_full = g('action_time_mlp_in.weight')  # [Da, 2*Da] in PyTorch
        self._atmlp_in_wt = atmlp_in_w_full[:, Da:].contiguous()      # time part [Da, Da] for pre-compute (FP16)
        self._atmlp_in_b  = g('action_time_mlp_in.bias')              # [Da]

        # W_a and mlp_out stay FP16 (too small for FP8 benefit at [10,1024]×[1024,1024])
        self._atmlp_in_wa = atmlp_in_w_full[:, :Da].t().contiguous()  # [Da, Da]
        self._atmlp_out_w = g('action_time_mlp_out.weight').t().contiguous()  # [Da, Da]
        self._atmlp_out_b = g('action_time_mlp_out.bias')

        # Final norm: standard RMSNorm with weight (Pi0, not AdaRMS)
        final_norm_raw = g_raw(f'{dp}.model.norm.weight').float()
        self._final_norm_w = (1.0 + final_norm_raw).to(fp16)

        # Decoder buffers (S_dec = Sa + 2)
        self._dec_rope = torch.empty(S_dec, 256, dtype=fp16, device='cuda')
        self._ae_x   = torch.empty(S_dec, Da, dtype=fp16, device='cuda')
        self._ae_xn  = torch.empty(S_dec, Da, dtype=fp16, device='cuda')
        self._ae_temp = torch.empty(Sa, Da, dtype=fp16, device='cuda')        # action_time_mlp temp
        self._ae_action_buf = torch.empty(Sa, Da, dtype=fp16, device='cuda')  # action tokens
        self._ae_gate = torch.empty(S_dec, Da, dtype=fp16, device='cuda')
        self._ae_qkv = torch.empty(S_dec, 2560, dtype=fp16, device='cuda')
        self._ae_logits = torch.empty(S_dec * 8, total_keys_max, dtype=fp16, device='cuda')
        self._ae_attn = torch.empty(S_dec * 8, 256, dtype=fp16, device='cuda')
        self._ae_hid  = torch.empty(S_dec, 2 * Ha, dtype=fp16, device='cuda')
        self._ae_fg   = torch.empty(S_dec, 2 * Ha, dtype=fp16, device='cuda')
        self._ae_xn_fp8  = torch.zeros(S_dec * Da, dtype=torch.uint8, device='cuda')
        self._ae_hid_fp8 = torch.zeros(S_dec * Ha, dtype=torch.uint8, device='cuda')
        self._ae_ctx_fp8 = torch.zeros(S_dec * 8 * 256, dtype=torch.uint8, device='cuda')
        self._g_noise = torch.zeros(Sa, 32, dtype=fp16, device='cuda')

        # State token buffer (computed in graph from state input)
        self._state_token = torch.zeros(1, Da, dtype=fp16, device='cuda')
        self._state_buf   = torch.zeros(1, 32, dtype=fp16, device='cuda')  # observation state input

        # Calibration scale buffers
        self._enc_calib_scales = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        self._ae_calib_scales  = torch.zeros(La * 4, dtype=torch.float32, device='cuda')

        logger.info("Weights loaded for Pi0TorchFrontendThor (S_dec=%d)", S_dec)

    # -----------------------------------------------------------------------
    # set_prompt
    # -----------------------------------------------------------------------

    def set_prompt(self, prompt_text):
        """Tokenize prompt, compute time conditioning, calibrate scales, capture graphs."""
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
        self.total_keys = Se + self.S_dec

        # Stage 3.3 — build AttentionBackend. Pi0 decoder uses state-masked
        # kernel (state token row 0 can only attend to enc_seq+1 keys).
        # total_keys = Se + S_dec (Pi0 differs from Pi0.5 which uses Se + Sa;
        # Pi0 has an extra row for the state token, so the KV cache reserves
        # one more slot per step).
        attn_scale = 1.0 / math.sqrt(float(self.HDe))
        layer_stride = int(self.total_keys) * int(self.HDe) * 2  # fp16 bytes
        kc_ptr = self._Kc.reshape(-1).data_ptr()
        vc_ptr = self._Vc.reshape(-1).data_ptr()
        self._attn = ThorFlashAttnBackend(
            make_pi0_attention_spec(
                num_views=self.num_views,
                enc_seq_max=self.Se,
                S_dec=self.S_dec,
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
            torch.cat([self._kc_t[dec_start:dec_start + self.S_dec, :, None],
                       self._ks_t[dec_start:dec_start + self.S_dec, :, None]], dim=2)
            .reshape(self.S_dec, 256))

        # ---- Pi0 time conditioning: pre-compute time projections ----
        Sa, Da, La = self.Sa, self.Da, self.La
        steps = 10

        # Pre-compute time_proj_expanded for all steps: [steps * Sa, Da]
        time_proj_list = []
        for step in range(steps):
            t_val = 1.0 - step / steps
            t_tensor = torch.tensor([t_val], device='cuda')
            fraction = torch.linspace(0, 1, Da // 2, device='cuda', dtype=torch.float64)
            period = 4e-3 * (4.0 / 4e-3) ** fraction
            scaling = 1.0 / period * 2 * math.pi
            sin_input = scaling * t_tensor.double()
            time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1).to(fp16)

            # time_proj[s] = time_emb @ W_time.T + bias → [1, Da]
            tp = (time_emb @ self._atmlp_in_wt.t() + self._atmlp_in_b.unsqueeze(0))
            # Expand to [Sa, Da]
            tp_expanded = tp.expand(Sa, -1).contiguous()
            time_proj_list.append(tp_expanded)

        self._time_proj_all = torch.cat(time_proj_list, dim=0)  # [steps*Sa, Da]

        # ---- Capture SigLIP graph ----
        self._capture_siglip_graph()

        # ---- Calibrate FP8 scales ----
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
            self._enc_alpha_host = [
                float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
                for i in range(Le * 4)]
            self._ae_calib_scales = torch.tensor(
                cached["ae_scales"], dtype=torch.float32, device='cuda')
            logger.info("Calibration loaded from cache (enc=%d, ae=%d scales)",
                        Le * 4, La * 4)
            return

        HDe = self.HDe; NHe = self.NHe; De = self.De; He = self.He

        # Encoder calibration (identical to Pi0.5)
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
        self._enc_alpha_host = [
            float(np.float32(self._enc_calib_scales[i].item()) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]
        logger.info("Encoder calibrated: %d scales", Le * 4)

        # Decoder calibration (Pi0 pattern: uses RMSNorm, not AdaRMS)
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        S_dec = self.S_dec
        ae_bufs = {
            'noise':       self._g_noise.data_ptr(),
            'x':           self._ae_x.data_ptr(),
            'xn':          self._ae_xn.data_ptr(),
            'temp':        self._ae_temp.data_ptr(),
            'action_buf':  self._ae_action_buf.data_ptr(),
            'gate':        self._ae_gate.data_ptr(),
            'qkv':         self._ae_qkv.data_ptr(),
            'logits':      self._ae_logits.data_ptr(),
            'attn_out':    self._ae_attn.data_ptr(),
            'hid':         self._ae_hid.data_ptr(),
            'fg':          self._ae_fg.data_ptr(),
            'xn_fp8':      self._ae_xn_fp8.data_ptr(),
            'hid_fp8':     self._ae_hid_fp8.data_ptr(),
            'ctx_fp8':     self._ae_ctx_fp8.data_ptr(),
            'state_token': self._state_token.data_ptr(),
        }
        ae_weights = {
            'ain_w':        self._ain_w.data_ptr(),
            'ain_b':        self._ain_b.data_ptr(),
            'wa_w':         self._atmlp_in_wa.data_ptr(),
            'atmlp_out_w':  self._atmlp_out_w.data_ptr(),
            'atmlp_out_b':  self._atmlp_out_b.data_ptr(),
            'time_proj_all': self._time_proj_all.data_ptr(),
            'qw':           self._dec_qkv_flat.data_ptr(),
            'Kc':           self._Kc.reshape(-1).data_ptr(),
            'Vc':           self._Vc.reshape(-1).data_ptr(),
            'ow':           self._dec_o_flat.data_ptr(),
            'gw':           self._dec_gu_flat.data_ptr(),
            'dw':           self._dec_d_flat.data_ptr(),
            'aow':          self._aow.data_ptr(),
            'aob':          self._aob.data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'rope':         self._dec_rope.data_ptr(),
            'w_scales':     self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'Sa': Sa, 'S_dec': S_dec, 'D': Da, 'H': Ha,
            'NH': 8, 'HD': 256, 'steps': 10, 'layers': La,
            'enc_seq': Se, 'total_keys': total_keys,
        }

        # Decoder scratch buffers
        _ae_calib_buf = torch.zeros(La * 4, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(S_dec * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(S_dec * max(Da, Ha), dtype=torch.uint8, device='cuda')
        _ae_norm_scratch = torch.empty(S_dec * Da, dtype=fp16, device='cuda')
        _ae_x_scratch = torch.empty(S_dec * Da, dtype=fp16, device='cuda')
        _ae_ones = torch.ones(Da, dtype=fp16, device='cuda')
        ae_bufs['calib_buf'] = _ae_calib_buf.data_ptr()
        ae_bufs['d_scale'] = _ae_d_scale.data_ptr()
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.data_ptr()
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.data_ptr()
        ae_bufs['norm_scratch'] = _ae_norm_scratch.data_ptr()
        ae_bufs['x_scratch'] = _ae_x_scratch.data_ptr()
        ae_bufs['ones'] = _ae_ones.data_ptr()

        # Use random noise for calibration
        self._state_token.zero_()  # zero state for calibration
        self._g_noise.normal_()
        ae_max = torch.zeros(La * 4, dtype=torch.float32, device='cuda')
        decoder_forward_calibrate_pi0(
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
    # Patch embedding
    # -----------------------------------------------------------------------

    def _patch_embed_ops(self, stream_int):
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
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)
        for _ in range(3):
            self._patch_embed_ops(0)
            self._sig_x.zero_()
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights,
                           self._sig_dims, stream=0, attn=self._attn)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

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
        Se = self.Se
        total_keys = self.total_keys
        Le = self.Le; La = self.La; De = self.De; He = self.He
        NHe = self.NHe; HDe = self.HDe
        Sa = self.Sa; Da = self.Da; Ha = self.Ha; S_dec = self.S_dec

        # Encoder dicts
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
            'act_scales':  self._enc_calib_scales.data_ptr(),
            'alpha_host':  self._enc_alpha_host,
        }
        enc_dims = {
            'Se': Se, 'D': De, 'H': He, 'NH': NHe, 'HD': HDe,
            'L': Le, 'total_keys': total_keys,
        }

        # Decoder dicts (Pi0 pattern)
        ae_bufs = {
            'noise':       self._g_noise.data_ptr(),
            'x':           self._ae_x.data_ptr(),
            'xn':          self._ae_xn.data_ptr(),
            'temp':        self._ae_temp.data_ptr(),
            'action_buf':  self._ae_action_buf.data_ptr(),
            'gate':        self._ae_gate.data_ptr(),
            'qkv':         self._ae_qkv.data_ptr(),
            'logits':      self._ae_logits.data_ptr(),
            'attn_out':    self._ae_attn.data_ptr(),
            'hid':         self._ae_hid.data_ptr(),
            'fg':          self._ae_fg.data_ptr(),
            'xn_fp8':      self._ae_xn_fp8.data_ptr(),
            'hid_fp8':     self._ae_hid_fp8.data_ptr(),
            'ctx_fp8':     self._ae_ctx_fp8.data_ptr(),
            'state_token': self._state_token.data_ptr(),
        }
        ae_weights = {
            'ain_w':        self._ain_w.data_ptr(),
            'ain_b':        self._ain_b.data_ptr(),
            'wa_w':         self._atmlp_in_wa.data_ptr(),
            'atmlp_out_w':  self._atmlp_out_w.data_ptr(),
            'atmlp_out_b':  self._atmlp_out_b.data_ptr(),
            'time_proj_all': self._time_proj_all.data_ptr(),
            'qw':           self._dec_qkv_flat.data_ptr(),
            'Kc':           self._Kc.reshape(-1).data_ptr(),
            'Vc':           self._Vc.reshape(-1).data_ptr(),
            'ow':           self._dec_o_flat.data_ptr(),
            'gw':           self._dec_gu_flat.data_ptr(),
            'dw':           self._dec_d_flat.data_ptr(),
            'aow':          self._aow.data_ptr(),
            'aob':          self._aob.data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'rope':         self._dec_rope.data_ptr(),
            'w_scales':     self._ae_w_dev.data_ptr(),
            'act_scales':   self._ae_calib_scales.data_ptr(),
        }
        ae_dims = {
            'Sa': Sa, 'S_dec': S_dec, 'D': Da, 'H': Ha,
            'NH': 8, 'HD': 256, 'steps': 10, 'layers': La,
            'enc_seq': Se, 'total_keys': total_keys,
        }

        # Warmup
        for _ in range(3):
            self._Kc.zero_(); self._Vc.zero_()
            # State proj (warmup with zeros)
            fvk.gmm_fp16(self._ctx, self._state_buf.data_ptr(),
                         self._state_proj_w.data_ptr(),
                         self._state_token.data_ptr(),
                         1, self.Da, 32, 0.0, 0)
            fvk.add_bias_fp16(self._state_token.data_ptr(),
                              self._state_proj_b.data_ptr(), 1, self.Da, 0)
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=0, attn=self._attn)
            decoder_forward_pi0(self._ctx, fvk, ae_bufs, ae_weights,
                                ae_dims, stream=0, attn=self._attn)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        self._enc_ae_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._enc_ae_graph.capture_begin()
            self._Kc.zero_(); self._Vc.zero_()
            # State proj
            fvk.gmm_fp16(self._ctx, self._state_buf.data_ptr(),
                         self._state_proj_w.data_ptr(),
                         self._state_token.data_ptr(),
                         1, self.Da, 32, 0.0, s_int)
            fvk.add_bias_fp16(self._state_token.data_ptr(),
                              self._state_proj_b.data_ptr(), 1, self.Da, s_int)
            encoder_forward(self._gemm, fvk, enc_bufs, enc_weights,
                            enc_dims, stream=s_int, attn=self._attn)
            decoder_forward_pi0(self._ctx, fvk, ae_bufs, ae_weights,
                                ae_dims, stream=s_int, attn=self._attn)
            self._enc_ae_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("Enc+AE CUDA graph captured (Se=%d, S_dec=%d)", Se, S_dec)

    # -----------------------------------------------------------------------
    # Autotune
    # -----------------------------------------------------------------------

    def _autotune_enc_ae(self, n_trials=5, n_bench=10):
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

        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)

        logger.info("Autotune: up to %d trials for best Enc+AE graph...", n_trials)

        for trial in range(n_trials):
            self._capture_enc_ae_graph()

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

            if p50 < 38.0:  # Pi0 fast regime (no AdaRMS, slightly faster than Pi0.5's 38.5ms)
                logger.info("Autotune done: Enc+AE = %.2f ms (trial %d)", p50, trial)
                return

        logger.info("Autotune done: Enc+AE = %.2f ms (best of %d)", p50, n_trials)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def infer(self, observation, debug=False):
        """Run inference: images + state -> CUDA graph replay -> actions.

        Args:
            observation: dict with 'image', 'wrist_image', and 'state'.
                         Each image is (224,224,3) uint8 or float16 numpy.
                         state is (32,) or (action_dim,) float numpy.
        Returns:
            {"actions": np.ndarray}  shape (Sa, LIBERO_ACTION_DIM)
        """
        t0 = time.perf_counter()
        nv = self.num_views

        # ---- Collect and upload images ----
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

        # ---- Upload state (Pi0-specific, minimal host overhead) ----
        state = observation.get('state', None)
        if state is not None:
            if isinstance(state, torch.Tensor):
                self._state_buf.copy_(state.to(dtype=fp16, device='cuda').view(1, -1))
            else:
                # Direct numpy → pinned GPU copy via CudaBuffer-style upload
                state_fp16 = np.asarray(state, dtype=np.float16).reshape(1, -1)
                self._state_buf.copy_(torch.from_numpy(state_fp16).to(device='cuda', dtype=fp16, non_blocking=True))
        else:
            self._state_buf.zero_()

        # ---- Graph 1: SigLIP + PostLN ----
        self._siglip_graph.replay()

        # ---- Lazy real-data recalibration on first call ----
        if not self._real_data_calibrated:
            torch.cuda.synchronize()
            self._recalibrate_with_real_data()
            self._real_data_calibrated = True

        # ---- Graph 2: Encoder + Decoder ----
        self._g_noise.normal_()
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
    # Real-data recalibration
    # -----------------------------------------------------------------------

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples=None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration API (Pi0 Thor).

        N=1: legacy implicit-recalibrate path via one ``infer`` call.
        N>=2: per-sample amax across obs_list on encoder + decoder
        calibrate passes, reduced via ``np.percentile(axis=0)``; the
        enc+ae CUDA graph is recaptured once at the end. Mirrors the
        Pi0.5 Thor multi-frame implementation with Pi0's state-token
        handling (each sample's ``obs['state']`` is projected per-sample
        so the decoder calibrate pass sees realistic state embeddings).
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
        """Pi0 Thor multi-sample dataset calibration.

        Same shape as :meth:`Pi05TorchFrontendThor._calibrate_multi_frame`
        but uses :func:`decoder_forward_calibrate_pi0` (Pi0's decoder
        kernel) and re-projects ``obs['state']`` per sample before the
        decoder pass.
        """
        from flash_rt.core.calibration import (
            accumulate_amax,
            check_scale_ceiling,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Pi0 Thor: calibrating FP8 across %d real samples "
            "(percentile=%.2f)...", n, percentile)

        Se = self.Se; Le = self.Le; La = self.La
        total_keys = self.total_keys
        De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        S_dec = self.S_dec
        nv = self.num_views

        # Reusable scratch.
        _norm_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _x_scratch = torch.empty(Se * De, dtype=fp16, device='cuda')
        _calib_buf = torch.zeros(Le * 4, dtype=torch.float32, device='cuda')
        _d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _fp8_scratch_enc = torch.zeros(
            Se * max(De, He), dtype=torch.uint8, device='cuda')
        _ones = torch.ones(De, dtype=fp16, device='cuda')

        _ae_calib_buf = torch.zeros(La * 4, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(S_dec * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(
            S_dec * max(Da, Ha), dtype=torch.uint8, device='cuda')
        _ae_norm_scratch = torch.empty(S_dec * Da, dtype=fp16, device='cuda')
        _ae_x_scratch = torch.empty(S_dec * Da, dtype=fp16, device='cuda')
        _ae_ones = torch.ones(Da, dtype=fp16, device='cuda')

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
            'ain_w':        self._ain_w.data_ptr(),
            'ain_b':        self._ain_b.data_ptr(),
            'wa_w':         self._atmlp_in_wa.data_ptr(),
            'atmlp_out_w':  self._atmlp_out_w.data_ptr(),
            'atmlp_out_b':  self._atmlp_out_b.data_ptr(),
            'time_proj_all': self._time_proj_all.data_ptr(),
            'qw':           self._dec_qkv_flat.data_ptr(),
            'Kc':           self._Kc.reshape(-1).data_ptr(),
            'Vc':           self._Vc.reshape(-1).data_ptr(),
            'ow':           self._dec_o_flat.data_ptr(),
            'gw':           self._dec_gu_flat.data_ptr(),
            'dw':           self._dec_d_flat.data_ptr(),
            'aow':          self._aow.data_ptr(),
            'aob':          self._aob.data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'rope':         self._dec_rope.data_ptr(),
            'w_scales':     self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'Sa': Sa, 'S_dec': S_dec, 'D': Da, 'H': Ha,
            'NH': 8, 'HD': 256, 'steps': 10, 'layers': La,
            'enc_seq': Se, 'total_keys': total_keys,
        }

        per_sample_enc: list[np.ndarray] = []
        per_sample_ae:  list[np.ndarray] = []
        noise_gen = torch.Generator(device='cuda').manual_seed(0)

        for i, obs in enumerate(obs_list):
            # --- Prepare + run SigLIP on this sample's images ---
            if 'images' in obs:
                img_list = obs['images']
            else:
                img_list = [obs['image']]
                if nv >= 2:
                    img_list.append(obs.get(
                        'wrist_image', obs.get('image')))
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

            # --- Encoder calibrate pass ---
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

            # --- Project this sample's state into state_token ---
            state = obs.get('state')
            if state is None:
                state_np = np.zeros(32, dtype=np.float32)
            else:
                state_np = np.asarray(state, dtype=np.float32).reshape(-1)
                if state_np.size < 32:
                    state_np = np.pad(state_np, (0, 32 - state_np.size))
                state_np = state_np[:32]
            self._state_buf.copy_(
                torch.from_numpy(state_np.astype(np.float16)).cuda())
            fvk.gmm_fp16(self._ctx, self._state_buf.data_ptr(),
                         self._state_proj_w.data_ptr(),
                         self._state_token.data_ptr(),
                         1, self.Da, 32, 0.0, 0)
            fvk.add_bias_fp16(self._state_token.data_ptr(),
                              self._state_proj_b.data_ptr(), 1, self.Da, 0)

            # --- Decoder calibrate pass ---
            ae_bufs = {
                'noise':       self._g_noise.data_ptr(),
                'x':           self._ae_x.data_ptr(),
                'xn':          self._ae_xn.data_ptr(),
                'temp':        self._ae_temp.data_ptr(),
                'action_buf':  self._ae_action_buf.data_ptr(),
                'gate':        self._ae_gate.data_ptr(),
                'qkv':         self._ae_qkv.data_ptr(),
                'logits':      self._ae_logits.data_ptr(),
                'attn_out':    self._ae_attn.data_ptr(),
                'hid':         self._ae_hid.data_ptr(),
                'fg':          self._ae_fg.data_ptr(),
                'xn_fp8':      self._ae_xn_fp8.data_ptr(),
                'hid_fp8':     self._ae_hid_fp8.data_ptr(),
                'ctx_fp8':     self._ae_ctx_fp8.data_ptr(),
                'state_token': self._state_token.data_ptr(),
                'calib_buf':      _ae_calib_buf.data_ptr(),
                'd_scale':        _ae_d_scale.data_ptr(),
                'hidden_scratch': _ae_hidden_scratch.data_ptr(),
                'fp8_scratch':    _ae_fp8_scratch.data_ptr(),
                'norm_scratch':   _ae_norm_scratch.data_ptr(),
                'x_scratch':      _ae_x_scratch.data_ptr(),
                'ones':           _ae_ones.data_ptr(),
            }
            self._ae_calib_scales.zero_()
            noise = torch.empty_like(self._g_noise).normal_(generator=noise_gen)
            self._g_noise.copy_(noise)
            decoder_forward_calibrate_pi0(
                self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
                self._ae_calib_scales.data_ptr(), stream=0)
            torch.cuda.synchronize()
            per_sample_ae.append(self._ae_calib_scales.cpu().numpy().copy())

            if verbose and (i + 1) % max(1, n // 10) == 0:
                logger.info("  calibration sample %d/%d", i + 1, n)

        # --- Reduce + upload ---
        final_enc = accumulate_amax(per_sample_enc, percentile=percentile)
        final_ae  = accumulate_amax(per_sample_ae,  percentile=percentile)

        if verbose:
            logger.info("encoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample_enc, final_enc)))
            logger.info("decoder %s",
                        format_summary(summarize_amax_dispersion(
                            per_sample_ae, final_ae)))

        self._enc_calib_scales.copy_(
            torch.from_numpy(final_enc.astype(np.float32)))
        self._ae_calib_scales.copy_(
            torch.from_numpy(final_ae.astype(np.float32)))

        enc_ws = self._enc_w_dev.cpu().tolist()
        self._enc_alpha_host = [
            float(np.float32(final_enc[i]) * np.float32(enc_ws[i]))
            for i in range(Le * 4)]

        check_scale_ceiling(final_enc, label=f"pi0_thor_enc_N{n}")
        check_scale_ceiling(final_ae,  label=f"pi0_thor_ae_N{n}")

        self._capture_enc_ae_graph()
        self._real_data_calibrated = True
        logger.info(
            "Pi0 Thor multi-frame calibration + graph recapture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """Thor frontends do not yet surface a structured PrecisionSpec."""
        return None

    def _recalibrate_with_real_data(self):
        Se = self.Se; Le = self.Le; La = self.La
        total_keys = self.total_keys
        De = self.De; He = self.He; NHe = self.NHe; HDe = self.HDe

        # Encoder recalibration (identical to Pi0.5)
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

        # Decoder recalibration (Pi0 pattern)
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        S_dec = self.S_dec
        ae_bufs = {
            'noise':       self._g_noise.data_ptr(),
            'x':           self._ae_x.data_ptr(),
            'xn':          self._ae_xn.data_ptr(),
            'temp':        self._ae_temp.data_ptr(),
            'action_buf':  self._ae_action_buf.data_ptr(),
            'gate':        self._ae_gate.data_ptr(),
            'qkv':         self._ae_qkv.data_ptr(),
            'logits':      self._ae_logits.data_ptr(),
            'attn_out':    self._ae_attn.data_ptr(),
            'hid':         self._ae_hid.data_ptr(),
            'fg':          self._ae_fg.data_ptr(),
            'xn_fp8':      self._ae_xn_fp8.data_ptr(),
            'hid_fp8':     self._ae_hid_fp8.data_ptr(),
            'ctx_fp8':     self._ae_ctx_fp8.data_ptr(),
            'state_token': self._state_token.data_ptr(),
        }
        ae_weights = {
            'ain_w':        self._ain_w.data_ptr(),
            'ain_b':        self._ain_b.data_ptr(),
            'wa_w':         self._atmlp_in_wa.data_ptr(),
            'atmlp_out_w':  self._atmlp_out_w.data_ptr(),
            'atmlp_out_b':  self._atmlp_out_b.data_ptr(),
            'time_proj_all': self._time_proj_all.data_ptr(),
            'qw':           self._dec_qkv_flat.data_ptr(),
            'Kc':           self._Kc.reshape(-1).data_ptr(),
            'Vc':           self._Vc.reshape(-1).data_ptr(),
            'ow':           self._dec_o_flat.data_ptr(),
            'gw':           self._dec_gu_flat.data_ptr(),
            'dw':           self._dec_d_flat.data_ptr(),
            'aow':          self._aow.data_ptr(),
            'aob':          self._aob.data_ptr(),
            'final_norm_w': self._final_norm_w.data_ptr(),
            'rope':         self._dec_rope.data_ptr(),
            'w_scales':     self._ae_w_dev.data_ptr(),
        }
        ae_dims = {
            'Sa': Sa, 'S_dec': S_dec, 'D': Da, 'H': Ha,
            'NH': 8, 'HD': 256, 'steps': 10, 'layers': La,
            'enc_seq': Se, 'total_keys': total_keys,
        }

        _ae_calib_buf = torch.zeros(La * 4, dtype=torch.float32, device='cuda')
        _ae_d_scale = torch.zeros(1, dtype=torch.float32, device='cuda')
        _ae_hidden_scratch = torch.empty(S_dec * Ha, dtype=fp16, device='cuda')
        _ae_fp8_scratch = torch.zeros(S_dec * max(Da, Ha), dtype=torch.uint8, device='cuda')
        _ae_norm_scratch = torch.empty(S_dec * Da, dtype=fp16, device='cuda')
        _ae_x_scratch = torch.empty(S_dec * Da, dtype=fp16, device='cuda')
        _ae_ones = torch.ones(Da, dtype=fp16, device='cuda')
        ae_bufs['calib_buf'] = _ae_calib_buf.data_ptr()
        ae_bufs['d_scale'] = _ae_d_scale.data_ptr()
        ae_bufs['hidden_scratch'] = _ae_hidden_scratch.data_ptr()
        ae_bufs['fp8_scratch'] = _ae_fp8_scratch.data_ptr()
        ae_bufs['norm_scratch'] = _ae_norm_scratch.data_ptr()
        ae_bufs['x_scratch'] = _ae_x_scratch.data_ptr()
        ae_bufs['ones'] = _ae_ones.data_ptr()

        # Compute state_proj with real state data
        fvk.gmm_fp16(self._ctx, self._state_buf.data_ptr(),
                     self._state_proj_w.data_ptr(),
                     self._state_token.data_ptr(),
                     1, self.Da, 32, 0.0, 0)
        fvk.add_bias_fp16(self._state_token.data_ptr(),
                          self._state_proj_b.data_ptr(), 1, self.Da, 0)

        self._ae_calib_scales.zero_()
        self._g_noise.normal_()
        decoder_forward_calibrate_pi0(
            self._ctx, fvk, ae_bufs, ae_weights, ae_dims,
            self._ae_calib_scales.data_ptr(), stream=0)
        torch.cuda.synchronize()

        # Recapture graph with updated scales
        self._capture_enc_ae_graph()
        logger.info("Recalibrated with real data + graph recaptured")

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def get_latency_stats(self):
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
