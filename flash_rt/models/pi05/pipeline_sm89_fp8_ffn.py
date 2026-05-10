"""FlashRT — SM89 Pi0.5 inference pipeline with FP8 FFN optimization.

Key optimizations based on Session 136 analysis:
- encoder_seq_len padding: 265 -> 272 (FP8 kernel alignment requirement)
- Encoder FFN uses FP8 GEMM (Gate/Up/Down): 1.57x speedup, saves 12.57ms
- Vision, Encoder Attention, Decoder remain FP16 (FP8 not beneficial)

Architecture:
- Vision: 27 SigLIP layers (FP16 - small matrices)
- Encoder: 18 Gemma-2B layers
  - Attention: FP16 (precision needed for QKV)
  - FFN: FP8 (large matrices: 272x16384x2048)
- Decoder: 18 Gemma-300M layers (FP16 - seq=11 too small for FP8)
- Diffusion: 10-step flow-matching with AdaRMSNorm

Performance target: <50ms (from 59ms FP16 baseline)
"""

from __future__ import annotations

import ctypes
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.cuda_buffer import CudaBuffer
import flash_rt.flash_rt_kernels as frk

logger = logging.getLogger(__name__)


# Fixed Pi0.5 model dimensions
VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3  # 588

ENC_L = 18
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8
ENC_NKV = 1
ENC_HD = 256

DEC_L = 18
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256

ACTION_DIM = 32
CHUNK_SIZE_DEFAULT = 10
NUM_STEPS_DEFAULT = 10

FP16 = np.float16
FP32 = np.float32


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


class Pi05PipelineSm89Fp8Ffn:
    """Pi0.5 inference pipeline with FP8 FFN optimization for SM89.
    
    Key optimizations:
    1. encoder_seq_len padding (265 -> 272) for FP8 kernel alignment
    2. Encoder FFN uses FP8 GEMM (large matrices: 272x16384x2048)
    3. Other components use FP16 (FP8 not beneficial)
    
    Args:
        gemm: GemmRunner for FP16 GEMM operations.
        fvk: flash_rt_kernels module.
        weights: FP16 weight pointers dict.
        weights_fp8: FP8 weight pointers dict (Encoder FFN only).
        fp8_scales: FP8 scales dict (Encoder FFN only).
        num_views: Number of camera views.
        max_prompt_len: Maximum prompt token length.
        chunk_size: Action chunk size (default 10).
        num_steps: Diffusion denoise steps (default 10).
    """
    
    def __init__(
        self,
        gemm,
        fvk,
        weights,
        weights_fp8=None,
        fp8_scales=None,
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        self.gemm = gemm
        self.fvk = fvk
        self.weights = weights
        self.weights_fp8 = weights_fp8 or {}
        self.fp8_scales = fp8_scales or {}
        self._ctx = fvk.FvkContext()
        
        self.num_views = int(num_views)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)
        self.S_dec = self.chunk_size + 1
        self.num_steps = int(num_steps)
        
        # Fixed layer configuration
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        # Derived sizes with FP8 padding
        self.vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        encoder_seq_raw = self.vision_seq + self.max_prompt_len  # 265
        
        # FP8 kernel requires M % 16 == 0, pad to 272
        self.encoder_seq_padded = ((encoder_seq_raw + 15) // 16) * 16
        self.encoder_seq_len = self.encoder_seq_padded  # Use padded value for buffers
        
        logger.info(
            "FP8 FFN pipeline: encoder_seq_raw=%d, padded=%d (M%%16=%d)",
            encoder_seq_raw, self.encoder_seq_padded, self.encoder_seq_padded % 16)
        
        # FP8 support
        self.use_fp8_ffn = bool(weights_fp8)  # Enable if FP8 weights provided
        
        if self.use_fp8_ffn:
            # Pre-allocated FP8 activation buffers for Encoder FFN
            # CUDA Graph compatible: fixed size tensors
            self.encoder_x_fp8 = torch.empty(
                self.encoder_seq_padded, ENC_D,
                dtype=torch.float8_e4m3fn, device='cuda')
            self.encoder_hidden_fp8 = torch.empty(
                self.encoder_seq_padded, ENC_H,
                dtype=torch.float8_e4m3fn, device='cuda')
            
            # Scale tensor for quantization (scale=1.0 for activation)
            self.scale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            self.scale_one_ptr = self.scale_one.data_ptr()
            
            logger.info("FP8 FFN enabled: x_fp8=%dMB, hidden_fp8=%dMB",
                        self.encoder_x_fp8.numel() * 1 // 1024**2,
                        self.encoder_hidden_fp8.numel() * 1 // 1024**2)
        
        # Allocate buffers
        self.bufs = self._allocate_buffers()
        
        # Build RoPE table
        self._build_rope_table()
        
        # CUDART for D2D copies
        self._cudart = ctypes.CDLL("libcudart.so")
        
        # Pre-create torch tensors for CUDA Graph compatibility
        self._create_preallocated_tensors()
        
        logger.info(
            "Pi05PipelineSm89Fp8Ffn initialised (num_views=%d, vision_seq=%d, "
            "encoder_seq=%d padded, chunk_size=%d, num_steps=%d, fp8=%s)",
            self.num_views, self.vision_seq, self.encoder_seq_len,
            self.chunk_size, self.num_steps, self.use_fp8_ffn)
    
    def _create_preallocated_tensors(self):
        """Create pre-allocated torch tensors for CUDA Graph compatibility."""
        sd = self.S_dec
        es = self.encoder_seq_len
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        
        # Decoder scratch tensors
        self._dec_qkv_tensor = torch.empty(sd, dec_qkv_dim, dtype=torch.float16, device='cuda')
        self._dec_rope_tensor = torch.empty(sd, 256, dtype=torch.float16, device='cuda')
        self._dec_enc_K_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        self._dec_enc_V_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        
        # RoPE views
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
        
        # Encoder scratch tensors for attention
        enc_qkv_dim = (ENC_NH + 2 * ENC_NKV) * ENC_HD
        self._enc_qkv_tensor = torch.empty(es, enc_qkv_dim, dtype=torch.float16, device='cuda')
        self._enc_rope_tensor = torch.empty(es, 256, dtype=torch.float16, device='cuda')
        self._enc_rope_cos = self._enc_rope_tensor[:, :128]
        self._enc_rope_sin = self._enc_rope_tensor[:, 128:]
        
        # Vision batch attention tensors (Session 142 optimization)
        # batch SDPA: (num_views, VIS_NH, VIS_SEQ_PER_VIEW, VIS_HD)
        vs = self.vision_seq
        self._vis_qkv_tensor = torch.empty(vs, 3 * VIS_D, dtype=torch.float16, device='cuda')
        self._vis_Q_batch = torch.empty(self.num_views, VIS_NH, VIS_SEQ_PER_VIEW, VIS_HD, dtype=torch.float16, device='cuda')
        self._vis_K_batch = torch.empty(self.num_views, VIS_NH, VIS_SEQ_PER_VIEW, VIS_HD, dtype=torch.float16, device='cuda')
        self._vis_V_batch = torch.empty(self.num_views, VIS_NH, VIS_SEQ_PER_VIEW, VIS_HD, dtype=torch.float16, device='cuda')
        self._vis_attn_out_batch = torch.empty(self.num_views, VIS_NH, VIS_SEQ_PER_VIEW, VIS_HD, dtype=torch.float16, device='cuda')
        self._vis_attn_out = torch.empty(vs, VIS_D, dtype=torch.float16, device='cuda')
        
        logger.info("Created pre-allocated tensors for CUDA Graph (Vision batch enabled)")
    
    def _allocate_buffers(self) -> dict:
        """Allocate all pipeline working buffers."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len  # Padded value (272)
        sd = self.S_dec
        sa = self.chunk_size
        B = {}
        
        # Vision (SigLIP) - same as original
        B["observation_images_normalized"] = CudaBuffer.device_empty(
            nv * 224 * 224 * 3, FP16)
        B["vision_x"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_x_norm"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_QKV"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP16)
        B["vision_hidden"] = CudaBuffer.device_empty(vs * VIS_H, FP16)
        B["vision_patches"] = CudaBuffer.device_empty(vs * VIS_PATCH_FLAT, FP16)
        B["vision_pos_embed_expanded"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_attn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_ffn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        
        # Encoder (Gemma-2B) - use padded seq_len for buffers
        B["encoder_x"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_x_norm"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_QKV"] = CudaBuffer.device_empty(
            es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, FP16)
        B["encoder_hidden"] = CudaBuffer.device_empty(es * ENC_H, FP16)
        B["encoder_gate_merged"] = CudaBuffer.device_empty(es * 2 * ENC_H, FP16)
        B["encoder_K_cache"] = CudaBuffer.device_empty(
            ENC_L * es * ENC_NKV * ENC_HD, FP16)  # Padded size for cache
        B["encoder_V_cache"] = CudaBuffer.device_empty(
            ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_attn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_ffn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        
        # RMSNorm ones buffers
        B["encoder_ones"] = CudaBuffer.from_numpy(np.ones(ENC_D, dtype=FP16))
        B["decoder_ones"] = CudaBuffer.from_numpy(np.ones(DEC_D, dtype=FP16))
        
        # Decoder (Gemma-300M) - with AdaRMSNorm buffers
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(
            sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        B["decoder_attn_out"] = CudaBuffer.device_empty(sd * DEC_NH * DEC_HD, FP16)
        
        # AdaRMSNorm scratch buffers
        B["x_normed_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["gate_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        # Precomputed style buffers (uploaded from frontend)
        B["decoder_time_emb"] = CudaBuffer.device_empty(
            self.num_steps * sd * DEC_D, FP16)
        B["decoder_style_attn"] = CudaBuffer.device_empty(
            self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_ffn"] = CudaBuffer.device_empty(
            self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_final"] = CudaBuffer.device_empty(
            self.num_steps * sd * 3 * DEC_D, FP16)
        
        # Diffusion
        B["diffusion_noise"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        B["decoder_action_buf"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        
        return B
    
    def _build_rope_table(self):
        """Build RoPE cos/sin tables."""
        # Use padded encoder_seq_len for RoPE table
        max_pos = self.encoder_seq_len + self.S_dec
        
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]
        
        cos = np.cos(phase).astype(FP16)
        sin = np.sin(phase).astype(FP16)
        
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)
        self._rope_table_np = interleaved
        
        # Encoder RoPE (use padded size)
        enc_rope = interleaved[:self.encoder_seq_len]
        self.bufs["encoder_rope"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(enc_rope))
        
        # Decoder RoPE
        dec_rope = interleaved[
            self.encoder_seq_len:self.encoder_seq_len + self.S_dec]
        self.bufs["decoder_rope"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(dec_rope))
    
    def _build_pos_embed_expanded(self):
        """Expand position embedding across num_views."""
        pos_src_ptr = self.weights["vision_position_embedding"]
        per_view_nbytes = VIS_SEQ_PER_VIEW * VIS_D * 2
        dst_buf = self.bufs["vision_pos_embed_expanded"]
        
        for v in range(self.num_views):
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(dst_buf.ptr.value + v * per_view_nbytes),
                ctypes.c_void_p(pos_src_ptr),
                per_view_nbytes, 3)
        self._cudart.cudaDeviceSynchronize()
    
    def upload_precomputed_styles(self, styles: dict):
        """Upload frontend-precomputed decoder style buffers."""
        B = self.bufs
        
        if "time_emb" in styles:
            B["decoder_time_emb"].upload(np.ascontiguousarray(styles["time_emb"]))
        
        if "style_attn" in styles:
            B["decoder_style_attn"].upload(np.ascontiguousarray(styles["style_attn"]))
        
        if "style_ffn" in styles:
            B["decoder_style_ffn"].upload(np.ascontiguousarray(styles["style_ffn"]))
        
        if "style_final" in styles:
            B["decoder_style_final"].upload(np.ascontiguousarray(styles["style_final"]))
        
        logger.info("Uploaded precomputed styles for %d diffusion steps", self.num_steps)
    
    def _style_slice_ptr(self, buf_name: str, step: int, layer: int | None = None) -> int:
        """Compute device pointer for a per-step style slice."""
        base = _p(self.bufs[buf_name])
        sd = self.S_dec
        
        if buf_name == "decoder_time_emb":
            return base + step * sd * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * sd * 3 * DEC_D * 2
        
        per_layer = sd * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer
    
    # ══════════════════════════════════════════════════════════════════
    #  AdaRMSNorm (optimized kernel implementation)
    # ══════════════════════════════════════════════════════════════════
    
    def _ada_rms_norm_fp16(self, x_ptr: int, style_ptr: int,
                           out_ptr: int, gate_out_ptr: int,
                           seq: int, dim: int, stream: int):
        """AdaRMSNorm using optimized fvk kernel."""
        self.fvk.adarms_fp16(x_ptr, style_ptr, out_ptr, gate_out_ptr, seq, dim, stream)
    
    def _gate_residual_fp16(self, x_ptr: int, residual_ptr: int,
                            gate_ptr: int, seq_dim: int, stream: int):
        """Gate * residual add."""
        self.fvk.gate_mul_residual(x_ptr, residual_ptr, gate_ptr, seq_dim, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase A: Vision Encoder (SigLIP 27 layers) - FP16 only
    # ══════════════════════════════════════════════════════════════════
    
    def vision_encoder(self, stream: int = 0):
        """Run SigLIP vision encoder - FP16 (no FP8 optimization needed)."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        nv = self.num_views
        
        # Patch embedding
        fvk.patch_im2col(
            _p(B["observation_images_normalized"]),
            _p(B["vision_patches"]),
            nv, stream)
        
        gemm.fp16_nn(
            _p(B["vision_patches"]),
            W["vision_patch_embedding_w"],
            _p(B["vision_x"]),
            seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_pos_embed_expanded"]),
            W["vision_patch_embedding_b"],
            seq, VIS_D, stream=stream)
        
        for i in range(VIS_L):
            self._vision_layer(i, stream)
    
    def _vision_layer(self, i: int, stream: int):
        """One SigLIP layer - FP16 with batch attention optimization."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        nv = self.num_views
        
        # Pre-attention LayerNorm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W["vision_pre_attn_norm_w"][i],
            W["vision_pre_attn_norm_b"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, 1e-5, stream=stream)
        
        # QKV GEMM (FP16)
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W["vision_attn_qkv_w"][i],
            _p(B["vision_QKV"]),
            seq, 3 * VIS_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_QKV"]),
            W["vision_attn_qkv_b"][i],
            seq, 3 * VIS_D, stream=stream)
        
        # Copy to pre-allocated tensor and run batch SDPA
        fvk.gpu_copy(self._vis_qkv_tensor.data_ptr(), _p(B["vision_QKV"]), seq * 3 * VIS_D * 2, stream)
        
        Q = self._vis_qkv_tensor[:, :VIS_D].view(seq, VIS_NH, VIS_HD)
        K = self._vis_qkv_tensor[:, VIS_D:2*VIS_D].view(seq, VIS_NH, VIS_HD)
        V = self._vis_qkv_tensor[:, 2*VIS_D:].view(seq, VIS_NH, VIS_HD)
        
        # Batch attention: reshape to (num_views, VIS_NH, VIS_SEQ_PER_VIEW, VIS_HD)
        self._vis_Q_batch.copy_(Q.view(nv, VIS_SEQ_PER_VIEW, VIS_NH, VIS_HD).permute(0, 2, 1, 3))
        self._vis_K_batch.copy_(K.view(nv, VIS_SEQ_PER_VIEW, VIS_NH, VIS_HD).permute(0, 2, 1, 3))
        self._vis_V_batch.copy_(V.view(nv, VIS_SEQ_PER_VIEW, VIS_NH, VIS_HD).permute(0, 2, 1, 3))
        
        # Single batch SDPA kernel
        attn_result = F.scaled_dot_product_attention(self._vis_Q_batch, self._vis_K_batch, self._vis_V_batch)
        
        # Reshape output back to (seq, VIS_D)
        self._vis_attn_out.copy_(attn_result.permute(0, 2, 1, 3).reshape(seq, VIS_D))
        
        fvk.gpu_copy(_p(B["vision_attn_out"]), self._vis_attn_out.data_ptr(), seq * VIS_D * 2, stream)
        
        # O projection
        gemm.fp16_nn(
            _p(B["vision_attn_out"]),
            W["vision_attn_o_w"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_x_norm"]),
            W["vision_attn_o_b"][i],
            seq, VIS_D, stream=stream)
        
        # Pre-FFN LayerNorm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W["vision_pre_ffn_norm_w"][i],
            W["vision_pre_ffn_norm_b"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, 1e-5, stream=stream)
        
        # FFN up (FP16)
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W["vision_ffn_up_w"][i],
            _p(B["vision_hidden"]),
            seq, VIS_H, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_hidden"]),
            W["vision_ffn_up_b"][i],
            seq, VIS_H, stream=stream)
        fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
        
        # FFN down (FP16)
        gemm.fp16_nn(
            _p(B["vision_hidden"]),
            W["vision_ffn_down_w"][i],
            _p(B["vision_ffn_out"]),
            seq, VIS_D, VIS_H, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_ffn_out"]),
            W["vision_ffn_down_b"][i],
            seq, VIS_D, stream=stream)
        
        fvk.residual_add_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]), seq * VIS_D, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase B: Gemma-2B Encoder (18 layers)
    #  Attention: FP16 (precision needed)
    #  FFN: FP8 (large matrices: 272x16384x2048)
    # ══════════════════════════════════════════════════════════════════
    
    def transformer_encoder(self, stream: int = 0):
        """Run Gemma-2B encoder - Attention FP16, FFN FP8 if enabled."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len  # Padded value (272)
        
        # Vision final norm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W["vision_final_norm_w"],
            W["vision_final_norm_b"],
            _p(B["vision_x_norm"]),
            vs, VIS_D, 1e-5, stream=stream)
        
        # Multi-modal projector
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W["encoder_multi_modal_projector_w"],
            _p(B["encoder_x"]),
            vs, ENC_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["encoder_x"]),
            W["encoder_multi_modal_projector_b"],
            vs, ENC_D, stream=stream)
        
        # Encoder layers
        for i in range(self.enc_layers):
            self._encoder_layer(i, es, stream)
    
    def _encoder_layer(self, i: int, seq: int, stream: int):
        """One encoder layer - Attention FP16, FFN FP8 if enabled."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        W8 = self.weights_fp8  # FP8 weights (may be empty)
        S8 = self.fp8_scales  # FP8 scales
        B = self.bufs
        
        # RMSNorm
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        # QKV GEMM (FP16 - precision needed)
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_attn_qkv_w"][i],
            _p(B["encoder_QKV"]),
            seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)
        
        # Split QKV using pre-allocated tensor
        enc_qkv_dim = (ENC_NH + 2 * ENC_NKV) * ENC_HD
        fvk.gpu_copy(self._enc_qkv_tensor.data_ptr(), _p(B["encoder_QKV"]),
                     seq * enc_qkv_dim * 2, stream)
        
        Q = self._enc_qkv_tensor[:, :ENC_NH * ENC_HD].view(seq, ENC_NH, ENC_HD)
        KV = self._enc_qkv_tensor[:, ENC_NH * ENC_HD:].view(seq, 2, ENC_NKV, ENC_HD)
        K = KV[:, 0, :, :]
        V = KV[:, 1, :, :]
        
        # Apply RoPE using optimized kernel (8x faster than torch ops)
        fvk.gpu_copy(self._enc_rope_tensor.data_ptr(), _p(B["encoder_rope"]), seq * 256 * 2, stream)
        
        # Use optimized RoPE kernel on flat tensors
        Q_flat = Q.reshape(seq, ENC_NH * ENC_HD)  # [S, NH*HD]
        K_flat = K.reshape(seq, ENC_NKV * ENC_HD)  # [S, NKV*HD]
        fvk.rope_rotate_half_fp16(Q_flat.data_ptr(), self._enc_rope_tensor[:, :ENC_HD//2].data_ptr(),
                                  self._enc_rope_tensor[:, ENC_HD//2:].data_ptr(), seq, ENC_NH, ENC_HD, stream)
        fvk.rope_rotate_half_fp16(K_flat.data_ptr(), self._enc_rope_tensor[:, :ENC_HD//2].data_ptr(),
                                  self._enc_rope_tensor[:, ENC_HD//2:].data_ptr(), seq, ENC_NKV, ENC_HD, stream)
        Q_rope = Q_flat.reshape(seq, ENC_NH, ENC_HD)
        K_rope = K_flat.reshape(seq, ENC_NKV, ENC_HD)
        
        # Write K/V to cache
        K_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        V_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        
        fvk.gpu_copy(_p(B["encoder_K_cache"]) + K_cache_offset,
                     K_rope.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(_p(B["encoder_V_cache"]) + V_cache_offset,
                     V.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        
        # Torch attention (GQA)
        K_expanded = K_rope.expand(-1, ENC_NH, -1)
        V_expanded = V.expand(-1, ENC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = K_expanded.transpose(0, 1)
        V_t = V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(seq, ENC_D)
        
        fvk.gpu_copy(_p(B["encoder_attn_out"]), attn_out.data_ptr(), seq * ENC_D * 2, stream)
        
        # O projection (FP16)
        gemm.fp16_nn(
            _p(B["encoder_attn_out"]),
            W["encoder_attn_o_w"][i],
            _p(B["encoder_x_norm"]),
            seq, ENC_D, ENC_D, stream=stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN - FP8 if weights available, FP16 otherwise
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        if self.use_fp8_ffn and i in W8.get("gate", {}):
            self._encoder_ffn_fp8(i, seq, stream)
        else:
            self._encoder_ffn_fp16(i, seq, stream)
    
    def _encoder_ffn_fp16(self, i: int, seq: int, stream: int):
        """Encoder FFN with FP16 GEMM (fallback)."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_ffn_gate_w"][i],
            _p(B["encoder_gate_merged"]),
            seq, ENC_H, ENC_D, stream=stream)
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_ffn_up_w"][i],
            _p(B["encoder_hidden"]),
            seq, ENC_H, ENC_D, stream=stream)
        
        fvk.gate_geglu_fp16(
            _p(B["encoder_gate_merged"]),
            _p(B["encoder_hidden"]),
            _p(B["encoder_hidden"]),
            seq * ENC_H, stream=stream)
        
        gemm.fp16_nn(
            _p(B["encoder_hidden"]),
            W["encoder_ffn_down_w"][i],
            _p(B["encoder_ffn_out"]),
            seq, ENC_D, ENC_H, stream=stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_ffn_out"]), seq * ENC_D, stream)
    
    def _encoder_ffn_fp8(self, i: int, seq: int, stream: int):
        """Encoder FFN with FP8 GEMM (optimized version).
        
        Key steps:
        1. Quantize x_norm to FP8 (scale=1.0)
        2. FP8 GEMM for Gate/Up
        3. Gate-GeGLU fusion
        4. Quantize hidden to FP8
        5. FP8 GEMM for Down
        """
        fvk = self.fvk
        W8 = self.weights_fp8
        S8 = self.fp8_scales
        B = self.bufs
        
        # Quantize x_norm to FP8
        fvk.quantize_fp8_static_fp16(
            _p(B["encoder_x_norm"]),
            self.encoder_x_fp8.data_ptr(),
            self.scale_one_ptr,
            seq * ENC_D, stream)
        
        # Gate (FP8 GEMM): x_fp8 @ W8_gate -> gate_merged (BF16)
        frk.cutlass_ada_fp8_gemm_bf16_simple(
            self.encoder_x_fp8.data_ptr(),
            W8["gate"][i],
            _p(B["encoder_gate_merged"]),
            seq, ENC_H, ENC_D,
            1.0, S8["gate"][i], stream)
        
        # Up (FP8 GEMM): x_fp8 @ W8_up -> hidden (BF16)
        frk.cutlass_ada_fp8_gemm_bf16_simple(
            self.encoder_x_fp8.data_ptr(),
            W8["up"][i],
            _p(B["encoder_hidden"]),
            seq, ENC_H, ENC_D,
            1.0, S8["up"][i], stream)
        
        # Gate-GeGLU fusion
        fvk.gate_geglu_fp16(
            _p(B["encoder_gate_merged"]),
            _p(B["encoder_hidden"]),
            _p(B["encoder_hidden"]),
            seq * ENC_H, stream=stream)
        
        # Quantize hidden to FP8
        fvk.quantize_fp8_static_fp16(
            _p(B["encoder_hidden"]),
            self.encoder_hidden_fp8.data_ptr(),
            self.scale_one_ptr,
            seq * ENC_H, stream)
        
        # Down (FP8 GEMM): hidden_fp8 @ W8_down -> ffn_out (BF16)
        frk.cutlass_ada_fp8_gemm_bf16_simple(
            self.encoder_hidden_fp8.data_ptr(),
            W8["down"][i],
            _p(B["encoder_ffn_out"]),
            seq, ENC_D, ENC_H,
            1.0, S8["down"][i], stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_ffn_out"]), seq * ENC_D, stream)
    
    def _apply_rope_interleaved(self, x, cos, sin):
        """Apply interleaved RoPE."""
        x_even = x[..., :x.shape[-1]//2]
        x_odd = x[..., x.shape[-1]//2:]
        
        if x.dim() == 3:
            cos_exp = cos.unsqueeze(1)
            sin_exp = sin.unsqueeze(1)
        else:
            cos_exp = cos
            sin_exp = sin
        
        rotated_even = x_even * cos_exp - x_odd * sin_exp
        rotated_odd = x_even * sin_exp + x_odd * cos_exp
        
        return torch.cat([rotated_even, rotated_odd], dim=-1)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase C: Decoder + Diffusion (AdaRMSNorm) - FP16 only
    # ══════════════════════════════════════════════════════════════════
    
    def _assemble_decoder_x(self, step: int, stream: int):
        """Build decoder_x = [time_emb[0]; action_in_proj(noise) + time_emb[1:]]."""
        B = self.bufs
        W = self.weights
        fvk = self.fvk
        gemm = self.gemm
        sa = self.chunk_size
        
        # Get time embedding for this step
        time_emb_ptr = self._style_slice_ptr("decoder_time_emb", step)
        
        # Copy time_emb[0] to decoder_x[0]
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(_p(B["decoder_x"])),
            ctypes.c_void_p(time_emb_ptr),
            DEC_D * 2, 3, stream)
        
        # action_in_proj(noise) -> decoder_x[1:]
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        
        gemm.fp16_nn(
            _p(B["diffusion_noise"]),
            W["decoder_action_in_proj_w"],
            x_action_ptr,
            sa, DEC_D, ACTION_DIM, stream=stream)
        fvk.add_bias_fp16(
            x_action_ptr,
            W["decoder_action_in_proj_b"],
            sa, DEC_D, stream=stream)
        
        # Add time_emb[1:] to action projection
        time_emb_action_ptr = time_emb_ptr + DEC_D * 2
        fvk.residual_add_fp16(
            x_action_ptr,
            time_emb_action_ptr,
            sa * DEC_D, stream)
    
    def transformer_decoder(self, stream: int = 0):
        """Run decoder + 10-step diffusion with AdaRMSNorm - FP16."""
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        es = self.encoder_seq_len
        sd = self.S_dec
        
        for step in range(self.num_steps):
            self._assemble_decoder_x(step, stream)
            
            for i in range(self.dec_layers):
                self._decoder_layer(i, step, es, sd, stream)
            
            # Final AdaRMSNorm + output projection
            style_final_ptr = self._style_slice_ptr("decoder_style_final", step)
            
            self._ada_rms_norm_fp16(
                _p(B["decoder_x"]),
                style_final_ptr,
                _p(B["x_normed_buf"]),
                _p(B["gate_buf"]),
                sd, DEC_D, stream)
            
            x_out_action_ptr = _p(B["x_normed_buf"]) + DEC_D * 2
            
            gemm.fp16_nn(
                x_out_action_ptr,
                W["decoder_action_out_proj_w"],
                _p(B["diffusion_noise"]),
                sd - 1, ACTION_DIM, DEC_D, stream=stream)
            self.fvk.add_bias_fp16(
                _p(B["diffusion_noise"]),
                W["decoder_action_out_proj_b"],
                sd - 1, ACTION_DIM, stream)
    
    def _decoder_layer(self, i: int, step: int, enc_seq: int, sd: int, stream: int):
        """One decoder layer with AdaRMSNorm - FP16."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        style_attn_ptr = self._style_slice_ptr("decoder_style_attn", step, i)
        style_ffn_ptr = self._style_slice_ptr("decoder_style_ffn", step, i)
        
        # AdaRMSNorm (attention)
        self._ada_rms_norm_fp16(
            _p(B["decoder_x"]),
            style_attn_ptr,
            _p(B["x_normed_buf"]),
            _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        gemm.fp16_nn(
            _p(B["x_normed_buf"]),
            W["decoder_attn_qkv_w"][i],
            _p(B["decoder_QKV"]),
            sd, dec_qkv_dim, DEC_D, stream=stream)
        
        # Split QKV
        qkv_tensor = self._dec_qkv_tensor
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["decoder_QKV"]), sd * dec_qkv_dim * 2, stream)
        
        Q = qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)
        KV = qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)
        K_dec = KV[:, 0, :, :]
        V_dec = KV[:, 1, :, :]
        
        # Apply RoPE using optimized kernel (8x faster than torch ops)
        fvk.gpu_copy(self._dec_rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        
        # Use optimized RoPE kernel on flat tensors
        Q_flat = qkv_tensor[:, :DEC_NH * DEC_HD]  # [S, NH*HD]
        K_flat = qkv_tensor[:, DEC_NH * DEC_HD:DEC_NH * DEC_HD + DEC_NKV * DEC_HD]  # [S, NKV*HD]
        fvk.rope_rotate_half_fp16(Q_flat.data_ptr(), self._dec_rope_tensor[:, :DEC_HD//2].data_ptr(),
                                  self._dec_rope_tensor[:, DEC_HD//2:].data_ptr(), sd, DEC_NH, DEC_HD, stream)
        fvk.rope_rotate_half_fp16(K_flat.data_ptr(), self._dec_rope_tensor[:, :DEC_HD//2].data_ptr(),
                                  self._dec_rope_tensor[:, DEC_HD//2:].data_ptr(), sd, DEC_NKV, DEC_HD, stream)
        Q_rope = Q_flat.view(sd, DEC_NH, DEC_HD)
        K_dec_rope = K_flat.view(sd, DEC_NKV, DEC_HD)
        
        # Cross-attention with encoder KV cache
        enc_K_ptr = _p(B["encoder_K_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        enc_V_ptr = _p(B["encoder_V_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        
        fvk.gpu_copy(self._dec_enc_K_tensor.data_ptr(), enc_K_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(self._dec_enc_V_tensor.data_ptr(), enc_V_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        
        enc_K_expanded = self._dec_enc_K_tensor.expand(-1, DEC_NH, -1)
        enc_V_expanded = self._dec_enc_V_tensor.expand(-1, DEC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = enc_K_expanded.transpose(0, 1)
        V_t = enc_V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(sd, DEC_NH * DEC_HD)
        
        fvk.gpu_copy(_p(B["decoder_attn_out"]), attn_out.data_ptr(), sd * DEC_NH * DEC_HD * 2, stream)
        
        # O projection
        gemm.fp16_nn(
            _p(B["decoder_attn_out"]),
            W["decoder_attn_o_w"][i],
            _p(B["x_normed_buf"]),
            sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        
        self._gate_residual_fp16(
            _p(B["decoder_x"]),
            _p(B["x_normed_buf"]),
            _p(B["gate_buf"]),
            sd * DEC_D, stream)
        
        # AdaRMSNorm (FFN)
        self._ada_rms_norm_fp16(
            _p(B["decoder_x"]),
            style_ffn_ptr,
            _p(B["x_normed_buf"]),
            _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        gemm.fp16_nn(
            _p(B["x_normed_buf"]),
            W["decoder_ffn_gate_w"][i],
            _p(B["decoder_gate_merged"]),
            sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(
            _p(B["x_normed_buf"]),
            W["decoder_ffn_up_w"][i],
            _p(B["decoder_hidden"]),
            sd, DEC_H, DEC_D, stream=stream)
        
        fvk.gate_geglu_fp16(
            _p(B["decoder_gate_merged"]),
            _p(B["decoder_hidden"]),
            _p(B["decoder_hidden"]),
            sd * DEC_H, stream=stream)
        
        gemm.fp16_nn(
            _p(B["decoder_hidden"]),
            W["decoder_ffn_down_w"][i],
            _p(B["decoder_attn_out"]),
            sd, DEC_D, DEC_H, stream=stream)
        
        self._gate_residual_fp16(
            _p(B["decoder_x"]),
            _p(B["decoder_attn_out"]),
            _p(B["gate_buf"]),
            sd * DEC_D, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Full pipeline
    # ══════════════════════════════════════════════════════════════════
    
    def run_pipeline(self, stream: int = 0, sync: bool = True):
        """Run full inference pipeline."""
        self._copy_lang_embeds_to_encoder_x(stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        self.transformer_decoder(stream)
        if sync:
            self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    
    def set_language_embeds(self, lang_embeds_np: np.ndarray):
        """Set language embedding from prompt."""
        prompt_len = lang_embeds_np.shape[0]
        assert prompt_len <= self.max_prompt_len
        
        arr = np.ascontiguousarray(lang_embeds_np)
        self._lang_embeds_buf = CudaBuffer.from_numpy(arr)
        self._current_prompt_len = prompt_len
    
    def _copy_lang_embeds_to_encoder_x(self, stream: int):
        """Copy language embeddings to encoder_x buffer."""
        if not hasattr(self, "_lang_embeds_buf"):
            return
        
        start_byte = self.vision_seq * ENC_D * 2
        dst_ptr = _p(self.bufs["encoder_x"]) + start_byte
        
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            self._lang_embeds_buf.ptr,
            self._lang_embeds_buf.nbytes, 3, stream)
    
    @property
    def input_images_buf(self):
        return self.bufs["observation_images_normalized"]
    
    @property
    def input_noise_buf(self):
        return self.bufs["diffusion_noise"]
    
    @property
    def output_noise_buf(self):
        return self.bufs["diffusion_noise"]