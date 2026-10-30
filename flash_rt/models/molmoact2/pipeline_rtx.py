"""FlashRT — MolmoAct2 RTX (consumer Blackwell SM120 / Ada SM89) pipeline.

Native-kernel execution plan for ``allenai/MolmoAct2`` continuous action
inference.  The complete steady-state graph is:

    uint8 cameras ──resize/patchify──► ViT (25L, hd72) ──► 2x2 pooling
      adapter (masked window attention) ──► projector MLP ──► token embeds
      ──► frozen Qwen2.5-style VLM prefill (36L, GQA 32/8, hd128, qk-norm)
      ──► per-layer cross-attn K/V projection ──► 36-block action expert
      ──► 10-step flow-matching Euler loop ──► action chunk.

Every model-math stage is a repository-native kernel: cuBLASLt GEMMs through
``GemmRunner`` (bf16 / FP8 E4M3), vendored Flash-Attention 2 (``flash_rt_fa2``)
for ViT / causal prefill / action self- and cross-attention, and the
``flash_rt_kernels`` norm / rope / activation / elementwise surface plus the
MolmoAct2-specific helpers in ``csrc/kernels/molmoact2_kernels.cu``.  The whole
pipeline is captured into one CUDA Graph over static buffers.

Layout contract: all GEMM weights are stored as ``[K, N]`` (FlashRT "kn"
layout) so ``GemmRunner.bf16_nn`` / ``fp8_nn_dev`` consume them directly.
"""

from __future__ import annotations

import ctypes
import logging
import os

import numpy as np

from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.cuda_graph import CUDAGraph

logger = logging.getLogger(__name__)

# ── MolmoAct2 fixed dimensions ────────────────────────────────────────────
NUM_VIEWS = 3
IMG_H, IMG_W = 360, 640          # production camera resolution (BimanualYAM)
TILE = 378                       # processor resize target
PATCH = 14
PATCH_DIM = PATCH * PATCH * 3    # 588
PATCH_DIM_PAD = 592               # 16-byte rows for the patch-embed GEMM
GRID = TILE // PATCH             # 27
POS_TOKENS = GRID * GRID         # 729
POOL = 2
POOL_GRID = ((GRID + POOL - 1) // POOL)  # 14 (27 -> 28 padded)
POOL_TOKENS = POOL_GRID * POOL_GRID      # 196
POOL_WINDOW = POOL * POOL                # 4

TOTAL_PATCHES = NUM_VIEWS * POS_TOKENS       # 2187
TOTAL_POOLED = NUM_VIEWS * POOL_TOKENS       # 588

# Vision tower
VIS_D = 1152
VIS_FFN = 4304
# The repository NVFP4 GELU-epilogue GEMM requires M % 128 == 0 and
# N % 32 == 0; the vision MLP width 4304 is padded to 4352 (zero weight
# rows/columns) and the patch count 2187 to 2304 rows.
VIS_FFN_PAD = 4352
VIS_PAD = 2304
VIS_NH = 16
VIS_HD = 72
VIS_LAYERS = 25
VIT_FEATURE_LAYERS = (24, 18)     # concatenation order from adapter_config.vit_layers
VIS_FEAT_D = VIS_D * len(VIT_FEATURE_LAYERS)  # 2304

# VLM text tower
TXT_D = 2560
TXT_FFN = 9728
TXT_NH = 32
TXT_NKV = 8
TXT_HD = 128
TXT_LAYERS = 36
TXT_Q_DIM = TXT_NH * TXT_HD       # 4096
TXT_KV_DIM = TXT_NKV * TXT_HD     # 1024
TXT_QKV_DIM = TXT_Q_DIM + 2 * TXT_KV_DIM
TXT_ROPE_THETA = 5000000.0
EPS = 1e-6

# Action expert
ACT_D = 768
ACT_H = 3072
ACT_NH = 8
ACT_HD = 96
ACT_LAYERS = 36
ACT_HORIZON = 30
ACT_MAX_DIM = 32
ACT_STEPS = 10
# Row stride (elements) of the head-major action context (96 + 8 padding)
ACT_MK_CTX_STRIDE = 104
ACT_KV_DIM = ACT_NH * ACT_HD      # 768
CONTEXT_DIM = TXT_KV_DIM

BF16 = np.float16                 # sizing-only (2 bytes); see pi05 pipeline note
FP8 = np.uint8
FP32 = np.float32
INT64 = np.int64
INT32 = np.int32


def _p(buf: CudaBuffer, off_bytes: int = 0) -> int:
    return buf.ptr.value + off_bytes


class MolmoAct2Pipeline:
    """Static-shape MolmoAct2 inference pipeline.

    Args:
        gemm: ``fvk.GemmRunner`` instance (cuBLASLt BF16 / FP8 GEMMs).
        fvk:  ``flash_rt.flash_rt_kernels`` module.
        fa2:  ``flash_rt.flash_rt_fa2`` module (vendored FA2).
        weights: pointer dict produced by the frontend; see
            ``MolmoAct2TorchFrontendRtx._build_pipeline_weights``.
        prompt_len: static tokenised prompt length for the production task.
        num_steps: flow-matching steps (checkpoint default 10).
        use_fp8: run the large GEMMs on FP8 E4M3 weights/activations.
    """

    def __init__(self, gemm, fvk, fa2, weights, *, prompt_len: int,
                 num_steps: int = ACT_STEPS, use_fp8: bool = True,
                 act_dim: int = 14, action_horizon: int = ACT_HORIZON,
                 vis_fused: bool = False):
        self.gemm = gemm
        self.fvk = fvk
        self.fa2 = fa2
        self.weights = weights
        self.S = int(prompt_len)
        self.num_steps = int(num_steps)
        self.use_fp8 = bool(use_fp8)
        self.act_dim = int(act_dim)
        self.horizon = int(action_horizon)
        # Fused NVFP4 GELU-epilogue vision MLP pads the 2187 patches to 2304
        # rows; the residual scratch grows with it.
        self.vis_pad = VIS_PAD if vis_fused else TOTAL_PATCHES
        # FP8 w2 route: the MLP width is zero-padded 4304 -> 4352 so the
        # w2 reduction runs on a 128-multiple K (zero products are exact).
        self.vis_ffn = (VIS_FFN_PAD if weights.get("vis_w2_pad")
                        else VIS_FFN)

        self._cudart = ctypes.CDLL("libcudart.so")
        self._cudart.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p]
        self._cudart.cudaMemcpyAsync.restype = ctypes.c_int
        self._cudart.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self._cudart.cudaMemcpy2DAsync.restype = ctypes.c_int

        self.fp8_calibrated = False
        self.fp8_scales: dict[str, CudaBuffer] = {}
        self.fp8_scale_host: dict[str, float] = {}
        self.shared_scale_ok: dict[str, bool] = {}
        self.skinny_ags_off: dict[tuple[str, int], int] = {}
        # FLASHRT_MOLMO_SKINNY_SCOPE: outd (default) | off | all | ffn |
        # attn | out | outfg — selects which action-expert sites use the
        # skinny NVFP4 (W4A4) family. "outd" (self/cross out-projections and
        # the FFN down projection) keeps delivery similarity >= 0.99 on the
        # reference observations; scopes that quantize the QKV/query/up-gate
        # projections fall below it.
        self.skinny_scope = os.environ.get(
            "FLASHRT_MOLMO_SKINNY_SCOPE", "outd").strip().lower()
        self._current_step = -1
        self.input_images_ptr = 0
        self._vit_layer_hook = None

        self.bufs = self._allocate_buffers()
        self._build_rope_tables()
        self._upload_precomputed()
        self._graph = None
        self._graph_stream = None

    # ══════════════════════════════════════════════════════════════════
    #  Buffers
    # ══════════════════════════════════════════════════════════════════
    def _alloc(self, name: str, count: int, dtype, zero: bool = False):
        buf = (CudaBuffer.device_zeros(count, dtype) if zero
               else CudaBuffer.empty(count, dtype))
        self.bufs[name] = buf
        return buf

    def _allocate_buffers(self) -> dict:
        B: dict[str, CudaBuffer] = {}
        self.bufs = B
        S = self.S
        # ── prompt / VLM prefill ──
        self._alloc("ids", S, INT64)
        self._alloc("emb", S * TXT_D, BF16)
        self._alloc("x", S * TXT_D, BF16)
        self._alloc("tmp", S * TXT_D, BF16)
        self._alloc("xalt", S * TXT_D, BF16)
        self._alloc("qkv", S * TXT_QKV_DIM, BF16)
        self._alloc("Q", S * TXT_Q_DIM, BF16)
        self._alloc("K_cache", TXT_LAYERS * S * TXT_KV_DIM, BF16)
        kv_pad = ((S + 127) // 128) * 128
        self.kv_pad = kv_pad
        # padded FP8 caches: raw-cast operands for the SM120 FP8 attention
        # (pad rows stay zero; the first S rows also serve the context GEMMs)
        self._alloc("K_cache_fp8", TXT_LAYERS * kv_pad * TXT_KV_DIM, FP8, True)
        self._alloc("V_cache_fp8", TXT_LAYERS * kv_pad * TXT_KV_DIM, FP8, True)
        self._alloc("Q_fp8", kv_pad * TXT_Q_DIM, FP8, True)
        self._alloc("o_fp4", kv_pad * TXT_Q_DIM // 2, FP8)
        self._alloc("o_fp4_sf", int(np.ceil(kv_pad / 128)) * 512 *
                    ((TXT_Q_DIM // 16 + 3) // 4), FP8)
        self._alloc("fp8_one", 1, FP32)
        self.bufs["fp8_one"].upload(np.ones(1, np.float32))
        # Text attention route: the repository SM120 FP8 causal GQA kernel
        # with a fused NVFP4 o-proj operand (default) or the vendored bf16
        # FA2 path (FLASHRT_MOLMO_FP8_ATTN=0).
        self.fp8_attn = os.environ.get(
            "FLASHRT_MOLMO_FP8_ATTN", "1") == "1"
        # 8-wide vectorized SwiGLU->NVFP4 producer (byte-identical output;
        # 1.6x the scalar merged kernel at the 692x9728 text shape).
        self.silu_vec = os.environ.get(
            "FLASHRT_MOLMO_SILU_VEC", "1") == "1"
        # Vision-tower attention route: "fa2" (delivered) or "native"
        # (72-deep BF16 kernel; measured 1.05x slower at the production shape).
        # Vision w1: NVFP4 GEMM with the bias+GELU->static-FP8 epilogue
        # (bytes identical to GEMM + molmo_bias_gelu_quant_fp8_vec).
        # Action expert: the skinny out-projection gate/residual launch also
        # emits the next modulated RMSNorm's FP8 operand (bit-identical).
        self.act_resnorm = os.environ.get(
            "FLASHRT_MOLMO_ACT_RESNORM", "1") == "1"
        self._pending_norm = None
        self.skip_bf16_qkv = os.environ.get(
            "FLASHRT_MOLMO_SKIP_BF16_QKV", "1") == "1"
        self.kv_only_tail = os.environ.get(
            "FLASHRT_MOLMO_KV_ONLY_TAIL", "1") == "1"
        self.ctx_batched = os.environ.get(
            "FLASHRT_MOLMO_CTX_BATCHED", "1") == "1"
        self.ctx_group = int(os.environ.get("FLASHRT_MOLMO_CTX_GROUP", "2"))
        self.vis_res_norm = os.environ.get(
            "FLASHRT_MOLMO_VIS_RESNORM", "1") == "1"
        self.vis_gelu_fp8out = os.environ.get(
            "FLASHRT_MOLMO_VIS_GELU_FP8OUT", "1") == "1"
        self.vis_gelu_inv = {}
        # Vision attention: "f16pv" (default) runs each KV tile's P*V in fp16
        # with an fp16 accumulator (2x the f32-accumulate MMA rate on SM120
        # GeForce) folded into the fp32 running output; "native" is the
        # all-bf16/fp32-accumulate kernel (needed for the fused FP8 output).
        self.vit_attn = os.environ.get(
            "FLASHRT_MOLMO_VIT_ATTN", "f16pv").strip().lower()
        # Persistent-CTA vision attention (FLASHRT_MOLMO_VIT_ATTN=persist):
        # a fixed CTA count walks the (view, head, q-tile) work items. Same
        # math and per-item tile order as the per-item grid (bit-identical),
        # measured slower; kept as an opt-in schedule experiment.
        self.vit_persist_blocks = int(os.environ.get(
            "FLASHRT_MOLMO_VIT_PERSIST_BLOCKS", "72"))
        # Text attention tile: 0 = 128x128/8-warp (delivered), 1 = 64x64/4-warp
        # with two CTAs per SM (98.3 vs 114.7 us/layer at L=768).
        self.attn_q64 = os.environ.get(
            "FLASHRT_MOLMO_ATTN_Q64", "1") == "1"
        # Action cross-attention split budget. The FA2 scheduler targets
        # num_sms*2 blocks: at the inherited 72 it picks 8 KV splits for the
        # 30x692x8x96 shape and the fp32 partial buffer (8*8*30*96) dominates
        # the call (14.3 us). 18 -> 4 splits measures 12.3 us; 3 splits is
        # the same, 8 splits is slower.
        self.xattn_sms = int(os.environ.get(
            "FLASHRT_MOLMO_XATTN_SMS", "18"))
        # Experimental small-M NVFP4 o-projection route (hand-written
        # 96x128 tile); off by default, measured slower than CUTLASS.
        self.smallm_gemm = os.environ.get(
            "FLASHRT_MOLMO_SMALLM_GEMM", "0") == "1"
        # Text attention with two producer warps (V transpose off the
        # compute warps). Bit-identical; measured ~5% slower.
        self.attn_wsp = os.environ.get(
            "FLASHRT_MOLMO_ATTN_WSP", "0") == "1"
        # Last-CTA fused K-split reduction + gate/residual for the skinny
        # action output projections (bit-identical; measured slower).
        self.fused_red = os.environ.get(
            "FLASHRT_MOLMO_FUSED_RED", "0") == "1"
        # Action expert as one persistent kernel per flow step (all layers,
        # grid barriers between phases, L2 prefetch of the next layer);
        # "0" keeps the multi-launch route.
        self.act_layer_kernel = os.environ.get(
            "FLASHRT_MOLMO_ACTION_MK", "1") == "1"
        self.act_mk = None
        self._alloc("V_cache", TXT_LAYERS * S * TXT_KV_DIM, BF16)
        self._alloc("O", S * TXT_Q_DIM, BF16)
        self._alloc("merged", S * (2 * TXT_FFN), BF16)
        self._alloc("act", S * TXT_FFN, BF16)
        # Per-layer action context K/V at a kv_pad row stride, so the shared
        # context projection runs as one GEMM over all layers' FP8 caches.
        self._alloc("ctxK", TXT_LAYERS * kv_pad * ACT_KV_DIM, BF16)
        self._alloc("ctxV", TXT_LAYERS * kv_pad * ACT_KV_DIM, BF16)
        self._alloc("ctx_tmp", TXT_LAYERS * kv_pad * ACT_D, BF16)
        self._alloc("ctx_flat", TXT_LAYERS * kv_pad * ACT_D, BF16)
        self._alloc("lse_txt", 1 * TXT_NH * S, FP32)

        # ── vision ──
        # zero pad columns are written once here and never touched again
        self._alloc("patches", TOTAL_PATCHES * PATCH_DIM_PAD, BF16, zero=True)
        self._alloc("vx", self.vis_pad * VIS_D, BF16)
        self._alloc("vtmp", self.vis_pad * VIS_D, BF16)
        self._alloc("vqkv", TOTAL_PATCHES * 3 * VIS_D, BF16)
        self._alloc("vQ", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("vK", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("vV", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("vO", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("vmlp", TOTAL_PATCHES * self.vis_ffn, BF16)
        # fused NVFP4 GELU-epilogue vision MLP: packed FP4 operand +
        # swizzled per-16 output block scales (consumed by the w2 GEMM)
        self._alloc("vis_ff4", VIS_PAD * VIS_FFN_PAD // 2, FP8)
        self._alloc("vis_ff4_sf", int(np.ceil(VIS_PAD / 128)) * 512 *
                    ((VIS_FFN_PAD // 16 + 3) // 4), FP8)
        self._alloc("feat24", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("feat18", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("concat", TOTAL_PATCHES * VIS_FEAT_D, BF16)
        self._alloc("pool_mean", TOTAL_POOLED * VIS_FEAT_D, BF16)
        self._alloc("pool_q", TOTAL_POOLED * VIS_D, BF16)
        self._alloc("pool_k", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("pool_v", TOTAL_PATCHES * VIS_D, BF16)
        self._alloc("pool_attn", TOTAL_POOLED * VIS_D, BF16)
        self._alloc("pool_out", TOTAL_POOLED * VIS_D, BF16)
        self._alloc("proj1", TOTAL_POOLED * TXT_FFN, BF16)
        self._alloc("proj3", TOTAL_POOLED * TXT_FFN, BF16)
        self._alloc("proj", TOTAL_POOLED * TXT_D, BF16)
        self._alloc("lse_vis", NUM_VIEWS * VIS_NH * POS_TOKENS, FP32)
        # ── action expert ──
        self._alloc("traj", self.horizon * ACT_MAX_DIM, BF16)
        self._alloc("vel", self.horizon * ACT_MAX_DIM, BF16)
        self._alloc("ax", self.horizon * ACT_D, BF16)
        self._alloc("axn", self.horizon * ACT_D, BF16)
        self._alloc("aqkv", self.horizon * 3 * ACT_D, BF16)
        self._alloc("aQ", self.horizon * ACT_D, BF16)
        self._alloc("aK", self.horizon * ACT_D, BF16)
        self._alloc("aV", self.horizon * ACT_D, BF16)
        self._alloc("aO", self.horizon * ACT_D, BF16)
        self._alloc("atmp", self.horizon * ACT_D, BF16)
        self._alloc("aup", self.horizon * ACT_H, BF16)
        self._alloc("agate", self.horizon * ACT_H, BF16)
        self._alloc("aact", self.horizon * ACT_H, BF16)
        self._alloc("acts_out", self.horizon * ACT_MAX_DIM, BF16)
        self._alloc("actions", self.horizon * self.act_dim, FP32)
        self._alloc("lse_act", ACT_NH * self.horizon, FP32)
        # FA2 split-KV scratch for the action cross-attention (<= 8 splits)
        self._alloc("lse_act_split", 8 * ACT_NH * self.horizon, FP32)
        self._alloc("o_accum", 8 * ACT_NH * self.horizon * ACT_HD, FP32)
        self._alloc("seqused_k", 1, INT64)
        self.bufs["seqused_k"].upload(np.array([self.S], np.int32))
        # bf16 ones: gate-broadcast neutral element for fused residual adds
        self._alloc("vis_o_fp8", TOTAL_PATCHES * VIS_D, FP8)
        self._alloc("concat_fp8", TOTAL_PATCHES * VIS_FEAT_D, FP8)
        self._alloc("mean_fp8", TOTAL_POOLED * VIS_D, FP8)
        self._alloc("attn_fp8", TOTAL_POOLED * VIS_D, FP8)
        self._alloc("p1_fp8", TOTAL_POOLED * TXT_FFN, FP8)
        self._alloc("out_fp8", TOTAL_POOLED * VIS_D, FP8)
        self._alloc("vis_ones", VIS_D, BF16)
        # skinny NVFP4 action-expert GEMMs: K-split FP32 partials (two live
        # sets for the concurrent up/gate projections) + per-site a_gs.
        sk = self.weights.get("skinny", {})
        sk_max = max((v[4] * self.horizon * v[5] for v in sk.values()),
                     default=16)
        self._alloc("skinny_counters", 32, INT32, zero=True)
        self._alloc("skinny_partials", sk_max, FP32)
        self._alloc("skinny_partials2", sk_max, FP32)
        self._alloc("skinny_ags", ACT_LAYERS * 7 * ACT_STEPS, FP32)
        self.bufs["vis_ones"].upload(np.full(VIS_D, 0x3F80, np.uint16))
        # fp8 activation scratch (largest activation: S x 19456)
        self._alloc("fp8_act", max(S * (2 * TXT_FFN), TOTAL_POOLED * TXT_FFN,
                                   TOTAL_PATCHES * self.vis_ffn,
                                   self.horizon * ACT_H), FP8)
        self._alloc("fp8_act_scale", 1, FP32)
        # NVFP4 activation scratch: packed nibbles + block scales + unit
        # global scale. Sized for the largest eligible activation.
        nv_n = max(S * TXT_D, S * TXT_Q_DIM, S * (2 * TXT_FFN),
                   TOTAL_PATCHES * VIS_D, TOTAL_PATCHES * self.vis_ffn,
                   TOTAL_POOLED * VIS_FEAT_D, TOTAL_POOLED * TXT_FFN,
                   self.horizon * ACT_H)
        self._alloc("nv_act", max(nv_n // 2, 16), FP8)
        self._alloc("nv_sf", max(nv_n // 16, 16), FP8)
        self._alloc("nv_one", 1, FP32)
        self.bufs["nv_one"].upload(np.ones(1, np.float32))
        # SwiGLU epilogue fold (text FFN): the fold GEMM runs whole 128-row
        # tiles and writes its SFD while the ff1 operand's SF is still being
        # read, so it gets its own SF buffer; its full-width output reuses
        # the (then unused) bf16 merged buffer.
        self.swiglu_fold = dict(self.weights.get("swiglu_fold", {}))
        # Vectorized FP8-only text q/k norm + RoPE producer (see
        # molmo_qkv_norm_rope_fp8_hd128); 0 keeps the generic kernel.
        self.qkv_vec = os.environ.get("FLASHRT_MOLMO_QKV_VEC", "1") == "1"
        self.swiglu_ceil_sf = os.environ.get(
            "FLASHRT_MOLMO_SWIGLU_CEIL_SF", "1") == "1"
        if self.swiglu_fold:
            self.S_pad128 = (S + 127) // 128 * 128
            self._alloc("nv_sf_ff2", self.fvk.nvfp4_sf_swizzled_bytes(
                self.S_pad128, TXT_FFN), FP8)
        self._alloc("gemm_f32", max(S * (2 * TXT_FFN),
                                    TOTAL_PATCHES * self.vis_ffn,
                                    TOTAL_PATCHES * 3 * VIS_D,
                                    TOTAL_POOLED * TXT_FFN,
                                    TOTAL_PATCHES * VIS_D,
                                    self.horizon * ACT_H), FP32)
        return B

    def _build_rope_tables(self):
        S = self.S
        inv = 1.0 / (TXT_ROPE_THETA ** (np.arange(0, TXT_HD // 2, dtype=np.float32)
                                        / (TXT_HD // 2)))
        pos = np.arange(S, dtype=np.float32)
        freqs = np.outer(pos, inv)                     # [S, HD/2]
        cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
        sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)

        def _bf16_bytes(a):
            import ml_dtypes
            return np.ascontiguousarray(a.astype(ml_dtypes.bfloat16))

        self.bufs["txt_cos"] = CudaBuffer.from_numpy(_bf16_bytes(cos))
        self.bufs["txt_sin"] = CudaBuffer.from_numpy(_bf16_bytes(sin))

        half = ACT_HD // 2
        inv_a = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32)
                                   / max(half, 1)))
        freqs_a = np.outer(np.arange(self.horizon, dtype=np.float32), inv_a)
        ca = np.cos(freqs_a)
        sa = np.sin(freqs_a)
        cos_a = np.concatenate([ca, ca], axis=-1)
        sin_a = np.concatenate([sa, sa], axis=-1)
        self.bufs["act_cos"] = CudaBuffer.from_numpy(_bf16_bytes(cos_a))
        self.bufs["act_sin"] = CudaBuffer.from_numpy(_bf16_bytes(sin_a))

    def _upload_precomputed(self):
        """Upload frontend-precomputed static tensors.

        Expected ``weights['precomputed']`` (real bf16 bit pattern, uint16):
          pos_emb_expanded [TOTAL_PATCHES, VIS_D]
          pool_idx         [TOTAL_POOLED, POOL_WINDOW] int32 (-1 masked)
          mod_self/mod_cross/mod_ffn [num_steps, ACT_LAYERS, 3, ACT_D]
              (scale, shift, gate per section, per step/layer)
          mod_final        [num_steps, 2, ACT_D] (scale, shift)
          ones_act         [ACT_D], ones_head [ACT_HD]
          img_row_start    [NUM_VIEWS] int64  (first <image_patch> row)
          noise            [horizon, ACT_MAX_DIM]
        """
        pre = self.weights["precomputed"]
        for name in ("pos_emb_expanded", "pool_idx", "mod_self", "mod_cross",
                     "mod_ffn", "mod_final", "noise", "state_q01", "state_q99",
                     "act_q01", "act_q99"):
            arr = np.ascontiguousarray(pre[name])
            self.bufs[name] = CudaBuffer.from_numpy(arr)
        for name in ("state_mask", "state_zero_mask", "act_mask"):
            if pre.get(name) is not None:
                self.bufs[name] = CudaBuffer.from_numpy(
                    np.ascontiguousarray(pre[name]))
        self.bufs["state_in"] = CudaBuffer.empty(int(pre["state_dim"]), FP32)
        self.bufs["ones_act"] = CudaBuffer.from_numpy(
            np.ones(ACT_D, dtype=np.float16))
        self.bufs["ones_head"] = CudaBuffer.from_numpy(
            np.ones(ACT_HD, dtype=np.float16))

    # ══════════════════════════════════════════════════════════════════
    #  GEMM / attention helpers
    # ══════════════════════════════════════════════════════════════════
    def _scale_name(self, name: str) -> str:
        if self._current_step >= 0 and name.startswith("act_"):
            return f"{name}__s{self._current_step}"
        return name

    def _scale_buf(self, name: str) -> CudaBuffer:
        key = self._scale_name(name)
        buf = self.fp8_scales.get(key)
        if buf is None:
            buf = CudaBuffer.device_zeros(1, FP32)
            self.fp8_scales[key] = buf
        return buf

    def _gemm(self, name: str, A: int, out: int, M: int, N: int, K: int,
              stream: int, act_n: int | None = None,
              scale_key: str | None = None, residual: int = 0,
              fuse_residual: bool = False) -> None:
        """Dispatch one weight GEMM in the configured precision.

        ``act_n`` is the number of activation elements to quantize (M*K) and
        defaults to ``M*K``.  ``scale_key`` lets two call sites share a weight
        but keep independent activation scales (the action expert applies one
        shared context projection to 36 different K/V layers).
        """
        if name in self.weights.get("nvfp4", {}):
            # Production NVFP4 prefill route (CUTLASS SM120 W4A4): dynamic
            # per-16 activation block scales (global scale 1), per-tensor
            # weight alpha, fused bias and optional residual epilogue.
            w_ptr, sf_ptr, alpha_ptr, bias_ptr = self.weights["nvfp4"][name]
            aq = _p(self.bufs["nv_act"])
            asf = _p(self.bufs["nv_sf"])
            self.fvk.nvfp4_quant(
                A, K, aq, asf, _p(self.bufs["nv_one"]), M, K, stream=stream)
            sched = 1 if K >= 6144 else 0
            # 128-deep K tile for K % 256 == 128 (see _gemm_nvfp4_pre)
            gemm = (self.fvk.nvfp4_gemm_bias_res_k128_bf16out
                    if K % 256 == 128 else
                    self.fvk.nvfp4_gemm_bias_res_bf16out)
            if fuse_residual and residual:
                # D = alpha*A@B + bias + C in one epilogue (C read, D written;
                # the caller ping-pongs the residual stream so D != C).
                gemm(aq, w_ptr, asf, sf_ptr, bias_ptr, residual, out,
                     M, N, K, alpha_ptr, 1.0, sched, stream=stream)
                return
            gemm(aq, w_ptr, asf, sf_ptr, bias_ptr, 0, out,
                 M, N, K, alpha_ptr, 0.0, sched, stream=stream)
            if residual:
                self.fvk.residual_add(residual, out, M * N, stream=stream)
            return
        if self.use_fp8 and name in self.weights["fp8"]:
            n = int(act_n if act_n is not None else M * K)
            w_ptr, w_scale = self.weights["fp8"][name]
            act_fp8 = _p(self.bufs["fp8_act"])
            scale = self._scale_buf(scale_key or name)
            if self.fp8_calibrated:
                self.fvk.quantize_fp8_static(
                    A, act_fp8, scale.ptr.value, n, stream=stream)
                self.gemm.fp8_nn_dev(
                    act_fp8, w_ptr, out, M, N, K, scale.ptr.value,
                    w_scale, stream=stream)
            else:
                dyn = _p(self.bufs["fp8_act_scale"])
                self.fvk.quantize_fp8_device(
                    A, act_fp8, dyn, n, stream=stream)
                self.fvk.fp8_accumulate_scale_max(
                    dyn, scale.ptr.value, stream=stream)
                self.gemm.fp8_nn_dev(
                    act_fp8, w_ptr, out, M, N, K, dyn, w_scale,
                    stream=stream)
        else:
            self.gemm.bf16_nn(A, self.weights["bf16"][name], out, M, N, K,
                              stream=stream)
        if residual:
            self.fvk.residual_add(residual, out, M * N, stream=stream)

    def _gemm_bias(self, name: str, A: int, out: int, M: int, N: int, K: int,
                   stream: int) -> None:
        """GEMM with the site bias (fused in the NVFP4 epilogue, split for
        FP8 / BF16)."""
        if name in self.weights.get("nvfp4", {}):
            # The NVFP4 epilogue already applies the site bias.
            self._gemm(name, A, out, M, N, K, stream)
            return
        bias = self.weights["bias"][name]
        if self.use_fp8 and name in self.weights["fp8"]:
            self._gemm(name, A, out, M, N, K, stream)
            self.fvk.molmo_add_bias_bf16(out, bias, M, N, stream=stream)
        else:
            acc = _p(self.bufs["gemm_f32"])
            self.gemm.bf16_nn_fp32(
                A, self.weights["bf16"][name], acc, M, N, K, stream=stream)
            self.fvk.molmo_bias_round_bf16(acc, bias, out, M, N,
                                           stream=stream)

    def _gemm_fp8_pre(self, name: str, A_fp8: int, out: int, M: int, N: int,
                      K: int, scale_ptr: int, stream: int) -> None:
        """FP8 GEMM over an activation already emitted as static-scale FP8."""
        w_ptr, w_scale = self.weights["fp8"][name]
        self.gemm.fp8_nn_dev(A_fp8, w_ptr, out, M, N, K, scale_ptr, w_scale,
                             stream=stream)

    def _norm_quant(self, *args):
        """nvfp4_norm_quant with byte-identical warp-per-row kernels: text rows
        (2560) split over four warps, vision rows (1152) one warp per row."""
        dim = args[9]
        rc = self.fvk.nvfp4_norm_quant_warp(
            *args, mw_wpr=4 if dim == TXT_D else 0)
        if rc != 0:
            raise RuntimeError(f"nvfp4_norm_quant_warp rc={rc}")

    def _gemm_nvfp4_pre(self, name: str, A_fp4: int, A_sf: int, out: int,
                        M: int, N: int, K: int, stream: int,
                        residual: int = 0,
                        fuse_residual: bool = False) -> None:
        """NVFP4 GEMM over a pre-quantized (swizzled) activation operand."""
        w_ptr, sf_ptr, alpha_ptr, bias_ptr = self.weights["nvfp4"][name]
        if self.smallm_gemm and name in self.weights.get("smallm", {}):
            bq, bsf = self.weights["smallm"][name]
            beta = 1.0 if (fuse_residual and residual) else 0.0
            self.fvk.molmo_smallm_gemm_bias_res_bf16out(
                A_fp4, A_sf, bq, bsf, bias_ptr, residual if beta else 0,
                out, M, N, K, alpha_ptr, beta, stream=stream)
            return
        sched = 1 if K >= 6144 else 0
        # A 256-deep K tile rounds K=1152 up to five iterations, so the last
        # one is half empty. The 128-deep variant executes the same
        # arithmetic (bit-identical outputs) in nine full iterations and
        # measured 6-8% faster at exactly those shapes (vision qkv / w1);
        # the wider tile stays better for K that is already 256-aligned.
        gemm = (self.fvk.nvfp4_gemm_bias_res_k128_bf16out
                if K % 256 == 128 else
                self.fvk.nvfp4_gemm_bias_res_bf16out)
        if fuse_residual and residual:
            gemm(
                A_fp4, w_ptr, A_sf, sf_ptr, bias_ptr, residual, out, M, N,
                K, alpha_ptr, 1.0, sched, stream=stream)
            return
        gemm(
            A_fp4, w_ptr, A_sf, sf_ptr, bias_ptr, 0, out, M, N, K,
            alpha_ptr, 0.0, sched, stream=stream)
        if residual:
            self.fvk.residual_add(residual, out, M * N, stream=stream)

    def _act_gemm(self, name: str, A: int, fp8_pre: bool, fp8_scale: int,
                  out: int, N: int, K: int, stream: int) -> bool:
        """One action-expert GEMM; returns True when the site bias was applied."""
        if fp8_pre:
            self._gemm_fp8_pre(name, A, out, self.horizon, N, K, fp8_scale,
                               stream)
            return False
        self._gemm_bias(name, A, out, self.horizon, N, K, stream)
        return True

    def _act_mod_norm(self, x: int, scale_ptr: int, shift_ptr: int,
                      stream: int, quant_name: str):
        """Produce the action block's modulated norm.

        Returns ``(ptr, fp8_scale_ptr, is_fp8)``: the fused FP8 producer is
        used when the downstream GEMM is a calibrated FP8 site.
        """
        # ACT_D is 768: bit-exact warp-per-row replicas of both reference
        # kernels (same virtual-warp reduction grouping, 2x faster).
        if (self.use_fp8 and self.fp8_calibrated
                and quant_name in self.weights["fp8"]):
            sc = self._scale_buf(quant_name)
            self.fvk.molmo_mod_rms_norm_rows768_quant_fp8_exact(
                x, scale_ptr, shift_ptr, _p(self.bufs["fp8_act"]),
                sc.ptr.value, self.horizon, EPS, stream=stream)
            return _p(self.bufs["fp8_act"]), sc.ptr.value, True
        self.fvk.molmo_mod_rms_norm_rows768_exact(
            x, scale_ptr, shift_ptr, _p(self.bufs["axn"]), self.horizon,
            EPS, stream=stream)
        return _p(self.bufs["axn"]), 0, False

    def _act_bias(self, name: str) -> int:
        return self.weights["bias"][name]

    # ── skinny NVFP4 action-expert GEMMs ────────────────────────────────
    # ── fused action SwiGLU producer (up+gate+silu, one launch) ──────────
    def _act_fused_ok(self, su: str, sg: str) -> bool:
        nk = self.weights.get("fp8_nk", {})
        return bool(self.weights.get("act_fused", False)
                    and su in nk and sg in nk
                    and su in self.weights.get("fp8", {})
                    and sg in self.weights.get("fp8", {}))

    def _act_fused_alphas(self, su: str, sg: str) -> tuple[float, float]:
        step = self._current_step
        nk = self.weights["fp8_nk"]
        au = self.fp8_scale_host.get(f"{su}__s{step}")
        ag = self.fp8_scale_host.get(f"{sg}__s{step}")
        if au is None or ag is None:
            raise RuntimeError(
                f"missing fused FFN activation scale for {su}/{sg} step {step}")
        return (au * nk[su][2], ag * nk[sg][2])

    def _skinny_ok(self, name: str) -> bool:
        if self.skinny_scope == "off":
            return False
        if self.skinny_scope == "attn":
            return name.startswith(("act_qkv_", "act_so_", "act_cq_",
                                    "act_co_"))
        if self.skinny_scope == "outd":
            return name.startswith(("act_so_", "act_co_", "act_down_"))
        if self.skinny_scope == "outfg":
            return name.startswith(("act_so_", "act_co_", "act_up_",
                                    "act_gate_"))
        if self.skinny_scope == "out":
            return name.startswith(("act_so_", "act_co_"))
        if self.skinny_scope == "ffn":
            return name.startswith(("act_up_", "act_gate_", "act_down_"))
        return True

    def _skinny_ags(self, name: str, step: int) -> int:
        return (self.bufs["skinny_ags"].ptr.value
                + 4 * self.skinny_ags_off[(name, step)])

    def _skinny_gemm(self, name: str, A: int, step: int, stream: int,
                     buf: str = "skinny_partials") -> int:
        w_pack, sf_pack, w_gs, cfg, splits, N, K = self.weights["skinny"][
            name]
        self.fvk.nvfp4_skinny_gemm_bf16_act(
            A, self._skinny_ags(name, step), w_pack, sf_pack,
            _p(self.bufs[buf]), self.horizon, N, K, cfg, True, stream)
        return splits

    def _act_norm_fusable(self, quant_name: str) -> bool:
        """Whether the next modulated norm can ride the gate/residual launch."""
        return (self.act_resnorm and self.use_fp8 and self.fp8_calibrated
                and not self.fused_red
                and quant_name in self.weights["fp8"])

    def _skinny_gate_res(self, site: str, A: int, x: int, gate: int,
                         step: int, stream: int, norm=None):
        """Skinny out-projection + gate/residual, fused or two-launch.

        With ``norm = (scale_ptr, shift_ptr, quant_name)`` the following
        modulated RMSNorm -> static FP8 producer runs in the same launch
        (bit-identical) and its ``_act_mod_norm`` result is returned.
        """
        w_pack, sf_pack, w_gs, cfg, splits, N, K = self.weights["skinny"][
            site]
        ags = self._skinny_ags(site, step)
        if norm is not None and self._act_norm_fusable(norm[2]):
            splits = self._skinny_gemm(site, A, step, stream)
            sc = self._scale_buf(norm[2])
            rc = self.fvk.molmo_skinny_gate_residual_mod_rms768_quant_fp8(
                _p(self.bufs["skinny_partials"]), splits, ags, w_gs,
                self._act_bias(site), x, gate, norm[0], norm[1],
                _p(self.bufs["fp8_act"]), sc.ptr.value, self.horizon, EPS,
                stream=stream)
            if rc != 0:
                raise RuntimeError(f"gate_residual_mod_rms rc={rc}")
            return _p(self.bufs["fp8_act"]), sc.ptr.value, True
        if self.fused_red:
            self.fvk.nvfp4_skinny_gemm_gate_residual_fused(
                A, ags, w_pack, sf_pack, _p(self.bufs["skinny_partials"]),
                ags, w_gs, self._act_bias(site), x, gate,
                _p(self.bufs["skinny_counters"]), self.horizon, N, K, cfg,
                stream=stream)
            return
        splits = self._skinny_gemm(site, A, step, stream)
        self.fvk.molmo_skinny_gate_residual(
            _p(self.bufs["skinny_partials"]), splits, ags, w_gs,
            self._act_bias(site), x, gate, self.horizon, N, stream=stream)
        return None

    def prepare_scale_host(self) -> None:
        """Cache every calibrated static scale on the host (setup only)."""
        for key, buf in self.fp8_scales.items():
            self.fp8_scale_host[key] = float(
                buf.download_new((1,), np.float32)[0])

        def eq(a, b):
            return (a in self.fp8_scale_host and b in self.fp8_scale_host
                    and self.fp8_scale_host[a] == self.fp8_scale_host[b])

        self.shared_scale_ok["pool_kv"] = eq("pool_k", "pool_v")
        self.shared_scale_ok["proj_w13"] = eq("proj_w1", "proj_w3")

    def prepare_vis_gelu_inv_scales(self) -> None:
        """Freeze 1/scale for the fused NVFP4 w1 -> GELU -> FP8 epilogue.

        Computed on device by the same expression the FP8 producers use, so
        the fused epilogue emits identical bytes. Runs once after
        calibration, before graph capture.
        """
        self.vis_gelu_inv = {}
        if not self.vis_gelu_fp8out:
            return
        nv = self.weights.get("nvfp4", {})
        for i in range(VIS_LAYERS):
            w1, w2 = f"vis_w1_{i}", f"vis_w2_{i}"
            if w1 not in nv or w2 not in self.weights["fp8"]:
                continue
            buf = CudaBuffer.device_zeros(1, FP32)
            rc = self.fvk.molmo_reciprocal_f32(
                self._scale_buf(w2).ptr.value, buf.ptr.value, 1, 0)
            if rc != 0:
                raise RuntimeError(f"molmo_reciprocal_f32 rc={rc}")
            self.vis_gelu_inv[i] = buf

    def prepare_skinny_scales(self) -> None:
        """Freeze each action site's static NVFP4 activation global scale.

        The calibrated FP8 activation scale is amax/448; the NVFP4 global
        scale is amax/(6*448), so a_gs = fp8_scale / 6.  Runs once after
        real-data calibration, before graph capture.
        """
        sk = self.weights.get("skinny", {})
        self.skinny_ags_off = {}
        if not sk:
            return
        entries = [(name, step) for name in sorted(sk)
                   for step in range(self.num_steps)]
        vals = np.zeros(len(entries), np.float32)
        for idx, (name, step) in enumerate(entries):
            buf = self.fp8_scales.get(f"{name}__s{step}")
            if buf is None:
                raise RuntimeError(
                    f"missing calibrated scale for skinny site {name} s{step}")
            vals[idx] = float(buf.download_new((1,), np.float32)[0]) / 6.0
            self.skinny_ags_off[(name, step)] = idx
        self.bufs["skinny_ags"].upload(vals)

    def prepare_action_kernel(self) -> None:
        """Build the persistent action-expert kernel's static tables.

        Needs the delivered action precision (calibrated FP8 qkv / query /
        fused up-gate sites and skinny NVFP4 out / down sites); otherwise the
        multi-launch route stays in use. Runs once after calibration, before
        graph capture.
        """
        self.act_mk = None
        if not (self.act_layer_kernel and self.fp8_calibrated and self.use_fp8):
            return
        fp8 = self.weights.get("fp8", {})
        nk = self.weights.get("fp8_nk", {})
        sk = self.weights.get("skinny", {})
        for i in range(ACT_LAYERS):
            if not (f"act_qkv_{i}" in fp8 and f"act_cq_{i}" in fp8
                    and f"act_up_{i}" in nk and f"act_gate_{i}" in nk
                    and all(f"act_{s}_{i}" in sk and self._skinny_ok(f"act_{s}_{i}")
                            for s in ("so", "co", "down"))
                    and not self._skinny_ok(f"act_qkv_{i}")
                    and not self._skinny_ok(f"act_cq_{i}")):
                return
        if not self._ctx_batched_route():
            return   # the head-major context comes from the batched route
        hm = TXT_LAYERS * ACT_NH * self.kv_pad * ACT_MK_CTX_STRIDE
        for name in ("ctxK_hm", "ctxV_hm"):
            if name not in self.bufs:
                self._alloc(name, hm, BF16, True)
        from flash_rt.models.molmoact2.action_mk import ActionLayerKernel
        self.act_mk = ActionLayerKernel(self, layers=ACT_LAYERS, d=ACT_D,
                                        hid=ACT_H, eps=EPS,
                                        ctx_stride=ACT_MK_CTX_STRIDE)

    def _fa2_fp8out(self, Q, K, V, O, lse, O_fp8, scale_ptr, batch, sq, sk,
                    nq, nkv, hd, stream) -> None:
        """FA2 with a fused static-scale FP8 copy of the output.

        Uses the repository ``fwd_bf16_fp8out`` entry: ``scale_ptr`` is the
        device scalar holding the static activation scale, i.e. the same
        buffer ``quantize_fp8_static`` reads, so the emitted bytes match a
        standalone quantize. The bf16 O is dead on this route and is not
        stored.
        """
        qs = (sq * nq * hd, nq * hd, hd)
        ks = (sk * nkv * hd, nkv * hd, hd)
        self.fa2.fwd_bf16_fp8out(
            Q, K, V, O, lse, 0, 0, batch, sq, sk, nq, nkv, hd,
            qs, ks, ks, qs, O_fp8, scale_ptr, qs, True,
            1.0 / (hd ** 0.5), 0, stream)

    def _fa2_short(self, Q, K, V, O, lse, sq, sk, nq, nkv, hd,
                   stream) -> None:
        """Short-query tile FA2 entry (repository small-q BF16 kernel)."""
        qs = (sq * nq * hd, nq * hd, hd)
        ks = (sk * nkv * hd, nkv * hd, hd)
        self.fa2.fwd_bf16_tile(
            Q, K, V, O, lse, 0, 1, sq, sk, nq, nkv, hd, qs, ks, ks, qs,
            1.0 / (hd ** 0.5), 0, stream)

    def _fa2_vision_strided(self, qkv, O, lse, stream) -> None:
        """FA2 over the packed [rows, 3*VIS_D] qkv projection.

        Q/K/V are contiguous per head inside every row, so the kernel can read
        them with (batch, row, head) strides and the standalone split copy is
        not needed. Numerically identical to splitting first.
        """
        row = 3 * VIS_D
        bs = POS_TOKENS * row
        qs = (bs, row, VIS_HD)
        os_ = (POS_TOKENS * VIS_NH * VIS_HD, VIS_NH * VIS_HD, VIS_HD)
        if self.vit_attn == "native":
            self.fvk.molmo_vit_attn_bf16(
                qkv, qkv + VIS_D * 2, qkv + 2 * VIS_D * 2, O,
                NUM_VIEWS, POS_TOKENS, VIS_NH, VIS_HD, 3 * VIS_D,
                VIS_NH * VIS_HD, 1.0 / (VIS_HD ** 0.5), stream)
            return
        if self.vit_attn == "f16pv":
            self.fvk.molmo_vit_attn_f16pv(
                qkv, qkv + VIS_D * 2, qkv + 2 * VIS_D * 2, O,
                NUM_VIEWS, POS_TOKENS, VIS_NH, VIS_HD, 3 * VIS_D,
                VIS_NH * VIS_HD, 1.0 / (VIS_HD ** 0.5), stream)
            return
        if self.vit_attn == "persist":
            self.fvk.molmo_vit_attn_bf16_persist(
                qkv, qkv + VIS_D * 2, qkv + 2 * VIS_D * 2, O,
                NUM_VIEWS, POS_TOKENS, VIS_NH, VIS_HD, 3 * VIS_D,
                VIS_NH * VIS_HD, 1.0 / (VIS_HD ** 0.5),
                self.vit_persist_blocks, stream)
            return
        self.fa2.fwd_bf16(
            qkv, qkv + VIS_D * 2, qkv + 2 * VIS_D * 2, O, lse, 0, 0,
            NUM_VIEWS, POS_TOKENS, POS_TOKENS, VIS_NH, VIS_NH, VIS_HD,
            qs, qs, qs, os_, 1.0 / (VIS_HD ** 0.5), 0, stream)

    def _fa2(self, Q, K, V, O, lse, batch, sq, sk, nq, nkv, hd, causal,
             stream) -> None:
        qs = (sq * nq * hd, nq * hd, hd)
        ks = (sk * nkv * hd, nkv * hd, hd)
        vs = ks
        os_ = qs
        scale = 1.0 / (hd ** 0.5)
        if causal:
            self.fa2.fwd_bf16_causal(
                Q, K, V, O, lse, 0, 0, batch, sq, sk, nq, nkv, hd,
                qs, ks, vs, os_, scale, 0, stream)
        else:
            self.fa2.fwd_bf16(
                Q, K, V, O, lse, 0, 0, batch, sq, sk, nq, nkv, hd,
                qs, ks, vs, os_, scale, 0, stream)

    # ══════════════════════════════════════════════════════════════════
    #  Stage: image preprocessing + ViT + adapter
    # ══════════════════════════════════════════════════════════════════
    def image_preprocess(self, stream: int = 0) -> None:
        img_u8 = self.input_images_ptr
        out = _p(self.bufs["patches"])
        per_img = POS_TOKENS * PATCH_DIM_PAD * 2
        for v in range(NUM_VIEWS):
            self.fvk.molmo_resize_patchify_bf16(
                img_u8 + v * IMG_H * IMG_W * 3, out + v * per_img,
                IMG_H, IMG_W, 3, TILE, TILE, PATCH, stream=stream,
                out_stride=PATCH_DIM_PAD)

    def vision_encoder(self, stream: int = 0) -> None:
        fvk = self.fvk
        b = self.bufs
        P = TOTAL_PATCHES
        patches = _p(b["patches"])
        x = _p(b["vx"])
        tmp = _p(b["vtmp"])
        qkv = _p(b["vqkv"])
        Q, K, V = _p(b["vQ"]), _p(b["vK"]), _p(b["vV"])
        O = _p(b["vO"])
        mlp = _p(b["vmlp"])
        W = self.weights
        if ("vis_patch_embed" in self.weights["bf16"]
                and "vis_patch_embed" not in self.weights.get("nvfp4", {})
                and not (self.use_fp8 and "vis_patch_embed" in self.weights["fp8"])):
            # bf16 GEMM to fp32, then bias round + positional add in one pass
            acc = _p(self.bufs["gemm_f32"])
            self.gemm.bf16_nn_fp32(patches, self.weights["bf16"]["vis_patch_embed"],
                                   acc, P, VIS_D, PATCH_DIM_PAD, stream=stream)
            fvk.molmo_bias_round_add_bf16(
                acc, self.weights["bias"]["vis_patch_embed"],
                _p(b["pos_emb_expanded"]), x, P, VIS_D, stream)
        else:
            self._gemm_bias("vis_patch_embed", patches, x, P, VIS_D,
                            PATCH_DIM_PAD, stream)
            fvk.residual_add(x, _p(b["pos_emb_expanded"]), P * VIS_D,
                             stream=stream)

        # Deferred FP8 projection residual (out, bias): the next NVFP4 norm
        # producer applies x += bf16(out + bias) in place (bit-identical to
        # the gate/residual kernel with a unit gate) instead of a separate
        # residual pass.
        pend = None
        fuse_res = self.fp8_calibrated and self.vis_res_norm

        def flush():
            nonlocal pend
            if pend is not None:
                if pend[1]:
                    fvk.molmo_gate_residual_bias(
                        x, pend[0], _p(b["vis_ones"]), pend[1], P, VIS_D,
                        stream=stream)
                else:   # NVFP4 projection output (bias applied in the GEMM)
                    fvk.residual_add(x, pend[0], P * VIS_D, stream=stream)
                pend = None

        def defer_ok(i):
            return (fuse_res and i + 1 < VIS_LAYERS
                    and i != VIT_FEATURE_LAYERS[1]
                    and self._vit_layer_hook is None)

        for i in range(VIS_LAYERS):
            ln1w = W["bf16"][f"vis_ln1_w_{i}"]
            ln1b = W["bias"][f"vis_ln1_b_{i}"]
            site = f"vis_qkv_{i}"
            fp8_site = self.use_fp8 and site in self.weights["fp8"]
            qkv_packed = False
            if not (self.fp8_calibrated and not fp8_site
                    and site in self.weights.get("nvfp4", {})):
                flush()
            if self.fp8_calibrated and fp8_site:
                sc = self._scale_buf(site)
                fvk.molmo_layer_norm_quant_fp8(
                    x, ln1w, ln1b, _p(b["fp8_act"]), sc.ptr.value, P,
                    VIS_D, EPS, stream)
                self._gemm_fp8_pre(site, _p(b["fp8_act"]), qkv, P,
                                   3 * VIS_D, VIS_D, sc.ptr.value, stream)
                fvk.molmo_qkv_split_bias(
                    qkv, W["bias"][site], Q, K, V, P, VIS_D, VIS_D, VIS_D,
                    stream=stream)
            elif (self.fp8_calibrated and site in self.weights.get(
                    "nvfp4", {})):
                self._norm_quant(
                    x, ln1w, ln1b, 0, EPS, _p(b["nv_act"]),
                    _p(b["nv_sf"]), _p(b["nv_one"]), P, VIS_D, stream,
                    pend[0] if pend else 0, pend[1] if pend else 0)
                pend = None
                self._gemm_nvfp4_pre(site, _p(b["nv_act"]), _p(b["nv_sf"]),
                                     qkv, P, 3 * VIS_D, VIS_D, stream)
                qkv_packed = True
            elif site in self.weights.get("nvfp4", {}) or not fp8_site:
                fvk.layer_norm(x, ln1w, ln1b, tmp, P, VIS_D, EPS,
                               stream=stream)
                self._gemm_bias(site, tmp, qkv, P, 3 * VIS_D, VIS_D, stream)
                fvk.qkv_split(qkv, Q, K, V, P, VIS_D, VIS_D, VIS_D,
                              stream=stream)
            else:
                fvk.layer_norm(x, ln1w, ln1b, tmp, P, VIS_D, EPS,
                               stream=stream)
                self._gemm(site, tmp, qkv, P, 3 * VIS_D, VIS_D, stream)
                fvk.molmo_qkv_split_bias(
                    qkv, W["bias"][site], Q, K, V, P, VIS_D, VIS_D, VIS_D,
                    stream=stream)
            site = f"vis_o_{i}"
            fp8_site = self.use_fp8 and site in self.weights["fp8"]
            o_fp8 = (self.fp8_calibrated and fp8_site
                     and os.environ.get("FLASHRT_MOLMO_FA2_FP8OUT") == "1")
            if qkv_packed and o_fp8:
                # The FP8-output epilogue entry takes contiguous Q/K/V.
                fvk.molmo_qkv_split_vec(qkv, Q, K, V, P, VIS_D, VIS_D,
                                        VIS_D, stream)
                qkv_packed = False
            if o_fp8:
                # FA2 epilogue emits the o-proj FP8 operand directly.
                sc = self._scale_buf(site)
                self._fa2_fp8out(
                    Q, K, V, O, _p(b["lse_vis"]), _p(b["vis_o_fp8"]),
                    sc.ptr.value,
                    NUM_VIEWS, POS_TOKENS, POS_TOKENS, VIS_NH, VIS_NH,
                    VIS_HD, stream)
            elif (qkv_packed and self.vit_attn == "native"
                  and self.fp8_calibrated and fp8_site
                  and site not in self.weights.get("nvfp4", {})):
                # Native attention emits the o-proj static FP8 operand
                # directly (bytes identical to a separate quantize pass).
                sc = self._scale_buf(site)
                rc = fvk.molmo_vit_attn_bf16_fp8out(
                    qkv, qkv + VIS_D * 2, qkv + 2 * VIS_D * 2,
                    _p(b["vis_o_fp8"]), sc.ptr.value, NUM_VIEWS, POS_TOKENS,
                    VIS_NH, VIS_HD, 3 * VIS_D, VIS_NH * VIS_HD,
                    1.0 / (VIS_HD ** 0.5), stream)
                if rc != 0:
                    raise RuntimeError(f"molmo_vit_attn_bf16_fp8out rc={rc}")
                o_fp8 = True
            elif qkv_packed:
                self._fa2_vision_strided(qkv, O, _p(b["lse_vis"]), stream)
            else:
                self._fa2(Q, K, V, O, _p(b["lse_vis"]), NUM_VIEWS,
                          POS_TOKENS, POS_TOKENS, VIS_NH, VIS_NH, VIS_HD,
                          False, stream)
            o_mlp_nv = (f"vis_w1_{i}" in self.weights.get("nvfp4", {})
                        and f"vis_w2_{i}" in self.weights.get("nvfp4", {}))
            if site in self.weights.get("nvfp4", {}) or not fp8_site:
                self._gemm_bias(site, O, tmp, P, VIS_D, VIS_D, stream)
                if (fuse_res and site in self.weights.get("nvfp4", {})
                        and o_mlp_nv and os.environ.get(
                            "FLASHRT_MOLMO_VIS_NORMQUANT", "1") != "0"):
                    # the fused MLP's ln2 norm_quant adds it (bit-identical)
                    pend = (tmp, 0)
                else:
                    fvk.residual_add(x, tmp, P * VIS_D, stream=stream)
            elif o_fp8:
                self._gemm_fp8_pre(site, _p(b["vis_o_fp8"]), tmp, P, VIS_D,
                                   VIS_D, sc.ptr.value, stream)
                if fuse_res:
                    pend = (tmp, W["bias"][site])
                else:
                    fvk.molmo_gate_residual_bias(
                        x, tmp, _p(b["vis_ones"]), W["bias"][site], P,
                        VIS_D, stream=stream)
            else:
                self._gemm(site, O, tmp, P, VIS_D, VIS_D, stream)
                fvk.molmo_gate_residual_bias(
                    x, tmp, _p(b["vis_ones"]), W["bias"][site], P, VIS_D,
                    stream=stream)

            ln2w = W["bf16"][f"vis_ln2_w_{i}"]
            ln2b = W["bias"][f"vis_ln2_b_{i}"]
            site = f"vis_w1_{i}"
            w2_site = f"vis_w2_{i}"
            w1_nv = site in self.weights.get("nvfp4", {})
            w2_nv = w2_site in self.weights.get("nvfp4", {})
            fp8_site = self.use_fp8 and site in self.weights["fp8"]
            w2_fp8 = self.use_fp8 and w2_site in self.weights["fp8"]
            w2_done = False
            gelu_fp8_done = False
            w1_takes_pend = (
                not (w1_nv and w2_nv) and self.fp8_calibrated and w2_fp8
                and w1_nv and os.environ.get(
                    "FLASHRT_MOLMO_VIS_NORMQUANT", "1") != "0")
            mlp_nq = (w1_nv and w2_nv and os.environ.get(
                "FLASHRT_MOLMO_VIS_NORMQUANT", "1") != "0")
            if not (w1_takes_pend or mlp_nq):
                flush()
            # With the fused bias+GELU->FP8 producer the w1 GEMM emits the raw
            # projection (the producer applies the bias); otherwise the
            # GEMM+bias+GELU chain completes here.
            if w1_nv and w2_nv:
                # Repository fused NVFP4 GEMM with bias+GELU epilogue: the
                # w1 projection emits the packed FP4 w2 operand (padded to
                # the kernel's M%128 / N%32 tiling) and its swizzled SF.
                w_ptr, sf_ptr, alpha_ptr, bias_ptr = self.weights["nvfp4"][
                    site]
                if os.environ.get("FLASHRT_MOLMO_VIS_NORMQUANT",
                                  "1") != "0":
                    # Pending o-proj residual: x += out over the padded rows
                    # too (pad rows never feed real rows).
                    self._norm_quant(
                        x, ln2w, ln2b, 0, EPS, _p(b["nv_act"]),
                        _p(b["nv_sf"]), _p(b["nv_one"]), VIS_PAD, VIS_D,
                        stream, pend[0] if pend else 0, 0)
                    pend = None
                else:
                    flush()
                    fvk.layer_norm(x, ln2w, ln2b, _p(b["vtmp"]), VIS_PAD,
                                   VIS_D, EPS, stream=stream)
                    fvk.nvfp4_quant(
                        _p(b["vtmp"]), VIS_D, _p(b["nv_act"]),
                        _p(b["nv_sf"]), _p(b["nv_one"]), VIS_PAD, VIS_D,
                        stream=stream)
                fvk.nvfp4_gemm_bias_gelu_fp4out(
                    _p(b["nv_act"]), w_ptr, _p(b["nv_sf"]), sf_ptr, bias_ptr,
                    _p(b["vis_ff4"]), _p(b["vis_ff4_sf"]), VIS_PAD,
                    VIS_FFN_PAD, VIS_D, alpha_ptr, _p(b["nv_one"]),
                    stream=stream)
                self._gemm_nvfp4_pre(
                    w2_site, _p(b["vis_ff4"]), _p(b["vis_ff4_sf"]), tmp,
                    VIS_PAD, VIS_D, VIS_FFN_PAD, stream)
                nq = f"vis_qkv_{i + 1}"
                if (defer_ok(i) and nq in self.weights.get("nvfp4", {})
                        and not (self.use_fp8 and nq in self.weights["fp8"])):
                    pend = (tmp, 0)   # next layer's ln1 norm_quant adds it
                else:
                    fvk.residual_add(x, tmp, P * VIS_D, stream=stream)
                w2_done = True
            elif self.fp8_calibrated and w2_fp8 and site in self.weights.get(
                    "nvfp4", {}) and os.environ.get(
                        "FLASHRT_MOLMO_VIS_NORMQUANT", "1") != "0":
                self._norm_quant(
                    x, ln2w, ln2b, 0, EPS, _p(b["nv_act"]),
                    _p(b["nv_sf"]), _p(b["nv_one"]), P, VIS_D, stream,
                    pend[0] if pend else 0, pend[1] if pend else 0)
                pend = None
                inv = self.vis_gelu_inv.get(i)
                if inv is not None and VIS_D % 256 == 128:
                    w_ptr, sf_ptr, alpha_ptr, bias_ptr = self.weights[
                        "nvfp4"][site]
                    rc = fvk.nvfp4_gemm_bias_gelu_fp8out_k128(
                        _p(b["nv_act"]), w_ptr, _p(b["nv_sf"]), sf_ptr,
                        bias_ptr, _p(b["fp8_act"]), P, self.vis_ffn, VIS_D,
                        alpha_ptr, inv.ptr.value, stream)
                    if rc != 0:
                        raise RuntimeError(
                            f"nvfp4_gemm_bias_gelu_fp8out_k128 rc={rc}")
                    gelu_fp8_done = True
                else:
                    self._gemm_nvfp4_pre(site, _p(b["nv_act"]),
                                         _p(b["nv_sf"]), mlp, P,
                                         self.vis_ffn, VIS_D, stream)
            elif self.fp8_calibrated and w2_fp8 and fp8_site:
                fvk.layer_norm(x, ln2w, ln2b, tmp, P, VIS_D, EPS,
                               stream=stream)
                self._gemm(site, tmp, mlp, P, self.vis_ffn, VIS_D, stream)
            elif self.fp8_calibrated and w2_fp8 and site in self.weights.get(
                    "nvfp4", {}):
                fvk.layer_norm(x, ln2w, ln2b, tmp, P, VIS_D, EPS,
                               stream=stream)
                self._gemm(site, tmp, mlp, P, self.vis_ffn, VIS_D, stream)
            else:
                fvk.layer_norm(x, ln2w, ln2b, tmp, P, VIS_D, EPS,
                               stream=stream)
                if site in self.weights.get("nvfp4", {}) or not fp8_site:
                    self._gemm_bias(site, tmp, mlp, P, self.vis_ffn, VIS_D,
                                    stream)
                    fvk.gelu_inplace(mlp, P * self.vis_ffn, stream=stream)
                else:
                    self._gemm(site, tmp, mlp, P, self.vis_ffn, VIS_D, stream)
                    fvk.bias_gelu_bf16_strict(
                        mlp, W["bias"][site], P, self.vis_ffn, stream=stream)
            site = w2_site
            fp8_site = self.use_fp8 and site in self.weights["fp8"]
            if w2_done:
                pass
            elif self.fp8_calibrated and fp8_site:
                # bias + GELU emitted directly as static-scale FP8
                sc = self._scale_buf(site)
                if not gelu_fp8_done:
                    fvk.molmo_bias_gelu_quant_fp8_vec(
                        mlp, W["bias"].get(f"vis_w1_{i}", 0),
                        _p(b["fp8_act"]), sc.ptr.value, P, self.vis_ffn,
                        stream)
                self._gemm_fp8_pre(site, _p(b["fp8_act"]), tmp, P, VIS_D,
                                   self.vis_ffn, sc.ptr.value, stream)
                if (fuse_res and i + 1 < VIS_LAYERS
                        and i != VIT_FEATURE_LAYERS[1]
                        and self._vit_layer_hook is None):
                    pend = (tmp, W["bias"][site])
                else:
                    fvk.molmo_gate_residual_bias(
                        x, tmp, _p(b["vis_ones"]), W["bias"][site], P,
                        VIS_D, stream=stream)
            elif site in self.weights.get("nvfp4", {}) or not fp8_site:
                self._gemm_bias(site, mlp, tmp, P, VIS_D, self.vis_ffn, stream)
                fvk.residual_add(x, tmp, P * VIS_D, stream=stream)
            else:
                self._gemm(site, mlp, tmp, P, VIS_D, self.vis_ffn, stream)
                fvk.molmo_gate_residual_bias(
                    x, tmp, _p(b["vis_ones"]), W["bias"][site], P, VIS_D,
                    stream=stream)

            if self._vit_layer_hook is not None:
                self._vit_layer_hook(i, self.bufs["vx"])
            if i == VIT_FEATURE_LAYERS[1]:       # layer 18 -> second half
                if self._concat_direct():
                    # straight into the right half of the adapter input
                    self._memcpy2d_d2d(_p(b["concat"]) + VIS_D * 2,
                                       2 * VIS_D * 2, x, VIS_D * 2,
                                       VIS_D * 2, P, stream)
                else:
                    self._memcpy_d2d(_p(b["feat18"]), x, P * VIS_D * 2, stream)
            # Layer 24 is the last layer: its residual stream is already the
            # adapter's first half, so no copy is needed for it.

        flush()
        # feat24 == the live residual stream when the tap is the last layer.
        first = (x if VIT_FEATURE_LAYERS[0] == VIS_LAYERS - 1
                 else _p(b["feat24"]))
        if self.fp8_calibrated and "pool_k" in self.weights["fp8"]:
            fvk.molmo_concat2_quant_fp8(
                first, _p(b["feat18"]), _p(b["concat"]), P,
                VIS_D, VIS_D, stream, _p(b["concat_fp8"]),
                self._scale_buf("pool_k").ptr.value)
        elif self._concat_direct():
            self._memcpy2d_d2d(_p(b["concat"]), 2 * VIS_D * 2, first,
                               VIS_D * 2, VIS_D * 2, P, stream)
        else:
            fvk.concat2_bf16(first, _p(b["feat18"]),
                             _p(b["concat"]), P, VIS_D, VIS_D, stream=stream)

    def _concat_direct(self) -> bool:
        """The bf16 adapter input is assembled by two strided copies (layer
        18 into the right half when it is produced, the last layer into the
        left half) instead of a copy plus concat2_bf16; same bytes."""
        return not (self.fp8_calibrated and "pool_k" in self.weights["fp8"])

    def image_adapter(self, stream: int = 0) -> None:
        fvk = self.fvk
        b = self.bufs
        P, QP = TOTAL_PATCHES, TOTAL_POOLED
        concat = _p(b["concat"])
        idx = _p(b["pool_idx"])
        mean = _p(b["pool_mean"])
        q = _p(b["pool_q"])
        k = _p(b["pool_k"])
        v = _p(b["pool_v"])
        attn = _p(b["pool_attn"])
        out = _p(b["pool_out"])
        p1, p3 = _p(b["proj1"]), _p(b["proj3"])
        proj = _p(b["proj"])

        cal = (self.fp8_calibrated and "pool_q" in self.weights["fp8"]
               and "proj_w1" in self.weights["fp8"]
               and self.shared_scale_ok.get("pool_kv")
               and self.shared_scale_ok.get("proj_w13"))
        if cal:
            fvk.molmo_pool_mean_bf16(
                concat, idx, mean, QP, POOL_WINDOW, VIS_FEAT_D, stream,
                _p(b["mean_fp8"]), self._scale_buf("pool_q").ptr.value)
            for name, src_fp8, src, dst, rows in (
                    ("pool_q", _p(b["mean_fp8"]), mean, q, QP),
                    ("pool_k", _p(b["concat_fp8"]), concat, k, P),
                    ("pool_v", _p(b["concat_fp8"]), concat, v, P)):
                self._gemm_fp8_pre(
                    name, src_fp8, dst, rows, VIS_D, VIS_FEAT_D,
                    self._scale_buf("pool_k" if name != "pool_q" else
                                    name).ptr.value, stream)
                fvk.molmo_add_bias_bf16(
                    dst, self.weights["bias"][name], rows, VIS_D, stream)
            fvk.molmo_pool_attn_bf16(
                q, k, v, idx, attn, QP, POOL_WINDOW, VIS_NH, VIS_HD,
                1.0 / (VIS_HD ** 0.5), stream, _p(b["attn_fp8"]),
                self._scale_buf("pool_o").ptr.value)
            self._gemm_fp8_pre("pool_o", _p(b["attn_fp8"]), out, QP, VIS_D,
                               VIS_D, self._scale_buf("pool_o").ptr.value,
                               stream)
            fvk.molmo_add_bias_bf16(out, self.weights["bias"]["pool_o"], QP,
                                    VIS_D, stream)
            # w1/w3 share one quantized copy of the same activation.
            sc_w1 = self._scale_buf("proj_w1")
            fvk.quantize_fp8_static(out, _p(b["out_fp8"]), sc_w1.ptr.value,
                                    QP * VIS_D, stream=stream)
            self._gemm_fp8_pre("proj_w1", _p(b["out_fp8"]), p1, QP, TXT_FFN,
                               VIS_D, sc_w1.ptr.value, stream)
            self._gemm_fp8_pre("proj_w3", _p(b["out_fp8"]), p3, QP, TXT_FFN,
                               VIS_D, sc_w1.ptr.value, stream)
            fvk.gelu_inplace(p1, QP * TXT_FFN, stream=stream)
            fvk.molmo_mul_bf16(p1, p3, p1, QP * TXT_FFN, stream,
                               _p(b["p1_fp8"]),
                               self._scale_buf("proj_w2").ptr.value)
            self._gemm_fp8_pre("proj_w2", _p(b["p1_fp8"]), proj, QP, TXT_D,
                               TXT_FFN, self._scale_buf("proj_w2").ptr.value,
                               stream)
        else:
            fvk.molmo_pool_mean_bf16(concat, idx, mean, QP, POOL_WINDOW,
                                     VIS_FEAT_D, stream=stream)
            self._gemm_bias("pool_q", mean, q, QP, VIS_D, VIS_FEAT_D, stream)
            nv = self.weights.get("nvfp4", {})
            if "pool_k" in nv and "pool_v" in nv:
                # one NVFP4 quantization of the shared input for both
                fvk.nvfp4_quant(concat, VIS_FEAT_D, _p(b["nv_act"]),
                                _p(b["nv_sf"]), _p(b["nv_one"]), P, VIS_FEAT_D,
                                stream=stream)
                for name, dst in (("pool_k", k), ("pool_v", v)):
                    self._gemm_nvfp4_pre(name, _p(b["nv_act"]), _p(b["nv_sf"]),
                                         dst, P, VIS_D, VIS_FEAT_D, stream)
            else:
                self._gemm_bias("pool_k", concat, k, P, VIS_D, VIS_FEAT_D,
                                stream)
                self._gemm_bias("pool_v", concat, v, P, VIS_D, VIS_FEAT_D,
                                stream)
            fvk.molmo_pool_attn_bf16(q, k, v, idx, attn, QP, POOL_WINDOW,
                                     VIS_NH, VIS_HD, 1.0 / (VIS_HD ** 0.5),
                                     stream=stream)
            self._gemm_bias("pool_o", attn, out, QP, VIS_D, VIS_D, stream)
            if "proj_w1" in nv and "proj_w3" in nv:
                fvk.nvfp4_quant(out, VIS_D, _p(b["nv_act"]), _p(b["nv_sf"]),
                                _p(b["nv_one"]), QP, VIS_D, stream=stream)
                for name, dst in (("proj_w1", p1), ("proj_w3", p3)):
                    self._gemm_nvfp4_pre(name, _p(b["nv_act"]), _p(b["nv_sf"]),
                                         dst, QP, TXT_FFN, VIS_D, stream)
            else:
                self._gemm("proj_w1", out, p1, QP, TXT_FFN, VIS_D, stream)
                self._gemm("proj_w3", out, p3, QP, TXT_FFN, VIS_D, stream)
            if "proj_w2" in self.weights.get("nvfp4", {}):
                # gelu + product + NVFP4 quantization in one pass (same bytes
                # as gelu_inplace -> molmo_mul_bf16 -> nvfp4_quant)
                rc = fvk.nvfp4_gelu_mul_quant(
                    p1, p3, _p(b["nv_act"]), _p(b["nv_sf"]), _p(b["nv_one"]),
                    QP, TXT_FFN, stream)
                if rc != 0:
                    raise RuntimeError(f"nvfp4_gelu_mul_quant rc={rc}")
                self._gemm_nvfp4_pre("proj_w2", _p(b["nv_act"]), _p(b["nv_sf"]),
                                     proj, QP, TXT_D, TXT_FFN, stream)
            else:
                fvk.gelu_inplace(p1, QP * TXT_FFN, stream=stream)
                fvk.molmo_mul_bf16(p1, p3, p1, QP * TXT_FFN, stream=stream)
                self._gemm("proj_w2", p1, proj, QP, TXT_D, TXT_FFN, stream)

    # ══════════════════════════════════════════════════════════════════
    #  Stage: token embeddings
    # ══════════════════════════════════════════════════════════════════
    def build_embeddings(self, stream: int = 0) -> None:
        fvk = self.fvk
        b = self.bufs
        S = self.S
        ids = _p(b["ids"])
        emb = _p(b["emb"])
        fvk.embedding_lookup_bf16(
            ids, self.weights["embedding"], emb, S, TXT_D, stream=stream)
        # Image features are added into the <image_patch> rows of the prompt.
        feats = _p(b["proj"])
        starts = self.weights["precomputed"]["img_row_start"]
        rows = POOL_TOKENS * TXT_D
        for v in range(NUM_VIEWS):
            dst = emb + int(starts[v]) * TXT_D * 2
            src = feats + v * rows * 2
            fvk.add_bf16_out(dst, src, dst, rows, stream=stream)

    def discretize_state(self, stream: int = 0) -> None:
        pre = self.weights["precomputed"]
        self.fvk.molmo_state_discretize_f32(
            _p(self.bufs["state_in"]), _p(self.bufs["state_q01"]),
            _p(self.bufs["state_q99"]),
            _p(self.bufs["state_mask"]) if pre.get("state_mask") is not None else 0,
            _p(self.bufs["state_zero_mask"]) if pre.get("state_zero_mask") is not None else 0,
            _p(self.bufs["ids"], int(pre["state_id_offset"]) * 8),
            int(pre["state_dim"]), int(pre["num_state_tokens"]),
            int(pre["state_token_start_id"]), stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #  Stage: VLM prefill
    # ══════════════════════════════════════════════════════════════════
    def _prefill_reads_emb(self) -> bool:
        """Layer 0 can read the embeddings in place of the residual stream.

        On the calibrated NVFP4 path layer 0 reads x only in its qkv norm
        producer and as the o-projection's residual source (written to the
        ping-pong buffer), so the embedding seed copy is unnecessary.
        """
        nv = self.weights.get("nvfp4", {})
        return (self.fp8_calibrated and self.fp8_attn
                and all(f"txt_{k}_0" in nv for k in ("qkv", "o", "ff2"))
                and TXT_LAYERS > 1)

    def prefill_layer(self, i: int, stream: int = 0, x_in: int = 0) -> None:
        fvk = self.fvk
        b = self.bufs
        W = self.weights
        S = self.S
        x = _p(b["x"])
        xa = _p(b["xalt"])
        tmp = _p(b["tmp"])
        qkv = _p(b["qkv"])
        Q = _p(b["Q"])
        O = _p(b["O"])
        merged = _p(b["merged"])
        act = _p(b["act"])
        kv_stride = S * TXT_KV_DIM * 2
        nv = self.weights.get("nvfp4", {})
        cal = self.fp8_calibrated
        # Fused residual epilogues write the next residual stream while the
        # ping-pong buffer supplies C, so the residual stream alternates
        # x -> xalt -> x within the layer (no standalone residual adds).
        nv_res = cal and all(f"txt_{k}_{i}" in nv for k in
                             ("o", "ff2"))
        kv_only = (i == TXT_LAYERS - 1 and cal and self.kv_only_tail
                   and f"txt_qkv_{i}" in nv)
        x_src = x_in or x
        if kv_only:
            # The last layer feeds only the K/V caches: project the K|V
            # weight rows (a 128-row-aligned slice of the packed weight and
            # its swizzled scales) into a [S, K|V] buffer.
            self._norm_quant(
                x, W["bf16"][f"txt_ln1_{i}"], 0, 1, EPS,
                _p(b["nv_act"]), _p(b["nv_sf"]), _p(b["nv_one"]), S, TXT_D,
                stream, 0)
            w_ptr, sf_ptr, alpha_ptr, bias_ptr = nv[f"txt_qkv_{i}"]
            rb = TXT_Q_DIM // 128
            nb4 = (TXT_D // 16 + 3) // 4
            fvk.nvfp4_gemm_bias_res_bf16out(
                _p(b["nv_act"]), w_ptr + TXT_Q_DIM * TXT_D // 2,
                _p(b["nv_sf"]), sf_ptr + rb * nb4 * 512,
                bias_ptr + TXT_Q_DIM * 2, 0, qkv, S, 2 * TXT_KV_DIM, TXT_D,
                alpha_ptr, 0.0, 0, stream=stream)
        elif cal and f"txt_qkv_{i}" in nv:
            self._norm_quant(
                x_src, W["bf16"][f"txt_ln1_{i}"], 0, 1, EPS,
                _p(b["nv_act"]), _p(b["nv_sf"]), _p(b["nv_one"]), S, TXT_D,
                stream, 0)
            self._gemm_nvfp4_pre(f"txt_qkv_{i}", _p(b["nv_act"]),
                                 _p(b["nv_sf"]), qkv, S, TXT_QKV_DIM,
                                 TXT_D, stream)
        else:
            fvk.molmo_rms_norm_bf16(x, W["bf16"][f"txt_ln1_{i}"], tmp, S,
                                    TXT_D, EPS, stream=stream)
            self._gemm(f"txt_qkv_{i}", tmp, qkv, S, TXT_QKV_DIM, TXT_D,
                       stream)
        kptr = _p(b["K_cache"], i * kv_stride)
        vptr = _p(b["V_cache"], i * kv_stride)
        fa = self.fp8_attn and cal
        if fa and self._ctx_uses_fp8_cache() and self.skip_bf16_qkv:
            # FP8 attention reads Q/K/V from the FP8 copies and the action
            # context projects the FP8 caches: the bf16 copies are dead.
            Q = kptr = vptr = 0
        kv_pad_stride = self.kv_pad * TXT_KV_DIM
        # FP8-only consumers: the vectorized head_dim-128 producer.
        vec = self.qkv_vec and not (Q or kptr or vptr)
        if kv_only and vec:
            rc = fvk.molmo_qkv_norm_rope_fp8_hd128(
                qkv, W["bf16"][f"txt_qnorm_{i}"], W["bf16"][f"txt_knorm_{i}"],
                _p(b["txt_cos"]), _p(b["txt_sin"]), 0,
                _p(b["K_cache_fp8"], i * kv_pad_stride),
                _p(b["V_cache_fp8"], i * kv_pad_stride), 0,
                self.bufs["fp8_one"].ptr.value if fa
                else self._scale_buf(f"act_ck_s{i}").ptr.value,
                self.bufs["fp8_one"].ptr.value if fa
                else self._scale_buf(f"act_cv_s{i}").ptr.value,
                S, 0, TXT_NKV, TXT_KV_DIM, EPS, stream)
            if rc != 0:
                raise RuntimeError(f"molmo_qkv_norm_rope_fp8_hd128 rc={rc}")
            return
        if kv_only:
            fvk.molmo_qkv_split_bias_norm_rope(
                qkv, 0, W["bf16"][f"txt_qnorm_{i}"],
                W["bf16"][f"txt_knorm_{i}"], _p(b["txt_cos"]),
                _p(b["txt_sin"]), Q, kptr, vptr, S, 0,
                TXT_KV_DIM, TXT_KV_DIM, 0, TXT_NKV, TXT_HD, EPS, 0, 1,
                _p(b["K_cache_fp8"], i * kv_pad_stride),
                _p(b["V_cache_fp8"], i * kv_pad_stride),
                self.bufs["fp8_one"].ptr.value if fa
                else self._scale_buf(f"act_ck_s{i}").ptr.value,
                self.bufs["fp8_one"].ptr.value if fa
                else self._scale_buf(f"act_cv_s{i}").ptr.value,
                0, self.bufs["fp8_one"].ptr.value, stream)
            return
        if cal:
            ksc = (self.bufs["fp8_one"].ptr.value if fa
                   else self._scale_buf(f"act_ck_s{i}").ptr.value)
            vsc = (self.bufs["fp8_one"].ptr.value if fa
                   else self._scale_buf(f"act_cv_s{i}").ptr.value)
        if cal and vec:
            rc = fvk.molmo_qkv_norm_rope_fp8_hd128(
                qkv, W["bf16"][f"txt_qnorm_{i}"], W["bf16"][f"txt_knorm_{i}"],
                _p(b["txt_cos"]), _p(b["txt_sin"]),
                _p(b["Q_fp8"]) if fa else 0,
                _p(b["K_cache_fp8"], i * kv_pad_stride),
                _p(b["V_cache_fp8"], i * kv_pad_stride),
                self.bufs["fp8_one"].ptr.value, ksc, vsc, S, TXT_NH, TXT_NKV,
                TXT_KV_DIM, EPS, stream)
            if rc != 0:
                raise RuntimeError(f"molmo_qkv_norm_rope_fp8_hd128 rc={rc}")
        elif cal:
            fvk.molmo_qkv_split_bias_norm_rope(
                qkv, 0, W["bf16"][f"txt_qnorm_{i}"],
                W["bf16"][f"txt_knorm_{i}"], _p(b["txt_cos"]),
                _p(b["txt_sin"]), Q, kptr, vptr, S, TXT_Q_DIM,
                TXT_KV_DIM, TXT_KV_DIM, TXT_NH, TXT_NKV, TXT_HD, EPS, 0, 1,
                _p(b["K_cache_fp8"], i * kv_pad_stride),
                _p(b["V_cache_fp8"], i * kv_pad_stride), ksc, vsc,
                _p(b["Q_fp8"]) if fa else 0,
                self.bufs["fp8_one"].ptr.value, stream)
        else:
            fvk.molmo_qkv_split_bias_norm_rope(
                qkv, 0, W["bf16"][f"txt_qnorm_{i}"],
                W["bf16"][f"txt_knorm_{i}"], _p(b["txt_cos"]),
                _p(b["txt_sin"]), Q, kptr, vptr, S, TXT_Q_DIM,
                TXT_KV_DIM, TXT_KV_DIM, TXT_NH, TXT_NKV, TXT_HD, EPS, 0, 1,
                0, 0, 0, 0, 0, 0, stream)
        if i == TXT_LAYERS - 1:
            # The action expert reads only the per-layer K/V caches, so the
            # last layer's attention output, o-projection and MLP are dead.
            return
        if fa:
            # SM120 FP8 causal GQA attention (repository kernel); emits the
            # o_proj NVFP4 operand directly (no bf16 O round-trip/quantize).
            if self.attn_wsp:
                attn_fn = fvk.fmha_fp8_causal_gqa_q64_nhd_d128_fp4out_wsp
            elif self.attn_q64:
                attn_fn = fvk.fmha_fp8_causal_gqa_q64_nhd_d128_fp4out
            else:
                attn_fn = fvk.fmha_fp8_causal_gqa_nhd_d128_fp4out
            rc = attn_fn(
                _p(b["Q_fp8"]),
                _p(b["K_cache_fp8"], i * kv_pad_stride),
                _p(b["V_cache_fp8"], i * kv_pad_stride),
                _p(b["o_fp4"]), _p(b["o_fp4_sf"]), S,
                S, TXT_NH, TXT_NKV, 1.0 / (TXT_HD ** 0.5),
                stream)
            if rc != 0:
                raise RuntimeError(f"fp8 attention rejected shape {rc}")
            o_out, o_x2 = (xa, xa) if nv_res else (tmp, x)
            self._gemm_nvfp4_pre(
                f"txt_o_{i}", _p(b["o_fp4"]), _p(b["o_fp4_sf"]), o_out,
                S, TXT_D, TXT_Q_DIM, stream, residual=x_src,
                fuse_residual=nv_res)
            x2 = o_x2
        else:
            self._fa2(Q, kptr, vptr, O, _p(b["lse_txt"]), 1, S, S, TXT_NH,
                      TXT_NKV, TXT_HD, True, stream)
            if nv_res:
                self._gemm(f"txt_o_{i}", O, xa, S, TXT_D, TXT_Q_DIM, stream,
                           residual=x, fuse_residual=True)
                x2 = xa
            else:
                self._gemm(f"txt_o_{i}", O, tmp, S, TXT_D, TXT_Q_DIM, stream,
                           residual=x)
                x2 = x

        fold_alpha = self.swiglu_fold.get(f"txt_ff1_{i}")
        if fold_alpha is not None:
            # ff1 GEMM over the interleaved gate|up weight with the SwiGLU
            # ->NVFP4 epilogue, then the even-column compaction into the ff2
            # operand (the Qwen3 RTX prefill route).
            if cal:
                self._norm_quant(
                    x2, W["bf16"][f"txt_ln2_{i}"], 0, 1, EPS,
                    _p(b["nv_act"]), _p(b["nv_sf"]), _p(b["nv_one"]), S,
                    TXT_D, stream, 0)
            else:
                fvk.molmo_rms_norm_bf16(x2, W["bf16"][f"txt_ln2_{i}"], tmp, S,
                                        TXT_D, EPS, stream=stream)
                fvk.nvfp4_quant(tmp, TXT_D, _p(b["nv_act"]), _p(b["nv_sf"]),
                                _p(b["nv_one"]), S, TXT_D, stream=stream)
            w_ptr, sf_ptr, _, _ = nv[f"txt_ff1_{i}"]
            fvk.fp4_w4a16_dual_gemm_silu_fp4out_sm120(
                _p(b["nv_act"]), w_ptr, w_ptr, _p(b["nv_sf"]), sf_ptr, sf_ptr,
                merged, _p(b["nv_sf_ff2"]), self.S_pad128, 2 * TXT_FFN,
                TXT_D, fold_alpha, fold_alpha, stream, self.swiglu_ceil_sf)
            fvk.fp4_swiglu_even_col_compact_v2(
                merged, _p(b["nv_act"]), S, TXT_FFN, stream)
            if cal:
                self._gemm_nvfp4_pre(f"txt_ff2_{i}", _p(b["nv_act"]),
                                     _p(b["nv_sf_ff2"]), x if nv_res else tmp,
                                     S, TXT_D, TXT_FFN, stream,
                                     residual=xa if nv_res else 0,
                                     fuse_residual=nv_res)
            else:
                self._gemm_nvfp4_pre(f"txt_ff2_{i}", _p(b["nv_act"]),
                                     _p(b["nv_sf_ff2"]), tmp, S, TXT_D,
                                     TXT_FFN, stream, residual=x2)
            return
        if cal and f"txt_ff1_{i}" in nv:
            self._norm_quant(
                x2, W["bf16"][f"txt_ln2_{i}"], 0, 1, EPS,
                _p(b["nv_act"]), _p(b["nv_sf"]), _p(b["nv_one"]), S, TXT_D,
                stream, 0)
            self._gemm_nvfp4_pre(f"txt_ff1_{i}", _p(b["nv_act"]),
                                 _p(b["nv_sf"]), merged, S, 2 * TXT_FFN,
                                 TXT_D, stream)
        else:
            fvk.molmo_rms_norm_bf16(x2, W["bf16"][f"txt_ln2_{i}"], tmp, S,
                                    TXT_D, EPS, stream=stream)
            self._gemm(f"txt_ff1_{i}", tmp, merged, S, 2 * TXT_FFN, TXT_D,
                       stream)
        if cal and f"txt_ff2_{i}" in nv:
            if self.silu_vec:
                fvk.silu_mul_merged_to_nvfp4_swizzled_vec_bf16(
                    merged, _p(b["nv_act"]), _p(b["nv_sf"]), S, TXT_FFN,
                    stream)
            else:
                fvk.silu_mul_merged_to_nvfp4_swizzled_bf16(
                    merged, _p(b["nv_act"]), _p(b["nv_sf"]), S, TXT_FFN,
                    stream)
            self._gemm_nvfp4_pre(f"txt_ff2_{i}", _p(b["nv_act"]),
                                 _p(b["nv_sf"]), x if nv_res else tmp, S,
                                 TXT_D, TXT_FFN, stream,
                                 residual=xa if nv_res else 0,
                                 fuse_residual=nv_res)
        else:
            fvk.silu_mul_merged_bf16(merged, act, S, TXT_FFN, stream=stream)
            self._gemm(f"txt_ff2_{i}", act, tmp, S, TXT_D, TXT_FFN, stream,
                       residual=x2)

    def prefill(self, stream: int = 0) -> None:
        if self._prefill_reads_emb():
            self.prefill_layer(0, stream, x_in=_p(self.bufs["emb"]))
        else:
            self._memcpy_d2d(_p(self.bufs["x"]), _p(self.bufs["emb"]),
                             self.S * TXT_D * 2, stream)
            self.prefill_layer(0, stream)
        for i in range(1, TXT_LAYERS):
            self.prefill_layer(i, stream)

    def _ctx_uses_fp8_cache(self) -> bool:
        return (self.fp8_calibrated and self.fp8_attn
                and "act_ck" in self.weights["fp8"]
                and "act_cv" in self.weights["fp8"])

    def _ctx_batched_route(self) -> bool:
        return bool(self.ctx_batched and self.fp8_calibrated and self.fp8_attn
                    and "act_ck" in self.weights["fp8"]
                    and "act_cv" in self.weights["fp8"])

    def action_context(self, stream: int = 0) -> None:
        fvk = self.fvk
        b = self.bufs
        W = self.weights
        S = self.S
        kv_stride = S * TXT_KV_DIM * 2
        ctx_stride = self.kv_pad * ACT_KV_DIM * 2
        flat = _p(b["ctx_flat"])
        tmp = _p(b["ctx_tmp"])
        if self._ctx_batched_route():
            # Every layer shares context_k/v_proj and the unit FP8 scale:
            # one GEMM per projection over a group of layers' kv_pad-strided
            # cache rows (pad rows are computed and ignored), then row-wise
            # norms; the group keeps the bf16 intermediate L2-resident.
            G = self.ctx_group
            one = self.bufs["fp8_one"].ptr.value
            for g0 in range(0, ACT_LAYERS, G):
                n = min(G, ACT_LAYERS - g0)
                R = n * self.kv_pad
                off = g0 * self.kv_pad
                self._gemm_fp8_pre(
                    "act_ck", _p(b["K_cache_fp8"], off * TXT_KV_DIM), tmp,
                    R, ACT_D, CONTEXT_DIM, one, stream)
                if self.act_mk is not None:
                    # head-major, row-padded layout of the persistent action
                    # kernel (one bulk copy per head and KV split)
                    hm = off * ACT_KV_DIM // ACT_HD * ACT_MK_CTX_STRIDE * 2
                    # byte-identical warp-per-row form of
                    # molmo_rms_norm768_heads96 / _headmajor
                    fvk.molmo_rms_norm768_ctx_warp(
                        tmp, _p(b["ctxK_hm"], hm), R, EPS, EPS, 1,
                        self.kv_pad, ACT_MK_CTX_STRIDE, stream)
                else:
                    # context norm + per-head k norm in one launch
                    fvk.molmo_rms_norm768_heads96(
                        tmp, _p(b["ctxK"], off * ACT_KV_DIM * 2), R, EPS,
                        EPS, stream=stream)
                self._gemm_fp8_pre(
                    "act_cv", _p(b["V_cache_fp8"], off * TXT_KV_DIM), tmp,
                    R, ACT_D, CONTEXT_DIM, one, stream)
                if self.act_mk is not None:
                    fvk.molmo_rms_norm768_ctx_warp(
                        tmp, _p(b["ctxV_hm"], hm), R, EPS, EPS, 0,
                        self.kv_pad, ACT_MK_CTX_STRIDE, stream)
                else:
                    fvk.molmo_rms_norm_bf16(
                        tmp, 0, _p(b["ctxV"], off * ACT_KV_DIM * 2), R,
                        ACT_D, EPS, stream=stream)
            return
        for i in range(ACT_LAYERS):
            kptr = _p(b["K_cache"], i * kv_stride)
            vptr = _p(b["V_cache"], i * kv_stride)
            if (self.fp8_calibrated and "act_ck" in self.weights["fp8"]):
                kfp = _p(b["K_cache_fp8"], i * self.kv_pad * TXT_KV_DIM)
                vfp = _p(b["V_cache_fp8"], i * self.kv_pad * TXT_KV_DIM)
                self._gemm_fp8_pre(
                    "act_ck", kfp, tmp, S, ACT_D, CONTEXT_DIM,
                    self.bufs["fp8_one"].ptr.value if self.fp8_attn else
                    self._scale_buf(f"act_ck_s{i}").ptr.value, stream)
            else:
                self._gemm("act_ck", kptr, tmp, S, ACT_D, CONTEXT_DIM,
                           stream, scale_key=f"act_ck_s{i}")
            fvk.molmo_rms_norm_bf16(tmp, 0, flat, S, ACT_D, EPS,
                                    stream=stream)
            # per-head k norm over HD=96: bit-exact warp-per-row replica of
            # the reference reduction (5.5x faster, no block barrier)
            fvk.molmo_rms_norm_rows96_exact(
                flat, _p(b["ctxK"], i * ctx_stride), S * ACT_NH, EPS,
                stream=stream)
            if (self.fp8_calibrated and "act_cv" in self.weights["fp8"]):
                self._gemm_fp8_pre(
                    "act_cv", vfp, tmp, S, ACT_D, CONTEXT_DIM,
                    self.bufs["fp8_one"].ptr.value if self.fp8_attn else
                    self._scale_buf(f"act_cv_s{i}").ptr.value, stream)
            else:
                self._gemm("act_cv", vptr, tmp, S, ACT_D, CONTEXT_DIM,
                           stream, scale_key=f"act_cv_s{i}")
            fvk.molmo_rms_norm_bf16(
                tmp, 0, _p(b["ctxV"], i * ctx_stride), S, ACT_D, EPS,
                stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #  Stage: action expert flow loop
    # ══════════════════════════════════════════════════════════════════
    def _act_block(self, i: int, stream: int) -> None:
        fvk = self.fvk
        b = self.bufs
        H, D, HID = self.horizon, ACT_D, ACT_H
        x = _p(b["ax"])
        qkv = _p(b["aqkv"])
        Q, K, V = _p(b["aQ"]), _p(b["aK"]), _p(b["aV"])
        O = _p(b["aO"])
        tmp = _p(b["atmp"])
        up, gate, act = _p(b["aup"]), _p(b["agate"]), _p(b["aact"])
        mod_self = _p(b["mod_self"])
        mod_cross = _p(b["mod_cross"])
        mod_ffn = _p(b["mod_ffn"])
        per_layer = 3 * D * 2
        per_step = ACT_LAYERS * per_layer
        step = self._current_step
        sec = step * per_step + i * per_layer
        sk = self.weights.get("skinny", {})
        use_sk = self.fp8_calibrated and bool(sk)

        # self-attention
        site = f"act_qkv_{i}"
        if use_sk and site in sk and self._skinny_ok(site):
            fvk.molmo_mod_rms_norm_bf16(
                x, 0, mod_self + sec, mod_self + sec + D * 2,
                _p(b["axn"]), H, D, EPS, stream=stream)
            splits = self._skinny_gemm(site, _p(b["axn"]), step, stream)
            fvk.molmo_skinny_qkv_split_bias(
                _p(b["skinny_partials"]), splits,
                self._skinny_ags(site, step), sk[site][2],
                self._act_bias(site), Q, K, V, H, D, D, D, stream=stream)
        else:
            pending, self._pending_norm = self._pending_norm, None
            if pending is not None and pending[0] == (step, i):
                a, a_s, pre = pending[1]
            else:
                a, a_s, pre = self._act_mod_norm(
                    x, mod_self + sec, mod_self + sec + D * 2, stream, site)
            bias_done = self._act_gemm(site, a, pre, a_s, qkv, 3 * D, D,
                                       stream)
            # fused split (+bias when the GEMM did not add it) + qk-norm + RoPE
            fvk.molmo_qkv_split_bias_norm_rope(
                qkv, 0 if bias_done else self._act_bias(site), 0, 0,
                _p(b["act_cos"]), _p(b["act_sin"]), Q, K, V, H, D, D, D,
                ACT_NH, ACT_NH, ACT_HD, EPS, 0 if bias_done else 1, 1,
                0, 0, 0, 0, 0, 0, stream)
        if use_sk and site in sk:
            fvk.molmo_qk_norm_rope_bf16(
                Q, 0, _p(b["act_cos"]), _p(b["act_sin"]), H, ACT_NH, ACT_HD,
                EPS, 0, stream=stream)
            fvk.molmo_qk_norm_rope_bf16(
                K, 0, _p(b["act_cos"]), _p(b["act_sin"]), H, ACT_NH, ACT_HD,
                EPS, 0, stream=stream)
        self._fa2_short(Q, K, V, O, _p(b["lse_act"]), H, H, ACT_NH,
                        ACT_NH, ACT_HD, stream)
        site = f"act_so_{i}"
        cq = f"act_cq_{i}"
        next_norm = None
        if use_sk and site in sk and self._skinny_ok(site):
            nrm = None
            if not (use_sk and cq in sk and self._skinny_ok(cq)):
                nrm = (mod_cross + sec, mod_cross + sec + D * 2, cq)
            next_norm = self._skinny_gate_res(
                site, O, x, mod_self + sec + 2 * D * 2, step, stream, nrm)
        elif site in self.weights.get("nvfp4", {}):
            self._gemm_bias(site, O, tmp, H, D, D, stream)
            fvk.molmo_gate_residual_bf16(
                x, tmp, mod_self + sec + 2 * D * 2, H, D, stream=stream)
        else:
            self._gemm(site, O, tmp, H, D, D, stream)
            fvk.molmo_gate_residual_bias(
                x, tmp, mod_self + sec + 2 * D * 2, self._act_bias(site),
                H, D, stream=stream)

        # cross-attention
        site = f"act_cq_{i}"
        if use_sk and site in sk and self._skinny_ok(site):
            fvk.molmo_mod_rms_norm_bf16(
                x, 0, mod_cross + sec, mod_cross + sec + D * 2,
                _p(b["axn"]), H, D, EPS, stream=stream)
            splits = self._skinny_gemm(site, _p(b["axn"]), step, stream)
            fvk.molmo_skinny_bias_rms_norm(
                _p(b["skinny_partials"]), splits,
                self._skinny_ags(site, step), sk[site][2],
                self._act_bias(site), Q, H * ACT_NH, ACT_NH, ACT_HD, EPS,
                stream=stream)
        else:
            a, a_s, pre = next_norm or self._act_mod_norm(
                x, mod_cross + sec, mod_cross + sec + D * 2, stream, site)
            self._act_gemm(site, a, pre, a_s, Q, D, D, stream)
            # hd == 96: bit-exact warp-per-row replicas of the reference
            # block reduction (bias folded into the load), 2.0x faster.
            if pre:
                fvk.molmo_bias_rms_norm_rows96_exact(
                    Q, self._act_bias(site), Q, H * ACT_NH, ACT_NH, EPS,
                    stream=stream)
            else:
                fvk.molmo_rms_norm_rows96_exact(
                    Q, Q, H * ACT_NH, EPS, stream=stream)
        ctx_stride = self.kv_pad * D * 2
        # Decode-shaped cross attention: the repository split-KV FA2 entry
        # (num_splits from the scheduler heuristic) beats the single-CTA
        # tile kernel at M=30; outputs match to 1 bf16 ulp.
        self.fa2.fwd_bf16_seqused_splitkv(
            Q, _p(b["ctxK"], i * ctx_stride),
            _p(b["ctxV"], i * ctx_stride), O, _p(b["lse_act"]),
            _p(b["seqused_k"]), _p(b["lse_act_split"]), _p(b["o_accum"]),
            1, H, self.S, ACT_NH, ACT_NH, ACT_HD,
            (H * ACT_NH * ACT_HD, ACT_NH * ACT_HD, ACT_HD),
            (self.S * ACT_NH * ACT_HD, ACT_NH * ACT_HD, ACT_HD),
            (self.S * ACT_NH * ACT_HD, ACT_NH * ACT_HD, ACT_HD),
            (H * ACT_NH * ACT_HD, ACT_NH * ACT_HD, ACT_HD),
            1.0 / (ACT_HD ** 0.5), self.xattn_sms, stream)
        site = f"act_co_{i}"
        next_norm = None
        if use_sk and site in sk and self._skinny_ok(site):
            nrm = None
            fu, fg = f"act_up_{i}", f"act_gate_{i}"
            if not (use_sk and fu in sk and fg in sk and self._skinny_ok(fu)):
                nrm = (mod_ffn + sec, mod_ffn + sec + D * 2, fu)
            next_norm = self._skinny_gate_res(
                site, O, x, mod_cross + sec + 2 * D * 2, step, stream, nrm)
        elif site in self.weights.get("nvfp4", {}):
            self._gemm_bias(site, O, tmp, H, D, D, stream)
            fvk.molmo_gate_residual_bf16(
                x, tmp, mod_cross + sec + 2 * D * 2, H, D, stream=stream)
        else:
            self._gemm(site, O, tmp, H, D, D, stream)
            fvk.molmo_gate_residual_bias(
                x, tmp, mod_cross + sec + 2 * D * 2, self._act_bias(site),
                H, D, stream=stream)

        # FFN
        site_u, site_g = f"act_up_{i}", f"act_gate_{i}"
        site_d = f"act_down_{i}"
        sk_up = (use_sk and site_u in sk and site_g in sk
                 and self._skinny_ok(site_u))
        sk_down = use_sk and site_d in sk and self._skinny_ok(site_d)
        act_is_fp8 = False
        if sk_up:
            fvk.molmo_mod_rms_norm_bf16(
                x, 0, mod_ffn + sec, mod_ffn + sec + D * 2, _p(b["axn"]),
                H, D, EPS, stream=stream)
            su = self._skinny_gemm(site_u, _p(b["axn"]), step, stream,
                                   "skinny_partials")
            sg = self._skinny_gemm(site_g, _p(b["axn"]), step, stream,
                                   "skinny_partials2")
            fvk.molmo_skinny_silu_mul(
                _p(b["skinny_partials"]), su, self._skinny_ags(site_u, step),
                sk[site_u][2], _p(b["skinny_partials2"]), sg,
                self._skinny_ags(site_g, step), sk[site_g][2],
                self._act_bias(site_g), self._act_bias(site_u), act, H, HID,
                stream=stream)
        else:
            a, a_s, pre = next_norm or self._act_mod_norm(
                x, mod_ffn + sec, mod_ffn + sec + D * 2, stream, site_u)
            if pre and sk_down and self._act_fused_ok(site_u, site_g):
                au, ag = self._act_fused_alphas(site_u, site_g)
                nk = self.weights["fp8_nk"]
                rc = fvk.action_swiglu_fp8(
                    a, nk[site_u][0], nk[site_g][0],
                    self._act_bias(site_u), self._act_bias(site_g), act,
                    H, D, HID, au, ag, stream)
                if rc != 0:
                    raise RuntimeError(f"action_swiglu_fp8 rc={rc}")
            else:
                self._act_gemm(site_u, a, pre, a_s, up, HID, D, stream)
                if pre:
                    self._gemm_fp8_pre(site_g, a, gate, H, HID, D, a_s,
                                       stream)
                    if sk_down:
                        fvk.molmo_silu_mul_bias(
                            gate, up, self._act_bias(site_g),
                            self._act_bias(site_u), act, H, HID,
                            stream=stream)
                    else:
                        down_sc = self._scale_buf(site_d)
                        fvk.molmo_silu_mul_quant_fp8(
                            gate, up, self._act_bias(site_g),
                            self._act_bias(site_u), _p(b["fp8_act"]),
                            down_sc.ptr.value, H, HID, stream=stream)
                        act_is_fp8 = True
                else:
                    self._gemm_bias(site_g, a, gate, H, HID, D, stream)
                    nv = self.weights.get("nvfp4", {})
                    bg = 0 if site_g in nv else self._act_bias(site_g)
                    bu = 0 if site_u in nv else self._act_bias(site_u)
                    fvk.molmo_silu_mul_bias(gate, up, bg, bu, act, H, HID,
                                            stream=stream)
        if sk_down:
            nrm = None
            nq = f"act_qkv_{i + 1}"
            if i + 1 < ACT_LAYERS and not (use_sk and nq in sk
                                           and self._skinny_ok(nq)):
                nsec = sec + per_layer
                nrm = (mod_self + nsec, mod_self + nsec + D * 2, nq)
            res = self._skinny_gate_res(
                site_d, act, x, mod_ffn + sec + 2 * D * 2, step, stream, nrm)
            if res is not None:
                self._pending_norm = ((step, i + 1), res)
        elif act_is_fp8:
            self._gemm_fp8_pre(site_d, _p(b["fp8_act"]), tmp, H, D, HID,
                               self._scale_buf(site_d).ptr.value, stream)
            fvk.molmo_gate_residual_bias(
                x, tmp, mod_ffn + sec + 2 * D * 2, self._act_bias(site_d),
                H, D, stream=stream)
        else:
            self._gemm_bias(site_d, act, tmp, H, D, HID, stream)
            fvk.molmo_gate_residual_bf16(
                x, tmp, mod_ffn + sec + 2 * D * 2, H, D, stream=stream)

    def flow_step(self, step: int, stream: int = 0) -> None:
        """One flow-matching step: trajectory -> velocity (no Euler update)."""
        fvk = self.fvk
        b = self.bufs
        H, D = self.horizon, ACT_D
        traj = _p(b["traj"])
        vel = _p(b["vel"])
        self._current_step = step
        self._gemm_bias("act_embed", traj, _p(b["ax"]), H, D, ACT_MAX_DIM,
                        stream)
        if self.act_mk is not None:
            self.act_mk.launch(step, stream)
        else:
            for i in range(ACT_LAYERS):
                self._act_block(i, stream)
        mod_final = _p(b["mod_final"])
        fbase = step * 2 * D * 2
        fvk.molmo_mod_rms_norm_bf16(
            _p(b["ax"]), 0, mod_final + fbase, mod_final + fbase + D * 2,
            _p(b["axn"]), H, D, EPS, stream=stream)
        self._gemm_bias("act_final", _p(b["axn"]), vel, H, ACT_MAX_DIM, D,
                        stream)
        fvk.molmo_mask_dims_bf16(vel, H, ACT_MAX_DIM, self.act_dim,
                                 stream=stream)
        self._current_step = -1

    def flow_loop(self, stream: int = 0) -> None:
        b = self.bufs
        H2 = self.horizon * ACT_MAX_DIM
        traj = _p(b["traj"])
        self._memcpy_d2d(traj, _p(b["noise"]), H2 * 2, stream)
        dt = 1.0 / float(self.num_steps)
        for step in range(self.num_steps):
            self.flow_step(step, stream)
            self.fvk.euler_step_bf16_out(
                traj, _p(b["vel"]), traj, dt, H2, stream=stream)

    def postprocess(self, stream: int = 0) -> None:
        self.fvk.molmo_action_denorm_bf16(
            _p(self.bufs["traj"]), _p(self.bufs["actions"]),
            _p(self.bufs["act_q01"]), _p(self.bufs["act_q99"]),
            _p(self.bufs["act_mask"]), self.horizon, ACT_MAX_DIM,
            self.act_dim, stream=stream)

    # ══════════════════════════════════════════════════════════════════
    #  Full pipeline / graph
    # ══════════════════════════════════════════════════════════════════
    def upload_inputs(self, stream: int = 0) -> None:
        """Pinned-host -> device staging copies (captured in the graph)."""
        if not getattr(self, "host_images_ptr", 0):
            return
        self._memcpy(self.input_images_ptr, self.host_images_ptr,
                     NUM_VIEWS * IMG_H * IMG_W * 3, 1, stream)
        self._memcpy(_p(self.bufs["state_in"]), self.host_state_ptr,
                     int(self.weights["precomputed"]["state_dim"]) * 4, 1,
                     stream)
        self._memcpy(_p(self.bufs["noise"]), self.host_noise_ptr,
                     self.horizon * ACT_MAX_DIM * 2, 1, stream)

    def download_actions(self, stream: int = 0) -> None:
        self._memcpy(self.host_actions_ptr, _p(self.bufs["actions"]),
                     self.horizon * self.act_dim * 4, 2, stream)

    def _memcpy(self, dst: int, src: int, nbytes: int, kind: int,
                stream: int) -> None:
        ret = self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, kind,
            ctypes.c_void_p(stream))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed: {ret}")

    def run_pipeline(self, stream: int = 0) -> None:
        self.upload_inputs(stream)
        self.image_preprocess(stream)
        self.vision_encoder(stream)
        self.image_adapter(stream)
        self.discretize_state(stream)
        self.build_embeddings(stream)
        self.prefill(stream)
        self.action_context(stream)
        self.flow_loop(stream)
        self.postprocess(stream)
        self.download_actions(stream)

    def _memcpy2d_d2d(self, dst: int, dpitch: int, src: int, spitch: int,
                      width: int, rows: int, stream: int) -> None:
        ret = self._cudart.cudaMemcpy2DAsync(
            ctypes.c_void_p(dst), dpitch, ctypes.c_void_p(src), spitch, width,
            rows, 3, ctypes.c_void_p(stream))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy2DAsync D2D failed: {ret}")

    def _memcpy_d2d(self, dst: int, src: int, nbytes: int, stream: int):
        ret = self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, 3,
            ctypes.c_void_p(stream))
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2D failed: {ret}")

    def record_infer_graph(self, external_stream_int: int | None = None) -> None:
        self._graph = CUDAGraph()
        if external_stream_int is None:
            stream = self._graph.create_stream()
            stream_int = stream.value or 0
            stream_handle = stream
        else:
            stream_int = int(external_stream_int)
            stream_handle = ctypes.c_void_p(stream_int)
        self._graph_stream = stream_handle
        for _ in range(3):
            self.run_pipeline(stream=stream_int)
        self._cudart.cudaStreamSynchronize(stream_handle)
        self._graph.begin_capture(stream_handle)
        self.run_pipeline(stream=stream_int)
        self._graph.end_capture(stream_handle)
        self._cudart.cudaStreamSynchronize(stream_handle)
        logger.info("MolmoAct2 CUDA Graph captured (S=%d, steps=%d)",
                    self.S, self.num_steps)

    def replay(self, stream: int = 0) -> None:
        self._graph.replay(ctypes.c_void_p(stream))
