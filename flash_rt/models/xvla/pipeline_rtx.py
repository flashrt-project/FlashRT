"""X-VLA (soft-prompted flow-matching VLA) — RTX / SM120 compute path.

Reference model: ``lerobot/xvla-libero`` (LeRobot `XVLAPolicy`) — Florence-2's
DaViT vision tower and BART encoder feeding a 24-layer soft-prompted
flow-matching transformer run for 10 denoising steps.

Stages, in the order the reference calls them:

1. ``vision_forward`` — DaViT-L over each camera view, then
   ``Florence2MultiModalProjector``: learned 2-D position add, a broadcast
   temporal vector, ``[spatial_avg_pool, temporal_avg_pool]`` -> 50 tokens,
   a bias-free 2048 -> 1024 projection and a LayerNorm.
2. ``encode_forward`` — BART encoder over ``vision_token(cam0) ++ text``
   (100 tokens, 12 layers, learned positions, erf-GELU FFN).
3. ``denoise_forward`` — 10 unrolled Euler steps of ``SoftPromptedTransformer``
   over ``[action(30) | vlm(100) | aux_visual(100) | soft_prompt(32)]``
   = 262 tokens, 24 pre-LN blocks, GELU-tanh MLP.

Everything is native: GEMMs go through the FlashRT GEMM runner (cuBLASLt) or
the SM120 block-scaled FP8/NVFP4 CUTLASS paths, attention through the vendored
FA2 (SM120 AOT), and norm / activation / quantize / elementwise through
``flash_rt_kernels``. Every call takes raw pointers and runs on the caller's
stream, so the whole forward captures into a CUDA Graph with no host work.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field

import torch

# ── Geometry, fixed by lerobot/xvla-libero/config.json ──
HIDDEN = 1024
ACT_HEADS = 16
ACT_MLP = 4096
ACT_LAYERS = 24
CHUNK = 30
ACTION_DIM = 20
DIM_PROPRIO = 20
DIM_TIME = 32
N_SOFT = 32
N_DENOISE = 10

VISION_TOKENS = 50  # per view: 1 pooled + 49 patch
N_VIEWS = 3  # cam0 and cam1 are real; cam2 is the configured empty camera
TEXT_LEN = 50
ENC_SEQ = VISION_TOKENS + TEXT_LEN  # 100
AUX_VIEWS = N_VIEWS - 1
AUX_TOKENS = AUX_VIEWS * VISION_TOKENS  # 100
# The transformer's non-action segments are the BART encoder output (100
# tokens, not the 50 vision tokens) plus the two auxiliary views' raw pooled
# features (100 tokens).
PREFIX = CHUNK + ENC_SEQ + AUX_TOKENS  # 230
SEQ = PREFIX + N_SOFT  # 262

ENC_LAYERS = 12
ENC_HEADS = 16
ENC_MLP = 4096
ENC_HEAD_DIM = HIDDEN // ENC_HEADS

GRIPPER = (9, 19)  # EE6DActionSpace

# The fp4-out FFN epilogue (bias + GELU + NVFP4 output) emits its own output
# scale factors, which removes the standalone FFN-midpoint quantize. Its
# persistent scheduler only accepts an M it can tile whole (`can_implement`
# returns kErrorMemoryAllocation at M=262, succeeds at 128 and 256), so the fc1
# activation carries SEQ rows of data padded up to a tile multiple.
USE_FP4OUT_FFN = True

# BART encoder GEMMs on NVFP4 W4A4 instead of block-128 FP8. The encoder is the
# last stage still on FP8 while the action transformer runs 4-bit; measured at
# its M=100 shapes the 4-bit tile is worth roughly 18 us per layer. Kept or
# dropped by the delivery gate on the real corpus.
USE_FP4_ENCODER = True
# The DaViT tower's four block GEMM families run NVFP4 W4A4. The block-128
# FP8 tower tier was measured slower at the same gate and is not part of this
# path; `xvla_precision` still selects FP8 for the action transformer/encoder.
# Fuse the DaViT FFN's bias + GELU(tanh) + NVFP4 output into the fc1 GEMM
# epilogue. The epilogue's GELU is the tanh approximation, so the 96-frame gate
# decides the scope: all four stages give 26.047 ms / min 0.984575 (fail),
# stage 2 alone 26.189 ms / min 0.991049 (pass), stage 1 alone 0.984125 (fail),
# stage 0 passes alone but not together with stage 2. Stage 2 (9 of the 12
# blocks) is the delivered scope; the env vars remain for A/B.
USE_VISION_GELU_FUSED = os.environ.get("XVLA_VISION_GELU_FUSED", "1") == "1"
USE_DECODER_GEMV = os.environ.get("XVLA_DECODER_GEMV", "1") == "1"
_FUSED_SCOPE = os.environ.get("XVLA_VISION_GELU_FUSED_STAGES", "2")
FUSED_STAGES = {int(x) for x in _FUSED_SCOPE.split(",") if x.strip()}
# Programmatic dependent launch between the graph's nodes: the NVFP4 CUTLASS
# GEMMs, the FA2 tiles and the fused norms start their launch and prologue
# under the previous node's tail (csrc/kernels/pdl_switch.cuh).
USE_PDL = os.environ.get("XVLA_PDL", "1") == "1"
FP4OUT_M = 384  # ceil(SEQ / 128) * 128
# The action decoder reads only the CHUNK action rows of the last block's
# output, and every op after the last block's attention is row-local (GEMM rows,
# LayerNorm, the per-16 NVFP4 scales). So that block's attention takes CHUNK
# queries against all SEQ keys and its proj / n2 / fc1 / fc2 run on CHUNK rows;
# each kept row is computed by exactly the same instructions, so the action
# chunk is bit-identical. The last block's K/V still need every token.
PRUNE_LAST_BLOCK = os.environ.get("XVLA_PRUNE_LAST", "1") == "1"
FP4OUT_M_LAST = 128  # ceil(CHUNK / 128) * 128
# The action-transformer and encoder NVFP4 GEMMs (M = 262 / 100 / 30) on the
# small-M kernels in csrc/kernels/xvla_fp4_gemm_sm120.cu instead of the 128x128
# CUTLASS tile: same operands, bit-identical outputs, M tiles of 32-112 rows.
# Measured in-graph on real operands at M=262: proj 7.54 -> 4.71 us, qkv
# 11.49 -> 9.04, fc1 (GELU + fp4 out) 18.27 -> 16.31, fc2 16.88 -> 12.36.
USE_SMALLM_FP4_GEMM = os.environ.get("XVLA_SMALLM_GEMM", "1") == "1"
# Attention on FA2's two-warp 32-row query tile <64,32,64,2>. FA2 computes each
# 16-row warp slice identically whatever the tile height, so outputs are
# bit-identical; the denoise shape gets 144 CTAs (four per SM) instead of 80.
USE_FA2_M32 = os.environ.get("XVLA_FA2_M32", "1") == "1"
LN_EPS = 1e-5

# ── DaViT vision tower ──
VIS_DIM = (256, 512, 1024, 2048)
VIS_HEADS = (8, 16, 32, 64)
VIS_GROUPS = (8, 16, 32, 64)
VIS_DEPTH = (1, 1, 9, 1)
VIS_KS = (7, 3, 3, 3)
VIS_STRIDE = (4, 2, 2, 2)
VIS_PAD = (3, 1, 1, 1)
VIS_PRENORM = (False, True, True, True)
VIS_MLP_RATIO = 4.0
WINDOW = 12
SRC_IMG = 256  # LIBERO camera frames; resized to `img` by the tower
VIS_TOK_DIM = 1024  # projection_dim
VIS_OUT_DIM = 2048  # last DaViT stage width
VIS_HALF = False  # DaViT runs bf16: its block GEMMs take the SM120 block-128 FP8
# path, whose activation quantizer emits bf16. 1.4-1.6x the fp16 GEMM throughput
# at these shapes (measured); the tower-output cosine cost is 0.9988 vs 1.0000.


@dataclass
class XVLADims:
    img: int = 224
    vision: list[tuple[int, int, int]] = field(default_factory=list)


def vision_stage_shapes(img: int) -> list[tuple[int, int, int]]:
    """(C, H, W) produced by each DaViT stage for an `img` x `img` input."""
    out, res = [], img
    for i in range(4):
        res = (res + 2 * VIS_PAD[i] - VIS_KS[i]) // VIS_STRIDE[i] + 1
        out.append((VIS_DIM[i], res, res))
    return out


def denoise_schedule(steps: int) -> list[float]:
    """The reference's integrator times, in loop order.

    ``XVLAModel.generate_actions`` iterates ``for i in range(steps, 0, -1)`` and
    uses ``t = i / steps``, so the loop starts at t = 1 and ends at t = 1/steps.
    """
    return [i / steps for i in range(steps, 0, -1)]


def timestep_table(times: list[float], dim: int, max_period: int = 100) -> torch.Tensor:
    """``timestep_embedding`` for each loop timestep, as the reference computes it.

    The reference evaluates this in fp32 inside the loop; evaluating the same
    formula once and storing bf16 matches what the reduced-precision token
    buffer would round to anyway.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    out = torch.empty(len(times), dim, dtype=torch.float32)
    for i, tv in enumerate(times):
        args = torch.tensor(tv) * freqs
        out[i] = torch.cat([torch.cos(args), torch.sin(args)])
    return out.to(torch.bfloat16).cuda().contiguous()


def _window_tokens(c: int, h: int, w: int) -> int:
    nwh = (h + WINDOW - 1) // WINDOW
    nww = (w + WINDOW - 1) // WINDOW
    return nwh * nww * WINDOW * WINDOW * c


def _im2col_elems(c: int, h: int, w: int) -> int:
    return h * w * c * 9


class XVLARTXPipeline:
    """Native SM120 compute path for one X-VLA inference."""

    def __init__(
        self,
        gemm,
        fvk,
        fa2,
        weights: dict,
        *,
        num_sms: int = 0,
        img: int = 224,
        use_fp8: bool = False,
        use_fp4: bool = False,
    ) -> None:
        self.gemm = gemm
        self.fvk = fvk
        self.fa2 = fa2
        self.W = weights
        self.num_sms = num_sms
        self.img = img
        self.use_fp8 = use_fp8
        self.use_fp4 = use_fp4
        self.dims = XVLADims(img=img, vision=vision_stage_shapes(img))
        self.smallm_fp4 = USE_SMALLM_FP4_GEMM and hasattr(fvk, "xvla_fp4_gemm_bias_bf16out")
        self.fa2_m32 = USE_FA2_M32 and hasattr(fa2, "fwd_bf16_tile64m32w2_qnvfp4")
        self.use_pdl = USE_PDL
        self._allocate()
        self._graph = None
        self._graph_stream = None
        self._prefill_window_padding()

    def _prefill_window_padding(self) -> None:
        """Write each spatial block's static QKV bias into its padded window
        slots.  The per-call pad kernel writes only real tokens from then on, so
        these values persist across graph replays; each block has its own q/k/v
        region."""
        V = self.W["vision"]
        nv = N_VIEWS - 1
        st = self._st()
        for ci in range(4):
            c, h, w = self.dims.vision[ci]
            nwin = ((h + WINDOW - 1) // WINDOW) * ((w + WINDOW - 1) // WINDOW)
            for b in range(VIS_DEPTH[ci]):
                q, k, v = self.win_qkv[(ci, b)]
                bias = V["blocks"][ci][b]["spatial"]["qkv_b"]
                self.fvk.xvla_vision_window_qkv_pad_fill(
                    self._p(q), self._p(k), self._p(v), self._p(bias),
                    nv * nwin, WINDOW, c, VIS_HEADS[ci], VIS_HALF, st)

    # ── buffers ──────────────────────────────────────────────────────────

    def _allocate(self) -> None:
        dev = torch.device("cuda")
        bf = torch.bfloat16
        B: dict[str, torch.Tensor] = {}

        vn = N_VIEWS - 1  # camera views the tower batches over
        B["images"] = torch.zeros(vn, 3, SRC_IMG, SRC_IMG, dtype=torch.float32, device=dev)
        # DaViT works in NHWC; the reference's NCHW<->NHWC permutes around each
        # LayerNorm are pure layout and are folded away here.
        B["vimg"] = torch.zeros(vn, self.img, self.img, 3, dtype=bf, device=dev)
        sh = self.dims.vision
        nmax = max(h * w for _, h, w in sh)
        nv = max(h * w * c for c, h, w in sh) * vn  # token stream, [view*rows, C]
        wmax = max(_window_tokens(c, h, w) for c, h, w in sh) * vn
        B["v_a"] = torch.zeros(nv, dtype=bf, device=dev)
        B["v_b"] = torch.zeros(nv, dtype=bf, device=dev)
        B["v_qkv_w"] = torch.zeros(max(h * w * 3 * c for c, h, w in sh) * vn, dtype=bf, device=dev)
        B["v_attn_w"] = torch.zeros(wmax, dtype=bf, device=dev)
        B["v_qkv_c"] = torch.zeros(max(h * w * 3 * c for c, h, w in sh) * vn, dtype=bf, device=dev)
        B["v_attn_c"] = torch.zeros(nv, dtype=bf, device=dev)
        B["v_mlp"] = torch.zeros(max(h * w * 4 * c for c, h, w in sh) * vn, dtype=bf, device=dev)
        B["v_col"] = torch.zeros(max(_im2col_elems(c, h, w) for c, h, w in sh) * vn, dtype=bf, device=dev)
        # NVFP4 W4A4 activation scratch. Widest (M x K) is stage 0's fc2
        # input at 6272 x 1024; sized per stage pair below.
        v_rows = max(h * w for c, h, w in sh) * vn
        v_cols = max(4 * c for c, h, w in sh)
        B["v_a4"] = torch.empty(v_rows, v_cols // 2, dtype=torch.uint8, device=dev)
        B["v_asf4"] = torch.empty(
            self.fvk.nvfp4_sf_swizzled_bytes(v_rows, v_cols),
            dtype=torch.uint8, device=dev)

        # Channel-attention workspace: partial[b][g][s][hg][hg]. The largest
        # stage is 256 channels x 32-wide groups x 7 token splits.
        B["v_ca"] = torch.zeros(vn * 512 * 32 * 32, dtype=torch.float32, device=dev)
        # FA2 window-attention operands: head_dim 32 zero-padded to 64.  Each
        # spatial half-block owns a private q/k/v region: its padded slots hold
        # that block's static QKV bias (filled once at construction), and the
        # next block's real-slot writes must not clobber them.
        wp = max(_window_tokens(c, h, w) for c, h, w in sh) * vn
        self.win_qkv = {}
        for ci, (c, h, w) in enumerate(sh):
            nwin = ((h + WINDOW - 1) // WINDOW) * ((w + WINDOW - 1) // WINDOW)
            elems = vn * nwin * WINDOW * WINDOW * VIS_HEADS[ci] * 64
            for b in range(VIS_DEPTH[ci]):
                self.win_qkv[(ci, b)] = tuple(
                    torch.zeros(elems, dtype=bf, device=dev) for _ in range(3))
        B["v_op"] = torch.zeros(wp // 32 * 64, dtype=bf, device=dev)
        B["v_wlse"] = torch.zeros(128, 64, 144, dtype=torch.float32, device=dev)
        # Packed projector input: 1 pooled + 49 patch tokens at the DaViT width.
        B["v_proj"] = torch.zeros(VISION_TOKENS, VIS_OUT_DIM, dtype=bf, device=dev)
        B["v_tok_h"] = torch.zeros(VISION_TOKENS, VIS_TOK_DIM, dtype=bf, device=dev)
        B["vision_tokens"] = torch.zeros(N_VIEWS, VISION_TOKENS, VIS_TOK_DIM, dtype=bf, device=dev)
        B["enc_x"] = torch.zeros(ENC_SEQ, HIDDEN, dtype=bf, device=dev)
        B["enc_h"] = torch.zeros(ENC_SEQ, HIDDEN, dtype=bf, device=dev)
        B["enc_qkv"] = torch.zeros(ENC_SEQ, 3 * HIDDEN, dtype=bf, device=dev)
        B["enc_attn"] = torch.zeros(ENC_SEQ, HIDDEN, dtype=bf, device=dev)
        B["enc_mlp"] = torch.zeros(ENC_SEQ, ENC_MLP, dtype=bf, device=dev)
        B["enc_lse"] = torch.zeros(1, ENC_HEADS, ENC_SEQ, dtype=torch.float32, device=dev)
        # Block-128 split-K partials: [k_split, M, N] fp32.
        B["splitk_scratch"] = torch.zeros(4, ENC_SEQ, HIDDEN, dtype=torch.float32, device=dev)

        B["act_tok"] = torch.zeros(CHUNK, ACTION_DIM + DIM_PROPRIO + DIM_TIME, dtype=bf, device=dev)
        B["x1"] = torch.zeros(CHUNK, ACTION_DIM, dtype=bf, device=dev)
        B["action"] = torch.zeros(CHUNK, ACTION_DIM, dtype=bf, device=dev)
        B["act_x"] = torch.zeros(SEQ, HIDDEN, dtype=bf, device=dev)
        # SoftPromptedTransformer is stateless across denoising steps: every
        # step rebuilds the sequence from the freshly projected prefix. This
        # buffer holds that prefix (rows CHUNK..SEQ) pristine, and each step
        # restores it before running the blocks.
        B["act_prefix"] = torch.zeros(SEQ - CHUNK, HIDDEN, dtype=bf, device=dev)
        B["act_h"] = torch.zeros(SEQ, HIDDEN, dtype=bf, device=dev)
        B["act_qkv"] = torch.zeros(SEQ, 3 * HIDDEN, dtype=bf, device=dev)
        B["act_attn"] = torch.zeros(SEQ, HIDDEN, dtype=bf, device=dev)
        B["act_mlp"] = torch.zeros(SEQ, ACT_MLP, dtype=bf, device=dev)
        B["act_lse"] = torch.zeros(1, ACT_HEADS, SEQ, dtype=torch.float32, device=dev)
        B["out"] = torch.zeros(CHUNK, ACTION_DIM, dtype=torch.float32, device=dev)
        if self.use_fp8 or self.use_fp4:
            # Activation quantization scratch for the four quantized transformer
            # GEMMs (K is 1024 or 4096).
            for k in (HIDDEN, ACT_MLP):
                B[f"a8_{k}"] = torch.empty(SEQ, k, dtype=torch.uint8, device=dev)
                B[f"asc_{k}"] = torch.empty(SEQ, k // 128, dtype=torch.float32, device=dev)
        if self.use_fp4:
            for k in (HIDDEN, ACT_MLP):
                B[f"a4_{k}"] = torch.empty(SEQ, k // 2, dtype=torch.uint8, device=dev)
                B[f"asf4_{k}"] = torch.empty(self.fvk.nvfp4_sf_swizzled_bytes(SEQ, k),
                                             dtype=torch.uint8, device=dev)
        if self.use_fp4 and USE_FP4OUT_FFN:
            # Padded to FP4OUT_M rows; the pad rows are never written and cost
            # nothing extra (262 rows already need three 128-row tiles).
            B["a4m_1024"] = torch.zeros(FP4OUT_M, HIDDEN // 2, dtype=torch.uint8, device=dev)
            B["mlp_pack"] = torch.zeros(FP4OUT_M, ACT_MLP // 2, dtype=torch.uint8, device=dev)
            B["mlp_sf"] = torch.zeros(
                self.fvk.nvfp4_sf_swizzled_bytes(FP4OUT_M, ACT_MLP),
                dtype=torch.uint8, device=dev)
        # Raw proprio staging: the reference's pad_vector / gripper zeroing run
        # natively in xvla_proprio_bf16, straight from this fp32 reading.
        B["state_in"] = torch.zeros(DIM_PROPRIO, dtype=torch.float32, device=dev)
        self.times = denoise_schedule(N_DENOISE)
        B["time_tab"] = timestep_table(self.times, DIM_TIME)

        self.B = B

    # ── small helpers ────────────────────────────────────────────────────

    def _st(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    @staticmethod
    def _p(t: torch.Tensor) -> int:
        return t.data_ptr()

    def _ln(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        self.fvk.layer_norm_pdl(self._p(x), self._p(w), self._p(b), self._p(out),
                            x.shape[0], x.shape[1], LN_EPS, self._st())

    def _gemm_bias(self, a, w, bias, out, m: int | None = None, wt=None, fp8=None,
                   fp4=None, splitk: int = 0, prequant: bool = False) -> None:
        """out[M,N] = a[M,K] @ w[N,K]^T + bias, at the configured weight precision.

        ``w`` is the natural ``nn.Linear`` layout. ``wt`` is an optional
        pre-transposed [K,N] copy for shapes the CUTLASS tile rejects
        (the 20-wide action decoder is below its minimum N).
        """
        M = a.shape[0] if m is None else m
        K = a.shape[1]
        N = w.shape[0]
        st = self._st()
        if fp4 is not None and self.use_fp4:
            # NVFP4 W4A4: 4-bit weights with per-16-element e4m3 scales (static,
            # from the checkpoint) and the same swizzled activation quantization
            # the repository's SM120 NVFP4 callers use.
            packed, sfb, alpha = fp4
            a_packed = self.B[f"a4_{K}"]
            a_sfa = self.B[f"asf4_{K}"]
            # The `_v2` entry is bit-identical to `quantize_bf16_to_nvfp4_swizzled`
            # but stages the row in shared memory instead of reading it twice
            # (12.9 -> 6.6 us at the 4096-wide FFN midpoint).
            if not prequant:
                self.fvk.quantize_bf16_to_nvfp4_swizzled_v3(self._p(a), self._p(a_packed),
                                                            self._p(a_sfa), M, K, st)
            self._fp4_gemm(a_packed, packed, out, bias, M, N, K, a_sfa, sfb, alpha)
            return
        if fp8 is not None:
            # Block-128 FP8: weights carry static per-128x128-block scales from
            # the checkpoint; activations are quantized per token per 128-K
            # block by a fixed, graph-captured kernel (no host sync), which is
            # the same dataflow the repository's SM120 FP8 callers use.
            q8, wsc = fp8
            a8 = self.B[f"a8_{K}"]
            asc = self.B[f"asc_{K}"]
            self.fvk.fp8_per_token_block128_quant_bf16_v2(self._p(a), self._p(a8),
                                                       self._p(asc), M, K, st)
            if splitk:
                # The repository's block-128 split-K entrant, used by its own
                # caller at M=41. Measured at this model's encoder shapes it wins
                # by 1.85x on the two N=1024 GEMMs (the CUTLASS tile sees only
                # eight CTAs when M=100) and loses on the N>=3072 ones. It has no
                # bias argument, so the bias follows as its own launch.
                rc = self.fvk.splitk_b128_fp8_gemm_32x64x128_w4(
                    self._p(a8), self._p(q8), self._p(asc), self._p(wsc),
                    self._p(out), M, N, K, K // 128, splitk,
                    self._p(self.B["splitk_scratch"]), st)
                if rc != 0:
                    raise RuntimeError(
                        "splitk_b128_fp8_gemm_32x64x128_w4 failed "
                        f"(rc={rc}) at {M}x{N}x{K}"
                    )
                self.fvk.add_bias_bf16_pdl(self._p(out), self._p(bias), M, N, st)
                return
            self.fvk.fp8_block128_gemm_cutlass_sm120_bf16out_bias(
                self._p(a8), self._p(q8), self._p(out), self._p(bias), M, N, K,
                self._p(asc), self._p(wsc), st)
            return
        if wt is not None:
            # D = A @ B with B stored (K, N).
            self.gemm.bf16_nn(self._p(a), self._p(wt), self._p(out), M, N, K, st)
        else:
            self.fvk.tq_cutlass_bf16_gemm(self._p(a), self._p(w), self._p(out), M, N, K, st)
        # `GemmRunner.bf16_nn_bias` returns CUBLAS_STATUS_NOT_SUPPORTED on
        # cuBLAS 13 (it is an AMD-only entry in this tree); every NVIDIA caller
        # pairs the GEMM with the native add_bias_bf16, so do the same.
        self.fvk.add_bias_bf16_pdl(self._p(out), self._p(bias), M, N, st)

    def _norm_quant_fp4(self, x, w, b, rows: int, cols: int) -> None:
        """LayerNorm straight into the swizzled NVFP4 activation buffer.

        Bit-identical to the repository's `layer_norm_to_nvfp4_swizzled_bf16`,
        which reads its input three times (mean, variance, normalize); this
        stages the row once in shared memory.
        """
        self.fvk.xvla_ln_quant_bf16(
            self._p(x), self._p(w), self._p(b), self._p(self.B[f"a4_{cols}"]),
            self._p(self.B[f"asf4_{cols}"]), rows, cols, LN_EPS, self._st())

    def _add_norm_quant_fp4(self, x, h, w, b, rows: int, cols: int,
                            packed=None, sf=None) -> None:
        """`x += h` then LayerNorm(x) straight into the NVFP4 activation buffer.

        Every block's sublayer output is consumed by the *next* LayerNorm, so
        the residual add folds into that norm's staging pass: two launches and
        one round trip through memory become one launch, with `x` carrying the
        running sum exactly as `residual_add` would have left it.
        """
        self.fvk.xvla_add_ln_quant_bf16(
            self._p(x), self._p(h), self._p(w), self._p(b),
            self._p(packed if packed is not None else self.B[f"a4_{cols}"]),
            self._p(sf if sf is not None else self.B[f"asf4_{cols}"]),
            rows, cols, LN_EPS, self._st())

    def _ffn_fp4(self, L: dict, h_out, rows: int = SEQ) -> None:
        """fc1 + bias + GELU, then quantize, then fc2.

        `a4_{HIDDEN}` already holds the packed fc1 input: `denoise_step` folds
        the sublayer residual into the n2 LayerNorm-quantize immediately before
        this call, so re-running the norm here would be a redundant launch.

        The FFN midpoint's standalone quantize uses the repository's `_v2`
        entry, which is bit-identical and half the time at 4096 columns.
        """
        fvk, W, B = self.fvk, self.W, self.B
        fc1_p, fc1_sf, fc1_a = L["fp4"]["fc1"]
        fc2_p, fc2_sf, fc2_a = L["fp4"]["fc2"]
        st = self._st()
        if USE_FP4OUT_FFN and "a4m_1024" in B and self.smallm_fp4:
            # Rows past `rows` are never read (fc2 runs on `rows`), so the
            # small-M epilogue writes only those.
            rc = fvk.xvla_fp4_gemm_bias_gelu_fp4out(
                self._p(B["a4m_1024"]), self._p(fc1_p), self._p(B["asf4_1024"]),
                self._p(fc1_sf), self._p(L["fc1_b"]), self._p(B["mlp_pack"]),
                self._p(B["mlp_sf"]), rows, ACT_MLP, HIDDEN, fc1_a, st)
            if rc != 0:
                raise RuntimeError(
                    f"xvla_fp4_gemm_bias_gelu_fp4out failed (rc={rc}) at "
                    f"{rows}x{ACT_MLP}x{HIDDEN}"
                )
            self._gemm_fc2(fc2_p, fc2_sf, fc2_a, L["fc2_b"], h_out, B["mlp_pack"],
                           B["mlp_sf"], rows)
            return
        if USE_FP4OUT_FFN and "a4m_1024" in B:
            fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
                self._p(B["a4m_1024"]), self._p(fc1_p), self._p(B["asf4_1024"]),
                self._p(fc1_sf), self._p(L["fc1_b"]), self._p(B["mlp_pack"]),
                self._p(B["mlp_sf"]), FP4OUT_M if rows > CHUNK else FP4OUT_M_LAST,
                ACT_MLP, HIDDEN, fc1_a, st)
            self._gemm_fc2(fc2_p, fc2_sf, fc2_a, L["fc2_b"], h_out, B["mlp_pack"],
                           B["mlp_sf"], rows)
            return
        if rows != SEQ:
            raise ValueError(
                f"the non-fp4out FFN path expects the full sequence ({SEQ} rows), "
                f"got {rows}"
            )
        fvk.fp4_w4a16_gemm_bias_gelu_bf16out_sm120(
            self._p(B[f"a4_{HIDDEN}"]), self._p(fc1_p),
            self._p(B[f"asf4_{HIDDEN}"]), self._p(fc1_sf), self._p(L["fc1_b"]),
            self._p(B["act_mlp"]), SEQ, ACT_MLP, HIDDEN, fc1_a, st)
        fvk.quantize_bf16_to_nvfp4_swizzled_v3(
            self._p(B["act_mlp"]), self._p(B[f"a4_{ACT_MLP}"]),
            self._p(B[f"asf4_{ACT_MLP}"]), SEQ, ACT_MLP, st)
        self._gemm_fc2(fc2_p, fc2_sf, fc2_a, L["fc2_b"], h_out,
                       B[f"a4_{ACT_MLP}"], B[f"asf4_{ACT_MLP}"])

    def _gemm_fc2(self, packed, sfb, alpha, bias, out, a_packed, a_sfa,
                  rows: int = SEQ) -> None:
        """The FFN down projection: N=HIDDEN (1024), K=ACT_MLP (4096).

        Measured against the other two SM120 NVFP4 entrants at this exact shape,
        the plain `..._bias` schedule (17.16 us) beats the pingpong schedule
        (17.73 us) by 3%; at the qkv, proj and fc1 shapes pingpong wins by 10-20%,
        so the schedule stays per shape. The header of the tile module documents
        these as explicit opt-ins for exactly this A/B.
        """
        self._fp4_gemm(a_packed, packed, out, bias, rows, HIDDEN, ACT_MLP, a_sfa, sfb, alpha,
                       schedule="bias")

    def _fp4_gemm(self, a_packed, packed, out, bias, M: int, N: int, K: int, a_sfa, sfb,
                  alpha: float, schedule: str = "pingpong_bias") -> None:
        """out[M,N] (bf16) = NVFP4 A[M,K] . W[N,K]^T + bias, both operands prepacked.

        The small-M kernel when built (bit-identical to either cutlass schedule),
        else the per-shape cutlass schedule.
        """
        if self.smallm_fp4:
            rc = self.fvk.xvla_fp4_gemm_bias_bf16out(
                self._p(a_packed), self._p(packed), self._p(out), self._p(bias),
                M, N, K, self._p(a_sfa), self._p(sfb), alpha, self._st())
            if rc != 0:
                raise RuntimeError(
                    f"xvla_fp4_gemm_bias_bf16out failed (rc={rc}) at {M}x{N}x{K}"
                )
            return
        getattr(self.fvk, f"fp4_w4a16_gemm_sm120_bf16out_{schedule}")(
            self._p(a_packed), self._p(packed), self._p(out), self._p(bias),
            M, N, K, self._p(a_sfa), self._p(sfb), alpha, self._st())

    def _gemm_fp4_prequant(self, key: int, L: dict, name: str, bias, out,
                           M: int, N: int, K: int) -> None:
        """NVFP4 GEMM on an activation already packed by `_norm_quant_fp4`."""
        packed, sfb, alpha = L["fp4"][name]
        self._fp4_gemm(self.B[f"a4_{key}"], packed, out, bias, M, N, K,
                       self.B[f"asf4_{key}"], sfb, alpha)

    def _attn(self, qkv: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, seq: int,
              heads: int, fp4_out=None, q_rows: int | None = None) -> None:
        """Full (unmasked) self-attention straight out of a fused QKV buffer.

        q/k/v are stride-views into ``qkv``, so no split or copy is needed.

        ``fp4_out`` is an optional ``(packed, scales, cols)`` triple: the kernel
        then quantizes its own O tile to NVFP4 out of the epilogue that already
        holds it, and the consumer GEMM reads that buffer directly instead of a
        standalone quantize launch writing it. Byte-identical to running
        ``quantize_bf16_to_nvfp4_swizzled`` over ``out`` afterwards (verified at
        this shape), and 0.5 ms cheaper across the denoise loop's 240 calls.
        """
        hd = HIDDEN // heads
        base = self._p(qkv)
        qs = (seq * 3 * HIDDEN, 3 * HIDDEN, hd)
        # Every attention in this model has a 64-wide head, so take the entry
        # whose template head width matches it: the short-query dispatch this
        # replaces routes head_dim<=96 to a 96-wide template with a 32-wide KV
        # tile, which measured 19.93 us against 14.56 us at the denoise shape.
        # `hd == 64` is static here (HIDDEN // heads, both module constants).
        attn = (self.fa2.fwd_bf16_tile64 if hd == 64 else self.fa2.fwd_bf16_tile)
        # seqused_k must be NULL for full attention; a pointer to zeros would
        # be read as a KV length of 0.
        qs_o = (seq * HIDDEN, HIDDEN, hd)
        if fp4_out is not None:
            packed, sfa, cols = fp4_out
            # Only the fused NVFP4 output is consumed (the projection below is
            # prequantized), so the bf16 O tile is dead traffic: 536 KB per call
            # at this shape. Pass a null O and the epilogue skips its store;
            # measured 25.077 -> 25.038 ms equal-scope, delivery metrics
            # bit-identical.
            fused = (self.fa2.fwd_bf16_tile64m32w2_qnvfp4 if self.fa2_m32
                     else self.fa2.fwd_bf16_tile64_qnvfp4)
            fused(
                base, base + HIDDEN * 2, base + HIDDEN * 4, 0, self._p(lse), 0,
                self._p(packed), self._p(sfa), cols // 2, (cols // 16 + 3) // 4,
                1, seq if q_rows is None else q_rows, seq, heads, heads, hd,
                qs, qs, qs, qs_o, hd ** -0.5, self.num_sms, self._st(), enable_pdl=self.use_pdl,
            )
            return
        attn(
            base, base + HIDDEN * 2, base + HIDDEN * 4, self._p(out), self._p(lse), 0,
            1, seq, seq, heads, heads, hd,
            qs, qs, qs, (seq * HIDDEN, HIDDEN, hd),
            hd ** -0.5, self.num_sms, self._st(),
            **({"enable_pdl": self.use_pdl} if hd == 64 else {}),
        )

    # ── stage 2: BART encoder ────────────────────────────────────────────

    def encode_forward(self) -> None:
        """``[vision(cam0) ++ text] -> 100 x 1024``, in place on ``enc_x``."""
        fvk, W, B, st = self.fvk, self.W, self.B, self._st()
        x, h = B["enc_x"], B["enc_h"]
        n = ENC_SEQ * HIDDEN

        fvk.xvla_add2_bf16(self._p(x), self._p(W["enc_pos"]), self._p(h), n, st)
        self._ln(h, W["enc_ln_embed_w"], W["enc_ln_embed_b"], x)

        # BART is post-norm: the LayerNorm follows each residual add, not the
        # sub-layer input (the action transformer below is pre-norm).
        enc_fp4 = USE_FP4_ENCODER and self.use_fp4 and bool(W["enc"][0].get("fp4"))
        enc_ln_fused = enc_fp4 and hasattr(self.fvk, "xvla_add_ln_quant_norm_bf16")

        a4, asf = B["a4_1024"], B["asf4_1024"]

        def enc_gemm(a, key, bias, out, splitk=0, prequant=False):
            L = W["enc"][li]
            L4 = L["fp4"].get(key) if enc_fp4 else None
            if L4 is not None:
                self._gemm_bias(a, L[f"{key}_w"], bias, out, fp4=L4, prequant=prequant)
            else:
                self._gemm_bias(a, L[f"{key}_w"], bias, out,
                                fp8=L["fp8"].get(key), splitk=splitk)

        for li in range(ENC_LAYERS):
            L = W["enc"][li]
            enc_gemm(x, "qkv", L["qkv_b"], B["enc_qkv"])
            self._attn(B["enc_qkv"], B["enc_attn"], B["enc_lse"], ENC_SEQ, ENC_HEADS,
                       fp4_out=((a4, asf, HIDDEN) if enc_ln_fused else None))
            enc_gemm(B["enc_attn"], "proj", L["proj_b"], h, splitk=1,
                     prequant=enc_ln_fused)
            if enc_ln_fused:
                # One launch for fold + LayerNorm + quantize, writing both the
                # normalised bf16 row (the encoder is post-norm, so x carries it)
                # and the packed operand. The kernel reproduces
                # layer_norm_kernel's reduction order exactly.
                fvk.xvla_add_ln_quant_norm_bf16(
                    self._p(x), self._p(h), self._p(x), self._p(L["ln1_w"]),
                    self._p(L["ln1_b"]), self._p(a4), self._p(asf),
                    ENC_SEQ, HIDDEN, LN_EPS, st)
            else:
                fvk.residual_add_pdl(self._p(x), self._p(h), n, st)
                self._ln(x, L["ln1_w"], L["ln1_b"], x)

            enc_gemm(x, "fc1", L["fc1_b"], B["enc_mlp"], prequant=enc_ln_fused)
            fvk.gelu_erf_bf16(self._p(B["enc_mlp"]), ENC_SEQ * ENC_MLP, st)
            enc_gemm(B["enc_mlp"], "fc2", L["fc2_b"], h, splitk=4)
            if enc_ln_fused:
                fvk.xvla_add_ln_quant_norm_bf16(
                    self._p(x), self._p(h), self._p(x), self._p(L["ln2_w"]),
                    self._p(L["ln2_b"]), self._p(a4), self._p(asf),
                    ENC_SEQ, HIDDEN, LN_EPS, st)
            else:
                fvk.residual_add_pdl(self._p(x), self._p(h), n, st)
                self._ln(x, L["ln2_w"], L["ln2_b"], x)

    def encode_assemble(self) -> None:
        """``enc_x = [vision(cam0) | text_embeds]`` — the encoder's input."""
        fvk, B, st = self.fvk, self.B, self._st()
        fvk.xvla_copy_bf16(self._p(B["vision_tokens"][0]), self._p(B["enc_x"]), VISION_TOKENS * HIDDEN, st)
        fvk.xvla_copy_bf16(self._p(self.W["text_emb"]), self._p(B["enc_x"][VISION_TOKENS]),
                           TEXT_LEN * HIDDEN, st)

    # ── stage 3: flow-matching denoise loop ──────────────────────────────

    def denoise_prepare(self) -> None:
        """Project the constant transformer segments and fold positional bias.

        ``act_x = [action(30) | vlm_proj(vlm) | aux_visual_proj(aux) | soft(32)]``
        with ``pos_emb`` added to the first 230 rows only — the reference appends
        the soft prompts *after* the position add, so they carry no position.
        """
        fvk, W, B, st = self.fvk, self.W, self.B, self._st()
        pre = B["act_prefix"]

        self._gemm_bias(B["enc_x"], W["vlm_proj_w"], W["vlm_proj_b"], pre[:ENC_SEQ])
        self._gemm_bias(B["vision_tokens"][1:].reshape(AUX_TOKENS, VIS_TOK_DIM),
                        W["aux_proj_w"], W["aux_proj_b"], pre[ENC_SEQ:PREFIX - CHUNK])
        fvk.xvla_copy_bf16(self._p(W["soft"]), self._p(pre[PREFIX - CHUNK]), N_SOFT * HIDDEN, st)
        fvk.xvla_proprio_bf16(self._p(B["state_in"]), self._p(B["act_tok"]), CHUNK,
                              ACTION_DIM + DIM_PROPRIO + DIM_TIME, ACTION_DIM,
                              DIM_PROPRIO, W["n_state"], GRIPPER[0], GRIPPER[1], st)
        fvk.xvla_add2_bf16(self._p(pre), self._p(W["pos_emb"][CHUNK]), self._p(pre),
                           (PREFIX - CHUNK) * HIDDEN, st)

    def _n2dst(self):
        """Destination for the n2 fold-norm-quantize: the padded fc1 pack input
        when the fp4-out FFN epilogue is in use, otherwise the standard one."""
        if self.use_fp4 and USE_FP4OUT_FFN and "a4m_1024" in self.B:
            return (self.B["a4m_1024"], self.B["asf4_1024"])
        return ()

    def denoise_step(self, step: int) -> None:
        fvk, W, B, st = self.fvk, self.W, self.B, self._st()
        x, h = B["act_x"], B["act_h"]
        n = SEQ * HIDDEN

        # Rebuild the stateless sequence: the prefix is the same fresh
        # projection every step, only the action segment changes.
        fvk.xvla_copy_bf16(self._p(B["act_prefix"]), self._p(x[CHUNK]), (SEQ - CHUNK) * HIDDEN, st)

        # Euler step folded into the action-token assembly: act_tok[:, :20]
        # becomes x_t with the gripper channels zeroed; proprio and the time
        # embedding occupy columns 20..71 and never change.
        tok_stride = ACTION_DIM + DIM_PROPRIO + DIM_TIME
        fvk.xvla_flow_step_bf16(
            self._p(B["x1"]), self._p(B["action"]), self._p(B["act_tok"]),
            self._p(B["time_tab"][step]), self.times[step], CHUNK,
            ACTION_DIM, tok_stride, GRIPPER[0], GRIPPER[1],
            ACTION_DIM + DIM_PROPRIO, DIM_TIME, st)
        self._gemm_bias(B["act_tok"], W["action_encoder_w"], W["action_encoder_b"], h)
        fvk.xvla_add2_bf16(self._p(h), self._p(W["pos_emb"]), self._p(x), CHUNK * HIDDEN, st)

        for li in range(ACT_LAYERS):
            L = W["act"][li]
            # Rows the block must produce: all SEQ, except the last block when
            # only the action rows are read afterwards (PRUNE_LAST_BLOCK).
            rows = (CHUNK if self.use_fp4 and PRUNE_LAST_BLOCK and li == ACT_LAYERS - 1
                    else SEQ)
            if self.use_fp4:
                # `h` still holds the previous block's fc2 output, so the
                # residual add that would have run at the end of that block
                # folds into this norm. The first block has no pending addend.
                if li == 0:
                    self._norm_quant_fp4(x, L["n1w"], L["n1b"], SEQ, HIDDEN)
                else:
                    self._add_norm_quant_fp4(x, h, L["n1w"], L["n1b"], SEQ, HIDDEN)
                self._gemm_fp4_prequant(HIDDEN, L, "qkv", L["qkv_b"], B["act_qkv"],
                                        SEQ, 3 * HIDDEN, HIDDEN)
            else:
                self._ln(x, L["n1w"], L["n1b"], h)
                self._gemm_bias(h, L["qkv_w"], L["qkv_b"], B["act_qkv"], fp8=L["fp8"].get("qkv") if self.use_fp8 else None)
            self._attn(B["act_qkv"], B["act_attn"], B["act_lse"], SEQ, ACT_HEADS,
                       fp4_out=((B["a4_1024"], B["asf4_1024"], HIDDEN)
                                if self.use_fp4 else None), q_rows=rows)
            self._gemm_bias(B["act_attn"], L["proj_w"], L["proj_b"], h, m=rows,
                fp8=L["fp8"].get("proj") if self.use_fp8 else None,
                fp4=L["fp4"].get("proj") if self.use_fp4 else None,
                prequant=self.use_fp4)

            if self.use_fp4:
                self._add_norm_quant_fp4(x, h, L["n2w"], L["n2b"], rows, HIDDEN,
                                         *self._n2dst())
                self._ffn_fp4(L, h, rows)
            else:
                fvk.residual_add_pdl(self._p(x), self._p(h), n, st)
                self._ln(x, L["n2w"], L["n2b"], h)
                self._gemm_bias(h, L["fc1_w"], L["fc1_b"], B["act_mlp"], fp8=L["fp8"].get("fc1") if self.use_fp8 else None)
                fvk.gelu_inplace(self._p(B["act_mlp"]), SEQ * ACT_MLP, st)
                self._gemm_bias(B["act_mlp"], L["fc2_w"], L["fc2_b"], h, fp8=L["fp8"].get("fc2") if self.use_fp8 else None)
                fvk.residual_add_pdl(self._p(x), self._p(h), n, st)

        if self.use_fp4:
            # The last block's fc2 output has no following norm to fold into.
            fvk.residual_add_pdl(self._p(x), self._p(h),
                             CHUNK * HIDDEN if PRUNE_LAST_BLOCK else n, st)
        self._ln(x[:CHUNK], W["norm_w"], W["norm_b"], h)
        if USE_DECODER_GEMV:
            fvk.xvla_action_decoder_gemv(
                self._p(h), self._p(W["action_decoder_w"]),
                self._p(W["action_decoder_b"]), self._p(B["action"]),
                CHUNK, ACTION_DIM, HIDDEN, self._st())
        else:
            self._gemm_bias(h, W["action_decoder_w"], W["action_decoder_b"], B["action"],
                            m=CHUNK, wt=W["action_decoder_wt"])

    def denoise_forward(self) -> None:
        B = self.B
        for step in range(N_DENOISE):
            self.denoise_step(step)
        self.fvk.xvla_finalize_bf16(self._p(B["action"]), self._p(B["out"]),
                                    CHUNK, ACTION_DIM, ACTION_DIM,
                                    GRIPPER[0], GRIPPER[1], self._st())

    # ── stage 1: DaViT vision tower ──────────────────────────────────────

    def _vgemm(self, a, w, out, M: int, N: int, K: int, bias=None,
               prequant: bool = False, fp4=None, a_packed=None,
               a_sfa_ptr: int | None = None) -> None:
        """DaViT GEMM: D[M,N] = A[M,K] @ W[K,N].

        `fp4` selects the tower's NVFP4 W4A4 path; the convolution's
        K = Cin*k*k is not a multiple of 256, so it stays on the bf16 runner.
        Passing `bias` folds the add_bias launch into whichever GEMM's epilogue.
        """
        st = self._st()
        if fp4 is not None:
            wp, wsf, wa = fp4
            ap = self.B["v_a4"] if a_packed is None else a_packed
            asf = self._p(self.B["v_asf4"]) if a_sfa_ptr is None else a_sfa_ptr
            if not prequant:
                self.fvk.quantize_bf16_to_nvfp4_swizzled_v3(
                    self._p(a), self._p(ap), asf, M, K, st)
            # Pingpong, not the plain schedule: swept at the tower's ten real
            # (M, N, K) shapes on real activations it wins by 5-30%, mean ~20%.
            self.fvk.fp4_w4a16_gemm_sm120_bf16out_pingpong_bias(
                self._p(ap), self._p(wp), self._p(out), self._p(bias),
                M, N, K, asf, self._p(wsf), wa, st)
            return
        self.gemm.bf16_nn(self._p(a), self._p(w), self._p(out), M, N, K, st)
        if bias is not None:
            # `bf16_nn_bias` returns CUBLAS_STATUS_NOT_SUPPORTED on cuBLAS 13
            # (an AMD-only entry in this tree, as in `_gemm_bias`); every NVIDIA
            # caller pairs the GEMM with the native add_bias_bf16.
            self.fvk.add_bias_bf16_pdl(self._p(out), self._p(bias), M, N, st)

    def _vnorm(self, x, w, b, out, rows: int, dim: int) -> None:
        # The conv-embed norms run one block per row with one element per thread
        # in the repository entry, which at dim <= 256 leaves half the block idle
        # while both halves pay for the reductions. ln_bf16 uses the fused
        # entry's arithmetic with the same tree, so at dim2 <= 128 it reproduces
        # layer_norm_kernel bit for bit and is measurably faster.
        if dim <= 256:
            self.fvk.xvla_ln_bf16(self._p(x), self._p(out), self._p(w), self._p(b),
                                  rows, dim, LN_EPS, self._st())
            return
        self.fvk.layer_norm_pdl(self._p(x), self._p(w), self._p(b), self._p(out),
                            rows, dim, LN_EPS, self._st())

    def _conv_embed(self, ci: int) -> None:
        """One DaViT ConvEmbed stage: LayerNorm, strided conv (im2col + GEMM), bias."""
        fvk, V, B, st = self.fvk, self.W["vision"], self.B, self._st()
        c, h, w = self.dims.vision[ci]
        prev = (3, self.img, self.img) if ci == 0 else self.dims.vision[ci - 1]
        _, hin, win_ = prev
        cin = prev[0]
        k, stride, pad = VIS_KS[ci], VIS_STRIDE[ci], VIS_PAD[ci]
        rows = h * w
        nv = N_VIEWS - 1
        nrm = B["v_b"]

        # Stage 0 normalizes after the conv, the rest before it.
        if VIS_PRENORM[ci]:
            self._vnorm(B["v_a"], V["convs"][ci]["nw"], V["convs"][ci]["nb"], nrm,
                        nv * hin * win_, cin)
            src = nrm
        else:
            src = B["v_a"]

        fvk.xvla_vision_im2col(self._p(src), self._p(B["v_col"]), hin, win_, cin,
                               k, stride, pad, h, w, nv, VIS_HALF, st)
        self._vgemm(B["v_col"], V["convs"][ci]["w_t"], B["v_a"], nv * rows, c,
                    cin * k * k, bias=V["convs"][ci]["b"])
        if not VIS_PRENORM[ci]:
            # Stage 0 normalizes after the conv; normalize in place so the
            # stage result always lands in v_a for the block loop.
            self._vnorm(B["v_a"], V["convs"][ci]["nw"], V["convs"][ci]["nb"],
                        B["v_a"], nv * rows, c)

    def _vision_block(self, ci: int, blk: dict, sub: str, h: int, w: int, c: int,
                      heads: int, groups: int, wb: int) -> None:
        """One half of a DaViT block: depthwise conv + residual, then either
        window attention (spatial) or grouped channel attention (channel)."""
        fvk, V, B, st = self.fvk, self.W["vision"], self.B, self._st()
        S = blk[sub]
        rows = h * w
        nv = N_VIEWS - 1

        fvk.xvla_vision_dwconv3x3_res(self._p(B["v_a"]), self._p(S["conv1_w"]),
                                      self._p(S["conv1_b"]), self._p(B["v_b"]), h, w,
                                      c, nv, VIS_HALF, st)
        # LayerNorm straight into the packed A operand the QKV GEMM reads: the
        # normed bf16 row is never written and the in-GEMM quantize launch
        # disappears with it. The dwconv folded its own residual into `v_b`, so
        # `v_a` is free here and the half-block's residual is `v_b` at the end.
        # 256 threads, matching layer_norm_kernel's reduction order: fusing the
        # norm into the operand must not move the delivery gate.
        self.fvk.xvla_ln_quant_bf16_nt256(
            self._p(B["v_b"]), self._p(S["n1w"]), self._p(S["n1b"]),
            self._p(B["v_a4"]), self._p(B["v_asf4"]), nv * rows, c, LN_EPS, st)
        a_qkv, qkv_kw = B["v_a4"], {"fp4": S["qkv4"], "prequant": True}

        if sub == "spatial":
            # The QKV and the projection run on the tokens in image order; only
            # the FA2 operands carry the padded window layout. A padded row has
            # q = v = 0, so its attention output is 0 and its projection is the
            # bias alone — the row is discarded either way, and computing it
            # cost 2.9x the rows at stages 2 and 3 plus a gather and a scatter.
            nwin = (h + WINDOW - 1) // WINDOW * ((w + WINDOW - 1) // WINDOW)
            nw = nv * nwin
            self._vgemm(a_qkv, S["qkv_w_t"], B["v_qkv_w"], nv * rows, 3 * c, c,
                        bias=S["qkv_b"], **qkv_kw)
            qw, kw_, vw = self.win_qkv[(ci, wb)]
            fvk.xvla_vision_window_qkv_pad(self._p(B["v_qkv_w"]),
                                           self._p(S["qkv_b"]), self._p(qw),
                                           self._p(kw_), self._p(vw),
                                           h, w, WINDOW, c, heads, nwin, nv,
                                           VIS_HALF, st)
            o64 = (nw * WINDOW * WINDOW * heads * 64)
            stq = (o64 // nw, heads * 64, 64)
            # Same 64-wide-head tile as the transformer attention (the DaViT
            # head is 32 wide and is zero-padded into 64 by the QKV pad).
            self.fa2.fwd_bf16_tile64(self._p(qw), self._p(kw_), self._p(vw),
                                   self._p(B["v_op"]),
                                   self._p(B["v_wlse"]), 0, nw,
                                   WINDOW * WINDOW, WINDOW * WINDOW, heads, heads, 64,
                                   stq, stq, stq, stq, 32 ** -0.5, self.num_sms, st,
                                   enable_pdl=self.use_pdl)
            # The unpad packs the NVFP4 operand itself: the row is staged in
            # shared memory rather than written out and read back.
            fvk.xvla_vision_window_out_unpad_to_nvfp4(
                self._p(B["v_a4"]), self._p(B["v_asf4"]), self._p(B["v_op"]),
                h, w, WINDOW, c, heads, nv, VIS_HALF, st)
            proj_kw = dict(fp4=S["proj4"], prequant=True)
            self._vgemm(B["v_a4"], S["proj_w_t"], B["v_attn_w"], nv * rows, c, c,
                        bias=S["proj_b"], **proj_kw)
            attn = B["v_attn_w"]
        else:
            self._vgemm(a_qkv, S["qkv_w_t"], B["v_qkv_c"], nv * rows, 3 * c, c,
                        bias=S["qkv_b"], **qkv_kw)
            fvk.xvla_vision_channel_attn(self._p(B["v_qkv_c"]), self._p(B["v_attn_c"]),
                                         self._p(B["v_ca"]), rows, c, groups, nv,
                                         VIS_HALF, st)
            self._vgemm(B["v_attn_c"], S["proj_w_t"], B["v_a"], nv * rows, c, c,
                        bias=S["proj_b"], fp4=S["proj4"])
            attn = B["v_a"]

        fvk.residual_add_pdl(self._p(B["v_b"]), self._p(attn), nv * rows * c, st)

        # conv2 + residual lands in v_a and stays there: it is the FFN residual,
        # so the LayerNorm must read it without clobbering it. The normed copy
        # goes to v_b, and the down projection overwrites v_b (the normed copy
        # is dead by then), leaving v_a = residual + ffn as the block output.
        fvk.xvla_vision_dwconv3x3_res(self._p(B["v_b"]), self._p(S["conv2_w"]),
                                      self._p(S["conv2_b"]), self._p(B["v_a"]), h, w,
                                      c, nv, VIS_HALF, st)
        # Same fusion for the FFN's input; v_a carries the residual and is
        # only read, so nothing needs to be staged elsewhere.
        self.fvk.xvla_ln_quant_bf16_nt256(
            self._p(B["v_a"]), self._p(S["n2w"]), self._p(S["n2b"]),
            self._p(B["v_a4"]), self._p(B["v_asf4"]), nv * rows, c, LN_EPS, st)
        a_fc1, fc1_kw = B["v_a4"], {"fp4": S["fc1_4"], "prequant": True}
        if USE_VISION_GELU_FUSED and ci in FUSED_STAGES:
            # bias + GELU + NVFP4 output fused into the fc1 GEMM epilogue. The
            # epilogue uses the tanh approximation; only the delivery gate can
            # decide whether that is admissible here.
            M = nv * rows
            Mp = (M + 127) // 128 * 128
            wp, wsf, wa = S["fc1_4"]
            dp = self._p(B["v_mlp"])
            dsf = dp + (Mp * 4 * c) // 2
            fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
                self._p(B["v_a4"]), self._p(wp), self._p(B["v_asf4"]),
                self._p(wsf), self._p(S["fc1_b"]), dp, dsf,
                Mp, 4 * c, c, wa, st)
            fc2_kw = dict(fp4=S["fc2_4"], prequant=True,
                          a_packed=B["v_mlp"], a_sfa_ptr=dsf)
        else:
            self._vgemm(a_fc1, S["fc1_w_t"], B["v_mlp"], nv * rows, 4 * c, c,
                        bias=S["fc1_b"], **fc1_kw)
            # The DaViT FFN is the erf GELU, not the tanh approximation; GELU
            # straight into the packed operand fc2 reads: the midpoint is never
            # written, and the quantize stops being a second launch.
            fvk.xvla_vision_gelu_erf_to_nvfp4(
                self._p(B["v_mlp"]), self._p(B["v_a4"]), self._p(B["v_asf4"]),
                nv * rows, 4 * c, st)
            fc2_kw = dict(fp4=S["fc2_4"], prequant=True)
        self._vgemm(B["v_mlp"], S["fc2_w_t"], B["v_b"], nv * rows, c, 4 * c,
                    bias=S["fc2_b"], **fc2_kw)
        fvk.residual_add_pdl(self._p(B["v_a"]), self._p(B["v_b"]), nv * rows * c, st)

    def vision_forward(self) -> None:
        """Florence-2 DaViT over the two real views, then the multimodal projector.

        The configured third view is an empty camera: the reference zeroes that
        view's *projector output* (it initializes the assembled feature tensor to
        zeros and scatters only the valid views in), so the tower runs over the
        two real views and the third slot stays zero.

        Both views ride through the tower together — tokens are [view][h][w] —
        so every stage GEMM's M is doubled and the tower's few hundred small
        kernels run once rather than once per camera.
        """
        fvk, V, B, st = self.fvk, self.W["vision"], self.B, self._st()
        nv = N_VIEWS - 1

        fvk.xvla_vision_resize(self._p(B["images"]), self._p(B["vimg"]), nv, 3,
                               SRC_IMG, SRC_IMG, self.img, self.img, VIS_HALF, st)
        fvk.xvla_copy_bf16(self._p(B["vimg"]), self._p(B["v_a"]),
                           nv * self.img * self.img * 3, st)

        for ci in range(4):
            c, h, w = self.dims.vision[ci]
            self._conv_embed(ci)
            for b in range(VIS_DEPTH[ci]):
                blk = V["blocks"][ci][b]
                self._vision_block(ci, blk, "spatial", h, w, c, VIS_HEADS[ci],
                                   VIS_GROUPS[ci], b)
                self._vision_block(ci, blk, "channel", h, w, c, VIS_HEADS[ci],
                                   VIS_GROUPS[ci], b)

        # Projector, per view (the pooled token and the position map are
        # per-image): position + temporal add, [spatial_avg_pool, patches],
        # bias-free 2048->1024 projection, LayerNorm.
        cl, hl, wl = self.dims.vision[3]
        for v in range(nv):
            fvk.xvla_vision_projector_pack(
                self._p(B["v_a"][v * hl * wl * cl]), self._p(V["pos_map"]),
                self._p(V["temporal"]), self._p(B["v_proj"]), V["tokens"],
                VIS_OUT_DIM, VIS_HALF, st)
            self._vgemm(B["v_proj"], V["proj_w"], B["v_tok_h"], VISION_TOKENS,
                        VIS_TOK_DIM, VIS_OUT_DIM)
            # One fused LayerNorm narrows fp16 -> bf16 at the single
            # DaViT/encoder dtype boundary, so no separate cast kernel runs.
            self._vnorm(B["v_tok_h"], V["proj_norm_w"], V["proj_norm_b"],
                        B["vision_tokens"][v], VISION_TOKENS, VIS_TOK_DIM)

    # ── driver ───────────────────────────────────────────────────────────

    def run(self) -> torch.Tensor:
        with self.fvk.xvla_pdl_scope(self.use_pdl):
            return self._run()

    def _run(self) -> torch.Tensor:
        self.vision_forward()
        self.encode_assemble()
        self.encode_forward()
        self.denoise_prepare()
        self.denoise_forward()
        return self.B["out"]

    def warm_start(self) -> None:
        """Prepare the prompt-independent device state (call once, after weights)."""
        self.denoise_prepare()
