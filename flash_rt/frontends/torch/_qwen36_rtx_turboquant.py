"""FlashRT -- TurboQuant KV cache quantization (Phase 2B).

Reference Python implementation of TurboQuant for Qwen3.6 KV cache
compression. Faithful to the algorithms in Zandieh et al. 2025
(arXiv:2504.19874v1):

  - Algorithm 1 (TurboQuant_mse): MSE-optimal scalar quantization on
    randomly-rotated coordinates.
  - Algorithm 2 (TurboQuant_prod): unbiased inner-product estimator.
    Combines TurboQuant_mse(b-1) with a 1-bit Quantized Johnson-
    Lindenstrauss (QJL) sketch on the residual.

Used to compress the per-token K/V cache in the Qwen3.6 full-attention
layers, freeing VRAM to extend usable context from 32K -> 200K+
(NVFP4 main path) or 8K -> 64K+ (FP8 main path).

Per-vector format (head_dim d=256, bit-width b):

  TurboQuant_mse storage:
    idx       : (d,) uint8 storing b-bit indices (packed: 8/b coords/byte)
    norm      : fp16 scalar (||x||_2 — re-applied at dequant)
    bytes     : ceil(d*b/8) + 2

  TurboQuant_prod storage (b total bits per coord = b-1 mse + 1 qjl):
    idx_mse   : (d,) packed (b-1)-bit indices
    qjl       : (d,) packed 1-bit signs
    norm      : fp16 (vector norm, ||x||_2)
    rnorm     : fp16 (residual norm, ||r||_2)
    bytes     : ceil(d*(b-1)/8) + d/8 + 4

Default for Phase 2B first cut: pure b=3 Q_mse for both K and V.
  per-vector: 3*256/8 + 2 = 98 bytes
  per-token: 16 layers * 4 KV heads * 2 (K+V) * 98 = 12544 bytes = 12.25 KB
  256K context: 256K * 12.25 KB = 3.06 GB
  vs BF16: 16 * 4 * 256 * 2 * 2 = 65536 = 64 KB/token, 256K = 16 GB
  Compression: 5.33x

Module conventions:
  - Π (rotation) and S (JL projection) are per-(layer, head) random
    matrices generated once, deterministically, from a layer-seeded RNG.
  - Codebooks are pre-computed for b in {2, 3, 4} via Lloyd-Max
    optimization on the per-coordinate Beta distribution that emerges
    after the random rotation (paper §3.1, Lemma 1).

Validation entry points:
  - reference_quant_dequant_roundtrip: cosine, MSE bound check
  - reference_inner_product_unbiased: IP unbiased property (Q_prod)
  - paper_distortion_check: numerical match to paper Theorem 1/2
"""
from __future__ import annotations

import math
import os

import torch


# ====================================================================
# Offline setup: rotations, JL projections, codebooks
# ====================================================================

def make_rotation_matrix(d: int, seed: int,
                          device='cuda:0', dtype=torch.float32):
    """Generate a random orthogonal matrix Π ∈ R^{d×d} via QR(N(0,1)).

    Per paper §3.1: Π is uniform over the orthogonal group O(d).
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    A = torch.randn(d, d, generator=g, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Standard trick: enforce sign convention so Π is uniform on O(d)
    # (Mezzadri 2007). Multiply Q by sign(diag(R)).
    sign = torch.sign(torch.diag(R))
    sign[sign == 0] = 1.0
    Q = Q * sign.unsqueeze(0)
    return Q.contiguous()


def make_jl_matrix(d: int, seed: int,
                   device='cuda:0', dtype=torch.float32):
    """Generate a JL projection S ∈ R^{d×d} with i.i.d. N(0,1) entries.

    Used by TurboQuant_prod (Algorithm 2) — sign(S·r) is the QJL sketch
    of the residual vector r.
    """
    g = torch.Generator(device=device).manual_seed(int(seed) + 1_000_000)
    return torch.randn(d, d, generator=g, device=device, dtype=dtype)


def lloyd_max_codebook(b: int, d: int, n_iter: int = 100,
                        n_samples: int = 1_000_000,
                        seed: int = 42,
                        device='cuda:0') -> torch.Tensor:
    """Compute Lloyd-Max-optimal centroids for b-bit scalar quantization
    of a coordinate of a randomly-rotated unit vector in R^d.

    By Lemma 1 of the paper, each coord of Π·x (for x on the unit
    sphere) follows a Beta((d-1)/2, (d-1)/2) distribution scaled to
    [-1, 1]. For large d (d=256 in our case) this converges to
    N(0, 1/d) per the central limit theorem. We sample from the Normal
    approximation which is reproducible (torch.randn supports generator).

    Returns: centroids (2^b,) fp32, sorted ascending. Voronoi cell for
    centroid c[k] is [(c[k-1]+c[k])/2, (c[k]+c[k+1])/2].

    Closed-form for small b (paper §3.1 / §3.2):
      b=1:  ±√(2/π)/√d
      b=2:  ±0.453/√d, ±1.51/√d  (numerical Lloyd-Max)
    """
    n_centroids = 1 << b
    # Sample from N(0, 1/d) using a fully reproducible generator.
    # (torch.distributions.Beta does not accept a generator arg, so it
    # would silently consume global RNG state and produce different
    # codebooks across runs in the same process.)
    g = torch.Generator(device=device).manual_seed(seed)
    samples = torch.randn(n_samples, generator=g,
                           device=device, dtype=torch.float32)
    samples = samples / math.sqrt(d)
    # Clamp to [-1, 1] for safety (rare large-magnitude tails).
    samples = samples.clamp(-1.0, 1.0)

    # Init centroids by quantiles
    quantiles = torch.linspace(
        1.0 / (2 * n_centroids),
        1.0 - 1.0 / (2 * n_centroids),
        n_centroids, device=device)
    centroids = torch.quantile(samples, quantiles).contiguous()

    for _ in range(n_iter):
        # E-step: assign each sample to nearest centroid
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        idx = dists.argmin(dim=1)
        # M-step: new centroids = mean of assigned samples
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_centroids):
            mask = idx == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
            else:
                new_centroids[k] = centroids[k]
        if torch.allclose(new_centroids, centroids, atol=1e-7):
            break
        centroids = new_centroids
    return centroids.sort()[0].contiguous()


# ====================================================================
# TurboQuant_mse (Algorithm 1)
# ====================================================================

def tq_quant_mse(x: torch.Tensor, b: int, rotation: torch.Tensor,
                  codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a batch of vectors via TurboQuant_mse.

    Args:
      x         : (..., d) bf16/fp32 input
      b         : bit-width (1..4 supported by codebook)
      rotation  : (d, d) Π matrix
      codebook  : (2^b,) sorted centroids

    Returns:
      idx       : (..., d) int64 in [0, 2^b)
      norm      : (...,)  fp32 ||x||_2 per vector

    Math (paper Algorithm 1):
      norm = ||x||_2
      x_unit = x / norm                              # unit-sphere
      y = Π · x_unit                                 # random rotation
      idx[j] = argmin_k |y[j] - c[k]|
    """
    norm = x.float().norm(dim=-1, keepdim=False)     # (...)
    eps = 1e-12
    x_unit = (x.float() / (norm.unsqueeze(-1) + eps))
    y = x_unit @ rotation.T                          # (..., d)
    # Voronoi nearest centroid: midpoints between consecutive centroids
    # are the cell boundaries. Use vectorized argmin on |y - c|.
    dists = (y.unsqueeze(-1) - codebook.view(*([1] * y.ndim), -1)).abs()
    idx = dists.argmin(dim=-1)
    return idx, norm


def tq_dequant_mse(idx: torch.Tensor, rotation: torch.Tensor,
                    codebook: torch.Tensor,
                    norm: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize back to (..., d) fp32.

    Math:
      ỹ[j] = c[idx[j]]
      x̃_unit = Π^T · ỹ
      x̃ = norm * x̃_unit
    """
    y_hat = codebook[idx]                            # (..., d) fp32
    x_hat_unit = y_hat @ rotation                    # = ỹ @ Π = (Π^T ỹ)^T
    if norm is not None:
        x_hat = x_hat_unit * norm.unsqueeze(-1)
    else:
        x_hat = x_hat_unit
    return x_hat


# ====================================================================
# TurboQuant_prod (Algorithm 2): Q_mse(b-1) + 1-bit QJL
# ====================================================================

def tq_quant_prod(x: torch.Tensor, b: int, rotation: torch.Tensor,
                   codebook_bm1: torch.Tensor, jl: torch.Tensor):
    """Quantize a batch of vectors via TurboQuant_prod.

    Args:
      x             : (..., d) bf16/fp32 input
      b             : total bit-width budget (uses b-1 for MSE part)
      rotation      : (d, d) Π
      codebook_bm1  : (2^(b-1),) centroids for the (b-1)-bit MSE part
      jl            : (d, d) JL matrix S

    Returns:
      idx       : (..., d) int64 in [0, 2^(b-1))
      qjl       : (..., d) int8 in {-1, +1} (sign(S·r))
      norm      : (...,)   fp32 ||x||_2
      rnorm     : (...,)   fp32 ||r||_2 (residual L2 norm)
    """
    idx, norm = tq_quant_mse(x, b - 1, rotation, codebook_bm1)
    # Recover normalized x and its mse-quantized reconstruction (unit
    # sphere domain), compute residual on unit sphere.
    norm_safe = norm + 1e-12
    x_unit = x.float() / norm_safe.unsqueeze(-1)
    x_mse_unit = tq_dequant_mse(idx, rotation, codebook_bm1, norm=None)
    r_unit = x_unit - x_mse_unit
    rnorm = r_unit.norm(dim=-1, keepdim=False)
    # QJL on r_unit (already unit-sphere domain)
    Sr = r_unit @ jl.T                                # (..., d)
    qjl_signed = torch.where(
        Sr >= 0, torch.tensor(1, dtype=torch.int8, device=x.device),
        torch.tensor(-1, dtype=torch.int8, device=x.device))
    return idx, qjl_signed, norm, rnorm


def tq_dequant_prod(idx: torch.Tensor, qjl_signed: torch.Tensor,
                     norm: torch.Tensor, rnorm: torch.Tensor,
                     rotation: torch.Tensor, codebook_bm1: torch.Tensor,
                     jl: torch.Tensor) -> torch.Tensor:
    """Dequantize via TurboQuant_prod.

    Math (paper Algorithm 2 line 11):
      x̃_mse = DequantMSE(idx)              (unit sphere; we'll rescale)
      x̃_qjl = √(π/2)/d · γ · S^T · qjl     where γ = ||r||
      x̃_unit = x̃_mse + x̃_qjl
      x̃ = norm * x̃_unit
    """
    d = rotation.shape[0]
    x_mse_unit = tq_dequant_mse(idx, rotation, codebook_bm1, norm=None)
    # qjl is int8 ∈ {-1,+1}; promote to fp32 for matmul
    qjl_f = qjl_signed.float()
    Sq = qjl_f @ jl                                   # = (S^T qjl)^T
    coef = math.sqrt(math.pi / 2.0) / d
    x_qjl_unit = (coef * rnorm.unsqueeze(-1)) * Sq
    x_hat_unit = x_mse_unit + x_qjl_unit
    return x_hat_unit * norm.unsqueeze(-1)


# ====================================================================
# Convenience: per-layer-per-head setup struct
# ====================================================================

class TurboQuantSetup:
    """Holds per-layer rotation + JL + codebooks for a model.

    Layout:
      rotations[layer]   : (d, d) fp32 — shared across heads in a layer
                           (could be per-head; per-layer is simpler and
                           matches paper §4.3 setup for KV cache)
      jl[layer]          : (d, d) fp32
      codebook[b]        : (2^b,) fp32 — shared across all layers

    Pre-compute once at frontend init.
    """

    def __init__(self, num_layers: int, head_dim: int,
                 base_seed: int = 0xC0FFEE,
                 device='cuda:0',
                 b_v: int = 3, b_k_total: int = 3):
        """Args:
          num_layers : 16 (Qwen3.6 full-attn layers)
          head_dim   : 256
          b_v        : bit-width for V (Q_mse)
          b_k_total  : total bit-width for K (Q_prod uses b_k_total-1 for MSE)
        """
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.b_v = b_v
        self.b_k_total = b_k_total
        self.device = device

        # Rotations + JL per layer
        self.rotations = []
        self.jl = []
        for L in range(num_layers):
            R = make_rotation_matrix(head_dim, base_seed + L, device=device)
            S = make_jl_matrix(head_dim, base_seed + L, device=device)
            self.rotations.append(R)
            self.jl.append(S)

        # Codebooks shared across layers (depend only on b and d)
        self.codebooks = {}
        # V uses b_v
        self.codebooks[b_v] = lloyd_max_codebook(
            b_v, head_dim, device=device)
        # K uses Q_prod with (b_k_total - 1) for MSE part
        b_k_mse = max(1, b_k_total - 1)
        if b_k_mse not in self.codebooks:
            self.codebooks[b_k_mse] = lloyd_max_codebook(
                b_k_mse, head_dim, device=device)

    @property
    def b_k_mse(self) -> int:
        return max(1, self.b_k_total - 1)

    # ── Fast-path bf16 caches (Phase 3A B9: CUDA unpack + cuBLAS GEMM) ──
    # Pre-cast rotation / jl to bf16 once so the tensor-core GEMM in
    # read_kv_fast does not pay a per-call cast.  cb_*_bf16 unused (we
    # keep cb in fp32; the unpack kernel casts on emission).
    @property
    def rotation_bf16(self):
        if not hasattr(self, '_rotation_bf16'):
            self._rotation_bf16 = [
                R.to(torch.bfloat16).contiguous() for R in self.rotations]
        return self._rotation_bf16

    @property
    def jl_bf16(self):
        if not hasattr(self, '_jl_bf16'):
            self._jl_bf16 = [
                J.to(torch.bfloat16).contiguous() for J in self.jl]
        return self._jl_bf16

    # Convenience wrappers for V (Q_mse) and K (Q_prod)
    def quant_v(self, v, layer):
        return tq_quant_mse(
            v, self.b_v, self.rotations[layer], self.codebooks[self.b_v])

    def dequant_v(self, idx, norm, layer):
        return tq_dequant_mse(
            idx, self.rotations[layer], self.codebooks[self.b_v], norm)

    def quant_k(self, k, layer):
        return tq_quant_prod(
            k, self.b_k_total, self.rotations[layer],
            self.codebooks[self.b_k_mse], self.jl[layer])

    def dequant_k(self, idx, qjl, norm, rnorm, layer):
        return tq_dequant_prod(
            idx, qjl, norm, rnorm,
            self.rotations[layer], self.codebooks[self.b_k_mse],
            self.jl[layer])

    # Per-vector storage size (bytes)
    @property
    def bytes_per_v_vec(self) -> int:
        return math.ceil(self.head_dim * self.b_v / 8) + 2  # idx + fp16 norm

    @property
    def bytes_per_k_vec(self) -> int:
        idx_bytes = math.ceil(self.head_dim * self.b_k_mse / 8)
        qjl_bytes = self.head_dim // 8
        return idx_bytes + qjl_bytes + 4  # idx + qjl + 2 fp16 norms


# ====================================================================
# Production buffers + write/read hooks for KV cache integration
# ====================================================================
#
# Storage layout (per layer, per cache K/V):
#
#   V cache: split into three contiguous tensors per layer:
#     v_idx_cache   : (max_seq, num_kv, head_dim) uint8   <- b-bit indices
#                     (1 byte per idx; bit-packing is a follow-up)
#     v_norm_cache  : (max_seq, num_kv)            fp16
#
#   K cache:
#     k_idx_cache   : (max_seq, num_kv, head_dim) uint8   <- (b-1)-bit
#     k_qjl_cache   : (max_seq, num_kv, head_dim/8) uint8 <- 1-bit packed
#     k_norm_cache  : (max_seq, num_kv)            fp16   <- ||x||
#     k_rnorm_cache : (max_seq, num_kv)            fp16   <- ||r||
#
# This is FUNCTIONALLY equivalent to the bit-packed layout. With 1-byte
# idx storage at b=3:
#   per V vec : 256 + 2     = 258 bytes  (vs bit-packed 98)
#   per K vec : 256 + 32 + 4 = 292 bytes  (vs bit-packed 100)
#   per token : 16 × 4 × 550 = 35.2 KB    (vs bit-packed 12.4)
#   256K ctx  : 9.0 GB                    (vs bit-packed 3.2 GB)
# 9 GB still fits 32GB - 17 GB main = 15 GB headroom; bit-packing is
# a P optimization for either pushing past 256K or reducing VRAM.


def _pack_4bit(idx_uint8):
    """Pack a tensor of 4-bit values (..., d, last dim must be even) into
    (..., d/2) uint8. Each output byte holds two 4-bit values:
    (hi << 4) | lo. Vectorized; no Python loop."""
    if idx_uint8.shape[-1] % 2 != 0:
        raise RuntimeError('last dim must be even for 4-bit pack')
    flat = idx_uint8.view(*idx_uint8.shape[:-1], idx_uint8.shape[-1] // 2, 2)
    return (flat[..., 0] | (flat[..., 1] << 4)).contiguous()


def _unpack_4bit(packed):
    """Inverse of _pack_4bit. (..., d/2) uint8 -> (..., d) uint8."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.stack([lo, hi], dim=-1).view(
        *packed.shape[:-1], packed.shape[-1] * 2).contiguous()


_PACK_QJL_WEIGHTS = None


def _pack_qjl_1bit(qjl_signed):
    """Pack signed bits (-1/+1) into 1 bit per coord.
    qjl_signed: (..., d) int8, d divisible by 8.
    Returns: (..., d/8) uint8."""
    global _PACK_QJL_WEIGHTS
    if qjl_signed.shape[-1] % 8 != 0:
        raise RuntimeError('last dim must be multiple of 8 for 1-bit pack')
    if _PACK_QJL_WEIGHTS is None or _PACK_QJL_WEIGHTS.device != qjl_signed.device:
        _PACK_QJL_WEIGHTS = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128],
            dtype=torch.uint8, device=qjl_signed.device)
    bits = (qjl_signed > 0).to(torch.uint8)
    flat = bits.view(*bits.shape[:-1], bits.shape[-1] // 8, 8)
    return (flat * _PACK_QJL_WEIGHTS).sum(dim=-1).to(torch.uint8).contiguous()


def _unpack_qjl_1bit(packed):
    """Inverse of _pack_qjl_1bit. (..., d/8) uint8 -> (..., d) int8 in {-1,+1}."""
    global _PACK_QJL_WEIGHTS
    if _PACK_QJL_WEIGHTS is None or _PACK_QJL_WEIGHTS.device != packed.device:
        _PACK_QJL_WEIGHTS = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128],
            dtype=torch.uint8, device=packed.device)
    bits = ((packed.unsqueeze(-1) & _PACK_QJL_WEIGHTS) > 0).to(torch.int8)
    return (bits * 2 - 1).view(
        *packed.shape[:-1], packed.shape[-1] * 8).contiguous()


class TurboQuantKVCache:
    """Per-frontend K/V cache state holding packed buffers.

    Allocated lazily after the frontend's max_seq is known. Replaces
    self._attn.K_cache / V_cache for the TurboQuant path.

    Two storage modes:
      packed=False (default): idx as uint8 (1 byte per coord),
        qjl as int8 (-1/+1 stored explicitly, 1 byte per coord).
        Easy to reason about, larger storage.
      packed=True (B8): idx packed at 4 bits per coord,
        qjl packed at 1 bit per coord. ~3× smaller storage; uses
        torch bit-pack ops on read/write. Validates at b_v∈{2,3,4}
        and b_k_mse∈{2,3} (4-bit slot wastes 1 bit at b=3 but trivial
        impl; true 3-bit pack is a future optimization).
    """

    def __init__(self, setup: TurboQuantSetup, max_seq: int,
                 num_kv: int = 4, device='cuda:0',
                 packed: bool = False):
        import torch
        self.setup = setup
        self.max_seq = int(max_seq)
        self.num_kv = int(num_kv)
        self.device = device
        self.packed = bool(packed)
        d = setup.head_dim
        L = setup.num_layers

        # V cache (Q_mse): idx + per-vec norm
        idx_dim = (d // 2) if packed else d
        self.v_idx = torch.zeros(L, max_seq, num_kv, idx_dim,
                                  dtype=torch.uint8, device=device)
        self.v_norm = torch.zeros(L, max_seq, num_kv,
                                   dtype=torch.float16, device=device)

        # K cache (Q_prod): idx (b-1 bit slot) + qjl (1-bit) + 2 norms
        self.k_idx = torch.zeros(L, max_seq, num_kv, idx_dim,
                                  dtype=torch.uint8, device=device)
        qjl_dim = (d // 8) if packed else d
        qjl_dtype = torch.uint8 if packed else torch.int8
        self.k_qjl = torch.zeros(L, max_seq, num_kv, qjl_dim,
                                  dtype=qjl_dtype, device=device)
        self.k_norm = torch.zeros(L, max_seq, num_kv,
                                   dtype=torch.float16, device=device)
        self.k_rnorm = torch.zeros(L, max_seq, num_kv,
                                    dtype=torch.float16, device=device)

        # Phase 3B-β: per-layer dequant high-water mark.  Tracks the
        # exclusive end position up to which an external BF16 staging
        # buffer (per-layer, owned by the frontend) is up-to-date.
        # write_kv* invalidates the tail when an earlier slot is
        # rewritten; reads only need to dequant [valid_end, end_pos).
        self._dequant_valid_end = [0] * L

    def invalidate_layer(self, layer: int) -> None:
        """Reset the dequant high-water mark for one layer."""
        self._dequant_valid_end[layer] = 0

    def invalidate_all(self) -> None:
        """Reset the dequant high-water mark for every layer."""
        for L in range(len(self._dequant_valid_end)):
            self._dequant_valid_end[L] = 0

    def write_kv(self, layer: int, pos_start: int, pos_end: int,
                 k: torch.Tensor, v: torch.Tensor) -> None:
        """Quantize and write K/V for positions [pos_start, pos_end).

        k, v: (S, num_kv, head_dim) bf16 where S = pos_end - pos_start.
        Auto-packs idx (4-bit) + qjl (1-bit) when self.packed=True.
        """
        S = pos_end - pos_start
        if k.shape != (S, self.num_kv, self.setup.head_dim):
            raise RuntimeError(
                f'K shape {tuple(k.shape)} != ({S}, {self.num_kv}, '
                f'{self.setup.head_dim})')
        # β: invalidate tail of any per-layer staging that covered
        # [pos_start, valid_end).  Append-only writes are a no-op.
        if pos_start < self._dequant_valid_end[layer]:
            self._dequant_valid_end[layer] = pos_start
        # K via Q_prod
        idx_k, qjl_k, norm_k, rnorm_k = self.setup.quant_k(k, layer)
        idx_k_u8 = idx_k.to(torch.uint8)
        if self.packed:
            self.k_idx[layer, pos_start:pos_end].copy_(_pack_4bit(idx_k_u8))
            self.k_qjl[layer, pos_start:pos_end].copy_(_pack_qjl_1bit(qjl_k))
        else:
            self.k_idx[layer, pos_start:pos_end].copy_(idx_k_u8)
            self.k_qjl[layer, pos_start:pos_end].copy_(qjl_k)
        self.k_norm[layer, pos_start:pos_end].copy_(norm_k.to(torch.float16))
        self.k_rnorm[layer, pos_start:pos_end].copy_(
            rnorm_k.to(torch.float16))
        # V via Q_mse
        idx_v, norm_v = self.setup.quant_v(v, layer)
        idx_v_u8 = idx_v.to(torch.uint8)
        if self.packed:
            self.v_idx[layer, pos_start:pos_end].copy_(_pack_4bit(idx_v_u8))
        else:
            self.v_idx[layer, pos_start:pos_end].copy_(idx_v_u8)
        self.v_norm[layer, pos_start:pos_end].copy_(norm_v.to(torch.float16))

    def read_kv(self, layer: int, pos_end: int) -> tuple[torch.Tensor,
                                                          torch.Tensor]:  # noqa
        return self._read_kv_unpacked(layer, pos_end)

    def write_kv_fast(self, layer: int, pos_start: int, pos_end: int,
                      k: torch.Tensor, v: torch.Tensor) -> None:
        """Phase 3A B9-S10: capture-safe quantize+pack via 4 small CUDA
        kernels + 3 explicit GEMM wrappers by default.

        The kernel GEMM route uses cuBLASLt tactics (A=3, B=0, C=3)
        validated against the reference on real 2048-token long-prefill
        chunks.
        """
        if not self.packed:
            raise RuntimeError(
                'write_kv_fast requires packed=True (B8 layout)')
        # β: invalidate tail of any per-layer staging that covered
        # [pos_start, valid_end).  Append-only writes are a no-op.
        if pos_start < self._dequant_valid_end[layer]:
            self._dequant_valid_end[layer] = pos_start
        from flash_rt import flash_rt_kernels as fvk
        S = pos_end - pos_start
        nkv = self.num_kv
        M = S * nkv
        d = self.setup.head_dim

        # Lazy scratch.  Decode writes one token, but chunked long
        # prefill can write larger S.  Grow the scratch to the largest
        # observed M; fixed 64-token capacity silently overflowed when
        # experimenting with 128-token prefill chunks.
        cap_M = max(64 * nkv, M)
        if (not hasattr(self, '_w_kv_unit')
                or int(getattr(self, '_w_cap_M', 0)) < cap_M):
            t = lambda *shape: torch.empty(  # noqa: E731
                *shape, dtype=torch.float32, device=self.device)
            self._w_kv_unit = t(2 * cap_M, d)        # [k_unit; v_unit]
            self._w_kv_rotated = t(2 * cap_M, d)    # [y_k; y_v]
            self._w_dq_in = t(cap_M, d)             # cb_k[idx_k]
            self._w_dq_out = t(cap_M, d)            # cb_k[idx_k] @ rotation
            self._w_residual = t(cap_M, d)
            self._w_Sr = t(cap_M, d)
            self._w_norms = t(3 * cap_M)            # [norm_k|norm_v|rnorm_k]
            self._w_cap_M = cap_M
        else:
            cap_M = int(self._w_cap_M)

        rot_fp32 = self.setup.rotations[layer]
        jl_fp32 = self.setup.jl[layer]
        cb_k = self.setup.codebooks[self.setup.b_k_mse]
        cb_v = self.setup.codebooks[self.setup.b_v]
        s = torch.cuda.current_stream().cuda_stream
        if os.environ.get('FLASHRT_QWEN36_TQ_KERNEL_WRITE', '1') != '1':
            raise ValueError(
                'FLASHRT_QWEN36_TQ_KERNEL_WRITE=0 is no longer supported '
                'on the kernel-only Qwen3.6 path')
        use_kernel_gemm = (
            hasattr(fvk, 'tq_fp32_gemm_lt_bt_algo')
            and hasattr(fvk, 'tq_fp32_gemm_lt_algo')
        )
        if not use_kernel_gemm:
            raise RuntimeError(
                'TurboQuant write requires FlashRT cuBLASLt GEMM wrappers')

        # Slice the scratch (views, no alloc).
        k_unit = self._w_kv_unit[:M]
        v_unit = self._w_kv_unit[M:2 * M]
        y_k = self._w_kv_rotated[:M]
        y_v = self._w_kv_rotated[M:2 * M]
        norm_k = self._w_norms[0:cap_M][:M]
        norm_v = self._w_norms[cap_M:2 * cap_M][:M]
        rnorm_k = self._w_norms[2 * cap_M:3 * cap_M][:M]
        dq_in = self._w_dq_in[:M]
        dq_out = self._w_dq_out[:M]
        residual = self._w_residual[:M]
        Sr = self._w_Sr[:M]

        # K1: bf16 (k, v) → fp32 unit + norm (1 launch)
        fvk.tq_write_k1_unit_norm(
            k.data_ptr(), v.data_ptr(),
            k_unit.data_ptr(), v_unit.data_ptr(),
            norm_k.data_ptr(), norm_v.data_ptr(),
            M, self.setup.b_k_mse, self.setup.b_v, s,
        )
        # GEMM A: [k_unit; v_unit] @ rotation^T → [y_k; y_v]
        fvk.tq_fp32_gemm_lt_bt_algo(
            self._w_kv_unit[:2 * M].data_ptr(), rot_fp32.data_ptr(),
            self._w_kv_rotated[:2 * M].data_ptr(),
            2 * M, d, d, 3, s,
        )

        # K2: argmin → idx, pack 4-bit to cache, gather cb_k[idx_k] for dq
        fvk.tq_write_k2_argmin_pack(
            y_k.data_ptr(), y_v.data_ptr(),
            cb_k.data_ptr(), cb_v.data_ptr(),
            self.k_idx[layer].data_ptr(),
            self.v_idx[layer].data_ptr(),
            dq_in.data_ptr(),
            pos_start, nkv, M,
            self.setup.b_k_mse, self.setup.b_v, s,
        )
        # GEMM B: cb_k[idx_k] @ rotation → dq_k
        fvk.tq_fp32_gemm_lt_algo(
            dq_in.data_ptr(), rot_fp32.data_ptr(), dq_out.data_ptr(),
            M, d, d, 0, s,
        )

        # K3: residual = k_unit - dq_k ; rnorm_k = ‖residual‖
        fvk.tq_write_k3_residual_rnorm(
            k_unit.data_ptr(), dq_out.data_ptr(),
            residual.data_ptr(), rnorm_k.data_ptr(),
            M, s,
        )
        # GEMM C: residual @ jl^T → Sr
        fvk.tq_fp32_gemm_lt_bt_algo(
            residual.data_ptr(), jl_fp32.data_ptr(), Sr.data_ptr(),
            M, d, d, 3, s,
        )

        # K4: pack qjl bits + write all norms (fp16) to cache slot
        fvk.tq_write_k4_qjl_norms(
            Sr.data_ptr(),
            norm_k.data_ptr(), rnorm_k.data_ptr(), norm_v.data_ptr(),
            self.k_qjl[layer].data_ptr(),
            self.k_norm[layer].data_ptr(),
            self.k_rnorm[layer].data_ptr(),
            self.v_norm[layer].data_ptr(),
            pos_start, nkv, M, s,
        )

    # ────────────────────────────────────────────────────────────────
    # Phase 3A B9 fast path: CUDA unpack + cuBLAS bf16 GEMM + combine.
    # Writes into caller-supplied (max_seq, num_kv, head_dim) bf16
    # staging buffers — saves the read_kv allocator round-trip.
    # ────────────────────────────────────────────────────────────────
    def _ensure_fast_dequant_scratch(self, cap_M: int) -> None:
        """Grow fp32 dequant scratch to the active window, not max_seq."""
        d = self.setup.head_dim
        cap_M = int(cap_M)
        if int(getattr(self, '_fast_cap_M', 0)) >= cap_M:
            return
        self._fast_yk_yv = torch.empty(
            2 * cap_M, d, dtype=torch.float32, device=self.device)
        self._fast_qjl = torch.empty(
            cap_M, d, dtype=torch.float32, device=self.device)
        self._fast_rotated_fp32 = torch.empty(
            2 * cap_M, d, dtype=torch.float32, device=self.device)
        self._fast_kqjl_fp32 = torch.empty(
            cap_M, d, dtype=torch.float32, device=self.device)
        self._fast_cap_M = cap_M

    def read_kv_fast(self, layer: int, pos_end: int,
                     k_stage: torch.Tensor, v_stage: torch.Tensor) -> None:
        """B9 dequant: packed[layer, :pos_end] → k_stage[:pos_end] / v_stage.

        Requires self.packed=True. Caller passes pre-allocated bf16
        staging tensors of shape (max_seq, num_kv, head_dim); only the
        first ``pos_end`` rows are written.
        """
        if not self.packed:
            raise RuntimeError(
                'read_kv_fast requires packed=True (B8 layout)')
        if pos_end <= 0:
            return
        # cuBLASLt's bitwise match to torch.matmul is shape/tactic
        # sensitive.  The validated production unit is the 2048-token
        # long-prefill window, so larger full-prefix reads are assembled
        # from the same exact window kernel.
        chunk = 2048
        if pos_end > chunk:
            for ps in range(0, pos_end, chunk):
                self.read_kv_fast_window(
                    layer, ps, min(ps + chunk, pos_end),
                    k_stage, v_stage)
            return
        from flash_rt import flash_rt_kernels as fvk
        d = self.setup.head_dim
        nkv = self.num_kv
        M = pos_end * nkv

        # Scratch is sized to the active dequant window.  The full-prefix
        # path above decomposes long reads into 2048-token windows, so
        # allocating by max_seq would waste multiple GB at 128K/256K.
        self._ensure_fast_dequant_scratch(M)

        yk = self._fast_yk_yv[:M]
        yv = self._fast_yk_yv[M:2 * M]
        qjl_f = self._fast_qjl[:M]
        rotated = self._fast_rotated_fp32[:2 * M]
        kmse = rotated[:M]
        vunit = rotated[M:]
        kqjl = self._fast_kqjl_fp32[:M]

        s = torch.cuda.current_stream().cuda_stream
        cb_k = self.setup.codebooks[self.setup.b_k_mse]
        cb_v = self.setup.codebooks[self.setup.b_v]
        rot_fp32 = self.setup.rotations[layer]
        jl_fp32 = self.setup.jl[layer]

        # 1) unpack packed → 3 fp32 tensors directly (no bf16 round-trip)
        fvk.tq_unpack_packed_fp32(
            self.k_idx[layer, :pos_end].data_ptr(),
            self.k_qjl[layer, :pos_end].data_ptr(),
            self.v_idx[layer, :pos_end].data_ptr(),
            cb_k.data_ptr(), cb_v.data_ptr(),
            yk.data_ptr(), qjl_f.data_ptr(), yv.data_ptr(),
            M, self.setup.b_k_mse, self.setup.b_v, s,
        )
        # 2) fp32 GEMMs via explicit wrappers.  PyTorch chooses different
        # tactics for the two shapes: Lt algo 1 matches the rotation GEMM,
        # while ordinary cuBLAS matches the JL GEMM bit-for-bit.
        fvk.tq_fp32_gemm_lt_algo(
            self._fast_yk_yv[:2 * M].data_ptr(), rot_fp32.data_ptr(),
            rotated.data_ptr(), 2 * M, d, d, 1, s,
        )
        fvk.tq_fp32_gemm_fp32(
            qjl_f.data_ptr(), jl_fp32.data_ptr(), kqjl.data_ptr(),
            M, d, d, s,
        )
        # 3) combine fp32 in → bf16 out
        coef = math.sqrt(math.pi / 2.0) / d
        k_flat = k_stage[:pos_end].view(M, d)
        v_flat = v_stage[:pos_end].view(M, d)
        fvk.tq_combine_kv_fp32_in(
            kmse.data_ptr(), kqjl.data_ptr(), vunit.data_ptr(),
            self.k_norm[layer, :pos_end].data_ptr(),
            self.k_rnorm[layer, :pos_end].data_ptr(),
            self.v_norm[layer, :pos_end].data_ptr(),
            k_flat.data_ptr(), v_flat.data_ptr(),
            M, coef, s,
        )

    def read_kv_fast_window(self, layer: int,
                              pos_start: int, pos_end: int,
                              k_stage: torch.Tensor,
                              v_stage: torch.Tensor) -> None:
        """β: dequant the slice [pos_start, pos_end) into the same offset
        of the caller-supplied per-layer staging buffers.

        Identical math to read_kv_fast — same kernels, same GEMMs — but
        operates on the (pos_end - pos_start) row sub-range so the
        dequant cost scales with the new-row count, not cur_pos.

        Output: k_stage[pos_start:pos_end] / v_stage[pos_start:pos_end]
        are written; rows outside [pos_start, pos_end) are untouched.

        Bit-tight equivalence (gate Gβ-3): for any (ps, pe), this writes
        the same bytes as read_kv_fast(layer, pe).slice([ps:pe]).
        """
        if not self.packed:
            raise RuntimeError(
                'read_kv_fast_window requires packed=True (B8 layout)')
        if pos_end <= pos_start:
            return
        from flash_rt import flash_rt_kernels as fvk
        d = self.setup.head_dim
        nkv = self.num_kv
        M = (pos_end - pos_start) * nkv

        self._ensure_fast_dequant_scratch(M)

        yk = self._fast_yk_yv[:M]
        yv = self._fast_yk_yv[M:2 * M]
        qjl_f = self._fast_qjl[:M]
        rotated = self._fast_rotated_fp32[:2 * M]
        kmse = rotated[:M]
        vunit = rotated[M:]
        kqjl = self._fast_kqjl_fp32[:M]

        s = torch.cuda.current_stream().cuda_stream
        cb_k = self.setup.codebooks[self.setup.b_k_mse]
        cb_v = self.setup.codebooks[self.setup.b_v]
        rot_fp32 = self.setup.rotations[layer]
        jl_fp32 = self.setup.jl[layer]

        # Window pointers via slicing (no alloc, just stride/offset math).
        k_idx_w = self.k_idx[layer, pos_start:pos_end]
        k_qjl_w = self.k_qjl[layer, pos_start:pos_end]
        v_idx_w = self.v_idx[layer, pos_start:pos_end]
        k_norm_w = self.k_norm[layer, pos_start:pos_end]
        k_rnorm_w = self.k_rnorm[layer, pos_start:pos_end]
        v_norm_w = self.v_norm[layer, pos_start:pos_end]

        fvk.tq_unpack_packed_fp32(
            k_idx_w.data_ptr(), k_qjl_w.data_ptr(), v_idx_w.data_ptr(),
            cb_k.data_ptr(), cb_v.data_ptr(),
            yk.data_ptr(), qjl_f.data_ptr(), yv.data_ptr(),
            M, self.setup.b_k_mse, self.setup.b_v, s,
        )
        fvk.tq_fp32_gemm_lt_algo(
            self._fast_yk_yv[:2 * M].data_ptr(), rot_fp32.data_ptr(),
            rotated.data_ptr(), 2 * M, d, d, 1, s,
        )
        fvk.tq_fp32_gemm_fp32(
            qjl_f.data_ptr(), jl_fp32.data_ptr(), kqjl.data_ptr(),
            M, d, d, s,
        )
        coef = math.sqrt(math.pi / 2.0) / d
        k_flat = k_stage[pos_start:pos_end].view(M, d)
        v_flat = v_stage[pos_start:pos_end].view(M, d)
        fvk.tq_combine_kv_fp32_in(
            kmse.data_ptr(), kqjl.data_ptr(), vunit.data_ptr(),
            k_norm_w.data_ptr(), k_rnorm_w.data_ptr(), v_norm_w.data_ptr(),
            k_flat.data_ptr(), v_flat.data_ptr(),
            M, coef, s,
        )

    def _read_kv_unpacked(self, layer: int, pos_end: int) -> tuple[
            torch.Tensor, torch.Tensor]:
        """Dequantize K/V cache rows [0, pos_end). Returns (K, V) in bf16.

        Output shape: (pos_end, num_kv, head_dim) each.
        Auto-unpacks idx (4-bit) + qjl (1-bit) when self.packed=True.
        """
        if self.packed:
            idx_k = _unpack_4bit(self.k_idx[layer, :pos_end]).long()
            qjl_k = _unpack_qjl_1bit(self.k_qjl[layer, :pos_end])
            idx_v = _unpack_4bit(self.v_idx[layer, :pos_end]).long()
        else:
            idx_k = self.k_idx[layer, :pos_end].long()
            qjl_k = self.k_qjl[layer, :pos_end]
            idx_v = self.v_idx[layer, :pos_end].long()
        norm_k = self.k_norm[layer, :pos_end].float()
        rnorm_k = self.k_rnorm[layer, :pos_end].float()
        k_hat = self.setup.dequant_k(idx_k, qjl_k, norm_k, rnorm_k, layer)
        norm_v = self.v_norm[layer, :pos_end].float()
        v_hat = self.setup.dequant_v(idx_v, norm_v, layer)

        return k_hat.to(torch.bfloat16), v_hat.to(torch.bfloat16)
