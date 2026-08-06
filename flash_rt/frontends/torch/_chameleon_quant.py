"""Low-bit weight quantizers for the Chameleon-7B GEMMs (Orin SM87).

Checkpoint-agnostic and frontend-agnostic: every function takes plain tensor
lists and returns plain tensors, so the same code serves any Chameleon
frontend.

Layout contract — get this wrong and the GEMM silently returns garbage:

* The declarative weight spec hands us per-projection FP16 weights in
  ``[K, N]`` row-major (``Cat``/``FusedGateUp`` followed by ``T()``).
* The CUTLASS SM80 rowwise GEMMs consume the **B** operand as ``[N, K]``
  ColumnMajor with an ``[N]`` FP32 ``RowBroadcast`` scale, so we transpose to
  ``[N, K]`` and quantize each of the N output rows symmetrically.
* The **A** operand is ``[M, K]`` RowMajor with ``M`` consecutive FP32
  ``ColBroadcast`` scales, produced at runtime by the fused norm/quant kernels.

INT4 additionally applies the QuaRot rotation ``W_rot = (H_K @ W)/sqrt(K)``
offline; the matching activation rotation happens online in the fused
RMSNorm+FHT kernels. Values are packed 2/byte with the even index in the low
nibble (``cutlass::int4b_t`` order).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

INT8_QUANT_MAX = 127.0
INT8_QUANT_EPS = 1e-12
INT4_QUANT_MAX = 7.0
INT4_QUANT_EPS = 1e-10

#: The seven per-layer GEMMs, in the order the pipeline consumes them.
PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "d")

#: Projections whose K == D (a power of two) and so can take a full-width
#: Hadamard rotation. ``d`` has K == Dff == 11008 and needs block-H128.
INT4_POW2_PROJECTIONS = ("q", "k", "v", "o", "gate", "up")


def split_fused_projections(qkv_w: List[torch.Tensor],
                            gu_w: List[torch.Tensor],
                            o_w: List[torch.Tensor],
                            d_w: List[torch.Tensor],
                            *, D: int, Dff: int) -> Dict[str, List[torch.Tensor]]:
    """Materialize the seven per-projection ``[K, N]`` weight lists.

    ``qkv_w[li]`` is ``[D, 3D]`` row-major and ``gu_w[li]`` is ``[D, 2*Dff]``:
    after ``Cat(dim=0) -> T().contiguous()`` the fused row stride is ``3D``
    (resp. ``2*Dff``), so a *byte-offset* split would read the projections
    column-interleaved. Only a ``fused[:, lo:hi].contiguous()`` slice recovers
    the original ``q_proj`` / ``k_proj`` / ``v_proj`` blocks.
    """
    out: Dict[str, List[torch.Tensor]] = {k: [] for k in PROJECTIONS}
    for li, (qkv, gu) in enumerate(zip(qkv_w, gu_w)):
        if tuple(qkv.shape) != (D, 3 * D):
            raise RuntimeError(
                f"layer {li}: expected fused qkv_w {(D, 3 * D)}, "
                f"got {tuple(qkv.shape)}")
        if tuple(gu.shape) != (D, 2 * Dff):
            raise RuntimeError(
                f"layer {li}: expected fused gu_w {(D, 2 * Dff)}, "
                f"got {tuple(gu.shape)}")
        out["q"].append(qkv[:, 0:D].contiguous())
        out["k"].append(qkv[:, D:2 * D].contiguous())
        out["v"].append(qkv[:, 2 * D:3 * D].contiguous())
        out["gate"].append(gu[:, 0:Dff].contiguous())
        out["up"].append(gu[:, Dff:2 * Dff].contiguous())
    out["o"] = list(o_w)
    out["d"] = list(d_w)
    return out


def quantize_per_row_int8(w_kn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric INT8 of a ``[K, N]`` FP16 weight.

    Returns ``(q [N, K] int8, scale [N] fp32)``, both contiguous on the
    weight's device.
    """
    w_f32 = w_kn.float().transpose(0, 1).contiguous()                    # [N, K]
    scale = torch.clamp(w_f32.abs().amax(dim=1) / INT8_QUANT_MAX,
                        min=INT8_QUANT_EPS).float().contiguous()         # [N]
    q = torch.clamp(torch.round(w_f32 / scale[:, None]),
                    -127, 127).to(torch.int8).contiguous()               # [N, K]
    return q, scale


def hadamard_gpu(n: int, device="cuda") -> torch.Tensor:
    """Unnormalised Sylvester Hadamard ``H_n`` (fp32). ``n`` must be a power of 2."""
    H = torch.ones(1, 1, dtype=torch.float32, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def _pack_int4_rows(w_rot_nk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric INT4 of an already-rotated ``[N, K]`` fp32 weight."""
    scale = (w_rot_nk.abs().amax(1) / INT4_QUANT_MAX).clamp_min(INT4_QUANT_EPS)
    q = torch.clamp(torch.round(w_rot_nk / scale[:, None]), -7, 7).to(torch.int8)
    lo = (q[:, 0::2] & 0xF).to(torch.uint8)
    hi = (q[:, 1::2] & 0xF).to(torch.uint8)
    return (lo | (hi << 4)).contiguous(), scale.float().contiguous()


class QuantizedWeights:
    """Owns the quantized tensors and exposes the pointer dicts the pipeline wants.

    ``ptr[proj][li]`` is the packed weight pointer and ``scale_ptr[proj][li]``
    the matching ``[N]`` FP32 scale pointer. The tensors are kept alive in
    ``_store`` for the object's lifetime — dropping this object invalidates
    every pointer, including any already baked into a captured CUDA graph.
    """

    def __init__(self) -> None:
        self._store: List[torch.Tensor] = []
        self.ptr: Dict[str, List[int]] = {k: [] for k in PROJECTIONS}
        self.scale_ptr: Dict[str, List[int]] = {k: [] for k in PROJECTIONS}
        self.precision: Dict[str, str] = {}

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._store)

    def _set(self, proj: str, li: int, q: torch.Tensor, s: torch.Tensor) -> None:
        self._store.append(q)
        self._store.append(s)
        while len(self.ptr[proj]) <= li:
            self.ptr[proj].append(0)
            self.scale_ptr[proj].append(0)
        self.ptr[proj][li] = q.data_ptr()
        self.scale_ptr[proj][li] = s.data_ptr()

    def _drop(self, ptr: int) -> None:
        if ptr:
            self._store = [t for t in self._store if t.data_ptr() != ptr]


def quantize_int8_all(proj: Dict[str, List[torch.Tensor]],
                      *, num_layers: int) -> QuantizedWeights:
    """INT8-quantize all seven projections for all layers."""
    qw = QuantizedWeights()
    for li in range(num_layers):
        for key in PROJECTIONS:
            w = proj[key][li]
            if w.dtype != torch.float16:
                w = w.to(torch.float16)
            qw._set(key, li, *quantize_per_row_int8(w))
        qw.precision.update({k: "int8" for k in PROJECTIONS})
    logger.info("INT8 quantized %d LLM GEMM weights (%d layers x %d proj), %.2f GB",
                num_layers * len(PROJECTIONS), num_layers, len(PROJECTIONS),
                qw.bytes() / 2 ** 30)
    return qw


def quantize_int4_quarot(qw: QuantizedWeights,
                         proj: Dict[str, List[torch.Tensor]],
                         *, num_layers: int, D: int,
                         include_down: bool = False) -> QuantizedWeights:
    """Replace the INT8 tensors of the rotatable projections with QuaRot INT4.

    ``include_down`` additionally rotates the FFN down projection with a
    **block-diagonal** ``H_128`` because its K (11008) is not a power of two.
    Unrotated per-row W4A4 measures cos 0.9494 and fails the gate, so the
    rotation is mandatory rather than an optimization.
    """
    Hm = hadamard_gpu(D) / (float(D) ** 0.5)
    keys = list(INT4_POW2_PROJECTIONS)
    Hb = None
    if include_down:
        keys.append("d")
        Hb = hadamard_gpu(128) / (128.0 ** 0.5)

    for li in range(num_layers):
        for key in keys:
            w = proj[key][li].to(torch.float32)                      # [K, N]
            if key == "d":
                Kd = w.shape[0]
                w_rot = (Hb.t() @ w.reshape(Kd // 128, 128, -1)
                         ).reshape(Kd, -1).t().contiguous()          # [N, Kd]
            else:
                if w.shape[0] != D:
                    raise RuntimeError(
                        f"{key} layer {li}: expected K={D}, got {tuple(w.shape)}")
                w_rot = (Hm @ w).t().contiguous()                    # [N, D]
            qw._drop(qw.ptr[key][li])
            qw._set(key, li, *_pack_int4_rows(w_rot))
            qw.precision[key] = "int4"
            del w, w_rot
    del Hm, Hb
    torch.cuda.empty_cache()
    logger.info("QuaRot INT4: rotated+packed %d weights (%d layers x %d proj)%s",
                num_layers * len(keys), num_layers, len(keys),
                "" if include_down else "; down stays INT8")
    return qw


def quantize_int8_hadamard(qw: QuantizedWeights,
                           proj: Dict[str, List[torch.Tensor]],
                           *, num_layers: int, D: int) -> QuantizedWeights:
    """Re-quantize the rotatable projections as **Hadamard-rotated INT8** (W8A8+QuaRot).

    This configuration sits between two tiers:

    * plain per-row INT8 — 8-bit resolution, but *unconditioned*, so a row whose
      amax is set by a massive-activation channel loses its remaining ~4090
      channels to rounding;
    * QuaRot INT4 — conditioned by the rotation, but only 15 levels.

    Rotating at 8 bits gets both. The rotation is free at inference time: the
    weight side folds offline here (``W_rot = H·W/sqrt(K)``) and the activation
    side is fused into the norm kernels (``rms_norm_fht_int8_fp16`` and friends).

    Crucially it keeps **plain per-row scales**, so the unmodified
    ``cutlass_int8_rowwise_*`` GEMMs are reused — the alternative outlier fixes
    (group-128 / block-scaled) would each need a bespoke GEMM, and the measured
    ceiling for a hand-written block-scaled s4 kernel on 16-SM Orin was only
    41 TOPS. Principle #17: pick the rotation that keeps you on the fast path.

    The FFN down projection is left as plain INT8: its K (11008) is not a power
    of two, and its input is the un-rotated BF16 SiLU output.
    """
    Hm = hadamard_gpu(D) / (float(D) ** 0.5)
    for li in range(num_layers):
        for key in INT4_POW2_PROJECTIONS:
            w = proj[key][li].to(torch.float32)                   # [K, N]
            if w.shape[0] != D:
                raise RuntimeError(
                    f"{key} layer {li}: expected K={D}, got {tuple(w.shape)}")
            w_rot = (Hm @ w).half()                               # [K, N]
            qw._drop(qw.ptr[key][li])
            qw._set(key, li, *quantize_per_row_int8(w_rot))
            qw.precision[key] = "int8+hadamard"
            del w, w_rot
    del Hm
    torch.cuda.empty_cache()
    logger.info("W8A8+Hadamard: rotated %d weights (%d layers x %d proj); "
                "down stays plain INT8",
                num_layers * len(INT4_POW2_PROJECTIONS), num_layers,
                len(INT4_POW2_PROJECTIONS))
    return qw


__all__ = [
    "INT8_QUANT_MAX", "INT4_QUANT_MAX", "PROJECTIONS", "INT4_POW2_PROJECTIONS",
    "QuantizedWeights", "split_fused_projections", "quantize_per_row_int8",
    "hadamard_gpu", "quantize_int8_all", "quantize_int8_hadamard",
    "quantize_int4_quarot",
]
