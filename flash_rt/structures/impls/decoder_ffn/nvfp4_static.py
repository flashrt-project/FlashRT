"""NVFP4 (W4A4) implementation of the ``decoder_ffn`` structure — native build.

The local native build's FP4 GEMM tier serves the MLP seam: activation
quantize to NVFP4 (per-16-block swizzled scales), gate/up GEMM, fused
SiLU + quantize, down GEMM. Unlike the ``w8a16``/``w4a16`` hub decode
bands (M in [1, 8]), the native fp4 GEMM covers the MaskGIT-scale M, so
this backend is the prefill/large-M form.

Kernel resolution follows the PR-175 tiering: hub artifact first, the
local native build second, the retained host module always the floor.
This impl consumes the local native build (``flash_rt.flash_rt_kernels``)
because the hub's fp4 packages do not ship a torch-2.13 variant.

Boundary: normed activations in, FFN output out (BF16) — the host's
input-layernorm precedes the MLP, so no norm is fused here. Weights are
checkpoint-native ``[out, in]`` (``w_gate``/``w_up``: ``[F, D]``,
``w_down``: ``[D, F]``); no calibration data is required (per-16-block
weight scales at bind time, per-16-block activation scales per call).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

SUPPORT = {
    "D": {"min": 512, "max": 16384, "multiple_of": 64},
    "F": {"min": 1024, "max": 16384, "multiple_of": 64},
    # the native fp4 GEMM serves any M the host throws at it
    "M": {"min": 1, "max": 1 << 30},
    "m_classes": ("micro", "small", "medium", "large"),
}

#: kernels this backend needs from the local native build
_NATIVE_SYMBOLS = (
    "fp4_w4a16_gemm_sm120_bf16out",
    "fp4_w4a16_gemm_sm120_bf16out_pingpong",
    "quantize_bf16_to_nvfp4_swizzled",
    "quantize_bf16_to_nvfp4_swizzled_mse",
    "silu_mul_merged_to_nvfp4_swizzled_bf16",
)


def _swizzled_sf_bytes(rows: int, cols: int) -> int:
    assert cols % 16 == 0
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 128 * 64


@lru_cache(maxsize=1)
def _native():
    """The locally built native extension, or None when absent.

    Absence is a bind refusal, not a silent host path: a seam bound
    without its kernels would fall back on every call and the ledger
    would count a lie.
    """
    try:
        from flash_rt import flash_rt_kernels as fk
    except ImportError:
        return None
    if any(getattr(fk, s, None) is None for s in _NATIVE_SYMBOLS):
        return None
    return fk


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w_gate, w_up, w_down = (weights["w_gate"], weights["w_up"],
                            weights["w_down"])
    dim_f, dim_d = w_gate.shape
    if w_up.shape != (dim_f, dim_d) or w_down.shape != (dim_d, dim_f):
        raise ValueError(
            f"inconsistent weight dims: gate {tuple(w_gate.shape)}, "
            f"up {tuple(w_up.shape)}, down {tuple(w_down.shape)}")
    for name, dim in (("D", dim_d), ("F", dim_f)):
        bounds = SUPPORT[name]
        if not bounds["min"] <= dim <= bounds["max"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"[{bounds['min']}, {bounds['max']}]")
        if dim % bounds["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a multiple of "
                f"{bounds['multiple_of']}")
    return dim_d, dim_f


class BoundDecoderFfnNvfp4:
    """MLP-seam callable: normed activations in, FFN output out (BF16)."""

    def __init__(self, fk, gate_up_packed, gate_up_sf,
                 down_packed, down_sf, dim_d, dim_f):
        self._fk = fk
        self._gate_up_packed = gate_up_packed
        self._gate_up_sf = gate_up_sf
        self._down_packed = down_packed
        self._down_sf = down_sf
        self._dim_d = dim_d
        self._dim_f = dim_f
        self._gu_variant = "pingpong" if 2 * dim_f >= 4096 else "default"
        # per-row-count workspace cache: the host (MaskGIT) keeps a
        # constant row count per generation, so the first call allocates
        # and every later step reuses — no per-call empty()/zeros()
        self._ws: dict[int, dict[str, torch.Tensor]] = {}

    def _workspace(self, m: int, d: int):
        ws = self._ws.get(m)
        if ws is not None:
            return ws
        dev = self._gate_up_packed.device
        ws = {
            "inp_packed": torch.empty(m, d // 2, dtype=torch.uint8,
                                      device=dev),
            "inp_sf": torch.zeros(_swizzled_sf_bytes(m, d),
                                  dtype=torch.uint8, device=dev),
            "dg": torch.empty(m, 2 * self._dim_f, dtype=torch.bfloat16,
                              device=dev),
            "act_packed": torch.empty(m, self._dim_f // 2,
                                      dtype=torch.uint8, device=dev),
            "act_sf": torch.zeros(_swizzled_sf_bytes(m, self._dim_f),
                                  dtype=torch.uint8, device=dev),
            "out": torch.empty(m, d, dtype=torch.bfloat16, device=dev),
        }
        self._ws[m] = ws
        return ws

    def ffn(self, normed: torch.Tensor) -> torch.Tensor:
        fk = self._fk
        shape = normed.shape
        x = normed.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        m = x.shape[0]
        d = x.shape[-1]
        st = torch.cuda.current_stream().cuda_stream
        ws = self._workspace(m, d)

        fk.quantize_bf16_to_nvfp4_swizzled(
            x.data_ptr(), ws["inp_packed"].data_ptr(),
            ws["inp_sf"].data_ptr(), m, d, st)

        if self._gu_variant == "pingpong":
            fk.fp4_w4a16_gemm_sm120_bf16out_pingpong(
                ws["inp_packed"].data_ptr(), self._gate_up_packed.data_ptr(),
                ws["dg"].data_ptr(), m, 2 * self._dim_f, d,
                ws["inp_sf"].data_ptr(), self._gate_up_sf.data_ptr(),
                1.0, st)
        else:
            fk.fp4_w4a16_gemm_sm120_bf16out(
                ws["inp_packed"].data_ptr(), self._gate_up_packed.data_ptr(),
                ws["dg"].data_ptr(), m, 2 * self._dim_f, d,
                ws["inp_sf"].data_ptr(), self._gate_up_sf.data_ptr(),
                1.0, st)

        fk.silu_mul_merged_to_nvfp4_swizzled_bf16(
            ws["dg"].data_ptr(), ws["act_packed"].data_ptr(),
            ws["act_sf"].data_ptr(), m, self._dim_f, st)

        fk.fp4_w4a16_gemm_sm120_bf16out(
            ws["act_packed"].data_ptr(), self._down_packed.data_ptr(),
            ws["out"].data_ptr(), m, d, self._dim_f,
            ws["act_sf"].data_ptr(), self._down_sf.data_ptr(), 1.0, st)
        return ws["out"].reshape(shape).to(normed.dtype)

    __call__ = ffn


class FusedGluMlpNvfp4(GuardedSeam, torch.nn.Module):
    """MLP-seam module backed by the native NVFP4 FFN kernels.

    ``original`` is retained whole (host MLP naming varies across model
    families), and attribute lookups fall through to it so hosts that
    introspect the module they call keep working.
    """

    _frt_host_attr = "host_mlp"
    _frt_can_fallback = True

    def __init__(self, bound: BoundDecoderFfnNvfp4,
                 original: torch.nn.Module | None = None):
        super().__init__()
        self._bound = bound
        if original is not None:
            self.host_mlp = original
        guard = self._frt_arm(dtypes=CAST_OK,
                              device=bound._gate_up_packed.device,
                              k=int(bound._dim_d))
        guard.notes["backend"] = "nvfp4_static"

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_mlp":
                raise
            return getattr(super().__getattr__("host_mlp"), name)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(hidden)
        if admitted is not PROCEED:
            return admitted
        return self._bound.ffn(hidden)


@torch.no_grad()
def bind_mlp_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    variant: Mapping[str, str],
    original: torch.nn.Module | None = None,
):
    """Bind the MLP-seam slice of ``decoder_ffn`` with native NVFP4.

    ``weights`` uses checkpoint-native ``[out, in]`` projection layout
    (``w_gate``/``w_up``: ``[F, D]``, ``w_down``: ``[D, F]``). Weights
    are packed at bind time with MSE per-16-block scales; activations
    are quantized per call, so no calibration data is required.
    """
    if variant.get("activation", "silu") != "silu":
        raise ValueError(
            f"refused: nvfp4_static serves the SiLU gated FFN only, "
            f"got activation {variant.get('activation')!r}")
    fk = _native()
    if fk is None:
        raise ValueError(
            "refused: nvfp4_static needs the locally built "
            "flash_rt_kernels (flash_rt_kernels + fp4 GEMM symbols); "
            "rebuild with -DGPU_ARCH=120")
    dim_d, dim_f = _check(weights)
    dev = weights["w_gate"].device

    def _pack(w: torch.Tensor):
        w = w.to("cuda", torch.bfloat16).contiguous()
        n, k = w.shape
        packed = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
        sf = torch.zeros(_swizzled_sf_bytes(n, k), dtype=torch.uint8,
                         device="cuda")
        fk.quantize_bf16_to_nvfp4_swizzled_mse(
            w.data_ptr(), packed.data_ptr(), sf.data_ptr(), n, k,
            torch.cuda.current_stream().cuda_stream)
        return packed, sf

    gate_up = torch.cat([weights["w_gate"], weights["w_up"]], dim=0)
    gate_up_packed, gate_up_sf = _pack(gate_up)
    down_packed, down_sf = _pack(weights["w_down"])
    torch.cuda.synchronize()

    bound = BoundDecoderFfnNvfp4(
        fk, gate_up_packed, gate_up_sf, down_packed, down_sf,
        dim_d, dim_f)

    # bind-time smoke: one launch through the real entry point before the
    # seam is handed out (AGENTS.md §2.8). The probe carries the host's
    # real rank/shape class — 2D, BF16, CUDA.
    probe = bound.ffn(torch.zeros(16, dim_d, device=dev,
                                  dtype=torch.bfloat16))
    if probe.shape != (16, dim_d) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: nvfp4_static bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return FusedGluMlpNvfp4(bound, original=original)
