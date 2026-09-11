"""NVFP4 (W4A4) ``linear_proj`` implementation — native build.

Single projection as one fp4 GEMM: activation quantize to NVFP4
(per-16-block swizzled scales), ``fp4_w4a16_gemm_sm120_bf16out``, BF16
output. Serves the attention Q/K/V/O projections of hosts whose forward
M exceeds the hub decode bands; kernel resolution is the PR-175 tiering
(hub first, local native build second, host floor) — this impl consumes
the local build because the hub's fp4 packages ship no torch-2.13
variant.

``weights`` is checkpoint-native ``[N, K]`` (out, in). No calibration
data required: per-16-block weight scales at bind time, per-call
activation scales.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

SUPPORT = {
    "K": {"min": 512, "max": 16384, "multiple_of": 16},
    "N": {"min": 128, "max": 65536, "multiple_of": 8},
}

_NATIVE_SYMBOLS = (
    "fp4_w4a16_gemm_sm120_bf16out",
    "quantize_bf16_to_nvfp4_swizzled",
    "quantize_bf16_to_nvfp4_swizzled_mse",
)


def _swizzled_sf_bytes(rows: int, cols: int) -> int:
    assert cols % 16 == 0
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 128 * 64


@lru_cache(maxsize=1)
def _native():
    try:
        from flash_rt import flash_rt_kernels as fk
    except ImportError:
        return None
    if any(getattr(fk, s, None) is None for s in _NATIVE_SYMBOLS):
        return None
    return fk


def _check(weights: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    w = weights["w"]
    if w.dim() != 2:
        raise ValueError(f"w must be [N, K], got {tuple(w.shape)}")
    n, k = w.shape
    for name, dim in (("K", k), ("N", n)):
        bounds = SUPPORT[name]
        if dim < bounds["min"]:
            raise ValueError(
                f"{name}={dim} outside support envelope "
                f"(min {bounds['min']})")
        if bounds.get("multiple_of") and dim % bounds["multiple_of"]:
            raise ValueError(
                f"{name}={dim} must be a multiple of "
                f"{bounds['multiple_of']}")
    return n, k


class LinearProjNvfp4(GuardedSeam, torch.nn.Module):
    """Single projection: fp4 GEMM with runtime activation scales."""

    _frt_can_fallback = True

    def __init__(self, fk, w_packed, w_sf, bias, n, k):
        super().__init__()
        self._fk = fk
        self._w_packed = w_packed
        self._w_sf = w_sf
        self._bias = bias
        self._n = n
        self._k = k
        self._ws: dict[int, dict[str, torch.Tensor]] = {}
        self._frt_arm(dtypes=CAST_OK, device=w_packed.device, k=int(k))

    def _workspace(self, m: int):
        ws = self._ws.get(m)
        if ws is not None:
            return ws
        dev = self._w_packed.device
        ws = {
            "a_packed": torch.empty(m, self._k // 2, dtype=torch.uint8,
                                    device=dev),
            "a_sf": torch.zeros(_swizzled_sf_bytes(m, self._k),
                                dtype=torch.uint8, device=dev),
            "y": torch.empty(m, self._n, dtype=torch.bfloat16, device=dev),
        }
        self._ws[m] = ws
        return ws

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        fk = self._fk
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        m = flat.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        ws = self._workspace(m)
        fk.quantize_bf16_to_nvfp4_swizzled(
            flat.data_ptr(), ws["a_packed"].data_ptr(),
            ws["a_sf"].data_ptr(), m, self._k, st)
        fk.fp4_w4a16_gemm_sm120_bf16out(
            ws["a_packed"].data_ptr(), self._w_packed.data_ptr(),
            ws["y"].data_ptr(), m, self._n, self._k,
            ws["a_sf"].data_ptr(), self._w_sf.data_ptr(), 1.0, st)
        y = ws["y"].reshape(*shape[:-1], self._n)
        if self._bias is not None:
            y = y + self._bias
        return y.type_as(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_linear":
                raise
            return getattr(super().__getattr__("host_linear"), name)


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
    *,
    original: torch.nn.Module | None = None,
):
    """Bind one projection from a dense ``[N, K]`` weight."""
    fk = _native()
    if fk is None:
        raise ValueError(
            "refused: linear_proj nvfp4_static needs the locally built "
            "flash_rt_kernels (fp4 GEMM symbols); rebuild with "
            "-DGPU_ARCH=120")
    n, k = _check(weights)
    w = weights["w"].to("cuda", torch.bfloat16).contiguous()
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(_swizzled_sf_bytes(n, k), dtype=torch.uint8,
                     device="cuda")
    fk.quantize_bf16_to_nvfp4_swizzled_mse(
        w.data_ptr(), packed.data_ptr(), sf.data_ptr(), n, k,
        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    bias = None
    b = weights.get("bias")
    if b is not None and b.numel():
        bias = b.to("cuda", torch.bfloat16).contiguous()
    bound = LinearProjNvfp4(fk, packed, sf, bias, n, k)
    if original is not None:
        bound.host_linear = original

    # bind-time smoke (AGENTS.md §2.8)
    probe = bound.forward(torch.zeros(16, k, device="cuda",
                                      dtype=torch.bfloat16))
    if probe.shape != (16, n) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: linear_proj nvfp4_static bind smoke produced "
            f"shape {tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound
