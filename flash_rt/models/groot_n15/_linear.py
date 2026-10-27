"""FlashRT -- projection lowerings of the GR00T N1.5 forward.

The forward prepares an activation once with ``a = lin.input(x, M, K)`` and
then projects it with ``lin(a, w, out, M)`` (with the bias) or
``lin.matmul(a, w, out, M)`` (without one): ``out[M, w.n] = x[M, w.k] @ W^T
(+ b)``. A lowering decides how; the stage order of the forward does not
change with it.

* :class:`Bf16Linear` -- the reference's bits: projections with a bias run
  torch's addmm algorithm, projections without one torch's matmul algorithm.
* :class:`Fp8Linear` -- projections whose weights were packed to E4M3 (one
  scale per output channel, set when the checkpoint is loaded; which ones is
  the target's ``fp8_projections``) quantize the activation per token on the
  device (``groot_n15_quant_rows_fp8``, once per activation, no
  calibration), run a cuBLASLt FP8 GEMM with unit scales and a bf16 result,
  then one pass applies the token x channel scales and the bias
  (``groot_n15_rescale_rows_bf16``). The GEMM is cuBLASLt's heuristic first
  choice, so a shape runs the same algorithm in every process: the
  candidates are equally accurate (their results differ by bf16 output
  rounding), but those differences would otherwise make the actions differ
  from one run to the next. The other projections run the BF16 algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: flash_rt_kernels entries each tier's lowering launches.
LINEAR_SYMBOLS = {
    "bf16": ("bf16_matmul_cublaslt_ref_bf16",),
    "fp8": ("bf16_matmul_cublaslt_ref_bf16", "GemmRunner"),
}


@dataclass(frozen=True)
class LinearWeights:
    """Device pointers of one projection.

    ``w`` bf16 ``[n, k]`` (0 once packed to FP8), ``b`` bf16 ``[n]`` (0 for
    none); a packed projection has ``w8`` E4M3 ``[n, k]`` and ``scale`` its
    fp32 per-channel scales ``[n]``.
    """

    w: int
    b: int
    n: int
    k: int
    w8: int = 0
    scale: int = 0


@dataclass
class Activation:
    """A bf16 activation ``[M, K]`` prepared for the projections that read it."""

    x: int
    M: int
    K: int
    quantized: bool = False


class Bf16Linear:
    """BF16 projections on the algorithms torch runs (the reference's bits)."""

    precision = "bf16"

    def __init__(self, g: Any, fvk: Any) -> None:
        self._bias = g.groot_n15_linear_bias_bf16
        self._mm = fvk.bf16_matmul_cublaslt_ref_bf16

    def input(self, x: int, M: int, K: int, stream: int) -> Activation:
        return Activation(x, M, K)

    def __call__(self, a: Activation, w: LinearWeights, out: int, M: int,
                 stream: int) -> None:
        if not w.w:
            raise ValueError("projection was packed to FP8; the bf16 lowering cannot run it")
        self._bias(a.x, w.w, w.b, out, M, w.n, w.k, stream)

    def matmul(self, a: Activation, w: LinearWeights, out: int, M: int,
               stream: int) -> None:
        if not w.w:
            raise ValueError("projection was packed to FP8; the bf16 lowering cannot run it")
        self._mm(a.x, w.w, out, M, w.n, w.k, False, stream)


class Fp8Linear(Bf16Linear):
    """FP8 for the packed projections, the BF16 algorithms for the rest.

    ``x8`` (bytes, at least the largest ``M * K`` quantized), ``row_scale``
    (fp32, ``rows`` of them) and ``one`` (an fp32 1.0) are frontend-owned
    scratch. Only the most recent activation may be quantized into it.
    """

    precision = "fp8"

    def __init__(self, g: Any, fvk: Any, *, x8: int, x8_bytes: int, row_scale: int,
                 rows: int, one: int) -> None:
        super().__init__(g, fvk)
        self._quant = g.groot_n15_quant_rows_fp8
        self._rescale = g.groot_n15_rescale_rows_bf16
        self._runner = fvk.GemmRunner()
        self._x8, self._x8_bytes = int(x8), int(x8_bytes)
        self._rs, self._rows, self._one = int(row_scale), int(rows), int(one)
        self._current: Activation | None = None

    def input(self, x: int, M: int, K: int, stream: int) -> Activation:
        self._current = Activation(x, M, K)
        return self._current

    def _fp8_input(self, a: Activation, stream: int) -> int:
        if a is not self._current:
            raise RuntimeError("an FP8 projection read an activation older than the last input()")
        if not a.quantized:
            if a.M * a.K > self._x8_bytes or a.M > self._rows:
                raise ValueError(f"FP8 activation [{a.M}, {a.K}] exceeds the quantization scratch")
            self._quant(a.x, self._x8, self._rs, a.M, a.K, stream)
            a.quantized = True
        return self._x8

    def _gemm(self, a: Activation, w: LinearWeights, out: int, M: int, stream: int) -> None:
        x8 = self._fp8_input(a, stream)
        self._runner.fp8_nt_dev(x8, w.w8, out, M, w.n, w.k, self._one, self._one, stream)

    def __call__(self, a: Activation, w: LinearWeights, out: int, M: int,
                 stream: int) -> None:
        if not w.w8:
            super().__call__(a, w, out, M, stream)
            return
        self._gemm(a, w, out, M, stream)
        self._rescale(out, self._rs, w.scale, w.b, M, w.n, stream)

    def matmul(self, a: Activation, w: LinearWeights, out: int, M: int,
               stream: int) -> None:
        if not w.w8:
            super().matmul(a, w, out, M, stream)
            return
        self._gemm(a, w, out, M, stream)
        self._rescale(out, self._rs, w.scale, 0, M, w.n, stream)


def make_linear(precision: str, g: Any, fvk: Any, **scratch: Any) -> Bf16Linear:
    """Target hook: the projection lowering of a precision tier."""
    if precision == "bf16":
        return Bf16Linear(g, fvk)
    if precision == "fp8":
        return Fp8Linear(g, fvk, **scratch)
    raise ValueError(f"GR00T N1.5 has no {precision!r} projection lowering")


def linear_symbols(precision: str) -> tuple[str, ...]:
    return LINEAR_SYMBOLS[precision]
