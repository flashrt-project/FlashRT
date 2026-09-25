"""BF16 hipBLASLt GEMM provider for AMD RDNA 3.5."""

from __future__ import annotations

import os

import torch


class Rdna35GemmBackend:
    """Run BF16 GEMMs through an instance-local hipBLASLt algorithm cache."""

    def __init__(self, kernels, dtype: torch.dtype = torch.bfloat16):
        if dtype is not torch.bfloat16:
            raise TypeError(f"RDNA 3.5 GEMM supports BF16 only, got {dtype}")
        self.dtype = dtype
        self.fvk = kernels
        self._runner = kernels.RdnaGemmRunner()
        self.autotune = (
            os.getenv("FLASHRT_RDNA35_GEMM_AUTOTUNE", "1") == "1"
        )
        if self.autotune:
            try:
                pool = int(os.getenv("FLASHRT_RDNA35_GEMM_ALGOS", "16"))
            except ValueError as exc:
                raise ValueError(
                    "FLASHRT_RDNA35_GEMM_ALGOS must be a positive integer"
                ) from exc
            if pool < 1:
                raise ValueError(
                    "FLASHRT_RDNA35_GEMM_ALGOS must be a positive integer")
            self._runner.enable_lazy_autotune(pool)
        self.hip_smallm = (
            os.getenv("FLASHRT_RDNA35_HIP_SMALLM", "1") == "1"
        )
        self._smallm_weights: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @staticmethod
    def _stream(tensor: torch.Tensor) -> int:
        return int(torch.cuda.current_stream(tensor.device).cuda_stream)

    def _smallm_gemm(
        self, out, x, weight_nt, bias=None, *, accumulate=False,
    ) -> None:
        function = (
            self.fvk.smallm_wmma_bf16_residual_rdna
            if accumulate else self.fvk.smallm_wmma_bf16_rdna
        )
        function(
            out.data_ptr(), x.data_ptr(), weight_nt.data_ptr(),
            0 if bias is None else bias.data_ptr(), x.shape[0], out.shape[1],
            x.shape[1], self._stream(x))

    def prepare_smallm_weight(self, weight: torch.Tensor) -> None:
        """Create the persistent N-by-K layout consumed by gfx11 WMMA.

        Only the measured decoder action projection is currently routed, so
        enabling this adds 64 KiB rather than duplicating all model weights.
        Preparation happens before graph capture and never on the hot path.
        """
        if not self.hip_smallm:
            return
        if (
            weight.device.type != "cuda"
            or weight.dtype != self.dtype
            or weight.ndim != 2
            or weight.shape != (1024, 32)
            or not weight.is_contiguous()
        ):
            raise ValueError("unsupported RDNA 3.5 small-M weight")
        self._smallm_weights[weight.data_ptr()] = (
            weight, weight.t().contiguous())

    def linear(self, out, x, weight, bias=None):
        tensors = (out, x, weight) if bias is None else (
            out, x, weight, bias)
        if any(t.device.type != "cuda" for t in tensors):
            raise ValueError("RDNA 3.5 GEMM tensors must be on the ROCm device")
        if any(t.device != x.device for t in tensors):
            raise ValueError("RDNA 3.5 GEMM tensors must be on the same device")
        if any(t.dtype != self.dtype for t in tensors):
            raise TypeError("RDNA 3.5 GEMM tensors must be BF16")
        if x.ndim < 2 or out.ndim < 2 or weight.ndim != 2:
            raise ValueError(
                f"unsupported linear rank: x={x.ndim} weight={weight.ndim} "
                f"out={out.ndim}")
        if not x.is_contiguous():
            raise ValueError("RDNA 3.5 GEMM input must be contiguous")
        if out.stride(-1) != 1:
            raise ValueError(
                "RDNA 3.5 GEMM output must have unit column stride")
        if weight.stride(1) != 1:
            raise ValueError(
                "RDNA 3.5 GEMM weight must be row-major with unit column stride")
        expected = (*x.shape[:-1], weight.shape[1])
        if x.shape[-1] != weight.shape[0] or out.shape != expected:
            raise ValueError(
                f"unsupported linear shape: x={tuple(x.shape)} "
                f"weight={tuple(weight.shape)} out={tuple(out.shape)}")
        x2 = x.reshape(-1, x.shape[-1])
        if out.ndim == 2:
            out2 = out
        elif out.is_contiguous():
            out2 = out.reshape(-1, out.shape[-1])
        else:
            raise ValueError(
                "RDNA 3.5 batched GEMM output must be contiguous")
        output_stride = out2.stride(0)
        if output_stride < out2.shape[1]:
            raise ValueError(
                "RDNA 3.5 GEMM output row stride is too small")
        if bias is not None and (
            bias.shape != (out2.shape[1],) or not bias.is_contiguous()
        ):
            raise ValueError("unsupported RDNA 3.5 GEMM bias")
        packed = self._smallm_weights.get(weight.data_ptr())
        weight_nt = (
            packed[1]
            if packed is not None and packed[0] is weight
            else None
        )
        if (
            weight_nt is not None
            and x2.shape[0] <= 48
            and out2.is_contiguous()
        ):
            self._smallm_gemm(out2, x2, weight_nt, bias)
            return
        stream = int(torch.cuda.current_stream(x.device).cuda_stream)
        if bias is None:
            self._runner.bf16_nn(
                x2.data_ptr(), weight.data_ptr(), out2.data_ptr(),
                x2.shape[0], out2.shape[1], x2.shape[1], weight.stride(0),
                output_stride, stream)
        else:
            self._runner.bf16_nn_bias(
                x2.data_ptr(), weight.data_ptr(), out2.data_ptr(),
                bias.data_ptr(), x2.shape[0], out2.shape[1], x2.shape[1],
                weight.stride(0), output_stride, stream)

    def linear_residual(self, out, x, weight, bias=None) -> bool:
        """Fuse a prepared small-M projection into an in-place residual."""
        tensors = (out, x, weight) if bias is None else (
            out, x, weight, bias)
        if any(t.device.type != "cuda" for t in tensors):
            raise ValueError("RDNA 3.5 GEMM tensors must be on the ROCm device")
        if any(t.device != x.device for t in tensors):
            raise ValueError("RDNA 3.5 GEMM tensors must be on the same device")
        if any(t.dtype != self.dtype for t in tensors):
            raise TypeError("RDNA 3.5 GEMM tensors must be BF16")
        if x.ndim < 2 or weight.ndim != 2:
            raise ValueError(
                f"unsupported linear-residual rank: x={x.ndim} "
                f"weight={weight.ndim}")
        expected = (*x.shape[:-1], weight.shape[1])
        if (
            x.shape[-1] != weight.shape[0]
            or out.shape != expected
        ):
            raise ValueError(
                f"unsupported linear-residual shape: x={tuple(x.shape)} "
                f"weight={tuple(weight.shape)} out={tuple(out.shape)}")
        if bias is not None and (
            bias.shape != (out.shape[-1],) or not bias.is_contiguous()
        ):
            raise ValueError("unsupported RDNA 3.5 GEMM bias")
        if not x.is_contiguous() or not out.is_contiguous():
            return False
        x2 = x.reshape(-1, x.shape[-1])
        out2 = out.reshape(-1, out.shape[-1])
        packed = self._smallm_weights.get(weight.data_ptr())
        weight_nt = (
            packed[1]
            if packed is not None and packed[0] is weight
            else None
        )
        if (
            weight_nt is None
            or x2.shape[0] > 48
        ):
            return False
        self._smallm_gemm(
            out2, x2, weight_nt, bias, accumulate=True)
        return True
