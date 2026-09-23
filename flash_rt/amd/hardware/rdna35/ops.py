"""RDNA tensor operation bindings; no Pi0.5 model or layer traversal."""
from __future__ import annotations
import os
import torch
import torch.nn.functional as F


class Rdna35TensorOps:
    def __init__(self, kernels, dtype=torch.bfloat16):
        self.fvk = kernels
        self.dtype = dtype
        self.fused_rope = os.getenv("FLASHRT_RDNA35_HIP_ROPE", "1") == "1"
        self.fused_decoder_ops = (
            os.getenv("FLASHRT_RDNA35_HIP_DECODER", "1") == "1"
        )
        self.merged_decoder_ffn = (
            os.getenv("FLASHRT_RDNA35_HIP_FFN_GATE_UP", "1") == "1"
        )
        self.merged_encoder_ffn = (
            os.getenv("FLASHRT_RDNA35_HIP_ENCODER_FFN", "1") == "1"
        )
        self.fused_large_ops = (
            os.getenv("FLASHRT_RDNA35_HIP_LARGE_OPS", "1") == "1"
        )

    @staticmethod
    def _stream(tensor: torch.Tensor) -> int:
        return int(torch.cuda.current_stream(tensor.device).cuda_stream)

    def qkv_rope(
        self, q_out, k_out, v_out, qkv, rope_cos, rope_sin, position_start,
    ) -> None:
        self.fvk.qkv_rope_rdna(
            q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(),
            qkv.data_ptr(), rope_cos.data_ptr(), rope_sin.data_ptr(),
            qkv.shape[0], position_start, self._stream(qkv))

    def layer_norm(self, out, x, weight, bias, eps=1e-6) -> None:
        if self.fused_large_ops and x.shape[-1] == 1152:
            width = x.shape[-1]
            self.fvk.layer_norm_rdna(
                out.data_ptr(), x.data_ptr(), weight.data_ptr(),
                bias.data_ptr(), x.numel() // width, width, eps,
                self._stream(x))
            return
        value = F.layer_norm(
            x.float(), (x.shape[-1],), weight.float(), bias.float(), eps)
        out.copy_(value.to(self.dtype))

    def rms_norm(self, out, x, weight=None, eps=1e-6) -> None:
        if self.fused_large_ops and weight is None and x.shape[-1] == 2048:
            width = x.shape[-1]
            self.fvk.rms_norm_rdna(
                out.data_ptr(), x.data_ptr(), x.numel() // width, width, eps,
                self._stream(x))
            return
        xf = x.float()
        value = xf * torch.rsqrt(
            xf.square().mean(dim=-1, keepdim=True) + eps)
        if weight is not None:
            value = value * (1.0 + weight.float())
        out.copy_(value.to(self.dtype))

    def adarms(
        self, out, x, cond, weight, bias, eps=1e-6, modulation=None,
    ):
        if cond is not None:
            if modulation is None:
                modulation = torch.empty(
                    1, 3 * x.shape[-1], dtype=self.dtype, device=x.device)
            torch.addmm(bias, cond, weight, out=modulation)
        elif modulation is None:
            raise ValueError("precomputed AdaRMS requires a modulation tensor")
        if self.fused_decoder_ops and x.shape[-1] == 1024:
            width = x.shape[-1]
            self.fvk.adarms_rdna(
                out.data_ptr(), x.data_ptr(), modulation.data_ptr(),
                x.numel() // width, width, eps, self._stream(x))
            return modulation[:, 2 * width:]
        scale, shift, gate = modulation.chunk(3, dim=-1)
        xf = x.float()
        value = xf * torch.rsqrt(
            xf.square().mean(dim=-1, keepdim=True) + eps)
        value = value * (1.0 + scale.float()) + shift.float()
        out.copy_(value.to(self.dtype))
        return gate.to(self.dtype)

    def gelu(self, out, x) -> None:
        if self.fused_large_ops and x.shape[-1] == 4304:
            self.fvk.gelu_rdna(
                out.data_ptr(), x.data_ptr(), x.numel(), self._stream(x))
            return
        out.copy_(F.gelu(x.float(), approximate="tanh").to(self.dtype))

    def gelu_mul(self, out, gate, up) -> None:
        if (
            self.fused_decoder_ops
            and out.shape[-1] == 4096
            and gate.is_contiguous()
            and up.is_contiguous()
        ):
            self.fvk.gelu_mul_rdna(
                out.data_ptr(), gate.data_ptr(), up.data_ptr(), out.numel(),
                self._stream(out))
            return
        out.copy_((
            F.gelu(gate.float(), approximate="tanh") * up.float()
        ).to(self.dtype))

    def gelu_mul_merged(self, out, gate_up) -> None:
        if not (self.merged_decoder_ffn or self.merged_encoder_ffn):
            raise RuntimeError("merged FFN requires its HIP kernel")
        self.fvk.gelu_mul_merged_rdna(
            out.data_ptr(), gate_up.data_ptr(), out.shape[0], out.shape[1],
            self._stream(out))

    def silu(self, out, x) -> None:
        if self.fused_decoder_ops and x.shape[-1] == 1024:
            self.fvk.silu_rdna(
                out.data_ptr(), x.data_ptr(), x.numel(), self._stream(x))
            return
        out.copy_(F.silu(x.float()).to(self.dtype))

    def residual(self, out, update, residual, gate=None) -> None:
        if (
            (self.fused_decoder_ops and out.shape[-1] == 1024)
            or (
                self.fused_large_ops
                and gate is None
                and out.shape[-1] in (1152, 2048)
            )
        ):
            width = out.shape[-1]
            self.fvk.residual_rdna(
                out.data_ptr(), update.data_ptr(), residual.data_ptr(),
                0 if gate is None else gate.data_ptr(), out.numel(), width,
                self._stream(out))
            return
        value = update.float()
        if gate is not None:
            value = value * gate.float()
        out.copy_((residual.float() + value).to(self.dtype))

    def residual_rms(
        self, out_sum, out_norm, update, residual, eps=1e-6,
    ) -> None:
        if self.fused_large_ops and out_sum.shape[-1] == 2048:
            width = out_sum.shape[-1]
            self.fvk.residual_rms_rdna(
                out_sum.data_ptr(), out_norm.data_ptr(), update.data_ptr(),
                residual.data_ptr(), out_sum.numel() // width, width, eps,
                self._stream(out_sum))
            return
        self.residual(out_sum, update, residual)
        self.rms_norm(out_norm, out_sum, eps=eps)

    def residual_adarms(
        self, out_sum, out_norm, update, residual, gate, cond, weight, bias,
        eps=1e-6, modulation=None,
    ):
        if self.fused_decoder_ops and cond is None:
            if modulation is None:
                raise ValueError("fused residual-AdaRMS requires modulation")
            if out_sum.shape[-1] == 1024:
                width = out_sum.shape[-1]
                self.fvk.residual_adarms_rdna(
                    out_sum.data_ptr(), out_norm.data_ptr(), update.data_ptr(),
                    residual.data_ptr(), gate.data_ptr(), modulation.data_ptr(),
                    out_sum.numel() // width, width, eps,
                    self._stream(out_sum))
                return modulation[:, 2 * width:]
        self.residual(out_sum, update, residual, gate)
        return self.adarms(
            out_norm, out_sum, cond, weight, bias, eps, modulation)
