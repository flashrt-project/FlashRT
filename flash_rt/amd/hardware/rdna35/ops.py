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

    def _normalize(self, out, x, weight=None, bias=None, *, kind, eps,
                   weight_offset=0.0, round_affine=False, fused=False,
                   residual=None, out_sum=None):
        """Common norm math; affine rounding is an explicit numerical contract.

        ``round_affine`` rounds the normalized value, offset weight and product
        to the input dtype. Otherwise reduction and affine arithmetic stay FP32
        until the output store. A residual sum is always rounded first.
        """
        if fused:
            from ..rdna.normalization import normalize
            normalize(x, weight, bias, kind=kind, eps=eps,
                      weight_offset=weight_offset, round_affine=round_affine,
                      residual=residual, out=out, out_sum=out_sum)
            return
        if residual is not None:
            x = x + residual
            out_sum.copy_(x)
        if kind == "layer" and not round_affine:
            # ATen accumulates BF16 LayerNorm in FP32, including the affine.
            # Avoid separate input/parameter casts and their graph nodes.
            value = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        else:
            xf = x.float()
            value = (F.layer_norm(xf, (x.shape[-1],), eps=eps)
                     if kind == "layer" else
                     xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))
            if round_affine:
                value = value.to(x.dtype)
            if weight is not None:
                scale = weight if round_affine else weight.float()
                if weight_offset:
                    scale = weight_offset + scale
                value = value * scale
            if bias is not None:
                value = value + (bias if round_affine else bias.float())
        value = value.to(self.dtype)
        return value if out is None else out.copy_(value)

    def layer_norm(self, out, x, weight=None, bias=None, eps=1e-6):
        """LayerNorm with optional destination; allocate only when ``out=None``."""
        if (self.fused_large_ops and x.shape[-1] == 1152
                and weight is not None and bias is not None):
            if out is None:
                out = torch.empty_like(x)
            width = x.shape[-1]
            self.fvk.layer_norm_rdna(
                out.data_ptr(), x.data_ptr(), weight.data_ptr(),
                bias.data_ptr(), x.numel() // width, width, eps,
                self._stream(x))
            return out
        return self._normalize(out, x, weight, bias, kind="layer", eps=eps)

    def rms_norm(self, out, x, weight=None, eps=1e-6, *,
                 weight_offset=1.0, round_affine=False, fused=False) -> None:
        """RMSNorm; legacy callers retain FP32 affine with ``1 + weight``."""
        if self.fused_large_ops and weight is None and x.shape[-1] == 2048:
            width = x.shape[-1]
            self.fvk.rms_norm_rdna(
                out.data_ptr(), x.data_ptr(), x.numel() // width, width, eps,
                self._stream(x))
            return
        self._normalize(out, x, weight, kind="rms", eps=eps,
                        weight_offset=weight_offset,
                        round_affine=round_affine, fused=fused)

    def ada_layer_norm(self, out, x, scale, shift, eps=1e-5, *, fused=True):
        """Centered LayerNorm with dtype-rounded ``norm * (1 + scale) + shift``."""
        self._normalize(out, x, scale, shift, kind="layer", eps=eps,
                        weight_offset=1.0, round_affine=True, fused=fused)

    def residual_layer_norm(self, out, out_sum, update, residual,
                            scale=None, shift=None, eps=1e-5, *, fused=True):
        self._normalize(out, update, scale, shift, kind="layer", eps=eps,
                        weight_offset=1.0, round_affine=True, fused=fused,
                        residual=residual, out_sum=out_sum)

    @staticmethod
    def rotate_half(out, x, cos, sin, *, fused=True):
        """SHD rotate-half with FP32 products, unlike adjacent-pair QKV RoPE."""
        if fused:
            from ..rdna.normalization import rope
            rope(x, cos, sin, out=out)
            return
        first, second = x.float().chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        out.copy_((x.float() * cos[:, None].float()
                   + rotated * sin[:, None].float()).to(x.dtype))

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
        self._normalize(out, x, scale, shift, kind="rms", eps=eps,
                        weight_offset=1.0)
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

    def rms_norm_affine(self, x, weight, eps=1e-6, *, modulation=None):
        """BF16 affine, or FP32 AdaRMS affine/modulation before the store."""
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise TypeError("RDNA affine RMSNorm requires BF16 input and weight")
        if not x.is_contiguous() or x.shape[-1] % 4:
            raise ValueError("RDNA affine RMSNorm requires contiguous rows with a 4-aligned width")
        out = torch.empty_like(x)
        rows, width = x.numel() // x.shape[-1], x.shape[-1]
        if modulation is None:
            self.fvk.rms_norm_affine_rdna(
                x.data_ptr(), weight.data_ptr(), out.data_ptr(), rows, width, eps, self._stream(x)
            )
        else:
            gamma, beta = modulation
            if gamma.dtype != torch.float32 or beta.dtype != torch.float32:
                raise TypeError("RDNA AdaRMS modulation must be FP32")
            self.fvk.ada_rms_norm_fp32_rdna(
                x.data_ptr(), weight.data_ptr(), gamma.data_ptr(), beta.data_ptr(), out.data_ptr(),
                rows, width, eps, self._stream(x),
            )
        return out

    def residual_rms_affine(
        self, update, residual, weight, eps=1e-6, *, modulation=None, shared=None
    ):
        """Round the residual stream before affine RMSNorm or FP32 AdaRMSNorm."""
        summed, out = torch.empty_like(residual), torch.empty_like(residual)
        gamma, beta = modulation if modulation is not None else (None, None)
        width = residual.shape[-1]
        self.fvk.residual_rms_affine_rdna(
            update.data_ptr(), residual.data_ptr(), 0 if shared is None else shared.data_ptr(),
            weight.data_ptr(), 0 if gamma is None else gamma.data_ptr(),
            0 if beta is None else beta.data_ptr(), summed.data_ptr(), out.data_ptr(),
            residual.numel() // width, width, eps, self._stream(residual),
        )
        return summed, out

    def rotate_half_hip(self, x, cos, sin, *, fp32=False):
        """SHD or batch-1 BSHD input; preserve FP32 products and strided Q/K."""
        if x.dtype not in (torch.bfloat16, torch.float32) or cos.dtype != sin.dtype:
            raise TypeError("RDNA rotate-half requires BF16/FP32 input and matching tables")
        dtype = torch.float32 if fp32 else x.dtype
        if dtype == torch.float32 and cos.dtype != torch.float32:
            raise TypeError("FP32 RDNA rotate-half requires FP32 tables")
        flat = x[0] if x.ndim == 4 else x
        cosine, sine = (cos[0], sin[0]) if cos.ndim == 3 else (cos, sin)
        out = torch.empty(x.shape, dtype=dtype, device=x.device)
        self.fvk.rotate_half_rdna(
            flat.data_ptr(), cosine.data_ptr(), sine.data_ptr(), out.data_ptr(), flat.shape[0],
            flat.shape[1], flat.shape[2], flat.stride(0), flat.stride(1), cosine.stride(0),
            sine.stride(0), x.dtype == torch.float32, cos.dtype == torch.float32,
            dtype == torch.float32, self._stream(x),
        )
        return out

    def residual_rms_router(self, update, residual, weight, modulation, eps):
        """768-wide residual/AdaRMSNorm with a BF16-rounded FP32 router input."""
        summed, out = torch.empty_like(residual), torch.empty_like(residual)
        router_input = torch.empty(residual.shape, device=residual.device, dtype=torch.float32)
        gamma, beta = modulation
        self.fvk.residual_rms_router_rdna(
            update.data_ptr(), residual.data_ptr(), weight.data_ptr(), gamma.data_ptr(),
            beta.data_ptr(), summed.data_ptr(), out.data_ptr(), router_input.data_ptr(),
            residual.numel() // 768, eps, self._stream(residual),
        )
        return summed, out, router_input

    def silu_bf16(self, x, up=None):
        """Round SiLU to BF16 before the optional BF16 up multiplication."""
        out = torch.empty_like(x)
        self.fvk.silu_bf16_rdna(
            x.data_ptr(), 0 if up is None else up.data_ptr(), out.data_ptr(), x.numel(),
            self._stream(x),
        )
        return out
