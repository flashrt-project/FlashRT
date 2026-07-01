"""FlashRT -- MiniMax-Remover NVFP4 (W4A4) Linear layer.

Replaces every eligible ``nn.Linear`` in the Transformer3DModel with a
NVFP4 W4A4 GEMM backed by the generic FlashRT NVFP4 kernels:

* Weights are quantised once at load time to NVFP4 via
  ``bf16_weight_to_nvfp4_swizzled`` (packed [N, K/2] e2m1 + tile-swizzled
  block-scale factors + a host-scalar global ``alpha``).
* Activations are quantised dynamically per call via
  ``quantize_bf16_to_nvfp4_swizzled`` (per-16-element UE4M3 block scales
  computed on-GPU, no CPU sync, no offline calibration -- this is the
  key difference from the static FP8 path).
* The GEMM ``fp4_w4a16_gemm_sm120_bf16out`` is the SM120-native W4A4
  MMA producing a bf16 output (the kernel name keeps the legacy
  ``w4a16`` token but both operands are FP4).

Shape constraints (SM120 FP4 MMA): K >= 64, K % 16 == 0, N % 16 == 0.
All MiniMax-Remover Linears (K in {5120, 13824}, N in {64, 5120, 13824})
satisfy this; ineligible layers (none in practice) keep the original
Linear. NVFP4 needs no calibration, so the calibration shims below are
no-ops kept only for API compatibility.
"""

import os
from typing import Optional

import torch
import torch.nn as nn

_BF16 = torch.bfloat16
_FP16 = torch.float16

_FP4_MIN_K = 64
_FP4_ALIGN = 16


def _sf_swizzled_bytes(rows, dim, kern):
    return kern.nvfp4_sf_swizzled_bytes(rows, dim)


def _quantize_weight_nvfp4(w, kern):
    """Quantise an [N, K] bf16 weight to NVFP4.

    Returns ``(packed[N, K/2] uint8, sfb uint8, alpha python float)``.
    """
    assert w.dtype == _BF16, f"FP4 weight quant requires bf16, got {w.dtype}"
    N, K = w.shape
    assert K % _FP4_ALIGN == 0 and N % _FP4_ALIGN == 0 and K >= _FP4_MIN_K, (
        f"FP4 weight shape unsupported N={N} K={K} "
        f"(need K>=64, N%16==0, K%16==0)")
    w = w.contiguous()
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w.device)
    sfb = torch.empty(_sf_swizzled_bytes(N, K, kern), dtype=torch.uint8, device=w.device)
    scratch_amax = torch.empty(1, dtype=torch.float32, device=w.device)
    out_global = torch.empty(1, dtype=torch.float32, device=w.device)
    kern.bf16_weight_to_nvfp4_swizzled(
        w.data_ptr(), packed.data_ptr(), sfb.data_ptr(),
        scratch_amax.data_ptr(), out_global.data_ptr(), N, K, 0)
    torch.cuda.synchronize(w.device)
    alpha = float(out_global.item())
    return packed, sfb, alpha


def _pick_gemm(kern):
    mode = os.environ.get("FLASHRT_FP4_GEMM", "pingpong").lower()
    if mode == "plain":
        return kern.fp4_w4a16_gemm_sm120_bf16out
    if mode == "widen":
        return kern.fp4_w4a16_gemm_sm120_bf16out_widen
    return kern.fp4_w4a16_gemm_sm120_bf16out_pingpong


class FlashRTNvfp4Linear(nn.Module):
    """Linear layer backed by the FlashRT NVFP4 (W4A4) GEMM.

    Computes ``y = x @ weight^T + bias`` where the weight is stored as
    NVFP4 and the activation is quantised to NVFP4 dynamically per call.
    The SM120 FP4 kernels are bf16-native; the residual stream is fp16,
    so this layer casts fp16 -> bf16 on entry and bf16 -> fp16 on exit
    (two large-tensor casts, memory-bound, ~0.04 ms/layer).
    """

    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=torch.float16, kern=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._kern = kern

        self.weight_packed = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            requires_grad=False)
        self.weight_sfb = nn.Parameter(
            torch.empty(_sf_swizzled_bytes(out_features, in_features, kern),
                        dtype=torch.uint8, device=device),
            requires_grad=False)
        self.weight_alpha = 1.0
        self.register_buffer(
            "weight_alpha_buf", torch.ones(1, dtype=torch.float32, device=device),
            persistent=False)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=_BF16, device=device),
                requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.calibrating = False
        self._gemm = _pick_gemm(kern)

    @property
    def weight(self):
        return self.weight_packed

    @classmethod
    def from_linear(cls, linear, kern):
        in_f, out_f = linear.weight.shape[1], linear.weight.shape[0]
        layer = cls(in_f, out_f, bias=linear.bias is not None,
                    device=linear.weight.device, dtype=_FP16, kern=kern)
        w_bf16 = linear.weight.data.to(_BF16)
        packed, sfb, alpha = _quantize_weight_nvfp4(w_bf16, kern)
        layer.weight_packed.data = packed
        layer.weight_sfb.data = sfb
        layer.weight_alpha = alpha
        layer.weight_alpha_buf.data = torch.tensor([alpha], dtype=torch.float32,
                                                   device=w_bf16.device)
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(_BF16)
        return layer

    def forward(self, x):
        kern = self._kern
        in_dtype = x.dtype
        if x.dtype != _BF16:
            x = x.to(_BF16)

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        if x2d.stride(0) != self.in_features or x2d.stride(1) != 1:
            x2d = x2d.contiguous()
        m = x2d.shape[0]
        k, n = self.in_features, self.out_features
        stream = torch.cuda.current_stream().cuda_stream

        a_packed = torch.empty(m, k // 2, dtype=torch.uint8, device=x2d.device)
        a_sfa = torch.empty(_sf_swizzled_bytes(m, k, kern), dtype=torch.uint8, device=x2d.device)
        kern.quantize_bf16_to_nvfp4_swizzled(
            x2d.data_ptr(), a_packed.data_ptr(), a_sfa.data_ptr(), m, k, stream)

        out = torch.empty(m, n, dtype=_BF16, device=x2d.device)
        self._gemm(
            a_packed.data_ptr(), self.weight_packed.data_ptr(), out.data_ptr(),
            m, n, k, a_sfa.data_ptr(), self.weight_sfb.data_ptr(),
            self.weight_alpha, stream)
        if self.bias is not None:
            kern.add_bias_bf16(out.data_ptr(), self.bias.data_ptr(), m, n, stream)
        out = out.view(*orig_shape[:-1], n)

        if in_dtype != _BF16:
            out = out.to(in_dtype)
        return out

    def freeze_act_scale(self, margin=1.0):
        pass


def _is_nvfp4_eligible(linear):
    if not isinstance(linear, nn.Linear):
        return False
    in_f, out_f = linear.weight.shape[1], linear.weight.shape[0]
    return (in_f % _FP4_ALIGN == 0 and out_f % _FP4_ALIGN == 0
            and in_f >= _FP4_MIN_K)


def install_flashrt_nvfp4(model, kern, verbose=False, target="all"):
    """Recursively replace eligible Linears in ``model`` with NVFP4 layers.

    ``target``:
      * ``"all"`` (default) -- every attention / ffn / proj Linear.
      * ``"ffn_only"``      -- only the FFN up/down projections (higher
        precision, the speedup comes from the large GEMMs anyway).
    Timestep / condition embedders are always skipped (precision
    sensitive and small). Returns the number of replaced layers.
    """
    replaced = 0
    skipped_shape = 0
    skip_substr = ("time_embedder", "condition_embedder")

    if target == "ffn_only":
        include_patterns = ("ffn.net.0.proj", "ffn.net.2")
    else:
        include_patterns = ()

    def _should_replace(name):
        if any(s in name for s in skip_substr):
            return False
        if include_patterns:
            return any(p in name for p in include_patterns)
        return True

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _should_replace(name):
            continue
        targets.append((name, module))

    for name, linear in targets:
        if not _is_nvfp4_eligible(linear):
            skipped_shape += 1
            continue
        parent = model
        for p in name.split(".")[:-1]:
            parent = getattr(parent, p)
        attr = name.split(".")[-1]
        setattr(parent, attr, FlashRTNvfp4Linear.from_linear(linear, kern))
        replaced += 1

    return replaced


def set_calibration(model, on):
    for module in model.modules():
        if isinstance(module, FlashRTNvfp4Linear):
            module.calibrating = on


def freeze_calibration(model, margin=1.0):
    n = 0
    for module in model.modules():
        if isinstance(module, FlashRTNvfp4Linear):
            module.freeze_act_scale(margin)
            n += 1
    return n
