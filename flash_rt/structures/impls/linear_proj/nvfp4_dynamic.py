"""NVFP4 (W4A4, dynamic activation scales) ``linear_proj`` implementation.

Weights are packed to NVFP4 (E2M1 data plus per-16-element-block scale
factors) at bind time; activations are quantized to the same format at
runtime, per call, with dynamically computed block scales — no
calibration data at either end. This is the execution form behind the
27B enablement line: checkpoints whose upstream loader decompresses
4-bit weights to BF16 inside ``forward`` (and therefore cannot fit the
card) run on the same card once their projections consume the packed
layout directly.

The ``flashrt/fp4-gemm`` entry point ``fp4_w4a16_linear_bf16`` takes the
pre-quantized activation tensor plus its scale factors — despite the
``a16`` in its historical name, the GEMM it runs is W4A4. ``variant=2``
is the qualified dispatch across the decode and short-prefill shapes
this impl serves (same-token 1.0000 against an exact reference on the
27B host, decode and prefill both through this path).

There is no host fallback: the module this replaces holds packed
weights the host cannot execute. The guard therefore refuses instead of
falling back, and adoption of a whole checkpoint is a load-time
transform, not a reversible attachment.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp4-gemm",
    "version": ">=1",
}

#: mirrors the kernel's own shape checks (``torch_binding.cpp``: K
#: divisible by 16 for the per-block scale factors, positive dims) —
#: no invented size walls: the adoption path serves whatever the
#: checkpoint author packed, and the 27B host's 17408-wide FFN is a
#: qualified shape, not an edge case
SUPPORT = {
    "K": {"min": 16, "multiple_of": 16},
    "N": {"min": 1},
}


@lru_cache(maxsize=1)
def _kernel():
    from flash_rt.structures.impls import hub_kernel

    return hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])


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
    b = weights.get("b")
    if b is not None and tuple(b.shape) != (n,):
        raise ValueError(
            f"bias shape {tuple(b.shape)} does not match N={n}")
    return n, k


def _quantize_activation(kern, flat: torch.Tensor):
    """Use the direct BF16 producer when the installed artifact carries it."""
    if flat.dtype is torch.bfloat16:
        direct = getattr(kern, "quantize_fp4_sfa_bf16", None)
        if direct is not None:
            return direct(flat.contiguous())
    return kern.quantize_fp4_sfa_fp16(
        flat.to(torch.float16).contiguous())


class LinearProjNvfp4Dynamic(GuardedSeam, torch.nn.Module):
    """Packed-weight projection: FP4 GEMM with runtime activation scales."""

    _frt_can_fallback = False

    def __init__(self, w_packed, w_sfb, bias, n, k):
        super().__init__()
        self.register_buffer("_w_packed", w_packed)
        self.register_buffer("_w_sfb", w_sfb)
        self._bias = bias
        self._n = n
        self._k = k
        kern = _kernel()
        self._kern = kern
        self._gemm = kern.fp4_w4a16_linear_bf16
        # M=1 decode rows route to the warp-split GEMV where the build
        # carries it and the shape qualifies (its own contract: N%8,
        # K a multiple of 64*warps). Absence is not a refusal - the
        # tiled GEMM serves every shape correctly, the GEMV just fills
        # the SMs it underfills at long-K decode shapes.
        # The entry's presence in the build is not its qualification to
        # run: the aarch64 package carries it and the kernel refuses at
        # call time on anything below SM120, which surfaces as a runtime
        # error on the first M=1 row rather than as a choice made here.
        # The engine adapters already withhold it off SM120; deciding it
        # once, where the arm is selected, means an impl bound directly -
        # through structures.get(), or any hand assembly - behaves the
        # same as one bound through a door.
        gemv = getattr(kern, "fp4_w4a4_gemv_warpsplit_bf16", None)
        cc = (torch.cuda.get_device_capability(w_packed.device)
              if w_packed.is_cuda else (0, 0))
        self._gemv = (gemv if gemv is not None and cc >= (12, 0)
                      and n % 8 == 0 and k % (64 * 4) == 0 else None)
        self._frt_arm(dtypes=CAST_OK, device=w_packed.device, k=int(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        a_packed, a_sfa = _quantize_activation(self._kern, flat)
        if flat.shape[0] == 1 and self._gemv is not None:
            y = self._gemv(a_packed, self._w_packed, a_sfa, self._w_sfb)
        else:
            y = self._gemm(a_packed, self._w_packed, a_sfa, self._w_sfb,
                           variant=2)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*shape[:-1], self._n).type_as(x)


@torch.no_grad()
def bind_proj_seam(
    weights: Mapping[str, torch.Tensor],
) -> tuple[LinearProjNvfp4Dynamic, float]:
    """Bind one projection from a dense ``[N, K]`` weight.

    The weight is packed to the Hub kernel's NVFP4 layout on the GPU;
    the returned float is the pack-and-unpack relative L2 against the
    input weight — the *conversion* cost of regridding into this layout,
    reported so a caller adopting a whole checkpoint can put it in the
    receipt instead of losing it.
    """
    n, k = _check(weights)
    kern = _kernel()
    w = weights["w"].to("cuda", torch.float16).contiguous()
    w_packed, w_sfb = kern.quantize_fp4_sfa_fp16(w, is_sfb=True)
    # the conversion check accumulates in row slabs: a whole-tensor
    # FP32 dequant doubles the bind's transient footprint, and on
    # head-class weights that spike is what fails under a tight budget
    deq = kern.dequantize_fp4_sfa_fp16(w_packed, w_sfb)
    num_sq = den_sq = 0.0
    for i in range(0, n, 4096):
        diff = deq[i:i + 4096].float() - w[i:i + 4096].float()
        num_sq += float(diff.square().sum())
        den_sq += float(w[i:i + 4096].float().square().sum())
    del deq
    rel = (num_sq ** 0.5) / max(den_sq ** 0.5, 1e-12)
    bias = weights.get("b")
    if bias is not None:
        bias = bias.detach().to("cuda", torch.bfloat16)
    bound = LinearProjNvfp4Dynamic(w_packed, w_sfb, bias, n, k)
    # bind-time smoke: one M=1 launch through the real entry point before
    # the seam is handed out — a stale build or missing symbol surfaces
    # as a clean bind refusal, not later inside the host's forward
    probe = bound(torch.zeros(1, k, device=w_packed.device,
                              dtype=torch.bfloat16))
    if probe.shape != (1, n) or not torch.isfinite(probe).all():
        raise ValueError(
            f"refused: nvfp4 bind smoke produced shape "
            f"{tuple(probe.shape)}, finite={bool(torch.isfinite(probe).all())}")
    return bound, rel
