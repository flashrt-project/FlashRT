# Fused MoE token generation kernels.
#
# Vendored gpt-oss MoE pipeline. See `nki_kernels/_vendor_meta.md` for the
# rationale; the gpt-oss megakernel imports these instead of nkilib so that
# 24 per-layer invocations don't produce duplicate op names. Each entry
# point accepts a `name_prefix` kwarg that the megakernel sets to
# `f"L{layer_idx}_moe_"`.
from flash_rt.neuron._vendor.qwen36.nki_kernels._legacy.norm.rmsnorm_tkg import rmsnorm_tkg

from .moe_tkg import moe_tkg
from .router_topk import (
    XHBMLayout_H_T__0,
    XHBMLayout_T_H__1,
    XSBLayout_tp102__0,
    XSBLayout_tp201__2,
    XSBLayout_tp2013__1,
    router_topk,
)

__all__ = [
    "moe_tkg",
    "rmsnorm_tkg",
    "router_topk",
    "XHBMLayout_H_T__0",
    "XHBMLayout_T_H__1",
    "XSBLayout_tp102__0",
    "XSBLayout_tp201__2",
    "XSBLayout_tp2013__1",
]
