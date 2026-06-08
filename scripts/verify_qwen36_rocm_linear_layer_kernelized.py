from __future__ import annotations
import os

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_qwen36_rocm_linear_layer_bringup import (
    causal_conv1d_reference,
    gated_deltanet_reference,
    gated_rms_norm_silu,
    rms_norm,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    )


def main() -> None:
    import flash_rt.flash_rt_rocm_kernels as kernels
    from flash_rt.frontends.torch.qwen36_rocm_weights import (
        extract_weights_qwen36_fp8_rocm,
    )

    handles = extract_weights_qwen36_fp8_rocm(
        os.environ.get("QWEN36_MODEL", "Qwen/Qwen3.6-27B-FP8"),
        max_layers=1,
        weight_mode="bf16_dequant",
    )
    by_ptr = {int(t.data_ptr()): t for t in handles.anchors}
    layer = handles.ptrs["layers"][0]
    eps = float(handles.ptrs["rms_norm_eps"])

    def t(name: str) -> torch.Tensor:
        return by_ptr[int(layer[name])]

    def linear(x: torch.Tensor, name: str) -> torch.Tensor:
        return kernels.hipblaslt_linear_bf16(x.contiguous(), t(name + "_w"))

    seq = 4
    hidden = 5120
    h = torch.randn(seq, hidden, device="cuda", dtype=torch.bfloat16)

    # Shared projection front half.
    x = rms_norm(h, t("input_norm_eff_w"), eps)
    qkv = linear(x, "in_proj_qkv")
    z = linear(x, "in_proj_z").view(seq, 48, 128)
    a = kernels.hipblaslt_linear_bf16(x.contiguous(), t("in_proj_a_w"))
    b = kernels.hipblaslt_linear_bf16(x.contiguous(), t("in_proj_b_w"))

    # Reference linear-attn core.
    qkv_conv_ref = causal_conv1d_reference(qkv, t("conv1d_w"))
    q16 = qkv_conv_ref[:, :2048].view(seq, 16, 128)
    k16 = qkv_conv_ref[:, 2048:4096].view(seq, 16, 128)
    v48 = qkv_conv_ref[:, 4096:].view(seq, 48, 128)
    attn_ref = gated_deltanet_reference(q16, k16, v48, a, b, t("A_log"), t("dt_bias"))
    norm_ref = gated_rms_norm_silu(
        attn_ref.reshape(seq * 48, 128),
        z.reshape(seq * 48, 128),
        t("head_norm_w"),
        eps,
    )

    # Kernelized linear-attn core.
    qkv_conv = torch.empty_like(qkv)
    kernels.qwen36_causal_conv1d_bf16_out(qkv, t("conv1d_w"), qkv_conv, seq, 10240, 4, True)
    q = torch.empty(seq, 48, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    kernels.qwen36_lin_split_qkv_broadcast_bf16_out(qkv_conv, q, k, v, seq)
    g = torch.empty(seq, 48, device="cuda", dtype=torch.bfloat16)
    beta = torch.empty_like(g)
    kernels.qwen36_gdn_gating_bf16_out(a, b, t("A_log"), t("dt_bias"), g, beta, seq, 48)
    state = torch.zeros(48, 128, 128, device="cuda", dtype=torch.float32)
    attn = torch.empty(seq, 48, 128, device="cuda", dtype=torch.bfloat16)
    kernels.qwen36_gated_deltanet_recurrent_bf16_out(q, k, v, g, beta, state, attn, seq, 48, 128)
    norm = torch.empty(seq * 48, 128, device="cuda", dtype=torch.bfloat16)
    kernels.qwen36_rms_norm_gated_silu_bf16_out(
        attn.reshape(seq * 48, 128),
        z.reshape(seq * 48, 128),
        t("head_norm_w"),
        norm,
        seq * 48,
        128,
        eps,
    )

    # Shared tail.
    attn_proj = linear(norm.view(seq, 6144), "out_proj")
    h_post = (h + attn_proj).to(torch.bfloat16)
    x_mlp = rms_norm(h_post, t("post_attn_norm_eff_w"), eps)
    gate = linear(x_mlp, "mlp_gate")
    up = linear(x_mlp, "mlp_up")
    act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    out = (h_post + linear(act, "mlp_down")).to(torch.bfloat16)

    ref_attn_proj = linear(norm_ref.view(seq, 6144), "out_proj")
    ref_h_post = (h + ref_attn_proj).to(torch.bfloat16)
    ref_x_mlp = rms_norm(ref_h_post, t("post_attn_norm_eff_w"), eps)
    ref_gate = linear(ref_x_mlp, "mlp_gate")
    ref_up = linear(ref_x_mlp, "mlp_up")
    ref_act = (F.silu(ref_gate.float()) * ref_up.float()).to(torch.bfloat16)
    ref_out = (ref_h_post + linear(ref_act, "mlp_down")).to(torch.bfloat16)

    torch.cuda.synchronize()
    for name, got, ref in (
        ("conv", qkv_conv, qkv_conv_ref),
        ("attn", attn, attn_ref),
        ("norm", norm, norm_ref),
        ("layer_out", out, ref_out),
    ):
        diff = (got.float() - ref.float()).abs()
        print(
            name,
            "cos",
            f"{cosine(got, ref):.8f}",
            "max_abs",
            float(diff.max().item()),
            "mean_abs",
            float(diff.mean().item()),
            "finite",
            bool(torch.isfinite(got.float()).all().item()),
        )


if __name__ == "__main__":
    main()
