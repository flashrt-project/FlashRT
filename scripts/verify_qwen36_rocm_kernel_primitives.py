from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_qwen36_rocm_linear_layer_reference import (
    causal_conv1d_reference,
    gated_deltanet_reference,
    gated_rms_norm_silu,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    )


def check(name: str, out: torch.Tensor, ref: torch.Tensor, *, atol: float = 0.0) -> None:
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs()
    print(
        name,
        "cos",
        f"{cosine(out, ref):.8f}",
        "max_abs",
        float(diff.max().item()),
        "mean_abs",
        float(diff.mean().item()),
        "finite",
        bool(torch.isfinite(out.float()).all().item()),
    )
    if not torch.isfinite(out.float()).all():
        raise RuntimeError(f"{name} produced non-finite values")
    if float(diff.max().item()) > atol:
        raise RuntimeError(f"{name} max_abs exceeded {atol}")


def main() -> None:
    import flash_rt.flash_rt_rocm_kernels as k

    torch.manual_seed(0)
    rows = 3

    qkv = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    conv_w = torch.randn(10240, 4, device="cuda", dtype=torch.bfloat16) * 0.05
    conv_out = torch.empty_like(qkv)
    k.qwen36_causal_conv1d_bf16_out(qkv, conv_w, conv_out, rows, 10240, 4, True)
    check("causal_conv1d", conv_out, causal_conv1d_reference(qkv, conv_w), atol=0.03125)

    q = torch.empty(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
    kk = torch.empty_like(q)
    v = torch.empty_like(q)
    k.qwen36_lin_split_qkv_broadcast_bf16_out(conv_out, q, kk, v, rows)
    q_ref = conv_out[:, :2048].view(rows, 16, 128).repeat_interleave(3, dim=1)
    k_ref = conv_out[:, 2048:4096].view(rows, 16, 128).repeat_interleave(3, dim=1)
    v_ref = conv_out[:, 4096:].view(rows, 48, 128)
    check("split_q", q, q_ref)
    check("split_k", kk, k_ref)
    check("split_v", v, v_ref)

    a = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(48, device="cuda", dtype=torch.bfloat16) * 0.01
    dt_bias = torch.randn(48, device="cuda", dtype=torch.bfloat16) * 0.01
    g = torch.empty_like(a)
    beta = torch.empty_like(a)
    k.qwen36_gdn_gating_bf16_out(a, b, A_log, dt_bias, g, beta, rows, 48)
    g_ref = (-torch.exp(A_log.float())[None, :] * F.softplus(a.float() + dt_bias.float()[None, :])).to(torch.bfloat16)
    beta_ref = torch.sigmoid(b.float()).to(torch.bfloat16)
    check("gating_g", g, g_ref, atol=0.0078125)
    check("gating_beta", beta, beta_ref, atol=0.00390625)

    state = torch.zeros(48, 128, 128, device="cuda", dtype=torch.float32)
    out = torch.empty(rows, 48, 128, device="cuda", dtype=torch.bfloat16)
    k.qwen36_gated_deltanet_recurrent_bf16_out(q, kk, v, g, beta, state, out, rows, 48, 128)
    ref = gated_deltanet_reference(
        conv_out[:, :2048].view(rows, 16, 128),
        conv_out[:, 2048:4096].view(rows, 16, 128),
        v_ref,
        a,
        b,
        A_log,
        dt_bias,
    )
    check("gated_deltanet", out, ref, atol=0.0625)

    z = torch.randn(rows * 48, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    norm_out = torch.empty_like(z)
    k.qwen36_rms_norm_gated_silu_bf16_out(out.reshape(rows * 48, 128), z, weight, norm_out, rows * 48, 128, 1e-6)
    norm_ref = gated_rms_norm_silu(out.reshape(rows * 48, 128), z, weight, 1e-6)
    check("rms_norm_gated_silu", norm_out, norm_ref, atol=0.03125)


if __name__ == "__main__":
    main()
