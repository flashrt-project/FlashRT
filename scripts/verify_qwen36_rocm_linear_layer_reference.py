from __future__ import annotations
import os

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(torch.bfloat16)


def causal_conv1d_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv for Qwen3.6 linear attention.

    Args:
        x: (seq, 10240) BF16.
        weight: (10240, 4) BF16.

    Returns:
        (seq, 10240) BF16 after causal conv and SiLU activation.
    """

    seq, dim = x.shape
    k = weight.shape[1]
    padded = F.pad(x.float().transpose(0, 1), (k - 1, 0))
    windows = padded.unfold(dimension=1, size=k, step=1)[:, :seq, :]
    y = (windows * weight.float()[:, None, :]).sum(dim=-1).transpose(0, 1)
    return F.silu(y).to(torch.bfloat16)


def gated_deltanet_reference(
    q16: torch.Tensor,
    k16: torch.Tensor,
    v48: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Readable recurrent Gated-DeltaNet reference for shape checks.

    This follows the delta-rule structure used by the RTX path:
    L2-normalize q/k, compute beta and decay gate, then update a per-head
    (128, 128) recurrent state sequentially. It is intentionally a PyTorch
    reference, not the production kernel.
    """

    seq = q16.shape[0]
    q = F.normalize(q16.float(), p=2, dim=-1).repeat_interleave(3, dim=1)
    k = F.normalize(k16.float(), p=2, dim=-1).repeat_interleave(3, dim=1)
    v = v48.float()
    beta = torch.sigmoid(b.float())
    g = -torch.exp(A_log.float())[None, :] * F.softplus(a.float() + dt_bias.float()[None, :])
    decay = torch.exp(torch.clamp(g, min=-30.0, max=0.0))

    state = torch.zeros(48, 128, 128, device=q.device, dtype=torch.float32)
    outs = []
    for i in range(seq):
        pred = torch.einsum("hd,hde->he", k[i], state)
        delta = (v[i] - pred) * beta[i, :, None]
        state = state * decay[i, :, None, None] + torch.einsum("hd,he->hde", k[i], delta)
        out = torch.einsum("hd,hde->he", q[i], state)
        outs.append(out)
    return torch.stack(outs, dim=0).to(torch.bfloat16)


def gated_rms_norm_silu(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = rms_norm(x, weight, eps).float()
    return (y * F.silu(z.float())).to(torch.bfloat16)


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
    assert layer["type"] == "linear_attention"
    eps = float(handles.ptrs["rms_norm_eps"])

    def t(name: str) -> torch.Tensor:
        return by_ptr[int(layer[name])]

    def linear(x: torch.Tensor, name: str) -> torch.Tensor:
        return kernels.hipblaslt_linear_bf16(x.contiguous(), t(name + "_w"))

    seq = 4
    hidden = 5120
    h = torch.randn(seq, hidden, device="cuda", dtype=torch.bfloat16)

    x = rms_norm(h, t("input_norm_eff_w"), eps)
    qkv = linear(x, "in_proj_qkv")
    z = linear(x, "in_proj_z").view(seq, 48, 128)
    a = kernels.hipblaslt_linear_bf16(x.contiguous(), t("in_proj_a_w"))
    b = kernels.hipblaslt_linear_bf16(x.contiguous(), t("in_proj_b_w"))
    qkv_conv = causal_conv1d_reference(qkv, t("conv1d_w"))
    q16 = qkv_conv[:, :2048].view(seq, 16, 128)
    k16 = qkv_conv[:, 2048:4096].view(seq, 16, 128)
    v48 = qkv_conv[:, 4096:].view(seq, 48, 128)
    attn = gated_deltanet_reference(q16, k16, v48, a, b, t("A_log"), t("dt_bias"))
    normed = gated_rms_norm_silu(attn.reshape(seq * 48, 128), z.reshape(seq * 48, 128), t("head_norm_w"), eps)
    attn_proj = linear(normed.view(seq, 6144), "out_proj")
    h_post = (h + attn_proj).to(torch.bfloat16)

    x_mlp = rms_norm(h_post, t("post_attn_norm_eff_w"), eps)
    gate = linear(x_mlp, "mlp_gate")
    up = linear(x_mlp, "mlp_up")
    act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    mlp_out = linear(act, "mlp_down")
    out = (h_post + mlp_out).to(torch.bfloat16)
    torch.cuda.synchronize()

    print("linear_layer_out", tuple(out.shape), out.dtype)
    print("finite", bool(torch.isfinite(out.float()).all().item()))
    print("mean_abs", float(out.float().abs().mean().item()))
    print("max_abs", float(out.float().abs().max().item()))
    print("attn_mean_abs", float(attn.float().abs().mean().item()))


if __name__ == "__main__":
    main()
