"""LingBot-VLA normalization helpers.

Pure-PyTorch reference implementations of:
    * ``Qwen2RMSNorm``    used by the VLM tower (input/post-attn LN)
    * ``AdaRMSNorm``      used by the Action Expert (input/post-attn LN
                           with FiLM scale+shift conditioned on the
                           timestep embedding)

Math copied from ``lingbotvla/models/vla/pi0/modeling_lingbot_vla.py``
(Qwen2RMSNorm @ L227, AdaRMSNorm @ L1043). Both compute the variance and
normalization in fp32 and cast the final result back to the input dtype
— matching this is required for bit-exact cosine vs upstream.

These functions are stateless (take raw weight tensors instead of an
nn.Module); they exist to make + forward paths composable from the
device-bound weight target without needing to instantiate upstream modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_rt.models.lingbot.kernel_ops import linear_bf16, linear_fp8


def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional Qwen2RMSNorm: ``weight * (x / sqrt(mean(x^2) + eps))``.

    Math is identical to upstream ``Qwen2RMSNorm.forward``:
        1. Cast to fp32.
        2. variance = mean(x^2, dim=-1, keepdim=True).
        3. x = x * rsqrt(variance + eps).
        4. y = weight * x.to(input_dtype).

    Args:
        hidden_states: ``[..., hidden]``.
        weight:        ``[hidden]`` RMS gain (no bias).
        eps:           variance epsilon (default 1e-6 matches Qwen2.5-VL).

    Returns:
        Same shape/dtype as ``hidden_states``.
    """
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def ada_rms_norm(
    hidden_states: torch.Tensor,
    cond: torch.Tensor,
    *,
    weight: torch.Tensor,
    gamma_weight: torch.Tensor,
    gamma_bias: torch.Tensor,
    beta_weight: torch.Tensor,
    beta_bias: torch.Tensor,
    eps: float = 1e-6,
    site_prefix: "str | None" = None,
) -> torch.Tensor:
    """Functional AdaRMSNorm: RMSNorm + FiLM ``(1 + γ(cond)) * x + β(cond)``.

    Math is identical to upstream ``AdaRMSNorm.forward``:
        x_norm = weight * (x / sqrt(mean(x^2) + eps))    in fp32
        γ      = Linear(gamma_w, gamma_b)(cond)          [B, hidden]
        β      = Linear(beta_w,  beta_b )(cond)          [B, hidden]
        y      = (1 + γ.unsqueeze(1)) * x_norm + β.unsqueeze(1)
        return y.to(input_dtype)

    Note the FiLM linears use ``(weight @ cond.T + bias).T`` semantics;
    we call ``F.linear`` which is the same matmul. The result is
    broadcast over the token axis via ``unsqueeze(1)``.

    Args:
        hidden_states: ``[B, L, hidden]``.
        cond:          ``[B, cond_dim]`` (timestep embedding).
        weight:        ``[hidden]`` RMS gain.
        gamma_weight:  ``[hidden, cond_dim]``.
        gamma_bias:    ``[hidden]``.
        beta_weight:   ``[hidden, cond_dim]``.
        beta_bias:     ``[hidden]``.
        eps:           variance epsilon.

    Returns:
        ``[B, L, hidden]`` same dtype as ``hidden_states``.
    """
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x                                   # still fp32

    if site_prefix is None:
        sid_g = sid_b = None
    else:
        sid_g = f"{site_prefix}.gamma"
        sid_b = f"{site_prefix}.beta"
    gamma = linear_fp8(cond, gamma_weight, gamma_bias,
                       site_id=sid_g).unsqueeze(1)  # [B, 1, hidden]
    beta = linear_fp8(cond, beta_weight, beta_bias,
                      site_id=sid_b).unsqueeze(1)
    out = (1 + gamma.to(torch.float32)) * x + beta.to(torch.float32)
    return out.to(input_dtype)


__all__ = ["rms_norm", "ada_rms_norm"]
