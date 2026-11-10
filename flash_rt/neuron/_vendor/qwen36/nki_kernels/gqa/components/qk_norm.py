# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-RoPE q/k RMSNorm for the GQA attention block (head_dim=256, token generation).

Sits between the QKV projection and RoPE: norms each Q head and each K head over head_dim, and
passes the V heads through unchanged.

Layout, which is what makes this a free-axis reduce:
    x_sb [T, N, D] -- T = B*S tokens on the partition axis; N heads head-major on the free axis,
    then head_dim D. Because D sits entirely on the free axis and 256 <= the 512 free limit, RMSNorm
    over D is a single free-axis reduction per token. A head_dim-on-partition norm would cap d at 128.

Math per normed head, with gamma the standard layernorm weight (not 1+weight):
    y[t, :] = x[t, :] * rsqrt(mean_D(x[t, :]^2) + eps) * gamma

bf16 or fp32 IO; the square, reduction, rsqrt and scale all run in fp32.

SBUF-in / SBUF-out: the projection's [B*S, I] result is the same buffer viewed as [T, N, D], so a
caller passes it here with no copy.
"""

import nki.isa as nisa
import nki.language as nl

# Qwen3.6 GQA per-rank (TP=4) head config: head_dim 256, 4 Q heads, 1 K head; N = Q + 2*K heads.
HEAD_DIM = 256
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
NUM_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 6 -> [q0|q1|q2|q3|k0|v0] on the N axis


def kernel_assert(condition, error_text):
    """Assert with an NKI-formatted error message (identifies kernel-origin failures)."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def rms_norm_over_free(x_head, gamma_head, eps_t, out_head):
    """Free-axis RMSNorm of one head tile [T, D] over head_dim, fp32 reduce with an IO-dtype store.

    Args:
        x_head:     [T, D] SBUF, one head's tokens.
        gamma_head: [T, D] SBUF per-head_dim weight, partition-broadcast to the T token rows.
        eps_t:      [T, 1] SBUF fp32 epsilon, memset once and shared across heads.
        out_head:   [T, D] SBUF, written.
    """
    T, D = x_head.shape

    # Sum of squares over the free axis, accumulated in fp32.
    sq = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=x_head)

    ss = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=[1], keepdims=True)

    # inv = rsqrt(ss * (1/D) + eps) = 1 / sqrt(mean_D(x^2) + eps).
    inv = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, scale=1.0 / D, bias=eps_t)

    # y = (x * inv) * gamma in one Vector-engine pass: inv is the per-token scalar [T, 1]
    # (free-broadcast), gamma is the full [T, D] weight. fp32 math, cast to out_head.dtype.
    nisa.scalar_tensor_tensor(
        dst=out_head,
        data=x_head,
        op0=nl.multiply,
        operand0=inv,
        op1=nl.multiply,
        operand1=gamma_head,
    )


def qk_norm_compose(
    qkv_sb,
    gamma_q_sb,
    gamma_k_sb,
    num_q_heads=NUM_Q_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    eps=1e-6,
    out_sb=None,
):
    """Pre-RoPE q/k RMSNorm over a head-major QKV tile, a free-axis reduce over head_dim.

    Norms the Q heads with gamma_q and the K heads with gamma_k; the V heads are copied through.

    Args:
        qkv_sb:     [T, N, D] SBUF head-major QKV, N = num_q_heads + 2*num_kv_heads.
        gamma_q_sb: [T, D] SBUF Q-layernorm weight, partition-broadcast to the T token rows.
        gamma_k_sb: [T, D] SBUF K-layernorm weight, same broadcast.
        num_q_heads, num_kv_heads, head_dim: head config.
        eps:        RMSNorm epsilon.
        out_sb:     optional output; allocated if None.
    """
    T, N, D = qkv_sb.shape
    kernel_assert(D == head_dim, "qkv_sb last dim must equal head_dim")
    kernel_assert(
        N == num_q_heads + 2 * num_kv_heads,
        "N must equal num_q_heads + 2*num_kv_heads (head-major [q.. | k.. | v..])",
    )
    kernel_assert(gamma_q_sb.shape == (T, D), "gamma_q_sb must be [T, head_dim]")
    kernel_assert(gamma_k_sb.shape == (T, D), "gamma_k_sb must be [T, head_dim]")

    if out_sb is None:
        out_sb = nl.ndarray((T, N, D), dtype=qkv_sb.dtype, buffer=nl.sbuf)

    eps_t = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=float(eps))

    k_end = num_q_heads + num_kv_heads
    for n in range(num_q_heads):
        rms_norm_over_free(
            qkv_sb[0:T, n, 0:D], gamma_q_sb[0:T, 0:D], eps_t, out_sb[0:T, n, 0:D]
        )
    for n in range(num_q_heads, k_end):
        rms_norm_over_free(
            qkv_sb[0:T, n, 0:D], gamma_k_sb[0:T, 0:D], eps_t, out_sb[0:T, n, 0:D]
        )
    for n in range(k_end, N):
        nisa.tensor_copy(dst=out_sb[0:T, n, 0:D], src=qkv_sb[0:T, n, 0:D])

    return out_sb
