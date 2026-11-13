"""Caller-buffer BF16 RMSNorm, SwiGLU, residual and Qwen-style KV writes."""
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, R, O, WIDTH: tl.constexpr, EPS: tl.constexpr,
          RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    value = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    if RESIDUAL:
        value += tl.load(R + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        # Preserve the FP32 sum for normalization; only the stored residual
        # rounds to BF16, matching the official Qwen3 fused residual norm.
        tl.store(X + row * WIDTH + col, value, col < WIDTH)
    inv = tl.rsqrt(tl.sum(value * value, 0) / WIDTH + EPS)
    weight = tl.load(W + col, col < WIDTH, 0).to(tl.float32)
    tl.store(O + row * WIDTH + col, value * inv * weight, col < WIDTH)


# Reuse compiled tiles when a new prompt changes only the masked extent.
@triton.jit(do_not_specialize=["SIZE"])
def _silu(GU, O, I: tl.constexpr, SIZE, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // I, index % I
    gate = tl.load(GU + row * (2 * I) + col, index < SIZE, 0).to(tl.float32)
    up = tl.load(GU + row * (2 * I) + I + col, index < SIZE, 0).to(tl.float32)
    activated = gate / (1.0 + tl.exp(-gate))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(O + index, activated * up, index < SIZE)


@triton.jit
def _qk_rope_cache(QKV, QW, KW, COS, SIN, POS, Q, KC, VC,
                   NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
                   EPS: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, D)
    base = tl.load(POS)
    position = base + row
    offset = row * ((NQ + 2 * NK) * D) + head * D
    value = tl.load(QKV + offset + dim).to(tl.float32)
    partner = tl.load(QKV + offset + (dim + D // 2) % D).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 0) / D + EPS)
    if head < NQ:
        weight = tl.load(QW + dim).to(tl.float32)
        other = tl.load(QW + (dim + D // 2) % D).to(tl.float32)
    else:
        weight = tl.load(KW + dim).to(tl.float32)
        other = tl.load(KW + (dim + D // 2) % D).to(tl.float32)
    # Qwen3's Q/K norms return BF16 before RoPE. Preserve that boundary
    # while fusing the cache write; retaining FP32 here changes attention.
    value = (value * inv * weight).to(tl.bfloat16).to(tl.float32)
    partner = (partner * inv * other).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(COS + position * (D // 2) + dim % (D // 2)).to(tl.float32)
    sin = tl.load(SIN + position * (D // 2) + dim % (D // 2)).to(tl.float32)
    rotated = value * cos + tl.where(dim < D // 2, -partner, partner) * sin
    if head < NQ:
        tl.store(Q + (row * NQ + head) * D + dim, rotated)
    else:
        kh = head - NQ
        tl.store(KC + (position * NK + kh) * D + dim, rotated)
        val = tl.load(QKV + row * ((NQ + 2 * NK) * D) + (NQ + NK + kh) * D + dim)
        tl.store(VC + (position * NK + kh) * D + dim, val)


def rms_norm(x, weight, out, *, eps, residual=None):
    _norm[(x.shape[0],)](x, weight, residual, out, x.shape[1], eps,
                         residual is not None, triton.next_power_of_2(x.shape[1]),
                         num_warps=4, enable_fp_fusion=False)


def silu_mul(gate_up, out):
    """Preserve eager BF16 SiLU rounding before multiplication."""
    _silu[(triton.cdiv(out.numel(), 256),)](
        gate_up, out, out.shape[1], out.numel(), 256, num_warps=4)


def qk_norm_rope_cache(qkv, qw, kw, cos, sin, position, q, k_cache, v_cache, *, eps):
    nq, d = q.shape[1:]
    nk = k_cache.shape[1]
    _qk_rope_cache[(q.shape[0], nq + nk)](
        qkv, qw, kw, cos, sin, position, q, k_cache, v_cache, nq, nk, d, eps,
        num_warps=4, enable_fp_fusion=False)
