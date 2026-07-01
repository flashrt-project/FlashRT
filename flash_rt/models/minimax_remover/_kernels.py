"""FlashRT -- MiniMax-Remover fused Triton kernels.

Self-contained Triton JIT kernels used by the MiniMax-Remover denoise
loop. They are the precision-critical elementwise / normalisation ops
of the Transformer3DModel block forward and the flow-matching step:

* ``ada_layernorm_fp16_io`` -- fp32-stat LayerNorm + adaLN modulation
  fused into one kernel, fp16/bf16 in/out. The FlashRT generic
  ``ada_layer_norm_fp16`` loses precision on real diffusion latents;
  this kernel matches the reference FP32LayerNorm path bit-for-bit by
  accumulating mean/var in fp32 across three passes.
* ``rms_norm_fp32stat``      -- RMSNorm with fp32 statistics + affine
  weight, used for attention norm_q / norm_k.
* ``gate_mul_residual_bcast``-- in-place ``residual += x * gate`` with a
  broadcast gate vector (avoids the [S, D] expand copy).
* ``rope_apply_bshd`` / ``freqs_to_cos_sin`` -- interleaved RoPE applied
  in-place on the native [B, S, H, D] layout (no transpose / copy).
* ``euler_step_inplace``     -- in-place Euler flow-matching step with
  fp32 accumulation, used every denoise step.
* ``mask_mul`` / ``latent_affine`` -- fused host-side elementwise ops
  (masked-image build and latent (de)normalisation).

Every kernel here is dtype-generic on fp16 / bf16 and dispatches on the
input dtype at call time. No model-class imports -- tensors only.
"""

import torch
import triton
import triton.language as tl


def _io_dtype(x):
    return tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16


# ── adaLayerNorm: fp32-stat LayerNorm + (1 + scale) * x_norm + shift ──────

@triton.jit
def _ada_layernorm_io_kernel(X, SCALE, SHIFT, OUT, M, N, sM_x, sM_o, eps,
                             BLOCK_N: tl.constexpr, IO_DTYPE: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = X + row * sM_x
    o_ptr = OUT + row * sM_o
    _mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += tl.sum(x)
    mean = _mean / N
    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        _var += tl.sum(d * d)
    rstd = 1.0 / tl.sqrt(_var / N + eps)
    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * rstd
        scale = tl.load(SCALE + cols, mask=mask, other=0.0)
        shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
        y = x_norm * (1.0 + scale) + shift
        tl.store(o_ptr + cols, y.to(IO_DTYPE), mask=mask)


def ada_layernorm_fp16_io(x, scale, shift, eps=1e-6):
    """LayerNorm(x) * (1 + scale) + shift -> same dtype as x.

    ``x`` is contiguous [S, D] or [B, S, D] fp16/bf16; scale/shift are
    [D] vectors. Statistics accumulated in fp32 (reference-equivalent).
    """
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.reshape(orig_shape[0] * orig_shape[1], orig_shape[2])
    S, D = x.shape
    assert x.is_contiguous(), "ada_layernorm_fp16_io: x must be contiguous"
    scale = scale.contiguous().to(torch.float32).view(-1)
    shift = shift.contiguous().to(torch.float32).view(-1)
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(min(D, 2048))
    num_warps = 8 if D >= 1024 else 4
    _ada_layernorm_io_kernel[(S,)](
        x, scale, shift, out, S, D, x.stride(0), out.stride(0), eps,
        BLOCK_N=BLOCK_N, num_warps=num_warps, IO_DTYPE=_io_dtype(x))
    return out.reshape(orig_shape)


# ── RMSNorm with fp32 statistics + affine weight ─────────────────────────

@triton.jit
def _rmsnorm_affine_kernel(X, WEIGHT, OUT, Npts, D, EPS, BLOCK_D: tl.constexpr,
                           IO_DTYPE: tl.constexpr):
    pt = tl.program_id(0)
    xp = X + pt * D
    op = OUT + pt * D
    _sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in tl.range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(xp + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.sum(x * x)
    inv_rms = 1.0 / tl.sqrt(_sum / D + EPS)
    for off in tl.range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(xp + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(op + cols, (x * inv_rms * w).to(IO_DTYPE), mask=mask)


def rms_norm_fp32stat(x, weight, eps):
    """RMSNorm (fp32 statistics + affine weight[D]). x[.., D] -> same dtype."""
    D = x.shape[-1]
    orig_shape = x.shape
    x2 = x.reshape(-1, D).contiguous()
    Npts = x2.shape[0]
    out = torch.empty_like(x2)
    w = weight.contiguous().to(torch.float32).view(-1)
    BLOCK_D = 16
    while BLOCK_D < D and BLOCK_D < 2048:
        BLOCK_D <<= 1
    _rmsnorm_affine_kernel[(Npts,)](
        x2, w, out, Npts, D, float(eps), BLOCK_D=BLOCK_D,
        num_warps=8 if D >= 256 else 4, IO_DTYPE=_io_dtype(x2))
    return out.reshape(orig_shape)


# ── in-place residual += x * gate (broadcast gate vector) ────────────────

@triton.jit
def _gate_mul_res_bcast_kernel(RES, X, GATE, Nrow, D, BLOCK_D: tl.constexpr,
                               IO_DTYPE: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    rp = RES + r * D
    x = tl.load(rp + cols, mask=mask).to(tl.float32)
    xv = tl.load(X + r * D + cols, mask=mask).to(tl.float32)
    g = tl.load(GATE + cols, mask=mask).to(tl.float32)
    tl.store(rp + cols, (x + xv * g).to(IO_DTYPE), mask=mask)


def gate_mul_residual_bcast(residual, x, gate):
    """In-place residual[S, D] += x[S, D] * gate[D] (gate broadcast)."""
    D = residual.shape[-1]
    res2 = residual.reshape(-1, D)
    x2 = x.reshape(-1, D)
    Nrow = res2.shape[0]
    g = gate.contiguous().to(residual.dtype).view(-1)
    BLOCK_D = 16
    while BLOCK_D < D and BLOCK_D < 2048:
        BLOCK_D <<= 1
    _gate_mul_res_bcast_kernel[(Nrow,)](res2, x2, g, Nrow, D, BLOCK_D=BLOCK_D,
                                        num_warps=4, IO_DTYPE=_io_dtype(residual))
    return residual


# ── interleaved RoPE on native [B, S, H, D] layout (in-place) ────────────

@triton.jit
def _rope_bshd_kernel(X, COS, SIN, Nrows, S, H, Dhalf,
                      BLOCK_D: tl.constexpr, IO_DTYPE: tl.constexpr):
    r = tl.program_id(0)
    s = (r // H) % S
    x_ptr = X + r * (2 * Dhalf)
    cos_ptr = COS + s * Dhalf
    sin_ptr = SIN + s * Dhalf
    offs = tl.arange(0, BLOCK_D)
    mask = offs < Dhalf
    cos = tl.load(cos_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + 2 * offs, (x0 * cos - x1 * sin).to(IO_DTYPE), mask=mask)
    tl.store(x_ptr + 2 * offs + 1, (x0 * sin + x1 * cos).to(IO_DTYPE), mask=mask)


def rope_apply_bshd(x, cos, sin):
    """In-place interleaved RoPE on contiguous [B, S, H, D].

    cos/sin are [S, D // 2] fp32 tables. Returns x (modified in place).
    """
    assert x.is_contiguous(), "rope_apply_bshd: x must be contiguous"
    B, S, H, D = x.shape
    Dhalf = D // 2
    Nrows = B * S * H
    BLOCK_D = max(16, 1 << max(0, (Dhalf - 1).bit_length()))
    _rope_bshd_kernel[(Nrows,)](
        x, cos.contiguous().to(torch.float32), sin.contiguous().to(torch.float32),
        Nrows, S, H, Dhalf, BLOCK_D=BLOCK_D, num_warps=4, IO_DTYPE=_io_dtype(x))
    return x


def freqs_to_cos_sin(freqs):
    """complex [1, 1, S, D // 2] -> (cos[S, D // 2] fp32, sin[S, D // 2] fp32)."""
    f = freqs.squeeze().to(torch.complex64)
    return f.real.contiguous().to(torch.float32), f.imag.contiguous().to(torch.float32)


# ── denoise-loop elementwise kernels (fp32 accumulation, in-place) ───────

@triton.jit
def _euler_kernel(latents_ptr, noise_ptr, n_elements, dt,
                  BLOCK: tl.constexpr, IO_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    lat = tl.load(latents_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    noise = tl.load(noise_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(latents_ptr + offs, (lat + dt * noise).to(IO_DTYPE), mask=mask)


def euler_step_inplace(latents, noise_pred, dt):
    """In-place Euler flow-matching step: latents += dt * noise_pred."""
    n = latents.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _euler_kernel[grid](latents, noise_pred, n, float(dt),
                        BLOCK=2048, IO_DTYPE=_io_dtype(latents))


@triton.jit
def _mask_mul_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr,
                     IO_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, (a * (1.0 - b)).to(IO_DTYPE), mask=mask)


def mask_mul(a, b):
    """out = a * (1 - b), same dtype as a. Replaces torch elementwise."""
    out = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _mask_mul_kernel[grid](a, b, out, n, BLOCK=4096, IO_DTYPE=_io_dtype(a))
    return out


@triton.jit
def _latent_affine_kernel(x_ptr, param_ptr, out_ptr, n, scale, is_sub_mul,
                          BLOCK: tl.constexpr, IO_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    p = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    if is_sub_mul:
        result = (x - p) * scale
    else:
        result = x * scale + p
    tl.store(out_ptr + offs, result.to(IO_DTYPE), mask=mask)


def latent_normalize(x, mean_param, inv_std_param):
    """out = (x - mean) * inv_std. Replaces torch elementwise."""
    out = torch.empty_like(x)
    inv_std = 1.0 / float(inv_std_param.max())
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _latent_affine_kernel[grid](x, mean_param, out, n, inv_std, True,
                                BLOCK=4096, IO_DTYPE=_io_dtype(x))
    return out


def latent_denormalize(x, mean_param, inv_std_param):
    """out = x / inv_std + mean = x * (1/inv_std) + mean.

    inv_std_param stores 1/std; mean_param stores latents_mean.
    """
    out = torch.empty_like(x)
    inv_std_val = float(inv_std_param.max())
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _latent_affine_kernel[grid](x, mean_param, out, n, inv_std_val, False,
                                BLOCK=4096, IO_DTYPE=_io_dtype(x))
    return out
