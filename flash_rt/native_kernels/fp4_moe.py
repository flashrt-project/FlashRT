"""Fused glue for the prefill-size MoE on vLLM's CUTLASS NVFP4 grouped GEMM (sm120).

Three Triton kernels replace shuffle + quant, silu*up + quant, and un-shuffle + weighted combine:
  gather_quant_nvfp4   x [m, K] bf16 + dst->src row map  ->  fp4 codes [M, K/2], swizzled E4M3 block scales
  silu_mul_quant_nvfp4 c1 [M, 2n] (group-interleaved gate8|up8) -> fp4 codes [M, n/2], swizzled scales
  combine_topk         c3 [M, K] sorted rows + dst->src map + top-k weights + shared -> out [m, K]
Scale layout and rounding follow vLLM's scaled_fp4_experts_quant (per-expert regions padded to 128 rows,
128x4 block interleave, E4M3 scale = global * absmax / 6, E2M1 round-to-nearest-even via cvt.rn.satfinite).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1x2(hi, lo):
    # packs two fp32 values into one byte of E2M1 codes: lo -> low nibble, hi -> high nibble (returned as int16)
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; cvt.u16.u8 $0, t; }",
        "=h,r,r",
        [hi, lo],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quant_rows(x, gscale, KB: tl.constexpr):
    """x [R, KB*16] fp32 -> (packed codes [R, KB*8] uint8, scales [R, KB] uint8 (E4M3 bits))."""
    R: tl.constexpr = x.shape[0]
    x3 = tl.reshape(x, (R, KB, 16))
    amax = tl.max(tl.abs(x3), axis=2)  # [R, KB]
    sf = (gscale * (amax / 6.0)).to(tl.float8e4nv)
    sff = sf.to(tl.float32)
    inv = tl.where(sff > 0, gscale / sff, 0.0)
    q = x3 * inv[:, :, None]  # [R, KB, 16]
    q2 = tl.reshape(q, (R, KB * 8, 2))
    lo, hi = tl.split(q2)
    packed = _e2m1x2(hi, lo).to(tl.uint8)
    return packed, sf.to(tl.uint8, bitcast=True)


@triton.jit
def _store_swizzled(sf_ptr, sf, row_local, sf_base, m, KB: tl.constexpr):
    # scale element (r, c) of an expert region at sf_base rows: flat = base*KB + ((((m1*(KB/4) + k1)*32 + m3)*4 + m2)*4 + k2)
    r = row_local
    m1 = r // 128
    m2 = (r // 32) % 4
    m3 = r % 32
    c = tl.arange(0, KB)
    k1 = c // 4
    k2 = c % 4
    off = (
        (sf_base * KB)[:, None]
        + (
            ((m1[:, None] * (KB // 4) + k1[None, :]) * 32 + m3[:, None]) * 4
            + m2[:, None]
        )
        * 4
        + k2[None, :]
    )
    tl.store(sf_ptr + off, sf, mask=m[:, None])


@triton.jit
def _gather_quant_kernel(
    x_ptr,
    map_ptr,
    row_expert_ptr,
    eoff_ptr,
    bsoff_ptr,
    g_ptr,
    out_ptr,
    sf_ptr,
    n_rows,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    KB: tl.constexpr = K // 16
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = rows < n_rows
    src = tl.load(map_ptr + rows, mask=m, other=0).to(tl.int64)
    e = tl.load(row_expert_ptr + rows, mask=m, other=0)
    r_local = rows - tl.load(eoff_ptr + e, mask=m, other=0)
    sf_base = tl.load(bsoff_ptr + e, mask=m, other=0)
    g = tl.load(g_ptr)
    cols = tl.arange(0, K)
    x = tl.load(
        x_ptr + src[:, None] * K + cols[None, :], mask=m[:, None], other=0.0
    ).to(tl.float32)
    packed, sf = _quant_rows(x, g, KB)
    tl.store(
        out_ptr + rows.to(tl.int64)[:, None] * (K // 2) + tl.arange(0, K // 2)[None, :],
        packed,
        mask=m[:, None],
    )
    _store_swizzled(sf_ptr, sf, r_local, sf_base, m, KB)


@triton.jit
def _silu_quant_kernel(
    c_ptr,
    row_expert_ptr,
    eoff_ptr,
    bsoff_ptr,
    g_ptr,
    out_ptr,
    sf_ptr,
    n_rows,
    N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # c rows: [N/8 groups][gate8 | up8] bf16 -> act [N] -> fp4
    KB: tl.constexpr = N // 16
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = rows < n_rows
    e = tl.load(row_expert_ptr + rows, mask=m, other=0)
    r_local = rows - tl.load(eoff_ptr + e, mask=m, other=0)
    sf_base = tl.load(bsoff_ptr + e, mask=m, other=0)
    g = tl.load(g_ptr)
    j = tl.arange(0, N)
    gi = (j // 8) * 16 + (j % 8)
    base = c_ptr + rows.to(tl.int64)[:, None] * (2 * N)
    gate = tl.load(base + gi[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    up = tl.load(base + (gi + 8)[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate) * up
    act = act.to(tl.bfloat16).to(
        tl.float32
    )  # the bf16 activation the reference path quantizes
    packed, sf = _quant_rows(act, g, KB)
    tl.store(
        out_ptr + rows.to(tl.int64)[:, None] * (N // 2) + tl.arange(0, N // 2)[None, :],
        packed,
        mask=m[:, None],
    )
    _store_swizzled(sf_ptr, sf, r_local, sf_base, m, KB)


@triton.jit
def _act_amax_kernel(c_ptr, out_ptr, n_rows, N: tl.constexpr, BLOCK_R: tl.constexpr):
    # max |silu(gate) * up| over the group-interleaved rows (bf16-rounded like the quantized activation)
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = rows < n_rows
    j = tl.arange(0, N)
    gi = (j // 8) * 16 + (j % 8)
    base = c_ptr + rows.to(tl.int64)[:, None] * (2 * N)
    gate = tl.load(base + gi[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    up = tl.load(base + (gi + 8)[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    act = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    tl.atomic_max(out_ptr, tl.max(tl.abs(act)))


def act_amax(c1: torch.Tensor) -> torch.Tensor:
    M = c1.shape[0]
    N = c1.shape[1] // 2
    out = torch.zeros((1,), dtype=torch.float32, device=c1.device)
    _act_amax_kernel[(triton.cdiv(M, 16),)](c1, out, M, N=N, BLOCK_R=16, num_warps=4)
    return out


@triton.jit
def _combine_kernel(
    c_ptr,
    map_ptr,
    tw_ptr,
    shared_ptr,
    out_ptr,
    n_tok,
    TOPK: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    m = t < n_tok
    cols = tl.arange(0, K)
    acc = tl.load(
        shared_ptr + t.to(tl.int64)[:, None] * K + cols[None, :],
        mask=m[:, None],
        other=0.0,
    ).to(tl.float32)
    for k in tl.static_range(TOPK):
        row = tl.load(map_ptr + t * TOPK + k, mask=m, other=0).to(tl.int64)
        w = tl.load(tw_ptr + t * TOPK + k, mask=m, other=0.0).to(tl.float32)
        v = tl.load(
            c_ptr + row[:, None] * K + cols[None, :], mask=m[:, None], other=0.0
        ).to(tl.float32)
        acc += w[:, None] * v
    tl.store(
        out_ptr + t.to(tl.int64)[:, None] * K + cols[None, :],
        acc.to(tl.bfloat16),
        mask=m[:, None],
    )


def row_experts(
    expert_offsets: torch.Tensor, n_rows: int, num_experts: int
) -> torch.Tensor:
    counts = expert_offsets[1:] - expert_offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(num_experts, device=expert_offsets.device, dtype=torch.int32),
        counts.to(torch.int64),
        output_size=n_rows,
    )


def gather_quant_nvfp4(
    x, a_map, row_expert, expert_offsets, bs_offsets, gscale, sf_rows_total
):
    M = a_map.shape[0]
    K = x.shape[1]
    out = torch.empty((M, K // 2), dtype=torch.uint8, device=x.device)
    sf = torch.zeros(
        (sf_rows_total * (K // 16),), dtype=torch.uint8, device=x.device
    )  # padding rows read as 0
    _gather_quant_kernel[(triton.cdiv(M, 4),)](
        x,
        a_map,
        row_expert,
        expert_offsets,
        bs_offsets,
        gscale,
        out,
        sf,
        M,
        K=K,
        BLOCK_R=4,
        num_warps=4,
    )
    return out, sf.view(torch.float8_e4m3fn).view(sf_rows_total, K // 16)


def silu_mul_quant_nvfp4(
    c1, row_expert, expert_offsets, bs_offsets, gscale, sf_rows_total
):
    M = c1.shape[0]
    N = c1.shape[1] // 2
    out = torch.empty((M, N // 2), dtype=torch.uint8, device=c1.device)
    sf = torch.zeros((sf_rows_total * (N // 16),), dtype=torch.uint8, device=c1.device)
    _silu_quant_kernel[(triton.cdiv(M, 8),)](
        c1,
        row_expert,
        expert_offsets,
        bs_offsets,
        gscale,
        out,
        sf,
        M,
        N=N,
        BLOCK_R=8,
        num_warps=4,
    )
    return out, sf.view(torch.float8_e4m3fn).view(sf_rows_total, N // 16)


def combine_topk(c3, c_map, tw, shared, out):
    m, topk = tw.shape
    K = c3.shape[1]
    _combine_kernel[(triton.cdiv(m, 4),)](
        c3, c_map, tw, shared, out, m, TOPK=topk, K=K, BLOCK_T=4, num_warps=4
    )
    return out
