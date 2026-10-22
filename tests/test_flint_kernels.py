#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the Flint (SM120) W4A16 kernels.

Each kernel is compared against a straight PyTorch transcription of the
official numerics contract (exact NVFP4 dequantization, BF16 activations,
fp32 accumulation, one rounding per output; see docs/flint_usage.md) on
synthetic data. Exact comparisons are used where the contract is exact
(repack, routing ids); accumulation-order differences are bounded by one
BF16 rounding otherwise.

Run:  python -m pytest tests/test_flint_kernels.py

Needs a CUDA device and ``flash_rt_flint`` (built with FLASHRT_ENABLE_FLINT on
GPU_ARCH=120); skipped otherwise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Flint kernels need a CUDA device", allow_module_level=True)
fl = pytest.importorskip("flash_rt.flash_rt_flint")

from flash_rt.models.flint.weights import (  # noqa: E402
    E2M1_VALUES, dequantize, split_nvfp4_blocks, ue4m3_table)

DEV = "cuda"
EPS = 1e-6
STREAM = 0


def _stream():
    return torch.cuda.current_stream().cuda_stream


def bf16(x):
    return x.to(torch.bfloat16)


def rand_blocks(rows, K, seed, lo=0x28, hi=0x44):
    """Random ggml block_nvfp4 superblocks with realistic scale bytes."""
    rng = np.random.default_rng(seed)
    blk = rng.integers(0, 256, size=(rows, K // 64, 36), dtype=np.uint8)
    blk[:, :, :4] = rng.integers(lo, hi, size=(rows, K // 64, 4), dtype=np.uint8)
    return blk


def to_dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def repack_dev(blocks, group_rows=None, group_stride=None, row_offset=0, out=None):
    rows, nb, _ = blocks.shape
    K = nb * 64
    src = to_dev(blocks)
    if out is None:
        codes = torch.empty(rows, K // 2, dtype=torch.uint8, device=DEV)
        scales = torch.empty(rows, K // 16, dtype=torch.uint8, device=DEV)
    else:
        codes, scales = out
    gr = group_rows or rows
    fl.repack_nvfp4(src.data_ptr(), rows, K, codes.data_ptr(), scales.data_ptr(), gr,
                    group_stride or gr, row_offset, _stream())
    return codes, scales


def deq_ref(codes, scales):
    """fp64 exact dequant (without the global scale) of device codes/scales."""
    return torch.from_numpy(dequantize(codes.cpu().numpy(), scales.cpu().numpy())).double()


def close_bf16(got, want, name, rel=2.0 ** -7, abs_frac=1e-3):
    got = got.double().cpu()
    want = want.double().cpu()
    assert bool(torch.isfinite(got).all()), f"{name}: non-finite output"
    assert bool(torch.isfinite(want).all()), f"{name}: non-finite reference"
    tol = rel * want.abs() + abs_frac * want.abs().max()
    bad = (got - want).abs() > tol
    assert not bool(bad.any()), f"{name}: {int(bad.sum())} elements out of tolerance, max|diff|={(got - want).abs().max():.3e}"


# ---------------------------------------------------------------------------
def test_repack_byte_identical_and_grouped():
    blk = rand_blocks(96, 512, 1, lo=0, hi=256)
    codes, scales = repack_dev(blk)
    rc, rs = split_nvfp4_blocks(blk)
    assert np.array_equal(codes.cpu().numpy(), rc)
    assert np.array_equal(scales.cpu().numpy(), rs)
    # grouped placement: two 48-row "experts" interleaved with a second tensor
    codes2 = torch.zeros(192, 256, dtype=torch.uint8, device=DEV)
    scales2 = torch.zeros(192, 32, dtype=torch.uint8, device=DEV)
    repack_dev(blk, 48, 96, 0, (codes2, scales2))
    got = codes2.view(2, 96, 256)[:, :48].reshape(96, 256)
    assert np.array_equal(got.cpu().numpy(), rc)
    assert int(codes2.view(2, 96, 256)[:, 48:].abs().sum()) == 0


def test_ggml_order_dequant_matches_reference_decode():
    """Repacked codes decode to the ggml block_nvfp4 reference values."""
    blk = rand_blocks(8, 128, 2, lo=0, hi=0x7F)
    codes, scales = repack_dev(blk)
    tab = ue4m3_table()
    want = np.zeros((8, 128), dtype=np.float32)
    for r in range(8):
        for b in range(2):
            for s in range(4):
                d = tab[blk[r, b, s]]
                for j in range(8):
                    q = blk[r, b, 4 + 8 * s + j]
                    want[r, 64 * b + 16 * s + j] = E2M1_VALUES[q & 15] * d
                    want[r, 64 * b + 16 * s + j + 8] = E2M1_VALUES[q >> 4] * d
    got = dequantize(codes.cpu().numpy(), scales.cpu().numpy())
    assert np.array_equal(got, want)


@pytest.mark.parametrize("N,K", [(5120, 2048), (2048, 4096), (2048, 6144), (512, 2048)])
def test_gemv_bf16(N, K):
    blk = rand_blocks(N, K, 10 + K)
    codes, scales = repack_dev(blk)
    x = bf16(torch.randn(K, device=DEV))
    gs = 0.00123
    out = torch.empty(N, dtype=torch.bfloat16, device=DEV)
    fl.gemv_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), out.data_ptr(), N, K, gs, 0, _stream())
    ref = (deq_ref(codes, scales) @ x.double().cpu()) * gs
    close_bf16(out, ref, f"gemv {N}x{K}")


def test_gemv_f32_lm_head_shape():
    N, K = 1000, 2048
    codes, scales = repack_dev(rand_blocks(N, K, 21))
    x = bf16(torch.randn(K, device=DEV))
    out = torch.empty(N, dtype=torch.float32, device=DEV)
    fl.gemv_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), out.data_ptr(), N, K, 0.5, 1, _stream())
    ref = (deq_ref(codes, scales) @ x.double().cpu()) * 0.5
    assert torch.allclose(out.double().cpu(), ref, rtol=1e-5, atol=1e-4 * ref.abs().max().item())


def test_gemv_swiglu_dense_rule():
    N, K = 2 * 384, 2048
    codes, scales = repack_dev(rand_blocks(N, K, 22))  # rows already in interleaved-by-8 order
    x = bf16(torch.randn(K, device=DEV))
    gs = 0.002
    out = torch.empty(N // 2, dtype=torch.bfloat16, device=DEV)
    fl.gemv_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), out.data_ptr(), N, K, gs, 2, _stream())
    y = ((deq_ref(codes, scales) @ x.double().cpu()) * gs).view(N // 16, 2, 8)
    g = bf16(y[:, 0].reshape(-1)).double()
    u = bf16(y[:, 1].reshape(-1)).double()
    ref = g / (1 + torch.exp(-g)) * u
    close_bf16(out, ref, "gemv swiglu")


def test_add_rmsnorm_reference_rounding():
    rows = 3
    x = bf16(torch.randn(rows, 2048, device=DEV) * 3)
    res = bf16(torch.randn(rows, 2048, device=DEV) * 10)
    w = torch.rand(2048, device=DEV) + 0.5
    res_k = res.clone()
    out = torch.empty_like(x)
    fl.add_rmsnorm_bf16(x.data_ptr(), res_k.data_ptr(), w.data_ptr(), out.data_ptr(), rows, EPS, 0, _stream())
    s = x.double() + res.double()
    assert torch.equal(res_k, bf16(s))
    ref = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + EPS) * w.double()
    same = (out == bf16(ref)).float().mean().item()
    assert same > 0.99, same
    close_bf16(out, ref, "add_rmsnorm", rel=2.0 ** -8, abs_frac=0)
    # first layer: residual := x
    res_f = torch.zeros_like(x)
    fl.add_rmsnorm_bf16(x.data_ptr(), res_f.data_ptr(), w.data_ptr(), out.data_ptr(), rows, EPS, 1, _stream())
    assert torch.equal(res_f, x)


def rope_tables(max_pos, theta=10000.0):
    inv = 1.0 / theta ** (np.arange(0, 64, dtype=np.float64) * 2 / 128)
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
    return (torch.from_numpy(np.cos(ang).astype(np.float32)).to(DEV),
            torch.from_numpy(np.sin(ang).astype(np.float32)).to(DEV))


@pytest.mark.parametrize("rope,ring", [(1, 1024), (0, 0)])
def test_qk_norm_rope_kv(rope, ring):
    pos_v = 1500
    qkv = bf16(torch.randn(5120, device=DEV) * 4)
    qw = torch.rand(128, device=DEV) + 0.5
    kw = torch.rand(128, device=DEV) + 0.5
    cos_t, sin_t = rope_tables(2048)
    pos = torch.tensor([pos_v], dtype=torch.int32, device=DEV)
    slots = ring if ring else 2048
    cache = torch.zeros(slots, 1024, dtype=torch.bfloat16, device=DEV)
    q_out = torch.empty(4096, dtype=torch.bfloat16, device=DEV)
    fl.qk_norm_rope_kv_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                            pos.data_ptr(), q_out.data_ptr(), cache.data_ptr(), ring, rope, EPS, _stream())

    def ref_head(v, w):
        v = v.float()
        n = bf16(v * torch.rsqrt((v * v).mean() + EPS) * w).float()
        if rope:
            c, s = cos_t[pos_v], sin_t[pos_v]
            n1, n2 = n[:64], n[64:]
            n = torch.cat([n1 * c - n2 * s, n2 * c + n1 * s])
        return bf16(n)

    q = qkv[:4096].view(32, 128)
    want_q = torch.stack([ref_head(q[h], qw) for h in range(32)]).view(-1)
    slot = pos_v % ring if ring else pos_v
    k = qkv[4096:4608].view(4, 128)
    v = qkv[4608:].view(4, 128)
    want_k = torch.stack([ref_head(k[h], kw) for h in range(4)]).view(-1)
    for name, got, want in (("q", q_out, want_q), ("k", cache[slot, :512], want_k)):
        same = (got == want).float().mean().item()
        assert same > 0.98, (name, same)
        close_bf16(got, want, name, rel=2.0 ** -7, abs_frac=0)
    assert torch.equal(cache[slot, 512:], v.reshape(-1))


def attn_ref(q, cache, pos, ring, window, sinks):
    """fp64 reference with sinks; q [32,128], cache [slots,1024] bf16."""
    kstart = max(0, pos - window + 1) if window else 0
    ps = torch.arange(kstart, pos + 1)
    slots = ps % ring if ring else ps
    kv = cache[slots.to(DEV)].double().cpu()
    K = kv[:, :512].view(-1, 4, 128)
    V = kv[:, 512:].view(-1, 4, 128)
    out = torch.empty(32, 128, dtype=torch.float64)
    for h in range(32):
        s = (K[:, h // 8] @ q[h].double().cpu()) / math.sqrt(128)
        m = max(s.max().item(), sinks[h].item())
        e = torch.exp(s - m)
        out[h] = (e[:, None] * V[:, h // 8]).sum(0) / (e.sum() + math.exp(sinks[h].item() - m))
    return out


@pytest.mark.parametrize("pos,ring,window,n_split", [(0, 0, 0, 4), (37, 0, 0, 4), (5000, 0, 0, 16),
                                                     (700, 1024, 513, 4), (3000, 1024, 513, 8)])
def test_decode_attention(pos, ring, window, n_split):
    torch.manual_seed(pos)
    slots = ring if ring else pos + 1
    cache = bf16(torch.randn(slots, 1024, device=DEV))
    q = bf16(torch.randn(32, 128, device=DEV))
    sinks = torch.randn(32, device=DEV) * 3
    p = torch.tensor([pos], dtype=torch.int32, device=DEV)
    o_part = torch.empty(n_split, 32, 128, device=DEV)
    ml = torch.empty(n_split, 32, 2, device=DEV)
    out = torch.empty(32, 128, dtype=torch.bfloat16, device=DEV)
    fl.decode_attention_bf16(q.data_ptr(), cache.data_ptr(), p.data_ptr(), ring, window, sinks.data_ptr(),
                             1 / math.sqrt(128), n_split, o_part.data_ptr(), ml.data_ptr(), out.data_ptr(),
                             _stream())
    ref = attn_ref(q, cache, pos, ring, window, sinks.cpu())
    err = ((out.double().cpu() - ref).abs().max() / ref.abs().max()).item()
    assert err < 8e-3, err


def test_router_topk():
    torch.manual_seed(3)
    xn = bf16(torch.randn(2048, device=DEV))
    wr = bf16(torch.randn(256, 2048, device=DEV) * 0.02)
    logits = torch.empty(256, device=DEV)
    counter = torch.zeros(1, dtype=torch.int32, device=DEV)
    ids = torch.empty(6, dtype=torch.int32, device=DEV)
    tw = torch.empty(6, device=DEV)
    for _ in range(2):  # the counter must self-reset
        fl.router_topk_bf16(xn.data_ptr(), wr.data_ptr(), logits.data_ptr(), counter.data_ptr(), ids.data_ptr(),
                            tw.data_ptr(), _stream())
    lg = bf16(wr.double() @ xn.double()).float().cpu()
    order = torch.sort(lg, descending=True, stable=True).indices[:6]
    assert ids.cpu().tolist() == order.tolist()
    sg = torch.sigmoid(lg[order].double())
    want = bf16(sg / (sg.sum() + 1e-20)).float()
    assert torch.equal(tw.cpu(), want)
    assert int(counter.item()) == 0


def test_moe_gate_up_and_down_combine():
    torch.manual_seed(4)
    E = 256
    eg = rand_blocks(E * 512, 2048, 30)
    eu = rand_blocks(E * 512, 2048, 31)
    ed = rand_blocks(E * 2048, 512, 32)
    gu_c = torch.empty(E * 1024, 1024, dtype=torch.uint8, device=DEV)
    gu_s = torch.empty(E * 1024, 128, dtype=torch.uint8, device=DEV)
    repack_dev(eg, 8, 16, 0, (gu_c, gu_s))   # gate|up interleaved by 8 rows
    repack_dev(eu, 8, 16, 8, (gu_c, gu_s))
    d_c, d_s = repack_dev(ed)
    sh_gu_c = torch.empty(2048, 1024, dtype=torch.uint8, device=DEV)
    sh_gu_s = torch.empty(2048, 128, dtype=torch.uint8, device=DEV)
    repack_dev(rand_blocks(1024, 2048, 33), 8, 16, 0, (sh_gu_c, sh_gu_s))
    repack_dev(rand_blocks(1024, 2048, 34), 8, 16, 8, (sh_gu_c, sh_gu_s))
    sh_d_c, sh_d_s = repack_dev(rand_blocks(2048, 1024, 35))
    egs = torch.rand(E, device=DEV) * 1e-3 + 1e-4
    egd = torch.rand(E, device=DEV) * 1e-3 + 1e-4
    sgs, sgd = 7e-4, 9e-4
    xn = bf16(torch.randn(2048, device=DEV))
    ids = torch.tensor([200, 3, 77, 128, 9, 255], dtype=torch.int32, device=DEV)
    tw = bf16(torch.tensor([0.3, 0.2, 0.2, 0.1, 0.1, 0.1], device=DEV)).float()
    act = torch.empty(4096, dtype=torch.bfloat16, device=DEV)
    fl.moe_gate_up_bf16(xn.data_ptr(), ids.data_ptr(), gu_c.data_ptr(), gu_s.data_ptr(), egs.data_ptr(),
                        sh_gu_c.data_ptr(), sh_gu_s.data_ptr(), sgs, act.data_ptr(), _stream())
    out = torch.empty(2048, dtype=torch.bfloat16, device=DEV)
    fl.moe_down_combine_bf16(act.data_ptr(), ids.data_ptr(), tw.data_ptr(), d_c.data_ptr(), d_s.data_ptr(),
                             egd.data_ptr(), sh_d_c.data_ptr(), sh_d_s.data_ptr(), sgd, out.data_ptr(), _stream())
    x64 = xn.double().cpu()
    want_act = []
    for j, e in enumerate(ids.tolist()):
        W = deq_ref(gu_c[e * 1024:(e + 1) * 1024], gu_s[e * 1024:(e + 1) * 1024])
        y = ((W @ x64) * egs[e].item()).view(64, 2, 8)
        g, u = y[:, 0].reshape(-1), y[:, 1].reshape(-1)
        want_act.append(g / (1 + torch.exp(-g)) * u)
    Wsh = deq_ref(sh_gu_c, sh_gu_s)
    ysh = ((Wsh @ x64) * sgs).view(128, 2, 8)
    g, u = bf16(ysh[:, 0].reshape(-1)).double(), bf16(ysh[:, 1].reshape(-1)).double()
    want_act.append(g / (1 + torch.exp(-g)) * u)
    want_act = torch.cat(want_act)
    close_bf16(act, want_act, "moe gate_up", rel=2.0 ** -6)
    a64 = act.double().cpu()  # chain from the kernel's own bf16 activations
    routed = torch.zeros(2048, dtype=torch.float64)
    for j, e in enumerate(ids.tolist()):
        W = deq_ref(d_c[e * 2048:(e + 1) * 2048], d_s[e * 2048:(e + 1) * 2048])
        y = bf16((W @ a64[j * 512:(j + 1) * 512]) * egd[e].item()).double()
        routed += tw[j].item() * y
    shared = bf16((deq_ref(sh_d_c, sh_d_s) @ a64[3072:]) * sgd).double()
    want = bf16(routed).double() + shared
    close_bf16(out, want, "moe down combine", rel=2.0 ** -6)


def test_argmax_next():
    n = 150144
    logits = torch.randn(n, device=DEV)
    logits[12345] = 100.0
    logits[99999] = 100.0  # tie: lowest index wins
    nb = fl.ARGMAX_BLOCKS
    cand_v = torch.empty(nb, device=DEV)
    cand_i = torch.empty(nb, dtype=torch.int32, device=DEV)
    counter = torch.zeros(1, dtype=torch.int32, device=DEV)
    tokens = torch.full((16,), 7, dtype=torch.int32, device=DEV)
    pos = torch.tensor([3], dtype=torch.int32, device=DEV)
    n_prompt = torch.tensor([5], dtype=torch.int32, device=DEV)
    for _ in range(3):  # pos 3 -> 4 (prompt token kept), 4 -> 5 (written), 5 -> 6 (written)
        fl.argmax_next_f32(logits.data_ptr(), n, cand_v.data_ptr(), cand_i.data_ptr(), counter.data_ptr(),
                           tokens.data_ptr(), pos.data_ptr(), n_prompt.data_ptr(), _stream())
    assert pos.item() == 6 and counter.item() == 0
    assert tokens.cpu().tolist()[:8] == [7, 7, 7, 7, 7, 12345, 12345, 7]


# ---------------------------------------------------------------------------
# Prefill (many-token) kernels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M", [1, 77, 300])
def test_gemm_dense_epilogues(M):
    N, K = 256, 2048
    codes, scales = repack_dev(rand_blocks(N, K, 40 + M))
    A = bf16(torch.randn(M, K, device=DEV))
    W = deq_ref(codes, scales)
    ref = (A.double().cpu() @ W.T) * 0.003
    out = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    fl.gemm_w4a16(A.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.003, out.data_ptr(), N,
                  fl.EPI_BF16, _stream())
    close_bf16(out, ref, "gemm bf16")
    out32 = torch.empty(M, N, dtype=torch.float32, device=DEV)
    fl.gemm_w4a16(A.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.003, out32.data_ptr(), N,
                  fl.EPI_F32, _stream())
    assert torch.allclose(out32.double().cpu(), ref, rtol=1e-5, atol=1e-5 * ref.abs().max().item())
    act = torch.empty(M, N // 2, dtype=torch.bfloat16, device=DEV)
    fl.gemm_w4a16(A.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.003, act.data_ptr(), N // 2,
                  fl.EPI_SWIGLU_DENSE, _stream())
    y = ref.view(M, N // 16, 2, 8)
    g, u = bf16(y[:, :, 0].reshape(M, -1)).double(), bf16(y[:, :, 1].reshape(M, -1)).double()
    close_bf16(act, g / (1 + torch.exp(-g)) * u, "gemm swiglu")


GEMM_CFGS = [(16, 4, 1, 1), (16, 4, 8, 1), (16, 4, 2, 2), (16, 3, 2, 4), (32, 4, 4, 1), (32, 3, 2, 2), (32, 2, 1, 4),
             (64, 3, 1, 1), (64, 3, 2, 1), (64, 3, 1, 2), (128, 2, 1, 1), (128, 3, 1, 1)]


@pytest.mark.parametrize("bm,stages,split,kb", GEMM_CFGS)
@pytest.mark.parametrize("M", [5, 77])
def test_gemm_configs_match_reference(bm, stages, split, kb, M):
    N, K = 256, 2048
    codes, scales = repack_dev(rand_blocks(N, K, 60 + M))
    A = bf16(torch.randn(M, K, device=DEV))
    ref = (A.double().cpu() @ deq_ref(codes, scales).T) * 0.003
    out32 = torch.empty(M, N, dtype=torch.float32, device=DEV)
    fl.gemm_w4a16_cfg(A.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.003, out32.data_ptr(), N,
                      fl.EPI_F32, bm, stages, split, kb, _stream())
    assert torch.allclose(out32.double().cpu(), ref, rtol=1e-5, atol=1e-5 * ref.abs().max().item())
    if split == 1:  # the k-order of a CTA does not depend on BM / stages: bit-identical across configs
        base = torch.empty_like(out32)
        fl.gemm_w4a16_cfg(A.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.003, base.data_ptr(), N,
                          fl.EPI_F32, 128, 3, 1, 1, _stream())
        assert torch.equal(out32, base)


@pytest.mark.parametrize("tile_rows,override", [(16, None), (32, None), (64, None), (16, (16, 3, 2, 4)),
                                               (32, (32, 4, 4, 1)), (64, (64, 3, 2, 2))])
def test_grouped_tile_rows(tile_rows, override):
    torch.manual_seed(tile_rows)
    n, E, H, I = 50, 256, 2048, 512
    xn = bf16(torch.randn(n, H, device=DEV))
    ids = torch.stack([torch.randperm(8, device=DEV)[:6] for _ in range(n)]).int()  # 8 hot experts: multi-tile
    counts = torch.bincount(ids.view(-1).long(), minlength=E).int()
    sorted_tok = torch.empty(n * 6, dtype=torch.int32, device=DEV); inv = torch.empty_like(sorted_tok)
    max_tiles = n * 6 // tile_rows + E
    tiles = torch.zeros(3 * max_tiles, dtype=torch.int32, device=DEV); ntiles = torch.zeros(1, dtype=torch.int32, device=DEV)
    fl.moe_scan_place(ids.data_ptr(), n, tile_rows, counts.data_ptr(), sorted_tok.data_ptr(), inv.data_ptr(),
                      tiles.data_ptr(), ntiles.data_ptr(), 0, 0, _stream())
    d_c, d_s = repack_dev(rand_blocks(8 * H, I, 70))  # only experts 0..7 are routed
    gs = torch.rand(E, device=DEV) * 1e-3 + 1e-4
    a = bf16(torch.randn(n * 6, I, device=DEV))
    y = torch.zeros(n * 6, H, dtype=torch.bfloat16, device=DEV)
    if override:
        fl.set_gemm_override(0, 0, 0, 0, *override)
    try:
        fl.gemm_w4a16_grouped(a.data_ptr(), 0, I, tiles.data_ptr(), ntiles.data_ptr(), max_tiles, d_c.data_ptr(),
                              d_s.data_ptr(), H, I, gs.data_ptr(), y.data_ptr(), H, fl.EPI_BF16, tile_rows, tile_rows,
                              _stream())
    finally:
        fl.set_gemm_override(0, 0, 0, 0, 0, 0, 0, 0)
    e_of = ids.view(-1).cpu()[torch.argsort(inv.cpu())]
    for slot in (0, 17, n * 6 - 1):
        e = e_of[slot].item()
        W = deq_ref(d_c[e * H:(e + 1) * H], d_s[e * H:(e + 1) * H])
        close_bf16(y[slot], (W @ a[slot].double().cpu()) * gs[e].item(), f"tile_rows {tile_rows} slot {slot}")


def test_router_place_and_grouped_gemm():
    torch.manual_seed(5)
    n, E, H, I = 37, 256, 2048, 512
    xn = bf16(torch.randn(n, H, device=DEV))
    wr = bf16(torch.randn(E, H, device=DEV) * 0.02)
    ids = torch.empty(n, 6, dtype=torch.int32, device=DEV)
    tw = torch.empty(n, 6, device=DEV)
    counts = torch.zeros(E, dtype=torch.int32, device=DEV)
    fl.router_topk_rows_bf16(xn.data_ptr(), n, wr.data_ptr(), ids.data_ptr(), tw.data_ptr(), counts.data_ptr(),
                             0, 0, _stream())
    sorted_tok = torch.empty(n * 6, dtype=torch.int32, device=DEV)
    inv = torch.empty(n * 6, dtype=torch.int32, device=DEV)
    max_tiles = n * 6 // 64 + E
    tiles = torch.zeros(3 * max_tiles, dtype=torch.int32, device=DEV)
    ntiles = torch.zeros(1, dtype=torch.int32, device=DEV)
    fl.moe_scan_place(ids.data_ptr(), n, 64, counts.data_ptr(), sorted_tok.data_ptr(), inv.data_ptr(),
                      tiles.data_ptr(), ntiles.data_ptr(), 0, 0, _stream())
    idl, st, iv = ids.view(-1).cpu(), sorted_tok.cpu(), inv.cpu()
    # grouped gate/up (routed SwiGLU rule) on the sorted slots
    eg = rand_blocks(E * I, H, 50); eu = rand_blocks(E * I, H, 51)
    gu_c = torch.empty(E * 2 * I, H // 2, dtype=torch.uint8, device=DEV)
    gu_s = torch.empty(E * 2 * I, H // 16, dtype=torch.uint8, device=DEV)
    repack_dev(eg, 8, 16, 0, (gu_c, gu_s)); repack_dev(eu, 8, 16, 8, (gu_c, gu_s))
    gs = torch.rand(E, device=DEV) * 1e-3 + 1e-4
    act = torch.zeros(n * 6, I, dtype=torch.bfloat16, device=DEV)
    fl.gemm_w4a16_grouped(xn.data_ptr(), sorted_tok.data_ptr(), H, tiles.data_ptr(), ntiles.data_ptr(), max_tiles,
                          gu_c.data_ptr(), gu_s.data_ptr(), 2 * I, H, gs.data_ptr(), act.data_ptr(), I,
                          fl.EPI_SWIGLU_ROUTED, 64, 64, _stream())
    for slot in (0, 5, n * 6 - 1):
        tok = st[slot].item(); e = idl[torch.argsort(iv)][slot].item()
        W = deq_ref(gu_c[e * 2 * I:(e + 1) * 2 * I], gu_s[e * 2 * I:(e + 1) * 2 * I])
        y = ((W @ xn[tok].double().cpu()) * gs[e].item()).view(I // 8, 2, 8)
        gg, uu = y[:, 0].reshape(-1), y[:, 1].reshape(-1)
        close_bf16(act[slot], gg / (1 + torch.exp(-gg)) * uu, f"grouped slot {slot}", rel=2.0 ** -6)


@pytest.mark.parametrize("n", [37, 300])
def test_router_topk_rows_and_scan_place(n):
    torch.manual_seed(6)
    E, H = 256, 2048
    xn = bf16(torch.randn(n, H, device=DEV))
    wr = bf16(torch.randn(E, H, device=DEV) * 0.02)
    ids = torch.empty(n, 6, dtype=torch.int32, device=DEV)
    tw = torch.empty(n, 6, device=DEV)
    counts = torch.zeros(E, dtype=torch.int32, device=DEV)
    fl.router_topk_rows_bf16(xn.data_ptr(), n, wr.data_ptr(), ids.data_ptr(), tw.data_ptr(), counts.data_ptr(),
                             0, 0, _stream())
    lg = bf16(xn.double() @ wr.double().t()).double().cpu()
    srt = torch.sort(lg, dim=1, descending=True, stable=True)
    got = ids.cpu().long()
    ulp = srt.values[:, 5].abs() * 2.0 ** -7
    clear = (srt.values[:, 5] - srt.values[:, 6]) > ulp  # selection not decided by one bf16 ulp
    assert clear.float().mean() > 0.5
    assert torch.equal(got[clear], srt.indices[clear, :6])
    sg = torch.sigmoid(srt.values[:, :6])
    want = bf16(sg / (sg.sum(1, keepdim=True) + 1e-20)).float()
    assert torch.allclose(tw.cpu()[clear], want[clear], rtol=2.0 ** -7, atol=0)
    assert torch.equal(counts.cpu(), torch.bincount(got.view(-1), minlength=E).int())
    # scan/place: tile list from the counts, expert-grouped permutation, counts re-zeroed
    max_tiles = n * 6 // 64 + E
    st_ = torch.empty(n * 6, dtype=torch.int32, device=DEV)
    iv_ = torch.empty(n * 6, dtype=torch.int32, device=DEV)
    tl_ = torch.zeros(3 * max_tiles, dtype=torch.int32, device=DEV)
    nt_ = torch.zeros(1, dtype=torch.int32, device=DEV)
    fl.moe_scan_place(ids.data_ptr(), n, 64, counts.data_ptr(), st_.data_ptr(), iv_.data_ptr(), tl_.data_ptr(),
                      nt_.data_ptr(), 0, 0, _stream())
    assert int(counts.abs().sum().item()) == 0
    cnt = torch.bincount(got.view(-1), minlength=E).tolist()
    want, off = [], 0
    for e in range(E):
        for k in range(0, cnt[e], 64):
            want += [e, off + k, min(64, cnt[e] - k)]
        off += cnt[e]
    assert int(nt_.item()) == len(want) // 3
    assert tl_.cpu()[:len(want)].tolist() == want
    idl, st, iv = ids.view(-1).cpu(), st_.cpu(), iv_.cpu()
    for i in range(n * 6):
        assert st[iv[i]].item() == i // 6
    assert torch.equal(torch.sort(iv).values, torch.arange(n * 6, dtype=torch.int32))
    e_of_slot = idl[torch.argsort(iv)]
    assert torch.equal(e_of_slot, torch.sort(e_of_slot).values)



@pytest.mark.parametrize("n", [37, 3000])   # cluster path, wide path
def test_padding_rows_are_not_routed(n):
    torch.manual_seed(10)
    E, H, valid = 256, 2048, n - 7
    xn = bf16(torch.randn(n, H, device=DEV))
    wr = bf16(torch.randn(E, H, device=DEV) * 0.02)
    pos = torch.tensor([100], dtype=torch.int32, device=DEV)
    n_end = torch.tensor([100 + valid], dtype=torch.int32, device=DEV)
    out = []
    for p, q in ((0, 0), (pos.data_ptr(), n_end.data_ptr())):
        ids = torch.empty(n, 6, dtype=torch.int32, device=DEV)
        tw = torch.empty(n, 6, device=DEV)
        counts = torch.zeros(E, dtype=torch.int32, device=DEV)
        fl.router_topk_rows_bf16(xn.data_ptr(), n, wr.data_ptr(), ids.data_ptr(), tw.data_ptr(), counts.data_ptr(),
                                 p, q, _stream())
        out.append((ids, tw, counts))
    (ids_a, tw_a, _), (ids, tw, counts) = out
    assert torch.equal(ids, ids_a) and torch.equal(tw, tw_a)
    cnt = torch.bincount(ids[:valid].reshape(-1).long(), minlength=E).int()
    assert torch.equal(counts, cnt)
    max_tiles = n * 6 // 64 + E
    st_ = torch.empty(n * 6, dtype=torch.int32, device=DEV)
    iv_ = torch.full((n * 6,), -1, dtype=torch.int32, device=DEV)
    tl_ = torch.zeros(3 * max_tiles, dtype=torch.int32, device=DEV)
    nt_ = torch.zeros(1, dtype=torch.int32, device=DEV)
    fl.moe_scan_place(ids.data_ptr(), n, 64, counts.data_ptr(), st_.data_ptr(), iv_.data_ptr(), tl_.data_ptr(),
                      nt_.data_ptr(), pos.data_ptr(), n_end.data_ptr(), _stream())
    iv, st = iv_.cpu(), st_.cpu()
    assert int(nt_.item()) == sum(-(-c // 64) for c in cnt.tolist())
    assert torch.equal(iv[valid * 6:], torch.zeros(7 * 6, dtype=torch.int32))
    assert torch.equal(torch.sort(iv[:valid * 6]).values, torch.arange(valid * 6, dtype=torch.int32))
    assert all(st[iv[i]].item() == i // 6 for i in range(valid * 6))


def test_router_cluster_path_matches_wide_path():
    torch.manual_seed(8)
    E, H, n = 256, 2048, 12000  # 375 token blocks: wide path; the first 37 rows alone: cluster path
    xn = bf16(torch.randn(n, H, device=DEV))
    wr = bf16(torch.randn(E, H, device=DEV) * 0.02)
    res = []
    for rows in (n, 37):
        ids = torch.empty(rows, 6, dtype=torch.int32, device=DEV)
        tw = torch.empty(rows, 6, device=DEV)
        counts = torch.zeros(E, dtype=torch.int32, device=DEV)
        fl.router_topk_rows_bf16(xn.data_ptr(), rows, wr.data_ptr(), ids.data_ptr(), tw.data_ptr(), counts.data_ptr(),
                                 0, 0, _stream())
        res.append((ids[:37].cpu(), tw[:37].cpu()))
        assert int(counts.sum().item()) == rows * 6
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])


def test_qk_rows_match_decode():
    n, ring = 5, 1024
    qkv = bf16(torch.randn(n, 5120, device=DEV) * 3)
    qw = torch.rand(128, device=DEV) + 0.5
    kw = torch.rand(128, device=DEV) + 0.5
    cos_t, sin_t = rope_tables(4096)
    pos = torch.tensor([2000], dtype=torch.int32, device=DEV)
    cache_r = torch.zeros(ring, 1024, dtype=torch.bfloat16, device=DEV)
    q_r = torch.empty(n, 4096, dtype=torch.bfloat16, device=DEV)
    fl.qk_norm_rope_kv_rows_bf16(qkv.data_ptr(), n, qw.data_ptr(), kw.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                                 pos.data_ptr(), q_r.data_ptr(), cache_r.data_ptr(), ring, 1, EPS, _stream())
    cache_d = torch.zeros_like(cache_r)
    q_d = torch.empty(4096, dtype=torch.bfloat16, device=DEV)
    for r in range(n):
        p1 = torch.tensor([2000 + r], dtype=torch.int32, device=DEV)
        fl.qk_norm_rope_kv_bf16(qkv[r].data_ptr(), qw.data_ptr(), kw.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                                p1.data_ptr(), q_d.data_ptr(), cache_d.data_ptr(), ring, 1, EPS, _stream())
        assert torch.equal(q_d, q_r[r])
    assert torch.equal(cache_d, cache_r)


@pytest.mark.parametrize("pos0,n,ring,window", [(0, 50, 0, 0), (300, 70, 0, 0), (0, 40, 1024, 513),
                                                (900, 130, 1024, 513), (5000, 33, 1024, 513), (2000, 200, 0, 0)])
@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n_split", [1, 3])
def test_prefill_attention(pos0, n, ring, window, variant, n_split):
    torch.manual_seed(pos0 + n)
    slots = ring if ring else pos0 + n
    cache = bf16(torch.randn(slots, 1024, device=DEV))
    q = bf16(torch.randn(n, 32, 128, device=DEV))
    sinks = torch.randn(32, device=DEV) * 3
    pos = torch.tensor([pos0], dtype=torch.int32, device=DEV)
    out = torch.empty(n, 32, 128, dtype=torch.bfloat16, device=DEV)
    units = (n + 15) // 16 * 4
    o_part = torch.empty(units * n_split * 128 * 128, device=DEV)
    ml = torch.empty(units * n_split * 128 * 2, device=DEV)
    counters = torch.zeros(units, dtype=torch.int32, device=DEV)
    for _ in range(2):  # counters self-reset
        fl.prefill_attention_bf16(q.data_ptr(), n, cache.data_ptr(), pos.data_ptr(), ring, window, sinks.data_ptr(),
                                   1 / math.sqrt(128), out.data_ptr(), n_split, o_part.data_ptr(), ml.data_ptr(),
                                   counters.data_ptr(), variant, _stream())
    assert int(counters.abs().sum().item()) == 0
    for r in (0, n // 2, n - 1):
        ref = attn_ref(q[r], cache, pos0 + r, ring, window, sinks.cpu())
        err = ((out[r].double().cpu() - ref).abs().max() / ref.abs().max()).item()
        assert err < 8e-3, (r, err)
    if variant in (1, 4):  # 32-key tiles with other pipelines: same accumulation order as variant 0
        base = torch.empty_like(out)
        fl.prefill_attention_bf16(q.data_ptr(), n, cache.data_ptr(), pos.data_ptr(), ring, window, sinks.data_ptr(),
                                  1 / math.sqrt(128), base.data_ptr(), n_split, o_part.data_ptr(), ml.data_ptr(),
                                  counters.data_ptr(), 0, _stream())
        assert torch.equal(out, base)


def test_moe_combine_rows():
    n = 9
    y = bf16(torch.randn(n * 6, 2048, device=DEV))
    inv = torch.randperm(n * 6, device=DEV).to(torch.int32)
    tw = bf16(torch.rand(n, 6, device=DEV)).float()
    sh = bf16(torch.randn(n, 2048, device=DEV))
    out = torch.empty(n, 2048, dtype=torch.bfloat16, device=DEV)
    fl.moe_combine_rows_bf16(y.data_ptr(), inv.data_ptr(), tw.data_ptr(), sh.data_ptr(), out.data_ptr(), n, _stream())
    yy = y.double().cpu()[inv.cpu().long()].view(n, 6, 2048)
    routed = (yy * tw.double().cpu()[:, :, None]).sum(1)
    want = bf16(bf16(routed).double() + sh.double().cpu())
    close_bf16(out, want, "combine", rel=2.0 ** -7, abs_frac=0)





def test_moe_combine_norm_rows_matches_two_kernels():
    torch.manual_seed(11)
    n = 37
    y = bf16(torch.randn(n * 6, 2048, device=DEV))
    inv = torch.randperm(n * 6, device=DEV).to(torch.int32)
    tw = bf16(torch.rand(n, 6, device=DEV)).float()
    sh = bf16(torch.randn(n, 2048, device=DEV))
    res = bf16(torch.randn(n, 2048, device=DEV) * 4)
    w = torch.rand(2048, device=DEV) + 0.5
    x = torch.empty(n, 2048, dtype=torch.bfloat16, device=DEV)
    res_a, out_a = res.clone(), torch.empty_like(x)
    fl.moe_combine_rows_bf16(y.data_ptr(), inv.data_ptr(), tw.data_ptr(), sh.data_ptr(), x.data_ptr(), n, _stream())
    fl.add_rmsnorm_bf16(x.data_ptr(), res_a.data_ptr(), w.data_ptr(), out_a.data_ptr(), n, EPS, 0, _stream(), 1)
    res_b, out_b = res.clone(), torch.empty_like(x)
    fl.moe_combine_norm_rows_bf16(y.data_ptr(), inv.data_ptr(), tw.data_ptr(), sh.data_ptr(), res_b.data_ptr(),
                                  w.data_ptr(), out_b.data_ptr(), n, EPS, _stream())
    assert torch.equal(res_b, res_a) and torch.equal(out_b, out_a)


@pytest.mark.parametrize("override", [None, (16, 4, 2, 2), (32, 2, 1, 4), (64, 3, 2, 1), (128, 2, 1, 1)])
@pytest.mark.parametrize("M,ring,rope", [(5, 1024, 1), (77, 0, 0), (300, 1024, 1)])
def test_gemm_qkrope_matches_gemm_plus_rows(M, ring, rope, override):
    torch.manual_seed(12)
    N, K = 5120, 2048
    codes, scales = repack_dev(rand_blocks(N, K, 62))
    a = bf16(torch.randn(M, K, device=DEV))
    qw = torch.rand(128, device=DEV) + 0.5
    kw = torch.rand(128, device=DEV) + 0.5
    cos_t, sin_t = rope_tables(4096)
    pos = torch.tensor([700], dtype=torch.int32, device=DEV)
    slots = ring if ring else 2048
    if override:
        fl.set_gemm_override(*override, 0, 0, 0, 0)
    try:
        qkv = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        fl.gemm_w4a16(a.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.01, qkv.data_ptr(), N,
                      fl.EPI_BF16, _stream())
        q_a = torch.empty(M, 4096, dtype=torch.bfloat16, device=DEV)
        c_a = torch.zeros(slots, 1024, dtype=torch.bfloat16, device=DEV)
        fl.qk_norm_rope_kv_rows_bf16(qkv.data_ptr(), M, qw.data_ptr(), kw.data_ptr(), cos_t.data_ptr(),
                                     sin_t.data_ptr(), pos.data_ptr(), q_a.data_ptr(), c_a.data_ptr(), ring, rope, EPS,
                                     _stream())
        q_b = torch.empty_like(q_a)
        c_b = torch.zeros_like(c_a)
        fl.gemm_w4a16_qkrope(a.data_ptr(), M, K, codes.data_ptr(), scales.data_ptr(), N, K, 0.01, qw.data_ptr(),
                             kw.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(), pos.data_ptr(), q_b.data_ptr(),
                             c_b.data_ptr(), ring, rope, EPS, _stream())
    finally:
        fl.set_gemm_override(0, 0, 0, 0, 0, 0, 0, 0)
    assert torch.equal(q_b, q_a) and torch.equal(c_b, c_a)

# ---------------------------------------------------------------------------
# Fused decode step kernels vs the unfused composition
# ---------------------------------------------------------------------------
def _frac_equal(a, b):
    return (a == b).float().mean().item()


def test_gemv_fused_norm_matches_unfused():
    torch.manual_seed(7)
    N, K = 5120, 2048
    codes, scales = repack_dev(rand_blocks(N, K, 60))
    x = bf16(torch.randn(K, device=DEV) * 2)
    res = bf16(torch.randn(K, device=DEV) * 8)
    w = torch.rand(K, device=DEV) + 0.5
    res_a = res.clone(); xn = torch.empty_like(x); out_a = torch.empty(N, dtype=torch.bfloat16, device=DEV)
    fl.add_rmsnorm_bf16(x.data_ptr(), res_a.data_ptr(), w.data_ptr(), xn.data_ptr(), 1, EPS, 0, _stream())
    fl.gemv_w4a16(codes.data_ptr(), scales.data_ptr(), xn.data_ptr(), out_a.data_ptr(), N, K, 0.002, 0, _stream())
    res_out = torch.zeros_like(x); out_b = torch.empty_like(out_a)
    fl.gemv_fused_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), res.data_ptr(), res_out.data_ptr(), w.data_ptr(),
                   0, EPS, out_b.data_ptr(), N, K, 0.002, 0, _stream())
    assert torch.equal(res_out, res_a)
    assert _frac_equal(out_a, out_b) > 0.99
    close_bf16(out_b, out_a.double(), "gemv_fused norm", rel=2.0 ** -7)


def test_router_norm_matches_unfused():
    torch.manual_seed(8)
    x = bf16(torch.randn(2048, device=DEV)); res = bf16(torch.randn(2048, device=DEV) * 4)
    w = torch.rand(2048, device=DEV) + 0.5
    wr = bf16(torch.randn(256, 2048, device=DEV) * 0.02)
    lg = torch.empty(256, device=DEV); ctr = torch.zeros(1, dtype=torch.int32, device=DEV)
    ids_a = torch.empty(6, dtype=torch.int32, device=DEV); tw_a = torch.empty(6, device=DEV)
    res_a = res.clone(); xn_a = torch.empty_like(x)
    fl.add_rmsnorm_bf16(x.data_ptr(), res_a.data_ptr(), w.data_ptr(), xn_a.data_ptr(), 1, EPS, 0, _stream())
    fl.router_topk_bf16(xn_a.data_ptr(), wr.data_ptr(), lg.data_ptr(), ctr.data_ptr(), ids_a.data_ptr(), tw_a.data_ptr(),
                        _stream())
    ids_b = torch.empty_like(ids_a); tw_b = torch.empty_like(tw_a); res_b = torch.zeros_like(x); xn_b = torch.empty_like(x)
    for _ in range(2):
        fl.router_norm_bf16(x.data_ptr(), res.data_ptr(), w.data_ptr(), EPS, res_b.data_ptr(), xn_b.data_ptr(),
                            wr.data_ptr(), lg.data_ptr(), ctr.data_ptr(), ids_b.data_ptr(), tw_b.data_ptr(), _stream())
    assert torch.equal(res_b, res_a) and torch.equal(xn_b, xn_a)
    assert torch.equal(ids_b, ids_a) and torch.equal(tw_b, tw_a) and ctr.item() == 0


@pytest.mark.parametrize("rows", [2, 4, 8])
def test_lm_head_greedy_matches_three_kernels(rows):
    torch.manual_seed(9)
    N, K, gs = 4001, 2048, 0.25          # a partial last block exercises idle warps
    codes, scales = repack_dev(rand_blocks(N, K, 61))
    x = bf16(torch.randn(K, device=DEV) * 2)
    res = bf16(torch.randn(K, device=DEV) * 8)
    w = torch.rand(K, device=DEV) + 0.5

    def seq():   # tokens, pos, n_prompt: pos 3 -> 4 keeps the prompt token, then 5 and 6 are written
        return (torch.full((16,), 7, dtype=torch.int32, device=DEV), torch.tensor([3], dtype=torch.int32, device=DEV),
                torch.tensor([5], dtype=torch.int32, device=DEV))

    def three(tokens, pos, n_prompt):
        res_o, xn, lg = torch.zeros_like(x), torch.empty_like(x), torch.empty(N, device=DEV)
        cv = torch.empty(fl.ARGMAX_BLOCKS, device=DEV)
        ci = torch.empty(fl.ARGMAX_BLOCKS, dtype=torch.int32, device=DEV)
        ctr = torch.zeros(1, dtype=torch.int32, device=DEV)
        for _ in range(3):
            fl.add_rmsnorm_pdl_bf16(x.data_ptr(), res.data_ptr(), w.data_ptr(), EPS, res_o.data_ptr(), xn.data_ptr(),
                                    _stream())
            fl.gemv_fused_w4a16(codes.data_ptr(), scales.data_ptr(), xn.data_ptr(), 0, 0, 0, 0, EPS, lg.data_ptr(), N, K,
                                gs, 1, _stream())
            fl.argmax_next_f32(lg.data_ptr(), N, cv.data_ptr(), ci.data_ptr(), ctr.data_ptr(), tokens.data_ptr(),
                               pos.data_ptr(), n_prompt.data_ptr(), _stream())
        return lg, res_o, ctr

    def fused(tokens, pos, n_prompt):
        res_o, lg = torch.zeros_like(x), torch.empty(N, device=DEV)
        cap = -(-N // 16)
        cv = torch.empty(cap, device=DEV)
        ci = torch.empty(cap, dtype=torch.int32, device=DEV)
        ctr = torch.zeros(1, dtype=torch.int32, device=DEV)
        fl.set_lm_head_rows(rows)
        try:
            for _ in range(3):
                fl.lm_head_greedy_w4a16(codes.data_ptr(), scales.data_ptr(), x.data_ptr(), res.data_ptr(),
                                        res_o.data_ptr(), w.data_ptr(), EPS, lg.data_ptr(), N, gs, cv.data_ptr(),
                                        ci.data_ptr(), cap, ctr.data_ptr(), tokens.data_ptr(), pos.data_ptr(),
                                        n_prompt.data_ptr(), _stream())
        finally:
            fl.set_lm_head_rows(4)
        return lg, res_o, ctr

    win = int(three(*seq())[0].argmax())
    lo = win // 3                        # an exact tie in another block: the lower index must win
    codes[lo] = codes[win]
    scales[lo] = scales[win]
    sa, sb = seq(), seq()
    lg_a, res_a, ctr_a = three(*sa)
    lg_b, res_b, ctr_b = fused(*sb)
    assert torch.equal(lg_b, lg_a) and torch.equal(res_b, res_a)
    assert sb[1].item() == 6 and ctr_b.item() == 0
    assert sb[0].tolist() == sa[0].tolist() == [7] * 5 + [lo, lo] + [7] * 9


@pytest.mark.parametrize("pos,ring,window,rope,n_split,warps", [
    (0, 0, 0, 0, 4, 4), (777, 0, 0, 0, 16, 4), (5000, 1024, 513, 1, 4, 4), (300, 1024, 513, 1, 4, 8),
    (20000, 0, 0, 0, 64, 4), (5000, 1024, 513, 1, 1, 16), (1500, 0, 0, 0, 1, 8), (9000, 0, 0, 0, 24, 8),
    (3, 0, 0, 0, 1, 4)])
def test_decode_attention_fused_matches_unfused(pos, ring, window, rope, n_split, warps):
    torch.manual_seed(pos + 1)
    slots = ring if ring else pos + 1
    cache0 = bf16(torch.randn(slots, 1024, device=DEV))
    qkv = bf16(torch.randn(5120, device=DEV) * 3)
    qw = torch.rand(128, device=DEV) + 0.5; kw = torch.rand(128, device=DEV) + 0.5
    cos_t, sin_t = rope_tables(max(pos + 1, 16))
    sinks = torch.randn(32, device=DEV) * 2
    p = torch.tensor([pos], dtype=torch.int32, device=DEV)
    o_part = torch.empty(n_split, 32, 128, device=DEV); ml = torch.empty(n_split, 32, 2, device=DEV)
    # unfused
    cache_a = cache0.clone(); q = torch.empty(4096, dtype=torch.bfloat16, device=DEV)
    fl.qk_norm_rope_kv_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                            p.data_ptr(), q.data_ptr(), cache_a.data_ptr(), ring, rope, EPS, _stream())
    out_a = torch.empty(32, 128, dtype=torch.bfloat16, device=DEV)
    fl.decode_attention_bf16(q.data_ptr(), cache_a.data_ptr(), p.data_ptr(), ring, window, sinks.data_ptr(),
                             1 / math.sqrt(128), n_split, o_part.data_ptr(), ml.data_ptr(), out_a.data_ptr(), _stream())
    # fused (run twice: counters must self-reset)
    cache_b = cache0.clone(); out_b = torch.empty_like(out_a)
    ctr = torch.zeros(4, dtype=torch.int32, device=DEV)
    for _ in range(2):
        fl.decode_attention_fused_bf16(qkv.data_ptr(), qw.data_ptr(), kw.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
                                       p.data_ptr(), cache_b.data_ptr(), ring, window, rope, EPS, sinks.data_ptr(),
                                       1 / math.sqrt(128), n_split, o_part.data_ptr(), ml.data_ptr(), ctr.data_ptr(),
                                       out_b.data_ptr(), _stream(), warps)
    assert torch.equal(cache_b, cache_a)
    assert torch.count_nonzero(ctr).item() == 0
    # P is rounded to bf16 relative to each warp's running max, so a different warp/split layout
    # legitimately changes last bits; bit-equality is only expected for the same layout.
    if warps == 4:
        assert _frac_equal(out_a, out_b) > 0.99, _frac_equal(out_a, out_b)
    close_bf16(out_b, out_a.double(), "fused attn", rel=2.0 ** -7)
    ref = attn_ref(q.view(32, 128), cache_a, pos, ring, window, sinks.cpu())
    assert ((out_b.double().cpu() - ref).abs().max() / ref.abs().max()).item() < 8e-3
