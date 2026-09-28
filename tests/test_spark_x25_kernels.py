#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the repository's Spark-X2.5 kernels.

Each kernel is compared against a straight PyTorch transcription of the
reference (modeling_spark.py plus FlashRT's NVFP4 convention). The point is to
isolate "did I write the kernel right" from "did I wire the model right", so
the tables here are small and the tolerances are exact where the reference is
exact.

Run:  python tests/test_spark_x25_kernels.py
      python -m pytest tests/test_spark_x25_kernels.py

Needs a CUDA device and the SM120 core kernels; both are skipped rather than
failed when absent, so the file is collection-safe on a CPU-only or non-SM120
machine (``flash_rt_sparkx25`` is not built there).
"""
from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import numpy as np  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("Spark-X2.5 kernels need a CUDA device", allow_module_level=True)

sk = pytest.importorskip("flash_rt.flash_rt_sparkx25")
fvk = pytest.importorskip("flash_rt.flash_rt_kernels")

DEV = "cuda"
FAILS = []

#: ``check()`` raises on a mismatch by default, so a pytest run reports the
#: kernel that failed instead of a bare count. The ``__main__`` entry point
#: clears it to collect every mismatch and print them together, which is the
#: form a bring-up run wants. Without one of the two, a mismatch only printed
#: a FAIL line and every test still passed.
RAISE_ON_FAIL = True


def check(name, got, want, atol=0.0, rtol=0.0):
    got = got.float()
    want = want.float()
    if atol == 0.0 and rtol == 0.0:
        ok = torch.equal(got, want)
    else:
        ok = torch.allclose(got, want, atol=atol, rtol=rtol)
    err = (got - want).abs().max().item()
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name:38s} max|diff|={err:.3e}")
    if not ok:
        FAILS.append(name)
        if RAISE_ON_FAIL:
            raise AssertionError(f"{name}: max|diff|={err:.3e} (atol={atol}, rtol={rtol})")
    return ok


# ── NVFP4 encoding, transcribed from FlashRT's nvfp4_convert.cuh ──────────
def fp4_e2m1(v: torch.Tensor) -> torch.Tensor:
    a = v.abs()
    mag = torch.zeros_like(a)
    for lo, hi, m in ((0.0, 0.25, 0), (0.25, 0.75, 1), (0.75, 1.25, 2),
                      (1.25, 1.75, 3), (1.75, 2.5, 4), (2.5, 3.5, 5),
                      (3.5, 5.0, 6), (5.0, float("inf"), 7)):
        mag = torch.where((a >= lo) & (a < hi), torch.full_like(a, m), mag)
    sign = torch.where(v < 0, torch.full_like(a, 8), torch.zeros_like(a))
    return (sign + mag).to(torch.uint8)


def ue4m3_ceil(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> UE4M3 byte with round-up, including the >240 -> 0xFE quirk."""
    out = torch.zeros_like(v, dtype=torch.uint8)
    pos = v > 0
    if not pos.any():
        return out
    x = v[pos]
    # decode table for all 128 UE4M3 codes
    codes = torch.arange(256, dtype=torch.int32, device=v.device)
    e = (codes >> 3) & 0xF
    m = codes & 0x7
    dec = torch.where(e == 0, (m.float() / 8.0) * (2.0 ** -6),
                      (1.0 + m.float() / 8.0) * (2.0 ** (e.float() - 7)))
    dec[0x80:] = 0.0  # sign bit unused in this path
    dec = dec[:128]
    # smallest code whose decoded value >= x
    idx = torch.searchsorted(dec.contiguous(), x.contiguous(), right=False)
    idx = idx.clamp(max=127)
    byte = idx.to(torch.uint8)
    byte = torch.where(x > 240.0, torch.full_like(byte, 0xFE), byte)
    out[pos] = byte
    return out


def ue4m3_decode(byte: torch.Tensor) -> torch.Tensor:
    b = byte.to(torch.int32)
    e = (b >> 3) & 0xF
    m = b & 0x7
    return torch.where(e == 0, (m.float() / 8.0) * (2.0 ** -6),
                       (1.0 + m.float() / 8.0) * (2.0 ** (e.float() - 7)))


# ── 1. gelu_mul + NVFP4 quantize ─────────────────────────────────────────
def test_gelu_mul_nvfp4():
    print("gelu_mul_to_nvfp4_swizzled_bf16")
    rows, cols = 4, 64
    torch.manual_seed(0)
    gate = (torch.randn(rows, cols, device=DEV) * 2).to(torch.bfloat16)
    up = (torch.randn(rows, cols, device=DEV) * 2).to(torch.bfloat16)

    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device=DEV)
    ncb = ((cols // 16) + 3) // 4
    sf = torch.zeros(((rows + 127) // 128) * ncb * 512, dtype=torch.uint8, device=DEV)
    sk.gelu_mul_to_nvfp4_swizzled_bf16(
        gate.data_ptr(), up.data_ptr(), packed.data_ptr(), sf.data_ptr(),
        rows, cols, cols, cols, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # reference: bf16 gelu, bf16 product, then amax/6 -> UE4M3 ceil -> e2m1
    g = gate.float()
    gelu = 0.5 * g * (1.0 + torch.erf(g * 0.7071067811865476))
    gelu_bf = gelu.to(torch.bfloat16)
    prod = (gelu_bf.float() * up.float()).to(torch.bfloat16)

    want = torch.zeros(rows, cols // 2, dtype=torch.uint8, device=DEV)
    for r in range(rows):
        row = prod[r].float().view(-1, 16)
        amax = row.abs().amax(dim=1)
        byte = ue4m3_ceil(amax / 6.0)
        scale = ue4m3_decode(byte)
        q = prod[r].float() / scale.repeat_interleave(16)
        codes = fp4_e2m1(q)
        lo, hi = codes[0::2], codes[1::2]
        want[r] = hi * 16 + lo
    check("packed fp4 bytes", packed, want)

    # spot-check the swizzled scale factor for row 0
    want_sf0 = byte = ue4m3_ceil(
        prod[0].float().view(-1, 16).abs().amax(dim=1) / 6.0)
    got_sf0 = sf[:4]
    check("row0 UE4M3 block scales", got_sf0, want_sf0)


# ── 2. partial RoPE + KV cache write ─────────────────────────────────────
def test_qkv_post_rope():
    print("qkv_post_rope_kvwrite_bf16")
    q_heads, kv_heads, hd = 4, 2, 16
    rope_dim = 8           # partial: only the first 8 of 16 rotate
    rows, ring, mirrored = 3, 4, 1
    max_pos = 32
    torch.manual_seed(1)
    qkv_dim = q_heads * hd + 2 * kv_heads * hd
    qkv = (torch.randn(rows, qkv_dim, device=DEV) * 1.5).to(torch.bfloat16)

    inv = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    pos = torch.arange(max_pos, dtype=torch.float64)
    freqs = torch.outer(pos, inv)
    cos_t = freqs.cos().float().contiguous().cuda()
    sin_t = freqs.sin().float().contiguous().cuda()

    pos_start = 5
    q_buf = torch.zeros(rows, q_heads * hd, device=DEV, dtype=torch.bfloat16)
    # linear cache big enough for every position this test writes
    k_cache = torch.zeros(pos_start + rows, kv_heads * hd, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    # the kernel reads the position from device memory so a captured CUDA Graph
    # does not embed it in the launch
    pos_dev = torch.tensor([pos_start], dtype=torch.int32, device=DEV)
    # absolute indexing: this test's reference keeps every position in place
    lin_w = 0
    # the kernel writes both a linear cache (what prefill reads) and, when a
    # ring pointer is given, a mirrored 2W ring (what decode reads)
    # the trailing four pointers are the optional E4M3 mirror
    # (k8, v8, k8_scale, v8_scale); null here, the test covers bf16 only
    sk.qkv_post_rope_kvwrite_bf16(
        qkv.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
        q_buf.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        0, 0,
        0, 0, 0, 0,
        rows, q_heads, kv_heads, hd, rope_dim, pos_dev.data_ptr(), ring,
        lin_w,
        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    def ref_rope(x, cos, sin, rd):
        # `cos`/`sin` carry only rd/2 entries: the checkpoint duplicates
        # freqs along the last axis, so cos[j] == cos[j + rd/2].
        xf = x.float()
        rot, pas = xf[..., :rd], xf[..., rd:]
        half = rd // 2
        x1, x2 = rot[..., :half], rot[..., half:]
        c = cos[:half]
        s = sin[:half]
        out_rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        return torch.cat([out_rot, pas], dim=-1).to(torch.bfloat16)

    q_src = qkv[:, : q_heads * hd].view(rows, q_heads, hd)
    want_q = torch.zeros_like(q_buf)
    k_src = qkv[:, q_heads * hd: q_heads * hd + kv_heads * hd].view(rows, kv_heads, hd)
    v_src = qkv[:, q_heads * hd + kv_heads * hd:].view(rows, kv_heads, hd)
    for r in range(rows):
        p = pos_start + r
        want_q[r] = ref_rope(q_src[r], cos_t[p], sin_t[p], rope_dim).reshape(-1)
    check("Q with partial RoPE", q_buf, want_q)

    want_k = torch.zeros_like(k_cache)
    want_v = torch.zeros_like(v_cache)
    for r in range(rows):
        p = pos_start + r
        kk = ref_rope(k_src[r], cos_t[p], sin_t[p], rope_dim)
        want_k[p] = kk.reshape(-1)
        want_v[p] = v_src[r].reshape(-1)
    check("K linear cache", k_cache, want_k)
    check("V linear cache", v_cache, want_v)


def test_qkv_ring():
    """The mirrored ring must hold every position twice, at s and s+W."""
    print("qkv_post_rope_kvwrite_bf16 (mirrored ring)")
    q_heads, kv_heads, hd, rope_dim, W = 1, 1, 4, 4, 8
    max_pos = 3 * W
    torch.manual_seed(5)
    rows = max_pos
    qkv = (torch.randn(rows, q_heads * hd + 2 * kv_heads * hd, device=DEV)).to(torch.bfloat16)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float64), inv)
    cos_t = freqs.cos().float().contiguous().cuda()
    sin_t = freqs.sin().float().contiguous().cuda()
    q_buf = torch.zeros(rows, q_heads * hd, device=DEV, dtype=torch.bfloat16)
    k_lin = torch.zeros(max_pos, kv_heads * hd, device=DEV, dtype=torch.bfloat16)
    v_lin = torch.zeros_like(k_lin)
    k_ring = torch.zeros(W, kv_heads * hd, device=DEV, dtype=torch.bfloat16)
    v_ring = torch.zeros_like(k_ring)
    pos_dev = torch.tensor([0], dtype=torch.int32, device=DEV)
    lin_w = 0
    # prefill writes only the linear cache (a multi-row ring write would race),
    # so no ring pointers here
    sk.qkv_post_rope_kvwrite_bf16(
        qkv.data_ptr(), cos_t.data_ptr(), sin_t.data_ptr(),
        q_buf.data_ptr(), k_lin.data_ptr(), v_lin.data_ptr(), 0, 0,
        0, 0, 0, 0,
        rows, q_heads, kv_heads, hd, rope_dim, pos_dev.data_ptr(), W,
        lin_w,
        torch.cuda.current_stream().cuda_stream)
    sk.seed_ring_bf16(k_lin.data_ptr(), v_lin.data_ptr(),
                      k_ring.data_ptr(), v_ring.data_ptr(),
                      max(0, max_pos - W), min(max_pos, W), kv_heads * hd, W,
                      lin_w,
                      torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    # A ring only retains the most recent W positions; earlier slots have been
    # overwritten. Check the surviving window, in both copies.
    ok = True
    for p in range(max_pos - W, max_pos):
        s_ = p % W
        ok &= torch.equal(k_ring[s_], k_lin[p])
        ok &= torch.equal(v_ring[s_], v_lin[p])
    check("ring holds the surviving window", torch.tensor(1.0 if ok else 0.0),
          torch.tensor(1.0))
    # every one of the W window positions must be present
    present = {tuple(k_ring[s_].tolist()) for s_ in range(W)}
    check("ring holds all W distinct window slots", torch.tensor(float(len(present))),
          torch.tensor(float(W)))


# ── 5. fused gated GeGLU decode GEMM ────────────────────────────────────
def test_m1_gated_geglu():
    """The fused M=1 gated GeGLU GEMM must reproduce the shipped two-kernel path.

    Same weights, same activation, same numerics: gate_up GEMM to bf16, then
    gelu(gate)*up with the bf16 round trips, then the per-16-block NVFP4 pack.
    """
    print("fp4_w4a4_mma_sm120_gated_geglu_fp4out (M=1 decode)")
    M, K, inter = 1, 2560, 1024
    N = 2 * inter
    torch.manual_seed(9)
    Wg = (torch.randn(inter, K, device=DEV) * 0.02).to(torch.bfloat16)
    Wu = (torch.randn(inter, K, device=DEV) * 0.02).to(torch.bfloat16)
    A = torch.randn(M, K, device=DEV).to(torch.bfloat16)

    ap = torch.empty(M, K // 2, dtype=torch.uint8, device=DEV)
    asf = torch.zeros(int(fvk.nvfp4_sf_swizzled_bytes(M, K)), dtype=torch.uint8, device=DEV)
    fvk.quantize_bf16_to_nvfp4_swizzled(A.data_ptr(), ap.data_ptr(), asf.data_ptr(),
                                        M, K, torch.cuda.current_stream().cuda_stream)

    def quant(w):
        n, k = w.shape
        pk = torch.empty(n, k // 2, dtype=torch.uint8, device=DEV)
        sf = torch.zeros(int(fvk.nvfp4_sf_swizzled_bytes(n, k)), dtype=torch.uint8, device=DEV)
        sc = torch.zeros(1, device=DEV)
        gs = torch.zeros(1, device=DEV)
        fvk.bf16_weight_to_nvfp4_swizzled(w.contiguous().data_ptr(), pk.data_ptr(), sf.data_ptr(),
                                          sc.data_ptr(), gs.data_ptr(), n, k,
                                          torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return pk, sf, float(gs.item())

    # shipped path: concatenated [gate ; up], bf16 GEMM, then the fused activation
    pc, sfc, gc = quant(torch.cat([Wg, Wu], 0))
    du = torch.zeros(M, N, device=DEV, dtype=torch.bfloat16)
    fvk.fp4_w4a4_mma_sm120_full_n_bf16out(ap.data_ptr(), pc.data_ptr(), du.data_ptr(),
                                          N, K, asf.data_ptr(), sfc.data_ptr(), gc, 0)
    do = torch.zeros(M, inter // 2, dtype=torch.uint8, device=DEV)
    so = torch.zeros(int(fvk.nvfp4_sf_swizzled_bytes(M, inter)), dtype=torch.uint8, device=DEV)
    sk.gelu_mul_to_nvfp4_swizzled_bf16(du.data_ptr(), du.data_ptr() + inter * 2,
                                       do.data_ptr(), so.data_ptr(), M, inter, inter, inter, 0)

    # fused path: interleaved weight, gated epilogue in the M=1 MMA
    wil = torch.empty(N, K, device=DEV, dtype=torch.bfloat16)
    wil[0::2] = Wg
    wil[1::2] = Wu
    pi, sfi, gi = quant(wil)
    dp = torch.zeros(M, inter // 2, dtype=torch.uint8, device=DEV)
    sp = torch.zeros_like(so)
    fvk.fp4_w4a4_mma_sm120_gated_geglu_fp4out(ap.data_ptr(), pi.data_ptr(), dp.data_ptr(),
                                              sp.data_ptr(), asf.data_ptr(), sfi.data_ptr(),
                                              gi, N, K, 0)
    torch.cuda.synchronize()
    check("packed gated activation == two-kernel path", dp, do)
    check("swizzled output SF == two-kernel path", sp, so)


# ── 3. attention output gate ─────────────────────────────────────────────
def test_attn_out_gate():
    print("attn_out_gate_bf16")
    rows, heads, hd = 3, 4, 8
    torch.manual_seed(2)
    attn = (torch.randn(rows, heads * hd, device=DEV)).to(torch.bfloat16)
    gate = (torch.randn(rows, heads, device=DEV) * 2).to(torch.bfloat16)
    out = torch.zeros_like(attn)
    sk.attn_out_gate_bf16(attn.data_ptr(), gate.data_ptr(), out.data_ptr(),
                          rows, heads, hd, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    sig = torch.sigmoid(gate.float()).to(torch.bfloat16)
    want = (attn.float() * sig.repeat_interleave(hd, dim=1).float()).to(torch.bfloat16)
    check("gated attention output", out, want)


# ── 4. attention gate projection ─────────────────────────────────────────
def test_boundary_norm():
    """The repo boundary kernel must match FlashRT's v2 byte for byte.

    v2 is what decode ran before; this one is the same arithmetic restructured
    around a single row. What has to hold is that everything downstream of it --
    the qkv/gate_up GEMM's activation operand -- is unchanged: the packed E2M1
    nibbles and the swizzled UE4M3 scales must be identical, and h_post must be
    identical. The row sum of squares is associated differently, so `rms` can
    differ in the last fp32 ulp; this test measures whether that ever reaches
    the bytes rather than assuming it does not.
    """
    print("residual_add_rms_norm_to_nvfp4_bf16 (decode row)")
    cols = 2560
    ncb = (cols // 16 + 3) // 4
    sf_bytes = ncb * 512

    for scale in (1.0, 12.0, 0.05):
        torch.manual_seed(3)
        # realistic magnitudes: the residual stream is O(1), the sub-block
        # output is smaller, and the norm weight is O(1)
        h_in = (torch.randn(cols, device=DEV) * scale).to(torch.bfloat16)
        attn = (torch.randn(cols, device=DEV) * scale * 0.3).to(torch.bfloat16)
        w = (torch.randn(cols, device=DEV) * 0.5 + 1.0).to(torch.bfloat16)

        def run(fn):
            post = torch.zeros(cols, device=DEV, dtype=torch.bfloat16)
            pk = torch.zeros(cols // 2, dtype=torch.uint8, device=DEV)
            sf = torch.zeros(sf_bytes, dtype=torch.uint8, device=DEV)
            fn(post, pk, sf)
            torch.cuda.synchronize()
            return post, pk, sf

        ref = run(lambda po, pk, sf: fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16_v2(
            h_in.data_ptr(), attn.data_ptr(), po.data_ptr(), w.data_ptr(),
            pk.data_ptr(), sf.data_ptr(), 1, cols, 1e-6, 0))
        got = run(lambda po, pk, sf: sk.residual_add_rms_norm_to_nvfp4_bf16(
            h_in.data_ptr(), attn.data_ptr(), po.data_ptr(), w.data_ptr(),
            pk.data_ptr(), sf.data_ptr(), cols, 1e-6, 0))

        check(f"h_post (scale {scale})", got[0], ref[0])
        check(f"packed fp4 bytes (scale {scale})", got[1], ref[1])
        check(f"swizzled UE4M3 scales (scale {scale})", got[2], ref[2])


def test_argmax():
    """The decode argmax must match torch over the real vocab, ties included.

    It replaces FlashRT's single-stride scan for a 16-byte-load version, so the
    things that can go wrong are the vector tail and the tie-break. The
    reference (and FlashRT's kernel) resolves equal logits to the LOWEST index,
    which torch.argmax does too.
    """
    print("argmax_bf16 (M=1 greedy sample)")
    vocab = 131072
    tok = torch.zeros(1, dtype=torch.int64, device=DEV)

    torch.manual_seed(5)
    x = (torch.randn(vocab, device=DEV) * 4).to(torch.bfloat16)
    sk.argmax_bf16(x.data_ptr(), tok.data_ptr(), vocab, 0)
    torch.cuda.synchronize()
    check("random logits", tok, x.float().argmax().reshape(1))

    # All-equal logits: every lane ties, so only the tie order decides.
    x = torch.full((vocab,), 0.5, device=DEV, dtype=torch.bfloat16)
    sk.argmax_bf16(x.data_ptr(), tok.data_ptr(), vocab, 0)
    torch.cuda.synchronize()
    check("all-tied logits", tok, torch.zeros(1, dtype=torch.int64, device=DEV))

    # Ties spread across the stride boundaries of a 1024-thread scan.
    x = torch.zeros(vocab, device=DEV, dtype=torch.bfloat16)
    for i in (1023, 1024, 8 * 1024, 65536, 131071):
        x[i] = 9.0
    sk.argmax_bf16(x.data_ptr(), tok.data_ptr(), vocab, 0)
    torch.cuda.synchronize()
    check("tied maxima across strides", tok, x.float().argmax().reshape(1))

    # A maximum in the final vector, so the tail loop and the main loop meet.
    x = torch.zeros(vocab, device=DEV, dtype=torch.bfloat16)
    x[vocab - 1] = 7.0
    sk.argmax_bf16(x.data_ptr(), tok.data_ptr(), vocab, 0)
    torch.cuda.synchronize()
    check("maximum at the last element", tok, x.float().argmax().reshape(1))


def test_decode_gemm_splits():
    """The decode GEMM's warp-split shapes must agree with the `full_n` kernel.

    `SparkX25Runtime._DECODE_SPLIT` routes the three narrow-N decode shapes to
    FlashRT's warp-split-K kernel because `full_n` launches one warp per block
    over N/8 blocks and cannot fill a 36-SM part at N=2560. The two kernels
    associate the K accumulation differently -- warp-split sums `warps` fp32
    partials in shared memory -- so the outputs are not bit-identical; they must
    agree to within the bf16 rounding of that reassociation.

    A wrong launch (bad swizzle, wrong warp count for K/64) shows up here as a
    large disagreement, which is the failure the runtime would otherwise take
    silently all the way to a decode token.
    """
    print("decode GEMM: warp-split-K vs full_n at the production shapes")
    from flash_rt.models.spark_x25.pipeline_rtx import _DECODE_SPLIT

    torch.manual_seed(11)
    for (N, K), (warps, stages) in sorted(_DECODE_SPLIT.items()):
        W = (torch.randn(N, K, device=DEV) * 0.03).to(torch.bfloat16)
        A = torch.randn(1, K, device=DEV).to(torch.bfloat16)

        ap = torch.empty(1, K // 2, dtype=torch.uint8, device=DEV)
        asf = torch.zeros(int(fvk.nvfp4_sf_swizzled_bytes(1, K)),
                          dtype=torch.uint8, device=DEV)
        fvk.quantize_bf16_to_nvfp4_swizzled(A.data_ptr(), ap.data_ptr(),
                                            asf.data_ptr(), 1, K, 0)
        pk = torch.empty(N, K // 2, dtype=torch.uint8, device=DEV)
        sf = torch.zeros(int(fvk.nvfp4_sf_swizzled_bytes(N, K)),
                         dtype=torch.uint8, device=DEV)
        sc = torch.zeros(1, device=DEV)
        gs = torch.zeros(1, device=DEV)
        fvk.bf16_weight_to_nvfp4_swizzled(W.contiguous().data_ptr(), pk.data_ptr(),
                                          sf.data_ptr(), sc.data_ptr(), gs.data_ptr(),
                                          N, K, 0)
        torch.cuda.synchronize()

        ref = torch.zeros(N, device=DEV, dtype=torch.bfloat16)
        fvk.fp4_w4a4_mma_sm120_full_n_bf16out(ap.data_ptr(), pk.data_ptr(),
                                              ref.data_ptr(), N, K, asf.data_ptr(),
                                              sf.data_ptr(), float(gs.item()), 0)
        got = torch.zeros(N, device=DEV, dtype=torch.bfloat16)
        rc = fvk.fp4_w4a4_mma_sm120_warpsplit_bf16out(
            ap.data_ptr(), pk.data_ptr(), got.data_ptr(), N, K, asf.data_ptr(),
            sf.data_ptr(), float(gs.item()), warps, stages, 0)
        torch.cuda.synchronize()
        assert rc == 0, f"warpsplit launch rejected N={N} K={K} warps={warps}"
        cos = F.cosine_similarity(got.float(), ref.float(), dim=0).item()
        # bf16 ulp at the output scale, times a few for the reassociation
        tol = 8.0 * (ref.float().abs().max().item() * 2 ** -8)
        check(f"N={N} K={K} w{warps}s{stages} == full_n", got, ref, atol=tol)
        print(f"      cos={cos:.7f}  max|out|={ref.float().abs().max().item():.3f}")


def test_gproj():
    """gproj must match a bf16 Linear within bf16 rounding."""
    print("gproj_bf16")
    rows, g_dim, K = 5, 16, 2560
    torch.manual_seed(3)
    x = (torch.randn(rows, K, device=DEV) * 0.5).to(torch.bfloat16)
    W = (torch.randn(g_dim, K, device=DEV) * 0.02).to(torch.bfloat16)
    out = torch.zeros(rows, g_dim, device=DEV, dtype=torch.bfloat16)
    sk.gproj_bf16(x.data_ptr(), W.data_ptr(), out.data_ptr(), rows, g_dim, K,
                  torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    want = F.linear(x.float(), W.float())
    # The kernel accumulates in fp32 and rounds once, exactly like the
    # reference, so the result is the bf16 rounding of the fp32 product
    # (not merely close to it).
    check("g_proj == bf16(fp32 reference)", out,
          want.to(torch.bfloat16), atol=0.0)
    check("g_proj stays within bf16 of fp32",
          (out.float() - want).abs().max(), torch.tensor(0.0), atol=2e-3)


if __name__ == "__main__":
    # Collect every mismatch instead of stopping at the first one: this is the
    # bring-up path, where seeing all the failing kernels in one run matters
    # more than which one aborted.
    RAISE_ON_FAIL = False
    torch.cuda.init()
    test_gelu_mul_nvfp4()
    test_qkv_post_rope()
    test_qkv_ring()
    test_m1_gated_geglu()
    test_attn_out_gate()
    test_boundary_norm()
    test_argmax()
    test_decode_gemm_splits()
    test_gproj()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("all kernel tests passed")
