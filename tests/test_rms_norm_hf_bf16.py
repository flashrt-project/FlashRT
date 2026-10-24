#!/usr/bin/env python3
"""HF-parity BF16 kernels equal the PyTorch ops they replace, bit for bit.

  rms_norm_hf_bf16 / residual_add_rms_norm_hf_bf16  vs  Qwen3RMSNorm
  qk_rms_norm_rope_hf_bf16                          vs  q/k_norm + rotate-half RoPE
  silu_mul_hf_bf16                                  vs  F.silu(g) * u

The reference is torch eager on the same device, so the comparison includes
PyTorch's reduction order; row counts below 16 are covered because torch
widens its reduction block there. The wrong reduction tree must disagree,
which shows the comparison can tell the trees apart.

Usage:
  PYTHONPATH=. pytest -q tests/test_rms_norm_hf_bf16.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")
F = torch.nn.functional

BF16 = torch.bfloat16
EPS = 1e-6


def _fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError as exc:
        pytest.skip(f"flash_rt_kernels not built: {exc}")
    if not hasattr(fvk, "rms_norm_hf_bf16"):
        pytest.skip("flash_rt_kernels predates the *_hf_bf16 kernels")
    return fvk


def _skip_pdl_below_sm90(pdl):
    if pdl and torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("programmatic dependent launch needs SM90+")


def _tree():
    from flash_rt.models.qwen3.prefill_bf16 import torch_reduce_tree
    return torch_reduce_tree()


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _hf_rms(x, w):
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    return w * (x32 * torch.rsqrt(var + EPS)).to(x.dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


ROWS = [1, 2, 3, 7, 8, 15, 16, 17, 255, 1024]


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dim", [1000, 2560, 4096])
def test_rms_norm_matches_hf(rows, dim, pdl):
    fvk = _fvk()
    _skip_pdl_below_sm90(pdl)
    torch.manual_seed(rows * 31 + dim)
    x = (torch.randn(rows, dim, device="cuda") * torch.rand(rows, 1, device="cuda") * 8).to(BF16)
    d = (torch.randn(rows, dim, device="cuda") * 2).to(BF16)
    w = (torch.randn(dim, device="cuda") * 0.3 + 1).to(BF16)
    out = torch.empty_like(x)
    fvk.rms_norm_hf_bf16(x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, dim,
                         EPS, _tree(), pdl, _stream())
    assert torch.equal(out, _hf_rms(x, w))

    r = x.clone()
    out2 = torch.empty_like(x)
    fvk.residual_add_rms_norm_hf_bf16(r.data_ptr(), d.data_ptr(), w.data_ptr(),
                                      out2.data_ptr(), rows, dim, EPS, _tree(),
                                      pdl, _stream())
    ref = x + d
    assert torch.equal(r, ref)
    assert torch.equal(out2, _hf_rms(ref, w))


def _rope_tables(seq, hd=128, theta=1e6):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    f = torch.arange(seq, device="cuda", dtype=torch.float32)[:, None] * inv.cuda()[None]
    emb = torch.cat((f, f), -1)
    return emb.cos().to(BF16), emb.sin().to(BF16)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("seq", [1, 2, 8, 255, 1024])
def test_qk_norm_rope_matches_hf(seq, pdl):
    fvk = _fvk()
    _skip_pdl_below_sm90(pdl)
    nq, nk, hd = 32, 8, 128
    torch.manual_seed(seq)
    q = (torch.randn(seq, nq * hd, device="cuda") * 3).to(BF16)
    k = (torch.randn(seq, nk * hd, device="cuda") * 3).to(BF16)
    wq = (torch.randn(hd, device="cuda") * 0.2 + 1).to(BF16)
    wk = (torch.randn(hd, device="cuda") * 0.2 + 1).to(BF16)
    cos, sin = _rope_tables(seq)
    qn, kn = _hf_rms(q.view(seq, nq, hd), wq), _hf_rms(k.view(seq, nk, hd), wk)
    q_ref = qn * cos[:, None] + _rotate_half(qn) * sin[:, None]
    k_ref = kn * cos[:, None] + _rotate_half(kn) * sin[:, None]
    ch, sh = cos[:, :hd // 2].contiguous(), sin[:, :hd // 2].contiguous()
    fvk.qk_rms_norm_rope_hf_bf16(q.data_ptr(), nq * hd, nq, k.data_ptr(), nk * hd, nk,
                                 wq.data_ptr(), wk.data_ptr(), ch.data_ptr(), sh.data_ptr(),
                                 seq, hd, EPS, _tree(), pdl, _stream())
    assert torch.equal(q.view(seq, nq, hd), q_ref)
    assert torch.equal(k.view(seq, nk, hd), k_ref)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("n", [8, 9728 * 8, 9728 * 1024])
def test_silu_mul_matches_hf(n, pdl):
    fvk = _fvk()
    _skip_pdl_below_sm90(pdl)
    torch.manual_seed(n % 997)
    g = (torch.randn(n, device="cuda") * 4).to(BF16)
    u = (torch.randn(n, device="cuda") * 2).to(BF16)
    ref = F.silu(g) * u
    out = torch.empty_like(g)
    fvk.silu_mul_hf_bf16(g.data_ptr(), u.data_ptr(), out.data_ptr(), n, pdl, _stream())
    assert torch.equal(out, ref)
    fvk.silu_mul_hf_bf16(g.data_ptr(), u.data_ptr(), g.data_ptr(), n, pdl, _stream())
    assert torch.equal(g, ref)


def test_wrong_reduction_tree_is_detected():
    # A different tree moves a row's fp32 sum by an ulp, which flips a bf16
    # rounding only rarely; enough rows make at least one flip certain.
    fvk = _fvk()
    torch.manual_seed(7)
    rows, dim = 8192, 2560
    x = (torch.randn(rows, dim, device="cuda") * torch.rand(rows, 1, device="cuda") * 8).to(BF16)
    w = (torch.randn(dim, device="cuda") * 0.3 + 1).to(BF16)
    out = torch.empty_like(x)
    fvk.rms_norm_hf_bf16(x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, dim,
                         EPS, 1 - _tree(), False, _stream())
    assert not torch.equal(out, _hf_rms(x, w))


def test_rejects_unsupported_arguments():
    fvk = _fvk()
    x = torch.empty(4, 128, device="cuda", dtype=BF16)
    with pytest.raises(RuntimeError):
        fvk.rms_norm_hf_bf16(x.data_ptr(), x.data_ptr(), x.data_ptr(), 4, 128,
                             EPS, _tree(), False, _stream())
    with pytest.raises(RuntimeError):
        fvk.rms_norm_hf_bf16(x.data_ptr(), x.data_ptr(), x.data_ptr(), 4, 256,
                             EPS, 2, False, _stream())
    with pytest.raises(RuntimeError):
        fvk.qk_rms_norm_rope_hf_bf16(x.data_ptr(), 64, 1, x.data_ptr(), 64, 1,
                                     x.data_ptr(), x.data_ptr(), x.data_ptr(),
                                     x.data_ptr(), 4, 64, EPS, _tree(), False,
                                     _stream())
