"""Validation of the FA2 FP16 causal forward path (Chameleon-7B on SM87)."""

import math

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    import flash_rt.flash_rt_fa2 as fa2
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"flash_rt_fa2 is not built: {exc}", allow_module_level=True)

if not hasattr(fa2, "fwd_fp16_causal"):
    pytest.skip("fwd_fp16_causal not exported", allow_module_level=True)

torch.manual_seed(0)

HD = 128
SCALE = 1.0 / math.sqrt(HD)


def _run_fp16_causal(q, k, v, num_sms=0):
    """q/k/v: [B, S, NH, HD] fp16 contiguous. Returns O [B, Sq, NHq, HD]."""
    B, sq, nhq, hd = q.shape
    sk = k.shape[1]
    nhkv = k.shape[2]
    o = torch.empty(B, sq, nhq, hd, device="cuda", dtype=torch.float16)
    lse = torch.empty(B, nhq, sq, device="cuda", dtype=torch.float32)
    n_splits = min(128, (sk + 63) // 64)
    lse_accum = torch.empty(n_splits, B, nhq, sq, device="cuda", dtype=torch.float32)
    o_accum = torch.empty(n_splits, B, nhq, sq, hd, device="cuda", dtype=torch.float32)
    fa2.fwd_fp16_causal(
        Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
        O=o.data_ptr(), softmax_lse=lse.data_ptr(),
        softmax_lse_accum=lse_accum.data_ptr(), o_accum=o_accum.data_ptr(),
        batch=B, seqlen_q=sq, seqlen_k=sk,
        num_heads_q=nhq, num_heads_kv=nhkv, head_dim=hd,
        q_strides=(q.stride(0), q.stride(1), q.stride(2)),
        k_strides=(k.stride(0), k.stride(1), k.stride(2)),
        v_strides=(v.stride(0), v.stride(1), v.stride(2)),
        o_strides=(o.stride(0), o.stride(1), o.stride(2)),
        softmax_scale=SCALE, num_sms=num_sms)
    torch.cuda.synchronize()
    return o


def _ref_causal(q, k, v):
    """Torch SDPA causal reference. Inputs [B, S, NH, HD] fp16."""
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    if kt.shape[1] != qt.shape[1]:
        rep = qt.shape[1] // kt.shape[1]
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=(q.shape[1] == k.shape[1]),
                                         scale=SCALE)
    return out.transpose(1, 2)


def _cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def test_prefill_matches_sdpa_causal():
    B, S, NH = 1, 128, 4
    q = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16)
    o = _run_fp16_causal(q, k, v)
    ref = _ref_causal(q, k, v)
    assert _cos(o, ref) >= 0.999, f"prefill cosine {_cos(o, ref)} < 0.999"


def test_q_len_1_decode_matches_last_row():
    B, SK, NH = 1, 256, 4
    q = torch.randn(B, 1, NH, HD, device="cuda", dtype=torch.float16)
    k = torch.randn(B, SK, NH, HD, device="cuda", dtype=torch.float16)
    v = torch.randn(B, SK, NH, HD, device="cuda", dtype=torch.float16)
    o = _run_fp16_causal(q, k, v)
    # q_len=1 attends to all SK keys (causal mask degenerates to full row).
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * SCALE
    ref = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v.float())
    assert _cos(o, ref) >= 0.999, f"decode cosine {_cos(o, ref)} < 0.999"


def test_causality_future_keys_do_not_leak():
    B, S, NH = 1, 64, 4
    q = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, NH, HD, device="cuda", dtype=torch.float16)
    o1 = _run_fp16_causal(q, k, v)

    # Perturb key/value at the LAST position: rows before it must not change.
    k2 = k.clone()
    v2 = v.clone()
    k2[:, -1] += 5.0
    v2[:, -1] *= -1.0
    o2 = _run_fp16_causal(q, k2, v2)

    assert torch.equal(o1[:, :-1], o2[:, :-1]), \
        "output rows before the perturbed position changed — causal mask broken"
    assert not torch.equal(o1[:, -1], o2[:, -1]), \
        "last row should depend on the last key/value"
