"""The head_dim-64 tile entries opt in to the ragged-tail skip.

The skip is the compile-time ``SkipDeadTail`` flag on the vendored FA2
kernel: the X-VLA tile instantiations pass ``true``, while every existing
entry uses the default ``false`` and keeps the unmodified upstream MMA
sequence. This checks that the two paths agree on the production ragged
shapes, so enabling the skip does not change the attention result.
"""

import pytest

torch = pytest.importorskip("torch")
fa2 = pytest.importorskip("flash_rt.flash_rt_fa2")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

NQ, NKV, HD = 16, 16, 64
HQ = NQ * HD


def _strides(t):
    """(batch, row, head) element strides of a (B, S, H, D) tensor viewed B,H,S,D."""
    return (t.stride(0), t.stride(2), t.stride(1))


def _require(name):
    if not hasattr(fa2, name):
        pytest.skip(f"{name} not built in this FA2 matrix")
    return getattr(fa2, name)


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(262, 262), (262, 100), (30, 262)])
def test_tile64_skip_matches_upstream_entry(seqlen_q, seqlen_k):
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, seqlen_q, NQ, HD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, seqlen_k, NKV, HD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, seqlen_k, NKV, HD, device=dev, dtype=torch.bfloat16)
    q4, k4, v4 = (x.transpose(1, 2) for x in (q, k, v))

    scale = float(HD) ** -0.5
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    stream = torch.cuda.current_stream().cuda_stream
    o_st = (seqlen_q * HQ, HQ, HD)
    shared = dict(
        batch=1, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
        num_heads_q=NQ, num_heads_kv=NKV, head_dim=HD,
        q_strides=_strides(q4), k_strides=_strides(k4),
        v_strides=_strides(v4), o_strides=o_st,
        softmax_scale=scale, num_sms=sms, stream=stream,
    )

    # The existing entry compiles SkipDeadTail=false.
    ref_o = torch.zeros(seqlen_q, HQ, device=dev, dtype=torch.bfloat16)
    ref_lse = torch.zeros(1, NQ, seqlen_q, device=dev, dtype=torch.float32)
    _require("fwd_bf16")(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                         ref_o.data_ptr(), ref_lse.data_ptr(), 0, 0, **shared)

    # The X-VLA tile compiles SkipDeadTail=true.
    got_o = torch.zeros_like(ref_o)
    got_lse = torch.zeros_like(ref_lse)
    _require("fwd_bf16_tile64")(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                got_o.data_ptr(), got_lse.data_ptr(), 0,
                                **shared)
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(
        ref_o.float().flatten(), got_o.float().flatten(), dim=0)
    assert float(cos) >= 0.999, f"cosine {float(cos)} < 0.999"
    assert torch.isfinite(got_o.float()).all()
