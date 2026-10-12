"""Does the FA2 quantized-output epilogue match a standalone quantize pass?

A consumer GEMM that wants an FP8 activation normally reads O twice: once as
the attention kernel's bf16 store, once as a separate quantize_fp8_static pass
that rewrites it as e4m3. The *_fp8out entries emit that e4m3 copy from the
registers that already feed the bf16 store, so the second pass disappears.

That is only a safe substitution if two things hold exactly, which is what this
file checks:

1. The bf16 output is untouched. The epilogue must not perturb the existing
   path, so O and softmax_lse have to stay bit-identical to the entry without
   the suffix -- not close, identical, because downstream quantization buckets
   turn a one-ULP drift into a different integer.

2. The FP8 copy is what quantize_fp8_static would have produced. Same reason:
   a consumer calibrated against that kernel must see the same bytes.

Both the non-split and the split-KV paths are covered, because the combine
kernel writes the final output through a different thread layout, and so are
shapes whose last block is only partly populated.

Skipped where FA2 or the FP8 quantizer is not built, since that is a target
property rather than a failure.
"""

import pytest

torch = pytest.importorskip("torch")

fa2 = pytest.importorskip("flash_rt.flash_rt_fa2")
fvk = pytest.importorskip("flash_rt.flash_rt_kernels")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

NQ, NKV, HD = 8, 4, 256
HQ = NQ * HD


def _strides(t):
    """(batch, row, head) element strides of a (B, S, H, D) tensor viewed B,H,S,D."""
    return (t.stride(0), t.stride(2), t.stride(1))


def _require(name):
    if not hasattr(fa2, name):
        pytest.skip(f"{name} not built in this FA2 matrix")
    return getattr(fa2, name)


def _run_pair(causal, seqlen_q, seqlen_k, split):
    """Return (reference, candidate) outputs for one shape."""
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn(1, seqlen_q, NQ, HD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, seqlen_k, NKV, HD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, seqlen_k, NKV, HD, device=dev, dtype=torch.bfloat16)
    q4, k4, v4 = (x.transpose(1, 2) for x in (q, k, v))

    scale = float(HD) ** -0.5
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    stream = torch.cuda.current_stream().cuda_stream

    lse_accum = torch.full((128, 1, NQ, seqlen_q), float("-inf"), device=dev)
    o_accum = torch.zeros(128, 1, NQ, seqlen_q, HD, device=dev, dtype=torch.float32)
    accum = (lse_accum.data_ptr(), o_accum.data_ptr()) if split else (0, 0)

    o_st = (seqlen_q * HQ, HQ, HD)
    shared = dict(
        batch=1, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
        num_heads_q=NQ, num_heads_kv=NKV, head_dim=HD,
        q_strides=_strides(q4), k_strides=_strides(k4),
        v_strides=_strides(v4), o_strides=o_st,
        softmax_scale=scale, num_sms=sms, stream=stream,
    )

    ref_o = torch.zeros(seqlen_q, HQ, device=dev, dtype=torch.bfloat16)
    ref_lse = torch.zeros(1, NQ, seqlen_q, device=dev, dtype=torch.float32)
    base = _require("fwd_bf16_causal" if causal else "fwd_bf16")
    base(q.data_ptr(), k.data_ptr(), v.data_ptr(), ref_o.data_ptr(), ref_lse.data_ptr(),
         accum[0], accum[1], **shared)

    # the scale a consumer would have calibrated: amax over the reference output
    a_scale = (ref_o.float().abs().amax() / 448.0).reshape(1).contiguous()
    ref_fp8 = torch.empty(seqlen_q, HQ, device=dev, dtype=torch.float8_e4m3fn)
    torch.cuda.synchronize()
    fvk.quantize_fp8_static(ref_o.data_ptr(), ref_fp8.data_ptr(), a_scale.data_ptr(),
                            seqlen_q * HQ, stream)

    got_o = torch.zeros_like(ref_o)
    got_lse = torch.zeros_like(ref_lse)
    got_fp8 = torch.empty_like(ref_fp8)
    lse_accum.fill_(float("-inf"))
    o_accum.zero_()
    cand = _require("fwd_bf16_causal_fp8out" if causal else "fwd_bf16_fp8out")
    cand(q.data_ptr(), k.data_ptr(), v.data_ptr(), got_o.data_ptr(), got_lse.data_ptr(),
         accum[0], accum[1],
         O_fp8=got_fp8.data_ptr(), o_fp8_scale=a_scale.data_ptr(),
         o_fp8_strides=o_st, skip_bf16_output=False, **shared)
    torch.cuda.synchronize()

    return (ref_o, ref_lse, ref_fp8), (got_o, got_lse, got_fp8)


SHAPES = [
    # (causal, seqlen_q, seqlen_k, split)
    (True, 41, 41, True),      # short block, split-KV combine
    (True, 7, 305, True),      # few queries against a long history
    (True, 256, 256, True),
    (True, 41, 41, False),     # same shape unsplit: single-pass epilogue
    (True, 1, 568, False),     # one query row, partly populated block
    (False, 41, 41, False),    # full attention
    (False, 129, 129, False),  # non-multiple of the block size
]


@pytest.mark.parametrize("causal,seqlen_q,seqlen_k,split", SHAPES)
def test_bf16_output_is_unchanged(causal, seqlen_q, seqlen_k, split):
    """The epilogue must not perturb the path it rides on."""
    (ref_o, ref_lse, _), (got_o, got_lse, _) = _run_pair(causal, seqlen_q, seqlen_k, split)
    assert torch.equal(got_o, ref_o), "bf16 O drifted from the entry without the epilogue"
    assert torch.equal(got_lse, ref_lse), "softmax_lse drifted"


@pytest.mark.parametrize("causal,seqlen_q,seqlen_k,split", SHAPES)
def test_fp8_copy_matches_quantize_fp8_static(causal, seqlen_q, seqlen_k, split):
    """The e4m3 copy must be the bytes a standalone quantize pass would write."""
    (_, _, ref_fp8), (_, _, got_fp8) = _run_pair(causal, seqlen_q, seqlen_k, split)
    assert torch.equal(got_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), (
        "FP8 copy differs from quantize_fp8_static over the same output"
    )


def test_null_fp8_pointer_is_the_plain_entry():
    """O_fp8 = 0 must leave the kernel exactly as the entry without the suffix."""
    dev = "cuda"
    torch.manual_seed(1)
    seqlen = 64
    q = torch.randn(1, seqlen, NQ, HD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, seqlen, NKV, HD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, seqlen, NKV, HD, device=dev, dtype=torch.bfloat16)
    q4, k4, v4 = (x.transpose(1, 2) for x in (q, k, v))
    o_st = (seqlen * HQ, HQ, HD)
    shared = dict(
        batch=1, seqlen_q=seqlen, seqlen_k=seqlen,
        num_heads_q=NQ, num_heads_kv=NKV, head_dim=HD,
        q_strides=_strides(q4), k_strides=_strides(k4),
        v_strides=_strides(v4), o_strides=o_st,
        softmax_scale=float(HD) ** -0.5,
        num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    lse = torch.zeros(1, NQ, seqlen, device=dev, dtype=torch.float32)

    ref = torch.zeros(seqlen, HQ, device=dev, dtype=torch.bfloat16)
    _require("fwd_bf16_causal")(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                ref.data_ptr(), lse.data_ptr(), 0, 0, **shared)
    got = torch.zeros_like(ref)
    _require("fwd_bf16_causal_fp8out")(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                       got.data_ptr(), lse.data_ptr(), 0, 0, **shared)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
