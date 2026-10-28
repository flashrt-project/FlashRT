"""Dedicated LaWAM FA2 calls must not change legacy calls in the same process."""
import pytest

torch = pytest.importorskip("torch")

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal,hd", [(False,64),(True,128)])
def test_lawam_attention_is_scoped_and_matches_reference(dtype, causal, hd):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        from flash_rt import flash_rt_fa2 as fa2
    except ImportError:
        pytest.skip("FA2 not built")
    name = "fwd_" + ("fp16" if dtype == torch.float16 else "bf16") + ("_causal" if causal else "")
    if not hasattr(fa2, "lawam_" + name):
        pytest.skip("LaWAM FA2 entries not built")
    torch.manual_seed(19)
    seq, heads = 219, 4
    q, k, v = [torch.randn(1, seq, heads, hd, device="cuda", dtype=dtype) for _ in range(3)]
    out = torch.empty_like(q)
    lse = torch.empty(1, heads, seq, device="cuda")
    strides = (seq*heads*hd, heads*hd, hd)
    kw = dict(Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=out.data_ptr(),
              softmax_lse=lse.data_ptr(), softmax_lse_accum=0, o_accum=0, batch=1,
              seqlen_q=seq, seqlen_k=seq, num_heads_q=heads, num_heads_kv=heads,
              head_dim=hd, q_strides=strides, k_strides=strides, v_strides=strides,
              o_strides=strides, softmax_scale=hd**-0.5, num_sms=0, stream=0)
    legacy, lawam = getattr(fa2, name), getattr(fa2, "lawam_" + name)
    legacy(**kw)
    before = out.clone()
    lawam(**kw)
    tuned = out.clone()
    legacy(**kw)
    torch.testing.assert_close(out, before, rtol=0, atol=0)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2).float(), k.transpose(1,2).float(), v.transpose(1,2).float(),
        is_causal=causal).transpose(1,2)
    for actual in (before, tuned):
        torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=0.01)
    if dtype == torch.bfloat16 and not causal:
        quantized = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.01], device="cuda")
        fa2.lawam_fwd_bf16_fp8out(**kw, O_fp8=quantized.data_ptr(),
                                o_fp8_scale=scale.data_ptr(), o_fp8_strides=strides,
                                skip_bf16_output=False)
        torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.01)
        torch.testing.assert_close(quantized.float() * scale, reference, rtol=0.1, atol=0.015)
