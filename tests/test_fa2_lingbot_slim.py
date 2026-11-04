"""Smoke test for the opt-in LingBot slim FA2 module.

The slim module is built only with ``-DFLASHRT_FA2_LINGBOT_SLIM=ON``.  The
default build ships the generic module, so this file skips there; the numerical
check also skips when no CUDA device is available.  It exercises the two public
entries the LingBot native transformer calls and compares them with PyTorch
SDPA on small shapes.
"""

from __future__ import annotations

import math

import pytest

REQUIRED = {"fwd_bf16", "fwd_bf16_tile"}
GENERIC_ONLY = {
    "fwd_bf16_seqused",
    "fwd_bf16_window",
    "fwd_bf16_causal",
    "fwd_fp16",
    "fwd_fp16_causal",
    "has_xvla_tile64",
}

fa2 = pytest.importorskip("flash_rt.flash_rt_fa2")
_NAMES = {name for name in dir(fa2) if not name.startswith("_")}

pytestmark = pytest.mark.skipif(
    bool(_NAMES & GENERIC_ONLY) or not REQUIRED <= _NAMES,
    reason="generic FA2 module (slim build not selected)",
)

torch = pytest.importorskip("torch")


def test_slim_surface_has_only_the_two_entries():
    assert REQUIRED <= _NAMES
    assert not (_NAMES & GENERIC_ONLY), _NAMES & GENERIC_ONLY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seqlen_q,entry", [(128, "fwd_bf16"), (16, "fwd_bf16_tile")])
def test_forward_matches_sdpa(seqlen_q: int, entry: str):
    device = torch.device("cuda")
    batch, heads, seqlen_k, head_dim = 1, 2, 256, 128
    q = torch.randn(batch, seqlen_q, heads, head_dim,
                    device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen_k, heads, head_dim,
                    device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen_k, heads, head_dim,
                    device=device, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    lse = torch.empty(batch, heads, seqlen_q, device=device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_dim)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    stream = torch.cuda.current_stream().cuda_stream
    ptrs = (q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
            lse.data_ptr())

    if entry == "fwd_bf16":
        fa2.fwd_bf16(*ptrs, 0, 0, batch, seqlen_q, seqlen_k, heads, heads,
                     head_dim, q.stride(), k.stride(), v.stride(),
                     out.stride(), scale, num_sms, stream)
    else:
        fa2.fwd_bf16_tile(*ptrs, 0, batch, seqlen_q, seqlen_k, heads, heads,
                          head_dim, q.stride(), k.stride(), v.stride(),
                          out.stride(), scale, num_sms, stream)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float(), scale=scale, is_causal=False,
    ).transpose(1, 2).to(torch.bfloat16)
    assert torch.isfinite(out).all()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    assert cos > 0.999, (entry, cos)
