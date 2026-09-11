"""Equivalence test for the M-row W4A16 GEMV.

This kernel exists so a speculative verify computes the same function as the
decode step it verifies, so the bar is exactness, not closeness: row m of its
output must equal what the M=1 GEMV produces for row m on its own, bit for bit.
A verify that is merely close keeps tokens plain greedy would not have emitted.

The M=1 case is checked as well, since it is the claim that the extension left
the original arithmetic alone.
"""

import pytest
import torch

# The dense projections a decode step runs, so the shapes are the real ones.
SHAPES = [
    (8192, 2048),      # in_proj_qkv, q_proj
    (4096, 2048),      # in_proj_z
    (2048, 4096),      # gdn out_proj, o_proj
    (512, 2048),       # shared gate/up
    (2048, 512),       # shared down
    (256, 2048),       # router-sized, below one warp's row block
]


def _load_fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the M-row W4A16 test")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:                # pragma: no cover - environmental
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    for name in ("w4a16_mrows_edge_sm120_bf16",
                 "w4a16_matvec_edge_sm120_bf16",
                 "bf16_weight_to_nvfp4_swizzled"):
        if not hasattr(fvk, name):
            pytest.skip(f"{name} not in this build")
    return fvk


def _quantise(fvk, w, N, K, dev):
    from flash_rt.frontends.torch._nexn2_rtx_nvfp4_weights import _sf_swz_bytes
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    sf = torch.zeros(_sf_swz_bytes(N, K), dtype=torch.uint8, device=dev)
    scr = torch.zeros(1, dtype=torch.float32, device=dev)
    og = torch.zeros(1, dtype=torch.float32, device=dev)
    fvk.bf16_weight_to_nvfp4_swizzled(
        w.contiguous().data_ptr(), packed.data_ptr(), sf.data_ptr(),
        scr.data_ptr(), og.data_ptr(), N, K,
        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(dev)
    return packed, sf, float(og.item())


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [1, 2, 3, 4, 8])
def test_rows_match_the_per_row_gemv_exactly(N, K, M):
    fvk = _load_fvk()
    dev = "cuda:0"
    st = torch.cuda.current_stream().cuda_stream
    g = torch.Generator(device=dev).manual_seed(N + K + M)
    w = torch.randn(N, K, generator=g, device=dev, dtype=torch.bfloat16)
    packed, sf, alpha = _quantise(fvk, w, N, K, dev)
    x = torch.randn(M, K, generator=g, device=dev, dtype=torch.bfloat16)

    ref = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    for m in range(M):
        y1 = torch.empty(1, N, dtype=torch.bfloat16, device=dev)
        fvk.w4a16_matvec_edge_sm120_bf16(
            x[m:m + 1].contiguous().data_ptr(), packed.data_ptr(),
            sf.data_ptr(), y1.data_ptr(), N, K, alpha, st)
        torch.cuda.synchronize(dev)
        ref[m].copy_(y1[0])

    got = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    rc = fvk.w4a16_mrows_edge_sm120_bf16(
        x.contiguous().data_ptr(), packed.data_ptr(), sf.data_ptr(),
        got.data_ptr(), M, N, K, alpha, st)
    assert rc == 0, f"kernel returned {rc}"
    torch.cuda.synchronize(dev)

    assert torch.equal(got, ref), (
        f"{int((got != ref).sum())} of {got.numel()} elements differ from the "
        "per-row GEMV")


def test_rejects_an_M_it_cannot_hold():
    """A window wider than the kernel stages should be refused, not truncated."""
    fvk = _load_fvk()
    dev = "cuda:0"
    N, K, M = 512, 2048, 9
    g = torch.Generator(device=dev).manual_seed(1)
    w = torch.randn(N, K, generator=g, device=dev, dtype=torch.bfloat16)
    packed, sf, alpha = _quantise(fvk, w, N, K, dev)
    x = torch.randn(M, K, generator=g, device=dev, dtype=torch.bfloat16)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    rc = fvk.w4a16_mrows_edge_sm120_bf16(
        x.data_ptr(), packed.data_ptr(), sf.data_ptr(), out.data_ptr(),
        M, N, K, alpha, torch.cuda.current_stream().cuda_stream)
    assert rc != 0, "an over-wide window was accepted"
