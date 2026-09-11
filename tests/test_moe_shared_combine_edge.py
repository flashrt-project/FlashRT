"""Equivalence test for the MoE shared-expert combine.

out = routed + shared * sigmoid(gate_logit[row])

The routed sum arrives in fp32 from the weighted reduction and the kernel keeps
it there, rounding once at the store. That is the reason this kernel exists
rather than the bf16 gate-mul-residual one already in the tree, so the
reference computes it the same way -- in fp32, cast once -- and the comparison
is exact where it can be.
"""

import pytest
import torch

HID = 2048


def _load_fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the MoE combine test")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:                # pragma: no cover - environmental
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    if not hasattr(fvk, "moe_shared_gate_combine_edge_bf16"):
        pytest.skip("moe_shared_gate_combine_edge_bf16 not in this build")
    return fvk


@pytest.mark.parametrize("S", [1, 17, 256, 2048])
def test_matches_the_fp32_chain(S):
    fvk = _load_fvk()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(23 + S)
    routed = torch.randn(S, HID, generator=g, device=dev,
                         dtype=torch.float32)
    shared = torch.randn(S, HID, generator=g, device=dev,
                         dtype=torch.bfloat16)
    glog = torch.randn(S, 1, generator=g, device=dev, dtype=torch.bfloat16)

    out = torch.empty(S, HID, dtype=torch.bfloat16, device=dev)
    fvk.moe_shared_gate_combine_edge_bf16(
        routed.data_ptr(), shared.data_ptr(), glog.data_ptr(),
        out.data_ptr(), S, HID, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(dev)

    sgate = torch.sigmoid(glog.float())
    ref = (routed + shared.float() * sgate).to(torch.bfloat16)

    # Judged on absolute and relative distance together, which is what this
    # output needs and what two simpler criteria each got wrong here:
    #
    #   * a relative norm over the whole tensor passes almost anything, since
    #     one bf16 ULP is already 4e-3 relative;
    #   * a ULP distance is meaningless where the two sides straddle zero. The
    #     kernel fuses the multiply into the add, so it rounds once where the
    #     tensor chain rounds twice, and on rows where `routed` and the gated
    #     shared expert nearly cancel the results are 9.3e-10 against exactly
    #     0 -- an enormous ULP distance for an absolute difference of nothing.
    #
    # So: an atol that treats a cancellation to zero as agreement, and an rtol
    # of two bf16 ULP for everything else. Both are far tighter than a wrong
    # gate, a wrong row index or a wrong dtype would produce.
    assert torch.allclose(out.float(), ref.float(), rtol=8e-3, atol=1e-6), (
        "combine drifted: max abs "
        f"{(out.float() - ref.float()).abs().max().item():.3e}")

    # And the bulk must be bit-identical, which is what catches a systematic
    # drift that stays inside the tolerance above.
    n_diff = int((out != ref).sum())
    assert n_diff <= out.numel() // 10000, \
        f"{n_diff} of {out.numel()} elements differ; expected a handful"


def test_reproducible():
    fvk = _load_fvk()
    dev = "cuda:0"
    S = 512
    g = torch.Generator(device=dev).manual_seed(29)
    routed = torch.randn(S, HID, generator=g, device=dev, dtype=torch.float32)
    shared = torch.randn(S, HID, generator=g, device=dev,
                         dtype=torch.bfloat16)
    glog = torch.randn(S, 1, generator=g, device=dev, dtype=torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream

    def run():
        out = torch.empty(S, HID, dtype=torch.bfloat16, device=dev)
        fvk.moe_shared_gate_combine_edge_bf16(
            routed.data_ptr(), shared.data_ptr(), glog.data_ptr(),
            out.data_ptr(), S, HID, st)
        torch.cuda.synchronize(dev)
        return out

    a = run()
    for _ in range(3):
        assert torch.equal(a, run())
