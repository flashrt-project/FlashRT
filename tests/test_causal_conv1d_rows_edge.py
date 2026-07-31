"""Equivalence test for the row-blocked causal conv1d.

The row-blocked entry computes the same convolution as the per-token one, and
writes the silu the same way rather than merely equivalently, so this compares
them exactly: any difference is a bug, not a reduction order. Both are checked
against a direct torch reference as well, so a shared misreading of the layout
or the causal edge does not pass by agreeing with itself.

Lengths that are not multiples of the eight rows a thread walks are included,
since that tail is where a row-blocked kernel goes wrong.
"""

import pytest
import torch
import torch.nn.functional as F

CONV, K = 8192, 4


def _load_fvk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the conv1d test")
    try:
        from flash_rt import flash_rt_kernels as fvk
    except Exception as exc:                # pragma: no cover - environmental
        pytest.skip(f"flash_rt_kernels is not built: {exc}")
    if not hasattr(fvk, "causal_conv1d_qwen36_rows_bf16"):
        pytest.skip("causal_conv1d_qwen36_rows_bf16 not in this build")
    return fvk


def _torch_ref(x, w, k, silu=True):
    """y[s, c] = sum_i x[s + i - (k-1), c] * w[c, i], zero before the start."""
    S, C = x.shape[1], x.shape[2]
    xt = x[0].t().unsqueeze(0)                       # (1, C, S)
    xp = F.pad(xt.float(), (k - 1, 0))
    y = F.conv1d(xp, w.float().unsqueeze(1), groups=C)
    y = y[0].t().unsqueeze(0)                        # (1, S, C)
    if silu:
        y = y / (1.0 + torch.exp(-y))
    return y.to(torch.bfloat16)


@pytest.mark.parametrize("S", [1, 7, 64, 129, 512, 2048, 4097])
def test_rows_matches_per_token_entry_exactly(S):
    fvk = _load_fvk()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(S)
    x = torch.randn(1, S, CONV, generator=g, device=dev, dtype=torch.bfloat16)
    w = torch.randn(CONV, K, generator=g, device=dev, dtype=torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream

    a = torch.empty(1, S, CONV, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    fvk.causal_conv1d_qwen36_bf16(
        x.data_ptr(), w.data_ptr(), 0, a.data_ptr(), 1, S, CONV, K, True, st)
    fvk.causal_conv1d_qwen36_rows_bf16(
        x.data_ptr(), w.data_ptr(), 0, b.data_ptr(), 1, S, CONV, K, True, st)
    torch.cuda.synchronize(dev)

    assert torch.equal(a, b), (
        f"{int((a != b).sum())} of {a.numel()} elements differ from the "
        "per-token entry")

    # And both against a reference that shares none of their code, so a common
    # misreading of the causal edge cannot pass.
    ref = _torch_ref(x, w, K)
    rel = ((b.float() - ref.float()).norm()
           / ref.float().norm().clamp_min(1e-9)).item()
    assert rel < 1e-2, f"both kernels disagree with the reference by {rel:.3e}"


@pytest.mark.parametrize("S", [63, 2048])
def test_no_silu_variant_matches(S):
    fvk = _load_fvk()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(S + 1)
    x = torch.randn(1, S, CONV, generator=g, device=dev, dtype=torch.bfloat16)
    w = torch.randn(CONV, K, generator=g, device=dev, dtype=torch.bfloat16)
    st = torch.cuda.current_stream().cuda_stream

    a = torch.empty(1, S, CONV, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    fvk.causal_conv1d_qwen36_bf16(
        x.data_ptr(), w.data_ptr(), 0, a.data_ptr(), 1, S, CONV, K, False, st)
    fvk.causal_conv1d_qwen36_rows_bf16(
        x.data_ptr(), w.data_ptr(), 0, b.data_ptr(), 1, S, CONV, K, False, st)
    torch.cuda.synchronize(dev)
    assert torch.equal(a, b)


def test_beyond_the_old_grid_ceiling():
    """The per-token entry launches one grid row per token, so a single call
    stopped at 65535. The row-blocked grid divides the sequence, so this shape
    is reachable at all -- which is the point, not the speed."""
    fvk = _load_fvk()
    dev = "cuda:0"
    S, C = 70000, 512                     # narrow, so the buffer stays small
    g = torch.Generator(device=dev).manual_seed(2)
    x = torch.randn(1, S, C, generator=g, device=dev, dtype=torch.bfloat16)
    w = torch.randn(C, K, generator=g, device=dev, dtype=torch.bfloat16)
    out = torch.empty(1, S, C, dtype=torch.bfloat16, device=dev)
    fvk.causal_conv1d_qwen36_rows_bf16(
        x.data_ptr(), w.data_ptr(), 0, out.data_ptr(), 1, S, C, K, True,
        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize(dev)
    assert torch.isfinite(out.float()).all()
    ref = _torch_ref(x, w, K)
    rel = ((out.float() - ref.float()).norm()
           / ref.float().norm().clamp_min(1e-9)).item()
    assert rel < 1e-2, f"past the old ceiling the result drifted {rel:.3e}"
