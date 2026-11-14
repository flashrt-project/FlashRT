import math

import pytest
import torch


def _sampling_kernel():
    if torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires NVIDIA CUDA")
    try:
        from flash_rt import flash_rt_kernels as kernels
    except ImportError:
        pytest.skip("flash_rt_kernels is not built")
    if not hasattr(kernels, "delayed_codebook_sample_embed_bf16"):
        pytest.skip("audio sampling kernel is not built")
    return kernels


def _sample(kernels, logits, codebook, codes, embed, *, delay, seed, step):
    kernels.delayed_codebook_sample_embed_bf16(
        logits.data_ptr(), codebook.data_ptr(), codes.data_ptr(),
        embed.data_ptr(), logits.shape[0], logits.shape[1],
        codebook.shape[1], delay, 1024, 1.0, seed, step,
        torch.cuda.current_stream().cuda_stream,
    )


def test_sampling_kernel_seed_step_and_delay_mask():
    kernels = _sampling_kernel()
    nc, vocab, hidden = 8, 1026, 1
    logits = torch.zeros(nc, vocab, device="cuda", dtype=torch.bfloat16)
    codebook = torch.zeros(
        nc * vocab, hidden, device="cuda", dtype=torch.bfloat16)
    first = torch.empty(nc, device="cuda", dtype=torch.long)
    second = torch.empty_like(first)
    other_step = torch.empty_like(first)
    embed = torch.empty(1, hidden, device="cuda", dtype=torch.bfloat16)

    _sample(kernels, logits, codebook, first, embed,
            delay=nc, seed=1234, step=7)
    _sample(kernels, logits, codebook, second, embed,
            delay=nc, seed=1234, step=7)
    _sample(kernels, logits, codebook, other_step, embed,
            delay=nc, seed=1234, step=8)
    torch.cuda.synchronize()

    assert torch.equal(first, second)
    assert not torch.equal(first, other_step)
    assert bool(((first >= 0) & (first < vocab)).all())

    delayed = torch.empty_like(first)
    _sample(kernels, logits, codebook, delayed, embed,
            delay=0, seed=1234, step=0)
    torch.cuda.synchronize()
    assert bool((delayed[1:] == 1024).all())


def test_sampling_kernel_matches_categorical_frequency():
    kernels = _sampling_kernel()
    draws = 4096
    logits = torch.tensor(
        [[math.log(3.0), 0.0]], device="cuda", dtype=torch.bfloat16)
    codebook = torch.zeros(2, 1, device="cuda", dtype=torch.bfloat16)
    codes = torch.empty(draws, device="cuda", dtype=torch.long)
    embeds = torch.empty(draws, 1, device="cuda", dtype=torch.bfloat16)

    for step in range(draws):
        _sample(kernels, logits, codebook, codes[step:], embeds[step:],
                delay=1, seed=0xC0FFEE, step=step)
    torch.cuda.synchronize()

    p_zero = float((codes == 0).float().mean())
    assert 0.72 < p_zero < 0.78
    assert set(codes.cpu().tolist()) == {0, 1}


def test_sampling_cdf_endpoint_uses_last_positive_probability():
    kernels = _sampling_kernel()
    # This seed makes the native FP32 uniform round to exactly 1.0. The
    # finite negative logits underflow to zero without any vocabulary filter.
    logits = torch.tensor([[0., 0., -200., -200.]], device="cuda", dtype=torch.bfloat16)
    codebook = torch.arange(4, device="cuda", dtype=torch.bfloat16).reshape(4, 1)
    codes = torch.empty(1, device="cuda", dtype=torch.long)
    embed = torch.empty(1, 1, device="cuda", dtype=torch.bfloat16)
    _sample(kernels, logits, codebook, codes, embed,
            delay=1, seed=3558559446808474027, step=0)
    assert codes.item() == 1  # The old fallback incorrectly selected index 3.
    assert embed.item() == 1
    kernels.delayed_codebook_argmax_embed_bf16(
        logits.data_ptr(), codebook.data_ptr(), codes.data_ptr(), embed.data_ptr(),
        1, 4, 1, 1, 1024, torch.cuda.current_stream().cuda_stream)
    assert codes.item() == 0  # Greedy still selects the first tied maximum.
