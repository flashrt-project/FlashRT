"""Device-only weight quantization must preserve the pre-reload contract."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _legacy_quantize(weight):
    scale = max(weight.float().abs().max().item() / 448.0, 1e-12)
    quantized = (weight.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, torch.tensor([scale], dtype=torch.float32, device=weight.device)


@pytest.mark.parametrize("peak", [0.0, 1e-15, 0.03125, 0.17, 0.3, 1.0, 2.125, 1000.0])
def test_device_quantization_matches_legacy_bytes(peak):
    from flash_rt.frontends.torch.pi05_rtx import _quantize_fp8_e4m3

    weight = torch.linspace(-peak, peak, 100002, device="cuda").bfloat16()
    # Include noncontiguous input, both signs, zero, and FP8 midpoint cases.
    weight = weight.reshape(2, -1).t()
    expected, expected_scale = _legacy_quantize(weight)
    actual, actual_scale = _quantize_fp8_e4m3(weight)
    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_device_quantization_graph_replay_updates_scale_and_bytes():
    from flash_rt.frontends.torch.pi05_rtx import _quantize_fp8_e4m3

    weight = torch.linspace(-0.3, 0.3, 100002, device="cuda").bfloat16()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _quantize_fp8_e4m3(weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, scale = _quantize_fp8_e4m3(weight)
    for peak in (0.3, 0.17, 0.0):
        weight.copy_(torch.linspace(-peak, peak, weight.numel(), device="cuda").bfloat16())
        graph.replay()
        expected, expected_scale = _legacy_quantize(weight)
        assert torch.equal(scale, expected_scale)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
