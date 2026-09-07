"""Host-constructor contracts, not model calibration or quality fixtures."""

import pytest
import torch

from flash_rt.structures.impls.graph_lowering.pi052_denoise import (
    Pi05DenoiseGraphLoweringAdapter,
)


class FlowHost(torch.nn.Module):
    paligemma_with_expert = None

    def embed_suffix(self):
        pass

    def denoise_step(self):
        pass

    def sample_actions(self, data, **kwargs):
        return torch.tensor(data, **kwargs)


@pytest.mark.parametrize("data", [0.3, 2, True, [0.3, 0.2]])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_constructor_value_shape_dtype_and_restore(data, dtype):
    host = FlowHost()
    original = torch.tensor
    expected = host.sample_actions(data, device="cpu", dtype=dtype)
    lowering = Pi05DenoiseGraphLoweringAdapter().lower(host, lambda: None)
    try:
        actual = host.sample_actions(data, device="cpu", dtype=dtype)
        assert torch.equal(actual, expected)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.tensor is original
    finally:
        lowering.undo()
    assert "sample_actions" not in host.__dict__
    assert torch.equal(host.sample_actions(data, device="cpu", dtype=dtype), expected)


def test_scalar_is_fresh_and_unrecognized_host_is_unchanged():
    original = torch.tensor
    adapter = Pi05DenoiseGraphLoweringAdapter()
    assert adapter.lower(torch.nn.Identity(), lambda: None) is None
    host = FlowHost()
    lowering = adapter.lower(host, lambda: None)
    try:
        host.sample_actions(0.5, device="cpu").add_(1)
        assert host.sample_actions(0.5, device="cpu").item() == 0.5
        with pytest.raises(RuntimeError, match="Could not infer dtype"):
            host.sample_actions(object(), device="cpu")
        assert torch.tensor is original
    finally:
        lowering.undo()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture gate")
@pytest.mark.parametrize("compiled", [False, True])
def test_scalar_and_list_capture_repeated_inputs(compiled):
    host = FlowHost()
    lowering = Pi05DenoiseGraphLoweringAdapter().lower(host, lambda: None)
    inputs = torch.ones(3, device="cuda")

    def hot():
        scalar = host.sample_actions(0.25, device=inputs.device, dtype=inputs.dtype)
        schedule = host.sample_actions([0.5, 0.25], device=inputs.device,
                                       dtype=inputs.dtype)
        return inputs * scalar + schedule.sum()

    expected = hot()
    run = torch.compile(hot) if compiled else hot
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
        for value in (1, 3, 1):
            with torch.cuda.stream(stream):
                inputs.fill_(value)
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output, expected + (value - 1) * 0.25,
                                       atol=0, rtol=0)
    finally:
        lowering.undo()
        torch._dynamo.reset()
