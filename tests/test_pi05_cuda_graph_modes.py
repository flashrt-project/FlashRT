"""CPU-only contract tests for Pi0.5 CUDA-graph and stream routing."""

import ctypes
from types import SimpleNamespace

import pytest

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline
from flash_rt.models.pi05.pipeline_rtx_cfg_batched import Pi05CFGBatchedPipeline


class _CudaRuntimeSpy:
    def __init__(self):
        self.synchronized = []

    def cudaStreamSynchronize(self, stream):
        self.synchronized.append(
            stream.value if isinstance(stream, ctypes.c_void_p) else stream)


class _GraphSpy:
    def __init__(self):
        self.replayed = []

    def replay(self, stream):
        self.replayed.append(stream)


def _pipeline(pipeline_cls, output_name):
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline._graph = None
    pipeline._cudart = _CudaRuntimeSpy()
    pipeline.bufs = {output_name: SimpleNamespace(ptr=SimpleNamespace(value=1234))}
    pipeline.run_streams = []
    pipeline.run_pipeline = lambda *, stream: pipeline.run_streams.append(stream)
    if isinstance(pipeline, Pi05CFGBatchedPipeline):
        pipeline.COND_SLOT = 0
        pipeline.chunk_size = 50
    return pipeline


@pytest.mark.parametrize(
    ("pipeline_cls", "output_name", "expected_ptr"),
    [
        (Pi05Pipeline, "diffusion_noise", 1234),
        (Pi05BatchedPipeline, "diffusion_noise_b2", 1234),
        (Pi05CFGBatchedPipeline, "diffusion_noise_b2", 1234),
    ],
)
def test_no_graph_full_forward_uses_requested_stream(
    pipeline_cls, output_name, expected_ptr,
):
    pipeline = _pipeline(pipeline_cls, output_name)

    result = pipeline.forward(stream=73)

    assert pipeline.run_streams == [73]
    assert pipeline._cudart.synchronized == [73]
    assert result == expected_ptr


def test_cfg_batched_no_graph_returns_conditioned_slot():
    pipeline = _pipeline(Pi05CFGBatchedPipeline, "diffusion_noise_b2")
    pipeline.COND_SLOT = 1
    pipeline.chunk_size = 50

    result = pipeline.forward(stream=91)

    assert result == 1234 + 50 * 32 * 2


@pytest.mark.parametrize(
    ("pipeline_cls", "output_name"),
    [
        (Pi05Pipeline, "diffusion_noise"),
        (Pi05BatchedPipeline, "diffusion_noise_b2"),
        (Pi05CFGBatchedPipeline, "diffusion_noise_b2"),
    ],
)
def test_graph_forward_still_replays_captured_stream(pipeline_cls, output_name):
    pipeline = _pipeline(pipeline_cls, output_name)
    pipeline._graph = _GraphSpy()
    pipeline._graph_stream = 17

    pipeline.forward(stream=73)

    assert pipeline._graph.replayed == [17]
    assert pipeline.run_streams == []
    assert pipeline._cudart.synchronized == [17]


def test_no_graph_decode_only_uses_requested_stream():
    pipeline = _pipeline(Pi05Pipeline, "diffusion_noise")
    pipeline.transformer_streams = []
    pipeline.transformer_decoder = (
        lambda *, stream: pipeline.transformer_streams.append(stream))

    result = pipeline.forward_decode_only(stream=29)

    assert pipeline.transformer_streams == [29]
    assert pipeline._cudart.synchronized == [29]
    assert result == 1234


def test_disabled_graph_mode_does_not_record(monkeypatch):
    frontend = Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
    frontend.use_cuda_graph = False
    frontend.pipeline = SimpleNamespace(record_infer_graph=lambda **_: pytest.fail(
        "record_infer_graph must not run when CUDA graphs are disabled"))
    monkeypatch.setattr(
        "flash_rt.subgraphs.capture.apply_frontend_capture_hooks",
        lambda _: pytest.fail("capture hooks must not run"))

    frontend._record_infer_graph_if_enabled(41)


def test_default_graph_mode_records_on_requested_stream(monkeypatch):
    frontend = Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
    frontend.use_cuda_graph = True
    recorded = []
    frontend.pipeline = SimpleNamespace(
        record_infer_graph=lambda **kwargs: recorded.append(kwargs))
    hooked = []
    monkeypatch.setattr(
        "flash_rt.subgraphs.capture.apply_frontend_capture_hooks",
        lambda value: hooked.append(value))

    frontend._record_infer_graph_if_enabled(41)

    assert hooked == [frontend]
    assert recorded == [{"external_stream_int": 41}]


@pytest.mark.parametrize(
    ("cache_frames", "expected"),
    [
        (1, [True, True, True, True]),
        (2, [True, False, True, False]),
    ],
)
def test_cache_frame_schedule_alternates_full_and_decode_only(
    cache_frames, expected,
):
    frontend = Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
    frontend._cache_frames = cache_frames
    frontend._frame_count = 0

    actual = [frontend._use_full_pipeline_for_next_frame() for _ in expected]

    assert actual == expected
