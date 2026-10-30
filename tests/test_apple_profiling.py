"""Profiling must pair ranges on failure and never enable per-op GPU waits."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from flash_rt.hardware.apple_native.profiling import StageProfiler, profile_pipeline


def test_stage_range_closes_on_error():
    events = []

    class Signposts:
        @contextmanager
        def interval(self, name):
            events.append(("begin", name))
            try:
                yield
            finally:
                events.append(("end", name))

    profiler = StageProfiler(signposts=Signposts(), synchronize=lambda: events.append("sync"))
    with pytest.raises(ValueError):
        with profiler.range("vision.blocks"):
            raise ValueError("inference failed")
    assert events == ["sync", ("begin", "vision.blocks"), "sync", ("end", "vision.blocks")]
    assert profiler.samples_ms["vision.blocks"][0] >= 0


def test_trace_cleanup_and_no_per_op_wait(monkeypatch):
    events = []
    pipeline = SimpleNamespace(_stage_profiler=None)

    @contextmanager
    def trace(**kwargs):
        events.append(kwargs)
        try:
            yield
        finally:
            events.append("stop")

    monkeypatch.setattr(torch.mps.profiler, "profile", trace)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("drain"))
    with pytest.raises(ValueError):
        with profile_pipeline(pipeline, signposts=False) as profiler:
            assert pipeline._stage_profiler is profiler
            with pytest.raises(RuntimeError, match="already active"):
                with profile_pipeline(pipeline, signposts=False):
                    pass
            with profiler.range("vision.patch"):
                raise ValueError("inference failed")
    assert pipeline._stage_profiler is None
    assert events == [{"mode": "interval,event", "wait_until_completed": False}, "drain", "stop"]
    assert profiler.samples_ms == {}  # Host submission intervals are not GPU times.


def test_inactive_profiling_does_not_synchronize(monkeypatch):
    def fail():
        raise AssertionError("unexpected GPU synchronization")
    monkeypatch.setattr(torch.mps, "synchronize", fail)
    with profile_pipeline(SimpleNamespace(_stage_profiler=None), signposts=False, mps_ops=False) as profiler:
        with profiler.range("vision.blocks"):
            pass
    assert profiler.samples_ms == {}
