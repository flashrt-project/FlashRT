"""The whole-graph lowering registry: recognition, atomicity, undo.

Capture consults this registry so that making a host graph-safe is the
library's job, per host family — the alternative was every user hand
writing the same hundred-odd lines of pins, which is the point at which
the automatic tier stops being automatic.
"""

from __future__ import annotations

import pytest
import torch

from flash_rt.structures.impls.graph_lowering import protocol


def test_unrecognized_host_is_captured_as_is():
    # no family recognizing a model is not a fallback: the host may
    # simply be graph-safe already
    model = torch.nn.Linear(4, 4)
    from flash_rt.structures.impls.graph_lowering.qwen3_vl import (
        Qwen3VLGraphLoweringAdapter)

    assert Qwen3VLGraphLoweringAdapter().lower(model, lambda: None) is None


def test_a_failing_adapter_unpins_everything_before_it(monkeypatch):
    applied, undone = [], []

    class Good:
        def lower(self, model, forward):
            applied.append("good")
            return protocol.GraphLowering(
                undo=lambda: undone.append("good"),
                family="good", pins=("x",))

    class Bad:
        def lower(self, model, forward):
            raise protocol.GraphLoweringRefused("cannot pin safely")

    monkeypatch.setattr(protocol, "_ADAPTERS", [Good(), Bad()])
    with pytest.raises(protocol.GraphLoweringRefused):
        protocol.lower_for_capture(object(), lambda: None)
    # the earlier family's pins came back off — no half-pinned host
    assert undone == ["good"]


def test_capture_records_the_family_and_stage_can_restore(monkeypatch):
    # the certification names what was pinned, and the stage carries
    # the undo so an A/B harness can put the host back
    calls = []
    lowering = protocol.GraphLowering(
        undo=lambda: calls.append("undo"), family="fake", pins=("a", "b"),
        details={"tokens": 7})

    from flash_rt.structures.stages import CapturedStage

    stage = CapturedStage(graph=None, stream=None, output=None,
                          windows={}, lowerings=(lowering,))
    stage.restore_host()
    assert calls == ["undo"]
    assert stage.lowerings == ()
