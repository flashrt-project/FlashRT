from __future__ import annotations

import torch
from torch import nn

from flash_rt.structures.autobuild import AutoPlan
from flash_rt.structures.frontdoor import _Arm
from flash_rt.structures.guard import CAST_OK, PROCEED, GuardedSeam


class RoutedCore(GuardedSeam, nn.Module):
    _frt_can_fallback = False

    def __init__(self):
        super().__init__()
        self._frt_arm(dtypes=CAST_OK, device=torch.device("cpu"), k=4)

    def forward(self, value):
        admitted = self._frt_admit(value)
        if admitted is not PROCEED:
            return admitted
        return value


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Identity()


def _plan(state, core):
    def enable():
        state["routed"] = True

    def disable():
        state["routed"] = False

    return AutoPlan(
        observed={"attention.processor::core": core},
        toggles=[(enable, disable)],
    )


def test_regular_gate_arm_does_not_enable_routed_seams():
    host = Host().eval()
    state = {"routed": False}
    plan = _plan(state, RoutedCore())
    arm = _Arm(
        host, {"proj": nn.Identity()}, plan, "raise", routed=False)

    arm.on()
    assert not state["routed"]
    arm.off()
    assert not state["routed"]


def test_routed_gate_arm_can_observe_without_module_swaps():
    host = Host().eval()
    state = {"routed": False}
    core = RoutedCore()
    plan = _plan(state, core)
    arm = _Arm(host, {}, plan, "raise", routed=True)

    arm.on()
    assert state["routed"]
    assert torch.equal(core(torch.ones(2, 4)), torch.ones(2, 4))
    summary = arm.handle.summary()
    assert summary["guarded_calls"] == 1
    assert summary["fallbacks"] == 0
    assert summary["clean"]
    assert summary["seams_fell_back"] == []
    arm.off()
    assert not state["routed"]
