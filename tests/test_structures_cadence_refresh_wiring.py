"""The cross-attention bank refresh wired into its producer's forward.

The manual refresh contract leaves the banks outside the hot path; a
whole-pipeline capture then records an encoder whose output feeds
nothing. These tests pin the wired form: the producer is identified by
tensor identity on one probe forward, and after wiring the assembled
model tracks a new observation with no manual refresh call.
"""

import pytest
import torch

from flash_rt.structures.impls.cadence_static.cross_attention import (
    bind_cross_attention_kv,
    capture_cross_attention_kv,
    discover_cross_attention_kv,
    wire_refresh_to_producer,
)


class Cross(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8)
        self.to_k = torch.nn.Linear(4, 8)
        self.to_v = torch.nn.Linear(4, 8)

    def forward(self, x, enc):
        return self.to_q(x) + self.to_k(enc) + self.to_v(enc)


class Host(torch.nn.Module):
    """An encoder on the slow cadence feeding a three-step fast loop."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.cross = Cross()

    def forward(self, x, obs):
        enc = self.encoder(obs)
        out = x
        for _ in range(3):
            out = self.cross(out, enc)
        return out


def assemble(model, forward):
    sites = discover_cross_attention_kv(model)
    assert {c.path for c in sites} == {"cross.to_k", "cross.to_v"}
    caps = capture_cross_attention_kv(sites, forward)
    swaps, statics = bind_cross_attention_kv(sites, caps)
    for path, static in swaps.items():
        parent = model.get_submodule(path.rsplit(".", 1)[0])
        setattr(parent, path.rsplit(".", 1)[1], static)
    return statics


def test_wired_refresh_tracks_a_new_observation():
    torch.manual_seed(0)
    model = Host().eval()
    reference = Host().eval()
    reference.load_state_dict(model.state_dict())
    x = torch.randn(2, 8)
    obs = torch.randn(2, 4)

    statics = assemble(model, lambda: model(x, obs))
    producer, handle = wire_refresh_to_producer(
        model, statics, lambda: model(x, obs))
    assert producer is model.encoder

    obs2 = torch.randn(2, 4)
    with torch.no_grad():
        got = model(x, obs2)
        want = reference(x, obs2)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    assert all(s._frt_guard.notes["refreshes"] >= 1 for s in statics)
    handle.remove()


def test_unwired_statics_hold_the_calibration_observation():
    torch.manual_seed(1)
    model = Host().eval()
    reference = Host().eval()
    reference.load_state_dict(model.state_dict())
    x = torch.randn(2, 8)
    obs = torch.randn(2, 4)

    assemble(model, lambda: model(x, obs))
    obs2 = torch.randn(2, 4)
    with torch.no_grad():
        stale = model(x, obs2)
        fresh = reference(x, obs2)
        calib = reference(x, obs)
    # without wiring the banks hold the calibration observation: the
    # output ignores obs2's encoding — that is the manual contract
    assert torch.allclose(stale, calib, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(stale, fresh, rtol=1e-3, atol=1e-3)


def test_wire_refuses_without_statics():
    model = Host().eval()
    with pytest.raises(ValueError):
        wire_refresh_to_producer(model, (), lambda: None)
