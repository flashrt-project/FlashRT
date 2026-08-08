"""Unit tests for flash_rt.structures.guard and .swap.

CPU-only, no kernels and no checkpoints: the subject is the mechanism a
bound structure uses to answer for itself at runtime, which is testable
with a synthetic structure whose output is unmistakably not the host's.

The property under test throughout is that nothing here is silent. A seam
called outside the form it was bound for may run the host module instead —
that is the useful behaviour — but it may not do so without saying it
once, counting every time, and eventually taking itself out of the model.
"""

import threading
import warnings

import pytest
import torch
from torch import nn

from flash_rt.structures.guard import (
    CAST_OK, PROCEED, SELF_DETACH_AFTER, GuardRefused, GuardedSeam)
from flash_rt.structures.swap import attach


class FakeFused(GuardedSeam, nn.Module):
    """A structure bound for a fixed row count, returning all ones.

    Row count is the one contract violation a plain ``nn.Linear`` can
    still compute through, which is what makes it the useful trigger:
    a width, dtype or device mismatch fails in the host too, so it cannot
    show that the fallback reproduces the host.
    """

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, original: nn.Module, rows: int | None = None):
        super().__init__()
        self.host_linear = original
        self._frt_arm(dtypes=CAST_OK, device=original.weight.device,
                      k=int(original.weight.shape[1]), rows=rows)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_linear":
                raise
            return getattr(super().__getattr__("host_linear"), name)

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        return torch.ones(*x.shape[:-1], self.host_linear.out_features,
                          device=x.device, dtype=x.dtype)


class CapacityFused(GuardedSeam, nn.Module):
    """A preallocated seam that admits every logical row count it covers."""

    _frt_host_attr = "host_linear"
    _frt_can_fallback = True

    def __init__(self, original: nn.Module, capacity: int):
        super().__init__()
        self.host_linear = original
        self._frt_arm(
            dtypes=CAST_OK, device=original.weight.device,
            k=int(original.weight.shape[1]), row_capacity=capacity)

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        return torch.ones(*x.shape[:-1], self.host_linear.out_features,
                          device=x.device, dtype=x.dtype)


class Composed(GuardedSeam, nn.Module):
    """A structure with no equivalent host module to revert to."""

    _frt_can_fallback = False

    def __init__(self, k: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(k, k))
        self._frt_arm(dtypes=CAST_OK, device=self.w.device, k=k)

    def forward(self, x):
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        return x @ self.w


class Host(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        self.proj = nn.Linear(k, k)
        self.tail = nn.Linear(k, k, bias=False)

    def forward(self, x):
        return self.tail(self.proj(x))


K = 8


@pytest.fixture()
def host():
    torch.manual_seed(0)
    return Host(K).eval()


def _rows(n):
    return torch.randn(n, K)


def test_admitted_call_runs_the_structure(host):
    handle = attach(host, {"proj": FakeFused(host.proj, rows=4)})
    out = host.proj(_rows(4))
    assert torch.allclose(out, torch.ones_like(out))
    entry = handle.report()["proj"]
    assert (entry["calls"], entry["fallbacks"]) == (1, 0)
    assert handle.summary()["clean"]
    handle.raise_on_fallback()


def test_row_capacity_admits_smaller_shapes_and_reports_capacity(host):
    handle = attach(
        host, {"proj": CapacityFused(host.proj, capacity=8)})
    for rows in (8, 3, 6):
        out = host.proj(_rows(rows))
        assert out.shape[0] == rows
        assert torch.equal(out, torch.ones_like(out))
    entry = handle.report()["proj"]
    assert (entry["calls"], entry["fallbacks"]) == (3, 0)
    assert entry["form"]["rows"] is None
    assert entry["form"]["row_capacity"] == 8


def test_row_capacity_falls_back_only_above_capacity(host):
    original = host.proj
    handle = attach(
        host, {"proj": CapacityFused(host.proj, capacity=8)})
    x = _rows(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = host.proj(x)
    assert torch.equal(got, original(x))
    entry = handle.report()["proj"]
    assert entry["fallbacks"] == 1
    assert "capacity 8" in entry["last_reason"]


def test_refused_call_reproduces_the_host_exactly(host):
    original = host.proj
    handle = attach(host, {"proj": FakeFused(host.proj, rows=4)})
    x = _rows(9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = host.proj(x)
    assert torch.equal(out, original(x))
    assert any("fell back to the host module" in str(w.message)
               for w in caught)
    entry = handle.report()["proj"]
    assert entry["fallbacks"] == 1
    assert "row" in entry["last_reason"]
    assert not handle.summary()["clean"]
    with pytest.raises(RuntimeError, match="fell back"):
        handle.raise_on_fallback()


def test_warns_once_but_counts_always(host):
    handle = attach(host, {"proj": FakeFused(host.proj, rows=4)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            host.proj(_rows(9))
    assert sum("fell back to the host module" in str(w.message)
               for w in caught) == 1
    assert handle.report()["proj"]["fallbacks"] == 5


def test_persistent_fallback_restores_the_host_module(host):
    replacement = FakeFused(host.proj, rows=4)
    handle = attach(host, {"proj": replacement})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(SELF_DETACH_AFTER):
            host.proj(_rows(9))
    assert handle.report()["proj"]["detached"]
    assert host.proj is not replacement
    assert any("restored at that path permanently" in str(w.message)
               for w in caught)
    before = handle.report()["proj"]["calls"]
    host.proj(_rows(9))
    assert handle.report()["proj"]["calls"] == before
    handle.detach()
    assert isinstance(host.proj, nn.Linear)


def test_strict_mode_refuses_instead_of_falling_back(host):
    attach(host, {"proj": FakeFused(host.proj, rows=4)},
           on_guard_fail="raise")
    with pytest.raises(GuardRefused, match="row"):
        host.proj(_rows(9))


def test_structure_without_a_host_path_refuses(host):
    attach(host, {"proj": Composed(K)})
    with pytest.raises(GuardRefused, match="no equivalent host module"):
        host.proj(torch.randn(4, K * 2))


def test_second_thread_is_refused_not_interleaved(host):
    original = host.proj
    handle = attach(host, {"proj": FakeFused(host.proj, rows=4)})
    x = _rows(4)
    host.proj(x)                      # first thread claims the seam
    box = {}

    def other():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            box["out"] = host.proj(x)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert "second thread" in handle.report()["proj"]["last_reason"]
    assert torch.equal(box["out"], original(x))
    assert torch.allclose(host.proj(x), torch.ones(4, K))


def test_state_dict_keeps_the_host_schema_while_attached(host):
    before = {name: t.clone() for name, t in host.state_dict().items()}
    handle = attach(host, {"proj": FakeFused(host.proj)})
    attached = host.state_dict()
    assert sorted(attached) == sorted(before)
    assert all(torch.equal(attached[n], before[n]) for n in before)
    handle.detach()
    host.load_state_dict(before)
    assert all(torch.equal(host.state_dict()[n], before[n]) for n in before)


def test_migration_and_loading_are_refused_while_attached(host):
    snapshot = {name: t.clone() for name, t in host.state_dict().items()}
    handle = attach(host, {"proj": FakeFused(host.proj)})
    with pytest.raises(GuardRefused, match="detach"):
        host.to(torch.float64)
    with pytest.raises(RuntimeError, match="cannot load_state_dict"):
        host.load_state_dict(snapshot)
    handle.detach()
    host.to(torch.float32)
    host.load_state_dict(snapshot)


def test_training_mode_is_refused(host):
    host.train()
    with pytest.raises(ValueError, match="training mode"):
        attach(host, {"proj": FakeFused(host.proj)})
    host.eval()
    attach(host, {"proj": FakeFused(host.proj)}, allow_training=True)


def test_failed_resolution_leaves_the_model_untouched(host):
    kept = host.proj
    with pytest.raises(AttributeError):
        attach(host, {"proj": FakeFused(host.proj),
                      "absent.path": FakeFused(host.proj)})
    assert host.proj is kept


def test_nested_guards_report_but_cannot_self_detach(host):
    class Outer(GuardedSeam, nn.Module):
        _frt_can_fallback = False

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self._frt_arm(dtypes=CAST_OK, device=torch.device("cpu"), k=K)

        def forward(self, x):
            admitted = self._frt_admit(x)
            if admitted is not PROCEED:
                return admitted
            return self.inner(x)

    handle = attach(host, {"proj": Outer(FakeFused(host.proj))})
    host.proj(_rows(4))
    assert sorted(handle.report()) == ["proj", "proj::inner"]
    assert handle.report()["proj::inner"]["calls"] == 1


def test_detach_is_idempotent_and_runs_revert_once(host):
    calls = []
    handle = attach(host, {"proj": FakeFused(host.proj)},
                    revert=[lambda: calls.append(1)])
    handle.detach()
    handle.detach()
    assert calls == [1]
    assert isinstance(host.proj, nn.Linear)
