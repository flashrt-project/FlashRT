"""Torch-free Apple plan binding: clear failures before any Metal call (runs on any host)."""
import json
import subprocess
import sys

import pytest


def test_binding_imports_without_torch():
    code = ("import json, sys; import flash_rt.hardware.apple_native.native; "
            "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in {'torch', 'mlx'})))")
    result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []


def test_native_plan_fails_clearly(monkeypatch, tmp_path):
    from flash_rt.hardware.apple_native import native
    monkeypatch.setattr(native.sys, 'platform', 'linux')
    with pytest.raises(RuntimeError, match='macOS'):
        native.NativePlan()
    monkeypatch.setattr(native.sys, 'platform', 'darwin')
    with pytest.raises(RuntimeError, match='not found'):
        native.NativePlan(tmp_path / 'missing.dylib')
    monkeypatch.setenv('FLASHRT_APPLE_PLAN_LIBRARY', str(tmp_path / 'other.dylib'))
    assert native.default_library() == tmp_path / 'other.dylib'
    with pytest.raises(RuntimeError, match='other.dylib'):
        native.NativePlan()


class _Lib:
    """Records plan dispatch calls; ``flags=False`` models a library built before concurrent dispatches."""

    def __init__(self, flags=True):
        self.calls = []
        self.frt_apple_plan_dispatch = lambda *args: self.calls.append(('dispatch', args[-1])) or 0
        if flags:
            self.frt_apple_plan_dispatch_flags = _Fn(lambda *args: self.calls.append(('flags', args[-1])) or 0)


class _Fn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def _plan(lib):
    from flash_rt.hardware.apple_native import native
    plan = native.NativePlan.__new__(native.NativePlan)
    plan.lib, plan.handle, plan.labels, plan._recording = lib, None, {0: []}, 0
    return plan


def test_dispatch_barrier_default_keeps_the_original_call():
    lib = _Lib()
    plan = _plan(lib)
    plan.dispatch('k', [(0, 0, 0)], (1,), 32, label='a')
    plan.dispatch('k', [(0, 0, 0)], (1,), 32, label='b', barrier=False)
    assert lib.calls == [('dispatch', b'a'), ('flags', 1)]
    assert plan.labels[0] == ['a', 'b']


def test_concurrent_dispatch_needs_a_rebuilt_library():
    plan = _plan(_Lib(flags=False))
    plan.dispatch('k', [(0, 0, 0)], (1,), 32)
    with pytest.raises(RuntimeError, match='frt_apple_plan_dispatch_flags'):
        plan.dispatch('k', [(0, 0, 0)], (1,), 32, barrier=False)
