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
