"""CPU tests of production methods without importing the CUDA extension."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def method(name):
    path = Path(__file__).parents[1] / "flash_rt/frontends/torch/pi05_rtx.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Pi05TorchFrontendRtx")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "np": np, "ENC_D": 4}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_noise_fill_never_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("noise staging must not download")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    fill = method("_fill_noise")
    buf = torch.empty(2, 4, dtype=torch.bfloat16)
    assert fill(None, buf, torch.ones_like(buf), None) is None
    assert torch.equal(buf, torch.ones_like(buf))
    fill(None, buf, None, torch.Generator().manual_seed(1))
    fill(None, buf, None, None)
    with pytest.raises(ValueError):
        fill(None, buf, torch.zeros(1), None)
    with pytest.raises(ValueError):
        fill(None, buf, buf, torch.Generator())
