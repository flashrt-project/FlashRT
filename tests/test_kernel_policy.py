"""Process-wide kernel fast-path policy.

The register-resident producers and PDL are validated by the VLA-JEPA model,
but they are shared C++ entry points. They are opt-in: the default must keep
the pre-existing kernels for every other caller, and only the model that
validated them may flip the process flag. ``FLASHRT_*`` environment
variables stay as explicit per-process overrides.

Run:
    PYTHONPATH=. python -m pytest tests/test_kernel_policy.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kernels = pytest.importorskip("flash_rt.flash_rt_kernels")

pytestmark = pytest.mark.skipif(
    not hasattr(kernels, "set_kernel_fast_paths"),
    reason="build without the kernel-fast-path policy binding")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _restore_policy():
    kernels.set_kernel_fast_paths(False, False)
    yield
    kernels.set_kernel_fast_paths(False, False)


def test_default_policy_keeps_the_shared_kernels_on_the_old_path():
    assert kernels.get_kernel_fast_paths() == (False, False)


def test_opt_in_is_process_wide_and_reversible():
    kernels.set_kernel_fast_paths(True, True)
    assert kernels.get_kernel_fast_paths() == (True, True)
    kernels.set_kernel_fast_paths(False, False)
    assert kernels.get_kernel_fast_paths() == (False, False)


def _run(code: str, **env_extra: str) -> str:
    """Run a snippet in a fresh process (the env override is cached once)."""
    env = dict(os.environ)
    for key in ("FLASHRT_PDL",):
        env.pop(key, None)
    env.update(env_extra)
    script = f"import sys; sys.path.insert(0, {_ROOT!r})\n" + code
    return subprocess.check_output(
        [sys.executable, "-I", "-c", script], env=env, text=True).strip()


def test_pdl_default_is_off():
    assert _run(
        "import flash_rt.flash_rt_kernels as k; print(k.pdl_enabled())") == "False"


def test_pdl_env_can_force_on():
    assert _run(
        "import flash_rt.flash_rt_kernels as k; print(k.pdl_enabled())",
        FLASHRT_PDL="1") == "True"


def test_pdl_env_can_force_off_even_after_opt_in():
    assert _run(
        "import flash_rt.flash_rt_kernels as k\n"
        "k.set_kernel_fast_paths(True, True)\n"
        "print(k.pdl_enabled())",
        FLASHRT_PDL="0") == "False"


def test_only_the_vla_model_opts_into_the_fast_paths():
    """Shared defaults stay old for other models: the VLA-JEPA frontend is
    the only Python caller of the process-wide opt-in."""
    callers = []
    for dirpath, _, filenames in os.walk(os.path.join(_ROOT, "flash_rt")):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if "set_kernel_fast_paths(" in text:
                callers.append(os.path.relpath(path, _ROOT))
    assert callers == [os.path.join("flash_rt", "frontends", "torch",
                                    "vla_jepa_rtx.py")]
