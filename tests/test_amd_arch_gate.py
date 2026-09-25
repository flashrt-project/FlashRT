"""The CDNA gates accept exact gfx942/gfx950 build/device pairs only.

The AMD backend supports exact gfx942 and gfx950 targets. Their FP8 formats
are incompatible, so the extension and device must match. These tests pin
the exact base-target comparison and reject look-alike names.

The runtime gate is exercised through its comparison rule rather than by
constructing a frontend (that needs a device, a built extension and a
checkpoint); the build gate is exercised by running the script.
"""
import pathlib
import re
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_BUILD = _REPO / "scripts" / "amd" / "build_amd.sh"

# (device_arch, build_arch, must_be_accepted)
_ARCH_CASES = [
    ("gfx950", "gfx950", True),
    ("gfx950:sramecc+:xnack-", "gfx950", True),      # feature suffix is fine
    ("gfx950", "gfx950:xnack-", True),
    ("gfx9500", "gfx950", False),                    # prefix trap
    ("gfx950", "gfx9500", False),                    # prefix trap, build side
    ("gfx942", "gfx942", True),                      # CDNA3
    ("gfx942:sramecc+:xnack-", "gfx942", True),
    ("gfx942", "gfx950", False),                     # cross-generation
    ("gfx950", "gfx942", False),
    ("gfx90a", "gfx950", False),
    ("none", "gfx950", False),                       # device probe failed
    ("gfx950", "unknown", False),                    # build stamp missing
]


def _accepted(device_arch: str, build_arch: str) -> bool:
    """The frontends' gate rule, kept in one place for the test.

    Mirrors ``flash_rt/amd/frontends/torch/pi05.py`` (and the GROOT
    frontend on its branch): compare only the base target, so a feature
    suffix passes and a longer look-alike name does not.
    """
    device = device_arch.split(":", 1)[0]
    build = build_arch.split(":", 1)[0]
    return device == build and build in {"gfx942", "gfx950"}


@pytest.mark.parametrize("device_arch,build_arch,accepted", _ARCH_CASES)
def test_gate_rule_accepts_only_supported_exact_pairs(device_arch, build_arch, accepted):
    assert _accepted(device_arch, build_arch) is accepted


def test_frontends_use_shared_capability_gate():
    """Guard against a regression back to ``startswith("gfx950")``.

    A prefix test silently admits ``gfx9500``; the frontends must split
    the feature suffix off and compare the base target exactly.
    """
    for source in ("flash_rt/amd/frontends/torch/pi05.py",
                   "flash_rt/amd/frontends/torch/groot_n17.py"):
        text = (_REPO / source).read_text()
        assert "load_capabilities" in text
        assert not re.search(r'startswith\(\s*["\']gfx(?:942|950)["\']\s*\)', text)


@pytest.mark.skipif(not _BUILD.exists(), reason="build script not present")
@pytest.mark.parametrize("arch", ["gfx9500", "gfx9420", "gfx90a", "sm_120"])
def test_build_script_rejects_unsupported_arch(arch):
    """The build script must refuse an unregistered AMD target.

    Building for another architecture yields a module that can never pass
    the runtime gate, so failing at build time is the cheaper error.
    """
    proc = subprocess.run(["bash", str(_BUILD), arch],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, (
        f"build script accepted GPU_ARCH={arch}")
    output = proc.stderr + proc.stdout
    assert all(arch in output for arch in ("gfx942", "gfx950", "gfx1151")), (
        "the rejection message should name every supported architecture")


@pytest.mark.skipif(not _BUILD.exists(), reason="build script not present")
def test_build_script_documents_all_supported_targets():
    text = _BUILD.read_text()
    assert "gfx942|gfx950|gfx1151" in text
