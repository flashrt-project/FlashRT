"""The gfx950 gates must reject look-alike architectures.

The AMD backend is gfx950-only: its MFMA tile shapes and FP8 paths
compute wrong results — not a slow fallback — on other AMD targets. Two
gates enforce that, and both used to compare with a prefix test, which
would also accept a hypothetical ``gfx9500``. These tests pin the exact
base-target comparison at both gates.

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
    ("gfx942", "gfx950", False),                     # CDNA3
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
    return (device_arch.split(":", 1)[0] == "gfx950"
            and build_arch.split(":", 1)[0] == "gfx950")


@pytest.mark.parametrize("device_arch,build_arch,accepted", _ARCH_CASES)
def test_gate_rule_accepts_only_gfx950(device_arch, build_arch, accepted):
    assert _accepted(device_arch, build_arch) is accepted


@pytest.mark.parametrize("source", [
    "flash_rt/amd/frontends/torch/pi05.py",
    "flash_rt/amd/frontends/torch/groot_n17.py",
])
def test_frontend_gate_uses_base_target_comparison(source):
    """Guard against a regression back to ``startswith("gfx950")``.

    A prefix test silently admits ``gfx9500``; the frontends must split
    the feature suffix off and compare the base target exactly.
    """
    path = _REPO / source
    if not path.exists():
        pytest.skip(f"{source} is not present on this branch")
    text = path.read_text()
    if "device_arch()" not in text:
        pytest.skip(f"{source} does not carry an architecture gate")
    assert 'split(":", 1)[0] == "gfx950"' in text, (
        f"{source} must compare the base architecture target exactly")
    assert not re.search(r'startswith\(\s*["\']gfx950["\']\s*\)', text), (
        f"{source} still uses a prefix test, which accepts gfx9500")


@pytest.mark.skipif(not _BUILD.exists(), reason="build script not present")
@pytest.mark.parametrize("arch", ["gfx9500", "gfx942", "gfx90a", "sm_120"])
def test_build_script_rejects_non_gfx950(arch):
    """The build script must refuse a non-gfx950 target.

    Building for another architecture yields a module that can never pass
    the runtime gate, so failing at build time is the cheaper error.
    """
    proc = subprocess.run(["bash", str(_BUILD), arch],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0, (
        f"build script accepted GPU_ARCH={arch}")
    assert "gfx950" in (proc.stderr + proc.stdout), (
        "the rejection message should name the supported architecture")


@pytest.mark.skipif(not _BUILD.exists(), reason="build script not present")
def test_build_script_override_is_documented():
    """The escape hatch for a future port must stay discoverable."""
    text = _BUILD.read_text()
    assert "FLASHRT_AMD_ALLOW_ARCH" in text
