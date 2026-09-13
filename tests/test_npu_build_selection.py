"""What the Ascend build script compiles, and what it leaves behind.

Which models' units are built is a selection, and a selection that only decides
what to *add* is not one: building with a model on and then again with it off
would leave its shared objects in place, so the output directory would disagree
with the selection that produced it and a loader would bind them.

These drive the **real script**, end to end, against a fake toolchain: a
``bisheng`` and a ``c++`` that create the file named by their ``-o`` argument and
compile nothing. So they exercise the source paths and the artifact names as well
as the prune, they need neither CANN nor a device, and the script carries no test
hook of its own.
"""
import os
import pathlib
import stat
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "npu" / "build.sh"

PI05 = ["libflashrt_npu.so", "libflashrt_npu_cube.so", "libflashrt_npu_decoder.so",
        "libflashrt_npu_attn.so"]
GROOT_N17 = ["libflashrt_npu_groot_n17_dit_attn.so",
             "libflashrt_npu_groot_n17_dit_norm.so",
             "libflashrt_npu_groot_n17_image.so"]

#: Creates whatever its `-o` argument names and records the sources it was given,
#: which is all the script needs of a compiler to be driven to completion.
_FAKE_COMPILER = """#!/usr/bin/env bash
out=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "-o" ]]; then out="$arg"; fi
    case "$arg" in
        *.cpp) echo "$arg" >> "$FAKE_COMPILER_LOG" ;;
    esac
    prev="$arg"
done
[[ -n "$out" ]] && : > "$out"
exit 0
"""


@pytest.fixture
def build(tmp_path):
    """Returns a callable that runs the real script against a fake toolchain."""
    toolkit = tmp_path / "toolkit"
    (toolkit / "bin").mkdir(parents=True)
    fake = toolkit / "bin" / "bisheng"
    fake.write_text(_FAKE_COMPILER)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    shims = tmp_path / "shims"
    shims.mkdir()
    host = shims / "c++"
    host.write_text(_FAKE_COMPILER)
    host.chmod(host.stat().st_mode | stat.S_IEXEC)
    out_dir = tmp_path / "lib"
    log = tmp_path / "sources.txt"

    def run(**env):
        environment = dict(
            os.environ,
            ASCEND_TOOLKIT_HOME=str(toolkit),
            FLASHRT_NPU_BUILD_DIR=str(out_dir),
            FAKE_COMPILER_LOG=str(log),
            PATH=f"{shims}{os.pathsep}{os.environ['PATH']}")
        environment.update(env)
        return subprocess.run(["bash", str(SCRIPT)], capture_output=True,
                              text=True, env=environment)

    run.out_dir = out_dir
    run.sources = lambda: (log.read_text().split() if log.exists() else [])
    run.built = lambda: sorted(p.name for p in out_dir.iterdir()
                               if p.suffix == ".so")
    return run


def test_the_default_selection_builds_the_four_pi05_units(build):
    done = build()
    assert done.returncode == 0, done.stderr
    assert build.built() == sorted(PI05)
    assert "pi05=ON" in done.stdout and "groot_n17=OFF" in done.stdout


def test_selecting_this_model_adds_its_three_and_compiles_its_directory(build):
    done = build(FLASHRT_ENABLE_NPU_GROOT_N17="ON")
    assert done.returncode == 0, done.stderr
    assert build.built() == sorted(PI05 + GROOT_N17)
    compiled = build.sources()
    for unit in ("dit_attn_910b.cpp", "dit_norm_910b.cpp", "area_resize_910b.cpp"):
        assert any(source.endswith(f"kernels/groot_n17/{unit}")
                   for source in compiled), f"{unit} came from the model directory"


def test_selecting_only_this_model_builds_only_its_three(build):
    done = build(FLASHRT_ENABLE_NPU_PI05="OFF", FLASHRT_ENABLE_NPU_GROOT_N17="ON")
    assert done.returncode == 0, done.stderr
    assert build.built() == sorted(GROOT_N17)


def test_a_deselected_models_units_are_removed_not_left_behind(build):
    """The sequential build: with this model on, then with the default. Seven
    libraries must not survive into a build that reports four."""
    assert build(FLASHRT_ENABLE_NPU_GROOT_N17="ON").returncode == 0
    assert build.built() == sorted(PI05 + GROOT_N17)
    done = build()
    assert done.returncode == 0, done.stderr
    assert build.built() == sorted(PI05)
    for name in GROOT_N17:
        assert name in done.stdout, "a removal is reported, not silent"


def test_the_reverse_sequential_build_removes_pi05s_units(build):
    assert build().returncode == 0
    assert build(FLASHRT_ENABLE_NPU_PI05="OFF",
                 FLASHRT_ENABLE_NPU_GROOT_N17="ON").returncode == 0
    assert build.built() == sorted(GROOT_N17)


def test_nothing_but_the_standard_names_is_touched(build):
    """An override lives wherever the caller put it, and the directory may hold
    other things; the script owns the names it produces and nothing else."""
    assert build(FLASHRT_ENABLE_NPU_GROOT_N17="ON").returncode == 0
    (build.out_dir / "libflashrt_npu_experiment.so").write_bytes(b"")
    (build.out_dir / "notes.txt").write_text("mine")
    assert build().returncode == 0
    left = sorted(p.name for p in build.out_dir.iterdir())
    assert "libflashrt_npu_experiment.so" in left and "notes.txt" in left


def test_a_selection_with_no_model_is_refused_and_deletes_nothing(build):
    assert build().returncode == 0
    done = build(FLASHRT_ENABLE_NPU_PI05="OFF", FLASHRT_ENABLE_NPU_GROOT_N17="OFF")
    assert done.returncode != 0
    assert "no model selected" in done.stderr
    assert build.built() == sorted(PI05), "a refused selection deletes nothing"


@pytest.mark.parametrize("value", ["", "1", "yes", "on"])
def test_a_switch_that_is_not_on_or_off_is_refused_by_name(build, value):
    done = build(FLASHRT_ENABLE_NPU_GROOT_N17=value)
    assert done.returncode != 0
    assert "NPU_ENABLE_GROOT_N17 must be ON or OFF" in done.stderr


def test_a_part_these_kernels_are_not_validated_for_is_refused(build):
    done = build(ASCEND_SOC_VERSION="Ascend910B2")
    assert done.returncode != 0
    assert "Ascend910B4 only" in done.stderr


def test_the_abi_and_part_are_compiled_into_every_unit(build):
    """Each library carries its own identity, which is what lets a loader refuse
    a partial rebuild instead of calling a stale contract."""
    done = build(FLASHRT_ENABLE_NPU_GROOT_N17="ON")
    assert done.returncode == 0, done.stderr
    assert build.built() == sorted(PI05 + GROOT_N17)
    from flash_rt.npu.core import abi

    assert abi.ABI_VERSION == 1, (
        "the published ABI; adding units with new symbols does not change it")
