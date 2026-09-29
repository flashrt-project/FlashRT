"""Validate Spark's opt-in gate against a configured CMake build."""
import os
from pathlib import Path
import subprocess

import pytest


def test_spark_target_matches_model_gate():
    value = os.environ.get("FLASHRT_BUILD_DIR")
    if not value:
        pytest.skip("set FLASHRT_BUILD_DIR to a configured CMake build")
    build = Path(value)
    cache = (build / "CMakeCache.txt").read_text()
    enabled = "FLASHRT_ENABLE_SPARK_X25:BOOL=ON" in cache
    targets = subprocess.check_output(
        ["cmake", "--build", str(build), "--target", "help"], text=True)
    assert ("flash_rt_sparkx25" in targets) == enabled
