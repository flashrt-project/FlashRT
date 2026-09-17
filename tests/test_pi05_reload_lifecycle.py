"""CPU lifecycle tests independent of native extension availability."""
import importlib.util
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

spec = importlib.util.spec_from_file_location(
    "pi05_lifecycle", Path(__file__).parents[1] / "flash_rt/models/pi05/_lifecycle.py")
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


class Frontend:
    def __init__(self):
        self._lifecycle_lock = threading.RLock()
        self._reload_failed = False

    @lifecycle.serialized
    def infer(self):
        return "action"

    @lifecycle.reload_guard
    def reload(self, mutate=False, entered=None, release=None):
        if entered is not None:
            entered.set()
            assert release.wait(5)
        self._reload_mutating = mutate
        raise ValueError("invalid checkpoint")


@pytest.mark.parametrize("mutate", [False, True])
def test_failure_before_and_after_mutation(mutate):
    frontend = Frontend()
    with pytest.raises(ValueError):
        frontend.reload(mutate=mutate)
    if mutate:
        with pytest.raises(RuntimeError, match="construct a new frontend"):
            frontend.infer()
    else:
        assert frontend.infer() == "action"


def test_reload_and_inference_are_serialized():
    frontend = Frontend()
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    def infer():
        attempted.set()
        return frontend.infer()
    with ThreadPoolExecutor(max_workers=2) as executor:
        reload = executor.submit(frontend.reload, False, entered, release)
        try:
            assert entered.wait(5)
            action = executor.submit(infer)
            assert attempted.wait(5)
            assert not action.done()
        finally:
            release.set()
        with pytest.raises(ValueError):
            reload.result(timeout=5)
        assert action.result(timeout=5) == "action"
