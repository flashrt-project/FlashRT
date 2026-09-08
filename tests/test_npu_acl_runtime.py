"""Runtime ownership and async ordering without requiring an NPU."""
import ctypes as C
from types import SimpleNamespace

import pytest
from flash_rt.npu.core.acl_runtime import NativeReplay


class Runtime:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def call(self, name, *args):
        self.calls.append(name)
        if name == self.fail:
            raise RuntimeError("injected")
        if name in {"aclrtGetCurrentContext", "aclrtCreateEventWithFlag"}:
            C.cast(args[0], C.POINTER(C.c_void_p))[0] = C.c_void_p(7)
        if name == "aclrtEventElapsedTime":
            C.cast(args[0], C.POINTER(C.c_float))[0] = 1.5


def make(runtime):
    host = SimpleNamespace(ptr=C.c_void_p(1), nbytes=4)
    return NativeReplay(runtime, C.c_void_p(2), 3, object(),
                        [(host, 4)], [(5, host)])


def test_enqueue_defers_completion_and_close_drains():
    rt = Runtime()
    replay = make(rt)
    replay.enqueue()
    assert "aclrtSynchronizeStream" not in rt.calls
    with pytest.raises(RuntimeError, match="pending"):
        replay.enqueue()
    replay.close()
    assert rt.calls.index("aclrtSynchronizeStream") < rt.calls.index("aclrtDestroyEvent")
    assert replay.last_replay_ms == 1.5
    with pytest.raises(RuntimeError, match="closed"):
        replay.enqueue()


def test_submission_failure_drains_upload_before_cleanup():
    rt = Runtime(fail="aclmdlRIExecuteAsync")
    replay = make(rt)
    with pytest.raises(RuntimeError, match="injected"):
        replay.enqueue()
    assert rt.calls[-1] == "aclrtSynchronizeStream"
    replay.close()
    assert rt.calls.index("aclrtSynchronizeStream") < rt.calls.index("aclrtDestroyEvent")


def test_static_row_precision_metadata_roundtrip(tmp_path):
    import numpy as np
    from flash_rt.core.precision_spec import ModelPrecisionSpec, PrecisionSpec
    spec = ModelPrecisionSpec(weight_specs={
        'linear.weight': PrecisionSpec(dtype='int8', granularity='per_channel',
                                      axis=0, scale=np.array([0.1, 0.2], np.float32))})
    spec.validate()
    path = tmp_path / 'precision.json'
    spec.to_json(str(path))
    restored = ModelPrecisionSpec.from_json(str(path))
    restored.validate()
    assert restored.weight_specs['linear.weight'].axis == 0
    np.testing.assert_array_equal(restored.weight_specs['linear.weight'].scale,
                                  spec.weight_specs['linear.weight'].scale)
