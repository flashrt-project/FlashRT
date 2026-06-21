"""CPU-only tests for the public calibration API contract."""

import pytest

from flash_rt.core.calibration_api import implicit_calibrate


def test_implicit_calibrate_single_sample_runs_infer_once():
    class Frontend:
        def __init__(self):
            self.calls = []

        def infer(self, obs):
            self.calls.append(obs)

    frontend = Frontend()
    obs = {"frame": 1}
    implicit_calibrate(frontend, [obs])
    assert frontend.calls == [obs]


def test_implicit_calibrate_multi_sample_error_points_to_shim():
    class Frontend:
        def infer(self, obs):
            raise AssertionError("N>=2 must not call infer")

    with pytest.raises(NotImplementedError) as exc:
        implicit_calibrate(Frontend(), [{"frame": 1}, {"frame": 2}])
    msg = str(exc.value)
    assert "implicit_calibrate compatibility shim" in msg
    assert "native multi-sample calibration" in msg
    assert "Thor multi-sample support is planned" not in msg


def test_vla_model_calibrate_forwards_public_kwargs():
    from flash_rt.api import VLAModel

    class Frontend:
        seen = None

        def calibrate(self, observations, *, percentile, max_samples, verbose):
            type(self).seen = {
                "observations": observations,
                "percentile": percentile,
                "max_samples": max_samples,
                "verbose": verbose,
            }

    obs = [{"frame": 1}, {"frame": 2}]
    model = VLAModel(Frontend(), framework="torch")
    model.calibrate(obs, percentile=99.0, max_samples=1, verbose=True)

    assert Frontend.seen == {
        "observations": obs,
        "percentile": 99.0,
        "max_samples": 1,
        "verbose": True,
    }


def test_groot_n17_calibrate_accepts_public_max_samples_kwarg():
    from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor

    frontend = GrootN17TorchFrontendThor.__new__(GrootN17TorchFrontendThor)
    frontend._backbone_features = object()
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return kwargs

    frontend._snapshot_precision_spec = snapshot
    frontend.calibrate([{"aux": 1}, {"aux": 2}], max_samples=1)

    assert calls == [{
        "method": "single_frame",
        "n": 1,
        "percentile": None,
    }]
    assert frontend._precision_spec == calls[0]
