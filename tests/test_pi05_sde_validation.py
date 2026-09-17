"""Reject invalid sigma before allocating or copying any device data."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1])
def test_invalid_sigma_is_rejected_before_device_access(bad):
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    frontend = SimpleNamespace(_sde=True, _num_steps=10)
    sigma = [0.0] * 10
    sigma[4] = bad
    with pytest.raises(ValueError, match="finite non-negative"):
        Pi05TorchFrontendRtx._fill_sde(frontend, None, sigma, None, 0)
