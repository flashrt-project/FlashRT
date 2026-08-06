from __future__ import annotations

import inspect
import os

from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor


def test_chameleon_trt_vqgan_is_opt_in_by_default():
    sig = inspect.signature(ChameleonTorchFrontendThor.__init__)
    assert sig.parameters["use_trt_vqgan"].default is False


def test_chameleon_fa4_attn_is_opt_in_by_default():
    sig = inspect.signature(ChameleonTorchFrontendThor.__init__)
    assert sig.parameters["use_fa4_attn"].default is None
    os.environ.pop("FLASHRT_CHAMELEON_FA4_ATTN", None)
    assert bool(os.environ.get("FLASHRT_CHAMELEON_FA4_ATTN", "0") in ("1", "true", "on")) is False
