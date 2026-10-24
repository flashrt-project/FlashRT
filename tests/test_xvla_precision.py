"""Optional first-light smoke for X-VLA on a real checkpoint.

This test is intentionally local-only: it needs the multi-GB
``lerobot/xvla-libero`` checkpoint and a BART tokenizer. Run it with::

    XVLA_CKPT=/path/to/xvla-libero \
    XVLA_REF_ACTIONS=/path/to/ref_actions.npz \
    pytest -q tests/test_xvla_precision.py -s

``XVLA_REF_ACTIONS`` is optional; when present it must be an ``.npz`` with an
``actions`` array of shape ``[N, 30, 20]`` for the same observations. The test
checks finite shape-preserving output; the full per-frame cosine evidence is
in ``docs/xvla_rtx_sm120.md`` and the campaign receipt rather than a
repository fixture.
"""

import os

import numpy as np
import pytest
import torch

pytest.importorskip("flash_rt")
pytest.importorskip("transformers")

CKPT = os.environ.get("XVLA_CKPT")
if not CKPT:
    pytest.skip("set XVLA_CKPT to run the X-VLA first-light smoke",
                allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("X-VLA requires CUDA", allow_module_level=True)
if torch.cuda.get_device_capability(0) not in ((12, 0), (12, 1)):
    pytest.skip("X-VLA is validated on SM120/SM121 only", allow_module_level=True)


def test_xvla_predict_smoke():
    import flash_rt

    precision = os.environ.get("XVLA_PRECISION", "nvfp4")
    model = flash_rt.load_model(
        CKPT,
        config="xvla",
        framework="torch",
        hardware="rtx_sm120",
        num_views=2,
        xvla_precision=precision,
    )
    images = [np.zeros((256, 256, 3), dtype=np.float32) for _ in range(2)]
    state = np.zeros(8, dtype=np.float32)
    actions = model.predict(images, prompt="pick up the object", state=state)
    actions = np.asarray(actions)
    assert actions.shape == (30, 20)
    assert np.isfinite(actions).all()
