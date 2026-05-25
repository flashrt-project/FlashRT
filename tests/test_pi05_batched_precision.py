"""Precision gate for Phase 3a (generic batched Pi05 pipeline).

Asserts that ``Pi05BatchedPipeline`` (B=2, both slots fed the same obs
and prompt) reproduces the single-sample ``Pi05Pipeline`` output to
within the same tolerance the CFG precision gate expects. This isolates
any numerical drift that is purely due to the batched GEMM / attention
path (FP8 quantization, cuBLASLt tactic differences at M=B*seq vs
M=seq) from CFG-specific algorithmic drift tested in
``test_pi05_cfg_batched_inference.py``.

Run both branches in separate child processes; sharing a CUDA context
between two frontend instances in one process is known-flaky.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)


_CHILD = r"""
import os, sys, numpy as np, torch
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
mode, ckpt, out_path, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
obs = {"image": np.zeros((224,224,3), dtype=np.uint8),
       "wrist_image": np.zeros((224,224,3), dtype=np.uint8),
       "state": np.zeros(8, dtype=np.float32)}
rt = Pi05TorchFrontendRtx(ckpt, num_views=2)
if mode == "batched":
    rt.set_batched_mode(enable=True)
    rt.set_prompt_batch(["pick up the cup", "pick up the cup"])
    rt.calibrate([obs])
    torch.manual_seed(seed)
    out_list = rt.infer_batch([obs, obs])
    out = out_list[0]["actions"]
else:
    rt.set_prompt("pick up the cup")
    rt.calibrate([obs])
    torch.manual_seed(seed)
    out = rt.infer(obs)["actions"]
np.save(out_path, out)
"""


def _run_child(mode: str, out_path: str, seed: int = 424242) -> None:
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, mode, CKPT_PI05, out_path, str(seed)],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"child ({mode}) failed rc={r.returncode}\n"
            f"stdout tail:\n{r.stdout[-2000:]}\n"
            f"stderr tail:\n{r.stderr[-2000:]}")


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_batched_cos_vs_b1():
    """Batched Pi05 (slot 0 w/ identical obs+prompt) must match B=1 serial.

    Same prompt, same dummy obs, same noise seed on both paths. Slot 0
    of the batched output is compared against the B=1 serial output.
    Target: cos >= 0.99 (dummy obs; matches the tolerance from
    docs/rl_inference.md).
    """
    with tempfile.TemporaryDirectory() as td:
        b1_p = os.path.join(td, "b1.npy")
        b2_p = os.path.join(td, "b2.npy")
        _run_child("serial", b1_p)
        _run_child("batched", b2_p)
        a = np.load(b1_p).astype(np.float64).flatten()
        b = np.load(b2_p).astype(np.float64).flatten()
        cos = float(np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        assert cos >= 0.99, (
            f"Phase 3a batched path drifts from B=1: cos={cos:.6f} < 0.99\n"
            f"b1[:5]={a[:5]}\nb2[:5]={b[:5]}")
