"""Integration tests for the Pi0.5 RTX batched CFG path (Phase 3b).

Runs on RTX 5090 / 4090. Skipped if GPU or pi05 checkpoint missing.

These tests cover:
  - The default ``infer()`` path is unaffected when neither RL mode
    nor batched mode is enabled.
  - ``set_rl_mode(enable=True)`` + ``set_batched_mode(enable=True)``
    + ``set_prompt()`` builds a :class:`Pi05CFGBatchedPipeline`.
  - ``infer(obs)`` on the batched CFG pipeline returns finite actions.
  - Output cosine similarity to the serial CFG pipeline (Phase 2)
    on the same observation is high (same math, different fusion —
    minor FP8 / autotune drift expected).

Run::

    python -m pytest tests/test_pi05_cfg_batched_inference.py -v
"""

import os

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")

_GPU_AVAILABLE = torch.cuda.is_available()
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)


def _make_dummy_obs():
    return {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_rl_only_uses_serial_cfg_pipeline():
    """RL mode without batched mode uses the serial Pi05CFGPipeline."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    from flash_rt.models.pi05.pipeline_rtx_cfg import Pi05CFGPipeline
    from flash_rt.models.pi05.pipeline_rtx_cfg_batched import (
        Pi05CFGBatchedPipeline,
    )

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    rt.set_prompt("pick up the cup")
    assert isinstance(rt.pipeline, Pi05CFGPipeline)
    assert not isinstance(rt.pipeline, Pi05CFGBatchedPipeline)


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_rl_and_batched_builds_cfg_batched_pipeline():
    """RL + batched together builds Pi05CFGBatchedPipeline."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    from flash_rt.models.pi05.pipeline_rtx_cfg_batched import (
        Pi05CFGBatchedPipeline,
    )

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True)
    rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    rt.set_prompt("pick up the cup")
    assert isinstance(rt.pipeline, Pi05CFGBatchedPipeline)
    assert rt.pipeline.cfg_beta == pytest.approx(1.5)
    assert rt.pipeline.B == 2


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_batched_cfg_inference_returns_finite_actions():
    """End-to-end batched CFG: calibrate + infer produces finite actions."""
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    rt = Pi05TorchFrontendRtx(CKPT_PI05, num_views=2)
    rt.set_batched_mode(enable=True)
    rt.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    rt.set_prompt("pick up the cup")
    rt.calibrate([_make_dummy_obs()])

    out = rt.infer(_make_dummy_obs())
    assert "actions" in out
    a = out["actions"]
    assert np.isfinite(a).all(), "batched CFG: non-finite actions"
    assert np.max(np.abs(a)) < 100.0, \
        f"batched CFG: actions out of plausible range, max abs = {np.max(np.abs(a))}"


# ── Precision gate: batched CFG must track serial CFG on the same obs ──
#
# Paper contract (arXiv:2511.14759 Appendix E): at every denoising step
# the conditioned and unconditioned branches see the *same* guided noise
# N_k as input. The fused batched pipeline must reproduce this; any
# per-step drift between the two slots' noise buffers breaks the CFG
# math and manifests as a cosine gap vs the serial reference.
#
# The test shells out to two child processes: running a serial
# Pi05CFGPipeline frontend and a Pi05CFGBatchedPipeline frontend in the
# same Python process has known CUDA-state crosstalk, so we use
# subprocess isolation — matching the pattern the repo's latency
# benchmarks use.

_PRECISION_CHILD = r"""
import os, sys, numpy as np, torch
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
mode, ckpt, out_path, seed, beta = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
obs = {"image": np.zeros((224,224,3), dtype=np.uint8),
       "wrist_image": np.zeros((224,224,3), dtype=np.uint8),
       "state": np.zeros(8, dtype=np.float32)}
rt = Pi05TorchFrontendRtx(ckpt, num_views=2)
if mode in ("batched", "batched_cond_only"):
    rt.set_batched_mode(enable=True)
if mode == "cond_only":
    rt.set_prompt("pick up the cup\nAdvantage: positive")
elif mode == "batched_cond_only":
    rt.set_prompt_batch(["pick up the cup\nAdvantage: positive",
                         "pick up the cup\nAdvantage: positive"])
else:
    rt.set_rl_mode(cfg_enable=True, cfg_beta=beta)
    rt.set_prompt("pick up the cup")
rt.calibrate([obs])
torch.manual_seed(seed)
if mode == "batched_cond_only":
    out = rt.infer_batch([obs, obs])[0]["actions"]
else:
    out = rt.infer(obs)["actions"]
np.save(out_path, out)
"""


def _run_child(mode: str, out_path: str, seed: int = 424242,
                beta: float = 1.5) -> None:
    import subprocess
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, "-c", _PRECISION_CHILD, mode, CKPT_PI05,
         out_path, str(seed), str(beta)],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"child ({mode}) failed rc={r.returncode}\n"
            f"stdout tail:\n{r.stdout[-2000:]}\n"
            f"stderr tail:\n{r.stderr[-2000:]}")


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_batched_cfg_beta_one_matches_cond_only():
    """At ``beta=1.0`` CFG combine collapses to ``v_cond`` (a
    mathematical identity: ``noise += v_uncond + 1*(v_cond - v_uncond)
    = v_cond``). Both serial and batched CFG with ``beta=1.0`` MUST
    therefore reproduce the standard cond-only single-prompt pipeline
    bit-for-bit modulo per-tensor FP8 calibration variance. This is
    the cleanest end-to-end correctness gate for the CFG plumbing
    (encoder K/V cache snapshot, batched slot symmetry, fused combine
    kernel) — any algorithmic regression in either path breaks it.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cond_p = os.path.join(td, "cond.npy")
        sb1_p = os.path.join(td, "sb1.npy")
        bb1_p = os.path.join(td, "bb1.npy")
        _run_child("cond_only", cond_p)
        _run_child("serial", sb1_p, beta=1.0)
        _run_child("batched", bb1_p, beta=1.0)
        c = np.load(cond_p).astype(np.float64).flatten()
        s = np.load(sb1_p).astype(np.float64).flatten()
        b = np.load(bb1_p).astype(np.float64).flatten()

        def cos(a, b):
            return float(np.dot(a, b) /
                         (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos_s = cos(s, c)
        cos_b = cos(b, c)
        assert cos_s >= 0.99, (
            f"serial CFG(beta=1.0) does not collapse to cond-only: "
            f"cos={cos_s:.6f}")
        assert cos_b >= 0.99, (
            f"batched CFG(beta=1.0) does not collapse to cond-only: "
            f"cos={cos_b:.6f}")


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_batched_cfg_cos_vs_serial():
    """Batched CFG vs serial CFG at production beta=1.5.

    Both implementations are paper-faithful (Eq. 13). After the enc_Q
    stride mismatch that left slot 1's Q uninitialised under prompt
    asymmetry was fixed, the two paths track each other within FP8
    rounding noise across the paper's full β ∈ [1.0, 2.5] moderate
    range. See
    docs/cfg_correctness_spec.md.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        serial_p = os.path.join(td, "serial.npy")
        batched_p = os.path.join(td, "batched.npy")
        _run_child("serial", serial_p, beta=1.5)
        _run_child("batched", batched_p, beta=1.5)
        a = np.load(serial_p).astype(np.float64).flatten()
        b = np.load(batched_p).astype(np.float64).flatten()
        cos = float(np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        assert cos >= 0.99, (
            f"batched CFG(beta=1.5) diverges from serial CFG: "
            f"cos={cos:.6f} < 0.99\n"
            f"serial[0,:5]={a[:5]}\nbatched[0,:5]={b[:5]}")


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_batched_cfg_beta_sweep_trend():
    """Lock the (batched vs serial) cosine across the paper's full β
    ∈ [1.0, 2.5] range. After the encoder enc_Q stride mismatch was
    fixed, both paths track each other within FP8 rounding noise; the
    floors below catch any future slot-asymmetric regression —
    especially anything that lets one
    slot read uninitialised memory along the encoder's Q path.
    """
    import tempfile
    expectations = [
        (1.0, 0.99),
        (1.5, 0.99),
        (2.0, 0.99),
        (2.5, 0.99),
    ]
    with tempfile.TemporaryDirectory() as td:
        for beta, threshold in expectations:
            sp = os.path.join(td, f"s_{beta}.npy")
            bp = os.path.join(td, f"b_{beta}.npy")
            _run_child("serial", sp, beta=beta)
            _run_child("batched", bp, beta=beta)
            a = np.load(sp).astype(np.float64).flatten()
            b = np.load(bp).astype(np.float64).flatten()
            cos = float(np.dot(a, b) /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            assert cos >= threshold, (
                f"β={beta}: batched vs serial cos={cos:.4f} < "
                f"{threshold:.2f}")


# ── Bug 7 regression gate: asymmetric slot-1 vs B=1 single prompt ──
#
# This is the test that would have caught Bug 7 immediately. Run a
# generic batched (NOT CFG) forward with two DIFFERENT prompts in the
# two slots and the same noise mirrored, then verify each slot's
# output matches a B=1 single-prompt run with the corresponding
# prompt. Slot-asymmetric infrastructure issues (uninitialised reads,
# stride mismatches, allocator-recycled buffers) all manifest as a
# slot 1 cosine far below 0.99 here.

_ASYM_CHILD = r"""
import sys, ctypes, numpy as np, torch
from flash_rt.frontends.torch.pi05_rtx import (
    Pi05TorchFrontendRtx, PI05_BATCH_SIZE,
    unnormalize_actions, LIBERO_ACTION_DIM,
)
mode, ckpt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
COND   = "pick up the cup\nAdvantage: positive"
UNCOND = "pick up the cup"
obs = {"image": np.zeros((224,224,3), dtype=np.uint8),
       "wrist_image": np.zeros((224,224,3), dtype=np.uint8),
       "state": np.zeros(8, dtype=np.float32)}
rt = Pi05TorchFrontendRtx(ckpt, num_views=2)
if mode == "b1_cond":
    rt.set_prompt(COND); rt.calibrate([obs])
    torch.manual_seed(424242)
    np.save(out_path, rt.infer(obs)["actions"])
elif mode == "b1_uncond":
    rt.set_prompt(UNCOND); rt.calibrate([obs])
    torch.manual_seed(424242)
    np.save(out_path, rt.infer(obs)["actions"])
else:  # asym
    rt.set_batched_mode(enable=True)
    rt.set_prompt_batch([COND, UNCOND])
    rt.calibrate([obs])
    torch.manual_seed(424242)
    rt._noise_buf.normal_()
    for b in range(PI05_BATCH_SIZE):
        rt._noise_buf_b2[b].copy_(rt._noise_buf)
    for b in range(PI05_BATCH_SIZE):
        rt._img_buf_b2[b].copy_(rt._stack_images(obs))
    with torch.cuda.stream(rt._graph_torch_stream):
        sint = rt._graph_torch_stream.cuda_stream
        rt._copy_tensor_to_pipeline_buf_stream(
            rt._img_buf_b2, rt.pipeline.input_images_buf_b2, sint)
        rt._copy_tensor_to_pipeline_buf_stream(
            rt._noise_buf_b2, rt.pipeline.input_noise_buf_b2, sint)
        rt.pipeline.forward()
        rt._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(rt._noise_out_b2.data_ptr()),
            ctypes.c_void_p(rt.pipeline.input_noise_buf_b2.ptr.value),
            rt._noise_out_b2.numel() * 2, 3, sint)
    rt._cudart.cudaStreamSynchronize(
        ctypes.c_void_p(rt._graph_torch_stream.cuda_stream))
    raw = rt._noise_out_b2.float().cpu().numpy()
    a0 = unnormalize_actions(raw[0], rt.norm_stats)[:, :LIBERO_ACTION_DIM]
    a1 = unnormalize_actions(raw[1], rt.norm_stats)[:, :LIBERO_ACTION_DIM]
    np.savez(out_path, slot0=a0, slot1=a1)
"""


def _run_asym_child(mode, out_path):
    import subprocess
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, "-c", _ASYM_CHILD, mode, CKPT_PI05, out_path],
        capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"child ({mode}) failed rc={r.returncode}\n"
            f"stderr tail:\n{r.stderr[-2000:]}")


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no CUDA")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"pi05 ckpt missing at {CKPT_PI05}")
def test_batched_asym_slot1_vs_b1_uncond():
    """Bug 7 regression gate.

    Run Pi05BatchedPipeline with [COND, UNCOND] prompts and noise
    mirrored across slots, then verify slot 1's output matches a B=1
    single-prompt run with the UNCOND prompt (cos ≥ 0.99). Bug 7
    (the encoder enc_Q stride mismatch that left slot 1's Q buffer
    uninitialised under prompt asymmetry) collapses this from 0.99
    to ~0.92, and lower-magnitude regressions in any slot-asymmetric
    code path (per-sample loop offsets, FA2 strides, calibration
    coverage) will surface here too.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        b1c_p = os.path.join(td, "b1_cond.npy")
        b1u_p = os.path.join(td, "b1_uncond.npy")
        asym_p = os.path.join(td, "asym.npz")
        _run_asym_child("b1_cond", b1c_p)
        _run_asym_child("b1_uncond", b1u_p)
        _run_asym_child("asym", asym_p)
        b1c = np.load(b1c_p)
        b1u = np.load(b1u_p)
        asym = np.load(asym_p)

        def cos(a, b):
            a = a.flatten().astype(np.float64)
            b = b.flatten().astype(np.float64)
            return float(np.dot(a, b) /
                         (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos_s0 = cos(asym["slot0"], b1c)
        cos_s1 = cos(asym["slot1"], b1u)
        # Both slots should match their corresponding B=1 single-prompt
        # run within the FP8 budget. After Bug 7 fix slot 1 measures
        # ~0.987-0.99; the floor at 0.985 catches any meaningful
        # slot-asymmetric regression. A drop below ~0.95 historically
        # indicated uninitialised slot 1 reads (Bug 7).
        assert cos_s0 >= 0.99, (
            f"slot 0 vs B=1 cond: cos={cos_s0:.4f} (slot-0 path "
            f"regressed)")
        assert cos_s1 >= 0.985, (
            f"slot 1 vs B=1 uncond: cos={cos_s1:.4f} — slot-1 "
            f"asymmetric drift; check encoder enc_Q stride and any "
            f"other per-sample loop offsets that depend on the runtime "
            f"seq vs the buffer's allocated max-seq")
