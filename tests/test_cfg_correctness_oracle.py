"""C1-C5 oracle tests for Pi0.5 CFG correctness.

These tests anchor FlashRT's serial / batched FP8 CFG implementations
against an independent FP32 PyTorch reference (R_fp32, see
``flash_rt/refs/pi05_cfg_reference.py``) and against the fused
combine kernel's mathematical contract. The full test hierarchy is
documented in ``docs/precision_spec.md``; below is a summary.

  C1 — kernel-level: cfg_combine_into_residual matches FP32 numpy
       reference elementwise (max_abs_diff = 0).
  C2 — per-step velocity component-wise: each implementation's per-step
       v_cond / v_uncond match R_fp32 within FP8 budget (cosine ≥ τ_v).
  C3 — per-slot end-to-end: batched slot 0 / slot 1 match B=1
       single-prompt runs (lives in test_pi05_batched_precision.py;
       this file's C3 gate measures the per-slot fidelity from the
       trace).
  C4 — CFG identity at β=1: v_guided = v_cond ⇒ end-to-end action
       chunk equals cond-only single-prompt pipeline (lives in
       test_pi05_cfg_batched_inference.py::test_batched_cfg_beta_one_*).
  C5 — full β-range vs R_fp32: cosine ≥ τ_act for β ∈ {1.0, 1.5, 2.0,
       2.5}.

The fixture file backing these tests is
``tests/fixtures/cfg_reference_outputs.npz`` — regenerate it via
``tools/generate_cfg_oracle_fixtures.py`` after any change that
legitimately affects per-tensor numerics. Tests load fixture data
without needing GPU access; the actual model runs are baked at
generation time.
"""

import os
from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cfg_reference_outputs.npz"
_FIXTURES_AVAILABLE = FIXTURE_PATH.exists()

BETAS = [1.0, 1.5, 2.0, 2.5]

# Per-correctness-condition cosine thresholds. Values are calibrated
# from the current main-branch numerics with margin; tightening them
# is the explicit goal of the M2 / M3 milestones (see
# docs/precision_spec.md). Lowering them is a release blocker.
TAU_C1_MAX_ABS_DIFF = 0.0          # bf16 round-to-nearest exact

# C2 — per-step noise trajectory cosine vs R_fp32.
# Why noise (and not v_cond / v_uncond)? FlashRT's per-step
# ``decoder_action_buf`` carries the velocity AFTER the model's
# output projection has folded the diffusion schedule's ``-dt``
# scale (``out_proj`` weights pre-scaled by ``-1/num_steps``); the
# upstream openpi reference returns the raw ``v_t`` from
# ``denoise_step``. So per-step v differs by both sign and scale
# even when both implementations are mathematically identical, and
# cosine over those raw values is meaningless. The per-step noise
# state ``a^k`` is the physically-grounded shared quantity: each
# implementation integrates ``a^{k+1} = a^k + dt · v_guided^{(k)}``
# regardless of how ``v`` is internally represented.
#
# The thresholds below are min-over-step cosines; FP8 error
# accumulates monotonically with k (k=0 is always 1.0 because that
# is the externally-staged initial noise byte-for-byte). The floor
# at the final step (k=num_steps) is what we lock.
TAU_C2_NOISE_PER_STEP = {
    # (path, beta) -> per-step floor (min over k). Margin ≈ 0.005 below
    # the measured value at fixture-generation time. After the enc_Q
    # stride mismatch was fixed, batched per-step noise tracking matches
    # serial within FP8 rounding noise across the entire paper
    # β ∈ [1.0, 2.5] range.
    ("serial", 1.0):  0.99,  ("serial", 1.5):  0.98,
    ("serial", 2.0):  0.97,  ("serial", 2.5):  0.95,
    ("batched", 1.0): 0.99,  ("batched", 1.5): 0.98,
    ("batched", 2.0): 0.97,  ("batched", 2.5): 0.95,
}

TAU_C5_ACTIONS_VS_REF = {
    # (path, beta) -> cosine floor for end-to-end vs FP32 ref.
    # Both serial and batched are FP8 paths; their cosine vs the FP32
    # reference is ultimately bounded by the FP8 quantisation budget
    # accumulated over the 10 denoising steps and amplified by
    # (β - 1). Floors below set the SAME cap for both paths at each
    # β — after Bug 7's fix, batched now tracks the FP32 reference
    # at least as tightly as serial across the full paper range.
    ("serial", 1.0):  0.99, ("serial", 1.5):  0.98,
    ("serial", 2.0):  0.97, ("serial", 2.5):  0.96,
    ("batched", 1.0): 0.99, ("batched", 1.5): 0.99,
    ("batched", 2.0): 0.98, ("batched", 2.5): 0.97,
}


@pytest.fixture(scope="module")
def fixtures():
    if not _FIXTURES_AVAILABLE:
        pytest.skip(
            f"oracle fixtures not generated — run "
            f"tools/generate_cfg_oracle_fixtures.py first")
    return dict(np.load(FIXTURE_PATH))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    n = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / n)


# ──────────────────────────────────────────────────────────────────
# C1 — kernel: cfg_combine_into_residual
# ──────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    "FLASHVLA_GPU_TESTS" not in os.environ
    or not os.environ.get("FLASHVLA_GPU_TESTS"),
    reason="C1 needs CUDA + flash_rt_kernels extension")
def test_c1_cfg_combine_kernel_matches_fp32():
    """C1: residual += v_uncond + β·(v_cond - v_uncond), in BF16, vs FP32 ref.

    Kernel order (csrc/kernels/elementwise.cu cfg_combine_kernel):
        bf16 inputs → upcast fp32 → g = vu + β·(vc - vu) → r + g → bf16 store
    The reference here mirrors that exact order and starts from bf16-quantized
    inputs so the only remaining slack is the final round-to-nearest, which
    the kernel does in hardware. ``r + u + β(c-u)`` is mathematically equal
    to the paper-form ``r + (1-β)u + βc`` but the two FP32 evaluation paths
    differ by ~1 fp32 ULP, which can flip the final bf16 rounding by 1 bf16
    ULP — that is what we need to match against, not the paper form.
    """
    import torch

    from flash_rt import flash_rt_kernels as fvk

    n = 320
    rng = np.random.default_rng(0)
    residual_np = rng.standard_normal(n).astype(np.float32)
    v_cond_np = rng.standard_normal(n).astype(np.float32)
    v_uncond_np = rng.standard_normal(n).astype(np.float32)
    beta = 1.5

    # Quantize inputs to bf16 first (kernel sees bf16). Reference then runs
    # in the kernel's operation order on those upcast fp32 values, with a
    # single bf16 round-to-nearest at the store.
    def _to_bf16_to_fp32(x):
        return torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()
    r32 = _to_bf16_to_fp32(residual_np)
    c32 = _to_bf16_to_fp32(v_cond_np)
    u32 = _to_bf16_to_fp32(v_uncond_np)
    g = u32 + np.float32(beta) * (c32 - u32)
    ref = r32 + g
    ref_bf16 = (
        torch.from_numpy(ref).to(torch.bfloat16).to(torch.float32).numpy())

    res_t = torch.from_numpy(residual_np).to(
        device="cuda", dtype=torch.bfloat16)
    vc_t = torch.from_numpy(v_cond_np).to(
        device="cuda", dtype=torch.bfloat16)
    vu_t = torch.from_numpy(v_uncond_np).to(
        device="cuda", dtype=torch.bfloat16)
    fvk.cfg_combine_into_residual(
        res_t.data_ptr(), vc_t.data_ptr(), vu_t.data_ptr(),
        beta, n, 0)
    torch.cuda.synchronize()
    out = res_t.to(torch.float32).cpu().numpy()

    diff = np.max(np.abs(out - ref_bf16))
    assert diff <= TAU_C1_MAX_ABS_DIFF, (
        f"cfg_combine_into_residual diverges from bf16 ref: max_abs={diff}")


# ──────────────────────────────────────────────────────────────────
# C2 — per-step velocity component-wise (FP8 path vs FP32 ref)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("path", ["serial", "batched"])
def test_c2_per_step_noise_vs_ref(fixtures, beta, path):
    """C2: per-step noise trajectory cosine vs R_fp32.

    Compares each step's diffusion noise state ``a^k`` between
    FlashRT's FP8 path and the FP32 reference. Both implementations
    integrate ``a^{k+1} = a^k + dt · v_guided^{(k)}`` from the same
    initial noise; deviation grows monotonically with k as FP8 / FP8
    cuBLASLt rounding accumulates. ``cos(ref[0], impl[0])`` is always
    ≈ 1.0 (the initial noise is staged externally byte-for-byte —
    a sanity check on noise injection, not on math). The locked
    threshold is the floor over all k.
    """
    tag = f"b{beta:.1f}"
    ref = fixtures[f"ref_noise_{tag}"]      # (T+1, H, D)
    impl = fixtures[f"{path}_noise_{tag}"]  # (T+1, H, D)
    assert ref.shape == impl.shape, (
        f"shape mismatch: ref={ref.shape}, impl={impl.shape}")
    per_step_cos = [_cos(ref[k], impl[k]) for k in range(ref.shape[0])]
    # Sanity: initial noise is staged externally, must be byte-equal.
    assert per_step_cos[0] >= 0.999, (
        f"{path} β={beta}: initial noise (k=0) cos={per_step_cos[0]:.4f}; "
        f"the noise injection stage diverged before any decoder kernel "
        f"ran — fix the fixture generator before reading further C2 "
        f"signal")
    threshold = TAU_C2_NOISE_PER_STEP[(path, beta)]
    worst = float(np.min(per_step_cos))
    assert worst >= threshold, (
        f"{path} β={beta}: min per-step noise cos={worst:.4f} < "
        f"{threshold:.4f}\n  per-step cosines = "
        f"{[f'{c:.3f}' for c in per_step_cos]}")


# ──────────────────────────────────────────────────────────────────
# C5 — full β-range end-to-end vs FP32 ref
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("path", ["serial", "batched"])
def test_c5_actions_vs_fp32_reference(fixtures, beta, path):
    """C5: end-to-end action chunk cosine vs FP32 reference.

    The release contract is ``cos ≥ 0.99 ∀ β ∈ [1.0, 2.5]`` for both
    paths (see docs/precision_spec.md). The current per-(path, β)
    floors below sit at the measured numerics — they catch
    regressions but do NOT signal release readiness.
    """
    tag = f"b{beta:.1f}"
    ref = fixtures[f"ref_actions_{tag}"]
    impl = fixtures[f"{path}_actions_{tag}"]
    cos = _cos(ref, impl)
    threshold = TAU_C5_ACTIONS_VS_REF[(path, beta)]
    assert cos >= threshold, (
        f"{path} β={beta}: actions vs FP32 ref cos={cos:.4f} < "
        f"{threshold:.4f}")


# ──────────────────────────────────────────────────────────────────
# Inter-implementation gate (regression catcher)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("beta", BETAS)
def test_serial_vs_batched_consistency(fixtures, beta):
    """Cross-check serial vs batched at each β. Locks the current
    measured cosine and catches future implementation drift even when
    both paths drift together (which the per-vs-ref tests would miss).
    """
    tag = f"b{beta:.1f}"
    s = fixtures[f"serial_actions_{tag}"]
    b = fixtures[f"batched_actions_{tag}"]
    cos = _cos(s, b)
    # Same floors as C5 batched (which is the harder path).
    floor = max(TAU_C5_ACTIONS_VS_REF[("batched", beta)] - 0.02, 0.85)
    assert cos >= floor, (
        f"serial-vs-batched β={beta}: cos={cos:.4f} < {floor:.4f}")
