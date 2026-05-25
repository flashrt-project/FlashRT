# Pi0.5 CFG correctness specification

This document is the **release contract** for FlashRT's classifier-free
guidance (CFG) inference paths on Pi0.5 / RECAP. It defines:

* the math each implementation must reproduce (Section 2),
* the verification framework that proves it (Section 3),
* per-condition acceptance thresholds (Section 4),
* and the long-term trajectory that lifts the current numerics to the
  release floor (Section 5).

---

## 1. Implementations covered

| Implementation | Class | Path |
|---|---|---|
| **Reference** ($\mathcal{R}_\text{fp32}$) | `Pi05CFGReference` | `flash_rt/refs/pi05_cfg_reference.py` |
| Serial CFG (FP8) | `Pi05CFGPipeline` | `flash_rt/models/pi05/pipeline_rtx_cfg.py` |
| Batched CFG (FP8, B=2) | `Pi05CFGBatchedPipeline` | `flash_rt/models/pi05/pipeline_rtx_cfg_batched.py` |

The reference implementation wraps the upstream `openpi`
`PI0Pytorch` model in a manual CFG denoising loop. It runs in BF16 /
FP32 and is the **ground truth** all FP8 paths verify against.

---

## 2. Mathematical contract

The Pi0.5 / RECAP policy (arXiv:2511.14759, Appendix E, Eq. 13) gives
the per-step CFG-guided velocity as

$$
v_\text{guided}^{(k)}(a^k; o, \ell, \beta)
   = (1 - \beta)\, v_\text{uncond}^{(k)} + \beta\, v_\text{cond}^{(k)},
\qquad k = 0, 1, \dots, T-1,
$$

with action expert outputs $v_\text{cond}^{(k)} =
f_\theta(a^k, k, o, \ell, I_t = \text{True})$ and
$v_\text{uncond}^{(k)} = f_\theta(a^k, k, o, \ell, I_t = \text{None})$.

The Euler step then advances the noise:

$$
a^{k+1} = a^k + \Delta t \cdot v_\text{guided}^{(k)},
\qquad \Delta t = -\frac{1}{T},\quad a^0 \sim \mathcal{N}(0, I).
$$

The action chunk delivered to the robot is $\hat a_{t:t+H} = a^T$.

The π0.6 paper recommends $\beta \in \{1.0\}$ as the default and
$\beta \in [1.5, 2.5]$ as the "moderate" CFG range. The release
contract below covers this entire interval.

---

## 3. Correctness conditions

Five conditions, layered. Each is independently falsifiable, and the
union $C_1 \wedge C_2 \wedge C_3 \wedge C_4 \wedge C_5$ is the
release gate.

### C1 — Combine kernel exactness

$$
\Big| \texttt{cfg\_combine\_into\_residual}(n, v_c, v_u, \beta)
   - \text{round}_\text{bf16}\!\big(n + (1-\beta)\, v_u + \beta\, v_c\big)
\Big|_\infty = 0.
$$

The bf16 round-to-nearest exactness holds because the kernel
accumulates internally in fp32 and only rounds on the final store.
Verified by comparing kernel output to a numpy fp32 reference plus
explicit bf16 round.

**Test**: `tests/test_cfg_correctness_oracle.py::test_c1_cfg_combine_kernel_matches_fp32`.

### C2 — Per-step noise trajectory ≈ FP32 reference

For each implementation $\Pi \in \{\text{serial}, \text{batched}\}$,
each $\beta$, each step $k$:

$$
\cos\!\big\langle a^{(k)}_{\Pi},\, a^{(k)}_{\mathcal{R}_\text{fp32}} \big\rangle \geq \tau_\text{noise}(\Pi, \beta).
$$

Why noise (and not $v_\text{cond}, v_\text{uncond}$ directly):
FlashRT's `decoder_action_buf` is the raw action-projection output
**after** absorbing the schedule's $-\Delta t$ scale into the output
weight (see `pipeline_rtx.py`'s `out_proj` pre-scale). The reference
returns the unscaled $v_t$ from `denoise_step`. So per-step `v` differs
in sign and scale even when both implementations are mathematically
identical. The integrated noise state $a^{(k)}$ is the
implementation-independent physical quantity — both paths integrate
the same Euler update, and per-step noise tells us exactly when and
how fast their trajectories diverge.

C2 is the workhorse oracle: it pinpoints **at which step** divergence
appears, decoupling the diffusion accumulator from any single-kernel
defect.

**Test**: `tests/test_cfg_correctness_oracle.py::test_c2_per_step_noise_vs_ref`.

### C3 — Per-slot end-to-end (batched only)

For batched CFG with prompts $(p_0, p_1) = (\text{cond}, \text{uncond})$
and identical noise mirrored to both slots:

$$
\cos\!\big\langle \text{slot}_b\!\big[\hat a_\text{batched}((p_0, p_1), \omega)\big],\, \hat a_{B=1}(p_b, \omega) \big\rangle \geq 0.99,
\qquad b \in \{0, 1\}.
$$

This decouples slot symmetry from the CFG combine. Currently
slot 0 satisfies C3 (cos ≥ 0.999); slot 1 currently does **not**
(measured cos ≈ 0.92 — the M2 fix target).

**Test**: existing `tests/test_pi05_batched_precision.py` (slot 0)
plus an asymmetric-prompt smoke check for slot 1. The slot-1 case is
tracked as a candidate public `tests/` gate.

### C4 — CFG identity at $\beta = 1$

When $\beta = 1$ the combine collapses:
$v_\text{guided} = v_\text{uncond} + 1 \cdot (v_\text{cond} -
v_\text{uncond}) = v_\text{cond}$. So both serial and batched CFG with
$\beta = 1$ must reproduce the standard cond-only single-prompt
pipeline modulo per-tensor FP8 calibration variance:

$$
\cos\!\big\langle \hat a_\Pi(\beta = 1, \omega),\, \hat a_\text{cond-only}(p_\text{cond}, \omega) \big\rangle \geq 0.999.
$$

This is the cleanest plumbing-correctness gate: any algorithmic
regression in the encoder K/V cache snapshot, the slot-symmetry of
the batched path, the noise mirror, or the fused combine kernel
breaks it.

**Test**: `tests/test_pi05_cfg_batched_inference.py::test_batched_cfg_beta_one_matches_cond_only`.

### C5 — Full $\beta$-range end-to-end vs reference

For each $\Pi \in \{\text{serial}, \text{batched}\}$,
$\beta \in \{1.0, 1.5, 2.0, 2.5\}$:

$$
\cos\!\big\langle \hat a_\Pi(\beta, \omega),\, \hat a_{\mathcal{R}_\text{fp32}}(\beta, \omega) \big\rangle \geq \tau_\text{act}(\Pi, \beta).
$$

The release floor is $\tau_\text{act} = 0.99$ for all
$(\Pi, \beta)$. Current measurements lock per-cell floors below this
(see Section 4) — those are **regression catchers**, not release
signals.

**Test**: `tests/test_cfg_correctness_oracle.py::test_c5_actions_vs_fp32_reference`.

---

## 4. Threshold table

Live values are kept in
`tests/test_cfg_correctness_oracle.py` next to each test; this table
is a snapshot.

### C2 — per-step noise cosine floor (min over $k$)

| $\beta$ | serial | batched |
|---|---|---|
| 1.0 | 0.99 | 0.99 |
| 1.5 | 0.98 | 0.98 |
| 2.0 | 0.97 | 0.97 |
| 2.5 | 0.95 | 0.95 |

### C5 — final-action cosine floor vs FP32 reference

| $\beta$ | serial | batched |
|---|---|---|
| 1.0 | 0.99 | 0.99 |
| 1.5 | 0.98 | 0.99 |
| 2.0 | 0.97 | 0.98 |
| 2.5 | 0.96 | 0.97 |

After the encoder enc_Q stride mismatch that left slot 1's Q buffer
uninitialised under prompt asymmetry was fixed, the batched path tracks
the FP32 reference at least as tightly as serial across the full paper
β range. Both paths share the FP8 quantisation budget; the per-β floors
above pin the budget without any path-specific carve-out.

---

## 5. Trajectory to release (M2 → M5)

| Milestone | Goal | Status |
|---|---|---|
| **M1** | Oracle infrastructure: $\mathcal{R}_\text{fp32}$, per-step probes, fixture, C1–C5 tests | ✅ done |
| **M2** | Locate slot-1 / FP8 root cause and fix | ✅ done — Bug 7 (encoder enc_Q stride mismatch). Batched-vs-serial cosine at β=2.5 jumped from 0.881 to 0.997; batched-vs-FP32-ref at β=2.5 from 0.895 to 0.976. |
| **M3** | Tighten remaining gaps | ✅ done — `Pi05CFGBatchedPipeline.__init__` precision warning removed; thresholds aligned across paths. |
| **M4** | Documentation + release prep | ✅ done — `docs/rl_inference.md` updated; release notes in repo. |
| **M5** | Real-LIBERO validation | pending — C5 with real obs ≥ 0.999 |

The M2 diagnostic plan covered the root-cause hypothesis tree,
per-step probes, sanitizer runs, and autotune experiments; the public
result is summarized in the milestones above.

---

## 6. Reproducing the fixtures

```
docker exec <container> bash -c \
  "cd <repo_root> && \
   python tools/generate_cfg_oracle_fixtures.py"
```

This populates `tests/fixtures/cfg_reference_outputs.npz` with
12 (path × β) × 4 (actions / v_cond / v_uncond / noise) = 48 arrays.
Regenerate after any change that legitimately affects per-tensor
numerics (FP8 calibration, weights, kernel fusion semantics).
