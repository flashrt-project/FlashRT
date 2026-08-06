#!/usr/bin/env python3
"""Pi0.5 Thor 1-view / 2-view / 3-view latency + cos benchmark.

Populates the ``Pi0.5 Thor (SM110)`` 1v/2v/3v table in the README,
matching the existing ``Pi0.5 RTX 5090`` table structure.

Matrix (per view count):
  Frontend          Calibration     Precision
  ----------------  --------------  ---------
  Pi0.5 Torch FP4   N=8 LIBERO       NVFP4 encoder (17 live layers + AWQ + P1)
  Pi0.5 Torch FP8   N=8 LIBERO       FP8 baseline
  Pi0.5 JAX  FP4    N=8 LIBERO       NVFP4 encoder (17 live layers + AWQ + P1)
  Pi0.5 JAX  FP8    N=1 (lazy)       FP8 baseline  [JAX FP8 does not
                                                     support N>=2 dataset
                                                     calibrate today]

Reports per cell:
  calibrate_ms  - wall-clock of the calibrate() call (0 for lazy N=1)
  p50_ms        - median of 50 CUDA-graph replays (matches the README
                   RTX table's methodology)
  p95_ms        - 95th percentile
  cos_vs_fp32_ref - 3v only (we only have a 3-view PyTorch FP32 reference)

Each cell runs in its own Python subprocess (Thor resource model will
not accept multiple 7 GB VLA weight loads in one process).

Usage::

    python3 tests/bench_pi05_thor_views.py
    python3 tests/bench_pi05_thor_views.py --views 2 3          # subset
    python3 tests/bench_pi05_thor_views.py --frontends torch_fp4  # one row
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bench_pi05_views")

ROOT = Path(__file__).resolve().parents[1]

PI05_TORCH_CKPT = os.environ.get("PI05_CKPT", "/workspace/pytorch_checkpoints/pi05_libero_converted")
PI05_JAX_CKPT = os.environ.get("PI05_JAX_CKPT", "/workspace/checkpoints/pi05_libero_jax")


TORCH_SCRIPT = r"""
import sys, json, time, pathlib, numpy as np, torch
sys.path.insert(0, "ROOTDIR")

# Clear calibration cache so the reported calibrate_ms is a real
# forward-pass cost, not a disk read.
cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

# Load per-view-count LIBERO obs bundle.
d = np.load(f"/tmp/libero_obs_{NV}v_n8.npz")
obs_list = []
for i in range(int(d["n"])):
    o = {"image": d[f"img_{i}"], "state": d[f"state_{i}"]}
    if NV >= 2: o["wrist_image"]       = d[f"wrist_{i}"]
    if NV >= 3: o["wrist_image_right"] = d[f"wrist_right_{i}"]
    obs_list.append(o)

# Build frontend
if USE_FP4:
    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4
    pipe = Pi05TorchFrontendThorFP4(
        CKPT, num_views=NV, autotune=3,
        use_fp4_encoder_ffn=True, fp4_layers=tuple(range(17)),
        use_awq=True, use_p1_split_gu=True)
else:
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
    pipe = Pi05TorchFrontendThor(CKPT, num_views=NV, autotune=3)

# 3v uses pytorch_reference.npz tokens + noise to allow cos vs FP32 ref.
# 1v/2v have no FP32 ref so we use a plain string prompt; cos is not
# reported for those.
if NV == 3:
    ref = np.load("/tmp/pytorch_reference.npz", allow_pickle=True)
    tok_mask = ref["arg8_tokenized_prompt_mask"][0]
    tokens = ref["arg7_tokenized_prompt"][0][:int(tok_mask.sum())].astype(np.int64)
    pipe.set_prompt(tokens.tolist())
    deploy_obs = {
        "image":             ref["arg0_base_rgb"][0],
        "wrist_image":       ref["arg1_left_wrist_rgb"][0],
        "wrist_image_right": ref["arg2_right_wrist_rgb"][0],
    }
    ref_raw   = ref["pytorch_raw_output"][0].astype(np.float32)
    ref_noise = ref["arg9_noise"][0].astype(np.float16)
else:
    pipe.set_prompt("pick up the red block and place it in the tray")
    deploy_obs = obs_list[0]   # use a LIBERO obs as deployment
    ref_raw = None
    ref_noise = None

# Calibrate with 8 LIBERO samples.
t0 = time.perf_counter()
pipe.calibrate(obs_list, percentile=99.9)
cal_ms = (time.perf_counter() - t0) * 1000

# Warmup (Graph capture happens here implicitly in first infer).
for _ in range(5): pipe.infer(deploy_obs)

# 3v: measure cos vs FP32 reference using matched noise.
# Pi0.5 torch frontend (the current Pi0.5 torch frontend) draws _g_noise via
# np.random.randn(Sa, 32) on CPU then H2D-copies; the legacy
# torch.Tensor.normal_ monkey-patch no longer fires. Mirror the JAX
# path's _PatchedRNG pattern (lines below).
cos_ref = None; maxdiff = None
if NV == 3 and ref_raw is not None:
    Sa = ref_noise.shape[0]
    _orig_randn = np.random.randn
    class _PatchedRNG:
        on = False
        def __call__(self, *a, **kw):
            if self.on and a == (Sa, 32):
                return ref_noise.astype(np.float64)
            return _orig_randn(*a, **kw)
    p = _PatchedRNG()
    np.random.randn = p
    p.on = True
    pipe.infer(deploy_obs)
    p.on = False
    np.random.randn = _orig_randn
    out = pipe._g_noise.float().cpu().numpy()
    a, b = out.flatten().astype(np.float64), ref_raw.flatten().astype(np.float64)
    cos_ref = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    maxdiff = float(np.max(np.abs(out - ref_raw)))

# Latency: 50 CUDA-graph replays. Match the README's RTX table
# (--warmup 50 --iters 500) approach of reporting replay P50 + P95.
lat = []
for _ in range(50):
    t0 = time.perf_counter()
    pipe.infer(deploy_obs)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
p50 = lat[25]
p95 = lat[47]

out = {
    "frontend": ("torch_fp4" if USE_FP4 else "torch_fp8"),
    "num_views": NV,
    "calibrate_ms": round(cal_ms, 1),
    "p50_ms": round(p50, 2),
    "p95_ms": round(p95, 2),
}
if cos_ref is not None: out["cos_vs_fp32_ref"] = round(cos_ref, 6)
if maxdiff is not None: out["maxdiff_vs_ref"] = round(maxdiff, 4)
print("__RESULT__ " + json.dumps(out))
"""


JAX_SCRIPT = r"""
import sys, json, time, pathlib, numpy as np
sys.path.insert(0, "ROOTDIR")

# JAX Pi0.5: FP8 (N=1 implicit) or FP4 (N=8 LIBERO multi-frame).
# Clear cache so calibrate_ms reflects a real forward-pass cost.
cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

d = np.load(f"/tmp/libero_obs_{NV}v_n8.npz")
obs_list = []
for i in range(int(d["n"])):
    o = {"image": d[f"img_{i}"], "state": d[f"state_{i}"]}
    # The JAX Pi0.5 frontend's infer() reads wrist_image / wrist_image_right
    # unconditionally — duplicate the base image when the view count is
    # smaller so the 1v / 2v bench cells work through the same interface.
    o["wrist_image"]       = d[f"wrist_{i}"] if NV >= 2 else o["image"]
    o["wrist_image_right"] = (d[f"wrist_right_{i}"] if NV >= 3
                              else o["wrist_image"])
    obs_list.append(o)
deploy_obs = dict(obs_list[0])

if USE_FP4:
    from flash_rt.frontends.jax.pi05_thor_fp4 import Pi05JaxFrontendThorFP4
    pipe = Pi05JaxFrontendThorFP4(
        JAX_CKPT, num_views=NV, autotune=3, weight_cache=True,
        use_fp4_encoder_ffn=True, fp4_layers=tuple(range(17)),
        use_awq=True, use_p1_split_gu=True)
else:
    from flash_rt.frontends.jax.pi05_thor import Pi05JaxFrontendThor
    pipe = Pi05JaxFrontendThor(JAX_CKPT, num_views=NV, autotune=3)

# 3v matches pytorch_reference.npz so we can report cos.
if NV == 3:
    ref = np.load("/tmp/pytorch_reference.npz", allow_pickle=True)
    tok_mask = ref["arg8_tokenized_prompt_mask"][0]
    tokens = ref["arg7_tokenized_prompt"][0][:int(tok_mask.sum())].astype(np.int64)
    pipe.set_prompt(tokens.tolist())
    deploy_obs = {
        "image":             ref["arg0_base_rgb"][0],
        "wrist_image":       ref["arg1_left_wrist_rgb"][0],
        "wrist_image_right": ref["arg2_right_wrist_rgb"][0],
    }
    ref_raw   = ref["pytorch_raw_output"][0].astype(np.float32)
    ref_noise = ref["arg9_noise"][0].astype(np.float16)
else:
    pipe.set_prompt("pick up the red block and place it in the tray")
    ref_raw = None
    ref_noise = None

# FP4: N=8 LIBERO multi-frame; FP8: N=1 implicit (JAX FP8 path lacks N>=2).
t0 = time.perf_counter()
pipe.calibrate(obs_list if USE_FP4 else [deploy_obs], percentile=99.9)
cal_ms = (time.perf_counter() - t0) * 1000

for _ in range(5): pipe.infer(deploy_obs)

cos_ref = None; maxdiff = None
if NV == 3 and ref_raw is not None:
    _orig_randn = np.random.randn
    class _PatchedRNG:
        on = False
        def __call__(self, *a, **kw):
            if self.on and a == (10, 32):
                return ref_noise.astype(np.float64)
            return _orig_randn(*a, **kw)
    p = _PatchedRNG()
    np.random.randn = p
    p.on = True
    pipe.infer(deploy_obs)
    p.on = False
    np.random.randn = _orig_randn
    out = pipe.g_noise.download_new((10, 32), np.float16).astype(np.float32)
    a, b = out.flatten().astype(np.float64), ref_raw.flatten().astype(np.float64)
    cos_ref = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    maxdiff = float(np.max(np.abs(out - ref_raw)))

lat = []
for _ in range(50):
    t0 = time.perf_counter()
    pipe.infer(deploy_obs)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
p50 = lat[25]; p95 = lat[47]

out = {
    "frontend": ("jax_fp4" if USE_FP4 else "jax_fp8"),
    "num_views": NV,
    "calibrate_ms": round(cal_ms, 1),
    "p50_ms": round(p50, 2),
    "p95_ms": round(p95, 2),
}
if cos_ref is not None: out["cos_vs_fp32_ref"] = round(cos_ref, 6)
if maxdiff is not None: out["maxdiff_vs_ref"] = round(maxdiff, 4)
print("__RESULT__ " + json.dumps(out))
"""


def _run(*, frontend: str, nv: int, ckpt: str) -> dict:
    if frontend.startswith("torch"):
        body = TORCH_SCRIPT
        use_fp4 = (frontend == "torch_fp4")
        header = (
            f"CKPT = {ckpt!r}\n"
            f"NV = {int(nv)}\n"
            f"USE_FP4 = {bool(use_fp4)}\n"
        )
    elif frontend.startswith("jax"):
        body = JAX_SCRIPT
        use_fp4 = (frontend == "jax_fp4")
        header = (
            f"JAX_CKPT = {ckpt!r}\n"
            f"NV = {int(nv)}\n"
            f"USE_FP4 = {bool(use_fp4)}\n"
        )
    else:
        raise ValueError(f"unknown frontend {frontend!r}")

    script = header + body.replace("ROOTDIR", str(ROOT))
    r = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    return {
        "frontend": frontend, "num_views": nv, "error": (
            f"no __RESULT__; stderr tail: "
            + "\n".join(r.stderr.splitlines()[-8:]))
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--frontends", nargs="+",
                    default=["torch_fp4", "torch_fp8", "jax_fp4"])
    args = ap.parse_args()

    rows = []
    for fr in args.frontends:
        for nv in args.views:
            logger.info("── %s  num_views=%d ──", fr, nv)
            ckpt = (PI05_JAX_CKPT if fr.startswith("jax") else PI05_TORCH_CKPT)
            r = _run(frontend=fr, nv=nv, ckpt=ckpt)
            logger.info("  %s", json.dumps(r))
            rows.append(r)

    # Summary table.
    print("\n" + "=" * 95)
    print(f"{'frontend':>10}  {'nv':>3}  {'calibrate':>10}  "
          f"{'p50':>9}  {'p95':>9}  {'cos_ref':>10}  {'maxdiff':>8}")
    print("-" * 95)
    for r in rows:
        if "error" in r:
            print(f"{r.get('frontend','?'):>10}  {r.get('num_views','?'):>3}  "
                  f"ERROR  {r['error'][:60]}")
            continue
        cos = r.get("cos_vs_fp32_ref", "-")
        mdf = r.get("maxdiff_vs_ref", "-")
        cos_s = f"{cos:.6f}" if isinstance(cos, float) else str(cos)
        mdf_s = f"{mdf:.4f}" if isinstance(mdf, float) else str(mdf)
        print(f"{r['frontend']:>10}  {r['num_views']:>3}  "
              f"{r['calibrate_ms']:>8.1f} ms  "
              f"{r['p50_ms']:>6.2f} ms  {r['p95_ms']:>6.2f} ms  "
              f"{cos_s:>10}  {mdf_s:>8}")
    print("=" * 95)
    print(json.dumps(rows, indent=2))

    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
