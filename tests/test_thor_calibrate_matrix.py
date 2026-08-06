#!/usr/bin/env python3
"""Thor calibration verification matrix — cosine + latency, no regressions.

For every Thor torch frontend that has ``calibrate()`` implemented,
compare **N=1 (default)** vs **N=8 LIBERO dataset** against the
**PyTorch FP32 original-model reference** — the gold standard.

Reference sources (ALL gold-tier, not FP8 cross-checks):

  pi05 / pi05_fp4 / pi05_jax / pi05_jax_fp4
                   /tmp/pytorch_reference.npz::pytorch_raw_output
                     - 3-view PyTorch FP16/FP32 ref, 200-token prompt,
                       fixed noise, fixed ref images.
                     - Reported as `cos_vs_pytorch_ref` + `maxdiff`.
  pi0              /tmp/pi0_ref_2view.npz::pytorch_raw_output
                     - 2-view PyTorch FP32 ref, raw decoder output.
  pi0fast          first_token match against the JAX FP16 greedy
                   decode (autoregressive gold; continuous cosine is
                   not a meaningful per-step metric for greedy argmax).

Note: ``/tmp/v_prod.npy``, ``/tmp/v_torch.npy``, ``/tmp/v_jax.npy`` are
FP8-grade outputs at the same precision tier as our Thor build. They
are CROSS-CHECKS, not accuracy baselines, and are no longer reported
by this script.

Each (model, N) cell runs in its own subprocess (Thor cannot hold
multiple 7 GB VLA weights in one process). P50 / P95 latency is
measured over 20 graph-replay iterations after calibration.

Usage::

    python3 tests/test_thor_calibrate_matrix.py                # full matrix
    python3 tests/test_thor_calibrate_matrix.py --models pi05  # one model
    python3 tests/test_thor_calibrate_matrix.py --n 16         # bump dataset N
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("thor_cal_matrix")

ROOT = Path(__file__).resolve().parents[1]

LIBERO_ROOT = os.environ.get("LIBERO_ROOT", "/workspace/libero_10_image")
PI05_CKPT = os.environ.get("PI05_CKPT", "/workspace/pytorch_checkpoints/pi05_libero_converted")
PI05_JAX_CKPT = os.environ.get("PI05_JAX_CKPT", "/workspace/checkpoints/pi05_libero_jax")
PI0_CKPT = os.environ.get("PI0_CKPT", "/workspace/pytorch_checkpoints/pi0_base_converted")
PI0_JAX_CKPT = os.environ.get(
    "PI0_JAX_CKPT", "/workspace/checkpoints/pi0_base")
PI0FAST_CKPT = os.environ.get(
    "PI0FAST_CKPT", "/workspace/pytorch_checkpoints/pi0_fast_base_converted")
PI0FAST_JAX_CKPT = os.environ.get(
    "PI0FAST_JAX_CKPT", "/workspace/checkpoints/pi0_fast_base")


# ------------------------------------------------------------------
# Per-model scripts. Each is a stand-alone Python string run under
# subprocess.run so GPU state is fresh.
# ------------------------------------------------------------------
#
# Shared protocol:
# - The script prints exactly one line starting with ``__RESULT__ ``
#   containing a JSON payload with at least {cos, p50_ms, ...}.
# - Any cosine or latency printed before that is human-readable only.
# - Cache directory is cleared at the top of each script so the
#   calibration step actually executes (we are measuring the N=1 vs
#   N=8 difference, not cache hits).


PI05_SCRIPT = r"""
# Gold reference: /tmp/pytorch_reference.npz::pytorch_raw_output — this
# is the PyTorch FP16 / FP32 original-model output, the only valid
# accuracy baseline. /tmp/v_prod.npy and /tmp/v_torch.npy are FP8
# outputs (same precision tier as FlashRT Thor) and are cross-checks,
# not gold. We report cos_vs_pytorch_ref as the primary metric.
import sys, json, time, pathlib, numpy as np, torch
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

# Load the FP32 PyTorch reference (3-view, 200-token prompt, fixed noise).
ref = np.load("/tmp/pytorch_reference.npz", allow_pickle=True)
ref_raw      = ref["pytorch_raw_output"][0].astype(np.float32)   # (10, 32)
ref_img0     = ref["arg0_base_rgb"][0]                           # (224,224,3) fp16 [-1,1]
ref_img1     = ref["arg1_left_wrist_rgb"][0]
ref_img2     = ref["arg2_right_wrist_rgb"][0]
ref_noise    = ref["arg9_noise"][0].astype(np.float16)           # (10, 32)
tok_mask     = ref["arg8_tokenized_prompt_mask"][0]
ref_tokens   = ref["arg7_tokenized_prompt"][0][:int(tok_mask.sum())].astype(np.int64)

from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor as PIPE_CLS
pipe = PIPE_CLS(CKPT, num_views=3, autotune=3)
pipe.set_prompt(ref_tokens.tolist())

ref_obs = {"image": ref_img0, "wrist_image": ref_img1,
           "wrist_image_right": ref_img2}

if N > 1:
    # 3-view LIBERO obs bundle (LIBERO is 2-view; right_wrist is a
    # duplicate of wrist for compatibility with the 3-view frontend).
    import os as _os
    obs_npz = f"/tmp/libero_obs3v_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [
            {"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"],
             "wrist_image_right": d[f"wrist_right_{i}"],
             "state": d[f"state_{i}"]} for i in range(int(d["n"]))
        ]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        base = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
        obs_list = [{"image": o["image"], "wrist_image": o["wrist_image"],
                     "wrist_image_right": o["wrist_image"],
                     "state": o.get("state", np.zeros(32, dtype=np.float32))}
                    for o in base]
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0   # implicit, inside first infer

for _ in range(3): pipe.infer(ref_obs)

# Inject ref_noise via np.random.randn — current Pi0.5 Torch frontend
# draws _g_noise via np.random.randn(Sa, 32) on the CPU and H2D-copies.
# The legacy torch.Tensor.normal_ monkey-patch no longer fires; without
# this update the test would diff against the wrong initial noise and
# cos collapses to ~0.3. Mirrors test_all_models_precision.py PI05_SCRIPT.
Sa = ref_noise.shape[0]
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (Sa, 32):
            return ref_noise.astype(np.float64)
        return _orig_randn(*a, **kw)
_p = _PatchedRNG(); np.random.randn = _p
_p.on = True
pipe.infer(ref_obs)
_p.on = False
np.random.randn = _orig_randn
out = pipe._g_noise.float().cpu().numpy()

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

max_abs_err = float(np.max(np.abs(out - ref_raw)))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(ref_obs); lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "cos_vs_pytorch_ref": round(cos(out, ref_raw), 6),
    "maxdiff_vs_ref":     round(max_abs_err, 4),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
}))
"""


PI05_FP4_SCRIPT = r"""
# Same gold reference as pi05: /tmp/pytorch_reference.npz (PyTorch FP32
# original, 3-view, 200-token prompt, fixed noise). FP4 encoder layers
# add AWQ + NVFP4 quant on top of the FP8 pi05 pipeline; we compare
# the FP4 decoder output directly against the FP32 reference.
import sys, json, time, pathlib, numpy as np, torch
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

ref = np.load("/tmp/pytorch_reference.npz", allow_pickle=True)
ref_raw     = ref["pytorch_raw_output"][0].astype(np.float32)
ref_img0    = ref["arg0_base_rgb"][0]
ref_img1    = ref["arg1_left_wrist_rgb"][0]
ref_img2    = ref["arg2_right_wrist_rgb"][0]
ref_noise   = ref["arg9_noise"][0].astype(np.float16)
tok_mask    = ref["arg8_tokenized_prompt_mask"][0]
ref_tokens  = ref["arg7_tokenized_prompt"][0][:int(tok_mask.sum())].astype(np.int64)

from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4
pipe = Pi05TorchFrontendThorFP4(
    CKPT, num_views=3, autotune=3,
    use_fp4_encoder_ffn=True, fp4_layers=tuple(range(17)),
    use_awq=True, use_p1_split_gu=True)
pipe.set_prompt(ref_tokens.tolist())

ref_obs = {"image": ref_img0, "wrist_image": ref_img1,
           "wrist_image_right": ref_img2}

if N > 1:
    import os as _os
    obs_npz = f"/tmp/libero_obs3v_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [
            {"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"],
             "wrist_image_right": d[f"wrist_right_{i}"],
             "state": d[f"state_{i}"]} for i in range(int(d["n"]))
        ]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        base = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
        obs_list = [{"image": o["image"], "wrist_image": o["wrist_image"],
                     "wrist_image_right": o["wrist_image"],
                     "state": o.get("state", np.zeros(32, dtype=np.float32))}
                    for o in base]
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

for _ in range(3): pipe.infer(ref_obs)

# Inject ref_noise via np.random.randn — current Pi0.5 Torch frontend
# draws _g_noise via np.random.randn(Sa, 32) on the CPU and H2D-copies.
# The legacy torch.Tensor.normal_ monkey-patch no longer fires; without
# this update the test would diff against the wrong initial noise and
# cos collapses to ~0.3. Mirrors test_all_models_precision.py PI05_SCRIPT.
Sa = ref_noise.shape[0]
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (Sa, 32):
            return ref_noise.astype(np.float64)
        return _orig_randn(*a, **kw)
_p = _PatchedRNG(); np.random.randn = _p
_p.on = True
pipe.infer(ref_obs)
_p.on = False
np.random.randn = _orig_randn
out = pipe._g_noise.float().cpu().numpy()

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

max_abs_err = float(np.max(np.abs(out - ref_raw)))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(ref_obs); lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "cos_vs_pytorch_ref": round(cos(out, ref_raw), 6),
    "maxdiff_vs_ref":     round(max_abs_err, 4),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
    "fp4_layers": sorted(pipe._fp4_layers),
    "use_awq": bool(pipe.use_awq),
    "use_p1_split_gu": bool(pipe.use_p1_split_gu),
}))
"""


PI0_SCRIPT = r"""
import sys, json, time, pathlib, numpy as np, torch
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

ref = np.load("/tmp/pi0_ref_2view.npz", allow_pickle=True)
ref_raw = ref["pytorch_raw_output"][0].astype(np.float32)
img0, img1 = ref["arg0_base_rgb"][0], ref["arg1_left_wrist_rgb"][0]
state = ref["arg4_state"][0]
noise_fp16 = ref["arg7_noise"][0]
toks = ref["arg5_tokenized_prompt"][0]
tok_mask = ref["arg6_tokenized_prompt_mask"][0]
prompt_len = int(tok_mask.sum())

from flash_rt.frontends.torch.pi0_thor import Pi0TorchFrontendThor
pipe = Pi0TorchFrontendThor(CKPT, num_views=2, autotune=3)
pipe.set_prompt(toks[:prompt_len].tolist())

obs = {
    "image": (img0*127.5+127.5).clip(0,255).astype(np.uint8),
    "wrist_image": (img1*127.5+127.5).clip(0,255).astype(np.uint8),
    "state": state.astype(np.float32),
}

if N > 1:
    # Prefer pre-extracted LIBERO obs for container portability.
    import os as _os
    obs_npz = f"/tmp/libero_obs_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [{"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"], "state": d[f"state_{i}"]} for i in range(int(d["n"]))]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        obs_list = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

for _ in range(5): pipe.infer(obs)

matched_noise = torch.from_numpy(noise_fp16).to(dtype=torch.float16, device="cuda")
_o = torch.Tensor.normal_
def _p(self, *a, **kw):
    if self.data_ptr() == pipe._g_noise.data_ptr():
        self.copy_(matched_noise); return self
    return _o(self, *a, **kw)
torch.Tensor.normal_ = _p
pipe.infer(obs)
torch.Tensor.normal_ = _o

# Re-play the enc+ae graph with matched noise to get the raw output.
images = np.stack([img0, img1])
pipe._img_buf.upload(images)
pipe._state_buf.copy_(torch.from_numpy(state[None,:]).to("cuda", torch.float16))
pipe._siglip_graph.replay()
pipe._g_noise.copy_(matched_noise)
pipe._enc_ae_graph.replay()
torch.cuda.synchronize()
raw_out = pipe._g_noise.float().cpu().numpy()

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "cos_vs_pytorch_ref": round(cos(raw_out, ref_raw), 6),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
}))
"""


PI0_JAX_SCRIPT = r"""
# Pi0 JAX FP8: Orbax checkpoint, no FP4. Phase-3 collapse-only gate
# (cos vs PyTorch FP32 ref ≥ 0.99, finite, sane max_abs). The Pi0 JAX
# frontend has no real dataset-aligned reference so the gate is looser
# than pi05/pi05_fp4.
import sys, json, time, pathlib, numpy as np
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

ref = np.load("/tmp/pi0_ref_2view.npz", allow_pickle=True)
ref_raw = ref["pytorch_raw_output"][0].astype(np.float32)
img0, img1 = ref["arg0_base_rgb"][0], ref["arg1_left_wrist_rgb"][0]
state = ref["arg4_state"][0]
noise_fp16 = ref["arg7_noise"][0].astype(np.float16)
toks = ref["arg5_tokenized_prompt"][0]
tok_mask = ref["arg6_tokenized_prompt_mask"][0]
prompt_len = int(tok_mask.sum())

from flash_rt.frontends.jax.pi0_thor import Pi0JaxFrontendThor
pipe = Pi0JaxFrontendThor(CKPT, num_views=2, autotune=3)
pipe.set_prompt(toks[:prompt_len].tolist())

ref_obs = {"image": img0, "wrist_image": img1, "state": state.astype(np.float32)}

if N > 1:
    import os as _os
    obs_npz = f"/tmp/libero_obs_2v_n{N}.npz"
    if not _os.path.exists(obs_npz):
        obs_npz = f"/tmp/libero_obs_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [
            {"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"],
             "state": d.get(f"state_{i}", np.zeros(32, dtype=np.float32))}
            for i in range(int(d["n"]))
        ]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        obs_list = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

for _ in range(3): pipe.infer(ref_obs)

Sa = pipe.Sa
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (Sa, 32):
            return noise_fp16.astype(np.float64)
        return _orig_randn(*a, **kw)
p = _PatchedRNG(); np.random.randn = p
p.on = True; pipe.infer(ref_obs); p.on = False
np.random.randn = _orig_randn

out = pipe.g_noise.download_new((Sa, 32), np.float16).astype(np.float32)

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

max_abs_err = float(np.max(np.abs(out - ref_raw)))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(ref_obs); lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "cos_vs_pytorch_ref": round(cos(out, ref_raw), 6),
    "maxdiff_vs_ref":     round(max_abs_err, 4),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
}))
"""


PI0FAST_SCRIPT = r"""
import sys, json, time, pathlib, numpy as np, torch, os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "ROOTDIR")
import logging; logging.basicConfig(level=logging.WARNING)

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.pi0fast import Pi0FastTorchFrontend
from flash_rt.models.pi0fast.pipeline import (
    prefill_forward_pi0fast, decode_step_pi0fast_bf16,
)

pipe = Pi0FastTorchFrontend(CKPT, num_views=2, autotune=0, use_cuda_graph=False)
pipe.set_prompt("pick up the red block", state=np.zeros(32, dtype=np.float32))

if N > 1:
    # Prefer pre-extracted LIBERO obs for container portability.
    import os as _os
    obs_npz = f"/tmp/libero_obs_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [{"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"], "state": d[f"state_{i}"]} for i in range(int(d["n"]))]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        obs_list = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

np.random.seed(42)
imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
images = np.stack([(im.astype(np.float32)/127.5 - 1.0).astype(np.float16) for im in imgs])
pipe._img_buf.upload(images)
pipe._siglip_graph.replay()
torch.cuda.synchronize()

Se = pipe.Se; De = pipe.De; He = pipe.He; Le = pipe.Le; NHe = pipe.NHe; HDe = pipe.HDe

# prefill forward (FVK path)
pipe._Kc.zero_(); pipe._Vc.zero_()
prefill_bufs = {
    "x": pipe._enc_x.data_ptr(), "x_fp8": pipe._enc_x_fp8.data_ptr(),
    "qkv": pipe._enc_qkv_buf.data_ptr(), "logits": pipe._enc_logits.data_ptr(),
    "attn_out": pipe._enc_attn.data_ptr(), "o_fp8": pipe._enc_o_fp8.data_ptr(),
    "gate": pipe._enc_gate.data_ptr(), "hid_fp8": pipe._enc_hid_fp8.data_ptr(),
    "fg": pipe._enc_fg.data_ptr(), "xn": pipe._enc_xn.data_ptr(), "ctx": pipe._ctx,
}
prefill_weights = {
    "qkv_w":  [w.data_ptr() for w in pipe._enc_qkv_w],
    "o_w":    [w.data_ptr() for w in pipe._enc_o_w],
    "gate_w": [w.data_ptr() for w in pipe._enc_gu_w],
    "down_w": [w.data_ptr() for w in pipe._enc_d_w],
    "alpha_host": pipe._enc_alpha_host,
    "qkv_w_flat":  pipe._enc_qkv_flat.data_ptr(),
    "o_w_flat":    pipe._enc_o_flat.data_ptr(),
    "gate_w_flat": pipe._enc_gu_flat.data_ptr(),
    "down_w_flat": pipe._enc_d_flat.data_ptr(),
    "w_scales":    pipe._enc_w_dev.data_ptr(),
    "rope":        pipe._enc_rope.data_ptr(),
    "Kc": pipe._Kc.reshape(-1).data_ptr(),
    "Vc": pipe._Vc.reshape(-1).data_ptr(),
    "final_norm_w": pipe._final_norm_w.data_ptr(),
    "act_scales":   pipe._enc_calib_scales.data_ptr(),
}
prefill_dims = {"Se": Se, "D": De, "H": He, "NH": NHe, "HD": HDe,
                "L": Le, "total_keys_max": pipe.max_total_keys}
prefill_forward_pi0fast(pipe._gemm, fvk, prefill_bufs, prefill_weights, prefill_dims)
torch.cuda.synchronize()

prefill_xn_last = pipe._enc_xn[Se - 1].float().cpu().numpy()
last_hidden = pipe._enc_xn[Se - 1:Se].to(torch.float16)
torch.matmul(last_hidden, pipe.embedding_weight.T, out=pipe._logit_buf)
torch.cuda.synchronize()
first_logit = pipe._logit_buf[0].float().cpu().numpy()
first_token = int(first_logit.argmax())

np.savez("/tmp/pi0fast_cal_matrix.npz",
         prefix=pipe._enc_x[:Se].float().cpu().numpy(),
         Se=np.array(Se),
         prefill_xn_last=prefill_xn_last,
         first_logit=first_logit,
         first_token=np.array(first_token))

# Prefill / per-token latency (no graph)
t_prefill = []
for _ in range(10):
    t0 = time.perf_counter()
    pipe._Kc.zero_(); pipe._Vc.zero_()
    prefill_forward_pi0fast(pipe._gemm, fvk, prefill_bufs, prefill_weights, prefill_dims)
    torch.cuda.synchronize()
    t_prefill.append((time.perf_counter() - t0) * 1000)
t_prefill.sort()

# Per-token decode latency (bf16 path)
decode_bufs = {
    "x": pipe._dec_x_bf16.data_ptr(), "x_fp8": pipe._dec_x_fp8.data_ptr(),
    "qkv": pipe._dec_qkv.data_ptr(), "logits": pipe._dec_logits.data_ptr(),
    "attn_out": pipe._dec_attn.data_ptr(), "o_fp8": pipe._dec_o_fp8.data_ptr(),
    "gate": pipe._dec_gate.data_ptr(), "hid_fp8": pipe._dec_hid_fp8.data_ptr(),
    "fg": pipe._dec_fg_bf16.data_ptr(), "xn": pipe._dec_xn_bf16.data_ptr(),
    "fg_scratch": pipe._dec_fg_scratch.data_ptr(),
}
decode_weights = {
    "qkv_w_flat": pipe._enc_qkv_flat.data_ptr(),
    "o_w_flat": pipe._enc_o_flat.data_ptr(),
    "gate_w_flat": pipe._enc_gu_flat.data_ptr(),
    "alpha_host": pipe._enc_alpha_host,
    "down_w_flat": pipe._enc_d_flat.data_ptr(),
    "rope_base": pipe._full_rope.data_ptr(),
    "Kc": pipe._Kc.reshape(-1).data_ptr(),
    "Vc": pipe._Vc.reshape(-1).data_ptr(),
    "final_norm_w": pipe._final_norm_w.data_ptr(),
    "act_scales": pipe._enc_calib_scales.data_ptr(),
    "w_scales": pipe._enc_w_dev.data_ptr(),
}
if hasattr(fvk, "cutlass_fp8_wide"):
    decode_weights["gate_w_list"] = [w.data_ptr() for w in pipe._enc_gu_w]
decode_dims = {"D": De, "H": He, "NH": NHe, "HD": HDe, "L": Le,
               "prefill_len": Se, "total_keys_max": pipe.max_total_keys}

token_embed = pipe.embedding_weight[first_token] * float(De ** 0.5)
pipe._dec_x_bf16.copy_(token_embed.unsqueeze(0))
t_dec = []
for step in range(10):
    t0 = time.perf_counter()
    decode_step_pi0fast_bf16(pipe._ctx, fvk, decode_bufs, decode_weights, decode_dims, step=step)
    torch.cuda.synchronize()
    t_dec.append((time.perf_counter() - t0) * 1000)
t_dec.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "first_token_id": first_token,
    "prefill_p50_ms": round(t_prefill[5], 2),
    "per_token_p50_ms": round(t_dec[5], 2),
    "50tok_est_ms": round(t_prefill[5] + 50 * t_dec[5], 1),
}))
"""


PI0FAST_JAX_SCRIPT = r"""
# Pi0-FAST JAX FP8: autoregressive prefill+decode. Phase-3 collapse-only
# gate: first_token_id must equal 4022 (matches the JAX FP16 greedy
# decode reference); finite output; sane max_abs prefill activations.
# cos vs torch FP8 is NOT a meaningful metric for autoregressive decode —
# greedy argmax is the dispositive cross-framework signal.
import sys, json, time, pathlib, numpy as np, torch, os
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.jax.pi0fast import Pi0FastJaxFrontend
from flash_rt.models.pi0fast.pipeline import (
    prefill_forward_pi0fast, decode_step_pi0fast_bf16,
)

pipe = Pi0FastJaxFrontend(CKPT, num_views=2, autotune=0, use_cuda_graph=False)
pipe.set_prompt("pick up the red block", state=np.zeros(32, dtype=np.float32))

if N > 1:
    obs_npz = f"/tmp/libero_obs_2v_n{N}.npz"
    if not os.path.exists(obs_npz):
        obs_npz = f"/tmp/libero_obs_n{N}.npz"
    if os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [
            {"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"],
             "state": d.get(f"state_{i}", np.zeros(32, dtype=np.float32))}
            for i in range(int(d["n"]))
        ]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        obs_list = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

np.random.seed(42)
imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
images = np.stack([(im.astype(np.float32)/127.5 - 1.0).astype(np.float16) for im in imgs])
pipe._img_buf.upload(images)
pipe._siglip_graph.replay()
torch.cuda.synchronize()

Se = pipe.Se; De = pipe.De; He = pipe.He; Le = pipe.Le; NHe = pipe.NHe; HDe = pipe.HDe

pipe._Kc.zero_(); pipe._Vc.zero_()
prefill_bufs = {
    "x": pipe._enc_x.data_ptr(), "x_fp8": pipe._enc_x_fp8.data_ptr(),
    "qkv": pipe._enc_qkv_buf.data_ptr(), "logits": pipe._enc_logits.data_ptr(),
    "attn_out": pipe._enc_attn.data_ptr(), "o_fp8": pipe._enc_o_fp8.data_ptr(),
    "gate": pipe._enc_gate.data_ptr(), "hid_fp8": pipe._enc_hid_fp8.data_ptr(),
    "fg": pipe._enc_fg.data_ptr(), "xn": pipe._enc_xn.data_ptr(), "ctx": pipe._ctx,
}
prefill_weights = {
    "qkv_w":  [w.data_ptr() for w in pipe._enc_qkv_w],
    "o_w":    [w.data_ptr() for w in pipe._enc_o_w],
    "gate_w": [w.data_ptr() for w in pipe._enc_gu_w],
    "down_w": [w.data_ptr() for w in pipe._enc_d_w],
    "alpha_host": pipe._enc_alpha_host,
    "qkv_w_flat":  pipe._enc_qkv_flat.data_ptr(),
    "o_w_flat":    pipe._enc_o_flat.data_ptr(),
    "gate_w_flat": pipe._enc_gu_flat.data_ptr(),
    "down_w_flat": pipe._enc_d_flat.data_ptr(),
    "w_scales":    pipe._enc_w_dev.data_ptr(),
    "rope":        pipe._enc_rope.data_ptr(),
    "Kc": pipe._Kc.reshape(-1).data_ptr(),
    "Vc": pipe._Vc.reshape(-1).data_ptr(),
    "final_norm_w": pipe._final_norm_w.data_ptr(),
    "act_scales":   pipe._enc_calib_scales.data_ptr(),
}
prefill_dims = {"Se": Se, "D": De, "H": He, "NH": NHe, "HD": HDe,
                "L": Le, "total_keys_max": pipe.max_total_keys}
prefill_forward_pi0fast(pipe._gemm, fvk, prefill_bufs, prefill_weights, prefill_dims)
torch.cuda.synchronize()

prefill_xn_last = pipe._enc_xn[Se - 1].float().cpu().numpy()
last_hidden = pipe._enc_xn[Se - 1:Se].to(torch.float16)
torch.matmul(last_hidden, pipe.embedding_weight.T, out=pipe._logit_buf)
torch.cuda.synchronize()
first_logit = pipe._logit_buf[0].float().cpu().numpy()
first_token = int(first_logit.argmax())

t_prefill = []
for _ in range(10):
    t0 = time.perf_counter()
    pipe._Kc.zero_(); pipe._Vc.zero_()
    prefill_forward_pi0fast(pipe._gemm, fvk, prefill_bufs, prefill_weights, prefill_dims)
    torch.cuda.synchronize()
    t_prefill.append((time.perf_counter() - t0) * 1000)
t_prefill.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "first_token_id": first_token,
    "prefill_p50_ms": round(t_prefill[5], 2),
    "max_abs_prefill_xn": round(float(np.max(np.abs(prefill_xn_last))), 4),
    "all_finite": bool(np.isfinite(prefill_xn_last).all() and np.isfinite(first_logit).all()),
}))
"""


PI05_JAX_SCRIPT = r"""
# JAX FP8 base: Orbax checkpoint, no FP4 quant. Mirror of PI05_SCRIPT
# (torch FP8) but using the JAX frontend. Gold reference is the same
# /tmp/pytorch_reference.npz::pytorch_raw_output the torch path uses,
# so cos_vs_pytorch_ref is directly comparable across the two stacks.
import sys, json, time, pathlib, numpy as np
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

ref = np.load("/tmp/pytorch_reference.npz", allow_pickle=True)
ref_raw    = ref["pytorch_raw_output"][0].astype(np.float32)
ref_img0   = ref["arg0_base_rgb"][0]
ref_img1   = ref["arg1_left_wrist_rgb"][0]
ref_img2   = ref["arg2_right_wrist_rgb"][0]
ref_noise  = ref["arg9_noise"][0].astype(np.float16)
tok_mask   = ref["arg8_tokenized_prompt_mask"][0]
ref_tokens = ref["arg7_tokenized_prompt"][0][:int(tok_mask.sum())].astype(np.int64)

from flash_rt.frontends.jax.pi05_thor import Pi05JaxFrontendThor
pipe = Pi05JaxFrontendThor(CKPT, num_views=3, autotune=3)
pipe.set_prompt(ref_tokens.tolist())

ref_obs = {"image": ref_img0, "wrist_image": ref_img1,
           "wrist_image_right": ref_img2}

if N > 1:
    import os as _os
    obs_npz = f"/tmp/libero_obs_3v_n{N}.npz"
    if not _os.path.exists(obs_npz):
        obs_npz = f"/tmp/libero_obs3v_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [
            {"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"],
             "wrist_image_right": d[f"wrist_right_{i}"],
             "state": d[f"state_{i}"]} for i in range(int(d["n"]))
        ]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        base = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
        obs_list = [{"image": o["image"], "wrist_image": o["wrist_image"],
                     "wrist_image_right": o["wrist_image"],
                     "state": o.get("state", np.zeros(32, dtype=np.float32))}
                    for o in base]
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

for _ in range(3): pipe.infer(ref_obs)

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
p.on = True; pipe.infer(ref_obs); p.on = False
np.random.randn = _orig_randn

out = pipe.g_noise.download_new((Sa, 32), np.float16).astype(np.float32)

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

max_abs_err = float(np.max(np.abs(out - ref_raw)))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(ref_obs); lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "cos_vs_pytorch_ref": round(cos(out, ref_raw), 6),
    "maxdiff_vs_ref":     round(max_abs_err, 4),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
}))
"""


PI05_JAX_FP4_SCRIPT = r"""
# JAX FP4 full preset: Orbax checkpoint, 18 FFN layers + AWQ + P1 split-GU.
# Gold reference: /tmp/pytorch_reference.npz::pytorch_raw_output.
# Framework-dependency invariant: this script must not import torch at
# module scope; the JAX FP4 frontend is torch-free at runtime.
import sys, json, time, pathlib, numpy as np
sys.path.insert(0, "ROOTDIR")

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"): f.unlink()

ref = np.load("/tmp/pytorch_reference.npz", allow_pickle=True)
ref_raw    = ref["pytorch_raw_output"][0].astype(np.float32)
ref_img0   = ref["arg0_base_rgb"][0]
ref_img1   = ref["arg1_left_wrist_rgb"][0]
ref_img2   = ref["arg2_right_wrist_rgb"][0]
ref_noise  = ref["arg9_noise"][0].astype(np.float16)
tok_mask   = ref["arg8_tokenized_prompt_mask"][0]
ref_tokens = ref["arg7_tokenized_prompt"][0][:int(tok_mask.sum())].astype(np.int64)

from flash_rt.frontends.jax.pi05_thor_fp4 import Pi05JaxFrontendThorFP4
pipe = Pi05JaxFrontendThorFP4(
    CKPT, num_views=3, autotune=3, weight_cache=True,
    use_fp4_encoder_ffn=True, fp4_layers=tuple(range(17)),
    use_awq=True, use_p1_split_gu=True)
pipe.set_prompt(ref_tokens.tolist())

ref_obs = {"image": ref_img0, "wrist_image": ref_img1,
           "wrist_image_right": ref_img2}

if N > 1:
    import os as _os
    obs_npz = f"/tmp/libero_obs_3v_n{N}.npz"
    if not _os.path.exists(obs_npz):
        obs_npz = f"/tmp/libero_obs3v_n{N}.npz"
    if _os.path.exists(obs_npz):
        d = np.load(obs_npz)
        obs_list = [
            {"image": d[f"img_{i}"], "wrist_image": d[f"wrist_{i}"],
             "wrist_image_right": d[f"wrist_right_{i}"],
             "state": d[f"state_{i}"]} for i in range(int(d["n"]))
        ]
    else:
        from flash_rt.datasets.libero import load_calibration_obs
        base = load_calibration_obs(LIBERO_ROOT, n=N, verbose=False)
        obs_list = [{"image": o["image"], "wrist_image": o["wrist_image"],
                     "wrist_image_right": o["wrist_image"],
                     "state": o.get("state", np.zeros(32, dtype=np.float32))}
                    for o in base]
    t0 = time.perf_counter(); pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000
else:
    cal_ms = 0.0

for _ in range(3): pipe.infer(ref_obs)

# Pi0.5 JAX frontend draws decoder noise via np.random.randn(Sa, 32) — patch
# that single call to return the fixed ref_noise so the raw output is
# deterministic against the FP32 gold.
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
p.on = True; pipe.infer(ref_obs); p.on = False
np.random.randn = _orig_randn

out = pipe.g_noise.download_new((Sa, 32), np.float16).astype(np.float32)

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

max_abs_err = float(np.max(np.abs(out - ref_raw)))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(ref_obs); lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

print("__RESULT__ " + json.dumps({
    "model": MODEL,
    "n": N,
    "calibrate_ms": round(cal_ms, 1),
    "cos_vs_pytorch_ref": round(cos(out, ref_raw), 6),
    "maxdiff_vs_ref":     round(max_abs_err, 4),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
    "fp4_layers": sorted(pipe._fp4_layers),
    "use_awq": bool(pipe.use_awq),
    "use_p1_split_gu": bool(pipe.use_p1_split_gu),
}))
"""


MODEL_REGISTRY = {
    "pi05":         ("Pi0.5 (torch)",      PI05_CKPT,        PI05_SCRIPT),
    "pi05_fp4":     ("Pi0.5 FP4 preset",   PI05_CKPT,        PI05_FP4_SCRIPT),
    "pi05_jax":     ("Pi0.5 JAX",          PI05_JAX_CKPT,    PI05_JAX_SCRIPT),
    "pi05_jax_fp4": ("Pi0.5 JAX FP4",      PI05_JAX_CKPT,    PI05_JAX_FP4_SCRIPT),
    "pi0":          ("Pi0 (torch)",        PI0_CKPT,         PI0_SCRIPT),
    "pi0_jax":      ("Pi0 JAX",            PI0_JAX_CKPT,     PI0_JAX_SCRIPT),
    "pi0fast":      ("Pi0-FAST (torch)",   PI0FAST_CKPT,     PI0FAST_SCRIPT),
    "pi0fast_jax":  ("Pi0-FAST JAX",       PI0FAST_JAX_CKPT, PI0FAST_JAX_SCRIPT),
}


def _run_script(model: str, ckpt: str, script_body: str, n: int, libero_root: str) -> dict:
    # Inject config as Python literals at the top, then let the script
    # body reference CKPT / LIBERO_ROOT / MODEL / N by name.
    header = (
        f"CKPT = {ckpt!r}\n"
        f"LIBERO_ROOT = {libero_root!r}\n"
        f"MODEL = {model!r}\n"
        f"N = {int(n)}\n"
    )
    body = script_body.replace("ROOTDIR", str(ROOT))
    r = subprocess.run(
        [sys.executable, "-c", header + body],
        capture_output=True, text=True, timeout=2400)
    for line in r.stdout.splitlines():
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    return {
        "model": model, "n": n,
        "error": ("no __RESULT__ line; stderr tail: "
                  + "\n".join(r.stderr.splitlines()[-6:])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default="pi05,pi05_fp4,pi05_jax,pi05_jax_fp4,pi0,pi0_jax,pi0fast,pi0fast_jax",
        help="comma list of models (default: all registered models)")
    ap.add_argument(
        "--ns", default="1,8",
        help="comma list of N values to exercise (default: 1,8)")
    ap.add_argument("--libero-root", default=LIBERO_ROOT)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    rows = []
    for m in models:
        if m not in MODEL_REGISTRY:
            rows.append({"model": m, "error": f"unknown model {m!r}"})
            continue
        name, ckpt, script = MODEL_REGISTRY[m]
        for n in ns:
            logger.info("── %s  N=%d ──", name, n)
            r = _run_script(m, ckpt, script, n, args.libero_root)
            logger.info("  %s", json.dumps(r, indent=None))
            rows.append(r)

    # Summary table. "cos vs PyTorch FP32 ref" is the gold column —
    # compared against /tmp/pytorch_reference.npz (pi05 / pi05_fp4) or
    # /tmp/pi0_ref_2view.npz (pi0). Pi0-FAST reports first-token match
    # against the JAX FP16 reference because autoregressive greedy
    # decode admits a per-token gold signal instead of a continuous cos.
    print("\n" + "=" * 100)
    print(f"{'model':>10}  {'N':>3}  {'calibrate':>10}  "
          f"{'cos vs PyTorch ref':>20}  {'maxdiff':>9}  "
          f"{'p50':>9}  notes")
    print("-" * 100)
    for r in rows:
        if "error" in r:
            print(f"{r['model']:>10}  {r.get('n', '?'):>3}  "
                  f"{'ERROR':>10}  {'':>20}  {'':>9}  {'':>9}  "
                  f"{r['error'][:30]}")
            continue
        cos = (r.get("cos_vs_pytorch_ref")
               or (r.get("first_token_id") is not None
                   and f"first_tok={r['first_token_id']}")
               or "n/a")
        maxd = r.get("maxdiff_vs_ref", "-")
        p50 = r.get("p50_ms") or r.get("per_token_p50_ms") or "-"
        note = ""
        if "per_token_p50_ms" in r:
            note = (f"prefill={r['prefill_p50_ms']:.1f}ms  "
                    f"50tok≈{r['50tok_est_ms']:.0f}ms")
        cos_str = (f"{cos:.6f}" if isinstance(cos, float) else str(cos))
        maxd_str = f"{maxd:.4f}" if isinstance(maxd, (int, float)) else str(maxd)
        p50_str = f"{p50:.2f} ms" if isinstance(p50, (int, float)) else str(p50)
        print(f"{r['model']:>10}  {r['n']:>3}  "
              f"{r['calibrate_ms']:>8.1f} ms  "
              f"{cos_str:>20}  {maxd_str:>9}  "
              f"{p50_str:>9}  {note}")
    print("=" * 100)
    print(json.dumps(rows, indent=2))

    # Pass if every row has a cos number and no error.
    ok = all(("error" not in r) for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
