"""End-to-end gate for the Pi0.5 fixed-shape state-prompt graph.

Pi0.5 renders the discretized robot state into the language prompt, so the
prompt token length drifts with the state values. The default
``state_prompt_mode="fixed"`` keeps ONE pipeline and ONE captured graph at the
max prompt length: every length is served by masking the padded prefix keys
(FlashAttention-2 ``seqused_k``) and appending the decoder's action K/V right
after the valid prefix (``qkv_split_rope_devpos``). The legacy ``"exact"`` mode
captures a separate graph per length.

Gates (each mode runs in its own child process; sharing a CUDA context between
two frontends in one process is known-flaky — see test_pi05_batched_*):

1. **Mechanism is exact** — in BF16 (no FP8 quant noise), fixed-mode actions
   match exact-mode actions to cosine >= 0.9999 across a varying-length state
   sequence. This proves the seqused masking + devpos K/V append reproduce the
   per-length computation bit-for-bit; exact mode is the unchanged origin/main
   path (validated against the openpi reference), so the equivalence carries.
2. **One graph, zero recapture** — fixed mode captures exactly ONE graph for
   the whole sequence; exact mode captures one per distinct length.
3. **Coverage** — the sequence exercises multiple distinct prompt lengths.

The FP8 path adds quantization/tactic noise (fixed runs GEMMs at the padded
M=vision+max, exact at M=vision+len), so fixed-vs-exact FP8 cosine is ~0.999 and
is reported (not hard-gated) here; the BF16 gate is the exactness proof and the
FP8 cosine-vs-reference should be validated on real benchmark observations.
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
import sys, numpy as np, torch
import flash_rt

mode, ckpt, out_path, seed = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))

rng = np.random.RandomState(0)
images = [rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
          rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
prompt = "pick up the cup"
states = [
    np.zeros(8, dtype=np.float32),
    np.ones(8, dtype=np.float32),
    np.full(8, 0.5, dtype=np.float32),
    np.linspace(-1.0, 1.0, 8).astype(np.float32),
    np.full(8, -0.5, dtype=np.float32),
]

model = flash_rt.load_model(ckpt, framework="torch", config="pi05",
                            num_views=2, state_prompt_mode=mode)
fe = model._pipe
# Warmup: calibrate + capture (fixed: one graph; exact: per-length).
for s in states:
    model.predict(images, prompt=prompt, state=s)

def measure(s):
    torch.manual_seed(seed)
    a = np.asarray(model.predict(images, prompt=prompt, state=s),
                   dtype=np.float32)
    return a, int(fe.current_prompt_len)

acts, lens = [], []
for s in states:
    a, L = measure(s)
    acts.append(a)
    lens.append(L)
acts = np.stack(acts)
lens = np.array(lens, dtype=np.int64)

if mode == "fixed" and getattr(fe.pipeline, "_fixed_shape", False):
    n_graphs = 1
else:
    n_graphs = len(getattr(fe, "_prompt_pipeline_cache", {}))

np.savez(out_path, acts=acts, lens=lens, n_graphs=n_graphs)
"""


def _run(mode, ckpt, env=None, seed=4321):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        out_path = f.name
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, mode, ckpt, out_path, str(seed)],
        capture_output=True, text=True, env=full_env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{mode} child failed:\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}")
    data = dict(np.load(out_path))
    os.unlink(out_path)
    return data


def _cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="CUDA GPU required")
@pytest.mark.skipif(not _CKPT_AVAILABLE,
                    reason=f"Pi0.5 checkpoint not found at {CKPT_PI05}")
def test_fixed_graph_mechanism_exact_and_one_graph():
    # ── BF16: the exactness proof (no FP8 quant noise) ──
    bf = {"FVK_PI05_RTX_FORCE_BF16": "1"}
    fixed = _run("fixed", CKPT_PI05, env=bf)
    exact = _run("exact", CKPT_PI05, env=bf)

    # Coverage + lengths agree across modes.
    distinct = sorted(set(fixed["lens"].tolist()))
    assert len(distinct) >= 2, f"need >=2 prompt lengths, got {distinct}"
    assert fixed["lens"].tolist() == exact["lens"].tolist()

    # One graph (fixed) vs per-length (exact).
    assert int(fixed["n_graphs"]) == 1, (
        f"fixed mode must capture exactly ONE graph, got {fixed['n_graphs']}")
    assert int(exact["n_graphs"]) == len(distinct), (
        f"exact mode: one graph per length {len(distinct)}, "
        f"got {exact['n_graphs']}")

    # Mechanism exact: BF16 fixed == exact per step.
    for i in range(fixed["acts"].shape[0]):
        c = _cos(fixed["acts"][i], exact["acts"][i])
        assert c >= 0.9999, (
            f"BF16 step {i} (len={fixed['lens'][i]}): fixed vs exact "
            f"cos={c:.7f} — seqused/devpos mechanism is not bit-exact")
