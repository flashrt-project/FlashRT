#!/usr/bin/env python3
"""FlashRT — Full precision test: all models, all backends.

Tests current flash_rt_kernels.so against saved reference outputs.
Each model runs in separate subprocess. Uses monkey-patch noise injection
for deterministic comparison (same pattern as _vf_*.py production tests).

Compares:
  Pi0.5: Production vs FlashRT Torch vs FlashRT JAX (saved outputs in /tmp/v_*.npy)
  Pi0:   FlashRT Torch vs PI0Pytorch reference (/tmp/pi0_ref_2view.npz)
  Pi0 JAX: FlashRT JAX vs FlashRT Torch (raw decoder output, same noise)
  GROOT: FlashRT Torch vs PyTorch reference (groot_ref/groot_ref_e2e_full.pt)

Usage:
    python3 tests/test_all_models_precision.py
    python3 tests/test_all_models_precision.py --model pi0
    python3 tests/test_all_models_precision.py --model pi0_jax
"""
import argparse, subprocess, sys, os, json
import numpy as np

FLASH_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Centralised env-var resolution for checkpoint paths. Per-model templates
# below embed placeholder strings like "<your_pi05_torch_ckpt>" which
# ``run_model`` substitutes from ``tests/_helpers/paths.resolve(...)``
# before launching the subprocess. Public users export FLASH_RT_PI05_CKPT
# etc. (see tests/_helpers/paths.py for the full key list and defaults).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers.paths import resolve  # noqa: E402

# Placeholder string -> resolver key. Only keys whose placeholder actually
# appears in the chosen template get resolved, so a missing GROOT fixture
# won't break a Pi0-only invocation.
_PLACEHOLDER_TO_KEY = {
    "<your_pi05_torch_ckpt>": "PI05_CKPT",
    "<your_pi0_torch_ckpt>":  "PI0_CKPT",
    "<your_pi05_jax_ckpt>":   "PI05_JAX_CKPT",
    "<your_jax_ckpts>":       "JAX_CKPTS",
    "<your_groot_ckpt>":      "GROOT_CKPT",
    "<your_groot_ref>":       "GROOT_REF",
}

def cosine(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ══════════════════════════════════════════════════════════════
# Pi0.5: monkey-patch noise, full infer path
# Re-run FlashRT Torch with matched_noise, compare vs saved prod/jax
# ══════════════════════════════════════════════════════════════
PI05_SCRIPT = '''
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, "ROOTDIR")
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor as ThorPipelineTorch

np.random.seed(42)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
obs = {"image": img, "wrist_image": wrist}

pipe = ThorPipelineTorch("<your_pi05_torch_ckpt>", num_views=2, autotune=3)
pipe.set_prompt("pick up the red block and place it in the tray")
for _ in range(5): pipe.infer(obs)

# Inject matched_noise via np.random.randn — Pi0.5 torch frontend
# (the current Pi0.5 torch frontend) draws _g_noise via np.random.randn(Sa, 32) on
# the CPU and H2D-copies. The legacy torch.Tensor.normal_ monkey-patch
# no longer fires. Mirror the PI05_JAX_SCRIPT pattern below.
matched_noise = np.load("/tmp/matched_noise.npy").astype(np.float16)
Sa = matched_noise.shape[0]
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (Sa, 32):
            return matched_noise.astype(np.float64)
        return _orig_randn(*a, **kw)
p = _PatchedRNG()
np.random.randn = p

p.on = True
r = pipe.infer(obs)
p.on = False
np.random.randn = _orig_randn

out = r["actions"]
prod = np.load("/tmp/v_prod.npy")
jax = np.load("/tmp/v_jax.npy")
old_torch = np.load("/tmp/v_torch.npy")

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter()-t0)*1000)
lat.sort()

print(json.dumps({
    "vs_prod": round(cos(out, prod), 6),
    "vs_jax": round(cos(out, jax), 6),
    "vs_old_torch": round(cos(out, old_torch), 6),
    "p50_ms": round(lat[10], 1),
}))
'''

# ══════════════════════════════════════════════════════════════
# Pi0.5 FP4 preset: same monkey-patch flow as Pi0.5, with
# use_fp4=True (auto = 17 live layers + AWQ + P1 split-GU).
# Compared vs pytorch fp32 ref, FP8 prod canary, and old torch.
# ══════════════════════════════════════════════════════════════
PI05_FP4_SCRIPT = '''
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, "ROOTDIR")
import flash_rt

np.random.seed(42)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
obs = {"image": img, "wrist_image": wrist}

m = flash_rt.load_model("<your_pi05_torch_ckpt>",
                         framework="torch", config="pi05",
                         num_views=2, autotune=3, use_fp4=True)
pipe = m._pipe
pipe.set_prompt("pick up the red block and place it in the tray")
for _ in range(5): pipe.infer(obs)

# Inject matched_noise via np.random.randn — see PI05_SCRIPT comment.
matched_noise = np.load("/tmp/matched_noise.npy").astype(np.float16)
Sa = matched_noise.shape[0]
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (Sa, 32):
            return matched_noise.astype(np.float64)
        return _orig_randn(*a, **kw)
p = _PatchedRNG()
np.random.randn = p

p.on = True
r = pipe.infer(obs)
p.on = False
np.random.randn = _orig_randn

out = r["actions"]
prod = np.load("/tmp/v_prod.npy")
jax = np.load("/tmp/v_jax.npy")
old_torch = np.load("/tmp/v_torch.npy")

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter()-t0)*1000)
lat.sort()

print(json.dumps({
    "vs_pytorch_ref": round(cos(out, old_torch), 6),
    "vs_prod": round(cos(out, prod), 6),
    "vs_jax": round(cos(out, jax), 6),
    "p50_ms": round(lat[10], 1),
    "fp4_layers": sorted(pipe._fp4_layers),
    "use_awq": bool(pipe.use_awq),
    "use_p1_split_gu": bool(pipe.use_p1_split_gu),
}))
'''

# ══════════════════════════════════════════════════════════════
# Pi0: monkey-patch noise, full infer path, vs PI0Pytorch ref
# ══════════════════════════════════════════════════════════════
PI0_SCRIPT = '''
import sys, os, time, json, pathlib, torch, numpy as np
sys.path.insert(0, "ROOTDIR")

for f in (pathlib.Path.home()/".flash_rt"/"calibration").glob("70bdf6f4*"):
    f.unlink()

ref = np.load("/tmp/pi0_ref_2view.npz", allow_pickle=True)
ref_raw = ref["pytorch_raw_output"][0].astype(np.float32)
img0, img1 = ref["arg0_base_rgb"][0], ref["arg1_left_wrist_rgb"][0]
state = ref["arg4_state"][0]
noise_fp16 = ref["arg7_noise"][0]
toks = ref["arg5_tokenized_prompt"][0]
tok_mask = ref["arg6_tokenized_prompt_mask"][0]
prompt_len = int(tok_mask.sum())

from flash_rt.frontends.torch.pi0_thor import Pi0TorchFrontendThor as ThorPipelineTorchPi0
pipe = ThorPipelineTorchPi0("<your_pi0_torch_ckpt>", num_views=2, autotune=3)
pipe.set_prompt(toks[:prompt_len].tolist())

obs = {"image": (img0*127.5+127.5).clip(0,255).astype(np.uint8),
       "wrist_image": (img1*127.5+127.5).clip(0,255).astype(np.uint8),
       "state": state.astype(np.float32)}
for _ in range(5): pipe.infer(obs)

# Monkey-patch to inject reference noise
matched_noise = torch.from_numpy(noise_fp16).to(dtype=torch.float16, device="cuda")
_orig = torch.Tensor.normal_
def _patched(self, *a, **kw):
    if self.data_ptr() == pipe._g_noise.data_ptr():
        self.copy_(matched_noise); return self
    return _orig(self, *a, **kw)
torch.Tensor.normal_ = _patched
r = pipe.infer(obs)
torch.Tensor.normal_ = _orig

# Compare raw decoder output (before unnormalize)
# infer() returns unnormalized actions. We need raw output.
# Re-do with direct graph replay to get raw:
images = np.stack([img0, img1])
pipe._img_buf.upload(images)
pipe._state_buf.copy_(torch.from_numpy(state[None,:]).to("cuda", torch.float16))
pipe._siglip_graph.replay()
pipe._g_noise.copy_(matched_noise)
pipe._enc_ae_graph.replay()
torch.cuda.synchronize()
raw_out = pipe._g_noise.float().cpu().numpy()
np.save("/tmp/pi0_torch_raw.npy", raw_out)

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter()-t0)*1000)
lat.sort()

print(json.dumps({
    "vs_pytorch_ref": round(cos(raw_out, ref_raw), 6),
    "p50_ms": round(lat[10], 1),
}))
'''

# ══════════════════════════════════════════════════════════════
# GROOT: fixed seed noise, vs PyTorch reference
# ══════════════════════════════════════════════════════════════
GROOT_SCRIPT = '''
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, "ROOTDIR")

ref = torch.load("<your_groot_ref>/groot_ref_e2e_full.pt",
                 map_location="cpu", weights_only=False)
ref_actions = ref["actions"][0].float().numpy()
img_np = ref["img_np"]
prompt = ref["prompt"]
T_ref = ref_actions.shape[0]

from flash_rt.frontends.torch.groot_thor import GrootTorchFrontendThor as ThorPipelineTorchGroot
pipe = ThorPipelineTorchGroot("<your_groot_ckpt>", num_views=2, autotune=3)
pipe.set_prompt(prompt)

obs = {"image": img_np, "wrist_image": img_np}
for _ in range(5): pipe.infer(obs)

# Fixed seed for DiT noise (same as gen_e2e_ref.py)
torch.manual_seed(123)
result = pipe.infer(obs)
fvk_actions = result["actions"][:T_ref]

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter()-t0)*1000)
lat.sort()

print(json.dumps({
    "vs_pytorch_ref": round(cos(fvk_actions, ref_actions), 6),
    "p50_ms": round(lat[10], 1),
}))
'''

# ══════════════════════════════════════════════════════════════
# Pi0 JAX: Orbax checkpoint, vs Torch raw output + PI0Pytorch ref
# ══════════════════════════════════════════════════════════════
PI0_JAX_SCRIPT = '''
import sys, os, time, json, pathlib, numpy as np
sys.path.insert(0, "ROOTDIR")

ref = np.load("/tmp/pi0_ref_2view.npz", allow_pickle=True)
ref_raw = ref["pytorch_raw_output"][0].astype(np.float32)
img0, img1 = ref["arg0_base_rgb"][0], ref["arg1_left_wrist_rgb"][0]
state = ref["arg4_state"][0]
noise_fp16 = ref["arg7_noise"][0].astype(np.float16)
toks = ref["arg5_tokenized_prompt"][0]
tok_mask = ref["arg6_tokenized_prompt_mask"][0]
prompt_len = int(tok_mask.sum())

from flash_rt.frontends.jax.pi0_thor import Pi0JaxFrontendThor as ThorPipelineJaxPi0
pipe = ThorPipelineJaxPi0("<your_jax_ckpts>/pi0_base", num_views=2, autotune=3)
pipe.set_prompt(toks[:prompt_len].tolist())

obs = {"image": (img0*127.5+127.5).clip(0,255).astype(np.uint8),
       "wrist_image": (img1*127.5+127.5).clip(0,255).astype(np.uint8),
       "state": state.astype(np.float32)}
for _ in range(5): pipe.infer(obs)

# Monkey-patch noise for deterministic comparison
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (10, 32):
            return noise_fp16.astype(np.float64)
        return _orig_randn(*a, **kw)
p = _PatchedRNG()
np.random.randn = p

p.on = True
pipe.infer(obs)
p.on = False
np.random.randn = _orig_randn

# Get raw decoder output
jax_raw = pipe.g_noise.download_new((10, 32), np.float16).astype(np.float32)

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter()-t0)*1000)
lat.sort()

# Also load Torch raw output if available (saved by pi0 test)
torch_raw_path = "/tmp/pi0_torch_raw.npy"
results = {
    "vs_pytorch_ref": round(cos(jax_raw, ref_raw), 6),
    "p50_ms": round(lat[10], 1),
}
if os.path.exists(torch_raw_path):
    torch_raw = np.load(torch_raw_path)
    results["vs_torch_raw"] = round(cos(jax_raw, torch_raw), 6)

print(json.dumps(results))
'''

# ══════════════════════════════════════════════════════════════
# Pi0.5 JAX: Orbax checkpoint, vs saved Torch / prod / old_torch refs
# Mirrors Pi0.5 torch flow with matched_noise, but through Pi05JaxFrontendThor.
# ══════════════════════════════════════════════════════════════
PI05_JAX_SCRIPT = '''
import sys, os, time, json, numpy as np
sys.path.insert(0, "ROOTDIR")
from flash_rt.frontends.jax.pi05_thor import Pi05JaxFrontendThor as ThorPipelineJaxPi05

pipe = ThorPipelineJaxPi05("<your_pi05_jax_ckpt>", num_views=2, autotune=3)
pipe.set_prompt("pick up the red block and place it in the tray")

np.random.seed(42)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
obs = {"image": img, "wrist_image": wrist}
for _ in range(5): pipe.infer(obs)

# Monkey-patch noise: Pi05 JAX pulls noise via np.random.randn(Sa, 32).
matched_noise = np.load("/tmp/matched_noise.npy").astype(np.float16)
Sa = matched_noise.shape[0]
_orig_randn = np.random.randn
class _PatchedRNG:
    on = False
    def __call__(self, *a, **kw):
        if self.on and a == (Sa, 32):
            return matched_noise.astype(np.float64)
        return _orig_randn(*a, **kw)
p = _PatchedRNG()
np.random.randn = p

p.on = True
r = pipe.infer(obs)
p.on = False
np.random.randn = _orig_randn

out = r["actions"]
prod      = np.load("/tmp/v_prod.npy")
jax_ref   = np.load("/tmp/v_jax.npy")
old_torch = np.load("/tmp/v_torch.npy")

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

lat = []
for _ in range(20):
    t0 = time.perf_counter(); pipe.infer(obs); lat.append((time.perf_counter()-t0)*1000)
lat.sort()

print(json.dumps({
    "vs_prod":      round(cos(out, prod), 6),
    "vs_jax":       round(cos(out, jax_ref), 6),
    "vs_old_torch": round(cos(out, old_torch), 6),
    "p50_ms":       round(lat[10], 1),
}))
'''

MODELS = {
    'pi05':     ('Pi0.5',              PI05_SCRIPT),
    'pi05_fp4': ('Pi0.5 FP4 preset',   PI05_FP4_SCRIPT),
    'pi05_jax': ('Pi0.5 JAX',          PI05_JAX_SCRIPT),
    'pi0':      ('Pi0',                PI0_SCRIPT),
    'pi0_jax':  ('Pi0 JAX',            PI0_JAX_SCRIPT),
    'groot':    ('GROOT N1.6',         GROOT_SCRIPT),
}

def run_model(key):
    name, script = MODELS[key]
    script = script.replace('ROOTDIR', FLASH_VLA_ROOT)
    # Resolve only the placeholders that actually appear in this template,
    # so a missing key for an unselected model doesn't kill the run.
    for placeholder, env_key in _PLACEHOLDER_TO_KEY.items():
        if placeholder in script:
            # JAX_CKPTS is a base prefix; downstream code joins with /pi0_base etc.
            must_exist = env_key not in {"JAX_CKPTS", "TORCH_CKPTS"}
            script = script.replace(placeholder, resolve(env_key, must_exist=must_exist))
    r = subprocess.run(['python3', '-c', script], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {'error': '\n'.join(r.stderr.strip().split('\n')[-5:])}
    for line in reversed(r.stdout.strip().split('\n')):
        if line.strip().startswith('{'):
            return json.loads(line.strip())
    return {'error': 'No JSON output\n' + r.stdout[-200:]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS.keys()), default=None)
    args = parser.parse_args()
    targets = [args.model] if args.model else list(MODELS.keys())

    print("=" * 60)
    print("FlashRT — Full Precision & Latency Test")
    print("=" * 60)

    # FP4 is lossy vs FP8 prod — loosen cos thresholds accordingly
    # (production config validated on LIBERO Spatial 491/500 = 98.2%)
    def threshold_for(key, metric):
        if key == 'pi05_fp4':
            return 0.996 if metric == 'vs_prod' else 0.997
        return 0.995 if 'pytorch_ref' in metric else 0.998

    results = {}
    for key in targets:
        name, _ = MODELS[key]
        print(f"\n── {name} ──")
        r = run_model(key)
        results[key] = r
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
        else:
            for k, v in r.items():
                if k.startswith('vs_'):
                    threshold = threshold_for(key, k)
                    s = 'PASS' if v >= threshold else 'FAIL'
                    print(f"  {k}: {v:.6f}  [{s} @ {threshold:.3f}]")
                elif k == 'p50_ms':
                    print(f"  P50: {v:.1f} ms")
                elif k in ('fp4_layers', 'use_awq', 'use_p1_split_gu'):
                    print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key in targets:
        name, _ = MODELS[key]
        r = results[key]
        if 'error' in r:
            print(f"  {name:12s}  [ERROR]")
        else:
            cos_vals = [f"{k}={v:.4f}" for k, v in r.items() if k.startswith('vs_')]
            print(f"  {name:12s}  {', '.join(cos_vals)}  P50={r['p50_ms']:.1f}ms")
    print("=" * 60)

if __name__ == '__main__':
    main()
