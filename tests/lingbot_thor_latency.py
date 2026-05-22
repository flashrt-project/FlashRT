"""LingBot-VLA Thor E2E latency + FA4-activation check, run against THIS repo's
flash_rt (the regularized one-.so build). Proves Part A (lingbot merged into
flash_rt_kernels) + Part B (FA4 vendored, import-sys fix) end to end:
FA4 active -> ~102ms@25; FA4 falling back to fmha -> ~120ms@25.
"""
import sys, time
# This repo first, so `import flash_rt` resolves here (the regularized build).
sys.path.insert(0, "/workspace/lingbot-vla/_flashrt_build/flashrt-public")
from types import SimpleNamespace
from pathlib import Path
import torch

import flash_rt
from flash_rt.executors.torch_weights import SafetensorsSource
from flash_rt.executors.weight_loader import WeightLoader
from flash_rt.frontends.torch._lingbot_thor_spec import build_spec
from flash_rt.models.lingbot.buffer_binder import bind_target_to_device
from flash_rt.models.lingbot.kernel_ops import clear_fp8_weight_cache
from flash_rt.models.lingbot.graph_runner import sample_actions_graph
from flash_rt.models.lingbot import calibration as calib
from flash_rt.models.lingbot import kernel_ops as ko

print("flash_rt from:", flash_rt.__file__, flush=True)
import flash_rt.flash_rt_kernels as _k
print("lingbot_* symbols in flash_rt_kernels:", len([x for x in dir(_k) if x.startswith("lingbot_")]), flush=True)
print("FA4 loaded (vendored, no env):", ko._get_fa4() is not None, flush=True)

clear_fp8_weight_cache()
src = SafetensorsSource("/workspace/lingbot-vla/lingbot-vla-4b/model.safetensors", device="cpu", strip_prefix="")
t = SimpleNamespace(); WeightLoader(src, target=t, spec=build_spec()).run()
bind_target_to_device(t, dtype=torch.bfloat16, device="cuda")
calib.set_static_scales(calib.load_calibration(
    "/workspace/lingbot-vla/calibration/lingbot_thor_static.json", device=torch.device("cuda")))
art = Path("/workspace/lingbot-vla/baseline_artifacts_10")
inp = {k: torch.load(art/("inputs/"+k+".pt")).cuda()
       for k in ["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"]}

def measure(NS):
    replay, out, g = sample_actions_graph(target=t, num_steps=NS, warmup_iters=3, **inp)
    replay(); torch.cuda.synchronize()
    finite = bool(torch.isfinite(out).all().item())
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        replay(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    p = sorted(ts)[len(ts)//2] * 1000
    del replay, out, g; torch.cuda.empty_cache(); clear_fp8_weight_cache()
    return p, finite

for NS in (25, 10):
    p, finite = measure(NS)
    print(f"  ns={NS} P50={p:6.1f} ms | out_finite={finite}", flush=True)
