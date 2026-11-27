"""Inkling 2 Mini on RTX SM120: the routed experts of verification rows (expert_router_route + expert_flat_fused, w13
and w2 in one kernel) against the split passes (expert_route, expert_w13_flat, expert_w2_flat), and the fused kernel
on an SM partition smaller than its grid. Needs FLASHRT_INKLING2_DIR = the released checkpoint directory, an SM120
GPU and the module built with FLASHRT_ENABLE_INKLING2; skipped otherwise."""
from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path

import pytest

CKPT = os.environ.get("FLASHRT_INKLING2_DIR", "")
if not CKPT or not Path(CKPT).is_dir():
    pytest.skip("set FLASHRT_INKLING2_DIR to the Inkling-2-Mini checkpoint", allow_module_level=True)
torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("needs an SM120 GPU", allow_module_level=True)
pytest.importorskip("flash_rt.flash_rt_inkling2")

TEXT = ("A bounded worker queue applies updates in timestamp order, retries transient failures and reports "
        "backpressure, idempotency and fairness metrics; the service keys events by user and sequence number. ")
LAYERS = (3, 21)            # NVFP4 and MXFP8 experts

# One process: the fused pass of 8 rows on the whole GPU, then on a stream of a green context of 16 SMs (the grid
# keeps one block per SM of the device, so most blocks wait for an SM while the others run).
PARTITION = r"""
import ctypes, sys, torch
from flash_rt.frontends.torch._inkling2_rtx_runtime import Inkling2RtxRuntime
from flash_rt.models.flint.tokenizer import FlintTokenizer
ckpt, text, nsm, layers = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(v) for v in sys.argv[4:]]
cu = ctypes.CDLL("libcuda.so.1")
if not hasattr(cu, "cuGreenCtxCreate"):
    print("SKIP the driver has no green contexts"); sys.exit(0)
rt = Inkling2RtxRuntime(ckpt, max_seq=2048)
rt.prefill(FlintTokenizer.from_directory(ckpt).encode(text)[:700])
cases = []
for layer in layers:
    p = f"layers.{layer}."
    mx = rt.w[p + "mlp.experts.w13_weight"].dtype != torch.uint8
    calls = rt._experts(p, 8, mx, rt.mlp_in, False, router_w=rt.w[p + "mlp.gate.weight"].data_ptr())
    rt._run(calls)
    torch.cuda.synchronize()
    cases.append((calls, rt.expert_out[:64].clone()))
def need(r, what):           # CUDA_ERROR_NOT_SUPPORTED (801): no green contexts here; any other error fails
    if r == 801:
        print(f"SKIP {what}: not supported"); sys.exit(0)
    if r:
        print(f"FAIL {what}: CUresult {r}"); sys.exit(1)
prim, g, gctx, gs = (ctypes.c_void_p() for _ in range(4))
need(cu.cuCtxGetCurrent(ctypes.byref(prim)), "cuCtxGetCurrent")
dev = ctypes.c_int()
need(cu.cuDeviceGet(ctypes.byref(dev), torch.cuda.current_device()), "cuDeviceGet")
full, part, rest = (ctypes.create_string_buffer(256) for _ in range(3))
need(cu.cuDeviceGetDevResource(dev, full, 1), "cuDeviceGetDevResource")
nb = ctypes.c_uint(1)
need(cu.cuDevSmResourceSplitByCount(part, ctypes.byref(nb), full, rest, 0, nsm), "cuDevSmResourceSplitByCount")
sms = ctypes.c_uint.from_buffer(part, 96).value
desc = ctypes.c_void_p()
need(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), part, 1), "cuDevResourceGenerateDesc")
need(cu.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), "cuGreenCtxCreate")
need(cu.cuCtxFromGreenCtx(ctypes.byref(gctx), g), "cuCtxFromGreenCtx")
need(cu.cuGreenCtxStreamCreate(ctypes.byref(gs), g, 1, 0), "cuGreenCtxStreamCreate")
for calls, ref in cases:
    rt.expert_out.zero_()
    torch.cuda.synchronize()
    cu.cuCtxSetCurrent(gctx)
    for fn, d in calls:
        fn(d, gs.value)
    r = cu.cuStreamSynchronize(gs)
    cu.cuCtxSetCurrent(prim)
    print("RESULT", sms, r, torch.equal(rt.expert_out[:64], ref), flush=True)
"""


def test_fused_experts_complete_on_an_sm_partition():
    """The w2 phase waits on counts of w13 chunks; every chunk is claimed by a running block, so the pass completes
    with most of its blocks waiting for an SM (and gives the whole GPU's output)."""
    import flash_rt
    root = str(Path(flash_rt.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p))
    try:
        r = subprocess.run([sys.executable, "-c", PARTITION, CKPT, TEXT * 20, "16", *map(str, LAYERS)],
                           capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("the fused expert pass did not complete on a 16-SM partition")
    skip = [l for l in r.stdout.splitlines() if l.startswith("SKIP")]
    if skip:
        pytest.skip(skip[0])
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    res = [l.split()[1:] for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert len(res) == len(LAYERS)
    assert all(0 < int(sms) < torch.cuda.get_device_properties(0).multi_processor_count for sms, _, _ in res)
    assert all(err == "0" and same == "True" for _, err, same in res), res


@pytest.fixture(scope="module")
def rt():
    from flash_rt.frontends.torch._inkling2_rtx_runtime import Inkling2RtxRuntime
    from flash_rt.models.flint.tokenizer import FlintTokenizer
    r = Inkling2RtxRuntime(CKPT, max_seq=2048)
    r.prefill(FlintTokenizer.from_directory(CKPT).encode(TEXT * 20)[:700])     # real rows in the MLP inputs
    yield r
    r.graphs.clear()
    r.programs.clear()
    del r
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("layer", LAYERS)
def test_fused_experts_equal_split_passes(rt, layer):
    """Activations, routing and expert outputs of 2..8 rows bit for bit (the split passes read the logits the
    router pass of the fused calls wrote)."""
    p = f"layers.{layer}."
    mx = rt.w[p + "mlp.experts.w13_weight"].dtype != torch.uint8
    gate = rt.w[p + "mlp.gate.weight"].data_ptr()
    for R in range(2, 9):
        out = []
        for router_w in (gate, 0):
            for b in (rt.act, rt.ids, rt.route_w, rt.expert_out):
                b.zero_()
            rt._run(rt._experts(p, R, mx, rt.mlp_in, False, router_w=router_w))
            torch.cuda.synchronize()
            out.append([b.clone() for b in (rt.act[:R * 8], rt.ids[:R], rt.route_w[:R], rt.expert_out[:R * 8])])
        assert out[0][3].abs().sum() > 0
        assert all(torch.equal(a, b) for a, b in zip(*out)), (layer, R)
