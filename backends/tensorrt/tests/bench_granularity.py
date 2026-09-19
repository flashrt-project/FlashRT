"""What operator granularity costs, against the layer and the stage plugin.

The encoder is available at three granularities. This builds the finest one:
an engine of `--depth` blocks, each block the three operator plugins that
carry an encoder layer's arithmetic -- attention, the output projection, the
FFN -- with fp16 between them, per-layer weights, and nothing else.

The arm is a lower bound, not an encoder: it has no QKV projection, no RoPE and
no normalization, and it is not numerically meaningful. If it is already slower
than `engine_encoder_layer_chain.py` (18 layer plugins) or
`engine_encoder_stage.py` (one stage plugin), which do all of that work and are
bitwise equal to FlashRT, the boundary is what costs, not the arithmetic.

usage:
  bench_granularity.py <plugin.so> <encoder_all.safetensors> [--depth N]
                       [--down-variant N] [--o-variant N] [--json FILE]
"""
import argparse
import json
import os
import sys
import time

try:
    import tensorrt  # noqa: F401
except ImportError:  # JetPack installs the TensorRT bindings for the system Python
    sys.path.append(f"/usr/lib/python3.{sys.version_info.minor}/dist-packages")

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("plugin")
ap.add_argument("encoder")
ap.add_argument("--depth", type=int, default=18, help="blocks in the stack")
ap.add_argument("--down-variant", type=int, default=6, help="tile for the FFN down GEMM")
ap.add_argument("--o-variant", type=int, default=6, help="tile for the output projection")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--json", default=None)
args = ap.parse_args()

logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(args.plugin)
dev = torch.device("cuda")
stream = torch.cuda.current_stream()

E = load_file(args.encoder)
M, D, H, NH, HD = E["meta"].tolist()[:5]
L = E["meta"].tolist()[8]
keep = []
# The last layer records no FFN or projection weights, so the stack cycles
# through the layers that do.
WEIGHTED = [l for l in range(L) if f"L{l}.down_packed" in E]


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


def fields(ints, floats=()):
    fc = trt.PluginFieldCollection()
    for name, value in ints:
        keep.append(np.array([value], dtype=np.int32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.INT32))
    for name, value in floats:
        keep.append(np.array([value], dtype=np.float32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.FLOAT32))
    return fc


def plugin(net, op, ints, floats, inputs, name):
    p = registry.get_creator(op, "1", "").create_plugin(op, fields(ints, floats),
                                                        trt.TensorRTPhase.BUILD)
    node = net.add_plugin_v3(inputs, [], p)
    node.name = name
    return node.get_output(0)


def constant(net, name, tensor):
    arr = blob(tensor)
    layer = net.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
    layer.name = name
    return layer.get_output(0)


def build():
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = net.add_input("x", trt.float16, (M, D))
    for i in range(args.depth):
        layer = WEIGHTED[i % len(WEIGHTED)]
        w = {n: constant(net, f"L{layer}.{n}.{i}", E[f"L{layer}.{n}"])
             for n in ("o_packed", "o_sfb", "gu_il_packed", "gu_il_sfb",
                       "down_packed", "down_sfb")}
        kv = net.add_slice(x, (0, 0), (M, HD), (1, 1))
        kv.name = f"kv_{i}"
        attn = plugin(net, "FlashrtFa4Attention",
                      [("mode", 0), ("NH", NH), ("head_dim", HD), ("batch", 1)],
                      [("scale", 0.0)], [x, kv.get_output(0), kv.get_output(0)], f"attn_{i}")
        proj = plugin(net, "FlashrtNvfp4Linear",
                      [("N", D), ("K", D), ("norm_mode", 0), ("epilogue", 0),
                       ("variant", args.o_variant), ("opt_mask", 0)], [("eps", 1e-6)],
                      [attn, w["o_packed"], w["o_sfb"]], f"o_proj_{i}")
        x = plugin(net, "FlashrtNvfp4Mlp",
                   [("D", D), ("H", H), ("norm_mode", 0), ("gate_mode", 0),
                    ("gate_variant", 0), ("down_variant", args.down_variant),
                    ("epilogue", 0), ("opt_mask", 0)], [("eps", 1e-6)],
                   [proj, w["gu_il_packed"], w["gu_il_sfb"], w["down_packed"], w["down_sfb"]],
                   f"ffn_{i}")
    x.name = "y"
    net.mark_output(x)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
    t0 = time.time()
    ser = builder.build_serialized_network(net, config)
    assert ser is not None, "operator stack build failed"
    print(f"built in {time.time() - t0:.1f}s, {ser.nbytes / 1e6:.0f} MB", flush=True)
    return trt.Runtime(logger).deserialize_cuda_engine(ser)


engine = build()
ctx = engine.create_execution_context()
x_in = E["x_in"][:M].to(dev).contiguous()
y = torch.empty(M, D, dtype=torch.float16, device=dev)
ctx.set_tensor_address("x", x_in.data_ptr())
ctx.set_tensor_address("y", y.data_ptr())
assert ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
finite = bool(torch.isfinite(y).all())

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
for _ in range(10):
    g.replay()
stream.synchronize()
ms = []
for _ in range(args.iters):
    t = time.perf_counter()
    g.replay()
    stream.synchronize()
    ms.append((time.perf_counter() - t) * 1e3)
ms.sort()
median, p90 = ms[len(ms) // 2], ms[int(len(ms) * 0.9)]
print(f"operator stack: {args.depth} blocks x (attention + projection + FFN)  "
      f"median {median:.3f} ms  p90 {p90:.3f}  {median / args.depth * 1000:.1f} us/block  "
      f"finite={finite}")
if args.json:
    with open(args.json, "w") as f:
        json.dump({"depth": args.depth, "M": M, "D": D, "H": H, "median_ms": median,
                   "p90_ms": p90, "finite": finite}, f, indent=2)
    print("wrote", args.json)
print("BENCH_GRANULARITY_DONE")
