"""one pi0.5 encoder layer as a TensorRT engine with the FlashRT
plugin. Checks bitwise parity against the FlashRT reference dump, then again
under an outer CUDA graph, and reports latency for both.

usage: engine_encoder_layer.py <plugin.so> <layer.safetensors> <engine_out>
"""
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

plugin_path, dump_path, engine_path = sys.argv[1:4]
logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)

T = load_file(dump_path)
meta = T["meta"].tolist()
Se, D, H, NH, HD = meta[0], meta[1], meta[2], meta[3], meta[4]
o_variant, down_variant = meta[6], meta[7]
alpha = float(T["alpha_qkv"][0])
print(f"Se={Se} D={D} H={H} NH={NH} HD={HD} variants o={o_variant} down={down_variant} alpha={alpha}")


def u8(t):
    """Byte blob carried as INT32 (TensorRT constants reject UINT8)."""
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
x = network.add_input("x", trt.float16, (-1, D))
rope = network.add_input("rope", trt.float16, (-1, HD))


def const(name, arr):
    layer = network.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
    layer.name = name
    return layer.get_output(0)


weights = [
    const("qkv_w", u8(T["qkv_w"])),
    const("qkv_scale", T["act_scale_qkv"].numpy().astype(np.float32)),
    const("o_packed", u8(T["o_packed"])),
    const("o_sfb", u8(T["o_sfb"])),
    const("awq_inv_s", T["awq_inv_s_gu"].numpy().astype(np.float16)),
    const("gu_il_packed", u8(T["gu_il_packed"])),
    const("gu_il_sfb", u8(T["gu_il_sfb"])),
    const("down_packed", u8(T["down_packed"])),
    const("down_sfb", u8(T["down_sfb"])),
]

# PluginField keeps a raw pointer: the arrays must outlive plugin creation.
field_arrays = []
fields = trt.PluginFieldCollection()
for name, value in (("D", D), ("H", H), ("NH", NH), ("HD", HD),
                    ("attn_o_variant", o_variant), ("down_variant", down_variant), ("last", 0)):
    field_arrays.append(np.array([value], dtype=np.int32))
    fields.append(trt.PluginField(name, field_arrays[-1], trt.PluginFieldType.INT32))
field_arrays.append(np.array([alpha], dtype=np.float32))
fields.append(trt.PluginField("qkv_alpha", field_arrays[-1], trt.PluginFieldType.FLOAT32))
creator = registry.get_creator("Pi05EncoderLayer", "1", "")
plugin = creator.create_plugin("Pi05EncoderLayer", fields, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3([x, rope] + weights, [], plugin)
layer.name = "flashrt_pi05_encoder_layer"
for i, name in enumerate(("x_out", "k", "v")):
    out = layer.get_output(i)
    out.name = name
    network.mark_output(out)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
profile = builder.create_optimization_profile()
profile.set_shape("x", (16, D), (Se, D), (1024, D))
profile.set_shape("rope", (16, HD), (Se, HD), (1024, HD))
config.add_optimization_profile(profile)
t0 = time.time()
serialized = builder.build_serialized_network(network, config)
assert serialized is not None, "engine build failed"
open(engine_path, "wb").write(serialized)
print(f"engine built in {time.time() - t0:.1f}s, {serialized.nbytes / 1e6:.1f} MB")

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
ctx = engine.create_execution_context()
dev = torch.device("cuda")
x_in = T["x_in"].to(dev)
rope_in = T["rope"].to(dev)
ref = {"x_out": T["x_out"].to(dev), "k": T["k_out"].to(dev).view(Se, HD), "v": T["v_out"].to(dev).view(Se, HD)}

in_x = x_in.clone()
in_rope = rope_in.clone()
outs = {n: torch.empty((Se, D) if n == "x_out" else (Se, HD), dtype=torch.float16, device=dev) for n in ref}
ctx.set_input_shape("x", (Se, D))
ctx.set_input_shape("rope", (Se, HD))
ctx.set_tensor_address("x", in_x.data_ptr())
ctx.set_tensor_address("rope", in_rope.data_ptr())
for n, t in outs.items():
    ctx.set_tensor_address(n, t.data_ptr())
stream = torch.cuda.current_stream()


def check(tag):
    same = {n: torch.equal(outs[n], ref[n]) for n in ref}
    print(f"{tag}: bitwise {same}")
    return all(same.values())


assert ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
ok_eager = check("TRT eager")


def timed(fn, n=300, warm=30):
    for _ in range(warm):
        fn()
    stream.synchronize()
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        stream.synchronize()
        ms.append((time.perf_counter() - t) * 1e3)
    ms.sort()
    return ms[len(ms) // 2], ms[int(len(ms) * 0.9)]


eager_ms = timed(lambda: ctx.execute_async_v3(stream.cuda_stream))

# Outer CUDA graph, the way the openpi TensorRT runtime wraps an engine.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
for t in outs.values():
    t.zero_()
graph.replay()
torch.cuda.synchronize()
ok_graph = check("TRT + outer CUDA graph")
graph_ms = timed(graph.replay)
print(f"latency eager median {eager_ms[0]:.3f} ms p90 {eager_ms[1]:.3f} | graph replay median {graph_ms[0]:.3f} ms p90 {graph_ms[1]:.3f}")
print("ENGINE_ENCODER_LAYER_" + ("PASS" if ok_eager and ok_graph else "FAIL"))
