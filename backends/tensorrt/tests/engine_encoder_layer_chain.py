"""M2: the whole pi0.5 encoder as a chain of FlashRT layer plugins in one
TensorRT engine. Bitwise check of the final residual stream and every layer's
K/V against the FlashRT library, eager and under an outer CUDA graph.

usage: engine_encoder_layer_chain.py <plugin.so> <encoder_dump.safetensors>
"""
import sys
import time

sys.path.append("/usr/lib/python3.12/dist-packages")

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

plugin_path, dump_path = sys.argv[1:3]
logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)
T = load_file(dump_path)
Se, D, H, NH, HD, total_keys, o_variant, down_variant, L_full = T["meta"].tolist()
import os
L = int(os.environ.get("STAGE_LAYERS", L_full))
NO_CONCAT = os.environ.get("NO_CONCAT") == "1"
print(f"Se={Se} L={L} (of {L_full}) no_concat={NO_CONCAT}")


def u8(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
x = network.add_input("x", trt.float16, (-1, D))
rope = network.add_input("rope", trt.float16, (-1, HD))
keep = []


def const(name, arr):
    arr = np.ascontiguousarray(arr)
    keep.append(arr)
    layer = network.add_constant(arr.shape, trt.Weights(arr))
    layer.name = name
    return layer.get_output(0)


dummy_i32 = np.zeros(1, dtype=np.int32)
dummy_f16 = np.zeros(D, dtype=np.float16)
k_outs, v_outs = [], []
for l in range(L):
    last = l == L_full - 1
    P = f"L{l}."
    ins = [x, rope,
           const(P + "qkv_w", u8(T[P + "qkv_w"])),
           const(P + "qkv_scale", T[P + "act_scale_qkv"].numpy().astype(np.float32))]
    if last:
        ins += [const(P + n, dummy_i32) for n in ("o_packed", "o_sfb")]
        ins += [const(P + "awq_inv_s", dummy_f16)]
        ins += [const(P + n, dummy_i32) for n in ("gu_il_packed", "gu_il_sfb", "down_packed", "down_sfb")]
    else:
        ins += [const(P + "o_packed", u8(T[P + "o_packed"])),
                const(P + "o_sfb", u8(T[P + "o_sfb"])),
                const(P + "awq_inv_s", T[P + "awq_inv_s_gu"].numpy().astype(np.float16)),
                const(P + "gu_il_packed", u8(T[P + "gu_il_packed"])),
                const(P + "gu_il_sfb", u8(T[P + "gu_il_sfb"])),
                const(P + "down_packed", u8(T[P + "down_packed"])),
                const(P + "down_sfb", u8(T[P + "down_sfb"]))]
    fields = trt.PluginFieldCollection()
    for name, value in (("D", D), ("H", H), ("NH", NH), ("HD", HD), ("attn_o_variant", o_variant),
                        ("down_variant", down_variant), ("last", int(last))):
        arr = np.array([value], dtype=np.int32); keep.append(arr)
        fields.append(trt.PluginField(name, arr, trt.PluginFieldType.INT32))
    arr = np.array([float(T[P + "alpha_qkv"][0])], dtype=np.float32); keep.append(arr)
    fields.append(trt.PluginField("qkv_alpha", arr, trt.PluginFieldType.FLOAT32))
    plugin = registry.get_creator("Pi05EncoderLayer", "1", "").create_plugin(
        "Pi05EncoderLayer", fields, trt.TensorRTPhase.BUILD)
    layer = network.add_plugin_v3(ins, [], plugin)
    layer.name = f"flashrt_pi05_encoder_layer_{l}"
    x = layer.get_output(0)
    k_outs.append(layer.get_output(1)); v_outs.append(layer.get_output(2))

x.name = "x_out"; network.mark_output(x)
if NO_CONCAT:
    for l in range(L):
        k_outs[l].name = f"k{l}"; network.mark_output(k_outs[l])
        v_outs[l].name = f"v{l}"; network.mark_output(v_outs[l])
else:
    for name, outs in (("k", k_outs), ("v", v_outs)):
        cat = network.add_concatenation(outs); cat.axis = 0
        cat.get_output(0).name = name
        network.mark_output(cat.get_output(0))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
profile = builder.create_optimization_profile()
profile.set_shape("x", (16, D), (Se, D), (1024, D))
profile.set_shape("rope", (16, HD), (Se, HD), (1024, HD))
config.add_optimization_profile(profile)
t0 = time.time()
serialized = builder.build_serialized_network(network, config)
assert serialized is not None, "engine build failed"
print(f"engine built in {time.time() - t0:.1f}s, {serialized.nbytes / 1e6:.1f} MB")

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
ctx = engine.create_execution_context()
dev = torch.device("cuda")
ref_x = T["x_out"].to(dev) if L == L_full else None
ref_k = torch.cat([T[f"L{l}.k_out"] for l in range(L)]).to(dev).view(L * Se, HD)
ref_v = torch.cat([T[f"L{l}.v_out"] for l in range(L)]).to(dev).view(L * Se, HD)
in_x = T["x_in"].to(dev).clone(); in_rope = T["rope"].to(dev).clone()
outs = {"x_out": torch.empty(Se, D, dtype=torch.float16, device=dev)}
if NO_CONCAT:
    for l in range(L):
        outs[f"k{l}"] = torch.empty(Se, HD, dtype=torch.float16, device=dev)
        outs[f"v{l}"] = torch.empty(Se, HD, dtype=torch.float16, device=dev)
else:
    outs["k"] = torch.empty(L * Se, HD, dtype=torch.float16, device=dev)
    outs["v"] = torch.empty(L * Se, HD, dtype=torch.float16, device=dev)
ctx.set_input_shape("x", (Se, D)); ctx.set_input_shape("rope", (Se, HD))
ctx.set_tensor_address("x", in_x.data_ptr()); ctx.set_tensor_address("rope", in_rope.data_ptr())
for n, t in outs.items():
    ctx.set_tensor_address(n, t.data_ptr())
side = torch.cuda.Stream()


def check(tag):
    k_all = torch.cat([outs[f"k{l}"] for l in range(L)]) if NO_CONCAT else outs["k"]
    v_all = torch.cat([outs[f"v{l}"] for l in range(L)]) if NO_CONCAT else outs["v"]
    same = {"k": torch.equal(k_all, ref_k), "v": torch.equal(v_all, ref_v)}
    if ref_x is not None:
        same["x"] = torch.equal(outs["x_out"], ref_x)
    print(f"{tag}: bitwise {same}")
    return all(same.values())


with torch.cuda.stream(side):
    assert ctx.execute_async_v3(side.cuda_stream)
    side.synchronize()
    ok_eager = check("TRT eager")

    def timed(fn, n=100, warm=10):
        for _ in range(warm):
            fn()
        side.synchronize()
        ms = []
        for _ in range(n):
            t = time.perf_counter(); fn(); side.synchronize()
            ms.append((time.perf_counter() - t) * 1e3)
        ms.sort()
        return ms[len(ms) // 2], ms[int(len(ms) * 0.9)]

    eager_ms = timed(lambda: ctx.execute_async_v3(side.cuda_stream))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        ctx.execute_async_v3(side.cuda_stream)
    for t in outs.values():
        t.zero_()
    graph.replay(); side.synchronize()
    ok_graph = check("TRT + outer CUDA graph")
    graph_ms = timed(graph.replay)
print(f"encoder stage latency: eager median {eager_ms[0]:.2f} ms p90 {eager_ms[1]:.2f} | "
      f"graph median {graph_ms[0]:.2f} ms p90 {graph_ms[1]:.2f}")
print("M2_ENCODER_" + ("PASS" if ok_eager and ok_graph else "FAIL"))
