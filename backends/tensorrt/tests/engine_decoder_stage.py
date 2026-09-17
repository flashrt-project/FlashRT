"""the whole pi0.5 action decoder (10 denoise steps) as one FlashRT
plugin in a TensorRT engine. The prefix K/V rows are engine inputs; the
full cache lives in the plugin workspace. Bitwise check of the final raw
actions against the FlashRT library, eager and under an outer CUDA graph.

usage: engine_decoder_stage.py <plugin.so> <decoder_steps.safetensors>
"""
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

plugin_path, dump_path = sys.argv[1:3]
logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)
T = load_file(dump_path)
m = T["meta"].tolist()
S, D, H, NH, HD, L, steps, enc_seq, total_keys = m[:9]
v_qkv, v_o, v_gu, v_down = m[9:13]
dt = float(T["dt"][0])
print(f"S={S} L={L} steps={steps} enc_seq={enc_seq}")


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
noise = network.add_input("noise", trt.float16, (S, 32))
pk = network.add_input("prefix_k", trt.float16, (-1,))
pv = network.add_input("prefix_v", trt.float16, (-1,))
keep = []


def const(name, arr):
    arr = np.ascontiguousarray(arr)
    keep.append(arr)
    layer = network.add_constant(arr.shape, trt.Weights(arr))
    layer.name = name
    return layer.get_output(0)


f16 = lambda n: T[n].numpy().astype(np.float16).reshape(-1) if n in ("sa", "sf", "fs") else T[n].numpy().astype(np.float16)  # noqa: E731
ins = [noise, pk, pv] + [const(n, f16(n)) for n in ("ain_w", "ain_b", "aow", "aob", "rope", "sa", "sf", "fs")]
ins += [const(n, blob(T[n])) for n in ("qw_fp4", "qw_sfb", "ow_fp4", "ow_sfb", "gwil_fp4", "gwil_sfb", "dw_fp4", "dw_sfb")]
fields = trt.PluginFieldCollection()
for name, value in (("S", S), ("D", D), ("H", H), ("NH", NH), ("HD", HD), ("L", L), ("steps", steps),
                    ("v_qkv", v_qkv), ("v_o", v_o), ("v_gu", v_gu), ("v_down", v_down)):
    arr = np.array([value], dtype=np.int32); keep.append(arr)
    fields.append(trt.PluginField(name, arr, trt.PluginFieldType.INT32))
arr = np.array([dt], dtype=np.float32); keep.append(arr)
fields.append(trt.PluginField("dt", arr, trt.PluginFieldType.FLOAT32))
plugin = registry.get_creator("Pi05Decoder", "1", "").create_plugin("Pi05Decoder", fields, trt.TensorRTPhase.BUILD)
layer = network.add_plugin_v3(ins, [], plugin)
layer.name = "flashrt_pi05_decoder"
out = layer.get_output(0); out.name = "actions"; network.mark_output(out)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
profile = builder.create_optimization_profile()
for name in ("prefix_k", "prefix_v"):
    profile.set_shape(name, (L * 16 * HD,), (L * enc_seq * HD,), (L * 1024 * HD,))
config.add_optimization_profile(profile)
t0 = time.time()
serialized = builder.build_serialized_network(network, config)
assert serialized is not None, "engine build failed"
print(f"engine built in {time.time() - t0:.1f}s, {serialized.nbytes / 1e6:.1f} MB")
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
ctx = engine.create_execution_context()

dev = torch.device("cuda")
in_noise = T["noise_in"].to(dev).clone()
in_k = torch.cat([T[f"L{l}.k_prefix"].reshape(-1) for l in range(L)]).to(dev).contiguous()
in_v = torch.cat([T[f"L{l}.v_prefix"].reshape(-1) for l in range(L)]).to(dev).contiguous()
ref = T["noise_out"].to(dev)
actions = torch.empty(S, 32, dtype=torch.float16, device=dev)
ctx.set_input_shape("prefix_k", tuple(in_k.shape)); ctx.set_input_shape("prefix_v", tuple(in_v.shape))
for n, t in (("noise", in_noise), ("prefix_k", in_k), ("prefix_v", in_v), ("actions", actions)):
    ctx.set_tensor_address(n, t.data_ptr())
side = torch.cuda.Stream()



def check(tag):
    same = torch.equal(actions, ref)
    print(f"{tag}: actions bitwise {same} (max |d| {(actions.float() - ref.float()).abs().max().item():.3g})")
    return same


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
    actions.zero_()
    graph.replay(); side.synchronize()
    ok_graph = check("TRT + outer CUDA graph")
    graph_ms = timed(graph.replay)
print(f"decoder stage latency (10 steps): eager median {eager_ms[0]:.2f} ms p90 {eager_ms[1]:.2f} | "
      f"graph median {graph_ms[0]:.2f} ms p90 {graph_ms[1]:.2f}")
print("ENGINE_DECODER_STAGE_" + ("PASS" if ok_eager and ok_graph else "FAIL"))
