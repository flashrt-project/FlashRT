"""pi0.5 SigLIP as TensorRT engines on FlashRT plugins, checked bitwise
against the FlashRT library forward (tools/reference/dump_siglip.py), eager and under
an outer CUDA graph, with latency for both.

  layers: 27 chained Pi05SiglipLayer nodes, x_embed -> SigLIP output
  stage:  one Pi05Siglip node, fp16 images -> projected image tokens

usage: engine_siglip.py <plugin.so> <siglip_all.safetensors>
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
S, D, H, NH, HD, L, NV, SPV, H_PAD, DE, UP_VARIANT, DOWN_VARIANT = T["meta"].tolist()
assert DOWN_VARIANT == 0
alpha = T["alpha"].tolist()
LAYER = ("ln_attn_w", "ln_attn_b", "qkv_w", "qkv_b", "o_w", "o_b", "ln_ffn_w", "ln_ffn_b",
         "awq_inv_s", "up_packed", "up_sfb", "up_b", "down_packed", "down_sfb", "down_b")
BLOBS = {"qkv_w", "o_w", "up_packed", "up_sfb", "down_packed", "down_sfb"}
print(f"S={S} views={NV} L={L}")


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


keep = []  # PluginField arrays must outlive plugin creation


def fields(ints, floats):
    fc = trt.PluginFieldCollection()
    for name, value in ints:
        keep.append(np.array([value], dtype=np.int32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.INT32))
    for name, values in floats:
        keep.append(np.array(values, dtype=np.float32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.FLOAT32))
    return fc


def build(kind):
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    def const(name, arr):
        layer = net.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
        layer.name = name
        return layer.get_output(0)

    def fp16(name, key=None):
        return const(name, T[key or name].numpy().astype(np.float16))

    def layer_inputs(l):
        p = f"L{l}."
        return [const(p + n, blob(T[p + n])) if n in BLOBS else fp16(p + n) for n in LAYER]

    common = [("D", D), ("H_pad", H_PAD), ("NH", NH), ("HD", HD), ("spv", SPV), ("up_variant", UP_VARIANT)]
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
    prof = builder.create_optimization_profile()
    if kind == "layers":
        x = net.add_input("x", trt.float16, (-1, D))
        creator = registry.get_creator("Pi05SiglipLayer", "1", "")
        cur = x
        for l in range(L):
            plugin = creator.create_plugin("Pi05SiglipLayer", fields(
                common, [("qkv_alpha", [alpha[4 * l]]), ("o_alpha", [alpha[4 * l + 1]])]),
                trt.TensorRTPhase.BUILD)
            node = net.add_plugin_v3([cur] + layer_inputs(l), [], plugin)
            node.name = f"flashrt_siglip_layer_{l}"
            cur = node.get_output(0)
        out = cur
        prof.set_shape("x", (SPV, D), (S, D), (S, D))
    else:
        images = net.add_input("images", trt.float16, (-1, 224, 224, 3))
        head = [fp16("pe_w"), fp16("pe_b"), fp16("pos_emb"), fp16("postln_w"),
                fp16("postln_b"), fp16("proj_w"), fp16("proj_b")]
        tail = []
        for l in range(L):
            tail += layer_inputs(l)
        alphas = []
        for l in range(L):
            alphas += [alpha[4 * l], alpha[4 * l + 1]]
        plugin = registry.get_creator("Pi05Siglip", "1", "").create_plugin(
            "Pi05Siglip", fields(common + [("De", DE), ("L", L)], [("alpha", alphas)]),
            trt.TensorRTPhase.BUILD)
        node = net.add_plugin_v3([images] + head + tail, [], plugin)
        node.name = "flashrt_siglip_stage"
        out = node.get_output(0)
        prof.set_shape("images", (1, 224, 224, 3), (NV, 224, 224, 3), (NV, 224, 224, 3))
    out.name = "out"
    net.mark_output(out)
    config.add_optimization_profile(prof)
    t0 = time.time()
    ser = builder.build_serialized_network(net, config)
    assert ser is not None, f"{kind} build failed"
    print(f"[{kind}] engine built in {time.time() - t0:.1f}s, {ser.nbytes / 1e6:.0f} MB")
    return trt.Runtime(logger).deserialize_cuda_engine(ser)


dev = torch.device("cuda")
stream = torch.cuda.current_stream()


def timed(fn, n=100, warm=10):
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


ok = True
for kind in os.environ.get("KINDS", "layers,stage").split(","):
    engine = build(kind)
    ctx = engine.create_execution_context()
    if kind == "layers":
        inp, in_name, ref = T["x_embed"].to(dev), "x", T["x_sig"].to(dev)
        ctx.set_input_shape("x", (S, D))
    else:
        # FlashRT's uint8 lookup table, the same values a host-side normalization gives
        lut = (torch.arange(256, dtype=torch.float32) / 127.5 - 1.0).to(torch.float16)
        assert torch.equal(lut, T["lut"])
        inp, in_name, ref = lut[T["images_u8"].long()].to(dev).contiguous(), "images", T["tokens"].to(dev)
        ctx.set_input_shape("images", (NV, 224, 224, 3))
    out = torch.empty_like(ref)
    ctx.set_tensor_address(in_name, inp.data_ptr())
    ctx.set_tensor_address("out", out.data_ptr())
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    eager_ok = torch.equal(out, ref)
    d = (out.float() - ref.float()).abs().max().item()
    eager_ms = timed(lambda: ctx.execute_async_v3(stream.cuda_stream))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_ok = torch.equal(out, ref)
    graph_ms = timed(g.replay)
    print(f"[{kind}] bitwise eager={eager_ok} (max|d| {d:.3g}) graph={graph_ok} | eager median "
          f"{eager_ms[0]:.3f} ms p90 {eager_ms[1]:.3f} | graph median {graph_ms[0]:.3f} ms p90 {graph_ms[1]:.3f}")
    ok = ok and eager_ok and graph_ok
    del ctx, engine, g
    torch.cuda.empty_cache()
print("ENGINE_SIGLIP_" + ("PASS" if ok else "FAIL"))
