"""Latency of the FlashRT operator plugins at the pi0.5 shapes.

One engine per operator, real recorded weights and activations, timed under an
outer CUDA graph. The same numbers measured for TensorRT's own path are what
the operator comparison in docs/tensorrt_ops.md reports.

usage: bench_ops.py <plugin.so> <encoder_all.safetensors> <siglip_all.safetensors> [--json out.json]
"""
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

plugin_path, encoder_path, siglip_path = sys.argv[1:4]
json_out = None
if "--json" in sys.argv:
    json_out = sys.argv[sys.argv.index("--json") + 1]
save_dir = None
if "--save-engines" in sys.argv:
    save_dir = sys.argv[sys.argv.index("--save-engines") + 1]
    os.makedirs(save_dir, exist_ok=True)

logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)
dev = torch.device("cuda")
stream = torch.cuda.current_stream()

E = load_file(encoder_path)
S = load_file(siglip_path)
Se, D_e, H_e, NH_e, HD_e, _total, o_variant, down_variant, _L = E["meta"].tolist()
M_s, D_s, _H_s, NH_s, HD_s, _L_s, views, spv, H_s_pad, _De, up_variant = S["meta"].tolist()[:11]

keep = []  # PluginField arrays must outlive plugin creation


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


def make_fields(ints, floats=()):
    fc = trt.PluginFieldCollection()
    for name, value in ints:
        keep.append(np.array([value], dtype=np.int32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.INT32))
    for name, value in floats:
        keep.append(np.array([value], dtype=np.float32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.FLOAT32))
    return fc


def build(op, fields, inputs, out_cols, m):
    """inputs: list of (name, tensor, kind) with kind in {"in", "blob", "half"}."""
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors, runtime_inputs = [], []
    for name, t, kind in inputs:
        if kind == "in":
            x = net.add_input(name, trt.float16, (-1, int(t.shape[1])))
            runtime_inputs.append((name, t))
        else:
            arr = blob(t) if kind == "blob" else t.numpy()
            layer = net.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
            layer.name = name
            x = layer.get_output(0)
        tensors.append(x)
    plugin = registry.get_creator(op, "1", "").create_plugin(op, fields, trt.TensorRTPhase.BUILD)
    node = net.add_plugin_v3(tensors, [], plugin)
    node.name = f"flashrt_{op}"
    out = node.get_output(0)
    out.name = "y"
    net.mark_output(out)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
    prof = builder.create_optimization_profile()
    for name, t in runtime_inputs:
        prof.set_shape(name, (m, int(t.shape[1])), (m, int(t.shape[1])), (m, int(t.shape[1])))
    config.add_optimization_profile(prof)
    ser = builder.build_serialized_network(net, config)
    assert ser is not None, f"{op} build failed"
    if save_dir:
        # Saved so trtexec can time this operator exactly like the TensorRT-native
        # subgraphs: same flags, same CUDA graph, same GPU-compute clock.
        tag = build.tag
        with open(os.path.join(save_dir, f"{tag}.plan"), "wb") as f:
            f.write(ser)
        shapes = ",".join(f"{n}:{m}x{int(t.shape[1])}" for n, t in runtime_inputs)
        with open(os.path.join(save_dir, f"{tag}.shapes"), "w") as f:
            f.write(shapes + "\n")
    engine = trt.Runtime(logger).deserialize_cuda_engine(ser)
    return engine, runtime_inputs, out_cols


def timed(fn, n=200, warm=20):
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


def run_case(label, op, fields, inputs, out_cols, m, note="", tag=None):
    build.tag = tag or label.split()[0]
    engine, runtime_inputs, cols = build(op, fields, inputs, out_cols, m)
    ctx = engine.create_execution_context()
    held = []
    for name, t in runtime_inputs:
        d = t.to(dev).contiguous()
        held.append(d)
        ctx.set_input_shape(name, tuple(d.shape))
        ctx.set_tensor_address(name, d.data_ptr())
    y = torch.empty(m, cols, dtype=torch.float16, device=dev)
    ctx.set_tensor_address("y", y.data_ptr())
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    finite = bool(torch.isfinite(y).all())

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    med, p90 = timed(g.replay)
    print(f"{label:22s} {op:20s} {med * 1000:8.1f} us  p90 {p90 * 1000:8.1f}  finite={finite}  {note}")
    del ctx, engine, g
    torch.cuda.empty_cache()
    return {"case": label, "op": op, "median_us": med * 1000, "p90_us": p90 * 1000,
            "finite": finite, "note": note}


results = []

# Encoder FFN: RMSNorm x AWQ -> NVFP4, interleaved GeGLU gate/up, down into the residual.
results.append(run_case(
    f"encoder_mlp M={Se}", "FlashrtNvfp4Mlp",
    make_fields([("D", D_e), ("H", H_e), ("norm_mode", 1), ("gate_mode", 0),
                 ("gate_variant", 0), ("down_variant", down_variant), ("epilogue", 1),
                 ("opt_mask", (1 << 2) | (1 << 5))], [("eps", 1e-6)]),
    [("x", E["x_in"], "in"),
     ("gu_packed", E["L0.gu_il_packed"], "blob"), ("gu_sfb", E["L0.gu_il_sfb"], "blob"),
     ("down_packed", E["L0.down_packed"], "blob"), ("down_sfb", E["L0.down_sfb"], "blob"),
     ("awq", E["L0.awq_inv_s_gu"], "half"), ("residual", E["x_in"], "in")],
    D_e, Se, f"D={D_e} H={H_e}"))

# Encoder attention output projection: quantize -> NVFP4 GEMM -> += residual.
results.append(run_case(
    f"encoder_o M={Se}", "FlashrtNvfp4Linear",
    make_fields([("N", D_e), ("K", D_e), ("norm_mode", 0), ("epilogue", 1),
                 ("variant", o_variant), ("opt_mask", 1 << 4)], [("eps", 1e-6)]),
    [("x", E["x_in"], "in"),
     ("w_packed", E["L0.o_packed"], "blob"), ("w_sfb", E["L0.o_sfb"], "blob"),
     ("residual", E["x_in"], "in")],
    D_e, Se, f"N={D_e} K={D_e}"))

# The same FFN without the normalization, matching a boundary that starts at
# the already normalized activation.
results.append(run_case(
    f"encoder_mlp_plain M={Se}", "FlashrtNvfp4Mlp",
    make_fields([("D", D_e), ("H", H_e), ("norm_mode", 0), ("gate_mode", 0),
                 ("gate_variant", 0), ("down_variant", down_variant), ("epilogue", 0),
                 ("opt_mask", 0)], [("eps", 1e-6)]),
    [("x", E["x_in"], "in"),
     ("gu_packed", E["L0.gu_il_packed"], "blob"), ("gu_sfb", E["L0.gu_il_sfb"], "blob"),
     ("down_packed", E["L0.down_packed"], "blob"), ("down_sfb", E["L0.down_sfb"], "blob")],
    D_e, Se, f"D={D_e} H={H_e}, no norm/residual", tag="encoder_mlp_plain"))

# The same projection without the residual: the boundary TensorRT's own
# subgraph has (quantize + GEMM only).
results.append(run_case(
    f"encoder_o_plain M={Se}", "FlashrtNvfp4Linear",
    make_fields([("N", D_e), ("K", D_e), ("norm_mode", 0), ("epilogue", 0),
                 ("variant", o_variant), ("opt_mask", 0)], [("eps", 1e-6)]),
    [("x", E["x_in"], "in"),
     ("w_packed", E["L0.o_packed"], "blob"), ("w_sfb", E["L0.o_sfb"], "blob")],
    D_e, Se, f"N={D_e} K={D_e}, no residual", tag="encoder_o_plain"))

# SigLIP FFN: LayerNorm x AWQ -> NVFP4, up with bias and GELU, down with bias and residual.
results.append(run_case(
    f"siglip_mlp M={M_s}", "FlashrtNvfp4Mlp",
    make_fields([("D", D_s), ("H", H_s_pad), ("norm_mode", 2), ("gate_mode", 1),
                 ("gate_variant", up_variant), ("down_variant", 0), ("epilogue", 2),
                 ("opt_mask", 0b111111)], [("eps", 1e-6)]),
    [("x", S["L0.x_out"], "in"),
     ("up_packed", S["L0.up_packed"], "blob"), ("up_sfb", S["L0.up_sfb"], "blob"),
     ("down_packed", S["L0.down_packed"], "blob"), ("down_sfb", S["L0.down_sfb"], "blob"),
     ("gamma", S["L0.ln_ffn_w"], "half"), ("beta", S["L0.ln_ffn_b"], "half"),
     ("awq", S["L0.awq_inv_s"], "half"), ("up_b", S["L0.up_b"], "half"),
     ("down_b", S["L0.down_b"], "half"), ("residual", S["L0.x_out"], "in")],
    D_s, M_s, f"D={D_s} H={H_s_pad}"))

results.append(run_case(
    f"siglip_mlp_plain M={M_s}", "FlashrtNvfp4Mlp",
    make_fields([("D", D_s), ("H", H_s_pad), ("norm_mode", 0), ("gate_mode", 1),
                 ("gate_variant", up_variant), ("down_variant", 0), ("epilogue", 0),
                 ("opt_mask", 1 << 3)], [("eps", 1e-6)]),
    [("x", S["L0.x_out"], "in"),
     ("up_packed", S["L0.up_packed"], "blob"), ("up_sfb", S["L0.up_sfb"], "blob"),
     ("down_packed", S["L0.down_packed"], "blob"), ("down_sfb", S["L0.down_sfb"], "blob"),
     ("up_b", S["L0.up_b"], "half")],
    D_s, M_s, f"D={D_s} H={H_s_pad}, no norm/residual", tag="siglip_mlp_plain"))

# Encoder attention: FlashAttention-4, head_dim 256, grouped queries over one KV head.
q = torch.zeros(Se, NH_e * HD_e, dtype=torch.float16)
kv = torch.zeros(Se, HD_e, dtype=torch.float16)
q.normal_(0, 0.3)
kv.normal_(0, 0.3)
results.append(run_case(
    f"encoder_attn Sq={Se}", "FlashrtFa4Attention",
    make_fields([("mode", 0), ("NH", NH_e), ("head_dim", HD_e), ("batch", 1)], [("scale", 0.0)]),
    [("q", q, "in"), ("k", kv, "in"), ("v", kv, "in")],
    NH_e * HD_e, Se, f"NH={NH_e} HD={HD_e} 1 KV head"))

# SigLIP attention: FlashAttention-4, head_dim 72, one batch entry per camera.
qs = torch.zeros(M_s, NH_s * HD_s, dtype=torch.float16)
qs.normal_(0, 0.3)
results.append(run_case(
    f"siglip_attn S={spv}x{views}", "FlashrtFa4Attention",
    make_fields([("mode", 1), ("NH", NH_s), ("head_dim", HD_s), ("batch", views)],
                [("scale", 0.0)]),
    [("q", qs, "in"), ("k", qs, "in"), ("v", qs, "in")],
    NH_s * HD_s, M_s, f"NH={NH_s} HD={HD_s}"))

if json_out:
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", json_out)
print("BENCH_OPS_DONE")
