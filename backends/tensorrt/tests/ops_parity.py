"""Bitwise check of the operator plugins against FlashRT.

The recording tools save one encoder layer and one SigLIP layer at the
boundaries the operator plugins implement (`ops.*` in the dumps), taken from
the per-layer reference that is itself checked bit for bit against the FlashRT
library. Each operator here is given that recorded input and compared with that
recorded output, eagerly and under an outer CUDA graph.

usage: ops_parity.py <plugin.so> <encoder_all.safetensors> <siglip_all.safetensors>
"""
import os
import sys

try:
    import tensorrt  # noqa: F401
except ImportError:  # JetPack installs the TensorRT bindings for the system Python
    sys.path.append(f"/usr/lib/python3.{sys.version_info.minor}/dist-packages")

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

plugin_path, encoder_path, siglip_path = sys.argv[1:4]
logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)
dev = torch.device("cuda")
stream = torch.cuda.current_stream()

E = load_file(encoder_path)
S = load_file(siglip_path)
Se, D_e, H_e, NH_e, HD_e, _total, o_variant, down_variant, _L = E["meta"].tolist()
M_s, D_s, _H_s, NH_s, HD_s, _L_s, views, _spv, H_s_pad, _De, up_variant, down_variant_s = \
    S["meta"].tolist()[:12]
le, ls = int(E["ops.meta"][0]), int(S["ops.meta"][0])
print(f"encoder layer {le} Se={Se}  siglip layer {ls} S={M_s}")

keep = []


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


def check(label, op, ints, floats, inputs, expect):
    """inputs: (name, tensor, kind) with kind "in" (runtime) or "blob"/"half" (constant)."""
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors, runtime = [], []
    for name, t, kind in inputs:
        if kind == "in":
            x = net.add_input(name, trt.float16, tuple(t.shape))
            runtime.append((name, t))
        else:
            arr = blob(t) if kind == "blob" else t.numpy()
            layer = net.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
            layer.name = name
            x = layer.get_output(0)
        tensors.append(x)
    plugin = registry.get_creator(op, "1", "").create_plugin(op, fields(ints, floats),
                                                             trt.TensorRTPhase.BUILD)
    node = net.add_plugin_v3(tensors, [], plugin)
    node.name = f"flashrt_{op}"
    out = node.get_output(0)
    out.name = "y"
    net.mark_output(out)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
    ser = builder.build_serialized_network(net, config)
    assert ser is not None, f"{label}: build failed"
    engine = trt.Runtime(logger).deserialize_cuda_engine(ser)
    ctx = engine.create_execution_context()

    held = []
    for name, t in runtime:
        d = t.to(dev).contiguous()
        held.append(d)
        ctx.set_tensor_address(name, d.data_ptr())
    ref = expect.to(dev)
    y = torch.empty_like(ref)
    ctx.set_tensor_address("y", y.data_ptr())

    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    eager = torch.equal(y, ref)
    diff = int((y != ref).sum())

    y.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    g.replay()
    stream.synchronize()
    graph = torch.equal(y, ref)
    print(f"{label:28s} {op:20s} bitwise eager={eager} graph={graph}"
          + ("" if eager else f"  differing elements {diff}/{ref.numel()}"))
    del ctx, engine, g
    torch.cuda.empty_cache()
    return eager and graph


ok = True

# Encoder attention: FlashAttention-4 over the recorded queries and the layer's
# own K/V rows.
ok &= check(
    "encoder attention", "FlashrtFa4Attention",
    [("mode", 0), ("NH", NH_e), ("head_dim", HD_e), ("batch", 1)], [("scale", 0.0)],
    [("q", E["ops.attn_q"], "in"),
     ("k", E[f"L{le}.k_out"].reshape(Se, HD_e), "in"),
     ("v", E[f"L{le}.v_out"].reshape(Se, HD_e), "in")],
    E["ops.attn_out"])

# Encoder output projection: quantize -> NVFP4 GEMM -> += residual.
ok &= check(
    "encoder output projection", "FlashrtNvfp4Linear",
    [("N", D_e), ("K", D_e), ("norm_mode", 0), ("epilogue", 1), ("variant", o_variant),
     ("opt_mask", 1 << 4)], [("eps", 1e-6)],
    [("x", E["ops.attn_out"], "in"),
     ("w_packed", E[f"L{le}.o_packed"], "blob"), ("w_sfb", E[f"L{le}.o_sfb"], "blob"),
     ("residual", E["ops.o_res"], "in")],
    E["ops.o_out"])

# Encoder FFN: RMSNorm x AWQ -> NVFP4, interleaved GeGLU gate/up, down into the
# residual.
ok &= check(
    "encoder FFN", "FlashrtNvfp4Mlp",
    [("D", D_e), ("H", H_e), ("norm_mode", 1), ("gate_mode", 0), ("gate_variant", 0),
     ("down_variant", down_variant), ("epilogue", 1), ("opt_mask", (1 << 2) | (1 << 5))],
    [("eps", 1e-6)],
    [("x", E["ops.o_out"], "in"),
     ("gu_packed", E[f"L{le}.gu_il_packed"], "blob"),
     ("gu_sfb", E[f"L{le}.gu_il_sfb"], "blob"),
     ("down_packed", E[f"L{le}.down_packed"], "blob"),
     ("down_sfb", E[f"L{le}.down_sfb"], "blob"),
     ("awq", E[f"L{le}.awq_inv_s_gu"], "half"),
     ("residual", E["ops.o_out"], "in")],
    E["ops.ffn_out"])

# SigLIP attention: FlashAttention-4, head_dim 72, one batch entry per camera.
ok &= check(
    "siglip attention", "FlashrtFa4Attention",
    [("mode", 1), ("NH", NH_s), ("head_dim", HD_s), ("batch", views)], [("scale", 0.0)],
    [("q", S["ops.attn_q"], "in"), ("k", S["ops.attn_k"], "in"),
     ("v", S["ops.attn_v"], "in")],
    S["ops.attn_out"])

# SigLIP FFN: LayerNorm x AWQ -> NVFP4, up with bias and GELU, down with bias
# and residual. The LayerNorm epsilon is the pipeline's, not this test's.
ok &= check(
    "siglip FFN", "FlashrtNvfp4Mlp",
    [("D", D_s), ("H", H_s_pad), ("norm_mode", 2), ("gate_mode", 1),
     ("gate_variant", up_variant), ("down_variant", down_variant_s), ("epilogue", 2),
     ("opt_mask", 0b111111)], [("eps", 1e-5)],
    [("x", S["ops.ffn_in"], "in"),
     ("up_packed", S[f"L{ls}.up_packed"], "blob"), ("up_sfb", S[f"L{ls}.up_sfb"], "blob"),
     ("down_packed", S[f"L{ls}.down_packed"], "blob"),
     ("down_sfb", S[f"L{ls}.down_sfb"], "blob"),
     ("gamma", S[f"L{ls}.ln_ffn_w"], "half"), ("beta", S[f"L{ls}.ln_ffn_b"], "half"),
     ("awq", S[f"L{ls}.awq_inv_s"], "half"), ("up_b", S[f"L{ls}.up_b"], "half"),
     ("down_b", S[f"L{ls}.down_b"], "half"), ("residual", S["ops.ffn_in"], "in")],
    S["ops.ffn_out"])

print("OPS_PARITY_PASS" if ok else "OPS_PARITY_FAIL")
sys.exit(0 if ok else 1)
