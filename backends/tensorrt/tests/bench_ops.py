"""Latency of the FlashRT operator plugins at the pi0.5 shapes.

One engine per operator, real recorded weights and activations, timed under an
outer CUDA graph. The recordings come from the dump the stage tests use, so an
operator here runs the bits production runs.

`--save-engines` writes every engine and the shapes it was profiled for, which
is what lets trtexec time these operators exactly like the TensorRT-native
subgraphs they are compared with. `--sweep` walks the tile variants a GEMM
exposes and reports the best one at that operator's own shape.

usage:
  bench_ops.py <plugin.so> <encoder_all.safetensors> <siglip_all.safetensors>
               [--only TAG[,TAG...]] [--sweep TAG[,TAG...]] [--set TAG.FIELD=N]
               [--save-engines DIR] [--json FILE] [--iters N]
"""
import argparse
import ctypes
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

# frt_variant_kind in kernels/frt_ops.h.
VARIANT_GEMM, VARIANT_GATE_BIAS_GELU, VARIANT_DOWN_BIAS_RES = 0, 1, 2

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("plugin")
ap.add_argument("encoder")
ap.add_argument("siglip")
ap.add_argument("--only", default="")
ap.add_argument("--sweep", default="")
ap.add_argument("--set", action="append", default=[], metavar="TAG.FIELD=N")
ap.add_argument("--save-engines", default=None, metavar="DIR")
ap.add_argument("--json", default=None, metavar="FILE")
ap.add_argument("--iters", type=int, default=200)
args = ap.parse_args()
if args.save_engines:
    os.makedirs(args.save_engines, exist_ok=True)

logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(args.plugin)
lib = ctypes.CDLL(args.plugin)
lib.frt_nvfp4_num_variants.restype = ctypes.c_int32
lib.frt_nvfp4_variant_name.restype = ctypes.c_char_p
dev = torch.device("cuda")
stream = torch.cuda.current_stream()

E = load_file(args.encoder)
S = load_file(args.siglip)
Se, D_e, H_e, NH_e, HD_e, _total, o_variant, down_variant, _L = E["meta"].tolist()
M_s, D_s, _H_s, NH_s, HD_s, _L_s, views, spv, H_s_pad, _De, up_variant = S["meta"].tolist()[:11]

keep = []  # PluginField arrays must outlive plugin creation


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


def make_fields(ints, floats):
    fc = trt.PluginFieldCollection()
    for name, value in ints.items():
        keep.append(np.array([value], dtype=np.int32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.INT32))
    for name, value in floats.items():
        keep.append(np.array([value], dtype=np.float32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.FLOAT32))
    return fc


def tile_fields(case):
    """Which attributes of this case are tile variant indices, and of which table.

    The index selects a tile in the table of the kernel the mode and the
    epilogue pick, so the table follows from the other attributes.
    """
    ints, fields = case["ints"], {}
    if case["op"] == "FlashrtNvfp4Linear":
        fields["variant"] = VARIANT_GEMM
    elif case["op"] == "FlashrtNvfp4Mlp":
        if ints["gate_mode"] == 1:  # FRT_GATE_BIAS_GELU
            fields["gate_variant"] = VARIANT_GATE_BIAS_GELU
        fields["down_variant"] = (VARIANT_DOWN_BIAS_RES if ints["epilogue"] == 2
                                  else VARIANT_GEMM)
    return fields


def build(case, ints, tag):
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensors, runtime_inputs = [], []
    for name, t, kind in case["inputs"]:
        if kind == "in":
            x = net.add_input(name, trt.float16, (-1, int(t.shape[1])))
            runtime_inputs.append((name, t))
        else:
            arr = blob(t) if kind == "blob" else t.numpy()
            layer = net.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
            layer.name = name
            x = layer.get_output(0)
        tensors.append(x)
    op = case["op"]
    fields = make_fields(ints, case.get("floats", {}))
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
    rows = case["rows"]
    for name, t in runtime_inputs:
        prof.set_shape(name, (rows, int(t.shape[1])), (rows, int(t.shape[1])),
                       (rows, int(t.shape[1])))
    config.add_optimization_profile(prof)
    ser = builder.build_serialized_network(net, config)
    assert ser is not None, f"{op} build failed"
    if args.save_engines and tag:
        # Saved so trtexec can time this operator exactly like the
        # TensorRT-native subgraphs: same flags, same CUDA graph, same clock.
        with open(os.path.join(args.save_engines, f"{tag}.plan"), "wb") as f:
            f.write(ser)
        shapes = ",".join(f"{n}:{rows}x{int(t.shape[1])}" for n, t in runtime_inputs)
        with open(os.path.join(args.save_engines, f"{tag}.shapes"), "w") as f:
            f.write(shapes + "\n")
    engine = trt.Runtime(logger).deserialize_cuda_engine(ser)
    return engine, runtime_inputs


def timed(fn, n, warm=20):
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


def measure(case, ints, tag, iters):
    """Build and time one configuration. Returns None if the kernel refuses it."""
    engine, runtime_inputs = build(case, ints, tag)
    ctx = engine.create_execution_context()
    held = []
    for name, t in runtime_inputs:
        d = t.to(dev).contiguous()
        held.append(d)
        ctx.set_input_shape(name, tuple(d.shape))
        ctx.set_tensor_address(name, d.data_ptr())
    y = torch.empty(case["rows"], case["out_cols"], dtype=torch.float16, device=dev)
    ctx.set_tensor_address("y", y.data_ptr())
    ok = ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    finite = ok and bool(torch.isfinite(y).all())
    out = None
    if ok and finite:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        med, p90 = timed(g.replay, iters)
        del g
        out = {"case": case["label"], "tag": case["tag"], "op": case["op"],
               "median_us": med * 1000, "p90_us": p90 * 1000, "finite": finite,
               "note": case.get("note", ""),
               "tiles": {f: ints[f] for f in tile_fields(case)}}
    del ctx, engine
    torch.cuda.empty_cache()
    return out


def sweep(case, iters):
    """Walk each tile table at this operator's own shape, one field at a time."""
    fields = tile_fields(case)
    if not fields:
        print(f"{case['tag']}: no tile variants to sweep")
        return None
    ints = dict(case["ints"])
    trail = []
    for field, kind in fields.items():
        n = lib.frt_nvfp4_num_variants(kind)
        best = None
        for idx in range(n):
            probe = dict(ints)
            probe[field] = idx
            try:
                r = measure(case, probe, None, iters)
            except Exception as exc:  # a tile the kernel cannot build at this shape
                r = None
                note = str(exc).splitlines()[0][:60]
            else:
                note = "" if r else "rejected"
            name = lib.frt_nvfp4_variant_name(kind, idx).decode()
            us = f"{r['median_us']:8.1f}" if r else "       -"
            mark = ""
            if r and (best is None or r["median_us"] < best["median_us"]):
                best, mark = dict(r, variant=idx), "  <"
            print(f"  {case['tag']:20s} {field:13s} {idx:3d} {us} us  {name}{mark} {note}",
                  flush=True)
            if r:
                trail.append({"field": field, "variant": idx, "name": name,
                              "median_us": r["median_us"]})
        if best is None:
            print(f"  {case['tag']}: every {field} rejected")
            return None
        ints[field] = best["variant"]
        print(f"  {case['tag']:20s} {field:13s} best {best['variant']} "
              f"{best['median_us']:.1f} us  {lib.frt_nvfp4_variant_name(fields[field], best['variant']).decode()}",
              flush=True)
    return {"tag": case["tag"], "chosen": {f: ints[f] for f in fields},
            "default": {f: case["ints"][f] for f in fields}, "trail": trail}


# ---------------------------------------------------------------- the cases ---
# Every case is one operator at the shape pi0.5 runs it. The `_plain` cases drop
# the normalization and the residual so the boundary matches the TensorRT-native
# subgraph cut from the same model.
q_e = torch.empty(Se, NH_e * HD_e, dtype=torch.float16).normal_(0, 0.3)
kv_e = torch.empty(Se, HD_e, dtype=torch.float16).normal_(0, 0.3)
q_s = torch.empty(M_s, NH_s * HD_s, dtype=torch.float16).normal_(0, 0.3)

CASES = [
    # Encoder FFN: RMSNorm x AWQ -> NVFP4, interleaved GeGLU gate/up, down into
    # the residual.
    dict(tag="encoder_mlp", label=f"encoder_mlp M={Se}", op="FlashrtNvfp4Mlp",
         ints=dict(D=D_e, H=H_e, norm_mode=1, gate_mode=0, gate_variant=0,
                   down_variant=down_variant, epilogue=1, opt_mask=(1 << 2) | (1 << 5)),
         floats=dict(eps=1e-6),
         inputs=[("x", E["x_in"], "in"),
                 ("gu_packed", E["L0.gu_il_packed"], "blob"),
                 ("gu_sfb", E["L0.gu_il_sfb"], "blob"),
                 ("down_packed", E["L0.down_packed"], "blob"),
                 ("down_sfb", E["L0.down_sfb"], "blob"),
                 ("awq", E["L0.awq_inv_s_gu"], "half"),
                 ("residual", E["x_in"], "in")],
         out_cols=D_e, rows=Se, note=f"D={D_e} H={H_e}"),
    # Encoder attention output projection: quantize -> NVFP4 GEMM -> += residual.
    dict(tag="encoder_o", label=f"encoder_o M={Se}", op="FlashrtNvfp4Linear",
         ints=dict(N=D_e, K=D_e, norm_mode=0, epilogue=1, variant=o_variant,
                   opt_mask=1 << 4),
         floats=dict(eps=1e-6),
         inputs=[("x", E["x_in"], "in"),
                 ("w_packed", E["L0.o_packed"], "blob"),
                 ("w_sfb", E["L0.o_sfb"], "blob"),
                 ("residual", E["x_in"], "in")],
         out_cols=D_e, rows=Se, note=f"N={D_e} K={D_e}"),
    dict(tag="encoder_mlp_plain", label=f"encoder_mlp_plain M={Se}", op="FlashrtNvfp4Mlp",
         ints=dict(D=D_e, H=H_e, norm_mode=0, gate_mode=0, gate_variant=0,
                   down_variant=down_variant, epilogue=0, opt_mask=0),
         floats=dict(eps=1e-6),
         inputs=[("x", E["x_in"], "in"),
                 ("gu_packed", E["L0.gu_il_packed"], "blob"),
                 ("gu_sfb", E["L0.gu_il_sfb"], "blob"),
                 ("down_packed", E["L0.down_packed"], "blob"),
                 ("down_sfb", E["L0.down_sfb"], "blob")],
         out_cols=D_e, rows=Se, note=f"D={D_e} H={H_e}, no norm/residual"),
    dict(tag="encoder_o_plain", label=f"encoder_o_plain M={Se}", op="FlashrtNvfp4Linear",
         ints=dict(N=D_e, K=D_e, norm_mode=0, epilogue=0, variant=o_variant, opt_mask=0),
         floats=dict(eps=1e-6),
         inputs=[("x", E["x_in"], "in"),
                 ("w_packed", E["L0.o_packed"], "blob"),
                 ("w_sfb", E["L0.o_sfb"], "blob")],
         out_cols=D_e, rows=Se, note=f"N={D_e} K={D_e}, no residual"),
    # SigLIP FFN: LayerNorm x AWQ -> NVFP4, up with bias and GELU, down with
    # bias and residual.
    dict(tag="siglip_mlp", label=f"siglip_mlp M={M_s}", op="FlashrtNvfp4Mlp",
         ints=dict(D=D_s, H=H_s_pad, norm_mode=2, gate_mode=1, gate_variant=up_variant,
                   down_variant=0, epilogue=2, opt_mask=0b111111),
         # the vision tower's LayerNorm epsilon, not this file's default
         floats=dict(eps=1e-5),
         inputs=[("x", S["L0.x_out"], "in"),
                 ("up_packed", S["L0.up_packed"], "blob"),
                 ("up_sfb", S["L0.up_sfb"], "blob"),
                 ("down_packed", S["L0.down_packed"], "blob"),
                 ("down_sfb", S["L0.down_sfb"], "blob"),
                 ("gamma", S["L0.ln_ffn_w"], "half"), ("beta", S["L0.ln_ffn_b"], "half"),
                 ("awq", S["L0.awq_inv_s"], "half"), ("up_b", S["L0.up_b"], "half"),
                 ("down_b", S["L0.down_b"], "half"), ("residual", S["L0.x_out"], "in")],
         out_cols=D_s, rows=M_s, note=f"D={D_s} H={H_s_pad}"),
    dict(tag="siglip_mlp_plain", label=f"siglip_mlp_plain M={M_s}", op="FlashrtNvfp4Mlp",
         ints=dict(D=D_s, H=H_s_pad, norm_mode=0, gate_mode=1, gate_variant=up_variant,
                   down_variant=0, epilogue=0, opt_mask=1 << 3),
         floats=dict(eps=1e-5),
         inputs=[("x", S["L0.x_out"], "in"),
                 ("up_packed", S["L0.up_packed"], "blob"),
                 ("up_sfb", S["L0.up_sfb"], "blob"),
                 ("down_packed", S["L0.down_packed"], "blob"),
                 ("down_sfb", S["L0.down_sfb"], "blob"),
                 ("up_b", S["L0.up_b"], "half")],
         out_cols=D_s, rows=M_s, note=f"D={D_s} H={H_s_pad}, no norm/residual"),
    # Encoder attention: FlashAttention-4, head_dim 256, grouped queries over
    # one KV head.
    dict(tag="encoder_attn", label=f"encoder_attn Sq={Se}", op="FlashrtFa4Attention",
         ints=dict(mode=0, NH=NH_e, head_dim=HD_e, batch=1), floats=dict(scale=0.0),
         inputs=[("q", q_e, "in"), ("k", kv_e, "in"), ("v", kv_e, "in")],
         out_cols=NH_e * HD_e, rows=Se, note=f"NH={NH_e} HD={HD_e} 1 KV head"),
    # SigLIP attention: FlashAttention-4, head_dim 72, one batch entry per camera.
    dict(tag="siglip_attn", label=f"siglip_attn S={spv}x{views}", op="FlashrtFa4Attention",
         ints=dict(mode=1, NH=NH_s, head_dim=HD_s, batch=views), floats=dict(scale=0.0),
         inputs=[("q", q_s, "in"), ("k", q_s, "in"), ("v", q_s, "in")],
         out_cols=NH_s * HD_s, rows=M_s, note=f"NH={NH_s} HD={HD_s}"),
]

by_tag = {c["tag"]: c for c in CASES}
for override in args.set:
    key, value = override.split("=")
    tag, field = key.split(".")
    by_tag[tag]["ints"][field] = int(value)

selected = [by_tag[t] for t in args.only.split(",")] if args.only else CASES
results, sweeps = [], []
for case in selected:
    if case["tag"] in args.sweep.split(","):
        s = sweep(case, args.iters)
        if s:
            sweeps.append(s)
            case["ints"].update(s["chosen"])
    r = measure(case, case["ints"], case["tag"], args.iters)
    if r is None:
        print(f"{case['label']:22s} {case['op']:20s}  rejected")
        continue
    tiles = " ".join(f"{k}={v}" for k, v in r["tiles"].items())
    print(f"{r['case']:22s} {r['op']:20s} {r['median_us']:8.1f} us  "
          f"p90 {r['p90_us']:8.1f}  {tiles}  {r['note']}", flush=True)
    results.append(r)

if args.json:
    with open(args.json, "w") as f:
        json.dump({"results": results, "sweeps": sweeps}, f, indent=2)
    print("wrote", args.json)
print("BENCH_OPS_DONE")
