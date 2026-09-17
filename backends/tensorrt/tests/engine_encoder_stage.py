"""the full pi0.5 encoder as one Pi05Encoder stage node (the chain of
Pi05EncoderLayer nodes is tested by engine_encoder_layer_chain.py). Each is checked bitwise against the FlashRT library encoder
output, eager and under an outer CUDA graph, with latency for both.

usage: engine_encoder_stage.py <plugin.so> <encoder_all.safetensors>
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
Se, D, H, NH, HD, total_keys, o_variant, down_variant, L = T["meta"].tolist()
alphas = [float(T[f"L{l}.alpha_qkv"][0]) for l in range(L)]
TAIL = ("o_packed", "o_sfb", "awq_inv_s_gu", "gu_il_packed", "gu_il_sfb", "down_packed", "down_sfb")
print(f"Se={Se} L={L}")


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


keep = []  # PluginField arrays must outlive plugin creation


def make_fields(ints, floats):
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
    x = net.add_input("x", trt.float16, (-1, D))
    rope = net.add_input("rope", trt.float16, (-1, HD))

    def const(name, arr):
        layer = net.add_constant(arr.shape, trt.Weights(np.ascontiguousarray(arr)))
        layer.name = name
        return layer.get_output(0)

    def layer_consts(l):
        p = f"L{l}."
        head = [const(p + "qkv_w", blob(T[p + "qkv_w"])),
                const(p + "qkv_scale", T[p + "act_scale_qkv"].numpy().astype(np.float32))]
        if l == L - 1:
            return head, []
        tail = []
        for n in TAIL:
            arr = T[p + n].numpy().astype(np.float16) if n == "awq_inv_s_gu" else blob(T[p + n])
            tail.append(const(p + n, arr))
        return head, tail

    if kind == "layers":
        cur = x
        ks, vs = [], []
        creator = registry.get_creator("Pi05EncoderLayer", "1", "")
        for l in range(L):
            head, tail = layer_consts(l)
            last = int(l == L - 1)
            if last:  # the plugin input list is fixed; the last layer ignores the FFN inputs
                tail = [const(f"L{l}.dummy{i}", np.zeros(1, dtype=np.float16 if i == 2 else np.int32))
                        for i in range(len(TAIL))]
            fc = make_fields([("D", D), ("H", H), ("NH", NH), ("HD", HD), ("attn_o_variant", o_variant),
                              ("down_variant", down_variant), ("last", last)],
                             [("qkv_alpha", [alphas[l]])])
            plugin = creator.create_plugin("Pi05EncoderLayer", fc, trt.TensorRTPhase.BUILD)
            node = net.add_plugin_v3([cur, rope] + head + tail, [], plugin)
            node.name = f"flashrt_encoder_layer_{l}"
            cur = node.get_output(0)
            ks.append(node.get_output(1))
            vs.append(node.get_output(2))
        # stack [Se, HD] x L -> [L, Se, HD]
        def stack(ts):
            ups = []
            for t in ts:
                sh = net.add_shuffle(t)
                sh.reshape_dims = (1, -1, HD)
                ups.append(sh.get_output(0))
            cat = net.add_concatenation(ups)
            cat.axis = 0
            return cat.get_output(0)
        outs = (cur, stack(ks), stack(vs))
    else:
        heads, tails = [], []
        for l in range(L):
            h, t = layer_consts(l)
            heads += h
            tails += t
        fc = make_fields([("D", D), ("H", H), ("NH", NH), ("HD", HD), ("L", L),
                          ("attn_o_variant", o_variant), ("down_variant", down_variant)],
                         [("qkv_alpha", alphas)])
        plugin = registry.get_creator("Pi05Encoder", "1", "").create_plugin("Pi05Encoder", fc, trt.TensorRTPhase.BUILD)
        node = net.add_plugin_v3([x, rope] + heads + tails, [], plugin)
        node.name = "flashrt_encoder_stage"
        outs = (node.get_output(0), node.get_output(1), node.get_output(2))
    for t, name in zip(outs, ("x_out", "k", "v")):
        t.name = name
        net.mark_output(t)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # Levels above 0 re-time the plugin per stream for minutes with no benefit.
    config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (16, D), (Se, D), (1024, D))
    prof.set_shape("rope", (16, HD), (Se, HD), (1024, HD))
    config.add_optimization_profile(prof)
    t0 = time.time()
    ser = builder.build_serialized_network(net, config)
    assert ser is not None, f"{kind} build failed"
    print(f"[{kind}] engine built in {time.time() - t0:.1f}s, {ser.nbytes / 1e6:.0f} MB")
    return trt.Runtime(logger).deserialize_cuda_engine(ser)


dev = torch.device("cuda")
x_in = T["x_in"].to(dev)
rope_in = T["rope"].to(dev)
ref_x = T["x_out"].to(dev)
ref_k = torch.cat([T[f"L{l}.k_out"] for l in range(L)]).view(L * Se, HD).to(dev)
ref_v = torch.cat([T[f"L{l}.v_out"] for l in range(L)]).view(L * Se, HD).to(dev)
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
for kind in ("stage",):  # the layer chain is covered by engine_encoder_layer_chain.py
    engine = build(kind)
    ctx = engine.create_execution_context()
    ix, ir = x_in.clone(), rope_in.clone()
    ox = torch.empty(Se, D, dtype=torch.float16, device=dev)
    ok_ = torch.empty(L * Se, HD, dtype=torch.float16, device=dev)
    ov = torch.empty(L * Se, HD, dtype=torch.float16, device=dev)
    ctx.set_input_shape("x", (Se, D))
    ctx.set_input_shape("rope", (Se, HD))
    for name, t in (("x", ix), ("rope", ir), ("x_out", ox), ("k", ok_), ("v", ov)):
        ctx.set_tensor_address(name, t.data_ptr())
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    eager_ok = torch.equal(ox, ref_x) and torch.equal(ok_, ref_k) and torch.equal(ov, ref_v)
    for nm, got, ref in (("x", ox, ref_x), ("k", ok_, ref_k), ("v", ov, ref_v)):
        dif = (got.float() - ref.float()).abs()
        rows = (dif.reshape(got.shape[0], -1).amax(dim=1) > 0).nonzero().flatten().tolist()
        print(f"[{kind}] {nm}: bitwise={torch.equal(got, ref)} max|d|={dif.max().item():.3g} "
              f"first differing rows={rows[:6]} n={len(rows)}")
    eager_ms = timed(lambda: ctx.execute_async_v3(stream.cuda_stream))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    ox.zero_(); ok_.zero_(); ov.zero_()
    g.replay()
    torch.cuda.synchronize()
    graph_ok = torch.equal(ox, ref_x) and torch.equal(ok_, ref_k) and torch.equal(ov, ref_v)
    graph_ms = timed(g.replay)
    print(f"[{kind}] bitwise eager={eager_ok} graph={graph_ok} | eager median {eager_ms[0]:.3f} ms "
          f"p90 {eager_ms[1]:.3f} | graph median {graph_ms[0]:.3f} ms p90 {graph_ms[1]:.3f}")
    ok = ok and eager_ok and graph_ok
    del ctx, engine, g
    torch.cuda.empty_cache()
print("ENGINE_ENCODER_" + ("PASS" if ok else "FAIL"))
