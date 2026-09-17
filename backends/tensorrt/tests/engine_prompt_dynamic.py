"""prompt-dynamic pi0.5 engine (tools/export_onnx.py --prompts, built with
trtexec) against FlashRT: for every reference prompt, the engine's raw actions
from the same images and pinned noise must match the library bit for bit,
eagerly and under an outer CUDA graph captured for that prompt length.

usage: engine_prompt_dynamic.py <plugin.so> <pi05.engine> <siglip_all> <prompts> [decoder_steps]
"""
import json
import struct
import sys
import time

try:
    import tensorrt  # noqa: F401
except ImportError:  # JetPack installs the TensorRT bindings for the system Python
    sys.path.append(f"/usr/lib/python3.{sys.version_info.minor}/dist-packages")

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402

plugin_path, engine_path, sig_path, prompts_path = sys.argv[1:5]
DT = {"F16": np.float16, "U8": np.uint8, "I32": np.int32}


def reader(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))

    def read(key):
        h = header[key]
        a, b = h["data_offsets"]
        f.seek(8 + n + a)
        return np.frombuffer(f.read(b - a), dtype=DT[h["dtype"]]).reshape(h["shape"])
    return header, read


logger = trt.Logger(trt.Logger.WARNING)
trt.get_plugin_registry().load_library(plugin_path)
with open(engine_path, "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
dev = torch.device("cuda")
_, read_sig = reader(sig_path)
ph, read_p = reader(prompts_path)
lut = (np.arange(256, dtype=np.float32) / 127.5 - 1.0).astype(np.float16)
images = torch.from_numpy(lut[read_sig("images_u8")]).to(dev).contiguous()
noise = torch.from_numpy(read_p("noise_in").copy()).to(dev)
if len(sys.argv) > 5:
    _, read_d = reader(sys.argv[5])
    same = np.array_equal(read_d("noise_out"), read_p("p0.actions")) and np.array_equal(read_d("noise_in"),
                                                                                         read_p("noise_in"))
    print("prompt 0 reference == decoder dump actions:", same)
out = torch.empty(10, 32, dtype=torch.float16, device=dev)
stream = torch.cuda.Stream()
n_prompts = len([k for k in ph if k.endswith(".tokens")])


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
    return ms[len(ms) // 2]


ok = True
with torch.cuda.stream(stream):
    for i in range(n_prompts):
        tokens = torch.from_numpy(read_p(f"p{i}.tokens").copy()).to(dev)
        ref = torch.from_numpy(read_p(f"p{i}.actions").copy()).to(dev)
        ctx.set_input_shape("lang_tokens", tuple(tokens.shape))
        for n, t in (("images", images), ("lang_tokens", tokens), ("noise", noise), ("actions", out)):
            ctx.set_tensor_address(n, t.data_ptr())
        out.zero_()
        assert ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        eager_ok = torch.equal(out, ref)
        eager_ms = timed(lambda: ctx.execute_async_v3(stream.cuda_stream))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            ctx.execute_async_v3(stream.cuda_stream)
        out.zero_()
        g.replay()
        stream.synchronize()
        graph_ok = torch.equal(out, ref)
        graph_ms = timed(g.replay)
        if not eager_ok:
            refs = {j: torch.from_numpy(read_p(f"p{j}.actions").copy()).to(dev) for j in range(n_prompts)}
            ctx.set_input_shape("lang_tokens", tuple(tokens.shape))
            assert ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            dist = {j: round((out.float() - r.float()).abs().max().item(), 5) for j, r in refs.items()}
            print(f"  prompt {i} engine output max|d| vs each reference: {dist}")
        print(f"prompt {i}: {tokens.numel()} tokens | actions bitwise eager={eager_ok} graph={graph_ok} | "
              f"eager {eager_ms:.2f} ms, graph {graph_ms:.2f} ms")
        ok = ok and eager_ok and graph_ok
        del g
print("ENGINE_PROMPT_DYNAMIC_" + ("PASS" if ok else "FAIL"))
