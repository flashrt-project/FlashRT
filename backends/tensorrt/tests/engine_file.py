"""M4 step 2: run a serialized pi0.5 engine (built by trtexec from
tools/export_onnx.py output) and check its actions bitwise against the
FlashRT library inference, eagerly and under an outer CUDA graph.

usage: engine_file.py <plugin.so> <pi05.engine> <siglip_all> <decoder_steps>
"""
import json
import struct
import sys
import time

sys.path.append("/usr/lib/python3.12/dist-packages")

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402

plugin_path, engine_path, sig_path, dec_path = sys.argv[1:5]


def read(path, key, dtype):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))[key]
        a, b = h["data_offsets"]
        f.seek(8 + n + a)
        return np.frombuffer(f.read(b - a), dtype=dtype).reshape(h["shape"])


logger = trt.Logger(trt.Logger.WARNING)
trt.get_plugin_registry().load_library(plugin_path)
with open(engine_path, "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
dev = torch.device("cuda")
lut = (np.arange(256, dtype=np.float32) / 127.5 - 1.0).astype(np.float16)
images = torch.from_numpy(lut[read(sig_path, "images_u8", np.uint8)]).to(dev).contiguous()
noise = torch.from_numpy(read(dec_path, "noise_in", np.float16).copy()).to(dev)
ref = torch.from_numpy(read(dec_path, "noise_out", np.float16).copy()).to(dev)
out = torch.empty_like(ref)
for n, t in (("images", images), ("noise", noise), ("actions", out)):
    ctx.set_tensor_address(n, t.data_ptr())
stream = torch.cuda.Stream()


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


with torch.cuda.stream(stream):
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
print(f"[engine file] actions bitwise eager={eager_ok} graph={graph_ok} | eager median {eager_ms[0]:.2f} ms "
      f"p90 {eager_ms[1]:.2f} | graph median {graph_ms[0]:.2f} ms p90 {graph_ms[1]:.2f}")
print("M4_ENGINE_FILE_" + ("PASS" if eager_ok and graph_ok else "FAIL"))
