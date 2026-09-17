"""M4 step 1: the whole pi0.5 policy as one TensorRT engine on FlashRT stage
plugins, images + noise -> raw actions:

  images fp16 [3, 224, 224, 3] -> Pi05Siglip -> image tokens [768, 2048]
  concat(image tokens, prompt embeddings) -> Pi05Encoder -> K/V [L*976, 256]
  flatten K/V, noise [10, 32] -> Pi05Decoder (10 steps) -> actions [10, 32]

The three reference dumps come from the same FlashRT inference and chain bit
for bit (SigLIP tokens = encoder input rows, encoder K/V = decoder prefix), so
the engine output is checked bitwise against the library's final actions,
eagerly and under an outer CUDA graph, with latency for both.

usage: engine_policy.py <plugin.so> <siglip_all> <encoder_all> <decoder_steps>
"""
import os
import sys
import time

sys.path.append("/usr/lib/python3.12/dist-packages")

import numpy as np  # noqa: E402
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

plugin_path, sig_path, enc_path, dec_path = sys.argv[1:5]
logger = trt.Logger(trt.Logger.WARNING)
registry = trt.get_plugin_registry()
registry.load_library(plugin_path)
Ts, Te, Td = load_file(sig_path), load_file(enc_path), load_file(dec_path)

S_sig, D_sig, _, NH_sig, HD_sig, L_sig, NV, SPV, H_PAD, DE, UP_VARIANT, _ = Ts["meta"].tolist()
Se, D, H, NH, HD, total_keys, o_variant, down_variant, L = Te["meta"].tolist()
S, D_dec, H_dec, NH_dec, HD_dec, L_dec, steps, enc_seq = Td["meta"].tolist()[:8]
v_qkv, v_o, v_gu, v_down = Td["meta"].tolist()[9:13]
assert enc_seq == Se and L_dec == L and DE == D
n_lang = Se - S_sig
print(f"views={NV} image tokens={S_sig} prompt tokens={n_lang} Se={Se} steps={steps}")

keep = []


def blob(t):
    b = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


def fields(ints, floats):
    fc = trt.PluginFieldCollection()
    for name, value in ints:
        keep.append(np.array([value], dtype=np.int32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.INT32))
    for name, values in floats:
        keep.append(np.array(values, dtype=np.float32))
        fc.append(trt.PluginField(name, keep[-1], trt.PluginFieldType.FLOAT32))
    return fc


builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))


def const(name, arr):
    arr = np.ascontiguousarray(arr)
    keep.append(arr)
    layer = net.add_constant(arr.shape, trt.Weights(arr))
    layer.name = name
    return layer.get_output(0)


def f16(T, key, name=None):
    return const(name or key, T[key].numpy().astype(np.float16))


images = net.add_input("images", trt.float16, (NV, 224, 224, 3))
noise = net.add_input("noise", trt.float16, (S, 32))

# SigLIP stage
SIG_LAYER = ("ln_attn_w", "ln_attn_b", "qkv_w", "qkv_b", "o_w", "o_b", "ln_ffn_w", "ln_ffn_b",
             "awq_inv_s", "up_packed", "up_sfb", "up_b", "down_packed", "down_sfb", "down_b")
SIG_BLOBS = {"qkv_w", "o_w", "up_packed", "up_sfb", "down_packed", "down_sfb"}
alpha = Ts["alpha"].tolist()
sig_in = [images] + [f16(Ts, n, "siglip." + n) for n in ("pe_w", "pe_b", "pos_emb", "postln_w", "postln_b",
                                                          "proj_w", "proj_b")]
for l in range(L_sig):
    p = f"L{l}."
    sig_in += [const("siglip." + p + n, blob(Ts[p + n])) if n in SIG_BLOBS else f16(Ts, p + n, "siglip." + p + n)
               for n in SIG_LAYER]
sig_alpha = []
for l in range(L_sig):
    sig_alpha += [alpha[4 * l], alpha[4 * l + 1]]
plugin = registry.get_creator("Pi05Siglip", "1", "").create_plugin("Pi05Siglip", fields(
    [("D", D_sig), ("H_pad", H_PAD), ("NH", NH_sig), ("HD", HD_sig), ("spv", SPV), ("up_variant", UP_VARIANT),
     ("De", DE), ("L", L_sig)], [("alpha", sig_alpha)]), trt.TensorRTPhase.BUILD)
node = net.add_plugin_v3(sig_in, [], plugin)
node.name = "flashrt_siglip"
image_tokens = node.get_output(0)

# prompt embeddings after the image tokens
lang = const(
    "prompt_embeddings", Te["x_in"][S_sig:].numpy().astype(np.float16))
cat = net.add_concatenation([image_tokens, lang])
cat.axis = 0
cat.name = "prefix_embeddings"
x = cat.get_output(0)

# encoder stage
heads, tails = [], []
TAIL = ("o_packed", "o_sfb", "awq_inv_s_gu", "gu_il_packed", "gu_il_sfb", "down_packed", "down_sfb")
for l in range(L):
    p = f"L{l}."
    heads += [const("enc." + p + "qkv_w", blob(Te[p + "qkv_w"])),
              const("enc." + p + "qkv_scale", Te[p + "act_scale_qkv"].numpy().astype(np.float32))]
    if l < L - 1:
        tails += [const("enc." + p + n, Te[p + n].numpy().astype(np.float16) if n == "awq_inv_s_gu"
                        else blob(Te[p + n])) for n in TAIL]
enc_alpha = [float(Te[f"L{l}.alpha_qkv"][0]) for l in range(L)]
plugin = registry.get_creator("Pi05Encoder", "1", "").create_plugin("Pi05Encoder", fields(
    [("D", D), ("H", H), ("NH", NH), ("HD", HD), ("L", L), ("attn_o_variant", o_variant),
     ("down_variant", down_variant)], [("qkv_alpha", enc_alpha)]), trt.TensorRTPhase.BUILD)
node = net.add_plugin_v3([x, f16(Te, "rope", "enc.rope")] + heads + tails, [], plugin)
node.name = "flashrt_encoder"
enc_k, enc_v = node.get_output(1), node.get_output(2)


def flatten(t, name):
    sh = net.add_shuffle(t)
    sh.reshape_dims = (-1,)
    sh.name = name
    return sh.get_output(0)


# decoder stage
dec_in = [noise, flatten(enc_k, "prefix_k"), flatten(enc_v, "prefix_v")]
dec_in += [const("dec." + n, Td[n].numpy().astype(np.float16).reshape(-1) if n in ("sa", "sf", "fs")
                 else Td[n].numpy().astype(np.float16)) for n in ("ain_w", "ain_b", "aow", "aob", "rope",
                                                                  "sa", "sf", "fs")]
dec_in += [const("dec." + n, blob(Td[n])) for n in ("qw_fp4", "qw_sfb", "ow_fp4", "ow_sfb", "gwil_fp4",
                                                     "gwil_sfb", "dw_fp4", "dw_sfb")]
plugin = registry.get_creator("Pi05Decoder", "1", "").create_plugin("Pi05Decoder", fields(
    [("S", S), ("D", D_dec), ("H", H_dec), ("NH", NH_dec), ("HD", HD_dec), ("L", L), ("steps", steps),
     ("v_qkv", v_qkv), ("v_o", v_o), ("v_gu", v_gu), ("v_down", v_down)], [("dt", [float(Td["dt"][0])])]),
    trt.TensorRTPhase.BUILD)
node = net.add_plugin_v3(dec_in, [], plugin)
node.name = "flashrt_decoder"
actions = node.get_output(0)
actions.name = "actions"
net.mark_output(actions)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.builder_optimization_level = int(os.environ.get("BUILDER_OPT_LEVEL", "0"))
t0 = time.time()
ser = builder.build_serialized_network(net, config)
assert ser is not None, "engine build failed"
print(f"engine built in {time.time() - t0:.1f}s, {ser.nbytes / 1e6:.0f} MB")
engine_path = os.environ.get("ENGINE_OUT")
if engine_path:
    with open(engine_path, "wb") as f:
        f.write(ser)
engine = trt.Runtime(logger).deserialize_cuda_engine(ser)
del ser
ctx = engine.create_execution_context()

dev = torch.device("cuda")
lut = (torch.arange(256, dtype=torch.float32) / 127.5 - 1.0).to(torch.float16)
in_images = lut[Ts["images_u8"].long()].to(dev).contiguous()
in_noise = Td["noise_in"].to(dev).contiguous()
ref = Td["noise_out"].to(dev)
out = torch.empty(S, 32, dtype=torch.float16, device=dev)
for n, t in (("images", in_images), ("noise", in_noise), ("actions", out)):
    ctx.set_tensor_address(n, t.data_ptr())
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
graph_ms = timed(g.replay, n=200)
print(f"[full] actions bitwise eager={eager_ok} (max|d| {d:.3g}) graph={graph_ok} | eager median "
      f"{eager_ms[0]:.2f} ms p90 {eager_ms[1]:.2f} | graph median {graph_ms[0]:.2f} ms p90 {graph_ms[1]:.2f}")
print("M4_FULL_" + ("PASS" if eager_ok and graph_ok else "FAIL"))
