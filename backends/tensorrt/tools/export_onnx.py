#!/usr/bin/env python3
"""Export the pi0.5 Thor FP4 policy as an ONNX graph of FlashRT stage ops.

The graph uses three custom ops in the "flashrt" domain, implemented by the
FlashRT TensorRT plugin library, joined by standard ONNX ops:

  images fp16 [views, 224, 224, 3] --Pi05Siglip--> image tokens [views*256, 2048]
  Concat(image tokens, prompt embeddings) --Pi05Encoder--> K, V [L*prefix, 256]
  Reshape(K, V) + noise fp16 [10, 32] --Pi05Decoder--> actions fp16 [10, 32]

By default the prompt embeddings and RoPE rows of the dumped prompt are baked
in. With --prompts (tools/reference/dump_prompts.py output) the prompt is an input:

  lang_tokens int32 [n] -> Gather(embedding) -> fp16(fp32 * sqrt(D))
  encoder rope = rope_table[0:Se], decoder rope = rope_table[Se:Se + 10]

where Se = image tokens + n must be even (repeat the last token if not).

Weights are initializers in an external data file: fp16 tensors as float16,
packed FP8/NVFP4 bytes as int32 blobs (TensorRT constants take no uint8).
The inputs are FlashRT reference dumps (tools/reference/dump_siglip.py,
tools/reference/dump_encoder.py, tools/reference/dump_decoder.py) of one calibrated model and
prompt. Needs numpy and onnx only.

Build with TensorRT:
  trtexec --onnx=pi05.onnx --dynamicPlugins=libflashrt_trt_pi05.so \
          --stronglyTyped --builderOptimizationLevel=0 \
          --memPoolSize=workspace:2048 --saveEngine=pi05.engine

Prompt-dynamic builds also need shapes, e.g.
          --minShapes=lang_tokens:2 --optShapes=lang_tokens:14 --maxShapes=lang_tokens:256

usage: export_onnx.py <siglip_all> <encoder_all> <decoder_steps> <out_dir> [--prompts prompts]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DTYPES = {"F16": np.float16, "F32": np.float32, "I64": np.int64, "I32": np.int32, "U8": np.uint8,
          "F8_E4M3": np.uint8}


class Safetensors:
    """Lazy raw reader; float8 tensors come back as their bytes."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def __getitem__(self, key):
        h = self.header[key]
        a, b = h["data_offsets"]
        self.f.seek(self.base + a)
        arr = np.frombuffer(self.f.read(b - a), dtype=DTYPES[h["dtype"]])
        return arr if h["dtype"] == "F8_E4M3" else arr.reshape(h["shape"])


def blob(a):
    b = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    pad = (-b.size) % 4
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return b.view(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("siglip_all")
    ap.add_argument("encoder_all")
    ap.add_argument("decoder_steps")
    ap.add_argument("out_dir")
    ap.add_argument("--prompts", help="prompt-dynamic graph from tools/reference/dump_prompts.py tables")
    args = ap.parse_args()
    sig_path, enc_path, dec_path, out_dir = args.siglip_all, args.encoder_all, args.decoder_steps, args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    Ts, Te, Td = Safetensors(sig_path), Safetensors(enc_path), Safetensors(dec_path)
    S_sig, D_sig, _, NH_sig, HD_sig, L_sig, NV, SPV, H_PAD, DE, UP_VARIANT, DOWN_VARIANT = Ts["meta"].tolist()
    Se, D, H, NH, HD, _, o_variant, down_variant, L = Te["meta"].tolist()
    dm = Td["meta"].tolist()
    S, D_dec, H_dec, NH_dec, HD_dec, L_dec, steps, enc_seq = dm[:8]
    v_qkv, v_o, v_gu, v_down = dm[9:13]
    assert DOWN_VARIANT == 0 and enc_seq == Se and L_dec == L and DE == D

    inits = []

    def add(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def f16(T, key, name):
        return add(name, T[key].astype(np.float16))

    # SigLIP
    sig_layer = ("ln_attn_w", "ln_attn_b", "qkv_w", "qkv_b", "o_w", "o_b", "ln_ffn_w", "ln_ffn_b",
                 "awq_inv_s", "up_packed", "up_sfb", "up_b", "down_packed", "down_sfb", "down_b")
    sig_blobs = {"qkv_w", "o_w", "up_packed", "up_sfb", "down_packed", "down_sfb"}
    sig_in = ["images"] + [f16(Ts, n, "siglip." + n) for n in ("pe_w", "pe_b", "pos_emb", "postln_w",
                                                                "postln_b", "proj_w", "proj_b")]
    for l in range(L_sig):
        p = f"L{l}."
        sig_in += [add("siglip." + p + n, blob(Ts[p + n])) if n in sig_blobs
                   else f16(Ts, p + n, "siglip." + p + n) for n in sig_layer]
    alpha = Ts["alpha"].tolist()
    sig_alpha = [a for l in range(L_sig) for a in (alpha[4 * l], alpha[4 * l + 1])]
    nodes = [helper.make_node(
        "Pi05Siglip", sig_in, ["image_tokens"], name="flashrt_siglip", domain="flashrt",
        D=D_sig, H_pad=H_PAD, NH=NH_sig, HD=HD_sig, spv=SPV, up_variant=UP_VARIANT, De=DE, L=L_sig,
        alpha=sig_alpha)]

    # prompt embeddings follow the image tokens
    graph_inputs = [helper.make_tensor_value_info("images", TensorProto.FLOAT16, [NV, 224, 224, 3])]
    if args.prompts:
        Tp = Safetensors(args.prompts)
        graph_inputs.append(helper.make_tensor_value_info("lang_tokens", TensorProto.INT32, ["n_tokens"]))
        add("embedding", Tp["embedding"].astype(np.float16))
        add("rope_table", Tp["rope_table"].astype(np.float16))
        add("embed_scale", np.array(D ** 0.5, dtype=np.float32))
        add("one_1d", np.array([1], dtype=np.int64))
        add("zero_1d", np.array([0], dtype=np.int64))
        add("axes_0", np.array([0], dtype=np.int64))
        add("action_rows", np.array([S], dtype=np.int64))
        nodes += [
            helper.make_node("Gather", ["embedding", "lang_tokens"], ["token_embeddings"], name="embed", axis=0),
            helper.make_node("Cast", ["token_embeddings"], ["token_embeddings_f32"], name="embed_to_f32",
                             to=TensorProto.FLOAT),
            helper.make_node("Mul", ["token_embeddings_f32", "embed_scale"], ["prompt_embeddings_f32"],
                             name="embed_scale_mul"),
            helper.make_node("Cast", ["prompt_embeddings_f32"], ["prompt_embeddings"], name="embed_to_f16",
                             to=TensorProto.FLOAT16),
        ]
    else:
        add("prompt_embeddings", Te["x_in"][S_sig:].astype(np.float16))
    nodes.append(helper.make_node("Concat", ["image_tokens", "prompt_embeddings"], ["prefix_embeddings"],
                                  name="prefix_concat", axis=0))
    if args.prompts:
        nodes += [
            helper.make_node("Shape", ["prefix_embeddings"], ["prefix_shape"], name="prefix_shape"),
            helper.make_node("Slice", ["prefix_shape", "zero_1d", "one_1d"], ["prefix_len_1d"], name="prefix_len"),
            helper.make_node("Add", ["prefix_len_1d", "action_rows"], ["prefix_end"], name="prefix_end"),
            helper.make_node("Slice", ["rope_table", "zero_1d", "prefix_len_1d", "axes_0"], ["enc_rope"],
                             name="encoder_rope"),
            helper.make_node("Slice", ["rope_table", "prefix_len_1d", "prefix_end", "axes_0"], ["dec_rope"],
                             name="decoder_rope"),
        ]
        enc_rope, dec_rope = "enc_rope", "dec_rope"
    else:
        enc_rope, dec_rope = f16(Te, "rope", "enc.rope"), f16(Td, "rope", "dec.rope")

    # encoder
    heads, tails = [], []
    tail = ("o_packed", "o_sfb", "awq_inv_s_gu", "gu_il_packed", "gu_il_sfb", "down_packed", "down_sfb")
    for l in range(L):
        p = f"L{l}."
        heads += [add("enc." + p + "qkv_w", blob(Te[p + "qkv_w"])),
                  add("enc." + p + "qkv_scale", Te[p + "act_scale_qkv"].astype(np.float32))]
        if l < L - 1:
            tails += [f16(Te, p + n, "enc." + p + n) if n == "awq_inv_s_gu" else add("enc." + p + n, blob(Te[p + n]))
                      for n in tail]
    enc_alpha = [float(Te[f"L{l}.alpha_qkv"][0]) for l in range(L)]
    nodes.append(helper.make_node(
        "Pi05Encoder", ["prefix_embeddings", enc_rope] + heads + tails,
        ["encoder_x", "prefix_k", "prefix_v"], name="flashrt_encoder", domain="flashrt",
        D=D, H=H, NH=NH, HD=HD, L=L, attn_o_variant=o_variant, down_variant=down_variant,
        qkv_alpha=enc_alpha))
    add("flat_shape", np.array([-1], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", ["prefix_k", "flat_shape"], ["prefix_k_flat"], name="flatten_k"))
    nodes.append(helper.make_node("Reshape", ["prefix_v", "flat_shape"], ["prefix_v_flat"], name="flatten_v"))

    # decoder
    dec_in = ["noise", "prefix_k_flat", "prefix_v_flat"]
    dec_in += [add("dec." + n, Td[n].astype(np.float16)) for n in ("ain_w", "ain_b", "aow", "aob")]
    dec_in.append(dec_rope)
    dec_in += [add("dec." + n, Td[n].astype(np.float16).reshape(-1)) for n in ("sa", "sf", "fs")]
    dec_in += [add("dec." + n, blob(Td[n])) for n in ("qw_fp4", "qw_sfb", "ow_fp4", "ow_sfb", "gwil_fp4",
                                                       "gwil_sfb", "dw_fp4", "dw_sfb")]
    nodes.append(helper.make_node(
        "Pi05Decoder", dec_in, ["actions"], name="flashrt_decoder", domain="flashrt",
        S=S, D=D_dec, H=H_dec, NH=NH_dec, HD=HD_dec, L=L, steps=steps, v_qkv=v_qkv, v_o=v_o, v_gu=v_gu,
        v_down=v_down, dt=[float(Td["dt"][0])]))

    graph = helper.make_graph(
        nodes, "pi05_flashrt_thor_fp4",
        graph_inputs + [helper.make_tensor_value_info("noise", TensorProto.FLOAT16, [S, 32])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT16, [S, 32])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("flashrt", 1)],
                              producer_name="flashrt-tensorrt")
    path = os.path.join(out_dir, "pi05.onnx")
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="pi05.onnx.data", size_threshold=0)
    print("wrote", path, "initializers", len(inits))


if __name__ == "__main__":
    main()
