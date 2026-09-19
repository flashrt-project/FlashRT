"""Time TensorRT's own Attention operator at the pi0.5 attention shapes.

The tutorial export writes attention as a MatMul/Softmax/MatMul chain, so a
comparison against it only says FlashAttention-4 beats that export. This builds
the same attention out of the ONNX `Attention` operator instead, which
TensorRT lowers to its own fused attention, and times it the same way.

usage:
  attention_arm.py --out DIR [--only TAG[,TAG...]] [--rows N] [--views N]
                   [--tokens N] [--trtexec PATH] [--json FILE]
"""
import argparse
import json
import os
import sys

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_native_ops import TRTEXEC_FLAGS, find_trtexec, run_trtexec  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
ap.add_argument("--rows", type=int, default=526, help="encoder sequence length")
ap.add_argument("--views", type=int, default=2, help="cameras")
ap.add_argument("--tokens", type=int, default=256, help="image tokens per camera")
ap.add_argument("--opset", type=int, default=24)
ap.add_argument("--trtexec", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()

# tag: (batch, query heads, key/value heads, query length, key length, head size)
CASES = {
    "siglip_attn": (args.views, 16, 16, args.tokens, args.tokens, 72),
    "encoder_attn": (1, 8, 1, args.rows, args.rows, 256),
}


def build(tag, batch, q_heads, kv_heads, q_len, kv_len, head_size):
    def tensor(name, heads, length):
        # The batch and the length stay symbolic, as they are in the subgraphs
        # cut from the export, so both arms are built the same way.
        return helper.make_tensor_value_info(
            name, TensorProto.FLOAT16, [f"{name}_batch", heads, f"{name}_len", head_size])

    node = helper.make_node("Attention", ["q", "k", "v"], ["y"], name=f"attention_{tag}",
                            is_causal=0)
    graph = helper.make_graph([node], f"trt_{tag}",
                              [tensor("q", q_heads, q_len), tensor("k", kv_heads, kv_len),
                               tensor("v", kv_heads, kv_len)],
                              [tensor("y", q_heads, q_len)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", args.opset)])
    model.ir_version = 10  # the parser rejects newer IR versions than it knows
    path = os.path.join(args.out, f"op_{tag}_native_attention.onnx")
    onnx.save(model, path)
    return path


def shapes_for(tag, batch, q_heads, kv_heads, q_len, kv_len, head_size):
    return (f"q:{batch}x{q_heads}x{q_len}x{head_size},"
            f"k:{batch}x{kv_heads}x{kv_len}x{head_size},"
            f"v:{batch}x{kv_heads}x{kv_len}x{head_size}")


os.makedirs(args.out, exist_ok=True)
trtexec = find_trtexec(args.trtexec)
selected = args.only.split(",") if args.only else list(CASES)
results = []
for tag in selected:
    batch, q_heads, kv_heads, q_len, kv_len, head_size = CASES[tag]
    path = build(tag, batch, q_heads, kv_heads, q_len, kv_len, head_size)
    shapes = shapes_for(tag, batch, q_heads, kv_heads, q_len, kv_len, head_size)
    # Written next to the model the way bench_ops.py writes them next to an
    # engine, so ab_ops.py can alternate the two arms without knowing either.
    with open(os.path.splitext(path)[0] + ".shapes", "w") as f:
        f.write(shapes + "\n")
    log = os.path.join(args.out, f"native_attention_{tag}.log")
    print("==", tag, shapes, flush=True)
    median = run_trtexec([trtexec, f"--onnx={path}", "--stronglyTyped", f"--shapes={shapes}"]
                         + TRTEXEC_FLAGS, log)
    print(f"   median={'-' if median is None else round(median, 1)} us  ({log})")
    results.append({"case": tag, "median_us": median, "shapes": shapes})

if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.json)
print("ATTENTION_ARM_DONE")
