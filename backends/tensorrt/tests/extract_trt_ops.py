"""Cut single-operator subgraphs out of an ONNX export and time them with trtexec.

TensorRT's own quantized path is the reference the FlashRT operator plugins are
measured against, so the subgraphs are cut from the exported model unchanged:
the quantization nodes, the weights and the layouts are the export's, only the
boundaries are ours. The boundaries are in trt_native_ops.py.

The subgraphs are written next to a copy of (or a link to) the export's
external weight file so their initializers keep resolving.

usage:
  extract_trt_ops.py --onnx MODEL.onnx --out DIR [--rows N] [--views N]
                     [--tokens N] [--only TAG[,TAG...]] [--trtexec PATH]
"""
import argparse
import json
import os
import subprocess
import sys

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_native_ops import CASES, TRTEXEC_FLAGS, dims, find_trtexec, run_trtexec, shapes_for  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rows", type=int, default=526, help="encoder sequence length")
ap.add_argument("--views", type=int, default=2, help="cameras")
ap.add_argument("--tokens", type=int, default=256, help="image tokens per camera")
ap.add_argument("--only", default="")
ap.add_argument("--trtexec", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()

trtexec = find_trtexec(args.trtexec)
os.makedirs(args.out, exist_ok=True)
external = os.path.splitext(args.onnx)[0] + ".data"
link = os.path.join(args.out, os.path.basename(external))
if os.path.exists(external) and not os.path.exists(link):
    # The export directory is usually read-only, so the subgraphs live here and
    # reach the weights through a link.
    os.symlink(external, link)

selected = args.only.split(",") if args.only else list(CASES)
model = onnx.load(args.onnx, load_external_data=False)
known = {v.name for v in model.graph.value_info} | {v.name for v in model.graph.input} | \
        {v.name for v in model.graph.output}
for tag in selected:
    case = CASES[tag]
    for name, template in zip(case["inputs"] + case["outputs"],
                              case["input_shapes"] + case["output_shapes"]):
        if name in known:
            continue
        # Shape inference over the whole model is too expensive, so a boundary
        # tensor gets its shape from the case's template instead.
        model.graph.value_info.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT16, dims(template, f"d{len(known)}")))
        known.add(name)

extractor = onnx.utils.Extractor(model)
results = []
for tag in selected:
    case = CASES[tag]
    sub = os.path.join(args.out, f"op_{tag}.onnx")
    if not os.path.exists(sub):
        print("== extracting", tag, flush=True)
        onnx.save(extractor.extract_model(case["inputs"], case["outputs"]), sub)
    shapes = shapes_for(tag, args.rows, args.views, args.tokens)
    log = os.path.join(args.out, f"native_{tag}.log")
    print("==", tag, flush=True)
    median = run_trtexec([trtexec, f"--onnx={sub}", "--stronglyTyped", f"--shapes={shapes}"]
                         + TRTEXEC_FLAGS, log)
    print(f"   median={'-' if median is None else round(median, 1)} us  ({log})")
    results.append({"case": tag, "median_us": median, "shapes": shapes})

if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.json)
print("EXTRACT_DONE")
