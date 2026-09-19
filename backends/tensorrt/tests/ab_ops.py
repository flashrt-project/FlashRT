"""Alternate the FlashRT operator engines with TensorRT's own subgraphs.

Both arms are timed by trtexec with the same flags, so the two numbers are the
same measurement. A process on Thor lands in a fast or a slow clock mode and
stays there, so the arms alternate over several rounds and the fastest round of
each is what the comparison reports.

The FlashRT engines come from bench_ops.py --save-engines, the TensorRT
subgraphs from extract_trt_ops.py, both into the same directory.

usage:
  ab_ops.py --ops-dir DIR --plugin libflashrt_trt_pi05.so [--rounds N]
            [--rows N] [--views N] [--tokens N] [--only TAG[,TAG...]]
            [--trtexec PATH] [--json FILE]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_native_ops import CASES, TRTEXEC_FLAGS, find_trtexec, run_trtexec, shapes_for  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ops-dir", required=True)
ap.add_argument("--plugin", required=True)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--rows", type=int, default=526)
ap.add_argument("--views", type=int, default=2)
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--only", default="")
ap.add_argument("--native", choices=("export", "attention"), default="export",
                help="the TensorRT arm: the subgraph cut from the export, or the "
                     "graph attention_arm.py builds from the ONNX Attention operator")
ap.add_argument("--trtexec", default=None)
ap.add_argument("--json", default=None)
args = ap.parse_args()

trtexec = find_trtexec(args.trtexec)
selected = args.only.split(",") if args.only else list(CASES)
for tag in selected:
    if tag not in CASES and args.native != "attention":
        raise SystemExit(f"{tag} has no boundary in the export; use --native attention")
results = {}
for r in range(args.rounds):
    for tag in selected:
        # Cases the export has a boundary for name their engine there; the
        # synthetic attention arms are named after the operator itself.
        plugin_tag = CASES[tag]["plugin"] if tag in CASES else tag
        plan = os.path.join(args.ops_dir, f"{plugin_tag}.plan")
        with open(os.path.join(args.ops_dir, f"{plugin_tag}.shapes")) as f:
            plan_shapes = f.read().strip()
        if args.native == "attention":
            native = os.path.join(args.ops_dir, f"op_{tag}_native_attention.onnx")
            with open(os.path.splitext(native)[0] + ".shapes") as f:
                native_shapes = f.read().strip()
        else:
            native = os.path.join(args.ops_dir, f"op_{tag}.onnx")
            native_shapes = shapes_for(tag, args.rows, args.views, args.tokens)
        ours = run_trtexec([trtexec, f"--loadEngine={plan}", f"--dynamicPlugins={args.plugin}",
                            f"--shapes={plan_shapes}"] + TRTEXEC_FLAGS,
                           os.path.join(args.ops_dir, f"ab_flashrt_{plugin_tag}_r{r}.log"))
        theirs = run_trtexec([trtexec, f"--onnx={native}", "--stronglyTyped",
                              f"--shapes={native_shapes}"] + TRTEXEC_FLAGS,
                             os.path.join(args.ops_dir,
                                          f"ab_{args.native}_{tag}_r{r}.log"))
        results.setdefault(tag, {"flashrt": [], "tensorrt": []})
        results[tag]["flashrt"].append(ours)
        results[tag]["tensorrt"].append(theirs)
        fmt = lambda v: "     -  " if v is None else f"{v:8.1f}"  # noqa: E731
        print(f"round {r} {tag:14s} flashrt {fmt(ours)} us | tensorrt {fmt(theirs)} us",
              flush=True)

print()
for tag, arms in results.items():
    ours = [v for v in arms["flashrt"] if v]
    theirs = [v for v in arms["tensorrt"] if v]
    if ours and theirs:
        print(f"{tag:14s} flashrt {min(ours):7.1f} us | tensorrt {min(theirs):7.1f} us | "
              f"{min(theirs) / min(ours):.2f}x")
if args.json:
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.json)
print("AB_OPS_DONE")
