#!/usr/bin/env python3
"""Check the TensorRT Edge-LLM pi0.5 example against FlashRT.

usage: check_pi05.py <pi05_policy_inference> <engine_dir> <tokenizer_dir> [--plugin <lib>]

<engine_dir> is the output directory of tools/build_pi05_engine.sh (pi05.engine,
siglip_all.safetensors, prompts.safetensors). The script writes the recorded
calibration observation as PNG files, runs pi05_policy_inference for the four
recorded LIBERO prompts with the recorded noise, and checks the prompt tokens
and the raw actions bit for bit against FlashRT's references.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from safetensors import safe_open

# Same prompts as tools/reference/dump_prompts.py.
LIBERO_PROMPTS = (
    "pick up the red block and place it in the tray",
    "put the bowl on the plate",
    "open the top drawer of the cabinet and put the bowl inside",
    "turn on the stove",
)


def cublas_dir():
    # FlashRT's references use the cuBLAS build PyTorch ships; load the same one.
    try:
        import nvidia
        d = os.path.join(list(nvidia.__path__)[0], "cu13", "lib")
        return d if os.path.isdir(d) else ""
    except ImportError:
        return ""


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    p = argparse.ArgumentParser()
    p.add_argument("runner")
    p.add_argument("engine_dir")
    p.add_argument("tokenizer_dir")
    p.add_argument("--plugin", default=os.path.join(root, "build", "tensorrt", "libflashrt_trt_pi05.so"))
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    d = os.path.abspath(args.engine_dir)
    work = os.path.join(d, "edgellm")
    os.makedirs(work, exist_ok=True)
    # Read single tensors: the recordings also hold FP8 tensors numpy cannot represent.
    with safe_open(os.path.join(d, "siglip_all.safetensors"), "np") as f:
        images = f.get_tensor("images_u8")
    with safe_open(os.path.join(d, "prompts.safetensors"), "np") as f:
        refs = {k: f.get_tensor(k) for k in f.keys() if k.endswith((".tokens", ".actions"))}
    paths = []
    for v in range(images.shape[0]):
        path = os.path.join(work, f"view{v}.png")
        Image.fromarray(images[v]).save(path)
        assert np.array_equal(np.asarray(Image.open(path)), images[v])
        paths.append(path)

    env = dict(os.environ)
    lib = cublas_dir()
    if lib:
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    ok = True
    for i, text in enumerate(LIBERO_PROMPTS):
        out = os.path.join(work, f"prompt{i}.json")
        cmd = [args.runner, "--engine", os.path.join(d, "pi05.engine"), "--plugin", args.plugin,
               "--tokenizer", args.tokenizer_dir, "--images", ",".join(paths), "--prompt", text,
               "--noise", os.path.join(d, "prompts.safetensors"), "--pixel_norm", "flashrt",
               "--iters", str(args.iters), "--output", out]
        r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            sys.exit(f"prompt {i}: pi05_policy_inference failed")
        res = json.load(open(out))
        raw = np.array(res["raw_actions"], dtype=np.float32).astype(np.float16)
        ref = refs[f"p{i}.actions"]
        tokens_ok = res["tokens"] == refs[f"p{i}.tokens"].tolist()
        bitwise = np.array_equal(raw, ref)
        diff = np.abs(raw.astype(np.float32) - ref.astype(np.float32)).max()
        lat = res.get("latency_ms", {})
        print(f"prompt {i}: {len(res['tokens'])} tokens | tokens match {tokens_ok} | raw actions bitwise {bitwise} "
              f"(max diff {diff:.3g}) | median {lat.get('median', float('nan')):.2f} ms, "
              f"cuda graph {lat.get('cuda_graph')}")
        ok &= tokens_ok and bitwise
    print("EDGELLM_PI05_PASS" if ok else "EDGELLM_PI05_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
