"""The TensorRT-native side of the operator comparison.

Each entry names the tensors that bound one operator in the openpi Thor
tutorial's ONNX export, so a subgraph cut there runs TensorRT's own quantized
kernels over the same weights, at the same shape, between the same boundaries
as the matching FlashRT operator plugin.

Shapes are templates. A dimension written as `{rows}`, `{views}` or `{tokens}`
depends on the task and becomes a symbolic dimension that the profile pins; a
literal stays static, which some TensorRT layers require of the axis they block
along (the FP4 dynamic quantizer wants its K known at build time).
"""
import os
import re
import shutil
import subprocess

CASES = {
    "encoder_mlp": dict(
        inputs=["/layers.0/post_attention_layernorm/Cast_2_output_0"],
        input_shapes=["1x{rows}x2048"],
        outputs=["/layers.0/mlp/down_proj/MatMul_output_0"],
        output_shapes=["1x{rows}x2048"],
        plugin="encoder_mlp_plain"),
    "encoder_o": dict(
        inputs=["/layers.0/self_attn/Reshape_7_output_0"],
        input_shapes=["1x{rows}x2048"],
        outputs=["/layers.0/self_attn/o_proj/MatMul_output_0"],
        output_shapes=["1x{rows}x2048"],
        plugin="encoder_o_plain"),
    # The vision tower runs FP8 in the tutorial export and NVFP4 in FlashRT;
    # the comparison has to say so.
    "siglip_mlp": dict(
        inputs=["/vision_tower/vision_model/encoder/layers.0/layer_norm2/"
                "LayerNormalization_output_0"],
        input_shapes=["{views}x{tokens}x1152"],
        outputs=["/vision_tower/vision_model/encoder/layers.0/mlp/fc2/MatMul_output_0"],
        output_shapes=["{views}x{tokens}x1152"],
        plugin="siglip_mlp_plain"),
    "siglip_attn": dict(
        inputs=["/vision_tower/vision_model/encoder/layers.0/self_attn/Transpose_output_0",
                "/vision_tower/vision_model/encoder/layers.0/self_attn/Transpose_2_output_0",
                "/vision_tower/vision_model/encoder/layers.0/self_attn/Transpose_1_output_0"],
        input_shapes=["{views}x16x{tokens}x72", "{views}x16x72x{tokens}",
                      "{views}x16x{tokens}x72"],
        outputs=["/vision_tower/vision_model/encoder/layers.0/self_attn/MatMul_1_output_0"],
        output_shapes=["{views}x16x{tokens}x72"],
        plugin="siglip_attn"),
}

TRTEXEC_FLAGS = ["--useCudaGraph", "--noDataTransfers", "--iterations=200", "--avgRuns=10"]


def find_trtexec(explicit=None):
    for candidate in (explicit, os.environ.get("TRTEXEC"), shutil.which("trtexec"),
                      "/usr/src/tensorrt/bin/trtexec"):
        if candidate and os.path.exists(candidate):
            return candidate
    raise SystemExit("trtexec not found: pass --trtexec or set TRTEXEC")


def dims(template, unique):
    """A value_info shape from a template: literals stay, `{...}` becomes a symbol.

    Every symbol is unique, because TensorRT reads a shared dimension name as
    one value and the optimization profile then contradicts itself.
    """
    out = []
    for i, part in enumerate(template.split("x")):
        out.append(f"{unique}_{i}" if "{" in part else int(part))
    return out


def shapes_for(tag, rows, views, tokens):
    case = CASES[tag]
    return ",".join(f"{n}:{t.format(rows=rows, views=views, tokens=tokens)}"
                    for n, t in zip(case["inputs"], case["input_shapes"]))


def run_trtexec(cmd, log):
    """Run trtexec and return its median GPU compute time in microseconds."""
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    median = None
    for line in open(log):
        if "GPU Compute Time" in line and "median" in line:
            median = float(re.search(r"median = ([0-9.]+) ms", line).group(1)) * 1000
    return median
