# ONNX / ModelOpt Quantization Bridge

FlashRT consumes external FP8 calibration through a JSON quantization manifest.
The manifest is generated offline from a ModelOpt/ONNX QDQ graph, then loaded
by the Pi0.5 RTX/Orin FP8 frontend before graph capture. Runtime inference
still uses static FlashRT scales and FP8 GEMMs.

## Isolated Environment

Create a local environment for ONNX/ModelOpt tooling:

```bash
scripts/setup_onnx_modelopt_env.sh .venv-onnx-modelopt
source .venv-onnx-modelopt/bin/activate
```

Inside a PyTorch deployment container, reuse the container's torch install:

```bash
SYSTEM_SITE_PACKAGES=1 scripts/setup_onnx_modelopt_env.sh .venv-onnx-modelopt
source .venv-onnx-modelopt/bin/activate
```

For ONNX-only manifest extraction, skip ModelOpt:

```bash
INSTALL_MODELOPT=0 scripts/setup_onnx_modelopt_env.sh .venv-onnx-modelopt
```

NVIDIA documents ModelOpt installation through the `nvidia-modelopt` package
and NVIDIA PyPI extra index. The isolated environment keeps those dependencies
out of the FlashRT runtime environment.

The ONNX tooling path requires ModelOpt's optional ONNX dependencies, including
`onnx-graphsurgeon`. The setup script installs the `onnx` extra by default;
set `MODELOPT_EXTRAS=all` only when the full Torch/HuggingFace plugin stack is
needed.

In containers that already ship PyTorch and an older `nvidia-modelopt`, use
`SYSTEM_SITE_PACKAGES=1`. The setup script upgrades ModelOpt in the venv with
`--no-deps` and installs only the ONNX-side dependencies needed for QDQ export
and manifest parsing. This avoids pulling a second PyTorch/CUDA stack.

FlashRT does not require ModelOpt at inference time. The runtime consumes the
exported ONNX QDQ scales through the JSON manifest only.

## ModelOpt Handoff Requirement

FlashRT expects the delivered ONNX to already contain QDQ around every runtime
FP8 GEMM input and weight tensor. In local validation with ModelOpt 0.44, the
ONNX FP8 quantizer did not insert FP8 QDQ for standalone MatMul-only graphs
(`Total number of quantized nodes: 0`), even when `nodes_to_quantize` matched
all MatMul nodes. Therefore the quantization team should provide a real Pi0.5
QDQ ONNX artifact from their production ModelOpt flow, not just an unquantized
MatMul ONNX skeleton.

The expected contract is:

- every FlashRT runtime site maps to one or more ONNX MatMul/Gemm nodes;
- each mapped node has `DequantizeLinear` on activation input 0 and weight
  input 1, or an equivalent QDQ pattern traceable back to scale initializers;
- fused FlashRT sites such as QKV and Gate+Up may map to several ONNX nodes;
  scalar scales are combined with `max(scale_i)`.

## Manifest Schema

The runtime manifest uses FlashRT internal GEMM site names. A separate mapping
file bridges ONNX initializer names to FlashRT sites, so graph-specific fusion
rules stay outside inference code.

Mapping file:

```json
{
  "model": "pi05",
  "sites": {
    "encoder_ffn_down_w_16": {
      "weight_scale": "encoder.layers.16.ffn.down.weight_scale",
      "activation_scale": "encoder.layers.16.ffn.down.input_scale",
      "onnx_nodes": ["encoder.layers.16.ffn.down/MatMul"]
    }
  }
}
```

For Q/DQ ONNX graphs, the mapping can also point at the quantized MatMul/Gemm
node. The exporter traces each selected node input back through
`DequantizeLinear` and reads the scale initializer automatically:

```json
{
  "model": "pi05",
  "sites": {
    "encoder_attn_qkv_w_0": {
      "onnx_nodes": [
        "encoder.layers.0.attn.q/MatMul",
        "encoder.layers.0.attn.k/MatMul",
        "encoder.layers.0.attn.v/MatMul"
      ],
      "activation_input": 0,
      "weight_input": 1
    },
    "encoder_ffn_gate_up_w_0": {
      "onnx_nodes": [
        "encoder.layers.0.ffn.gate/MatMul",
        "encoder.layers.0.ffn.up/MatMul"
      ]
    }
  }
}
```

When several ONNX nodes map to one fused FlashRT site, scalar FP8 scales are
combined with `max(scale_i)`. For scales derived as `amax / fp8_max`, this
matches quantizing the fused tensor with a single per-tensor scale.

Export manifest:

```bash
python scripts/pi05_onnx_modelopt_mapping.py \
  --onnx modelopt_qdq.onnx \
  --out pi05_modelopt_mapping.json

python scripts/onnx_modelopt_export_manifest.py \
  --onnx modelopt_qdq.onnx \
  --mapping pi05_modelopt_mapping.json \
  --out pi05_quant_manifest.json
```

By default the generated manifest records only input file names, not full local
paths. Pass `--include-source-paths` only when the artifact is meant for local
debugging.

Runtime load:

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint,
    config="pi05",
    hardware="rtx_sm120",
    use_fp8=True,
    quant_manifest="pi05_quant_manifest.json",
    quant_policy="strict",
)
```

`strict` requires every runtime FP8 site to have a manifest scale.
`compatible` uses manifest entries where present and falls back to FlashRT
local calibration for missing activation scales and local amax for missing
weight scales.

For integration validation without a ModelOpt artifact, Pi0.5 can export its
current calibrated FlashRT scales:

```python
model.calibrate(observations, percentile=100.0)
model.export_quant_manifest("flashrt_quant_manifest.json")
```

Reload that file with `quant_policy="strict"` to validate manifest coverage and
runtime behavior independently from ONNX export.

## Current Support

Pi0.5 currently accepts per-tensor FP8 E4M3 weight and activation scales. The
manifest parser can represent per-channel scales for future frontends, but the
current Pi0.5 FP8 GEMM path rejects them explicitly instead of silently changing
precision or layout.
