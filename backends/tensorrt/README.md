# FlashRT TensorRT Backend

TensorRT plugins that run FlashRT's pipelines inside TensorRT engines, built
with the standard ONNX → `trtexec` workflow. First pipeline: Pi0.5 on Jetson
AGX Thor (NVFP4 + FlashAttention-4), bitwise identical to FlashRT's runtime.

- Usage: [docs/tensorrt_usage.md](../../docs/tensorrt_usage.md)
- Design, operator reference, keeping in sync: [docs/tensorrt_backend.md](../../docs/tensorrt_backend.md)
- TensorRT Edge-LLM: [docs/tensorrt_edgellm.md](../../docs/tensorrt_edgellm.md)

| path | contents |
|---|---|
| `CMakeLists.txt` | plugin library from `csrc/` kernels and `csrc/stages/pi05_thor/` |
| `plugins/` | `IPluginV3` implementations |
| `tools/` | engine build script, ONNX export, FA4 and tokenizer export, calibration fixture |
| `tools/reference/` | calibration and reference recording through FlashRT's frontend |
| `tests/` | native and engine parity tests, regression script |
| `integrations/openpi/` | openpi policy hook |
| `integrations/edgellm/` | TensorRT Edge-LLM example overlay and check |
