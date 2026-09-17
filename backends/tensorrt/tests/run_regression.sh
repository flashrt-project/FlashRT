#!/bin/bash
# Bitwise and latency regression of the FlashRT TensorRT backend on Jetson Thor.
#
#   backends/tensorrt/tests/run_regression.sh <reference_dir> [<openpi_reference_dir>]
#
# <reference_dir> holds FlashRT reference dumps recorded with the 3-view,
# 208-token-prompt configuration (the openpi tutorial engine's shape):
#   encoder_layer1.safetensors  tools/reference/pi05_pipeline.py --dump-layer 1
#   encoder_all.safetensors     tools/reference/dump_encoder.py
#   decoder_steps.safetensors   tools/reference/dump_decoder.py
#   siglip_all.safetensors      tools/reference/dump_siglip.py
# <openpi_reference_dir> is a build_pi05_engine.sh output directory (optional).
#
# Environment: PYTHON (FlashRT Python), ONNX_PYTHON, BUILD_DIR (default
# <repo>/build/tensorrt, configured with -DFLASHRT_TRT_BUILD_TESTS=ON), TRTEXEC.
#
# Preconditions for the bitwise gates:
#  - Nothing else runs on the GPU (contention also inflates latency 3-4x).
#  - The cuBLAS build is PyTorch's (loaded below): the decoder's small GEMMs
#    pick different kernels in other builds.
set -uo pipefail
M=$1; M6=${2:-}
T=$(cd "$(dirname "$0")" && pwd); BACKEND=$(dirname "$T"); ROOT=$(cd "$BACKEND/../.." && pwd)
PYTHON=${PYTHON:-python3}; ONNX_PYTHON=${ONNX_PYTHON:-$PYTHON}
B=${BUILD_DIR:-$ROOT/build/tensorrt}; P=$B/libflashrt_trt_pi05.so
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" BUILDER_OPT_LEVEL=0 NO_CONCAT=1
# FlashRT compiles FA4 for Thor under this CuTe DSL chip name (nvidia-cutlass-dsl 4.5).
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_101a}
CUBLAS_DIR=$($PYTHON -c "import nvidia, os; d = os.path.join(list(nvidia.__path__)[0], 'cu13', 'lib'); print(d if os.path.isdir(d) else '')" 2>/dev/null || true)
export LD_LIBRARY_PATH="${CUBLAS_DIR:+$CUBLAS_DIR:}${LD_LIBRARY_PATH:-}"
WORK=$(mktemp -d)
run() { echo "=== $1"; shift; timeout 1800 "$@" 2>&1 | grep -E "bitwise|parity|PARITY|median|PASS|FAIL|PASSED|FAILED|wrote|Traceback|Error:|Segmentation" | grep -v "scale(s) exceed" | cut -c1-220; echo "exit=${PIPESTATUS[0]}"; }

run "FA4 AOT modules vs FlashRT JIT FA4" $PYTHON "$T/fa4_aot_parity.py" "$B/libfa4_aot_runner.so"
run "native encoder layer" "$B/encoder_layer_parity" "$M/encoder_layer1.safetensors" 200
run "native encoder (18 layers)" "$B/encoder_stage_parity" "$M/encoder_all.safetensors" 100
run "native decoder (10 steps)" "$B/decoder_step_parity" "$M/decoder_steps.safetensors" 50
run "native SigLIP" "$B/siglip_parity" "$M/siglip_all.safetensors" 100
run "engine: encoder layer plugin" $PYTHON "$T/engine_encoder_layer.py" "$P" "$M/encoder_layer1.safetensors" "$WORK/layer.engine"
run "engine: 18 chained encoder layer plugins" $PYTHON "$T/engine_encoder_layer_chain.py" "$P" "$M/encoder_all.safetensors"
run "engine: encoder stage plugin" $PYTHON "$T/engine_encoder_stage.py" "$P" "$M/encoder_all.safetensors"
run "engine: 10 chained decoder step plugins" $PYTHON "$T/engine_decoder_step_chain.py" "$P" "$M/decoder_steps.safetensors"
run "engine: decoder stage plugin" $PYTHON "$T/engine_decoder_stage.py" "$P" "$M/decoder_steps.safetensors"
run "engine: SigLIP layer chain and stage plugins" $PYTHON "$T/engine_siglip.py" "$P" "$M/siglip_all.safetensors"
run "engine: whole policy (network API)" $PYTHON "$T/engine_policy.py" "$P" "$M/siglip_all.safetensors" "$M/encoder_all.safetensors" "$M/decoder_steps.safetensors"
run "ONNX export" $ONNX_PYTHON "$BACKEND/tools/export_onnx.py" "$M/siglip_all.safetensors" "$M/encoder_all.safetensors" "$M/decoder_steps.safetensors" "$WORK/onnx"
run "trtexec build" "$TRTEXEC" --onnx="$WORK/onnx/pi05.onnx" --dynamicPlugins="$P" --stronglyTyped --builderOptimizationLevel=0 --memPoolSize=workspace:2048 --saveEngine="$WORK/pi05.engine" --skipInference
run "engine: whole policy (ONNX + trtexec)" $PYTHON "$T/engine_file.py" "$P" "$WORK/pi05.engine" "$M/siglip_all.safetensors" "$M/decoder_steps.safetensors"
if [ -n "$M6" ]; then
    run "engine: decoder step chain (openpi checkpoint)" $PYTHON "$T/engine_decoder_step_chain.py" "$P" "$M6/decoder_steps.safetensors"
    run "engine: prompt-dynamic policy (openpi checkpoint)" $PYTHON "$T/engine_prompt_dynamic.py" "$P" "$M6/pi05.engine" "$M6/siglip_all.safetensors" "$M6/prompts.safetensors" "$M6/decoder_steps.safetensors"
fi
rm -rf "$WORK"
echo REGRESSION_DONE
