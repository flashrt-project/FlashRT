#!/bin/bash
# Build a FlashRT Pi0.5 TensorRT engine for Jetson Thor from a checkpoint.
#
#   backends/tensorrt/tools/build_pi05_engine.sh <checkpoint> <num_views> <fixture.npz> <out_dir>
#
#   1. calibrate FlashRT's production FP4 pipeline on the fixture observations
#      and record the SigLIP, encoder and decoder stages and the prompt tables,
#   2. export ONNX (inputs images, lang_tokens, noise; output actions),
#   3. build the engine with trtexec and check it bit for bit against FlashRT.
#
# Environment:
#   PYTHON      Python with FlashRT built and importable (default: python3)
#   ONNX_PYTHON Python with numpy + onnx (default: $PYTHON)
#   PLUGIN      plugin library (default: <repo>/build/tensorrt/libflashrt_trt_pi05.so)
#   TRTEXEC     trtexec (default: trtexec on PATH, then /usr/src/tensorrt/bin/trtexec)
#   MAX_TOKENS  largest prompt the engine accepts (default: 256)
#   SKIP_CHECK  set to 1 to skip the bitwise engine check
set -euo pipefail
[ $# -eq 4 ] || { sed -n 2,20p "$0"; exit 2; }
CKPT=$1; VIEWS=$2; FIXTURE=$3; OUT=$4
T=$(cd "$(dirname "$0")" && pwd)
BACKEND=$(dirname "$T")
ROOT=$(cd "$BACKEND/../.." && pwd)
PYTHON=${PYTHON:-python3}
ONNX_PYTHON=${ONNX_PYTHON:-$PYTHON}
PLUGIN=${PLUGIN:-$ROOT/build/tensorrt/libflashrt_trt_pi05.so}
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
MAX_TOKENS=${MAX_TOKENS:-256}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# FlashRT compiles FA4 for Thor under this CuTe DSL chip name (nvidia-cutlass-dsl 4.5).
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_101a}
[ -f "$PLUGIN" ] || { echo "plugin library not found: $PLUGIN (build backends/tensorrt first)"; exit 1; }
mkdir -p "$OUT/onnx"

A="--checkpoint $CKPT --fixture $FIXTURE --num-views $VIEWS --prompt-len 0"
step() {
    echo "== $1"; shift
    "$@" 2>&1 | grep -E "reference|checks|bitwise|wrote|Error|Traceback|PASS|FAIL" || true
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "step failed"; exit 1; }
}
step "calibrate + record SigLIP" $PYTHON "$T/reference/dump_siglip.py" $A --out "$OUT/siglip_all.safetensors"
step "calibrate + record encoder" $PYTHON "$T/reference/dump_encoder.py" $A --out "$OUT/encoder_all.safetensors"
step "calibrate + record decoder" $PYTHON "$T/reference/dump_decoder.py" $A --out "$OUT/decoder_steps.safetensors"
step "prompt tables + references" $PYTHON "$T/reference/dump_prompts.py" $A --out "$OUT/prompts.safetensors"
step "ONNX export" $ONNX_PYTHON "$T/export_onnx.py" "$OUT/siglip_all.safetensors" "$OUT/encoder_all.safetensors" \
    "$OUT/decoder_steps.safetensors" "$OUT/onnx" --prompts "$OUT/prompts.safetensors"
echo "== trtexec build"
"$TRTEXEC" --onnx="$OUT/onnx/pi05.onnx" --dynamicPlugins="$PLUGIN" --stronglyTyped --builderOptimizationLevel=0 \
    --memPoolSize=workspace:2048 --minShapes=lang_tokens:2 --optShapes=lang_tokens:14 \
    --maxShapes=lang_tokens:$MAX_TOKENS --saveEngine="$OUT/pi05.engine" --skipInference 2>&1 \
    | grep -E "PASSED|FAILED|\[E\]" | cut -c1-100
if [ "${SKIP_CHECK:-0}" != 1 ]; then
    # FlashRT's references use the cuBLAS build PyTorch ships; load the same one.
    CUBLAS_DIR=$($PYTHON -c "import nvidia, os; d = os.path.join(list(nvidia.__path__)[0], 'cu13', 'lib'); print(d if os.path.isdir(d) else '')" 2>/dev/null || true)
    LD_LIBRARY_PATH="${CUBLAS_DIR:+$CUBLAS_DIR:}${LD_LIBRARY_PATH:-}" step "engine check" $PYTHON "$BACKEND/tests/engine_prompt_dynamic.py" \
        "$PLUGIN" "$OUT/pi05.engine" "$OUT/siglip_all.safetensors" "$OUT/prompts.safetensors" "$OUT/decoder_steps.safetensors"
fi
echo "engine: $OUT/pi05.engine"
echo "plugin: $PLUGIN"
