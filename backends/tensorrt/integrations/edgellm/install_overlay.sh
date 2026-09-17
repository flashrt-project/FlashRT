#!/bin/bash
# Install the pi0.5 example into a TensorRT Edge-LLM checkout as an experimental
# model, then build it:
#
#   backends/tensorrt/integrations/edgellm/install_overlay.sh <TensorRT-Edge-LLM dir>
#   cmake --build <TensorRT-Edge-LLM dir>/build --target pi05_policy_inference -j 2
#
# The checkout must be configured with -DBUILD_EXPERIMENTAL_MODELS=ON.
set -euo pipefail
EDGELLM=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
ln -sfn "$HERE/pi05" "$EDGELLM/experimental_models/pi05"
grep -qx "add_subdirectory(pi05)" "$EDGELLM/experimental_models/CMakeLists.txt" \
    || echo "add_subdirectory(pi05)" >> "$EDGELLM/experimental_models/CMakeLists.txt"
echo "installed: $EDGELLM/experimental_models/pi05 -> $HERE/pi05"
