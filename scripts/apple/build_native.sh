#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${1:-$root/.cache/apple-native}"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Apple Silicon macOS is required" >&2
  exit 1
fi
mkdir -p "$build"
# OS Signpost bridge can be used by the torch baseline without a Metal context.
xcrun clang++ -std=c++17 -O2 -dynamiclib -mmacosx-version-min=14.0 \
  -fvisibility=hidden -fvisibility-inlines-hidden \
  "-Wl,-exported_symbols_list,$root/cpp/apple/signpost.exports" \
  "$root/cpp/apple/src/signpost.mm" -o "$build/libflashrt_apple_signpost.dylib"
bash "$root/scripts/apple/check_exports.sh" "$build/libflashrt_apple_signpost.dylib" "$root/cpp/apple/signpost.exports"
# Small SDK-only entry point for machines without CMake or the offline Metal tools.
xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=14.0 \
  -fvisibility=hidden -fvisibility-inlines-hidden \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" -c "$root/cpp/apple/src/pipeline.mm" \
  -o "$build/pipeline.o"
xcrun ar rcs "$build/libflashrt_apple.a" "$build/pipeline.o"
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
  -fvisibility=hidden -fvisibility-inlines-hidden \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" -dynamiclib "$root/cpp/apple/src/plan.cpp" "$build/pipeline.o" \
  "-Wl,-exported_symbols_list,$root/cpp/apple/plan.exports" \
  -framework Foundation -framework Metal \
  -o "$build/libflashrt_apple_plan.dylib"
bash "$root/scripts/apple/check_exports.sh" "$build/libflashrt_apple_plan.dylib" "$root/cpp/apple/plan.exports"
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" "$root/cpp/apple/tests/pipeline_smoke.cpp" \
  "$build/libflashrt_apple.a" -framework Foundation -framework Metal \
  -o "$build/flashrt_apple_smoke"
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" "$root/cpp/apple/tests/bf16_gemm_smoke.cpp" \
  "$build/libflashrt_apple.a" -framework Foundation -framework Metal \
  -o "$build/flashrt_apple_bf16_smoke"
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" "$root/cpp/apple/tests/attention_smoke.cpp" \
  "$build/libflashrt_apple.a" -framework Foundation -framework Metal \
  -o "$build/flashrt_apple_attention_smoke"
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" "$root/cpp/apple/tests/gemm_tune.cpp" \
  "$build/libflashrt_apple.a" -framework Foundation -framework Metal \
  -o "$build/flashrt_apple_gemm_tune"
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
  -I "$root/cpp/apple/include" -I "$root/csrc/apple" "$root/cpp/apple/tests/kernel_registry_smoke.cpp" \
  "$build/libflashrt_apple.a" -framework Foundation -framework Metal \
  -o "$build/flashrt_apple_kernel_registry_smoke"
echo "Built host libraries and smoke executables in $build (Metal shaders compile at runtime)"
