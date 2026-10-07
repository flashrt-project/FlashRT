#!/usr/bin/env bash
set -euo pipefail
SOURCE="${1:?Usage: bash install-native.sh /absolute/path/to/FlashRT-pi05-thor-limit-5421c93}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
cd "$SOURCE"
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()==(11,0); print(torch.__version__,torch.version.cuda)'
test -d third_party/cutlass || git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
python -m pip install --index-url "$PIP_INDEX_URL" numpy==1.26.4 safetensors==0.8.0 nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 sentencepiece pillow pybind11 ninja
python -m pip install --no-deps --no-build-isolation -e .
cmake -S . -B build -DGPU_ARCH=110 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"${BUILD_JOBS:-2}" --target flash_rt_kernels flash_rt_fp4 fmha_fp16_strided
export PYTHONPATH="$SOURCE"
export CUTE_DSL_ARCH=sm_101a
export LINGBOT_FA4_SRC="$SOURCE/csrc/attention/flash_attn_4_src"
python -c 'import flash_rt; from flash_rt import flash_rt_kernels,flash_rt_fp4; print(flash_rt.__file__)'
echo "Build complete. Export PYTHONPATH, CUTE_DSL_ARCH and LINGBOT_FA4_SRC in each runtime shell as above."
