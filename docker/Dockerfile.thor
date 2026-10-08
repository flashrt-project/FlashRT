# FlashRT source build for Jetson AGX Thor, SM110 / linux-arm64.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM scratch AS tokenizer_cache
FROM ${BASE_IMAGE}
ARG BUILD_JOBS=2
ARG PIP_INDEX_URL=https://pypi.org/simple
LABEL org.opencontainers.image.title="FlashRT Thor"
LABEL org.opencontainers.image.source="https://github.com/flashrt-project/FlashRT"
ENV CUTE_DSL_ARCH=sm_101a
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build ffmpeg libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
      numpy==1.26.4 safetensors==0.8.0 sentencepiece pillow pybind11 ninja \
      nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 huggingface-hub modelscope==1.40.0 modelscope-hub==0.4.2 requests \
    && python -m pip uninstall -y torchao
WORKDIR /opt/FlashRT
COPY CMakeLists.txt ./
COPY csrc/ csrc/
COPY cmake/ cmake/
RUN git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass \
    && rm -rf third_party/cutlass/.git \
    && cmake -S . -B build -DGPU_ARCH=110 -DCMAKE_CUDA_ARCHITECTURES=OFF \
         -DCMAKE_BUILD_TYPE=Release -DFLASHRT_ENABLE_PI05_THOR=ON \
    && cmake --build build -j${BUILD_JOBS} --target \
         flash_rt_kernels flash_rt_fp4 fmha_fp16_strided flash_rt_pi05_thor \
    && rm -rf build
COPY flash_rt/ flash_rt/
COPY pyproject.toml README.md LICENSE ./
RUN python -m pip install --no-deps --no-build-isolation -e . \
    && python -c "from flash_rt import flash_rt_kernels, flash_rt_fp4, flash_rt_pi05_thor"
COPY docker/install-reference.sh docker/requirements-openpi.txt docker/requirements-groot.txt docker/
RUN bash docker/install-reference.sh /opt/reference all
COPY . .
ARG TOKENIZER_URL=https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
RUN --mount=type=bind,from=tokenizer_cache,target=/tokenizer-cache \
    mkdir -p /opt/tokenizer \
    && if [ -s /tokenizer-cache/paligemma_tokenizer.model ]; then \
         cp /tokenizer-cache/paligemma_tokenizer.model /opt/tokenizer/paligemma_tokenizer.model; \
       else \
         curl -fL --retry 3 "$TOKENIZER_URL" -o /opt/tokenizer/paligemma_tokenizer.model; \
       fi
RUN mkdir -p /root/.cache/openpi/big_vision && cp /opt/tokenizer/paligemma_tokenizer.model /root/.cache/openpi/big_vision/paligemma_tokenizer.model
ENV FLASH_RT_PALIGEMMA_TOKENIZER=/opt/tokenizer/paligemma_tokenizer.model
ENV LINGBOT_FA4_SRC=/opt/FlashRT/csrc/attention/flash_attn_4_src
ENV PYTHONPATH=/opt/FlashRT
CMD ["bash"]
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
