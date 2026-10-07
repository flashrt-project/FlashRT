ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba
FROM ${BASE_IMAGE} AS selected_source
COPY runtime-source.tar.gz versions.json unpack-runtime.py /source/
RUN python /source/unpack-runtime.py --out /snapshot

FROM ${BASE_IMAGE}
LABEL org.flashrt.release-source-only="true"
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG SOURCE_COMMIT=8efce63a2e77048426d6f30c71ec7118f08eeeb8
ARG BUILD_JOBS=2
ENV CUTE_DSL_ARCH=sm_101a
RUN apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY --from=selected_source /snapshot /opt/FlashRT-PAI/FlashRT-pi05-thor-limit-5421c93
WORKDIR /opt/FlashRT-PAI/FlashRT-pi05-thor-limit-5421c93
RUN git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
RUN python -m pip install --index-url ${PIP_INDEX_URL} 'numpy==1.26.4' 'safetensors==0.8.0' sentencepiece pillow pytest pybind11 ninja 'nvidia-cutlass-dsl==4.5.1' 'quack-kernels==0.4.1' huggingface-hub
RUN python -m pip install --no-deps --no-build-isolation -e .
RUN cmake -S . -B build -DGPU_ARCH=110 -DCMAKE_BUILD_TYPE=Release && cmake --build build -j${BUILD_JOBS} --target flash_rt_kernels flash_rt_fp4 fmha_fp16_strided
ENV PYTHONPATH=/opt/FlashRT-PAI/FlashRT-pi05-thor-limit-5421c93
ENV LINGBOT_FA4_SRC=/opt/FlashRT-PAI/FlashRT-pi05-thor-limit-5421c93/csrc/attention/flash_attn_4_src
RUN python -c "import flash_rt;from flash_rt import flash_rt_kernels,flash_rt_fp4;import flash_rt.hardware.thor.fa4_backend"
COPY setup-reference-envs.sh requirements-*.txt /opt/jal/
# Optional torchao in the base conflicts with the official pinned diffusers.
RUN python -m pip uninstall -y torchao && bash /opt/jal/setup-reference-envs.sh /opt/reference
COPY *.py *.sh /opt/jal/
COPY versions.json official-files.json mirror-files.json /opt/jal/
COPY fixtures/ /opt/jal/fixtures/
WORKDIR /opt/jal
ENTRYPOINT ["bash", "/opt/jal/flashrt-jal.sh"]
CMD ["help"]
