# FlashRT official Thor image

The FlashRT-maintained ARM64 image includes compiled kernels, pinned official reference code, numerical validators and input fixtures for **OpenPI π0.5** and **GR00T N1.7 LIBERO**. Model weights stay in your mounted model directory.

## 1. Pull

On a Thor with Docker and NVIDIA Container Toolkit:

```bash
export IMAGE=ghcr.io/flashrt-project/flashrt-thor:thor-v0.1.1
export MODELS="$PWD/models"
export OUT="$PWD/results"
mkdir -p "$MODELS" "$OUT"
docker pull "$IMAGE"
docker run --rm --runtime=nvidia --gpus all "$IMAGE" doctor
```

The package is public; registry login is not required. Use the versioned tag for reproducible runs.

## 2. Prepare weights

For China pip access, optionally set `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` before these commands. GR00T and Cosmos download from versioned ModelScope mirrors with checksum verification. OpenPI downloads and converts the official `pi05_libero` checkpoint.

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare pi05
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare groot
```

Already prepared? Use the directory layout in the [π0.5 guide](pi05.md) and [GR00T guide](groot-n17.md), then skip this step. Existing verified GR00T files are reused.

## 3. Validate and benchmark

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate all
```

Use `validate pi05` or `validate groot` to run one model. The command freshly runs the official reference and FlashRT FP8/FP4, saves numerical errors and latency reports, and returns nonzero when numerical acceptance fails. Read [results](results.md) and the [JAL comparison contract](comparison.md); model inference, preprocessing and complete-call latency are separate measurements.

Reference loading, calibration and lazy kernel/graph initialization happen before measured inference. The first validation therefore takes longer than the latency shown in the benchmark table.

The release image passed fresh numerical checks, verified calibration-cache reuse and an anonymous registry pull followed by actual container validation. The [release evidence](results.md#published-image-validation) records the tested digest and measurements. Image filesystem layers and metadata were audited before publication.

To clone, install and compile yourself, or build the Docker image from source, follow [Thor setup](README.md). Maintainer publication steps are in the [release guide](release.md).
