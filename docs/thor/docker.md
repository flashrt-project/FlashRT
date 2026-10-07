# Run FlashRT on Thor with Docker

Requirements: Jetson AGX Thor, JetPack, Docker and NVIDIA Container Toolkit.
The image was verified on L4T R39.2.1 / CUDA 13.2. Registry login is not required.

## 1. Pull and check the runtime

```bash
export IMAGE=ghcr.io/flashrt-project/flashrt-thor:thor-v0.1.1
export MODELS="$PWD/models"
export OUT="$PWD/results"
mkdir -p "$MODELS" "$OUT"
docker pull "$IMAGE"
docker run --rm --runtime=nvidia --gpus all "$IMAGE" doctor
```

Expected: Thor GPU, capability `[11, 0]` and `fa4_available: true`.

## 2. Prepare weights

For China pip access, optionally set `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`.

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare pi05
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare groot
```

OpenPI weights are downloaded, checksum-verified and converted to BF16.
GR00T and Cosmos download from ModelScope. To reuse weights, follow the
directory layouts in the [π0.5](pi05.md) and [GR00T](groot-n17.md) tutorials.

## 3. Validate and benchmark

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate all
```

Use `validate pi05` or `validate groot` for one model. First-run loading,
calibration and initialization happen before measured inference.

Expected final output: `combined accuracy status: 0`. The image runs fresh
official references and checks FlashRT FP8 and FP4. Follow each model's
**Check the results** section to print numerical status and latency from `$OUT`.

| FP4 model | Expected latency |
|---|---:|
| π0.5 LIBERO | about 20 ms |
| GR00T LIBERO model inference | about 24.3 ms |
| GR00T LIBERO complete call | about 27.9 ms |

[Verified public-image reports](results.md#published-image-validation).

## Optional: build the Docker image yourself

```bash
git clone --branch docs/thor-release-and-community https://github.com/flashrt-project/FlashRT.git
cd FlashRT/repro/thor
docker build -f Dockerfile.thor -t flashrt-thor:local .
export IMAGE=flashrt-thor:local
```

Use this `IMAGE` in the same preparation and validation commands above.
For installation without Docker, use the native option in either model tutorial.
