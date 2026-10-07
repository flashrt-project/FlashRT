# FlashRT maintained Thor image

Goal: a public ARM64 image maintained by FlashRT. Users pull it, mount model/result directories, then run `prepare` and `validate`. Dependencies, compiled kernels, reference code and regression fixtures are inside the image; model weights remain external.

A source-only release candidate, repackaged with existing compiled kernels, passed both models in FP8/FP4. The complete `Dockerfile.thor` build from public source also finished and passed all four numerical checks; see [results](results.md). No public registry image is available for `docker pull` yet. Build instructions are in [Thor setup](README.md); model commands are in the two model guides.

Before public release:

1. Select a team-owned GHCR or Docker Hub namespace and release tag.
2. Build from only the intended FlashRT source snapshot. The current private runtime image contains a full repository Git history; do not push it publicly as-is. Removing files in a later Docker layer does not remove earlier layer contents.
3. Run both model validators on Thor and preserve the reports.
4. Push the release image, make the package public, and test pulling without registry credentials on another Thor.
5. Replace the local build steps in the guide with the verified published image name. Keep released tags stable; release changes under a new tag.

After publication, the user flow is:

```bash
export IMAGE="<published FlashRT image>"
export MODELS="$PWD/models"
export OUT="$PWD/results"
mkdir -p "$MODELS" "$OUT"
docker pull "$IMAGE"
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" "$IMAGE" prepare groot
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate groot
```

Replace `groot` with `pi05` for OpenPI. With already prepared weights, skip `prepare`.

Public GHCR packages support anonymous pulls ([GitHub documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)). “FlashRT maintained image” does not imply the separately curated [Docker Official Images](https://docs.docker.com/docker-hub/repos/manage/trusted-content/official-images/) badge or NVIDIA endorsement.

First registry/account setup and publication: [release guide](release.md).
