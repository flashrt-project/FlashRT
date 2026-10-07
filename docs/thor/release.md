# First public FlashRT Thor release

Use a GitHub account or organization controlled by your team. GHCR can publish a public image even when the linked source repository is private. The image itself must contain only material intended for public release.

The project repository already lives under `flashrt-project`. Prefer that organization for the Thor image if your account has package publishing permission there; creating another organization is optional.

## 1. Create the team identity

Sign in to GitHub. If the team already has an organization, use it; otherwise follow avatar → Settings → Organizations → New organization ([official steps](https://docs.github.com/en/organizations/collaborating-with-groups-in-organizations/creating-a-new-organization-from-scratch)). Select GitHub Free to start. Choose a stable team name and add another trusted owner for recovery. Use that organization as the image namespace.

If starting with a personal account, use your own username as the namespace. You do not need to create a Docker Hub account for GHCR.

## 2. Authenticate on the build Thor

Create a GitHub personal access token (classic) with `write:packages`, following the [official registry instructions](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry). Authorize organization SSO if required. Enter it only in the local terminal; do not put it in source files, build arguments or chat.

```bash
export REGISTRY_USER="<your GitHub username>"
read -rsp 'GHCR token: ' REGISTRY_TOKEN; echo
printf '%s' "$REGISTRY_TOKEN" | docker login ghcr.io -u "$REGISTRY_USER" --password-stdin
unset REGISTRY_TOKEN
```

The token owner must have permission to publish in the chosen organization.

## 3. Build and validate the release

From `repro/thor` in this repository on Thor:

```bash
docker build -f Dockerfile.thor -t flashrt-jal:release .
export IMAGE=flashrt-jal:release
export MODELS="$PWD/models"
export OUT="$PWD/release-results"
mkdir -p "$MODELS" "$OUT"
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate all
```

Prepare both models first if not already downloaded, using their guides. The release Dockerfile retains only the selected FlashRT snapshot and uses the checked source-only runtime archive.

On the validation machine, if repackaging the already validated local runtime, `docker build -f Dockerfile.release -t flashrt-jal:release .` is an alternative. This copies only the intended runtime directory and official reference environments into a fresh base image, excluding the original private repository layers.

Before pushing, all four numerical checks must pass. Review the final image contents and license notices; retain the build log and accuracy reports with the release. The repackaged source-only candidate was built on Thor and passed both models in FP8 and FP4. Its build/validation log is in `../../repro/thor/evidence/release-validation.log`, with accuracy reports in `../../repro/thor/evidence/release`. A fresh full source build should also run these checks before publication.

For this review branch, the new archive-based Docker recipe has passed its [source-stage check](../../repro/thor/evidence/public-source-build.json) on Thor. Its full image build has not been rerun. Build it and run `validate all` before publishing an image from this branch.

## 4. Publish

Choose your actual namespace and a version tag. For the first release, a name such as `flashrt-thor` and a version such as `thor-v0.1.0` is suitable; these are proposed names, not an existing published release.

```bash
export PUBLIC_IMAGE="ghcr.io/<your team>/flashrt-thor:thor-v0.1.0"
bash publish-image.sh "$IMAGE" "$PUBLIC_IMAGE"
```

On GitHub, open your user/organization Packages page → `flashrt-thor` → Package settings → Change visibility → Public. First publication is private by default. Check that the public package can be read without repository access.

## 5. Verify the user experience

On another Thor, use a fresh Docker credential directory:

```bash
CLEAN_DOCKER_CONFIG="$(mktemp -d)"
DOCKER_CONFIG="$CLEAN_DOCKER_CONFIG" docker pull "$PUBLIC_IMAGE"
```

Then run each model guide using `IMAGE="$PUBLIC_IMAGE"`. Save the second machine's reports. Once this succeeds, put the real pull address into README and attach the reports to the release. Keep this version tag unchanged; publish later fixes under another version.

This is a FlashRT maintained official project image. Docker Official Images certification and NVIDIA endorsement are separate programs and are not required for public pull access.
