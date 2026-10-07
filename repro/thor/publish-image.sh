#!/usr/bin/env bash
set -euo pipefail
LOCAL_IMAGE="${1:?Usage: publish-image.sh LOCAL_IMAGE PUBLIC_IMAGE:VERSION}"
PUBLIC_IMAGE="${2:?Set a team-owned public image and release version}"
test "$(docker image inspect "$LOCAL_IMAGE" --format '{{.Architecture}}')" = arm64
test "$(docker image inspect "$LOCAL_IMAGE" --format '{{index .Config.Labels "org.flashrt.release-source-only"}}')" = true || { echo 'Use a source-only release build from Dockerfile.thor; private runtime candidates cannot be published.' >&2; exit 2; }
docker tag "$LOCAL_IMAGE" "$PUBLIC_IMAGE"
docker push "$PUBLIC_IMAGE"
echo 'Set the registry package visibility to public, then verify anonymous pull on another Thor.'
