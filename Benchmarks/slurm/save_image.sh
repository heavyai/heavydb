#!/bin/bash
# Pull a HeavyDB product image from a registry and save it as an enroot .sqsh,
# using pyxis (srun --container-image/--container-save) so no root-owned enroot
# directories are needed. Same approach as velox-testing's save_image_locally.sh.
#
#   save_image.sh <image> <output.sqsh> [-- extra srun args, e.g. --account=X --partition=Y]
#
# Example (Docker Hub, private heavyai/scantest repo):
#   save_image.sh heavyai/scantest:ubuntu22.04-cuda-render-aarch64-10.0.0-20260929-39952e6c25 \
#       $HOME/images/heavydb-arm.sqsh -- --partition=interactive
#
# Registry credentials go in ~/.config/enroot/.credentials, one line per registry:
#   machine auth.docker.io login <dockerhub-user> password <access-token>
#   machine ghcr.io        login <github-user>    password <PAT with read:packages>
# (Docker Hub images are named "heavyai/scantest:tag"; other registries use
#  "host#path:tag", e.g. "ghcr.io#org/image:tag" -- pyxis syntax.)
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
image=$1
out=$(realpath -m "$2")
shift 2
[ "${1:-}" = "--" ] && shift

mkdir -p "$(dirname "$out")"
rm -f "$out"

# The container only needs to start and exit; --container-save writes the sqsh at exit.
srun --nodes=1 --ntasks=1 --time=00:30:00 --job-name=heavydb-image-save \
  --container-mounts="$(dirname "$out")":"$(dirname "$out")" \
  --container-image="$image" \
  --container-save="$out" \
  "$@" \
  -- true

ls -lh "$out"
