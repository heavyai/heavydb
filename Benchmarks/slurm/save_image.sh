#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pull a HeavyDB product image from a registry and save it as an enroot .sqsh,
# using pyxis (srun --container-image/--container-save) so no root-owned enroot
# directories are needed. Same approach as velox-testing's save_image_locally.sh.
#
#   save_image.sh <image> <output.sqsh> [-- extra srun args, e.g. --account=X --partition=Y]
#
# Example (a private Docker Hub repo):
#   save_image.sh <user>@registry-1.docker.io#<org>/<repo>:<tag> \
#       $HOME/images/heavydb-arm.sqsh -- --partition=interactive
#
# NOTE: enroot only uses a credentials entry when the image names the user, so
# private Docker Hub images need the form  USER@registry-1.docker.io#<org>/<repo>:<tag>
# (a bare <org>/<repo>:<tag> is pulled anonymously and fails with 401).
#
# Registry credentials go in ~/.config/enroot/.credentials, one line per registry:
#   machine auth.docker.io login <dockerhub-user> password <access-token>
#   machine ghcr.io        login <github-user>    password <PAT with read:packages>
# (Other registries use
#  "host#path:tag", e.g. "ghcr.io#org/image:tag" -- pyxis syntax.)
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n "4,24p" "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
image=$1
out=$(realpath -m "$2")  # pure path math; works even if the dir is absent here
shift 2
[ "${1:-}" = "--" ] && shift

# One job: pyxis pulls the image and writes the sqsh at exit. The output directory
# must already exist and be visible on the compute node (e.g. /scratch).
srun --nodes=1 --ntasks=1 --time=00:30:00 --job-name=heavydb-image-save \
  --container-image="$image" \
  --container-save="$out" \
  "$@" \
  -- true
