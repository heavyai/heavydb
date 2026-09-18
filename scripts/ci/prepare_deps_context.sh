#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stages a deps tarball and its supporting scripts into a Docker build context
# directory (one of docker/build/{ubuntu_param,rockylinux8_param}).
#
# Usage: prepare_deps_context.sh <param_dir> <tarball_path> [--distro=<distro>]
#
# Extracted from the "Prepare Docker build context" step in
# .github/workflows/deps-image-builder.yml so the same logic runs locally
# (via dev.sh build-deps-image) and in CI.

set -euo pipefail

PARAM_DIR="${1:-}"
TARBALL="${2:-}"
DISTRO=ubuntu

for arg in "${@:3}"; do
  case "$arg" in
    --distro=*) DISTRO="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [ -z "$PARAM_DIR" ] || [ -z "$TARBALL" ]; then
  echo "Usage: $(basename "$0") <param_dir> <tarball_path> [--distro=<distro>]" >&2
  exit 1
fi
if [ ! -d "$PARAM_DIR" ]; then
  echo "ERROR: param_dir not found: $PARAM_DIR" >&2
  exit 1
fi
if [ ! -f "$TARBALL" ]; then
  echo "ERROR: tarball not found: $TARBALL" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cp "$TARBALL" "$PARAM_DIR/"
cp "$SCRIPTS_DIR/mapd-deps-prebuilt.sh" "$PARAM_DIR/"
cp "$SCRIPTS_DIR/common-functions.sh" "$PARAM_DIR/"
cp "$SCRIPTS_DIR/nvidia-graphics-env.sh" "$PARAM_DIR/"
