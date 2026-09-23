#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs `docker build` to produce a deps container image from a staged build
# context (populated by prepare_deps_context.sh).
#
# Usage: build_deps_image.sh <param_dir> <base_image> <output_image> <tag> <lib_type> [--distro=rocky]
#
# Extracted from the "Build Docker image" step in
# .github/workflows/deps-image-builder.yml so the same logic runs locally
# (via dev.sh build-deps-image) and in CI.
#
# PREBUILT_CUSTOM_ARG=--enable creates /etc/profile.d/xx-mapd-deps.sh inside
# the resulting image so cmake and other deps tools are on PATH.

set -euo pipefail

PARAM_DIR="${1:-}"
BASE_IMAGE="${2:-}"
OUTPUT_IMAGE="${3:-}"
TAG="${4:-}"
LIB_TYPE="${5:-}"
DISTRO=ubuntu

for arg in "${@:6}"; do
  case "$arg" in
    --distro=*) DISTRO="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if [ -z "$PARAM_DIR" ] || [ -z "$BASE_IMAGE" ] || [ -z "$OUTPUT_IMAGE" ] || [ -z "$TAG" ] || [ -z "$LIB_TYPE" ]; then
  echo "Usage: $(basename "$0") <param_dir> <base_image> <output_image> <tag> <lib_type> [--distro=rocky]" >&2
  exit 1
fi

TARBALL=$(find "$PARAM_DIR" -maxdepth 1 -name 'mapd-deps-*.tar.xz' | head -1)
if [ -z "$TARBALL" ]; then
  echo "ERROR: no mapd-deps-*.tar.xz found in $PARAM_DIR (run prepare_deps_context.sh first)" >&2
  exit 1
fi
TARBALL_NAME=$(basename "$TARBALL")

# Rocky's full dnf upgrade fails on the base's pinned CUDA packages without --nobest.
PREBUILT_OPTS=""
if [ "$DISTRO" = "rocky" ] || [ "$DISTRO" = "rockylinux8" ]; then
  PREBUILT_OPTS="--update-options=--nobest"
fi

docker build \
  --build-arg BASE_CUDA_IMAGE="$BASE_IMAGE" \
  --build-arg DEPS_TARBALL="$TARBALL_NAME" \
  --build-arg "PREBUILT_CUSTOM_ARG=--enable --${LIB_TYPE} --update-packages ${PREBUILT_OPTS} --custom=${TAG}" \
  -t "${OUTPUT_IMAGE}:${TAG}" \
  "$PARAM_DIR"
