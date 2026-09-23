# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cmd_build_deps_image: build a local deps container image from scratch.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

cmd_build_deps_image() {
  local distro="ubuntu22.04"
  local cuda_version="12.9.2"
  local lib_type="static"
  local tag=""
  local nproc_arg=""

  for arg in "$@"; do
    case "$arg" in
      --help|-h)
        cat <<'EOF'
Usage: dev-tools/dev.sh build deps [options]

Builds a local deps container image using the same scripts as CI
(deps-image-builder.yml). This is a multi-hour operation.

Options:
  --distro=ubuntu22.04|rockylinux8   default: ubuntu22.04
  --cuda-version=<ver>               default: 12.9.2
  --lib-type=static|shared           default: static
  --tag=<tag>                        image tag; default: YYYYMMDD
  --nproc=<n>                        parallel jobs; default: min(nproc,24)
EOF
        return 0 ;;
      --distro=*)      distro="${arg#*=}" ;;
      --cuda-version=*) cuda_version="${arg#*=}" ;;
      --lib-type=*)    lib_type="${arg#*=}" ;;
      --tag=*)         tag="${arg#*=}" ;;
      --nproc=*)       nproc_arg="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  _validate_distro "$distro"

  [ -z "$tag" ] && tag="$(date -u +%Y%m%d)"

  # Derive per-distro values (mirrors deps-image-builder.yml matrix setup).
  local cuda_base deps_script param_dir output_image ubuntu_version
  case "$distro" in
    ubuntu22.04)
      ubuntu_version="22.04"
      cuda_base="nvcr.io/nvidia/cuda:${cuda_version}-devel-ubuntu${ubuntu_version}"
      deps_script="mapd-deps-ubuntu.sh"
      param_dir="$REPO_ROOT/docker/build/ubuntu_param"
      output_image="ghcr.io/${GHCR_OWNER}/${GHCR_REPO}/core-build-ubuntu${ubuntu_version}-${lib_type}-cuda${cuda_version}-$(uname -m)"
      ;;
    rockylinux8)
      cuda_base="nvcr.io/nvidia/cuda:${cuda_version}-devel-rockylinux8"
      deps_script="mapd-deps-rockylinux.sh"
      param_dir="$REPO_ROOT/docker/build/rockylinux8_param"
      output_image="ghcr.io/${GHCR_OWNER}/${GHCR_REPO}/core-build-rockylinux8-${lib_type}-cuda${cuda_version}-$(uname -m)"
      ;;
    *)
      echo "ERROR: unhandled distro '$distro' in cmd_build_deps_image" >&2
      exit 1 ;;
  esac

  # nproc: default to system core count, capped at 24.
  local nproc
  nproc=$(_resolve_nproc "$nproc_arg")

  echo "Building deps image: ${output_image}:${tag}" >&2
  echo "  distro=${distro}  cuda=${cuda_version}  lib_type=${lib_type}  nproc=${nproc}" >&2
  echo "  This typically takes 1-2 hours." >&2

  # Stage 1: build the deps tarball inside the CUDA base image.
  echo "--- Stage 1: building deps tarball (this is the long step) ---" >&2
  local update_opts="" tarball_output_dir
  [ "$distro" = "rockylinux8" ] && update_opts="--update-options=--nobest"
  tarball_output_dir=$(mktemp -d)
  docker run --rm \
    -v "$REPO_ROOT:/repo:ro" \
    -v "$REPO_ROOT/ThirdParty:/work/ThirdParty:ro" \
    -v "$tarball_output_dir:/tarball-output" \
    -e USER=root \
    "$cuda_base" \
    bash -c "
      echo '#!/bin/sh' > /usr/sbin/sudo
      echo 'exec \"\$@\"' >> /usr/sbin/sudo
      chmod +x /usr/sbin/sudo
      cp -a /repo/scripts /work/scripts
      cd /work/scripts
      SUFFIX='${tag}' ./${deps_script} \
        --${lib_type} \
        --update-packages \
        ${update_opts} \
        --savespace \
        --compress \
        --nproc=${nproc}
      mv /work/scripts/mapd-deps-*.tar.xz /tarball-output/
    "

  local tarball
  tarball=$(find "$tarball_output_dir" -maxdepth 1 -name 'mapd-deps-*.tar.xz' | head -1)
  if [ -z "$tarball" ]; then
    echo "ERROR: deps tarball not found in $tarball_output_dir after build" >&2
    rm -rf "$tarball_output_dir"
    exit 1
  fi

  # Stage 2: stage tarball + supporting scripts into the Docker build context.
  echo "--- Stage 2: preparing Docker build context ---" >&2
  "$SCRIPTS_DIR/ci/prepare_deps_context.sh" "$param_dir" "$tarball" --distro="$distro"

  # Stage 3: docker build the deps image.
  echo "--- Stage 3: building Docker image ---" >&2
  "$SCRIPTS_DIR/ci/build_deps_image.sh" \
    "$param_dir" "$cuda_base" "$output_image" "$tag" "$lib_type" \
    --distro="$distro"

  # Cleanup: remove staged files from the param dir and the tarball temp dir.
  rm -f "$param_dir/$(basename "$tarball")" \
        "$param_dir/mapd-deps-prebuilt.sh" \
        "$param_dir/common-functions.sh" \
        "$param_dir/nvidia-graphics-env.sh"
  rm -rf "$tarball_output_dir"

  echo "Done. Image: ${output_image}:${tag}" >&2
}
