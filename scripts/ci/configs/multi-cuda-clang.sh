# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: release + CUDA, clang, static. Source of truth for
# build-multi-cuda-clang in .github/workflows/pr-required-checks.yml.
# Build-only — no test variant pairs with this (heavydb compile check).
# No IMAGE_NAME: this config does not push an image in CI.

export CC=clang
export CXX=clang++

export CMAKE_FLAGS="-GNinja \
-DENABLE_CUDA=ON \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=release \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=OFF"
export MAKE_TARGETS="mapd_java_components heavydb"

export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
