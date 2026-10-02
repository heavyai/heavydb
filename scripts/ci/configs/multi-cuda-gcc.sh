# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: release + CUDA, gcc, static. Source of truth for
# build-multi-cuda-gcc in .github/workflows/pr-required-checks.yml.
# Pairs with the multi-cuda-gcc test variant.
#
# Locally, append `-DHEAVYAI_PARTITION=N` (N in 1..4) to CMAKE_FLAGS to
# match a specific CI partition; omit it for an unpartitioned build.

export CMAKE_FLAGS="-DENABLE_CUDA=ON \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=release \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=ON"
export MAKE_TARGETS="mapd_java_components heavydb sanity_tests_build_only"

export IMAGE_NAME="pr-build/multi-cuda-gcc"
export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config scripts"
