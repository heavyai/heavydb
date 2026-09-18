# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: release + CUDA + rendering (Ninja, mold linker). Source
# of truth for build-multi-render in
# .github/workflows/pr-required-checks.yml. Pairs with the multi-render
# test variant.

export CMAKE_FLAGS="-GNinja \
-DENABLE_CUDA=ON \
-DENABLE_RENDERING=ON \
-DCMAKE_BUILD_TYPE=release \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_RENDER_TESTS=ON \
-DENABLE_TESTS=OFF \
-DENABLE_ONLY_ONE_ARCH=ON \
-DUSE_ALTERNATE_LINKER=mold"
export MAKE_TARGETS="mapd_java_components render_sanity_tests_build_only"

export IMAGE_NAME="pr-build/multi-render"
export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.2.2-x86_64:rc.v9.0.0"
export INCLUDE_PATHS="build Tests config scripts"
