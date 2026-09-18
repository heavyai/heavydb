# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: RelWithDebInfo + CPU-only static, for valgrind memcheck.
# Source of truth for build-valgrind in .github/workflows/weekly.yml.
# Pairs with the valgrind test variant.

export CMAKE_FLAGS="-DENABLE_CUDA=OFF \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=RELWITHDEBINFO \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=ON"
export MAKE_TARGETS="mapd_java_components ExecuteTest"

export IMAGE_NAME="pr-build/valgrind"
export BASE_IMAGE="ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config scripts"
