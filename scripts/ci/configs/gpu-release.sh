# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: release + CUDA, static, tests enabled. Source of truth for
# build-gpu-release in .github/workflows/weekly.yml. One build shared by the
# arrow-ipc, ctas, read-only and extended-geo weekly integration tests.

export CMAKE_FLAGS="-DENABLE_CUDA=ON \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=release \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=ON"
export MAKE_TARGETS="mapd_java_components heavydb initheavy heavysql ArrowIpcIntegrationTest CtasIntegrationTest GeospatialTest"

export IMAGE_NAME="pr-build/gpu-release"
export BASE_IMAGE="ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config scripts"
