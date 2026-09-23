# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: release + CPU-only static, tests enabled. Source of truth for
# build-minio in .github/workflows/weekly.yml. Pairs with the minio test
# variants (ImportExportTest / ForeignTableDmlTest against a minio service).

export CMAKE_FLAGS="-DENABLE_CUDA=OFF \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=release \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=ON"
export MAKE_TARGETS="mapd_java_components ImportExportTest ForeignTableDmlTest"

export IMAGE_NAME="pr-build/minio"
export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config scripts"
