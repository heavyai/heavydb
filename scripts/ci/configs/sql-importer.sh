# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: release + CPU-only static, no tests. Source of truth for
# build-sql-importer in .github/workflows/nightly.yml. Builds the Java
# utility jars + server; pairs with the sql-importer test variant (runs
# com.mapd.utility.SQLImporter against a locally-started server).

export CMAKE_FLAGS="-DENABLE_CUDA=OFF \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=release \
-DPREFER_STATIC_LIBS=ON \
-DMAPD_IMMERSE_DOWNLOAD=OFF \
-DENABLE_TESTS=OFF"
export MAKE_TARGETS="mapd_java_components heavydb initheavy"

export IMAGE_NAME="pr-build/sql-importer"
export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config scripts"
