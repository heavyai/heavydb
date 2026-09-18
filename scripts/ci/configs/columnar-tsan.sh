# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: ExecuteTest under TSAN (unpartitioned). Source of truth for
# build-columnar-tsan in .github/workflows/nightly.yml. Pairs with the
# columnar-tsan test variant (ExecuteTest --enable-columnar-output).

export CMAKE_FLAGS="-GNinja \
-DENABLE_TSAN=ON \
-DENABLE_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELWITHDEBINFO \
-DPREFER_STATIC_LIBS=ON \
-DHAVE_STD_REGEX=ON"
export MAKE_TARGETS="mapd_java_components ExecuteTest"

export IMAGE_NAME="pr-build/columnar-tsan"
export BASE_IMAGE="ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config QueryEngine Shared Geospatial Logger scripts"
export STRIP=debug
