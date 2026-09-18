# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: TSAN + RelWithDebInfo, CPU-only static, FOLLY off. Source of
# truth for build-disk-cache-tsan in .github/workflows/weekly.yml. Pairs with
# the disk-cache-tsan test variants (ExecuteTest/UpdelStorageTest/CtasUpdateTest
# run with --use-disk-cache).

export CMAKE_FLAGS="-GNinja \
-DENABLE_TSAN=ON \
-DENABLE_FOLLY=OFF \
-DENABLE_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELWITHDEBINFO \
-DPREFER_STATIC_LIBS=ON \
-DHAVE_STD_REGEX=ON \
-DCMAKE_JOB_POOLS=link=4 \
-DCMAKE_JOB_POOL_LINK=link"
export MAKE_TARGETS="mapd_java_components UpdelStorageTest ExecuteTest CtasUpdateTest"

export IMAGE_NAME="pr-build/disk-cache-tsan"
export BASE_IMAGE="ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config QueryEngine Shared Geospatial Logger scripts"
export STRIP=debug
