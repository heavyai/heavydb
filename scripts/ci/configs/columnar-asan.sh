# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: ExecuteTest under ASAN (unpartitioned). Source of truth for
# build-columnar-asan in .github/workflows/nightly.yml. Pairs with the
# columnar-asan test variant (ExecuteTest --enable-columnar-output).

export CMAKE_FLAGS="-GNinja \
-DENABLE_ASAN=ON \
-DENABLE_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELWITHDEBINFO \
-DPREFER_STATIC_LIBS=ON"
export MAKE_TARGETS="mapd_java_components ExecuteTest"

export IMAGE_NAME="pr-build/columnar-asan"
export BASE_IMAGE="ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests config QueryEngine Shared Geospatial Logger scripts"
export STRIP=debug

# Build-time initheavy (init_test_dir) under ASAN trips a use-after-poison on
# heavydb's intentional memory poisoning and the report deadlocks (hangs the
# build); suppress user poisoning, as the asan test jobs do.
export ASAN_OPTIONS="allow_user_poisoning=false:detect_leaks=0:halt_on_error=0"
