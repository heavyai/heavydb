# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: TSAN + RelWithDebInfo, CPU-only static. Source of truth
# for build-tsan in .github/workflows/pr-required-checks.yml. Pairs with
# the tsan test variant.
#
# Locally, append `-DHEAVYAI_PARTITION=N` (N in 1..4) to CMAKE_FLAGS to
# match a specific CI partition; omit it for an unpartitioned build.

export CMAKE_FLAGS="-GNinja \
-DENABLE_TSAN=ON \
-DENABLE_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELWITHDEBINFO \
-DPREFER_STATIC_LIBS=ON \
-DHAVE_STD_REGEX=ON \
-DCMAKE_JOB_POOLS=link=4 \
-DCMAKE_JOB_POOL_LINK=link"
export MAKE_TARGETS="mapd_java_components sanity_tests_build_only"

export IMAGE_NAME="pr-build/tsan"
export BASE_IMAGE="ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.2.2-x86_64:rc.v9.0.0"
export INCLUDE_PATHS="build Tests config scripts"
export STRIP=debug
