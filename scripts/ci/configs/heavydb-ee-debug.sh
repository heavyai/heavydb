# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: full product build, rendering + CUDA, RelWithDebInfo. Source
# of truth for build-heavydb-ee-debug in .github/workflows/nightly.yml.
# Build-only (sanity_tests_build_only): compiles tests but does not run them
# (the Jenkins heavydb_ee_debug used sanity_tests=build_only). No IMAGE_NAME:
# this config does not push an image.

# Immerse/HeavyIQ downloads OFF: building the product with rendering would
# otherwise fetch the frontend/web-server/heavyiq assets (unavailable in CI);
# this is a compile check, not a packaging build.
# Ninja + a link JOB_POOL + mold: relwithdebinfo links carry heavy debug info;
# building the full product + all test binaries at -j$(nproc) OOM-killed the
# cpu32 runner (exit 137). Cap concurrent links to 2 and use the
# memory-frugal mold linker.
export CMAKE_FLAGS="-GNinja \
-DENABLE_RENDERING=ON \
-DENABLE_RENDER_TESTS=ON \
-DENABLE_CUDA=ON \
-DCMAKE_BUILD_TYPE=relwithdebinfo \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=ON \
-DMAPD_IMMERSE_DOWNLOAD=OFF \
-DHEAVYIQ_DOWNLOAD=OFF \
-DUSE_ALTERNATE_LINKER=mold \
-DCMAKE_JOB_POOLS=link=2 \
-DCMAKE_JOB_POOL_LINK=link"
# Build the product + render tests only. Building the full sanity_tests
# (~85 binaries) in relwithdebinfo overflows the cpu32 runner's disk (ENOSPC);
# the partitioned release/tsan/asan jobs already cover sanity-test compilation.
# This job's unique value is the rendering+CUDA+relwithdebinfo product build.
export MAKE_TARGETS="mapd_java_components heavydb render_sanity_tests_build_only"

export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
