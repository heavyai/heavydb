# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: debug + static. Source of truth for build-debug-static
# in .github/workflows/pr-required-checks.yml. Pairs with columnar /
# sharding test variants.

export CMAKE_FLAGS="-DPREFER_STATIC_LIBS=ON -DCMAKE_BUILD_TYPE=debug"
export MAKE_TARGETS="mapd_java_components ExecuteTest"

export IMAGE_NAME="pr-build/debug-static"
export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
export INCLUDE_PATHS="build Tests scripts"
export STRIP=debug
