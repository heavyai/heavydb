# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build config: debug + no-CUDA + gcc static. Source of truth for
# build-multi-static-debug in .github/workflows/pr-required-checks.yml.
# Build-only — heavydb compile check; no test variant pairs with this.
# No IMAGE_NAME: this config does not push an image in CI.

export CMAKE_FLAGS="-DENABLE_CUDA=OFF \
-DENABLE_RENDERING=OFF \
-DCMAKE_BUILD_TYPE=debug \
-DPREFER_STATIC_LIBS=ON \
-DENABLE_TESTS=OFF"
export MAKE_TARGETS="mapd_java_components heavydb"

export BASE_IMAGE="ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest"
