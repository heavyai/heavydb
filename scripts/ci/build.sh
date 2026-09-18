#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Build the heavydb tree for a single cmake configuration.
#
# Invoked from .github/workflows/pr-required-checks.yml; also locally runnable
# from a checkout inside the deps container.
#
# Required env vars:
#   CMAKE_FLAGS   - flags passed to cmake (e.g. "-DPREFER_STATIC_LIBS=ON -DCMAKE_BUILD_TYPE=debug")
#   MAKE_TARGETS  - make targets to build, space-separated (e.g. "mapd_java_components ExecuteTest")
#
# Optional env vars:
#   BUILD_DIR     - build directory relative to repo root (default: build)
#   MAKE_JOBS     - parallelism for make (default: $(nproc))
#   MAPD_DEPS_SH  - mapd-deps env file to source (default: /usr/local/mapd-deps/mapd-deps.sh)
#
set -euo pipefail

: "${CMAKE_FLAGS:?CMAKE_FLAGS must be set}"
: "${MAKE_TARGETS:?MAKE_TARGETS must be set}"

BUILD_DIR="${BUILD_DIR:-build}"
MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# actions/checkout marks safe.directory under a temporary HOME that expires
# with its step. Re-apply so cmake/maven git plugins don't trip on dubious
# ownership when this script runs.
git config --global --add safe.directory "$REPO_ROOT"

# Force mvn to use the project's GCS-mirrored settings.xml. The repo's
# project-settings-extension can't bootstrap because it's hosted on Maven
# Central, which rate-limits unauthenticated requests with HTTP 429.
# Writing .mvn/maven.config short-circuits the extension.
echo "--settings=${REPO_ROOT}/java/.mvn/settings.xml" > "${REPO_ROOT}/java/.mvn/maven.config"

source "$MAPD_DEPS_SH"

mkdir -p "${REPO_ROOT}/${BUILD_DIR}"
cd "${REPO_ROOT}/${BUILD_DIR}"

# shellcheck disable=SC2086    # word splitting is intentional for CMAKE_FLAGS
cmake $CMAKE_FLAGS ..

# Build via `cmake --build` so this works whether the generator is Make or
# Ninja (TSAN switches to Ninja for its job-pool support). Run one
# --target at a time rather than passing multiple in one invocation: it
# matches the original `make A && make B` semantics and keeps memory
# pressure bounded (e.g. Java/Calcite link doesn't overlap with C++ test
# link).
for t in $MAKE_TARGETS; do
  cmake --build . --parallel "$MAKE_JOBS" --target "$t"
done
