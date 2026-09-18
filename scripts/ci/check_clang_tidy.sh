#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Build with a compile_commands.json and run clang-tidy (-format -fix) over the
# first-party sources, then fail if it produced any diff. Mirrors
# check_clang_format.sh but for clang-tidy. Run inside the deps container.
#
# Optional env vars:
#   BUILD_DIR    - build directory (default: build)
#   MAKE_JOBS    - ninja parallelism (default: nproc)
#   MAPD_DEPS_SH - mapd-deps env file (default: /usr/local/mapd-deps/mapd-deps.sh)
#
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
git config --global --add safe.directory "$REPO_ROOT"
echo "--settings=${REPO_ROOT}/java/.mvn/settings.xml" > "${REPO_ROOT}/java/.mvn/maven.config"

source "$MAPD_DEPS_SH"

mkdir -p "${REPO_ROOT}/${BUILD_DIR}"
cd "${REPO_ROOT}/${BUILD_DIR}"

# Disable the Immerse/HeavyIQ component downloads: building all targets would
# otherwise fetch the frontend/web-server/heavyiq assets (unavailable in CI);
# clang-tidy only needs the C++ compile_commands.json.
cmake -DENABLE_TESTS=OFF -DENABLE_RENDERING=ON -DENABLE_ONLY_ONE_ARCH=ON \
  -DMAPD_IMMERSE_DOWNLOAD=OFF -DHEAVYIQ_DOWNLOAD=OFF \
  -DUSE_ALTERNATE_LINKER=mold -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -GNinja ..
# clang-tidy only needs the translation units compiled (and any generated
# headers produced) — it reads compile_commands.json, it does not need linked
# binaries. `-k 0` keeps building all compilable targets even if some final
# links fail, and we ignore the overall exit so a link-only failure doesn't
# abort the tidy run. NOTE: building "all" currently hits unresolved GDAL
# driver symbols (libkml/HDF5/netCDF) at link in the deps image — harmless for
# tidy, but worth a separate look.
ninja -k 0 -j"$MAKE_JOBS" || echo "ninja reported build/link errors; continuing to clang-tidy (compiles are what it needs)"

mkdir -p clang-tidy
# Exclude generated build/ and vendored ThirdParty/ translation units.
echo 'map(select(.file | test(".*/(build|ThirdParty)/.*") | not))' > jq.filter
jq -f jq.filter compile_commands.json > clang-tidy/compile_commands.json

pushd clang-tidy >/dev/null
python3 ../../ThirdParty/clang/run-clang-tidy.py -format -fix || true
popd >/dev/null

# Never count ThirdParty churn as a clang-tidy finding.
git checkout -- ../ThirdParty

git -C "$REPO_ROOT" diff
git -C "$REPO_ROOT" diff --exit-code > "clang-tidy-diff-${GITHUB_SHA:-local}.txt"
