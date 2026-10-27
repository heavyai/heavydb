#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Report binary size and run time for tests, optionally against their
# pre-port versions, as a markdown table for PR descriptions.
#
# Usage (inside the deps container, after `dev.sh build heavydb`):
#   scripts/ci/test_metrics.sh [--baseline-ref <git-ref>] [--runs N] <TestName>...
#
# New: each <TestName> is built and run through `ctest -R ^<TestName>$`, so
#      the time includes starting/stopping the server for thin-client tests.
# Old: with --baseline-ref, Tests/<TestName>.cpp is taken from that ref and
#      built as Old<TestName>, linked with THRIFT_HANDLER_TEST_LIBRARIES, and
#      run directly from build/Tests (in-process, no server). The extra
#      targets come from a CMake file under $BUILD_DIR/baseline injected with
#      CMAKE_PROJECT_INCLUDE; nothing in the repo is modified.
#
# Reported times are the median of N runs (default 3): "wall" is the whole
# process/ctest run, "gtest" is the sum of gtest's own "(N ms total)".
#
# Optional env vars:
#   BUILD_DIR    - build tree (default: /workspace/build if it exists, else build)
#   NPROC        - parallel make jobs (default: 8)
#   MAPD_DEPS_SH - mapd-deps env file (default: /usr/local/mapd-deps/mapd-deps.sh)
#   OUT          - also write the table to this file (default: $BUILD_DIR/test_metrics.md)
#
set -uo pipefail

baseline_ref=""
runs=3
tests=()
while [ $# -gt 0 ]; do
  case "$1" in
    --baseline-ref) baseline_ref="$2"; shift 2 ;;
    --runs) runs="$2"; shift 2 ;;
    -h|--help) sed -n '5,28p' "$0"; exit 0 ;;
    *) tests+=("$1"); shift ;;
  esac
done
if [ ${#tests[@]} -eq 0 ]; then
  echo "ERROR: no tests given (see --help)" >&2
  exit 2
fi

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then BUILD_DIR=/workspace/build; else BUILD_DIR=build; fi
fi
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"
NPROC="${NPROC:-8}"
source "$MAPD_DEPS_SH"

BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
SRC_DIR="$(cd "$BUILD_DIR/.." && pwd)"
OUT="${OUT:-$BUILD_DIR/test_metrics.md}"
BASE_DIR="$BUILD_DIR/baseline"

mkdir -p "$BASE_DIR" "$BUILD_DIR/logs"
cd "$BUILD_DIR"

# ---- Build -----------------------------------------------------------------

build_targets=("${tests[@]}")
cmake_args=()
if [ -n "$baseline_ref" ]; then
  # Imported targets (absl::str_format) are directory-scoped, so find them again here.
  cat > "$BASE_DIR/baseline.cmake" <<EOF
find_package(absl REQUIRED)
get_directory_property(_old_libs DIRECTORY \${CMAKE_SOURCE_DIR}/Tests DEFINITION THRIFT_HANDLER_TEST_LIBRARIES)
EOF
  for t in "${tests[@]}"; do
    git -C "$SRC_DIR" show "$baseline_ref:Tests/$t.cpp" > "$BASE_DIR/Old$t.cpp" || exit 1
    # Created at top level, so repeat what Tests/CMakeLists.txt sets for its targets.
    cat >> "$BASE_DIR/baseline.cmake" <<EOF
add_executable(Old$t $BASE_DIR/Old$t.cpp)
set_target_properties(Old$t PROPERTIES RUNTIME_OUTPUT_DIRECTORY \${CMAKE_BINARY_DIR}/Tests)
target_include_directories(Old$t PRIVATE \${CMAKE_SOURCE_DIR}/Tests \${CMAKE_BINARY_DIR}/Tests)
target_compile_definitions(Old$t PRIVATE "BASE_PATH=\\"./tmp\\"")
target_link_libraries(Old$t \${_old_libs})
EOF
    build_targets+=("Old$t")
  done
  # Deferred to the end of the top-level CMakeLists, after Tests/ is processed.
  # CMAKE_PROJECT_INCLUDE runs for every project() call (e.g. third-party
  # subprojects too), so only act on the top-level one.
  cat > "$BASE_DIR/project_include.cmake" <<EOF
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  cmake_language(DEFER CALL include "$BASE_DIR/baseline.cmake")
endif()
EOF
  cmake_args=(-DCMAKE_PROJECT_INCLUDE="$BASE_DIR/project_include.cmake")
fi

# Don't leave the injected include in the cache for later builds, even on failure.
[ -n "$baseline_ref" ] && trap 'cmake -UCMAKE_PROJECT_INCLUDE . > /dev/null 2>&1' EXIT
cmake -DENABLE_TESTS=ON "${cmake_args[@]}" . > "$BUILD_DIR/logs/test_metrics_cmake.log" 2>&1 || {
  tail -20 "$BUILD_DIR/logs/test_metrics_cmake.log"; exit 1; }
make -j "$NPROC" "${build_targets[@]}" || exit 1

# ---- Measure ---------------------------------------------------------------

now() { date +%s.%N; }
median() { sort -n | awk '{a[NR]=$1} END {print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }
gtest_ms() { grep -E "tests? from .* ran\." "$1" | grep -oE '\([0-9]+ ms total\)' | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}'; }
mb() { awk -v b="$1" 'BEGIN {printf "%.0f", b/1048576}'; }

# run_timed <log> <cmd...>: prints "<wall_seconds> <gtest_ms> <rc>"
run_timed() {
  local log="$1"; shift
  local s e rc
  s=$(now); "$@" > "$log" 2>&1; rc=$?; e=$(now)
  echo "$(awk -v s="$s" -v e="$e" 'BEGIN {printf "%.2f", e-s}') $(gtest_ms "$log") $rc"
}

{
  echo "| Test | Old size (MB) | New size (MB) | Old wall (s) | New wall (s) | Old gtest (s) | New gtest (s) |"
  echo "|---|---|---|---|---|---|---|"
} > "$OUT"

for t in "${tests[@]}"; do
  old_size="-"; old_wall="-"; old_gtest="-"
  if [ -n "$baseline_ref" ]; then
    old_size=$(mb "$(stat -c %s "Tests/Old$t")")
    walls=(); gts=()
    for _ in $(seq "$runs"); do
      read -r w g rc < <(cd Tests && run_timed "$BUILD_DIR/baseline/old_$t.log" "./Old$t")
      [ "$rc" -ne 0 ] && echo "WARNING: Old$t exited $rc (see baseline/old_$t.log)" >&2
      walls+=("$w"); gts+=("$g")
    done
    old_wall=$(printf '%s\n' "${walls[@]}" | median)
    old_gtest=$(printf '%s\n' "${gts[@]}" | median | awk '{printf "%.1f", $1/1000}')
  fi
  new_size=$(mb "$(stat -c %s "Tests/$t")")
  walls=(); gts=()
  for _ in $(seq "$runs"); do
    read -r w g rc < <(run_timed "$BUILD_DIR/baseline/new_$t.log" ctest -V -R "^$t\$" --timeout 1800)
    [ "$rc" -ne 0 ] && echo "WARNING: ctest $t exited $rc (see baseline/new_$t.log)" >&2
    walls+=("$w"); gts+=("$g")
  done
  new_wall=$(printf '%s\n' "${walls[@]}" | median)
  new_gtest=$(printf '%s\n' "${gts[@]}" | median | awk '{printf "%.1f", $1/1000}')
  echo "| $t | $old_size | $new_size | $old_wall | $new_wall | $old_gtest | $new_gtest |" >> "$OUT"
done

cat "$OUT"
