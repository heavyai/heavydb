#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run a single gtest-style test binary from a heavydb build tree, with
# canonical --gtest_output XML emission.
#
# Invoked from .github/workflows/pr-required-checks.yml; also locally runnable.
#
# Required env vars:
#   BINARY        - path to the test binary relative to BUILD_DIR (e.g. Tests/ExecuteTest)
#   RESULTS_FILE  - filename for the gtest XML output, written into BUILD_DIR
#
# Optional env vars:
#   BUILD_DIR    - working directory for the test (default: /workspace/build if it
#                  exists, else build)
#   EXTRA_ARGS   - additional flags to pass to the test binary (default: empty)
#   MAPD_DEPS_SH - mapd-deps env file to source (default: /usr/local/mapd-deps/mapd-deps.sh)
#   OPEN_FILES_LIMIT - soft open-file limit to request before running the test
#                      binary (default: unchanged)
#   NO_ASLR      - if "true", wrap the binary with `setarch -R` so the
#                  ADDR_NO_RANDOMIZE personality is set. Sanitizer-instrumented
#                  binaries (TSAN/ASAN) trip on randomized address space; mirror
#                  the run_ctest.sh handling so the columnar-asan/columnar-tsan
#                  gtest jobs can run ExecuteTest directly. Requires the container
#                  to be launched with --security-opt seccomp=unconfined.
#
set -euo pipefail

: "${BINARY:?BINARY must be set}"
: "${RESULTS_FILE:?RESULTS_FILE must be set}"

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then
    BUILD_DIR=/workspace/build
  else
    BUILD_DIR=build
  fi
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"
OPEN_FILES_LIMIT="${OPEN_FILES_LIMIT:-}"

raise_open_files_limit() {
  local open_files_limit="$1"
  if [ -z "$open_files_limit" ]; then
    return
  fi

  case "$open_files_limit" in
    *[!0-9]*)
      echo "WARNING: ignoring non-numeric OPEN_FILES_LIMIT=$open_files_limit" >&2
      ;;
    *)
      local current_open_files_limit
      local hard_open_files_limit
      local requested_open_files_limit
      current_open_files_limit="$(ulimit -Sn)"
      hard_open_files_limit="$(ulimit -Hn)"
      requested_open_files_limit="$open_files_limit"

      if [ "$current_open_files_limit" != "unlimited" ]; then
        if [ "$hard_open_files_limit" != "unlimited" ] &&
           [ "$requested_open_files_limit" -gt "$hard_open_files_limit" ]; then
          requested_open_files_limit="$hard_open_files_limit"
        fi

        if [ "$requested_open_files_limit" -gt "$current_open_files_limit" ]; then
          if ulimit -Sn "$requested_open_files_limit"; then
            echo "Raised open-file limit from $current_open_files_limit to $(ulimit -Sn)"
          else
            echo "WARNING: failed to raise open-file limit from $current_open_files_limit to $requested_open_files_limit" >&2
          fi
        fi
      fi
      ;;
  esac
}

source "$MAPD_DEPS_SH"

raise_open_files_limit "$OPEN_FILES_LIMIT"

ABS_BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
# ExecuteTest and friends bake relative paths into their fixtures (e.g. COPY
# from '../../Tests/Import/datafiles/...'), which only resolve correctly
# when CWD is the binary's own directory (build/Tests/). Running from
# BUILD_DIR makes those silently fail — COPY into the table errors, the
# table is left empty, and queries return 0 rows. The fatal symptom we hit:
# Select.LimitAndOffsetFromSubquery's run_simple_agg CHECK-fails on empty
# results, but most other tests use loose assertions and silently produce
# wrong results. Always cd into the binary's directory; emit XML to the
# absolute BUILD_DIR/RESULTS_FILE so the artifact path is unchanged.
cd "$ABS_BUILD_DIR/$(dirname "$BINARY")"

bin="./$(basename "$BINARY")"
aslr_wrap=()
if [ "${NO_ASLR:-false}" = "true" ]; then
  aslr_wrap=( setarch "$(uname -m)" -R )
fi

# shellcheck disable=SC2086    # word splitting is intentional for EXTRA_ARGS
"${aslr_wrap[@]}" "$bin" $EXTRA_ARGS --gtest_output="xml:${ABS_BUILD_DIR}/${RESULTS_FILE}"
