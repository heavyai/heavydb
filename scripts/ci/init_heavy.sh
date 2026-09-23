#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Initialize a heavydb storage tree for tests via `initheavy`, with retries.
#
# Invoked from .github/workflows/pr-required-checks.yml; also locally runnable
# from a build tree.
#
# Optional env vars:
#   BUILD_DIR  - directory containing bin/initheavy (default: build, or /workspace/build
#                if it exists — for use inside a pr-build image)
#   TMP_DIR    - storage path passed to initheavy -f (relative to BUILD_DIR; default: tmp)
#   RETRIES    - number of attempts (default: 3)
#   SLEEP_SECS    - sleep between attempts (default: 5)
#   MAPD_DEPS_SH  - mapd-deps env file to source (default: /usr/local/mapd-deps/mapd-deps.sh)
#   NO_ASLR    - if "true", wraps initheavy with `setarch -R` so the
#                ADDR_NO_RANDOMIZE personality is set. TSAN-instrumented
#                initheavy trips on randomized address space ("unexpected
#                memory mapping" FATAL) the same way the test binaries
#                do; mirror the run_ctest.sh setarch handling here so
#                tsan jobs can initialise their catalog. Requires the
#                container to be launched with --security-opt
#                seccomp=unconfined, since Docker's default profile
#                blocks personality(ADDR_NO_RANDOMIZE).
#
set -euo pipefail

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then
    BUILD_DIR=/workspace/build
  else
    BUILD_DIR=build
  fi
fi
TMP_DIR="${TMP_DIR:-tmp}"
RETRIES="${RETRIES:-3}"
SLEEP_SECS="${SLEEP_SECS:-5}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

source "$MAPD_DEPS_SH"

cd "$BUILD_DIR"
mkdir -p "$TMP_DIR"

# Build the initheavy invocation; under TSAN we need ADDR_NO_RANDOMIZE.
INITHEAVY_CMD=( bin/initheavy -f "$TMP_DIR" )
if [ "${NO_ASLR:-false}" = "true" ]; then
  INITHEAVY_CMD=( setarch "$(uname -m)" -R "${INITHEAVY_CMD[@]}" )
fi

attempt=1
while [ "$attempt" -le "$RETRIES" ]; do
  if "${INITHEAVY_CMD[@]}"; then
    echo "initheavy succeeded on attempt $attempt"
    # Some tests are spawned by ctest with WORKING_DIRECTORY=Tests/ and
    # use a BASE_PATH baked at compile time as "./tmp". From their CWD
    # that's Tests/tmp, not the build/tmp we just initialized. Symlink
    # Tests/tmp -> ../tmp so both resolve to the same initialized state.
    # (For tests invoked directly from build/ with CWD=build/, this
    # symlink is invisible.)
    if [ -d Tests ]; then
      rm -rf "Tests/$TMP_DIR"
      ln -sf "../$TMP_DIR" "Tests/$TMP_DIR"
      echo "Created Tests/$TMP_DIR -> $(readlink "Tests/$TMP_DIR") (target dir: $(ls -ld "$TMP_DIR" | awk '{print $1}'))"
      # Sanity: the test binary's BASE_PATH=./tmp from CWD=build/Tests
      # must resolve to a directory containing the catalog DB.
      ls "Tests/$TMP_DIR/mapd_catalogs/" >/dev/null 2>&1 \
        && echo "Tests/$TMP_DIR/mapd_catalogs/ accessible via symlink" \
        || echo "WARNING: Tests/$TMP_DIR/mapd_catalogs/ NOT accessible via symlink" >&2
    fi
    exit 0
  fi
  echo "initheavy attempt $attempt failed; retrying after ${SLEEP_SECS}s" >&2
  sleep "$SLEEP_SECS"
  attempt=$((attempt + 1))
done

echo "ERROR: initheavy failed after $RETRIES attempts" >&2
exit 1
