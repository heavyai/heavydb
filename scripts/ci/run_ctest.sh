#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run ctest from a build dir with an optional label/regex filter and a
# canonical XML output destination.
#
# Invoked from .github/workflows/pr-required-checks.yml; also locally
# runnable.
#
# Optional env vars:
#   BUILD_DIR     - working directory for ctest (default: /workspace/build
#                   if it exists, else build)
#   LABEL_REGEX   - passed as ctest --label-regex (default: unset, runs all)
#   TESTS_REGEX   - passed as ctest --tests-regex (default: unset)
#   RESULTS_DIR   - directory ctest will write per-test xml output into
#                   (default: $BUILD_DIR)
#   MAPD_DEPS_SH  - mapd-deps env file to source (default:
#                   /usr/local/mapd-deps/mapd-deps.sh)
#   AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - exported to
#                   the test environment for tests that need S3.
#   TSAN_OPTIONS - exported to the test environment for tsan runs.
#   NO_ASLR       - if "true", wraps ctest with `setarch -R` so the
#                   ADDR_NO_RANDOMIZE personality is set for ctest and
#                   all subprocesses (the test binaries). TSAN runtime
#                   trips on randomized address space ("unexpected memory
#                   mapping" FATAL) — this avoids needing to write to
#                   /proc/sys/kernel/randomize_va_space, which is read-only
#                   in unprivileged containers.
#   CTEST_TIMEOUT_SECS - per-test timeout in seconds (default: 1200 / 20m).
#                   A hung test would otherwise stall ctest indefinitely;
#                   this kills the test and lets ctest move on.
#
set -euo pipefail

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then
    BUILD_DIR=/workspace/build
  else
    BUILD_DIR=build
  fi
fi
RESULTS_DIR="${RESULTS_DIR:-$BUILD_DIR}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

source "$MAPD_DEPS_SH"

cd "$BUILD_DIR"

ctest_args=( --verbose --output-on-failure )
# Fail when a filter matches nothing — without this, ctest exits 0 on a
# zero-match (the default in cmake < 3.26), so a missing CTestTestfile or
# wrong filter would appear green silently.
ctest_args+=( --no-tests=error )
# Per-test timeout: prevents a single hung test from stalling the whole
# ctest run for the rest of the job's runtime.
ctest_args+=( --timeout "${CTEST_TIMEOUT_SECS:-1200}" )
if [ -n "${LABEL_REGEX:-}" ]; then
  ctest_args+=( --label-regex "$LABEL_REGEX" )
fi
if [ -n "${TESTS_REGEX:-}" ]; then
  ctest_args+=( --tests-regex "$TESTS_REGEX" )
fi

if [ "${NO_ASLR:-false}" = "true" ]; then
  exec setarch "$(uname -m)" -R ctest "${ctest_args[@]}"
fi
ctest "${ctest_args[@]}"
