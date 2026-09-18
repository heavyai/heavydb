#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Start a heavydb server, run a test binary or arbitrary client against it,
# then always stop the server. For the integration-style tests that need a
# live server (arrow-ipc, ctas, read-only, sql-importer) and the Java
# concurrency tests.
#
# Invoked from the nightly/weekly/concurrency workflows; also runnable via
# `dev-tools/dev.sh test <variant>` for a test config with RUNNER=server.
#
# Provide exactly one of:
#   BINARY      - gtest binary relative to BUILD_DIR (e.g. Tests/ArrowIpcIntegrationTest).
#                 Run from the binary's own directory (fixtures use relative paths);
#                 if RESULTS_FILE is set, --gtest_output is emitted to BUILD_DIR.
#   CLIENT_CMD  - arbitrary shell command run from BUILD_DIR against the server
#                 (e.g. a heavysql pipeline or a `java -cp ... com.mapd.tests.X` run).
#
# Optional env vars:
#   BUILD_DIR    - build tree (default: /workspace/build if it exists, else build)
#   DATA         - heavydb data dir relative to BUILD_DIR (default: tmp)
#   SERVER_CMD   - full command to start the server (default:
#                  "bin/heavydb --data $DATA $SERVER_FLAGS"). Override e.g. for
#                  read-only: "../startheavy --read-only".
#   SERVER_FLAGS - extra flags appended to the default heavydb command
#   SERVER_WAIT  - seconds to wait for the server to come up (default: 60)
#   EXTRA_ARGS   - extra args for BINARY
#   RESULTS_FILE - gtest XML output filename (BINARY mode), written into BUILD_DIR
#   MAPD_DEPS_SH - mapd-deps env file (default: /usr/local/mapd-deps/mapd-deps.sh)
#   NO_ASLR      - if "true", launch the server under `setarch -R`
#                  (ADDR_NO_RANDOMIZE). A TSAN-instrumented heavydb FATALs at
#                  startup on randomized address space ("unexpected memory
#                  mapping"); mirror init_heavy.sh/run_ctest.sh. Requires the
#                  container launched with --security-opt seccomp=unconfined.
#
set -euo pipefail

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then BUILD_DIR=/workspace/build; else BUILD_DIR=build; fi
fi
DATA="${DATA:-tmp}"
SERVER_FLAGS="${SERVER_FLAGS:-}"
SERVER_WAIT="${SERVER_WAIT:-60}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

if [ -z "${BINARY:-}" ] && [ -z "${CLIENT_CMD:-}" ]; then
  echo "ERROR: set either BINARY or CLIENT_CMD" >&2
  exit 2
fi

source "$MAPD_DEPS_SH"

ABS_BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
cd "$ABS_BUILD_DIR"

# shellcheck disable=SC2086    # intentional word-splitting for SERVER_FLAGS
SERVER_CMD="${SERVER_CMD:-bin/heavydb --data $DATA $SERVER_FLAGS}"

if [ "${NO_ASLR:-false}" = "true" ]; then
  # TSAN-instrumented server FATALs on randomized address space; pin
  # ADDR_NO_RANDOMIZE for it (and its children) via setarch -R.
  SERVER_CMD="setarch $(uname -m) -R $SERVER_CMD"
fi

echo "Starting server: $SERVER_CMD"
# eval (not bare word-splitting) so quoted flags survive — e.g. heavydb's
# --allowed-import-paths='[".."]' JSON value. Inputs are CI-controlled.
eval "$SERVER_CMD &"
SERVER_PID=$!
# Always stop the server, even if the test fails or the script is interrupted.
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep "$SERVER_WAIT"

rc=0
if [ -n "${BINARY:-}" ]; then
  cd "$ABS_BUILD_DIR/$(dirname "$BINARY")"
  gtest_out=()
  [ -n "${RESULTS_FILE:-}" ] && gtest_out=( --gtest_output="xml:${ABS_BUILD_DIR}/${RESULTS_FILE}" )
  # shellcheck disable=SC2086    # intentional word-splitting for EXTRA_ARGS
  "./$(basename "$BINARY")" $EXTRA_ARGS "${gtest_out[@]}" || rc=$?
else
  # shellcheck disable=SC2086
  bash -c "$CLIENT_CMD" || rc=$?
fi

exit "$rc"
