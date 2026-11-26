#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Check that a set-based COUNT(DISTINCT) over a high-cardinality GROUP BY stays within the
# configured CPU memory, is reported with the right phase in SHOW QUERIES, and can be
# stopped with KILL QUERY while it is allocating. For the weekly
# count-distinct-memory job.
#
# Self-contained: scripts/ci/count_distinct_memory/tests.sh does its own initheavy and
# starts, restarts and stops its own heavydb servers (it needs different pool sizes and
# the server's pid for memory sampling), so this does not use run_server.sh.
#
# Optional env vars:
#   BUILD_DIR    - build tree with bin/heavydb, bin/initheavy, bin/heavysql
#                  (default: /workspace/build if it exists, else build)
#   GPU          - 1 to run the server in GPU mode (the query is then planned for GPU and
#                  retried on CPU); default: 1 if nvidia-smi sees a GPU, else 0
#   WORK_DIR     - scratch dir for data and logs (default: $BUILD_DIR/count_distinct_memory)
#   NUM_GROUPS, THREADS, EXECUTORS, POOL_BYTES, RUN_POOL_BYTES, SLACK_MB
#                - see scripts/ci/count_distinct_memory/tests.sh
#   MAPD_DEPS_SH - mapd-deps env file (default: /usr/local/mapd-deps/mapd-deps.sh)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then BUILD_DIR=/workspace/build; else BUILD_DIR=build; fi
fi
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

source "$MAPD_DEPS_SH"

export BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
export WORK_DIR="${WORK_DIR:-$BUILD_DIR/count_distinct_memory}"
mkdir -p "$WORK_DIR"

if [ -z "${GPU:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU=1
  else
    GPU=0
  fi
fi
export GPU
echo "count-distinct-memory: BUILD_DIR=$BUILD_DIR WORK_DIR=$WORK_DIR GPU=$GPU"

exec "$SCRIPT_DIR/count_distinct_memory/tests.sh"
