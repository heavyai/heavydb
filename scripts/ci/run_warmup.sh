#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Exercise the --db-query-list warmup path: bring up a server, create and
# populate a table, shut down, then restart with a warmup query list and
# confirm the server comes up healthy. For the weekly warmup-queries job.
#
# Self-contained (does its own initheavy); does not require init_heavy.sh.
#
# Optional env vars:
#   BUILD_DIR    - build tree (default: /workspace/build if it exists, else build)
#   DATA         - data dir relative to BUILD_DIR (default: warmup_data)
#   SERVER_WAIT  - seconds to wait for each server start (default: 60)
#   MAPD_DEPS_SH - mapd-deps env file (default: /usr/local/mapd-deps/mapd-deps.sh)
#
set -euo pipefail

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then BUILD_DIR=/workspace/build; else BUILD_DIR=build; fi
fi
DATA="${DATA:-warmup_data}"
SERVER_WAIT="${SERVER_WAIT:-60}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

source "$MAPD_DEPS_SH"

cd "$BUILD_DIR"

SERVER_PID=""
stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
trap stop_server EXIT

mkdir -p "$DATA"
bin/initheavy --skip-geo "$DATA"

# Phase 1: populate a table.
bin/heavydb --data "$DATA" &
SERVER_PID=$!
sleep "$SERVER_WAIT"

cat > populate_tables.sql <<'EOF'
CREATE TABLE test (x INT) WITH (FRAGMENT_SIZE=2);
INSERT INTO TEST VALUES (1);
INSERT INTO TEST VALUES (2);
INSERT INTO TEST VALUES (3);
INSERT INTO TEST VALUES (4);
INSERT INTO TEST VALUES (5);
EOF
bin/heavysql -p HyperInteractive < populate_tables.sql
stop_server
sleep 30

# Phase 2: restart with a warmup query list and confirm the server is healthy.
cat > warmup_queries.sql <<'EOF'
USER admin heavyai {
SELECT COUNT(*) FROM test;
}
EOF
bin/heavydb --data "$DATA" --db-query-list=warmup_queries.sql &
SERVER_PID=$!
sleep "$SERVER_WAIT"

echo "\status" | bin/heavysql -p HyperInteractive
