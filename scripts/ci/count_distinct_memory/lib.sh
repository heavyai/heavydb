#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for the count-distinct memory checks (tests.sh).
#
# Everything runs against a real heavydb server (real ERM configuration, real
# KILL QUERY / SHOW QUERIES), started and stopped inside one script run.
#
# Required env:  BUILD_DIR  directory containing bin/heavydb, bin/initheavy, bin/heavysql
# Optional env:  WORK_DIR   scratch dir (default: mktemp -d)
#                ROWS       number of distinct rows/groups to load (default 2000000)
#                GPU=1      do not pass --cpu-only (needs a CUDA build and a visible GPU)
#                RSS_CAP_MB safety cap: the harness SIGKILLs the server above this RSS,
#                           and reports it as "oom-killed" (default 6000)

set -u

: "${BUILD_DIR:?set BUILD_DIR to a build directory with bin/heavydb}"
WORK_DIR="${WORK_DIR:-$(mktemp -d)}"
ROWS="${ROWS:-2000000}"
RSS_CAP_MB="${RSS_CAP_MB:-6000}"
PORT="${PORT:-16274}"
HTTP_PORT="${HTTP_PORT:-16278}"
DATA_DIR="$WORK_DIR/data"
SERVER_PID=""
WATCHDOG_PID=""
OOM_KILLED_FLAG="$WORK_DIR/oom_killed"

hsql() {  # hsql "<sql>"  -> stdout of heavysql (quiet, no banner)
  timeout "${HSQL_TIMEOUT:-300}" "$BUILD_DIR/bin/heavysql" -p HyperInteractive -u admin --port "$PORT" -q heavyai <<<"$1" 2>&1
}

init_data_dir() {
  rm -rf "$DATA_DIR" "$OOM_KILLED_FLAG"; mkdir -p "$DATA_DIR"
  "$BUILD_DIR/bin/initheavy" -f "$DATA_DIR" >/dev/null 2>&1 || true
}

start_server() {  # start_server [extra heavydb flags...]   (keeps existing data dir)
  rm -f "$OOM_KILLED_FLAG"
  "$BUILD_DIR/bin/heavydb" --data "$DATA_DIR" --port "$PORT" --http-port "$HTTP_PORT" \
      --calcite-port $((PORT + 100)) $([ "${GPU:-0}" = 1 ] || echo --cpu-only) --enable-watchdog=false \
      --enable-runtime-query-interrupt=true --enable-non-kernel-time-query-interrupt=true \
      --allow-cpu-retry=true --allow-query-step-cpu-retry=true \
      --allowed-import-paths "[\"$WORK_DIR\"]" \
      "$@" >"$WORK_DIR/server.out" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    if hsql "SELECT 1;" 2>/dev/null | grep -q '^1$'; then return 0; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died at startup"; return 1; }
    sleep 1
  done
  echo "server did not come up"; return 1
}

stop_server() {
  [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  pkill -9 -f "calcite-.*jar" 2>/dev/null
  SERVER_PID=""
}

rss_mb() {  # current anonymous RSS of the server in MB (0 if gone)
  awk '/RssAnon/ {printf "%d", $2/1024}' "/proc/$SERVER_PID/status" 2>/dev/null || echo 0
}

# Watchdog standing in for the kernel OOM killer: kills the server above RSS_CAP_MB.
start_rss_watchdog() {
  (
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      if [ "$(rss_mb)" -gt "$RSS_CAP_MB" ]; then
        touch "$OOM_KILLED_FLAG"; kill -9 "$SERVER_PID" 2>/dev/null; break
      fi
      sleep 0.2
    done
  ) &
  WATCHDOG_PID=$!
}
stop_rss_watchdog() { [ -n "$WATCHDOG_PID" ] && { kill "$WATCHDOG_PID" 2>/dev/null; wait "$WATCHDOG_PID" 2>/dev/null; }; WATCHDOG_PID=""; }

# Sample RSS every 0.25 s into $1 until the given pid exits.
sample_rss_until_exit() {  # sample_rss_until_exit <outfile> <pid>
  local out=$1 pid=$2
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date +%s.%N) $(rss_mb)" >>"$out"; sleep 0.25
  done
}

peak_rss_mb() { awk 'BEGIN{m=0} $2>m{m=$2} END{print m}' "$1"; }

# Table with $ROWS rows; k is a unique BIGINT with a wide range (forces a baseline-hash
# group-by whose size comes from the cardinality estimator); d is a DOUBLE, which gives
# COUNT(DISTINCT d) the set-based (UnorderedSet) implementation on CPU.
load_table() {
  python3 -I - "$WORK_DIR/t.csv" "$ROWS" <<'EOF'
import sys
path, n = sys.argv[1], int(sys.argv[2])
with open(path, "w") as f:
    for i in range(n):
        f.write(f"{i * 7919 + 1000000007},{i % 1000}.5\n")
EOF
  hsql "DROP TABLE IF EXISTS t;" >/dev/null
  hsql "CREATE TABLE t (k BIGINT, d DOUBLE) WITH (fragment_size=500000);" >/dev/null
  hsql "COPY t FROM '$WORK_DIR/t.csv' WITH (header='false');" | tail -2
}

QUERY_SET_PATH="SELECT COUNT(*), SUM(c) FROM (SELECT k, COUNT(DISTINCT d) c FROM t GROUP BY k) x;"

# Table with $NUM_GROUPS distinct values of the group-by key `grp`, 2 rows per group.
# grp TEXT DICT(32) is the group-by key; the COUNT(DISTINCT <string expr>) argument has no
# value range, so it uses the set-based implementation.
load_table_groups() {
  python3 -I - "$WORK_DIR/g.csv" "$NUM_GROUPS" <<'EOF2'
import sys
path, n = sys.argv[1], int(sys.argv[2])
with open(path, "w") as f:
    for i in range(n):
        g = f"{i:09d}"
        f.write(f"{g},A,1,n1\n{g},B,2,n1\n")
EOF2
  hsql "DROP TABLE IF EXISTS cdm;" >/dev/null
  hsql "CREATE TABLE cdm (grp TEXT ENCODING DICT(32), kind TEXT ENCODING DICT(8), band SMALLINT, net TEXT ENCODING DICT(16)) WITH (fragment_size=1000000);" >/dev/null
  hsql "COPY cdm FROM '$WORK_DIR/g.csv' WITH (header='false');" | tail -2
}

# Set-based COUNT(DISTINCT <string expression>) per group, and a control with the same
# group-by and filter that has no set-based aggregate.
QUERY_DISTINCT="SELECT COUNT(*) AS total, SUM(CASE WHEN nb = 1 THEN 1 ELSE 0 END) AS single FROM (SELECT grp, MAX(CASE WHEN kind = 'A' AND band = 1 THEN 1 ELSE 0 END) AS has_a, COUNT(DISTINCT CAST(kind AS VARCHAR) || '_' || CAST(band AS VARCHAR)) AS nb FROM cdm WHERE net = 'n1' GROUP BY grp) t LIMIT 1000;"
QUERY_CONTROL="SELECT COUNT(*) AS total, SUM(CASE WHEN nb = 1 THEN 1 ELSE 0 END) AS single FROM (SELECT grp, MAX(CASE WHEN kind = 'A' AND band = 1 THEN 1 ELSE 0 END) AS has_a, MAX(band) AS nb FROM cdm WHERE net = 'n1' GROUP BY grp) t LIMIT 1000;"
