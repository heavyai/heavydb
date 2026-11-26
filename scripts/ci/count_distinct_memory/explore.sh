#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exploration run: how much RSS does the set-based COUNT(DISTINCT) query use vs the pool?
source "$(dirname "$0")/lib.sh"
POOL_BYTES="${POOL_BYTES:-268435456}"
THREADS="${THREADS:-4}"
pick_query() { case "${QSEL:-setpath}" in distinct) echo "$QUERY_DISTINCT";; control) echo "$QUERY_CONTROL";; *) echo "$QUERY_SET_PATH";; esac; }
trap 'stop_rss_watchdog; stop_server' EXIT

init_data_dir
start_server || exit 1                      # default pool for the load
if [ -n "${NUM_GROUPS:-}" ]; then load_table_groups; else load_table; fi
stop_server
start_server --cpu-buffer-mem-bytes "$POOL_BYTES" --cpu-threads "$THREADS" \
             --num-executors "${EXECUTORS:-4}" --verbose=true --allow-memory-status-log=true || exit 1
hsql "SELECT COUNT(*) FROM $([ -n "${NUM_GROUPS:-}" ] && echo cdm || echo t);" >/dev/null   # warm dictionaries/metadata
echo "baseline RSS: $(rss_mb) MB (pool $((POOL_BYTES/1048576)) MB, threads $THREADS, rows $ROWS)"
start_rss_watchdog
( hsql "$(pick_query)" >"$WORK_DIR/query.out" 2>&1; echo "$?" >"$WORK_DIR/query.rc" ) &
QPID=$!
sample_rss_until_exit "$WORK_DIR/rss.txt" "$QPID"
stop_rss_watchdog
echo "query output:"; cat "$WORK_DIR/query.out" | head -5
echo "peak RSS during query: $(peak_rss_mb "$WORK_DIR/rss.txt") MB; oom-killed: $([ -e "$OOM_KILLED_FLAG" ] && echo yes || echo no)"
LOG=$(ls "$DATA_DIR"/storage/log/heavydb.INFO 2>/dev/null || ls "$DATA_DIR"/log/heavydb.INFO 2>/dev/null)
echo "--- log excerpts ($LOG)"
grep -E "Retrying|OutOfMemory|OUT_OF_CPU|CardinalityEstimation|cardinality|requested between|Allocate|count distinct" "$LOG" | cut -c1-230 | tail -20
cp "$LOG" "${KEEP_LOG:-/dev/null}" 2>/dev/null
echo "--- RSS timeline (s since query start, MB)"
T0=$(head -1 "$WORK_DIR/rss.txt" | awk '{print $1}')
awk -v t0="$T0" '{printf "%.1f:%d ", $1-t0, $2} END{print ""}' "$WORK_DIR/rss.txt" | fold -w 150
