#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set-based COUNT(DISTINCT) host-memory limits, status and KILL QUERY.
# Black-box checks against a real heavydb server; runs on CPU or, with GPU=1, on a GPU
# build where the query is planned for GPU and retried on CPU.
#
#   BUILD_DIR=<dir with bin/heavydb, bin/initheavy, bin/heavysql> ./tests.sh [T0 T1 T2 T3]
#
# Each test expresses the desired behaviour. T0 is a control that must always pass; on a build
# without the fix for the unaccounted set heap T1-T3 FAIL.
#
#   T0  control      same query shape with MAX() instead of COUNT(DISTINCT <expr>): host memory
#                    stays within the configured pool.   (sanity check for the harness)
#   T1  limits       COUNT(DISTINCT <string expr>) over many groups must stay within the
#                    configured CPU pool (+slack) or fail cleanly - never exhaust host RAM.
#   T2  status       SHOW QUERIES must not report RUNNING_REDUCTION while the kernel step runs.
#   T3  kill         KILL QUERY issued while the step is allocating must stop the allocation
#                    and the query promptly.
#
# Scale knobs (defaults are the smallest that show everything, ~1.3 GB RSS, ~10 s/test):
#   NUM_GROUPS (distinct group keys, 2 rows each) default 2000000
#   THREADS (--cpu-threads) default 8, EXECUTORS (--num-executors) default 4
#   POOL_BYTES (--cpu-buffer-mem-bytes, T0/T1) default 268435456, SLACK_MB default 128
#   RUN_POOL_BYTES pool for T2/T3, large enough that the query is admitted, default 2 GiB
source "$(dirname "$0")/lib.sh"

NUM_GROUPS="${NUM_GROUPS:-2000000}"
THREADS="${THREADS:-8}"
EXECUTORS="${EXECUTORS:-4}"
POOL_BYTES="${POOL_BYTES:-268435456}"
POOL_MB=$((POOL_BYTES / 1048576))
# T2/T3 need the query to be ADMITTED so that it actually allocates; with a correct memory
# budget a 256 MB pool rejects it up front (that is T1's PASS). Use a pool large enough to run it.
RUN_POOL_BYTES="${RUN_POOL_BYTES:-2147483648}"
SLACK_MB="${SLACK_MB:-128}"
export HSQL_TIMEOUT="${HSQL_TIMEOUT:-120}"
RESULTS=()
trap 'stop_rss_watchdog; stop_server' EXIT

record() { RESULTS+=("$1|$2|$3"); echo "[$1] $2 -- $3"; }

# Load once with the default pool (a 256 MB pool is too small for the dictionary import),
# then restart with the constrained configuration for every test.
setup_data() {
  init_data_dir
  start_server || exit 1
  load_table_groups
  stop_server
}

fresh_server() {  # fresh_server [pool_bytes]
  local pool=${1:-$POOL_BYTES}
  stop_rss_watchdog; stop_server
  start_server --cpu-buffer-mem-bytes "$pool" --cpu-threads "$THREADS" \
               --num-executors "$EXECUTORS" || { echo "server start failed"; return 1; }
  hsql "SELECT COUNT(*) FROM cdm;" >/dev/null   # warm dictionaries and metadata
  BASE_MB=$(rss_mb)
}

start_query() {  # start_query "<sql>"  -> background; sets QPID
  rm -f "$WORK_DIR/query.out" "$WORK_DIR/query.end"
  ( hsql "$1" >"$WORK_DIR/query.out" 2>&1; date +%s.%N >"$WORK_DIR/query.end" ) &
  QPID=$!
}

query_status_row() {  # first SHOW QUERIES row that is our query: "<session>|<status>"
  hsql "SHOW QUERIES;" | grep -a "FROM cdm" | head -1 | tr -d '\000' | awk -F'|' '{print $1 "|" $2}'
}

# ------------------------------------------------------------------------------------------
# Classifies what the query returned (query.out of the last start_query). A query that fails
# for an unrelated reason (SQL error, missing table, dropped connection) must not count as
# "survived": it has to return the expected result, or, if `reject_ok`, be rejected for
# needing too much CPU result memory.
#   outcome_of <result|result_or_reject>  -> prints "result", "rejected" or "unexpected: <text>"
outcome_of() {
  local mode=$1 out
  out=$(tr -d '\000' <"$WORK_DIR/query.out")
  if printf '%s\n' "$out" | grep -q "^${NUM_GROUPS}|"; then echo result; return 0; fi
  if [ "$mode" = result_or_reject ] &&
     printf '%s\n' "$out" | grep -q "more CPU result memory"; then echo rejected; return 0; fi
  echo "unexpected: $(printf '%s' "$out" | head -c 120 | tr '\n' ' ')"; return 1
}

run_limits_test() {  # run_limits_test <id> <label> <sql> <result|result_or_reject>
  local id=$1 label=$2 sql=$3 mode=$4
  fresh_server || { record "$id" "$label" "FAIL (server did not start)"; return; }
  RSS_CAP_MB=$((BASE_MB + POOL_MB + SLACK_MB))     # stands in for the kernel OOM killer
  start_rss_watchdog
  start_query "$sql"
  sample_rss_until_exit "$WORK_DIR/rss.txt" "$QPID"
  stop_rss_watchdog
  local peak growth alive=no killed=no outcome outcome_ok=yes
  peak=$(peak_rss_mb "$WORK_DIR/rss.txt"); growth=$((peak - BASE_MB))
  kill -0 "$SERVER_PID" 2>/dev/null && alive=yes
  [ -e "$OOM_KILLED_FLAG" ] && killed=yes
  outcome=$(outcome_of "$mode") || outcome_ok=no
  local detail="baseline ${BASE_MB} MB, peak ${peak} MB (+${growth} MB) vs pool ${POOL_MB} MB + slack ${SLACK_MB} MB; server alive: ${alive}; killed at cap: ${killed}; outcome: ${outcome}"
  if [ "$alive" = yes ] && [ "$killed" = no ] && [ "$outcome_ok" = yes ]; then
    record "$id" "$label" "PASS ($detail)"
  else
    record "$id" "$label" "FAIL ($detail)"
  fi
}

test_T0() { run_limits_test T0 "control (MAX) stays within pool" "$QUERY_CONTROL" result; }
test_T1() {
  run_limits_test T1 "COUNT(DISTINCT expr) stays within pool or fails cleanly" \
                  "$QUERY_DISTINCT" result_or_reject
}

# Wait (up to $2 s) until the server's RSS has grown by $1 MB over baseline.
wait_for_growth() {
  local need=$1 max=$2 i=0
  while [ $((i * 2)) -lt $((max * 10)) ]; do
    [ $(($(rss_mb) - BASE_MB)) -ge "$need" ] && return 0
    kill -0 "$QPID" 2>/dev/null || return 1
    sleep 0.2; i=$((i + 1))
  done
  return 1
}

test_T2() {
  fresh_server "$RUN_POOL_BYTES" || { record T2 "status" "FAIL (server did not start)"; return; }
  RSS_CAP_MB=$((BASE_MB + 4000)); start_rss_watchdog      # safety net only
  start_query "$QUERY_DISTINCT"
  if ! wait_for_growth 250 30; then
    stop_rss_watchdog; record T2 "status" "INCONCLUSIVE (memory never grew by 250 MB)"; return
  fi
  local row status sid statuses=""
  for _ in 1 2 3 4; do
    row=$(query_status_row); sid=${row%%|*}; status=${row##*|}
    statuses="$statuses $status"; sleep 0.3
  done
  [ -n "$sid" ] && hsql "KILL QUERY '$sid';" >/dev/null 2>&1
  wait "$QPID" 2>/dev/null; stop_rss_watchdog
  if echo "$statuses" | grep -q RUNNING_REDUCTION; then
    record T2 "SHOW QUERIES reports the real phase" "FAIL (while the kernel step was allocating, statuses seen:${statuses})"
  else
    record T2 "SHOW QUERIES reports the real phase" "PASS (statuses seen:${statuses})"
  fi
}

test_T3() {
  fresh_server "$RUN_POOL_BYTES" || { record T3 "kill" "FAIL (server did not start)"; return; }
  RSS_CAP_MB=$((BASE_MB + 4000)); start_rss_watchdog      # safety net only
  start_query "$QUERY_DISTINCT"
  if ! wait_for_growth 300 30; then
    stop_rss_watchdog; record T3 "kill" "INCONCLUSIVE (memory never grew by 300 MB)"; return
  fi
  local row sid k0 tk
  row=$(query_status_row); sid=${row%%|*}
  k0=$(rss_mb); tk=$(date +%s.%N)
  hsql "KILL QUERY '$sid';" >/dev/null 2>&1
  local peak=$k0 ended=no i
  for i in $(seq 1 50); do                                   # watch for up to 10 s
    local r; r=$(rss_mb); [ "$r" -gt "$peak" ] && peak=$r
    if ! kill -0 "$QPID" 2>/dev/null; then ended=yes; break; fi
    sleep 0.2
  done
  local stop_s="n/a"
  [ -e "$WORK_DIR/query.end" ] && stop_s=$(awk "BEGIN{printf \"%.1f\", $(cat "$WORK_DIR/query.end") - $tk}")
  kill -9 "$QPID" 2>/dev/null; wait "$QPID" 2>/dev/null; stop_rss_watchdog
  local after=$((peak - k0))
  local detail="KILL issued at +$((k0 - BASE_MB)) MB; RSS kept growing by ${after} MB after the kill; query ended: ${ended}, ${stop_s} s after the kill"
  if [ "$ended" = yes ] && [ "$after" -le 100 ] && awk "BEGIN{exit !(${stop_s:-99} <= 2.0)}"; then
    record T3 "KILL QUERY stops allocation promptly" "PASS ($detail)"
  else
    record T3 "KILL QUERY stops allocation promptly" "FAIL ($detail; want <=100 MB and <=2 s)"
  fi
}

# ------------------------------------------------------------------------------------------
setup_data
echo "config: groups=$NUM_GROUPS threads=$THREADS executors=$EXECUTORS pool=${POOL_MB}MB slack=${SLACK_MB}MB"
SELECTED=("$@"); [ ${#SELECTED[@]} -eq 0 ] && SELECTED=(T0 T1 T2 T3)
for t in "${SELECTED[@]}"; do "test_$t"; done

if [ "${GPU:-0}" = 1 ]; then
  retries=$(cat "$DATA_DIR"/log/heavydb.INFO.*.log 2>/dev/null | grep -a "Retrying current query step" | wc -l)
  echo "GPU run: 'Retrying current query step ... on CPU' seen in the server logs: ${retries} time(s)"
fi
echo; echo "=== summary"
rc=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r id label verdict <<<"$r"; printf '%-3s %-6s %s\n' "$id" "${verdict%% *}" "$label"
  case "$verdict" in PASS*) ;; *) rc=1;; esac
done
exit $rc
