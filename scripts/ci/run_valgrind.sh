#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run a gtest binary under valgrind memcheck with a PER-TEST timeout watchdog.
#
# gtest has no built-in per-test timeout, and isolating each case in its own
# valgrind process is far too expensive (heavy valgrind + DB init per case). So
# we run the whole binary under one valgrind invocation and watch its gtest
# progress: if a single test case runs longer than PER_TEST_TIMEOUT with no
# completion, we kill the run, record that case as a TIMEOUT FAILURE, add it to
# the exclude filter, and resume with the remaining cases. Repeated for each
# hang. At the end, if any case timed out the job fails (exit 1) — so hangs
# surface as failures instead of being silently dropped or eating the whole job
# timeout, and no coverage is statically removed.
#
# Assumes the catalog has already been initialized (scripts/ci/init_heavy.sh).
#
# Required env vars:
#   BINARY        - gtest binary relative to BUILD_DIR (e.g. Tests/ExecuteTest)
#
# Optional env vars:
#   BUILD_DIR        - build tree (default: /workspace/build if it exists, else build)
#   RESULTS_FILE     - gtest XML filename written into BUILD_DIR (default: test-results-valgrind.xml)
#   VALGRIND_XML     - valgrind XML filename written into BUILD_DIR (default: valgrind-memcheck.xml)
#   SUPPRESSIONS     - valgrind suppressions file (default: <repo>/config/valgrind.suppressions)
#   GTEST_FILTER     - base gtest filter (default: empty); watchdog appends hung cases to its negative set
#   PER_TEST_TIMEOUT - seconds a single gtest case may run under valgrind before
#                      it's killed and recorded as a timeout failure (default: 900 = 15m;
#                      comfortably above the slowest legit case observed ~6m)
#   WATCHDOG_POLL    - seconds between progress checks (default: 20)
#   MAX_HANGS        - safety cap on how many cases may be killed before aborting (default: 25)
#   MAPD_DEPS_SH     - mapd-deps env file (default: /usr/local/mapd-deps/mapd-deps.sh)
#
set -euo pipefail

: "${BINARY:?BINARY must be set}"

if [ -z "${BUILD_DIR:-}" ]; then
  if [ -d /workspace/build ]; then BUILD_DIR=/workspace/build; else BUILD_DIR=build; fi
fi
RESULTS_FILE="${RESULTS_FILE:-test-results-valgrind.xml}"
VALGRIND_XML="${VALGRIND_XML:-valgrind-memcheck.xml}"
GTEST_FILTER="${GTEST_FILTER:-}"
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-900}"
WATCHDOG_POLL="${WATCHDOG_POLL:-20}"
MAX_HANGS="${MAX_HANGS:-25}"
MAPD_DEPS_SH="${MAPD_DEPS_SH:-/usr/local/mapd-deps/mapd-deps.sh}"

command -v valgrind >/dev/null 2>&1 || { echo "ERROR: valgrind not installed in this image" >&2; exit 1; }

source "$MAPD_DEPS_SH"

ABS_BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
SUPPRESSIONS="${SUPPRESSIONS:-$(cd "$ABS_BUILD_DIR/.." && pwd)/config/valgrind.suppressions}"

cd "$ABS_BUILD_DIR/$(dirname "$BINARY")"
BIN="./$(basename "$BINARY")"

# Compose a gtest filter that runs the base set minus the accumulated hung cases.
# gtest filter syntax: "positive[-negative]". Add hung cases to the negative part.
compose_filter() {
  local base="$1" hung="$2"
  if [ -z "$hung" ]; then printf '%s' "$base"; return; fi
  if [ -z "$base" ]; then printf -- '-%s' "$hung"; return; fi
  case "$base" in
    *-*) printf '%s:%s' "$base" "$hung" ;;   # base already has a negative section
    *)   printf '%s-%s' "$base" "$hung" ;;   # base is positive-only
  esac
}

hung=""                 # colon-joined hung case names (accumulated across passes)
declare -a timed_out=() # for the final report
final_rc=0

while : ; do
  filt="$(compose_filter "$GTEST_FILTER" "$hung")"
  runlog="$(mktemp)"
  echo ">>> valgrind pass (per-test timeout ${PER_TEST_TIMEOUT}s); excluded hung so far: ${hung:-none}"

  # Launch in a new session so the whole tree (valgrind + the guest + any
  # Calcite/JVM children) can be killed as a process group. stdbuf forces
  # line-buffered stdout so the watchdog sees [ RUN ]/[ OK ] lines promptly.
  export VG_SUPP="$SUPPRESSIONS" VG_XML="${ABS_BUILD_DIR}/${VALGRIND_XML}" \
         VG_BIN="$BIN" VG_OUT="xml:${ABS_BUILD_DIR}/${RESULTS_FILE}" VG_FILTER="$filt"
  # --error-exitcode=1 makes valgrind fail the job on errors. By default
  # --errors-for-leak-kinds is "definite,possible"; in a long-lived server
  # like heavydb, *possible* leaks at shutdown (singletons, thread-locals,
  # global caches) are unavoidable and not bugs, yet they would trip the exit
  # code while being invisible (--show-leak-kinds=definite hides them) ->
  # exit 1 with an empty XML. Restrict BOTH the shown set and the error set
  # to "definite" so the failure signal matches what is reported: the job
  # fails iff there is a genuine invalid-memory error or a definite leak, and
  # whatever caused the failure is serialized into the XML with its stack.
  setsid bash -c '
    args=( --suppressions="$VG_SUPP" --gen-suppressions=all
           --leak-check=full --show-leak-kinds=definite --errors-for-leak-kinds=definite
           --tool=memcheck --error-exitcode=1 --xml=yes --xml-file="$VG_XML" "$VG_BIN" )
    [ -n "$VG_FILTER" ] && args+=( "--gtest_filter=$VG_FILTER" )
    args+=( "--gtest_output=$VG_OUT" )
    exec stdbuf -oL -eL valgrind "${args[@]}"
  ' >"$runlog" 2>&1 &
  pgid=$!

  cur=""; cur_since=$(date +%s); killed=0
  while kill -0 "$pgid" 2>/dev/null; do
    sleep "$WATCHDOG_POLL"
    # The in-flight case is the last "[ RUN ]" with no later completion line.
    run_ln=$(grep -anE '\[ RUN      \]' "$runlog" 2>/dev/null | tail -1 | cut -d: -f1 || true)
    end_ln=$(grep -anE '\[       OK \]|\[  FAILED  \]|\[  SKIPPED \]' "$runlog" 2>/dev/null | tail -1 | cut -d: -f1 || true)
    if [ -n "$run_ln" ] && { [ -z "$end_ln" ] || [ "$run_ln" -gt "$end_ln" ]; }; then
      running=$(sed -n "${run_ln}p" "$runlog" | sed -E 's/.*\[ RUN      \] //' | awk '{print $1}')
      if [ "$running" != "$cur" ]; then cur="$running"; cur_since=$(date +%s); fi
      if [ $(( $(date +%s) - cur_since )) -ge "$PER_TEST_TIMEOUT" ]; then
        echo "!!! per-test timeout: '$cur' exceeded ${PER_TEST_TIMEOUT}s under valgrind — killing and recording as failure"
        kill -KILL -- -"$pgid" 2>/dev/null || kill -KILL "$pgid" 2>/dev/null || true
        timed_out+=( "$cur" )
        hung="${hung:+$hung:}$cur"
        killed=1
        break
      fi
    else
      cur=""  # between cases (or still in startup/DB init) — don't start the clock
    fi
  done

  wait "$pgid" 2>/dev/null && rc=0 || rc=$?

  # Surface this pass's full valgrind+gtest output to the job log (it was
  # captured to $runlog so the watchdog could parse progress). Includes
  # gtest [ RUN ]/[ OK ] lines and valgrind's "ERROR SUMMARY" — needed to
  # see what failed (the GitHub log otherwise only shows watchdog messages).
  echo "----- valgrind pass output (begin) -----"
  cat "$runlog" || true
  echo "----- valgrind pass output (end); valgrind exit=$rc -----"
  rm -f "$runlog" || true

  if [ "$killed" -eq 1 ]; then
    if [ "${#timed_out[@]}" -ge "$MAX_HANGS" ]; then
      echo "ERROR: hit MAX_HANGS=${MAX_HANGS} hung cases; aborting." >&2
      break
    fi
    continue   # resume with the hung case excluded
  fi

  final_rc=$rc   # process finished on its own (rc!=0 => a test failed or memcheck found errors)
  break
done

if [ "${#timed_out[@]}" -gt 0 ]; then
  echo "=============================================================="
  echo "Cases that hung under valgrind (>${PER_TEST_TIMEOUT}s) — recorded as FAILURES:"
  printf '  - %s\n' "${timed_out[@]}"
  echo "=============================================================="
  exit 1
fi

exit "$final_rc"
