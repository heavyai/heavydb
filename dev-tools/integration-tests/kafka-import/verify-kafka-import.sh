#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run KafkaImporter, poll until expected rows arrive in HeavyDB, then report.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /usr/local/mapd-deps/mapd-deps.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/mapd-deps/mapd-deps.sh
  set -u
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

EXPECTED_ROWS=100
POLL_INTERVAL=2
POLL_ATTEMPTS=30
KI_LOG=/tmp/kafkaimporter.log

log() { echo "[kafka-import] $*"; }
die() { echo "[kafka-import] ERROR: $*" >&2; exit 1; }

run_heavysql() {
  heavysql \
    -s "${HEAVYDB_HOST}" \
    --port "${HEAVYDB_PORT}" \
    -u "${HEAVYDB_USER}" \
    -p "${HEAVYDB_PASSWORD}" \
    -n -q "${HEAVYDB_DB}" <<< "$1"
}

log "Verifying heavysql connectivity and count query format..."
RAW=$(run_heavysql "SELECT COUNT(*) FROM flights;" 2>&1 || true)
log "  raw heavysql output: '${RAW}'"

log "Starting KafkaImporter (broker=${KAFKA_BROKER} topic=${KAFKA_TOPIC})"
KafkaImporter \
  --table flights \
  --database "${HEAVYDB_DB}" \
  --user "${HEAVYDB_USER}" \
  --passwd "${HEAVYDB_PASSWORD}" \
  --host "${HEAVYDB_HOST}" \
  --port "${HEAVYDB_PORT}" \
  --brokers "${KAFKA_BROKER}" \
  --topic "${KAFKA_TOPIC}" \
  --group-id kafka-import-test \
  --delim , \
  --null NA \
  --batch 1 \
  --print_error \
  --log-directory /tmp/ki-logs \
  --log-severity-clog INFO \
  > "${KI_LOG}" 2>&1 &
IMPORTER_PID=$!

sleep 2
log "KafkaImporter output (first 2s):"
cat "${KI_LOG}" || true
if ! kill -0 "${IMPORTER_PID}" 2>/dev/null; then
  die "KafkaImporter failed to start — check arguments"
fi

log "Polling for ${EXPECTED_ROWS} rows in flights table..."
COUNT=0
for i in $(seq 1 "${POLL_ATTEMPTS}"); do
  if ! kill -0 "${IMPORTER_PID}" 2>/dev/null; then
    log "KafkaImporter exited at attempt ${i}"
    break
  fi
  RAW=$(run_heavysql "SELECT COUNT(*) FROM flights;" 2>&1 || true)
  COUNT=$(echo "${RAW}" | tr -d ' \r\n')
  if [[ "${COUNT}" =~ ^[0-9]+$ ]] && (( COUNT >= EXPECTED_ROWS )); then
    kill "${IMPORTER_PID}" 2>/dev/null || true
    log "SUCCESS: ${COUNT} rows imported into flights table"
    exit 0
  fi
  log "  attempt ${i}/${POLL_ATTEMPTS}: raw='${RAW}' count=${COUNT:-?}"
  sleep "${POLL_INTERVAL}"
done

log "KafkaImporter final output:"
cat "${KI_LOG}" || true
kill "${IMPORTER_PID}" 2>/dev/null || true
die "Timed out waiting for ${EXPECTED_ROWS} rows — got ${COUNT:-0}"
