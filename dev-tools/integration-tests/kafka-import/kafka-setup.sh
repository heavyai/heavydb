#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create the Kafka topic and produce 100_flights.csv rows (header stripped).
set -euo pipefail

KAFKA_BIN="/opt/kafka/bin"
BOOTSTRAP="${BOOTSTRAP_SERVER:-kafka:9092}"
TOPIC="${KAFKA_TOPIC:-flights}"
SAMPLE_FILE="/heavydb/SampleData/100_flights.csv"

log() { echo "[kafka-import] $*"; }
die() { echo "[kafka-import] ERROR: $*" >&2; exit 1; }

[[ -f "${SAMPLE_FILE}" ]] || die "Sample file not found: ${SAMPLE_FILE}"

log "Creating topic ${TOPIC} on ${BOOTSTRAP}"
"${KAFKA_BIN}/kafka-topics.sh" \
  --bootstrap-server "${BOOTSTRAP}" \
  --create \
  --topic "${TOPIC}" \
  --partitions 1 \
  --replication-factor 1 \
  --if-not-exists

ROW_COUNT=$(( $(wc -l < "${SAMPLE_FILE}") - 1 ))
log "Producing ${ROW_COUNT} rows to topic ${TOPIC} (header stripped)"
tail -n +2 "${SAMPLE_FILE}" | \
  "${KAFKA_BIN}/kafka-console-producer.sh" \
    --bootstrap-server "${BOOTSTRAP}" \
    --topic "${TOPIC}"

log "Production complete: ${ROW_COUNT} rows sent"
