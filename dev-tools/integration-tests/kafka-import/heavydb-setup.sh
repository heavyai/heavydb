#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create the flights table in HeavyDB ready for KafkaImporter.
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

CREATE_SQL="${SCRIPT_DIR}/create_flights_table.sql"

log() { echo "[kafka-import] $*"; }
die() { echo "[kafka-import] ERROR: $*" >&2; exit 1; }

[[ -f "${CREATE_SQL}" ]] || die "SQL file not found: ${CREATE_SQL}"

log "Creating flights table in ${HEAVYDB_DB} on ${HEAVYDB_HOST}:${HEAVYDB_PORT}"
heavysql \
  -s "${HEAVYDB_HOST}" \
  --port "${HEAVYDB_PORT}" \
  -u "${HEAVYDB_USER}" \
  -p "${HEAVYDB_PASSWORD}" \
  -n -q "${HEAVYDB_DB}" < "${CREATE_SQL}"

log "Flights table created"
