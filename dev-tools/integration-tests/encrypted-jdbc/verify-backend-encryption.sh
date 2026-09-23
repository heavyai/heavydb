#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wait for heavydb, then run heavysql checks against default initheavy sample tables.
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

HEAVYSQL_BIN="${HEAVYDB_BUILD}/bin/heavysql"

log() {
  echo "[encrypted-jdbc] $*"
}

die() {
  echo "[encrypted-jdbc] ERROR: $*" >&2
  exit 1
}

[[ -x "${HEAVYSQL_BIN}" ]] || die "heavysql not found at ${HEAVYSQL_BIN}"
[[ -f "${CA_CERT}" ]] || die "CA certificate not found at ${CA_CERT}"

log "Querying ${HEAVYDB_HOST}:${HEAVYDB_PORT} database ${HEAVYDB_DB}"

run_heavysql_query() {
  local sql="$1"
  "${HEAVYSQL_BIN}" \
    --ca-cert "${CA_CERT}" \
    -s "${HEAVYDB_HOST}" \
    --port "${HEAVYDB_PORT}" \
    -u "${HEAVYDB_USER}" \
    -p "${HEAVYDB_PASSWORD}" \
    -n -q "${HEAVYDB_DB}" <<< "${sql}"
}

OUTPUT="$(
  run_heavysql_query "$(cat <<'SQL'
SHOW TABLES;
SQL
)"
)"

printf '%s\n' "${OUTPUT}"

for table in heavyai_us_states heavyai_countries; do
  grep -q "${table}" <<< "${OUTPUT}" || die "default table not found: ${table}"
done

log "Default sample tables present"
