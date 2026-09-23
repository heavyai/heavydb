#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate dev TLS material, initialize storage, and start heavydb.
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

HEAVYDB_DATA="${HEAVYDB_DATA:-/tmp/heavydb-data}"
OPENSSL_BIN="${OPENSSL_BIN:-/usr/local/mapd-deps/bin/openssl}"
KEYTOOL_BIN="${KEYTOOL_BIN:-keytool}"

GEN_SCRIPT="${HEAVYDB_SOURCE}/dev-tools/integration-tests/generate-calcite-test-keystore.sh"
HEAVYDB_BIN="${HEAVYDB_BUILD}/bin/heavydb"
INITHEAVY_BIN="${HEAVYDB_BUILD}/bin/initheavy"
CONFIG_FILE="${CA_INFO_DIR}/heavyai-tls.conf"

log() {
  echo "[encrypted-jdbc] $*"
}

die() {
  echo "[encrypted-jdbc] ERROR: $*" >&2
  exit 1
}

[[ -x "${HEAVYDB_BIN}" ]] || die "heavydb not found at ${HEAVYDB_BIN} (check HEAVYDB_BUILD mount)"
[[ -x "${INITHEAVY_BIN}" ]] || die "initheavy not found at ${INITHEAVY_BIN}"
[[ -f "${GEN_SCRIPT}" ]] || die "TLS script not found at ${GEN_SCRIPT}"

mkdir -p "${CA_INFO_DIR}" "${HEAVYDB_DATA}"

if ! touch "${CA_INFO_DIR}/.write-test" 2>/dev/null; then
  die "CA_INFO_DIR ${CA_INFO_DIR} is not writable as uid $(id -u):gid $(id -g) (check TLS volume mount)"
fi
rm -f "${CA_INFO_DIR}/.write-test"

log "Generating TLS material in ${CA_INFO_DIR}"
OUT_DIR="${CA_INFO_DIR}" \
  OPENSSL_BIN="${OPENSSL_BIN}" \
  KEYTOOL_BIN="${KEYTOOL_BIN}" \
  bash "${GEN_SCRIPT}"

[[ -f "${CONFIG_FILE}" ]] || die "Expected config file missing: ${CONFIG_FILE}"

log "Initializing storage at ${HEAVYDB_DATA} (initheavy -f)"
"${INITHEAVY_BIN}" -f "${HEAVYDB_DATA}"

log "Starting heavydb"
log "  config: ${CONFIG_FILE}"
log "  data:   ${HEAVYDB_DATA}"

cd "${HEAVYDB_BUILD}"
exec "${HEAVYDB_BIN}" --config "${CONFIG_FILE}" --data "${HEAVYDB_DATA}"
