#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Initialize storage and start HeavyDB (unencrypted).
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
HEAVYDB_BIN="${HEAVYDB_BUILD}/bin/heavydb"
INITHEAVY_BIN="${HEAVYDB_BUILD}/bin/initheavy"

log() { echo "[kafka-import] $*"; }
die() { echo "[kafka-import] ERROR: $*" >&2; exit 1; }

[[ -x "${HEAVYDB_BIN}" ]] || die "heavydb not found at ${HEAVYDB_BIN} (check HEAVYDB_BUILD mount)"
[[ -x "${INITHEAVY_BIN}" ]] || die "initheavy not found at ${INITHEAVY_BIN}"

mkdir -p "${HEAVYDB_DATA}"
log "Initializing storage at ${HEAVYDB_DATA}"
"${INITHEAVY_BIN}" -f "${HEAVYDB_DATA}"

log "Starting heavydb (unencrypted)"
log "  data: ${HEAVYDB_DATA}"
cd "${HEAVYDB_BUILD}"
exec "${HEAVYDB_BIN}" --data "${HEAVYDB_DATA}"
