#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared container environment defaults for encrypted-jdbc integration test services.
# Source this after mapd-deps.sh (if present) in each container entrypoint script.

HEAVYDB_SOURCE="${HEAVYDB_SOURCE:-/heavydb}"
HEAVYDB_BUILD="${HEAVYDB_BUILD:-/heavydb-build}"
CA_INFO_DIR="${CA_INFO_DIR:-/heavydb-ca-info}"
HEAVYDB_HOST="${HEAVYDB_HOST:-heavydb}"
HEAVYDB_PORT="${HEAVYDB_PORT:-6274}"
HEAVYDB_USER="${HEAVYDB_USER:-admin}"
HEAVYDB_PASSWORD="${HEAVYDB_PASSWORD:-HyperInteractive}"
HEAVYDB_DB="${HEAVYDB_DB:-heavyai}"
CA_CERT="${CA_INFO_DIR}/ca/ca.crt"
export PATH="${HEAVYDB_BUILD}/bin:${PATH}"
