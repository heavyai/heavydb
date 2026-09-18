#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared container environment defaults for kafka-import integration test services.
# Source this after mapd-deps.sh (if present) in each container entrypoint script.

HEAVYDB_SOURCE="${HEAVYDB_SOURCE:-/heavydb-internal}"
HEAVYDB_BUILD="${HEAVYDB_BUILD:-/heavydb-build}"
HEAVYDB_HOST="${HEAVYDB_HOST:-heavydb}"
HEAVYDB_PORT="${HEAVYDB_PORT:-6274}"
HEAVYDB_USER="${HEAVYDB_USER:-admin}"
HEAVYDB_PASSWORD="${HEAVYDB_PASSWORD:-HyperInteractive}"
HEAVYDB_DB="${HEAVYDB_DB:-heavyai}"
KAFKA_BROKER="${KAFKA_BROKER:-kafka:9092}"
KAFKA_TOPIC="${KAFKA_TOPIC:-flights}"
export PATH="${HEAVYDB_BUILD}/bin:${PATH}"
