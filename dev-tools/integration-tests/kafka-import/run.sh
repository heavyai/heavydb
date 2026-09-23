#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start HeavyDB + Kafka via docker/dev-kafka-import-test, run KafkaImporter verification, tear down.
#
# Examples:
#   ./run.sh --build-dir build
#   ./run.sh --build-dir xb
#
# --build-dir is CMAKE_BINARY_DIR (must contain bin/heavydb).
# HEAVYDB_SOURCE is three levels above this script (repo root).
# In-container paths are fixed: /heavydb, /heavydb-build, /tmp/heavydb-data (default storage).
# By default HeavyDB storage lives inside the container and is removed with compose down.
# Optional host bind mount: --host-storage → storage/kafka-import on the host.
#
# Requires Docker Compose v2.29+ for `up --wait` and service_completed_successfully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../load-build-env.sh"

DEFAULT_IMAGE='ghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest'
DEFAULT_KAFKA_IMAGE='apache/kafka:3.7.2'
USE_HOST_STORAGE="${USE_HOST_STORAGE:-0}"
BUILD_DIR_ARG=""

usage() {
  cat <<EOF
Usage: ${0##*/} --build-dir DIR [--image IMAGE] [--kafka-image IMAGE] [--host-storage]

Required:
  --build-dir      CMake binary directory (must contain bin/heavydb)

Options:
  --image          Docker image for HeavyDB services (default: ${DEFAULT_IMAGE})
  --kafka-image    Docker image for the Kafka broker (default: ${DEFAULT_KAFKA_IMAGE})
  --host-storage   Bind-mount storage/kafka-import on the host (default: container-only storage)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || { echo "[kafka-import] ERROR: --build-dir requires a value" >&2; exit 1; }
      BUILD_DIR_ARG="$2"
      shift 2
      ;;
    --image)
      [[ $# -ge 2 ]] || { echo "[kafka-import] ERROR: --image requires a value" >&2; exit 1; }
      HEAVYDB_IMAGE="$2"
      shift 2
      ;;
    --kafka-image)
      [[ $# -ge 2 ]] || { echo "[kafka-import] ERROR: --kafka-image requires a value" >&2; exit 1; }
      KAFKA_IMAGE="$2"
      shift 2
      ;;
    --host-storage)
      USE_HOST_STORAGE=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "[kafka-import] ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export HEAVYDB_IMAGE="${HEAVYDB_IMAGE:-${DEFAULT_IMAGE}}"
export KAFKA_IMAGE="${KAFKA_IMAGE:-${DEFAULT_KAFKA_IMAGE}}"

load_integration_test_build_env "kafka-import" "${SCRIPT_DIR}"

COMPOSE_DIR="${HEAVYDB_SOURCE}/docker/dev-kafka-import-test"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"
COMPOSE_HOST_STORAGE_FILE="${COMPOSE_DIR}/docker-compose.host-storage.yml"

export HEAVYDB_DATA="${HEAVYDB_SOURCE}/storage/kafka-import"
export HEAVYDB_CONTAINER_NAME="heavydb-dev-kafka-import-test"
export HEAVYDB_SETUP_CONTAINER_NAME="heavydb-dev-kafka-import-test-setup"
export HEAVYDB_VERIFY_CONTAINER_NAME="heavydb-dev-kafka-import-test-verify"
export KAFKA_CONTAINER_NAME="kafka-dev-kafka-import-test"
export KAFKA_SETUP_CONTAINER_NAME="kafka-dev-kafka-import-test-setup"

export USER_SPEC="$(id -u):$(id -g)"

prepare_heavydb_data() {
  if [[ -e "${HEAVYDB_DATA}" ]] && [[ ! -w "${HEAVYDB_DATA}" ]]; then
    echo "[kafka-import] ERROR: ${HEAVYDB_DATA} is not writable." >&2
    echo "[kafka-import] Remove it, then re-run (sudo rm -rf if needed):" >&2
    echo "[kafka-import]   rm -rf ${HEAVYDB_DATA}" >&2
    exit 1
  fi
  if [[ -d "${HEAVYDB_DATA}" ]] && find "${HEAVYDB_DATA}" -mindepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
    echo "[kafka-import] ERROR: ${HEAVYDB_DATA} contains files not writable by the current user." >&2
    echo "[kafka-import] This often happens after another test wrote storage as root." >&2
    echo "[kafka-import] Remove it, then re-run (sudo rm -rf if needed):" >&2
    echo "[kafka-import]   rm -rf ${HEAVYDB_DATA}" >&2
    exit 1
  fi
  mkdir -p "${HEAVYDB_DATA}"
}

COMPOSE_FILES=(-f "${COMPOSE_FILE}")
if [[ "${USE_HOST_STORAGE}" == 1 ]]; then
  COMPOSE_FILES+=(-f "${COMPOSE_HOST_STORAGE_FILE}")
fi

if [[ -n "${COMPOSE:-}" ]]; then
  read -ra COMPOSE_CMD <<< "${COMPOSE}"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v compose >/dev/null 2>&1 && compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "[kafka-import] ERROR: docker compose not found" >&2
  exit 1
fi

compose() {
  (cd "${COMPOSE_DIR}" && "${COMPOSE_CMD[@]}" "${COMPOSE_FILES[@]}" "$@")
}

log() {
  echo "[kafka-import] $*"
}

down() {
  log "Stopping containers..."
  compose down --remove-orphans || true
}

trap down EXIT INT TERM

log "HEAVYDB_BUILD=${HEAVYDB_BUILD}"
log "HEAVYDB_IMAGE=${HEAVYDB_IMAGE}"
if [[ "${USE_HOST_STORAGE}" == 1 ]]; then
  log "HeavyDB storage: host bind mount ${HEAVYDB_DATA}"
  prepare_heavydb_data
else
  log "HeavyDB storage: container /tmp/heavydb-data (removed with compose down)"
fi

log "Starting heavydb and kafka (wait until healthy)..."
if ! compose up -d --wait heavydb kafka; then
  log "Startup failed; recent logs:"
  compose logs --tail=40 heavydb kafka || true
  exit 1
fi

log "Running setup and verification..."
if ! compose run --rm verify-kafka-import; then
  log "FAILED: verify-kafka-import"
  compose logs --tail=40 heavydb-setup kafka-setup verify-kafka-import || true
  exit 1
fi

log "All checks passed."
exit 0
