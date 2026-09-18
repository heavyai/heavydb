#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start HeavyDB with encryption via docker/dev-encrypted-jdbc, run client verification, tear down.
#
# Examples:
#   ./run.sh --build-dir build
#   ./run.sh --build-dir xb
#
# --build-dir is CMAKE_BINARY_DIR (must contain bin/heavydb).
# HEAVYDB_SOURCE is three levels above this script (repo root).
# In-container paths are fixed: /heavydb-internal, /heavydb-build, /tmp/heavydb-data (default storage), /heavydb-ca-info (TLS volume).
# TLS assets live on a named compose volume (removed on exit).
# Host ~/.m2 is bind-mounted for JDBC verify by default; --empty-m2 uses a container-only cache instead.
# By default HeavyDB storage lives inside the container and is removed with compose down.
# Optional host bind mount: --host-storage → storage/encrypted-jdbc on the host.
#
# Requires Docker Compose v2.29+ for `up --wait` (healthcheck on heavydb).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../load-build-env.sh"

DEFAULT_IMAGE='ghcr.io/heavyai/heavydb-internal/core-build-ubuntu22.04-static-cuda12.9.2-x86_64:latest'
USE_HOST_STORAGE="${USE_HOST_STORAGE:-0}"
USE_EMPTY_M2="${USE_EMPTY_M2:-0}"
BUILD_DIR_ARG=""

usage() {
  cat <<EOF
Usage: ${0##*/} --build-dir DIR [--image IMAGE] [--host-storage] [--empty-m2]

Required:
  --build-dir      CMake binary directory (must contain bin/heavydb)

Options:
  --image          Docker image for all services (default: ${DEFAULT_IMAGE})
  --host-storage   Bind-mount storage/encrypted-jdbc on the host (default: container-only storage)
  --empty-m2       Use an empty container-only Maven cache (default: bind-mount host ~/.m2)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || { echo "[encrypted-jdbc] ERROR: --build-dir requires a value" >&2; exit 1; }
      BUILD_DIR_ARG="$2"
      shift 2
      ;;
    --image)
      [[ $# -ge 2 ]] || { echo "[encrypted-jdbc] ERROR: --image requires a value" >&2; exit 1; }
      HEAVYDB_IMAGE="$2"
      shift 2
      ;;
    --host-storage)
      USE_HOST_STORAGE=1
      shift
      ;;
    --empty-m2)
      USE_EMPTY_M2=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "[encrypted-jdbc] ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export HEAVYDB_IMAGE="${HEAVYDB_IMAGE:-${DEFAULT_IMAGE}}"

load_integration_test_build_env "encrypted-jdbc" "${SCRIPT_DIR}"

COMPOSE_DIR="${HEAVYDB_SOURCE}/docker/dev-encrypted-jdbc"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"
COMPOSE_HOST_STORAGE_FILE="${COMPOSE_DIR}/docker-compose.host-storage.yml"
COMPOSE_EMPTY_M2_FILE="${COMPOSE_DIR}/docker-compose.empty-m2.yml"

export HEAVYDB_DATA="${HEAVYDB_SOURCE}/storage/encrypted-jdbc"
if [[ "${USE_EMPTY_M2}" == 1 ]]; then
  export MAVEN_REPO="${MAVEN_REPO:-${HOME}/.m2}"
else
  export MAVEN_REPO="${HOME}/.m2"
fi
export USER_SPEC="$(id -u):$(id -g)"
export HEAVYDB_CONTAINER_NAME="heavydb-dev-encrypted-jdbc"
export HEAVYDB_VERIFY_CONTAINER_NAME="heavydb-dev-encrypted-jdbc-verify"
export HEAVYDB_JDBC_CONTAINER_NAME="heavydb-dev-encrypted-jdbc-jdbc"

prepare_heavydb_data() {
  if [[ -e "${HEAVYDB_DATA}" ]] && [[ ! -w "${HEAVYDB_DATA}" ]]; then
    echo "[encrypted-jdbc] ERROR: ${HEAVYDB_DATA} is not writable." >&2
    echo "[encrypted-jdbc] Docker creates missing bind-mount directories as root." >&2
    echo "[encrypted-jdbc] Remove it, then re-run (sudo rm -rf if needed):" >&2
    echo "[encrypted-jdbc]   rm -rf ${HEAVYDB_DATA}" >&2
    exit 1
  fi
  if [[ -d "${HEAVYDB_DATA}" ]] && find "${HEAVYDB_DATA}" -mindepth 1 ! -writable -print -quit 2>/dev/null | grep -q .; then
    echo "[encrypted-jdbc] ERROR: ${HEAVYDB_DATA} contains files not writable by the current user." >&2
    echo "[encrypted-jdbc] This often happens after a prior run wrote storage as root." >&2
    echo "[encrypted-jdbc] Remove it, then re-run (sudo rm -rf if needed):" >&2
    echo "[encrypted-jdbc]   rm -rf ${HEAVYDB_DATA}" >&2
    exit 1
  fi
  mkdir -p "${HEAVYDB_DATA}"
}

COMPOSE_FILES=(-f "${COMPOSE_FILE}")
if [[ "${USE_HOST_STORAGE}" == 1 ]]; then
  COMPOSE_FILES+=(-f "${COMPOSE_HOST_STORAGE_FILE}")
fi
if [[ "${USE_EMPTY_M2}" == 1 ]]; then
  COMPOSE_FILES+=(-f "${COMPOSE_EMPTY_M2_FILE}")
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
  echo "[encrypted-jdbc] ERROR: docker compose not found" >&2
  exit 1
fi

compose() {
  (cd "${COMPOSE_DIR}" && "${COMPOSE_CMD[@]}" "${COMPOSE_FILES[@]}" "$@")
}

log() {
  echo "[encrypted-jdbc] $*"
}

down() {
  log "Stopping containers..."
  compose down --remove-orphans || true
}

remove_tls_volume() {
  compose volume rm -f encrypted_jdbc_ca_info >/dev/null 2>&1 || true
}

remove_m2_volume() {
  compose volume rm -f encrypted_jdbc_m2_repository >/dev/null 2>&1 || true
}

cleanup() {
  down
  log "Removing TLS volume encrypted_jdbc_ca_info..."
  remove_tls_volume
  if [[ "${USE_EMPTY_M2}" == 1 ]]; then
    log "Removing Maven volume encrypted_jdbc_m2_repository..."
    remove_m2_volume
  fi
}

trap cleanup EXIT INT TERM

log "HEAVYDB_BUILD=${HEAVYDB_BUILD}"
log "HEAVYDB_IMAGE=${HEAVYDB_IMAGE}"
log "TLS volume: encrypted_jdbc_ca_info (removed on exit)"
if [[ "${USE_EMPTY_M2}" == 1 ]]; then
  log "Maven cache: encrypted_jdbc_m2_repository (empty, removed on exit)"
else
  log "MAVEN_REPO=${MAVEN_REPO}"
fi
log "USER_SPEC=${USER_SPEC}"
if [[ "${USE_HOST_STORAGE}" == 1 ]]; then
  log "HeavyDB storage: host bind mount ${HEAVYDB_DATA}"
  prepare_heavydb_data
else
  log "HeavyDB storage: container /tmp/heavydb-data (removed with compose down)"
fi

log "Starting heavydb (wait until healthy)..."
if ! compose up -d --wait heavydb; then
  log "heavydb failed to start; recent logs:"
  compose logs --tail=80 heavydb || true
  exit 1
fi
backend_exit=0
jdbc_exit=0

log "Running verify-backend-encryption..."
if ! compose run --rm verify-backend-encryption; then
  backend_exit=1
fi

log "Running verify-jdbc..."
if ! compose run --rm verify-jdbc; then
  jdbc_exit=1
fi

if (( backend_exit != 0 || jdbc_exit != 0 )); then
  log "FAILED: verify-backend-encryption=${backend_exit}, verify-jdbc=${jdbc_exit}"
  exit 1
fi

log "All checks passed."
exit 0
