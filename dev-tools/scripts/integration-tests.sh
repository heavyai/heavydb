# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Docker integration test entry points for dev-tools/dev.sh.
# Sourced by dev-tools/dev.sh — REPO_ROOT and SCRIPT_DIR are available here.
#
# Each cmd_* function execs the matching run.sh under dev-tools/integration-tests/.
# Does not source test env.sh or export test variables. --build-dir must be
# supplied on the command line.

cmd_integration_encrypted_jdbc() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test integration-encrypted-jdbc [run.sh options]

TLS + JDBC client Docker integration test.

Required run.sh option:
  --build-dir DIR   CMake binary directory (must contain bin/heavydb)

All other flags are forwarded unchanged. See dev-tools/integration-tests/README.md.

Example:
  dev-tools/dev.sh test integration-encrypted-jdbc --build-dir build --host-storage
EOF
      return 0
      ;;
    esac
  done

  local run_sh="$SCRIPT_DIR/integration-tests/encrypted-jdbc/run.sh"
  if [[ ! -x "$run_sh" ]]; then
    echo "ERROR: encrypted-jdbc runner not found at $run_sh" >&2
    exit 1
  fi
  exec "$run_sh" "$@"
}

cmd_integration_kafka_import() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test integration-kafka-import [run.sh options]

KafkaImporter Docker integration test.

Required run.sh option:
  --build-dir DIR   CMake binary directory (must contain bin/heavydb)

All other flags are forwarded unchanged. See dev-tools/integration-tests/README.md.

Example:
  dev-tools/dev.sh test integration-kafka-import --build-dir build --host-storage
EOF
      return 0
      ;;
    esac
  done

  local run_sh="$SCRIPT_DIR/integration-tests/kafka-import/run.sh"
  if [[ ! -x "$run_sh" ]]; then
    echo "ERROR: kafka-import runner not found at $run_sh" >&2
    exit 1
  fi
  exec "$run_sh" "$@"
}
