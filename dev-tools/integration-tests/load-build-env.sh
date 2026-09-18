#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set HEAVYDB_BUILD from --build-dir and HEAVYDB_SOURCE from the test script location.
#
# Call load_integration_test_build_env LOG_PREFIX TEST_SCRIPT_DIR after parsing
# --build-dir into BUILD_DIR_ARG.

load_integration_test_build_env() {
  local log_prefix="${1:?}"
  local script_dir="${2:?}"

  if [[ -z "${BUILD_DIR_ARG:-}" ]]; then
    echo "[${log_prefix}] ERROR: --build-dir is required (CMake binary directory)." >&2
    exit 1
  fi

  HEAVYDB_SOURCE="$(cd "${script_dir}/../../.." && pwd)"
  HEAVYDB_BUILD="$(cd "${BUILD_DIR_ARG}" && pwd)"
  export HEAVYDB_SOURCE HEAVYDB_BUILD

  if [[ ! -x "${HEAVYDB_BUILD}/bin/heavydb" ]]; then
    echo "[${log_prefix}] ERROR: ${HEAVYDB_BUILD}/bin/heavydb not found." >&2
    echo "[${log_prefix}] Build HeavyDB in ${HEAVYDB_BUILD}, then re-run." >&2
    exit 1
  fi
}
