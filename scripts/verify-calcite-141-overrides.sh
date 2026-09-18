#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Verifies the vendor-only Calcite 1.41 override state.
#
# This is intended to be run on the Calcite override/vendor PR. It is expected
# to report mismatches on follow-up compatibility branches that intentionally
# modify these override files after vendoring them.

CALCITE_VERSION="${CALCITE_VERSION:-1.41.0}"
CALCITE_REF="${CALCITE_REF:-calcite-${CALCITE_VERSION}}"
CALCITE_RAW_BASE="https://raw.githubusercontent.com/apache/calcite/${CALCITE_REF}/core/src/main/java"

# calcite-core 1.41.0 depends on avatica-core 1.27.0. The raw URL uses the
# release tag commit for refs/tags/rel/avatica-1.27.0.
AVATICA_VERSION="${AVATICA_VERSION:-1.27.0}"
AVATICA_REF="${AVATICA_REF:-7754d942f858e5521966c1771cf2e111e8a7ef87}"
AVATICA_RAW_BASE="https://raw.githubusercontent.com/apache/calcite-avatica/${AVATICA_REF}/core/src/main/java"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
work_dir="$(mktemp -d /tmp/calcite-141-override-check.XXXXXX)"

cleanup() {
  local tmp_leaf
  tmp_leaf="$(basename "${work_dir}")"
  case "${tmp_leaf}" in
    calcite-141-override-check.*) rm -fr /tmp/"${tmp_leaf}" ;;
  esac
}
trap cleanup EXIT

require_tool() {
  local tool="$1"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: required tool '${tool}' is not available" >&2
    exit 2
  fi
}

require_tool curl
require_tool cmp
require_tool diff
require_tool git
require_tool mktemp
require_tool sed

failures=0

check_remote_file() {
  local source_name="$1"
  local raw_base="$2"
  local package_path="$3"
  local local_path="java/calcite/src/main/java/${package_path}"
  local local_file="${repo_root}/${local_path}"
  local upstream_file="${work_dir}/${source_name}/${package_path}"
  local url="${raw_base}/${package_path}"

  mkdir -p "$(dirname "${upstream_file}")"

  if [[ ! -f "${local_file}" ]]; then
    echo "MISSING ${local_path}"
    failures=1
    return
  fi

  curl -fsSL --retry 3 --retry-delay 1 -o "${upstream_file}" "${url}"

  if cmp -s "${upstream_file}" "${local_file}"; then
    echo "OK ${local_path} (${source_name})"
    return
  fi

  echo "MISMATCH ${local_path} (${source_name})"
  echo "  upstream: ${url}"
  diff -u "${upstream_file}" "${local_file}" | sed -n '1,120p' || true
  failures=1
}

check_calcite_core() {
  check_remote_file "calcite-core-${CALCITE_VERSION}" "${CALCITE_RAW_BASE}" "$1"
}

check_avatica_core() {
  check_remote_file "avatica-core-${AVATICA_VERSION}" "${AVATICA_RAW_BASE}" "$1"
}

echo "Checking Calcite 1.41 override files against Apache release sources..."
echo

check_avatica_core org/apache/calcite/avatica/util/DateTimeUtils.java
check_calcite_core org/apache/calcite/prepare/PlannerImpl.java
check_calcite_core org/apache/calcite/rex/RexSimplify.java
check_calcite_core org/apache/calcite/sql/fun/SqlArrayValueConstructor.java
check_calcite_core org/apache/calcite/sql/fun/SqlSingleValueAggFunction.java
check_calcite_core org/apache/calcite/sql/fun/SqlStdOperatorTable.java
check_calcite_core org/apache/calcite/sql/type/SqlTypeFactoryImpl.java
check_calcite_core org/apache/calcite/sql/validate/SqlValidatorImpl.java
check_calcite_core org/apache/calcite/sql2rel/SqlToRelConverter.java
check_calcite_core org/apache/calcite/sql2rel/StandardConvertletTable.java

if [[ "${failures}" -ne 0 ]]; then
  echo
  echo "One or more vendored override files differ from the Apache release source."
  exit 1
fi

echo
echo "All vendored Calcite override files match the Apache release sources byte-for-byte."
