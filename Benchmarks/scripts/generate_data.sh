#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate a benchmark dataset (Parquet, decimal types, partitioned into
# <out>/<table>/<table>-N.parquet + metadata.json) with velox-testing's generator
# (https://github.com/rapidsai/velox-testing, benchmark_data_tools/).
#
#   VELOX_TESTING_DIR=/path/to/velox-testing \
#     generate_data.sh <scale-factor> <output-dir> [generate_data_files.py options]
#
# e.g.  generate_data.sh 10 /home/misiug/Data/tpch-sf10-decimal -j 16
# Do NOT pass --convert-decimals-to-floats: the benchmarks want decimal columns.
# Needs tpchgen-cli (on PATH or in <velox-testing>/benchmark_data_tools/.local_installs/bin;
# install with benchmark_data_tools/scripts/install_tpchgen_cli.sh) and python3 with duckdb.
set -euo pipefail

[ $# -ge 2 ] || { sed -n '5,14p' "$0"; exit 1; }
sf="$1"; out="$2"; shift 2
vt="${VELOX_TESTING_DIR:-}"
[ -n "$vt" ] && [ -f "$vt/benchmark_data_tools/generate_data_files.py" ] || {
  echo "set VELOX_TESTING_DIR to a velox-testing checkout" >&2; exit 1; }
# generate_data_files.py deletes its output directory first: never point it at existing data.
[ ! -e "$out" ] || { echo "refusing to overwrite existing $out" >&2; exit 1; }
for a in "$@"; do
  [ "$a" != "--convert-decimals-to-floats" ] && [ "$a" != "-c" ] || {
    echo "decimal columns are required; drop $a" >&2; exit 1; }
done

export PATH="$vt/benchmark_data_tools/.local_installs/bin:$PATH"
command -v tpchgen-cli >/dev/null || { echo "tpchgen-cli not found (see header)" >&2; exit 1; }
exec python3 "$vt/benchmark_data_tools/generate_data_files.py" \
  -b tpch -s "$sf" -d "$out" "$@"
