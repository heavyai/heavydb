#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a benchmark locally in the HeavyDB product Docker image.
#   run_local.sh <benchmark> <dataset-root> [run_benchmark.py options...]
# e.g. run_local.sh tpch /home/misiug/Data/scale-1 --queries q01,q06 --iterations 5
# Image: $HEAVYDB_IMAGE, else the newest local heavyai-* image (see README).
set -euo pipefail
[ $# -ge 2 ] || { sed -n '5,8p' "$0"; exit 1; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="$1"; data="$2"; shift 2
exec python3 -u "$here/run_benchmark.py" run "$bench" --data "$data" --runtime docker "$@"
