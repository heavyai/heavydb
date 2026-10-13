#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Template rendered by slurm/submit.py (string.Template placeholders; shell variables are written with a doubled dollar sign).
# One job = one self-contained node run: heavydb starts inside an enroot container
# on the allocated node, the benchmark runs against it, results land on shared storage.
$SBATCH_DIRECTIVES

set -euo pipefail

DATASETS=($DATASETS)
DATA="$${DATASETS[$${SLURM_ARRAY_TASK_ID:-0}]}"
SCRATCH="$${SLURM_TMPDIR:-$${TMPDIR:-/tmp}}/hdb-bench-$${SLURM_JOB_ID}"
mkdir -p "$$SCRATCH" "$OUTPUT_DIR"

cd "$BENCH_ROOT"
$PRE_COMMANDS
python3 -u run_benchmark.py run $BENCHMARK \
  --data "$$DATA" \
  --runtime enroot --image "$IMAGE" \
  --work-dir "$$SCRATCH" \
  --output "$OUTPUT_DIR" \
  --label "$LABEL" \
  $EXTRA_ARGS
rm -rf "$$SCRATCH"
