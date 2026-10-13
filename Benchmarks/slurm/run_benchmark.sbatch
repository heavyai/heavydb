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

# Mirror everything (stdout+stderr) into a log next to the results, so progress and errors
# are visible even if Slurm's own --output file is lost.
LOG="$OUTPUT_DIR/run-$${SLURM_JOB_ID}_$${SLURM_ARRAY_TASK_ID:-0}.log"
exec > >(tee -a "$$LOG") 2>&1
finish() {
  rc=$$?
  trap - EXIT TERM INT HUP
  echo "[$$(date '+%F %T')] job finished: exit code $$rc; cleaning up"
  # keep the server log: scratch is node-local and gets deleted
  cp -f "$$SCRATCH/heavydb.log" "$OUTPUT_DIR/heavydb-$${SLURM_JOB_ID}_$${SLURM_ARRAY_TASK_ID:-0}.log" 2>/dev/null || true
  # stop anything the harness left behind (failure, scancel, timeout), then remove scratch
  enroot list 2>/dev/null | grep -E "^hdb-bench-$${SLURM_JOB_ID:-none}-" | xargs -r -n1 enroot remove -f >/dev/null 2>&1 || true
  rm -rf "$$SCRATCH"
}
# signals run the EXIT trap too (scancel/timeout send TERM; give the harness a moment to clean up first)
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
echo "[$$(date '+%F %T')] job $${SLURM_JOB_ID:-?} task $${SLURM_ARRAY_TASK_ID:-0} on $$(hostname)"
echo "data=$$DATA image=$IMAGE scratch=$$SCRATCH"
echo "python: $$(python3 --version 2>&1)  enroot: $$(enroot version 2>&1 | head -1)  gpus: $$(nvidia-smi -L 2>&1 | wc -l)"

cd "$BENCH_ROOT"
$PRE_COMMANDS
python3 -u run_benchmark.py run $BENCHMARK \
  --data "$$DATA" \
  --runtime enroot --image "$IMAGE" \
  --work-dir "$$SCRATCH" \
  --output "$OUTPUT_DIR" \
  --label "$LABEL" \
  $EXTRA_ARGS
