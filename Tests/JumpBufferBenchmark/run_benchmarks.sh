#!/bin/bash

set -e

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
SCRIPT_NAME=$(basename $0)
cat << EOF
Script for running Jump Buffers microbenchmarks.

This script should be executed from within the "build/Tests" directory.
Example: "sh ../../Tests/JumpBufferBenchmark/$SCRIPT_NAME"

Usage:
$SCRIPT_NAME [-h | --help]

Optional Environment Variables:
MIN_TRANSFER_BUFFER_SIZE_MB
MAX_TRANSFER_BUFFER_SIZE_MB
TRANSFER_BUFFER_SIZE_MULTIPLIER

MIN_JUMP_BUFFER_SIZE_MB
MAX_JUMP_BUFFER_SIZE_MB
JUMP_BUFFER_SIZE_MULTIPLIER

MIN_PARALLEL_COPY_THREADS
MAX_PARALLEL_COPY_THREADS
PARALLEL_COPY_THREADS_MULTIPLIER

EOF

  exit 0
fi

BENCH_RUN_ID=$(date +%s)
BENCH_RESULT_FILE_NAME="${BENCH_RUN_ID}_bench_result.json"

START_TIMESTAMP=$(date +%s)
echo "Running benchmarks..."
./JumpBufferBenchmark --benchmark_time_unit=ms --benchmark_out=$BENCH_RESULT_FILE_NAME --benchmark_out_format=json
echo -e "\n\nBenchmark run completed..."
END_TIMESTAMP=$(date +%s)
echo "Benchmark run took $((END_TIMESTAMP - START_TIMESTAMP)) seconds"

echo "Creating benchmark report..."
python3 ../../Tests/JumpBufferBenchmark/create_benchmark_report.py --input-file $BENCH_RESULT_FILE_NAME --tag $BENCH_RUN_ID
echo "Benchmark report created..."

echo -e "Printing benchmark report...\n"
cat "${BENCH_RUN_ID}_bench_report.txt"
