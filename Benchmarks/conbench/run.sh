#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is a Jenkins post-build script for running and recording benchmarks to conbench.

# Assumptions:
# * The PWD is the root git project directory.
# * The current branch is $GIT_BRANCH
# * The build directory is build-$GIT_COMMIT
# * ~/.conbench contains the credentials to the network conbench server.

# If you need to get the last 10 commits:
# git log | egrep '^commit ' | head -10 | awk '{print $2}' | tac > commits.txt

mkdir -p build-$GIT_COMMIT/Tests/tmp

cd Benchmarks/conbench

ln -sf ~/.conbench .

echo "Starting conbench StringDictionaryBenchmark"
conbench StringDictionaryBenchmark --run-name "StringDictionaryBenchmark: $GIT_COMMIT"

echo "Starting conbench TPC_DS_10GB"
TPCDS_ASSETS_DIR=~/tpc-ds conbench TPC_DS_10GB --run-name "TPC_DS_10GB: $GIT_COMMIT"
