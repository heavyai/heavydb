# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ExecuteTest with --enable-columnar-output=true. Source
# of truth for test-columnar in .github/workflows/pr-required-checks.yml.

export PARENT_CONFIG=debug-static
export RUNNER=gtest

export BINARY="Tests/ExecuteTest"
export EXTRA_ARGS="--enable-columnar-output=true"
export RESULTS_FILE="test-results-columnar.xml"
