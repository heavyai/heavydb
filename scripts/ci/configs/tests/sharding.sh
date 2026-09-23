# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ExecuteTest with --with-sharding. Source of truth for
# test-sharding in .github/workflows/pr-required-checks.yml.

export PARENT_CONFIG=debug-static
export RUNNER=gtest

export BINARY="Tests/ExecuteTest"
export EXTRA_ARGS="--with-sharding"
export RESULTS_FILE="test-results-sharding.xml"
