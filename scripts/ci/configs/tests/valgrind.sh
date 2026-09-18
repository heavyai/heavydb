# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ExecuteTest under valgrind memcheck. Source of truth for
# test-valgrind in weekly.yml.

export PARENT_CONFIG=valgrind
export RUNNER=valgrind

export BINARY="Tests/ExecuteTest"
export RESULTS_FILE="test-results-valgrind.xml"
# Skip two string-pattern tests that are known-noisy under valgrind (QE-1018).
export GTEST_FILTER="-Select.CanReuseCompiledCodeForStringPatternMatchQuery:Select.IREquivalenceForStringPatternMatchQuery"
