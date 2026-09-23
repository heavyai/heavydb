# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: sanity_tests in release+CUDA mode (one partition slice).
# Source of truth for test-multi-cuda-gcc in
# .github/workflows/pr-required-checks.yml.

export PARENT_CONFIG=multi-cuda-gcc
export RUNNER=ctest

export LABEL_REGEX="_sanity_"
export CTEST_TIMEOUT_SECS="7200"
