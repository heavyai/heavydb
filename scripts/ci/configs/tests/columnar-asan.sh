# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ExecuteTest --enable-columnar-output under ASAN. Source of
# truth for test-columnar-asan in .github/workflows/nightly.yml. Reuses the
# asan build image.

export PARENT_CONFIG=columnar-asan
export RUNNER=gtest

export BINARY="Tests/ExecuteTest"
export EXTRA_ARGS="--enable-columnar-output=true"
export RESULTS_FILE="test-results-columnar-asan.xml"
export LSAN_OPTIONS="suppressions=/workspace/config/asan.suppressions"
export ASAN_OPTIONS="allow_user_poisoning=false"
export NO_ASLR="true"
