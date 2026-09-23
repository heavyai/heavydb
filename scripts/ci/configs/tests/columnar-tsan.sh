# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ExecuteTest --enable-columnar-output under TSAN. Source of
# truth for test-columnar-tsan in .github/workflows/nightly.yml. Reuses the
# tsan build image.

export PARENT_CONFIG=columnar-tsan
export RUNNER=gtest

export BINARY="Tests/ExecuteTest"
export EXTRA_ARGS="--enable-columnar-output=true"
export RESULTS_FILE="test-results-columnar-tsan.xml"
export TSAN_OPTIONS="suppressions=/workspace/config/tsan.suppressions,history_size=7,second_deadlock_stack=1,halt_on_error=0"
export NO_ASLR="true"
