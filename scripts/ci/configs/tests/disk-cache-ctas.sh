# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: CtasUpdateTest --use-disk-cache under TSAN. Source of truth for
# test-disk-cache-ctas in weekly.yml. Reuses the disk-cache-tsan build.

export PARENT_CONFIG=disk-cache-tsan
export RUNNER=gtest

export BINARY="Tests/CtasUpdateTest"
export EXTRA_ARGS="--use-disk-cache"
export RESULTS_FILE="test-results-disk-cache-ctas.xml"
export TSAN_OPTIONS="suppressions=/workspace/config/tsan.suppressions,history_size=7,second_deadlock_stack=1,halt_on_error=1"
export NO_ASLR="true"
