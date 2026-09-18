# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: UpdelStorageTest --use-disk-cache under TSAN. Source of truth for
# test-disk-cache-updel in weekly.yml. Reuses the disk-cache-tsan build.

export PARENT_CONFIG=disk-cache-tsan
export RUNNER=gtest

export BINARY="Tests/UpdelStorageTest"
export EXTRA_ARGS="--use-disk-cache"
export RESULTS_FILE="test-results-disk-cache-updel.xml"
export TSAN_OPTIONS="suppressions=/workspace/config/tsan.suppressions,history_size=7,second_deadlock_stack=1,halt_on_error=1"
export NO_ASLR="true"
