# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ASAN sanity_tests (one partition's slice). Source of truth
# for test-asan in .github/workflows/nightly.yml.

export PARENT_CONFIG=asan
export RUNNER=ctest

export LABEL_REGEX="_sanity_"
export LSAN_OPTIONS="suppressions=/workspace/config/asan.suppressions"
export ASAN_OPTIONS="allow_user_poisoning=false"
export NO_ASLR="true"
export CTEST_TIMEOUT_SECS="7200"
