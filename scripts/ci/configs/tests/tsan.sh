# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: TSAN sanity_tests (one partition's slice). Source of
# truth for test-tsan in .github/workflows/pr-required-checks.yml.
#
# TSAN_OPTIONS references /workspace/config/tsan.suppressions, which is
# the path inside a pr-build image or inside `dev-tools/dev.sh enter-deps`
# (which mounts the checkout at /workspace). Host-native invocations
# will need to override TSAN_OPTIONS.

export PARENT_CONFIG=tsan
export RUNNER=ctest

export LABEL_REGEX="_sanity_"
export TSAN_OPTIONS="suppressions=/workspace/config/tsan.suppressions,history_size=7,second_deadlock_stack=1,halt_on_error=0"
export NO_ASLR="true"
