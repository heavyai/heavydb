# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: render_sanity_tests (needs real GPU + Vulkan stack).
# Source of truth for test-multi-render in
# .github/workflows/pr-required-checks.yml.

export PARENT_CONFIG=multi-render
export RUNNER=ctest

export LABEL_REGEX="render_sanity"
