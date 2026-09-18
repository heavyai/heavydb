# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: GeospatialTest across all UTM zones (no server needed).
# Source of truth for test-extended-geo in weekly.yml.

export PARENT_CONFIG=gpu-release
export RUNNER=gtest

export BINARY="Tests/GeospatialTest"
export EXTRA_ARGS="--all-utm-zones"
export RESULTS_FILE="test-results-extended-geo.xml"
