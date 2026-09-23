# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: CtasIntegrationTest against a live server with permissive
# import/export paths. Source of truth for test-ctas in weekly.yml.

export PARENT_CONFIG=gpu-release
export RUNNER=server

export SERVER_FLAGS="--allowed-import-paths='[\"/\"]' --allowed-export-paths='[\"/\"]'"
export BINARY="Tests/CtasIntegrationTest"
