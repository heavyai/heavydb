# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ForeignTableDmlTest VirtualAddressing cases against minio.
# Source of truth for test-minio (ForeignTableDmlTest) in weekly.yml.

export PARENT_CONFIG=minio
export RUNNER=gtest

export BINARY="Tests/ForeignTableDmlTest"
export EXTRA_ARGS="--gtest_filter=*VirtualAddressing* --run-minio-tests --minio-hostname=minio"
export RESULTS_FILE="test-results-minio-foreign.xml"
