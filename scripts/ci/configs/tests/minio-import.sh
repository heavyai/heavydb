# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test variant: ImportExportTest Minio cases against a minio service.
# Source of truth for test-minio (ImportExportTest) in weekly.yml.

export PARENT_CONFIG=minio
export RUNNER=gtest

export BINARY="Tests/ImportExportTest"
export EXTRA_ARGS="--gtest_filter=*Minio* --run-minio-tests --minio-hostname=minio"
export RESULTS_FILE="test-results-minio-import.xml"
