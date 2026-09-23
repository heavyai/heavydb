#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Write a Tests/aws/s3client.conf file with the credentials ExecuteTest
# (and friends) expect for S3-backed test cases. Kept out of the build image
# so the secret never lands in a GHCR layer.
#
# Required env vars:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#
# Optional env vars:
#   AWS_CONF_DIR  - directory to write s3client.conf into (default: Tests/aws,
#                   relative to CWD; usually /workspace/Tests/aws when running
#                   inside a pr-build image)
#   AWS_REGION    - AWS region (default: us-west-1)
#
set -euo pipefail

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID must be set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY must be set}"

AWS_CONF_DIR="${AWS_CONF_DIR:-Tests/aws}"
AWS_REGION="${AWS_REGION:-us-west-1}"

mkdir -p "$AWS_CONF_DIR"
cat > "${AWS_CONF_DIR}/s3client.conf" <<EOF
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_REGION=${AWS_REGION}
EOF
