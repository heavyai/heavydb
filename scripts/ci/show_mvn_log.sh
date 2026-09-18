#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Tail the captured Maven build log into the GitHub Actions step output,
# grouped so it folds in the UI. Invoked as a post-failure step from
# .github/workflows/pr-required-checks.yml. Useful because CMakeLists.txt
# tells mvn to write to build/mvn_build.log with -l, so a Maven failure
# otherwise leaves the GHA log silent.
#
# Optional env vars:
#   BUILD_DIR  - directory containing mvn_build.log (default: build)
#   TAIL_LINES - number of trailing lines to print (default: 400)
#
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
TAIL_LINES="${TAIL_LINES:-400}"

log="${BUILD_DIR}/mvn_build.log"
echo "::group::tail of ${log}"
if [ -f "$log" ]; then
  tail -n "$TAIL_LINES" "$log"
else
  echo "(no ${log} present)"
fi
echo "::endgroup::"
