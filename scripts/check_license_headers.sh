#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the license-header pre-commit hooks against the files changed relative to a
# base ref. Used both by CI (.github/workflows/pr-required-checks.yml) and locally:
#
#     scripts/check_license_headers.sh                # compare against origin/master
#     scripts/check_license_headers.sh upstream/master # compare against another ref
#
# With --fix, modifications are applied to the working tree; without it the hooks
# still apply fixes but a non-zero exit means a changed file was missing/outdated.
set -euo pipefail

BASE_REF="${1:-${GITHUB_BASE_REF:-master}}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Resolve the base ref to a concrete commit, fetching it if needed (shallow CI clones).
if ! git rev-parse --verify --quiet "${BASE_REF}^{commit}" >/dev/null; then
  git fetch --quiet origin "${BASE_REF}:${BASE_REF}" 2>/dev/null \
    || git fetch --quiet origin "${BASE_REF}" 2>/dev/null || true
fi
FROM_REF="$(git merge-base "${BASE_REF}" HEAD 2>/dev/null || echo "${BASE_REF}")"

# verify-copyright derives "what changed" from the target branch; make it explicit so
# this works under workflow_dispatch (where GITHUB_BASE_REF is unset) and locally.
export TARGET_BRANCH="${BASE_REF}"

echo "Checking license headers on files changed since ${FROM_REF} (base: ${BASE_REF})"
exec pre-commit run --from-ref "${FROM_REF}" --to-ref HEAD --show-diff-on-failure
