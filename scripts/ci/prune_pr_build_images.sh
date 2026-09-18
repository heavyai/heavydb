#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Delete GHCR container image versions older than MAX_AGE_DAYS, while always
# preserving at least MIN_VERSIONS_TO_KEEP of the most-recent versions as a
# safety net. Used by .github/workflows/prune-pr-build-images.yml.
#
# Required env vars:
#   PACKAGE_NAME            - GHCR package path under the org/user, INCLUDING
#                             the repo segment for repo-linked packages.
#                             For an image pushed to
#                             ghcr.io/heavyai/heavydb/pr-build/debug-static
#                             this is "heavydb/pr-build/debug-static".
#
# Optional env vars:
#   OWNER                   - GHCR org or user (default: $GITHUB_REPOSITORY_OWNER)
#   MAX_AGE_DAYS            - default 14
#   MIN_VERSIONS_TO_KEEP    - default 2 (newest versions are always kept,
#                             regardless of age)
#   DRY_RUN                 - "true" logs what would be deleted; default false
#   GH_TOKEN                - token with packages:write; in CI this is
#                             $GITHUB_TOKEN
#
set -euo pipefail

: "${PACKAGE_NAME:?PACKAGE_NAME must be set}"

OWNER="${OWNER:-${GITHUB_REPOSITORY_OWNER:-}}"
if [ -z "$OWNER" ]; then
  echo "ERROR: OWNER must be set (or GITHUB_REPOSITORY_OWNER inferred)" >&2
  exit 1
fi

MAX_AGE_DAYS="${MAX_AGE_DAYS:-14}"
MIN_VERSIONS_TO_KEEP="${MIN_VERSIONS_TO_KEEP:-2}"
DRY_RUN="${DRY_RUN:-false}"

if ! command -v gh >/dev/null; then
  echo "ERROR: gh CLI not found in PATH" >&2
  exit 1
fi
if ! command -v jq >/dev/null; then
  echo "ERROR: jq not found in PATH" >&2
  exit 1
fi

cutoff_epoch="$(date -u -d "${MAX_AGE_DAYS} days ago" +%s)"
cutoff_iso="$(date -u -d "@${cutoff_epoch}" --iso-8601=seconds)"

# URL-encode the package name (it contains "/").
encoded_name="$(printf '%s' "$PACKAGE_NAME" | jq -sRr '@uri')"
endpoint="/orgs/${OWNER}/packages/container/${encoded_name}/versions"

echo "Pruning ${OWNER}/${PACKAGE_NAME}: deleting versions older than ${cutoff_iso} (${MAX_AGE_DAYS} days), keeping at least ${MIN_VERSIONS_TO_KEEP} newest."

# Probe first — package may not exist yet on the very first run of a new
# build config. A 404 here means "no work to do," not "failure."
if ! gh api "$endpoint" --silent >/dev/null 2>&1; then
  echo "Package ${OWNER}/${PACKAGE_NAME} not found (yet) or inaccessible. Skipping."
  exit 0
fi

# `gh api --paginate` returns one JSON array per page, so we --slurp into a
# single flat array, sort by updated_at desc, drop the first N to honor the
# safety floor, and emit the rest as one JSON object per line.
candidates="$(
  gh api --paginate "$endpoint" \
    | jq -s --argjson keep "$MIN_VERSIONS_TO_KEEP" '
        add // []
        | sort_by(.updated_at) | reverse
        | .[$keep:]
        | .[]
        | {id, updated_at, tags: (.metadata.container.tags // [])}
      ' \
    | jq -c .
)"

if [ -z "$candidates" ]; then
  echo "No versions to consider (package empty or only contains safety-net entries)."
  exit 0
fi

deleted=0
kept=0
while IFS= read -r v; do
  [ -z "$v" ] && continue
  id="$(printf '%s' "$v" | jq -r '.id')"
  updated_at="$(printf '%s' "$v" | jq -r '.updated_at')"
  tags="$(printf '%s' "$v" | jq -r '.tags | join(",")')"
  ts="$(date -u -d "$updated_at" +%s)"
  if [ "$ts" -lt "$cutoff_epoch" ]; then
    if [ "$DRY_RUN" = "true" ]; then
      echo "[DRY] would delete id=${id} updated_at=${updated_at} tags=${tags}"
    else
      echo "deleting id=${id} updated_at=${updated_at} tags=${tags}"
      gh api -X DELETE "${endpoint}/${id}" --silent
    fi
    deleted=$((deleted + 1))
  else
    kept=$((kept + 1))
  fi
done <<< "$candidates"

echo "Done. ${deleted} deleted, ${kept} kept (newer than cutoff)."
