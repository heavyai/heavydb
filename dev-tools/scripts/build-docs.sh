# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Docs build helpers and cmd_build_docs.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.
#
# Validates the Fern documentation site under fern/ (docs.yml, narrative pages
# in docs/pages/, images in docs/images/). Fern docs are previewed/published
# via the Fern CLI directly; there is no local static HTML build step here.
# Optionally regenerates the C++ API reference pages under
# docs/pages/api/cpp/ from source using Fern's library docs generator.
#
# Requires the `fern` CLI on PATH: npm install -g fern-api
# --regenerate-api additionally requires Docker.

_build_docs() {
  local regenerate_api="${1:-0}"
  local fern_dir="$REPO_ROOT/fern"

  if ! command -v fern >/dev/null 2>&1; then
    echo "ERROR: fern CLI not found on PATH. Install with: npm install -g fern-api" >&2
    return 1
  fi

  : "${_BUILD_LOG_DIR:=$REPO_ROOT/build/logs}"
  mkdir -p "$_BUILD_LOG_DIR"

  if [ "$regenerate_api" -eq 1 ]; then
    echo "Regenerating C++ API reference pages (fern docs md generate --local)..." >&2
    _run_logged "docs api-generate" "${_BUILD_LOG_DIR}/docs-api-generate.log" \
      bash -c "cd '$fern_dir' && fern docs md generate --local"
  fi

  echo "Validating Fern docs site (fern check)..." >&2
  _run_logged "docs check" "${_BUILD_LOG_DIR}/docs-check.log" \
    bash -c "cd '$fern_dir' && fern check"

  echo "Docs OK." >&2
  echo "  Preview: (cd fern && fern docs dev)" >&2
  echo "  Publish: (cd fern && fern generate --docs)" >&2
}

cmd_build_docs() {
  local regenerate_api=0

  for arg in "$@"; do
    case "$arg" in
      --help|-h)
        cat <<'EOF'
Usage: dev-tools/dev.sh build docs [options]

Validates the Fern documentation site (fern/) with `fern check`. Narrative
pages live under docs/pages/ and images under docs/images/; C++ API
reference pages (docs/pages/api/cpp/) are generated from source via Fern's
library docs generator.

There is no local static HTML build step — Fern docs are previewed with
`fern docs dev` and published with `fern generate --docs` (both run from
the fern/ directory).

Options:
  --regenerate-api   Regenerate the C++ API reference pages before validating
                     (fern docs md generate --local). Requires Docker.

Requires the fern CLI on PATH: npm install -g fern-api
EOF
        return 0 ;;
      --regenerate-api) regenerate_api=1 ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done

  _build_docs "$regenerate_api"
}
