# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Docs build helpers and cmd_build_docs.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.
#
# Builds Sphinx HTML into <heavydb_build_dir>/docs/html using the docs Docker
# image (docs/Dockerfile), not the heavydb deps container. Optionally runs
# Doxygen on the host first when a configured heavydb build exists and
# doxygen is on PATH.
#
# Environment:
#   HEAVYDB_SPHINX_IMAGE  Docker image name (default: heavydb-sphinx-doc)

# Build developer docs into $1 (heavydb output dir). Optional Doxygen when
# Doxyfile exists and doxygen is present on the host.
_build_docs() {
  local heavydb_build_dir="$1"
  local docs_dir="$REPO_ROOT/docs"
  local image_name="${HEAVYDB_SPHINX_IMAGE:-heavydb-sphinx-doc}"

  mkdir -p "$heavydb_build_dir"
  : "${_BUILD_LOG_DIR:=$heavydb_build_dir/logs}"
  mkdir -p "$_BUILD_LOG_DIR"

  # Optional Doxygen: only when a cmake-configured heavydb build has a Doxyfile
  # and doxygen is available on the host. Never fails the docs step.
  if [ -f "$heavydb_build_dir/Doxyfile" ]; then
    if command -v doxygen >/dev/null 2>&1; then
      echo "Running optional Doxygen on host..." >&2
      if ! _run_logged "docs doxygen" "${_BUILD_LOG_DIR}/docs-doxygen.log" \
        bash -c "
          set -euo pipefail
          cd \"${heavydb_build_dir}\"
          doxygen Doxyfile
        "; then
        echo "WARN: Doxygen failed; continuing with Sphinx (API pages may be placeholders)." >&2
      fi
    else
      echo "Skipping Doxygen (doxygen not found on host PATH)." >&2
    fi
  else
    echo "Skipping Doxygen (no Doxyfile in $heavydb_build_dir)." >&2
  fi

  local version
  version=$("$REPO_ROOT/scripts/parse-version.sh" 2>/dev/null || true)

  echo "Building Sphinx docs with $image_name..." >&2
  echo "  output: $heavydb_build_dir/docs/html" >&2

  local make_args=(html "BUILDDIR=/build/docs")
  if [ -n "$version" ]; then
    make_args+=("SPHINXOPTS=-j auto -D version=${version}")
  fi

  # Image build/pull (when missing) and Sphinx share docs.log so pull noise
  # stays out of the main shell. Quiet docker build keeps the log compact.
  _run_logged "docs" "${_BUILD_LOG_DIR}/docs.log" \
    bash -c '
      set -euo pipefail
      image_name="$1"
      docs_dir="$2"
      build_dir="$3"
      shift 3
      if ! docker image inspect "$image_name" >/dev/null 2>&1; then
        echo "Building Docker image ${image_name} (quiet)..."
        DOCKER_BUILDKIT=1 docker build -t "$image_name" "$docs_dir"
        echo "Image ${image_name} ready."
      fi
      exec docker run --rm \
        -v "$docs_dir:/doc" \
        -v "$build_dir:/build" \
        -w /doc \
        "$image_name" \
        make "$@"
    ' bash "$image_name" "$docs_dir" "$heavydb_build_dir" "${make_args[@]}"

  echo "HTML output: $heavydb_build_dir/docs/html" >&2
}

cmd_build_docs() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh build docs [options]

Builds Sphinx developer documentation into build/docs/html/
using the Sphinx docs Docker image (not the heavydb deps container).
Optionally runs Doxygen on the host first when a configured heavydb
build (Doxyfile) exists and doxygen is on PATH.

Options:
  --output-dir=<path>    Heavydb build directory (Doxygen XML source and
                         docs HTML destination). Default: build/ inside the repo.
  --verbose, -v          Stream Doxygen and Sphinx output to the shell as well
                         as the log files, instead of printing progress dots

Environment:
  HEAVYDB_SPHINX_IMAGE   Sphinx Docker image name (default: heavydb-sphinx-doc).
                         Built from docs/Dockerfile when missing.
EOF
      return 0 ;;
    esac
  done

  local output_dir=""

  for arg in "$@"; do
    case "$arg" in
      --output-dir=*)  output_dir="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done

  : "${output_dir:=$REPO_ROOT/build}"
  mkdir -p "$output_dir"
  _BUILD_LOG_DIR="$output_dir/logs"
  mkdir -p "$_BUILD_LOG_DIR"

  _build_docs "$output_dir"
}
