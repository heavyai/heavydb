#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Dev-facing CLI for building heavydb and running tests against the same
# configs CI uses. Sources scripts/ci/configs/*.sh (per-job env blocks)
# and delegates to the scripts sourced below for each subcommand.
#
# Usage: dev-tools/dev.sh <subcommand> [args...]
#
# Subcommands:
#   build [target] [options]   Build things — see 'dev.sh build --help'
#   test  [target] [options]   Run tests   — see 'dev.sh test --help'
#   shell [target]             Open a shell — see 'dev.sh shell --help'
#   list                       List CI configs and test variants
#
# Prerequisites:
#   - Docker installed and running.
#   - A local deps image (ghcr.io/heavyai/heavydb/core-build-*) for
#     build and test commands; run 'dev.sh build deps' to build one locally.
#   - Once per machine: 'gh auth refresh -h github.com -s read:packages'
#     (needed to pull CI images for 'shell <config>' / 'test in-image').
#   - nvidia-container-toolkit for GPU access (auto-detected via nvidia-smi).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
CONFIGS_DIR="$SCRIPTS_DIR/ci/configs"
GHCR_OWNER="heavyai"
GHCR_REPO="heavydb"
# Set to 1 by 'build --verbose'; read by _run_logged. Defined here so the
# helpers also work when dev.sh is sourced rather than run through cmd_build.
_BUILD_VERBOSE=0
# NODE_VERSION and GO_VERSION are read from the respective external repos at
# build time (.nvmrc in immerse, toolchain directive in webserver/go.mod).
# They must not be pinned here — bump them in the source repo instead.

usage() {
  cat <<'EOF'
Usage: dev-tools/dev.sh <subcommand> [args...]

Subcommands:
  build [target] [options]   Build things. Targets:
                               (none)    immerse + webserver + geos-dsos + heavydb + docs
                               all       all components + heavydb + docs
                               immerse|webserver|heavyiq|geos-dsos|docs   component only
                               heavydb   heavydb only
                               image     product Docker image from a pre-existing tarball
                               deps      deps container image
                               ci <cfg>  CI-style build
                             Run "dev.sh build --help" for full options.

  test [target] [options]    Run tests. Targets:
                               sanity    sanity tests against existing build (default)
                               all       full test suite
                               immerse   Immerse npm tests
                               webserver webserver verify (lint/format)
                               heavyiq   HeavyIQ pytest suite
                               pyheavydb pyheavydb pytest suite
                               integration-encrypted-jdbc  TLS + JDBC Docker integration test
                               integration-kafka-import    KafkaImporter Docker integration test
                               in-image  run a test variant inside a CI image
                               <variant> CI test variant
                             Run "dev.sh test --help" for full options.

  shell [target]             Open a shell. Targets:
                               (none)    enter deps container
                               <config>  pull + enter a CI image
                               pull <c>  just pull a CI image

  list                       List CI configs and test variants.
EOF
}

# --- Shared helpers -----------------------------------------------------------

require_config() {
  local name="$1" kind="$2" path
  case "$kind" in
    build) path="$CONFIGS_DIR/${name}.sh" ;;
    test)  path="$CONFIGS_DIR/tests/${name}.sh" ;;
    *) echo "internal error: bad config kind '$kind'" >&2; exit 2 ;;
  esac
  if [ ! -f "$path" ]; then
    echo "ERROR: $kind config '$name' not found at $path" >&2
    echo "Run 'dev-tools/dev.sh list' to see available configs." >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

docker_gpus_flag() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    printf '%s' "--gpus all"
  fi
}

# Extract the distro token from an arbitrary string (image name, tarball name,
# etc.). Prints one of ubuntu22.04, rockylinux8, or nothing.
_parse_distro() {
  case "$1" in
    *ubuntu22.04*) echo "ubuntu22.04" ;;
    *rockylinux8*) echo "rockylinux8" ;;
  esac
}

# Resolve the number of parallel jobs from an optional explicit value,
# capping at 24. Usage: nproc=$(_resolve_nproc "$nproc_arg")
_resolve_nproc() {
  local arg="${1:-}"
  local n
  if [ -n "$arg" ]; then
    n="$arg"
  else
    n=$(nproc 2>/dev/null || echo 8)
    [ "$n" -gt 24 ] && n=24
  fi
  echo "$n"
}

# Wipe root-owned files inside a host directory using the deps container.
# Usage: _docker_wipe_dir <deps_image> <host_dir>
_docker_wipe_dir() {
  local image="$1" host_dir="$2"
  docker run --rm -v "${host_dir}:/target" "$image" \
    bash -c 'find /target -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
}

# Validate that effective_distro is one of the known values.
# Usage: _validate_distro <effective_distro> <deps_image>
_validate_distro() {
  local effective_distro="$1" deps_image="${2:-}"
  case "$effective_distro" in
    ubuntu22.04|rockylinux8) ;;
    *)
      echo "ERROR: unsupported distro '${effective_distro:-unknown}'${deps_image:+ (deps image: $deps_image)}" >&2
      echo "Expected one of: ubuntu22.04, rockylinux8" >&2
      exit 1 ;;
  esac
}

# Run a bash script inside the local deps container.
# Usage: _run_in_deps_container <deps_image> <output_dir> <bash_script> [extra docker args...]
#
# Mounts:
#   $REPO_ROOT → /workspace       (read-write; cmake calls git to embed commit hash)
#   $output_dir → /workspace/build (read-write; tests write results here)
#
# Extra docker args (e.g. -v for additional mounts) can be appended after
# the three required positional arguments.
#
# The git common-dir is also mounted when running in a worktree so that git
# inside the container can follow the .git pointer to the main repo objects.
_run_in_deps_container() {
  local deps_image="$1"
  local output_dir="$2"
  local inner_script="$3"
  shift 3
  # Remaining args are passed directly to docker run (e.g. extra -v mounts).

  local gpus
  gpus="$(docker_gpus_flag)"

  # For git worktrees the .git file references an absolute host path; mount
  # the common git dir at that same path so git inside the container can
  # follow it.
  local git_mounts=()
  local git_common_dir
  git_common_dir=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
  [ -n "$git_common_dir" ] && [[ "$git_common_dir" != /* ]] && git_common_dir="$REPO_ROOT/$git_common_dir"
  if [ -n "$git_common_dir" ] && [ "$git_common_dir" != "$REPO_ROOT/.git" ]; then
    git_mounts=(-v "${git_common_dir}:${git_common_dir}:ro")
  fi

  # shellcheck disable=SC2086
  docker run --rm $gpus \
    -w /workspace \
    -v "$REPO_ROOT:/workspace" \
    -v "$output_dir:/workspace/build" \
    "${git_mounts[@]}" \
    "$@" \
    -e USER=root \
    "$deps_image" \
    bash -c "$inner_script"
}

# Source fragment for the deps environment inside a container.
# /usr/local/mapd-deps/mapd-deps.sh is the canonical path on both Ubuntu and
# Rocky Linux; /etc/profile.d/xx-mapd-deps.sh is an identical copy placed
# there so login shells pick it up automatically.
# Use inside a bash -c string passed to _run_in_deps_container.
_DEPS_ENV_SOURCE='[ -e /usr/local/mapd-deps/mapd-deps.sh ] && source /usr/local/mapd-deps/mapd-deps.sh'

# Finds a locally built deps image.
# Usage: _resolve_deps_image [lib_type] [distro] [cuda_version]
#   lib_type:     static (default) or shared
#   distro:       ubuntu22.04, rockylinux8, or "" (any)
#   cuda_version: e.g. 12.9.2, or "" (any)
# All supplied dimensions are used as filters. Exits with an error if no
# image is found or if the result is ambiguous (multiple candidates match).
# Use --deps-image to specify an image explicitly and bypass this function.
_resolve_deps_image() {
  local lib_type="${1:-static}" distro="${2:-}" cuda_version="${3:-}" arch
  arch=$(uname -m)

  # Start with all core-build images for this lib_type and arch.
  # Include the image ID so we can collapse multiple tags of the same image.
  local raw_candidates
  raw_candidates=$(docker images --format '{{.ID}}\t{{.Repository}}:{{.Tag}}' \
    | grep "ghcr.io/heavyai/heavydb/core-build.*${lib_type}.*${arch}" \
    | grep -v '<none>' || true)

  # Narrow by distro if given.
  [ -n "$distro" ] && raw_candidates=$(echo "$raw_candidates" | grep "$distro" || true)

  # Narrow by CUDA version if given.
  [ -n "$cuda_version" ] && raw_candidates=$(echo "$raw_candidates" | grep "cuda${cuda_version}" || true)

  # Deduplicate by image ID: multiple tags of the same image count as one.
  # docker images outputs newest-first, so first-seen = most recently created.
  local candidates
  candidates=$(echo "$raw_candidates" \
    | awk -F'\t' '
        { id=$1; img=$2 }
        !seen[id]++ { print img }
      ' | sort || true)

  local count
  count=$(echo "$candidates" | grep -c . || true)

  if [ "$count" -eq 0 ]; then
    echo "ERROR: no local deps image found for lib_type='$lib_type'" \
         "${distro:+distro='$distro' }${cuda_version:+cuda='$cuda_version' }arch='$arch'" >&2
    echo "Run: dev-tools/dev.sh build deps${distro:+ --distro=$distro}${cuda_version:+ --cuda-version=$cuda_version}" >&2
    exit 1
  fi

  if [ "$count" -gt 1 ]; then
    echo "ERROR: multiple deps images found. Specify one with --deps-image, or narrow the" >&2
    echo "auto-detection by adding the missing dimension(s):" >&2
    [ -z "$distro" ]       && echo "  --distro=ubuntu22.04|rockylinux8" >&2
    [ -z "$cuda_version" ] && echo "  --cuda-version=<ver>   e.g. --cuda-version=12.9.2" >&2
    if [ -z "$distro" ] || [ -z "$cuda_version" ]; then
      echo "If all dimensions are already specified, use --deps-image to pick a specific tag:" >&2
    fi
    echo "Candidates:" >&2
    echo "$candidates" | sed 's/^/  /' >&2
    exit 1
  fi

  echo "$candidates"
}

# Reads the npm upgrade spec from the engines.npm field in a repo's package.json.
# Exits 1 with an error message if the field is absent.
_read_npm_spec() {
  local repo_dir="$1"
  local spec
  spec=$(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get('engines', {}).get('npm', ''))
except Exception as e:
    sys.exit(1)
" "$repo_dir/package.json" 2>/dev/null || true)
  [ -n "$spec" ] || { echo "ERROR: no engines.npm entry found in $repo_dir/package.json" >&2; exit 1; }
  printf 'npm@%s\n' "$spec"
}

# Reads the Go toolchain version from a repo's go.mod (toolchain directive
# preferred, falls back to the go directive). Prints the version and exits 1
# with an error message if neither is found.
_read_go_version() {
  local repo_dir="$1"
  local ver
  ver=$(awk '/^toolchain go/{print substr($2,3);exit} /^go [0-9]/{v=$2} END{if(v)print v}' \
    "$repo_dir/go.mod" 2>/dev/null || true)
  [ -n "$ver" ] || { echo "ERROR: no 'go' or 'toolchain' directive found in $repo_dir/go.mod" >&2; exit 1; }
  printf '%s\n' "$ver"
}

# Reads the Node.js version from a repo's .nvmrc. Prints the version and exits
# 1 with an error message if the file is missing or empty.
_read_node_version() {
  local repo_dir="$1"
  [ -f "$repo_dir/.nvmrc" ] || { echo "ERROR: no .nvmrc found in $repo_dir" >&2; exit 1; }
  local ver
  ver=$(tr -d 'v \t\r\n' < "$repo_dir/.nvmrc")
  [ -n "$ver" ] || { echo "ERROR: .nvmrc in $repo_dir is empty" >&2; exit 1; }
  printf '%s\n' "$ver"
}

# Downloads a Node.js tarball to a local cache dir if not already present.
# Caller must set NODE_VERSION (via _read_node_version) before calling.
# Default cache: <repo-parent>/.heavyai-dev/tools/  (override: HEAVYAI_DEV_CACHE)
# After this call the binary is at:
#   <cache>/node-v${NODE_VERSION}-linux-{arch}/bin/node
_ensure_tools_node() {
  local cache_dir="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/tools"
  mkdir -p "$cache_dir"

  local node_arch; node_arch=$(uname -m | sed 's/x86_64/x64/; s/aarch64/arm64/')
  local node_dir="$cache_dir/node-v${NODE_VERSION}-linux-${node_arch}"
  if [ ! -x "$node_dir/bin/node" ]; then
    echo "Downloading Node.js ${NODE_VERSION} → $cache_dir" >&2
    curl -fsSL \
      "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
      | tar -xJ -C "$cache_dir"
  fi
}

# Downloads a Go tarball to a local cache dir if not already present.
# Caller must set GO_VERSION (via _read_go_version) before calling.
# Default cache: <repo-parent>/.heavyai-dev/tools/  (override: HEAVYAI_DEV_CACHE)
# After this call the binary is at:
#   <cache>/go-${GO_VERSION}/bin/go
_ensure_tools_go() {
  local cache_dir="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/tools"
  mkdir -p "$cache_dir"

  local go_arch; go_arch=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')
  local go_dir="$cache_dir/go-${GO_VERSION}"
  if [ ! -x "$go_dir/bin/go" ]; then
    echo "Downloading Go ${GO_VERSION} → $cache_dir" >&2
    local tmp; tmp=$(mktemp -d "${TMPDIR:-/tmp}/heavyai-go.XXXXXX")
    curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${go_arch}.tar.gz" | tar -xz -C "$tmp"
    mv "$tmp/go" "$go_dir"
    rmdir "$tmp" 2>/dev/null || true
  fi
}

# Clones a repo if it doesn't exist; fetches and checks out $ref when provided.
# If the repo already exists and no ref is requested, uses the current checkout.
_ensure_repo() {
  local url="$1" dest="$2" ref="$3"
  if [ ! -d "$dest/.git" ]; then
    echo "Cloning $url into $dest..." >&2
    git clone "$url" "$dest"
  fi
  if [ -n "$ref" ]; then
    echo "Fetching and checking out $ref in $(basename "$dest")..." >&2
    git -C "$dest" fetch origin
    git -C "$dest" checkout "$ref" 2>/dev/null \
      || git -C "$dest" checkout -b "$ref" "origin/$ref"
  else
    local sha
    sha=$(git -C "$dest" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "Using existing $(basename "$dest") checkout ($sha)" >&2
  fi
}

# Cleans root-owned files and directories inside a component repo clone using
# the deps container (files created inside Docker are owned by root and cannot
# be removed directly from the host). Accepts a repo dir followed by paths
# relative to that dir (files or directories, globs not supported).
_clean_repo_artifacts() {
  local deps_image="$1" repo_dir="$2"
  shift 2
  [ $# -eq 0 ] || [ ! -d "$repo_dir" ] && return 0
  echo "Cleaning artifacts in $repo_dir ..." >&2
  local args=()
  for path in "$@"; do args+=("/target/$path"); done
  docker run --rm -v "$repo_dir:/target" "$deps_image" \
    bash -c "rm -rf $(printf '%s ' "${args[@]}")"
}

# Runs <cmd> [args...], capturing all stdout+stderr in <log_file>.
# Prints "  → log: <log_file>" before starting and, while the command runs, a
# dot every 5 seconds so the shell stays readable. On failure, tails the last
# 50 lines of the log to stderr.
#
# With _BUILD_VERBOSE=1 (dev.sh build --verbose) the output is streamed to the
# shell as it is produced instead of the dots; the log file is still written,
# and the failure path skips the tail since the output is already on screen.
#
# Either way the non-zero exit code is returned so the caller's set -euo
# pipefail fires exactly as if the command had been run inline.
_run_logged() {
  local label="$1" log_file="$2"
  shift 2

  local rc=0
  if [ "$_BUILD_VERBOSE" -eq 1 ]; then
    echo "  → log: $log_file (also streaming below)" >&2
    # PIPESTATUS[0] is the command's own status; tee would otherwise mask it.
    set +e
    "$@" 2>&1 | tee "$log_file" >&2
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
      echo "ERROR: $label failed (exit $rc). Full output above; log: $log_file" >&2
      return "$rc"
    fi
    return 0
  fi

  echo "  → log: $log_file" >&2
  "$@" >"$log_file" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    printf '.' >&2
  done
  printf '\n' >&2
  wait "$pid" || rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "ERROR: $label failed (exit $rc). Last 50 lines of $log_file:" >&2
    tail -n 50 "$log_file" >&2
    return "$rc"
  fi
}

# --- Shared container-script fragments ----------------------------------------
# Reused verbatim inside the `docker run ... bash -c "..."` bodies of the
# source-from-container builds (build-component.sh, pyheavydb-tests.sh) so the
# deps-env and python-venv setup isn't duplicated. Single-quoted so any $ they
# contain is evaluated in the container, not here; interpolate with ${NAME}.

# The image's legacy deps script reads optional variables without nounset guards,
# so enable nounset after sourcing it. Then trust the mounted repo at /work.
DEV_CONTAINER_DEPS_ENV='set -eo pipefail
source /usr/local/mapd-deps/mapd-deps.sh
set -u
git config --global --add safe.directory /work'

# Install a python3 venv toolchain and set PYBIN. Ubuntu uses the distro default
# python3 (3.10 on 22.04 — which HeavyIQ build_prod needs);
# Rocky8's default python3 is too old, so install 3.11 there.
DEV_CONTAINER_PYVENV='if command -v apt-get >/dev/null 2>&1; then
  apt-get update -q >/dev/null
  apt-get install -y -q --no-install-recommends python3-venv >/dev/null
  PYBIN=python3
else
  dnf install -y -q python3.11 python3.11-pip >/dev/null
  PYBIN=python3.11
fi'

# --- Source subcommand implementations ----------------------------------------

# shellcheck source=dev-tools/scripts/ci.sh
source "$SCRIPT_DIR/scripts/ci.sh"
# shellcheck source=dev-tools/scripts/build-deps-image.sh
source "$SCRIPT_DIR/scripts/build-deps-image.sh"
# shellcheck source=dev-tools/scripts/build-component.sh
source "$SCRIPT_DIR/scripts/build-component.sh"
# shellcheck source=dev-tools/scripts/build-docs.sh
source "$SCRIPT_DIR/scripts/build-docs.sh"
# shellcheck source=dev-tools/scripts/build-heavydb.sh
source "$SCRIPT_DIR/scripts/build-heavydb.sh"
# shellcheck source=dev-tools/scripts/build-all.sh
source "$SCRIPT_DIR/scripts/build-all.sh"
# shellcheck source=dev-tools/scripts/sanity-tests.sh
source "$SCRIPT_DIR/scripts/sanity-tests.sh"
# shellcheck source=dev-tools/scripts/pyheavydb-tests.sh
source "$SCRIPT_DIR/scripts/pyheavydb-tests.sh"
# shellcheck source=dev-tools/scripts/component-tests.sh
source "$SCRIPT_DIR/scripts/component-tests.sh"
# shellcheck source=dev-tools/scripts/integration-tests.sh
source "$SCRIPT_DIR/scripts/integration-tests.sh"

# --- Top-level dispatch functions ---------------------------------------------

cmd_test_dispatch() {
  local target="${1:-}"
  case "$target" in
    --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test [target] [options]

Targets:
  sanity [options]            Sanity tests against an existing heavydb build (default)
  all [options]               Full test suite
  pyheavydb [options]         pyheavydb pytest suite (--unit / --integration)
  immerse [options]           Immerse npm tests (ESLint + Jest)
  webserver [options]         WebServer verify (goimports + golint)
  heavyiq [--unit] [options]  HeavyIQ pytest suite
  in-image <cfg> <variant>    Run a test variant inside a pre-built CI image
                              (use when you want to reproduce CI exactly with
                              pre-compiled binaries; no local build needed)
  <variant>                   Run a CI test variant against your local build.
                              Automatically runs inside the deps container from
                              the host; compiles the required test binary first.
                              Requires a prior 'build heavydb' (for CMakeCache).
                              Options: --deps-image=, --build-dir=, --nproc=
  integration-encrypted-jdbc [options]  TLS + JDBC Docker integration test
  integration-kafka-import [options]    KafkaImporter Docker integration test

Run "dev-tools/dev.sh test <target> --help" for target-specific options.
EOF
      return 0 ;;
    "")      cmd_sanity_tests "$@" ;;
    sanity)  shift; cmd_sanity_tests "$@" ;;
    all)       shift; cmd_sanity_tests --all "$@" ;;
    pyheavydb) shift; cmd_pyheavydb_tests "$@" ;;
    immerse)   shift; _test_immerse "$@" ;;
    webserver) shift; _test_webserver "$@" ;;
    heavyiq)   shift; _test_heavyiq "$@" ;;
    integration-encrypted-jdbc) shift; cmd_integration_encrypted_jdbc "$@" ;;
    integration-kafka-import) shift; cmd_integration_kafka_import "$@" ;;
    in-image)  shift; cmd_test_in_image "$@" ;;
    *)         cmd_test "$@" ;;
  esac
}

cmd_shell_dispatch() {
  local target="${1:-}"
  case "$target" in
    --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh shell [target]

Targets:
  (none)          Enter the deps container with the repo mounted at /workspace
  pull <config>   Pull a CI-built image from GHCR
  <config>        Pull (if needed) and drop into a bash shell in a CI image
EOF
      return 0 ;;
    "")   cmd_enter_deps ;;
    pull) shift; cmd_pull "$@" ;;
    *)    cmd_shell "$@" ;;
  esac
}

# --- Shared cmake helper ------------------------------------------------------
#
# Run 'cmake -DENABLE_TESTS=ON' with workarounds needed on all platforms:
#   - git safe.directory: cmake calls git internally; the bind-mounted
#     /workspace is owned by the host user, not the container root.
#   - XercesC_VERSION injection: cmake 3.26 lists XercesC_VERSION as a
#     REQUIRED_VAR but doesn't re-detect it from a cached XercesC_INCLUDE_DIR.
#   - CCACHE_EXE pinning: the NVIDIA container runtime can inject the host's
#     /usr/bin/ccache, causing cmake_check_build_system to activate it and
#     make to fail when the injection disappears at compile time.
#
# Usage: _cmake_enable_tests <build_dir> <src_dir>
# Must be called from within <build_dir>.
_cmake_enable_tests() {
  local build_dir="$1" src_dir="$2"

  git config --global --add safe.directory "$src_dir" 2>/dev/null || true

  local _hdr="" _cached_inc _xerces_ver_arg=()
  _cached_inc=$(grep -m1 '^XercesC_INCLUDE_DIR:' "$build_dir/CMakeCache.txt" 2>/dev/null \
    | cut -d= -f2 || true)
  for _try in \
      "${_cached_inc:+${_cached_inc}/xercesc/util/XercesVersion.hpp}" \
      "${MAPD_DEPS_DIR:+${MAPD_DEPS_DIR}/include/xercesc/util/XercesVersion.hpp}" \
      /usr/local/mapd-deps/include/xercesc/util/XercesVersion.hpp \
      /usr/include/xercesc/util/XercesVersion.hpp; do
    [ -n "$_try" ] && [ -f "$_try" ] && { _hdr="$_try"; break; }
  done
  if [ -n "$_hdr" ]; then
    local _maj _min _rev
    _maj=$(awk '/^#define XERCES_VERSION_MAJOR/{print $3;exit}'    "$_hdr")
    _min=$(awk '/^#define XERCES_VERSION_MINOR/{print $3;exit}'    "$_hdr")
    _rev=$(awk '/^#define XERCES_VERSION_REVISION/{print $3;exit}' "$_hdr")
    [ -n "$_maj" ] && [ -n "$_min" ] && [ -n "$_rev" ] && \
      _xerces_ver_arg=("-DXercesC_VERSION=${_maj}.${_min}.${_rev}")
  fi

  cmake -DENABLE_TESTS=ON -DCCACHE_EXE=CCACHE_EXE-NOTFOUND \
    "${_xerces_ver_arg[@]}" "$src_dir"
}

# --- Dispatcher ---------------------------------------------------------------
# Guard: only run the CLI dispatch when executed directly, not when sourced.
# Sourcing dev.sh (e.g. from a bash -c inner script) gives access to all the
# shared helper functions above without triggering the CLI.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
cmd="${1:-}"
shift || true
case "$cmd" in
  # Primary interface
  build)             cmd_build "$@" ;;
  test)              cmd_test_dispatch "$@" ;;
  shell)             cmd_shell_dispatch "$@" ;;
  list)              cmd_list "$@" ;;
  ""|help|-h|--help) usage ;;
  *)
    echo "Unknown subcommand: $cmd" >&2
    usage
    exit 1
    ;;
esac
fi
