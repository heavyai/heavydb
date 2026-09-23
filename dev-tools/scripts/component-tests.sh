# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Component test helpers: _test_immerse, _test_webserver, _test_heavyiq.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

_test_immerse() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test immerse [options]

Runs the Immerse frontend test suite (ESLint + Jest unit + Jest component tests)
inside the deps container.

Options:
  --deps-image=<image>   Deps container image. Auto-detected if omitted.
  --distro=...           Filter deps image auto-detection.
  --repos-dir=<path>     Parent dir for the immerse clone.
  --ref=<branch|sha>     Immerse ref to test. Default: existing local checkout.
EOF
      return 0 ;;
    esac
  done

  local deps_image="" distro="" repos_dir="" ref=""
  for arg in "$@"; do
    case "$arg" in
      --deps-image=*)  deps_image="${arg#*=}" ;;
      --distro=*)      distro="${arg#*=}" ;;
      --repos-dir=*)   repos_dir="${arg#*=}" ;;
      --ref=*)         ref="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  [ -n "$distro" ] && _validate_distro "$distro"

  : "${repos_dir:=$(cd "$REPO_ROOT/.." && pwd)}"
  local repo_dir="$repos_dir/immerse"
  _ensure_repo "git@github.com:heavyai/immerse.git" "$repo_dir" "$ref"

  NODE_VERSION=$(_read_node_version "$repo_dir")

  local npm_spec
  npm_spec=$(_read_npm_spec "$repo_dir")

  _ensure_tools_node
  local cache_dir="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/tools"
  local node_arch; node_arch=$(uname -m | sed 's/x86_64/x64/; s/aarch64/arm64/')
  local node_dir="$cache_dir/node-v${NODE_VERSION}-linux-${node_arch}"
  local npm_cache="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/npm"
  mkdir -p "$npm_cache"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "static" "$distro")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  local extra_env=()
  [ -n "${PRIVATE_PACKAGES_TOKEN:-}" ] && extra_env+=(-e "PRIVATE_PACKAGES_TOKEN=${PRIVATE_PACKAGES_TOKEN}")
  [ -z "${PRIVATE_PACKAGES_TOKEN:-}" ] \
    && echo "WARN: PRIVATE_PACKAGES_TOKEN not set — private npm packages may fail" >&2
  extra_env+=(-e "NPM_UPGRADE_SPEC=${npm_spec}")

  echo "=== Immerse tests ===" >&2
  docker run --rm \
    -v "$repo_dir:/work" \
    -v "$node_dir:/opt/dev-tools/node:ro" \
    -v "$npm_cache:/root/.npm" \
    -w /work \
    -e USER=root \
    -e PUPPETEER_SKIP_DOWNLOAD=true \
    -e PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    "${extra_env[@]}" \
    "${deps_image}" \
    bash -c '
      set -euo pipefail
      export PATH="/opt/dev-tools/node/bin:$PATH"
      git config --global --add safe.directory /work
      if [ -n "${PRIVATE_PACKAGES_TOKEN:-}" ]; then
        git config --global \
          "url.https://x-access-token:${PRIVATE_PACKAGES_TOKEN}@github.com/.insteadOf" \
          "ssh://git@github.com/"
      fi
      export NPM_CONFIG_PREFIX="$HOME/.npm-global"
      export PATH="$HOME/.npm-global/bin:$PATH"
      npm install -g "${NPM_UPGRADE_SPEC}" --loglevel=error
      hash -r
      npm ci --loglevel=error
      npm test
    '
}

_test_webserver() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test webserver [options]

Runs the web server quality gate (goimports formatting check + golint) inside
the deps container. Note: the webserver repo has no Go unit tests currently.

Options:
  --deps-image=<image>   Deps container image. Auto-detected if omitted.
  --distro=...           Filter deps image auto-detection.
  --repos-dir=<path>     Parent dir for the webserver clone.
  --ref=<branch|sha>     Webserver ref to test. Default: existing local checkout.
EOF
      return 0 ;;
    esac
  done

  local deps_image="" distro="" repos_dir="" ref=""
  for arg in "$@"; do
    case "$arg" in
      --deps-image=*)  deps_image="${arg#*=}" ;;
      --distro=*)      distro="${arg#*=}" ;;
      --repos-dir=*)   repos_dir="${arg#*=}" ;;
      --ref=*)         ref="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  [ -n "$distro" ] && _validate_distro "$distro"

  : "${repos_dir:=$(cd "$REPO_ROOT/.." && pwd)}"
  local repo_dir="$repos_dir/webserver"
  _ensure_repo "git@github.com:heavyai/webserver.git" "$repo_dir" "$ref"

  GO_VERSION=$(_read_go_version "$repo_dir")

  _ensure_tools_go
  local cache_dir="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/tools"
  local go_dir="$cache_dir/go-${GO_VERSION}"
  local go_mod_cache="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/go-mod"
  mkdir -p "$go_mod_cache"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "static" "$distro")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  echo "=== WebServer verify (goimports + golint) ===" >&2
  docker run --rm \
    --network=host \
    -v "$repo_dir:/work" \
    -v "$go_dir:/opt/dev-tools/go:ro" \
    -v "$go_mod_cache:/root/go/pkg/mod" \
    -w /work \
    -e USER=root \
    "${deps_image}" \
    bash -c '
      set -euo pipefail
      export PATH="/opt/dev-tools/go/bin:$PATH"
      export GOPATH="/root/go"
      export GOCACHE="/tmp/go-build"
      export GOPROXY=https://proxy.golang.org,direct
      git config --global --add safe.directory /work
      ./scripts/verify.sh
    '
}

_test_heavyiq() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test heavyiq [--unit] [options]

Runs the HeavyIQ pytest suite inside the deps container.

By default runs all tests; tests requiring a HeavyDB server are automatically
skipped if no server is reachable. Use --unit to explicitly skip server tests.

Options:
  --unit               Skip all tests that need a HeavyDB server (--no-heavydb).
  --deps-image=<image> Deps container image. Auto-detected if omitted.
  --distro=...         Filter deps image auto-detection.
  --repos-dir=<path>   Parent dir for the heavyiq clone.
  --ref=<branch|sha>   HeavyIQ ref to test. Default: existing local checkout.
EOF
      return 0 ;;
    esac
  done

  local deps_image="" distro="" repos_dir="" ref="" unit_only=0
  for arg in "$@"; do
    case "$arg" in
      --unit)          unit_only=1 ;;
      --deps-image=*)  deps_image="${arg#*=}" ;;
      --distro=*)      distro="${arg#*=}" ;;
      --repos-dir=*)   repos_dir="${arg#*=}" ;;
      --ref=*)         ref="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  [ -n "$distro" ] && _validate_distro "$distro"

  : "${repos_dir:=$(cd "$REPO_ROOT/.." && pwd)}"
  local repo_dir="$repos_dir/heavyiq"
  _ensure_repo "git@github.com:heavyai/heavyiq.git" "$repo_dir" "$ref"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "static" "$distro")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  local no_heavydb_flag=""
  [ "$unit_only" -eq 1 ] && no_heavydb_flag="--no-heavydb"

  # HeavyIQ expects repo-root-relative assets plus writable config/data paths.
  # Mount source read-only and construct that layout under container-local /tmp
  # so tests cannot overwrite a developer's checkout or configuration.
  echo "=== HeavyIQ tests${unit_only:+ (unit only)} ===" >&2
  docker run --rm \
    -v "$repo_dir:/work:ro" \
    -w /tmp \
    -e USER=root \
    "${deps_image}" \
    bash -c "
      ${DEV_CONTAINER_DEPS_ENV}
      ${DEV_CONTAINER_PYVENV}
      \"\$PYBIN\" -m venv /tmp/heavyiq-venv
      . /tmp/heavyiq-venv/bin/activate
      pip install -q --upgrade pip
      pip install -q -r /work/requirements.txt -r /work/requirements-dev.txt
      mkdir -p /tmp/heavyiq-test
      shopt -s dotglob nullglob
      for source_path in /work/*; do
        name=\${source_path#/work/}
        if [ "\$name" = config.toml ] || [ "\$name" = test_storage ]; then
          continue
        fi
        ln -s "\$source_path" "/tmp/heavyiq-test/\$name"
      done
      shopt -u dotglob nullglob
      mkdir -p /tmp/heavyiq-test/test_storage
      cp /work/config.pytest.toml /tmp/heavyiq-test/config.toml
      cd /tmp/heavyiq-test
      PYTHONDONTWRITEBYTECODE=1 pytest /work/tests \
        --rootdir=/work \
        -p no:cacheprovider \
        --disable-warnings -rs \
        --config-path=/work/config.pytest.toml \
        ${no_heavydb_flag}
    "
}
