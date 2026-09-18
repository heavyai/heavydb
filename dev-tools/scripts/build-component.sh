# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Component build helpers and cmd_build_component.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

# Builds the pyheavydb wheel from source inside the deps container (thrift
# codegen + python -m build). The wheel lands at
# <repos_dir>/pyheavydb/dist/pyheavydb-*.whl.
_build_pyheavydb_wheel() {
  local repos_dir="$1" ref="$2" deps_image="$3" requirement="$4"
  local repo_dir="$repos_dir/pyheavydb"
  _ensure_repo "git@github.com:heavyai/pyheavydb.git" "$repo_dir" "$ref"

  echo "Building pyheavydb wheel from source in deps container..." >&2
  _run_logged "pyheavydb wheel" "${_BUILD_LOG_DIR}/pyheavydb.log" \
    docker run --rm \
      -v "$repo_dir:/work" \
      -w /work \
      -e USER=root \
      -e "PYHEAVYDB_REQUIREMENT=$requirement" \
      "$deps_image" \
      bash -c "
        ${DEV_CONTAINER_DEPS_ENV}
        echo \"pyheavydb bindings generated with \$(thrift -version)\"
        ${DEV_CONTAINER_PYVENV}
        rm -rf dist venv
        \"\$PYBIN\" -m venv venv && . venv/bin/activate
        pip install -q --upgrade pip build
        # Makefile: make thrift (codegen via the deps image thrift) + python -m build.
        make build
        mkdir -p /tmp/pyheavydb-validation
        pip download -q --no-index --find-links=dist --only-binary=:all: \
          --no-deps --dest /tmp/pyheavydb-validation \
          \${PYHEAVYDB_REQUIREMENT}
      "
}

# Reads the single pyheavydb requirement declared by HeavyIQ. The declaration,
# including its version constraint, remains authoritative in every explicit source mode.
_heavyiq_pyheavydb_requirement() {
  local requirements_file="$1"
  local matches
  matches=$(grep -Ei '^[[:space:]]*pyheavydb([[:space:]]|[<>=!~])' "$requirements_file" || true)

  local count
  count=$(printf '%s\n' "$matches" | grep -c . || true)
  if [ "$count" -ne 1 ]; then
    echo "ERROR: expected exactly one pyheavydb declaration in $requirements_file" >&2
    printf '%s\n' "$matches" >&2
    return 1
  fi

  local requirement="${matches%%#*}"
  requirement=$(printf '%s' "$requirement" | xargs)
  printf '%s\n' "$requirement"
}

_find_single_pyheavydb_wheel() {
  local wheel_dir="$1"
  local wheels=()
  mapfile -t wheels < <(find "$wheel_dir" -maxdepth 1 -name 'pyheavydb-*.whl' -type f | sort)
  if [ "${#wheels[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one pyheavydb wheel in $wheel_dir" >&2
    printf '  %s\n' "${wheels[@]}" >&2
    return 1
  fi
  printf '%s\n' "${wheels[0]}"
}

_download_pyheavydb_wheel() {
  local requirement="$1" source="$2" wheel_dir="$3" deps_image="$4"
  local index_url
  case "$source" in
    pypi)     index_url="https://pypi.org/simple/" ;;
    testpypi) index_url="https://test.pypi.org/simple/" ;;
    *) echo "ERROR: unsupported pyheavydb package index: $source" >&2; return 1 ;;
  esac

  rm -rf "$wheel_dir"
  mkdir -p "$wheel_dir"
  echo "Downloading $requirement from $source..." >&2
  _run_logged "pyheavydb ($source)" "${_BUILD_LOG_DIR}/pyheavydb-${source}.log" \
    docker run --rm \
      -v "$wheel_dir:/out" \
      -e "PYHEAVYDB_REQUIREMENT=$requirement" \
      "$deps_image" \
      bash -c "
        set -euo pipefail
        ${DEV_CONTAINER_PYVENV}
        \${PYBIN} -m venv /tmp/pyheavydb-download-venv
        . /tmp/pyheavydb-download-venv/bin/activate
        pip install -q --upgrade pip
        pip --isolated download -q --only-binary=:all: --no-deps \
          --index-url '$index_url' \
          --dest /out \
          \${PYHEAVYDB_REQUIREMENT}
      "
}

_build_heavyiq() {
  local repos_dir="$1" ref="$2" output_dir="$3" distro="$4" deps_image="$5"
  local pyheavydb_source="$6" pyheavydb_ref="$7"
  local repo_dir="$repos_dir/heavyiq"
  _ensure_repo "git@github.com:heavyai/heavyiq.git" "$repo_dir" "$ref"

  case "$pyheavydb_source" in
    project|pypi|testpypi|build-local-wheel) ;;
    *)
      echo "ERROR: --pyheavydb-source must be project, pypi, testpypi, or build-local-wheel" >&2
      return 1
      ;;
  esac

  local pyheavydb_requirement=""
  if [ "$pyheavydb_source" != "project" ]; then
    pyheavydb_requirement=$(_heavyiq_pyheavydb_requirement "$repo_dir/requirements.txt")
    echo "HeavyIQ declares $pyheavydb_requirement" >&2
    if ! grep -q -- '--pyheavydb-wheel=' "$repo_dir/scripts/common_fn.sh"; then
      echo "ERROR: explicit pyheavydb sources require HeavyIQ's --pyheavydb-wheel hook." >&2
      echo "Use a HeavyIQ ref that includes the authoritative dependency change." >&2
      return 1
    fi
  fi

  local pyheavydb_wheel="" pyheavydb_dist=""
  case "$pyheavydb_source" in
    project)
      if [ -n "$pyheavydb_ref" ]; then
        echo "ERROR: --pyheavydb-ref requires --pyheavydb-source=build-local-wheel" >&2
        return 1
      fi
      echo "Using HeavyIQ's declared dependency resolver configuration." >&2
      ;;
    pypi|testpypi)
      if [ -n "$pyheavydb_ref" ]; then
        echo "ERROR: --pyheavydb-ref cannot be combined with --pyheavydb-source=$pyheavydb_source" >&2
        return 1
      fi
      pyheavydb_dist="$output_dir/.python-deps/pyheavydb-$pyheavydb_source"
      _download_pyheavydb_wheel \
        "$pyheavydb_requirement" "$pyheavydb_source" "$pyheavydb_dist" "$deps_image"
      pyheavydb_wheel=$(_find_single_pyheavydb_wheel "$pyheavydb_dist")
      ;;
    build-local-wheel)
      _build_pyheavydb_wheel \
        "$repos_dir" "$pyheavydb_ref" "$deps_image" "$pyheavydb_requirement"
      pyheavydb_dist="$repos_dir/pyheavydb/dist"
      pyheavydb_wheel=$(_find_single_pyheavydb_wheel "$pyheavydb_dist")
      ;;
  esac

  local pyheavydb_wheel_name=""
  local pyheavydb_docker_args=()
  if [ -n "$pyheavydb_wheel" ]; then
    pyheavydb_wheel_name=$(basename "$pyheavydb_wheel")
    pyheavydb_docker_args+=(
      -v "$pyheavydb_dist:/pyheavydb-dist:ro"
      -e "PYHEAVYDB_WHEEL_NAME=$pyheavydb_wheel_name"
    )
  fi

  local build_script
  case "$distro" in
    rockylinux8) build_script="./scripts/build_prod_rhel.sh" ;;
    *)           build_script="./scripts/build_prod.sh" ;;
  esac

  echo "Building HeavyIQ in deps container..." >&2
  _run_logged "heavyiq" "${_BUILD_LOG_DIR}/heavyiq.log" \
    docker run --rm \
      -v "$repo_dir:/work" \
      "${pyheavydb_docker_args[@]}" \
      -w /work \
      -e USER=root \
      "$deps_image" \
      bash -c "
        ${DEV_CONTAINER_DEPS_ENV}
        ${DEV_CONTAINER_PYVENV}
        rm -rf venv packages
        build_args=()
        if [ -n \"\${PYHEAVYDB_WHEEL_NAME:-}\" ]; then
          build_args+=(\"--pyheavydb-wheel=/pyheavydb-dist/\${PYHEAVYDB_WHEEL_NAME}\")
        fi
        bash ${build_script} \"\${build_args[@]}\"
      "

  local dependency_artifacts=()
  mapfile -t dependency_artifacts < <(
    tar -tzf "$repo_dir/dist.tgz" \
      | grep -E '(^|/)requirements\.packages\.txt$|^\./packages/.+[^/]$|\.whl$' \
      | sort -u || true
  )
  if [ -z "$pyheavydb_wheel_name" ]; then
    if [ "${#dependency_artifacts[@]}" -ne 0 ]; then
      echo "ERROR: HeavyIQ artifact contains an unexpected dependency artifact:" >&2
      printf '  %s\n' "${dependency_artifacts[@]}" >&2
      return 1
    fi
  else
    local expected_pyheavydb="./packages/$pyheavydb_wheel_name"
    if [ "${#dependency_artifacts[@]}" -ne 1 ] \
        || [ "${dependency_artifacts[0]}" != "$expected_pyheavydb" ]; then
      echo "ERROR: expected only the selected pyheavydb wheel: $expected_pyheavydb" >&2
      printf '  %s\n' "${dependency_artifacts[@]}" >&2
      return 1
    fi
  fi

  if ! cmp -s "$repo_dir/requirements.txt" \
      <(tar -xOf "$repo_dir/dist.tgz" ./requirements.txt); then
    echo "ERROR: HeavyIQ artifact does not preserve the project's requirements.txt" >&2
    return 1
  fi

  local suffix
  case "$distro" in
    rockylinux8) suffix="rocky-$(uname -m)" ;;
    *)           suffix="ubuntu-$(uname -m)" ;;
  esac
  cp "$repo_dir/dist.tgz" "$output_dir/dist-${suffix}.tgz"
  echo "HeavyIQ artifact: $output_dir/dist-${suffix}.tgz" >&2
}

_build_immerse() {
  local repos_dir="$1" ref="$2" output_dir="$3" deps_image="$4"
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

  echo "Building Immerse in deps container (Node.js ${NODE_VERSION} from tool cache)..." >&2

  local extra_env=()
  [ -n "${PRIVATE_PACKAGES_TOKEN:-}" ] && extra_env+=(-e "PRIVATE_PACKAGES_TOKEN=${PRIVATE_PACKAGES_TOKEN}")
  [ -n "${GOOGLE_API_KEY:-}" ]         && extra_env+=(-e "GOOGLE_API_KEY=${GOOGLE_API_KEY}")
  [ -n "${MAPBOX_TOKEN:-}" ]           && extra_env+=(-e "MAPBOX_TOKEN=${MAPBOX_TOKEN}")
  [ -n "${RAYGUN_AUTH_TOKEN:-}" ]      && extra_env+=(-e "RAYGUN_AUTH_TOKEN=${RAYGUN_AUTH_TOKEN}")
  extra_env+=(-e "NPM_UPGRADE_SPEC=${npm_spec}")
  [ -z "${PRIVATE_PACKAGES_TOKEN:-}" ] \
    && echo "WARN: PRIVATE_PACKAGES_TOKEN not set — private npm packages may fail" >&2
  [ -z "${MAPBOX_TOKEN:-}" ] \
    && echo "WARN: MAPBOX_TOKEN not set — Mapbox maps will not work in the built image" >&2
  [ -z "${GOOGLE_API_KEY:-}" ] \
    && echo "WARN: GOOGLE_API_KEY not set — Google Maps / Street View will not work in the built image" >&2

  _run_logged "immerse" "${_BUILD_LOG_DIR}/immerse.log" \
    docker run --rm \
      -v "$repo_dir:/work" \
      -v "$node_dir:/opt/dev-tools/node:ro" \
      -v "$npm_cache:/root/.npm" \
      -w /work \
      -e USER=root \
      "${extra_env[@]}" \
      "$deps_image" \
      bash -c '
      set -euo pipefail
      echo "MAPBOX_TOKEN: ${MAPBOX_TOKEN:+present (${#MAPBOX_TOKEN} chars)} ${MAPBOX_TOKEN:-MISSING}"
      export PATH="/opt/dev-tools/node/bin:$PATH"
      # Puppeteer (a test-only transitive dep) downloads a Chromium binary in
      # its install script; there is no arm64 build for the pinned version, so
      # npm ci fails on aarch64. The production frontend build needs no browser,
      # so skip the download. (Both var names cover old and new puppeteer.)
      export PUPPETEER_SKIP_DOWNLOAD=true
      export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
      git config --global --add safe.directory /work
      apt-get install -y -q --no-install-recommends rsync zip > /dev/null 2>&1 || true
      if [ -n "${PRIVATE_PACKAGES_TOKEN:-}" ]; then
        git config --global \
          "url.https://x-access-token:${PRIVATE_PACKAGES_TOKEN}@github.com/.insteadOf" \
          "ssh://git@github.com/"
      fi
      # The npm bundled with the cached Node build may fail npm ci with EBADPLATFORM
      # on wrong-platform optional binaries. NPM_UPGRADE_SPEC comes from the
      # engines.npm field in the Immerse package.json.
      # The bundled node dir is mounted read-only, so install into a writable prefix.
      export NPM_CONFIG_PREFIX="$HOME/.npm-global"
      export PATH="$HOME/.npm-global/bin:$PATH"
      npm install -g "${NPM_UPGRADE_SPEC}" --loglevel=error
      # bash caches command paths: the npm install above hashed the old
      # /opt/dev-tools/node/bin/npm, so without hash -r the calls below would
      # keep using the old version despite the PATH change. (Immerse release.yml
      # dodges this by running the upgrade and npm ci in separate steps / shells.)
      hash -r
      echo "Using npm $(npm --version) for npm ci"
      npm ci --loglevel=error
      npm run deploy -- --production --branch release --loglevel=error
    '

  for zip in "$repo_dir"/packages/*.zip; do
    cp "$zip" "$output_dir/"
    echo "Immerse artifact: $output_dir/$(basename "$zip")" >&2
  done
}

_build_webserver() {
  local repos_dir="$1" ref="$2" output_dir="$3" deps_image="$4"
  local repo_dir="$repos_dir/webserver"
  _ensure_repo "git@github.com:heavyai/webserver.git" "$repo_dir" "$ref"

  GO_VERSION=$(_read_go_version "$repo_dir")

  _ensure_tools_go
  local cache_dir="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/tools"
  local go_dir="$cache_dir/go-${GO_VERSION}"
  local go_mod_cache="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/go-mod"
  mkdir -p "$go_mod_cache"

  echo "Building WebServer in deps container (Go ${GO_VERSION} from tool cache)..." >&2

  _run_logged "webserver" "${_BUILD_LOG_DIR}/webserver.log" \
    docker run --rm \
      --network=host \
      -v "$repo_dir:/work" \
      -v "$go_dir:/opt/dev-tools/go:ro" \
      -v "$go_mod_cache:/root/go/pkg/mod" \
      -w /work \
      -e USER=root \
      "$deps_image" \
      bash -c '
        set -euo pipefail
        export PATH="/opt/dev-tools/go/bin:$PATH"
        export GOPATH="/root/go"
        export GOCACHE="/tmp/go-build"
        export GOPROXY=https://proxy.golang.org,direct
        git config --global --add safe.directory /work
        ./scripts/build-all.sh
      '

  local tarball="$(uname)-$(uname -m)-heavy_web_server.tar.gz"
  cp "$repo_dir/build/${tarball}" "$output_dir/"
  echo "WebServer artifact: $output_dir/${tarball}" >&2
}

_build_geos_dsos() {
  local output_dir="$1" deps_image="$2"
  
  local geos_version
  geos_version=$(grep '^GEOS_VERSION=' "$REPO_ROOT/scripts/common-functions.sh" | cut -d= -f2)
  [ -n "$geos_version" ] || { echo "ERROR: GEOS_VERSION not found in scripts/common-functions.sh" >&2; exit 1; }

  local geos_cache="${HEAVYAI_DEV_CACHE:-$(cd "$REPO_ROOT/.." && pwd)/.heavyai-dev}/geos"
  mkdir -p "$geos_cache" "$output_dir"

  echo "Building GEOS DSOs (version ${geos_version}) in deps container..." >&2
  _run_logged "geos-dsos" "${_BUILD_LOG_DIR}/geos-dsos.log" \
    docker run --rm \
      -v "$REPO_ROOT:/src:ro" \
      -v "$output_dir:/out" \
      -v "$geos_cache:/cache" \
      -w /out \
      -e USER=root \
      -e GEOS_VERSION="$geos_version" \
      "$deps_image" \
      bash -c '
        set -euo pipefail
        source /usr/local/mapd-deps/mapd-deps.sh

        NPROC=$(nproc 2>/dev/null || echo 8)
        [ "$NPROC" -gt 24 ] && NPROC=24

        source /etc/os-release
        OS=${ID}${VERSION_ID}
        ARCH=$(uname -m)
        FILENAME=heavydb-libgeos-${OS}-${ARCH}.tar

        rm -rf "geos-${GEOS_VERSION}.tar.bz2" "geos-${GEOS_VERSION}"

        wget -q "https://download.osgeo.org/geos/geos-${GEOS_VERSION}.tar.bz2"
        tar xf "geos-${GEOS_VERSION}.tar.bz2"

        pushd "geos-${GEOS_VERSION}"

        mkdir build install

        pushd build
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DCMAKE_INSTALL_PREFIX=../install \
                 -DBUILD_SHARED_LIBS=on \
                 -DBUILD_GEOSOP=off \
                 -DBUILD_TESTING=off
        cmake --build . --parallel "${NPROC}" && cmake --install .
        popd

        pushd install
        if [ "${ID}" = "rocky" ] || [ "${ID}" = "rhel" ]; then
          pushd lib64
        else
          pushd lib
        fi
        rm -f "../../../${FILENAME}" "../../../${FILENAME}.xz"
        tar cf "../../../${FILENAME}" libgeos*
        xz -T"${NPROC}" "../../../${FILENAME}"
        popd
        popd

        popd

        rm -rf "geos-${GEOS_VERSION}.tar.bz2" "geos-${GEOS_VERSION}"
        echo "GEOS DSO artifact: /out/${FILENAME}.xz"
      '

  local artifact
  artifact=$(find "$output_dir" -maxdepth 1 -name 'heavydb-libgeos-*.tar.xz' 2>/dev/null | sort | tail -1 || true)
  if [ -z "$artifact" ]; then
    echo "ERROR: GEOS DSO build did not produce heavydb-libgeos-*.tar.xz in $output_dir" >&2
    exit 1
  fi
  echo "GEOS DSO artifact: $artifact" >&2
}

cmd_build_component() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh build <component> [options]

Builds a non-heavydb component from source and writes its artifact to
--output-dir, ready to be consumed by dev.sh build heavydb.

Components:
  heavyiq    Python admin UI (dist.tgz). Uses its declared dependencies and may
             include only an explicitly selected pyheavydb wheel.
  immerse    Node.js frontend (npm run deploy → packages/*.zip)
             Set PRIVATE_PACKAGES_TOKEN for private npm packages.
             Optionally set GOOGLE_API_KEY, MAPBOX_TOKEN, RAYGUN_AUTH_TOKEN.
  webserver  Go HTTP server   (Linux-x86_64-heavy_web_server.tar.gz)
  geos-dsos  GEOS shared libs (heavydb-libgeos-<os>-<arch>.tar.xz).
  docs       Sphinx developer docs (build/docs/html/). Uses the
             docs Docker image; optional host Doxygen when a heavydb build exists.

All components except docs build inside the local heavydb deps container.
Docs use the Sphinx docs image (docs/Dockerfile); optional Doxygen runs on the host.
Node.js and Go are downloaded to a local tool cache on first use.

Options:
  --deps-image=<image>   Deps container image. Auto-detected from local
                         Docker images if omitted.
  --ref=<branch|sha>     Ref to build; fetches from origin when given.
                         Default: use the existing local checkout as-is.
  --pyheavydb-source=<source>
                         (heavyiq only) Where to obtain HeavyIQ's declared
                         pyheavydb wheel when explicitly selecting one:
                         project (default), pypi, testpypi, or build-local-wheel.
  --pyheavydb-ref=<ref>  Ref for build-local-wheel. Supplying this option
                         without --pyheavydb-source implies build-local-wheel.
  --repos-dir=<path>     Parent dir where component repos are cloned.
                         Default: parent directory of this repo.
  --output-dir=<path>    Where to write the artifact.
                         Default: build/components/ inside the repo.
  --distro=ubuntu22.04|rockylinux8
                         Distro variant. For HeavyIQ: selects build_prod.sh vs
                         build_prod_rhel.sh. Also filters deps image auto-detection
                         when --deps-image is not given. Default: ubuntu22.04
  --cuda-version=<ver>   CUDA version filter for deps image auto-detection
                         (e.g. 12.9.2). Use when multiple CUDA versions are
                         available locally. Default: any
EOF
      return 0 ;;
    esac
  done

  [ $# -ge 1 ] || { echo "Usage: dev-tools/dev.sh build <component> [options]" >&2; exit 1; }
  local component="$1"
  shift

  local ref="" deps_image="" pyheavydb_ref="" cuda_version=""
  local pyheavydb_source="project"
  local repos_dir
  repos_dir="$(cd "$REPO_ROOT/.." && pwd)"
  local output_dir distro="ubuntu22.04"

  for arg in "$@"; do
    case "$arg" in
      --deps-image=*)    deps_image="${arg#*=}" ;;
      --ref=*)           ref="${arg#*=}" ;;
      --pyheavydb-ref=*) pyheavydb_ref="${arg#*=}" ;;
      --pyheavydb-source=*) pyheavydb_source="${arg#*=}" ;;
      --repos-dir=*)     repos_dir="${arg#*=}" ;;
      --output-dir=*)    output_dir="${arg#*=}" ;;
      --distro=*)        distro="${arg#*=}" ;;
      --cuda-version=*)  cuda_version="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  _validate_distro "$distro"

  if [ -n "$pyheavydb_ref" ] && [ "$pyheavydb_source" = "project" ]; then
    pyheavydb_source="build-local-wheel"
    echo "--pyheavydb-ref implies --pyheavydb-source=build-local-wheel" >&2
  fi

  : "${output_dir:=$REPO_ROOT/build/components}"
  mkdir -p "$output_dir"
  _BUILD_LOG_DIR="$(dirname "$output_dir")/logs"
  mkdir -p "$_BUILD_LOG_DIR"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "static" "$distro" "$cuda_version")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  case "$component" in
    heavyiq)
      _build_heavyiq "$repos_dir" "$ref" "$output_dir" "$distro" "$deps_image" \
        "$pyheavydb_source" "$pyheavydb_ref"
      ;;
    immerse)   _build_immerse   "$repos_dir" "$ref" "$output_dir" "$deps_image" ;;
    webserver) _build_webserver "$repos_dir" "$ref" "$output_dir" "$deps_image" ;;
    geos-dsos) _build_geos_dsos "$output_dir" "$deps_image" ;;
    docs)
      # Docs write under the heavydb build dir (parent of components/), not
      # into the component artifact directory. Does not use the deps image.
      _build_docs "$(dirname "$output_dir")" ;;
    *)
      echo "ERROR: unknown component '$component'. Use heavyiq, immerse, webserver, geos-dsos, or docs." >&2
      exit 1 ;;
  esac
}
