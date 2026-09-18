# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cmd_pyheavydb_tests: build pyheavydb from source and run its pytest suite.
# The unit tests run standalone; the integration tests need a running HeavyDB.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

# The container prologue shared by both test sets: source the deps env (for the
# thrift compiler) via the shared fragments, generate the thrift bindings, and
# install pyheavydb with its test extras into a fresh venv. Double-quoted so the
# shared ${DEV_CONTAINER_*} fragments interpolate here; \$ escapes keep the rest
# for the container.
_PYHEAVYDB_TEST_SETUP="
  ${DEV_CONTAINER_DEPS_ENV}
  ${DEV_CONTAINER_PYVENV}
  make thrift
  \"\$PYBIN\" -m venv /tmp/venv && . /tmp/venv/bin/activate
  pip install -q --upgrade pip
  pip install -q \".[test]\"
"

# Torn-down at the end of a managed integration run (and on interrupt).
_PYHD_SERVER=""
_PYHD_NET=""
_pyhd_cleanup() {
  [ -n "${_PYHD_SERVER:-}" ] && docker rm -f "$_PYHD_SERVER" >/dev/null 2>&1 || true
  [ -n "${_PYHD_NET:-}" ] && docker network rm "$_PYHD_NET" >/dev/null 2>&1 || true
  _PYHD_SERVER=""
  _PYHD_NET=""
}

# Finds a locally-built HeavyDB product image (as produced by
# `dev-tools/dev.sh build all --docker`), named
# heavyai-<distro>-cuda-<render|norender>-<arch>:<version>-YYYYMMDD-<10charSHA>
# (also tagged :latest). Prefers the :latest tag when present. Requires a match
# for the given distro (default: ubuntu22.04). Prints the ref or returns
# non-zero if none.
# Usage: _resolve_product_image [distro]
_resolve_product_image() {
  local distro="${1:-ubuntu22.04}" arch candidates latest
  arch=$(uname -m)
  _validate_distro "$distro"
  # Prefer CPU builds (no NVIDIA runtime needed) for integration tests that
  # only require a running server; fall back to CUDA builds if no CPU image
  # is available locally.
  local all_candidates cpu_candidates
  all_candidates=$(docker images --format '{{.Repository}}:{{.Tag}}' \
    | grep -E "^heavyai-${distro}-(cuda|cpu)-(render|norender)-${arch}:" \
    | grep -v '<none>' || true)
  [ -z "$all_candidates" ] && return 1
  cpu_candidates=$(echo "$all_candidates" | grep -- "-cpu-" || true)
  candidates="${cpu_candidates:-$all_candidates}"
  latest=$(echo "$candidates" | grep ':latest$' | head -1 || true)
  if [ -n "$latest" ]; then echo "$latest"; else echo "$candidates" | head -1; fi
}

cmd_pyheavydb_tests() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test pyheavydb [--unit] [--integration] [options]

Builds pyheavydb from source (thrift codegen) in the deps container, installs
it with its test extras, and runs its pytest suite.

Test selection (default: --unit):
  --unit          Run the unit tests — no server required:
                  test_cursor, test_exceptions, test_results_set
  --integration   Run the server-backed tests — need a running HeavyDB:
                  test_connection, test_integration, test_runtime_udf,
                  test_table_meta_data

Options:
  --deps-image=<image>   Deps container image to build/run in. Auto-detected
                         from local Docker images if omitted.
  --distro=ubuntu22.04|rockylinux8
                         Distro for deps/product image auto-detection.
                         Default: ubuntu22.04
  --repos-dir=<path>     Parent dir for the pyheavydb clone.
                         Default: parent directory of this repo.
  --ref=<branch|sha>     pyheavydb ref to test. Default: repo default branch.
  --heavydb-host=<host>  (integration) Test against an already-running HeavyDB
                         at <host>:6274. When omitted, a HeavyDB server is
                         started automatically for the run and torn down after.
  --heavydb-image=<img>  (integration) Server image to start in managed mode.
                         Default: a locally-built product image from
                         'dev-tools/dev.sh build all --docker'
                         (heavyai-<distro>-cuda-<render|norender>-<arch>:<version>-YYYYMMDD-<10charSHA>, also :latest).
                         The tests always run against a locally-built server,
                         never a public/released image.
EOF
      return 0 ;;
    esac
  done

  local deps_image="" distro="ubuntu22.04" repos_dir="" ref="" heavydb_host="" heavydb_image=""
  local run_unit=0 run_integration=0
  for arg in "$@"; do
    case "$arg" in
      --unit)            run_unit=1 ;;
      --integration)     run_integration=1 ;;
      --deps-image=*)    deps_image="${arg#*=}" ;;
      --distro=*)        distro="${arg#*=}" ;;
      --repos-dir=*)     repos_dir="${arg#*=}" ;;
      --ref=*)           ref="${arg#*=}" ;;
      --heavydb-host=*)  heavydb_host="${arg#*=}" ;;
      --heavydb-image=*) heavydb_image="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  _validate_distro "$distro"
  # Default to the standalone unit tests when nothing is selected.
  [ "$run_unit" -eq 0 ] && [ "$run_integration" -eq 0 ] && run_unit=1

  : "${repos_dir:=$(cd "$REPO_ROOT/.." && pwd)}"
  local repo_dir="$repos_dir/pyheavydb"
  _ensure_repo "git@github.com:heavyai/pyheavydb.git" "$repo_dir" "$ref"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "static" "$distro")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  if [ "$run_unit" -eq 1 ]; then
    echo "=== pyheavydb unit tests ===" >&2
    docker run --rm \
      -v "$repo_dir:/work" -w /work -e USER=root \
      "$deps_image" \
      bash -c "${_PYHEAVYDB_TEST_SETUP}
        pytest tests/test_cursor.py tests/test_exceptions.py tests/test_results_set.py"
  fi

  if [ "$run_integration" -eq 1 ]; then
    local test_host net_args
    if [ -n "$heavydb_host" ]; then
      # Test against an external server the caller is managing.
      echo "=== pyheavydb integration tests (external HeavyDB at ${heavydb_host}:6274) ===" >&2
      test_host="$heavydb_host"
      net_args=(--network host)
    else
      # Managed mode: start a locally-built HeavyDB on a private network, wait
      # for it to be ready, and tear it down afterward. The test container joins
      # the same network and reaches the server by name.
      local image="$heavydb_image"
      if [ -z "$image" ]; then
        image=$(_resolve_product_image "$distro") || {
          echo "ERROR: no locally-built HeavyDB product image found for distro='$distro' (heavyai-${distro}-*-$(uname -m))." >&2
          echo "Build one first:            dev-tools/dev.sh build all --docker --distro=${distro}" >&2
          echo "or pass an explicit image:  --heavydb-image=<image>" >&2
          exit 1
        }
        echo "Using locally-built product image: $image" >&2
      fi
      echo "=== pyheavydb integration tests (managed HeavyDB: ${image}) ===" >&2
      _PYHD_NET="pyheavydb-test-net"
      _PYHD_SERVER="pyheavydb-test-heavydb"
      trap _pyhd_cleanup EXIT INT TERM
      docker rm -f "$_PYHD_SERVER" >/dev/null 2>&1 || true
      docker network rm "$_PYHD_NET" >/dev/null 2>&1 || true
      docker network create "$_PYHD_NET" >/dev/null
      local gpus; gpus="$(docker_gpus_flag)"
      echo "Starting HeavyDB (${image})${gpus:+ with --gpus all}..." >&2
      # shellcheck disable=SC2086
      docker run -d --name "$_PYHD_SERVER" --network "$_PYHD_NET" $gpus "$image" >/dev/null
      printf 'Waiting for HeavyDB to become ready' >&2
      local ready=0 i
      for i in $(seq 1 90); do
        if docker logs "$_PYHD_SERVER" 2>&1 | grep -qiE "heavydb.*started"; then
          ready=1; break
        fi
        if [ "$(docker inspect -f '{{.State.Running}}' "$_PYHD_SERVER" 2>/dev/null)" != "true" ]; then
          echo >&2; echo "ERROR: HeavyDB container exited during startup:" >&2
          docker logs "$_PYHD_SERVER" 2>&1 | tail -20 >&2
          exit 1
        fi
        printf '.' >&2; sleep 2
      done
      echo >&2
      [ "$ready" -eq 1 ] || { echo "ERROR: HeavyDB did not become ready in time" >&2; exit 1; }
      sleep 3  # small grace period for query readiness after the startup banner
      test_host="$_PYHD_SERVER"
      net_args=(--network "$_PYHD_NET")
    fi

    set +e
    docker run --rm "${net_args[@]}" \
      -v "$repo_dir:/work" -w /work -e USER=root -e HEAVYDB_HOST="$test_host" \
      "$deps_image" \
      bash -c "${_PYHEAVYDB_TEST_SETUP}
        pytest tests/test_connection.py tests/test_integration.py tests/test_runtime_udf.py tests/test_table_meta_data.py"
    local rc=$?
    set -e
    _pyhd_cleanup
    trap - EXIT INT TERM
    [ "$rc" -eq 0 ] || return "$rc"
  fi
}
