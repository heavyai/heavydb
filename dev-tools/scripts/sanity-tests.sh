# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cmd_sanity_tests: run the heavydb "sanity" (or full) test suite against an
# existing build tree, inside the deps container. Reconfigure the build dir
# with -DENABLE_TESTS=ON, build the test binaries, then run the
# `sanity_tests` (or `all_tests`) make target.
#
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

cmd_sanity_tests() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh test sanity [options]

Runs the heavydb test suite against an already-built tree (as produced by
'build', 'build heavydb' or 'build all'). Reconfigures the build dir with -DENABLE_TESTS=ON,
compiles the test binaries, and runs the make test target.

Options:
  --deps-image=<image>    Deps container image to run in. Auto-detected from
                          local Docker images if omitted.
  --distro=ubuntu22.04|rockylinux8
                          Filter deps image auto-detection. Ignored when
                          --deps-image is given.
  --output-dir=<path>     Build dir to test (must contain a completed build).
                          Default: build/ inside the repo.
  --all                   Run the full `all_tests` target instead of the
                          curated `sanity_tests` target.
  --nproc=<n>             Parallel make jobs for building test binaries
                          (default: min(nproc, 24)).

Requires GPU access for CUDA builds (auto-detected via nvidia-smi; the
container gets --gpus all when a GPU is present).
EOF
      return 0 ;;
    esac
  done

  local deps_image="" distro="" output_dir="" run_all=0 nproc_arg=""
  for arg in "$@"; do
    case "$arg" in
      --deps-image=*) deps_image="${arg#*=}" ;;
      --distro=*)     distro="${arg#*=}" ;;
      --output-dir=*) output_dir="${arg#*=}" ;;
      --all)          run_all=1 ;;
      --nproc=*)      nproc_arg="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  [ -n "$distro" ] && _validate_distro "$distro"

  local effective_distro="${distro}"

  : "${output_dir:=$REPO_ROOT/build}"

  if [ ! -f "$output_dir/CMakeCache.txt" ]; then
    echo "ERROR: no completed build found at $output_dir (missing CMakeCache.txt)" >&2
    echo "Run: dev-tools/dev.sh build heavydb (or build all) first" >&2
    exit 1
  fi

  # Read distro, CUDA version, and lib type from CMakeCache so the correct
  # deps image is found automatically, matching how the build was configured.
  if [ -z "$effective_distro" ]; then
    local cache_distro
    cache_distro=$(grep -m1 '^MAPD_PACKAGE_DISTRO_NAME' "$output_dir/CMakeCache.txt" \
      | cut -d= -f2 || true)
    [ -n "$cache_distro" ] && effective_distro="$cache_distro"
  fi
  local cache_cuda_version
  cache_cuda_version=$(grep -m1 '^HEAVYAI_DEPS_CUDA_VERSION' "$output_dir/CMakeCache.txt" \
    | cut -d= -f2 || true)
  local cache_lib_type="static"
  local cache_prefer_static
  cache_prefer_static=$(grep -m1 '^PREFER_STATIC_LIBS:BOOL=' "$output_dir/CMakeCache.txt" \
    | cut -d= -f2 || true)
  [ "$cache_prefer_static" = "OFF" ] && cache_lib_type="shared"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "$cache_lib_type" "$effective_distro" "$cache_cuda_version")
    echo "Auto-detected deps image: $deps_image" >&2
  fi
  : "${effective_distro:=$(_parse_distro "$deps_image")}"
  _validate_distro "$effective_distro" "$deps_image"

  local nproc
  nproc=$(_resolve_nproc "$nproc_arg")

  local target_desc
  [ "$run_all" -eq 1 ] && target_desc="all_tests" || target_desc="sanity_tests"

  # Shadow a host-specific java/.mvn/maven.config with one pointing at the
  # container-internal settings path (matches build heavydb).
  local maven_config_tmp
  maven_config_tmp=$(mktemp)
  trap "rm -f '$maven_config_tmp'" EXIT
  printf -- '-s\n/workspace/java/.mvn/settings.xml\n' > "$maven_config_tmp"

  local run_cmd
  if [ "$run_all" -eq 1 ]; then
    run_cmd="make -j ${nproc} all_tests"
  else
    run_cmd="make -j ${nproc} sanity_tests_build_only && make sanity_tests"
  fi

  local gpus
  gpus="$(docker_gpus_flag)"

  echo "Running heavydb ${target_desc}:" >&2
  echo "  deps image:  $deps_image" >&2
  echo "  build dir:   $output_dir" >&2
  echo "  gpus:        ${gpus:-none}" >&2

  _run_in_deps_container "$deps_image" "$output_dir" "
      set -euo pipefail
      git config --global --add safe.directory /workspace
      mkdir -p ~/.m2
      cp /workspace/java/.mvn/settings.xml ~/.m2/settings.xml
      ${_DEPS_ENV_SOURCE}
      cd /workspace/build
      # The test binaries resolve data fixtures relative to the build tree as
      # <build>/../../Tests/... . The build dir is mounted at /workspace/build
      # (two levels deep), so that relative path resolves to /Tests — point it
      # at the source Tests dir so the fixtures are found.
      ln -sfn /workspace/Tests /Tests
      # Source dev.sh as a library (the BASH_SOURCE guard skips the CLI
      # dispatcher) to get _cmake_enable_tests and other shared helpers.
      # shellcheck disable=SC1090
      source /workspace/dev-tools/dev.sh
      _cmake_enable_tests /workspace/build /workspace
      ${run_cmd}
      rm -rf Tests/tmp
    " \
    -v "${maven_config_tmp}:/workspace/java/.mvn/maven.config:ro"

  rm -f "$maven_config_tmp"

  echo "${target_desc} complete." >&2
}
