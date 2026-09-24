# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cmd_build_all / cmd_build: build components then heavydb in one step.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

# Shared implementation used by cmd_build_all and cmd_build.
# First argument: 1 = include HeavyIQ, 0 = exclude HeavyIQ.
# Remaining arguments: the user-facing flags.
_do_build() {
  local want_heavyiq="$1"
  shift

  local want_immerse=1 want_webserver=1 want_geos_dsos=1
  local ref="" heavyiq_ref="" immerse_ref="" webserver_ref="" pyheavydb_ref=""
  local pyheavydb_source="project"
  local repos_dir="" distro="ubuntu22.04" cuda_version="" component_dir=""
  local deps_image="" compiler="gcc" render="renderer" cuda="cuda"
  local lib_type="static" build_type="Release"
  local output_dir="" nproc_arg="" no_package=0 clean=0
  local build_docker=0 product_base_image="" run_tests="none"

  for arg in "$@"; do
    case "$arg" in
      --ref=*)                 ref="${arg#*=}" ;;
      --heavyiq-ref=*)         heavyiq_ref="${arg#*=}" ;;
      --immerse-ref=*)         immerse_ref="${arg#*=}" ;;
      --webserver-ref=*)       webserver_ref="${arg#*=}" ;;
      --pyheavydb-ref=*)       pyheavydb_ref="${arg#*=}" ;;
      --pyheavydb-source=*)    pyheavydb_source="${arg#*=}" ;;
      --repos-dir=*)           repos_dir="${arg#*=}" ;;
      --distro=*)              distro="${arg#*=}" ;;
      --cuda-version=*)        cuda_version="${arg#*=}" ;;
      --component-dir=*)       component_dir="${arg#*=}" ;;
      --deps-image=*)          deps_image="${arg#*=}" ;;
      --compiler=*)            compiler="${arg#*=}" ;;
      --norendering)           render="norender" ;;
      --cuda)                  cuda="cuda" ;;
      --cpu)                   cuda="cpu" ;;
      --static)                lib_type="static" ;;
      --shared)                lib_type="shared" ;;
      --build-type=*)          build_type="${arg#*=}" ;;
      --output-dir=*)          output_dir="${arg#*=}" ;;
      --nproc=*)               nproc_arg="${arg#*=}" ;;
      --no-package)            no_package=1 ;;
      --clean)                 clean=1 ;;
      --docker)                build_docker=1 ;;
      --product-base-image=*)  product_base_image="${arg#*=}" ;;
      --run-tests=*)           run_tests="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  _validate_distro "$distro"

  if [ -n "$pyheavydb_ref" ] && [ "$pyheavydb_source" = "project" ]; then
    pyheavydb_source="build-local-wheel"
    echo "--pyheavydb-ref implies --pyheavydb-source=build-local-wheel" >&2
  fi

  case "$run_tests" in
    none|sanity|all) ;;
    *) echo "ERROR: --run-tests must be none, sanity, or all (got '$run_tests')" >&2; exit 1 ;;
  esac

  local base_dir
  base_dir="$(cd "$REPO_ROOT/.." && pwd)"
  : "${repos_dir:=$base_dir}"
  : "${component_dir:=$REPO_ROOT/build/components}"
  : "${output_dir:=$REPO_ROOT/build}"

  # Resolve the deps image once — shared by component builds, the heavydb build, and clean.
  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "$lib_type" "$distro" "$cuda_version")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  if [ "$clean" -eq 1 ]; then
    # Build output files are owned by root (created inside Docker). Use the deps
    # container to delete them, then recreate the directories as the host user.
    if [ "$component_dir" != "$output_dir" ] && [ -d "$component_dir" ]; then
      echo "Cleaning $component_dir ..." >&2
      _docker_wipe_dir "$deps_image" "$component_dir"
    fi
    if [ -d "$output_dir" ]; then
      echo "Cleaning $output_dir ..." >&2
      _docker_wipe_dir "$deps_image" "$output_dir"
    fi
    # Clean root-owned build artifacts inside component repo clones. These are
    # created by Docker builds and cannot be removed directly from the host.
    [ "$want_immerse" -eq 1 ] && \
      _clean_repo_artifacts "$deps_image" "$repos_dir/immerse" \
        frontend-prod dist packages
    if [ "$want_heavyiq" -eq 1 ]; then
      _clean_repo_artifacts "$deps_image" "$repos_dir/heavyiq" \
        dist dist.tgz packages venv
      if [ "$pyheavydb_source" = "build-local-wheel" ]; then
        _clean_repo_artifacts "$deps_image" "$repos_dir/pyheavydb" dist venv
      fi
    fi
    [ "$want_webserver" -eq 1 ] && \
      _clean_repo_artifacts "$deps_image" "$repos_dir/webserver" build
  fi
  mkdir -p "$component_dir"
  _BUILD_LOG_DIR="$output_dir/logs"
  mkdir -p "$_BUILD_LOG_DIR"

  # Build each requested component.
  # Each component uses its own ref when given, else the shared --ref.
  local effective_pyheavydb_ref=""
  if [ "$pyheavydb_source" = "build-local-wheel" ]; then
    effective_pyheavydb_ref="${pyheavydb_ref:-$ref}"
  fi
  if [ "$want_heavyiq" -eq 1 ]; then
    _build_heavyiq "$repos_dir" "${heavyiq_ref:-$ref}" "$component_dir" \
      "$distro" "$deps_image" "$pyheavydb_source" "$effective_pyheavydb_ref"
  fi
  [ "$want_immerse" -eq 1 ]   && _build_immerse   "$repos_dir" "${immerse_ref:-$ref}"   "$component_dir" "$deps_image"
  [ "$want_webserver" -eq 1 ] && _build_webserver "$repos_dir" "${webserver_ref:-$ref}" "$component_dir" "$deps_image"
  [ "$want_geos_dsos" -eq 1 ] && _build_geos_dsos "$component_dir" "$deps_image"

  # Build heavydb, passing through all options and selecting the components we built.
  local heavydb_args=(
    "--component-dir=$component_dir"
    "--output-dir=$output_dir"
    "--compiler=$compiler"
    "--$cuda"
    "--$lib_type"
    "--build-type=$build_type"
  )
  [ "$render" = "norender" ]    && heavydb_args+=("--norendering")
  [ -n "$deps_image" ]         && heavydb_args+=("--deps-image=$deps_image")
  [ -n "$distro" ]             && heavydb_args+=("--distro=$distro")
  [ -n "$cuda_version" ]       && heavydb_args+=("--cuda-version=$cuda_version")
  [ -n "$nproc_arg" ]          && heavydb_args+=("--nproc=$nproc_arg")
  [ "$no_package" -eq 1 ]      && heavydb_args+=("--no-package")
  [ "$build_docker" -eq 1 ]    && heavydb_args+=("--docker")
  [ -n "$product_base_image" ] && heavydb_args+=("--product-base-image=$product_base_image")
  [ "$want_heavyiq" -eq 1 ]    && heavydb_args+=("--heavyiq")
  [ "$want_immerse" -eq 1 ]    && heavydb_args+=("--immerse")
  if [ "$want_webserver" -eq 1 ]; then
    heavydb_args+=("--webserver")
  elif [ "$want_immerse" -eq 1 ]; then
    # Immerse's cmake subproject always downloads the webserver unless a local
    # file is provided. Auto-include an existing artifact, or build it first.
    if find "$component_dir" -maxdepth 1 -name '*heavy_web_server.tar.gz' 2>/dev/null | grep -q .; then
      echo "Immerse requires webserver — reusing existing artifact from $component_dir" >&2
      heavydb_args+=("--webserver")
    else
      echo "Immerse requires webserver — building it now (none found in $component_dir)" >&2
      _build_webserver "$repos_dir" "${webserver_ref:-$ref}" "$component_dir" "$deps_image"
      heavydb_args+=("--webserver")
    fi
  fi

  cmd_build_heavydb "${heavydb_args[@]}"

  # Docs run after heavydb so optional Doxygen can use the configured build tree.
  # Sphinx uses the docs Docker image; Doxygen (if any) runs on the host.
  _build_docs "$output_dir"

  # Optionally run the test suite against the tree we just built.
  if [ "$run_tests" != "none" ]; then
    local sanity_args=(
      "--deps-image=$deps_image"
      "--distro=$distro"
      "--output-dir=$output_dir"
    )
    [ -n "$nproc_arg" ]        && sanity_args+=("--nproc=$nproc_arg")
    [ "$run_tests" = "all" ]   && sanity_args+=("--all")
    cmd_sanity_tests "${sanity_args[@]}"
  fi
}

cmd_build_all() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh build all [options]

Builds ALL components from source (heavyiq, immerse, webserver, geos-dsos),
then builds heavydb, bundles them, and builds docs. To build without HeavyIQ
use "dev-tools/dev.sh build" instead.

Component options:
  --ref=<branch|sha>      Ref to check out for all component repos (the shared
                          default). Leave empty to use each repo's default branch.
  --heavyiq-ref=<ref>     Override the ref for a single component; each falls
  --immerse-ref=<ref>     back to --ref when not set. Use these to build each
  --webserver-ref=<ref>   component from a different branch/commit.
  --pyheavydb-source=<source>
                          Where to obtain HeavyIQ's declared pyheavydb:
                          project (default), pypi, testpypi, or build-local-wheel.
  --pyheavydb-ref=<ref>   Ref for build-local-wheel; supplying it implies that
                          source. Falls back to --ref when not set.
  --repos-dir=<path>      Parent dir for component repo clones.
                          Default: parent directory of this repo.
  --distro=ubuntu22.04|rockylinux8
                          Distro variant. Selects HeavyIQ build script and
                          filters deps image auto-detection. (default: ubuntu22.04)

HeavyDB options (same as build heavydb):
  --deps-image=<image>    Deps container image (auto-detected if omitted)
  --cuda-version=<ver>    Filter deps image auto-detection to a specific
                          CUDA version, e.g. 12.9.2. Ignored when
                          --deps-image is given.
  --compiler=gcc|clang    (default: gcc)
  --norendering           Disable rendering (default: on)
  --cuda / --cpu          (default: cuda)
  --static / --shared     (default: static)
  --build-type=Release|Debug|RelWithDebInfo   (default: Release)
  --component-dir=<path>  Where component artifacts are written/read.
                          Default: build/components/ inside the repo.
  --output-dir=<path>     Where the heavydb build output goes.
                          Default: build/ inside the repo.
  --nproc=<n>             Parallel make jobs for heavydb (default: min(nproc,24))
  --no-package            Skip cpack
  --clean                 Wipe component-dir and output-dir before building
                          (forces fresh component builds and cmake from scratch)
  --docker                Also build a product Docker image after cpack
  --product-base-image=<image>
                          Override the auto-detected product Docker base image
  --run-tests=none|sanity|all
                          After building, run the test suite via test sanity:
                          'sanity' runs the sanity_tests target, 'all' the
                          all_tests target. (default: none)
  --verbose, -v           Stream build output to the shell as well as the log
                          file, instead of printing progress dots
EOF
      return 0 ;;
    esac
  done
  _do_build 1 "$@"
}

cmd_build() {
  # --verbose applies to every target, so consume it here rather than in each
  # per-target option parser. Stripping it before the target is read also means
  # it can appear before or after the target name.
  local args=()
  for arg in "$@"; do
    case "$arg" in
      --verbose|-v) _BUILD_VERBOSE=1 ;;
      *)            args+=("$arg") ;;
    esac
  done
  set -- "${args[@]}"

  local target="${1:-}"
  case "$target" in
    --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh build [target] [options]

Targets:
  (none)              Build immerse + webserver + geos-dsos + heavydb + docs (default)
  all                 Build ALL components (including heavyiq) + heavydb + docs
  immerse             Build just the Immerse frontend component
  webserver           Build just the web server component
  heavyiq             Build just the HeavyIQ component
  geos-dsos           Build just the GEOS DSO component
  docs                Build Sphinx developer docs (optional Doxygen if available)
  heavydb             Build heavydb only (bundles existing component outputs)
  image               Build a product Docker image from a pre-existing tarball
  deps                Build the deps container image
  ci <config>         Run a CI-style build

Common options (accepted by every target):
  --verbose, -v       Stream build output to the shell as it is produced, in
                      addition to writing the log file. Without it, each step
                      logs to a file and prints a dot every 5 seconds.

All options from build heavydb and build <component> are accepted where relevant.
Run "dev-tools/dev.sh build <target> --help" for target-specific options.
EOF
      return 0 ;;
    ""|--*)
      _do_build 0 "$@" ;;
    all)
      shift; cmd_build_all "$@" ;;
    immerse|webserver|heavyiq|geos-dsos)
      shift; cmd_build_component "$target" "$@" ;;
    docs)
      shift; cmd_build_docs "$@" ;;
    heavydb)
      shift; cmd_build_heavydb "$@" ;;
    image)
      shift; cmd_build_image "$@" ;;
    deps)
      shift; cmd_build_deps_image "$@" ;;
    ci)
      shift; cmd_ci_build "$@" ;;
    *)
      echo "Unknown build target: '$target' (run with --help)" >&2; exit 1 ;;
  esac
}
