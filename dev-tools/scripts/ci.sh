# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CI-facing commands: list, build, test, shell, test in-image.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

cmd_list() {
  echo "Build configs (scripts/ci/configs/<name>.sh):"
  for f in "$CONFIGS_DIR"/*.sh; do
    [ -f "$f" ] || continue
    local name desc
    name="$(basename "$f" .sh)"
    desc="$(grep '^# ' "$f" | grep -v '^# SPDX-' | head -1 | sed 's/^# //' || true)"
    printf "  %-20s %s\n" "$name" "$desc"
  done | sort -k1
  echo
  echo "Test variants (scripts/ci/configs/tests/<name>.sh):"
  for f in "$CONFIGS_DIR"/tests/*.sh; do
    [ -f "$f" ] || continue
    local name desc
    name="$(basename "$f" .sh)"
    desc="$(grep '^# ' "$f" | grep -v '^# SPDX-' | head -1 | sed 's/^# //' || true)"
    printf "  %-20s %s\n" "$name" "$desc"
  done | sort -k1
}

cmd_ci_build() {
  [ $# -ge 1 ] || { echo "Usage: dev-tools/dev.sh build ci <config>" >&2; exit 1; }
  local cfg_path
  cfg_path="$(require_config "$1" build)"
  source "$cfg_path"
  exec "$SCRIPTS_DIR/ci/build.sh"
}

cmd_test() {
  [ $# -ge 1 ] || {
    echo "ERROR: no test variant specified" >&2
    echo "" >&2
    cmd_test_dispatch --help >&2
    exit 1
  }

  # Parse options and extract the variant name first so we can validate it
  # before doing any expensive docker operations.
  local variant="" deps_image="" build_dir="" nproc_arg=""
  local passthrough=()
  for arg in "$@"; do
    case "$arg" in
      --deps-image=*)  deps_image="${arg#*=}" ;;
      --build-dir=*)   build_dir="${arg#*=}" ;;
      --nproc=*)       nproc_arg="${arg#*=}" ;;
      *)               passthrough+=("$arg") ;;
    esac
  done
  variant="${passthrough[0]:-}"
  if [ "${#passthrough[@]}" -gt 1 ]; then
    echo "ERROR: unexpected arguments after variant '${variant}': ${passthrough[*]:1}" >&2
    echo "Run 'dev-tools/dev.sh test --help' for usage." >&2
    exit 1
  fi

  # Validate the variant exists before doing anything else.
  require_config "$variant" test >/dev/null

  # If already inside a container, run directly — no docker wrapping needed.
  # This covers CI (pr-build image), cmd_test_in_image, and dev.sh shell.
  if [ -f /.dockerenv ]; then
    _cmd_test_inner "$variant" "$nproc_arg"
    return
  fi

  # On the host: resolve deps image and re-invoke inside the deps container.
  : "${build_dir:=$REPO_ROOT/build}"

  if [ ! -f "$build_dir/CMakeCache.txt" ]; then
    echo "ERROR: no completed build found at $build_dir (missing CMakeCache.txt)" >&2
    echo "Run 'dev-tools/dev.sh build heavydb' first." >&2
    exit 1
  fi

  # All three dimensions needed for deps image auto-detection come from the
  # build cache written by 'build heavydb': distro, CUDA version, and lib type.
  local cache_distro cache_cuda_version cache_lib_type="static" cache_prefer_static
  cache_distro=$(grep -m1 '^MAPD_PACKAGE_DISTRO_NAME' "$build_dir/CMakeCache.txt" \
    | cut -d= -f2 || true)
  cache_cuda_version=$(grep -m1 '^HEAVYAI_DEPS_CUDA_VERSION' "$build_dir/CMakeCache.txt" \
    | cut -d= -f2 || true)
  cache_prefer_static=$(grep -m1 '^PREFER_STATIC_LIBS:BOOL=' "$build_dir/CMakeCache.txt" \
    | cut -d= -f2 || true)
  [ "$cache_prefer_static" = "OFF" ] && cache_lib_type="shared"

  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "$cache_lib_type" "$cache_distro" "$cache_cuda_version")
    echo "Auto-detected deps image: $deps_image" >&2
  fi

  # Check whether the test config needs ASLR disabled (TSAN tests using
  # setarch -R). setarch needs the ADDR_NO_RANDOMIZE personality bit, which
  # is blocked by Docker's default seccomp profile. Source the config in a
  # subshell so it doesn't pollute the current environment.
  local extra_docker_args=()
  local cfg_no_aslr
  cfg_no_aslr=$(
    cfg_path="$(require_config "$variant" test)"
    # shellcheck disable=SC1090
    source "$cfg_path"
    printf '%s' "${NO_ASLR:-false}"
  )
  if [ "$cfg_no_aslr" = "true" ]; then
    echo "Note: test '$variant' sets NO_ASLR=true — adding --security-opt seccomp=unconfined" >&2
    extra_docker_args+=(--security-opt seccomp=unconfined)
  fi

  local nproc_flag=""
  [ -n "$nproc_arg" ] && nproc_flag="--nproc=$(printf '%q' "$nproc_arg")"
  echo "Running test '$variant' inside deps container: $deps_image" >&2
  _run_in_deps_container "$deps_image" "$build_dir" \
    "cd /workspace && dev-tools/dev.sh test $(printf '%q' "$variant") ${nproc_flag}" \
    "${extra_docker_args[@]}"
}

# Inner implementation — runs directly when already in a container.
_cmd_test_inner() {
  local variant="$1"
  local nproc_arg="${2:-}"
  local cfg_path
  cfg_path="$(require_config "$variant" test)"
  source "$cfg_path"
  : "${RUNNER:?RUNNER must be set by the test config (gtest|ctest)}"
  local runner_script="$SCRIPTS_DIR/ci/run_${RUNNER}.sh"
  [ -x "$runner_script" ] || { echo "ERROR: no runner script for RUNNER=$RUNNER" >&2; exit 1; }

  # Source the deps environment (Rocky or Ubuntu).
  # shellcheck disable=SC1090
  eval "$_DEPS_ENV_SOURCE"

  # Reconfigure with ENABLE_TESTS=ON and compile the test binary.
  # cmake is a fast no-op when nothing has changed.
  local nproc
  nproc=$(_resolve_nproc "$nproc_arg")

  local src_dir build_dir
  if [ -d /workspace ]; then
    src_dir=/workspace
    build_dir=/workspace/build
  else
    src_dir="$REPO_ROOT"
    build_dir="$REPO_ROOT/build"
  fi

  (
    cd "$build_dir"
    _cmake_enable_tests "$build_dir" "$src_dir"
    # Build the targets specified by the config. If MAKE_TARGETS is not set,
    # fall back to BINARY (the test executable). cmake make targets use just
    # the basename (e.g. ExecuteTest, not Tests/ExecuteTest). If neither is
    # set, this is a ctest-based variant that expects all test binaries to be
    # present (e.g. multi-cuda-gcc sanity suite); build the default target
    # (equivalent to 'make all') so every binary is compiled.
    local targets="${MAKE_TARGETS:-$(basename "${BINARY:-}")}"
    if [ -n "$targets" ]; then
      # shellcheck disable=SC2086
      make -j "$nproc" $targets
    else
      echo "No MAKE_TARGETS or BINARY set — building all test targets (this may take a while on first run)." >&2
      make -j "$nproc"
    fi
  )

  # Initialise the heavydb storage directory and create the Tests/tmp symlink
  # so fixture paths resolve correctly (see scripts/ci/init_heavy.sh).
  "$SCRIPTS_DIR/ci/init_heavy.sh"
  exec "$runner_script"
}

resolve_latest_tag() {
  local image_name="$1" pkg encoded
  pkg="${GHCR_REPO}/${image_name}"
  encoded="$(printf '%s' "$pkg" | jq -sRr '@uri')"
  gh api "/orgs/${GHCR_OWNER}/packages/container/${encoded}/versions" \
    --jq 'sort_by(.updated_at) | reverse | .[0].metadata.container.tags[0]'
}

cmd_pull() {
  [ $# -ge 1 ] || { echo "Usage: dev-tools/dev.sh shell pull <config> [<tag>]" >&2; exit 1; }
  local cfg_path
  cfg_path="$(require_config "$1" build)"
  source "$cfg_path"
  if [ -z "${IMAGE_NAME:-}" ]; then
    echo "ERROR: config '$1' does not publish an image (no IMAGE_NAME)" >&2
    exit 1
  fi
  local tag="${2:-}"
  if [ -z "$tag" ]; then
    command -v gh >/dev/null || { echo "ERROR: gh CLI required to look up the latest tag" >&2; exit 1; }
    command -v jq >/dev/null || { echo "ERROR: jq required to look up the latest tag" >&2; exit 1; }
    tag="$(resolve_latest_tag "$IMAGE_NAME")"
    if [ -z "$tag" ] || [ "$tag" = "null" ]; then
      echo "ERROR: no published versions found for $IMAGE_NAME" >&2
      exit 1
    fi
    echo "Resolved latest tag for $IMAGE_NAME: $tag" >&2
  fi
  local ref="ghcr.io/${GHCR_OWNER}/${GHCR_REPO}/${IMAGE_NAME}:${tag}"
  echo "Pulling $ref" >&2
  docker pull "$ref" >&2
  printf '%s\n' "$ref"
}

cmd_shell() {
  [ $# -ge 1 ] || { echo "Usage: dev-tools/dev.sh shell <config> [<tag>]" >&2; exit 1; }
  local ref
  ref="$(cmd_pull "$@")"
  local gpus
  gpus="$(docker_gpus_flag)"
  echo "Entering shell in $ref ${gpus:+(with --gpus all)}" >&2
  # Bind-mount the host's scripts/ over the image's so the latest dev.sh
  # and configs are always available — older pr-build images may have
  # been packed before dev.sh existed.
  # shellcheck disable=SC2086
  exec docker run --rm -it $gpus \
    -v "$SCRIPTS_DIR:/workspace/scripts:ro" \
    "$ref" bash
}

cmd_test_in_image() {
  local local_image="" build_dir="" tag=""
  local positional=()
  for arg in "$@"; do
    case "$arg" in
      --image=*)     local_image="${arg#*=}" ;;
      --build-dir=*) build_dir="${arg#*=}" ;;
      *)             positional+=("$arg") ;;
    esac
  done
  set -- "${positional[@]}"
  [ $# -ge 2 ] || {
    cat >&2 <<'EOF'
Usage: dev-tools/dev.sh test in-image <config> <variant> [<tag>] [options]

Options:
  --image=<local-image>   Use a local Docker image instead of pulling from GHCR.
  --build-dir=<path>      Mount a local cmake build dir at /workspace/build inside
                          the container (needed for ctest-based tests when using
                          --image with a product image that lacks the build dir).
EOF
    exit 1
  }
  local config="$1" variant="$2"
  tag="${3:-}"

  require_config "$variant" test >/dev/null

  local ref
  if [ -n "$local_image" ]; then
    ref="$local_image"
    echo "Using local image: $ref" >&2
  elif [ -n "$tag" ]; then
    ref="$(cmd_pull "$config" "$tag")"
  else
    ref="$(cmd_pull "$config")"
  fi

  local gpus
  gpus="$(docker_gpus_flag)"
  echo "Running test '$variant' in $ref ${gpus:+(with --gpus all)}" >&2

  local extra_mounts=()
  if [ -n "$build_dir" ]; then
    build_dir="$(cd "$build_dir" && pwd)"
    extra_mounts+=(-v "$build_dir:/workspace/build")
    echo "Mounting build dir: $build_dir → /workspace/build" >&2
  fi

  local dev_tools_dir
  dev_tools_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  # shellcheck disable=SC2086
  exec docker run --rm $gpus \
    -w /workspace \
    -v "$SCRIPTS_DIR:/workspace/scripts:ro" \
    -v "$dev_tools_dir:/workspace/dev-tools:ro" \
    "${extra_mounts[@]}" \
    "$ref" \
    bash -c "dev-tools/dev.sh test '$variant'"
}

cmd_enter_deps() {
  local base="${BASE_IMAGE:-}"
  if [ -z "$base" ]; then
    base=$(_resolve_deps_image)
  fi
  local gpus
  gpus="$(docker_gpus_flag)"
  echo "Entering deps container ($base) ${gpus:+(with --gpus all)}" >&2
  # shellcheck disable=SC2086
  exec docker run --rm -it $gpus \
    -v "$REPO_ROOT:/workspace" -w /workspace \
    "$base" bash
}
