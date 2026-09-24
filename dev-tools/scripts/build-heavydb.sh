# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cmd_build_heavydb: build heavydb inside the local deps container.
# Sourced by dev-tools/dev.sh — all variables defined there are available here.

cmd_build_heavydb() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh build heavydb [options]

Builds heavydb inside the local deps container image using cmake+make+cpack.
Components are opt-in: pass --heavyiq,
--immerse, and/or --webserver to bundle artifacts from --component-dir,
or use --*-file= to provide an explicit path for any component.

Options:
  --deps-image=<image>    Deps container image to build in. Auto-detected
                          from local Docker images if omitted.
  --distro=ubuntu22.04|rockylinux8
                          Filter deps image auto-detection to a specific
                          distro. Ignored when --deps-image is given.
  --cuda-version=<ver>    Filter deps image auto-detection to a specific
                          CUDA version, e.g. 12.9.2. Ignored when
                          --deps-image is given.
  --compiler=gcc|clang    Compiler (default: gcc)
  --norendering           Disable rendering (default: on)
  --cuda                  Enable CUDA (default)
  --cpu                   CPU-only build
  --static                Prefer static libs (default)
  --shared                Prefer shared libs
  --build-type=Release|Debug|RelWithDebInfo   (default: Release)
  --heavyiq               Bundle HeavyIQ from --component-dir
  --immerse               Bundle Immerse frontend from --component-dir
  --webserver             Bundle web server from --component-dir
  --heavyiq-file=<path>   Explicit path to HeavyIQ dist.tgz (implies --heavyiq)
  --immerse-file=<path>   Explicit path to Immerse .zip (implies --immerse)
  --webserver-file=<path> Explicit path to web server .tar.gz (implies --webserver)
  --component-dir=<path>  Directory with artifacts from build <component>.
                          Default: build/components/ inside the repo.
  --output-dir=<path>     Where to write build output and the final tar.gz.
                          Default: build/ inside the repo.
  --nproc=<n>             Parallel make jobs (default: min(nproc, 24))
  --no-package            Skip cpack; build only, no tar.gz
  --clean                 Wipe --output-dir before building (forces a fresh
                          cmake configure and full recompile)
  --docker                Also build a product Docker image after cpack.
  --product-base-image=<image>
                          Base image for the product Docker build. Auto-detected
                          from the deps image name if omitted.
  --verbose, -v           Stream build output to the shell as well as the log
                          file, instead of printing progress dots
EOF
      return 0 ;;
    esac
  done

  local deps_image="" distro="" cuda_version="" compiler="gcc" render="renderer" cuda="cuda"
  local lib_type="static" build_type="Release"
  local component_dir="" output_dir="" nproc_arg="" no_package=0 clean=0
  local want_heavyiq=0 want_immerse=0 want_webserver=0
  local heavyiq_file_arg="" immerse_file_arg="" webserver_file_arg=""
  local build_docker=0 product_base_image=""

  for arg in "$@"; do
    case "$arg" in
      --deps-image=*)          deps_image="${arg#*=}" ;;
      --distro=*)              distro="${arg#*=}" ;;
      --cuda-version=*)        cuda_version="${arg#*=}" ;;
      --compiler=*)            compiler="${arg#*=}" ;;
      --norendering)           render="norender" ;;
      --cuda)                  cuda="cuda" ;;
      --cpu)                   cuda="cpu" ;;
      --static)                lib_type="static" ;;
      --shared)                lib_type="shared" ;;
      --build-type=*)          build_type="${arg#*=}" ;;
      --heavyiq)               want_heavyiq=1 ;;
      --immerse)               want_immerse=1 ;;
      --webserver)             want_webserver=1 ;;
      --heavyiq-file=*)        heavyiq_file_arg="${arg#*=}"; want_heavyiq=1 ;;
      --immerse-file=*)        immerse_file_arg="${arg#*=}"; want_immerse=1 ;;
      --webserver-file=*)      webserver_file_arg="${arg#*=}"; want_webserver=1 ;;
      --component-dir=*)       component_dir="${arg#*=}" ;;
      --output-dir=*)          output_dir="${arg#*=}" ;;
      --nproc=*)               nproc_arg="${arg#*=}" ;;
      --no-package)            no_package=1 ;;
      --clean)                 clean=1 ;;
      --docker)                build_docker=1 ;;
      --product-base-image=*)  product_base_image="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done
  [ -n "$distro" ] && _validate_distro "$distro"

  if [ "$build_docker" -eq 1 ] && [ "$no_package" -eq 1 ]; then
    echo "ERROR: --docker and --no-package are mutually exclusive (no tarball to containerise)." >&2
    exit 1
  fi

  # Resolve deps image early so we can derive the effective distro for default paths.
  if [ -z "$deps_image" ]; then
    deps_image=$(_resolve_deps_image "$lib_type" "$distro" "$cuda_version")
    echo "Auto-detected deps image: $deps_image" >&2
  fi
  local effective_distro="${distro:-$(_parse_distro "$deps_image")}"
  _validate_distro "$effective_distro" "$deps_image"
  : "${component_dir:=$REPO_ROOT/build/components}"
  : "${output_dir:=$REPO_ROOT/build}"

  # Resolve each requested component to a file path.
  local heavyiq_file="" immerse_file="" webserver_file=""
  if [ "$want_heavyiq" -eq 1 ]; then
    if [ -n "$heavyiq_file_arg" ]; then
      heavyiq_file="$heavyiq_file_arg"
    else
      # Match both ubuntu and rocky artifacts; prefer the one matching --distro.
      local heavyiq_pattern="dist-*.tgz"
      case "$distro" in
        rockylinux*) heavyiq_pattern="dist-rocky-*.tgz" ;;
        ubuntu*)     heavyiq_pattern="dist-ubuntu-*.tgz" ;;
      esac
      heavyiq_file=$(find "$component_dir" -maxdepth 1 -name "$heavyiq_pattern" 2>/dev/null | sort | head -1 || true)
      if [ -z "$heavyiq_file" ]; then
        echo "ERROR: --heavyiq requested but no $heavyiq_pattern found in $component_dir" >&2
        echo "Run: dev-tools/dev.sh build heavyiq" >&2
        exit 1
      fi
    fi
  fi
  if [ "$want_immerse" -eq 1 ]; then
    if [ -n "$immerse_file_arg" ]; then
      immerse_file="$immerse_file_arg"
    else
      immerse_file=$(find "$component_dir" -maxdepth 1 -name '*.zip' 2>/dev/null | sort | head -1 || true)
      if [ -z "$immerse_file" ]; then
        echo "ERROR: --immerse requested but no *.zip found in $component_dir" >&2
        echo "Run: dev-tools/dev.sh build immerse" >&2
        exit 1
      fi
    fi
  fi
  if [ "$want_webserver" -eq 1 ]; then
    if [ -n "$webserver_file_arg" ]; then
      webserver_file="$webserver_file_arg"
    else
      webserver_file=$(find "$component_dir" -maxdepth 1 -name '*heavy_web_server.tar.gz' 2>/dev/null | sort | head -1 || true)
      if [ -z "$webserver_file" ]; then
        echo "ERROR: --webserver requested but no *heavy_web_server.tar.gz found in $component_dir" >&2
        echo "Run: dev-tools/dev.sh build webserver" >&2
        exit 1
      fi
    fi
  fi

  local nproc
  nproc=$(_resolve_nproc "$nproc_arg")

  local cuda_flag rendering_flag prefer_static_flag cc cxx
  [ "$cuda" = "cuda" ]       && cuda_flag="ON"  || cuda_flag="OFF"
  [ "$render" = "renderer" ] && rendering_flag="ON" || rendering_flag="OFF"
  [ "$lib_type" = "static" ] && prefer_static_flag="ON" || prefer_static_flag="OFF"
  [ "$compiler" = "clang" ]  && { cc="clang"; cxx="clang++"; } || { cc="gcc"; cxx="g++"; }

  local immerse_download="OFF" heavyiq_download="OFF"
  local cmake_extra=""
  local extra_mounts=()

  if [ -n "$heavyiq_file" ]; then
    heavyiq_download="ON"
    cmake_extra="${cmake_extra} -DHEAVYIQ_FILE=/tmp/heavyiq-dist.tgz"
    extra_mounts+=(-v "${heavyiq_file}:/tmp/heavyiq-dist.tgz:ro")
    # HeavyIQ is an ExternalProject whose stamp marks the extraction done, so a
    # freshly built dist.tgz arriving at the same path is ignored and the stale
    # tree gets packaged instead. Clear the extraction and the stamps. These
    # dirs are root-owned (created inside Docker), so use the deps container.
    for stale_dir in \
        "$output_dir/HeavyIQ/external" \
        "$output_dir/heavyiq"; do
      if [ -d "$stale_dir" ]; then
        echo "Clearing cached HeavyIQ extraction: $stale_dir" >&2
        _docker_wipe_dir "$deps_image" "$stale_dir"
      fi
    done
  fi
  if [ -n "$immerse_file" ]; then
    immerse_download="ON"
    cmake_extra="${cmake_extra} -DMAPD_IMMERSE_FILE=/tmp/immerse-frontend.zip"
    extra_mounts+=(-v "${immerse_file}:/tmp/immerse-frontend.zip:ro")
    # Invalidate cmake's ExternalProject cache for Immerse so it re-extracts
    # the new zip. These dirs are root-owned (created inside Docker), so use
    # the deps container to remove them. Must clear both the extracted source
    # AND the stamp files cmake uses to track completion.
    for stale_dir in \
        "$output_dir/external" \
        "$output_dir/Immerse/external" \
        "$output_dir/frontend"; do
      if [ -d "$stale_dir" ]; then
        echo "Clearing cached Immerse extraction: $stale_dir" >&2
        _docker_wipe_dir "$deps_image" "$stale_dir"
      fi
    done
  fi
  if [ -n "$webserver_file" ]; then
    # Webserver packaging is gated by MAPD_IMMERSE_DOWNLOAD in cmake.
    immerse_download="ON"
    cmake_extra="${cmake_extra} -DMAPD_WEBSERVER_FILE=/tmp/immerse-webserver.tar.gz"
    extra_mounts+=(-v "${webserver_file}:/tmp/immerse-webserver.tar.gz:ro")
  fi

  mkdir -p "$output_dir"
  _BUILD_LOG_DIR="$output_dir/logs"
  mkdir -p "$_BUILD_LOG_DIR"

  # If the repo has a java/.mvn/maven.config that references a host-specific
  # settings path it will be unreachable inside the container. Shadow it with a
  # tmpfile that points to the container-internal path instead.
  local maven_config_tmp
  maven_config_tmp=$(mktemp)
  printf -- '-s\n/workspace/java/.mvn/settings.xml\n' > "$maven_config_tmp"
  extra_mounts+=(-v "${maven_config_tmp}:/workspace/java/.mvn/maven.config:ro")

  # For git worktrees the .git file contains an absolute host path. Mount the
  # common git dir at that same path so git inside the container can follow it.
  local git_common_dir
  git_common_dir=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
  # rev-parse returns a relative path when not in a worktree; make it absolute.
  [[ "$git_common_dir" != /* ]] && git_common_dir="$REPO_ROOT/$git_common_dir"
  if [ -n "$git_common_dir" ] && [ "$git_common_dir" != "$REPO_ROOT/.git" ]; then
    extra_mounts+=(-v "${git_common_dir}:${git_common_dir}:ro")
  fi

  # Extract the CUDA version from the deps image name so it can be recorded
  # in CMakeCache.txt via -DHEAVYAI_DEPS_CUDA_VERSION. This lets downstream
  # commands (e.g. 'test <variant>') identify the exact deps image used.
  local deps_cuda_version
  deps_cuda_version=$(echo "${deps_image%%:*}" \
    | sed -n 's/.*cuda\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')

  echo "Building heavydb:" >&2
  echo "  deps image:  $deps_image" >&2
  echo "  recipe:      $compiler / $render / $cuda / $lib_type / $build_type" >&2
  echo "  source:      $REPO_ROOT" >&2
  echo "  output:      $output_dir" >&2
  [ -n "$heavyiq_file" ]   && echo "  heavyiq:     $heavyiq_file" >&2
  [ -n "$immerse_file" ]   && echo "  immerse:     $immerse_file" >&2
  [ -n "$webserver_file" ] && echo "  webserver:   $webserver_file" >&2

  local cpack_cmd=""
  [ "$no_package" -eq 0 ] && cpack_cmd="cpack -G TGZ"

  # --clean: wipe root-owned build artifacts in a separate docker run BEFORE
  # the logged build, so the log file is never inside the directory being wiped.
  if [ "$clean" -eq 1 ]; then
    echo "Cleaning $output_dir ..." >&2
    # shellcheck disable=SC2086
    docker run --rm \
      -v "$output_dir:/workspace/build" \
      -e USER=root \
      "$deps_image" \
      bash -c 'find /workspace/build -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
    # Recreate the log directory which the clean step just removed.
    mkdir -p "$_BUILD_LOG_DIR"
  fi

  _run_logged "heavydb cmake+make" "${_BUILD_LOG_DIR}/heavydb.log" \
    docker run --rm \
      -v "$REPO_ROOT:/workspace" \
      -v "$output_dir:/workspace/build" \
      "${extra_mounts[@]}" \
      -e USER=root \
      "$deps_image" \
      bash -c "
        set -euo pipefail
        git config --global --add safe.directory /workspace
        mkdir -p ~/.m2
        cp /workspace/java/.mvn/settings.xml ~/.m2/settings.xml
        # Source the deps environment (expanded inline by the host shell).
        ${_DEPS_ENV_SOURCE}
        cd /workspace/build
        # Wipe the cmake cache if key options changed or the cached compiler is gone.
        if [ -f CMakeCache.txt ]; then
          cached_cuda=\$(grep '^ENABLE_CUDA:BOOL=' CMakeCache.txt | cut -d= -f2 || true)
          cached_type=\$(grep '^CMAKE_BUILD_TYPE:STRING=' CMakeCache.txt | cut -d= -f2 | tr '[:upper:]' '[:lower:]' || true)
          cached_static=\$(grep '^PREFER_STATIC_LIBS:BOOL=' CMakeCache.txt | cut -d= -f2 || true)
          cached_cc=\$(grep '^CMAKE_C_COMPILER:FILEPATH=' CMakeCache.txt | cut -d= -f2 || true)
          want_cuda=${cuda_flag}
          want_type=\$(echo ${build_type} | tr '[:upper:]' '[:lower:]')
          want_static=${prefer_static_flag}
          stale=0
          [ \"\${cached_cuda}\" != \"\${want_cuda}\" ]   && stale=1
          [ \"\${cached_type}\" != \"\${want_type}\" ]   && stale=1
          [ \"\${cached_static}\" != \"\${want_static}\" ] && stale=1
          [ -n \"\${cached_cc}\" ] && [ ! -f \"\${cached_cc}\" ] && stale=1
          if [ \"\${stale}\" -eq 1 ]; then
            echo 'CMakeCache.txt is stale (options changed or compiler not found) — removing' >&2
            rm CMakeCache.txt
          fi
        fi
        cmake \
          -DCMAKE_BUILD_TYPE=${build_type} \
          -DENABLE_CUDA=${cuda_flag} \
          -DENABLE_RENDERING=${rendering_flag} \
          -DPREFER_STATIC_LIBS=${prefer_static_flag} \
          -DMAPD_IMMERSE_DOWNLOAD=${immerse_download} \
          -DHEAVYIQ_DOWNLOAD=${heavyiq_download} \
          -DMAPD_PACKAGE_DISTRO_NAME=${effective_distro} \
          -DHEAVYAI_DEPS_CUDA_VERSION=${deps_cuda_version} \
          -DENABLE_TESTS=OFF \
          -DENABLE_RENDER_TESTS=OFF \
          ${cmake_extra} \
          /workspace
        make -j ${nproc}
        ${cpack_cmd}
      "
  rm -f "$maven_config_tmp"

  echo "Build complete. Output: $output_dir" >&2
  if [ "$no_package" -eq 0 ]; then
    find "$output_dir" -maxdepth 1 -name '*.tar.gz' ! -name '*-docker.tar.gz' | sort | while read -r f; do
      echo "Package: $f" >&2
    done
  fi

  # Optional product Docker image build — runs on the host after cpack, no DinD needed.
  if [ "$build_docker" -eq 1 ] && [ "$no_package" -eq 0 ]; then
    local tarball
    tarball=$(find "$output_dir" -maxdepth 1 -name '*.tar.gz' ! -name '*-docker.tar.gz' | sort -r | head -1 || true)
    if [ -z "$tarball" ]; then
      echo "ERROR: --docker requested but no tar.gz found in $output_dir" >&2
      exit 1
    fi

    # CPU product containers are not supported — no NVIDIA-approved CPU base image.
    # Compilation with --cpu is fine; only containerisation is blocked.
    case "$(basename "$tarball")" in
      *cpu*)
        echo "ERROR: --docker is not supported for CPU builds (no NVIDIA-approved CPU base image)." >&2
        echo "Omit --docker to produce a tarball only, or use a CUDA build for containerisation." >&2
        exit 1 ;;
    esac
    # Auto-detect product base image from deps image name (distro + cuda version).
    if [ -z "$product_base_image" ]; then
      local image_basename cuda_ver
      image_basename="${deps_image%%:*}"
      cuda_ver=$(echo "$image_basename" | sed 's/.*cuda\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/')
      case "$image_basename" in
        *rockylinux8*) product_base_image="nvcr.io/nvidia/cuda:${cuda_ver}-runtime-rockylinux8" ;;
        *ubuntu22.04*) product_base_image="nvcr.io/nvidia/cuda:${cuda_ver}-runtime-ubuntu22.04" ;;
        *)
          echo "ERROR: cannot auto-detect product base image from '$deps_image'" >&2
          echo "Use --product-base-image=<image>" >&2
          exit 1 ;;
      esac
    fi

    _build_product_docker_image \
      "$tarball" "$effective_distro" "$render" "$product_base_image" "$_BUILD_LOG_DIR"
  fi
}

# ---------------------------------------------------------------------------
# _resolve_product_base_image: find a locally-available NVIDIA CUDA runtime base
# image for the given distro. Only CUDA product containers are supported.
#
# Scans local Docker images for nvcr.io/nvidia/cuda:*-runtime-<distro>.
# Prints the image ref on success. Exits with an error if none is found or if
# multiple CUDA runtime versions are present (ambiguous).
#
# Args:
#   $1  effective_distro   ubuntu22.04 | rockylinux8
# ---------------------------------------------------------------------------
_resolve_product_base_image() {
  local effective_distro="$1"

  # Scan for any locally-available runtime image for this distro.
  local candidates
  candidates=$(docker images --format '{{.Repository}}:{{.Tag}}' \
    | grep -E "^nvcr\.io/nvidia/cuda:[0-9.]+-runtime-${effective_distro}$" \
    | grep -v '<none>' || true)

  local count
  count=$(echo "$candidates" | grep -c . || true)

  if [ "$count" -eq 0 ]; then
    echo "ERROR: no CUDA runtime base image found for '${effective_distro}' in local Docker images." >&2
    echo "Pull one with:" >&2
    echo "  docker pull nvcr.io/nvidia/cuda:<version>-runtime-${effective_distro}" >&2
    echo "or specify a base image explicitly:" >&2
    echo "  --product-base-image=nvcr.io/nvidia/cuda:<version>-runtime-${effective_distro}" >&2
    exit 1
  fi

  if [ "$count" -gt 1 ]; then
    echo "ERROR: multiple CUDA runtime base images found for '${effective_distro}':" >&2
    echo "$candidates" | sed 's/^/  /' >&2
    echo "Specify which one to use:" >&2
    echo "  --product-base-image=<image>" >&2
    exit 1
  fi

  echo "$candidates"
}

# ---------------------------------------------------------------------------
# _build_product_docker_image: shared helper used by cmd_build_heavydb (--docker)
# and cmd_build_image. All base-image resolution happens before this call.
#
# Args:
#   $1  tarball            absolute path to the product .tar.gz
#   $2  effective_distro   ubuntu22.04 | rockylinux8
#   $3  render             renderer | norender
#   $4  product_base_image base Docker image (required; passed as --build-arg BASE_IMAGE)
#   $5  log_dir            directory for docker-build.log
#   $6  override_tag       optional: if set, used as the image tag instead of
#                          the version-date-sha parsed from the tarball name
# ---------------------------------------------------------------------------
_build_product_docker_image() {
  local tarball="$1"
  local effective_distro="$2"
  local render="$3"
  local product_base_image="$4"
  local log_dir="$5"
  local override_tag="${6:-}"

  local dockerfile="$REPO_ROOT/docker/Dockerfile.cuda"

  local product_tarball_name host_arch
  host_arch="$(uname -m)"
  product_tarball_name="$(basename "$tarball")"

  # Derive a local image name encoding all variant dimensions so multiple builds
  # coexist: heavyai-<distro>-cuda-<render|norender>-<arch>:<version>-YYYYMMDD-<sha>
  # Also tag the same image as :latest for convenient local use.
  local tag
  if [ -n "$override_tag" ]; then
    tag="$override_tag"
  else
    local tarball_stem
    tarball_stem=$(basename "$tarball" .tar.gz | sed 's/^[^0-9]*//')
    tag=$(echo "$tarball_stem" | sed -E 's/^([0-9]+(\.[0-9]+)*[[:alnum:]]*)-([0-9]{8})-([[:xdigit:]]{10}).*/\1-\3-\4/')
    if [ "$tag" = "$tarball_stem" ]; then
      echo "ERROR: could not parse version-date-sha from tarball name: $(basename "$tarball")" >&2
      echo "Use --tag=<tag> to supply an explicit image tag." >&2
      exit 1
    fi
  fi

  local render_tag product_repo product_image product_image_latest
  if [ "$render" = "renderer" ]; then
    render_tag="render"
  else
    render_tag="norender"
  fi
  product_repo="heavyai-${effective_distro}-cuda-${render_tag}-${host_arch}"
  product_image="${product_repo}:${tag}"
  product_image_latest="${product_repo}:latest"

  echo "--- Building product Docker image ---" >&2
  echo "  tarball:       $tarball" >&2
  echo "  base image:    $product_base_image" >&2
  echo "  product image: $product_image" >&2
  echo "  also tagged:   $product_image_latest" >&2

  mkdir -p "$log_dir"
  local ctx_dir
  ctx_dir=$(mktemp -d)
  trap "rm -rf '$ctx_dir'" EXIT
  cp "$tarball" "$ctx_dir/$product_tarball_name"
  cp "$dockerfile" "$ctx_dir/Dockerfile"

  _run_logged "product docker build" "${log_dir}/docker-build.log" \
    docker build -t "$product_image" -t "$product_image_latest" \
      --build-arg BASE_IMAGE="$product_base_image" \
      --build-arg TARBALL="$product_tarball_name" \
      "$ctx_dir"
  rm -rf "$ctx_dir"
  trap - EXIT

  echo "Product image: $product_image" >&2
  echo "Product image: $product_image_latest" >&2
}

# ---------------------------------------------------------------------------
# cmd_build_image: build a product Docker image from a pre-existing tarball.
# No compilation — useful when you have downloaded a pre-built release package.
# ---------------------------------------------------------------------------
cmd_build_image() {
  for arg in "$@"; do
    case "$arg" in --help|-h)
      cat <<'EOF'
Usage: dev-tools/dev.sh build image [options]

Builds a product Docker image from a heavydb tarball without recompiling.
The distro and render variant are parsed from the tarball name. Only CUDA
product containers are supported. The matching local NVIDIA CUDA runtime
base image is located automatically.

Options:
  --tar=<path>                  Path to the product .tar.gz (required).
  --product-base-image=<image>  Explicit base image. Skips local image
                                detection entirely. The distro is parsed
                                from this image name.
  --tag=<tag>                   Override the image tag. Default: parsed from
                                the tarball name as <version>-<YYYYMMDD>-<sha>.
  --verbose, -v                 Stream the docker build output to the shell as
                                well as the log file, instead of progress dots.

Examples:
  # Typical use — base image auto-detected from tarball name:
  dev-tools/dev.sh build image \
      --tar=heavyai-7.0.0-20260101-abc1234567-ubuntu22.04-x86_64.tar.gz

  # Explicit base image (e.g. to select among multiple local CUDA versions):
  dev-tools/dev.sh build image \
      --tar=heavyai-7.0.0-20260101-abc1234567-ubuntu22.04-x86_64.tar.gz \
      --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04
EOF
      return 0 ;;
    esac
  done

  local tarball="" product_base_image="" override_tag=""

  for arg in "$@"; do
    case "$arg" in
      --tar=*)                  tarball="${arg#*=}" ;;
      --product-base-image=*)   product_base_image="${arg#*=}" ;;
      --tag=*)                  override_tag="${arg#*=}" ;;
      *) echo "Unknown option: $arg (run with --help)" >&2; exit 1 ;;
    esac
  done

  if [ -z "$tarball" ]; then
    echo "ERROR: --tar=<path> is required" >&2
    echo "" >&2
    cmd_build_image --help >&2
    exit 1
  fi
  if [ ! -f "$tarball" ]; then
    echo "ERROR: tarball not found: $tarball" >&2
    exit 1
  fi
  # Resolve to an absolute path so docker can find it regardless of cwd.
  tarball="$(cd "$(dirname "$tarball")" && pwd)/$(basename "$tarball")"

  local tarball_name
  tarball_name="$(basename "$tarball")"

  # CPU product containers are not supported — no NVIDIA-approved CPU base image.
  case "$tarball_name" in
    *cpu*)
      echo "ERROR: CPU product containers are not supported (no NVIDIA-approved CPU base image)." >&2
      echo "Use --product-base-image to override if you have an appropriate base image." >&2
      exit 1 ;;
  esac
  # Derive render variant from the tarball name.
  local render
  case "$tarball_name" in
    *render*) render="renderer" ;;
    *)        render="norender" ;;
  esac

  # Resolve distro and base image. --product-base-image takes full precedence:
  # distro is parsed from the image name, and local image detection is skipped.
  # Without --product-base-image, distro must be present in the tarball name
  # (the new naming format from #8427).
  local effective_distro
  if [ -n "$product_base_image" ]; then
    # Derive distro from the explicit base image name so the local product
    # image name always reflects the actual base.
    effective_distro=$(_parse_distro "$product_base_image")
    if [ -z "$effective_distro" ]; then
      echo "ERROR: cannot determine distro from --product-base-image: $product_base_image" >&2
      echo "Expected the image name to contain ubuntu22.04 or rockylinux8." >&2
      exit 1
    fi
    # Cross-check against the tarball name when both are parseable, to catch
    # mismatches like an ubuntu22.04 tarball paired with a rockylinux8 base image.
    local tar_distro
    tar_distro=$(_parse_distro "$tarball_name")
    if [ -n "$tar_distro" ] && [ "$tar_distro" != "$effective_distro" ]; then
      echo "ERROR: distro mismatch — tarball is '$tar_distro' but --product-base-image specifies '$effective_distro'." >&2
      exit 1
    fi
  else
    # No explicit base image — derive distro from the tarball name
    # (requires #8427 naming: ...-ubuntu22.04-... etc.).
    effective_distro=$(_parse_distro "$tarball_name")
    if [ -z "$effective_distro" ]; then
      echo "ERROR: cannot determine distro from tarball name: $tarball_name" >&2
      echo "Expected the filename to contain ubuntu22.04 or rockylinux8." >&2
      echo "If using an older tarball (with 'Linux' in the name), specify the base" >&2
      echo "image explicitly: --product-base-image=<image>" >&2
      exit 1
    fi
    product_base_image=$(_resolve_product_base_image "$effective_distro")
    echo "Auto-detected base image: $product_base_image" >&2
  fi

  local log_dir="$REPO_ROOT/build/logs"
  _build_product_docker_image \
    "$tarball" "$effective_distro" "$render" "$product_base_image" "$log_dir" "$override_tag"
}
