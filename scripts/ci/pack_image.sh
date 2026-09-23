#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Pack a curated set of build outputs into a thin Docker image and push it
# to GHCR. The image is `FROM` the deps base image; each entry of
# INCLUDE_PATHS is staged into a temp dir preserving its relative path,
# optionally stripped, and then COPY'd into /workspace.
#
# Invoked from .github/workflows/pr-required-checks.yml; also locally
# runnable if you have a docker daemon and GHCR credentials.
#
# Required env vars:
#   IMAGE_NAME     - short package name (e.g. "pr-build/debug-static")
#   IMAGE_TAG      - tag (e.g. "${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}")
#   BASE_IMAGE     - the FROM image (the deps build container)
#
# Optional env vars:
#   INCLUDE_PATHS  - space-separated paths or shell globs (relative to repo
#                    root) to include in the image, preserving directory
#                    structure under /workspace. Default: "build Tests
#                    scripts" (whole build tree, intermediates pruned).
#   EXCLUDE_PATTERNS - space-separated `find -name` patterns to delete from
#                    the staged tree after copying. Default prunes the
#                    common build intermediates (*.o, CMakeFiles, etc.)
#                    that are never needed at runtime. Override only if
#                    you need to keep one of those.
#   STRIP          - "debug" runs `strip --strip-debug` on ELF binaries
#                    found in the staged set (function names preserved,
#                    debug info dropped). "none" (default) leaves binaries
#                    untouched.
#   DEBUG_ARTIFACTS_DIR - if set together with STRIP=debug, the DWARF info
#                    that strip would discard is first extracted via
#                    `objcopy --only-keep-debug` into <dir>/<rel-path>.debug
#                    so it can be uploaded as a separate artifact. The dir
#                    lives outside $STAGING so the .debug files don't ship
#                    in the image. Lets us strip for image-size reasons
#                    while still having full DWARF available offline for
#                    forensics (e.g. resolving TSAN race PCs to source
#                    lines via addr2line).
#   REGISTRY       - registry host prefix (default: "ghcr.io")
#   REPO_SLUG      - "owner/repo" (default: from $GITHUB_REPOSITORY, else
#                    from git remote)
#   REGISTRY_USER, REGISTRY_PASSWORD - GHCR credentials. In CI these come
#                    from $GITHUB_ACTOR and $GITHUB_TOKEN.
#
# Outputs:
#   Echoes "image=<full image ref>" — captured by the workflow via
#   $GITHUB_OUTPUT.
#
set -euo pipefail

: "${IMAGE_NAME:?IMAGE_NAME must be set}"
: "${IMAGE_TAG:?IMAGE_TAG must be set}"
: "${BASE_IMAGE:?BASE_IMAGE must be set}"

INCLUDE_PATHS="${INCLUDE_PATHS:-build Tests scripts}"
# Two-tier pruning: source-file patterns are pruned only from build/ (where
# they appear as configured/generated copies), preserving the same patterns
# under shipped source dirs like QueryEngine/ — UdfTest etc. compile against
# those headers at runtime, so stripping them globally breaks the test.
EXCLUDE_PATTERNS_BUILD="${EXCLUDE_PATTERNS_BUILD:-*.cpp *.cc *.cxx *.h *.hpp *.hxx *.in}"
# Pure build intermediates — never appear in source, safe to prune anywhere.
EXCLUDE_PATTERNS="${EXCLUDE_PATTERNS:-*.o *.obj lib*.a CMakeFiles cmake_install.cmake compile_commands.json Makefile build.ninja rules.ninja .ninja_log .ninja_deps}"
STRIP="${STRIP:-none}"
REGISTRY="${REGISTRY:-ghcr.io}"

if [ -z "${REPO_SLUG:-}" ]; then
  if [ -n "${GITHUB_REPOSITORY:-}" ]; then
    REPO_SLUG="$GITHUB_REPOSITORY"
  else
    origin_url="$(git config --get remote.origin.url)"
    REPO_SLUG="$(echo "$origin_url" | sed -E 's#.*[:/]([^/]+/[^/]+)\.git$#\1#')"
  fi
fi

: "${REGISTRY_USER:=${GITHUB_ACTOR:-}}"
: "${REGISTRY_PASSWORD:=${GITHUB_TOKEN:-}}"

if [ -z "$REGISTRY_USER" ] || [ -z "$REGISTRY_PASSWORD" ]; then
  echo "ERROR: REGISTRY_USER and REGISTRY_PASSWORD must be set (or GITHUB_ACTOR/GITHUB_TOKEN in CI)" >&2
  exit 1
fi

# Install Docker client if not already present
if ! command -v docker >/dev/null 2>&1; then
  echo "docker CLI not on PATH in this image — installing docker.io"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends docker.io >/dev/null
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

# Stage each path. Unquoted $INCLUDE_PATHS gets word-splitting; nullglob
# makes unmatched globs silently expand to nothing instead of erroring;
# globstar makes ** in INCLUDE_PATHS expand recursively.
shopt -s nullglob globstar
for raw in $INCLUDE_PATHS; do
  matches=( $raw )
  if [ ${#matches[@]} -eq 0 ]; then
    echo "warning: INCLUDE_PATHS entry '$raw' matched nothing, skipping" >&2
    continue
  fi
  for f in "${matches[@]}"; do
    if [ ! -e "$f" ]; then
      # Literal paths (no glob) pass through the matches array even when
      # they don't exist; skip with a warning so a missing optional dir
      # (e.g. build/lib64 when there's no 64-bit lib output) isn't fatal.
      echo "warning: $f does not exist, skipping" >&2
      continue
    fi
    # cp --parents preserves the relative path under $STAGING.
    cp -a --parents "$f" "$STAGING/"
  done
done
shopt -u nullglob globstar

# Prune build intermediates from staging. These are never needed at runtime
# and dominate the size of an unfiltered build/ tree.
if [ -n "${EXCLUDE_PATTERNS_BUILD:-}" ] && [ -d "$STAGING/build" ]; then
  echo "Pruning source/config artifacts (${EXCLUDE_PATTERNS_BUILD}) from $STAGING/build/..."
  for pattern in $EXCLUDE_PATTERNS_BUILD; do
    find "$STAGING/build" -name "$pattern" -prune -exec rm -rf {} + 2>/dev/null || true
  done
fi
if [ -n "$EXCLUDE_PATTERNS" ]; then
  echo "Pruning intermediates ($EXCLUDE_PATTERNS) from staging..."
  for pattern in $EXCLUDE_PATTERNS; do
    find "$STAGING" -name "$pattern" -prune -exec rm -rf {} + 2>/dev/null || true
  done
fi

if [ "$STRIP" = "debug" ]; then
  if ! command -v file >/dev/null 2>&1; then
    echo "ERROR: STRIP=debug requested but 'file' is not installed in this container" >&2
    exit 1
  fi
  save_debug="${DEBUG_ARTIFACTS_DIR:-}"
  if [ -n "$save_debug" ]; then
    mkdir -p "$save_debug"
    echo "Stripping debug info; saving .debug files under $save_debug ..."
  else
    echo "Stripping debug info from ELF binaries..."
  fi
  stripped=0
  while IFS= read -r -d '' f; do
    # Match only real ELF binaries (executables and shared libs); skip
    # JARs, CSVs, JSON test fixtures, etc.
    if file -b "$f" 2>/dev/null | grep -q "^ELF "; then
      if [ -n "$save_debug" ]; then
        rel="${f#$STAGING/}"
        target_dir="$save_debug/$(dirname "$rel")"
        mkdir -p "$target_dir"
        objcopy --only-keep-debug "$f" "$save_debug/$rel.debug" 2>/dev/null || {
          echo "warning: objcopy --only-keep-debug failed on $f" >&2
        }
      fi
      strip --strip-debug "$f" 2>/dev/null || {
        echo "warning: strip failed on $f" >&2
        continue
      }
      # If we saved debug info separately, add a gnu_debuglink reference so
      # downstream symbolizers (gdb, llvm-symbolizer, TSAN's runtime) will
      # auto-pick up the matching .debug file when it lands beside the
      # binary at test time.
      if [ -n "$save_debug" ]; then
        bn="$(basename "$f")"
        # objcopy expects the debug link name without the directory; the
        # search path used at debug time is the same dir as the binary.
        objcopy --add-gnu-debuglink="$save_debug/$rel.debug" "$f" 2>/dev/null || {
          echo "warning: objcopy --add-gnu-debuglink failed on $f" >&2
        }
      fi
      stripped=$((stripped + 1))
    fi
  done < <(find "$STAGING" -type f -print0)
  echo "Stripped $stripped binaries."
  if [ -n "$save_debug" ]; then
    echo ".debug archive size (uncompressed): $(du -sh "$save_debug" | cut -f1)"
  fi
fi

# Free disk before docker build. The original build/ tree (which we just
# staged from) can be ~30 GB for TSAN partitions and isn't needed again —
# docker build only reads from $STAGING. Skippable via FREE_BUILD=false
# for local debugging.
if [ "${FREE_BUILD:-true}" = "true" ] && [ -d build ]; then
  echo "Freeing ./build to make room for docker build context..."
  rm -rf build
fi

# Write the Dockerfile outside the build context so it doesn't bake into
# the image itself.
DOCKERFILE="$(mktemp)"
trap 'rm -rf "$STAGING" "$DOCKERFILE"' EXIT
cat > "$DOCKERFILE" <<EOF
FROM ${BASE_IMAGE}
WORKDIR /workspace
COPY . /workspace/
EOF

IMAGE_REF="${REGISTRY}/${REPO_SLUG}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Staged size (uncompressed): $(du -sh "$STAGING" | cut -f1)"

echo "$REGISTRY_PASSWORD" | docker login "$REGISTRY" -u "$REGISTRY_USER" --password-stdin
docker build -f "$DOCKERFILE" -t "$IMAGE_REF" "$STAGING"
docker push "$IMAGE_REF"

echo "image=${IMAGE_REF}"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "image=${IMAGE_REF}" >> "$GITHUB_OUTPUT"
fi
