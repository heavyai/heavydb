#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# test-cli.sh: CLI validation tests for dev-tools/dev.sh.
#
# Flags:
#   (none)                    Syntax, help, error-handling, blocked-path tests.
#                             No Docker required.
#   --mock-docker             Also test Docker resolver error messages
#                             (mock docker returns no images).
#   --mock-docker-extended    Also test _resolve_deps_image / _resolve_product_base_image
#                             logic with various mock image-list scenarios (Tier 3).
#   --docker-smoke            End-to-end Docker build smoke test: pulls ubuntu:22.04,
#                             creates a minimal tarball, runs 'build image', verifies
#                             the resulting image exists (Tier 2). Requires real Docker.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEV_SH="$REPO_ROOT/dev-tools/dev.sh"

PASS=0
FAIL=0
ERRORS=()

MOCK_DOCKER=0
MOCK_DOCKER_EXTENDED=0
DOCKER_SMOKE=0

for arg in "$@"; do
  case "$arg" in
    --mock-docker)          MOCK_DOCKER=1 ;;
    --mock-docker-extended) MOCK_DOCKER_EXTENDED=1 ;;
    --docker-smoke)         DOCKER_SMOKE=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

NEED_MOCK=$(( MOCK_DOCKER || MOCK_DOCKER_EXTENDED ))

if [ "$NEED_MOCK" -eq 1 ] && [ "$DOCKER_SMOKE" -eq 1 ]; then
  echo "ERROR: --docker-smoke cannot be combined with --mock-docker or --mock-docker-extended" >&2
  echo "       (the mock docker stub would intercept the smoke test's real docker calls)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
FAKE_TAR_DIR=""
MOCK_BIN_DIR=""
SMOKE_IMAGE=""
SMOKE_DIR=""
_cleanup() {
  [ -n "$FAKE_TAR_DIR" ] && rm -rf "$FAKE_TAR_DIR"
  [ -n "$MOCK_BIN_DIR" ] && rm -rf "$MOCK_BIN_DIR"
  [ -n "$SMOKE_DIR" ]    && rm -rf "$SMOKE_DIR"
  if [ -n "$SMOKE_IMAGE" ]; then
    docker rmi "${SMOKE_IMAGE}:smoketest" "${SMOKE_IMAGE}:latest" \
               "nvcr.io/nvidia/cuda:0.0.0-runtime-ubuntu22.04" 2>/dev/null || true
  fi
}
trap _cleanup EXIT

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

assert_exit_0() {
  local desc="$1"; shift
  local rc=0
  "$@" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "✅ $desc"; ((PASS++))
  else
    echo "❌ $desc (expected exit 0, got $rc)"
    ERRORS+=("$desc"); ((FAIL++))
  fi
}

assert_exit_nonzero() {
  local desc="$1"; shift
  local rc=0
  "$@" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "✅ $desc"; ((PASS++))
  else
    echo "❌ $desc (expected non-zero exit, got 0)"
    ERRORS+=("$desc"); ((FAIL++))
  fi
}

# assert_output_contains <desc> <pattern> <cmd...>
# Passes if the command output contains <pattern> (exit code ignored).
assert_output_contains() {
  local desc="$1" pattern="$2"; shift 2
  local output rc=0
  output=$("$@" 2>&1) || rc=$?
  if echo "$output" | grep -qF -- "$pattern"; then
    echo "✅ $desc"; ((PASS++))
  else
    echo "❌ $desc (pattern not found: '$pattern')"
    echo "   Output: $(echo "$output" | head -3 | sed 's/^/     /')"
    ERRORS+=("$desc"); ((FAIL++))
  fi
}

# assert_succeeds_with <desc> <pattern> <cmd...>
# Passes only when the command exits 0 AND the output contains <pattern>.
# Use for success-path tests where both correct output and clean exit matter.
assert_succeeds_with() {
  local desc="$1" pattern="$2"; shift 2
  local output rc=0
  output=$("$@" 2>&1) || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "❌ $desc (expected exit 0, got $rc)"
    echo "   Output: $(echo "$output" | head -3 | sed 's/^/     /')"
    ERRORS+=("$desc"); ((FAIL++))
  elif echo "$output" | grep -qF -- "$pattern"; then
    echo "✅ $desc"; ((PASS++))
  else
    echo "❌ $desc (exited 0 but pattern not found: '$pattern')"
    echo "   Output: $(echo "$output" | head -3 | sed 's/^/     /')"
    ERRORS+=("$desc"); ((FAIL++))
  fi
}

# assert_fails_with <desc> <pattern> <cmd...>
# Passes only when the command exits non-zero AND the output contains <pattern>.
# Use for error-path tests where both the failure and the message matter.
assert_fails_with() {
  local desc="$1" pattern="$2"; shift 2
  local output rc=0
  output=$("$@" 2>&1) || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "❌ $desc (expected non-zero exit, got 0)"
    ERRORS+=("$desc"); ((FAIL++))
  elif echo "$output" | grep -qF -- "$pattern"; then
    echo "✅ $desc"; ((PASS++))
  else
    echo "❌ $desc (exited $rc but pattern not found: '$pattern')"
    echo "   Output: $(echo "$output" | head -3 | sed 's/^/     /')"
    ERRORS+=("$desc"); ((FAIL++))
  fi
}

# ---------------------------------------------------------------------------
# Setup: fake tarballs (content irrelevant for error-path and output tests)
# ---------------------------------------------------------------------------
FAKE_TAR_DIR="$(mktemp -d)"

# Tarballs for error tests
CUDA_TAR="$FAKE_TAR_DIR/heavyai-10.0.0dev-20260101-abc1234567-ubuntu22.04-x86_64-render.tar.gz"
CPU_TAR="$FAKE_TAR_DIR/heavyai-10.0.0dev-20260101-abc1234567-ubuntu22.04-x86_64-cpu.tar.gz"
UNVERSIONED_TAR="$FAKE_TAR_DIR/heavyai-nover-ubuntu22.04-x86_64.tar.gz"
touch "$CUDA_TAR" "$CPU_TAR" "$UNVERSIONED_TAR"

# Additional tarballs for _parse_distro and version-tag tests
ROCKY_TAR="$FAKE_TAR_DIR/heavyai-7.0.0-20260101-abc1234567-rockylinux8-x86_64-render.tar.gz"
NORENDER_TAR="$FAKE_TAR_DIR/heavyai-7.0.0-20260101-abc1234567-ubuntu22.04-x86_64.tar.gz"
touch "$ROCKY_TAR" "$NORENDER_TAR"

# ---------------------------------------------------------------------------
# Setup: scenario-aware mock docker (installed when --mock-docker* is given)
# ---------------------------------------------------------------------------
if [ "$NEED_MOCK" -eq 1 ]; then
  MOCK_BIN_DIR="$(mktemp -d)"
  cat > "$MOCK_BIN_DIR/docker" <<'MOCK'
#!/bin/bash
# Scenario-aware docker stub for test-cli.sh.
# Behaviour controlled by MOCK_DOCKER_SCENARIO (default: empty).
scenario="${MOCK_DOCKER_SCENARIO:-empty}"
# Use the host arch so _resolve_deps_image's grep filter matches.
arch="$(uname -m)"

if [ "${1:-}" != "images" ]; then exit 0; fi

# Distinguish _resolve_deps_image calls (format string starts with {{.ID}})
# from _resolve_product_base_image calls (format string starts with {{.Repository}}).
# docker images is invoked with --format as a separate argument (not --format=),
# so we collect the value of the argument that follows --format.
_fmt_arg=""
_prev=""
for _a in "$@"; do
  [ "$_prev" = "--format" ] && { _fmt_arg="$_a"; break; }
  _prev="$_a"
done
if echo "$_fmt_arg" | grep -qF '{{.ID}}'
then
  # --- Deps image candidates: ID<TAB>Repository:Tag ---
  case "$scenario" in
    one-deps-ubuntu22)
      printf 'abc123def456\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:latest\n' "$arch"
      ;;
    two-same-id-deps)
      # Same image ID, two tags → deduplication collapses to one
      printf 'abc123def456\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:latest\n' "$arch"
      printf 'abc123def456\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:20260101\n' "$arch"
      ;;
    two-diff-id-deps)
      # Two distinct images → ambiguity error
      printf 'abc123def456\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:latest\n' "$arch"
      printf 'fed987654321\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:20260101\n' "$arch"
      ;;
    multi-distro-deps)
      # Ubuntu and Rocky present → needs --distro to disambiguate
      printf 'abc123def456\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:latest\n' "$arch"
      printf 'fed987654321\tghcr.io/heavyai/heavydb/core-build-rockylinux8-static-cuda12.9.2-%s:latest\n' "$arch"
      ;;
    multi-cuda-deps)
      # Same distro, two CUDA versions → needs --cuda-version to disambiguate
      printf 'abc123def456\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.9.2-%s:latest\n' "$arch"
      printf 'fed987654321\tghcr.io/heavyai/heavydb/core-build-ubuntu22.04-static-cuda12.6.0-%s:latest\n' "$arch"
      ;;
    # default (empty): no output
  esac
else
  # --- Runtime base image candidates: Repository:Tag ---
  case "$scenario" in
    one-runtime-ubuntu22)
      echo "nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04"
      ;;
    # default (empty): no output
  esac
fi
MOCK
  chmod +x "$MOCK_BIN_DIR/docker"
  export PATH="$MOCK_BIN_DIR:$PATH"
fi

# ---------------------------------------------------------------------------
# Section 1: Bash syntax checks
# ---------------------------------------------------------------------------
echo ""
echo "=== Syntax checks ==="

for script in \
    dev-tools/dev.sh \
    dev-tools/scripts/build-all.sh \
    dev-tools/scripts/build-component.sh \
    dev-tools/scripts/build-deps-image.sh \
    dev-tools/scripts/build-docs.sh \
    dev-tools/scripts/build-heavydb.sh \
    dev-tools/scripts/ci.sh \
    dev-tools/scripts/component-tests.sh \
    dev-tools/scripts/pyheavydb-tests.sh \
    dev-tools/scripts/sanity-tests.sh \
    dev-tools/scripts/test-cli.sh; do
  assert_exit_0 "bash -n $script" bash -n "$REPO_ROOT/$script"
done

# ---------------------------------------------------------------------------
# Section 2: Help / usage exits 0
# ---------------------------------------------------------------------------
echo ""
echo "=== Help exits 0 ==="

assert_exit_0 "dev.sh (no args)"         bash "$DEV_SH"
assert_exit_0 "dev.sh --help"            bash "$DEV_SH" --help
assert_exit_0 "build --help"             bash "$DEV_SH" build --help
assert_exit_0 "build heavydb --help"     bash "$DEV_SH" build heavydb --help
assert_exit_0 "build deps --help"        bash "$DEV_SH" build deps --help
assert_exit_0 "build image --help"       bash "$DEV_SH" build image --help
assert_exit_0 "build all --help"         bash "$DEV_SH" build all --help
assert_exit_0 "test --help"              bash "$DEV_SH" test --help
assert_exit_0 "shell --help"             bash "$DEV_SH" shell --help

# ---------------------------------------------------------------------------
# Section 3: Help content — key options must appear in help text
# ---------------------------------------------------------------------------
echo ""
echo "=== Help content ==="

assert_output_contains "build heavydb --help has --cuda-version" "--cuda-version" \
  bash "$DEV_SH" build heavydb --help

assert_output_contains "build all --help has --cuda-version" "--cuda-version" \
  bash "$DEV_SH" build all --help

assert_output_contains "build image --help has --tar" "--tar=" \
  bash "$DEV_SH" build image --help

assert_output_contains "build image --help has --product-base-image" "--product-base-image" \
  bash "$DEV_SH" build image --help

assert_output_contains "build --help has --verbose" "--verbose" \
  bash "$DEV_SH" build --help

assert_output_contains "build heavydb --help has --verbose" "--verbose" \
  bash "$DEV_SH" build heavydb --help

# ---------------------------------------------------------------------------
# Section 4: Unknown / invalid inputs
# ---------------------------------------------------------------------------
echo ""
echo "=== Unknown / invalid inputs ==="

assert_exit_nonzero    "unknown subcommand"          bash "$DEV_SH" bogus-subcommand-xyz
assert_exit_nonzero    "unknown build target"        bash "$DEV_SH" build unknown-target-xyz
assert_fails_with "unknown build flag"          "Unknown option" \
  bash "$DEV_SH" build --not-a-real-option
assert_fails_with "unknown build heavydb flag"  "Unknown option" \
  bash "$DEV_SH" build heavydb --not-a-real-option
assert_fails_with "unknown build image flag"    "Unknown option" \
  bash "$DEV_SH" build image --not-a-real-option --tar="$CUDA_TAR"
assert_fails_with "build deps bad distro"       "unsupported distro" \
  bash "$DEV_SH" build deps --distro=windows
assert_fails_with "build heavydb bad distro"    "unsupported distro" \
  bash "$DEV_SH" build heavydb --distro=windows
assert_fails_with "build all bad distro"        "unsupported distro" \
  bash "$DEV_SH" build --distro=windows
# --verbose is consumed by cmd_build, so the per-target parsers must never see
# it: these reach the distro check rather than failing with "Unknown option".
assert_fails_with "--verbose before target"     "unsupported distro" \
  bash "$DEV_SH" build --verbose --distro=windows
assert_fails_with "--verbose after target"      "unsupported distro" \
  bash "$DEV_SH" build heavydb --verbose --distro=windows
assert_fails_with "build ci nonexistent config" "not found" \
  bash "$DEV_SH" build ci nonexistent-config-xyz
assert_fails_with "test nonexistent variant"    "not found" \
  bash "$DEV_SH" test nonexistent-variant-xyz

# ---------------------------------------------------------------------------
# Section 5: Blocked paths — build image
# (assert_fails_with: must exit non-zero AND print the expected message)
# ---------------------------------------------------------------------------
echo ""
echo "=== Blocked: build image ==="

assert_fails_with "no --tar"                "--tar" \
  bash "$DEV_SH" build image
assert_fails_with "tarball not found"       "not found" \
  bash "$DEV_SH" build image --tar=/does/not/exist/tarball.tar.gz
assert_fails_with "CPU tarball blocked"     "CPU product containers" \
  bash "$DEV_SH" build image --tar="$CPU_TAR"
assert_fails_with "distro mismatch"         "distro mismatch" \
  bash "$DEV_SH" build image --tar="$CUDA_TAR" \
  --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-rockylinux8
assert_fails_with "unversioned name no --tag" "version-date-sha" \
  bash "$DEV_SH" build image --tar="$UNVERSIONED_TAR" \
  --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04
assert_fails_with "unrecognised base image" "cannot determine distro" \
  bash "$DEV_SH" build image --tar="$CUDA_TAR" \
  --product-base-image=my.registry.io/custom:latest

# ---------------------------------------------------------------------------
# Section 6: Blocked paths — build heavydb
# ---------------------------------------------------------------------------
echo ""
echo "=== Blocked: build heavydb ==="

assert_fails_with "--docker + --no-package" "mutually exclusive" \
  bash "$DEV_SH" build heavydb --docker --no-package

# ---------------------------------------------------------------------------
# Section 9: Docker resolver errors — zero-images case (--mock-docker)
# ---------------------------------------------------------------------------
if [ "$MOCK_DOCKER" -eq 1 ]; then
  echo ""
  echo "=== Docker resolver errors: zero images (mock) ==="

  assert_fails_with "no local deps image" "no local deps image found" \
    bash "$DEV_SH" build heavydb

  assert_fails_with "no CUDA runtime base image" "no CUDA runtime base image found" \
    bash "$DEV_SH" build image --tar="$CUDA_TAR"

  # ---------------------------------------------------------------------------
  # Section 7: _parse_distro — image name derivation from tarball
  # Under --mock-docker, 'docker build' and 'docker tag' are stubbed (exit 0),
  # so 'build image' fully succeeds. assert_succeeds_with verifies both the
  # correct image name AND a clean exit.
  # ---------------------------------------------------------------------------
  echo ""
  echo "=== _parse_distro: distro derived from tarball name ==="

  assert_succeeds_with "_parse_distro: ubuntu22.04" "heavyai-ubuntu22.04-cuda-render" \
    bash "$DEV_SH" build image --tar="$CUDA_TAR" \
    --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04 --tag=t

  assert_succeeds_with "_parse_distro: rockylinux8" "heavyai-rockylinux8-cuda-render" \
    bash "$DEV_SH" build image --tar="$ROCKY_TAR" \
    --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-rockylinux8 --tag=t

  # ---------------------------------------------------------------------------
  # Section 8: Version-tag and render-variant parsing from tarball name
  # Also under --mock-docker; assert_succeeds_with for the same reason.
  # ---------------------------------------------------------------------------
  echo ""
  echo "=== Tarball name parsing ==="

  assert_succeeds_with "version tag parsed correctly" \
    "heavyai-ubuntu22.04-cuda-render-$(uname -m):10.0.0dev-20260101-abc1234567" \
    bash "$DEV_SH" build image --tar="$CUDA_TAR" \
    --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04

  assert_succeeds_with "norender variant from tarball name" "cuda-norender" \
    bash "$DEV_SH" build image --tar="$NORENDER_TAR" \
    --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04 --tag=t

  assert_succeeds_with "render variant from tarball name" "cuda-render" \
    bash "$DEV_SH" build image --tar="$CUDA_TAR" \
    --product-base-image=nvcr.io/nvidia/cuda:12.9.2-runtime-ubuntu22.04 --tag=t
fi

# ---------------------------------------------------------------------------
# Section 10: _resolve_deps_image logic — multiple mock scenarios
#             (--mock-docker-extended)
# ---------------------------------------------------------------------------
if [ "$MOCK_DOCKER_EXTENDED" -eq 1 ]; then
  echo ""
  echo "=== _resolve_deps_image logic (extended mock) ==="

  assert_output_contains "one deps image → auto-detected" "Auto-detected deps image" \
    env MOCK_DOCKER_SCENARIO=one-deps-ubuntu22 bash "$DEV_SH" build heavydb --no-package

  assert_output_contains "two same-ID images → deduplicated to one" "Auto-detected deps image" \
    env MOCK_DOCKER_SCENARIO=two-same-id-deps bash "$DEV_SH" build heavydb --no-package

  assert_exit_nonzero      "two distinct images → exits non-zero" \
    env MOCK_DOCKER_SCENARIO=two-diff-id-deps bash "$DEV_SH" build heavydb
  assert_output_contains   "two distinct images → ambiguity error" "multiple deps images" \
    env MOCK_DOCKER_SCENARIO=two-diff-id-deps bash "$DEV_SH" build heavydb

  assert_exit_nonzero      "multi-distro → exits non-zero" \
    env MOCK_DOCKER_SCENARIO=multi-distro-deps bash "$DEV_SH" build heavydb
  assert_output_contains   "multi-distro → ambiguity error" "multiple deps images" \
    env MOCK_DOCKER_SCENARIO=multi-distro-deps bash "$DEV_SH" build heavydb

  assert_output_contains "multi-distro + --distro → resolves" "Auto-detected deps image" \
    env MOCK_DOCKER_SCENARIO=multi-distro-deps \
    bash "$DEV_SH" build heavydb --distro=ubuntu22.04 --no-package

  assert_exit_nonzero      "multi-cuda → exits non-zero" \
    env MOCK_DOCKER_SCENARIO=multi-cuda-deps bash "$DEV_SH" build heavydb --distro=ubuntu22.04
  assert_output_contains   "multi-cuda → ambiguity error" "multiple deps images" \
    env MOCK_DOCKER_SCENARIO=multi-cuda-deps bash "$DEV_SH" build heavydb --distro=ubuntu22.04

  assert_output_contains "multi-cuda + --cuda-version → resolves" "Auto-detected deps image" \
    env MOCK_DOCKER_SCENARIO=multi-cuda-deps \
    bash "$DEV_SH" build heavydb --distro=ubuntu22.04 --cuda-version=12.9.2 --no-package

  assert_output_contains "one runtime image → auto-detected" "Auto-detected base image" \
    env MOCK_DOCKER_SCENARIO=one-runtime-ubuntu22 \
    bash "$DEV_SH" build image --tar="$CUDA_TAR"
fi

# ---------------------------------------------------------------------------
# Section 11: End-to-end Docker smoke test (--docker-smoke)
# Pulls ubuntu:22.04, tags it as a fake CUDA runtime, creates a minimal
# product tarball, runs 'build image', and verifies the resulting local image.
# ---------------------------------------------------------------------------
if [ "$DOCKER_SMOKE" -eq 1 ]; then
  echo ""
  echo "=== Docker smoke test ==="

  SMOKE_ARCH="$(uname -m)"
  SMOKE_IMAGE="heavyai-ubuntu22.04-cuda-render-${SMOKE_ARCH}"
  SMOKE_DIR="$(mktemp -d)"
  SMOKE_TARNAME="heavyai-1.0.0-20260101-abc1234567-ubuntu22.04-${SMOKE_ARCH}-render.tar.gz"

  # Build a fake CUDA runtime image from ubuntu:22.04 with stubbed package-manager
  # commands. This lets Dockerfile.cuda's 'command -v apt-get' branch succeed
  # instantly without touching external repos or PPAs — making the smoke test
  # fast and network-independent.
  assert_exit_0 "smoke: build fake CUDA runtime base image" \
    docker build -t nvcr.io/nvidia/cuda:0.0.0-runtime-ubuntu22.04 - << 'SMOKEBASE'
FROM ubuntu:22.04
# Stub commands that Dockerfile.cuda invokes after the apt-get installs.
# /usr/local/bin takes priority over /usr/bin so these intercept the calls.
# update-alternatives is stubbed because clang++-14 is never actually
# installed, so the real binary would fail trying to register it.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/apt-get && \
    printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/add-apt-repository && \
    printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/update-alternatives && \
    chmod +x /usr/local/bin/apt-get \
              /usr/local/bin/add-apt-repository \
              /usr/local/bin/update-alternatives
SMOKEBASE

  # Create a minimal but valid tar.gz (strip-components=1 compatible).
  mkdir -p "$SMOKE_DIR/heavyai-1.0.0/bin"
  printf '#!/bin/sh\necho heavydb\n' > "$SMOKE_DIR/heavyai-1.0.0/bin/heavydb"
  chmod +x "$SMOKE_DIR/heavyai-1.0.0/bin/heavydb"
  tar czf "$SMOKE_DIR/$SMOKE_TARNAME" -C "$SMOKE_DIR" heavyai-1.0.0/

  # Run 'build image' end-to-end using the fake base image.
  assert_exit_0 "smoke: build image succeeds" \
    bash "$DEV_SH" build image \
      --tar="$SMOKE_DIR/$SMOKE_TARNAME" \
      --product-base-image=nvcr.io/nvidia/cuda:0.0.0-runtime-ubuntu22.04 \
      --tag=smoketest

  # Verify the product image exists with the correct name.
  assert_output_contains "smoke: product image tagged :smoketest" "${SMOKE_IMAGE}:smoketest" \
    docker images --format "{{.Repository}}:{{.Tag}}"

  assert_output_contains "smoke: product image tagged :latest" "${SMOKE_IMAGE}:latest" \
    docker images --format "{{.Repository}}:{{.Tag}}"

  rm -rf "$SMOKE_DIR"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "Failed tests:"
  for e in "${ERRORS[@]}"; do
    echo "  - $e"
  done
  exit 1
fi
