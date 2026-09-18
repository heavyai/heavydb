#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Sign a product tarball using Sigstore cosign keyless signing (GitHub OIDC)
# and bundle it together with a verification guide into a single zip archive.
#
# Keyless signing uses the GitHub Actions OIDC token to obtain a short-lived
# certificate from Sigstore's Fulcio CA. No secrets or pre-generated keys are
# required. The signature and certificate are recorded in Sigstore's Rekor
# public transparency log, which allows consumers to verify both the
# cryptographic integrity of the tarball and that it was produced by this
# specific GitHub Actions workflow.
#
# The zip contains three files at its root:
#   <tarname>.tar.gz         — the product binary tarball (copy)
#   <tarname>.tar.gz.bundle  — cosign bundle (signature + certificate combined)
#   VERIFY.md                — step-by-step verification instructions
#
# Invoked from .github/workflows/rc-builder.yml; also locally runnable for
# testing, though keyless signing requires a valid OIDC token (CI only).
#
# Required env vars:
#   BINARY  - absolute path to the product tarball (.tar.gz)
#
# Outputs:
#   Writes "signed_bundle=<path>" to $GITHUB_OUTPUT when running in CI.
#   Prints the bundle path to stdout in all cases.
#
set -euo pipefail

: "${BINARY:?BINARY must be set to the absolute path of the product tarball}"

if [ ! -f "$BINARY" ]; then
  echo "::error::Tarball not found: $BINARY"
  exit 1
fi

TARNAME="$(basename "$BINARY")"
BUNDLEDIR="$(dirname "$BINARY")/signed-bundle"
mkdir -p "$BUNDLEDIR"

COSIGN_BUNDLE="${BUNDLEDIR}/${TARNAME}.bundle"

# ── Sign the tarball ──────────────────────────────────────────────────────────
# --yes suppresses the interactive Rekor recording prompt; recording is
# automatic and intentional — it provides an immutable audit trail.
# --bundle writes a single JSON file combining the signature and certificate,
# which is the format expected by `cosign verify-blob --bundle`.
cosign sign-blob \
  --yes \
  --bundle="$COSIGN_BUNDLE" \
  "$BINARY"

echo "Bundle: $(basename "$COSIGN_BUNDLE")"

# ── Copy the tarball into the bundle staging area ─────────────────────────────
cp "$BINARY" "${BUNDLEDIR}/${TARNAME}"

# ── Generate VERIFY.md ────────────────────────────────────────────────────────
# Use Python so the file body can contain literal backticks and triple-fences
# without shell-escaping gymnastics.
python3 - "$TARNAME" > "${BUNDLEDIR}/VERIFY.md" << 'PYEOF'
import sys
tarname = sys.argv[1]
print(f"""\
# Verifying the HeavyAI Product Tarball

This bundle was produced by the HeavyAI Product Builder CI workflow and
contains three files:

| File | Description |
|------|-------------|
| `{tarname}` | Product binary tarball |
| `{tarname}.bundle` | Cosign bundle (signature + certificate) |
| `VERIFY.md` | This file |

Run all commands below from the directory where you unzipped this bundle.

## Prerequisites

Install cosign if not already present. Download the appropriate binary for
your platform from:

  https://github.com/sigstore/cosign/releases

Or, on macOS with Homebrew:

```
brew install cosign
```

## Verification Steps

### 1. Verify the signature

```
cosign verify-blob \\
  --bundle={tarname}.bundle \\
  --certificate-identity-regexp="https://github.com/heavyai/heavydb/.github/workflows/rc-builder.yml@refs/heads/.*" \\
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \\
  {tarname}
```

A successful result will print:

```
Verified OK
```

## What the bundle proves

The `.bundle` file contains the cosign signature and the ephemeral certificate
issued by Sigstore's Fulcio CA at signing time. The certificate encodes the
identity of the GitHub Actions workflow that produced this artifact — including
the repository, workflow file path, and git ref. The signature is also recorded
in Sigstore's public Rekor transparency log, providing an immutable audit trail.

If verification succeeds, the tarball is authentic and has not been modified
since it was signed by CI.
""")
PYEOF

# ── Zip the three files into a single archive ─────────────────────────────────
ZIPNAME="${TARNAME%.tar.gz}-signed.zip"
(cd "$BUNDLEDIR" && zip "$ZIPNAME" "$TARNAME" "${TARNAME}.bundle" VERIFY.md)

ZIPPATH="${BUNDLEDIR}/${ZIPNAME}"
echo "Signed bundle: $ZIPPATH"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "signed_bundle=${ZIPPATH}" >> "$GITHUB_OUTPUT"
fi
