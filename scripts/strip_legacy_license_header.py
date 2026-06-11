#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strip legacy HEAVY.AI / OmniSci / MapD Apache-2.0 license headers.

This is a companion fixer for the ``rapidsai/pre-commit-hooks`` ``verify-copyright``
hook. ``verify-copyright`` only recognizes *NVIDIA* copyright lines, so on a file that
still carries the old HEAVY.AI block it would *insert* an NVIDIA SPDX header while
leaving the legacy block in place (two stacked headers). Running this fixer *before*
``verify-copyright`` removes the recognized legacy block so the result is a single,
clean NVIDIA SPDX header.

It is intentionally conservative: it only removes a *leading* comment block that
contains both a legacy copyright line and the Apache-2.0 boilerplate, so it never
touches unrelated leading comments or genuine third-party headers (e.g. Simba,
Apache Calcite). Run it standalone or via pre-commit:

    python3 scripts/strip_legacy_license_header.py path/to/File.cpp ...

Exits non-zero if any file was modified (pre-commit fixer convention).
"""

import re
import sys

# A legacy header is identified by a copyright line naming a legacy entity.
# The Apache boilerplate is NOT required — short-form copyright-only blocks
# (common in EE files) are also stripped.
COPYRIGHT_RE = re.compile(
    r"Copyright\s+\d{4}\s+(?:HEAVY\.AI|Heavy\.AI|OmniSci|MapD)", re.IGNORECASE
)

# Leading C-style /* ... */ block (optionally preceded by blank lines).
C_BLOCK_RE = re.compile(r"\A(?:[ \t]*\r?\n)*[ \t]*/\*.*?\*/[ \t]*\r?\n?", re.DOTALL)
# Leading run of #-comment lines.
HASH_BLOCK_RE = re.compile(r"\A(?:[ \t]*#[^\n]*\r?\n)+")
SHEBANG_RE = re.compile(r"\A#![^\n]*\r?\n")
# #pragma once: preserve as a prefix so verify-copyright places the SPDX header before it.
PRAGMA_ONCE_RE = re.compile(r"\A#pragma once[ \t]*\r?\n")
# Blank lines to skip after shebang / #pragma once before the copyright block.
LEADING_BLANK_RE = re.compile(r"\A(?:[ \t]*\r?\n)+")


def strip_header(text: str) -> str:
    """Return ``text`` with a leading legacy license block removed, if present."""
    prefix = ""
    body = text

    # Preserve a shebang line so we only inspect the comment block after it.
    shebang = SHEBANG_RE.match(body)
    if shebang:
        prefix = body[: shebang.end()]
        body = body[shebang.end() :]

    # Preserve #pragma once (must remain first after any shebang).
    pragma = PRAGMA_ONCE_RE.match(body)
    if pragma:
        prefix += body[: pragma.end()]
        body = body[pragma.end() :]

    # Skip blank lines between prefix and the copyright comment block.
    blank = LEADING_BLANK_RE.match(body)
    body_after_blank = body[blank.end() :] if blank else body

    for block_re in (C_BLOCK_RE, HASH_BLOCK_RE):
        match = block_re.match(body_after_blank)
        if not match:
            continue
        block = match.group(0)
        if COPYRIGHT_RE.search(block):
            rest = body_after_blank[match.end() :]
            # Drop blank lines left behind between the old header and the code.
            rest = re.sub(r"\A(?:[ \t]*\r?\n)+", "", rest)
            return prefix + rest
        # Only the first leading block is a candidate; stop after checking it.
        break

    return text


def main(argv: list[str]) -> int:
    changed = False
    for path in argv:
        try:
            # utf-8-sig transparently strips a leading UTF-8 BOM if present.
            with open(path, encoding="utf-8-sig") as f:
                original = f.read()
        except (OSError, UnicodeDecodeError):
            # Binary or unreadable file; nothing to strip.
            continue

        updated = strip_header(original)
        if updated != original:
            # Always write back as plain UTF-8 (no BOM).
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"stripped legacy license header: {path}")
            changed = True

    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
