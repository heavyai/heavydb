# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check query results against reference Parquet files (--expected DIR).

The comparison itself is velox-testing's (result_comparison.py: tolerances, ORDER BY and
LIMIT tie handling, canonical sorting). This module adapts heavysql's text output to it:
the rows are typed after the reference columns, and decimal columns, which heavysql prints
with their scale, get the same half-unit-in-the-last-place tolerance velox-testing uses for
engines that report decimal(p,s) types.

Needs pandas, numpy, pyarrow and sqlglot (see requirements-verify.txt); import this module
only when verification was asked for.
"""

import decimal
import re
from pathlib import Path

import pandas as pd

from .result_comparison import XFAIL_IF_EMPTY, compare_result_frames

PASSED, FAILED, XFAIL, NOT_VALIDATED = "passed", "failed", "expected-failure", "not-validated"

_FIXED_POINT = re.compile(r"-?\d+\.(\d{1,6})$")


def expected_file(directory, query):
    """Reference file of a query: q01.parquet, q1.parquet or 01.parquet (or <query>.parquet)."""
    m = re.fullmatch(r"[a-zA-Z]*0*(\d+)", query)
    names = [query.lower()]
    if m:
        n = int(m.group(1))
        names += ["q%02d" % n, "q%d" % n, "%02d" % n]
    for name in names:
        path = directory / (name + ".parquet")
        if path.exists():
            return path
    return None


def _fixed_scale(cells):
    """Number of decimals if every non-NULL cell is a fixed-point number with the same count."""
    scales = set()
    for c in cells:
        if c is None:
            continue
        m = _FIXED_POINT.match(c)
        if not m:
            return None
        scales.add(len(m.group(1)))
    return scales.pop() if len(scales) == 1 else None


def frame_from_rows(rows, expected):
    """DataFrame of heavysql's text rows, columns typed after `expected`; also returns the
    per-column type names used for the decimal tolerance (None where not a decimal)."""
    ncols = expected.shape[1]
    for r in rows[:1]:
        if len(r) != ncols:
            raise AssertionError("Column count mismatch: %d (actual) vs %d (expected)" % (len(r), ncols))
    cols, types = [], []
    for i in range(ncols):
        cells = [r[i] for r in rows]
        e = expected.iloc[:, i]
        series = pd.Series(cells, dtype=object)
        scale = None
        if pd.api.types.is_float_dtype(e):
            scale = _fixed_scale(cells)
            series = pd.to_numeric(series)
        elif pd.api.types.is_numeric_dtype(e) and not pd.api.types.is_bool_dtype(e):
            series = pd.to_numeric(series)
        elif any(isinstance(v, decimal.Decimal) for v in e.head(1)):
            series = series.map(lambda v: None if v is None else decimal.Decimal(v))
        cols.append(series)
        types.append(None if scale is None else "decimal(38,%d)" % scale)
    actual = pd.concat(cols, axis=1) if cols else pd.DataFrame()
    actual.columns = expected.columns
    return actual, types


class Verifier:
    """Compares each query's result rows with <expected_dir>/<query>.parquet."""

    def __init__(self, expected_dir, xfail=None):
        """xfail: query name -> reason; a failed check of such a query is reported as an
        expected failure (a pass is still a pass)."""
        self.xfail = {k.lower(): v for k, v in (xfail or {}).items()}
        self.dir = Path(expected_dir)
        if not self.dir.is_dir():
            raise SystemExit("--expected: not a directory: %s" % self.dir)

    def check(self, query, sql, rows):
        """Return {"status": passed|failed|expected-failure|not-validated, "message": str|None}."""
        res = self._check(query, sql, rows)
        reason = self.xfail.get(query.lower())
        if reason and res["status"] == FAILED:
            res = {"status": XFAIL, "message": "known: %s | %s" % (reason, res["message"])}
        return res

    def _check(self, query, sql, rows):
        path = expected_file(self.dir, query)
        if path is None:
            return {"status": FAILED, "message": "no reference file for %s in %s" % (query, self.dir)}
        try:
            expected = pd.read_parquet(path)
            if expected.empty and all(t is object for t in expected.dtypes):
                return {"status": NOT_VALIDATED,
                        "message": "reference file is empty (no schema): %s" % path.name}
            actual, types = frame_from_rows(rows, expected)
            if query.lower() in XFAIL_IF_EMPTY and actual.empty:
                return {"status": XFAIL,
                        "message": "%s returned no rows: known float calculation mismatch" % query}
            compare_result_frames(actual, expected, sql, types)
            return {"status": PASSED, "message": None}
        except Exception as e:  # any comparison/parse problem is a failed check, not a crash
            return {"status": FAILED, "message": ("%s: %s" % (type(e).__name__, e))[:500]}
