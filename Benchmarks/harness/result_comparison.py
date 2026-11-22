# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared result comparison logic for TPC-H/TPC-DS query validation.

Used by both the integration test path (live query engine results vs DuckDB
reference) and the benchmark validation path (result parquet files vs expected
parquet files). See compare_result_frames for full comparison semantics.

Copied unchanged from velox-testing (common/testing/result_comparison.py) so that
HeavyDB results are judged by the same rules as the other engines. Keep it in sync
with that file rather than editing it here; HeavyDB-specific glue is in verify.py.
"""

import datetime
import re
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import sqlglot

REL_TOL = 1e-5
ABS_TOL = 1e-8
MAX_MISMATCHES = 5

# Known queries that return empty results due to float precision issues.
# These are marked as expected failures rather than test failures.
XFAIL_IF_EMPTY: set[str] = {"q15"}

ValidationStatus = Literal["passed", "failed", "expected-failure", "not-validated"]


# ---------------------------------------------------------------------------
# SQL parsing helpers
# ---------------------------------------------------------------------------


def get_orderby_sort_spec(
    query_sql: str,
    expected_col_names: list[str],
) -> tuple[list[int], list[bool], list[bool]]:
    """Resolve ORDER BY expressions to positional columns and sort options."""
    expr = sqlglot.parse_one(query_sql, read="presto")
    order = next((e for e in expr.find_all(sqlglot.exp.Order)), None)
    if not order:
        return [], [], []

    sort_col_indices: list[int] = []
    ascending: list[bool] = []
    nulls_first: list[bool] = []

    for ordered in order.expressions:
        key = ordered.this
        is_desc = bool(ordered.args.get("desc"))

        # Resolve key → column index via numeric literal or column reference.
        resolved_idx: int | None = None
        if isinstance(key, sqlglot.exp.Literal):
            try:
                col_num = int(key.this)
                if 1 <= col_num <= len(expected_col_names):
                    resolved_idx = col_num - 1
            except (ValueError, TypeError):
                pass

        if resolved_idx is None and isinstance(key, sqlglot.exp.Column):
            name = key.name
            if name in expected_col_names:
                resolved_idx = expected_col_names.index(name)

        if resolved_idx is None:
            # Complex expression — skip ORDER BY handling for this query.
            warnings.warn(
                f"ORDER BY expression {key.sql()!r} couldn't be mapped to a result column; "
                "ORDER BY validation and tie-boundary handling will be skipped for this query."
            )
            return [], [], []

        sort_col_indices.append(resolved_idx)
        ascending.append(not is_desc)
        nulls_first.append(bool(ordered.args.get("nulls_first")))

    return sort_col_indices, ascending, nulls_first


def get_orderby_col_indices(query_sql: str, expected_col_names: list[str]) -> tuple[list[int], list[bool]]:
    """
    Extract ORDER BY column positions and directions from SQL using sqlglot.

    Returns (indices, ascending) where ascending[i] is True iff indices[i]
    is sorted ASC. Returns ([], []) when there is no ORDER BY, or when any
    ORDER BY expression is too complex to map to a result column (CASE,
    aggregate, etc.).
    """
    sort_col_indices, ascending, _ = get_orderby_sort_spec(query_sql, expected_col_names)

    return sort_col_indices, ascending


def get_limit(query_sql: str) -> int | None:
    """Return the LIMIT value from SQL, or None if there is no LIMIT (or the
    LIMIT expression isn't a simple integer literal)."""
    expr = sqlglot.parse_one(query_sql)
    limit_node = next((e for e in expr.find_all(sqlglot.exp.Limit)), None)
    if limit_node is None:
        return None
    try:
        return int(limit_node.args["expression"].this)
    except (KeyError, ValueError, TypeError, AttributeError):
        warnings.warn(
            f"LIMIT clause {limit_node.sql()!r} couldn't be parsed as an integer literal; "
            "treating as no LIMIT (boundary tie handling will be skipped)."
        )
        return None


# ---------------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------------


def _first_non_null(col: pd.Series):
    """Return the first value in col that isn't None/NaN/NaT/NA, or None if all null."""
    for v in col:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        return v
    return None


def _normalize_to_expected(actual: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce each column of actual to match the corresponding expected column.

    Runs two alignment passes per column:
      1. Dtype alignment — astype actual to expected's dtype if they differ.
         (e.g., prestodb returns DATE as str while DuckDB returns datetime64;
          INTEGER comes back as int32 from DuckDB but int64 from Presto.)
      2. Value-type alignment — even when dtypes match (or after the astype
         in step 1), the Python values inside may differ in type (e.g.,
         parquet round-trips where one engine produces str dates and the
         other datetime.date; pandas 3.0 also wraps str in pyarrow-backed
         string dtypes that don't equal object). Sample the first non-null
         on each side and coerce actual's values to match expected's.

    Comparison is positional — column labels do not need to match.
    """
    out = actual.copy()
    for i in range(actual.shape[1]):
        a_col = actual.iloc[:, i]
        e_col = expected.iloc[:, i]
        target_dtype = e_col.dtype

        # 1. Dtype alignment.
        if a_col.dtype != target_dtype:
            a_col = a_col.astype(target_dtype)
            out.isetitem(i, a_col)

        # 2. Value-type alignment (independent of dtype — pandas can hold
        #    Python str in pyarrow-backed string dtypes too, so an object
        #    dtype check alone isn't sufficient).
        a_sample = _first_non_null(a_col)
        e_sample = _first_non_null(e_col)
        if a_sample is None or e_sample is None:
            continue
        if isinstance(a_sample, str) and isinstance(e_sample, datetime.date):
            # actual has str dates, expected has datetime.date objects —
            # parse the strings and yield datetime.date to match.
            out.isetitem(i, pd.to_datetime(a_col).dt.date)
    return out


# ---------------------------------------------------------------------------
# Frame comparison helpers
# ---------------------------------------------------------------------------


def _is_float_col(series: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(series)


def _decimal_abs_tolerance(type_name: str | None) -> float | None:
    if not type_name:
        return None
    match = re.fullmatch(r"decimal\(\d+,(\d+)\)", type_name.lower().replace(" ", ""))
    if not match:
        return None
    return 0.5 * 10 ** -int(match.group(1))


def _orderby_differs_from_prev(orderby_df: pd.DataFrame) -> np.ndarray:
    """
    Return a boolean array of length len(orderby_df) - 1 where element i is
    True iff row i+1 differs from row i on any ORDER BY column. Float columns
    use tolerance equality (rel_tol=1e-5, abs_tol=1e-8); non-float columns
    use exact equality.

    Tolerance is appropriate here because the caller (_find_last_tie_start)
    is identifying rows the engine *could have* considered interchangeable
    at a LIMIT cutoff — a cross-engine notion, not strict bit-identity.
    """
    n = len(orderby_df)
    if n <= 1:
        return np.array([], dtype=bool)

    differs = np.zeros(n - 1, dtype=bool)
    for col_idx in range(orderby_df.shape[1]):
        col = orderby_df.iloc[:, col_idx]
        if _is_float_col(col):
            arr = col.to_numpy(dtype=np.float64, na_value=np.nan)
            prev, curr = arr[:-1], arr[1:]
            with np.errstate(invalid="ignore"):
                tol = ABS_TOL + REL_TOL * np.maximum(np.abs(prev), np.abs(curr))
                tied = np.abs(curr - prev) <= tol
            differs |= ~tied
        else:
            arr = col.to_numpy()
            differs |= arr[1:] != arr[:-1]
    return differs


def _validate_orderby(df: pd.DataFrame, sort_col_indices: list[int], ascending: list[bool]) -> None:
    """
    Validate that each ORDER BY column is monotonic in its specified direction
    within the tie groups of all earlier ORDER BY columns.

    Walks one column at a time: for column k, a violation is any adjacent
    pair (i, i+1) that is anti-monotonic AND where all of columns 0..k-1
    are equal at i and i+1 (same outer tie group). Strict equality is used
    throughout — every value comes from one engine and is bit-identical to
    itself.

    Null handling: SQL leaves the position of NULLs in ORDER BY engine-
    defined unless NULLS FIRST/LAST is explicit, so adjacent pairs where
    either side is NULL are not counted as violations. Two NULLs are treated
    as tied (do not break the outer tie group); a NULL adjacent to a non-NULL
    value is treated as a change for tie-group tracking only.

    Raises AssertionError pointing at the first offending row. A no-op when
    sort_col_indices is empty or the frame has 0-1 rows.
    """
    if not sort_col_indices or len(df) <= 1:
        return

    # Boolean of length n-1; True where any prior column changes between
    # row i and row i+1 (i.e., row i+1 starts a new outer tie group).
    prior_changed = np.zeros(len(df) - 1, dtype=bool)

    for col_idx, asc in zip(sort_col_indices, ascending):
        col_series = df.iloc[:, col_idx]
        # Float columns route nulls through NaN so numpy's well-defined
        # float comparisons (NaN < NaN -> False) keep them out of the
        # anti-monotonic mask. Object/string/date columns can't be compared
        # element-wise against NaN/None without raising, so we mask null-
        # involving pairs out below.
        if _is_float_col(col_series):
            col = col_series.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            col = col_series.to_numpy()
        prev, curr = col[:-1], col[1:]

        is_null = pd.isna(col_series).to_numpy()
        prev_null, curr_null = is_null[:-1], is_null[1:]
        either_null = prev_null | curr_null

        if either_null.any() and not _is_float_col(col_series):
            # Object-dtype path: comparing None/NaN with anything raises
            # TypeError. Restrict the inequality check to fully non-null
            # pairs, and treat null<->non-null transitions as a value
            # change (so we still break out of tie groups across them)
            # while null<->null stays tied.
            safe = ~either_null
            anti_monotonic = np.zeros(len(prev), dtype=bool)
            value_changed = np.zeros(len(prev), dtype=bool)
            if safe.any():
                ps, cs = prev[safe], curr[safe]
                anti_monotonic[safe] = (cs < ps) if asc else (cs > ps)
                value_changed[safe] = cs != ps
            value_changed |= prev_null ^ curr_null
        else:
            anti_monotonic = (curr < prev) if asc else (curr > prev)
            value_changed = curr != prev

        # A real violation requires both anti-monotonicity AND prior columns
        # to be tied; otherwise we're across an outer-tie boundary, where
        # this column's direction doesn't apply.
        bad = np.flatnonzero(anti_monotonic & ~prior_changed)
        if len(bad) > 0:
            i = int(bad[0])
            raise AssertionError(
                f"Engine violated ORDER BY on column index {col_idx} "
                f"({'ASC' if asc else 'DESC'}) at row {i + 1}: "
                f"got {curr[i]!r} after {prev[i]!r}"
            )
        prior_changed = prior_changed | value_changed


def restore_orderby(
    df: pd.DataFrame,
    sort_col_indices: list[int],
    ascending: list[bool],
    nulls_first: list[bool],
) -> pd.DataFrame:
    """Reconstruct SQL result order after unordered distributed storage."""
    if not sort_col_indices or len(df) <= 1:
        return df.reset_index(drop=True)

    # Apply stable sorts from the least-significant key to the most-significant
    # key. Access columns positionally because SQL results may contain duplicate
    # column labels.
    row_order = np.arange(len(df))
    for col_idx, asc, null_first in reversed(list(zip(sort_col_indices, ascending, nulls_first))):
        values = df.iloc[row_order, col_idx].reset_index(drop=True)
        permutation = values.sort_values(
            ascending=asc,
            na_position="first" if null_first else "last",
            kind="stable",
        ).index.to_numpy()
        row_order = row_order[permutation]

    return df.iloc[row_order].reset_index(drop=True)


def _canonical_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort df by [all non-float columns, all float columns] for a deterministic,
    engine-independent ordering. ORDER BY columns are not treated specially —
    they're just regular columns participating in the same sort.
    """
    df = df.reset_index(drop=True)
    non_float_pos = [i for i in range(df.shape[1]) if not _is_float_col(df.iloc[:, i])]
    float_pos = [i for i in range(df.shape[1]) if _is_float_col(df.iloc[:, i])]
    labels = df.columns[non_float_pos + float_pos].tolist()
    if not labels:
        return df
    return df.sort_values(by=labels, na_position="last").reset_index(drop=True)


def _column_mismatches(a_col: pd.Series, b_col: pd.Series, abs_tol: float = ABS_TOL) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (bad_indices, null_mismatch_mask) for a single-column comparison.

    bad_indices: positions where the two columns disagree (after null and
        tolerance rules). null_mismatch_mask: boolean array, True where exactly
        one side is null.
    """
    a_null = pd.isna(a_col).to_numpy()
    b_null = pd.isna(b_col).to_numpy()
    null_mismatch = a_null ^ b_null  # exactly one side is null

    if _is_float_col(a_col):
        a = a_col.to_numpy(dtype=np.float64, na_value=np.nan)
        b = b_col.to_numpy(dtype=np.float64, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            value_mismatch = np.abs(a - b) > abs_tol + REL_TOL * np.maximum(np.abs(a), np.abs(b))
        # NaN arithmetic returns NaN, and NaN > tol is False, so any cell
        # involving a null is automatically excluded from value_mismatch.
    else:
        a = a_col.to_numpy()
        b = b_col.to_numpy()
        # In object arrays NaN != NaN is True; explicitly mask out null cells.
        value_mismatch = (a != b) & ~(a_null | b_null)

    return np.flatnonzero(null_mismatch | value_mismatch), null_mismatch


def _format_mismatch(
    row_idx: int, col_idx: int, a_col: pd.Series, b_col: pd.Series, is_null_mismatch: bool, abs_tol: float = ABS_TOL
) -> str:
    v1, v2 = a_col.iloc[row_idx], b_col.iloc[row_idx]
    if is_null_mismatch:
        return f"Row {row_idx}, col {col_idx}: {v1!r} vs {v2!r} (null mismatch)"
    if _is_float_col(a_col):
        fv1, fv2 = float(v1), float(v2)
        diff = abs(fv1 - fv2)
        tol = abs_tol + REL_TOL * max(abs(fv1), abs(fv2))
        return f"Row {row_idx}, col {col_idx}: {fv1} vs {fv2} (diff={diff:.2e}, tol={tol:.2e})"
    return f"Row {row_idx}, col {col_idx}: {v1!r} vs {v2!r}"


def _assert_frames_equal(df1: pd.DataFrame, df2: pd.DataFrame, abs_tolerances: list[float] | None = None) -> None:
    """
    Column-by-column vectorized comparison. Float columns use rel_tol=1e-5,
    abs_tol=1e-8; non-float columns use exact equality. Both DataFrames must
    have matching shape and be pre-sorted / reset-indexed. Comparison is
    positional — column labels are not used.

    Null handling: both sides null → equal; exactly one side null → mismatch.
    Covers all three flavours of null TPC-DS can produce (float NaN, None,
    pd.NA) via pandas' uniform `pd.isna`.

    Raises AssertionError describing up to MAX_MISMATCHES differences.
    """
    if len(df1) != len(df2):
        raise AssertionError(f"Row count mismatch: {len(df1)} vs {len(df2)}")

    mismatches: list[str] = []
    for col_idx in range(df1.shape[1]):
        a_col = df1.iloc[:, col_idx]
        b_col = df2.iloc[:, col_idx]
        abs_tol = abs_tolerances[col_idx] if abs_tolerances else ABS_TOL
        bad_idx, null_mismatch = _column_mismatches(a_col, b_col, abs_tol)
        for idx in bad_idx[: MAX_MISMATCHES - len(mismatches)]:
            i = int(idx)
            mismatches.append(_format_mismatch(i, col_idx, a_col, b_col, bool(null_mismatch[i]), abs_tol))
        if len(mismatches) >= MAX_MISMATCHES:
            break

    if mismatches:
        truncated = f" (showing first {MAX_MISMATCHES})" if len(mismatches) >= MAX_MISMATCHES else ""
        raise AssertionError(f"Found {len(mismatches)} mismatch(es){truncated}:\n  " + "\n  ".join(mismatches))


def _find_last_tie_start(orderby_df: pd.DataFrame) -> int:
    """
    Return the row index where the contiguous tie block at the end of
    orderby_df begins. Rows from this index onward are all tolerance-tied
    (greedy chain via _orderby_differs_from_prev) with their preceding
    neighbor. Returns 0 for empty frames or frames where every row is
    tolerance-tied with its predecessor.
    """
    n = len(orderby_df)
    if n <= 1:
        return 0
    differs = _orderby_differs_from_prev(orderby_df)
    # differs[i] indicates row i+1 differs from row i. The last tie block
    # starts at one past the last "differs" position. If nothing differs, the
    # whole frame is one tie block starting at row 0.
    diff_positions = np.flatnonzero(differs)
    if len(diff_positions) == 0:
        return 0
    return int(diff_positions[-1]) + 1


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------


def compare_result_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    query_sql: str,
    actual_column_types: list[str] | None = None,
) -> None:
    """
    Full comparison pipeline. Raises AssertionError on any mismatch.

    All cross-frame access is positional, so column labels never need to
    align between actual and expected.

    Steps:
      1. Validate column count and capture expected column names.
      2. Normalize dtypes / value types.
      3. Validate row count.
      4. Parse ORDER BY / LIMIT from SQL.
      5. Validate each engine respected ORDER BY (per-frame, strict equality).
      6. With LIMIT: peel the tied tail in engine order, then canonical-sort
         each piece. Compare front fully; compare tail's ORDER BY values only.
      7. Without LIMIT: canonical-sort both frames and compare in full.
    """
    # 1. Column count check
    if len(actual.columns) != len(expected.columns):
        raise AssertionError(
            f"Column count mismatch: {len(actual.columns)} (actual) vs {len(expected.columns)} (expected)\n"
            f"  actual:   {list(actual.columns)}\n"
            f"  expected: {list(expected.columns)}"
        )
    expected_col_names = list(expected.columns)
    abs_tolerances = [_decimal_abs_tolerance(type_name) or ABS_TOL for type_name in actual_column_types or []]

    # 2. Coerce actual's dtypes/value types to match expected's
    actual = _normalize_to_expected(actual, expected)

    # 3. Row count
    if len(actual) != len(expected):
        raise AssertionError(f"Row count mismatch: {len(actual)} (actual) vs {len(expected)} (expected)")

    # 4. Parse ORDER BY / LIMIT.
    sort_col_indices, ascending, _ = get_orderby_sort_spec(query_sql, expected_col_names)
    limit = get_limit(query_sql)

    # 5. Per-frame ORDER BY validation — engine order is intact here.
    _validate_orderby(actual, sort_col_indices, ascending)
    _validate_orderby(expected, sort_col_indices, ascending)

    # 6. LIMIT tie boundary: the engine may have selected different rows from
    # a tolerance-tied bucket at the cutoff. Peel that tail (in engine order)
    # before canonical sort, then compare the front fully and the tail's
    # ORDER BY values only.
    if sort_col_indices and limit is not None:
        last_start_a = _find_last_tie_start(actual.iloc[:, sort_col_indices])
        last_start_e = _find_last_tie_start(expected.iloc[:, sort_col_indices])
        if last_start_a != last_start_e:
            raise AssertionError(
                f"Engines disagree on LIMIT tie-boundary position: "
                f"actual tied tail starts at {last_start_a}, expected at {last_start_e}"
            )
        last_start = last_start_a

        _assert_frames_equal(
            _canonical_sort(actual.iloc[:last_start]),
            _canonical_sort(expected.iloc[:last_start]),
            abs_tolerances,
        )
        tail_abs_tolerances = [abs_tolerances[i] for i in sort_col_indices] if abs_tolerances else None
        _assert_frames_equal(
            _canonical_sort(actual.iloc[last_start:]).iloc[:, sort_col_indices],
            _canonical_sort(expected.iloc[last_start:]).iloc[:, sort_col_indices],
            tail_abs_tolerances,
        )
        return

    # 7. No LIMIT: canonical-sort both frames and compare in full.
    _assert_frames_equal(_canonical_sort(actual), _canonical_sort(expected), abs_tolerances)


# ---------------------------------------------------------------------------
# Status-returning wrapper (used by benchmark validation path)
# ---------------------------------------------------------------------------


def validate_query_result(
    query_id: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    query_sql: str,
) -> tuple[ValidationStatus, str | None]:
    """
    Compare actual vs expected for one query. Returns (status, message).

    Status values:
      "passed"           — comparison succeeded
      "failed"           — comparison raised an exception
      "expected-failure" — query is in XFAIL_IF_EMPTY and returned no rows
    """
    if query_id.lower() in XFAIL_IF_EMPTY and actual.empty:
        return (
            "expected-failure",
            f"{query_id.upper()} returned no rows: known float calculation mismatch "
            "in MAX(total_revenue) subquery causes empty result with float data",
        )

    try:
        compare_result_frames(actual, expected, query_sql)
        return "passed", None
    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"[:500]
