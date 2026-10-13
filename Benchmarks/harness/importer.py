# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import a dataset (directory-per-table Parquet) into HeavyDB.

A table's source is a directory of one or more parquet files (file names are
not assumed) or a single file.  Modes:
  native   CREATE TABLE + COPY FROM (data is imported into HeavyDB storage)
  foreign  CREATE FOREIGN TABLE over the parquet path (no import step)
"""

import re
import time
from pathlib import Path

from .parquet_schema import read_column_types


def path_filter_regex(glob):
    """Glob (e.g. '*.parquet') -> regex for HeavyDB's regex_path_filter option, so stray
    non-data files in a table directory (logs, .jsonl, _SUCCESS) are ignored."""
    return "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in glob) + "$"


def _natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def table_files(root, table):
    p = Path(root) / table.path
    if p.is_file():
        return [p]
    files = sorted(p.glob(table.glob), key=_natural)
    if not files:
        raise SystemExit("no files matching %s in %s" % (table.glob, p))
    return files


_CONV = re.compile(r'Conversion from Parquet type "(.+?)" to HeavyDB type "(.+?)" is not allowed'
                   r".*?Parquet column: (\w+)", re.S)
_PARQUET_TO_HEAVY = {"INT32": "INTEGER", "INT64": "BIGINT", "INT16": "SMALLINT", "INT8": "TINYINT",
                     "DOUBLE": "DOUBLE", "FLOAT": "FLOAT", "DATE": "DATE", "BOOLEAN": "BOOLEAN"}
_DECIMAL = re.compile(r"Decimal\(precision=(\d+), scale=(\d+)\)", re.I)


def heavy_type_for(parquet_type):
    m = _DECIMAL.search(parquet_type)
    if m:
        return "DECIMAL(%s,%s)" % m.groups()
    return _PARQUET_TO_HEAVY.get(parquet_type.strip().upper())


def adjust_ddl(ddl, error):
    """Return (new_ddl, description) changing the offending column to the Parquet type,
    or None if the error is not a type-conversion error we can fix."""
    m = _CONV.search(str(error))
    if not m:
        return None
    parquet_type, old, col = m.groups()
    new = heavy_type_for(parquet_type)
    if not new:
        return None
    pat = re.compile(r"(\b%s\s+)(DECIMAL\s*\(\s*\d+\s*,\s*\d+\s*\)|\w+)" % re.escape(col))
    if not pat.search(ddl):
        return None
    return pat.sub(lambda mo: mo.group(1) + new, ddl, count=1), "%s %s -> %s" % (col, old, new)


_COL_DEF = r"(\b%s\s+)(DECIMAL\s*\(\s*\d+\s*,\s*\d+\s*\)|\w+)"


def adapt_ddl_to_files(ddl, first_file):
    """Set numeric/date column types in ddl to the types stored in the Parquet file.

    Returns (new_ddl, ["col OLD -> NEW", ...]).  Text columns keep the benchmark's declared
    encoding; columns missing from the file or in unreadable files are left alone.
    """
    types = read_column_types(first_file)
    changes = []
    if not types:
        return ddl, changes
    for col, new in types.items():
        if not new:
            continue
        m = re.search(_COL_DEF % re.escape(col), ddl)
        if not m:
            continue
        old = re.sub(r"\s+", "", m.group(2)).upper()
        if old != new.upper():
            ddl = ddl[:m.start(2)] + new + ddl[m.end(2):]
            changes.append("%s %s -> %s" % (col, old, new))
    return ddl, changes


def import_dataset(bench, client, runtime, data_root, mode="native", log=print, adapt_types=True):
    """Create and load all tables; return {table: {seconds, files, bytes, adjusted}}.

    HeavyDB does not convert Parquet column types on import.  When adapt_types is set,
    numeric/date column types in the benchmark schema are first set from the Parquet
    footer of the table's first file, and, as a fallback, a conversion error makes the importer change that column of the benchmark schema to
    the dataset's Parquet type and retry, so one schema.sql serves datasets that store
    keys as int32/int64 or money as double/decimal(p,s).  Adjustments are recorded in the report.
    """
    schemas = bench.schema_statements()
    stats = {}
    for t in bench.tables:
        if t.name not in schemas:
            raise SystemExit("table %s has no CREATE TABLE in %s" % (t.name, bench.schema_file))
        files = table_files(data_root, t)
        src = Path(data_root) / t.path
        ddl_native, adjusted = schemas[t.name], []
        if adapt_types:  # take column types from the data itself
            ddl_native, adjusted = adapt_ddl_to_files(ddl_native, files[0])
        t0 = time.perf_counter()
        for _ in range(64):
            client.execute("DROP TABLE IF EXISTS %s;" % t.name)
            try:
                _load(client, runtime, t, src, files, ddl_native, mode)
                break
            except Exception as e:
                fix = adjust_ddl(ddl_native, e) if adapt_types else None
                if not fix:
                    if "Conversion from Parquet type" in str(e):
                        raise SystemExit("%s\n\nThe dataset's Parquet column types differ from the "
                                         "benchmark schema and could not be adapted; pass a matching "
                                         "--schema file." % e)
                    raise
                ddl_native, change = fix
                adjusted.append(change)
        else:
            raise SystemExit("too many schema adjustments for table %s" % t.name)
        stats[t.name] = {"seconds": time.perf_counter() - t0, "files": len(files),
                         "bytes": sum(f.stat().st_size for f in files), "adjusted": adjusted}
        log("  %-10s %2d file(s) %8.1f MB  %7.2fs%s" % (
            t.name, len(files), stats[t.name]["bytes"] / 1e6, stats[t.name]["seconds"],
            ("  [column types adapted: %s]" % "; ".join(adjusted)) if adjusted else ""))
    return stats


def _load(client, runtime, t, src, files, ddl_native, mode):
    if mode == "foreign":
        ddl = re.sub(r"^CREATE\s+TABLE", "CREATE FOREIGN TABLE", ddl_native, flags=re.I)
        ddl = ddl.rstrip().rstrip(";")
        opts = "file_path='%s'" % runtime.server_path(src)
        if src.is_dir():
            opts += ", regex_path_filter='%s'" % path_filter_regex(t.glob)
        ddl += " SERVER default_local_parquet WITH (%s);" % opts
        client.execute(ddl)
    else:
        client.execute(ddl_native)
        _copy(client, runtime, t, src, files)


def _copy(client, runtime, table, src, files):
    opts = "WITH (source_type='parquet_file'%s)" % (
        ", regex_path_filter='%s'" % path_filter_regex(table.glob) if src.is_dir() else "")
    if src.is_file():
        client.execute("COPY %s FROM '%s' %s;" % (table.name, runtime.server_path(src), opts))
        return
    try:  # whole directory in one COPY
        client.execute("COPY %s FROM '%s' %s;" % (table.name, runtime.server_path(src), opts))
    except Exception as e:  # fall back to appending file by file
        if "Conversion from Parquet type" in str(e):
            raise  # a schema problem, not a directory problem: let the caller adapt the schema
        print("    directory COPY failed (%s); importing file by file" % str(e).splitlines()[-1][:100])
        client.execute("TRUNCATE TABLE %s;" % table.name)
        for f in files:
            client.execute("COPY %s FROM '%s' WITH (source_type='parquet_file');"
                           % (table.name, runtime.server_path(f)))
