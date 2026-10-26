# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL client: runs statements through heavysql inside the runtime."""

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass

_TIMING = re.compile(r"Execution time: (\d+) ms,\s*Total time: (\d+) ms")
_ROWS = re.compile(r"^(\d+) rows? returned", re.M)
# heavysql prints errors to stderr ("SQL Error: ...", "Thrift error: ..."), and its exit code
# only reflects the last statement, so look for them on stderr as well as at the exit code
# (stdout is not searched: a result row may legitimately start with "Error").
_ERROR = re.compile(r"^(SQL Error|Thrift error|Exception|Error)\b", re.M)
# Field separator used when the rows are kept for verification: a control character cannot
# occur in TPC-H text, unlike the default '|'.
ROW_DELIM = "\x1f"


def split_statements(sql):
    """Split a SQL script on top-level ';' (ignoring ';' in quotes and '--' comments)."""
    out, cur, quote, i = [], [], None, 0
    while i < len(sql):
        c = sql[i]
        if quote:
            cur.append(c)
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            cur.append(c)
        elif c == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = len(sql) if j < 0 else j  # drop the comment, keep the newline
            continue
        elif c == ";":
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt + ";")
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail + ";")
    return out


class SqlError(RuntimeError):
    pass


@dataclass
class Result:
    exec_ms: float = 0.0
    total_ms: float = 0.0
    wall_ms: float = 0.0
    rows: int = 0
    digest: str = ""
    output: str = ""
    table: list = None  # rows as lists of str (None for NULL); only set with execute(rows=True)


class HeavySqlClient:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, sql, timeout=None, rows=False):
        """Run SQL through heavysql and return the Result of the last statement.

        With rows=True the last statement's result rows are also returned in Result.table.
        heavysql prints NULL and an empty string alike, so both come back as None.

        A ';'-separated script is run one statement per heavysql call (heavysql's exit code
        only reflects its last statement), so the first failing statement raises SqlError.
        Timings are the server-side values heavysql prints with -t.
        """
        stmts = split_statements(sql)
        if not stmts:
            raise SqlError("empty SQL statement")
        for stmt in stmts[:-1]:
            self._execute_one(stmt, timeout)
        return self._execute_one(stmts[-1], timeout, rows)

    def _execute_one(self, sql, timeout, rows=False):
        cmd = self.runtime.heavysql_prefix() + self.runtime.heavysql_args()
        if rows:
            cmd += ["--delimiter", ROW_DELIM]
        if self.runtime.dry_run:
            print("DRY-RUN: heavysql <<< %s" % sql.replace("\n", " ")[:120])
            return Result()
        t0 = time.perf_counter()
        r = subprocess.run(cmd, input=sql + "\n", capture_output=True, text=True,
                           timeout=timeout)
        wall = (time.perf_counter() - t0) * 1000
        out = r.stdout
        if r.returncode != 0 or _ERROR.search(r.stderr):
            raise SqlError("%s\n%s%s" % (sql[:300], out.strip(), r.stderr.strip()))
        res = Result(wall_ms=wall, output=out)
        m = _TIMING.findall(out)
        if m:
            res.exec_ms, res.total_ms = float(m[-1][0]), float(m[-1][1])
        # Result rows are everything except the timing and "N rows returned." lines
        body = [l for l in out.splitlines() if l and not _TIMING.search(l) and not _ROWS.match(l)]
        m = _ROWS.search(out)
        res.rows = int(m.group(1)) if m else len(body)
        res.digest = hashlib.sha256("\n".join(sorted(body)).encode()).hexdigest()[:16]
        if rows:
            res.table = [[None if c == "NULL" else c for c in l.split(ROW_DELIM)] for l in body]
        return res
