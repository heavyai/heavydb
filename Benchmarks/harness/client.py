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


class HeavySqlClient:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, sql, timeout=None):
        """Run one SQL statement (or a ';'-separated script); return a Result.

        Raises SqlError if heavysql reports an error.  Timings are the
        server-side values heavysql prints with -t (last statement wins).
        """
        sql = sql.strip()
        if not sql.endswith(";"):
            sql += ";"
        cmd = self.runtime.heavysql_prefix() + self.runtime.heavysql_args()
        if self.runtime.dry_run:
            print("DRY-RUN: heavysql <<< %s" % sql.replace("\n", " ")[:120])
            return Result()
        t0 = time.perf_counter()
        r = subprocess.run(cmd, input=sql + "\n", capture_output=True, text=True,
                           timeout=timeout)
        wall = (time.perf_counter() - t0) * 1000
        out = r.stdout
        if r.returncode != 0 or re.search(r"^(Exception|Thrift error|Error)", out, re.M):
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
        return res
