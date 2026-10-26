# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run benchmark queries: warmup + timed iterations, with summary statistics."""

import statistics
import subprocess

from .client import SqlError


def summarize(values):
    if not values:
        return {}
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": max(values)}


def run_queries(bench, client, queries, iterations, warmup, log=print, out=None, verifier=None):
    """Run the queries; results are appended to `out` (if given) so they survive an exception.

    With a verifier, the rows of the last timed iteration are checked against the reference
    and the outcome is recorded as rec["verification"].
    """
    out = [] if out is None else out
    for i, (name, sql) in enumerate(queries):
        rec = {"query": name, "status": "ok"}
        try:
            first = None
            for _ in range(warmup):
                r = client.execute(sql)
                first = first or r
            execs, totals = [], []
            digest = rows = None
            for it in range(iterations):
                r = client.execute(sql, rows=bool(verifier) and it == iterations - 1)
                execs.append(r.exec_ms)
                totals.append(r.total_ms)
                digest, rows = r.digest, r.rows
            rec.update(rows=rows, digest=digest, exec_ms=summarize(execs), total_ms=summarize(totals),
                       exec_ms_all=execs, cold_total_ms=first.total_ms if first else None)
            line = "  %-8s rows=%-6s exec med %8.1f ms   total med %8.1f ms (min %.1f)" % (
                name, rows, rec["exec_ms"]["median"], rec["total_ms"]["median"], rec["total_ms"]["min"])
            if verifier:
                v = rec["verification"] = verifier.check(name, sql, r.table)
                line += "  [%s]" % v["status"].upper()
                if v["message"]:
                    line += "\n           %s" % v["message"].replace("\n", "\n           ")
            log(line)
        except SqlError as e:
            rec.update(status="error", error=str(e)[:500])
            log("  %-8s FAILED: %s" % (name, str(e).splitlines()[-1][:120]))
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            # not a query error: the server/container is gone, so the rest cannot run
            rec.update(status="error", error=str(e)[:500])
            log("  %-8s FAILED (server unavailable): %s" % (name, str(e).splitlines()[-1][:120]))
            out.append(rec)
            for n, _ in queries[i + 1:]:
                out.append({"query": n, "status": "skipped", "error": "server unavailable"})
            log("  skipping %d remaining quer%s" % (len(queries) - i - 1,
                                                    "y" if len(queries) - i - 1 == 1 else "ies"))
            break
        out.append(rec)
    return out
