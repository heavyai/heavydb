# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run benchmark queries: warmup + timed iterations, with summary statistics."""

import statistics

from .client import SqlError


def summarize(values):
    if not values:
        return {}
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": max(values)}


def run_queries(bench, client, queries, iterations, warmup, log=print):
    out = []
    for name, sql in queries:
        rec = {"query": name, "status": "ok"}
        try:
            first = None
            for _ in range(warmup):
                r = client.execute(sql)
                first = first or r
            execs, totals = [], []
            digest = rows = None
            for _ in range(iterations):
                r = client.execute(sql)
                execs.append(r.exec_ms)
                totals.append(r.total_ms)
                digest, rows = r.digest, r.rows
            rec.update(rows=rows, digest=digest, exec_ms=summarize(execs), total_ms=summarize(totals),
                       exec_ms_all=execs, cold_total_ms=first.total_ms if first else None)
            log("  %-8s rows=%-6s exec med %8.1f ms   total med %8.1f ms (min %.1f)" % (
                name, rows, rec["exec_ms"]["median"], rec["total_ms"]["median"], rec["total_ms"]["min"]))
        except SqlError as e:
            rec.update(status="error", error=str(e)[:500])
            log("  %-8s FAILED: %s" % (name, str(e).splitlines()[-1][:120]))
        out.append(rec)
    return out
