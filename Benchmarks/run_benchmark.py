#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark driver.

  run_benchmark.py list
  run_benchmark.py run tpch --data /home/misiug/Data/scale-1 [--queries q01,q06]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import importer, registry, report, runner  # noqa: E402
from harness.client import HeavySqlClient  # noqa: E402
from harness.runtime import RUNTIMES, make_runtime  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list available benchmarks")
    r = sub.add_parser("run", help="import data and run a benchmark")
    r.add_argument("benchmark")
    r.add_argument("--data", required=True, help="dataset root (one subdir per table)")
    r.add_argument("--runtime", choices=sorted(RUNTIMES), default="docker")
    r.add_argument("--image", help="docker image / enroot .sqsh file")
    r.add_argument("--build-dir", help="build tree for --runtime local")
    r.add_argument("--schema", help="schema file overriding the benchmark's schema.sql "
                   "(for datasets whose Parquet column types differ)")
    r.add_argument("--var", action="append", default=[], metavar="NAME=VALUE",
                   help="override a [vars] entry of benchmark.toml used in schema.sql (repeatable)")
    r.add_argument("--strict-schema", action="store_true",
                   help="do not adapt column types to the Parquet files (fail on a mismatch)")
    r.add_argument("--queries", help="comma-separated query names (default: all enabled)")
    r.add_argument("--iterations", type=int)
    r.add_argument("--warmup", type=int)
    r.add_argument("--import-mode", choices=["native", "foreign"], default="native")
    r.add_argument("--output", default=str(Path(__file__).resolve().parent / "results"),
                   help="results directory (default: Benchmarks/results)")
    r.add_argument("--label", default="")
    r.add_argument("--work-dir", help="scratch dir for server storage/logs (default: temp dir)")
    r.add_argument("--cpu-only", dest="cpu_only", action="store_true", default=None)
    r.add_argument("--gpu", dest="cpu_only", action="store_false")
    r.add_argument("--server-arg", action="append", default=[], help="extra heavydb flag (repeatable)")
    r.add_argument("--runtime-arg", action="append", default=[],
                   help="extra arg for docker run / enroot start (repeatable)")
    r.add_argument("--dry-run", action="store_true", help="print the commands without running")
    return p.parse_args(argv)


def select_queries(bench, wanted, log=print):
    qs = bench.queries()
    if wanted:
        names = {w.strip().lower() for w in wanted.split(",")}
        norm = lambda n: n.lower()  # noqa: E731
        qs = [(n, s) for n, s in qs if norm(n) in names or norm(n).lstrip("q").lstrip("0") in names]
        if not qs:
            raise SystemExit("no queries match %s" % wanted)
        return qs  # explicit selection overrides the disabled list
    enabled = []
    for n, s in qs:
        if n in bench.disabled:
            log("  skipping %s (disabled: %s)" % (n, bench.disabled[n]))
        else:
            enabled.append((n, s))
    return enabled


def cmd_run(a):
    bench = registry.load(a.benchmark)
    for kv in a.var:
        k, _, v = kv.partition("=")
        if k not in bench.vars:
            raise SystemExit("unknown --var %s (known: %s)" % (k, ", ".join(bench.vars) or "none"))
        bench.vars[k] = v
    if a.schema:
        bench.schema_file = str(Path(a.schema).resolve())
    data = Path(a.data).resolve()
    if not data.is_dir():
        raise SystemExit("data dir not found: %s" % data)
    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="hdb-bench-"))
    kw = dict(data_root=data, work_dir=work, cpu_only=a.cpu_only, server_args=a.server_arg,
              dry_run=a.dry_run)
    if a.runtime == "docker":
        kw.update(image=a.image, docker_args=a.runtime_arg)
    elif a.runtime == "enroot":
        kw.update(image=a.image, enroot_args=a.runtime_arg)
    else:
        kw.update(build_dir=a.build_dir)
    rt = make_runtime(a.runtime, **kw)
    queries = select_queries(bench, a.queries)
    iters = a.iterations or bench.iterations
    warm = a.warmup if a.warmup is not None else bench.warmup
    print("benchmark=%s data=%s runtime=%s mode=%s queries=%d iters=%d warmup=%d" % (
        bench.name, data, a.runtime, a.import_mode, len(queries), iters, warm))
    with rt:
        client = HeavySqlClient(rt)
        print("importing tables:")
        stats = importer.import_dataset(bench, client, rt, data, a.import_mode,
                                        adapt_types=not a.strict_schema)
        for s in bench.setup_sql:
            client.execute(s)
        print("running queries:")
        results = runner.run_queries(bench, client, queries, iters, warm)
        for s in bench.teardown_sql:
            client.execute(s)
        env_runtime = rt
        if a.dry_run:
            return 0
        path = report.write_report(a.output, a.label, bench, data, a.import_mode, stats, results,
                                   env_runtime)
    report.print_summary(results, stats)
    print("report: %s" % path)
    return 0 if all(r["status"] == "ok" for r in results) else 1


def main(argv=None):
    a = parse_args(argv)
    if a.cmd == "list":
        for n in registry.names():
            b = registry.load(n)
            print("%-12s %s (%d tables, %d queries)" % (n, b.description, len(b.tables), len(b.queries())))
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
