#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark driver.

  run_benchmark.py list
  run_benchmark.py run tpch --data /path/to/tpch-sf1 [--queries q01,q06]
"""

import argparse
import datetime
import os
import re
import shutil
import signal
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import importer, registry, report, runner  # noqa: E402
from harness.client import HeavySqlClient  # noqa: E402
from harness.dataset import read_metadata  # noqa: E402
from harness.runtime import RUNTIMES, make_runtime  # noqa: E402


def _positive_int(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1 (use --warmup 0 to skip warmup)")
    return n


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
    r.add_argument("--iterations", type=_positive_int)
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


def dataset_scale_factor(data):
    """scale_factor from <data>/metadata.json as a plain decimal string (e.g. '1000', '0.01')."""
    sf = read_metadata(data).get("scale_factor")
    try:
        return None if sf is None else format(Decimal(str(sf)).normalize(), "f")
    except ArithmeticError:
        return None


def _canon(name):
    """'Q01' / 'q01' / 'q1' -> 'q1': case-insensitive, leading zeros of the number ignored."""
    return re.sub(r"^([a-z]*?)0+(?=\d)", r"\1", name.strip().lower())


def _matches(token, name):
    """A token selects a query by name ('q1', 'q01', 'Q1') or by bare number ('1')."""
    t, n = _canon(token), _canon(name)
    return t == n or (t.isdigit() and n.lstrip("abcdefghijklmnopqrstuvwxyz") == t)


def select_queries(bench, wanted, log=print):
    qs = bench.queries()
    if wanted:
        tokens = [w for w in (x.strip() for x in wanted.split(",")) if w]
        unknown = [t for t in tokens if not any(_matches(t, n) for n, _ in qs)]
        if unknown or not tokens:
            raise SystemExit("unknown query name(s): %s (available: %s)" % (
                ", ".join(unknown) or wanted, ", ".join(n for n, _ in qs)))
        # explicit selection overrides the disabled list
        return [(n, s) for n, s in qs if any(_matches(t, n) for t in tokens)]
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
        if k not in bench.vars and k != "SF":
            raise SystemExit("unknown --var %s (known: SF, %s)" % (k, ", ".join(bench.vars) or "none"))
        bench.vars[k] = v
    if a.schema:
        bench.schema_file = str(Path(a.schema).resolve())
    data = Path(a.data).resolve()
    if not data.is_dir():
        raise SystemExit("data dir not found: %s" % data)
    auto_work = not a.work_dir
    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="hdb-bench-"))
    # scancel/timeouts send SIGTERM: turn it into an exit so the runtime cleans up
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
    try:
        return _run(a, bench, data, work, queries_arg=a.queries)
    finally:
        if auto_work:
            logs = work / "logs"
            if logs.is_dir():  # the temp dir is about to be deleted: keep the server logs
                dest = Path(a.output) / ("heavydb-logs-%s-%d" % (
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), os.getpid()))
                try:
                    shutil.copytree(logs, dest)
                    print("server logs: %s" % dest)
                except OSError as e:
                    print("warning: could not keep the server logs: %s" % e, file=sys.stderr)
            shutil.rmtree(work, ignore_errors=True)


def _run(a, bench, data, work, queries_arg):
    kw = dict(data_root=data, work_dir=work, cpu_only=a.cpu_only, server_args=a.server_arg,
              dry_run=a.dry_run)
    if a.runtime == "docker":
        kw.update(image=a.image, docker_args=a.runtime_arg)
    elif a.runtime == "enroot":
        kw.update(image=a.image, enroot_args=a.runtime_arg)
    else:
        kw.update(build_dir=a.build_dir)
    rt = make_runtime(a.runtime, **kw)
    queries = select_queries(bench, queries_arg)
    extra = {}
    if "SF" not in bench.vars:  # --var SF=... wins over the dataset's metadata.json
        sf = dataset_scale_factor(data)
        if sf is not None:
            extra["SF"] = sf
    queries = [(n, bench.render_query(n, t, extra)) for n, t in queries]
    iters = a.iterations if a.iterations is not None else bench.iterations
    warm = a.warmup if a.warmup is not None else bench.warmup
    print("benchmark=%s data=%s runtime=%s mode=%s queries=%d iters=%d warmup=%d" % (
        bench.name, data, a.runtime, a.import_mode, len(queries), iters, warm))
    stats, results = {}, []
    path, outcome = None, "aborted"
    if a.import_mode == "foreign":
        print("foreign mode: import times cover table creation only; the data is read on first query")
    with rt:
        try:
            client = HeavySqlClient(rt)
            print("importing tables:")
            importer.import_dataset(bench, client, rt, data, a.import_mode,
                                    adapt_types=not a.strict_schema, stats=stats)
            for s in bench.setup_sql:
                client.execute(s)
            print("running queries:")
            runner.run_queries(bench, client, queries, iters, warm, out=results)
            for s in bench.teardown_sql:
                client.execute(s)
            outcome = "complete"
        except BaseException as e:
            outcome = "aborted: %s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else "")
            raise
        finally:
            if not a.dry_run and (stats or results):
                try:  # keep whatever was measured, even if the run did not finish
                    path = report.write_report(a.output, a.label, bench, data, a.import_mode, stats,
                                               results, rt, status=outcome)
                    if outcome != "complete":
                        print("partial report (%s): %s" % (outcome, path), file=sys.stderr)
                except Exception as e:  # never mask the original error
                    print("warning: could not write the report: %s" % e, file=sys.stderr)
    if a.dry_run:
        return 0
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
