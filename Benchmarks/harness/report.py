# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result reporting: JSON + CSV files and a console summary."""

import csv
import datetime
import json
import os
import platform
import re
import subprocess
from pathlib import Path

from .dataset import read_metadata


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return ""


def environment(runtime):
    env = {"host": platform.node(), "cpus": os.cpu_count(), "git_sha": _git_sha(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "python": platform.python_version()}
    env.update(runtime.describe())
    return env


def write_report(out_dir, label, bench, data_root, import_mode, import_stats, results, runtime,
                status="complete"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset = re.sub(r"[^A-Za-z0-9._-]", "_", Path(data_root).name)
    stem = "%s-%s-%s-%s" % (bench.name, label or "run", dataset, stamp)
    if os.environ.get("SLURM_JOB_ID"):  # array tasks share the label, so tell them apart
        stem += "-j%s_%s" % (os.environ["SLURM_JOB_ID"], os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    scale = read_metadata(data_root).get("scale_factor")
    doc = {"benchmark": bench.name, "label": label, "timestamp": stamp, "data": str(data_root), "dataset": dataset,
           "status": status, "scale_factor": scale, "import_mode": import_mode, "import": import_stats,
           "environment": environment(runtime), "queries": results}
    # never overwrite an earlier report: claim the JSON name exclusively, adding -2, -3, ... if taken
    base, n = stem, 1
    while True:
        try:
            jf = open(out_dir / (stem + ".json"), "x")
            break
        except FileExistsError:
            n += 1
            stem = "%s-%d" % (base, n)
    with jf:
        jf.write(json.dumps(doc, indent=2))
    with open(out_dir / (stem + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "status", "rows", "exec_min_ms", "exec_median_ms", "total_min_ms",
                    "total_median_ms", "total_mean_ms", "total_max_ms", "cold_total_ms", "digest",
                    "verification"])
        for r in results:
            e, t = r.get("exec_ms", {}), r.get("total_ms", {})
            w.writerow([r["query"], r["status"], r.get("rows"), e.get("min"), e.get("median"),
                        t.get("min"), t.get("median"), t.get("mean"), t.get("max"),
                        r.get("cold_total_ms"), r.get("digest"),
                        r.get("verification", {}).get("status")])
    return out_dir / (stem + ".json")


def print_summary(results, import_stats):
    ok = [r for r in results if r["status"] == "ok"]
    print("\n%-8s %10s %12s %12s" % ("query", "rows", "exec med ms", "total med ms"))
    for r in results:
        if r["status"] == "ok":
            print("%-8s %10s %12.1f %12.1f" % (r["query"], r["rows"], r["exec_ms"]["median"],
                                              r["total_ms"]["median"]))
        else:
            print("%-8s %10s" % (r["query"], "SKIPPED" if r["status"] == "skipped" else "ERROR"))
    if ok:
        print("%-8s %10s %12.1f %12.1f" % ("TOTAL", "", sum(r["exec_ms"]["median"] for r in ok),
                                          sum(r["total_ms"]["median"] for r in ok)))
    if import_stats:
        print("import: %.1fs total" % sum(s["seconds"] for s in import_stats.values()))
    print("%d/%d queries succeeded" % (len(ok), len(results)))
    checks = [r["verification"]["status"] for r in results if "verification" in r]
    if checks:
        print("verification: " + ", ".join("%d %s" % (checks.count(s), s)
                                           for s in ("passed", "failed", "expected-failure", "not-validated")
                                           if s in checks))


def verification_failed(results):
    return any(r.get("verification", {}).get("status") == "failed" for r in results)
