#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render and submit a Slurm job that runs a benchmark in an enroot container.

The enroot image must be the HeavyDB product image converted by the user, e.g.
  enroot import -o heavydb.sqsh dockerd://<image>
Each --data path becomes one array task (one node run per dataset).

  submit.py tpch --image /shared/heavydb.sqsh --data /shared/tpch/sf1 --data /shared/tpch/sf100 \\
            --partition gpu --gres gpu:8 --time 02:00:00 -- --queries q01,q06 --iterations 5
"""

import argparse
import os
import shlex
import string
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("benchmark")
    p.add_argument("--image", required=True, help="enroot .sqsh image (HeavyDB product image)")
    p.add_argument("--data", action="append", required=True, help="dataset root; repeat for an array job")
    p.add_argument("--output-dir", default=str(BENCH_ROOT / "results"), help="results dir (shared FS)")
    p.add_argument("--label", default="slurm")
    p.add_argument("--scratch-dir", help="node-local dir for the server data and logs (a hdb-bench-<jobid> "
                   "subdir is created and removed); default: $SLURM_TMPDIR, $TMPDIR, then /tmp on the node")
    p.add_argument("--partition")
    p.add_argument("--account")
    p.add_argument("--gres", help="e.g. gpu:8 (omit for CPU-only)")
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--cpus-per-task", type=int)
    p.add_argument("--mem", help="e.g. 0 for all node memory")
    p.add_argument("--nodelist")
    p.add_argument("--job-name")
    p.add_argument("--pre-command", action="append", default=[],
                   help="shell line run before the benchmark (module load, env setup); repeatable")
    p.add_argument("--script-dir", help="where to write the rendered script (default: <output-dir>)")
    p.add_argument("--dry-run", action="store_true", help="print the script, do not submit")
    argv = sys.argv[1:] if argv is None else list(argv)
    extra = []
    if "--" in argv:  # everything after '--' is passed to run_benchmark.py run
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)

    out_dir = Path(a.output_dir).resolve()
    job = a.job_name or "bench-%s" % a.benchmark
    d = ["#SBATCH --job-name=%s" % job, "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
         "#SBATCH --time=%s" % a.time, "#SBATCH --output=%s/slurm-%%A_%%a.out" % out_dir,
         "#SBATCH --array=0-%d" % (len(a.data) - 1)]
    for flag, val in [("partition", a.partition), ("account", a.account), ("gres", a.gres),
                      ("cpus-per-task", a.cpus_per_task), ("mem", a.mem), ("nodelist", a.nodelist)]:
        if val is not None:
            d.append("#SBATCH --%s=%s" % (flag, val))
    tpl = string.Template((HERE / "run_benchmark.sbatch").read_text())
    script = tpl.substitute(
        SBATCH_DIRECTIVES="\n".join(d),
        DATASETS=" ".join(shlex.quote(str(Path(x).resolve())) for x in a.data),
        BENCH_ROOT=BENCH_ROOT, OUTPUT_DIR=out_dir, BENCHMARK=a.benchmark,
        SCRATCH_BASE=shlex.quote(a.scratch_dir) if a.scratch_dir else '"${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"',
        IMAGE=Path(a.image).resolve(), LABEL=a.label, PRE_COMMANDS="\n".join(a.pre_command),
        EXTRA_ARGS=" ".join(shlex.quote(x) for x in extra))
    if a.dry_run:
        print(script)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(a.script_dir or out_dir) / ("%s-%d.sbatch" % (job, os.getpid()))
    path.write_text(script)
    return subprocess.run(["sbatch", str(path)]).returncode


if __name__ == "__main__":
    sys.exit(main())
