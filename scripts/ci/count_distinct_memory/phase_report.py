#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel-thread process memory around query memory initialization.

Needs a server log written with --verbose=true --allow-memory-status-log=true.
Each kernel thread logs process memory four times: before query execution, before query
memory initialization, after query memory initialization, after query execution.
The growth between the 2nd and 3rd value is the init phase (allocateCountDistinctSet loop in
QueryMemoryInitializer::initRowGroups); between the 3rd and 4th it is the scan.

usage: phase_report.py <heavydb.INFO> [sql-prefix-of-the-query, default "SELECT COUNT(*) AS total"]
"""
import collections
import re
import sys

log = open(sys.argv[1], errors="replace").read().splitlines()
prefix = sys.argv[2] if len(sys.argv) > 2 else "SELECT COUNT(*) AS total"
start = max(i for i, l in enumerate(log) if "Start query execution: " + prefix in l)

per_thread = collections.OrderedDict()
for line in log[start:]:
    m = re.match(r'(\S+) \S+ \d+ \d+ (\d+) Execute.cpp:\d+ .*"ProcessMB": ([0-9.]+)', line)
    if m:
        per_thread.setdefault(m.group(2), []).append((m.group(1)[11:23], float(m.group(3))))
    if "requested between" in line:
        print(line[line.find("In Executor") :][:120])

print("\nthread  before-init -> after-init (init phase)   -> after-execution (scan phase)")
init_total = scan_total = 0.0
for tid, v in per_thread.items():
    if len(v) != 4:
        continue  # not a full kernel sequence (e.g. outer query)
    init, scan = v[2][1] - v[1][1], v[3][1] - v[2][1]
    if init < 20:
        continue  # estimator / tiny kernels
    init_total += init
    scan_total += scan
    print(f"{tid:>6}  {v[1][1]:7.0f} MB -> {v[2][1]:7.0f} MB (+{init:5.0f} MB, at {v[2][0]})"
          f" -> {v[3][1]:7.0f} MB (+{scan:4.0f} MB, at {v[3][0]})")
tot = init_total + scan_total
if tot:
    print(f"\nsum over kernel threads: init +{init_total:.0f} MB ({100*init_total/tot:.0f}%), "
          f"scan +{scan_total:.0f} MB ({100*scan_total/tot:.0f}%)")
