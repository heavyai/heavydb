# Count-distinct memory checks

End-to-end checks, against a real `heavydb` server, that one set-based
`COUNT(DISTINCT <runtime expression>)` over a high-cardinality `GROUP BY` cannot exhaust
host memory or leave a query that cannot be stopped. They run in the weekly
`count-distinct-memory` job (`.github/workflows/weekly.yml`) and locally with:

```
dev-tools/dev.sh test count-distinct-memory
# or, against an existing build:
BUILD_DIR=<dir with bin/heavydb bin/initheavy bin/heavysql> scripts/ci/run_count_distinct_memory.sh
```

On a GPU runner the server runs in GPU mode, so the query is planned for GPU, cannot run
there (set-based COUNT(DISTINCT)) and is retried on CPU; the run reports how many
retries it saw. `GPU=0` forces CPU mode. Faster, deterministic coverage of the same fix lives in
the in-process gtests `CountDistinctSetMemoryTest` (`Tests/GroupByTest.cpp`) and
`Interrupt.StatusReturnsToKernelAfterEstimatorStep` (`Tests/RuntimeInterruptTest.cpp`); these
checks add what only a real server shows.

| Check | Desired behaviour |
|---|---|
| T0 control | the same query with `MAX()` instead of `COUNT(DISTINCT)` stays within the CPU pool |
| T1 limits | the set-based query stays within pool + slack, or fails cleanly with the server up |
| T2 status | `SHOW QUERIES` does not report `RUNNING_REDUCTION` while the kernel step runs |
| T3 kill | `KILL QUERY` while the step is allocating stops further growth (<=100 MB) and the query (<=2 s) |

Without the fix T1-T3 fail (T1: the server is killed at the memory cap; T2: stuck on
`RUNNING_REDUCTION`; T3: the allocation continues for another 300+ MB). T0 always passes; it
shows the harness is not failing for unrelated reasons.

## Query shape and data

`cdm(grp TEXT DICT(32), kind TEXT DICT(8), band SMALLINT, net TEXT DICT(16))` with 2 rows per
group and `NUM_GROUPS` (default 2,000,000) distinct groups, and
`COUNT(DISTINCT CAST(kind AS VARCHAR) || '_' || CAST(band AS VARCHAR))` per `grp`. The
expression has no value range, so it uses the set-based implementation
(`CountDistinctImplType::UnorderedSet`): one heap-allocated set per group-by entry, per kernel
thread, created while the output buffer is initialized.

The table is loaded with the default pool (a 256 MB pool is too small for the dictionary
import), then the server is restarted with `--cpu-buffer-mem-bytes` (T0/T1: 256 MB; T2/T3: 2 GiB
so the query is admitted and actually allocates), 8 CPU threads and 4 executors. A watchdog
SIGKILLs the server above baseline + pool + slack (`SLACK_MB`, default 128) and reports it as
"killed at cap", standing in for the kernel OOM killer. Knobs: `NUM_GROUPS`, `THREADS`,
`EXECUTORS`, `POOL_BYTES`, `RUN_POOL_BYTES`, `SLACK_MB`, `WORK_DIR`, `PORT`, `GPU`.

## Diagnostics (not run by CI)

`QSEL=distinct NUM_GROUPS=2000000 THREADS=4 explore.sh` runs the query once with memory-status
logging; `phase_report.py <heavydb.INFO>` splits each kernel thread's process memory into the
query-memory-initialization phase and the scan phase. On this shape, about three
quarters of the growth is the init loop and the rest is the sets filling during the scan.

## Limits

* Not covered: sibling kernels stopping after one fails with `OUT_OF_CPU_MEM` (needs a
  per-thread allocation failure, i.e. ~192 threads or a nearly full pool), very large group counts
  (100M+), and a real kernel/cgroup OOM kill.
* T1 and T3 are RSS and timing thresholds; if they flake on a runner, widen `SLACK_MB` or the
  kill thresholds in `tests.sh` rather than disabling the check.
* The sandbox/CI container needs `python3`, `awk`, `java` (for Calcite) and `heavysql`;
  no root-only features are used.
