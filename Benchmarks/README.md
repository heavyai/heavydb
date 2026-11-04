# HeavyDB benchmarks

A small, extensible harness for SQL benchmarks. A benchmark is a directory
(`benchmarks/<name>/`) with a table list, a schema and a set of query files; the data
is any directory of Parquet files passed on the command line. Today: **TPC-H**.

HeavyDB runs inside the product container image by default (the image produced by
`dev-tools/dev.sh build --docker`); on Slurm the same image, converted to an enroot
`.sqsh`, is used. The harness itself is plain Python 3.11+ (stdlib only).

## Quick start (local, Docker)

```bash
# 1. product image (once): dev-tools/dev.sh build --docker   -> heavyai-<distro>-cuda-<tag>-<arch>
# 2. run
Benchmarks/scripts/run_local.sh tpch /path/to/tpch-sf1
Benchmarks/scripts/run_local.sh tpch /path/to/tpch-sf1 --queries q01,q06 --iterations 5
```

Same thing without the wrapper: `Benchmarks/run_benchmark.py run tpch --data <dir> [options]`.
`run_benchmark.py list` shows available benchmarks. Common options:

| option | meaning |
| --- | --- |
| `--runtime docker\|enroot\|local` | where HeavyDB runs (default `docker`) |
| `--image` | docker image (default `$HEAVYDB_IMAGE` or newest local `heavyai-*`) / enroot `.sqsh` |
| `--build-dir` | with `--runtime local`: use `bin/heavydb`, `heavysql`, `initheavy` from a build tree |
| `--queries q01,q06` | subset (default: every query not listed under `[disabled]`) |
| `--iterations N --warmup N` | timed runs per query / untimed runs first (first warmup = cold time) |
| `--import-mode native\|foreign` | `COPY FROM` into HeavyDB storage (default) or foreign tables over the Parquet files |
| `--cpu-only` / `--gpu` | default: GPU if `nvidia-smi` works |
| `--server-arg X` | extra `heavydb` flag, repeatable (e.g. `--server-arg=--enable-watchdog=false`) |
| `--output DIR --label L` | results JSON + CSV go to DIR (default `results/`), named `<benchmark>-<label>-<dataset>-<time>[-j<jobid>_<task>].json`; an existing report is never overwritten (`-2`, `-3`, ... is appended) |
| `--expected DIR` | verify each query's result against reference Parquet files (see below) |
| `--dry-run` | print the docker/enroot/heavysql commands only |

Output: per-query rows, server-side execution/total time (min/median/mean/max), a result
digest (to compare runs for correctness), import time per table, and environment info
(host, image id, git sha, GPU/CPU mode, scale factor from `metadata.json` if present).

## Data layout

`<dataset>/<table>/` holds one or more Parquet files per table. File names are not
assumed (`lineitem-1.parquet`, `lineitem_3.snappy.parquet`, `part.2.parquet` all work),
so partitioned datasets just work: the directory is handed to `COPY FROM` / the foreign table.
For TPC-H the tables are `region nation supplier customer part partsupp orders lineitem`.
HeavyDB does not convert Parquet column types on import, and generators differ (int32 vs
int64 keys, double vs decimal(p,s) money columns). So the importer reads each table's
column types from the Parquet footer of its first file (stdlib parser, no pyarrow needed) and
sets the numeric/date column types of `schema.sql` to match before creating the table; text
columns keep the schema's encoding. Changes are printed during import
(`[column types adapted: ...]`) and saved under `import.<table>.adjusted` in the JSON report,
so note that a double-typed dataset benchmarks `DOUBLE` columns rather than `DECIMAL`.
If the footer can't be read, a HeavyDB conversion error triggers the same adjustment as a
fallback. `--strict-schema` disables all of this, and `--schema FILE` / `--var NAME=VALUE`
(e.g. `DEC_PRECISION`) change the starting schema.
Tested with int32 and int64 keys, decimal(p,s) and double money columns, and both single-file
and partitioned tables.

### Import modes

- `native` (default): `COPY FROM` loads the Parquet data into HeavyDB storage. The reported import
  time is the load, and a rejected row or a load failure fails the run.
- `foreign`: `CREATE FOREIGN TABLE` over the Parquet files. The reported "import" time covers table
  creation only (a fraction of a second), row counts are not recorded, and the data is read on the
  first query, so a cold run includes that read.
  Column types are not validated at creation either: a mismatch between the schema and the files
  (for example int32 keys declared `BIGINT`) only fails the queries that read that column, with
  `Conversion from Parquet type ...`. The footer-based type adaptation above applies in both
  modes and avoids this; the error-driven fallback only exists for native imports, so with
  `--strict-schema` (or an unreadable footer) fix such errors with `--schema` / `--var`.

## Generating datasets

Use velox-testing's generator (`benchmark_data_tools/`), which writes decimal columns,
int64 keys, and splits large tables into `<table>-N.parquet` files:

```bash
VELOX_TESTING_DIR=/path/to/velox-testing \
  Benchmarks/scripts/generate_data.sh 10 /path/to/tpch-sf10-decimal -j 16 -v
```

The wrapper refuses to write into an existing directory (the generator deletes its output
dir first) and rejects `--convert-decimals-to-floats`. Datasets generated this way match
`schema.sql` as is, so no column types need adapting. Output at SF0.1 was verified to run
all enabled TPC-H queries.

## Adding a benchmark

1. `benchmarks/<name>/benchmark.toml`: `name`, `description`, `iterations`, `warmup`, `[[tables]]`
   entries (`name`, optional `path`/`glob`), optional `[disabled]` (`query = "reason"`),
   `setup_sql`/`teardown_sql` lists.
2. `benchmarks/<name>/schema.sql`: one `CREATE TABLE name (...);` per table
   (foreign-table DDL is derived from it).
3. `benchmarks/<name>/queries/*.sql`: one query per file; the file stem is the query name.

Nothing else changes: any dataset of the right shape can be passed with `--data`.
New execution environments (podman, apptainer, ...) are one class in `harness/runtime/`.

Query files may use `${NAME}` placeholders, filled from `[vars]` in `benchmark.toml`. `${SF}` is
always available: it comes from the dataset's `metadata.json` `scale_factor` (override with
`--var SF=VALUE`), and a query that uses it fails with a clear error if neither is set. TPC-H Q11
uses it for its scale-dependent threshold (`0.0001 / SF`). Write a literal dollar sign as `$$`.

## Verifying results

`--expected DIR` compares every query's result with a reference file in `DIR`
(`q01.parquet`, `q1.parquet` or `01.parquet`; one file per query, made for the same dataset and
scale factor, e.g. by DuckDB or Presto). The comparison is velox-testing's
(`harness/result_comparison.py`, copied unchanged): float columns use rel 1e-5 / abs 1e-8,
columns heavysql prints as fixed-point decimals use half a unit of their last digit, other columns
must be equal; rows are compared in a canonical order, after checking that `ORDER BY` was
respected, and ties at a `LIMIT` cutoff are compared by their `ORDER BY` values only.
It needs `pip install -r requirements-verify.txt` (on Slurm, activate it with `--pre-command`).

```bash
Benchmarks/run_benchmark.py run tpch --data /path/to/tpch-sf1 --expected /path/to/tpch-sf1-expected
```

Each query gets `verification` in the JSON report (`passed`, `failed`, `expected-failure`,
`not-validated`) and a column in the CSV; the run exits with 1 if any check `failed`. The rows
checked are those of the last timed iteration. heavysql prints NULL and an empty string alike, so
the two cannot be told apart. A query whose check is known to fail is listed with a reason under
`[verify_xfail]` in `benchmark.toml`: a failure then shows as `expected-failure` (a pass is still
a pass). For TPC-H these are Q4 (correlated `EXISTS` returns wrong, varying counts) and Q8
(decimal division truncates instead of rounding, on decimal data).

## Slurm (enroot)

Convert the product image yourself. On a cluster where `enroot import` can't write
`/var/lib/enroot`, use `slurm/save_image.sh <image> <out.sqsh> [-- srun args]`, which pulls
via pyxis (`srun --container-image/--container-save`) and needs registry credentials in
`~/.config/enroot/.credentials` (see the script header). Private Docker Hub images must be
named with the user, `USER@registry-1.docker.io#<org>/<repo>:<tag>`; a bare name is pulled
anonymously and fails with 401. Elsewhere
`enroot import -o heavydb.sqsh dockerd://<image>` works too.
Then:

```bash
Benchmarks/slurm/submit.py tpch --image /shared/heavydb.sqsh \
    --data /shared/tpch/sf1 --data /shared/tpch/sf100 \
    --partition gpu --gres gpu:8 --time 02:00:00 --output-dir /shared/results \
    -- --queries q01,q06 --iterations 5
```

Each `--data` becomes one array task = one node run: the job creates an enroot container,
starts HeavyDB in it with the dataset bind-mounted at `/data`, runs the same harness, and
writes results to `--output-dir`. `--dry-run` prints the sbatch script
(`slurm/run_benchmark.sbatch` is the template); `--pre-command` adds shell lines
(e.g. `module load enroot`). Arguments after `--` go to `run_benchmark.py run`.

## Notes

- **Server logs:** the server's own logs (`heavydb.INFO`, `log4j.log`, ...) are saved before its storage
  is deleted: to `<work-dir>/logs/`, or, for a temp work dir, to `<output>/heavydb-logs-<time>-<pid>/`.
  Slurm jobs copy them to `<output-dir>/heavydb-logs-<jobid>_<task>/` next to `heavydb-<jobid>_<task>.log`
  (the server's stdout/stderr).

- TPC-H Q21 is disabled: HeavyDB cannot plan its correlated `EXISTS`/`NOT EXISTS` with `<>`.
- Query text is the standard TPC-H text with the spec's validation parameters; columns used
  in GROUP BY are dictionary-encoded in `schema.sql` (HeavyDB cannot group by none-encoded text).
- Reading the docker socket needs docker group membership; the harness never needs root.
