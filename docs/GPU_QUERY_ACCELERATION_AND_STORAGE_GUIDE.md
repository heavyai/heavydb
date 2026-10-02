# GPU Query Acceleration and Native Storage Guide

These settings allow HeavyDB to execute complex analytical plans that previously failed,
fell back to CPU, or exceeded intermediate memory and hash-table limits, while keeping
more joins, intermediate results, and reductions on GPUs. In the final 8xH200 evaluation,
all 22 TPC-H SF=1000 queries matched their references with both CPU retry paths disabled.
A 20-repetition query-major run averaged 4.925 seconds of start-to-next-start query
intervals and 4.631 seconds of HeavyDB execution time per 22-query sequence. The same
source tree completed an SSB SF=1000 query-major sequence in 1.003 seconds of HeavyDB
execution time on average. These are engineering measurements, not official TPC-H or SSB
results. Additional storage and hydration options can reduce bytes read, first-use
hydration cost, and lukewarm restart time.

The features are currently opt-in while broader workload and deployment validation
continues; if that validation remains successful, the broadly applicable planner,
execution, and hydration paths are candidates to become defaults, while persistent-format
and topology-sensitive options may remain explicit.

This guide is for operators, performance engineers, and application owners. Option names,
defaults, and compatibility statements describe the source tree containing this document.
Reference measurements identify their exact source revision separately.

## Status and scope

- The major planner, hash-join, GPU reduction, dictionary-recovery, input, and storage
  features are opt-in.
- With their feature flags disabled, HeavyDB does not select the opt-in planner rules,
  bitmap hash tables, result-reduction pipeline, lazy dictionary scan paths, storage
  compression writer, or direct temporary ResultSet peer reads.
- CPU retry remains enabled by default. Keep both CPU retry paths enabled during an
  initial deployment.
- The largest query-correctness risk is
  `--trust-unenforced-table-constraints=true`. HeavyDB records the new primary-key,
  unique-key, and foreign-key declarations, but does not enforce them on writes. Do not
  enable this option until the actual data has been validated.
- Native compression and sidecar-only metadata change persistent table storage.
  Compression can be reversed with `OPTIMIZE TABLE ... STORAGE_COMPRESSION='NONE'`.
  Sidecar-only metadata must be converted back to physical metadata pages before an
  older HeavyDB binary can open the table.
- The persistent cubin cache is disabled unless an explicit directory is configured.
  HeavyDB does not default it to `$HOME`.
- Storage compression does not improve a fully GPU-resident hot query. It targets disk
  footprint, restart hydration, CPU memory pressure, and first-use transfer volume.

## Feature groups

The options are organized into six layers.

1. **Relational planning and metadata**

   HeavyDB can record declarative `PRIMARY KEY` (PK), `UNIQUE`, and `FOREIGN KEY` (FK)
   metadata.
   An opt-in Calcite rule set uses relational structure, nullability, and trusted key
   proofs to reduce data earlier. The rules do not match SQL text or TPC-H query numbers.

2. **GPU joins and reductions**

   Exact-membership and ranked bitmap hash tables cover eligible semi, anti, and inner
   joins. A device-resident reduction pipeline keeps eligible intermediate ResultSets on
   GPUs, reduces unnecessary host materialization, and supports aggregate,
   count-distinct, window, and multi-stage query shapes on GPUs.

3. **Input hydration and transport**

   GPU prefetch, batched fetch, native FileMgr CPU-buffer bypass, larger pinned transfer
   buffers, and selectable peer-copy transports reduce serialized reads and redundant
   copies. These paths retain compatibility fallbacks when a shape or allocation is not
   eligible.

4. **Native storage**

   Native FileMgr payloads can be written in framed Snappy, LZ4, GDeflate, or Bitcomp
   form when the build provides the required codec support. Adaptive mode selects the
   smaller Snappy or Bitcomp-default representation per FileBuffer. Advisory manifests
   avoid repeated scattered metadata reads. An additional sidecar-only mode can remove
   physical metadata pages for eligible tables.

5. **String dictionary recovery and scans**

   Recovered dictionary payloads can be mapped without immediately rebuilding their ID
   lookup hashes. Associated scan and transformation paths avoid forcing recovery when a
   bounded storage scan is sufficient.

6. **Compilation and result delivery**

   An optional persistent cubin cache reuses validated GPU machine code across server
   restarts. CUDA 13 and newer builds can parallelize driver JIT split compilation. A
   persistent native query client can receive complete results through Thrift or Arrow;
   local Arrow shared memory avoids copying the IPC payload through Thrift.

These feature groups depend on correctness and lifecycle support for ResultSet ownership,
stream-ordered producer/consumer handoff, cancellation cleanup, CPU fallback, cache
invalidation, lazy-fetch handling, and deterministic boundary materialization. That
support is not separately configurable.

### Always-on support with no separate switch

Some supporting capabilities are always available rather than controlled by independent
flags:

- The parser and catalog always understand the constraint syntax in the SQL data
  definition language (DDL). Nothing is stored until an operator declares a constraint,
  and optimizer trust remains separately gated.
- Readers for compressed chunks and FileMgr v3 sidecars remain active so data written by
  an enabled configuration is readable after its creation flag is turned off.
- GPU window, count-distinct, column-layout, and reduction primitives are execution
  capabilities used only by eligible plans. There is no per-function startup flag.
- Ownership, stream-ordering, cache-invalidation, cancellation, and CPU-boundary behavior
  provide correctness guarantees required by the opt-in paths.
- Calcite 1.41 integration and hint propagation are compatibility requirements, not
  performance switches.

If an always-on GPU capability needs to be isolated, first disable its parent planner or
result-pipeline feature. A CPU-only server/query run is the final control.

Unless a section says otherwise, options beginning with `--` are process-wide server
startup options and require a restart to change. `FRAGMENT_SIZE` is a table-layout
property. Constraint DDL and `OPTIMIZE TABLE` are SQL operations with their own locking
and persistence behavior.

### Build and host prerequisites

- GPU execution requires a CUDA-enabled HeavyDB build and a supported NVIDIA driver.
- `nvidia-smi` reports the newest CUDA driver API supported by the installed driver, not
  the installed CUDA toolkit. Verify the build toolkit separately with `nvcc --version`
  and the `CUDA_VERSION` value in `cuda.h`.
- `--cuda-jit-max-parallel-threads` requires a CUDA 13 or newer build when set to anything
  other than `1`. The default `1` retains serial driver JIT compilation; `0` lets the
  driver use all available CPUs.
- Snappy development files are required to build this revision, independently of Parquet
  importer support.
- nvCOMP is optional. Its absence disables native Snappy GPU decompression and the
  GDeflate, Bitcomp, adaptive, compressed-pipeline, and compressed-peer paths that depend
  on the corresponding nvCOMP libraries. Snappy and LZ4 tables remain readable through
  their CPU paths.
- GPUDirect Storage and the `nvidia_fs` kernel module are not required. The implemented
  native fast paths use ordinary files and the Linux page cache.
- NVLink/NVSwitch is optional. Cross-GPU transport falls back according to topology and
  `--peer-copy-mode`.
- Pinned host-memory limits must cover configured jump buffers and staged transfers.
- The HeavyDB process must inherit an open-file limit sized for the native catalog. A
  common 1,024-file soft limit is insufficient for large, highly fragmented datasets and
  can terminate the server with `Too many open files` during scan or import.
- Keep both the NVIDIA compute cache (`CUDA_CACHE_PATH`) and HeavyDB cubin cache on a
  low-latency local filesystem. A network-mounted home directory can turn first startup
  and JIT cache access into minutes of serialized filesystem latency.
- Enable and verify NVIDIA persistence mode on dedicated benchmark hosts. Time
  `nvidia-smi` before starting HeavyDB; a multi-minute first driver initialization is a
  host problem, not HeavyDB query compilation.
- Page size and NUMA policy are host-level benchmark variables. The validated GB300
  Grace host used a supported 64 KiB kernel with automatic NUMA balancing disabled;
  do not assume that configuration is available or beneficial on another platform.

## Interpreting performance measurements

Use distinct labels for distinct cache states:

| State | Server process | OS page cache | HeavyDB CPU/GPU buffers | Main costs measured |
|---|---|---|---|---|
| Cold | New | Dropped | Empty | Storage reads, hydration, just-in-time (JIT) compilation, planning, execution |
| Lukewarm | New | Retained | Empty | Restart, metadata, JIT, DRAM-to-GPU hydration, execution |
| Hot | Existing | Retained | Resident where reused | Planning, execution, reduction, result production |

These are controlled experimental states, not labels that a benchmark mode can establish
by itself. A harness warmup is a lukewarm measurement only when it starts immediately
after a verified server restart, the OS page cache is intentionally retained, and the
HeavyDB CPU and GPU buffer pools are empty. A sequential first pass through Q1-Q22
progressively warms tables shared by later queries. It is a suite-level warmup, not 22
independent per-query lukewarm measurements. Restart before each query when independent
per-query restart cost is the question.

Ordering within the harness also matters. In query-major mode, a harness can run Q1's
warmup and measured repetitions before advancing to Q2. Summing the per-query warmups from
that run does not produce a contiguous post-restart Q1-Q22 interval because Q1's measured
executions warm state used by later queries. Label that value a restarted query-major
first-invocation sum. Use a suite-major pass that submits each of Q1-Q22 once, without
interleaved repetitions, when a contiguous first-suite interval is required.

For engine execution comparisons, use HeavyDB `execution_ms`. For a TPC-H query-set
timing study, use the persistent client's `query_interval_ms`: each interval begins when
the executable query text is submitted and ends when the next query is submitted, except
for the terminal query, which ends when its complete result is received. A query-major
schedule uses the same mechanics but is an engineering locality sequence, not a formal
TPC-H query set. Client receive time, `total_ms`, and per-process wall time cover different
boundaries and must not be mixed in one result series.

Result format is also part of the measured boundary. Thrift is the conservative portable
transport. `arrow-wire` removes text formatting but still transports IPC bytes, while
local-only `arrow-shared-memory` lets the client attach to server-owned Arrow IPC storage.
Reference validation should happen after the timed stream so Python startup and DuckDB
comparison do not inflate query intervals.

### Reference benchmark profiles

The following measurements describe validated configurations, not general performance
guarantees.

#### TPC-H SF=1000

| Item | Reference value |
|---|---|
| Hardware | 8 NVIDIA H200 GPUs with 139 GB per GPU |
| Source revision | `5df877f662e7ecb0e8ab34cf03c740920dc08160` (`Reduce Arrow query-stream result overhead`) |
| Dataset and storage | TPC-H SF=1000 in HeavyDB native tables; the recorded inventory contains mixed legacy-compressed and uncompressed chunks, with FileMgr v2 physical metadata pages retained |
| Table fragment setting | `FRAGMENT_SIZE=16000000` |
| Execution policy | Strict GPU execution with query-step and whole-query CPU retry disabled |
| Correctness | All 22 query results matched the reference results |
| Client and result format | Persistent native query stream with local Arrow shared memory |
| Schedule and metric | Query-major; `query_interval_ms` with 20 measured repetitions |
| Hot query intervals | 4.925 s average, 4.912 s median, 4.874 s minimum, 5.038 s maximum, and 4.800 s sum of per-query best intervals |
| Hot engine execution | 4.631 s average subtotal of HeavyDB `execution_ms` per 22-query sequence |
| Result validation | 22 checked warmups plus 440 checked measurements; all 462 executions matched their references |
| Restarted query-major first-invocation sum | 82.463 s; this is not a contiguous suite-level lukewarm interval because measured repetitions of each query run before the next query's first invocation |

The query-major result preserves locality for optimization comparisons, but it is not an
official TPC-H power or throughput result. A formal query-set-shaped study must submit all
22 queries once per pass, include the required refresh and stream semantics for the chosen
TPC-H metric, and report the prescribed rounded intervals.

#### Star Schema Benchmark SF=1000

| Item | Reference value |
|---|---|
| Hardware | 8 NVIDIA H200 GPUs with 139 GB per GPU |
| Source revision | Integer-sum tree represented by `d6eb3f9501de700c19e28acee01b3ce62445798e`, subsequently included in `5df877f66` |
| Dataset and storage | SSB SF=1000 in HeavyDB native tables with validated PK/FK metadata |
| Execution policy | Strict GPU execution with query-step and whole-query CPU retry disabled |
| Client, schedule, and metric | Persistent native Thrift client; query-major; HeavyDB `execution_ms` |
| Hot execution | 20 repetitions: 1.003 s average, 1.000 s median, 0.977 s minimum, 1.049 s maximum, and 0.960 s sum of per-query best times |
| First warmup | 3.778 s subtotal across the 13 queries |
| Result validation | 13 checked warmups plus 260 checked measurements; all 273 executions matched their references |

The SSB harness follows the 13 query flights Q1.1 through Q4.3, but it is an engineering
workflow rather than a certified or vendor-comparable SSB result. Its Q1 integral sums use
the result-reduction pipeline's exact, null-aware shared-memory reduction; floating-point
and `SUM_IF` aggregates retain their established paths.

Both reference results use multiple feature groups and should not be attributed to one
option. In particular:

- Planner rewrites, ranked bitmap joins, and device-resident reductions primarily affect
  hot execution.
- Cubin caching affects repeated compilation after restart, not already-hot kernels.
- Manifests affect table-open metadata work.
- Prefetch, CPU-buffer bypass, jump buffers, and compression affect hydration.
- Compression normally has no hot-query benefit once the needed columns are already on
  the GPU.

The hot reference used the following materially non-default settings. The cubin path is
deployment-specific and must be replaced with a service-owned directory. Memory and
watchdog values are capacity settings from the reference host, not general defaults.
The Snappy setting below describes the writer configuration; it does not assert that
every pre-existing chunk in the reference catalog was rewritten to Snappy.

```text
--enable-experimental-query-rewrites=true
--enable-bitmap-hashjoin=true
--enable-ranked-bitmap-hashjoin=true
--trust-unenforced-table-constraints=true
--enable-result-reduction-pipeline=true
--enable-partitioned-baseline-gpu-reduction=false
--enable-deferred-lazy-fetch=true
--stringdict-parallel-sort=true
--enable-lazy-string-dictionary-hash-recovery=true
--allow-cpu-retry=false
--allow-query-step-cpu-retry=false

--enable-native-storage-compression=true
--native-storage-compression-codec=snappy
--native-storage-compression-frame-size=65536
--enable-file-mgr-manifests=true
--enable-file-buffer-metadata-sidecar-only=false

--enable-gpu-input-prefetch=true
--enable-gpu-input-batched-prefetch=true
--gpu-input-prefetch-workers=32
--enable-gpu-input-cpu-buffer-bypass=true
--gpu-input-cpu-buffer-bypass-mode=staged
--gpu-input-cpu-buffer-bypass-staging-buffer-bytes=67108864
--gpu-input-cpu-buffer-bypass-reader-threads=16
--gpu-input-compressed-batch-max-bytes=2147483648
--enable-gpu-input-compressed-pipeline=true
--enable-gpu-input-compressed-peer-exchange=true
--enable-gpu-aggregate-payload-host-mapping=false
--enable-gpu-selected-dense-aggregate-payload-fetch=false

--jump-buffer-size=1073741824
--jump-buffer-parallel-copy-threads=16
--jump-buffer-slots-per-device=1
--enable-lazy-jump-buffer-allocation=false
--enable-background-jump-buffer-allocation=false
--jump-buffer-min-d2h-transfer-threshold=1099511627776
--peer-copy-staging-buffer-size=268435456
--enable-temporary-resultset-peer-access=false
--enable-temporary-resultset-payload-peer-access=false

--hashtable-cache-total-bytes=34359738368
--max-cacheable-hashtable-size-bytes=17179869184
--gpu-cubin-cache-path=/var/cache/heavydb/cubin
--gpu-cubin-cache-max-size-in-bytes=4294967296
--cuda-jit-max-parallel-threads=0
--bigint-count=true
--executor-max-available-resource-use-ratio=1.0
--watchdog-max-projected-rows-per-device=512000000
--num-reader-threads=32
```

## Defaults and the baseline path

Use explicit `=true` and `=false` values in deployment configuration. This avoids
ambiguity across command-line and config-file parsers.

Examples in this guide use command-line form, such as
`--enable-result-reduction-pipeline=true`. In `heavyai.conf`, omit the leading dashes:

```text
enable-result-reduction-pipeline = true
```

With the opt-in boolean feature flags at their defaults:

- Experimental Calcite rewrites are off.
- Unenforced table constraints are not exposed as optimizer proofs.
- Bitmap and ranked bitmap hash joins are off.
- The GPU result-reduction pipeline and partitioned baseline reducer are off.
- Lazy string dictionary hash recovery and its scan/view fast paths are off.
- GPU input prefetch and CPU-buffer bypass are off.
- Compressed input overlap and compressed peer exchange are off.
- New native writes are uncompressed.
- Advisory manifests and sidecar-only metadata creation are off.
- Persistent cubin caching is off because its path is empty.
- CUDA driver JIT split compilation is off because its thread limit defaults to `1`.
- Temporary ResultSet kernel peer access is off.
- CPU retry and query-step CPU retry remain on.

Two qualifications matter:

1. A new binary still reads compressed or sidecar-only data that was previously created.
   Disabling a writer flag cannot make existing upgraded data unreadable.
2. `--peer-copy-mode` defaults to `direct`, and the existing CUDA jump-buffer mechanism
   defaults to 128 MiB per GPU. These are transport settings, not disabled booleans.

## Recommended evaluation sequence

Do not enable every switch at once. Use the same binary and the same data for each stage,
and retain the complete results from stage 0 as the reference.

### Stage 0: upgrade with feature selection disabled

Start the new binary with all documented opt-in features disabled and both CPU retry paths
enabled. Run application smoke tests, CPU queries, one-GPU queries, multi-GPU queries,
cancellation, and any rendering or geospatial workflows used by the deployment.

This stage does not change the native table format.

### Stage 1: non-persistent query acceleration

Start with:

```text
--enable-experimental-query-rewrites=true
--enable-ranked-bitmap-hashjoin=true
--enable-bitmap-hashjoin=false
--enable-result-reduction-pipeline=true
--enable-partitioned-baseline-gpu-reduction=false
--trust-unenforced-table-constraints=false
--allow-cpu-retry=true
--allow-query-step-cpu-retry=true
```

These settings do not rewrite stored table payloads, so they can be compared and rolled
back without a data conversion. To roll back, restart with experimental rewrites, both
bitmap hash-join flags, and the result-reduction pipeline set to false.

Keep `--trust-unenforced-table-constraints=false` initially. The planner can still use
nullability and structural proofs, and many accelerated query shapes remain available.

This first stage is a correctness-first rollout, not the strict reference profile above.
It deliberately leaves exact bitmap membership disabled and keeps both retry paths
enabled. On the reference SF=1000 workload, one eligible semi-join otherwise requested
2,967,662,096 entries, which exceeds the legacy 2^31-entry GPU hash-table limit.
With retries enabled that becomes a visible fallback; with strict GPU execution it is an
error. After stage 1 is reference-clean, evaluate exact membership separately:

```text
--enable-bitmap-hashjoin=true
--enable-ranked-bitmap-hashjoin=true
--allow-cpu-retry=true
--allow-query-step-cpu-retry=true
```

Only disable retry for a later strict GPU coverage gate after the exact and ranked layouts
have passed repeated result, capacity, and fallback checks on the target data.

### Stage 2: restart and hydration acceleration

After the stage 1 results match the reference, evaluate:

```text
--enable-file-mgr-manifests=true
--enable-gpu-input-prefetch=true
--enable-gpu-input-batched-prefetch=true
--gpu-input-prefetch-workers=32
--enable-gpu-input-cpu-buffer-bypass=true
--gpu-input-cpu-buffer-bypass-mode=staged
--gpu-input-cpu-buffer-bypass-staging-buffer-bytes=67108864
--gpu-input-cpu-buffer-bypass-reader-threads=16
--gpu-input-compressed-batch-max-bytes=2147483648
--enable-gpu-input-compressed-pipeline=true
--enable-gpu-input-compressed-peer-exchange=true
--enable-deferred-lazy-fetch=true
--stringdict-parallel-sort=true
--enable-lazy-string-dictionary-hash-recovery=true
--gpu-cubin-cache-path=/var/cache/heavydb/cubin
--gpu-cubin-cache-max-size-in-bytes=4294967296
--cuda-jit-max-parallel-threads=0
```

The `staged` bypass is the conservative first configuration. Compare `mmap` only after
validating the staged path; its performance depends more directly on page-cache and
filesystem behavior.

The compressed pipeline requires an nvCOMP-enabled build and overlaps one compressed
payload transfer with decode of the preceding payload. Compressed peer exchange shares a
single storage read among concurrent GPU consumers and falls back when peer access or the
batch shape is ineligible. Evaluate both independently if the workload is uncompressed or
does not repeatedly hydrate the same native chunk across GPUs.

Set `CUDA_CACHE_PATH` to a local service-owned directory in the process environment; it
controls NVIDIA's PTX compute cache and is separate from HeavyDB's cubin cache. Use
`--cuda-jit-max-parallel-threads=0` only with a CUDA 13 or newer build and only when the
host can tolerate JIT using all CPUs. A positive value caps the split-compilation thread
count; `1` disables split compilation.

This stage creates advisory manifest and cubin-cache files, but it does not change native
column payload encoding or remove physical metadata pages. To return to eager string
dictionary recovery, restart with `--enable-lazy-string-dictionary-hash-recovery=false`;
mapped dictionary payload files are unchanged.

### Stage 3: native compression on a disposable or backed-up table

Enable Snappy for new native chunks:

```text
--enable-native-storage-compression=true
--native-storage-compression-codec=snappy
--native-storage-compression-frame-size=65536
```

Newly created native tables are the simplest evaluation target. Existing chunks are not
rewritten merely because the server flag is enabled. Convert an existing test table only
after taking a storage snapshot:

```sql
OPTIMIZE TABLE test_table WITH (STORAGE_COMPRESSION='SNAPPY');
```

Measure disk bytes and cold/lukewarm execution before and after. Hot execution is a guard
against regression, not the expected source of the compression benefit.

Keep Snappy as the first compatibility profile. GDeflate requires nvCOMP CPU codec
support, sidecar-only metadata, and zero rollback epochs; Bitcomp and adaptive modes also
require matching nvCOMP support. Those codecs are separate storage-format experiments,
not drop-in additions to the conservative Snappy rollout.

### Stage 4: trusted key metadata, only after data validation

Enable:

```text
--trust-unenforced-table-constraints=true
```

only after every declared key and reference has been checked against the real data and a
process exists to keep future writes valid. If production data cannot be checked in
place, leave this option false. A synthetic copy can test the optimizer, but it cannot
prove that production data satisfies an unenforced constraint.

### Stage 5: advanced experimental options

Sidecar-only metadata, direct temporary ResultSet peer reads, host-mapped aggregate
payloads, selected-dense payload fetch, and partitioned baseline reduction should be
separate experiments. They are not part of the conservative initial profile.

Fragment-size tuning is also a separate physical-design experiment. It requires a table
copy with a different layout rather than a server restart; use the
[fragment-size tuning procedure](#fragment-size-tuning-with-gpu-retained-intermediates).

## Planner and constraint behavior

### Declarative constraints

HeavyDB accepts table-level constraint syntax, including:

```sql
CREATE TABLE parent (
  id BIGINT NOT NULL,
  code INTEGER NOT NULL,
  CONSTRAINT parent_pk PRIMARY KEY (id),
  CONSTRAINT parent_code_uq UNIQUE (code)
);

CREATE TABLE child (
  id BIGINT NOT NULL,
  parent_id BIGINT,
  CONSTRAINT child_parent_fk
    FOREIGN KEY (parent_id) REFERENCES parent (id)
);

ALTER TABLE child DROP CONSTRAINT child_parent_fk;

ALTER TABLE child ADD CONSTRAINT child_parent_fk
  FOREIGN KEY (parent_id) REFERENCES parent (id);
```

HeavyDB validates the referenced tables, columns, types, and presence of a declared
referenced unique key at DDL time. It also blocks schema operations that would leave
recorded dependencies dangling.

It does **not** reject inserts or updates that violate these declarations. Constraints
are stored with `enforced=false`. When trust is disabled, Calcite does not use them as
cardinality or join-elimination proofs. When trust is enabled, a false declaration can
produce a wrong query result, not merely a slower query.

Calcite only exposes a primary or unique key as a key statistic when all participating
columns are also `NOT NULL`.

### Constraint validation queries

Adapt these templates for every declared key. A valid result is no rows.

```sql
-- Nulls in a key intended to be non-null.
SELECT key_col
FROM parent
WHERE key_col IS NULL
LIMIT 1;

-- Duplicate scalar key.
SELECT key_col, COUNT(*) AS copies
FROM parent
GROUP BY key_col
HAVING COUNT(*) > 1
LIMIT 1;

-- Duplicate composite key.
SELECT key_a, key_b, COUNT(*) AS copies
FROM parent
GROUP BY key_a, key_b
HAVING COUNT(*) > 1
LIMIT 1;

-- Orphan scalar foreign key. A null child key is exempt under MATCH SIMPLE semantics.
SELECT c.parent_id
FROM child c
LEFT JOIN parent p ON p.id = c.parent_id
WHERE c.parent_id IS NOT NULL AND p.id IS NULL
LIMIT 1;
```

For a composite foreign key using ordinary `MATCH SIMPLE` semantics, any null child-key
component exempts that row from requiring a parent. Check for orphans only when every
child-key component is non-null, and compare every component in the join:

```sql
SELECT c.parent_a, c.parent_b
FROM child c
LEFT JOIN parent p
  ON p.key_a = c.parent_a AND p.key_b = c.parent_b
WHERE c.parent_a IS NOT NULL
  AND c.parent_b IS NOT NULL
  AND p.key_a IS NULL
LIMIT 1;
```

Re-run these checks after bulk loads and before enabling trust on a restored database.

### Experimental rule families

`--enable-experimental-query-rewrites=true` activates a coordinated rule sequence. The
main externally relevant transformations are:

| Rule family | Transformation | Expected performance effect |
|---|---|---|
| Join normalization | Splits filters, normalizes connected joins, and decomposes supported outer `MultiJoin` forms | Gives later rules a provable and executable join graph |
| Semi/anti joins | Converts safe `NOT IN`, marker-left-join, and existence shapes to semi or anti joins | Avoids materializing unused right-hand-side payloads |
| Redundant keysets | Removes a semi join only when a complete trusted FK proof makes it redundant | Eliminates unnecessary scans and hash builds |
| Join-tree keyset reduction | Builds and applies small keysets before larger joins | Reduces rows entering expensive joins and shuffles |
| Aggregate/join reduction | Pre-aggregates facts and preserves only keys and required aggregate state | Reduces intermediate cardinality and payload width |
| Key-preserving aggregates | Moves or removes aggregates only when grouping-key preservation is proven | Avoids repeated aggregation across joins |
| Filtered large-table joins | Moves selective filters and eligible build-side reduction earlier | Reads or carries less data through later operators |
| Count and distinct | Rewrites supported count-distribution and single-count-distinct shapes into grouped reductions | Makes common distinct/count plans GPU executable |
| Different-value statistics | Replaces repeated existence scans with paired min/max/count statistics where null semantics are proven | Collapses multiple scans and anti joins into one grouped fact pass |
| Payload deferral | Carries keys through selective work and joins wider payload columns later | Reduces temporary ResultSet size and transfer volume |
| Extrema and Top-N | Converts supported scalar min/max join shapes to window or Top-N forms | Avoids a large self-join or repeated scalar subquery work |
| Outer-join strength | Converts an outer join to a stronger join only when null rejection or aggregate behavior proves equivalence | Enables further reduction and join reordering |

Every rule has shape, type, nullability, and expression guards. Unsupported shapes remain
on the pre-existing plan. Because these are optimizer transformations, however, the
correct validation standard is result equivalence, not merely successful execution.

## Hash-join options

### Exact bitmap membership

`--enable-bitmap-hashjoin=true` enables compact exact membership tables for eligible
semi and anti joins. These tables answer whether a key exists; they do not carry a
general join payload.

This can reduce memory and probe cost for bounded integer-like key domains. Sparse or
large domains that fail eligibility checks fall back to another hash-table layout.

### Ranked bitmap joins

`--enable-ranked-bitmap-hashjoin=true` enables a bitmap plus rank index for eligible GPU
inner joins. The rank maps a present key to a compact payload position. The implementation
also supports filtered build sides and payload-free unique probes where the plan proves
they are valid.

Ranked bitmap selection is restricted to real GPU query builds with a usable build-side
expression graph; CPU cache tests and synthetic perfect/baseline hash-table paths retain
their existing implementations.

The structure is not an unlimited replacement for all hash tables. Domain range, entry
width, slab size, and allocation checks still apply. On ineligible shapes, HeavyDB uses
perfect or baseline hashing, retries a query step on CPU, retries the query on CPU, or
reports an error according to the retry configuration.

Experimental rewrites can produce sparse intermediate keysets for which ranked bitmap
joins are the preferred compact GPU layout. HeavyDB logs a startup warning when rewrites
are enabled, ranked bitmap is disabled, and both CPU retry paths are also disabled.

## Device-resident result reduction

`--enable-result-reduction-pipeline=true` enables a group of execution changes rather
than one algorithm:

- Eligible intermediate ResultSets retain device-resident storage.
- Downstream GPU operators can reuse device columns without an unconditional
  device-to-host (D2H) and host-to-device (H2D) round trip.
- Multi-fragment and multi-GPU aggregate results can be reduced asynchronously.
- Eligible non-grouped integral and decimal `SUM` aggregates use an exact, null-aware
  shared-memory reduction instead of emitting one host-reduced partial per GPU thread.
  Floating-point sums and `SUM_IF` retain their established paths.
- Count-distinct, perfect-hash, baseline-hash, keyless aggregate, window, sort, and
  materialization boundaries have explicit GPU/CPU handoff paths.
- CPU consumers still receive materialized host-visible results at the boundary where
  they require them.
- Unsupported or unsafe shapes use the established materialization and reduction path.

On one GPU, a single already-compatible ResultSet can be elided at the boundary. Multiple
fragment ResultSets still require a logical cross-fragment reduction even though no
cross-device copy is involved.

For an eligible GPU group-by over a temporary relation with a finite, structurally proven
unique-key bound, HeavyDB normally uses that bound directly. If the complete bound exceeds
the maximum GPU slab, the reduction pipeline can first try the largest slab-safe group
table with reserved headroom. A capacity miss raises the ordinary cardinality-retry signal
and falls through to the existing HyperLogLog (HLL) estimate and retry path. The
speculative attempt is therefore a performance choice, not a correctness assumption; it
does not run for CPU execution, validation-only execution, or explain-only execution.

The pipeline can increase peak GPU memory because intermediates live longer on device.
Keep CPU retry enabled during rollout and monitor GPU out-of-memory (OOM) and fallback
logs.

`--enable-partitioned-baseline-gpu-reduction=true` is a narrower optional path for
multi-ResultSet baseline-hash reductions. It partitions reduction work across GPU streams.
It can help a large or skewed baseline result, but partition setup can cost more than it
saves for small results. Leave it false unless a comparative profile shows a repeatable
benefit.

`--enable-gpu-result-reduction-pipeline` is a deprecated alias for
`--enable-result-reduction-pipeline`. Do not set both names.

## Fragment-size tuning with GPU-retained intermediates

`FRAGMENT_SIZE` sets the maximum number of rows in a table fragment. A fragment is a
horizontal group of rows whose column chunks are scheduled and processed together. It is
a table-layout property, not a server startup option:

```sql
CREATE TABLE fact_table (
  fact_key BIGINT NOT NULL,
  measure DOUBLE
) WITH (FRAGMENT_SIZE=16000000);
```

Changing this value for an existing dataset requires creating and loading a table with
the desired layout; `OPTIMIZE TABLE` does not repartition a table into a different
fragment size.

Fragment-size tradeoffs differ between host-reduced and GPU-retained execution:

| Layout choice | Potential benefit | Potential cost |
|---|---|---|
| Larger fragments | Fewer partial ResultSets, transfers, kernel launches, and per-fragment scheduling operations | Larger per-fragment intermediates, longer-lived compacted GPU buffers, greater load imbalance, and higher OOM risk when a fragment is skewed |
| Smaller fragments | Smaller individual allocations and more opportunities to distribute rows, memory pressure, and downstream processing across GPUs | More partial ResultSets, metadata activity, scheduling, kernel launches, and reduction or exchange fan-in |

On the conventional path, partial ResultSets commonly cross from device to host for CPU
reduction. Larger fragments can perform well there because they reduce the number of
device-to-host transfers and CPU partials. With the result-reduction pipeline enabled,
eligible partials and compacted payloads remain on the GPU until downstream consumers
finish with them. Fragment size therefore also controls peak device memory, cross-GPU
work distribution, and the shape of later exchanges and reductions.

In the reference SF=1000 evaluation, a fragment size of approximately 16 million rows
reduced skew and distributed work and memory pressure more evenly across eight GPUs.
Increasing the fragment size to 375 million rows caused Q21 to encounter GPU OOMs
consistently. These values are workload observations, not universal thresholds. Relevant
variables include row width, filter selectivity, group cardinality, data ordering,
sharding, GPU count and memory, and the number of simultaneous downstream intermediates.

Evaluate fragment size as a physical-design parameter:

1. Create representative table copies at several candidate sizes while keeping row
   ordering, sharding, schema, data, queries, and server flags constant.
2. Compare repeated hot measurements and at least one lukewarm restart measurement.
   Smaller fragments can improve GPU balance while increasing metadata and hydration
   overhead.
3. Record peak memory and utilization per GPU, CPU retry and OOM counts, query variance,
   and the number and size of partial ResultSets. A lower average that introduces a long
   tail or retries is not a stable improvement.
4. Test the largest and most skew-sensitive aggregate and join plans, not only scan-heavy
   queries. Validate result equivalence at every candidate size.
5. Treat 16 million rows as a starting point only for workloads similar to the reference
   configuration. Select the production value from the target hardware and data.

## GPU input hydration

### CPU prefetch

`--enable-gpu-input-cpu-prefetch=true` fetches eligible input chunks into the CPU buffer
pool before foreground GPU materialization. It is active only when GPU prefetch is off.
It can hide serialized FileMgr reads but retains permanent CPU buffer-pool residency.

### GPU prefetch

`--enable-gpu-input-prefetch=true` starts bounded background fetches for known,
fixed-width input chunks directly toward the target GPU. Chunk owners remain pinned
through execution. Varlen and unsupported inputs use the established foreground path.

`--gpu-input-prefetch-workers` bounds scalar prefetch workers per query fetch task. More
workers are not always better: HeavyDB already fetches across devices and columns, and
nested reader fanout can increase contention.

### Batched prefetch

`--enable-gpu-input-batched-prefetch=true` is meaningful only with GPU prefetch. It groups
eligible requests to reduce FileMgr, mmap, H2D, and decompression setup overhead.

The fast batch path requires compatible inputs, such as a common compressed codec or a
supported uncompressed mmap batch. Mixed or partial batches fall back to scalar fetches.

### Compressed transfer and decode

`--gpu-input-compressed-batch-max-bytes` bounds compressed payload bytes submitted in one
nvCOMP batch. The 2 GiB default is a payload ceiling, not a guaranteed allocation; larger
requests split at chunk boundaries and allocation failures retry with smaller batches.
A value of `0` removes the configured payload ceiling.

`--enable-gpu-input-compressed-pipeline=true` splits the configured workspace budget
across two batches so HeavyDB can overlap one H2D compressed-payload transfer with decode
of the preceding payload. It is useful only for eligible compressed native chunks in an
nvCOMP-enabled build.

`--enable-gpu-input-compressed-peer-exchange=true` lets concurrent GPU consumers share
one compressed storage read. HeavyDB copies the compressed representation directly to an
eligible peer GPU and decodes into that GPU's ordinary input buffer. Unsupported codecs,
topologies, or allocations use the normal per-device fetch path.

### CPU-buffer bypass

`--enable-gpu-input-cpu-buffer-bypass=true` applies on a native FileMgr GPU cache miss. It
avoids making every fetched chunk a permanent resident of HeavyDB's CPU buffer pool.

- `staged` reads through bounded pinned host staging storage before H2D transfer.
- `mmap` maps page-cache-backed native files and copies their page payloads toward the
  GPU without first installing a permanent HeavyDB CPU-buffer copy.

This is not GPUDirect Storage. The operating-system page cache remains part of the path.
Foreign tables, varlen data, unsupported buffers, and failed fast paths retain their
existing CPU materialization behavior.

`--gpu-input-cpu-buffer-bypass-staging-buffer-bytes=0` disables the bypass even if its
boolean flag is true.

`--enable-deferred-lazy-fetch=true` postpones fixed-width lazy table-column hydration
until a ResultSet consumer asks for the column. It reduces unnecessary input and payload
work when planner transformations carry keys through selective stages before wider
columns are needed. Unsupported shapes use ordinary lazy fetch.

### Advanced experimental payload-fetch modes

`--enable-gpu-aggregate-payload-host-mapping` and
`--enable-gpu-selected-dense-aggregate-payload-fetch` target selective, single-table,
fixed-width aggregate payloads. They are deliberately narrow experiments.

- Host mapping leaves selected payload columns in mapped host memory for GPU reads.
- Selected-dense fetch builds compact payload vectors for qualifying rows.
- Grouped, join, distinct, varlen, and unsupported aggregate shapes do not use them.
- Host mapping takes precedence over selected-dense selection when both are enabled.

On the reference workload, CPU-side selection and compaction cost more than a contiguous
column transfer. Keep both false for a general deployment unless workload-specific
profiling shows a repeatable benefit.

## Lazy string dictionary recovery and scan fast paths

`--enable-lazy-string-dictionary-hash-recovery=true` maps a recovered persistent string
dictionary and defers rebuilding its string-to-ID hash until an operation actually needs
the hash. The mapped payload and offsets remain the source of truth. If a request is not
eligible for a bounded scan, HeavyDB calls the ordinary hash-recovery path before
continuing.

While the flag is enabled, related dictionary operations can stay on storage-backed
`string_view` values:

- Bulk lookups of at most 1,024 strings can scan an unrecovered dictionary instead of
  forcing a complete hash rebuild.
- Eligible self-dictionary string-operation translation builds candidates without first
  recovering the hash. A one-operation `SUBSTRING` can return a view into the locked
  source payload; other operations use the established copied-string evaluator.
- A case-sensitive LIKE pattern containing only literals and `%` uses an allocation-free
  matcher. Escapes, `_`, bracket patterns, ILIKE, and simple substring matching use the
  canonical matchers over the same storage-backed view.
- A transformed self-dictionary candidate can skip a persisted-payload scan only when no
  persisted source string has the candidate's length. Equal strings must have equal
  lengths, so this is an exact rejection proof rather than a probabilistic filter.

The optimized paths hold the dictionary read lock while views reference mapped payload.
Candidate limits, transient-ID limits, generation bounds, and canonical fallbacks remain
enforced. With the flag false, recovery and scan evaluation use the pre-existing eager
hash and copied-string paths.

## Native storage compression

### Scope

Compression applies to HeavyDB native FileMgr chunk payloads. It does not turn Parquet
foreign-table scans into a direct Parquet-to-GPU path, and it does not use Parquet's
compression metadata.

Compression is process-wide for newly initialized native chunks. It is not a per-table
`CREATE TABLE` property. A table can contain a mix of compressed and uncompressed chunks,
and existing compressed chunks retain their recorded codec even if the startup writer
flag is later disabled.

The writer compresses a chunk only when:

- compression is enabled,
- the codec and frame size are valid,
- compression metadata fits, and
- the compressed payload is smaller than the logical payload.

Otherwise it writes the ordinary uncompressed payload.

### Codec choice

The default configured codec is Snappy with 64 KiB uncompressed frames.

- **Snappy** is the conservative GPU hydration codec. When HeavyDB is built with nvCOMP
  and the fetch is eligible, compressed bytes can be transferred and decoded into final
  GPU input buffers. CUDA OOM in this optional path falls back to a smaller, scalar, or
  CPU decode path.
- **LZ4** saves disk and transfer bytes, but this implementation disables raw-LZ4 nvCOMP
  decode because the stored raw framing is not treated as a proven-compatible nvCOMP
  bitstream. LZ4 therefore uses CPU decompression.
- **GDeflate** uses nvCOMP's CPU codec for writes and supports GPU hydration. It requires
  nvCOMP CPU codec support, sidecar-only metadata, and a table with
  `MAX_ROLLBACK_EPOCHS=0`. Compression levels range from 0 through 12; the default level
  `1` favors load throughput.
- **Bitcomp-sparse** and **Bitcomp-default** use type-aware nvCOMP Bitcomp storage when
  that API is available. `bitcomp` is an alias for `bitcomp-sparse`.
- **adaptive** (or `auto`) compresses each FileBuffer with both Snappy and
  Bitcomp-default and keeps the smaller representation. It requires Bitcomp support and
  trades additional load-time CPU work for per-buffer codec selection.
- **none** disables compression for new chunks.

Snappy is a required build dependency for this revision. nvCOMP is optional. A build
without nvCOMP retains CPU Snappy/LZ4 readability but rejects unavailable GDeflate,
Bitcomp, or adaptive writer configurations.

The 64 KiB default is the tested setting. The writer can increase the effective frame
size for a very large chunk when the per-frame size table would not fit in the metadata
page. Larger frames are not automatically faster; a 1 MiB frame saved little disk space
and increased decode time on the reference workload.

### Write and update cost

Initial native writes compress on the CPU. Updating or appending to an already-compressed
FileBuffer currently reconstructs the full logical payload and recompresses it. This is
appropriate for load-once, read-mostly analytical tables, but it can amplify work for
frequently updated data.

### Rewriting an existing table

Supported commands are:

```sql
-- Force a named codec. The server compression feature must be enabled.
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='SNAPPY');
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='LZ4');
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='GDEFLATE');
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='BITCOMP-SPARSE');
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='BITCOMP-DEFAULT');
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='ADAPTIVE');

-- Use the server's current enabled/disabled codec configuration.
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='CURRENT');
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='REWRITE');

-- Use the enabled configured codec. Fails if compression is disabled.
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='TRUE');

-- Rewrite to ordinary uncompressed native payloads.
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='NONE');
```

`STORAGE_COMPRESSION='FALSE'` requests no storage rewrite; it does not decompress a
table. `NATIVE_STORAGE_COMPRESSION` is accepted as an option-name alias, but
`STORAGE_COMPRESSION` is preferred for documentation.

The operation takes table write locks, rewrites all physical shards, checkpoints them,
and clears external, CPU, and GPU caches. It requires table delete privilege. Plan for
temporary extra disk capacity, a cold cache afterward, and an application maintenance
window.

### On-disk compatibility

Compressed chunks carry native-storage compression metadata version 2. An older binary
whose latest encoder metadata version is 1 rejects such a chunk rather than interpreting
compressed bytes as ordinary data. This is intentional fail-fast forward-compatibility
behavior.

- New binary reading old uncompressed table: supported.
- New binary reading compressed table with writer flag off: supported.
- Binary without the optional GDeflate or Bitcomp implementation reading a chunk stored
  with that codec: not supported. Adaptive chunks store the selected concrete codec.
- Old binary reading a table containing a compressed chunk: not supported.
- Downgrade path: while still running the new binary, rewrite every affected table with
  `STORAGE_COMPRESSION='NONE'`, checkpoint, and validate before starting the old binary.

Take a storage snapshot before either direction of conversion.

## FileMgr manifests and sidecar-only metadata

### Advisory manifests

`--enable-file-mgr-manifests=true` writes checksummed lookaside files containing page
headers and current FileBuffer metadata. On reopen, HeavyDB validates file identities,
sizes, epochs, format versions, and checksums before using them.

For an ordinary FileMgr version 2 table, a missing, stale, or invalid advisory manifest
falls back to scanning physical headers and metadata pages. The table data remains the
source of truth. The expected benefit is restart/table-open latency, not hot execution.

Do not indiscriminately delete files named as manifests: the same metadata manifest is
mandatory for a sidecar-only table described below.

### Sidecar-only metadata

`--enable-file-buffer-metadata-sidecar-only=true` permits eligible chunks to remove their
physical metadata page and retain current metadata only in the durable sidecar manifest.
Eligibility requires:

- a native table FileMgr with a table identity,
- `MAX_ROLLBACK_EPOCHS=0`, and
- at least one physical data page.

Rollback-enabled and zero-data-page chunks keep physical metadata pages. An existing
clean eligible chunk can convert at its next checkpoint; this is not limited to a newly
created table.

When any chunk becomes sidecar-only, the table's FileMgr format changes from version 2
to version 3. HeavyDB writes a pending checksummed manifest, syncs data and epoch state,
and then publishes the manifest. Reopen can recover a valid pending manifest. Missing,
corrupt, mismatched, or incomplete required metadata fails table open rather than
silently inventing metadata.

Once a table is version 3:

- the new binary loads its required manifest even if both manifest creation flags are
  false,
- disabling the sidecar creation flag does not recreate physical pages by itself, and
- an older binary fails with the table-storage forward-compatibility error.

Of the documented options, this has the greatest persistent-format commitment. Leave it
disabled for an initial evaluation.

### Converting a sidecar-only table back

The tested conversion mechanism is:

```sql
ALTER TABLE t SET MAX_ROLLBACK_EPOCHS = 1;
OPTIMIZE TABLE t;
```

The changed rollback setting makes the next metadata checkpoint recreate physical
metadata pages. Once no sidecar-only chunk remains, HeavyDB writes FileMgr version 2.
For a sharded table, verify every physical shard through the logical table operation.

Before a binary downgrade, also rewrite compressed payloads to `NONE`, restart the new
binary once to verify the table, and retain a snapshot until the old binary has opened
and queried it successfully.

### Storage-format inventory and conversion records

This revision does not expose a SQL command that inventories the codec and metadata mode
of every physical chunk. Do not infer the state of an existing catalog from the current
server flags: a logical table can contain both compressed and uncompressed chunks, and a
sharded table has multiple physical FileMgr roots.

Before enabling either persistent-format feature, record:

- every logical table selected for conversion,
- all physical shards covered by that table,
- the requested compression codec and completed `OPTIMIZE TABLE` operation,
- whether sidecar-only conversion was enabled at checkpoint, and
- the snapshot or backup that precedes each transition.

FileMgr version 3 is the on-disk fence for required sidecar metadata. If storage-level
inspection is part of an operational procedure, inspect every physical table/shard root
and treat `filemgr_version` as read-only evidence; do not edit it or use it as a substitute
for opening and querying the table with HeavyDB.

When the compression inventory is uncertain before a downgrade, conservatively rewrite
every candidate logical table with `STORAGE_COMPRESSION='NONE'` under the new binary. When
sidecar-only inventory is uncertain, run the documented rollback-epoch and checkpoint
conversion for every candidate table. Reopen and validate all physical shards through the
logical table before starting the old binary.

## GPU transfer paths

### Jump buffers

Jump buffers are pinned host staging buffers used for sufficiently large H2D and D2H
copies. CUDA defaults are:

- `--jump-buffer-size=134217728` (128 MiB per slot per GPU)
- `--jump-buffer-parallel-copy-threads=4`
- `--jump-buffer-slots-per-device=1`
- `--jump-buffer-min-h2d-transfer-threshold=33554432` (32 MiB)
- `--jump-buffer-min-d2h-transfer-threshold=67108864` (64 MiB)
- `--enable-lazy-jump-buffer-allocation=false`
- `--enable-background-jump-buffer-allocation=false`

Pinned memory consumption is approximately:

```text
jump-buffer-size * slots-per-device * GPU count
```

before allocator overhead. A benchmark setting of 1 GiB with one slot on eight GPUs
therefore reserves roughly 8 GiB of pinned host memory. Do not copy that setting into a
shared production host without accounting for it.

`--jump-buffer-size=0` disables jump buffers. Lazy allocation moves the allocation cost
to first use. Background allocation only has an effect when lazy allocation is enabled;
it warms slots after startup. Eager allocation gives the most predictable first query but
adds startup and shutdown work.

### Cross-GPU copy mode

`--peer-copy-mode` controls copies, not remote pointer dereferences:

- `direct`: use CUDA peer copies when peer access is available, with host fallback.
- `staged`: copy in chunks through a per-GPU-pair device staging allocation.
- `host`: force a host-staged D2H/H2D path.

`--peer-copy-staging-buffer-size` is used only by `staged`. Zero forces host fallback.
Direct peer copy is normally the first choice on NVLink/NVSwitch systems, but topology
and measured bandwidth should decide.

### Direct temporary ResultSet peer reads

The two temporary ResultSet flags allow a generated kernel to dereference another GPU's
temporary allocation. This is a stronger requirement than CUDA's ability to perform a
peer copy.

- `--enable-temporary-resultset-peer-access` covers eligible temporary join columns.
- `--enable-temporary-resultset-payload-peer-access` covers eligible segmented payload
  columns while leaving join-key hash-table builds on the copy path.

Direct access requires all of the following before HeavyDB exposes a remote pointer:

- CUDA reports peer-to-peer (P2P) access support,
- an active destination-kernel read of a source-GPU validation pattern succeeds for that
  ordered device pair,
- CUDA reports native peer atomics, used as the performance signal that the link is
  suitable for repeated fine-grained kernel reads,
- the source is a tracked CUDA virtual memory management (VMM) allocation,
- read access can be granted for the consuming device, and
- producer readiness is ordered before the consumer stream reads it.

When either temporary ResultSet peer-access option is enabled, HeavyDB validates every
ordered GPU pair before query execution and logs the validation and direct-read policy
matrices. The results are cached, so eligible hot-path checks are atomic reads rather
than repeated probes. If any condition fails, HeavyDB copies the fragment instead. The
active probe is necessary because PCIe topology and CUDA capability attributes alone
cannot prove that remote kernel loads are valid under the host's current IOMMU
configuration. Correct remote reads are not necessarily fast remote reads: PCIe pairs
that pass validation still use the copy-local path because remote kernel loads can be
far slower than copying a fragment once.

Keep both flags false unless the exact topology has passed repeated result-equivalence
and stress testing. They are not required for the validated conservative profile.

## Persistent GPU cubin cache

### What it caches

HeavyDB JIT-compiles eligible GPU query code to a cubin. The in-memory code cache is lost
at server shutdown. The optional persistent cache stores those cubin bytes so a later
server process can skip PTX-to-cubin compilation for an identical key.

It caches compiled query kernels, not table data, query results, planner output, or a
copy of GPU memory.

### Path and service accounts

The engine default is an empty path, which disables persistent cubin caching. The engine
does not infer `$HOME` and does not require an interactive user account.

Configure an explicit directory writable by the HeavyDB service identity, for example:

```text
--gpu-cubin-cache-path=/var/cache/heavydb/cubin
--gpu-cubin-cache-max-size-in-bytes=1073741824
```

Example host preparation, adjusted for the deployment's service user and group:

```bash
sudo install -d -o heavydb -g heavydb -m 0750 /var/cache/heavydb/cubin
```

Every service runs under an operating-system identity even when it has no login shell or
home directory. In a container, mount a writable cache volume at the configured path. If
the filesystem is read-only or no persistent volume is desired, leave the path empty.

Treat cubins as executable code artifacts. The cache directory must not be writable by
untrusted users or tenants. Prefer one directory per HeavyDB instance unless instances
share the same software and trust boundary.

### Invalidation and lifetime

The cache key includes:

- a cache format/version salt,
- an explicit GPU code-generation ABI version,
- LLVM version,
- CUDA driver version,
- GPU compute capability and target streaming multiprocessor (SM) architecture,
- hashes of `cuda_mapd_rt.fatbin` and `CudaTableFunctions.a`,
- hashes of available runtime, H3, and libdevice modules, and
- the serialized query code-cache key.

A new CUDA driver, architecture, linked GPU runtime module, code-generation ABI, or query
kernel produces a different cache generation or filename. An unrelated HeavyDB rebuild
does not invalidate the generation by itself. Instead, the new process regenerates PTX
for a candidate entry and reuses the linked cubin only when the PTX digest matches. Once
the module has loaded successfully on every selected device, that executable is recorded
as validated so later restarts of the same binary can skip PTX generation as well.

GPU user-defined function (UDF) or runtime-UDF modules disable persistent cache use
because their bodies are not yet included in the key. Distinct generated kernel keys can
therefore create many files; the positive max-size setting is what bounds that growth
across a large query population.

Each cache file contains a format marker, payload length, cubin checksum, PTX digest, and
the digest of the last executable that validated it. A malformed, truncated, mismatched,
or unloadable entry is removed and the query is compiled normally. Cache read/write
errors are warnings and fall back to JIT compilation.

Writes use a unique temporary file followed by rename. Temporary cubin files older than
24 hours are removed during pruning. Successful reads refresh file modification time.
When the configured limit is exceeded, the oldest cubins are deleted first.

Important value semantics:

- Empty `--gpu-cubin-cache-path`: cache disabled.
- Positive max size: bounded, oldest-first pruning.
- Max size `0`: pruning disabled, so growth is unbounded. It does **not** disable the
  cache.

It is safe to clear this directory while HeavyDB is stopped. The consequence is extra
compilation latency, not table-data loss. After an upgrade, incompatible generations age
out under the size policy; an operator may also clear the cache during the maintenance
window.

## Complete option reference

The tables below list server startup options. `FRAGMENT_SIZE` is documented separately
because it is a table property. Byte-valued options are expressed in bytes. Defaults are
for a CUDA build at the revision covered by this guide.

### Planning and hash joins

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-experimental-query-rewrites` | `false` | Enables the coordinated Calcite rule set. Set false and restart to use the baseline planner sequence. |
| `--trust-unenforced-table-constraints` | `false` | Lets Calcite treat declarative PK/UNIQUE/FK metadata as true. Keep false until the data has been validated. |
| `--enable-bitmap-hashjoin` | `false` | Enables exact bitmap membership tables for eligible semi/anti joins. |
| `--enable-ranked-bitmap-hashjoin` | `false` | Enables ranked bitmap tables for eligible GPU inner joins. |

### Result execution and reduction

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-result-reduction-pipeline` | `false` | Enables device-resident intermediates and asynchronous GPU reduction. |
| `--enable-gpu-result-reduction-pipeline` | `false` | Deprecated alias of the preceding option. Do not configure both. |
| `--enable-partitioned-baseline-gpu-reduction` | `false` | Partitions eligible baseline-hash GPU reductions. |
| `--allow-cpu-retry` | `true` | Safety path: retry a failed GPU query on CPU. Keep true initially. |
| `--allow-query-step-cpu-retry` | `true` | Safety path: retry eligible individual query steps on CPU. Keep true initially. |

### Input and payload fetch

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-gpu-input-cpu-prefetch` | `false` | Prefetches into the CPU pool; ignored when GPU prefetch is enabled. |
| `--enable-gpu-input-prefetch` | `false` | Prefetches eligible fixed-width chunks toward the target GPU. |
| `--enable-gpu-input-batched-prefetch` | `false` | Batches eligible GPU-prefetch requests; has no useful effect without GPU prefetch. |
| `--gpu-input-prefetch-workers` | `32` | Maximum scalar prefetch workers per query fetch task; must be at least 1. |
| `--enable-gpu-input-cpu-buffer-bypass` | `false` | Avoids permanent HeavyDB CPU buffer-pool residency on eligible native cache misses. |
| `--gpu-input-cpu-buffer-bypass-staging-buffer-bytes` | `67108864` | Per-fetch staging bound; zero disables bypass. |
| `--gpu-input-cpu-buffer-bypass-reader-threads` | `16` | Maximum native reader threads inside a staged fetch; must be at least 1. |
| `--gpu-input-cpu-buffer-bypass-mode` | `staged` | `staged` or experimental `mmap`. |
| `--gpu-input-compressed-batch-max-bytes` | `2147483648` | Compressed bytes per nvCOMP batch. Larger requests split; zero removes the configured payload ceiling. |
| `--enable-gpu-input-compressed-pipeline` | `false` | Overlaps one eligible compressed H2D transfer with decode of the preceding payload. |
| `--enable-gpu-input-compressed-peer-exchange` | `false` | Shares one eligible compressed storage read across peer-accessible GPU consumers. |
| `--enable-deferred-lazy-fetch` | `false` | Defers eligible fixed-width lazy table columns until a ResultSet consumer requests them. |
| `--enable-gpu-aggregate-payload-host-mapping` | `false` | Experimental path for mapped fixed-width aggregate payloads. |
| `--enable-gpu-selected-dense-aggregate-payload-fetch` | `false` | Experimental path for compact selected aggregate payloads. |

### String dictionaries

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--stringdict-parallel-sort` | `false` | Parallelizes eligible string-dictionary sorted-cache construction. |
| `--enable-lazy-string-dictionary-hash-recovery` | `false` | Defers recovered dictionary hash construction and enables bounded storage scans and guarded string-view transformations. False restores eager recovery and copied-string scan behavior. |

### Native storage

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-native-storage-compression` | `false` | Compresses eligible newly initialized native chunk payloads. False does not decompress existing chunks. |
| `--native-storage-compression-codec` | `snappy` | `none`, `lz4`, `snappy`, `gdeflate`, `bitcomp-sparse`, `bitcomp-default`, or `adaptive`, subject to build and table eligibility. |
| `--native-storage-compression-frame-size` | `65536` | Requested uncompressed frame size; must be valid for the codec. |
| `--native-storage-compression-gdeflate-level` | `1` | GDeflate CPU compression level from 0 through 12. |
| `--enable-file-mgr-manifests` | `false` | Writes/uses advisory page-header and physical-metadata manifests. Required sidecar manifests load regardless. |
| `--enable-file-buffer-metadata-sidecar-only` | `false` | Lets eligible checkpointed chunks remove physical metadata pages and upgrades their table to FileMgr v3. |

### Host/device and peer transfer

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--jump-buffer-size` | `134217728` | Pinned bytes per slot per GPU; zero disables jump buffers. |
| `--jump-buffer-parallel-copy-threads` | `4` | Host copy workers per GPU jump-buffer transfer; minimum 1. |
| `--jump-buffer-slots-per-device` | `1` | Concurrent slots per GPU; multiplies pinned memory; minimum 1. |
| `--enable-lazy-jump-buffer-allocation` | `false` | Allocates pinned slots at first use instead of startup. |
| `--enable-background-jump-buffer-allocation` | `false` | Warms slots in background only when lazy allocation is true. |
| `--jump-buffer-min-h2d-transfer-threshold` | `33554432` | Minimum H2D bytes for jump-buffer use. |
| `--jump-buffer-min-d2h-transfer-threshold` | `67108864` | Minimum D2H bytes for jump-buffer use. |
| `--peer-copy-mode` | `direct` | `direct`, `staged`, or `host`; this selects a copy transport, not remote kernel access. |
| `--peer-copy-staging-buffer-size` | `268435456` | Per-GPU-pair staging chunk in `staged` mode; zero selects host fallback. |
| `--enable-temporary-resultset-peer-access` | `false` | Allows capability-gated direct kernel reads of eligible peer-resident join columns. |
| `--enable-temporary-resultset-payload-peer-access` | `false` | Allows capability-gated direct kernel reads of eligible segmented payloads. |

### Persistent compilation

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--gpu-cubin-cache-path` | empty | Persistent cubin directory. Empty disables the cache. |
| `--gpu-cubin-cache-max-size-in-bytes` | `1073741824` | Oldest-first disk limit. Zero means unlimited growth, not disabled. |
| `--cuda-jit-max-parallel-threads` | `1` | CUDA 13+ driver split-compilation limit. `0` uses all CPUs; `1` disables split compilation. Non-default values are rejected by older CUDA builds. |

## Stored-data and rollback matrix

| Feature | Changes table/catalog storage? | Older binary behavior | Rollback |
|---|---|---|---|
| Planner rewrites | No | Not applicable | Restart with flag false |
| Bitmap/ranked bitmap joins | No | Not applicable | Restart with flag false |
| Result-reduction pipeline | No | Not applicable | Restart with flag false |
| Prefetch/bypass/jump/peer copy | No | Not applicable | Restart with prior settings |
| Cubin cache | Separate disposable files | Older binary ignores unless configured to same path | Stop server and remove cache files |
| Declarative constraints | Yes, catalog `key_metainfo` | Older planner ignores the new proofs and may not protect dependency DDL | Drop constraints before downgrade if old software will mutate schema/data |
| Advisory manifests | Extra checksummed files | Ordinary table data remains authoritative | Disable flag; do not delete if table is sidecar-only |
| Fragment size | Yes, table layout and fragment boundaries | Supported | Create and reload a table with the prior `FRAGMENT_SIZE`; `OPTIMIZE TABLE` does not refragment it |
| Native compression | Per-chunk metadata/payload format | Fails fast on a compressed chunk | `OPTIMIZE ... STORAGE_COMPRESSION='NONE'` under new binary |
| Sidecar-only metadata | FileMgr v3 plus mandatory manifest | Fails fast on table format v3 | Set rollback epochs above zero and checkpoint/optimize under new binary |

Do not alternate old and new binaries against a writable catalog after adding
constraints. The old binary does not understand their optimizer or dependency semantics
and can make changes that invalidate the declarations.

## Correctness validation without a production-data copy

### What generated data can establish

Publicly generated TPC-H data at SF=1 or SF=10 is enough to test:

- server startup and shutdown,
- planner rule reachability,
- one-GPU and multi-GPU execution,
- CPU-only execution and CPU retry,
- null, empty-table, tie, and duplicate adversarial cases that you add deliberately,
- compression write, reopen, rewrite, and downgrade,
- cubin reuse across restart,
- cancellation and timeout cleanup, and
- result equivalence against another SQL engine.

Use generated tables with the same data types, nullability, dictionary encoding, fragment
sizes, sharding, and declared constraints as the intended deployment where possible.
Cardinality and skew affect which physical path is selected, so include synthetic skewed
and multi-fragment cases rather than relying only on tiny uniform tables.

### What generated data cannot establish

Generated data cannot prove that real production rows satisfy an unenforced primary key,
unique key, or foreign key. If read-only validation queries cannot run against production,
keep constraint trust disabled.

Generated data also cannot establish performance for a different topology, storage
device, filesystem, compression ratio, fragment size, or output cardinality. Treat the
reference measurements as an evaluation target, not a capacity plan.

### Differential test procedure

1. Run the same binary with every documented opt-in feature disabled. Save complete typed
   results, row counts, status, and engine execution time.
2. Enable one feature group at a time in the stage order above.
3. Restart between configurations. Re-run cold/lukewarm measurements when evaluating
   storage, cache, or input paths.
4. Compare values and multiplicities, not incidental row order. SQL without `ORDER BY`
   does not guarantee order. Add an outer deterministic sort only in the comparison
   harness, without changing the production query's relational semantics.
5. Compare exact integer, decimal, date, text, and null values. Use a documented tolerance
   only for floating aggregates whose reduction order may differ.
6. Run every query repeatedly. A single passing execution does not expose stream-ordering
   or lifetime races.
7. Test CPU mode, one GPU, and the production GPU count. One-GPU elision and multi-GPU
   transport exercise different code.
8. Force at least one supported CPU-retry case and confirm the final result, cancellation,
   and cleanup behavior.
9. For storage tests, stop and reopen the server after each format transition. Verify the
   table before deleting the pre-change snapshot.
10. If the deployment uses row-level security, views, query hints, or table functions,
    include those exact policy and query combinations. A base-table result comparison
    does not validate the policy-bearing plan.
11. Profile H2D, D2H, and peer-transfer counts and bytes for representative retained-GPU
    plans. Include a GPU-retained intermediate that ultimately reaches a CPU consumer, and
    verify that boundary materialization does not copy the same payload more than once.
    Use a system profiler or temporary, clearly isolated instrumentation; remove hot-path
    diagnostics after the investigation.
12. Treat partitioned baseline reduction as an independent experiment. The conservative
    reference profile leaves it disabled, so enabling it requires its own repeated
    one-GPU, multi-GPU, CPU-boundary, cancellation, and memory-pressure checks.

Suggested adversarial SQL coverage includes:

- empty build and probe sides,
- all-null and mixed-null keysets,
- duplicate keys despite a deliberately untrusted declaration,
- scalar and composite joins,
- filtered and unfiltered right-hand-side relations,
- zero-row, one-row, and high-cardinality groups,
- exact and approximate count-distinct,
- window partitions with ties,
- ordered and unordered final results,
- large fragment sizes and skewed fragment cardinalities, and
- query cancellation during input fetch, hash build, kernel execution, and reduction.

### Reference validation coverage

An earlier reference revision completed the full 87-test CTest inventory on 2026-07-22,
with each test invoked through its registered harness: 87 of 87 passed with zero failures
in 9,905.10 seconds (2 hours, 45 minutes, 5.1 seconds). The run covered all three 754-case
`ExecuteTest` configurations, CPU fallback, cancellation, watchdog cleanup,
count-distinct, GPU windows, hash-table caches, compressed native input, FileMgr recovery,
geospatial lazy fetch, Arrow IPC against its managed live server, and multi-instance
execution. That full inventory predates the final JIT, integral-sum, and Arrow
shared-memory commits; it must be rerun after the final rebase. Focused tests and the
benchmark guards do not replace it.

The reference CTest pass ran serially with `-j1`. Current test registrations do not
declare CTest resource locks, while many harnesses share `Tests/tmp`, Calcite or server
ports, catalog state, or GPUs. Do not apply blanket test parallelism until those resources
have unique per-test identities or explicit CTest locks. Parallelize only a reviewed set
of isolated tests.

The final SF=1000 8-GPU TPC-H guard ran all 22 queries with one checked warmup and 20
checked measurements per query. All 462 executions matched their references. The final
SSB SF=1000 guard ran all 13 queries with one checked warmup and 20 checked measurements;
all 273 executions matched. Focused validation also covered 21 pipelined GPU aggregate
tests, StringDictionary and Arrow IPC integration, stable System V shared-memory segment
counts, CUDA 13 split-JIT limits `1`, `2`, and `0`, and focused native storage,
ResultSet, group-by, join, retry, and multi-GPU execution cases.

The SF=100 CPU-only sweep found two existing capacity guards rather than silent wrong
results: Q13 hit the legacy 128-million projected-row watchdog and passed after that cap
was lifted; Q18 hit the same pre-execution memory guard with experimental rewrites both
enabled and disabled. Rendering-enabled code and `QueryRendererTest` built, but
end-to-end Vulkan rendering encountered a host-specific device-loss timeout. A rendering
deployment must therefore run its own end-to-end render validation before rollout.

This coverage does not replace validation on the target build, driver, GPU topology,
schema, and workload.

## Monitoring and failure interpretation

During rollout, retain startup and query logs and watch for:

- CPU query-step or whole-query retries,
- GPU OOM and code-cache eviction,
- unsupported ranked-bitmap or reduction shapes,
- cubin checksum/load warnings followed by recompilation,
- manifest validation fallback on ordinary tables,
- required sidecar manifest errors on FileMgr v3 tables,
- native GPU decompression, compressed-batch split, or compressed-peer fallback,
- peer-access capability rejection and copy fallback,
- open-file exhaustion during native scans or imports,
- cancellation that does not release GPU/host allocations, and
- growing pinned host memory from jump-buffer sizing.

A fallback is not automatically a correctness problem, but it changes performance and can
hide a GPU coverage gap. Track fallback counts alongside latency where the engine exposes
them.

Ordinary INFO logs do not currently expose every event in that list. Startup logs include
the basic peer-capability matrix and configured feature values. Required sidecar failures,
manifest validation problems, and cubin cache failures are visible. Detailed nvCOMP
fallback reasons are primarily available at `VLOG(1)`, while individual native-atomic
peer-access rejections and copy fallbacks are not logged at normal severity. For a
controlled qualification run, use Nsight Systems or temporary targeted instrumentation
to measure transfer counts, bytes, decode paths, and fallback behavior. Do not leave
per-copy or per-fragment diagnostic branches in a production hot path. If continuous
fallback telemetry is required operationally, add bounded counters rather than enabling
verbose per-event logging.

When diagnosing a wrong result, disable features by boundary rather than changing SQL:

1. Disable direct temporary ResultSet peer access.
2. Disable partitioned baseline reduction.
3. Disable the result-reduction pipeline.
4. Disable ranked and exact bitmap joins.
5. Disable experimental planner rewrites.
6. Disable trusted constraints.

Use the first boundary that restores correctness to define a focused reproduction. Do not
keep a query-text exception as the fix; repair the relational proof or execution
invariant.

## Emergency rollback

For a query-only issue with no new storage formats:

1. Keep the same binary.
2. Restart with all new planner, hash, reduction, input, and peer-access booleans false.
3. Keep CPU retry true.
4. Clear only the disposable cubin cache if compilation behavior is suspect.

For a binary downgrade after native compression:

1. Stop writes and take a storage snapshot.
2. Consult the conversion inventory and start the new binary with compression read
   support.
3. Run `OPTIMIZE TABLE ... STORAGE_COMPRESSION='NONE'` for every affected logical table.
   If the inventory is incomplete, rewrite every candidate table rather than assuming the
   current writer flag describes its existing chunks.
4. Reopen and validate those tables with the new binary.
5. Only then start the old binary.

For a binary downgrade after sidecar-only metadata:

1. Complete the compression rollback above if needed.
2. Under the new binary, set `MAX_ROLLBACK_EPOCHS` to at least 1 for every affected table
   and every table whose sidecar state is uncertain.
3. Run `OPTIMIZE TABLE t;` to force metadata recomputation and checkpoint.
4. Reopen and validate under the new binary.
5. Retain the snapshot until the old binary has opened and queried every converted table.

If a required sidecar manifest is missing or corrupt, do not fabricate it or relabel the
`filemgr_version` file. Restore a consistent snapshot or use a tested recovery procedure.
The version fence is what prevents compressed bytes or missing metadata from becoming
silent data corruption.

## Current limitations and follow-up areas

- Declarative PK/UNIQUE/FK constraints are not write-enforced.
- Persistent cubin keys intentionally exclude runtime GPU UDF bodies; such sessions skip
  the persistent cache.
- Snappy, GDeflate, and Bitcomp GPU decode require the matching nvCOMP capabilities; raw
  native LZ4 currently falls back to CPU decode.
- GDeflate requires sidecar-only metadata and zero rollback epochs. Bitcomp and adaptive
  storage are unavailable in builds without nvCOMP Bitcomp support.
- Native compression updates reconstruct and recompress the full FileBuffer.
- The Parquet foreign-table path is not a direct storage-to-GPU path in this revision.
- Sidecar-only metadata is a format upgrade and is not recommended for a first rollout.
- Direct temporary ResultSet peer reads remain topology-sensitive and are disabled in the
  conservative profile.
- Partitioned baseline GPU reduction is not part of the validated reference profile and
  requires an independent boundary-materialization and performance qualification.
- This revision has no SQL-level per-chunk compression or sidecar inventory. Persistent
  format rollouts require an external conversion ledger and conservative rollback when
  that ledger is incomplete.
- Bitmap layouts remain subject to domain and allocation eligibility; there is no promise
  of an unbounded hash table.
- Fragment size affects partial-result count, GPU memory pressure, and work distribution
  under the GPU-retained pipeline. Test with production-like fragment sizes and skew.
- Row-level security and rendering need deployment-specific end-to-end validation; a
  green relational benchmark is not evidence for either application boundary.
- Hot execution, restart hydration, shutdown latency, and client result serialization are
  separate measurements and should remain separate performance objectives.
- Arrow shared-memory query results are local-only and rely on System V shared-memory
  lifecycle cleanup. Use Arrow wire or Thrift across a host boundary.
