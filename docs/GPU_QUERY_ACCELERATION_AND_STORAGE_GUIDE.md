# GPU Query Acceleration and Native Storage Guide

This guide brings together HeavyDB's query, transfer, and storage options to help you
choose experiments for your workload. The options can keep more joins, intermediate
results, and reductions on GPUs, including plans that previously fell back to CPU or
reached memory and hash-table limits. Each feature section describes where an option
can help, the conditions it needs, and ways to compare its behavior with the existing path.

In the recorded 8xH200 reference evaluation, all 22 TPC-H SF=1000 queries matched their
references with both CPU retry paths disabled.
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

Option names, defaults, and selection conditions reflect a source review of revision
`bae68469dab9bc34eca3ac448637dda2a25311e6` on 2026-10-15. Runtime measurements are recorded
separately with the revisions and configurations used for those runs. The
[test coverage section](#existing-automated-coverage-and-its-limits) describes the cases
covered and useful next steps for evaluating other workloads and combinations.

For configuration work, start with [switch dependencies](#how-the-switches-depend-on-each-other),
the [evaluation stages](#recommended-evaluation-sequence), and the
[complete option reference](#complete-option-reference). For qualification, read the
[strict and recovery profiles](#strict-and-fallback-validation-configurations) and
[existing test coverage](#existing-automated-coverage-and-its-limits).

## Status and scope

- The major planner, hash-join, GPU reduction, dictionary-recovery, input, and storage
  features are opt-in.
- With their feature flags disabled, HeavyDB does not select the opt-in planner rules,
  bitmap hash tables, result-reduction pipeline, lazy dictionary scan paths, storage
  compression writer, or direct temporary ResultSet peer reads.
- CPU retry remains enabled by default. Keeping both paths enabled during an initial
  deployment provides recovery while you evaluate the GPU paths.
- `--trust-unenforced-table-constraints=true` lets the optimizer rely on declared
  primary, unique, and foreign keys. HeavyDB records these declarations without enforcing
  them on writes, so enabling trust depends on validating the data and keeping future
  writes consistent with the declarations. Invalid trusted keys can produce wrong results.
- Native compression and sidecar-only metadata change persistent table storage.
  Compression can be reversed with
  `OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='NONE');`.
  Sidecar-only metadata must be converted back to physical metadata pages before an
  older HeavyDB binary can open the table.
- The persistent cubin cache is disabled unless an explicit directory is configured.
  HeavyDB does not default it to `$HOME`.
- Storage compression does not improve a fully GPU-resident hot query. It targets disk
  footprint, restart hydration, CPU memory pressure, and first-use transfer volume.

### Terms and configuration scope

- **Native storage** is HeavyDB's own table storage, managed by FileMgr. A column's
  **chunk** holds its data for a table fragment. Foreign tables, such as external Parquet
  datasets, use different storage paths.
- A **ResultSet** is the engine's result for a query or an intermediate query step.
  Keeping it device-resident means retaining its data in GPU memory for a later consumer.
- A **payload** is the column data needed to produce a result, beyond the keys used to
  join, group, or filter rows.
- **Hydration** loads stored columns into the CPU or GPU buffer pools. H2D means
  host-to-device transfer; D2H means device-to-host transfer.
- A **baseline-hash aggregate** uses the engine's general hash-table layout for grouping.
  Here, "baseline hash" names an execution layout; it does not mean all features are off.
- **Peer copies** move bytes between GPUs. **Direct peer reads** let a kernel on one GPU
  dereference memory owned by another GPU. They have different capability requirements.
- **nvCOMP** provides optional compression/decompression libraries. CPU codec support
  and GPU decode support are separate build capabilities, each requiring the matching
  nvCOMP libraries in addition to any CUDA requirements.

Options beginning with `--` configure the `heavydb` server process. Examples are option
fragments to add to your normal server launch configuration. These settings apply at
server startup; SQL statements run through a connected SQL client. Test executables
have their own option parsers, described in the
[test coverage section](#existing-automated-coverage-and-its-limits).

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
  an enabled configuration is readable after its creation flag is turned off, provided
  the binary still has the required codec support.
- GPU window, count-distinct, column-layout, and reduction primitives are execution
  capabilities used only by eligible plans. There is no per-function startup flag.
- Ownership, stream-ordering, cache-invalidation, cancellation, and CPU-boundary behavior
  provide correctness guarantees required by the opt-in paths.
- Calcite 1.41 integration and hint propagation are compatibility requirements, not
  performance switches.

To isolate execution behavior, compare the planner and result-pipeline switches
independently. Setting them false does not remove compatibility readers or every GPU
primitive from the binary. A CPU-only server/query run is a separate control.

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
- Include configured jump buffers and staged transfers in the pinned host-memory budget;
  the [jump-buffer section](#jump-buffers) gives the per-GPU sizing calculation.
- Size the process's open-file limit for the native catalog. Large, highly fragmented
  datasets can exceed the common 1,024-file soft limit, causing `Too many open files`
  during scans or imports.
- A low-latency local filesystem is a useful choice for both the NVIDIA compute cache
  (`CUDA_CACHE_PATH`) and HeavyDB cubin cache. A network-mounted home directory can add
  minutes of serialized filesystem latency to startup and JIT cache access.
- On dedicated benchmark hosts, NVIDIA persistence mode can help make startup
  measurements more repeatable. Timing `nvidia-smi` before starting HeavyDB helps
  distinguish driver initialization time from query compilation time.
- Page size and NUMA policy are host-level benchmark variables. The validated GB300
  Grace host used a supported 64 KiB kernel with automatic NUMA balancing disabled;
  the appropriate settings depend on the target platform and workload.

## Interpreting performance measurements

Use distinct labels for distinct cache states:

| State | Server process | OS page cache | HeavyDB CPU/GPU buffers | Main costs measured |
|---|---|---|---|---|
| Cold | New | Dropped | Empty | Storage reads, hydration, just-in-time (JIT) compilation, planning, execution |
| Lukewarm | New | Retained | Empty | Restart, metadata, JIT, DRAM-to-GPU hydration, execution |
| Hot | Existing | Retained | Resident where reused | Planning, execution, reduction, result production |

Recording the process and cache state alongside a measurement makes comparisons easier
to interpret. For a lukewarm run, start immediately after a server restart with the OS
page cache retained and HeavyDB's CPU and GPU buffer pools empty. A sequential first
pass through Q1-Q22 progressively warms tables shared by later queries, giving a
suite-level warmup measurement. Restarting before each query instead measures each
query's independent restart cost.

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
boundaries; reporting each separately keeps the comparison consistent.

Result format is also part of the measured boundary. Thrift is the conservative portable
transport. `arrow-wire` removes text formatting but still transports IPC bytes, while
local-only `arrow-shared-memory` lets the client attach to server-owned Arrow IPC storage.
Reference validation should happen after the timed stream so Python startup and DuckDB
comparison do not inflate query intervals.

### Reference benchmark profiles

The following measurements provide reference points for the listed hardware, datasets,
and configurations. The evaluation stages below offer a way to explore how these options
affect your workload.

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

Both reference results combine several feature groups. To help choose where to begin
with a workload, the main areas of influence are:

- Planner rewrites, ranked bitmap joins, and device-resident reductions primarily affect
  hot execution.
- Cubin caching affects repeated compilation after restart, not already-hot kernels.
- Manifests affect table-open metadata work.
- Prefetch, CPU-buffer bypass, jump buffers, and compression affect hydration.
- Compression normally has no hot-query benefit once the needed columns are already on
  the GPU.

The hot reference used the following explicit settings, including disabled experiments
and non-default capacity choices. To adapt it, choose a service-owned cubin directory
and memory/watchdog budgets suited to the target host. The Snappy setting describes the
writer configuration; the reference catalog also retained pre-existing chunks in other
formats. The split-JIT value `0` requires a CUDA 13+ build; use `1` with older CUDA builds.
For T4 CI runners or other hardware, the [evaluation stages](#recommended-evaluation-sequence)
provide a starting point for checking capacity and execution behavior.

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

Two defaults are useful to keep in mind when setting up a comparison:

1. A new binary still reads compressed or sidecar-only data that was previously created.
   Disabling a writer flag cannot make existing upgraded data unreadable.
2. `--peer-copy-mode` defaults to `direct`, and the existing CUDA jump-buffer mechanism
   defaults to 128 MiB per GPU. These are transport settings, not disabled booleans.

Each example shows the options relevant to that experiment. Other settings in your
configuration file or launch command remain in effect. Saving the complete effective
configuration and checking it against the startup log makes runs easier to reproduce.
For example, disabling the persistent cubin cache means clearing its path setting from
every loaded configuration and launch argument so the path is empty. Setting its size
to zero instead removes the size limit.

## How the switches depend on each other

`--enable-result-reduction-pipeline=true` is a parent switch for the ResultSet execution
family. It permits eligible intermediate results to remain on GPUs, enables asynchronous
result reduction, and changes how later GPU or CPU consumers receive those results.
Some supporting CPU reduction and materialization paths also use this switch.

The other switches retain their separately configured values. Partitioned reduction and
direct temporary ResultSet peer reads need the pipeline plus their own switches; planner
rewrites, bitmap joins, input prefetch, storage compression, and dictionary recovery are
independent choices. Where an example repeats the pipeline flag, it includes that
prerequisite for the particular experiment.

The following table distinguishes required combinations from independent choices. An
enabled feature still needs an eligible query, data layout, build, and device.

| Feature | Required combination or relationship |
|---|---|
| Experimental planner rewrites | Independent of the result pipeline. Enable `--enable-experimental-query-rewrites` to select the new rule sequence and optimizer metadata. |
| Trust in declared keys | For the Calcite proofs described here, both rewrites and `--trust-unenforced-table-constraints` must be true. The data must satisfy the declared constraints. The trust flag does not declare or enforce keys. |
| Exact and ranked bitmap joins | Each has its own switch. Neither requires the other switch or the result pipeline. Rewrites may produce additional eligible joins; ranked bitmap selection is GPU-only. |
| Partitioned baseline reduction | Requires the result pipeline and `--enable-partitioned-baseline-gpu-reduction`, eligible rowwise baseline-hash results on multiple GPUs, and sufficient input size. |
| Direct temporary ResultSet reads | Use the result pipeline plus the relevant peer-read switch. Join-column and segmented-payload switches are separate; neither enables the other. GPU capability and allocation checks still apply. |
| Deferred lazy fetch | Independently controlled by `--enable-deferred-lazy-fetch`. It defers eligible fixed-width lazy table-column loads; it does not enable the result pipeline or input prefetch. |
| GPU input prefetch | Independent of the result pipeline. Batched prefetch additionally requires `--enable-gpu-input-prefetch=true`; scalar GPU prefetch uses the worker-count limit. GPU prefetch takes precedence over CPU prefetch. |
| Native CPU-buffer bypass | Independent of prefetch and the result pipeline. Requires its own switch, a positive staging-buffer setting, native storage, and an eligible CPU-cache miss. |
| Compressed input overlap and peer exchange | Require the native bypass path, eligible compressed batches, and matching nvCOMP support. The scan/join configuration below enables GPU prefetch and batched prefetch to supply those batches. The two compressed-input switches are independent of each other. |
| Host mapping and selected-dense payload fetch | Independent of the result pipeline and input-prefetch switches. Both target narrow aggregate shapes. Host mapping takes precedence when both are true; test them separately. |
| Native compression and metadata sidecars | Independent of the query pipeline. Compression requires its writer switch and a supported codec. Sidecar-only metadata requires its own switch and an eligible table. GDeflate writer configuration additionally requires the sidecar-only switch true and zero table rollback epochs. |
| Advisory manifests | Independent of compression and sidecar creation. Required manifests for existing sidecar-only data are read and maintained even when the advisory-manifest flag is false. |
| Lazy dictionary recovery and parallel dictionary sorting | Independent of the result pipeline and of each other. Each controls a different dictionary operation. |
| Jump buffers, peer-copy transport, cubin cache, split JIT | Independently configured. Background jump-buffer allocation requires lazy allocation and nonzero jump-buffer size. Cubin caching requires a nonempty path; split JIT does not require the cubin cache. |

CPU retry settings control recovery from eligible GPU query or query-step failures.
The query-level and query-step settings are independent. Setting both false makes those
failures visible while CPU planning, decompression, host materialization, and other CPU
work within a successful query remain available. In this guide, "strict GPU" means a
query requested in GPU mode with these two retry paths disabled.

## Recommended evaluation sequence

Evaluating one feature group at a time makes it easier to connect a change in behavior
or performance to its settings. Use the same binary and data for each comparison, with
the complete stage 0 results as the reference.

The stages group related experiments; you can choose the ones relevant to your workload
using the dependency table above. Keep previously validated settings when applying a
later block, except where that block changes them. Storage conversion and trusted
constraints are optional, so independent query or transfer experiments can proceed with
them disabled. To make GPU failures visible during testing, use the separate
[strict validation configurations](#strict-and-fallback-validation-configurations).

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

This first stage keeps both CPU retry paths available while you compare query results.
It leaves exact bitmap membership disabled so that layout can be evaluated separately.
With exact bitmap membership
disabled, the recorded SF=1000 workload included a semi-join whose legacy hash layout
requested 2,967,662,096 entries, exceeding its 2^31-entry GPU hash-table limit. CPU retry
can recover a query that encounters this supported fallback condition; disabling retries
exposes the error. Exact bitmap membership can avoid that layout when its own eligibility
checks pass. Once stage 1 results match the reference, compare exact bitmap membership
with these settings:

```text
--enable-bitmap-hashjoin=true
--enable-ranked-bitmap-hashjoin=true
--allow-cpu-retry=true
--allow-query-step-cpu-retry=true
```

After repeated result, capacity, and fallback checks on the target data, a separate run
with both retries disabled can help identify queries that still depend on CPU recovery.

### Stage 2: restart and hydration acceleration

After the stage 1 results match the reference, evaluate the following groups separately.
They do not require trusted constraints or a storage-format conversion.

#### Native input hydration

```text
--enable-gpu-input-cpu-prefetch=false
--enable-gpu-input-prefetch=true
--enable-gpu-input-batched-prefetch=true
--gpu-input-prefetch-workers=32
--enable-gpu-input-cpu-buffer-bypass=true
--gpu-input-cpu-buffer-bypass-mode=staged
--gpu-input-cpu-buffer-bypass-staging-buffer-bytes=67108864
--gpu-input-cpu-buffer-bypass-reader-threads=16
--enable-gpu-input-compressed-pipeline=false
--enable-gpu-input-compressed-peer-exchange=false
```

The `staged` bypass provides an initial comparison point for `mmap`, whose performance
depends more directly on page-cache and filesystem behavior. Once the staged results
match the reference, compare `mmap` and also try batched prefetch with its flag false. The worker
count controls scalar GPU prefetch, not the batched path or CPU prefetch. Bypass can also
be evaluated with both GPU-prefetch flags false: ordinary foreground GPU fetches can use
it on eligible cache misses.

For a separate CPU-prefetch experiment, set `--enable-gpu-input-cpu-prefetch=true` and
both GPU-prefetch flags false. CPU prefetch populates the CPU pool, so it can remove the
CPU-cache misses needed to exercise bypass. Keep it false when measuring bypass.

Evaluate deferred loading independently with:

```text
--enable-deferred-lazy-fetch=true
```

#### Optional compressed input transport

This experiment uses supported compressed native chunks. If the catalog currently has
uncompressed data, stage 3 provides the preparation steps. With the native input
configuration above and matching nvCOMP GPU decode support, first test overlap:

```text
--gpu-input-compressed-batch-max-bytes=2147483648
--enable-gpu-input-compressed-pipeline=true
--enable-gpu-input-compressed-peer-exchange=false
```

Then test compressed peer exchange separately:

```text
--enable-gpu-input-compressed-pipeline=false
--enable-gpu-input-compressed-peer-exchange=true
```

Overlap needs more than one compatible sub-batch. Peer exchange needs concurrent
consumers of the same compressed chunk on peer-accessible GPUs. Comparing each option
individually helps identify its contribution before combining them. The batch size
controls compressed-payload grouping; decode output and scratch use additional memory.

#### Metadata and dictionary work

These three switches are independent; compare each with its false setting before using
them together:

```text
--enable-file-mgr-manifests=true
--stringdict-parallel-sort=true
--enable-lazy-string-dictionary-hash-recovery=true
```

To measure the restart benefit of manifests, checkpoint and reopen after creating them.
Lazy dictionary recovery is also best observed after persisting a dictionary and
restarting, when the new process can defer hash construction until it is needed.

#### Persistent compilation and optional split JIT

Use a local directory writable by the service identity:

```text
--gpu-cubin-cache-path=/var/cache/heavydb/cubin
--gpu-cubin-cache-max-size-in-bytes=4294967296
--cuda-jit-max-parallel-threads=1
```

The directory and 4 GiB capacity are examples, not required values. See
[cache path preparation](#path-and-service-accounts). Run a query, restart, and repeat it
to test persistent reuse. A repeat within one process can be satisfied by the in-memory
cache and therefore does not establish persistent reuse.

Set `CUDA_CACHE_PATH` to a local service-owned directory in the process environment; it
controls NVIDIA's PTX compute cache and is separate from HeavyDB's cubin cache.
Independently, on a CUDA 13 or newer build, test a bounded split-JIT setting such as:

```text
--cuda-jit-max-parallel-threads=2
```

`1` disables split compilation; `0` allows all available CPUs. A value other than `1`
is rejected at startup by older CUDA builds and by CPU-only builds. Test split JIT on
compilation misses: a persistent cubin hit can bypass the compiler whose cost is being
measured. It does not require persistent cubin caching to be enabled.

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
To measure compressed H2D transport, also use the eligible native bypass configuration
and matching nvCOMP GPU decoder. A compression writer setting alone can still yield CPU
decompression followed by uncompressed H2D transfer.

Keep Snappy as the first compatibility profile. GDeflate requires nvCOMP CPU codec
support, sidecar-only metadata, and zero rollback epochs; Bitcomp and adaptive modes also
require matching nvCOMP support. Those codecs are separate storage-format experiments,
not drop-in additions to the conservative Snappy rollout.

### Stage 4: trusted key metadata, only after data validation

Validate the declared constraints with trust disabled, using the
[validation queries](#constraint-validation-queries). Then enable:

```text
--enable-experimental-query-rewrites=true
--trust-unenforced-table-constraints=true
```

Both are needed for the Calcite metadata proofs described here. The result pipeline,
storage compression, and sidecar-only metadata are not prerequisites. Enable trust only
after every declared key and reference has been checked against the real data and a
process exists to keep future writes valid. If production data cannot be checked in
place, leave trust false. A synthetic copy can test the optimizer, but cannot prove that
production data satisfies an unenforced constraint.

### Stage 5: advanced experimental options

Each experiment below can be compared with a configuration whose results already match
the reference. Keeping the other experiment switches false and restarting between runs
helps isolate its effect. Repeated parent settings, such as the result pipeline, show
the prerequisites for that experiment; Stage 4 constraint trust is optional. Checking
which path the workload selects helps explain both improvements and unchanged timings;
[observing feature selection](#observing-feature-selection) suggests ways to do that.

For the query-execution experiments, set the experiment's flag back to false and restart
to return to the prior path. Sidecar-only metadata requires the storage conversion
described below to restore physical metadata pages.

#### Sidecar-only metadata settings

Start the server with:

```text
--enable-file-mgr-manifests=true
--enable-file-buffer-metadata-sidecar-only=true
```

Use a disposable or backed-up, populated native table. Set its rollback epochs to zero
and checkpoint it. Zero disables rollback retention for that table:

```sql
ALTER TABLE test_table SET MAX_ROLLBACK_EPOCHS = 0;
OPTIMIZE TABLE test_table;
```

The sidecar flag is process-wide: any eligible native table checkpoint can convert, not
only the table named in this example. Use an isolated evaluation catalog when other
tables must retain their format. Compression is not required, and the advisory-manifest
flag is not a prerequisite for maintaining required sidecars; it is explicit here for
the combined metadata configuration.

Eligible chunks must have physical data pages. The conversion upgrades the table to
FileMgr v3. Restart and verify the table after conversion. Setting the creation flag back
to false leaves existing sidecar-only chunks readable but does not restore physical
metadata pages; follow [the conversion procedure](#converting-a-sidecar-only-table-back)
before opening the table with an older binary.

#### Direct temporary ResultSet peer-read settings

For eligible temporary join columns:

```text
--enable-result-reduction-pipeline=true
--enable-temporary-resultset-peer-access=true
--enable-temporary-resultset-payload-peer-access=false
```

In a separate run, evaluate segmented payload reads while keeping join-key hash-table
builds on the copy path:

```text
--enable-result-reduction-pipeline=true
--enable-temporary-resultset-peer-access=false
--enable-temporary-resultset-payload-peer-access=true
```

Use multiple GPUs and retained intermediate results consumed on another GPU. The
startup peer-access matrix reports copy capability only. This revision does not run an
automatic remote-kernel validation probe or log a direct-read policy matrix. Verify
selection and correctness on the actual topology as described under
[direct-read eligibility](#direct-temporary-resultset-peer-reads). GPU pairs rejected by
the capability or allocation checks continue to use copies.
`--peer-copy-mode=direct` alone does not enable either experiment.

#### Host-mapped aggregate payload settings

```text
--enable-gpu-aggregate-payload-host-mapping=true
--enable-gpu-selected-dense-aggregate-payload-fetch=false
```

Use selective, single-table, non-grouped aggregates with eligible fixed-width payload
columns. This option does not require the result pipeline or input prefetch. Compare
repeated results and execution times with host mapping disabled.

#### Selected-dense aggregate payload fetch settings

```text
--enable-gpu-aggregate-payload-host-mapping=false
--enable-gpu-selected-dense-aggregate-payload-fetch=true
```

Keep host mapping false because it takes precedence when both flags are true. Use
selective, single-table, non-grouped aggregates with supported simple predicates and
fixed-width payload columns. This option also does not require the result pipeline or
input prefetch. See
[payload-fetch eligibility and tradeoffs](#advanced-experimental-payload-fetch-modes)
for both modes.

#### Partitioned baseline reduction settings

```text
--enable-result-reduction-pipeline=true
--enable-partitioned-baseline-gpu-reduction=true
```

Compare with the same pipeline configuration and
`--enable-partitioned-baseline-gpu-reduction=false`. The partitioned path requires
multiple eligible rowwise baseline-hash ResultSets on at least two GPUs and an input
hash-table storage size meeting the implementation's 256 MiB threshold. Small inputs,
columnar results, and one-GPU runs do not exercise it. Validate repeated results,
CPU-boundary materialization, cancellation, and memory pressure as described in the
[differential test procedure](#differential-test-procedure). The documented reference
benchmark profile leaves this option false.

#### Fragment-size settings

Set `FRAGMENT_SIZE` when creating each comparison table, for example with a
`WITH (FRAGMENT_SIZE=16000000)` clause, and load the same data into each layout. This is a
separate physical-design experiment; restarting the server or running `OPTIMIZE TABLE`
does not change an existing table's fragment size. Choose candidate sizes using the
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

Constraints are stored with `enforced=false`, so inserts and updates can introduce rows
that violate the declarations. With trust disabled, Calcite leaves these declarations
out of cardinality and join-elimination proofs. With trust enabled, it assumes they
hold, and an invalid declaration can produce a wrong query result. The checks below
help establish whether a declaration is suitable for optimizer use.

Calcite only exposes a primary or unique key as a key statistic when all participating
columns are also `NOT NULL`.

### Constraint validation queries

Adapt these templates for every declared key. A valid result is no rows.

Run them with `--trust-unenforced-table-constraints=false` so the optimizer evaluates
the data without assuming the declaration being checked. The same setting is useful
when revalidating data after an external load or restore.

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
on the pre-existing plan. Comparing complete results with the reference checks that a
selected transformation preserves the query's values and row multiplicities.

## Hash-join options

### Exact bitmap membership

`--enable-bitmap-hashjoin=true` enables compact exact membership tables for eligible
semi and anti joins. These tables answer whether a key exists; they do not carry a
general join payload.

This can reduce memory and probe cost for bounded integer-like key domains. Sparse or
large domains that fail eligibility checks fall back to another hash-table layout.
Exact bitmap membership can be selected for CPU or GPU execution. Its switch is
independent of ranked bitmap joins, experimental rewrites, and the result pipeline.

### Ranked bitmap joins

`--enable-ranked-bitmap-hashjoin=true` enables a bitmap plus rank index for eligible GPU
inner joins. The rank maps a present key to a compact payload position. The implementation
also supports filtered build sides and payload-free unique probes where the plan proves
they are valid.

Ranked bitmap selection is restricted to real GPU query builds with a usable build-side
expression graph; CPU cache tests and synthetic perfect/baseline hash-table paths retain
their existing implementations.
Hints that force a baseline or one-to-many hash layout also prevent ranked selection.
Enabling either bitmap switch permits an eligible layout; it does not force every join
to use that layout.

Selection depends on domain range, entry width, slab size, and available allocation. On
ineligible shapes, HeavyDB uses
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

This family includes eligible CPU result-reduction and materialization paths. The flag
does not mean that every result remains on a GPU, nor does it activate the separately
controlled partitioned reducer or temporary ResultSet peer reads. Input prefetch,
deferred lazy fetch, host-mapped payloads, and selected-dense payload fetch have their
own independent gates; see [switch dependencies](#how-the-switches-depend-on-each-other).

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
multi-ResultSet baseline-hash reductions within the enabled result pipeline. Selection
requires compatible rowwise results from at least two source GPUs, supported reduction
slots, and the 256 MiB input-entry threshold. Precisely, total input entry count must be
at least `floor(268435456 / row_size_bytes)`. Keyless, columnar, and variable-length output
layouts are excluded.
Each source ResultSet must expose one device rowwise fragment with a matching reduction
layout. The size threshold is fixed in this implementation and measures input-entry
storage rather than final output size.

The reducer partitions work across devices and streams. One-GPU runs and small fixtures
use another reducer, so a comparison of this path needs eligible multi-GPU input. The
reference performance profile left it false; a separate comparison can establish whether
it offers a repeatable benefit for your workload.

`--enable-gpu-result-reduction-pipeline` is a deprecated alias for
`--enable-result-reduction-pipeline`. Prefer the current name and configure just one
spelling to keep the effective setting clear.

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
   and the number and size of partial ResultSets. These measurements help weigh a lower
   average against changes in tail latency, retries, and memory pressure.
4. Test the largest and most skew-sensitive aggregate and join plans, not only scan-heavy
   queries. Validate result equivalence at every candidate size.
5. For workloads similar to the reference configuration, 16 million rows can be a starting
   point. Use the target hardware and data to choose the final value.

## GPU input hydration

### CPU prefetch

`--enable-gpu-input-cpu-prefetch=true` fetches eligible input chunks into the CPU buffer
pool before foreground GPU materialization. It is active only when GPU prefetch is off.
It can hide serialized FileMgr reads but populates HeavyDB's CPU buffer pool. Its worker
count is capped by the engine's CPU-thread count and 32; it does not use
`--gpu-input-prefetch-workers`. A prefetched CPU-resident chunk no longer qualifies for
the CPU-cache-miss bypass described below.

### GPU prefetch

`--enable-gpu-input-prefetch=true` prepares known fixed-width input chunks toward the
target GPU. Its scalar path starts bounded background fetches; batched mode submits
grouped requests. Chunk owners remain pinned through execution. Varlen and unsupported
inputs use the established foreground path.

`--gpu-input-prefetch-workers` bounds scalar prefetch workers per query fetch task. More
workers can increase concurrency, while nested reader fanout can increase contention
because HeavyDB already fetches across devices and columns. Comparing a few worker
counts can help find a useful balance for the host.

### Batched prefetch

`--enable-gpu-input-batched-prefetch=true` is meaningful only with GPU prefetch. It groups
eligible requests to reduce FileMgr, mmap, H2D, and decompression setup overhead.

The fast batch path requires compatible inputs, such as a common compressed codec or a
supported uncompressed mmap batch. A request containing several codecs is split into
compatible groups. CPU-resident, partial, foreign-storage, and other ineligible requests
fall back to scalar fetches without discarding every eligible group. Batched fetch does
not create one scalar prefetch worker per chunk, so the scalar worker limit is not a
general batch-concurrency limit.

### Compressed transfer and decode

These paths require eligible compressed native chunks, CPU-buffer bypass, and matching
nvCOMP decode support. The GPU-prefetch plus batched-prefetch configuration in stage 2
supplies scan/join batches. Other consumers, including eligible deferred lazy fetches,
can also issue batched buffer requests. Setting the compression writer flag does not
compress old chunks or itself activate a GPU input path.

`--gpu-input-compressed-batch-max-bytes` is a target for grouping compressed payload
bytes into nvCOMP batches. The 2 GiB default is not a hard GPU-memory cap: grouping splits
at chunk boundaries, and one chunk can exceed the target. Decompressed output, metadata,
and decode scratch require additional memory. A value of `0` removes the grouping limit.
Allocation failures retry smaller groups or fall back to scalar fetch and CPU decode.

`--enable-gpu-input-compressed-pipeline=true` halves a positive grouping target greater
than one byte for each of two lanes, so HeavyDB can overlap one H2D compressed-payload
transfer with decode of the preceding batch. Actual overlap requires more than one
batch. Small requests and unlimited targets can produce a single batch, so checking
the batch count helps explain when overlap is available. This switch is independent
of the result-reduction pipeline.

`--enable-gpu-input-compressed-peer-exchange=true` lets concurrent GPU consumers share
one compressed storage read. HeavyDB copies the compressed representation directly to an
eligible peer GPU and decodes into that GPU's ordinary input buffer. Unsupported codecs,
topologies, or allocations use the normal per-device fetch path.
It does not require compressed-pipeline overlap, and its direct peer copy does not
consult `--peer-copy-mode`. Test exchange separately from overlap and include concurrent
consumers of the same stored chunk on different GPUs.

### CPU-buffer bypass

`--enable-gpu-input-cpu-buffer-bypass=true` applies to eligible native FileMgr GPU fetches
when the chunk is absent from the CPU buffer pool and the staging-byte setting is
positive. It avoids installing that chunk in HeavyDB's CPU buffer pool. It does not
require either prefetch switch or the result-reduction pipeline.

- `staged` tries pinned staging through the jump-buffer producer path, then falls back
  to an ordinary host staging buffer before H2D transfer.
- `mmap` maps page-cache-backed native files and copies their page payloads toward the
  GPU without first installing a permanent HeavyDB CPU-buffer copy.

This is not GPUDirect Storage. The operating-system page cache remains part of the path.
Foreign tables, unsupported buffers, and ineligible requests retain their established
fetch behavior. Known allocation/eligibility failures use the supported fallback paths;
unhandled runtime errors still propagate.

`--gpu-input-cpu-buffer-bypass-staging-buffer-bytes=0` disables the bypass even if its
boolean flag is true. A positive value caps the ordinary scalar host staging buffer;
it does not cap all pinned memory, jump-buffer slots, decompression scratch, or batch
output. The reader-thread setting controls native read fanout within a fetch and is
separate from both scalar prefetch workers and general FileMgr reader threads.

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
- Grouped, join, and distinct aggregate shapes do not use them. Host mapping marks only
  eligible fixed-width payload columns; selected-dense mode rejects unsupported payload
  column types and predicates.
- Setting host mapping true disables selected-dense selection, even if a particular
  query later fails host-mapping eligibility. Enable exactly one for a comparison.

Both modes are independent of the result-reduction pipeline and GPU input prefetch.
Selection depends on the predicates, aggregate targets, and input-column types.
The optimization needs a payload column that is not also needed to evaluate the filter.
Selected-dense mode additionally requires supported simple column/constant predicates
without residual general predicates.

On the reference workload, CPU-side selection and compaction cost more than a contiguous
column transfer. Both options therefore remain false in the initial profile; a comparison
with your own selective aggregates can show whether that tradeoff is different.

## Lazy string dictionary recovery and scan fast paths

Persistent dictionary payloads and offsets are memory-mapped with either setting.
`--enable-lazy-string-dictionary-hash-recovery=true` defers rebuilding a recovered
dictionary's string-to-ID hash until an operation actually needs the hash. The mapped
payload and offsets retain the persistent dictionary contents. Requests that cannot use
a bounded scan trigger the ordinary hash-recovery path before continuing.

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

`--stringdict-parallel-sort=true` is a separate choice: eligible sorted-cache construction
uses a parallel CPU sort instead of the serial sort. It neither enables nor requires lazy
hash recovery, GPU execution, or the result-reduction pipeline. Compare first-use cache
construction as well as queries that reuse an existing sorted cache.

## Native storage compression

### Scope

Compression applies to HeavyDB native FileMgr chunk payloads. It does not turn Parquet
foreign-table scans into a direct Parquet-to-GPU path, and it does not use Parquet's
compression metadata.

Compression is process-wide for newly initialized native chunks. It is not a per-table
`CREATE TABLE` property. A table can contain a mix of compressed and uncompressed chunks,
and existing compressed chunks retain their recorded codec even if the startup writer
flag is later disabled.

For an uncompressed chunk, the writer selects compression only when:

- compression is enabled,
- the codec and frame size are valid,
- compression metadata fits, and
- the compressed payload is smaller than the logical payload.

With compression disabled, or when compression is unhelpful or its metadata cannot fit,
the writer uses the ordinary uncompressed payload. Invalid or unavailable codec
configurations can instead fail startup or a write. Codec and frame-size validation also
runs at startup when the writer flag is false, so disabling new compressed writes still
requires a supported codec configuration.

### Codec choice

The default configured codec is Snappy with 64 KiB uncompressed frames.

- **Snappy** is the conservative GPU hydration codec. When HeavyDB is built with nvCOMP
  and the fetch is eligible, compressed bytes can be transferred and decoded into final
  GPU input buffers. CUDA OOM in this optional path falls back to a smaller, scalar, or
  CPU decode path.
- **LZ4** saves disk space and disk-read bytes, but this implementation disables raw-LZ4 nvCOMP
  decode because the stored raw framing is not treated as a proven-compatible nvCOMP
  bitstream. LZ4 therefore uses CPU decompression and transfers the decompressed input
  to the GPU; it does not provide compressed H2D transport here.
- **GDeflate** uses nvCOMP's CPU codec for writes and supports GPU hydration. It requires
  nvCOMP CPU codec support, the sidecar-only creation switch enabled, and a table with
  `MAX_ROLLBACK_EPOCHS=0`. Compression levels range from 0 through 12; the default level
  `1` favors load throughput.
- **Bitcomp-sparse** and **Bitcomp-default** use type-aware nvCOMP Bitcomp storage when
  that API is available in the CUDA build. `bitcomp` is an alias for `bitcomp-sparse`.
  Unlike GDeflate, these codecs do not inherently require sidecar-only metadata or zero
  rollback epochs.
- **adaptive** (or `auto`) compresses each FileBuffer with both Snappy and
  Bitcomp-default and keeps the smaller representation. It requires Bitcomp support and
  trades additional load-time CPU work for per-buffer codec selection.
- **none** disables compression for new chunks.

Codec names and the `bitcomp`/`auto` aliases are case-insensitive. GDeflate writing to a
table that is not eligible for sidecar-only metadata raises an error. Adaptive chunks
record the concrete codec selected, which determines the reader support they need.

Snappy is a required build dependency for this revision. nvCOMP is optional. A build
without nvCOMP retains CPU Snappy/LZ4 readability but rejects unavailable GDeflate,
Bitcomp, or adaptive writer configurations.

The 64 KiB default is the reference setting. The writer can increase the effective frame
size for a very large chunk when the per-frame size table would not fit in the metadata
page. In the reference workload, moving to a 1 MiB frame saved little disk space and
increased decode time, illustrating the tradeoff to measure when tuning this value.

Frame size is a positive uncompressed byte count, subject to codec limits: at most 16 MiB
for Snappy and Bitcomp, 64 KiB for GDeflate, and the linked LZ4 library's maximum input
size for LZ4. Adaptive must satisfy both Snappy and Bitcomp limits. If metadata cannot fit
even after increasing the effective frame size within its codec limit, the writer can
retain an uncompressed payload. The GDeflate level setting affects only GDeflate writes;
it is not a general compression-quality setting for the other codecs.

For an isolated GDeflate experiment, the server settings are:

```text
--enable-native-storage-compression=true
--native-storage-compression-codec=gdeflate
--native-storage-compression-frame-size=65536
--native-storage-compression-gdeflate-level=1
--enable-file-buffer-metadata-sidecar-only=true
```

On a backed-up, populated native test table, then run:

```sql
ALTER TABLE test_table SET MAX_ROLLBACK_EPOCHS = 0;
OPTIMIZE TABLE test_table WITH (STORAGE_COMPRESSION='GDEFLATE');
```

Reopen and validate it. This changes both payload encoding and metadata storage; use the
[sidecar conversion procedure](#converting-a-sidecar-only-table-back) to reverse them in
the required order. For Bitcomp or adaptive experiments, select `bitcomp-sparse`,
`bitcomp-default`, or `adaptive` as the codec and use the corresponding named rewrite.
Keep the sidecar switch false unless sidecar conversion is itself part of that experiment.
Those codecs require the matching Bitcomp build support but not GDeflate's table setting.

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

This encoder metadata version is separate from the table's FileMgr version. A FileMgr
v2 table can contain compressed chunks that an older binary cannot read. Downgrade
planning therefore includes both the table's metadata format and its chunk codecs;
the [conversion inventory](#storage-format-inventory-and-conversion-records) helps track both.

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
falls back to scanning physical headers and metadata pages. This makes advisory
manifests useful for reducing restart and table-open latency, while leaving the ordinary
table data and metadata available for recovery.

For the sidecar-only tables described below, the metadata manifest holds the required
metadata copy. Retain that manifest until the table has been converted back to physical
metadata pages using the [conversion procedure](#converting-a-sidecar-only-table-back).

### Sidecar-only metadata

`--enable-file-buffer-metadata-sidecar-only=true` permits eligible chunks to remove their
physical metadata page and retain current metadata only in the durable sidecar manifest.
It is independent of the compression writer, advisory-manifest switch, and query
pipeline. The creation switch applies process-wide to eligible tables at checkpoint,
not just to the table named in an example.
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
corrupt, mismatched, or incomplete required metadata causes table open to fail so that
the required metadata can be recovered before use.

Once a table is version 3:

- the new binary loads its required manifest even if both manifest creation flags are
  false,
- disabling the sidecar creation flag does not recreate physical pages by itself, and
- an older binary fails with the table-storage forward-compatibility error.

Because this option changes where required metadata lives, it is useful to evaluate the
conversion and recovery procedure separately. The initial profile keeps it disabled.

### Converting a sidecar-only table back

Before a binary downgrade, first remove compression while still running a compatible new
binary:

```sql
OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='NONE');
```

This order matters for GDeflate: decompress it while the table still has the required
sidecar-only eligibility. Then restore physical metadata pages:

```sql
ALTER TABLE t SET MAX_ROLLBACK_EPOCHS = 1;
OPTIMIZE TABLE t;
```

The changed rollback setting makes the next metadata checkpoint recreate physical
metadata pages. Once no sidecar-only chunk remains, HeavyDB writes FileMgr version 2.
For a sharded table, verify every physical shard through the logical table operation.

Restart the new binary once to verify the converted table, and retain a snapshot until
the old binary has opened and queried it successfully. Disabling new compression and
sidecar creation prevents subsequent checkpoints from reintroducing those formats;
it does not replace either conversion step.

### Storage-format inventory and conversion records

This revision does not expose a SQL command that inventories the codec and metadata mode
of every physical chunk. A logical table can contain both compressed and uncompressed
chunks, and a sharded table has multiple physical FileMgr roots. Current server flags
describe creation behavior; conversion records help track the formats already stored.

A useful conversion record includes:

- every logical table selected for conversion,
- all physical shards covered by that table,
- the requested compression codec and completed `OPTIMIZE TABLE` operation,
- whether sidecar-only conversion was enabled at checkpoint, and
- the snapshot or backup that precedes each transition.

FileMgr version 3 is the on-disk fence for required sidecar metadata. If storage-level
inspection is part of an operational procedure, inspect every physical table/shard root
and read `filemgr_version` without modifying it. Pair that inspection with opening and
querying the logical table to verify the data across its physical shards.

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
therefore reserves roughly 8 GiB of pinned host memory. On a shared host, include that
reservation alongside the memory needed by HeavyDB and other services.

`--jump-buffer-size=0` disables jump buffers; `1` is invalid, and a nonzero size must be at
least two bytes. Thread and slot counts must each be at least one. The directional
thresholds determine which ordinary H2D/D2H copies attempt jump-buffer use. A threshold
of zero admits all positive copy sizes; it does not disable that direction.

Lazy allocation moves allocation cost to first use. Background allocation only has an
effect with lazy allocation enabled and nonzero buffer size; it warms slots after
startup. Eager allocation adds startup and shutdown work and can fail startup if pinned
allocation fails. A failed optional lazy allocation falls back to ordinary transfers.

### Cross-GPU copy mode

`--peer-copy-mode` selects transport in the general cross-device copy helper:

- `direct`: use CUDA peer copies when peer access is available; otherwise use host staging.
- `staged`: copy in chunks through a per-GPU-pair device staging allocation when peer
  access is available; unavailable access or staging allocation OOM uses host fallback.
- `host`: use a host-staged D2H/H2D path for copies through this helper.

`--peer-copy-staging-buffer-size` sets the device staging chunk size in `staged` mode and
the host staging chunk size when the host path is used. Zero disables device staging and
uses 64 MiB host chunks; it does not disable a `direct` peer copy when that mode is selected.
Direct peer copy is normally the first choice on NVLink/NVSwitch systems, but topology
and measured bandwidth should decide.

Two independently controlled paths operate outside this copy helper: direct temporary
ResultSet kernel reads and compressed input peer exchange, which uses a direct peer
copy. For a comparison that sends these documented ResultSet copy paths
through host staging, also disable the independent peer-read and compressed-exchange
options:

```text
--peer-copy-mode=host
--enable-temporary-resultset-peer-access=false
--enable-temporary-resultset-payload-peer-access=false
--enable-gpu-input-compressed-peer-exchange=false
```

### Direct temporary ResultSet peer reads

The two temporary ResultSet flags allow a generated kernel to dereference another GPU's
temporary allocation. This is a stronger requirement than CUDA's ability to perform a
peer copy.

- `--enable-temporary-resultset-peer-access` covers eligible temporary join columns.
- `--enable-temporary-resultset-payload-peer-access` covers eligible segmented payload
  columns while leaving join-key hash-table builds on the copy path.

Use the result-reduction pipeline to retain eligible device ResultSets for these paths.
The two peer-read switches are independent. Neither enables the other or changes
`--peer-copy-mode`.

Direct access requires all of the following before HeavyDB exposes a remote pointer:

- CUDA reports peer-to-peer (P2P) access support,
- CUDA reports native peer atomics, used as a stronger capability gate than peer-copy
  support alone,
- the source is a tracked CUDA virtual memory management (VMM) allocation,
- read access can be granted for the consuming device, and
- producer readiness is ordered before the consumer stream reads it.

The implementation caches CUDA capability decisions for ordered device pairs and checks
allocation access before exposing pointers. The startup peer-access matrix reports basic
CUDA peer capability; this revision has no automatic startup remote-read validation
kernel or direct-read policy matrix. Workload tests and transfer measurements provide
the complementary information about correctness and bandwidth on the host. If a
capability or allocation-access check fails, the fetch path copies the fragment instead.

The `GpuPeerAccessRuntime` tests exercise event-ordered peer DMA and remote kernel reads
for one device pair when that pair is eligible, and skip when the required capabilities
are unavailable. On a larger system, extending the checks to the pairs your workload uses
helps establish coverage. The conservative profile leaves both flags false; enabling
either is a separate experiment with repeated result comparisons, path checks, and
stress testing on the target topology.

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

The service identity can own this directory without a login shell or home directory.
In a container, mount a writable cache volume at the configured path. If the filesystem
is read-only or no persistent volume is desired, leave the path empty.

Because cubins contain executable code, restrict directory write access to the trusted
service identities. One directory per instance keeps ownership straightforward;
instances using the same software and trust boundary can share one.

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
When the configured limit is exceeded, cubins from older cache generations are deleted
before current-generation entries; each group is ordered by oldest modification time.

Cache path and size settings:

- Empty `--gpu-cubin-cache-path`: cache disabled.
- Positive max size: size-based pruning, prioritizing older generations and then age.
- Max size `0`: size-based cubin pruning disabled, so growth is unbounded. Stale temporary
  file cleanup still runs, and the cache remains enabled while its path is nonempty.

It is safe to clear this directory while HeavyDB is stopped. The consequence is extra
compilation latency, not table-data loss. After an upgrade, incompatible generations age
out under the size policy; an operator may also clear the cache during the maintenance
window.

## Complete option reference

The tables below list server startup options. `FRAGMENT_SIZE` is documented separately
because it is a table property. Byte-valued options are expressed in bytes. Defaults are
for a CUDA build at the revision covered by this guide, not the larger values used in
some reference benchmarks. MiB and GiB mean powers of 1024. This is the complete list of
server options discussed in this guide, not every HeavyDB server option.

For a boolean feature switch, false disables selection of that feature while required
readers for already-written storage remain available. Limits and mode selectors take
effect when their associated feature is enabled. Zero has an option-specific meaning,
listed in the relevant row below.

### Planning and hash joins

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-experimental-query-rewrites` | `false` | Selects the coordinated Calcite rule sequence and optimizer metadata. Independent of the result pipeline. False restores the baseline planner sequence; declared constraints remain stored. |
| `--trust-unenforced-table-constraints` | `false` | Allows proofs from valid declared PK/UNIQUE/FK metadata; the Calcite proofs described here also require experimental rewrites. Does not validate or enforce rows. Keep false for unvalidated data and for the validation queries themselves. |
| `--enable-bitmap-hashjoin` | `false` | Allows exact bitmap membership for eligible CPU/GPU semi and anti joins. Independent of ranked bitmap and the result pipeline; ineligible joins use another layout. |
| `--enable-ranked-bitmap-hashjoin` | `false` | Allows ranked bitmap for eligible GPU inner joins with a usable build-side expression graph. Independent of exact bitmap and the result pipeline; layout-forcing hints and eligibility checks can prevent selection. |

### Result execution and reduction

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-result-reduction-pipeline` | `false` | Parent switch for eligible retained device ResultSets, asynchronous reduction, and associated GPU/CPU materialization paths. Does not turn on the other options; partitioned reduction and temporary peer reads need additional switches. |
| `--enable-gpu-result-reduction-pipeline` | `false` | Deprecated alias of the preceding option. Prefer the current name and configure just one spelling. |
| `--enable-partitioned-baseline-gpu-reduction` | `false` | Additional reducer within the enabled result pipeline. Requires compatible rowwise baseline-hash inputs from multiple GPUs and the approximately 256 MiB entry-storage threshold; see the exact eligibility discussion above. False retains other reducers. |
| `--allow-cpu-retry` | `true` | Allows supported query-level CPU retry paths. False exposes those failures; it does not disable the separate step retry, explicit CPU queries, or CPU work inside successful GPU queries. |
| `--allow-query-step-cpu-retry` | `true` | Allows supported query-step CPU retries. Independent of query-level retry. Set both false for the defined strict GPU validation profile; keep both true for a separate recovery test. |

### Input and payload fetch

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-gpu-input-cpu-prefetch` | `false` | For GPU execution, prefetches into the CPU pool only when GPU prefetch is false. Can remove the CPU-cache-miss condition needed by bypass. Independent of the result pipeline. |
| `--enable-gpu-input-prefetch` | `false` | Prefetches eligible fixed-width input toward the target GPU; takes precedence over CPU prefetch. Independent of the result pipeline and bypass. |
| `--enable-gpu-input-batched-prefetch` | `false` | Groups eligible input requests when GPU prefetch is true. False keeps scalar GPU prefetch if its parent switch is true. Does not enable compression or bypass. |
| `--gpu-input-prefetch-workers` | `32` | Maximum scalar GPU prefetch workers per fetch task, further capped by chunk and CPU-thread counts; minimum 1. Does not tune CPU prefetch or total batched-fetch concurrency. |
| `--enable-gpu-input-cpu-buffer-bypass` | `false` | Uses eligible native GPU fetches absent from the CPU pool, with positive staging bytes. Independent of prefetch and the result pipeline. False uses ordinary CPU-buffer materialization. |
| `--gpu-input-cpu-buffer-bypass-staging-buffer-bytes` | `67108864` (64 MiB) | Zero disables bypass. Positive values cap the ordinary scalar host staging buffer, not total pinned memory, jump slots, or decode workspace. |
| `--gpu-input-cpu-buffer-bypass-reader-threads` | `16` | Native read fanout within a bypass fetch; minimum 1. Distinct from GPU prefetch workers and general FileMgr readers. |
| `--gpu-input-cpu-buffer-bypass-mode` | `staged` | `staged` or experimental `mmap` for eligible native reads. Requires bypass enabled and positive staging bytes. Does not change stored payload format. |
| `--gpu-input-compressed-batch-max-bytes` | `2147483648` (2 GiB) | Compressed-byte grouping target; splits at chunk boundaries, so one chunk can exceed it. Output and scratch are additional. Pipeline overlap halves a positive target greater than 1 per lane; zero removes the grouping limit. |
| `--enable-gpu-input-compressed-pipeline` | `false` | Overlaps compressed transfer and decode for eligible native bypass batches with matching nvCOMP support and multiple batches. Independent of the result pipeline and compressed peer exchange. False retains non-overlapped decode. |
| `--enable-gpu-input-compressed-peer-exchange` | `false` | Shares eligible compressed batch reads among concurrent consumers of the same stored chunks on peer GPUs. Requires bypass and matching nvCOMP support. Independent of compressed overlap; its direct copy bypasses the general peer-copy-mode selector. |
| `--enable-deferred-lazy-fetch` | `false` | Defers eligible fixed-width lazy table columns until needed by a ResultSet consumer. Independent of the result pipeline and prefetch; unsupported columns use established lazy fetch. |
| `--enable-gpu-aggregate-payload-host-mapping` | `false` | Experimental mapped-host payload reads for eligible single-table, non-grouped GPU aggregates. Independent of the result pipeline. When true, prevents selected-dense selection. |
| `--enable-gpu-selected-dense-aggregate-payload-fetch` | `false` | Experimental compact payload fetch for qualifying rows of narrowly supported GPU aggregates. Requires host mapping false; independent of the result pipeline and prefetch. |

### String dictionaries

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--stringdict-parallel-sort` | `false` | Uses a parallel CPU sort for eligible dictionary sorted-cache construction. Independent of lazy recovery and GPU execution; false uses serial sorting. |
| `--enable-lazy-string-dictionary-hash-recovery` | `false` | Defers recovered dictionary hash construction and enables bounded scans and guarded string-view transformations. False restores eager recovery and copied-string scan behavior. Payloads are memory-mapped with either setting. |

### Native storage

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--enable-native-storage-compression` | `false` | Enables compression for eligible new native chunks and named-codec rewrites. Independent of the query pipeline. False does not decompress existing chunks or change their retained codec on ordinary updates. |
| `--native-storage-compression-codec` | `snappy` | Case-insensitive `none`, `lz4`, `snappy`, `gdeflate`, `bitcomp-sparse` (alias `bitcomp`), `bitcomp-default`, `adaptive` (alias `auto`). Subject to build and table eligibility; does not enable the writer. Invalid configurations can fail even with the writer off. |
| `--native-storage-compression-frame-size` | `65536` (64 KiB) | Positive requested uncompressed frame bytes. Codec-specific maxima apply; the writer may increase effective size to fit metadata within those limits. Does not enable compression. |
| `--native-storage-compression-gdeflate-level` | `1` | GDeflate CPU compression level, 0 through 12. Zero is a valid level, not disabled. Affects GDeflate writes only; needs writer enabled, matching CPU codec support, the sidecar-only switch true, and zero table rollback epochs. |
| `--enable-file-mgr-manifests` | `false` | Writes/uses optional page-header and physical-metadata manifests. Independent of sidecar creation. Required sidecar manifests are read and maintained even when false. |
| `--enable-file-buffer-metadata-sidecar-only` | `false` | Allows eligible table chunks with zero rollback epochs to remove physical metadata pages at checkpoint, upgrading their table to FileMgr v3. Independent of the advisory switch and compression. False does not reverse existing sidecar-only storage. |

### Host/device and peer transfer

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--jump-buffer-size` | `134217728` (128 MiB) | Pinned bytes per slot per GPU. Zero disables jump buffers; otherwise minimum 2 bytes. Multiply by slots and GPU count when budgeting host memory. |
| `--jump-buffer-parallel-copy-threads` | `4` | Host copy workers per GPU jump-buffer transfer; minimum 1. |
| `--jump-buffer-slots-per-device` | `1` | Concurrent slots per GPU; multiplies pinned memory; minimum 1. |
| `--enable-lazy-jump-buffer-allocation` | `false` | With nonzero jump-buffer size, allocates pinned slots at first use instead of startup. False selects eager allocation. |
| `--enable-background-jump-buffer-allocation` | `false` | Warms slots only when lazy allocation is true and jump-buffer size is nonzero. False leaves lazy slots to first use when lazy mode is selected. |
| `--jump-buffer-min-h2d-transfer-threshold` | `33554432` (32 MiB) | Minimum ordinary H2D copy size for attempting jump-buffer use. Requires nonzero buffer size; zero threshold admits all positive sizes. |
| `--jump-buffer-min-d2h-transfer-threshold` | `67108864` (64 MiB) | Minimum ordinary D2H copy size for attempting jump-buffer use. Requires nonzero buffer size; zero threshold admits all positive sizes. |
| `--peer-copy-mode` | `direct` | `direct`, `staged`, or `host` for the general cross-device copy helper. Does not govern remote kernel reads or compressed input peer exchange. Unavailable peer access uses host fallback. |
| `--peer-copy-staging-buffer-size` | `268435456` (256 MiB) | Device staging chunk size in staged mode, and host chunk size in the host path. Zero disables device staging and selects 64 MiB host chunks; it does not disable direct-mode peer copies. |
| `--enable-temporary-resultset-peer-access` | `false` | Additional gate for eligible retained ResultSet join-column peer reads. Use with the result pipeline; CUDA access/native-atomic capability and tracked allocation checks still apply. No automatic remote-read validation probe. |
| `--enable-temporary-resultset-payload-peer-access` | `false` | Separate gate for eligible retained segmented-payload peer reads, with the same pipeline and capability prerequisites. Does not enable join-column peer access; false retains copying. |

### Persistent compilation

| Option | Default | Effect and disable behavior |
|---|---:|---|
| `--gpu-cubin-cache-path` | empty | Explicit service-writable persistent cubin directory. Empty disables persistent caching; no inferred home-directory path. Independent of the result pipeline and split JIT. GPU UDF/runtime-UDF modules prevent use. |
| `--gpu-cubin-cache-max-size-in-bytes` | `1073741824` (1 GiB) | Size-based cubin pruning, older generations before current and oldest modification time within each group. Needs a nonempty path. Zero permits unbounded cubin growth; stale temporary-file cleanup remains active. |
| `--cuda-jit-max-parallel-threads` | `1` | CUDA 13+ driver split-compilation limit: `0` uses all available CPUs, `1` disables splitting, other positive values cap threads. Older CUDA and CPU-only builds reject values other than 1. Only matters on JIT work; a cache hit may bypass it. |

### Other settings used by the reference workloads

These existing capacity and threading settings are included because the reference
command blocks use them. They are independent of feature selection. Keeping them equal
between comparison runs helps isolate an option's effect; their values can be chosen
from the target host's capacity and workload needs.

| Option | Default | Effect and limits |
|---|---:|---|
| `--bigint-count` | `false` | Selects 64-bit rather than 32-bit COUNT results. Keep this consistent between typed-result comparisons, and enable it when counts can exceed the 32-bit range. |
| `--hashtable-cache-total-bytes` | `4294967296` (4 GiB) | Budget applied to each hash-table cache type, with per-device accounting, when hash-table caching is enabled. Not one combined server-memory cap. Use a positive budget; zero is not a documented cache-disable procedure. |
| `--max-cacheable-hashtable-size-bytes` | `2147483648` (2 GiB) | Largest individual hash table eligible for caching, also subject to the cache budget. Does not increase the maximum hash table that execution can build. |
| `--executor-max-available-resource-use-ratio` | `0.8` | Maximum fraction of currently available resources the enabled executor resource manager can grant a query; must be greater than 0 and at most 1. Allocation and layout limits still apply. |
| `--watchdog-max-projected-rows-per-device` | `128000000` | Projected-row capacity guard when the watchdog is enabled; query hints can override the threshold. Raising it permits larger projections, which also need sufficient memory. |
| `--num-reader-threads` | `0` | General FileMgr reader count. Zero, or a value above hardware concurrency, selects hardware concurrency. Separate from bypass-reader fanout and GPU prefetch workers. |

## Stored-data and rollback matrix

| Feature | Changes table/catalog storage? | Older binary behavior | Rollback |
|---|---|---|---|
| Planner rewrites | No | Not applicable | Restart with flag false |
| Bitmap/ranked bitmap joins | No | Not applicable | Restart with flag false |
| Result-reduction pipeline | No | Not applicable | Restart with flag false |
| Prefetch/bypass/jump/peer copy | No | Not applicable | Restart with prior settings |
| Cubin cache | Separate disposable files | Older binary ignores unless configured to same path | Stop server and remove cache files |
| Declarative constraints | Yes, catalog `key_metainfo` | Older planner ignores the new proofs and may not protect dependency DDL | Drop constraints before downgrade if old software will mutate schema/data |
| Advisory manifests | Extra checksummed files | Ordinary table data remains authoritative | Disable flag; retain required manifests for sidecar-only tables |
| Fragment size | Yes, table layout and fragment boundaries | Supported | Create and reload a table with the prior `FRAGMENT_SIZE`; `OPTIMIZE TABLE` does not refragment it |
| Native compression | Per-chunk metadata/payload format | Fails fast on a compressed chunk | `OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='NONE');` under new binary |
| Sidecar-only metadata | FileMgr v3 plus mandatory manifest | Fails fast on table format v3 | Set rollback epochs above zero and checkpoint/optimize under new binary |

Once constraints are declared, keep writable catalog access on a binary that understands
their dependency semantics. An older binary can make schema or data changes that
invalidate those declarations; dropping the constraints before such a downgrade avoids
leaving stale metadata for a later upgrade.

## Correctness validation without a production-data copy

### Working with generated data

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
and multi-fragment cases alongside smaller uniform tables.

### Checks on the target deployment

Generated data is useful for testing optimizer behavior. Enabling trust in production
also depends on checking the actual rows against their declared primary, unique, and
foreign keys. If read-only validation queries are unavailable, trust can remain false
while the independent features are evaluated.

Performance comparisons benefit from the target topology, storage device, filesystem,
compression ratio, fragment size, and output cardinality. The reference measurements can
help choose experiments; measurements on the deployment provide the basis for capacity
planning.

### Strict and fallback validation configurations

Stage 1 is a conservative deployment profile. A separate strict GPU test can enable both
bitmap layouts and the result pipeline, disable both CPU retry paths, and add trust and
partitioned reduction on suitable fixtures:

```text
--enable-experimental-query-rewrites=true
--enable-ranked-bitmap-hashjoin=true
--enable-bitmap-hashjoin=true
--enable-result-reduction-pipeline=true
--enable-partitioned-baseline-gpu-reduction=true
--trust-unenforced-table-constraints=true
--allow-cpu-retry=false
--allow-query-step-cpu-retry=false
```

This profile extends stage 1 and can be compared with the stage 0 reference. It includes
partitioned reduction, which the historical benchmark profile left disabled. Adding it
as a full-suite CI mode is follow-up work described below. To explore its behavior:

- Request GPU execution for queries expected to support it. The execution request and
  retry settings are separate; successful GPU queries can still include CPU planning,
  decompression, or materialization.
- Check declared constraints with trust false before enabling trust. For fixtures with
  intentionally invalid keys, keep trust false so the optimizer evaluates the rows
  without relying on those declarations.
- Include an eligible multi-GPU rowwise baseline-hash workload large enough to select
  partitioned reduction, and confirm that selection. Compare it with the same profile
  and partitioned reduction false. Small or one-GPU fixtures exercise other reducers.
- Keep other experimental input, peer-read, and storage settings explicit and fixed.
  This eight-option block does not turn them on. Test those features separately before
  adding relevant combinations.

Run a separate recovery configuration with both retry options true. Deliberately trigger
a supported CPU fallback, confirm the final result and cleanup, and confirm that the
corresponding strict run exposes the expected failure. CPU-only features and tests whose
purpose is CPU recovery remain useful as separate controls. Keeping these runs distinct
helps distinguish expected CPU behavior from a GPU path that unexpectedly needs recovery.

### Existing automated coverage and its limits

The source tree includes tests that explicitly enable options. Most set C++ globals in
scoped helpers and restore them afterward. That exercises the enabled feature inside
those cases even though server defaults are false; other cases retain their own settings.
The table highlights existing coverage and ways to extend it for a workload. Coverage
of combinations across all 51 documented options remains partial.

| Feature area | Existing coverage in the reviewed revision | Useful follow-up for a workload |
|---|---|---|
| Result pipeline and deferred results | `ExecuteTest` helpers `ResultReductionPipelineTestMode` and `DeferredGpuIntermediateResultSetTestMode` enable the pipeline and disable both CPU retries. Cases cover reductions, retained intermediates, host consumers, and error/cleanup boundaries. | Add the SQL shapes, layouts, device counts, and flag combinations used by the workload. |
| Planner trust and bitmap layouts | `ExecuteTest` has trust-on/off constraint cases and scoped exact/ranked bitmap helpers, including duplicate keys, filtered builds, and cache invalidation. | Validate the deployment's declared keys and check which join layouts its queries select. |
| Input and payload modes | `ExperimentalGpuInputFetchModesPreserveAggregateResults` switches among host mapping, selected-dense fetch, CPU/GPU prefetch, and bypass. Compressed-native tests enable bypass, batching, overlap, and peer exchange; selected cases check buffer residency and decode-workspace cleanup. | Confirm selection of each mode. Peer exchange needs multiple GPUs; GPU decode needs matching libraries. A fallback run can still produce correct results. |
| Native storage and manifests | `FileMgrTest` guards explicitly enable compression, advisory manifests, and sidecar-only metadata. Cases cover persistence, reopen, corruption/recovery handling, and conversion. | Exercise SQL conversion, server restart, upgrade/downgrade, and backup procedures in the deployment environment. |
| Dictionary paths | `StringDictionaryTest` explicitly covers lazy recovery and compares parallel sorted permutations against serial results. | Measure restart and first-use behavior with representative dictionaries. |
| Jump buffers and copy transport | `CudaMgrTest` compares transfers with jump-buffer size zero and a small nonzero size; device-copy tests and a pipelined reduction test exercise copy paths. | Extend coverage to lazy/background allocation, multiple slot counts, and explicit direct/staged/host comparisons. |
| Peer transport primitives | `GpuSharedMemoryTest` cases `GpuPeerAccessRuntime.PeerDmaHonorsProducerEvent` and `RemoteKernelReadsHonorProducerEvent` check ordered transfers/reads on one eligible device pair, and can skip when capabilities are unavailable. | Add end-to-end queries with the temporary ResultSet peer-read flags and check the ordered GPU pairs used by the workload. |
| Partitioned reduction and temporary ResultSet peer-read switches | No test in the reviewed `Tests` tree explicitly enables `g_enable_partitioned_baseline_gpu_reduction`, `g_enable_temporary_resultset_peer_access`, or `g_enable_temporary_resultset_payload_peer_access`. | Add cases that enable each switch and assert selection of its path, building on the existing reduction and peer-primitive tests. |
| Persistent cubin cache and split JIT | No test in the reviewed `Tests` tree explicitly configures the persistent-cubin or split-JIT options. Historical manual/benchmark validation is described separately below. | Add automated checks for restart reuse, invalidation, pruning, and non-default split-JIT settings. |

The `heavydb` server and `ExecuteTest` have different option parsers. At this revision,
`ExecuteTest` does not register the eight server options in the strict block above.
Appending that block to its CI arguments therefore produces an unrecognized-option
error. Two ways to run a broader profile are to add explicit support to the test harness
or use a server integration harness that launches `heavydb` with these options and submits
the workload through a client. Scoped helpers can override a harness-wide default;
recording both the profile and individual test settings helps interpret the coverage.

The checked-in required-checks workflow builds CUDA jobs on CPU runners and runs GPU
tests on the T4 runner labels. Its TSAN jobs build with CUDA off and cover CPU behavior;
GPU paths need the GPU test runs. The strict block is not currently registered as a
separate profile. The normal dependency image selects CUDA 12.9.2, leaving CUDA 13 split
JIT for a separate build. Alongside the runner label, recording the visible GPU count,
native-atomic peer capability, and nvCOMP libraries helps describe what each run can cover.

Useful extensions are explicit enabled and recovery profiles, activation assertions for
the three uncovered switches, and runs on hardware that supports the selected paths.
The following configurations provide complementary comparisons:

| Qualification run | Required environment or fixture |
|---|---|
| Default and CPU behavior | Current binary with explicit defaults; CPU-only build for CPU compatibility/TSAN and CUDA build for GPU defaults. |
| Strict one-GPU execution | CUDA build, GPU-requested supported queries, retries false, valid constraints when trusted; compare exact typed results with the baseline. |
| Partitioned reduction | At least two visible GPUs, eligible rowwise baseline-hash input meeting the entry-storage threshold, plus an assertion that the partitioned reducer ran. |
| Direct peer reads and compressed exchange | Eligible GPU pairs, retained cross-device ResultSets or concurrent compressed-chunk consumers as appropriate; verify actual path use and repeated results. Record unsupported/skip separately from a passing enabled-path test. |
| Codec, persistence, and JIT behavior | Matching codec libraries, disposable native storage and restart/reopen checks; a CUDA 13+ build for split-JIT values other than 1. |

This matrix describes proposed coverage. Completed reference runs are recorded below
with their scope and results.

### Observing feature selection

Three observations help explain a run: the option's setting, the implementation selected,
and whether the result matched the reference. A correct result through a fallback is
useful coverage too; recording that path separately shows which part of the behavior
was exercised.

Depending on the feature, useful observations include:

- Planner and hash joins: inspect the plan and, in engine tests, assert the selected
  hash-table type. Together these describe both the relational transformation and the
  downstream layout.
- Retained results and partitioned reduction: inspect ResultSet device fragments and the
  selected reduction path; profile transfers at the final CPU boundary.
- Hydration and decode: start from controlled CPU/GPU cache misses, confirm the stored
  codec, and inspect buffer residency plus transfer/decode activity. Hot input bypasses
  the disk and hydration paths being measured.
- Peer reads and exchange: establish the visible device pairs and actual pointer/copy
  path, complementing the capabilities reported in the startup matrix.
- Dictionary recovery and manifests: persist data and reopen; distinguish a deferred
  hash, a manifest hit, and the supported eager/metadata-scan fallback.
- Cubin cache and split JIT: compare first compile, same-process reuse, and reuse after
  restart; identify a persistent-cache hit or actual compilation work before attributing
  latency to a setting.
- Storage formats: checkpoint, close/reopen, verify data, and exercise the documented
  reverse conversion. These steps show how the stored format behaves across restarts.

Some selection decisions are not exposed at normal log severity. Existing assertions,
a profiler, or targeted test instrumentation can help identify those paths. Where that
information is unavailable, recording selection as unconfirmed leaves a clear next step
for a later investigation.

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
6. Repeat queries to look for stream-ordering and lifetime races that may appear
   intermittently.
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

The following runtime results come from earlier reference revisions, separate from the
source review noted at the beginning of this guide. Each run covers the configuration
and cases described; the proposed matrix above identifies ways to extend that coverage.

An earlier reference revision completed the full 87-test CTest inventory on 2026-07-22,
with each test invoked through its registered harness: 87 of 87 passed with zero failures
in 9,905.10 seconds (2 hours, 45 minutes, 5.1 seconds). The run covered all three 754-case
`ExecuteTest` configurations, CPU fallback, cancellation, watchdog cleanup,
count-distinct, GPU windows, hash-table caches, compressed native input, FileMgr recovery,
geospatial lazy fetch, Arrow IPC against its managed live server, and multi-instance
execution. That full inventory predates the final JIT, integral-sum, and Arrow
shared-memory commits. Repeating the full inventory after the final rebase is a remaining
validation step alongside the focused tests and benchmark guards.

The reference CTest pass ran serially with `-j1`. Current test registrations do not
declare CTest resource locks, while many harnesses share `Tests/tmp`, Calcite or server
ports, catalog state, or GPUs. Serial execution avoids collisions in those shared
resources. To introduce parallelism, start with isolated tests or give shared resources
unique per-test identities or explicit CTest locks.

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
deployment still needs a successful end-to-end render run on its target host.

These results provide comparison points for validation on your build, driver, GPU
topology, schema, and workload.

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

A fallback can preserve correct results while changing latency and where the work runs.
Tracking fallback counts alongside timing, where the engine exposes them, helps explain
those changes.

Ordinary INFO logs do not currently expose every event in that list. Startup logs include
the basic peer-capability matrix and configured feature values. Required sidecar failures,
manifest validation problems, and cubin cache failures are visible. Detailed nvCOMP
fallback reasons are primarily available at `VLOG(1)`, while individual native-atomic
peer-access rejections and copy fallbacks are not logged at normal severity. For a
controlled qualification run, use Nsight Systems or temporary targeted instrumentation
to measure transfer counts, bytes, decode paths, and fallback behavior. Remove temporary
per-copy or per-fragment diagnostics after the investigation to avoid adding work to the
hot path. Bounded counters are a useful option when continuous fallback telemetry is
needed without verbose per-event logging.

When investigating a result mismatch, keeping the SQL fixed and disabling features by
boundary can help narrow the cause. One useful sequence is:

1. Disable direct temporary ResultSet peer access.
2. Disable partitioned baseline reduction.
3. Disable the result-reduction pipeline.
4. Disable ranked and exact bitmap joins.
5. Disable experimental planner rewrites.
6. Disable trusted constraints.

The first boundary that restores the expected result provides a starting point for a
focused reproduction. Sharing the query, data shape, effective settings, and that
boundary helps collaborators investigate the relevant planner or execution behavior.

## Emergency rollback

For a query-only issue with no new storage formats:

1. Keep the same binary.
2. Restart with all new planner, hash, reduction, input, and peer-access booleans false.
3. Keep both CPU retry options true.
4. Clear only the disposable cubin cache if compilation behavior is suspect.

For a binary downgrade after native compression:

1. Stop writes and take a storage snapshot.
2. Consult the conversion inventory and start the new binary with compression read
   support.
3. Run `OPTIMIZE TABLE t WITH (STORAGE_COMPRESSION='NONE');` for every affected logical table.
   If the inventory is incomplete, rewriting every candidate table covers formats that
   may have been written under earlier configurations.
4. Reopen and validate those tables with the new binary.
5. If any table also uses sidecar-only metadata, complete its conversion below while
   still running the new binary. Start the old binary only after both formats have been
   converted everywhere they were used.

For a binary downgrade after sidecar-only metadata:

1. Complete the compression rewrite and new-binary validation above if needed. In
   particular, remove GDeflate compression before increasing rollback epochs.
2. Under the new binary, set `MAX_ROLLBACK_EPOCHS` to at least 1 for every affected table
   and every table whose sidecar state is uncertain.
3. Run `OPTIMIZE TABLE t;` to force metadata recomputation and checkpoint.
4. Reopen and validate under the new binary.
5. Disable new compression and sidecar creation in the retained new-binary configuration
   so later use cannot reintroduce them. Then start the old binary and retain the snapshot
   until it has opened and queried every converted table.

If a required sidecar manifest is missing or corrupt, recovery needs a consistent
snapshot or a tested procedure that restores the metadata. Changing `filemgr_version`
or constructing a replacement manifest by hand cannot recover the missing information
and can bypass the checks that prevent data corruption.

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
- Bitmap layout selection depends on key-domain and allocation limits, which are useful
  to include in capacity tests for large or sparse joins.
- Fragment size affects partial-result count, GPU memory pressure, and work distribution
  under the GPU-retained pipeline. Test with production-like fragment sizes and skew.
- For deployments using row-level security or rendering, include end-to-end checks of
  the actual policies and render queries alongside the relational benchmarks.
- Hot execution, restart hydration, shutdown latency, and client result serialization are
  separate measurements and should remain separate performance objectives.
- Arrow shared-memory query results are local-only and rely on System V shared-memory
  lifecycle cleanup. Use Arrow wire or Thrift across a host boundary.

## Implementation references

For a closer look at a feature, the links below point to its implementation and tests.
The file and function names provide starting points in the source revision listed at
the beginning of this guide. When comparing another revision, these locations can help
trace changes in defaults, selection conditions, and fallback behavior.

| Subject | Starting points |
|---|---|
| Server option registration, validation, and logging | [CommandLineOptions.cpp](../ThriftHandler/CommandLineOptions.cpp), [CommandLineOptions.h](../ThriftHandler/CommandLineOptions.h), and [SystemParameters.h](../Shared/SystemParameters.h); defaults also come from the global definitions in the owning components below. |
| Planner metadata and trust | [HeavyDBParser.java](../java/calcite/src/main/java/com/mapd/calcite/parser/HeavyDBParser.java), [HeavyDBTable.java](../java/calcite/src/main/java/com/mapd/calcite/parser/HeavyDBTable.java), and [RelAlgExecutor.cpp](../QueryEngine/RelAlgExecutor.cpp). |
| Bitmap selection and cache limits | [HashJoin.cpp](../QueryEngine/JoinHashTable/HashJoin.cpp), [DataRecycler.h](../QueryEngine/DataRecycler/DataRecycler.h), and [Execute.cpp](../QueryEngine/Execute.cpp). |
| Result pipeline, partitioned eligibility, and input prefetch | [Execute.cpp](../QueryEngine/Execute.cpp): `can_use_result_reduction_pipeline`, `try_reduce_baseline_hash_result_sets_partitioned_on_gpu`, and the input-fetch code; [RelAlgExecutor.cpp](../QueryEngine/RelAlgExecutor.cpp) for retained intermediates and CPU retries. |
| Payload selection, deferred fetch, and temporary peer reads | [NativeCodegen.cpp](../QueryEngine/NativeCodegen.cpp), [ColumnFetcher.cpp](../QueryEngine/ColumnFetcher.cpp), and [ResultSet.cpp](../QueryEngine/ResultSet.cpp). |
| Native bypass, batching, overlap, and compressed exchange | [CpuBufferMgr.cpp](../DataMgr/BufferMgr/CpuBufferMgr/CpuBufferMgr.cpp): `fetchBuffer`, `fetchBuffers`, and the compressed transfer helpers. |
| Compression, sidecar persistence, and rewrite SQL | [FileBuffer.cpp](../DataMgr/FileMgr/FileBuffer.cpp), [FileMgr.cpp](../DataMgr/FileMgr/FileMgr.cpp), [ParserNode.cpp](../Parser/ParserNode.cpp), and [DataMgr/CMakeLists.txt](../DataMgr/CMakeLists.txt) for codec availability. |
| Jump buffers and peer capabilities/transport | [JumpBufferTransferMgr.cpp](../CudaMgr/JumpBufferTransferMgr.cpp) and [CudaMgr.cpp](../CudaMgr/CudaMgr.cpp): `canAccessPeerMemoryFromKernel`, `ensurePeerAccessToDevicePtr`, and the copy helpers. |
| Dictionary recovery and sorting | [StringDictionary.cpp](../StringDictionary/StringDictionary.cpp) and [StringDictionaryProxy.cpp](../StringDictionary/StringDictionaryProxy.cpp). |
| Cubin cache and split JIT | [NativeCodegen.cpp](../QueryEngine/NativeCodegen.cpp) and [NvidiaKernel.cpp](../QueryEngine/NvidiaKernel.cpp). |
| Automated coverage and harness registration | [ExecuteTest.cpp](../Tests/ExecuteTest.cpp), [FileMgrTest.cpp](../Tests/FileMgrTest.cpp), [StringDictionaryTest.cpp](../Tests/StringDictionaryTest.cpp), [CudaMgrTest.cpp](../Tests/CudaMgrTest.cpp), [GpuSharedMemoryTest.cpp](../Tests/GpuSharedMemoryTest.cpp), [Tests/CMakeLists.txt](../Tests/CMakeLists.txt), and [pr-required-checks.yml](../.github/workflows/pr-required-checks.yml). |

A useful path through the code is from option registration to the selection check, then
to the chosen implementation and its tests. For a workload comparison, recording the
configuration, selected path, result checks, and timing helps others reproduce the run
and understand where an improvement came from.
