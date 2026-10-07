---
name: Fix shared CPU pool accounting
overview: Close the missing joint constraint that lets CPU result memory and pinned chunk memory each pass an independent pre-flight check and then together overcommit the shared CPU buffer pool counter, which aborts the server via CHECK_LE. Also fix the grant-policy slot collision that removes the per-query result-memory cap and the auto-shrink retry that sizes result memory against the entire pool.
todos:
  - id: subtype
    content: Add ResourceSubtype::CPU_RESULT_MEM_IN_POOL and wire both subtype/type mapping functions so pool-backed result memory is a first-class resource under CPU_BUFFER_POOL_MEM; point cpu_result_mem_resource_type at it
    status: completed
  - id: index-fix
    content: Fix allocate_resources and deallocate_resources to index allocated_resources_ with .resource_subtype instead of .resource_type
    status: completed
  - id: cap-headroom
    content: Add get_max_cpu_result_mem_grant_per_request(chunk_headroom) helper; move the cpu_result_mem max-grant block in calc_min_max_resource_grants_for_request to after the chunk-gating block and use the helper there and in can_currently_satisfy_request_impl
    status: completed
  - id: joint-check
    content: Add the joint constraint (grant result mem + chunk bytes + allocated <= total) to both branches of can_currently_satisfy_chunk_request, and subtract the grant's result mem from remaining_buffer_mem_for_memory_level in the gated branch of determine_dynamic_resource_grant
    status: completed
  - id: autoshrink
    content: Change the auto-shrink retry to size slots against the pool's effective per-request result-mem cap minus chunk headroom, via a new public accessor, instead of the whole pool
    status: completed
  - id: survivable-throw
    content: Replace CHECK(can_satisfy_request) in allocate_resources with a throw, add a try/catch at the process_queue_loop call site routing to mark_request_error while still calling wake_request_by_id, and retighten the post-commit CHECK_LE to the type-level total
    status: completed
  - id: collateral
    content: Fix ResourceSubtypeStrings ordering (switch instead of array), add duplicate-policy-slot guard in init(), replace the "CPU result memory" string match with a typed error kind, fix the double-incremented CPU_BUFFER_POOL_MEM outstanding counter, and report allocated_cpu_result_mem from the new subtype
    status: completed
  - id: tests
    content: Update the three existing tests that assert the buggy cap, and add regression tests for the joint pool-backed path (non-gated and gated), the policy cap, auto-shrink headroom, survivable allocation throw, and subtype string round-trip
    status: completed
isProject: false
---

# Fix shared CPU buffer pool accounting between result memory and pinned chunks

## Implementation status

All seven approach steps are implemented, across four commits on
`simoneves/fix_shared_cpu_buffer_pool_accounting`:

| Commit | Subject | Contents |
| --- | --- | --- |
| `ee65eed977` | Fix mislabeled resource subtype strings in startup logs | Step 7's string `switch` and dead-code removal, plus the `ResourceSubtypeMapping` test |
| `65e441f523` | Carry a typed error kind on `ExecutorResourceMgrError` | Step 7's typed error kind: `ExecutorResourceMgrErrorKind`, the field on `RequestStats`, the typed catches |
| `ed20b3da62` | Stop double-counting pool-backed requests in buffer pool stats | Step 7's outstanding-request counter fix |
| `581e4f1547` | Fix shared CPU buffer pool accounting between result memory and chunks | Steps 1-6, the remaining collateral from step 7, and the rest of the test work |

The first two commits are behavior-preserving and reviewable on their own; the third is
the only user-visible metric change. The fourth is the crash fix and **cannot be
subdivided further**, because three of its parts are each unsafe to land alone:

- The duplicate-policy `CHECK` detects a collision that exists before the new subtype, so on its own it would make the server fail at startup in every pool-backed configuration.
- Restoring the per-query result memory ratio without the auto-shrink resizing makes the retry recompute the slot count it was given, hit the equality guard, and turn a currently-succeeding query into a hard error.
- The joint constraint, the headroom reservation and the auto-shrink resizing share the cap helper, so separating them means writing an interim helper that is deleted one commit later.

The split was verified by reconstructing each commit forward from `HEAD` and confirming
that the cumulative diff was byte-identical to the diff taken before splitting, so no
hunk was dropped, duplicated or reworded. `ExecutorResourceMgr.h` and
`ResourceGrantPolicy.cpp` are fully consumed by the first two commits and do not appear
in the crash fix at all.

`ExecutorResourceMgrTest` passes, 20 of 20, in a release container build. One regression
surfaced on the first run and has been folded back into the commits that introduced it;
see [The regression the first test run found](#the-regression-the-first-test-run-found).
One [verification gap](#verification-gaps) remains.

## Root cause

Verified against the source before the fix; all line numbers cited in this section and
the next refer to that pre-fix state and have since shifted.

With `use-cpu-mem-pool-for-output-buffers=true` (the default, `g_use_cpu_mem_pool_for_output_buffers{true}` at [QueryEngine/Execute.cpp](QueryEngine/Execute.cpp):183), [ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):725-730 maps CPU result memory onto `ResourceSubtype::PINNED_CPU_BUFFER_POOL_MEM` — the same counter slot used for pinned input chunks. Inside a single `allocate_resources` call, under one write lock, two independent pre-flight checks validate each contributor against the same pool total, and then both are committed additively into that one counter.

```mermaid
flowchart TD
    preflight["allocate_resources:976 can_currently_satisfy_request_impl"]
    checkA["check_request_against_policy:489
    cpu_result_mem + allocated <= total"]
    checkB["can_currently_satisfy_chunk_request:656
    chunk_bytes_not_in_pool + allocated <= total"]
    commitA["line 984: pinned += cpu_result_mem"]
    commitB["line 714/721: pinned += chunk_bytes"]
    assertion["line 757: CHECK_LE(pinned, total) -- SIGABRT"]

    preflight --> checkA
    preflight --> checkB
    checkA -->|passes| commitA
    checkB -->|passes| commitB
    commitA --> commitB
    commitB --> assertion
```

Neither check includes the other's contribution, so the assertion fires exactly when `allocated + cpu_result_mem + chunk_bytes > total` while each term individually fits. No concurrency, no accounting drift, and no gated/non-gated mismatch is required.

The same defect has a **second abort site** the previous plan did not identify: the gated (`buffer_mem_gated_per_slot`) branch at [ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):667-670 asserts `get_total_allocated_buffer_pool_mem_for_level(...) + buffer_mem_for_given_slots <= total` *after* line 984 has already folded result memory into that same total. The gated `remaining_buffer_mem_for_memory_level` computed at [ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):934-941 likewise ignores the result memory the same grant is about to commit.

### Corroboration from the reported values

- The two reported assertion values, 508,474,188,976 and 480,634,599,024 bytes, both exceed the executor pool reserve of 459,024,629,760 bytes.
- The larger value also exceeds the hard ceiling of the whole CPU buffer pool (483,183,820,800 bytes), so the counter provably is not tracking resident pinned chunks alone — it is an aggregate of two independently capped quantities.
- Four of the five occurrences logged zero allocation attempts, consistent with an abort during admission rather than during allocation.

## Corrections to the previous plan's analysis

These three points change what the fix has to do, so they are worth stating explicitly.

**1. The `0.8 x remaining = 342.00 GiB` decomposition is mechanically unreachable.** `min_resource_grant.cpu_result_mem` is assigned the *full* request at [ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):429, and `max_resource_grant.cpu_result_mem` equals the full request whenever the request is under the per-request cap (otherwise line 341 or 522 throws). So min always equals max, and `determine_dynamic_single_resource_grant`'s `std::max(min_resource_requested, ...)` at [ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):876-879 always returns the full request. **The `max_available_resource_use_ratio` backoff never applies to CPU result memory at all.** Granted result memory is always exactly what was asked for. The exact result-memory/chunk split of the reported values cannot be recovered from the supplied logs and should not be asserted.

**2. Therefore the auto-shrink retry is load-bearing, not a secondary cleanup.** Since nothing else can reduce result memory, the only mechanism that resizes an over-large request is the retry at [ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):120-170. It sizes slots as `cpu_buffer_pool_size_bytes / actual_buf_size_per_slot` — i.e. it targets **100% of the pool** for result memory alone, reserving nothing for chunks and ignoring current occupancy. Any non-zero chunk working set then guarantees the overrun. The existing tests pass only because at kilobyte-scale buffer sizes the `heavyai::get_page_size()` rounding leaves large slack; at realistic per-slot buffer sizes that slack nearly vanishes.

**3. The policy-overwrite bug is confirmed, and three existing tests encode it.** [ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):760-776 puts the result-memory policy and the pinned-pool policy in the same vector targeting the same subtype; `init` at [ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):70-80 assigns by plain overwrite in vector order, so the later pinned-pool ratio wins. In production that ratio is hardcoded to `1.0` ([Execute.cpp](QueryEngine/Execute.cpp):5493), discarding the configured `per_query_max_cpu_result_mem_ratio` of 0.8. Proof in-tree: [Tests/ExecutorResourceMgrTest.cpp](Tests/ExecutorResourceMgrTest.cpp):1195 asserts a per-query limit of `48.0 KB` for a 65536-byte pool, which is `0.75 x 65536` (the test's pinned ratio), not `0.8 x 65536`. Same at line 1253 (`750 bytes` = `0.75 x 1000`) and line 1319 (`96.0 KB` = `0.75 x 131072`).

**4. Resolving the question the previous plan left open for an owner.** Line 429 should be left as-is for now. Changing it to `request_info.min_cpu_result_mem` would let the dynamic grant scale result memory independently of `cpu_slots`, but the executor sizes per-slot output buffers from the granted slot count, so a result-memory reservation unmoored from the slot count is incoherent. The correct structural answer is to make CPU result memory a slot-dependent resource (see Out of scope) — not to change line 429 in isolation.

## Approach (as implemented)

All seven steps are committed; see [Implementation status](#implementation-status) for
which commit carries which. Where the implementation departed from the plan as written,
the step says so and the reason is collected under
[Deviations](#deviations-from-the-plan-as-written).

### 1. Make pool-backed result memory a first-class subtype

In [ExecutorResourceMgrCommon.h](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgrCommon.h), append an enumerator so indices 0-7 stay stable:

```cpp
enum class ResourceSubtype {
  // ... existing 0-7 unchanged ...
  CPU_RESULT_MEM_IN_POOL = 8,
  INVALID_SUBTYPE = 9,
  NUM_RESOURCE_SUBTYPES = 9,
};
```

Then in [ExecutorResourcePool.h](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.h): `map_resource_subtype_to_resource_type(CPU_RESULT_MEM_IN_POOL)` returns `ResourceType::CPU_BUFFER_POOL_MEM`, and `map_resource_type_to_resource_subtypes(CPU_BUFFER_POOL_MEM)` returns `{PINNED_CPU_BUFFER_POOL_MEM, PAGEABLE_CPU_BUFFER_POOL_MEM, CPU_RESULT_MEM_IN_POOL}`. Set `cpu_result_mem_resource_type` in `generate_executor_resource_mgr` to `{ResourceType::CPU_BUFFER_POOL_MEM, ResourceSubtype::CPU_RESULT_MEM_IN_POOL}`.

Because `get_allocated_resource_of_type` sums over that list ([ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):160-168), type-level totals keep including result memory while the two quantities become separately observable. This also dissolves the policy collision for free: the result-memory policy lands in slot 8 and the pinned-chunk policy in slot 4.

### 2. Fix the subtype/type index confusion

[ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):984-985 and :1055-1056 index the per-*subtype* array `allocated_resources_` with `cpu_result_mem_resource_type_.resource_type`. This works today only because the first five enumerators of both enums coincide; with the new subtype it must be `.resource_subtype`. Leave genuine type-level uses (`get_allocated_resource_of_type`, `get_total_resource`, concurrency policies, outstanding-request counters) alone.

### 3. Reserve chunk headroom inside the per-request result-memory cap

Three private helpers in `ExecutorResourcePool`, used at both existing throw sites, so no new exception type or new throw site is needed and the error message is automatically correct:

```cpp
size_t ExecutorResourcePool::get_max_cpu_result_mem_grant_per_request(
    const size_t chunk_headroom_bytes) const {
  size_t max_grant =
      get_max_resource_grant_per_request(cpu_result_mem_resource_type_.resource_subtype);
  if (cpu_result_mem_resource_type_.resource_type != ResourceType::CPU_BUFFER_POOL_MEM) {
    return max_grant;  // result mem has its own pool; no headroom needed
  }
  max_grant = std::min(max_grant, get_total_resource(ResourceType::CPU_BUFFER_POOL_MEM));
  return max_grant > chunk_headroom_bytes ? max_grant - chunk_headroom_bytes : size_t(0);
}
```

The `std::min` against the pool total was not in the plan as written and is required. A per-request ratio above 1.0 is permitted by the config and is exercised by the test suite, so the subtype cap can exceed the pool total. Without the clamp, a request whose result memory plus chunks can never fit would pass the per-request cap and then wait in the queue indefinitely rather than being rejected. The clamp also keeps the cap below the pool total, which is why the two pre-existing `CHECK_LE`s in `can_currently_satisfy_chunk_request` could be left alone (see [Deviations](#deviations-from-the-plan-as-written)).

`chunk_headroom_bytes` comes from a second helper, `calc_chunk_headroom_for_request(chunk_request_info, min_cpu_slots)`: `0` for a GPU chunk request, `chunk_request_info.total_bytes` for a non-gated CPU request, and `max_bytes_per_kernel * min_cpu_slots` (gated) — the full `total_bytes` would be far too conservative in the gated case, where only a slot's worth is pinned at a time. A third helper, the public `get_max_cpu_result_mem_grant_for_request(const RequestInfo&)`, composes the two; it is what step 5 calls.

Applied at:
- `calc_min_max_resource_grants_for_request`, at the existing `cpu_result_mem` max-grant block. **The block was not moved.** `calc_chunk_headroom_for_request` re-evaluates the gating predicate itself (`bytes_scales_per_kernel && total_bytes > pinned_cap`) rather than reading `buffer_mem_gated_per_slot` off a partially-populated grant, so the reorder the plan called for turned out to be unnecessary. The cost is that the gating predicate is now expressed in two places; they must stay in sync, and the helper carries a comment saying so.
- `can_currently_satisfy_request_impl` — this is the throw that actually fires for an over-large request, since `min_resource_grant.cpu_result_mem` is the full request.

Both sites already throw `QueryNeedsTooMuchCpuResultMem`, which is what triggers the auto-shrink retry, so this change makes the retry fire with the right target instead of after the fact.

### 4. Enforce the joint constraint in the dynamic (pool-state) checks

In `can_currently_satisfy_chunk_request` ([ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):622-658), include the result memory this same grant will commit, in **both** branches:

```cpp
const size_t result_mem_from_pool =
    (cpu_result_mem_resource_type_.resource_type == ResourceType::CPU_BUFFER_POOL_MEM &&
     chunk_request_info.device_memory_pool_type == ExecutorDeviceType::CPU)
        ? min_resource_grant.cpu_result_mem
        : size_t(0);

// gated branch (replaces line 645-646)
return allocated_buffer_mem_for_memory_level + result_mem_from_pool +
       min_buffer_pool_mem_required <= total_buffer_mem_for_memory_level;

// non-gated branch (replaces line 656-657)
return chunk_bytes_not_in_pool + result_mem_from_pool +
       allocated_buffer_mem_for_memory_level <= total_buffer_mem_for_memory_level;
```

Keying off `.resource_type` rather than the new subtype keeps this correct independently of step 1. Because `can_currently_satisfy_request_impl` is used both by `determine_dynamic_resource_grant` (admission) and by `allocate_resources` (pre-commit, same lock), one edit closes the hole at both call sites.

Also subtract the grant's own result memory from `remaining_buffer_mem_for_memory_level` in the gated branch of `determine_dynamic_resource_grant` ([ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):934-941), so `max_grantable_mem` cannot be sized against memory the same grant is about to take.

### 5. Fix the auto-shrink retry to target the capped headroom

This step is **mandatory and must land with step 1**, not after it. Restoring the 0.8 cap without it would make the retry recompute the same slot count, hit the `adjusted_cpu_slots == request_info.cpu_slots` guard at [ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):145-150, and turn a currently-succeeding query into a hard `std::runtime_error`.

Replace [ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):135-136:

```cpp
const auto adjusted_cpu_slots = cpu_buffer_pool_size_bytes / actual_buf_size_per_slot;
```

with a slot count derived from the pool's own effective per-request cap minus chunk headroom:

```cpp
const auto max_result_mem_bytes =
    executor_resource_pool_.get_max_cpu_result_mem_grant_for_request(request_info);
const auto adjusted_cpu_slots = max_result_mem_bytes / actual_buf_size_per_slot;
```

Of the two options the plan offered, this is the dedicated accessor rather than exposing `get_max_resource_grant_per_request(ResourceSubtype)`. No `ExecutorResourceMgr` passthrough was needed, since the pool is a direct member. The manager and the pool now agree on the target by construction rather than by duplicated arithmetic, which is what makes the shrink provably strict (see [Deviations](#deviations-from-the-plan-as-written)). The existing `adjusted_cpu_slots <= 0` guard converts the genuinely impossible case into the clean `std::runtime_error` it already produces.

### 6. Make the abort sites unreachable, and make any throw survivable

With step 4 in place, `allocate_resources` already performs a complete joint pre-flight at lines 976-978 **before any mutation**, so no restructuring of `add_chunk_requests_to_allocated_pool` is required — which is a simplification over the previous plan, and avoids the partial-mutation leak it was designed to work around. Specifically:

- Non-gated: pre-flight guarantees `chunk_bytes_not_in_pool + cpu_result_mem + allocated <= total`; the loop at lines 711-727 adds *at most* `chunk_bytes_not_in_pool`. So line 757 is unreachable.
- Gated: pre-flight guarantees `allocated + cpu_result_mem + buffer_mem_for_given_slots <= total`, which is exactly what line 667 asserts after line 984 commits. Unreachable.

Remaining work here:

- Replace `CHECK(can_satisfy_request)` at line 978 with a thrown `ExecutorResourceMgrError`. This is the check-then-commit guard; nothing has been mutated at that point. The plan noted that its message must not contain the substring `"CPU result memory"` or `request_resources` would attempt another auto-shrink on it; the typed error kind in step 7 removes that constraint, since the throw defaults to `ExecutorResourceMgrErrorKind::OTHER`.
- Wrap the `allocate_resources` call in `process_queue_loop` ([ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):442-446) in a `try`/`catch` routing to the existing `mark_request_error(chosen_request_id, ...)`, mirroring the `choose_next_request` handling at lines 421-427. Without this, any throw escapes the resource manager's own `std::thread` and hits `std::terminate`. **The `wake_request_by_id(chosen_request_id)` call at line 447 must still run on the error path**, or the waiting query thread blocks forever. The existing recovery path is then exercised: the waiting thread sees `request_stats.error`, throws, and `handle_resource_stat_error` removes the request from the `EXECUTING` stage.
- Keep the post-commit `CHECK_LE` at line 757 as a now-provable invariant, but change it to compare the type-level allocation (`get_total_allocated_buffer_pool_mem_for_level`) rather than the pinned subtype alone, since that is the real invariant once result memory lives in its own subtype.

### 7. Collateral correctness fixes in the same module

- **`ResourceSubtypeStrings` is misordered.** Indices 5 and 6 are swapped relative to the enum ([ExecutorResourceMgrCommon.h](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgrCommon.h):139-158), so pageable CPU memory logs as `pinned_gpu_buffer_pool_mem` and vice versa. This feeds `ResourceGrantPolicy::to_string()` ([ResourceGrantPolicy.cpp](QueryEngine/ExecutorResourceMgr/ResourceGrantPolicy.cpp):54), which `log_parameters()` prints at startup — so operator-facing startup logs describing per-request CPU grant caps are mislabeled as GPU. Replace the array with a `switch` so it cannot drift, and add the new subtype.
- **Guard against future slot collisions.** In `ExecutorResourcePool::init` ([ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):70-80), `CHECK` that no two supplied policies target the same subtype before assigning. A startup `CHECK` is right here: this is a programming error, not a runtime condition.
- **Replace the fragile string match.** `e.getErrorMsg().find("CPU result memory")` at [ExecutorResourceMgr.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourceMgr.cpp):118-119 should become a typed error kind carried on `ExecutorResourceMgrError`, since exceptions are stringified at lines 50-53. This becomes materially more important once step 6 adds a second throw out of the allocation path.
- **Double-counted outstanding request counter.** When pool-backed, `cpu_result_mem_resource_type_.resource_type` is `CPU_BUFFER_POOL_MEM`, so `allocate_resources` increments `outstanding_per_resource_num_requests_[CPU_BUFFER_POOL_MEM]` twice for one request (lines 997-1002 and 1003-1008). It is symmetric with `deallocate_resources`, so it is not a leak, but it inflates `ResourcePoolInfo::outstanding_cpu_buffer_pool_mem_requests`, which is surfaced through the executor-stats system table via [InternalExecutorStatsDataWrapper.cpp](DataMgr/ForeignStorage/InternalExecutorStatsDataWrapper.cpp). Fixed with an `inline bool counts_as_cpu_buffer_pool_chunk_request(resource_grant, chunk_request_info)` predicate in [ExecutorResourcePool.h](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.h), which suppresses the chunk-side increment when the request's result memory has already been counted against that type. It gates the increment in both `allocate_resources` and `deallocate_resources`, keeping them symmetric.
- **`ResourcePoolInfo` reporting.** `allocated_cpu_result_mem` reported `get_allocated_resource_of_type(CPU_BUFFER_POOL_MEM)` — the whole pool allocation — when pool-backed. Now reports `get_allocated_resource_of_subtype(cpu_result_mem_resource_type_.resource_subtype)`, which is the new subtype when pool-backed and `CPU_RESULT_MEM` otherwise. `EnableCPUBufferPool` passes an empty chunk request, so its expectations are unaffected.
- **Dead code.** `get_requested_chunks_not_in_pool` ([ExecutorResourcePool.cpp](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.cpp):584) had no callers and has been removed, declaration and definition, so it cannot drift from `get_chunk_bytes_not_in_pool`.

## Tests

The pool-backed path combining result memory with a chunk request had no coverage, which is why this shipped: `gen_resource_mgr_with_defaults` defaults `enable_cpu_buffer_pool` to `false` ([Tests/ExecutorResourceMgrTest.cpp](Tests/ExecutorResourceMgrTest.cpp):56-57), `default_chunk_request_info` is empty (line 41), and the one test that passed `true` (`EnableCPUBufferPool`) used that empty chunk info. Every new test below exercises that combination.

Two helpers were added to support the new cases: `gen_pool_backed_resource_mgr(cpu_buffer_pool_mem, num_cpu_slots)`, which builds a manager with `use_cpu_mem_pool_for_output_buffers` on and no separate result-memory pool, and `gen_cpu_chunk_request_info(num_chunks, bytes_per_chunk, bytes_scales_per_kernel)`, which builds a chunk request with a chosen total and per-kernel maximum.

Existing tests updated, because they asserted the buggy cap:
- `AdjustNumCPUSlotForCPUGroupbyQuery`: limit `48.0 KB` to `50.42 KB` (`0.75 x 65536` to `ceil(0.8 x 65536) - 796`).
- `AdjustedNumCPUSlotBecomeZero`: limit `750 bytes` to `784 bytes` (`0.75 x 1000` to `0.8 x 1000 - 16`). The headroom is 16 here rather than the full 796, because this case does take the gated path. The test's second half still reaches the `adjusted_cpu_slots <= 0` guard.
- `IdentitalAdjustedNumCPUSlot` was **renamed to `AdjustedNumCPUSlotStrictlyShrinks`** and rewritten. It was asserting the `adjusted_cpu_slots == request_info.cpu_slots` guard, which step 5 makes unreachable. Rejection requires `requested_slots x per_slot > cap`, and the retry computes `floor(cap / buf_per_slot)` with `buf_per_slot >= per_slot`, so the adjusted count is now strictly smaller whenever the original was rejected. The guard remains in the code as a defensive stop against unbounded recursion; the test now asserts the shrink and the pool invariant instead.

New tests added:
- `CpuResultMemAndChunksCompeteForPool` — the field regression. Pool-backed, result memory and chunks each under their limits but jointly over the pool. First block asserts a catchable rejection and that the pool is left untouched; second block asserts that a request which does fit jointly is admitted and that both contributors are accounted against the one pool (`800000` of a `1048576` byte pool).
- `CpuResultMemAndGatedChunksCompeteForPool` — the same competition through the gated path, covering the line 667 abort site. Asserts slots gated to 1 and `825000` bytes allocated.
- `PoolBackedCpuResultMemUsesResultMemRatio` — covers the overwrite bug directly: `800000` bytes is admitted (under `0.8 x pool`) where the pinned ratio would have rejected it, and `900000` is rejected.
- `AdjustNumCPUSlotReservesChunkHeadroom` — the retry yields 4 slots, not the 8 requested, because the cap reserves the request's `262144` bytes of chunks.
- `ResourceSubtypeMapping` — asserts each subtype's string explicitly (locking down the ordering fix) and that every subtype rolls up under a type whose subtype list contains it, which is the invariant a future added subtype would break.

The plan's concurrent-requests invariant case was dropped as redundant: each admission case above asserts `allocated <= total` on the shared pool, and the defect never required concurrency to reproduce.

## The regression the first test run found

Three tests failed on the first container run — `AdjustNumCPUSlotForCPUGroupbyQuery`,
`AdjustedNumCPUSlotStrictlyShrinks` and `AdjustNumCPUSlotReservesChunkHeadroom` — all
with the same cause, and all in the source rather than the tests.

`process_queue_loop`'s handler for a throw out of `choose_next_request` recorded the
error with `mark_request_error(id, msg)`, dropping the kind. `RequestStats::error_kind`
therefore stayed `OTHER`, and by the time `request_resources` decided whether to retry
with fewer CPU slots the information was gone, so the error propagated to the caller
instead. The retry never fired. Step 7's typed error kind introduced this: the previous
gate matched a substring of the *message*, which that handler does forward.

What makes this worth recording is why only three of four auto-shrink tests caught it.
There are two paths to the same retry, and they had different fidelity:

| Throw site | Thread | Kind survives? |
| --- | --- | --- |
| `calc_min_max_resource_grants_for_request` | caller's, inside `request_resources_with_timeout` | Yes — the typed catch sets the kind directly, never touching `RequestStats` |
| `can_currently_satisfy_request_impl` | resource manager's queue thread | No — routed through `mark_request_error`, which dropped it |

`AdjustedNumCPUSlotBecomeZero` passes through the first path and so kept working, which
is exactly why the gap was invisible until the cases that take the second path existed.
Both paths now carry the kind.

The fix is one forwarded argument, plus removal of the defaulted `error_kind` parameter
on `mark_request_error`, so that omitting it is a compile error rather than a silent
`OTHER`. A default that quietly discards information at a call site which holds that
information is what let the omission through in the first place.

It had to be folded into **two** commits, not one. The crash fix commit adds its own
`mark_request_error` call site in the `std::exception` fallback of the `try`/`catch` it
installs in `process_queue_loop`, so with the default removed it would not compile unless
that call site is fixed in the same commit. The declaration change and the
`choose_next_request` handler went into the typed-error-kind commit; the explicit `OTHER`
went into the crash fix commit. Arity of the declaration, the definition and every call
site was checked at each of the four commits so the history stays bisectable.

## Deviations from the plan as written

Four, all recorded in context above and collected here:

1. **The per-request result-memory cap is clamped to the pool total** (step 3). Not in the plan. Needed because a configured per-request ratio above 1.0 is legal, and without the clamp an unsatisfiable request would wait in the queue forever instead of being rejected.
2. **The two pre-existing `CHECK_LE`s in `can_currently_satisfy_chunk_request` were left unchanged.** Strengthening them to include `result_mem_from_pool` was tried first and reverted: with a per-request ratio above 1.0 the cap could exceed the pool total and fire the assertion. The clamp in item 1 is the correct place to enforce that, and it makes strengthening these unnecessary.
3. **The `cpu_result_mem` max-grant block was not moved** (step 3). `calc_chunk_headroom_for_request` re-derives the gating predicate instead of reading it off a partially-populated grant.
4. **`IdentitalAdjustedNumCPUSlot` was rewritten, not re-stringed** (Tests). The guard it covered is unreachable once the retry and the pool share a limit.

## Verification gaps

One remains. The other is closed:

- ~~**The three updated expected byte strings are hand-computed**~~ — **resolved.** The first test run reported limits of `50.42 KB`, `101.62 KB` and `563.20 KB`, matching the hand-computed values exactly, which confirms both the `format_num_bytes` rounding behaviour and the cap-minus-headroom arithmetic. The allocation totals asserted by the new tests also hold.
- **The `allocate_resources` pre-commit throw has no test.** The plan called for one. It is not reachable from the public API without a test seam, because step 6 makes the pre-flight consistent with admission by construction — which is the point of the change, but does leave the `process_queue_loop` `try`/`catch` and the preserved `wake_request_by_id` uncovered. Adding a seam (for example a test-only hook that shrinks the pool between admission and allocation) would cover it, and is worth doing if this path is ever modified again. Note that the regression above was in the sibling handler a few lines up, in the same method, and was caught only because three unrelated tests happened to depend on it — so this method's error paths have a demonstrated history of going untested.

## Behavior changes and risks

- Restoring the 0.8 result-memory cap is a real behavior change: queries that previously received up to 100% of the pool are now capped and will auto-shrink to more slots' worth of smaller buffers, or fail with a clear error instead of crashing the server. This is the configured and documented intent (and what the startup log already claims), but it belongs in release notes.
- Adding a subtype changes `NUM_RESOURCE_SUBTYPES` from 8 to 9. The `std::array`s in [ExecutorResourcePool.h](QueryEngine/ExecutorResourceMgr/ExecutorResourcePool.h) are sized by `ResourceSubtypeSize`, so they grow automatically. `init_max_resource_grants_per_requests` does assign a default `UNLIMITED` policy to `CPU_RESULT_MEM_IN_POOL` in the non-pool-backed configuration, since the subtype maps to `CPU_BUFFER_POOL_MEM` and that resource is valid there — but this is inert: nothing allocates into the slot when `cpu_result_mem_resource_type_.resource_subtype` is `CPU_RESULT_MEM`, so it stays zero and the type-level sum over the three CPU buffer pool subtypes is unaffected.
- Subtracting chunk headroom from the per-request result-memory cap makes admission stricter. A query with a very large chunk working set and `output_buffers_reusable_intra_thread == false` (so no auto-shrink is possible) will now be rejected where it previously crashed the server — correct, but a user-visible change from "server restart" to "query error".
- The throw replacing the `CHECK` in `allocate_resources` is on the resource manager's queue thread. Step 6's `try`/`catch` and the preserved `wake_request_by_id` are not optional, and are the part of this change with no test coverage.

## Out of scope

- **The structural follow-up.** CPU result memory should be modeled as a resource *dependent on* `cpu_slots`, using the `calc_max_dependent_resource_grant_for_request` / `buffer_mem_gated_per_slot` machinery that already exists for pageable chunk memory. Today the pool grants `cpu_slots` and `cpu_result_mem` independently, so with `output_buffers_reusable_intra_thread` it reserves `per_slot x requested_slots` while only granting `granted_slots` — a systematic over-reservation, and the reason an exception-driven, string-matched retry outside the lock exists at the manager level at all. Doing this properly would let the pool scale slots to fit under the lock and would make the retry in `request_resources` unnecessary. It also subsumes the `min_resource_grant.cpu_result_mem` question at line 429. Materially larger blast radius; should be its own change.
- The other fatal signatures in the reported incident (a GPU illegal-address fault, a `StringDictionary` bounds check with an out-of-range id, a SIGSEGV, and a `RelAlgOptimizer` check) have independent causes and need separate investigation. This plan addresses only the `ExecutorResourcePool` aborts.
- Latent `CHECK`s elsewhere in the same file that are currently proven safe by callers but remain abort-on-release (for example lines 644 and 651) are left alone; converting them is a broader hardening pass.
