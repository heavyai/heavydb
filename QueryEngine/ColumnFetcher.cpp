/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryEngine/ColumnFetcher.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/ArrayNoneEncoder.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "Shared/Intervals.h"
#include "Shared/likely.h"
#include "Shared/scope.h"
#include "Shared/sqltypes.h"

struct ResultSetDeviceColumnTransfer {
  const int8_t* buffer{nullptr};
  size_t num_bytes{0};
  bool owns_buffer{false};
};

struct ResultSetDeviceColumnCacheOwner {
  std::shared_ptr<CudaAllocator> allocator;
  std::shared_ptr<CudaStreamReadyEvent> ready_event;
};

bool g_enable_temporary_resultset_peer_access{false};
bool g_enable_temporary_resultset_payload_peer_access{false};
extern bool g_enable_result_reduction_pipeline;
extern bool g_enable_gpu_input_prefetch;
extern bool g_enable_gpu_input_batched_prefetch;

ResultSetDeviceColumnTransfer transferResultSetDeviceColumnIfAvailable(
    Executor* executor,
    const ResultSetPtr& buffer,
    const int col_id,
    const int device_id,
    DeviceAllocator* device_allocator,
    const int frag_id);

namespace {

size_t checked_size_add(const size_t lhs, const size_t rhs, const char* const context) {
  if (rhs > std::numeric_limits<size_t>::max() - lhs) {
    throw std::overflow_error(std::string(context) + " size addition overflow");
  }
  return lhs + rhs;
}

size_t checked_size_multiply(const size_t lhs,
                             const size_t rhs,
                             const char* const context) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error(std::string(context) + " size multiplication overflow");
  }
  return lhs * rhs;
}

void synchronize_queued_copies_after_error(CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                           const CUstream cuda_stream,
                                           const char* const context) noexcept {
  try {
    cuda_mgr->synchronizeStream(cuda_stream);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Could not drain CUDA stream after " << context
               << " failed: " << error.what();
  } catch (...) {
    LOG(ERROR) << "Could not drain CUDA stream after " << context
               << " failed with an unknown error";
  }
}

inline const ColumnarResults* columnarize_result(
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const ResultSetPtr& result,
    const size_t thread_idx,
    const size_t executor_id,
    const int frag_id,
    const ColumnarResults::RowOrderMode row_order_mode =
        ColumnarResults::RowOrderMode::Preserve,
    const std::optional<size_t> selected_column_idx = std::nullopt,
    const std::vector<size_t>& selected_column_indices = {}) {
  INJECT_TIMER(columnarize_result);
  CHECK_EQ(0, frag_id);

  std::vector<SQLTypeInfo> col_types;
  for (size_t i = 0; i < result->colCount(); ++i) {
    const auto& src_ti = result->getColType(i);
    CHECK_EQ(result->checkSlotUsesFlatBufferFormat(i), src_ti.usesFlatBuffer());
    auto ti = get_logical_type_info(src_ti);
    ti.setUsesFlatBuffer(src_ti.supportsFlatBuffer());
    col_types.push_back(ti);
  }
  return new ColumnarResults(row_set_mem_owner,
                             *result,
                             result->colCount(),
                             col_types,
                             executor_id,
                             thread_idx,
                             false,
                             row_order_mode,
                             selected_column_idx,
                             selected_column_indices);
}

std::shared_ptr<const ColumnarResults> find_columnarized_result_alias(
    ColumnCacheMap& column_cache,
    const TemporaryTables* temporary_tables,
    const shared::TableKey& table_key,
    const ResultSetPtr& result,
    const int frag_id) {
  if (!temporary_tables || table_key.table_id >= 0 || !result) {
    return nullptr;
  }
  for (const auto& [cached_table_key, cached_fragments] : column_cache) {
    if (cached_table_key == table_key || cached_table_key.table_id >= 0) {
      continue;
    }
    const auto cached_temp_it = temporary_tables->find(cached_table_key.table_id);
    if (cached_temp_it == temporary_tables->end() ||
        cached_temp_it->second.get() != result.get()) {
      continue;
    }
    const auto cached_frag_it = cached_fragments.find(frag_id);
    if (cached_frag_it != cached_fragments.end()) {
      return cached_frag_it->second;
    }
  }
  return nullptr;
}

bool has_column_buffer(const std::shared_ptr<const ColumnarResults>& columnar_results,
                       const int column_id) {
  if (!columnar_results || column_id < 0) {
    return false;
  }
  const auto& column_buffers = columnar_results->getColumnBuffers();
  const auto column_idx = static_cast<size_t>(column_id);
  return column_idx < column_buffers.size() &&
         (columnar_results->size() == 0 || column_buffers[column_idx]);
}

bool is_varlen_columnar_type(const SQLTypeInfo& type_info) {
  return type_info.is_array() ||
         (type_info.is_string() && type_info.get_compression() == kENCODING_NONE) ||
         type_info.is_geometry();
}

uint64_t segmented_column_index_shift(const uint64_t row_count) {
  constexpr uint64_t min_shift = 20;
  constexpr uint64_t target_index_entries = 16 * 1024;
  uint64_t shift = min_shift;
  while (shift < 63) {
    const uint64_t entries =
        (row_count >> shift) + ((row_count & ((uint64_t(1) << shift) - 1)) != 0);
    if (entries <= target_index_entries) {
      return shift;
    }
    ++shift;
  }
  return shift;
}

uint64_t segmented_column_index_entries(const uint64_t row_count, const uint64_t shift) {
  if (!row_count) {
    return 0;
  }
  if (shift >= 63) {
    return 1;
  }
  return (row_count >> shift) + ((row_count & ((uint64_t(1) << shift) - 1)) != 0);
}

std::string getMemoryLevelString(Data_Namespace::MemoryLevel memoryLevel) {
  switch (memoryLevel) {
    case DISK_LEVEL:
      return "DISK_LEVEL";
    case GPU_LEVEL:
      return "GPU_LEVEL";
    case CPU_LEVEL:
      return "CPU_LEVEL";
    default:
      return "UNKNOWN";
  }
}

size_t getColumnarResultsColumnBytes(const ColumnarResults* columnar_results,
                                     const int col_id) {
  CHECK(columnar_results);
  const auto& col_buffers = columnar_results->getColumnBuffers();
  CHECK_LT(static_cast<size_t>(col_id), col_buffers.size());
  const auto& col_ti = columnar_results->getColumnType(col_id);
  if (col_ti.usesFlatBuffer()) {
    CHECK(FlatBufferManager::isFlatBuffer(col_buffers[col_id]));
    return FlatBufferManager::getBufferSize(col_buffers[col_id]);
  }
  CHECK_GT(col_ti.get_size(), 0);
  return checked_size_multiply(columnar_results->size(),
                               static_cast<size_t>(col_ti.get_size()),
                               "columnar ResultSet column");
}

std::shared_ptr<CudaAllocator> makeResultSetColumnCacheAllocator(Executor* executor,
                                                                 const int device_id) {
  CHECK(executor);
  return std::make_shared<CudaAllocator>(
      executor->getDataMgr(), device_id, executor->getCudaStream(device_id));
}

size_t result_set_fragment_entry_count(
    const ResultSet::DeviceColumnarBufferFragment& fragment) {
  return fragment.entry_count;
}

size_t result_set_fragment_entry_count(
    const ResultSet::ColumnarBufferFragment& fragment) {
  return fragment.second;
}

void wait_for_result_set_fragment_ready(
    Executor* executor,
    const int consumer_device_id,
    const ResultSet::DeviceColumnarBufferFragment& fragment) {
  if (!fragment.ready_event) {
    return;
  }
  executor->getCudaAllocator(consumer_device_id)->waitForReadyEvent(fragment.ready_event);
}

template <typename Fragment>
struct SelectedResultSetFragment {
  Fragment fragment;
  size_t rowid_offset;
};

bool try_make_join_column_from_result_set_storages(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    DeviceAllocator* device_allocator,
    std::vector<std::shared_ptr<void>>& malloc_owner,
    JoinColumn& join_column,
    const bool preserve_result_set_fragment_offsets) {
  if (fragments.empty() || !fragments.front().resultSet) {
    return false;
  }
  const auto result_set = fragments.front().resultSet;
  if (!std::all_of(fragments.begin(), fragments.end(), [result_set](const auto& frag) {
        return frag.resultSet == result_set;
      })) {
    return false;
  }
  const auto& column_key = hash_col.getColumnKey();
  if (get_column_descriptor_maybe(column_key)) {
    return false;
  }
  const auto elem_sz = hash_col.get_type_info().get_size();
  if (elem_sz <= 0 || hash_col.get_type_info().is_varlen()) {
    return false;
  }
  const auto& temporary_table =
      get_temporary_table(executor->getTemporaryTables(), column_key.table_id);

  const auto select_fragments = [&](const auto& source_fragments) {
    using SourceFragment = std::decay_t<decltype(source_fragments.front())>;
    std::vector<SelectedResultSetFragment<SourceFragment>> selected_source_fragments;
    std::vector<size_t> source_offsets(source_fragments.size(), 0);
    for (size_t fragment_idx = 1; fragment_idx < source_fragments.size();
         ++fragment_idx) {
      source_offsets[fragment_idx] = checked_size_add(
          source_offsets[fragment_idx - 1],
          result_set_fragment_entry_count(source_fragments[fragment_idx - 1]),
          "temporary ResultSet fragment offset");
    }
    if (fragments.size() == 1 && fragments.front().fragmentId == 0 &&
        fragments.front().getNumTuples() == temporary_table->rowCount()) {
      selected_source_fragments.reserve(source_fragments.size());
      for (size_t fragment_idx = 0; fragment_idx < source_fragments.size();
           ++fragment_idx) {
        selected_source_fragments.push_back(
            {source_fragments[fragment_idx], source_offsets[fragment_idx]});
      }
    } else {
      selected_source_fragments.reserve(fragments.size());
      size_t local_offset = 0;
      for (const auto& fragment : fragments) {
        CHECK_GE(fragment.fragmentId, 0);
        CHECK_LT(static_cast<size_t>(fragment.fragmentId), source_fragments.size());
        const auto source_fragment_idx = static_cast<size_t>(fragment.fragmentId);
        const auto entry_count =
            result_set_fragment_entry_count(source_fragments[source_fragment_idx]);
        selected_source_fragments.push_back({source_fragments[source_fragment_idx],
                                             preserve_result_set_fragment_offsets
                                                 ? source_offsets[source_fragment_idx]
                                                 : local_offset});
        local_offset = checked_size_add(
            local_offset, entry_count, "temporary ResultSet selected fragment offset");
      }
    }
    return selected_source_fragments;
  };

  if (effective_mem_lvl == Data_Namespace::GPU_LEVEL) {
    std::vector<ResultSet::DeviceColumnarBufferFragment> device_source_fragments;
    const bool has_device_source_fragments =
        temporary_table->getDeviceColumnarBufferFragments(
            column_key.column_id, static_cast<size_t>(elem_sz), device_source_fragments);
    if (has_device_source_fragments) {
      CHECK(!device_source_fragments.empty());
      const bool selecting_whole_temporary_table =
          fragments.size() == 1 && fragments.front().fragmentId == 0 &&
          fragments.front().getNumTuples() == temporary_table->rowCount();
      const bool selecting_all_source_fragments = [&]() {
        if (selecting_whole_temporary_table) {
          return true;
        }
        if (fragments.size() != device_source_fragments.size()) {
          return false;
        }
        for (size_t fragment_idx = 0; fragment_idx < fragments.size(); ++fragment_idx) {
          const auto& fragment = fragments[fragment_idx];
          if (fragment.fragmentId != static_cast<int>(fragment_idx) ||
              fragment.getNumTuples() !=
                  device_source_fragments[fragment_idx].entry_count) {
            return false;
          }
        }
        return true;
      }();
      auto selected_source_fragments = select_fragments(device_source_fragments);
      if (selecting_all_source_fragments) {
        size_t total_rows{0};
        for (const auto& fragment : selected_source_fragments) {
          total_rows = checked_size_add(total_rows,
                                        fragment.fragment.entry_count,
                                        "temporary ResultSet device column rows");
        }
        if (total_rows) {
          const auto total_bytes =
              checked_size_multiply(total_rows,
                                    static_cast<size_t>(elem_sz),
                                    "temporary ResultSet device column");
          if (auto result_set_column_cache = executor->activeResultSetColumnCache();
              result_set_column_cache && total_bytes <= executor->maxGpuSlabSize()) {
            const ResultSetColumnCache::Key cache_key{result_set,
                                                      column_key.column_id,
                                                      effective_mem_lvl,
                                                      device_id,
                                                      -1,
                                                      true,
                                                      false,
                                                      false};
            const int8_t* cached_column = result_set_column_cache->get(cache_key);
            if (!cached_column) {
              try {
                auto allocator_owner =
                    makeResultSetColumnCacheAllocator(executor, device_id);
                auto device_column =
                    transferResultSetDeviceColumnIfAvailable(executor,
                                                             temporary_table,
                                                             column_key.column_id,
                                                             device_id,
                                                             allocator_owner.get(),
                                                             -1);
                if (device_column.buffer) {
                  CHECK(device_column.owns_buffer);
                  cached_column = result_set_column_cache->putOrGetExisting(
                      cache_key,
                      temporary_table,
                      device_column.buffer,
                      std::move(allocator_owner));
                }
              } catch (const std::exception& e) {
                LOG(WARNING)
                    << "Falling back to chunked temporary ResultSet join-column fetch: "
                    << "table_id=" << column_key.table_id
                    << " col_id=" << column_key.column_id << " bytes=" << total_bytes
                    << " device_id=" << device_id << " reason=" << e.what();
              }
            }
            if (cached_column) {
              const auto col_chunks_buff_sz = sizeof(JoinChunk);
              auto col_chunks_buff = reinterpret_cast<int8_t*>(
                  malloc_owner.emplace_back(checked_malloc(col_chunks_buff_sz), free)
                      .get());
              auto join_chunk_array = reinterpret_cast<JoinChunk*>(col_chunks_buff);
              join_chunk_array[0] = JoinChunk{cached_column, total_rows, 0};
              malloc_owner.emplace_back(temporary_table);
              join_column = {col_chunks_buff,
                             col_chunks_buff_sz,
                             size_t(1),
                             total_rows,
                             static_cast<size_t>(elem_sz)};
              return true;
            }
          }
        }
      }
      auto result_set_column_cache = executor->activeResultSetColumnCache();
      const bool cache_fragmented_join_column =
          selecting_all_source_fragments && !g_enable_temporary_resultset_peer_access &&
          result_set_column_cache;
      if (cache_fragmented_join_column) {
        const ResultSetColumnCache::JoinColumnKey cache_key{
            result_set, column_key.column_id, effective_mem_lvl, device_id};
        if (const auto cached_join_column =
                result_set_column_cache->getJoinColumn(cache_key)) {
          join_column = *cached_join_column;
          return true;
        }
      }
      const auto alloc_chunk_count =
          std::max<size_t>(selected_source_fragments.size(), 1);
      const auto col_chunks_buff_sz = checked_size_multiply(
          sizeof(JoinChunk), alloc_chunk_count, "temporary ResultSet join chunks");
      std::shared_ptr<void> col_chunks_owner(checked_malloc(col_chunks_buff_sz), free);
      auto col_chunks_buff = reinterpret_cast<int8_t*>(col_chunks_owner.get());
      if (!cache_fragmented_join_column) {
        malloc_owner.emplace_back(col_chunks_owner);
      }
      auto join_chunk_array = reinterpret_cast<JoinChunk*>(col_chunks_buff);

      auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
      CHECK(cuda_mgr);
      CHECK(device_allocator);
      std::shared_ptr<CudaAllocator> cache_allocator_owner;
      DeviceAllocator* copy_allocator = device_allocator;
      if (cache_fragmented_join_column) {
        cache_allocator_owner = makeResultSetColumnCacheAllocator(executor, device_id);
        copy_allocator = cache_allocator_owner.get();
      }
      malloc_owner.emplace_back(temporary_table);

      size_t num_elems = 0;
      size_t num_chunks = 0;
      const auto cuda_stream = executor->getCudaStream(device_id);
      bool queued_async_peer_copies = false;
      try {
        for (const auto& selected_source_fragment : selected_source_fragments) {
          const auto& source_fragment = selected_source_fragment.fragment;
          CHECK(source_fragment.buffer);
          if (source_fragment.entry_count == 0) {
            continue;
          }
          wait_for_result_set_fragment_ready(executor, device_id, source_fragment);
          const auto chunk_bytes =
              checked_size_multiply(source_fragment.entry_count,
                                    static_cast<size_t>(elem_sz),
                                    "temporary ResultSet join chunk");
          const int8_t* col_buff = source_fragment.buffer;
          const bool can_read_remote_fragment =
              source_fragment.device_id != device_id &&
              g_enable_temporary_resultset_peer_access &&
              cuda_mgr->canAccessPeerMemoryFromKernel(device_id,
                                                      source_fragment.device_id) &&
              cuda_mgr->ensurePeerAccessToDevicePtr(device_id,
                                                    source_fragment.device_id,
                                                    source_fragment.buffer,
                                                    chunk_bytes);
          if (source_fragment.device_id != device_id && !can_read_remote_fragment) {
            auto gpu_col_buffer = copy_allocator->alloc(chunk_bytes);
            cuda_mgr->copyDeviceToDevice(
                gpu_col_buffer,
                const_cast<int8_t*>(source_fragment.buffer),
                chunk_bytes,
                device_id,
                source_fragment.device_id,
                "Temporary ResultSet join column device fragment",
                cuda_stream,
                false);
            queued_async_peer_copies = true;
            col_buff = gpu_col_buffer;
          }
          join_chunk_array[num_chunks++] =
              JoinChunk{col_buff,
                        source_fragment.entry_count,
                        selected_source_fragment.rowid_offset};
          num_elems = checked_size_add(
              num_elems, source_fragment.entry_count, "temporary ResultSet join rows");
        }
      } catch (...) {
        if (queued_async_peer_copies) {
          synchronize_queued_copies_after_error(
              cuda_mgr, cuda_stream, "temporary ResultSet join-column copy");
        }
        throw;
      }
      if (queued_async_peer_copies) {
        cuda_mgr->synchronizeStream(cuda_stream);
      }

      join_column = {col_chunks_buff,
                     col_chunks_buff_sz,
                     num_chunks,
                     num_elems,
                     static_cast<size_t>(elem_sz)};
      if (cache_fragmented_join_column) {
        const ResultSetColumnCache::JoinColumnKey cache_key{
            result_set, column_key.column_id, effective_mem_lvl, device_id};
        std::vector<std::shared_ptr<void>> owners;
        owners.emplace_back(std::move(cache_allocator_owner));
        join_column = result_set_column_cache->putOrGetExistingJoinColumn(
            cache_key,
            temporary_table,
            join_column,
            std::move(col_chunks_owner),
            std::move(owners));
      }
      return true;
    }
  }

  std::vector<ResultSet::ColumnarBufferFragment> source_fragments;
  if (!temporary_table->getColumnarBufferFragments(
          column_key.column_id, static_cast<size_t>(elem_sz), source_fragments)) {
    return false;
  }

  auto selected_source_fragments = select_fragments(source_fragments);

  const auto alloc_chunk_count = std::max<size_t>(selected_source_fragments.size(), 1);
  const auto col_chunks_buff_sz = checked_size_multiply(
      sizeof(JoinChunk), alloc_chunk_count, "host temporary ResultSet join chunks");
  auto col_chunks_buff = reinterpret_cast<int8_t*>(
      malloc_owner.emplace_back(checked_malloc(col_chunks_buff_sz), free).get());
  auto join_chunk_array = reinterpret_cast<JoinChunk*>(col_chunks_buff);

  size_t num_elems = 0;
  size_t num_chunks = 0;
  for (const auto& selected_source_fragment : selected_source_fragments) {
    const auto& [source_buffer, entry_count] = selected_source_fragment.fragment;
    CHECK(source_buffer);
    if (entry_count == 0) {
      continue;
    }
    const auto chunk_bytes = checked_size_multiply(
        entry_count, static_cast<size_t>(elem_sz), "host temporary ResultSet join chunk");
    const int8_t* col_buff = source_buffer;
    if (effective_mem_lvl == Data_Namespace::GPU_LEVEL) {
      CHECK(device_allocator);
      auto gpu_col_buffer = device_allocator->alloc(chunk_bytes);
      device_allocator->copyToDevice(gpu_col_buffer,
                                     source_buffer,
                                     chunk_bytes,
                                     "Temporary ResultSet join column chunk");
      col_buff = gpu_col_buffer;
    }
    join_chunk_array[num_chunks++] =
        JoinChunk{col_buff, entry_count, selected_source_fragment.rowid_offset};
    num_elems =
        checked_size_add(num_elems, entry_count, "host temporary ResultSet join rows");
  }

  join_column = {col_chunks_buff,
                 col_chunks_buff_sz,
                 num_chunks,
                 num_elems,
                 static_cast<size_t>(elem_sz)};
  return true;
}
}  // namespace

const int8_t* ResultSetColumnCache::get(const Key& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = entries_.find(key);
  if (it == entries_.end()) {
    return nullptr;
  }
  return it->second.buffer;
}

const int8_t* ResultSetColumnCache::putOrGetExisting(const Key& key,
                                                     const ResultSetPtr& result_owner,
                                                     const int8_t* buffer,
                                                     std::shared_ptr<void> owner) {
  CHECK(buffer);
  CHECK(result_owner);
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] =
      entries_.emplace(key, Entry{buffer, std::move(owner), result_owner});
  if (inserted) {
    return buffer;
  }
  return it->second.buffer;
}

const int8_t* ResultSetColumnCache::getOrCreate(
    const Key& key,
    const ResultSetPtr& result_owner,
    std::shared_ptr<void> owner,
    const std::function<const int8_t*()>& create) {
  CHECK(result_owner);
  CHECK(owner);

  std::shared_ptr<std::mutex> creation_mutex;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto entry_it = entries_.find(key); entry_it != entries_.end()) {
      return entry_it->second.buffer;
    }
    auto mutex_it =
        entry_creation_mutexes_.try_emplace(key, std::make_shared<std::mutex>()).first;
    creation_mutex = mutex_it->second;
  }

  std::lock_guard<std::mutex> creation_lock(*creation_mutex);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto entry_it = entries_.find(key); entry_it != entries_.end()) {
      return entry_it->second.buffer;
    }
  }

  const auto buffer = create();
  if (!buffer) {
    return nullptr;
  }
  return putOrGetExisting(key, result_owner, buffer, std::move(owner));
}

std::optional<JoinColumn> ResultSetColumnCache::getJoinColumn(
    const JoinColumnKey& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = join_column_entries_.find(key);
  if (it == join_column_entries_.end()) {
    return std::nullopt;
  }
  return it->second.join_column;
}

JoinColumn ResultSetColumnCache::putOrGetExistingJoinColumn(
    const JoinColumnKey& key,
    const ResultSetPtr& result_owner,
    JoinColumn join_column,
    std::shared_ptr<void> chunks_owner,
    std::vector<std::shared_ptr<void>> owners) {
  CHECK(result_owner);
  CHECK(join_column.col_chunks_buff);
  CHECK(chunks_owner);
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = join_column_entries_.emplace(
      key,
      JoinColumnEntry{
          join_column, std::move(chunks_owner), std::move(owners), result_owner});
  if (inserted) {
    return join_column;
  }
  return it->second.join_column;
}

void ResultSetColumnCache::mergeColumnSelection(
    const ResultSetPtr& result_set,
    const std::vector<size_t>& selected_column_indices) {
  CHECK(result_set);
  if (selected_column_indices.empty()) {
    return;
  }

  auto selection = selected_column_indices;
  for (const auto column_idx : selection) {
    CHECK_LT(column_idx, result_set->colCount());
  }
  std::sort(selection.begin(), selection.end());
  selection.erase(std::unique(selection.begin(), selection.end()), selection.end());

  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = column_selections_.try_emplace(
      result_set.get(), ColumnSelectionEntry{result_set, {}, false});
  auto& entry = it->second;
  CHECK_EQ(entry.result_owner.get(), result_set.get());
  if (entry.all_columns) {
    return;
  }
  if (selection.size() >= result_set->colCount()) {
    entry.selected_column_indices.clear();
    entry.all_columns = true;
    return;
  }
  if (inserted) {
    entry.selected_column_indices = std::move(selection);
    return;
  }

  std::vector<size_t> merged_selection;
  merged_selection.reserve(entry.selected_column_indices.size() + selection.size());
  std::set_union(entry.selected_column_indices.begin(),
                 entry.selected_column_indices.end(),
                 selection.begin(),
                 selection.end(),
                 std::back_inserter(merged_selection));
  if (merged_selection.size() >= result_set->colCount()) {
    entry.selected_column_indices.clear();
    entry.all_columns = true;
  } else {
    entry.selected_column_indices = std::move(merged_selection);
  }
}

std::optional<std::vector<size_t>> ResultSetColumnCache::getColumnSelection(
    const ResultSet* result_set) const {
  CHECK(result_set);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = column_selections_.find(result_set);
  if (it == column_selections_.end() || it->second.all_columns) {
    return std::nullopt;
  }
  return it->second.selected_column_indices;
}

void ResultSetColumnCache::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  entry_creation_mutexes_.clear();
  join_column_entries_.clear();
  column_selections_.clear();
}

ColumnFetcher::ColumnFetcher(Executor* executor,
                             ColumnCacheMap& column_cache,
                             ResultSetColumnCache* result_set_column_cache)
    : executor_(executor)
    , columnarized_table_cache_(column_cache)
    , result_set_column_cache_(result_set_column_cache
                                   ? result_set_column_cache
                                   : &local_result_set_column_cache_) {}

//! Gets a column fragment chunk on CPU or on GPU depending on the effective
//! memory level parameter. For temporary tables, the chunk will be copied to
//! the GPU if needed. Returns a buffer pointer and an element count.
std::pair<const int8_t*, size_t> ColumnFetcher::getOneColumnFragment(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ColumnCacheMap& column_cache,
    const ColumnarResults::RowOrderMode row_order_mode) {
  static std::mutex columnar_conversion_mutex;
  auto timer = DEBUG_TIMER(__func__);
  if (fragment.isEmptyPhysicalFragment()) {
    return {nullptr, 0};
  }
  const auto& column_key = hash_col.getColumnKey();
  const auto cd = get_column_descriptor_maybe(column_key);
  CHECK(!cd || !(cd->isVirtualCol));
  const int8_t* col_buff = nullptr;
  if (cd) {  // real table
    /* chunk_meta_it is used here to retrieve chunk numBytes and
       numElements. Apparently, their values are often zeros. If we
       knew how to predict the zero values, calling
       getChunkMetadataMap could be avoided to skip
       synthesize_metadata calls. */
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(column_key.column_id);
    CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
    ChunkKey chunk_key{column_key.db_id,
                       fragment.physicalTableId,
                       column_key.column_id,
                       fragment.fragmentId};
    const auto chunk = Chunk_NS::Chunk::getChunk(
        cd,
        executor->getDataMgr(),
        chunk_key,
        effective_mem_lvl,
        effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        chunk_meta_it->second->numBytes,
        chunk_meta_it->second->numElements);
    chunks_owner.push_back(chunk);
    CHECK(chunk);
    auto ab = chunk->getBuffer();
    CHECK(ab->getMemoryPtr());
    col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  } else {  // temporary table
    const ColumnarResults* col_frag{nullptr};
    {
      std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex);
      const auto frag_id = fragment.fragmentId;
      shared::TableKey table_key{column_key.db_id, column_key.table_id};
      if (column_cache.empty() || !column_cache.count(table_key)) {
        column_cache.insert(std::make_pair(
            table_key,
            std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_result = column_cache[table_key];
      const auto& temporary_table =
          get_temporary_table(executor->temporary_tables_, table_key.table_id);
      auto cached_result_it = frag_id_to_result.find(frag_id);
      if (cached_result_it == frag_id_to_result.end() ||
          !has_column_buffer(cached_result_it->second, column_key.column_id)) {
        std::vector<size_t> selected_column_indices;
        if (const auto result_set_column_cache = executor->activeResultSetColumnCache()) {
          if (const auto selection =
                  result_set_column_cache->getColumnSelection(temporary_table.get());
              selection &&
              std::binary_search(selection->begin(),
                                 selection->end(),
                                 static_cast<size_t>(column_key.column_id))) {
            selected_column_indices = *selection;
          }
        }
        const auto alias =
            selected_column_indices.empty()
                ? find_columnarized_result_alias(column_cache,
                                                 executor->temporary_tables_,
                                                 table_key,
                                                 temporary_table,
                                                 frag_id)
                : nullptr;
        if (g_enable_result_reduction_pipeline &&
            has_column_buffer(alias, column_key.column_id)) {
          frag_id_to_result[frag_id] = alias;
        } else {
          if (!selected_column_indices.empty()) {
            temporary_table->materializeDeferredLazyFetchColumnsForAllRows(
                selected_column_indices);
          }
          frag_id_to_result[frag_id] = std::shared_ptr<const ColumnarResults>(
              columnarize_result(executor->row_set_mem_owner_,
                                 temporary_table,
                                 thread_idx,
                                 executor->executor_id_,
                                 frag_id,
                                 row_order_mode,
                                 std::nullopt,
                                 selected_column_indices));
        }
      }
      col_frag = frag_id_to_result.at(frag_id).get();
    }
    col_buff = transferColumnIfNeeded(
        col_frag,
        hash_col.getColumnKey().column_id,
        executor->getDataMgr(),
        effective_mem_lvl,
        effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        device_allocator);
    return {col_buff, col_frag->size()};
  }
  return {col_buff, fragment.getNumTuples()};
}

//! makeJoinColumn() creates a JoinColumn struct containing a array of
//! JoinChunk structs, col_chunks_buff, malloced in CPU memory. Although
//! the col_chunks_buff array is in CPU memory here, each JoinChunk struct
//! contains an int8_t* pointer from getOneColumnFragment(), col_buff,
//! that can point to either CPU memory or GPU memory depending on the
//! effective_mem_lvl parameter. See also the fetchJoinColumn() function
//! where col_chunks_buff is copied into GPU memory if needed. The
//! malloc_owner parameter will have the malloced array appended. The
//! chunks_owner parameter will be appended with the chunks.
JoinColumn ColumnFetcher::makeJoinColumn(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    std::vector<std::shared_ptr<void>>& malloc_owner,
    ColumnCacheMap& column_cache,
    const bool preserve_result_set_fragment_offsets,
    const std::map<std::pair<int, int>, size_t>* physical_fragment_rowid_offsets) {
  CHECK(!fragments.empty());

  JoinColumn result_set_join_column{};
  if (g_enable_result_reduction_pipeline && try_make_join_column_from_result_set_storages(
                                                executor,
                                                hash_col,
                                                fragments,
                                                effective_mem_lvl,
                                                device_id,
                                                device_allocator,
                                                malloc_owner,
                                                result_set_join_column,
                                                preserve_result_set_fragment_offsets)) {
    return result_set_join_column;
  }

  const auto col_chunks_buff_sz = checked_size_multiply(
      sizeof(JoinChunk), fragments.size(), "join-column chunk descriptors");
  // TODO: needs an allocator owner
  auto col_chunks_buff = reinterpret_cast<int8_t*>(
      malloc_owner.emplace_back(checked_malloc(col_chunks_buff_sz), free).get());
  auto join_chunk_array = reinterpret_cast<struct JoinChunk*>(col_chunks_buff);

  size_t num_elems = 0;
  size_t num_chunks = 0;
  const auto fragment_rowid_offset = [&](const auto& fragment,
                                         const size_t local_offset) {
    if (!physical_fragment_rowid_offsets) {
      return local_offset;
    }
    const auto offset_it = physical_fragment_rowid_offsets->find(
        {fragment.physicalTableId, fragment.fragmentId});
    CHECK(offset_it != physical_fragment_rowid_offsets->end());
    return offset_it->second;
  };
  const auto& column_key = hash_col.getColumnKey();
  const auto cd = get_column_descriptor_maybe(column_key);
  const bool use_batched_gpu_fetch =
      g_enable_gpu_input_prefetch && g_enable_gpu_input_batched_prefetch &&
      effective_mem_lvl == Data_Namespace::GPU_LEVEL && cd && !cd->columnType.is_varlen();
  if (use_batched_gpu_fetch) {
    CHECK(!cd->isVirtualCol);
    std::vector<Data_Namespace::BufferFetchRequest> requests;
    std::vector<size_t> fragment_row_counts;
    std::vector<size_t> fragment_rowid_offsets;
    requests.reserve(fragments.size());
    fragment_row_counts.reserve(fragments.size());
    fragment_rowid_offsets.reserve(fragments.size());
    for (const auto& fragment : fragments) {
      if (g_enable_non_kernel_time_query_interrupt &&
          executor->checkNonKernelTimeInterrupted()) {
        throw QueryExecutionError(ErrorCode::INTERRUPTED);
      }
      if (fragment.isEmptyPhysicalFragment()) {
        continue;
      }
      const auto chunk_meta_it =
          fragment.getChunkMetadataMap().find(column_key.column_id);
      CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
      requests.push_back({{column_key.db_id,
                           fragment.physicalTableId,
                           column_key.column_id,
                           fragment.fragmentId},
                          chunk_meta_it->second->numBytes});
      fragment_row_counts.push_back(fragment.getNumTuples());
      fragment_rowid_offsets.push_back(fragment_rowid_offset(fragment, num_elems));
      num_elems = checked_size_add(
          num_elems, fragment.getNumTuples(), "join-column element count");
    }

    auto buffers = executor->getDataMgr()->getChunkBuffers(
        requests, Data_Namespace::GPU_LEVEL, device_id);
    CHECK_EQ(buffers.size(), fragment_row_counts.size());
    size_t adopted_buffer_count{0};
    ScopeGuard unpin_unadopted_buffers = [&] {
      for (size_t i = adopted_buffer_count; i < buffers.size(); ++i) {
        if (buffers[i]) {
          buffers[i]->unPin();
        }
      }
    };
    for (size_t i = 0; i < buffers.size(); ++i) {
      auto buffer = buffers[i];
      CHECK(buffer);
      CHECK(buffer->getMemoryPtr());
      auto chunk = Chunk_NS::Chunk::getChunk(cd, buffer, nullptr);
      adopted_buffer_count = i + 1;
      chunks_owner.push_back(std::move(chunk));
      const auto elem_count = fragment_row_counts[i];
      join_chunk_array[num_chunks] =
          JoinChunk{reinterpret_cast<const int8_t*>(buffer->getMemoryPtr()),
                    elem_count,
                    fragment_rowid_offsets[i]};
      ++num_chunks;
    }
  } else {
    for (auto& frag : fragments) {
      if (g_enable_non_kernel_time_query_interrupt &&
          executor->checkNonKernelTimeInterrupted()) {
        throw QueryExecutionError(ErrorCode::INTERRUPTED);
      }
      const auto row_order_mode =
          g_enable_result_reduction_pipeline && hash_col.getColumnKey().table_id < 0
              ? ColumnarResults::RowOrderMode::Preserve
              : ColumnarResults::RowOrderMode::Unordered;
      auto [col_buff, elem_count] = getOneColumnFragment(
          executor,
          hash_col,
          frag,
          effective_mem_lvl,
          effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
          device_allocator,
          thread_idx,
          chunks_owner,
          column_cache,
          row_order_mode);
      if (col_buff != nullptr) {
        join_chunk_array[num_chunks] =
            JoinChunk{col_buff, elem_count, fragment_rowid_offset(frag, num_elems)};
        num_elems = checked_size_add(num_elems, elem_count, "join-column element count");
      } else {
        continue;
      }
      ++num_chunks;
    }
  }

  int elem_sz = hash_col.get_type_info().get_size();
  CHECK_GT(elem_sz, 0);

  return {col_chunks_buff,
          col_chunks_buff_sz,
          num_chunks,
          num_elems,
          static_cast<size_t>(elem_sz)};
}

const int8_t* ColumnFetcher::getOneTableColumnFragment(
    const shared::TableKey& table_key,
    const int frag_id,
    const int col_id,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* allocator) const {
  const auto fragments_it = all_tables_fragments.find(table_key);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto& fragment = (*fragments)[frag_id];
  if (fragment.isEmptyPhysicalFragment()) {
    return nullptr;
  }
  std::shared_ptr<Chunk_NS::Chunk> chunk;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  CHECK(table_key.table_id > 0);
  const auto cd = get_column_descriptor({table_key, col_id});
  CHECK(cd);
  const auto col_type =
      get_column_type(col_id, table_key.table_id, cd, executor_->temporary_tables_);
  const bool is_real_string =
      col_type.is_string() && col_type.get_compression() == kENCODING_NONE;
  const bool is_varlen =
      is_real_string ||
      col_type.is_array();  // TODO: should it be col_type.is_varlen_array() ?
  {
    ChunkKey chunk_key{
        table_key.db_id, fragment.physicalTableId, col_id, fragment.fragmentId};
    std::unique_ptr<std::lock_guard<std::mutex>> varlen_chunk_lock;
    if (is_varlen) {
      varlen_chunk_lock.reset(new std::lock_guard<std::mutex>(varlen_chunk_fetch_mutex_));
    }
    chunk = Chunk_NS::Chunk::getChunk(
        cd,
        executor_->getDataMgr(),
        chunk_key,
        memory_level,
        memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        chunk_meta_it->second->numBytes,
        chunk_meta_it->second->numElements);
    {
      std::lock_guard<std::mutex> chunk_list_lock(chunk_list_mutex_);
      chunk_holder.push_back(chunk);
    }
  }
  if (is_varlen) {
    CHECK_GT(table_key.table_id, 0);
    CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
    chunk_iter_holder.push_back(chunk->begin_iterator(chunk_meta_it->second));
    auto& chunk_iter = chunk_iter_holder.back();
    if (memory_level == Data_Namespace::CPU_LEVEL) {
      return reinterpret_cast<int8_t*>(&chunk_iter);
    } else {
      auto ab = chunk->getBuffer();
      ab->pin();
      auto& row_set_mem_owner = executor_->getRowSetMemoryOwner();
      row_set_mem_owner->addVarlenInputBuffer(ab);
      CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
      CHECK(allocator);
      auto chunk_iter_gpu = allocator->alloc(sizeof(ChunkIter));
      allocator->copyToDevice(chunk_iter_gpu,
                              reinterpret_cast<int8_t*>(&chunk_iter),
                              sizeof(ChunkIter),
                              "Chunk iterator");
      return chunk_iter_gpu;
    }
  } else {
    auto ab = chunk->getBuffer();
    CHECK(ab->getMemoryPtr());
    return ab->getMemoryPtr();  // @TODO(alex) change to use ChunkIter
  }
}

const int8_t* ColumnFetcher::getAllTableColumnFragments(
    const shared::TableKey& table_key,
    const int col_id,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  const auto fragments_it = all_tables_fragments.find(table_key);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  const ColumnarResults* table_column = nullptr;
  const InputColDescriptor col_desc(col_id, table_key.table_id, table_key.db_id, int(0));
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);

  if (!g_enable_result_reduction_pipeline) {
    std::vector<std::unique_ptr<ColumnarResults>> column_frags;
    {
      std::lock_guard<std::mutex> columnar_conversion_guard(columnar_fetch_mutex_);
      const auto column_it = columnarized_scan_table_cache_.find(col_desc);
      if (column_it == columnarized_scan_table_cache_.end()) {
        for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
          if (g_enable_non_kernel_time_query_interrupt &&
              executor_->checkNonKernelTimeInterrupted()) {
            throw QueryExecutionError(ErrorCode::INTERRUPTED);
          }
          std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
          std::list<ChunkIter> chunk_iter_holder;
          const auto& fragment = (*fragments)[frag_id];
          if (fragment.isEmptyPhysicalFragment()) {
            continue;
          }
          const auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
          CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
          const auto col_buffer = getOneTableColumnFragment(table_key,
                                                            static_cast<int>(frag_id),
                                                            col_id,
                                                            all_tables_fragments,
                                                            chunk_holder,
                                                            chunk_iter_holder,
                                                            Data_Namespace::CPU_LEVEL,
                                                            0,
                                                            device_allocator);
          column_frags.push_back(
              std::make_unique<ColumnarResults>(executor_->row_set_mem_owner_,
                                                col_buffer,
                                                fragment.getNumTuples(),
                                                chunk_meta_it->second->sqlType,
                                                executor_->executor_id_,
                                                thread_idx));
        }
        auto merged_results =
            ColumnarResults::mergeResults(executor_->row_set_mem_owner_, column_frags);
        table_column = merged_results.get();
        columnarized_scan_table_cache_.emplace(col_desc, std::move(merged_results));
      } else {
        table_column = column_it->second.get();
      }
    }
    return ColumnFetcher::transferColumnIfNeeded(table_column,
                                                 0,
                                                 executor_->getDataMgr(),
                                                 memory_level,
                                                 device_id,
                                                 device_allocator);
  }

  if (memory_level == Data_Namespace::GPU_LEVEL) {
    CHECK(device_allocator);
    const auto cd = get_column_descriptor({table_key, col_id});
    CHECK(cd);
    if (!is_varlen_columnar_type(cd->columnType)) {
      const auto byte_width = static_cast<size_t>(cd->columnType.get_size());
      std::vector<ChunkKey> chunk_keys;
      std::vector<size_t> row_counts;
      std::vector<size_t> chunk_num_bytes;
      std::vector<size_t> chunk_num_elements;
      std::vector<size_t> byte_offsets;
      chunk_keys.reserve(frag_count);
      row_counts.reserve(frag_count);
      chunk_num_bytes.reserve(frag_count);
      chunk_num_elements.reserve(frag_count);
      byte_offsets.reserve(frag_count);

      size_t total_bytes = 0;
      for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
        const auto& fragment = (*fragments)[frag_id];
        if (fragment.isEmptyPhysicalFragment()) {
          continue;
        }
        auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
        CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
        byte_offsets.push_back(total_bytes);
        row_counts.push_back(fragment.getNumTuples());
        chunk_num_bytes.push_back(chunk_meta_it->second->numBytes);
        chunk_num_elements.push_back(chunk_meta_it->second->numElements);
        chunk_keys.push_back(ChunkKey{
            table_key.db_id, fragment.physicalTableId, col_id, fragment.fragmentId});
        const auto fragment_bytes = checked_size_multiply(
            fragment.getNumTuples(), byte_width, "all-fragment GPU column chunk");
        total_bytes =
            checked_size_add(total_bytes, fragment_bytes, "all-fragment GPU column");
      }

      if (!total_bytes) {
        return nullptr;
      }

      auto gpu_col_buffer = device_allocator->alloc(total_bytes);
      auto cuda_mgr = executor_->getDataMgr()->getCudaMgr();
      CHECK(cuda_mgr);
      const size_t worker_count = std::max<size_t>(
          1,
          std::min({chunk_keys.size(), static_cast<size_t>(cpu_threads()), size_t(32)}));
      for (size_t batch_begin = 0; batch_begin < chunk_keys.size();
           batch_begin += worker_count) {
        if (g_enable_non_kernel_time_query_interrupt &&
            executor_->checkNonKernelTimeInterrupted()) {
          throw QueryExecutionError(ErrorCode::INTERRUPTED);
        }
        std::vector<std::future<void>> workers;
        const size_t batch_end = std::min(batch_begin + worker_count, chunk_keys.size());
        workers.reserve(batch_end - batch_begin);
        for (size_t i = batch_begin; i < batch_end; ++i) {
          workers.push_back(std::async(std::launch::async, [&, i] {
            auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                                   executor_->getDataMgr(),
                                                   chunk_keys[i],
                                                   Data_Namespace::GPU_LEVEL,
                                                   device_id,
                                                   chunk_num_bytes[i],
                                                   chunk_num_elements[i]);
            CHECK(chunk);
            cuda_mgr->copyDeviceToDevice(
                gpu_col_buffer + byte_offsets[i],
                chunk->getBuffer()->getMemoryPtr(),
                checked_size_multiply(row_counts[i], byte_width, "all-fragment GPU copy"),
                device_id,
                device_id,
                "All-fragment column buffer");
          }));
        }
        for (auto& worker : workers) {
          worker.get();
        }
      }
      return gpu_col_buffer;
    }
  }

  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_fetch_mutex_);
    auto column_it = columnarized_scan_table_cache_.find(col_desc);
    if (column_it == columnarized_scan_table_cache_.end()) {
      const auto cd = get_column_descriptor({table_key, col_id});
      CHECK(cd);
      std::vector<const int8_t*> column_buffers_by_frag_id(frag_count);
      std::vector<size_t> row_counts_by_frag_id(frag_count);
      std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunk_holders_by_frag_id(frag_count);
      const auto build_column_fragment = [&](const size_t frag_id) {
        const auto& fragment = (*fragments)[frag_id];
        if (fragment.isEmptyPhysicalFragment()) {
          return;
        }
        auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
        CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
        ChunkKey chunk_key{
            table_key.db_id, fragment.physicalTableId, col_id, fragment.fragmentId};
        auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               executor_->getDataMgr(),
                                               chunk_key,
                                               Data_Namespace::CPU_LEVEL,
                                               0,
                                               chunk_meta_it->second->numBytes,
                                               chunk_meta_it->second->numElements);
        CHECK(chunk);
        auto ab = chunk->getBuffer();
        CHECK(ab->getMemoryPtr());
        column_buffers_by_frag_id[frag_id] = ab->getMemoryPtr();
        row_counts_by_frag_id[frag_id] = fragment.getNumTuples();
        chunk_holders_by_frag_id[frag_id] = std::move(chunk);
      };

      const size_t worker_count = std::max<size_t>(
          1, std::min({frag_count, static_cast<size_t>(cpu_threads()), size_t(32)}));
      for (size_t batch_begin = 0; batch_begin < frag_count;
           batch_begin += worker_count) {
        if (g_enable_non_kernel_time_query_interrupt &&
            executor_->checkNonKernelTimeInterrupted()) {
          throw QueryExecutionError(ErrorCode::INTERRUPTED);
        }
        std::vector<std::future<void>> workers;
        const size_t batch_end = std::min(batch_begin + worker_count, frag_count);
        workers.reserve(batch_end - batch_begin);
        for (size_t frag_id = batch_begin; frag_id < batch_end; ++frag_id) {
          workers.push_back(
              std::async(std::launch::async, build_column_fragment, frag_id));
        }
        for (auto& worker : workers) {
          worker.get();
        }
      }
      std::vector<const int8_t*> column_buffers;
      std::vector<size_t> row_counts;
      column_buffers.reserve(frag_count);
      row_counts.reserve(frag_count);
      for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
        if (column_buffers_by_frag_id[frag_id]) {
          column_buffers.push_back(column_buffers_by_frag_id[frag_id]);
          row_counts.push_back(row_counts_by_frag_id[frag_id]);
        }
      }
      auto merged_results =
          ColumnarResults::mergeColumnBuffers(executor_->row_set_mem_owner_,
                                              column_buffers,
                                              row_counts,
                                              cd->columnType,
                                              thread_idx);
      table_column = merged_results.get();
      columnarized_scan_table_cache_.emplace(col_desc, std::move(merged_results));
    } else {
      table_column = column_it->second.get();
    }
  }
  return ColumnFetcher::transferColumnIfNeeded(table_column,
                                               0,
                                               executor_->getDataMgr(),
                                               memory_level,
                                               device_id,
                                               device_allocator);
}

const int8_t* ColumnFetcher::getTableColumnFragmentsSegmented(
    const shared::TableKey& table_key,
    const int col_id,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    const std::vector<size_t>& fragment_ids,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator) const {
  CHECK_EQ(memory_level, Data_Namespace::GPU_LEVEL);
  CHECK(device_allocator);
  auto sorted_fragment_ids = fragment_ids;
  std::sort(sorted_fragment_ids.begin(), sorted_fragment_ids.end());
  sorted_fragment_ids.erase(
      std::unique(sorted_fragment_ids.begin(), sorted_fragment_ids.end()),
      sorted_fragment_ids.end());
  SegmentedTableColumnCacheKey cache_key{
      {table_key, col_id}, device_id, sorted_fragment_ids};
  {
    std::lock_guard<std::mutex> cache_lock(segmented_table_column_cache_mutex_);
    const auto cache_it = segmented_table_column_cache_.find(cache_key);
    if (cache_it != segmented_table_column_cache_.end()) {
      return cache_it->second.descriptor;
    }
  }

  const auto fragments_it = all_tables_fragments.find(table_key);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto cd = get_column_descriptor({table_key, col_id});
  CHECK(cd);
  CHECK(!is_varlen_columnar_type(cd->columnType));
  const auto byte_width = static_cast<size_t>(cd->columnType.get_size());
  CHECK_GT(byte_width, size_t(0));

  std::vector<size_t> fragment_offsets(fragments->size());
  size_t global_row_count{0};
  for (size_t frag_id = 0; frag_id < fragments->size(); ++frag_id) {
    fragment_offsets[frag_id] = global_row_count;
    global_row_count = checked_size_add(global_row_count,
                                        (*fragments)[frag_id].getNumTuples(),
                                        "segmented table global row count");
  }

  std::vector<uint64_t> descriptor;
  descriptor.reserve(checked_size_add(
      3,
      checked_size_multiply(3, fragment_ids.size(), "segmented table column descriptor"),
      "segmented table column descriptor"));
  descriptor.push_back(0);  // fragment count
  descriptor.push_back(0);  // row-position index shift
  descriptor.push_back(0);  // row-position index entry count

  std::vector<Data_Namespace::BufferFetchRequest> fetch_requests;
  std::vector<size_t> row_counts;
  std::vector<size_t> fetched_fragment_ids;
  fetch_requests.reserve(sorted_fragment_ids.size());
  row_counts.reserve(sorted_fragment_ids.size());
  fetched_fragment_ids.reserve(sorted_fragment_ids.size());
  for (const auto frag_id : sorted_fragment_ids) {
    CHECK_LT(frag_id, fragments->size());
    const auto& fragment = (*fragments)[frag_id];
    if (fragment.isEmptyPhysicalFragment()) {
      continue;
    }
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
    CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
    ChunkKey chunk_key{
        table_key.db_id, fragment.physicalTableId, col_id, fragment.fragmentId};
    fetch_requests.push_back({std::move(chunk_key), chunk_meta_it->second->numBytes});
    row_counts.push_back(fragment.getNumTuples());
    fetched_fragment_ids.push_back(frag_id);
  }

  if (fetch_requests.empty()) {
    return nullptr;
  }

  auto buffers = executor_->getDataMgr()->getChunkBuffers(
      fetch_requests, Data_Namespace::GPU_LEVEL, device_id);
  CHECK_EQ(buffers.size(), row_counts.size());
  size_t adopted_buffer_count{0};
  ScopeGuard unpin_unadopted_buffers = [&] {
    for (size_t i = adopted_buffer_count; i < buffers.size(); ++i) {
      if (buffers[i]) {
        buffers[i]->unPin();
      }
    }
  };
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> fetched_chunks;
  {
    fetched_chunks.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      auto buffer = buffers[i];
      CHECK(buffer);
      CHECK(buffer->getMemoryPtr());
      auto chunk = Chunk_NS::Chunk::getChunk(cd, buffer, nullptr);
      adopted_buffer_count = i + 1;
      fetched_chunks.push_back(std::move(chunk));
      const auto row_count = row_counts[i];
      descriptor.push_back(reinterpret_cast<uint64_t>(buffer->getMemoryPtr()));
      descriptor.push_back(
          static_cast<uint64_t>(fragment_offsets[fetched_fragment_ids[i]]));
      descriptor.push_back(static_cast<uint64_t>(row_count));
    }
    descriptor.front() = (descriptor.size() - 3) / 3;
    const auto fragment_count = descriptor.front();
    if (!fragment_count) {
      return nullptr;
    }
    const auto index_shift = segmented_column_index_shift(global_row_count);
    const auto index_count =
        segmented_column_index_entries(global_row_count, index_shift);
    descriptor[1] = index_shift;
    descriptor[2] = index_count;
    descriptor.reserve(checked_size_add(descriptor.size(),
                                        static_cast<size_t>(index_count),
                                        "segmented table column index"));
    size_t indexed_fragment = 0;
    for (uint64_t bucket = 0; bucket < index_count; ++bucket) {
      const uint64_t bucket_start = index_shift >= 63 ? 0 : (bucket << index_shift);
      while (indexed_fragment + 1 < fragment_count) {
        const auto entry_offset = 3 + 3 * indexed_fragment;
        const uint64_t start = descriptor[entry_offset + 1];
        const uint64_t row_count = descriptor[entry_offset + 2];
        if (bucket_start < checked_size_add(static_cast<size_t>(start),
                                            static_cast<size_t>(row_count),
                                            "segmented table column fragment range")) {
          break;
        }
        ++indexed_fragment;
      }
      descriptor.push_back(static_cast<uint64_t>(indexed_fragment));
    }
  }
  const auto descriptor_bytes = checked_size_multiply(
      descriptor.size(), sizeof(uint64_t), "segmented table column descriptor bytes");
  auto device_descriptor = device_allocator->alloc(descriptor_bytes);
  device_allocator->copyToDevice(device_descriptor,
                                 descriptor.data(),
                                 descriptor_bytes,
                                 "Segmented all-fragment column descriptor");
  {
    std::lock_guard<std::mutex> cache_lock(segmented_table_column_cache_mutex_);
    const auto cache_it =
        segmented_table_column_cache_
            .try_emplace(std::move(cache_key),
                         SegmentedTableColumnCacheEntry{device_descriptor,
                                                        std::move(fetched_chunks)})
            .first;
    return cache_it->second.descriptor;
  }
}

struct SegmentedResultSetColumnFragment {
  const int8_t* buffer{nullptr};
  size_t entry_count{0};
};

std::vector<uint64_t> makeSegmentedColumnDescriptor(
    const std::vector<SegmentedResultSetColumnFragment>& fragments) {
  std::vector<uint64_t> descriptor;
  descriptor.reserve(checked_size_add(
      3,
      checked_size_multiply(3, fragments.size(), "segmented column descriptor"),
      "segmented column descriptor"));
  descriptor.push_back(0);  // fragment count
  descriptor.push_back(0);  // row-position index shift
  descriptor.push_back(0);  // row-position index entry count

  size_t total_rows = 0;
  for (const auto& fragment : fragments) {
    if (!fragment.buffer || fragment.entry_count == 0) {
      continue;
    }
    descriptor.push_back(reinterpret_cast<uint64_t>(fragment.buffer));
    descriptor.push_back(static_cast<uint64_t>(total_rows));
    descriptor.push_back(static_cast<uint64_t>(fragment.entry_count));
    total_rows =
        checked_size_add(total_rows, fragment.entry_count, "segmented column rows");
  }

  descriptor.front() = (descriptor.size() - 3) / 3;
  const auto fragment_count = descriptor.front();
  if (!fragment_count) {
    descriptor.clear();
    return descriptor;
  }

  const auto index_shift = segmented_column_index_shift(total_rows);
  const auto index_count = segmented_column_index_entries(total_rows, index_shift);
  descriptor[1] = index_shift;
  descriptor[2] = index_count;
  descriptor.reserve(checked_size_add(
      descriptor.size(), static_cast<size_t>(index_count), "segmented column index"));

  size_t indexed_fragment = 0;
  for (uint64_t bucket = 0; bucket < index_count; ++bucket) {
    const uint64_t bucket_start = index_shift >= 63 ? 0 : (bucket << index_shift);
    while (indexed_fragment + 1 < fragment_count) {
      const auto entry_offset = 3 + 3 * indexed_fragment;
      const uint64_t start = descriptor[entry_offset + 1];
      const uint64_t row_count = descriptor[entry_offset + 2];
      if (bucket_start < checked_size_add(static_cast<size_t>(start),
                                          static_cast<size_t>(row_count),
                                          "segmented column fragment range")) {
        break;
      }
      ++indexed_fragment;
    }
    descriptor.push_back(static_cast<uint64_t>(indexed_fragment));
  }
  return descriptor;
}

const int8_t* ColumnFetcher::getResultSetColumn(
    const InputColDescriptor* col_desc,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const int frag_id) const {
  CHECK(col_desc);
  const auto table_key = col_desc->getScanDesc().getTableKey();
  return getResultSetColumn(
      get_temporary_table(executor_->temporary_tables_, table_key.table_id),
      table_key,
      col_desc->getColId(),
      memory_level,
      device_id,
      device_allocator,
      thread_idx,
      frag_id);
}

void ColumnFetcher::setResultSetColumnSelection(
    const ResultSetPtr& result_set,
    const std::vector<size_t>& selected_column_indices) const {
  CHECK(result_set);
  if (selected_column_indices.empty() ||
      result_set->getQueryDescriptionType() != QueryDescriptionType::Projection ||
      result_set->didOutputColumnar() || !result_set->hasDeferredLazyFetchChunks()) {
    return;
  }
  auto selection = selected_column_indices;
  for (const auto column_idx : selection) {
    CHECK_LT(column_idx, result_set->colCount());
  }
  std::sort(selection.begin(), selection.end());
  selection.erase(std::unique(selection.begin(), selection.end()), selection.end());
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_fetch_mutex_);
    auto selection_it = result_set_column_selections_.find(result_set.get());
    if (selection_it == result_set_column_selections_.end()) {
      result_set_column_selections_.emplace(
          result_set.get(),
          selection.size() >= result_set->colCount() ? std::vector<size_t>{} : selection);
    } else if (!selection_it->second.empty()) {
      std::vector<size_t> merged_selection;
      merged_selection.reserve(selection_it->second.size() + selection.size());
      std::set_union(selection_it->second.begin(),
                     selection_it->second.end(),
                     selection.begin(),
                     selection.end(),
                     std::back_inserter(merged_selection));
      selection_it->second = merged_selection.size() >= result_set->colCount()
                                 ? std::vector<size_t>{}
                                 : std::move(merged_selection);
    }
  }
  result_set_column_cache_->mergeColumnSelection(result_set, selection);
}

void ColumnFetcher::setResultSetColumnSelections(
    const std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs) const {
  struct ResultColumns {
    ResultSetPtr result_set;
    std::vector<size_t> column_indices;
  };
  std::unordered_map<const ResultSet*, ResultColumns> result_columns;
  for (const auto& col_desc : input_col_descs) {
    CHECK(col_desc);
    if (col_desc->getScanDesc().getSourceType() != InputSourceType::RESULT) {
      continue;
    }
    const auto& table_key = col_desc->getScanDesc().getTableKey();
    auto result_set =
        get_temporary_table(executor_->temporary_tables_, table_key.table_id);
    CHECK(result_set);
    auto& columns = result_columns[result_set.get()];
    columns.result_set = std::move(result_set);
    CHECK_GE(col_desc->getColId(), 0);
    columns.column_indices.push_back(static_cast<size_t>(col_desc->getColId()));
  }
  for (auto& [result_set, columns] : result_columns) {
    CHECK_EQ(result_set, columns.result_set.get());
    setResultSetColumnSelection(columns.result_set, columns.column_indices);
  }
}

const int8_t* ColumnFetcher::getResultSetColumnSegmented(
    const InputColDescriptor* col_desc,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator*,
    const size_t thread_idx,
    const int frag_id,
    const bool allow_direct_peer_access) const {
  CHECK(col_desc);
  CHECK_EQ(memory_level, Data_Namespace::GPU_LEVEL);
  CHECK_GE(frag_id, -1);
  const auto table_key = col_desc->getScanDesc().getTableKey();
  const auto buffer =
      get_temporary_table(executor_->temporary_tables_, table_key.table_id);
  CHECK(buffer);
  const auto col_id = col_desc->getColId();
  CHECK_GE(col_id, 0);

  const auto logical_ti = get_logical_type_info(buffer->getColType(col_id));
  const auto elem_size = logical_ti.get_size();
  CHECK_GT(elem_size, 0);
  CHECK(!logical_ti.is_varlen());

  const bool direct_peer_descriptor =
      allow_direct_peer_access && g_enable_temporary_resultset_payload_peer_access;
  const ResultSetColumnCache::Key cache_key{buffer.get(),
                                            col_id,
                                            memory_level,
                                            device_id,
                                            frag_id,
                                            true,
                                            true,
                                            direct_peer_descriptor};
  if (const auto cached_column = result_set_column_cache_->get(cache_key)) {
    return cached_column;
  }

  auto cuda_mgr = executor_->getDataMgr()->getCudaMgr();
  CHECK(cuda_mgr);
  const auto transfer_stream = cuda_mgr->getDeviceTransferStream(device_id);
  auto allocator_owner = std::make_shared<CudaAllocator>(
      executor_->getDataMgr(), device_id, transfer_stream);
  std::vector<SegmentedResultSetColumnFragment> segmented_fragments;
  bool queued_async_peer_copies = false;

  std::vector<ResultSet::DeviceColumnarBufferFragment> device_fragments;
  const bool has_device_columnar_fragments =
      buffer->getDeviceColumnarBufferFragments(
          col_id, static_cast<size_t>(elem_size), device_fragments) &&
      !device_fragments.empty();
  if (has_device_columnar_fragments) {
    segmented_fragments.reserve(device_fragments.size());
    try {
      for (size_t fragment_idx = 0; fragment_idx < device_fragments.size();
           ++fragment_idx) {
        if (frag_id >= 0 && static_cast<size_t>(frag_id) != fragment_idx) {
          continue;
        }
        const auto& fragment = device_fragments[fragment_idx];
        if (!fragment.entry_count) {
          continue;
        }
        allocator_owner->waitForReadyEvent(fragment.ready_event);
        const auto num_bytes = checked_size_multiply(fragment.entry_count,
                                                     static_cast<size_t>(elem_size),
                                                     "segmented device column fragment");
        const int8_t* fragment_buffer = fragment.buffer;
        const bool can_read_remote_fragment =
            fragment.device_id != device_id && allow_direct_peer_access &&
            g_enable_temporary_resultset_payload_peer_access &&
            cuda_mgr->canAccessPeerMemoryFromKernel(device_id, fragment.device_id) &&
            cuda_mgr->ensurePeerAccessToDevicePtr(
                device_id, fragment.device_id, fragment.buffer, num_bytes);
        if (fragment.device_id != device_id && !can_read_remote_fragment) {
          auto local_fragment_buffer = allocator_owner->alloc(num_bytes);
          cuda_mgr->copyDeviceToDevice(local_fragment_buffer,
                                       const_cast<int8_t*>(fragment.buffer),
                                       num_bytes,
                                       device_id,
                                       fragment.device_id,
                                       "Temporary ResultSet segmented column fragment",
                                       transfer_stream,
                                       false);
          queued_async_peer_copies = true;
          fragment_buffer = local_fragment_buffer;
        }
        segmented_fragments.push_back(
            SegmentedResultSetColumnFragment{fragment_buffer, fragment.entry_count});
      }
    } catch (...) {
      if (queued_async_peer_copies) {
        synchronize_queued_copies_after_error(
            cuda_mgr, transfer_stream, "segmented temporary ResultSet column copy");
      }
      throw;
    }
  }

  if (segmented_fragments.empty()) {
    std::vector<ResultSet::ColumnarBufferFragment> host_fragments;
    if (buffer->getColumnarBufferFragments(
            col_id, static_cast<size_t>(elem_size), host_fragments) &&
        !host_fragments.empty()) {
      segmented_fragments.reserve(host_fragments.size());
      for (size_t fragment_idx = 0; fragment_idx < host_fragments.size();
           ++fragment_idx) {
        if (frag_id >= 0 && static_cast<size_t>(frag_id) != fragment_idx) {
          continue;
        }
        const auto [source_buffer, entry_count] = host_fragments[fragment_idx];
        if (!source_buffer || !entry_count) {
          continue;
        }
        const auto num_bytes = checked_size_multiply(entry_count,
                                                     static_cast<size_t>(elem_size),
                                                     "segmented host column fragment");
        auto local_fragment_buffer = allocator_owner->alloc(num_bytes);
        allocator_owner->copyToDevice(
            local_fragment_buffer,
            source_buffer,
            num_bytes,
            "Temporary ResultSet host segmented column fragment");
        segmented_fragments.push_back(
            SegmentedResultSetColumnFragment{local_fragment_buffer, entry_count});
      }
    }
  }

  if (segmented_fragments.empty()) {
    CHECK(frag_id == -1 || frag_id == 0);
    const auto contiguous_column = getResultSetColumn(buffer,
                                                      table_key,
                                                      col_id,
                                                      memory_level,
                                                      device_id,
                                                      allocator_owner.get(),
                                                      thread_idx,
                                                      frag_id);
    if (!contiguous_column) {
      return nullptr;
    }
    const auto row_count = buffer->rowCount();
    segmented_fragments.push_back(
        SegmentedResultSetColumnFragment{contiguous_column, row_count});
  }

  auto descriptor = makeSegmentedColumnDescriptor(segmented_fragments);
  if (descriptor.empty()) {
    return nullptr;
  }
  const auto descriptor_bytes = checked_size_multiply(
      descriptor.size(), sizeof(uint64_t), "segmented column descriptor bytes");
  auto device_descriptor = allocator_owner->alloc(descriptor_bytes);
  allocator_owner->copyToDevice(device_descriptor,
                                descriptor.data(),
                                descriptor_bytes,
                                "Temporary ResultSet segmented column descriptor");

  auto ready_event = allocator_owner->recordReadyEvent();
  executor_->getCudaAllocator(device_id)->waitForReadyEvent(ready_event);
  auto cache_owner =
      std::make_shared<ResultSetDeviceColumnCacheOwner>(ResultSetDeviceColumnCacheOwner{
          std::move(allocator_owner), std::move(ready_event)});
  return result_set_column_cache_->putOrGetExisting(
      cache_key, buffer, device_descriptor, std::move(cache_owner));
}

const int8_t* ColumnFetcher::linearizeColumnFragments(
    const shared::TableKey& table_key,
    const int col_id,
    const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  auto timer = DEBUG_TIMER(__func__);
  const auto fragments_it = all_tables_fragments.find(table_key);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  const InputColDescriptor col_desc(col_id, table_key.table_id, table_key.db_id, int(0));
  const auto cd = get_column_descriptor({table_key, col_id});
  CHECK(cd);
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  CHECK_GT(table_key.table_id, 0);
  bool is_varlen_chunk = cd->columnType.is_varlen() && !cd->columnType.is_fixlen_array();
  size_t total_num_tuples = 0;
  size_t total_data_buf_size = 0;
  size_t total_idx_buf_size = 0;
  ChunkIter cached_chunk_iter;
  bool has_cached_chunk_iter = false;
  {
    std::lock_guard<std::mutex> linearize_guard(linearized_col_cache_mutex_);
    auto linearized_iter_it = linearized_multi_frag_chunk_iter_cache_.find(col_desc);
    if (linearized_iter_it != linearized_multi_frag_chunk_iter_cache_.end()) {
      if (memory_level == CPU_LEVEL) {
        const auto cached_iter = getChunkiter(col_desc, 0);
        if (cached_iter) {
          cached_chunk_iter = *cached_iter;
          has_cached_chunk_iter = true;
        }
      } else {
        // in GPU execution, this becomes the matter when we deploy multi-GPUs
        // so we only share the chunk_iter iff kernels are launched on the same GPU device
        // otherwise we need to separately load merged chunk and its iter
        // todo(yoonmin): D2D copy of merged chunk and its iter?
        const auto cached_iter = getChunkiter(col_desc, device_id);
        if (cached_iter) {
          cached_chunk_iter = *cached_iter;
          has_cached_chunk_iter = true;
        }
      }
    }
  }
  if (has_cached_chunk_iter) {
    if (memory_level == CPU_LEVEL) {
      std::lock_guard<std::mutex> chunk_list_lock(chunk_list_mutex_);
      chunk_iter_holder.push_back(cached_chunk_iter);
      return reinterpret_cast<int8_t*>(&(chunk_iter_holder.back()));
    }
    CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
    CHECK(device_allocator);
    auto chunk_iter_gpu = device_allocator->alloc(sizeof(ChunkIter));
    device_allocator->copyToDevice(
        chunk_iter_gpu, &cached_chunk_iter, sizeof(ChunkIter), "Chunk iterator");
    return chunk_iter_gpu;
  }

  // collect target fragments
  // basically we load chunk in CPU first, and do necessary manipulation
  // to make semantics of a merged chunk correctly
  std::shared_ptr<Chunk_NS::Chunk> chunk;
  std::list<std::shared_ptr<Chunk_NS::Chunk>> local_chunk_holder;
  std::list<ChunkIter> local_chunk_iter_holder;
  std::list<size_t> local_chunk_num_tuples;
  {
    std::lock_guard<std::mutex> linearize_guard(varlen_chunk_fetch_mutex_);
    for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
      const auto& fragment = (*fragments)[frag_id];
      if (fragment.isEmptyPhysicalFragment()) {
        continue;
      }
      auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
      CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
      ChunkKey chunk_key{
          table_key.db_id, fragment.physicalTableId, col_id, fragment.fragmentId};
      chunk = Chunk_NS::Chunk::getChunk(cd,
                                        executor_->getDataMgr(),
                                        chunk_key,
                                        Data_Namespace::CPU_LEVEL,
                                        0,
                                        chunk_meta_it->second->numBytes,
                                        chunk_meta_it->second->numElements);
      local_chunk_holder.push_back(chunk);
      auto chunk_iter = chunk->begin_iterator(chunk_meta_it->second);
      local_chunk_iter_holder.push_back(chunk_iter);
      local_chunk_num_tuples.push_back(fragment.getNumTuples());
      total_num_tuples += fragment.getNumTuples();
      total_data_buf_size += chunk->getBuffer()->size();
      std::ostringstream oss;
      oss << "Load chunk for col_name: " << chunk->getColumnDesc()->columnName
          << ", col_id: " << chunk->getColumnDesc()->columnId << ", Frag-" << frag_id
          << ", numTuples: " << fragment.getNumTuples()
          << ", data_size: " << chunk->getBuffer()->size();
      if (chunk->getIndexBuf()) {
        auto idx_buf_size = chunk->getIndexBuf()->size() - sizeof(ArrayOffsetT);
        oss << ", index_size: " << idx_buf_size;
        total_idx_buf_size += idx_buf_size;
      }
      VLOG(2) << oss.str();
    }
  }

  auto& col_ti = cd->columnType;
  MergedChunk res{nullptr, nullptr};
  // Do linearize multi-fragmented column depending on column type
  // We cover array and non-encoded text columns
  // Note that geo column is actually organized as a set of arrays
  // and each geo object has different set of vectors that they require
  // Here, we linearize each array at a time, so eventually the geo object has a set of
  // "linearized" arrays
  {
    std::lock_guard<std::mutex> linearization_guard(linearization_mutex_);
    if (col_ti.is_array()) {
      if (col_ti.is_fixlen_array()) {
        VLOG(2) << "Linearize fixed-length multi-frag array column (col_id: "
                << cd->columnId << ", col_name: " << cd->columnName
                << ", device_type: " << getMemoryLevelString(memory_level)
                << ", device_id: " << device_id << "): " << cd->columnType.to_string();
        res = linearizeFixedLenArrayColFrags(table_key.db_id,
                                             chunk_holder,
                                             chunk_iter_holder,
                                             local_chunk_holder,
                                             local_chunk_iter_holder,
                                             local_chunk_num_tuples,
                                             memory_level,
                                             cd,
                                             device_id,
                                             total_data_buf_size,
                                             total_idx_buf_size,
                                             total_num_tuples,
                                             device_allocator,
                                             thread_idx);
      } else {
        CHECK(col_ti.is_varlen_array());
        VLOG(2) << "Linearize variable-length multi-frag array column (col_id: "
                << cd->columnId << ", col_name: " << cd->columnName
                << ", device_type: " << getMemoryLevelString(memory_level)
                << ", device_id: " << device_id << "): " << cd->columnType.to_string();
        res = linearizeVarLenArrayColFrags(table_key.db_id,
                                           chunk_holder,
                                           chunk_iter_holder,
                                           local_chunk_holder,
                                           local_chunk_iter_holder,
                                           local_chunk_num_tuples,
                                           memory_level,
                                           cd,
                                           device_id,
                                           total_data_buf_size,
                                           total_idx_buf_size,
                                           total_num_tuples,
                                           device_allocator,
                                           thread_idx);
      }
    }
    if (col_ti.is_string() && !col_ti.is_dict_encoded_string()) {
      VLOG(2) << "Linearize variable-length multi-frag non-encoded text column (col_id: "
              << cd->columnId << ", col_name: " << cd->columnName
              << ", device_type: " << getMemoryLevelString(memory_level)
              << ", device_id: " << device_id << "): " << cd->columnType.to_string();
      res = linearizeVarLenArrayColFrags(table_key.db_id,
                                         chunk_holder,
                                         chunk_iter_holder,
                                         local_chunk_holder,
                                         local_chunk_iter_holder,
                                         local_chunk_num_tuples,
                                         memory_level,
                                         cd,
                                         device_id,
                                         total_data_buf_size,
                                         total_idx_buf_size,
                                         total_num_tuples,
                                         device_allocator,
                                         thread_idx);
    }
  }
  CHECK(res.first);  // check merged data buffer
  if (!col_ti.is_fixlen_array()) {
    CHECK(res.second);  // check merged index buffer
  }
  auto merged_data_buffer = res.first;
  auto merged_index_buffer = res.second;

  // prepare ChunkIter for the linearized chunk
  auto merged_chunk = std::make_shared<Chunk_NS::Chunk>(
      merged_data_buffer, merged_index_buffer, cd, false);
  // to prepare chunk_iter for the merged chunk, we pass one of local chunk iter
  // to fill necessary metadata that is a common for all merged chunks
  auto merged_chunk_iter = prepareChunkIter(merged_data_buffer,
                                            merged_index_buffer,
                                            *(local_chunk_iter_holder.rbegin()),
                                            is_varlen_chunk,
                                            total_num_tuples);
  {
    std::lock_guard<std::mutex> chunk_list_lock(chunk_list_mutex_);
    chunk_holder.push_back(merged_chunk);
    chunk_iter_holder.push_back(merged_chunk_iter);
  }

  auto merged_chunk_iter_ptr = reinterpret_cast<int8_t*>(&(chunk_iter_holder.back()));
  if (memory_level == MemoryLevel::CPU_LEVEL) {
    addMergedChunkIter(col_desc, 0, merged_chunk_iter);
    return merged_chunk_iter_ptr;
  } else {
    CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
    CHECK(device_allocator);
    addMergedChunkIter(col_desc, device_id, merged_chunk_iter);
    // note that merged_chunk_iter_ptr resides in CPU memory space
    // having its content aware GPU buffer that we alloc. for merging
    // so we need to copy this chunk_iter to each device explicitly
    auto chunk_iter_gpu = device_allocator->alloc(sizeof(ChunkIter));
    device_allocator->copyToDevice(
        chunk_iter_gpu, merged_chunk_iter_ptr, sizeof(ChunkIter), "Chunk iterator");
    return chunk_iter_gpu;
  }
}

MergedChunk ColumnFetcher::linearizeVarLenArrayColFrags(
    int32_t db_id,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& local_chunk_holder,
    std::list<ChunkIter>& local_chunk_iter_holder,
    std::list<size_t>& local_chunk_num_tuples,
    MemoryLevel memory_level,
    const ColumnDescriptor* cd,
    const int device_id,
    const size_t total_data_buf_size,
    const size_t total_idx_buf_size,
    const size_t total_num_tuples,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  // for linearization of varlen col we have to deal with not only data buffer
  // but also its underlying index buffer which is responsible for offset of varlen value
  // basically we maintain per-device linearized (data/index) buffer
  // for data buffer, we linearize varlen col's chunks within a device-specific buffer
  // by just appending each chunk
  // for index buffer, we need to not only appending each chunk but modify the offset
  // value to affect various conditions like nullness, padding and so on so we first
  // append index buffer in CPU, manipulate it as we required and then copy it to specific
  // device if necessary (for GPU execution)
  AbstractBuffer* merged_index_buffer_in_cpu = nullptr;
  AbstractBuffer* merged_data_buffer = nullptr;
  bool has_cached_merged_idx_buf = false;
  bool has_cached_merged_data_buf = false;
  const InputColDescriptor icd(cd->columnId, cd->tableId, db_id, int(0));
  // check linearized buffer's cache first
  // if not exists, alloc necessary buffer space to prepare linearization
  int64_t linearization_time_ms = 0;
  auto clock_begin = timer_start();
  {
    std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
    auto cached_data_buf_cache_it = linearized_data_buf_cache_.find(icd);
    if (cached_data_buf_cache_it != linearized_data_buf_cache_.end()) {
      auto& cd_cache = cached_data_buf_cache_it->second;
      auto cached_data_buf_it = cd_cache.find(device_id);
      if (cached_data_buf_it != cd_cache.end()) {
        has_cached_merged_data_buf = true;
        merged_data_buffer = cached_data_buf_it->second;
        VLOG(2) << "Recycle merged data buffer for linearized chunks (memory_level: "
                << getMemoryLevelString(memory_level) << ", device_id: " << device_id
                << ")";
      } else {
        merged_data_buffer =
            executor_->getDataMgr()->alloc(memory_level, device_id, total_data_buf_size);
        VLOG(2) << "Allocate " << total_data_buf_size
                << " bytes of data buffer space for linearized chunks (memory_level: "
                << getMemoryLevelString(memory_level) << ", device_id: " << device_id
                << ")";
        cd_cache.insert(std::make_pair(device_id, merged_data_buffer));
      }
    } else {
      DeviceMergedChunkMap m;
      merged_data_buffer =
          executor_->getDataMgr()->alloc(memory_level, device_id, total_data_buf_size);
      VLOG(2) << "Allocate " << total_data_buf_size
              << " bytes of data buffer space for linearized chunks (memory_level: "
              << getMemoryLevelString(memory_level) << ", device_id: " << device_id
              << ")";
      m.insert(std::make_pair(device_id, merged_data_buffer));
      linearized_data_buf_cache_.insert(std::make_pair(icd, m));
    }

    auto cached_index_buf_it =
        linearlized_temporary_cpu_index_buf_cache_.find(cd->columnId);
    if (cached_index_buf_it != linearlized_temporary_cpu_index_buf_cache_.end()) {
      has_cached_merged_idx_buf = true;
      merged_index_buffer_in_cpu = cached_index_buf_it->second;
      VLOG(2)
          << "Recycle merged temporary idx buffer for linearized chunks (memory_level: "
          << getMemoryLevelString(memory_level) << ", device_id: " << device_id << ")";
    } else {
      auto idx_buf_size = total_idx_buf_size + sizeof(ArrayOffsetT);
      merged_index_buffer_in_cpu =
          executor_->getDataMgr()->alloc(Data_Namespace::CPU_LEVEL, 0, idx_buf_size);
      VLOG(2) << "Allocate " << idx_buf_size
              << " bytes of temporary idx buffer space on CPU for linearized chunks";
      // just copy the buf addr since we access it via the pointer itself
      linearlized_temporary_cpu_index_buf_cache_.insert(
          std::make_pair(cd->columnId, merged_index_buffer_in_cpu));
    }
  }

  // linearize buffers if we don't have corresponding buf in cache
  size_t sum_data_buf_size = 0;
  size_t cur_sum_num_tuples = 0;
  size_t total_idx_size_modifier = 0;
  auto chunk_holder_it = local_chunk_holder.begin();
  auto chunk_iter_holder_it = local_chunk_iter_holder.begin();
  auto chunk_num_tuple_it = local_chunk_num_tuples.begin();
  bool null_padded_first_elem = false;
  bool null_padded_last_val = false;
  // before entering the actual linearization part, we first need to check
  // the overflow case where the sum of index offset becomes larger than 2GB
  // which currently incurs incorrect query result due to negative array offset
  // note that we can separate this from the main linearization logic b/c
  // we just need to see few last elems
  // todo (yoonmin) : relax this to support larger chunk size (>2GB)
  for (; chunk_holder_it != local_chunk_holder.end();
       chunk_holder_it++, chunk_num_tuple_it++) {
    // check the offset overflow based on the last "valid" offset for each chunk
    auto target_chunk = chunk_holder_it->get();
    auto target_chunk_data_buffer = target_chunk->getBuffer();
    auto target_chunk_idx_buffer = target_chunk->getIndexBuf();
    auto target_idx_buf_ptr =
        reinterpret_cast<ArrayOffsetT*>(target_chunk_idx_buffer->getMemoryPtr());
    auto cur_chunk_num_tuples = *chunk_num_tuple_it;
    ArrayOffsetT original_offset = -1;
    size_t cur_idx = cur_chunk_num_tuples;
    // find the valid (e.g., non-null) offset starting from the last elem
    while (original_offset < 0) {
      original_offset = target_idx_buf_ptr[--cur_idx];
    }
    ArrayOffsetT new_offset = original_offset + sum_data_buf_size;
    if (new_offset < 0) {
      throw std::runtime_error(
          "Linearization of a variable-length column having chunk size larger than 2GB "
          "not supported yet");
    }
    sum_data_buf_size += target_chunk_data_buffer->size();
  }
  chunk_holder_it = local_chunk_holder.begin();
  chunk_num_tuple_it = local_chunk_num_tuples.begin();
  sum_data_buf_size = 0;

  for (; chunk_holder_it != local_chunk_holder.end();
       chunk_holder_it++, chunk_iter_holder_it++, chunk_num_tuple_it++) {
    if (g_enable_non_kernel_time_query_interrupt &&
        executor_->checkNonKernelTimeInterrupted()) {
      throw QueryExecutionError(ErrorCode::INTERRUPTED);
    }
    auto target_chunk = chunk_holder_it->get();
    auto target_chunk_data_buffer = target_chunk->getBuffer();
    auto cur_chunk_num_tuples = *chunk_num_tuple_it;
    auto target_chunk_idx_buffer = target_chunk->getIndexBuf();
    auto target_idx_buf_ptr =
        reinterpret_cast<ArrayOffsetT*>(target_chunk_idx_buffer->getMemoryPtr());
    auto idx_buf_size = target_chunk_idx_buffer->size() - sizeof(ArrayOffsetT);
    auto target_data_buffer_start_ptr = target_chunk_data_buffer->getMemoryPtr();
    auto target_data_buffer_size = target_chunk_data_buffer->size();

    // when linearizing idx buffers, we need to consider the following cases
    // 1. the first idx val is padded (a. null / b. empty varlen arr / c. 1-byte size
    // varlen arr, i.e., {1})
    // 2. the last idx val is null
    // 3. null value(s) is/are located in a middle of idx buf <-- we don't need to care
    if (cur_sum_num_tuples > 0 && target_idx_buf_ptr[0] > 0) {
      null_padded_first_elem = true;
      target_data_buffer_start_ptr += ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE;
      target_data_buffer_size -= ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE;
      total_idx_size_modifier += ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE;
    }
    // we linearize data_buf in device-specific buffer
    if (!has_cached_merged_data_buf) {
      merged_data_buffer->append(target_data_buffer_start_ptr,
                                 target_data_buffer_size,
                                 Data_Namespace::CPU_LEVEL,
                                 device_id);
    }

    if (!has_cached_merged_idx_buf) {
      // linearize idx buf in CPU first
      merged_index_buffer_in_cpu->append(target_chunk_idx_buffer->getMemoryPtr(),
                                         idx_buf_size,
                                         Data_Namespace::CPU_LEVEL,
                                         0);  // merged_index_buffer_in_cpu resides in CPU
      auto idx_buf_ptr =
          reinterpret_cast<ArrayOffsetT*>(merged_index_buffer_in_cpu->getMemoryPtr());
      // here, we do not need to manipulate the very first idx buf, just let it as is
      // and modify otherwise (i.e., starting from second chunk idx buf)
      if (cur_sum_num_tuples > 0) {
        if (null_padded_last_val) {
          // case 2. the previous chunk's last index val is null so we need to set this
          // chunk's first val to be null
          idx_buf_ptr[cur_sum_num_tuples] = -sum_data_buf_size;
        }
        const size_t worker_count = cpu_threads();
        std::vector<std::future<void>> conversion_threads;
        std::vector<std::vector<size_t>> null_padded_row_idx_vecs(worker_count,
                                                                  std::vector<size_t>());
        bool is_parallel_modification = false;
        std::vector<size_t> null_padded_row_idx_vec;
        const auto do_work = [&cur_sum_num_tuples,
                              &sum_data_buf_size,
                              &null_padded_first_elem,
                              &idx_buf_ptr](
                                 const size_t start,
                                 const size_t end,
                                 const bool is_parallel_modification,
                                 std::vector<size_t>* null_padded_row_idx_vec) {
          for (size_t i = start; i < end; i++) {
            if (LIKELY(idx_buf_ptr[cur_sum_num_tuples + i] >= 0)) {
              if (null_padded_first_elem) {
                // deal with null padded bytes
                idx_buf_ptr[cur_sum_num_tuples + i] -=
                    ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE;
              }
              idx_buf_ptr[cur_sum_num_tuples + i] += sum_data_buf_size;
            } else {
              // null padded row needs to reference the previous row idx so in
              // multi-threaded index modification we may suffer from thread
              // contention when thread-i needs to reference thread-j's row idx so we
              // collect row idxs for null rows here and deal with them after this
              // step
              null_padded_row_idx_vec->push_back(cur_sum_num_tuples + i);
            }
          }
        };
        if (cur_chunk_num_tuples > g_enable_parallel_linearization) {
          is_parallel_modification = true;
          for (auto interval :
               makeIntervals(size_t(0), cur_chunk_num_tuples, worker_count)) {
            conversion_threads.push_back(
                std::async(std::launch::async,
                           do_work,
                           interval.begin,
                           interval.end,
                           is_parallel_modification,
                           &null_padded_row_idx_vecs[interval.index]));
          }
          for (auto& child : conversion_threads) {
            child.wait();
          }
          for (auto& v : null_padded_row_idx_vecs) {
            std::copy(v.begin(), v.end(), std::back_inserter(null_padded_row_idx_vec));
          }
        } else {
          do_work(size_t(0),
                  cur_chunk_num_tuples,
                  is_parallel_modification,
                  &null_padded_row_idx_vec);
        }
        if (!null_padded_row_idx_vec.empty()) {
          // modify null padded row idxs by referencing the previous row
          // here we sort row idxs to correctly propagate modified row idxs
          std::sort(null_padded_row_idx_vec.begin(), null_padded_row_idx_vec.end());
          for (auto& padded_null_row_idx : null_padded_row_idx_vec) {
            if (idx_buf_ptr[padded_null_row_idx - 1] > 0) {
              idx_buf_ptr[padded_null_row_idx] = -idx_buf_ptr[padded_null_row_idx - 1];
            } else {
              idx_buf_ptr[padded_null_row_idx] = idx_buf_ptr[padded_null_row_idx - 1];
            }
          }
        }
      }
    }
    cur_sum_num_tuples += cur_chunk_num_tuples;
    sum_data_buf_size += target_chunk_data_buffer->size();
    if (target_idx_buf_ptr[*chunk_num_tuple_it] < 0) {
      null_padded_last_val = true;
    } else {
      null_padded_last_val = false;
    }
    if (null_padded_first_elem) {
      sum_data_buf_size -= ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE;
      null_padded_first_elem = false;  // set for the next chunk
    }
    if (!has_cached_merged_idx_buf && cur_sum_num_tuples == total_num_tuples) {
      auto merged_index_buffer_ptr =
          reinterpret_cast<ArrayOffsetT*>(merged_index_buffer_in_cpu->getMemoryPtr());
      merged_index_buffer_ptr[total_num_tuples] =
          total_data_buf_size -
          total_idx_size_modifier;  // last index value is total data size;
    }
  }

  // put linearized index buffer to per-device cache
  AbstractBuffer* merged_index_buffer = nullptr;
  size_t buf_size = total_idx_buf_size + sizeof(ArrayOffsetT);
  auto copyBuf =
      [&device_allocator](
          int8_t* src, int8_t* dest, size_t buf_size, MemoryLevel memory_level) {
        if (memory_level == Data_Namespace::CPU_LEVEL) {
          memcpy((void*)dest, src, buf_size);
        } else {
          CHECK(memory_level == Data_Namespace::GPU_LEVEL);
          device_allocator->copyToDevice(dest, src, buf_size, "Linearized column buffer");
        }
      };
  {
    std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
    auto merged_idx_buf_cache_it = linearized_idx_buf_cache_.find(icd);
    // for CPU execution, we can use `merged_index_buffer_in_cpu` as is
    // but for GPU, we have to copy it to corresponding device
    if (memory_level == MemoryLevel::GPU_LEVEL) {
      if (merged_idx_buf_cache_it != linearized_idx_buf_cache_.end()) {
        auto& merged_idx_buf_cache = merged_idx_buf_cache_it->second;
        auto merged_idx_buf_it = merged_idx_buf_cache.find(device_id);
        if (merged_idx_buf_it != merged_idx_buf_cache.end()) {
          merged_index_buffer = merged_idx_buf_it->second;
        } else {
          merged_index_buffer =
              executor_->getDataMgr()->alloc(memory_level, device_id, buf_size);
          copyBuf(merged_index_buffer_in_cpu->getMemoryPtr(),
                  merged_index_buffer->getMemoryPtr(),
                  buf_size,
                  memory_level);
          merged_idx_buf_cache.insert(std::make_pair(device_id, merged_index_buffer));
        }
      } else {
        merged_index_buffer =
            executor_->getDataMgr()->alloc(memory_level, device_id, buf_size);
        copyBuf(merged_index_buffer_in_cpu->getMemoryPtr(),
                merged_index_buffer->getMemoryPtr(),
                buf_size,
                memory_level);
        DeviceMergedChunkMap m;
        m.insert(std::make_pair(device_id, merged_index_buffer));
        linearized_idx_buf_cache_.insert(std::make_pair(icd, m));
      }
    } else {
      // `linearlized_temporary_cpu_index_buf_cache_` has this buf
      merged_index_buffer = merged_index_buffer_in_cpu;
    }
  }
  CHECK(merged_index_buffer);
  linearization_time_ms += timer_stop(clock_begin);
  VLOG(2) << "Linearization has been successfully done, elapsed time: "
          << linearization_time_ms << " ms.";
  return {merged_data_buffer, merged_index_buffer};
}

MergedChunk ColumnFetcher::linearizeFixedLenArrayColFrags(
    int32_t db_id,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& local_chunk_holder,
    std::list<ChunkIter>& local_chunk_iter_holder,
    std::list<size_t>& local_chunk_num_tuples,
    MemoryLevel memory_level,
    const ColumnDescriptor* cd,
    const int device_id,
    const size_t total_data_buf_size,
    const size_t total_idx_buf_size,
    const size_t total_num_tuples,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  int64_t linearization_time_ms = 0;
  auto clock_begin = timer_start();
  // linearize collected fragments
  AbstractBuffer* merged_data_buffer = nullptr;
  bool has_cached_merged_data_buf = false;
  const InputColDescriptor icd(cd->columnId, cd->tableId, db_id, int(0));
  {
    std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
    auto cached_data_buf_cache_it = linearized_data_buf_cache_.find(icd);
    if (cached_data_buf_cache_it != linearized_data_buf_cache_.end()) {
      auto& cd_cache = cached_data_buf_cache_it->second;
      auto cached_data_buf_it = cd_cache.find(device_id);
      if (cached_data_buf_it != cd_cache.end()) {
        has_cached_merged_data_buf = true;
        merged_data_buffer = cached_data_buf_it->second;
        VLOG(2) << "Recycle merged data buffer for linearized chunks (memory_level: "
                << getMemoryLevelString(memory_level) << ", device_id: " << device_id
                << ")";
      } else {
        merged_data_buffer =
            executor_->getDataMgr()->alloc(memory_level, device_id, total_data_buf_size);
        VLOG(2) << "Allocate " << total_data_buf_size
                << " bytes of data buffer space for linearized chunks (memory_level: "
                << getMemoryLevelString(memory_level) << ", device_id: " << device_id
                << ")";
        cd_cache.insert(std::make_pair(device_id, merged_data_buffer));
      }
    } else {
      DeviceMergedChunkMap m;
      merged_data_buffer =
          executor_->getDataMgr()->alloc(memory_level, device_id, total_data_buf_size);
      VLOG(2) << "Allocate " << total_data_buf_size
              << " bytes of data buffer space for linearized chunks (memory_level: "
              << getMemoryLevelString(memory_level) << ", device_id: " << device_id
              << ")";
      m.insert(std::make_pair(device_id, merged_data_buffer));
      linearized_data_buf_cache_.insert(std::make_pair(icd, m));
    }
  }
  if (!has_cached_merged_data_buf) {
    size_t sum_data_buf_size = 0;
    auto chunk_holder_it = local_chunk_holder.begin();
    auto chunk_iter_holder_it = local_chunk_iter_holder.begin();
    for (; chunk_holder_it != local_chunk_holder.end();
         chunk_holder_it++, chunk_iter_holder_it++) {
      if (g_enable_non_kernel_time_query_interrupt && check_interrupt()) {
        throw QueryExecutionError(ErrorCode::INTERRUPTED);
      }
      auto target_chunk = chunk_holder_it->get();
      auto target_chunk_data_buffer = target_chunk->getBuffer();
      merged_data_buffer->append(target_chunk_data_buffer->getMemoryPtr(),
                                 target_chunk_data_buffer->size(),
                                 Data_Namespace::CPU_LEVEL,
                                 device_id);
      sum_data_buf_size += target_chunk_data_buffer->size();
    }
    // check whether each chunk's data buffer is clean under chunk merging
    CHECK_EQ(total_data_buf_size, sum_data_buf_size);
  }
  linearization_time_ms += timer_stop(clock_begin);
  VLOG(2) << "Linearization has been successfully done, elapsed time: "
          << linearization_time_ms << " ms.";
  return {merged_data_buffer, nullptr};
}

const int8_t* ColumnFetcher::transferColumnIfNeeded(
    const ColumnarResults* columnar_results,
    const int col_id,
    Data_Namespace::DataMgr* data_mgr,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator) {
  if (!columnar_results) {
    return nullptr;
  }
  const auto& col_buffers = columnar_results->getColumnBuffers();
  CHECK_LT(static_cast<size_t>(col_id), col_buffers.size());
  if (memory_level == Data_Namespace::GPU_LEVEL) {
    const auto num_bytes = getColumnarResultsColumnBytes(columnar_results, col_id);
    CHECK(device_allocator);
    auto gpu_col_buffer = device_allocator->alloc(num_bytes);
    device_allocator->copyToDevice(
        gpu_col_buffer, col_buffers[col_id], num_bytes, "Columnarized column buffer");
    return gpu_col_buffer;
  }
  return col_buffers[col_id];
}

std::pair<const int8_t*, size_t> transferResultSetColumnFragmentIfNeeded(
    const ResultSetPtr& buffer,
    const int col_id,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const int frag_id) {
  if (!buffer || frag_id < 0) {
    return {nullptr, 0};
  }
  const auto logical_ti = get_logical_type_info(buffer->getColType(col_id));
  const auto elem_size = logical_ti.get_size();
  if (elem_size <= 0 || logical_ti.is_varlen()) {
    return {nullptr, 0};
  }
  std::vector<ResultSet::ColumnarBufferFragment> fragments;
  if (!buffer->getColumnarBufferFragments(
          col_id, static_cast<size_t>(elem_size), fragments)) {
    return {nullptr, 0};
  }
  if (static_cast<size_t>(frag_id) >= fragments.size()) {
    return {nullptr, 0};
  }
  const auto [source_buffer, entry_count] = fragments[frag_id];
  CHECK(source_buffer);
  const auto num_bytes = checked_size_multiply(
      entry_count, static_cast<size_t>(elem_size), "columnar ResultSet fragment");
  if (memory_level == Data_Namespace::GPU_LEVEL) {
    CHECK(device_allocator);
    auto gpu_col_buffer = device_allocator->alloc(num_bytes);
    device_allocator->copyToDevice(gpu_col_buffer,
                                   source_buffer,
                                   num_bytes,
                                   "Columnarized ResultSet column fragment");
    return {gpu_col_buffer, num_bytes};
  }
  return {source_buffer, num_bytes};
}

ResultSetDeviceColumnTransfer transferResultSetDeviceColumnToHostIfAvailable(
    const ResultSetPtr& buffer,
    const int col_id,
    const int frag_id,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const size_t thread_idx) {
  CHECK(buffer);
  CHECK(row_set_mem_owner);
  const auto logical_ti = get_logical_type_info(buffer->getColType(col_id));
  const auto elem_size = logical_ti.get_size();
  if (elem_size <= 0 || logical_ti.is_varlen()) {
    return {};
  }

  std::vector<ResultSet::DeviceColumnarBufferFragment> fragments;
  if (!buffer->getDeviceColumnarBufferFragments(
          col_id, static_cast<size_t>(elem_size), fragments) ||
      fragments.empty()) {
    return {};
  }

  if (frag_id >= 0) {
    if (static_cast<size_t>(frag_id) >= fragments.size()) {
      return {};
    }
    const auto& fragment = fragments[frag_id];
    const auto num_bytes = checked_size_multiply(fragment.entry_count,
                                                 static_cast<size_t>(elem_size),
                                                 "device ResultSet fragment to host");
    if (!num_bytes) {
      return {};
    }
    CHECK(fragment.buffer);
    CHECK(fragment.owner);
    fragment.owner->waitForReadyEvent(fragment.ready_event);
    auto* host_buffer = row_set_mem_owner->allocate(num_bytes, thread_idx);
    fragment.owner->copyFromDevice(host_buffer,
                                   fragment.buffer,
                                   num_bytes,
                                   "Temporary ResultSet column fragment to host");
    return {host_buffer, num_bytes, false};
  }

  size_t total_rows{0};
  for (const auto& fragment : fragments) {
    total_rows = checked_size_add(
        total_rows, fragment.entry_count, "device ResultSet rows to host");
  }
  if (!total_rows) {
    return {};
  }

  const auto total_bytes = checked_size_multiply(
      total_rows, static_cast<size_t>(elem_size), "device ResultSet column to host");
  auto* host_buffer = row_set_mem_owner->allocate(total_bytes, thread_idx);
  size_t byte_offset{0};
  for (const auto& fragment : fragments) {
    const auto fragment_bytes =
        checked_size_multiply(fragment.entry_count,
                              static_cast<size_t>(elem_size),
                              "device ResultSet fragment to host");
    if (!fragment_bytes) {
      continue;
    }
    CHECK(fragment.buffer);
    CHECK(fragment.owner);
    fragment.owner->waitForReadyEvent(fragment.ready_event);
    fragment.owner->copyFromDevice(host_buffer + byte_offset,
                                   fragment.buffer,
                                   fragment_bytes,
                                   "Temporary ResultSet all-fragment column to host");
    byte_offset = checked_size_add(
        byte_offset, fragment_bytes, "device ResultSet host column offset");
  }
  CHECK_EQ(total_bytes, byte_offset);
  return {host_buffer, total_bytes, false};
}

ResultSetDeviceColumnTransfer transferResultSetDeviceColumnIfAvailable(
    Executor* executor,
    const ResultSetPtr& buffer,
    const int col_id,
    const int device_id,
    DeviceAllocator* device_allocator,
    const int frag_id) {
  CHECK(executor);
  CHECK(buffer);
  CHECK(device_allocator);
  const auto logical_ti = get_logical_type_info(buffer->getColType(col_id));
  const auto elem_size = logical_ti.get_size();
  if (elem_size <= 0 || logical_ti.is_varlen()) {
    return {};
  }

  std::vector<ResultSet::DeviceColumnarBufferFragment> fragments;
  if (!buffer->getDeviceColumnarBufferFragments(
          col_id, static_cast<size_t>(elem_size), fragments) ||
      fragments.empty()) {
    return {};
  }

  const auto copy_device_fragment =
      [&](int8_t* dest,
          const ResultSet::DeviceColumnarBufferFragment& fragment,
          const size_t num_bytes,
          const std::string_view tag,
          const bool synchronize = true) {
        auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
        CHECK(cuda_mgr);
        executor->getCudaAllocator(device_id)->waitForReadyEvent(fragment.ready_event);
        cuda_mgr->copyDeviceToDevice(dest,
                                     const_cast<int8_t*>(fragment.buffer),
                                     num_bytes,
                                     device_id,
                                     fragment.device_id,
                                     tag,
                                     executor->getCudaStream(device_id),
                                     synchronize);
        return !synchronize;
      };

  if (frag_id >= 0) {
    if (static_cast<size_t>(frag_id) >= fragments.size()) {
      return {};
    }
    const auto& fragment = fragments[frag_id];
    const auto num_bytes = checked_size_multiply(fragment.entry_count,
                                                 static_cast<size_t>(elem_size),
                                                 "device ResultSet column fragment");
    if (!num_bytes) {
      return {};
    }
    if (fragment.device_id == device_id) {
      executor->getCudaAllocator(device_id)->waitForReadyEvent(fragment.ready_event);
      return {fragment.buffer, num_bytes, false};
    }
    auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
    CHECK(cuda_mgr);
    if (g_enable_temporary_resultset_peer_access &&
        cuda_mgr->canAccessPeerMemoryFromKernel(device_id, fragment.device_id) &&
        cuda_mgr->ensurePeerAccessToDevicePtr(
            device_id, fragment.device_id, fragment.buffer, num_bytes)) {
      executor->getCudaAllocator(device_id)->waitForReadyEvent(fragment.ready_event);
      return {fragment.buffer, num_bytes, false};
    }
    auto dest_buffer = device_allocator->alloc(num_bytes);
    copy_device_fragment(
        dest_buffer, fragment, num_bytes, "Temporary ResultSet column fragment");
    return {dest_buffer, num_bytes, true};
  }

  size_t total_rows{0};
  for (const auto& fragment : fragments) {
    total_rows = checked_size_add(
        total_rows, fragment.entry_count, "device ResultSet column rows");
  }
  if (!total_rows) {
    return {};
  }

  const auto total_bytes = checked_size_multiply(
      total_rows, static_cast<size_t>(elem_size), "device ResultSet column");
  auto dest_buffer = device_allocator->alloc(total_bytes);
  size_t byte_offset{0};
  bool queued_async_peer_copies = false;
  try {
    for (const auto& fragment : fragments) {
      const auto fragment_bytes =
          checked_size_multiply(fragment.entry_count,
                                static_cast<size_t>(elem_size),
                                "device ResultSet column fragment");
      if (!fragment_bytes) {
        continue;
      }
      executor->getCudaAllocator(device_id)->waitForReadyEvent(fragment.ready_event);
      executor->getDataMgr()->getCudaMgr()->copyDeviceToDevice(
          dest_buffer + byte_offset,
          const_cast<int8_t*>(fragment.buffer),
          fragment_bytes,
          device_id,
          fragment.device_id,
          "Temporary ResultSet all-fragment column",
          executor->getCudaStream(device_id),
          false);
      queued_async_peer_copies = true;
      byte_offset =
          checked_size_add(byte_offset, fragment_bytes, "device ResultSet column offset");
    }
  } catch (...) {
    if (queued_async_peer_copies) {
      auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
      CHECK(cuda_mgr);
      synchronize_queued_copies_after_error(cuda_mgr,
                                            executor->getCudaStream(device_id),
                                            "temporary ResultSet column copy");
    }
    throw;
  }
  if (queued_async_peer_copies) {
    auto cuda_mgr = executor->getDataMgr()->getCudaMgr();
    CHECK(cuda_mgr);
    cuda_mgr->synchronizeStream(executor->getCudaStream(device_id));
  }
  CHECK_EQ(total_bytes, byte_offset);
  return {dest_buffer, total_bytes, true};
}

void ColumnFetcher::addMergedChunkIter(const InputColDescriptor col_desc,
                                       const int device_id,
                                       const ChunkIter& chunk_iter) const {
  std::lock_guard<std::mutex> linearize_guard(linearized_col_cache_mutex_);
  auto chunk_iter_it = linearized_multi_frag_chunk_iter_cache_.find(col_desc);
  if (chunk_iter_it != linearized_multi_frag_chunk_iter_cache_.end()) {
    auto iter_device_it = chunk_iter_it->second.find(device_id);
    if (iter_device_it == chunk_iter_it->second.end()) {
      VLOG(2) << "Additional merged chunk_iter for col_desc (tbl: "
              << col_desc.getScanDesc().getTableKey() << ", col: " << col_desc.getColId()
              << "), device_id: " << device_id;
      chunk_iter_it->second.emplace(device_id, chunk_iter);
    }
  } else {
    DeviceMergedChunkIterMap iter_m;
    iter_m.emplace(device_id, chunk_iter);
    VLOG(2) << "New merged chunk_iter for col_desc (tbl: "
            << col_desc.getScanDesc().getTableKey() << ", col: " << col_desc.getColId()
            << "), device_id: " << device_id;
    linearized_multi_frag_chunk_iter_cache_.emplace(col_desc, iter_m);
  }
}

const ChunkIter* ColumnFetcher::getChunkiter(const InputColDescriptor col_desc,
                                             const int device_id) const {
  auto linearized_chunk_iter_it = linearized_multi_frag_chunk_iter_cache_.find(col_desc);
  if (linearized_chunk_iter_it != linearized_multi_frag_chunk_iter_cache_.end()) {
    auto dev_iter_map_it = linearized_chunk_iter_it->second.find(device_id);
    if (dev_iter_map_it != linearized_chunk_iter_it->second.end()) {
      VLOG(2) << "Recycle merged chunk_iter for col_desc (tbl: "
              << col_desc.getScanDesc().getTableKey() << ", col: " << col_desc.getColId()
              << "), device_id: " << device_id;
      return &(dev_iter_map_it->second);
    }
  }
  return nullptr;
}

ChunkIter ColumnFetcher::prepareChunkIter(AbstractBuffer* merged_data_buf,
                                          AbstractBuffer* merged_index_buf,
                                          ChunkIter& chunk_iter,
                                          bool is_true_varlen_type,
                                          const size_t total_num_tuples) const {
  ChunkIter merged_chunk_iter;
  if (is_true_varlen_type) {
    merged_chunk_iter.start_pos = merged_index_buf->getMemoryPtr();
    merged_chunk_iter.current_pos = merged_index_buf->getMemoryPtr();
    merged_chunk_iter.end_pos = merged_index_buf->getMemoryPtr() +
                                merged_index_buf->size() - sizeof(StringOffsetT);
    merged_chunk_iter.second_buf = merged_data_buf->getMemoryPtr();
  } else {
    merged_chunk_iter.start_pos = merged_data_buf->getMemoryPtr();
    merged_chunk_iter.current_pos = merged_data_buf->getMemoryPtr();
    merged_chunk_iter.end_pos = merged_data_buf->getMemoryPtr() + merged_data_buf->size();
    merged_chunk_iter.second_buf = nullptr;
  }
  merged_chunk_iter.num_elems = total_num_tuples;
  merged_chunk_iter.skip = chunk_iter.skip;
  merged_chunk_iter.skip_size = chunk_iter.skip_size;
  merged_chunk_iter.type_info = chunk_iter.type_info;
  return merged_chunk_iter;
}

void ColumnFetcher::freeLinearizedBuf() {
  std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
  CHECK(executor_);
  const auto data_mgr = executor_->getDataMgr();

  if (!linearized_data_buf_cache_.empty()) {
    for (auto& kv : linearized_data_buf_cache_) {
      for (auto& kv2 : kv.second) {
        data_mgr->free(kv2.second);
      }
    }
  }

  if (!linearized_idx_buf_cache_.empty()) {
    for (auto& kv : linearized_idx_buf_cache_) {
      for (auto& kv2 : kv.second) {
        data_mgr->free(kv2.second);
      }
    }
  }
}

void ColumnFetcher::freeTemporaryCpuLinearizedIdxBuf() {
  std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
  CHECK(executor_);
  const auto data_mgr = executor_->getDataMgr();
  if (!linearlized_temporary_cpu_index_buf_cache_.empty()) {
    for (auto& kv : linearlized_temporary_cpu_index_buf_cache_) {
      data_mgr->free(kv.second);
    }
  }
}

const int8_t* ColumnFetcher::getResultSetColumn(
    const ResultSetPtr& buffer,
    const shared::TableKey& table_key,
    const int col_id,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx,
    const int frag_id) const {
  CHECK_GE(frag_id, -1);
  CHECK(buffer);
  CHECK_GE(col_id, 0);
  const ColumnarResults* result{nullptr};
  if (g_enable_result_reduction_pipeline && memory_level == Data_Namespace::GPU_LEVEL) {
    const ResultSetColumnCache::Key cache_key{
        buffer.get(), col_id, memory_level, device_id, frag_id, true, false, false};
    if (const auto cached_column = result_set_column_cache_->get(cache_key)) {
      return cached_column;
    }
    auto allocator_owner = makeResultSetColumnCacheAllocator(executor_, device_id);
    auto device_column = transferResultSetDeviceColumnIfAvailable(
        executor_, buffer, col_id, device_id, allocator_owner.get(), frag_id);
    if (device_column.buffer) {
      if (!device_column.owns_buffer) {
        return device_column.buffer;
      }
      return result_set_column_cache_->putOrGetExisting(
          cache_key, buffer, device_column.buffer, std::move(allocator_owner));
    }
    if (frag_id >= 0) {
      const auto column_fragment =
          transferResultSetColumnFragmentIfNeeded(
              buffer, col_id, memory_level, device_id, allocator_owner.get(), frag_id)
              .first;
      if (column_fragment) {
        return result_set_column_cache_->putOrGetExisting(
            cache_key, buffer, column_fragment, std::move(allocator_owner));
      }
    }
  } else if (g_enable_result_reduction_pipeline &&
             memory_level == Data_Namespace::CPU_LEVEL) {
    if (frag_id >= 0) {
      auto [column_fragment, num_bytes] = transferResultSetColumnFragmentIfNeeded(
          buffer, col_id, memory_level, device_id, device_allocator, frag_id);
      if (column_fragment) {
        return column_fragment;
      }
    }
    const ResultSetColumnCache::Key cache_key{
        buffer.get(), col_id, memory_level, 0, frag_id, true, false, false};
    auto row_set_mem_owner = executor_->getRowSetMemoryOwner();
    if (const auto host_column = result_set_column_cache_->getOrCreate(
            cache_key, buffer, row_set_mem_owner, [&, row_set_mem_owner] {
              auto transfer = transferResultSetDeviceColumnToHostIfAvailable(
                  buffer, col_id, frag_id, row_set_mem_owner, thread_idx);
              CHECK(!transfer.owns_buffer);
              return transfer.buffer;
            })) {
      return host_column;
    }
  }

  CHECK(frag_id == -1 || frag_id == 0);
  const bool selectively_columnarize =
      buffer->hasDeferredLazyFetchChunks() &&
      buffer->getQueryDescriptionType() == QueryDescriptionType::Projection &&
      buffer->didOutputColumnar() && buffer->isDirectColumnarConversionPossible();
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_fetch_mutex_);
    if (selectively_columnarize) {
      auto& columns = selectively_columnarized_result_cache_[buffer.get()];
      if (!columns.count(col_id)) {
        columns.emplace(col_id,
                        std::shared_ptr<const ColumnarResults>(
                            columnarize_result(executor_->row_set_mem_owner_,
                                               buffer,
                                               thread_idx,
                                               executor_->executor_id_,
                                               0,
                                               ColumnarResults::RowOrderMode::Preserve,
                                               static_cast<size_t>(col_id))));
      }
      result = columns.at(col_id).get();
    } else {
      if (columnarized_table_cache_.empty() ||
          !columnarized_table_cache_.count(table_key)) {
        columnarized_table_cache_.insert(std::make_pair(
            table_key,
            std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_result = columnarized_table_cache_[table_key];
      int frag_id = 0;
      auto cached_result_it = frag_id_to_result.find(frag_id);
      if (cached_result_it == frag_id_to_result.end() ||
          !has_column_buffer(cached_result_it->second, col_id)) {
        const auto selection_it = result_set_column_selections_.find(buffer.get());
        auto selected_column_indices = selection_it == result_set_column_selections_.end()
                                           ? std::vector<size_t>{}
                                           : selection_it->second;
        if (!selected_column_indices.empty() &&
            !std::binary_search(selected_column_indices.begin(),
                                selected_column_indices.end(),
                                static_cast<size_t>(col_id))) {
          selected_column_indices.clear();
        }
        if (const auto alias =
                selected_column_indices.empty()
                    ? find_columnarized_result_alias(columnarized_table_cache_,
                                                     executor_->temporary_tables_,
                                                     table_key,
                                                     buffer,
                                                     frag_id)
                    : nullptr;
            g_enable_result_reduction_pipeline && has_column_buffer(alias, col_id)) {
          frag_id_to_result[frag_id] = alias;
        } else {
          if (!selected_column_indices.empty()) {
            buffer->materializeDeferredLazyFetchColumnsForAllRows(
                selected_column_indices);
          }
          frag_id_to_result[frag_id] = std::shared_ptr<const ColumnarResults>(
              columnarize_result(executor_->row_set_mem_owner_,
                                 buffer,
                                 thread_idx,
                                 executor_->executor_id_,
                                 frag_id,
                                 ColumnarResults::RowOrderMode::Preserve,
                                 std::nullopt,
                                 selected_column_indices));
        }
      }
      CHECK_NE(size_t(0), columnarized_table_cache_.count(table_key));
      result = frag_id_to_result.at(frag_id).get();
    }
  }
  if (g_enable_result_reduction_pipeline && memory_level == Data_Namespace::GPU_LEVEL) {
    const ResultSetColumnCache::Key cache_key{
        buffer.get(), col_id, memory_level, device_id, frag_id, false, false, false};
    if (const auto cached_column = result_set_column_cache_->get(cache_key)) {
      return cached_column;
    }
    auto allocator_owner = makeResultSetColumnCacheAllocator(executor_, device_id);
    const auto gpu_column = transferColumnIfNeeded(result,
                                                   col_id,
                                                   executor_->getDataMgr(),
                                                   memory_level,
                                                   device_id,
                                                   allocator_owner.get());
    if (!gpu_column) {
      return nullptr;
    }
    return result_set_column_cache_->putOrGetExisting(
        cache_key, buffer, gpu_column, std::move(allocator_owner));
  }
  return transferColumnIfNeeded(
      result, col_id, executor_->getDataMgr(), memory_level, device_id, device_allocator);
}
