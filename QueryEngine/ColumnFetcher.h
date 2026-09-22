/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/Allocators/DeviceAllocator.h"
#include "QueryEngine/ColumnBufferLayout.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/JoinHashTable/Runtime/HashJoinRuntime.h"

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace std {
template <>
struct hash<std::vector<int>> {
  size_t operator()(const std::vector<int>& vec) const {
    return vec.size() ^ boost::hash_range(vec.begin(), vec.end());
  }
};

template <>
struct hash<std::pair<int, int>> {
  size_t operator()(const std::pair<int, int>& p) const {
    return boost::hash<std::pair<int, int>>()(p);
  }
};

}  // namespace std

struct FetchResultFragmentInfo {
  std::vector<std::vector<int64_t>> num_rows;
  std::vector<std::vector<uint64_t>> frag_offsets;
  std::vector<std::vector<int32_t>> frag_ids;
};
struct FetchResult {
  std::vector<std::vector<const int8_t*>> col_buffers;
  ColumnBufferLayouts col_buffer_layouts;
  std::vector<std::vector<const int64_t*>> selected_rowids;
  FetchResultFragmentInfo fragment_info;
  std::vector<std::shared_ptr<void>> owners;
  DeferredLazyFetchChunks deferred_lazy_fetch_chunks;
  LazyFetchSourceMetadata lazy_fetch_source_metadata;
};

using MergedChunk = std::pair<AbstractBuffer*, AbstractBuffer*>;

class ResultSetColumnCache {
 public:
  struct Key {
    const ResultSet* result_set;
    int col_id;
    Data_Namespace::MemoryLevel memory_level;
    int device_id;
    int frag_id;
    bool columnar_fragment;
    bool segmented;
    bool direct_peer_access;

    bool operator==(const Key& other) const {
      return result_set == other.result_set && col_id == other.col_id &&
             memory_level == other.memory_level && device_id == other.device_id &&
             frag_id == other.frag_id && columnar_fragment == other.columnar_fragment &&
             segmented == other.segmented &&
             direct_peer_access == other.direct_peer_access;
    }
  };

  const int8_t* get(const Key& key) const;
  const int8_t* putOrGetExisting(const Key& key,
                                 const ResultSetPtr& result_owner,
                                 const int8_t* buffer,
                                 std::shared_ptr<void> owner);
  const int8_t* getOrCreate(const Key& key,
                            const ResultSetPtr& result_owner,
                            std::shared_ptr<void> owner,
                            const std::function<const int8_t*()>& create);

  struct JoinColumnKey {
    const ResultSet* result_set;
    int col_id;
    Data_Namespace::MemoryLevel memory_level;
    int device_id;

    bool operator==(const JoinColumnKey& other) const {
      return result_set == other.result_set && col_id == other.col_id &&
             memory_level == other.memory_level && device_id == other.device_id;
    }
  };

  std::optional<JoinColumn> getJoinColumn(const JoinColumnKey& key) const;
  JoinColumn putOrGetExistingJoinColumn(const JoinColumnKey& key,
                                        const ResultSetPtr& result_owner,
                                        JoinColumn join_column,
                                        std::shared_ptr<void> chunks_owner,
                                        std::vector<std::shared_ptr<void>> owners);
  void mergeColumnSelection(const ResultSetPtr& result_set,
                            const std::vector<size_t>& selected_column_indices);
  std::optional<std::vector<size_t>> getColumnSelection(
      const ResultSet* result_set) const;
  void clear();

 private:
  struct KeyHash {
    size_t operator()(const Key& key) const {
      return std::hash<const ResultSet*>{}(key.result_set) ^
             (static_cast<size_t>(key.col_id) << 1) ^
             (static_cast<size_t>(key.memory_level) << 6) ^
             (static_cast<size_t>(key.device_id) << 8) ^
             (static_cast<size_t>(key.frag_id) << 16) ^
             (static_cast<size_t>(key.columnar_fragment) << 31) ^
             (static_cast<size_t>(key.segmented) << 32) ^
             (static_cast<size_t>(key.direct_peer_access) << 33);
    }
  };

  struct Entry {
    const int8_t* buffer;
    std::shared_ptr<void> owner;
    ResultSetPtr result_owner;
  };

  struct JoinColumnKeyHash {
    size_t operator()(const JoinColumnKey& key) const {
      return std::hash<const ResultSet*>{}(key.result_set) ^
             (static_cast<size_t>(key.col_id) << 1) ^
             (static_cast<size_t>(key.memory_level) << 6) ^
             (static_cast<size_t>(key.device_id) << 8);
    }
  };

  struct JoinColumnEntry {
    JoinColumn join_column;
    std::shared_ptr<void> chunks_owner;
    std::vector<std::shared_ptr<void>> owners;
    ResultSetPtr result_owner;
  };

  struct ColumnSelectionEntry {
    ResultSetPtr result_owner;
    std::vector<size_t> selected_column_indices;
    bool all_columns{false};
  };

  mutable std::mutex mutex_;
  std::unordered_map<Key, Entry, KeyHash> entries_;
  std::unordered_map<Key, std::shared_ptr<std::mutex>, KeyHash> entry_creation_mutexes_;
  std::unordered_map<JoinColumnKey, JoinColumnEntry, JoinColumnKeyHash>
      join_column_entries_;
  std::unordered_map<const ResultSet*, ColumnSelectionEntry> column_selections_;
};

class ColumnFetcher {
 public:
  ColumnFetcher(Executor* executor,
                ColumnCacheMap& column_cache,
                ResultSetColumnCache* result_set_column_cache = nullptr);

  ResultSetColumnCache* getResultSetColumnCache() const {
    return result_set_column_cache_;
  }

  //! Gets one chunk's pointer and element count on either CPU or GPU.
  static std::pair<const int8_t*, size_t> getOneColumnFragment(
      Executor* executor,
      const Analyzer::ColumnVar& hash_col,
      const Fragmenter_Namespace::FragmentInfo& fragment,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      DeviceAllocator* device_allocator,
      const size_t thread_idx,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      ColumnCacheMap& column_cache,
      const ColumnarResults::RowOrderMode row_order_mode =
          ColumnarResults::RowOrderMode::Preserve);

  //! Creates a JoinColumn struct containing an array of JoinChunk structs.
  static JoinColumn makeJoinColumn(
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
      bool preserve_result_set_fragment_offsets = false,
      const std::map<std::pair<int, int>, size_t>* physical_fragment_rowid_offsets =
          nullptr);

  const int8_t* getOneTableColumnFragment(
      const shared::TableKey& table_key,
      const int frag_id,
      const int col_id,
      const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator) const;

  const int8_t* getAllTableColumnFragments(
      const shared::TableKey& table_key,
      const int col_id,
      const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator,
      const size_t thread_idx) const;

  const int8_t* getTableColumnFragmentsSegmented(
      const shared::TableKey& table_key,
      const int col_id,
      const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
      const std::vector<size_t>& fragment_ids,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator) const;

  const int8_t* getResultSetColumn(const InputColDescriptor* col_desc,
                                   const Data_Namespace::MemoryLevel memory_level,
                                   const int device_id,
                                   DeviceAllocator* device_allocator,
                                   const size_t thread_idx,
                                   const int frag_id) const;

  void setResultSetColumnSelection(
      const ResultSetPtr& result_set,
      const std::vector<size_t>& selected_column_indices) const;
  void setResultSetColumnSelections(
      const std::list<std::shared_ptr<const InputColDescriptor>>& input_col_descs) const;

  const int8_t* getResultSetColumnSegmented(
      const InputColDescriptor* col_desc,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator,
      const size_t thread_idx,
      const int frag_id,
      const bool allow_direct_peer_access = false) const;

  const int8_t* linearizeColumnFragments(
      const shared::TableKey& table_key,
      const int col_id,
      const std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator,
      const size_t thread_idx) const;

  void freeTemporaryCpuLinearizedIdxBuf();
  void freeLinearizedBuf();

 private:
  static const int8_t* transferColumnIfNeeded(
      const ColumnarResults* columnar_results,
      const int col_id,
      Data_Namespace::DataMgr* data_mgr,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator);

  MergedChunk linearizeVarLenArrayColFrags(
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
      const size_t thread_idx) const;

  MergedChunk linearizeFixedLenArrayColFrags(
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
      const size_t thread_idx) const;

  void addMergedChunkIter(const InputColDescriptor col_desc,
                          const int device_id,
                          const ChunkIter& chunk_iter) const;

  const ChunkIter* getChunkiter(const InputColDescriptor col_desc,
                                const int device_id = 0) const;

  ChunkIter prepareChunkIter(AbstractBuffer* merged_data_buf,
                             AbstractBuffer* merged_index_buf,
                             ChunkIter& chunk_iter,
                             bool is_true_varlen_type,
                             const size_t total_num_tuples) const;

  const int8_t* getResultSetColumn(const ResultSetPtr& buffer,
                                   const shared::TableKey& table_key,
                                   const int col_id,
                                   const Data_Namespace::MemoryLevel memory_level,
                                   const int device_id,
                                   DeviceAllocator* device_allocator,
                                   const size_t thread_idx,
                                   const int frag_id) const;

  Executor* executor_;

  struct SegmentedTableColumnCacheKey {
    shared::ColumnKey column_key;
    int device_id;
    std::vector<size_t> fragment_ids;

    bool operator==(const SegmentedTableColumnCacheKey& other) const {
      return column_key == other.column_key && device_id == other.device_id &&
             fragment_ids == other.fragment_ids;
    }
  };

  struct SegmentedTableColumnCacheKeyHash {
    size_t operator()(const SegmentedTableColumnCacheKey& key) const {
      size_t seed = std::hash<shared::ColumnKey>{}(key.column_key);
      const auto combine = [&seed](const size_t value) {
        seed ^= value + size_t{0x9e3779b9} + (seed << 6) + (seed >> 2);
      };
      combine(std::hash<int>{}(key.device_id));
      for (const auto fragment_id : key.fragment_ids) {
        combine(std::hash<size_t>{}(fragment_id));
      }
      return seed;
    }
  };

  struct SegmentedTableColumnCacheEntry {
    const int8_t* descriptor;
    std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  };

  mutable std::mutex columnar_fetch_mutex_;
  mutable std::mutex varlen_chunk_fetch_mutex_;
  mutable std::mutex linearization_mutex_;
  mutable std::mutex chunk_list_mutex_;
  mutable std::mutex linearized_col_cache_mutex_;
  mutable std::mutex segmented_table_column_cache_mutex_;
  mutable std::unordered_map<SegmentedTableColumnCacheKey,
                             SegmentedTableColumnCacheEntry,
                             SegmentedTableColumnCacheKeyHash>
      segmented_table_column_cache_;
  ColumnCacheMap& columnarized_table_cache_;
  mutable std::unordered_map<
      const ResultSet*,
      std::unordered_map<int, std::shared_ptr<const ColumnarResults>>>
      selectively_columnarized_result_cache_;
  mutable std::unordered_map<const ResultSet*, std::vector<size_t>>
      result_set_column_selections_;
  ResultSetColumnCache local_result_set_column_cache_;
  ResultSetColumnCache* result_set_column_cache_;
  mutable std::unordered_map<InputColDescriptor, std::unique_ptr<const ColumnarResults>>
      columnarized_scan_table_cache_;
  using DeviceMergedChunkIterMap = std::unordered_map<int, ChunkIter>;
  using DeviceMergedChunkMap = std::unordered_map<int, AbstractBuffer*>;
  mutable std::unordered_map<InputColDescriptor, DeviceMergedChunkIterMap>
      linearized_multi_frag_chunk_iter_cache_;
  mutable std::unordered_map<int, AbstractBuffer*>
      linearlized_temporary_cpu_index_buf_cache_;
  mutable std::unordered_map<InputColDescriptor, DeviceMergedChunkMap>
      linearized_data_buf_cache_;
  mutable std::unordered_map<InputColDescriptor, DeviceMergedChunkMap>
      linearized_idx_buf_cache_;
  friend class QueryCompilationDescriptor;
  friend class TableFunctionExecutionContext;  // TODO(adb)
};
