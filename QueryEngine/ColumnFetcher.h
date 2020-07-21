/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "DataMgr/Allocators/DeviceAllocator.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/HashJoinRuntime.h"

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

struct FetchResult {
  std::vector<std::vector<const int8_t*>> col_buffers;
  std::vector<std::vector<int64_t>> num_rows;
  std::vector<std::vector<uint64_t>> frag_offsets;
};

class ColumnFetcher {
 public:
  ColumnFetcher(Executor* executor, const ColumnCacheMap& column_cache);

  //! Gets one chunk's pointer and element count on either CPU or GPU.
  static std::pair<const int8_t*, size_t> getOneColumnFragment(
      Executor* executor,
      const Analyzer::ColumnVar& hash_col,
      const Fragmenter_Namespace::FragmentInfo& fragment,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      DeviceAllocator* device_allocator,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      ColumnCacheMap& column_cache);

  //! Creates a JoinColumn struct containing an array of JoinChunk structs.
  static JoinColumn makeJoinColumn(
      Executor* executor,
      const Analyzer::ColumnVar& hash_col,
      const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments,
      const Data_Namespace::MemoryLevel effective_mem_lvl,
      const int device_id,
      DeviceAllocator* device_allocator,
      std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      std::vector<std::shared_ptr<void>>& malloc_owner,
      ColumnCacheMap& column_cache);

  const int8_t* getOneTableColumnFragment(
      const int table_id,
      const int frag_id,
      const int col_id,
      const std::map<int, const TableFragments*>& all_tables_fragments,
      std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
      std::list<ChunkIter>& chunk_iter_holder,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator) const;

  const int8_t* getAllTableColumnFragments(
      const int table_id,
      const int col_id,
      const std::map<int, const TableFragments*>& all_tables_fragments,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator) const;

  const int8_t* getResultSetColumn(const InputColDescriptor* col_desc,
                                   const Data_Namespace::MemoryLevel memory_level,
                                   const int device_id,
                                   DeviceAllocator* device_allocator) const;

 private:
  static const int8_t* transferColumnIfNeeded(
      const ColumnarResults* columnar_results,
      const int col_id,
      Data_Namespace::DataMgr* data_mgr,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      DeviceAllocator* device_allocator);

  const int8_t* getResultSetColumn(const ResultSetPtr& buffer,
                                   const int table_id,
                                   const int col_id,
                                   const Data_Namespace::MemoryLevel memory_level,
                                   const int device_id,
                                   DeviceAllocator* device_allocator) const;

  Executor* executor_;
  using CacheKey = std::vector<int>;
  mutable std::mutex columnar_conversion_mutex_;
  mutable ColumnCacheMap columnarized_table_cache_;
  mutable std::unordered_map<
      InputColDescriptor,
      std::unordered_map<CacheKey, std::unique_ptr<const ColumnarResults>>>
      columnarized_ref_table_cache_;
  mutable std::unordered_map<InputColDescriptor, std::unique_ptr<const ColumnarResults>>
      columnarized_scan_table_cache_;

  friend class QueryCompilationDescriptor;
  friend class TableFunctionExecutionContext;  // TODO(adb)
};
