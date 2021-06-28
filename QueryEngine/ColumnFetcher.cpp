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

#include "QueryEngine/ColumnFetcher.h"

#include <memory>

#include "DataMgr/ArrayNoneEncoder.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "Shared/Intervals.h"
#include "Shared/likely.h"
#include "Shared/sqltypes.h"

namespace {

inline const ColumnarResults* columnarize_result(
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const ResultSetPtr& result,
    const size_t thread_idx,
    const int frag_id) {
  INJECT_TIMER(columnarize_result);
  CHECK_EQ(0, frag_id);

  std::vector<SQLTypeInfo> col_types;
  for (size_t i = 0; i < result->colCount(); ++i) {
    col_types.push_back(get_logical_type_info(result->getColType(i)));
  }
  return new ColumnarResults(
      row_set_mem_owner, *result, result->colCount(), col_types, thread_idx);
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
}  // namespace

ColumnFetcher::ColumnFetcher(Executor* executor, const ColumnCacheMap& column_cache)
    : executor_(executor), columnarized_table_cache_(column_cache) {}

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
    ColumnCacheMap& column_cache) {
  static std::mutex columnar_conversion_mutex;
  auto timer = DEBUG_TIMER(__func__);
  if (fragment.isEmptyPhysicalFragment()) {
    return {nullptr, 0};
  }
  const auto table_id = hash_col.get_table_id();
  const auto& catalog = *executor->getCatalog();
  const auto cd =
      get_column_descriptor_maybe(hash_col.get_column_id(), table_id, catalog);
  CHECK(!cd || !(cd->isVirtualCol));
  const int8_t* col_buff = nullptr;
  if (cd) {  // real table
    /* chunk_meta_it is used here to retrieve chunk numBytes and
       numElements. Apparently, their values are often zeros. If we
       knew how to predict the zero values, calling
       getChunkMetadataMap could be avoided to skip
       synthesize_metadata calls. */
    auto chunk_meta_it = fragment.getChunkMetadataMap().find(hash_col.get_column_id());
    CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
    ChunkKey chunk_key{catalog.getCurrentDB().dbId,
                       fragment.physicalTableId,
                       hash_col.get_column_id(),
                       fragment.fragmentId};
    const auto chunk = Chunk_NS::Chunk::getChunk(
        cd,
        &catalog.getDataMgr(),
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
      if (column_cache.empty() || !column_cache.count(table_id)) {
        column_cache.insert(std::make_pair(
            table_id, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_result = column_cache[table_id];
      if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
        frag_id_to_result.insert(
            std::make_pair(frag_id,
                           std::shared_ptr<const ColumnarResults>(columnarize_result(
                               executor->row_set_mem_owner_,
                               get_temporary_table(executor->temporary_tables_, table_id),
                               thread_idx,
                               frag_id))));
      }
      col_frag = column_cache[table_id][frag_id].get();
    }
    col_buff = transferColumnIfNeeded(
        col_frag,
        hash_col.get_column_id(),
        &catalog.getDataMgr(),
        effective_mem_lvl,
        effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        device_allocator);
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
    ColumnCacheMap& column_cache) {
  CHECK(!fragments.empty());

  size_t col_chunks_buff_sz = sizeof(struct JoinChunk) * fragments.size();
  // TODO: needs an allocator owner
  auto col_chunks_buff = reinterpret_cast<int8_t*>(
      malloc_owner.emplace_back(checked_malloc(col_chunks_buff_sz), free).get());
  auto join_chunk_array = reinterpret_cast<struct JoinChunk*>(col_chunks_buff);

  size_t num_elems = 0;
  size_t num_chunks = 0;
  for (auto& frag : fragments) {
    if (g_enable_non_kernel_time_query_interrupt && check_interrupt()) {
      throw QueryExecutionError(Executor::ERR_INTERRUPTED);
    }
    auto [col_buff, elem_count] = getOneColumnFragment(
        executor,
        hash_col,
        frag,
        effective_mem_lvl,
        effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        device_allocator,
        thread_idx,
        chunks_owner,
        column_cache);
    if (col_buff != nullptr) {
      num_elems += elem_count;
      join_chunk_array[num_chunks] = JoinChunk{col_buff, elem_count};
    } else {
      continue;
    }
    ++num_chunks;
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
    const int table_id,
    const int frag_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* allocator) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto& fragment = (*fragments)[frag_id];
  if (fragment.isEmptyPhysicalFragment()) {
    return nullptr;
  }
  std::shared_ptr<Chunk_NS::Chunk> chunk;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  CHECK(table_id > 0);
  const auto& cat = *executor_->getCatalog();
  auto cd = get_column_descriptor(col_id, table_id, cat);
  CHECK(cd);
  const auto col_type =
      get_column_type(col_id, table_id, cd, executor_->temporary_tables_);
  const bool is_real_string =
      col_type.is_string() && col_type.get_compression() == kENCODING_NONE;
  const bool is_varlen =
      is_real_string ||
      col_type.is_array();  // TODO: should it be col_type.is_varlen_array() ?
  {
    ChunkKey chunk_key{
        cat.getCurrentDB().dbId, fragment.physicalTableId, col_id, fragment.fragmentId};
    std::unique_ptr<std::lock_guard<std::mutex>> varlen_chunk_lock;
    if (is_varlen) {
      varlen_chunk_lock.reset(new std::lock_guard<std::mutex>(varlen_chunk_fetch_mutex_));
    }
    chunk = Chunk_NS::Chunk::getChunk(
        cd,
        &cat.getDataMgr(),
        chunk_key,
        memory_level,
        memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        chunk_meta_it->second->numBytes,
        chunk_meta_it->second->numElements);
    std::lock_guard<std::mutex> chunk_list_lock(chunk_list_mutex_);
    chunk_holder.push_back(chunk);
  }
  if (is_varlen) {
    CHECK_GT(table_id, 0);
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
      allocator->copyToDevice(
          chunk_iter_gpu, reinterpret_cast<int8_t*>(&chunk_iter), sizeof(ChunkIter));
      return chunk_iter_gpu;
    }
  } else {
    auto ab = chunk->getBuffer();
    CHECK(ab->getMemoryPtr());
    return ab->getMemoryPtr();  // @TODO(alex) change to use ChunkIter
  }
}

const int8_t* ColumnFetcher::getAllTableColumnFragments(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  std::vector<std::unique_ptr<ColumnarResults>> column_frags;
  const ColumnarResults* table_column = nullptr;
  const InputColDescriptor col_desc(col_id, table_id, int(0));
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_fetch_mutex_);
    auto column_it = columnarized_scan_table_cache_.find(col_desc);
    if (column_it == columnarized_scan_table_cache_.end()) {
      for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
        if (g_enable_non_kernel_time_query_interrupt && check_interrupt()) {
          throw QueryExecutionError(Executor::ERR_INTERRUPTED);
        }
        std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
        std::list<ChunkIter> chunk_iter_holder;
        const auto& fragment = (*fragments)[frag_id];
        if (fragment.isEmptyPhysicalFragment()) {
          continue;
        }
        auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
        CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
        auto col_buffer = getOneTableColumnFragment(table_id,
                                                    static_cast<int>(frag_id),
                                                    col_id,
                                                    all_tables_fragments,
                                                    chunk_holder,
                                                    chunk_iter_holder,
                                                    Data_Namespace::CPU_LEVEL,
                                                    int(0),
                                                    device_allocator);
        column_frags.push_back(
            std::make_unique<ColumnarResults>(executor_->row_set_mem_owner_,
                                              col_buffer,
                                              fragment.getNumTuples(),
                                              chunk_meta_it->second->sqlType,
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
                                               &executor_->getCatalog()->getDataMgr(),
                                               memory_level,
                                               device_id,
                                               device_allocator);
}

const int8_t* ColumnFetcher::getResultSetColumn(
    const InputColDescriptor* col_desc,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  CHECK(col_desc);
  const auto table_id = col_desc->getScanDesc().getTableId();
  return getResultSetColumn(get_temporary_table(executor_->temporary_tables_, table_id),
                            table_id,
                            col_desc->getColId(),
                            memory_level,
                            device_id,
                            device_allocator,
                            thread_idx);
}

const int8_t* ColumnFetcher::linearizeColumnFragments(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  auto timer = DEBUG_TIMER(__func__);
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  const InputColDescriptor col_desc(col_id, table_id, int(0));
  const auto& cat = *executor_->getCatalog();
  auto cd = get_column_descriptor(col_id, table_id, cat);
  CHECK(cd);
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  CHECK_GT(table_id, 0);
  bool is_varlen_chunk = cd->columnType.is_varlen() && !cd->columnType.is_fixlen_array();
  size_t total_num_tuples = 0;
  size_t total_data_buf_size = 0;
  size_t total_idx_buf_size = 0;
  {
    std::lock_guard<std::mutex> linearize_guard(linearized_col_cache_mutex_);
    auto linearized_iter_it = linearized_multi_frag_chunk_iter_cache_.find(col_desc);
    if (linearized_iter_it != linearized_multi_frag_chunk_iter_cache_.end()) {
      if (memory_level == CPU_LEVEL) {
        // in CPU execution, each kernel can share merged chunk since they operates in the
        // same memory space, so we can share the same chunk iter among kernels
        return getChunkiter(col_desc, 0);
      } else {
        // in GPU execution, this becomes the matter when we deploy multi-GPUs
        // so we only share the chunk_iter iff kernels are launched on the same GPU device
        // otherwise we need to separately load merged chunk and its iter
        // todo(yoonmin): D2D copy of merged chunk and its iter?
        if (linearized_iter_it->second.find(device_id) !=
            linearized_iter_it->second.end()) {
          // note that cached chunk_iter is located on CPU memory space...
          // we just need to copy it to each device
          // chunk_iter already contains correct buffer addr depending on execution device
          auto chunk_iter_gpu = device_allocator->alloc(sizeof(ChunkIter));
          device_allocator->copyToDevice(
              chunk_iter_gpu, getChunkiter(col_desc, device_id), sizeof(ChunkIter));
          return chunk_iter_gpu;
        }
      }
    }
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
          cat.getCurrentDB().dbId, fragment.physicalTableId, col_id, fragment.fragmentId};
      chunk = Chunk_NS::Chunk::getChunk(cd,
                                        &cat.getDataMgr(),
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
        res = linearizeFixedLenArrayColFrags(cat,
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
        res = linearizeVarLenArrayColFrags(cat,
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
      res = linearizeVarLenArrayColFrags(cat,
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
  auto merged_chunk =
      std::make_shared<Chunk_NS::Chunk>(merged_data_buffer, merged_index_buffer, cd);
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
    addMergedChunkIter(col_desc, 0, merged_chunk_iter_ptr);
    return merged_chunk_iter_ptr;
  } else {
    CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
    CHECK(device_allocator);
    addMergedChunkIter(col_desc, device_id, merged_chunk_iter_ptr);
    // note that merged_chunk_iter_ptr resides in CPU memory space
    // having its content aware GPU buffer that we alloc. for merging
    // so we need to copy this chunk_iter to each device explicitly
    auto chunk_iter_gpu = device_allocator->alloc(sizeof(ChunkIter));
    device_allocator->copyToDevice(
        chunk_iter_gpu, merged_chunk_iter_ptr, sizeof(ChunkIter));
    return chunk_iter_gpu;
  }
}

MergedChunk ColumnFetcher::linearizeVarLenArrayColFrags(
    const Catalog_Namespace::Catalog& cat,
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
  const InputColDescriptor icd(cd->columnId, cd->tableId, int(0));
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
      } else {
        merged_data_buffer =
            cat.getDataMgr().alloc(memory_level, device_id, total_data_buf_size);
        VLOG(2) << "Allocate " << total_data_buf_size
                << " bytes of buffer space for linearized chunks (memory_level: "
                << getMemoryLevelString(memory_level) << ", device_id: " << device_id
                << ")";
        cd_cache.insert(std::make_pair(device_id, merged_data_buffer));
      }
    } else {
      DeviceMergedChunkMap m;
      merged_data_buffer =
          cat.getDataMgr().alloc(memory_level, device_id, total_data_buf_size);
      VLOG(2) << "Allocate " << total_data_buf_size
              << " bytes of buffer space for linearized chunks (memory_level: "
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
    } else {
      merged_index_buffer_in_cpu = cat.getDataMgr().alloc(
          Data_Namespace::CPU_LEVEL, 0, total_idx_buf_size + sizeof(ArrayOffsetT));
      VLOG(2) << "Allocate " << total_data_buf_size
              << " bytes of temorary buffer space on CPU for linearized chunks";
      // just copy the buf addr since we access it via the pointer itself
      linearlized_temporary_cpu_index_buf_cache_.insert(
          std::make_pair(cd->columnId, merged_index_buffer_in_cpu));
    }
  }

  // linearize buffers if we don't have corresponding buf in cache
  size_t sum_data_buf_size = 0;
  size_t sum_idx_buf_size = 0;
  size_t cur_sum_num_tuples = 0;
  size_t total_idx_size_modifier = 0;
  auto chunk_holder_it = local_chunk_holder.begin();
  auto chunk_iter_holder_it = local_chunk_iter_holder.begin();
  auto chunk_num_tuple_it = local_chunk_num_tuples.begin();
  bool null_padded_first_elem = false;
  bool null_padded_last_val = false;
  for (; chunk_holder_it != local_chunk_holder.end();
       chunk_holder_it++, chunk_iter_holder_it++, chunk_num_tuple_it++) {
    if (g_enable_non_kernel_time_query_interrupt && check_interrupt()) {
      throw QueryExecutionError(Executor::ERR_INTERRUPTED);
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
    sum_idx_buf_size += idx_buf_size;
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
    if (cur_sum_num_tuples == total_num_tuples) {
      auto merged_index_buffer_ptr =
          reinterpret_cast<ArrayOffsetT*>(merged_index_buffer_in_cpu->getMemoryPtr());
      merged_index_buffer_ptr[total_num_tuples] =
          total_data_buf_size -
          total_idx_size_modifier;  // last index value is total data size;
    }
  }

  // put linearized index buffer to per-device cache
  std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
  AbstractBuffer* merged_index_buffer = nullptr;
  auto copyBuf =
      [&device_allocator](
          int8_t* src, int8_t* dest, size_t buf_size, MemoryLevel memory_level) {
        if (memory_level == Data_Namespace::CPU_LEVEL) {
          memcpy((void*)dest, src, buf_size);
        } else {
          CHECK(memory_level == Data_Namespace::GPU_LEVEL);
          device_allocator->copyToDevice(dest, src, buf_size);
        }
      };
  auto merged_idx_buf_cache_it = linearized_idx_buf_cache_.find(icd);
  size_t buf_size = total_idx_buf_size + sizeof(ArrayOffsetT);
  if (merged_idx_buf_cache_it != linearized_idx_buf_cache_.end()) {
    auto& merged_idx_buf_cache = merged_idx_buf_cache_it->second;
    auto merged_idx_buf_it = merged_idx_buf_cache.find(device_id);
    if (merged_idx_buf_it != merged_idx_buf_cache.end()) {
      merged_index_buffer = merged_idx_buf_it->second;
    } else {
      merged_index_buffer = cat.getDataMgr().alloc(memory_level, device_id, buf_size);
      copyBuf(merged_index_buffer_in_cpu->getMemoryPtr(),
              merged_index_buffer->getMemoryPtr(),
              buf_size,
              memory_level);
      merged_idx_buf_cache.insert(std::make_pair(device_id, merged_index_buffer));
    }
  } else {
    merged_index_buffer = cat.getDataMgr().alloc(memory_level, device_id, buf_size);
    copyBuf(merged_index_buffer_in_cpu->getMemoryPtr(),
            merged_index_buffer->getMemoryPtr(),
            buf_size,
            memory_level);
    DeviceMergedChunkMap m;
    m.insert(std::make_pair(device_id, merged_index_buffer));
    linearized_idx_buf_cache_.insert(std::make_pair(icd, m));
  }
  CHECK(merged_index_buffer);
  linearization_time_ms += timer_stop(clock_begin);
  VLOG(2) << "Linearization has been successfully done, elapsed time: "
          << linearization_time_ms << " ms.";
  return {merged_data_buffer, merged_index_buffer};
}

MergedChunk ColumnFetcher::linearizeFixedLenArrayColFrags(
    const Catalog_Namespace::Catalog& cat,
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
  auto merged_data_buffer =
      cat.getDataMgr().alloc(memory_level, device_id, total_data_buf_size);
  VLOG(2) << "Allocate " << total_data_buf_size
          << " bytes of buffer space for linearized chunks (memory_level: "
          << getMemoryLevelString(memory_level) << ", device_id: " << device_id << ")";
  size_t sum_data_buf_size = 0;
  auto chunk_holder_it = local_chunk_holder.begin();
  auto chunk_iter_holder_it = local_chunk_iter_holder.begin();
  for (; chunk_holder_it != local_chunk_holder.end();
       chunk_holder_it++, chunk_iter_holder_it++) {
    if (g_enable_non_kernel_time_query_interrupt && check_interrupt()) {
      throw QueryExecutionError(Executor::ERR_INTERRUPTED);
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
    const auto& col_ti = columnar_results->getColumnType(col_id);
    const auto num_bytes = columnar_results->size() * col_ti.get_size();
    CHECK(device_allocator);
    auto gpu_col_buffer = device_allocator->alloc(num_bytes);
    device_allocator->copyToDevice(gpu_col_buffer, col_buffers[col_id], num_bytes);
    return gpu_col_buffer;
  }
  return col_buffers[col_id];
}

void ColumnFetcher::addMergedChunkIter(const InputColDescriptor col_desc,
                                       const int device_id,
                                       int8_t* chunk_iter_ptr) const {
  std::lock_guard<std::mutex> linearize_guard(linearized_col_cache_mutex_);
  auto chunk_iter_it = linearized_multi_frag_chunk_iter_cache_.find(col_desc);
  if (chunk_iter_it != linearized_multi_frag_chunk_iter_cache_.end()) {
    auto iter_device_it = chunk_iter_it->second.find(device_id);
    if (iter_device_it == chunk_iter_it->second.end()) {
      VLOG(2) << "Additional merged chunk_iter for col_desc (tbl: "
              << col_desc.getScanDesc().getTableId() << ", col: " << col_desc.getColId()
              << "), device_id: " << device_id;
      chunk_iter_it->second.emplace(device_id, chunk_iter_ptr);
    }
  } else {
    DeviceMergedChunkIterMap iter_m;
    iter_m.emplace(device_id, chunk_iter_ptr);
    VLOG(2) << "New merged chunk_iter for col_desc (tbl: "
            << col_desc.getScanDesc().getTableId() << ", col: " << col_desc.getColId()
            << "), device_id: " << device_id;
    linearized_multi_frag_chunk_iter_cache_.emplace(col_desc, iter_m);
  }
}

const int8_t* ColumnFetcher::getChunkiter(const InputColDescriptor col_desc,
                                          const int device_id) const {
  auto linearized_chunk_iter_it = linearized_multi_frag_chunk_iter_cache_.find(col_desc);
  if (linearized_chunk_iter_it != linearized_multi_frag_chunk_iter_cache_.end()) {
    auto dev_iter_map_it = linearized_chunk_iter_it->second.find(device_id);
    if (dev_iter_map_it != linearized_chunk_iter_it->second.end()) {
      VLOG(2) << "Recycle merged chunk_iter for col_desc (tbl: "
              << col_desc.getScanDesc().getTableId() << ", col: " << col_desc.getColId()
              << "), device_id: " << device_id;
      return dev_iter_map_it->second;
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
  const auto& cat = *executor_->getCatalog();
  auto& data_mgr = cat.getDataMgr();

  if (!linearized_data_buf_cache_.empty()) {
    for (auto& kv : linearized_data_buf_cache_) {
      for (auto& kv2 : kv.second) {
        data_mgr.free(kv2.second);
      }
    }
  }

  if (!linearized_idx_buf_cache_.empty()) {
    for (auto& kv : linearized_idx_buf_cache_) {
      for (auto& kv2 : kv.second) {
        data_mgr.free(kv2.second);
      }
    }
  }
}

void ColumnFetcher::freeTemporaryCpuLinearizedIdxBuf() {
  std::lock_guard<std::mutex> linearized_col_cache_guard(linearized_col_cache_mutex_);
  const auto& cat = *executor_->getCatalog();
  auto& data_mgr = cat.getDataMgr();
  if (!linearlized_temporary_cpu_index_buf_cache_.empty()) {
    for (auto& kv : linearlized_temporary_cpu_index_buf_cache_) {
      data_mgr.free(kv.second);
    }
  }
}

const int8_t* ColumnFetcher::getResultSetColumn(
    const ResultSetPtr& buffer,
    const int table_id,
    const int col_id,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    DeviceAllocator* device_allocator,
    const size_t thread_idx) const {
  const ColumnarResults* result{nullptr};
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_fetch_mutex_);
    if (columnarized_table_cache_.empty() || !columnarized_table_cache_.count(table_id)) {
      columnarized_table_cache_.insert(std::make_pair(
          table_id, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_result = columnarized_table_cache_[table_id];
    int frag_id = 0;
    if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
      frag_id_to_result.insert(std::make_pair(
          frag_id,
          std::shared_ptr<const ColumnarResults>(columnarize_result(
              executor_->row_set_mem_owner_, buffer, thread_idx, frag_id))));
    }
    CHECK_NE(size_t(0), columnarized_table_cache_.count(table_id));
    result = columnarized_table_cache_[table_id][frag_id].get();
  }
  CHECK_GE(col_id, 0);
  return transferColumnIfNeeded(result,
                                col_id,
                                &executor_->getCatalog()->getDataMgr(),
                                memory_level,
                                device_id,
                                device_allocator);
}
