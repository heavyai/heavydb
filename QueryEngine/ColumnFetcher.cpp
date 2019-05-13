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

#include "ColumnFetcher.h"
#include "Execute.h"

ColumnFetcher::ColumnFetcher(Executor* executor, const ColumnCacheMap& column_cache)
    : executor_(executor), columnarized_table_cache_(column_cache) {}

std::pair<const int8_t*, size_t> ColumnFetcher::getOneColumnFragment(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ColumnCacheMap& column_cache) {
  static std::mutex columnar_conversion_mutex;
  if (fragment.isEmptyPhysicalFragment()) {
    return {nullptr, 0};
  }
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(hash_col.get_column_id());
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  const auto& catalog = *executor->getCatalog();
  const auto cd = get_column_descriptor_maybe(
      hash_col.get_column_id(), hash_col.get_table_id(), catalog);
  CHECK(!cd || !(cd->isVirtualCol));
  const int8_t* col_buff = nullptr;
  if (cd) {
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
        chunk_meta_it->second.numBytes,
        chunk_meta_it->second.numElements);
    chunks_owner.push_back(chunk);
    CHECK(chunk);
    auto ab = chunk->get_buffer();
    CHECK(ab->getMemoryPtr());
    col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  } else {
    const ColumnarResults* col_frag{nullptr};
    {
      std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex);
      const auto table_id = hash_col.get_table_id();
      const auto frag_id = fragment.fragmentId;
      if (column_cache.empty() || !column_cache.count(table_id)) {
        column_cache.insert(std::make_pair(
            table_id, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_result = column_cache[table_id];
      if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
        frag_id_to_result.insert(std::make_pair(
            frag_id,
            std::shared_ptr<const ColumnarResults>(columnarize_result(
                executor->row_set_mem_owner_,
                get_temporary_table(executor->temporary_tables_, hash_col.get_table_id()),
                frag_id))));
      }
      col_frag = column_cache[table_id][frag_id].get();
    }
    col_buff = transferColumnIfNeeded(
        col_frag,
        hash_col.get_column_id(),
        &catalog.getDataMgr(),
        effective_mem_lvl,
        effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id);
  }
  return {col_buff, fragment.getNumTuples()};
}

std::pair<const int8_t*, size_t> ColumnFetcher::getAllColumnFragments(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ColumnCacheMap& column_cache) {
  CHECK(!fragments.empty());
  const size_t elem_width = hash_col.get_type_info().get_size();
  std::vector<const int8_t*> col_frags;
  std::vector<size_t> elem_counts;
  for (auto& frag : fragments) {
    const int8_t* col_frag = nullptr;
    size_t elem_count = 0;
    std::tie(col_frag, elem_count) = getOneColumnFragment(executor,
                                                          hash_col,
                                                          frag,
                                                          Data_Namespace::CPU_LEVEL,
                                                          0,
                                                          chunks_owner,
                                                          column_cache);
    if (col_frag == nullptr) {
      continue;
    }
    CHECK_NE(elem_count, size_t(0));
    col_frags.push_back(col_frag);
    elem_counts.push_back(elem_count);
  }
  CHECK(!col_frags.empty());
  CHECK_EQ(col_frags.size(), elem_counts.size());
  const auto total_elem_count =
      std::accumulate(elem_counts.begin(), elem_counts.end(), size_t(0));
  auto col_buff =
      reinterpret_cast<int8_t*>(checked_malloc(total_elem_count * elem_width));
  for (size_t i = 0, offset = 0; i < col_frags.size(); ++i) {
    memcpy(col_buff + offset, col_frags[i], elem_counts[i] * elem_width);
    offset += elem_counts[i] * elem_width;
  }
  return {col_buff, total_elem_count};
}

const int8_t* ColumnFetcher::getOneTableColumnFragment(
    const int table_id,
    const int frag_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  static std::mutex varlen_chunk_mutex;  // TODO(alex): remove
  static std::mutex chunk_list_mutex;
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
      varlen_chunk_lock.reset(new std::lock_guard<std::mutex>(varlen_chunk_mutex));
    }
    chunk = Chunk_NS::Chunk::getChunk(
        cd,
        &cat.getDataMgr(),
        chunk_key,
        memory_level,
        memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        chunk_meta_it->second.numBytes,
        chunk_meta_it->second.numElements);
    std::lock_guard<std::mutex> chunk_list_lock(chunk_list_mutex);
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
      CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
      auto& data_mgr = cat.getDataMgr();
      auto chunk_iter_gpu =
          CudaAllocator::alloc(&data_mgr, sizeof(ChunkIter), device_id, nullptr);
      copy_to_gpu(&data_mgr, chunk_iter_gpu, &chunk_iter, sizeof(ChunkIter), device_id);
      return reinterpret_cast<int8_t*>(chunk_iter_gpu);
    }
  } else {
    auto ab = chunk->get_buffer();
    CHECK(ab->getMemoryPtr());
    return ab->getMemoryPtr();  // @TODO(alex) change to use ChunkIter
  }
}

const int8_t* ColumnFetcher::getAllTableColumnFragments(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  std::vector<std::unique_ptr<ColumnarResults>> column_frags;
  const ColumnarResults* table_column = nullptr;
  const InputColDescriptor col_desc(col_id, table_id, int(0));
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    auto column_it = columnarized_scan_table_cache_.find(col_desc);
    if (column_it == columnarized_scan_table_cache_.end()) {
      for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
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
                                                    int(0));
        column_frags.push_back(
            boost::make_unique<ColumnarResults>(executor_->row_set_mem_owner_,
                                                col_buffer,
                                                fragment.getNumTuples(),
                                                chunk_meta_it->second.sqlType));
      }
      auto merged_results =
          ColumnarResults::mergeResults(executor_->row_set_mem_owner_, column_frags);
      table_column = merged_results.get();
      columnarized_scan_table_cache_.emplace(col_desc, std::move(merged_results));
    } else {
      table_column = column_it->second.get();
    }
  }
  return ColumnFetcher::transferColumnIfNeeded(
      table_column, 0, &executor_->getCatalog()->getDataMgr(), memory_level, device_id);
}

const int8_t* ColumnFetcher::getResultSetColumn(
    const InputColDescriptor* col_desc,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  CHECK(col_desc);
  const auto table_id = col_desc->getScanDesc().getTableId();
  return getResultSetColumn(get_temporary_table(executor_->temporary_tables_, table_id),
                            table_id,
                            col_desc->getColId(),
                            memory_level,
                            device_id);
}

const int8_t* ColumnFetcher::transferColumnIfNeeded(
    const ColumnarResults* columnar_results,
    const int col_id,
    Data_Namespace::DataMgr* data_mgr,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) {
  const auto& col_buffers = columnar_results->getColumnBuffers();
  CHECK_LT(static_cast<size_t>(col_id), col_buffers.size());
  if (memory_level == Data_Namespace::GPU_LEVEL) {
    const auto& col_ti = columnar_results->getColumnType(col_id);
    const auto num_bytes = columnar_results->size() * col_ti.get_size();
    auto gpu_col_buffer = CudaAllocator::alloc(data_mgr, num_bytes, device_id, nullptr);
    copy_to_gpu(data_mgr, gpu_col_buffer, col_buffers[col_id], num_bytes, device_id);
    return reinterpret_cast<const int8_t*>(gpu_col_buffer);
  }
  return col_buffers[col_id];
}

const int8_t* ColumnFetcher::getResultSetColumn(
    const ResultSetPtr& buffer,
    const int table_id,
    const int col_id,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  const ColumnarResults* result{nullptr};
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    if (columnarized_table_cache_.empty() || !columnarized_table_cache_.count(table_id)) {
      columnarized_table_cache_.insert(std::make_pair(
          table_id, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_result = columnarized_table_cache_[table_id];
    int frag_id = 0;
    if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
      frag_id_to_result.insert(
          std::make_pair(frag_id,
                         std::shared_ptr<const ColumnarResults>(columnarize_result(
                             executor_->row_set_mem_owner_, buffer, frag_id))));
    }
    CHECK_NE(size_t(0), columnarized_table_cache_.count(table_id));
    result = columnarized_table_cache_[table_id][frag_id].get();
  }
  CHECK_GE(col_id, 0);
  return transferColumnIfNeeded(
      result, col_id, &executor_->getCatalog()->getDataMgr(), memory_level, device_id);
}
