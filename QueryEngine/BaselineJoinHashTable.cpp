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

#include "BaselineJoinHashTable.h"
#include "Execute.h"
#include "ExpressionRewrite.h"
#include "HashJoinKeyHandlers.h"
#include "JoinHashTableGpuUtils.h"

#include "Shared/unreachable.h"

#include <future>

std::vector<std::pair<BaselineJoinHashTable::HashTableCacheKey,
                      BaselineJoinHashTable::HashTableCacheValue>>
    BaselineJoinHashTable::hash_table_cache_;
std::mutex BaselineJoinHashTable::hash_table_cache_mutex_;

std::shared_ptr<BaselineJoinHashTable> BaselineJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    ColumnCacheMap& column_cache,
    Executor* executor) {
  const auto& query_info =
      get_inner_query_info(getInnerTableId(condition.get(), executor), query_infos).info;
  const auto total_entries = 2 * query_info.getNumTuplesUpperBound();
  if (total_entries > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw TooManyHashEntries();
  }
  const auto shard_count = memory_level == Data_Namespace::GPU_LEVEL
                               ? BaselineJoinHashTable::getShardCountForCondition(
                                     condition.get(), ra_exe_unit, executor)
                               : 0;
  const auto entries_per_device =
      get_entries_per_device(total_entries, shard_count, device_count, memory_level);
  auto join_hash_table =
      std::shared_ptr<BaselineJoinHashTable>(new BaselineJoinHashTable(condition,
                                                                       query_infos,
                                                                       ra_exe_unit,
                                                                       memory_level,
                                                                       entries_per_device,
                                                                       column_cache,
                                                                       executor));
  join_hash_table->checkHashJoinReplicationConstraint(
      getInnerTableId(condition.get(), executor));
  try {
    join_hash_table->reify(device_count);
  } catch (const TableMustBeReplicated& e) {
    // Throw a runtime error to abort the query
    join_hash_table->freeHashBufferMemory();
    throw std::runtime_error(e.what());
  } catch (const HashJoinFail& e) {
    // HashJoinFail exceptions log an error and trigger a retry with a join loop (if
    // possible)
    join_hash_table->freeHashBufferMemory();
    throw HashJoinFail(std::string("Could not build a 1-to-1 correspondence for columns "
                                   "involved in equijoin | ") +
                       e.what());
  } catch (const ColumnarConversionNotSupported& e) {
    throw HashJoinFail(std::string("Could not build hash tables for equijoin | ") +
                       e.what());
  } catch (const std::exception& e) {
    LOG(FATAL) << "Fatal error while attempting to build hash tables for join: "
               << e.what();
  }
  return join_hash_table;
}

BaselineJoinHashTable::BaselineJoinHashTable(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Data_Namespace::MemoryLevel memory_level,
    const size_t entry_count,
    ColumnCacheMap& column_cache,
    Executor* executor)
    : condition_(condition)
    , query_infos_(query_infos)
    , memory_level_(memory_level)
    , entry_count_(entry_count)
    , emitted_keys_count_(0)
    , executor_(executor)
    , ra_exe_unit_(ra_exe_unit)
    , column_cache_(column_cache)
    , layout_(JoinHashTableInterface::HashType::OneToOne) {}

size_t BaselineJoinHashTable::getShardCountForCondition(
    const Analyzer::BinOper* condition,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Executor* executor) {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition, *executor->getCatalog(), executor->getTemporaryTables());
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto pair_shard_count =
        get_shard_count(inner_outer_pair, ra_exe_unit, executor);
    if (pair_shard_count) {
      return pair_shard_count;
    }
  }
  return 0;
}

int64_t BaselineJoinHashTable::getJoinHashBuffer(const ExecutorDeviceType device_type,
                                                 const int device_id) noexcept {
  if (device_type == ExecutorDeviceType::CPU && !cpu_hash_table_buff_) {
    return 0;
  }
#ifdef HAVE_CUDA
  CHECK_LT(static_cast<size_t>(device_id), gpu_hash_table_buff_.size());
  return device_type == ExecutorDeviceType::CPU
             ? reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0])
             : reinterpret_cast<int64_t>(gpu_hash_table_buff_[device_id]->getMemoryPtr());
#else
  CHECK(device_type == ExecutorDeviceType::CPU);
  return reinterpret_cast<int64_t>(&(*cpu_hash_table_buff_)[0]);
#endif
}

BaselineJoinHashTable::CompositeKeyInfo BaselineJoinHashTable::getCompositeKeyInfo(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  std::vector<const void*> sd_inner_proxy_per_key;
  std::vector<const void*> sd_outer_proxy_per_key;
  std::vector<ChunkKey> cache_key_chunks;  // used for the cache key
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto outer_col = inner_outer_pair.second;
    const auto& inner_ti = inner_col->get_type_info();
    const auto& outer_ti = outer_col->get_type_info();
    ChunkKey cache_key_chunks_for_column{executor_->getCatalog()->getCurrentDB().dbId,
                                         inner_col->get_table_id(),
                                         inner_col->get_column_id()};
    if (inner_ti.is_string()) {
      CHECK(outer_ti.is_string());
      CHECK(inner_ti.get_compression() == kENCODING_DICT &&
            outer_ti.get_compression() == kENCODING_DICT);
      const auto sd_inner_proxy = executor_->getStringDictionaryProxy(
          inner_ti.get_comp_param(), executor_->getRowSetMemoryOwner(), true);
      const auto sd_outer_proxy = executor_->getStringDictionaryProxy(
          outer_ti.get_comp_param(), executor_->getRowSetMemoryOwner(), true);
      CHECK(sd_inner_proxy && sd_outer_proxy);
      sd_inner_proxy_per_key.push_back(sd_inner_proxy);
      sd_outer_proxy_per_key.push_back(sd_outer_proxy);
      cache_key_chunks_for_column.push_back(sd_outer_proxy->getGeneration());
    } else {
      sd_inner_proxy_per_key.emplace_back();
      sd_outer_proxy_per_key.emplace_back();
    }
    cache_key_chunks.push_back(cache_key_chunks_for_column);
  }
  return {sd_inner_proxy_per_key, sd_outer_proxy_per_key, cache_key_chunks};
}

void BaselineJoinHashTable::reify(const int device_count) {
  CHECK_LT(0, device_count);
#ifdef HAVE_CUDA
  gpu_hash_table_buff_.resize(device_count);
#endif  // HAVE_CUDA
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto composite_key_info = getCompositeKeyInfo(inner_outer_pairs);
  const auto type_and_found = HashTypeCache::get(composite_key_info.cache_key_chunks);
  const auto layout = type_and_found.second ? type_and_found.first
                                            : JoinHashTableInterface::HashType::OneToOne;

  if (condition_->is_overlaps_oper()) {
    try {
      reifyWithLayout(device_count, JoinHashTableInterface::HashType::OneToMany);
      return;
    } catch (const std::exception& e) {
      VLOG(1) << "Caught exception while building overlaps baseline hash table: "
              << e.what();
      throw;
    }
  }

  try {
    reifyWithLayout(device_count, layout);
  } catch (const std::exception& e) {
    VLOG(1) << "Caught exception while building baseline hash table: " << e.what();
    freeHashBufferMemory();
    HashTypeCache::set(composite_key_info.cache_key_chunks,
                       JoinHashTableInterface::HashType::OneToMany);
    reifyWithLayout(device_count, JoinHashTableInterface::HashType::OneToMany);
  }
}

void BaselineJoinHashTable::reifyWithLayout(
    const int device_count,
    const JoinHashTableInterface::HashType layout) {
  layout_ = layout;
  const auto& query_info = get_inner_query_info(getInnerTableId(), query_infos_).info;
  if (query_info.fragments.empty()) {
    return;
  }
  std::vector<BaselineJoinHashTable::ColumnsForDevice> columns_per_device;
  const auto shard_count = shardCount();
  for (int device_id = 0; device_id < device_count; ++device_id) {
    const auto fragments =
        shard_count
            ? only_shards_for_device(query_info.fragments, device_id, device_count)
            : query_info.fragments;
    const auto columns_for_device = fetchColumnsForDevice(fragments, device_id);
    columns_per_device.push_back(columns_for_device);
  }
  if (layout == JoinHashTableInterface::HashType::OneToMany) {
    CHECK(!columns_per_device.front().join_columns.empty());
    emitted_keys_count_ = columns_per_device.front().join_columns.front().num_elems;
    size_t tuple_count;
    std::tie(tuple_count, std::ignore) = approximateTupleCount(columns_per_device);
    const auto entry_count = 2 * std::max(tuple_count, size_t(1));

    entry_count_ =
        get_entries_per_device(entry_count, shard_count, device_count, memory_level_);
  }
  std::vector<std::future<void>> init_threads;
  for (int device_id = 0; device_id < device_count; ++device_id) {
    const auto fragments =
        shard_count
            ? only_shards_for_device(query_info.fragments, device_id, device_count)
            : query_info.fragments;
    init_threads.push_back(std::async(std::launch::async,
                                      &BaselineJoinHashTable::reifyForDevice,
                                      this,
                                      columns_per_device[device_id],
                                      layout,
                                      device_id));
  }
  for (auto& init_thread : init_threads) {
    init_thread.wait();
  }
  for (auto& init_thread : init_threads) {
    init_thread.get();
  }
}

std::pair<size_t, size_t> BaselineJoinHashTable::approximateTupleCount(
    const std::vector<ColumnsForDevice>& columns_per_device) const {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs);
  CountDistinctDescriptor count_distinct_desc{
      CountDistinctImplType::Bitmap,
      0,
      11,
      true,
      effective_memory_level == Data_Namespace::MemoryLevel::GPU_LEVEL
          ? ExecutorDeviceType::GPU
          : ExecutorDeviceType::CPU,
      1};
  const auto padded_size_bytes = count_distinct_desc.bitmapPaddedSizeBytes();

  CHECK(!columns_per_device.empty() && !columns_per_device.front().join_columns.empty());

  if (effective_memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    const auto composite_key_info = getCompositeKeyInfo(inner_outer_pairs);
    HashTableCacheKey cache_key{columns_per_device.front().join_columns.front().num_elems,
                                composite_key_info.cache_key_chunks,
                                condition_->get_optype()};
    const auto cached_count_info = getApproximateTupleCountFromCache(cache_key);
    if (cached_count_info.first >= 0) {
      return std::make_pair(cached_count_info.first, cached_count_info.second);
    }
    int thread_count = cpu_threads();
    std::vector<uint8_t> hll_buffer_all_cpus(thread_count * padded_size_bytes);
    auto hll_result = &hll_buffer_all_cpus[0];

    approximate_distinct_tuples(hll_result,
                                count_distinct_desc.bitmap_sz_bits,
                                padded_size_bytes,
                                columns_per_device.front().join_columns,
                                columns_per_device.front().join_column_types,
                                thread_count);
    for (int i = 1; i < thread_count; ++i) {
      hll_unify(hll_result,
                hll_result + i * padded_size_bytes,
                1 << count_distinct_desc.bitmap_sz_bits);
    }
    return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits), 0);
  }
#ifdef HAVE_CUDA
  const int device_count = columns_per_device.size();
  auto& data_mgr = executor_->getCatalog()->getDataMgr();
  std::vector<std::vector<uint8_t>> host_hll_buffers(device_count);
  for (auto& host_hll_buffer : host_hll_buffers) {
    host_hll_buffer.resize(count_distinct_desc.bitmapPaddedSizeBytes());
  }
  std::vector<std::future<void>> approximate_distinct_device_threads;
  for (int device_id = 0; device_id < device_count; ++device_id) {
    approximate_distinct_device_threads.emplace_back(std::async(
        std::launch::async,
        [device_id,
         &columns_per_device,
         &count_distinct_desc,
         &data_mgr,
         &host_hll_buffers,
         this] {
          ThrustAllocator allocator(&data_mgr, device_id);
          auto device_hll_buffer =
              allocator.allocateScopedBuffer(count_distinct_desc.bitmapPaddedSizeBytes());
          data_mgr.getCudaMgr()->zeroDeviceMem(
              device_hll_buffer, count_distinct_desc.bitmapPaddedSizeBytes(), device_id);
          const auto& columns_for_device = columns_per_device[device_id];
          auto join_columns_gpu =
              transfer_pod_vector_to_gpu(columns_for_device.join_columns, allocator);
          auto join_column_types_gpu =
              transfer_pod_vector_to_gpu(columns_for_device.join_column_types, allocator);
          const auto key_handler =
              GenericKeyHandler(columns_for_device.join_columns.size(),
                                true,
                                join_columns_gpu,
                                join_column_types_gpu,
                                nullptr,
                                nullptr);
          const auto key_handler_gpu = transfer_object_to_gpu(key_handler, allocator);
          approximate_distinct_tuples_on_device(
              reinterpret_cast<uint8_t*>(device_hll_buffer),
              count_distinct_desc.bitmap_sz_bits,
              key_handler_gpu,
              columns_for_device.join_columns[0].num_elems,
              executor_->blockSize(),
              executor_->gridSize());

          auto& host_hll_buffer = host_hll_buffers[device_id];
          copy_from_gpu(&data_mgr,
                        &host_hll_buffer[0],
                        reinterpret_cast<CUdeviceptr>(device_hll_buffer),
                        count_distinct_desc.bitmapPaddedSizeBytes(),
                        device_id);
        }));
  }
  for (auto& child : approximate_distinct_device_threads) {
    child.get();
  }
  CHECK_EQ(Data_Namespace::MemoryLevel::GPU_LEVEL, effective_memory_level);
  auto& result_hll_buffer = host_hll_buffers.front();
  auto hll_result = reinterpret_cast<int32_t*>(&result_hll_buffer[0]);
  for (int device_id = 1; device_id < device_count; ++device_id) {
    auto& host_hll_buffer = host_hll_buffers[device_id];
    hll_unify(hll_result,
              reinterpret_cast<int32_t*>(&host_hll_buffer[0]),
              1 << count_distinct_desc.bitmap_sz_bits);
  }
  return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits), 0);
#else
  UNREACHABLE();
  return {0, 0};
#endif  // HAVE_CUDA
}

JoinColumn BaselineJoinHashTable::fetchColumn(
    const Analyzer::ColumnVar* inner_col,
    const Data_Namespace::MemoryLevel& effective_memory_level,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    const int device_id) {
  static std::mutex fragment_fetch_mutex;
  const bool has_multi_frag = fragments.size() > 1;
  const auto& first_frag = fragments.front();

  const int8_t* col_buff = nullptr;
  size_t elem_count = 0;
  const size_t elem_width = inner_col->get_type_info().get_size();
  const auto& catalog = *executor_->getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  ThrustAllocator dev_buff_owner(&data_mgr, device_id);
  if (has_multi_frag) {
    try {
      std::tie(col_buff, elem_count) =
          getAllColumnFragments(*inner_col, fragments, chunks_owner);
    } catch (...) {
      throw FailedToFetchColumn();
    }
  }
  {
    std::lock_guard<std::mutex> fragment_fetch_lock(fragment_fetch_mutex);
    if (has_multi_frag) {
      if (effective_memory_level == Data_Namespace::GPU_LEVEL) {
        CHECK(col_buff != nullptr);
        CHECK_NE(elem_count, size_t(0));
        int8_t* dev_col_buff = nullptr;
        dev_col_buff = dev_buff_owner.allocate(elem_count * elem_width);
        copy_to_gpu(&data_mgr,
                    reinterpret_cast<CUdeviceptr>(dev_col_buff),
                    col_buff,
                    elem_count * elem_width,
                    device_id);
        col_buff = dev_col_buff;
      }
    } else {
      try {
        std::tie(col_buff, elem_count) =
            Executor::ExecutionDispatch::getColumnFragment(executor_,
                                                           *inner_col,
                                                           first_frag,
                                                           effective_memory_level,
                                                           device_id,
                                                           chunks_owner,
                                                           column_cache_);
      } catch (...) {
        throw FailedToFetchColumn();
      }
    }
  }
  return {col_buff, elem_count};
}

BaselineJoinHashTable::ColumnsForDevice BaselineJoinHashTable::fetchColumnsForDevice(
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    const int device_id) {
  const auto& catalog = *executor_->getCatalog();
  const auto inner_outer_pairs =
      normalize_column_pairs(condition_.get(), catalog, executor_->getTemporaryTables());
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs);

  std::vector<JoinColumn> join_columns;
  std::vector<std::shared_ptr<Chunk_NS::Chunk>> chunks_owner;
  std::vector<JoinColumnTypeInfo> join_column_types;
  std::vector<JoinBucketInfo> join_bucket_info;
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto inner_cd = get_column_descriptor_maybe(
        inner_col->get_column_id(), inner_col->get_table_id(), catalog);
    if (inner_cd && inner_cd->isVirtualCol) {
      throw FailedToJoinOnVirtualColumn();
    }
    join_columns.emplace_back(fetchColumn(
        inner_col, effective_memory_level, fragments, chunks_owner, device_id));
    const auto& ti = inner_col->get_type_info();
    join_column_types.emplace_back(JoinColumnTypeInfo{static_cast<size_t>(ti.get_size()),
                                                      0,
                                                      inline_fixed_encoding_null_val(ti),
                                                      isBitwiseEq(),
                                                      0,
                                                      get_join_column_type_kind(ti)});
  }
  return {join_columns, join_column_types, chunks_owner, join_bucket_info};
}

void BaselineJoinHashTable::reifyForDevice(const ColumnsForDevice& columns_for_device,
                                           const JoinHashTableInterface::HashType layout,
                                           const int device_id) {
  const auto& catalog = *executor_->getCatalog();
  const auto inner_outer_pairs =
      normalize_column_pairs(condition_.get(), catalog, executor_->getTemporaryTables());
  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs);
  const auto err = initHashTableForDevice(columns_for_device.join_columns,
                                          columns_for_device.join_column_types,
                                          columns_for_device.join_buckets,
                                          layout,
                                          effective_memory_level,
                                          device_id);
  if (err) {
    switch (err) {
      case ERR_FAILED_TO_FETCH_COLUMN:
        throw FailedToFetchColumn();
      case ERR_FAILED_TO_JOIN_ON_VIRTUAL_COLUMN:
        throw FailedToJoinOnVirtualColumn();
      default:
        throw HashJoinFail(
            std::string("Unrecognized error when initializing baseline hash table (") +
            std::to_string(err) + std::string(")"));
    }
  }
}

std::pair<const int8_t*, size_t> BaselineJoinHashTable::getAllColumnFragments(
    const Analyzer::ColumnVar& hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner) {
  std::lock_guard<std::mutex> linearized_multifrag_column_lock(
      linearized_multifrag_column_mutex_);
  auto linearized_column_cache_key =
      std::make_pair(hash_col.get_table_id(), hash_col.get_column_id());
  const auto cache_it = linearized_multifrag_columns_.find(linearized_column_cache_key);
  if (cache_it != linearized_multifrag_columns_.end()) {
    return cache_it->second;
  }
  const int8_t* col_buff;
  size_t total_elem_count;
  std::tie(col_buff, total_elem_count) =
      Executor::ExecutionDispatch::getAllColumnFragments(
          executor_, hash_col, fragments, chunks_owner, column_cache_);
  linearized_multifrag_column_owner_.addColBuffer(col_buff);
  const auto shard_count = shardCount();
  if (!shard_count) {
    const auto it_ok = linearized_multifrag_columns_.emplace(
        linearized_column_cache_key, LinearizedColumn{col_buff, total_elem_count});
    CHECK(it_ok.second);
  }
  return {col_buff, total_elem_count};
}

size_t BaselineJoinHashTable::shardCount() const {
  if (memory_level_ != Data_Namespace::GPU_LEVEL) {
    return 0;
  }
  return BaselineJoinHashTable::getShardCountForCondition(
      condition_.get(), ra_exe_unit_, executor_);
}

size_t BaselineJoinHashTable::getKeyComponentWidth() const {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    const auto inner_col = inner_outer_pair.first;
    const auto& inner_col_ti = inner_col->get_type_info();
    if (inner_col_ti.get_logical_size() > 4) {
      CHECK_EQ(size_t(8), inner_col_ti.get_logical_size());
      return 8;
    }
  }
  return 4;
}

size_t BaselineJoinHashTable::getKeyComponentCount() const {
  const auto catalog = executor_->getCatalog();
  const auto inner_outer_pairs =
      normalize_column_pairs(condition_.get(), *catalog, executor_->getTemporaryTables());
  return inner_outer_pairs.size();
}

Data_Namespace::MemoryLevel BaselineJoinHashTable::getEffectiveMemoryLevel(
    const std::vector<InnerOuter>& inner_outer_pairs) const {
  for (const auto& inner_outer_pair : inner_outer_pairs) {
    if (needs_dictionary_translation(
            inner_outer_pair.first, inner_outer_pair.second, executor_)) {
      return Data_Namespace::CPU_LEVEL;
    }
  }
  return memory_level_;
}

int BaselineJoinHashTable::initHashTableOnCpu(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const JoinHashTableInterface::HashType layout) {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto composite_key_info = getCompositeKeyInfo(inner_outer_pairs);
  CHECK(!join_columns.empty());
  HashTableCacheKey cache_key{join_columns.front().num_elems,
                              composite_key_info.cache_key_chunks,
                              condition_->get_optype()};
  initHashTableOnCpuFromCache(cache_key);
  if (cpu_hash_table_buff_) {
    return 0;
  }
  const auto key_component_width = getKeyComponentWidth();
  const auto key_component_count = getKeyComponentCount();
  const auto entry_size =
      (key_component_count +
       (layout == JoinHashTableInterface::HashType::OneToOne ? 1 : 0)) *
      key_component_width;
  const auto keys_for_all_rows = join_columns.front().num_elems;
  const size_t one_to_many_hash_entries =
      layout == JoinHashTableInterface::HashType::OneToMany
          ? 2 * entry_count_ + keys_for_all_rows
          : 0;
  const size_t hash_table_size =
      entry_size * entry_count_ + one_to_many_hash_entries * sizeof(int32_t);

  // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
  if (hash_table_size > std::numeric_limits<int32_t>::max()) {
    throw TooManyHashEntries();
  }

  VLOG(1) << "Initializing CPU Join Hash Table with " << entry_count_
          << " hash entries and " << one_to_many_hash_entries
          << " entries in the one to many buffer";
  VLOG(1) << "Total hash table size: " << hash_table_size << " Bytes";

  cpu_hash_table_buff_.reset(new std::vector<int8_t>(hash_table_size));
  int thread_count = cpu_threads();
  std::vector<std::future<void>> init_cpu_buff_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    init_cpu_buff_threads.emplace_back(
        std::async(std::launch::async,
                   [this,
                    key_component_count,
                    key_component_width,
                    thread_idx,
                    thread_count,
                    layout] {
                     switch (key_component_width) {
                       case 4:
                         init_baseline_hash_join_buff_32(
                             &(*cpu_hash_table_buff_)[0],
                             entry_count_,
                             key_component_count,
                             layout == JoinHashTableInterface::HashType::OneToOne,
                             -1,
                             thread_idx,
                             thread_count);
                         break;
                       case 8:
                         init_baseline_hash_join_buff_64(
                             &(*cpu_hash_table_buff_)[0],
                             entry_count_,
                             key_component_count,
                             layout == JoinHashTableInterface::HashType::OneToOne,
                             -1,
                             thread_idx,
                             thread_count);
                         break;
                       default:
                         CHECK(false);
                     }
                   }));
  }
  for (auto& child : init_cpu_buff_threads) {
    child.get();
  }
  std::vector<std::future<int>> fill_cpu_buff_threads;
  for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    fill_cpu_buff_threads.emplace_back(std::async(
        std::launch::async,
        [this,
         &composite_key_info,
         &join_columns,
         &join_column_types,
         key_component_count,
         key_component_width,
         layout,
         thread_idx,
         thread_count] {
          switch (key_component_width) {
            case 4: {
              const auto key_handler =
                  GenericKeyHandler(key_component_count,
                                    true,
                                    &join_columns[0],
                                    &join_column_types[0],
                                    &composite_key_info.sd_inner_proxy_per_key[0],
                                    &composite_key_info.sd_outer_proxy_per_key[0]);
              return fill_baseline_hash_join_buff_32(
                  &(*cpu_hash_table_buff_)[0],
                  entry_count_,
                  -1,
                  key_component_count,
                  layout == JoinHashTableInterface::HashType::OneToOne,
                  &key_handler,
                  join_columns[0].num_elems,
                  thread_idx,
                  thread_count);
            } break;
            case 8: {
              const auto key_handler =
                  GenericKeyHandler(key_component_count,
                                    true,
                                    &join_columns[0],
                                    &join_column_types[0],
                                    &composite_key_info.sd_inner_proxy_per_key[0],
                                    &composite_key_info.sd_outer_proxy_per_key[0]);
              return fill_baseline_hash_join_buff_64(
                  &(*cpu_hash_table_buff_)[0],
                  entry_count_,
                  -1,
                  key_component_count,
                  layout == JoinHashTableInterface::HashType::OneToOne,
                  &key_handler,
                  join_columns[0].num_elems,
                  thread_idx,
                  thread_count);
            } break;
            default:
              CHECK(false);
          }
          return -1;
        }));
  }
  int err = 0;
  for (auto& child : fill_cpu_buff_threads) {
    int partial_err = child.get();
    if (partial_err) {
      err = partial_err;
    }
  }
  if (err) {
    cpu_hash_table_buff_.reset();
    return err;
  }
  if (layout == JoinHashTableInterface::HashType::OneToMany) {
    auto one_to_many_buff = reinterpret_cast<int32_t*>(&(*cpu_hash_table_buff_)[0] +
                                                       entry_count_ * entry_size);
    init_hash_join_buff(one_to_many_buff, entry_count_, -1, 0, 1);
    switch (key_component_width) {
      case 4: {
        const auto composite_key_dict =
            reinterpret_cast<int32_t*>(&(*cpu_hash_table_buff_)[0]);
        fill_one_to_many_baseline_hash_table_32(one_to_many_buff,
                                                composite_key_dict,
                                                entry_count_,
                                                -1,
                                                key_component_count,
                                                join_columns,
                                                join_column_types,
                                                join_bucket_info,
                                                composite_key_info.sd_inner_proxy_per_key,
                                                composite_key_info.sd_outer_proxy_per_key,
                                                thread_count);
        break;
      }
      case 8: {
        const auto composite_key_dict =
            reinterpret_cast<int64_t*>(&(*cpu_hash_table_buff_)[0]);
        fill_one_to_many_baseline_hash_table_64(one_to_many_buff,
                                                composite_key_dict,
                                                entry_count_,
                                                -1,
                                                key_component_count,
                                                join_columns,
                                                join_column_types,
                                                join_bucket_info,
                                                composite_key_info.sd_inner_proxy_per_key,
                                                composite_key_info.sd_outer_proxy_per_key,
                                                thread_count);
        break;
      }
      default:
        CHECK(false);
    }
  }
  if (!err && getInnerTableId() > 0) {
    putHashTableOnCpuToCache(cache_key);
  }
  return err;
}

int BaselineJoinHashTable::initHashTableOnGpu(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const JoinHashTableInterface::HashType layout,
    const size_t key_component_width,
    const size_t key_component_count,
    const int device_id) {
  int err = 0;
#ifdef HAVE_CUDA
  const auto catalog = executor_->getCatalog();
  auto& data_mgr = catalog->getDataMgr();
  ThrustAllocator allocator(&data_mgr, device_id);
  auto dev_err_buff =
      reinterpret_cast<CUdeviceptr>(allocator.allocateScopedBuffer(sizeof(int)));
  copy_to_gpu(&data_mgr, dev_err_buff, &err, sizeof(err), device_id);
  switch (key_component_width) {
    case 4:
      init_baseline_hash_join_buff_on_device_32(
          reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          entry_count_,
          key_component_count,
          layout == JoinHashTableInterface::HashType::OneToOne,
          -1,
          executor_->blockSize(),
          executor_->gridSize());
      break;
    case 8:
      init_baseline_hash_join_buff_on_device_64(
          reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          entry_count_,
          key_component_count,
          layout == JoinHashTableInterface::HashType::OneToOne,
          -1,
          executor_->blockSize(),
          executor_->gridSize());
      break;
    default:
      UNREACHABLE();
  }
  auto join_columns_gpu = transfer_pod_vector_to_gpu(join_columns, allocator);
  auto hash_buff =
      reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr());
  auto join_column_types_gpu = transfer_pod_vector_to_gpu(join_column_types, allocator);

  const auto key_handler = GenericKeyHandler(key_component_count,
                                             true,
                                             join_columns_gpu,
                                             join_column_types_gpu,
                                             nullptr,
                                             nullptr);
  const auto key_handler_gpu = transfer_object_to_gpu(key_handler, allocator);
  switch (key_component_width) {
    case 4: {
      fill_baseline_hash_join_buff_on_device_32(
          hash_buff,
          entry_count_,
          -1,
          key_component_count,
          layout == JoinHashTableInterface::HashType::OneToOne,
          reinterpret_cast<int*>(dev_err_buff),
          key_handler_gpu,
          join_columns.front().num_elems,
          executor_->blockSize(),
          executor_->gridSize());
      copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
      break;
    }
    case 8: {
      fill_baseline_hash_join_buff_on_device_64(
          hash_buff,
          entry_count_,
          -1,
          key_component_count,
          layout == JoinHashTableInterface::HashType::OneToOne,
          reinterpret_cast<int*>(dev_err_buff),
          key_handler_gpu,
          join_columns.front().num_elems,
          executor_->blockSize(),
          executor_->gridSize());
      copy_from_gpu(&data_mgr, &err, dev_err_buff, sizeof(err), device_id);
      break;
    }
    default:
      UNREACHABLE();
  }
  if (err) {
    return err;
  }
  if (layout == JoinHashTableInterface::HashType::OneToMany) {
    const auto entry_size = key_component_count * key_component_width;
    auto one_to_many_buff = reinterpret_cast<int32_t*>(
        gpu_hash_table_buff_[device_id]->getMemoryPtr() + entry_count_ * entry_size);
    switch (key_component_width) {
      case 4: {
        const auto composite_key_dict =
            reinterpret_cast<int32_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr());
        init_hash_join_buff_on_device(one_to_many_buff,
                                      entry_count_,
                                      -1,
                                      executor_->blockSize(),
                                      executor_->gridSize());
        fill_one_to_many_baseline_hash_table_on_device_32(one_to_many_buff,
                                                          composite_key_dict,
                                                          entry_count_,
                                                          -1,
                                                          key_component_count,
                                                          key_handler_gpu,
                                                          join_columns.front().num_elems,
                                                          executor_->blockSize(),
                                                          executor_->gridSize());
        break;
      }
      case 8: {
        const auto composite_key_dict =
            reinterpret_cast<int64_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr());
        init_hash_join_buff_on_device(one_to_many_buff,
                                      entry_count_,
                                      -1,
                                      executor_->blockSize(),
                                      executor_->gridSize());
        fill_one_to_many_baseline_hash_table_on_device_64(one_to_many_buff,
                                                          composite_key_dict,
                                                          entry_count_,
                                                          -1,
                                                          key_handler_gpu,
                                                          join_columns.front().num_elems,
                                                          executor_->blockSize(),
                                                          executor_->gridSize());
        break;
      }
      default:
        UNREACHABLE();
    }
  }
#else
  UNREACHABLE();
#endif
  return err;
}

int BaselineJoinHashTable::initHashTableForDevice(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const JoinHashTableInterface::HashType layout,
    const Data_Namespace::MemoryLevel effective_memory_level,
    const int device_id) {
  const auto key_component_width = getKeyComponentWidth();
  const auto key_component_count = getKeyComponentCount();
  int err = 0;
#ifdef HAVE_CUDA
  const auto catalog = executor_->getCatalog();
  auto& data_mgr = catalog->getDataMgr();
  if (memory_level_ == Data_Namespace::GPU_LEVEL) {
    const auto entry_size =
        (key_component_count +
         (layout == JoinHashTableInterface::HashType::OneToOne ? 1 : 0)) *
        key_component_width;
    const auto keys_for_all_rows = emitted_keys_count_;
    const size_t one_to_many_hash_entries =
        layout == JoinHashTableInterface::HashType::OneToMany
            ? 2 * entry_count_ + keys_for_all_rows
            : 0;
    const size_t hash_table_size =
        entry_size * entry_count_ + one_to_many_hash_entries * sizeof(int32_t);

    // We can't allocate more than 2GB contiguous memory on GPU and each entry is 4 bytes.
    if (hash_table_size > std::numeric_limits<int32_t>::max()) {
      throw TooManyHashEntries();
    }

    VLOG(1) << "Initializing GPU Hash Table for device " << device_id << " with "
            << entry_count_ << " hash entries and " << one_to_many_hash_entries
            << " entries in the one to many buffer";
    VLOG(1) << "Total hash table size: " << hash_table_size << " Bytes";
    gpu_hash_table_buff_[device_id] =
        alloc_gpu_abstract_buffer(&data_mgr, hash_table_size, device_id);
  }
#else
  CHECK_EQ(Data_Namespace::CPU_LEVEL, effective_memory_level);
#endif
  if (effective_memory_level == Data_Namespace::CPU_LEVEL) {
    std::lock_guard<std::mutex> cpu_hash_table_buff_lock(cpu_hash_table_buff_mutex_);
    err = initHashTableOnCpu(join_columns, join_column_types, join_bucket_info, layout);
    // Transfer the hash table on the GPU if we've only built it on CPU
    // but the query runs on GPU (join on dictionary encoded columns).
    // Don't transfer the buffer if there was an error since we'll bail anyway.
    if (memory_level_ == Data_Namespace::GPU_LEVEL && !err) {
#ifdef HAVE_CUDA
      copy_to_gpu(
          &data_mgr,
          reinterpret_cast<CUdeviceptr>(gpu_hash_table_buff_[device_id]->getMemoryPtr()),
          &(*cpu_hash_table_buff_)[0],
          cpu_hash_table_buff_->size() * sizeof((*cpu_hash_table_buff_)[0]),
          device_id);
#else
      CHECK(false);
#endif
    }
  } else {
    err = initHashTableOnGpu(join_columns,
                             join_column_types,
                             join_bucket_info,
                             layout,
                             key_component_width,
                             key_component_count,
                             device_id);
  }
  return err;
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define LL_FP(v) executor_->ll_fp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

llvm::Value* BaselineJoinHashTable::codegenSlot(const CompilationOptions& co,
                                                const size_t index) {
  CHECK(getHashType() == JoinHashTableInterface::HashType::OneToOne);
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  auto key_buff_lv = codegenKey(co);
  const auto hash_ptr = hashPtr(index);
  const auto key_ptr_lv =
      LL_BUILDER.CreatePointerCast(key_buff_lv, llvm::Type::getInt8PtrTy(LL_CONTEXT));
  const auto key_size_lv = LL_INT(getKeyComponentCount() * key_component_width);
  return executor_->cgen_state_->emitExternalCall(
      "baseline_hash_join_idx_" + std::to_string(key_component_width * 8),
      get_int_type(64, LL_CONTEXT),
      {hash_ptr, key_ptr_lv, key_size_lv, LL_INT(entry_count_)});
}

HashJoinMatchingSet BaselineJoinHashTable::codegenMatchingSet(
    const CompilationOptions& co,
    const size_t index) {
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  auto key_buff_lv = codegenKey(co);
  CHECK(layout_ == JoinHashTableInterface::HashType::OneToMany);
  auto hash_ptr = JoinHashTable::codegenHashTableLoad(index, executor_);
  const auto composite_dict_ptr_type =
      llvm::Type::getIntNPtrTy(LL_CONTEXT, key_component_width * 8);
  const auto composite_key_dict =
      hash_ptr->getType()->isPointerTy()
          ? LL_BUILDER.CreatePointerCast(hash_ptr, composite_dict_ptr_type)
          : LL_BUILDER.CreateIntToPtr(hash_ptr, composite_dict_ptr_type);
  const auto key_component_count = getKeyComponentCount();
  const auto key = executor_->cgen_state_->emitExternalCall(
      "get_composite_key_index_" + std::to_string(key_component_width * 8),
      get_int_type(64, LL_CONTEXT),
      {key_buff_lv,
       LL_INT(key_component_count),
       composite_key_dict,
       LL_INT(entry_count_)});
  auto one_to_many_ptr = hash_ptr;
  if (one_to_many_ptr->getType()->isPointerTy()) {
    one_to_many_ptr =
        LL_BUILDER.CreatePtrToInt(hash_ptr, llvm::Type::getInt64Ty(LL_CONTEXT));
  } else {
    CHECK(one_to_many_ptr->getType()->isIntegerTy(64));
  }
  const auto composite_key_dict_size =
      entry_count_ * key_component_count * key_component_width;
  one_to_many_ptr =
      LL_BUILDER.CreateAdd(one_to_many_ptr, LL_INT(composite_key_dict_size));
  return JoinHashTable::codegenMatchingSet(
      {one_to_many_ptr, key, LL_INT(int64_t(0)), LL_INT(entry_count_ - 1)},
      false,
      false,
      false,
      entry_count_ * sizeof(int32_t),
      executor_);
}

llvm::Value* BaselineJoinHashTable::codegenKey(const CompilationOptions& co) {
  const auto key_component_width = getKeyComponentWidth();
  CHECK(key_component_width == 4 || key_component_width == 8);
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto key_size_lv = LL_INT(getKeyComponentCount() * key_component_width);
  llvm::Value* key_buff_lv{nullptr};
  switch (key_component_width) {
    case 4:
      key_buff_lv =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv);
      break;
    case 8:
      key_buff_lv =
          LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
      break;
    default:
      CHECK(false);
  }

  for (size_t i = 0; i < getKeyComponentCount(); ++i) {
    const auto key_comp_dest_lv = LL_BUILDER.CreateGEP(key_buff_lv, LL_INT(i));
    const auto& inner_outer_pair = inner_outer_pairs[i];
    const auto outer_col = inner_outer_pair.second;
    const auto col_lvs = executor_->codegen(outer_col, true, co);
    CHECK_EQ(size_t(1), col_lvs.size());
    const auto col_lv = LL_BUILDER.CreateSExt(
        col_lvs.front(), get_int_type(key_component_width * 8, LL_CONTEXT));
    LL_BUILDER.CreateStore(col_lv, key_comp_dest_lv);
  }
  return key_buff_lv;
}

llvm::Value* BaselineJoinHashTable::hashPtr(const size_t index) {
  auto hash_ptr = JoinHashTable::codegenHashTableLoad(index, executor_);
  const auto pi8_type = llvm::Type::getInt8PtrTy(LL_CONTEXT);
  return hash_ptr->getType()->isPointerTy()
             ? LL_BUILDER.CreatePointerCast(hash_ptr, pi8_type)
             : LL_BUILDER.CreateIntToPtr(hash_ptr, pi8_type);
}

#undef ROW_FUNC
#undef LL_INT
#undef LL_BUILDER
#undef LL_CONTEXT

int BaselineJoinHashTable::getInnerTableId() const noexcept {
  try {
    return getInnerTableId(condition_.get(), executor_);
  } catch (...) {
    CHECK(false);
  }
  return 0;
}

int BaselineJoinHashTable::getInnerTableRteIdx() const noexcept {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  CHECK(!inner_outer_pairs.empty());
  const auto first_inner_col = inner_outer_pairs.front().first;
  return first_inner_col->get_rte_idx();
}

JoinHashTableInterface::HashType BaselineJoinHashTable::getHashType() const noexcept {
  return layout_;
}

int BaselineJoinHashTable::getInnerTableId(const Analyzer::BinOper* condition,
                                           const Executor* executor) {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition, *executor->getCatalog(), executor->getTemporaryTables());
  CHECK(!inner_outer_pairs.empty());
  const auto first_inner_col = inner_outer_pairs.front().first;
  return first_inner_col->get_table_id();
}

void BaselineJoinHashTable::checkHashJoinReplicationConstraint(const int table_id) const {
  if (!g_cluster) {
    return;
  }
  if (table_id >= 0) {
    const auto inner_td = executor_->getCatalog()->getMetadataForTable(table_id);
    CHECK(inner_td);
    const auto shard_count = shardCount();
    if (!shard_count && !table_is_replicated(inner_td)) {
      throw TableMustBeReplicated(inner_td->tableName);
    }
  }
}

void BaselineJoinHashTable::initHashTableOnCpuFromCache(const HashTableCacheKey& key) {
  std::lock_guard<std::mutex> hash_table_cache_lock(hash_table_cache_mutex_);
  for (const auto& kv : hash_table_cache_) {
    if (kv.first == key) {
      cpu_hash_table_buff_ = kv.second.buffer;
      layout_ = kv.second.type;
      entry_count_ = kv.second.entry_count;
      emitted_keys_count_ = kv.second.emitted_keys_count;
      break;
    }
  }
}

void BaselineJoinHashTable::putHashTableOnCpuToCache(const HashTableCacheKey& key) {
  std::lock_guard<std::mutex> hash_table_cache_lock(hash_table_cache_mutex_);
  for (const auto& kv : hash_table_cache_) {
    if (std::get<0>(kv) == key) {
      return;
    }
  }
  hash_table_cache_.emplace_back(
      key,
      HashTableCacheValue{
          cpu_hash_table_buff_, layout_, entry_count_, emitted_keys_count_});
}

std::pair<ssize_t, size_t> BaselineJoinHashTable::getApproximateTupleCountFromCache(
    const HashTableCacheKey& key) const {
  std::lock_guard<std::mutex> hash_table_cache_lock(hash_table_cache_mutex_);
  for (const auto& kv : hash_table_cache_) {
    if (kv.first == key) {
      return std::make_pair(kv.second.entry_count, kv.second.emitted_keys_count);
    }
  }
  return std::make_pair(-1, 0);
}

bool BaselineJoinHashTable::isBitwiseEq() const {
  return condition_->get_optype() == kBW_EQ;
}

void BaselineJoinHashTable::freeHashBufferMemory() {
#ifdef HAVE_CUDA
  freeHashBufferGpuMemory();
#endif
  freeHashBufferCpuMemory();
}

void BaselineJoinHashTable::freeHashBufferGpuMemory() {
#ifdef HAVE_CUDA
  const auto& catalog = *executor_->getCatalog();
  auto& data_mgr = catalog.getDataMgr();
  for (auto& buf : gpu_hash_table_buff_) {
    if (buf) {
      free_gpu_abstract_buffer(&data_mgr, buf);
      buf = nullptr;
    }
  }
#else
  CHECK(false);
#endif  // HAVE_CUDA
}

void BaselineJoinHashTable::freeHashBufferCpuMemory() {
  cpu_hash_table_buff_.reset();
}

std::map<std::vector<ChunkKey>, JoinHashTableInterface::HashType>
    HashTypeCache::hash_type_cache_;
std::mutex HashTypeCache::hash_type_cache_mutex_;

void HashTypeCache::set(const std::vector<ChunkKey>& key,
                        const JoinHashTableInterface::HashType hash_type) {
  std::lock_guard<std::mutex> hash_type_cache_lock(hash_type_cache_mutex_);
  hash_type_cache_[key] = hash_type;
}

std::pair<JoinHashTableInterface::HashType, bool> HashTypeCache::get(
    const std::vector<ChunkKey>& key) {
  std::lock_guard<std::mutex> hash_type_cache_lock(hash_type_cache_mutex_);
  const auto it = hash_type_cache_.find(key);
  if (it == hash_type_cache_.end()) {
    return {JoinHashTableInterface::HashType::OneToOne, false};
  }
  return {it->second, true};
}
