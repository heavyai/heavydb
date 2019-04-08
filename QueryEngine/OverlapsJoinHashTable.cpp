/*
 * Copyright 2018 OmniSci, Inc.
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

#include "OverlapsJoinHashTable.h"
#include "ExpressionRewrite.h"
#include "HashJoinKeyHandlers.h"
#include "JoinHashTableGpuUtils.h"
#include "JoinHashTableInterface.h"

#include "Execute.h"

#include "Shared/unreachable.h"

std::shared_ptr<OverlapsJoinHashTable> OverlapsJoinHashTable::getInstance(
    const std::shared_ptr<Analyzer::BinOper> condition,
    const std::vector<InputTableInfo>& query_infos,
    const RelAlgExecutionUnit& ra_exe_unit,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_count,
    ColumnCacheMap& column_map,
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
  auto join_hash_table = std::make_shared<OverlapsJoinHashTable>(condition,
                                                                 query_infos,
                                                                 ra_exe_unit,
                                                                 memory_level,
                                                                 entries_per_device,
                                                                 column_map,
                                                                 executor);
  join_hash_table->checkHashJoinReplicationConstraint(
      getInnerTableId(condition.get(), executor));
  try {
    join_hash_table->reify(device_count);
  } catch (const HashJoinFail& e) {
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

void OverlapsJoinHashTable::reifyWithLayout(
    const int device_count,
    const JoinHashTableInterface::HashType layout) {
  CHECK(layout == JoinHashTableInterface::HashType::OneToMany);
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

  size_t tuple_count;
  std::tie(tuple_count, emitted_keys_count_) = approximateTupleCount(columns_per_device);
  const auto entry_count = 2 * std::max(tuple_count, size_t(1));

  entry_count_ =
      get_entries_per_device(entry_count, shard_count, device_count, memory_level_);

  std::vector<std::future<void>> init_threads;
  for (int device_id = 0; device_id < device_count; ++device_id) {
    const auto fragments =
        shard_count
            ? only_shards_for_device(query_info.fragments, device_id, device_count)
            : query_info.fragments;
    init_threads.push_back(std::async(std::launch::async,
                                      &OverlapsJoinHashTable::reifyForDevice,
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

BaselineJoinHashTable::ColumnsForDevice OverlapsJoinHashTable::fetchColumnsForDevice(
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
    const auto join_column_info = fetchColumn(
        inner_col, effective_memory_level, fragments, chunks_owner, device_id);
    join_columns.emplace_back(
        JoinColumn{join_column_info.col_buff, join_column_info.num_elems});
    const auto& ti = inner_col->get_type_info();
    join_column_types.emplace_back(JoinColumnTypeInfo{static_cast<size_t>(ti.get_size()),
                                                      0,
                                                      inline_int_null_value<int64_t>(),
                                                      isBitwiseEq(),
                                                      0,
                                                      get_join_column_type_kind(ti)});
    CHECK(ti.is_array()) << "Overlaps join currently only supported for arrays.";

    if (bucket_sizes_for_dimension_.empty()) {
      computeBucketSizes(bucket_sizes_for_dimension_,
                         join_columns.back(),
                         inner_outer_pairs,
                         join_column_info.num_elems);
    }
    const auto elem_ti = ti.get_elem_type();
    CHECK(elem_ti.is_fp());
    join_bucket_info.emplace_back(
        JoinBucketInfo{bucket_sizes_for_dimension_, elem_ti.get_type() == kDOUBLE});
  }
  return {join_columns, join_column_types, chunks_owner, join_bucket_info};
}

std::pair<size_t, size_t> OverlapsJoinHashTable::approximateTupleCount(
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
  // Number of keys must match dimension of buckets
  CHECK_EQ(columns_per_device.front().join_columns.size(),
           columns_per_device.front().join_buckets.size());
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

    std::vector<int32_t> num_keys_for_row;
    // TODO(adb): support multi-column overlaps join
    CHECK_EQ(columns_per_device.size(), 1);
    num_keys_for_row.resize(columns_per_device.front().join_columns[0].num_elems);

    approximate_distinct_tuples_overlaps(hll_result,
                                         num_keys_for_row,
                                         count_distinct_desc.bitmap_sz_bits,
                                         padded_size_bytes,
                                         columns_per_device.front().join_columns,
                                         columns_per_device.front().join_column_types,
                                         columns_per_device.front().join_buckets,
                                         thread_count);
    for (int i = 1; i < thread_count; ++i) {
      hll_unify(hll_result,
                hll_result + i * padded_size_bytes,
                1 << count_distinct_desc.bitmap_sz_bits);
    }
    return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits),
                          num_keys_for_row.size() > 0 ? num_keys_for_row.back() : 0);
  }
#ifdef HAVE_CUDA
  const int device_count = columns_per_device.size();
  auto& data_mgr = executor_->getCatalog()->getDataMgr();
  std::vector<std::vector<uint8_t>> host_hll_buffers(device_count);
  for (auto& host_hll_buffer : host_hll_buffers) {
    host_hll_buffer.resize(count_distinct_desc.bitmapPaddedSizeBytes());
  }
  std::vector<size_t> emitted_keys_count_device_threads(device_count, 0);
  std::vector<std::future<void>> approximate_distinct_device_threads;
  for (int device_id = 0; device_id < device_count; ++device_id) {
    approximate_distinct_device_threads.emplace_back(std::async(
        std::launch::async,
        [device_id,
         &columns_per_device,
         &count_distinct_desc,
         &data_mgr,
         &host_hll_buffers,
         &emitted_keys_count_device_threads,
         this] {
          ThrustAllocator allocator(&data_mgr, device_id);
          auto device_hll_buffer =
              allocator.allocateScopedBuffer(count_distinct_desc.bitmapPaddedSizeBytes());
          data_mgr.getCudaMgr()->zeroDeviceMem(
              device_hll_buffer, count_distinct_desc.bitmapPaddedSizeBytes(), device_id);
          const auto& columns_for_device = columns_per_device[device_id];
          auto join_columns_gpu =
              transfer_pod_vector_to_gpu(columns_for_device.join_columns, allocator);

          CHECK_GT(columns_for_device.join_buckets.size(), 0);
          const auto& bucket_sizes_for_dimension =
              columns_for_device.join_buckets[0].bucket_sizes_for_dimension;
          auto bucket_sizes_gpu = allocator.allocateScopedBuffer(
              bucket_sizes_for_dimension.size() * sizeof(double));
          copy_to_gpu(&data_mgr,
                      reinterpret_cast<CUdeviceptr>(bucket_sizes_gpu),
                      bucket_sizes_for_dimension.data(),
                      bucket_sizes_for_dimension.size() * sizeof(double),
                      device_id);
          const size_t row_counts_buffer_sz =
              columns_per_device.front().join_columns[0].num_elems * sizeof(int32_t);
          auto row_counts_buffer = allocator.allocateScopedBuffer(row_counts_buffer_sz);
          data_mgr.getCudaMgr()->zeroDeviceMem(
              row_counts_buffer, row_counts_buffer_sz, device_id);
          const auto key_handler =
              OverlapsKeyHandler(bucket_sizes_for_dimension.size(),
                                 join_columns_gpu,
                                 reinterpret_cast<double*>(bucket_sizes_gpu));
          const auto key_handler_gpu = transfer_object_to_gpu(key_handler, allocator);
          approximate_distinct_tuples_on_device_overlaps(
              reinterpret_cast<uint8_t*>(device_hll_buffer),
              count_distinct_desc.bitmap_sz_bits,
              reinterpret_cast<int32_t*>(row_counts_buffer),
              key_handler_gpu,
              columns_for_device.join_columns[0].num_elems,
              executor_->blockSize(),
              executor_->gridSize());

          auto& host_emitted_keys_count = emitted_keys_count_device_threads[device_id];
          copy_from_gpu(&data_mgr,
                        &host_emitted_keys_count,
                        reinterpret_cast<CUdeviceptr>(
                            row_counts_buffer +
                            (columns_per_device.front().join_columns[0].num_elems - 1) *
                                sizeof(int32_t)),
                        sizeof(int32_t),
                        device_id);

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
  size_t emitted_keys_count = 0;
  for (auto& emitted_keys_count_device : emitted_keys_count_device_threads) {
    emitted_keys_count += emitted_keys_count_device;
  }
  return std::make_pair(hll_size(hll_result, count_distinct_desc.bitmap_sz_bits),
                        emitted_keys_count);
#else
  UNREACHABLE();
  return {0, 0};
#endif  // HAVE_CUDA
}

size_t OverlapsJoinHashTable::getKeyComponentWidth() const {
  return 8;
}

size_t OverlapsJoinHashTable::getKeyComponentCount() const {
  return bucket_sizes_for_dimension_.size();
}

int OverlapsJoinHashTable::initHashTableOnCpu(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const JoinHashTableInterface::HashType layout) {
  const auto inner_outer_pairs = normalize_column_pairs(
      condition_.get(), *executor_->getCatalog(), executor_->getTemporaryTables());
  const auto composite_key_info = getCompositeKeyInfo(inner_outer_pairs);
  CHECK(!join_columns.empty());
  CHECK(!join_bucket_info.empty());
  HashTableCacheKey cache_key{join_columns.front().num_elems,
                              composite_key_info.cache_key_chunks,
                              condition_->get_optype()};
  initHashTableOnCpuFromCache(cache_key);
  if (cpu_hash_table_buff_) {
    return 0;
  }
  CHECK(layout == JoinHashTableInterface::HashType::OneToMany);
  const auto key_component_width = getKeyComponentWidth();
  const auto key_component_count = join_bucket_info[0].bucket_sizes_for_dimension.size();
  const auto entry_size = key_component_count * key_component_width;
  const auto keys_for_all_rows = emitted_keys_count_;
  const size_t one_to_many_hash_entries = 2 * entry_count_ + keys_for_all_rows;
  const size_t hash_table_size =
      entry_size * entry_count_ + one_to_many_hash_entries * sizeof(int32_t);

  VLOG(1) << "Initializing CPU Overlaps Join Hash Table with " << entry_count_
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
    fill_cpu_buff_threads.emplace_back(
        std::async(std::launch::async,
                   [this,
                    &join_columns,
                    &join_bucket_info,
                    key_component_count,
                    key_component_width,
                    layout,
                    thread_idx,
                    thread_count] {
                     switch (key_component_width) {
                       case 4: {
                         const auto key_handler = OverlapsKeyHandler(
                             key_component_count,
                             &join_columns[0],
                             join_bucket_info[0].bucket_sizes_for_dimension.data());
                         return overlaps_fill_baseline_hash_join_buff_32(
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
                         const auto key_handler = OverlapsKeyHandler(
                             key_component_count,
                             &join_columns[0],
                             join_bucket_info[0].bucket_sizes_for_dimension.data());
                         return overlaps_fill_baseline_hash_join_buff_64(
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

int OverlapsJoinHashTable::initHashTableOnGpu(
    const std::vector<JoinColumn>& join_columns,
    const std::vector<JoinColumnTypeInfo>& join_column_types,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const JoinHashTableInterface::HashType layout,
    const size_t key_component_width,
    const size_t key_component_count,
    const int device_id) {
  int err = 0;
  // TODO(adb): 4 byte keys
  CHECK_EQ(key_component_width, size_t(8));
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
      CHECK(false);
  }
  auto join_columns_gpu = transfer_pod_vector_to_gpu(join_columns, allocator);
  auto hash_buff =
      reinterpret_cast<int8_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr());
  CHECK_EQ(join_columns.size(), 1);
  auto& bucket_sizes_for_dimension = join_bucket_info[0].bucket_sizes_for_dimension;
  auto bucket_sizes_gpu =
      transfer_pod_vector_to_gpu(bucket_sizes_for_dimension, allocator);
  const auto key_handler = OverlapsKeyHandler(
      bucket_sizes_for_dimension.size(), join_columns_gpu, bucket_sizes_gpu);
  const auto key_handler_gpu = transfer_object_to_gpu(key_handler, allocator);
  switch (key_component_width) {
    case 8: {
      overlaps_fill_baseline_hash_join_buff_on_device_64(
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
  const auto entry_size = key_component_count * key_component_width;
  auto one_to_many_buff = reinterpret_cast<int32_t*>(
      gpu_hash_table_buff_[device_id]->getMemoryPtr() + entry_count_ * entry_size);
  switch (key_component_width) {
    case 8: {
      const auto composite_key_dict =
          reinterpret_cast<int64_t*>(gpu_hash_table_buff_[device_id]->getMemoryPtr());
      init_hash_join_buff_on_device(one_to_many_buff,
                                    entry_count_,
                                    -1,
                                    executor_->blockSize(),
                                    executor_->gridSize());
      overlaps_fill_one_to_many_baseline_hash_table_on_device_64(
          one_to_many_buff,
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
#else
  UNREACHABLE();
#endif
  return err;
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_INT(v) executor_->ll_int(v)
#define LL_FP(v) executor_->ll_fp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

llvm::Value* OverlapsJoinHashTable::codegenKey(const CompilationOptions& co) {
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

  const auto& inner_outer_pair = inner_outer_pairs[0];
  const auto outer_col = inner_outer_pair.second;
  const auto outer_col_ti = outer_col->get_type_info();

  if (outer_col_ti.is_geometry()) {
    // TODO(adb): for points we will use the coords array, but for other geometries we
    // will need to use the bounding box. For now only support points.
    CHECK_EQ(outer_col_ti.get_type(), kPOINT);
    CHECK_EQ(bucket_sizes_for_dimension_.size(), static_cast<size_t>(2));

    const auto col_lvs = executor_->codegen(outer_col, true, co);
    CHECK_EQ(col_lvs.size(), size_t(1));

    const auto outer_col_var = dynamic_cast<const Analyzer::ColumnVar*>(outer_col);
    CHECK(outer_col_var);
    const auto coords_cd = executor_->getCatalog()->getMetadataForColumn(
        outer_col_var->get_table_id(), outer_col_var->get_column_id() + 1);
    CHECK(coords_cd);

    const auto array_ptr = executor_->cgen_state_->emitExternalCall(
        "array_buff",
        llvm::Type::getInt8PtrTy(executor_->cgen_state_->context_),
        {col_lvs.front(), executor_->posArg(outer_col)});
    CHECK(coords_cd->columnType.get_elem_type().get_type() == kTINYINT)
        << "Only TINYINT coordinates columns are supported in geo overlaps hash join.";
    const auto arr_ptr =
        executor_->castArrayPointer(array_ptr, coords_cd->columnType.get_elem_type());

    for (size_t i = 0; i < 2; i++) {
      const auto key_comp_dest_lv = LL_BUILDER.CreateGEP(key_buff_lv, LL_INT(i));

      // Note that get_bucket_key_for_range_compressed will need to be specialized for
      // future compression schemes
      auto bucket_key =
          outer_col_ti.get_compression() == kENCODING_GEOINT
              ? executor_->cgen_state_->emitExternalCall(
                    "get_bucket_key_for_range_compressed",
                    get_int_type(64, LL_CONTEXT),
                    {arr_ptr, LL_INT(i), LL_FP(bucket_sizes_for_dimension_[i])})
              : executor_->cgen_state_->emitExternalCall(
                    "get_bucket_key_for_range_double",
                    get_int_type(64, LL_CONTEXT),
                    {arr_ptr, LL_INT(i), LL_FP(bucket_sizes_for_dimension_[i])});
      const auto col_lv = LL_BUILDER.CreateSExt(
          bucket_key, get_int_type(key_component_width * 8, LL_CONTEXT));
      LL_BUILDER.CreateStore(col_lv, key_comp_dest_lv);
    }
  } else {
    LOG(FATAL) << "Overlaps key currently only supported for geospatial types.";
  }
  return key_buff_lv;
}

void OverlapsJoinHashTable::computeBucketSizes(
    std::vector<double>& bucket_sizes_for_dimension,
    const JoinColumn& join_column,
    const std::vector<InnerOuter>& inner_outer_pairs,
    const size_t row_count) {
  // No coalesced keys for overlaps joins yet
  CHECK_EQ(inner_outer_pairs.size(), 1);

  const auto col = inner_outer_pairs[0].first;
  CHECK(col);
  const auto col_ti = col->get_type_info();
  CHECK(col_ti.is_array());

  // Compute the number of dimensions for this overlaps key
  int num_dims{-1};
  if (col_ti.is_fixlen_array()) {
    num_dims = col_ti.get_size() / col_ti.get_elem_type().get_size();
    num_dims /= 2;
  } else {
    CHECK(col_ti.is_varlen_array());
    num_dims = 2;
    // TODO(adb): how can we pick the number of dims in the varlen case? e.g.
    // backwards compatibility with existing bounds cols or generic range joins
  }
  CHECK_GT(num_dims, 0);
  std::vector<double> local_bucket_sizes(num_dims, std::numeric_limits<double>::max());

  VLOG(1) << "Computing bucketed hashjoin with minimum bucket size "
          << std::to_string(g_overlaps_hashjoin_bucket_threshold);

  const auto effective_memory_level = getEffectiveMemoryLevel(inner_outer_pairs);
  if (effective_memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    const int thread_count = cpu_threads();
    compute_bucket_sizes(local_bucket_sizes,
                         join_column,
                         g_overlaps_hashjoin_bucket_threshold,
                         thread_count);
  }
#ifdef HAVE_CUDA
  else {
    // Note that we compute the bucket sizes using only a single GPU
    const int device_id = 0;
    auto& data_mgr = executor_->getCatalog()->getDataMgr();
    ThrustAllocator allocator(&data_mgr, device_id);
    auto device_bucket_sizes_gpu =
        transfer_pod_vector_to_gpu(local_bucket_sizes, allocator);
    auto join_columns_gpu = transfer_object_to_gpu(join_column, allocator);

    compute_bucket_sizes_on_device(device_bucket_sizes_gpu,
                                   join_columns_gpu,
                                   g_overlaps_hashjoin_bucket_threshold,
                                   executor_->blockSize(),
                                   executor_->gridSize());
    copy_from_gpu(&data_mgr,
                  local_bucket_sizes.data(),
                  reinterpret_cast<CUdeviceptr>(device_bucket_sizes_gpu),
                  local_bucket_sizes.size() * sizeof(double),
                  device_id);
  }
#endif

  size_t ctr = 0;
  for (auto& bucket_sz : local_bucket_sizes) {
    VLOG(1) << "Computed bucket size for dim[" << ctr++ << "]: " << bucket_sz;
    bucket_sizes_for_dimension.push_back(1.0 / bucket_sz);
  }

  return;
}
