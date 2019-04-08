/*
 * Copyright 2019 OmniSci, Inc.
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

#include "QueryMemoryInitializer.h"

#include "Execute.h"
#include "GpuMemUtils.h"
#include "ResultRows.h"
#include "ResultSet.h"
#include "StreamingTopN.h"

#include <Shared/checked_alloc.h>
#include <glog/logging.h>

namespace {

void check_total_bitmap_memory(const QueryMemoryDescriptor& query_mem_desc) {
  const int32_t groups_buffer_entry_count = query_mem_desc.getEntryCount();
  if (g_enable_watchdog) {
    checked_int64_t total_bytes_per_group = 0;
    const size_t num_count_distinct_descs =
        query_mem_desc.getCountDistinctDescriptorsSize();
    for (size_t i = 0; i < num_count_distinct_descs; i++) {
      const auto count_distinct_desc = query_mem_desc.getCountDistinctDescriptor(i);
      if (count_distinct_desc.impl_type_ != CountDistinctImplType::Bitmap) {
        continue;
      }
      total_bytes_per_group += count_distinct_desc.bitmapPaddedSizeBytes();
    }
    int64_t total_bytes{0};
    // Using OutOfHostMemory until we can verify that SlabTooBig would also be properly
    // caught
    try {
      total_bytes =
          static_cast<int64_t>(total_bytes_per_group * groups_buffer_entry_count);
    } catch (...) {
      // Absurd amount of memory, merely computing the number of bits overflows int64_t.
      // Don't bother to report the real amount, this is unlikely to ever happen.
      throw OutOfHostMemory(std::numeric_limits<int64_t>::max() / 8);
    }
    if (total_bytes >= 2 * 1000 * 1000 * 1000L) {
      throw OutOfHostMemory(total_bytes);
    }
  }
}

int64_t* alloc_group_by_buffer(const size_t numBytes,
                               RenderAllocatorMap* render_allocator_map) {
  if (render_allocator_map) {
    // NOTE(adb): If we got here, we are performing an in-situ rendering query and are not
    // using CUDA buffers. Therefore we need to allocate result set storage using CPU
    // memory.
    const auto gpu_idx = 0;  // Only 1 GPU supported in CUDA-disabled rendering mode
    auto render_allocator_ptr = render_allocator_map->getRenderAllocator(gpu_idx);
    return reinterpret_cast<int64_t*>(render_allocator_ptr->alloc(numBytes));
  } else {
    return reinterpret_cast<int64_t*>(checked_malloc(numBytes));
  }
}

}  // namespace

QueryMemoryInitializer::QueryMemoryInitializer(
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const int device_id,
    const ExecutorDeviceType device_type,
    const bool output_columnar,
    const bool sort_on_gpu,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int64_t>& consistent_frag_sizes,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    RenderAllocatorMap* render_allocator_map,
    RenderInfo* render_info,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const Executor* executor)
    : row_set_mem_owner_(row_set_mem_owner)
    , init_agg_vals_(executor->plan_state_->init_agg_vals_)
    , consistent_frag_sizes_(consistent_frag_sizes)
    , num_buffers_(getNumBuffers(query_mem_desc, device_type, executor))
    , count_distinct_bitmap_mem_(0)
    , count_distinct_bitmap_mem_bytes_(0)
    , count_distinct_bitmap_crt_ptr_(nullptr)
    , count_distinct_bitmap_host_mem_(nullptr) {
  CHECK(!sort_on_gpu || output_columnar);

  if (consistent_frag_sizes_.empty()) {
    // No fragments in the input, no underlying buffers will be needed.
    return;
  }
  check_total_bitmap_memory(query_mem_desc);
  if (device_type == ExecutorDeviceType::GPU) {
    allocateCountDistinctGpuMem(device_id, query_mem_desc, executor);
  }

  if (render_allocator_map || !query_mem_desc.isGroupBy()) {
    allocateCountDistinctBuffers(query_mem_desc, false, executor);
    if (render_info && render_info->useCudaBuffers()) {
      return;
    }
  }

  if (ra_exe_unit.estimator) {
    return;
  }

  const auto thread_count = device_type == ExecutorDeviceType::GPU
                                ? executor->blockSize() * executor->gridSize()
                                : 1;

  const auto group_buffer_size =
      query_mem_desc.getBufferSizeBytes(ra_exe_unit, thread_count, device_type);
  OOM_TRACE_PUSH(+": group_buffer_size " + std::to_string(group_buffer_size));
  std::unique_ptr<int64_t, CheckedAllocDeleter> group_by_buffer_template(
      static_cast<int64_t*>(checked_malloc(group_buffer_size)));

  if (!query_mem_desc.lazyInitGroups(device_type)) {
    if (output_columnar) {
      initColumnarGroups(
          query_mem_desc, group_by_buffer_template.get(), init_agg_vals_, executor);
    } else {
      auto rows_ptr = group_by_buffer_template.get();
      auto actual_entry_count = query_mem_desc.getEntryCount();
      auto warp_size =
          query_mem_desc.interleavedBins(device_type) ? executor->warpSize() : 1;
      if (use_streaming_top_n(ra_exe_unit, query_mem_desc)) {
        const auto node_count_size = thread_count * sizeof(int64_t);
        memset(rows_ptr, 0, node_count_size);
        const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
        const auto rows_offset =
            streaming_top_n::get_rows_offset_of_heaps(n, thread_count);
        memset(rows_ptr + thread_count, -1, rows_offset - node_count_size);
        rows_ptr += rows_offset / sizeof(int64_t);
        actual_entry_count = n * thread_count;
        warp_size = 1;
      }
      initGroups(query_mem_desc,
                 rows_ptr,
                 init_agg_vals_,
                 actual_entry_count,
                 warp_size,
                 executor);
    }
  }

  if (query_mem_desc.interleavedBins(device_type)) {
    CHECK(query_mem_desc.hasKeylessHash());
  }

  const auto step = device_type == ExecutorDeviceType::GPU &&
                            query_mem_desc.threadsShareMemory() &&
                            query_mem_desc.isGroupBy()
                        ? executor->blockSize()
                        : size_t(1);
  const auto index_buffer_qw = device_type == ExecutorDeviceType::GPU && sort_on_gpu &&
                                       query_mem_desc.hasKeylessHash()
                                   ? query_mem_desc.getEntryCount()
                                   : size_t(0);
  const auto actual_group_buffer_size =
      group_buffer_size + index_buffer_qw * sizeof(int64_t);
  const auto group_buffers_count = !query_mem_desc.isGroupBy() ? 1 : num_buffers_;

  for (size_t i = 0; i < group_buffers_count; i += step) {
    OOM_TRACE_PUSH(+": group_by_buffer " + std::to_string(actual_group_buffer_size));
    auto group_by_buffer =
        alloc_group_by_buffer(actual_group_buffer_size, render_allocator_map);
    if (!query_mem_desc.lazyInitGroups(device_type)) {
      memcpy(group_by_buffer + index_buffer_qw,
             group_by_buffer_template.get(),
             group_buffer_size);
    }
    if (!render_allocator_map) {
      row_set_mem_owner_->addGroupByBuffer(group_by_buffer);
    }
    group_by_buffers_.push_back(group_by_buffer);
    for (size_t j = 1; j < step; ++j) {
      group_by_buffers_.push_back(nullptr);
    }
    const auto column_frag_offsets =
        get_col_frag_offsets(ra_exe_unit.target_exprs, frag_offsets);
    const auto column_frag_sizes =
        get_consistent_frags_sizes(ra_exe_unit.target_exprs, consistent_frag_sizes);
    result_sets_.emplace_back(
        new ResultSet(target_exprs_to_infos(ra_exe_unit.target_exprs, query_mem_desc),
                      getColLazyFetchInfo(ra_exe_unit.target_exprs, executor),
                      col_buffers,
                      column_frag_offsets,
                      column_frag_sizes,
                      device_type,
                      device_id,
                      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc),
                      row_set_mem_owner_,
                      executor));
    result_sets_.back()->allocateStorage(reinterpret_cast<int8_t*>(group_by_buffer),
                                         executor->plan_state_->init_agg_vals_);
    for (size_t j = 1; j < step; ++j) {
      result_sets_.emplace_back(nullptr);
    }
  }
}

void QueryMemoryInitializer::initGroups(const QueryMemoryDescriptor& query_mem_desc,
                                        int64_t* groups_buffer,
                                        const std::vector<int64_t>& init_vals,
                                        const int32_t groups_buffer_entry_count,
                                        const size_t warp_size,
                                        const Executor* executor) {
  const size_t key_count{query_mem_desc.groupColWidthsSize()};
  const size_t row_size{query_mem_desc.getRowSize()};
  const size_t col_base_off{query_mem_desc.getColOffInBytes(0)};

  auto agg_bitmap_size = allocateCountDistinctBuffers(query_mem_desc, true, executor);
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);

  const auto query_mem_desc_fixedup =
      ResultSet::fixupQueryMemoryDescriptor(query_mem_desc);

  if (query_mem_desc.hasKeylessHash()) {
    CHECK(warp_size >= 1);
    CHECK(key_count == 1);
    for (size_t warp_idx = 0; warp_idx < warp_size; ++warp_idx) {
      for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count);
           ++bin, buffer_ptr += row_size) {
        initColumnPerRow(query_mem_desc_fixedup,
                         &buffer_ptr[col_base_off],
                         bin,
                         init_vals,
                         agg_bitmap_size);
      }
    }
    return;
  }

  for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count);
       ++bin, buffer_ptr += row_size) {
    fill_empty_key(buffer_ptr, key_count, query_mem_desc.getEffectiveKeyWidth());
    initColumnPerRow(query_mem_desc_fixedup,
                     &buffer_ptr[col_base_off],
                     bin,
                     init_vals,
                     agg_bitmap_size);
  }
}

namespace {

template <typename T>
int8_t* initColumnarBuffer(T* buffer_ptr, const T init_val, const uint32_t entry_count) {
  static_assert(sizeof(T) <= sizeof(int64_t), "Unsupported template type");
  for (uint32_t i = 0; i < entry_count; ++i) {
    buffer_ptr[i] = init_val;
  }
  return reinterpret_cast<int8_t*>(buffer_ptr + entry_count);
}

}  // namespace

void QueryMemoryInitializer::initColumnarGroups(
    const QueryMemoryDescriptor& query_mem_desc,
    int64_t* groups_buffer,
    const std::vector<int64_t>& init_vals,
    const Executor* executor) {
  CHECK(groups_buffer);
  for (const auto target_expr : executor->plan_state_->target_exprs_) {
    const auto agg_info = target_info(target_expr);
    CHECK(!is_distinct_target(agg_info));
  }
  const int32_t agg_col_count = query_mem_desc.getColCount();
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);

  const auto groups_buffer_entry_count = query_mem_desc.getEntryCount();
  if (!query_mem_desc.hasKeylessHash()) {
    const size_t key_count{query_mem_desc.groupColWidthsSize()};
    for (size_t i = 0; i < key_count; ++i) {
      buffer_ptr = initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr),
                                               EMPTY_KEY_64,
                                               groups_buffer_entry_count);
    }
  }
  // initializing all aggregate columns:
  int32_t init_val_idx = 0;
  for (int32_t i = 0; i < agg_col_count; ++i) {
    if (query_mem_desc.getPaddedColumnWidthBytes(i) > 0) {
      CHECK_LT(init_val_idx, init_vals.size());
      switch (query_mem_desc.getPaddedColumnWidthBytes(i)) {
        case 1:
          buffer_ptr = initColumnarBuffer<int8_t>(
              buffer_ptr, init_vals[init_val_idx++], groups_buffer_entry_count);
          break;
        case 2:
          buffer_ptr = initColumnarBuffer<int16_t>(reinterpret_cast<int16_t*>(buffer_ptr),
                                                   init_vals[init_val_idx++],
                                                   groups_buffer_entry_count);
          break;
        case 4:
          buffer_ptr = initColumnarBuffer<int32_t>(reinterpret_cast<int32_t*>(buffer_ptr),
                                                   init_vals[init_val_idx++],
                                                   groups_buffer_entry_count);
          break;
        case 8:
          buffer_ptr = initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr),
                                                   init_vals[init_val_idx++],
                                                   groups_buffer_entry_count);
          break;
        case 0:
          break;
        default:
          CHECK(false);
      }

      buffer_ptr = align_to_int64(buffer_ptr);
    }
  }
}

void QueryMemoryInitializer::initColumnPerRow(const QueryMemoryDescriptor& query_mem_desc,
                                              int8_t* row_ptr,
                                              const size_t bin,
                                              const std::vector<int64_t>& init_vals,
                                              const std::vector<ssize_t>& bitmap_sizes) {
  int8_t* col_ptr = row_ptr;
  size_t init_vec_idx = 0;
  for (size_t col_idx = 0; col_idx < query_mem_desc.getColCount();
       col_ptr += query_mem_desc.getNextColOffInBytes(col_ptr, bin, col_idx++)) {
    const ssize_t bm_sz{bitmap_sizes[col_idx]};
    int64_t init_val{0};
    if (!bm_sz || !query_mem_desc.isGroupBy()) {
      if (query_mem_desc.getColumnWidth(col_idx).compact > 0) {
        CHECK_LT(init_vec_idx, init_vals.size());
        init_val = init_vals[init_vec_idx++];
      }
    } else {
      CHECK_EQ(static_cast<size_t>(query_mem_desc.getColumnWidth(col_idx).compact),
               sizeof(int64_t));
      init_val =
          bm_sz > 0 ? allocateCountDistinctBitmap(bm_sz) : allocateCountDistinctSet();
      ++init_vec_idx;
    }
    switch (query_mem_desc.getColumnWidth(col_idx).compact) {
      case 1:
        *col_ptr = static_cast<int8_t>(init_val);
        break;
      case 2:
        *reinterpret_cast<int16_t*>(col_ptr) = (int16_t)init_val;
        break;
      case 4:
        *reinterpret_cast<int32_t*>(col_ptr) = (int32_t)init_val;
        break;
      case 8:
        *reinterpret_cast<int64_t*>(col_ptr) = init_val;
        break;
      case 0:
        continue;
      default:
        CHECK(false);
    }
  }
}

void QueryMemoryInitializer::allocateCountDistinctGpuMem(
    const int device_id,
    const QueryMemoryDescriptor& query_mem_desc,
    const Executor* executor) {
  if (query_mem_desc.countDistinctDescriptorsLogicallyEmpty()) {
    return;
  }
  CHECK(executor);
  auto& data_mgr = executor->catalog_->getDataMgr();

  size_t total_bytes_per_entry{0};
  const size_t num_count_distinct_descs =
      query_mem_desc.getCountDistinctDescriptorsSize();
  for (size_t i = 0; i < num_count_distinct_descs; i++) {
    const auto count_distinct_desc = query_mem_desc.getCountDistinctDescriptor(i);
    if (count_distinct_desc.impl_type_ == CountDistinctImplType::Invalid) {
      continue;
    }
    CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap);
    total_bytes_per_entry += count_distinct_desc.bitmapPaddedSizeBytes();
  }

  count_distinct_bitmap_mem_bytes_ =
      total_bytes_per_entry * query_mem_desc.getEntryCount();
  count_distinct_bitmap_mem_ =
      alloc_gpu_mem(&data_mgr, count_distinct_bitmap_mem_bytes_, device_id, nullptr);
  data_mgr.getCudaMgr()->zeroDeviceMem(
      reinterpret_cast<int8_t*>(count_distinct_bitmap_mem_),
      count_distinct_bitmap_mem_bytes_,
      device_id);
  OOM_TRACE_PUSH(+": count_distinct_bitmap_mem_bytes_ " +
                 std::to_string(count_distinct_bitmap_mem_bytes_));
  count_distinct_bitmap_crt_ptr_ = count_distinct_bitmap_host_mem_ =
      static_cast<int8_t*>(checked_malloc(count_distinct_bitmap_mem_bytes_));
  row_set_mem_owner_->addCountDistinctBuffer(
      count_distinct_bitmap_host_mem_, count_distinct_bitmap_mem_bytes_, true);
}

// deferred is true for group by queries; initGroups will allocate a bitmap
// for each group slot
std::vector<ssize_t> QueryMemoryInitializer::allocateCountDistinctBuffers(
    const QueryMemoryDescriptor& query_mem_desc,
    const bool deferred,
    const Executor* executor) {
  const size_t agg_col_count{query_mem_desc.getColCount()};
  std::vector<ssize_t> agg_bitmap_size(deferred ? agg_col_count : 0);

  CHECK_GE(agg_col_count, executor->plan_state_->target_exprs_.size());
  for (size_t target_idx = 0, agg_col_idx = 0;
       target_idx < executor->plan_state_->target_exprs_.size() &&
       agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = executor->plan_state_->target_exprs_[target_idx];
    const auto agg_info = target_info(target_expr);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.is_agg &&
            (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT));
      CHECK_EQ(static_cast<size_t>(query_mem_desc.getColumnWidth(agg_col_idx).actual),
               sizeof(int64_t));
      const auto& count_distinct_desc =
          query_mem_desc.getCountDistinctDescriptor(target_idx);
      CHECK(count_distinct_desc.impl_type_ != CountDistinctImplType::Invalid);
      if (count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap) {
        const auto bitmap_byte_sz = count_distinct_desc.bitmapPaddedSizeBytes();
        if (deferred) {
          agg_bitmap_size[agg_col_idx] = bitmap_byte_sz;
        } else {
          init_agg_vals_[agg_col_idx] = allocateCountDistinctBitmap(bitmap_byte_sz);
        }
      } else {
        CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::StdSet);
        if (deferred) {
          agg_bitmap_size[agg_col_idx] = -1;
        } else {
          init_agg_vals_[agg_col_idx] = allocateCountDistinctSet();
        }
      }
    }
    if (agg_info.agg_kind == kAVG) {
      ++agg_col_idx;
    }
  }

  return agg_bitmap_size;
}

int64_t QueryMemoryInitializer::allocateCountDistinctBitmap(const size_t bitmap_byte_sz) {
  if (count_distinct_bitmap_host_mem_) {
    CHECK(count_distinct_bitmap_crt_ptr_);
    auto ptr = count_distinct_bitmap_crt_ptr_;
    count_distinct_bitmap_crt_ptr_ += bitmap_byte_sz;
    row_set_mem_owner_->addCountDistinctBuffer(ptr, bitmap_byte_sz, false);
    return reinterpret_cast<int64_t>(ptr);
  }
  OOM_TRACE_PUSH(+": count_distinct_buffer " + std::to_string(bitmap_byte_sz));
  auto count_distinct_buffer = static_cast<int8_t*>(checked_calloc(bitmap_byte_sz, 1));
  row_set_mem_owner_->addCountDistinctBuffer(count_distinct_buffer, bitmap_byte_sz, true);
  return reinterpret_cast<int64_t>(count_distinct_buffer);
}

int64_t QueryMemoryInitializer::allocateCountDistinctSet() {
  auto count_distinct_set = new std::set<int64_t>();
  row_set_mem_owner_->addCountDistinctSet(count_distinct_set);
  return reinterpret_cast<int64_t>(count_distinct_set);
}

std::vector<ColumnLazyFetchInfo> QueryMemoryInitializer::getColLazyFetchInfo(
    const std::vector<Analyzer::Expr*>& target_exprs,
    const Executor* executor) const {
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  for (const auto target_expr : target_exprs) {
    if (!executor->plan_state_->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(
          ColumnLazyFetchInfo{false, -1, SQLTypeInfo(kNULLT, false)});
    } else {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      CHECK(col_var);
      auto col_id = col_var->get_column_id();
      auto rte_idx = (col_var->get_rte_idx() == -1) ? 0 : col_var->get_rte_idx();
      auto cd = (col_var->get_table_id() > 0)
                    ? get_column_descriptor(
                          col_id, col_var->get_table_id(), *executor->catalog_)
                    : nullptr;
      if (cd && IS_GEO(cd->columnType.get_type())) {
        // Geo coords cols will be processed in sequence. So we only need to track the
        // first coords col in lazy fetch info.
        {
          auto cd0 = get_column_descriptor(
              col_id + 1, col_var->get_table_id(), *executor->catalog_);
          auto col0_ti = cd0->columnType;
          CHECK(!cd0->isVirtualCol);
          auto col0_var = makeExpr<Analyzer::ColumnVar>(
              col0_ti, col_var->get_table_id(), cd0->columnId, rte_idx);
          auto local_col0_id = executor->getLocalColumnId(col0_var.get(), false);
          col_lazy_fetch_info.emplace_back(
              ColumnLazyFetchInfo{true, local_col0_id, col0_ti});
        }
      } else {
        auto local_col_id = executor->getLocalColumnId(col_var, false);
        const auto& col_ti = col_var->get_type_info();
        col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col_id, col_ti});
      }
    }
  }
  return col_lazy_fetch_info;
}

size_t QueryMemoryInitializer::getNumBuffers(const QueryMemoryDescriptor& query_mem_desc,
                                             const ExecutorDeviceType device_type,
                                             const Executor* executor) const {
  return device_type == ExecutorDeviceType::CPU
             ? 1
             : executor->blockSize() *
                   (query_mem_desc.blocksShareMemory() ? 1 : executor->gridSize());
}

namespace {

// in-place compaction of output buffer
void compact_projection_buffer_for_cpu_columnar(
    const QueryMemoryDescriptor& query_mem_desc,
    int8_t* projection_buffer,
    const size_t projection_count) {
  // the first column (row indices) remains unchanged.
  CHECK(projection_count <= query_mem_desc.getEntryCount());
  constexpr size_t row_index_width = sizeof(int64_t);
  size_t buffer_offset1{projection_count * row_index_width};
  // other columns are actual non-lazy columns for the projection:
  for (size_t i = 0; i < query_mem_desc.getColCount(); i++) {
    if (query_mem_desc.getPaddedColumnWidthBytes(i) > 0) {
      auto column_proj_size =
          projection_count * query_mem_desc.getPaddedColumnWidthBytes(i);
      auto buffer_offset2 = query_mem_desc.getColOffInBytes(i);
      if (buffer_offset1 + column_proj_size >= buffer_offset2) {
        // overlapping
        std::memmove(projection_buffer + buffer_offset1,
                     projection_buffer + buffer_offset2,
                     column_proj_size);
      } else {
        std::memcpy(projection_buffer + buffer_offset1,
                    projection_buffer + buffer_offset2,
                    column_proj_size);
      }
      buffer_offset1 += align_to_int64(column_proj_size);
    }
  }
}

}  // namespace

void QueryMemoryInitializer::compactProjectionBuffersCpu(
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t projection_count) {
  const auto num_allocated_rows =
      std::min(projection_count, query_mem_desc.getEntryCount());

  // copy the results from the main buffer into projection_buffer
  compact_projection_buffer_for_cpu_columnar(
      query_mem_desc,
      reinterpret_cast<int8_t*>(group_by_buffers_[0]),
      num_allocated_rows);

  // update the entry count for the result set, and its underlying storage
  CHECK(!result_sets_.empty());
  result_sets_.front()->updateStorageEntryCount(num_allocated_rows);
}

void QueryMemoryInitializer::compactProjectionBuffersGpu(
    const QueryMemoryDescriptor& query_mem_desc,
    Data_Namespace::DataMgr* data_mgr,
    const GpuQueryMemory& gpu_query_mem,
    const size_t projection_count,
    const int device_id) {
  // store total number of allocated rows:
  const auto num_allocated_rows =
      std::min(projection_count, query_mem_desc.getEntryCount());

  // copy the results from the main buffer into projection_buffer
  copy_projection_buffer_from_gpu_columnar(
      data_mgr,
      gpu_query_mem,
      query_mem_desc,
      reinterpret_cast<int8_t*>(group_by_buffers_[0]),
      num_allocated_rows,
      device_id);

  // update the entry count for the result set, and its underlying storage
  CHECK(!result_sets_.empty());
  result_sets_.front()->updateStorageEntryCount(num_allocated_rows);
}

GpuGroupByBuffers QueryMemoryInitializer::createGroupByBuffersOnGpu(
    const CudaAllocator& cuda_allocator,
    RenderAllocator* render_allocator,
    const QueryMemoryDescriptor& query_mem_desc,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const bool can_sort_on_gpu) {
  return create_dev_group_by_buffers(cuda_allocator,
                                     group_by_buffers_,
                                     query_mem_desc,
                                     block_size_x,
                                     grid_size_x,
                                     device_id,
                                     can_sort_on_gpu,
                                     false,
                                     render_allocator);
}

void QueryMemoryInitializer::copyGroupByBuffersFromGpu(
    Data_Namespace::DataMgr* data_mgr,
    const QueryMemoryDescriptor& query_mem_desc,
    const GpuQueryMemory& gpu_query_mem,
    const RelAlgExecutionUnit& ra_exe_unit,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    const bool prepend_index_buffer) const {
  const auto thread_count = block_size_x * grid_size_x;
  const auto total_buff_size = query_mem_desc.getBufferSizeBytes(
      ra_exe_unit, thread_count, ExecutorDeviceType::GPU);
  copy_group_by_buffers_from_gpu(data_mgr,
                                 group_by_buffers_,
                                 total_buff_size,
                                 gpu_query_mem.group_by_buffers.second,
                                 query_mem_desc,
                                 block_size_x,
                                 grid_size_x,
                                 device_id,
                                 prepend_index_buffer);
}

void QueryMemoryInitializer::applyStreamingTopNOffsetCpu(
    const QueryMemoryDescriptor& query_mem_desc,
    const RelAlgExecutionUnit& ra_exe_unit) {
  CHECK_EQ(group_by_buffers_.size(), size_t(1));

  const auto rows_copy = streaming_top_n::get_rows_copy_from_heaps(
      group_by_buffers_[0],
      query_mem_desc.getBufferSizeBytes(ra_exe_unit, 1, ExecutorDeviceType::CPU),
      ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit,
      1);
  CHECK_EQ(rows_copy.size(),
           query_mem_desc.getEntryCount() * query_mem_desc.getRowSize());
  memcpy(group_by_buffers_[0], &rows_copy[0], rows_copy.size());
}

void QueryMemoryInitializer::applyStreamingTopNOffsetGpu(
    Data_Namespace::DataMgr* data_mgr,
    const QueryMemoryDescriptor& query_mem_desc,
    const GpuQueryMemory& gpu_query_mem,
    const RelAlgExecutionUnit& ra_exe_unit,
    const unsigned total_thread_count,
    const int device_id) {
#ifdef HAVE_CUDA
  CHECK_EQ(group_by_buffers_.size(), num_buffers_);

  const auto rows_copy = pick_top_n_rows_from_dev_heaps(
      data_mgr,
      reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
      ra_exe_unit,
      query_mem_desc,
      total_thread_count,
      device_id);
  CHECK_EQ(
      rows_copy.size(),
      static_cast<size_t>(query_mem_desc.getEntryCount() * query_mem_desc.getRowSize()));
  memcpy(group_by_buffers_[0], &rows_copy[0], rows_copy.size());
#else
  UNREACHABLE();
#endif
}
